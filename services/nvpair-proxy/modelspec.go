// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// modelSpec is what the capacity planner needs to know about one model: how
// much memory its weights take and how much each token of context costs.
// Nothing here is guessed silently: Source says where the numbers came from,
// and a model the engine describes too little of comes back without a cost so
// the user can type one in.
type modelSpec struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
	// Source is "metadata" (architecture read from the engine or the model
	// file), "measured" (from the memory a loaded model actually uses), or ""
	// when the per-token cost is unknown.
	Source       string `json:"source"`
	Architecture string `json:"architecture,omitempty"`
	Format       string `json:"format,omitempty"`

	// WeightsBytes is the memory the weights take once loaded, which the
	// planner counts: TensorBytes when the tensor table gives it, else the
	// download size as an estimate. WeightsSource says which ("tensors",
	// "download", or "" when neither is known).
	WeightsBytes  int64  `json:"weightsBytes"`
	WeightsSource string `json:"weightsSource,omitempty"`
	// DownloadBytes is the model's size on disk, as the engine lists it
	// (DownloadSource: "ollama-tags", "lmstudio-models") or as the GGUF files
	// add up ("gguf-files").
	DownloadBytes  int64  `json:"downloadBytes,omitempty"`
	DownloadSource string `json:"downloadSource,omitempty"`
	// TensorBytes is the sum of the weight tensors in the model's tensor
	// table: what the weights occupy in memory once loaded, since quantized
	// tensors load as stored. Zero when a tensor type is unknown (MLX and
	// other non-GGUF formats) or the table is unavailable.
	TensorBytes int64 `json:"tensorBytes,omitempty"`
	// Loaded* is what Ollama's /api/ps reports for a copy loaded now: its
	// whole footprint (weights, context cache and working buffers), the part
	// on the GPU, and the context it was loaded with.
	LoadedBytes         int64 `json:"loadedBytes,omitempty"`
	LoadedVRAMBytes     int64 `json:"loadedVramBytes,omitempty"`
	LoadedContextTokens int64 `json:"loadedContextTokens,omitempty"`
	MaxContextTokens    int64 `json:"maxContextTokens,omitempty"`
	// ContextSource is "web" when MaxContextTokens was read from the model's
	// page (model/web-facts) because the engine and the file gave none;
	// ContextPage is that page and ContextReader the model that read it.
	ContextSource string `json:"contextSource,omitempty"`
	ContextPage   string `json:"contextPage,omitempty"`
	ContextReader string `json:"contextReader,omitempty"`
	// WebMoE says the model's page calls it a mixture of experts while the
	// engine and the file describe no experts.
	WebMoE bool `json:"webMoe,omitempty"`

	// KVBytesPerToken is the f16 key/value cache cost of one token of
	// context in the layers whose cache grows with the window.
	KVBytesPerToken int64 `json:"kvBytesPerToken"`
	// Sliding-window layers cache at most SlidingWindow tokens per session.
	SlidingBytesPerToken int64 `json:"slidingBytesPerToken,omitempty"`
	SlidingWindow        int64 `json:"slidingWindow,omitempty"`
	// RecurrentBytesPerSession is the fixed state of linear-attention or SSM
	// layers, which does not grow with the window.
	RecurrentBytesPerSession int64 `json:"recurrentBytesPerSession,omitempty"`

	// ExpertCount is the number of routed experts in a mixture-of-experts
	// model and ExpertUsedCount how many of them run per token. ExpertBytes
	// is the size of their weights, which llama.cpp can keep in system
	// memory instead of on the GPU, and ExpertLayerBytes that size per layer
	// (blk.i), in the order llama.cpp's --n-cpu-moe takes them.
	ExpertCount      int     `json:"expertCount,omitempty"`
	ExpertUsedCount  int     `json:"expertUsedCount,omitempty"`
	ExpertBytes      int64   `json:"expertBytes,omitempty"`
	ExpertLayerBytes []int64 `json:"expertLayerBytes,omitempty"`
	// TotalParams counts every weight in the tensor table, and ActiveParams
	// those one token runs (all but the unused experts); 0 when unknown.
	TotalParams  int64 `json:"totalParams,omitempty"`
	ActiveParams int64 `json:"activeParams,omitempty"`

	// Defaults are the window, slots and reply length the engine runs this
	// model with now, or the model's own defaults, for the planner to start
	// from.
	Defaults *modelDefaults `json:"defaults,omitempty"`

	Layers          int `json:"layers,omitempty"`
	AttentionLayers int `json:"attentionLayers,omitempty"`
	KVHeads         int `json:"kvHeads,omitempty"`
	KeyLength       int `json:"keyLength,omitempty"`
	ValueLength     int `json:"valueLength,omitempty"`

	Notes []string `json:"notes,omitempty"`
}

func metaInt(meta map[string]ggufValue, key string) (int64, bool) {
	switch v := meta[key].(type) {
	case int64:
		return v, true
	case float64:
		return int64(v), true
	case []int64:
		// A per-layer array; callers that need every layer read it directly.
		if len(v) > 0 {
			return v[0], true
		}
	}
	return 0, false
}

// slidingPatterns is how many layers in each repeating block use a sliding
// window, for architectures whose GGUF does not record the pattern itself:
// Gemma 3 alternates five sliding layers with one global layer, Gemma 2
// alternates one and one.
var slidingPatterns = map[string][2]int{
	"gemma3":       {5, 6},
	"gemma3n":      {4, 5},
	"gemma2":       {1, 2},
	"cohere2":      {3, 4},
	"gpt-oss":      {1, 2},
	"llama4":       {3, 4},
	"exaone4":      {3, 4},
	"smallthinker": {3, 4},
}

// applyArchitecture fills the per-token cost from architecture metadata in
// GGUF key form ("<arch>.block_count", "<arch>.attention.head_count_kv", ...).
// It reports whether the metadata was complete enough to compute a cost.
func (s *modelSpec) applyArchitecture(meta map[string]ggufValue) bool {
	arch, _ := meta["general.architecture"].(string)
	if arch == "" {
		return false
	}
	s.Architecture = arch
	k := func(name string) string { return arch + "." + name }

	layers, ok := metaInt(meta, k("block_count"))
	if !ok || layers <= 0 {
		return false
	}
	s.Layers = int(layers)
	if ctx, ok := metaInt(meta, k("context_length")); ok && ctx > 0 {
		s.MaxContextTokens = ctx
	}
	heads, _ := metaInt(meta, k("attention.head_count"))
	embed, _ := metaInt(meta, k("embedding_length"))
	keyLen, hasKey := metaInt(meta, k("attention.key_length"))
	valLen, hasVal := metaInt(meta, k("attention.value_length"))
	if !hasKey && heads > 0 && embed > 0 {
		keyLen = embed / heads
	}
	if !hasVal {
		valLen = keyLen
	}
	if keyLen <= 0 || valLen <= 0 {
		return false
	}
	s.KeyLength, s.ValueLength = int(keyLen), int(valLen)

	// Key/value heads per layer: a scalar, or an array with 0 for layers
	// that have no attention (linear-attention and SSM layers).
	perLayer := make([]int64, layers)
	switch v := meta[k("attention.head_count_kv")].(type) {
	case []int64:
		for i := range perLayer {
			if i < len(v) {
				perLayer[i] = v[i]
			}
		}
	default:
		kv, ok := metaInt(meta, k("attention.head_count_kv"))
		if !ok {
			kv = heads
		}
		if kv <= 0 {
			return false
		}
		for i := range perLayer {
			perLayer[i] = kv
		}
	}
	// Hybrid models record that only every Nth layer is full attention.
	if interval, ok := metaInt(meta, k("full_attention_interval")); ok && interval > 1 {
		for i := range perLayer {
			if (i+1)%int(interval) != 0 {
				perLayer[i] = 0
			}
		}
	}

	attention := 0
	var maxHeads int64
	for _, h := range perLayer {
		if h > 0 {
			attention++
			if h > maxHeads {
				maxHeads = h
			}
		}
	}
	if attention == 0 {
		return false
	}
	s.AttentionLayers = attention
	s.KVHeads = int(maxHeads)
	perHeadToken := (keyLen + valLen) * 2 // f16

	// Sliding-window layers keep only the last window of tokens.
	if window, ok := metaInt(meta, k("attention.sliding_window")); ok && window > 0 {
		pattern, known := slidingPatterns[arch]
		if p, ok := metaInt(meta, k("attention.sliding_window_pattern")); ok && p > 1 {
			pattern, known = [2]int{int(p) - 1, int(p)}, true
		}
		if known {
			s.SlidingWindow = window
			idx := 0
			for i, h := range perLayer {
				if h == 0 {
					continue
				}
				if idx%pattern[1] < pattern[0] {
					s.SlidingBytesPerToken += h * perHeadToken
					perLayer[i] = 0
				}
				idx++
			}
		} else {
			s.Notes = append(s.Notes, fmt.Sprintf("%s uses a %d-token sliding window, but its layer pattern is unknown; every layer is counted at the full window, which overestimates", arch, window))
		}
	}
	for _, h := range perLayer {
		s.KVBytesPerToken += h * perHeadToken
	}

	// Linear-attention / SSM layers hold a fixed state per session, in f32.
	if recurrent := int(layers) - attention; recurrent > 0 {
		state, _ := metaInt(meta, k("ssm.state_size"))
		inner, _ := metaInt(meta, k("ssm.inner_size"))
		conv, _ := metaInt(meta, k("ssm.conv_kernel"))
		groups, _ := metaInt(meta, k("ssm.group_count"))
		if state > 0 && inner > 0 {
			perLayer := inner*state + max64(conv-1, 0)*(inner+2*groups*state)
			s.RecurrentBytesPerSession = int64(recurrent) * perLayer * 4
		} else {
			s.Notes = append(s.Notes, fmt.Sprintf("%d of %d layers are not attention layers and their per-session state is not described; it is left to the safety reserve", recurrent, layers))
		}
	}
	s.Source = "metadata"
	return true
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// downloadOverTensors is how much more than the weight tensors a download may
// hold (tokenizer and other metadata) before the rest is taken to be files
// that load too, such as a vision projector.
const downloadOverTensors = 1.10

// settleWeights chooses the weight size the planner counts from the download
// size and the tensor sum, and says where it came from.
func (s *modelSpec) settleWeights() {
	switch {
	case s.TensorBytes > 0 && float64(s.DownloadBytes) > float64(s.TensorBytes)*downloadOverTensors:
		s.WeightsBytes, s.WeightsSource = s.DownloadBytes, "download"
		s.Notes = append(s.Notes, fmt.Sprintf("the download holds %.1f GB more than the weight tensors (such as a vision projector), which may load too; the plan counts the download size", float64(s.DownloadBytes-s.TensorBytes)/float64(gib)))
	case s.TensorBytes > 0:
		s.WeightsBytes, s.WeightsSource = s.TensorBytes, "tensors"
	case s.DownloadBytes > 0:
		s.WeightsBytes, s.WeightsSource = s.DownloadBytes, "download"
	}
}

// jsonToMeta converts Ollama's model_info (JSON numbers, strings and arrays) to the
// GGUF value form applyArchitecture reads.
func jsonToMeta(info map[string]any) map[string]ggufValue {
	out := make(map[string]ggufValue, len(info))
	for key, v := range info {
		switch t := v.(type) {
		case float64:
			out[key] = int64(t)
		case string:
			out[key] = t
		case bool:
			out[key] = t
		case []any:
			if strs, ok := jsonStrings(t); ok {
				out[key] = strs
				continue
			}
			ints := make([]int64, 0, len(t))
			for _, e := range t {
				f, ok := e.(float64)
				if !ok {
					ints = nil
					break
				}
				ints = append(ints, int64(f))
			}
			if ints != nil {
				out[key] = ints
			}
		}
	}
	return out
}

// jsonStrings converts a short, non-empty JSON array of strings
// (general.tags) to the GGUF form; vocabularies are left out, as in a file.
func jsonStrings(values []any) ([]string, bool) {
	if len(values) == 0 || len(values) > ggufMaxKeptStrings {
		return nil, false
	}
	out := make([]string, 0, len(values))
	for _, e := range values {
		s, ok := e.(string)
		if !ok {
			return nil, false
		}
		out = append(out, s)
	}
	return out, true
}

// specClient is the planner's engine reader; tests replace homeDir.
type specClient struct {
	client  *http.Client
	homeDir func() (string, error)
}

func (c *specClient) do(ctx context.Context, method string, base *url.URL, path string, query url.Values, body any, dst any) error {
	u := *base
	u.Path = path
	u.RawQuery = query.Encode()
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, u.String(), rd)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s %s: HTTP %d", method, path, resp.StatusCode)
	}
	return json.NewDecoder(io.LimitReader(resp.Body, 64<<20)).Decode(dst)
}

// describe returns what the engine and the model file say about one model.
// knownParallel is the slot count PAIR configured, used to turn a loaded
// model's measured memory into a per-token cost (0 when unknown).
func (c *specClient) describe(ctx context.Context, engine string, base *url.URL, model string, knownParallel int) (modelSpec, error) {
	s := modelSpec{Engine: engine, Model: model}
	var err error
	switch engine {
	case "ollama":
		err = c.describeOllama(ctx, base, &s, knownParallel)
	case "llamacpp":
		err = c.describeLlamaCPP(ctx, base, &s)
	case "lmstudio":
		err = c.describeLMStudio(ctx, base, &s)
	default:
		err = fmt.Errorf("unknown engine %q", engine)
	}
	return s, err
}

func (c *specClient) describeOllama(ctx context.Context, base *url.URL, s *modelSpec, knownParallel int) error {
	var show struct {
		Parameters string         `json:"parameters"`
		ModelInfo  map[string]any `json:"model_info"`
		Tensors    []ollamaTensor `json:"tensors"`
		Details    struct {
			Format string `json:"format"`
		} `json:"details"`
	}
	if err := c.do(ctx, http.MethodPost, base, "/api/show", nil, map[string]any{"model": s.Model, "verbose": true}, &show); err != nil {
		return err
	}
	s.Format = show.Details.Format
	meta := jsonToMeta(show.ModelInfo)
	s.applyArchitecture(meta)
	if tensors, ok := ollamaTensors(show.Tensors); ok {
		s.applyExperts(meta, tensors)
	}
	if b, ok := ollamaTensorBytes(show.Tensors); ok {
		s.TensorBytes = b
	}

	var tags struct {
		Models []struct {
			Name string `json:"name"`
			Size int64  `json:"size"`
		} `json:"models"`
	}
	if err := c.do(ctx, http.MethodGet, base, "/api/tags", nil, nil, &tags); err == nil {
		for _, m := range tags.Models {
			if m.Name == s.Model {
				s.DownloadBytes, s.DownloadSource = m.Size, "ollama-tags"
			}
		}
	}
	s.settleWeights()

	// A loaded copy: its footprint is reported, and when the architecture
	// is missing (safetensors/MLX models, and some newer GGUF ones) it is
	// measured for the per-token cost.
	var ps struct {
		Models []struct {
			Name          string `json:"name"`
			Size          int64  `json:"size"`
			SizeVRAM      int64  `json:"size_vram"`
			ContextLength int64  `json:"context_length"`
		} `json:"models"`
	}
	psErr := c.do(ctx, http.MethodGet, base, "/api/ps", nil, nil, &ps)
	// The window a loaded copy runs with is in force; the model's num_ctx
	// is what Ollama loads it with otherwise.
	if psErr == nil {
		for _, m := range ps.Models {
			if m.Name == s.Model {
				s.setDefault(defaultWindow, m.ContextLength, "engine")
			}
		}
	}
	params := ollamaParameters(show.Parameters)
	s.setDefault(defaultWindow, params["num_ctx"], "model")
	s.setDefault(defaultReply, params["num_predict"], "model")
	if psErr == nil {
		for _, m := range ps.Models {
			if m.Name != s.Model {
				continue
			}
			s.LoadedBytes, s.LoadedVRAMBytes, s.LoadedContextTokens = m.Size, m.SizeVRAM, m.ContextLength
			if s.Source != "" || m.ContextLength <= 0 || s.WeightsBytes <= 0 || m.Size <= s.WeightsBytes {
				continue
			}
			slots := knownParallel
			if slots <= 0 {
				slots = 1
				s.Notes = append(s.Notes, "measured from the loaded model assuming one parallel slot; this overestimates the cost when Ollama runs more")
			}
			s.KVBytesPerToken = (m.Size - s.WeightsBytes) / (m.ContextLength * int64(slots))
			s.Source = "measured"
			s.Notes = append(s.Notes, "measured cost includes the engine's working buffers, so it is on the high side")
		}
	}
	if s.Source == "" {
		s.Notes = append(s.Notes, "Ollama does not describe this model's layers; load it once so PAIR can measure it, or enter its context cost per token")
	}
	return nil
}

var shardSuffix = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

// fileSpec reads a GGUF model file (and its shards' sizes) into s; the
// caller settles the weight size.
func fileSpec(path string, s *modelSpec) error {
	meta, tensors, err := readGGUFFile(path)
	if err != nil {
		return err
	}
	s.Format = "gguf"
	if !s.applyArchitecture(meta) {
		s.Notes = append(s.Notes, "the model file does not describe its attention layers; enter the context cost per token")
	}
	// A split model keeps part of the tensor table in each shard.
	shards := []string{path}
	if m := shardSuffix.FindStringSubmatch(path); m != nil {
		prefix := strings.TrimSuffix(path, m[0])
		shards, _ = filepath.Glob(prefix + "-*-of-" + m[2] + ".gguf")
	}
	var total int64
	complete := true
	for _, p := range shards {
		if st, err := os.Stat(p); err == nil {
			total += st.Size()
		}
		if p == path {
			continue
		}
		if _, more, err := readGGUFFile(p); err == nil {
			tensors = append(tensors, more...)
		} else {
			complete = false
		}
	}
	s.DownloadBytes, s.DownloadSource = total, "gguf-files"
	s.TensorBytes = 0
	if b, ok := tensorBytes(tensors); ok && complete {
		s.TensorBytes = b
	}
	s.applyExperts(meta, tensors)
	return nil
}

func (c *specClient) describeLlamaCPP(ctx context.Context, base *url.URL, s *modelSpec) error {
	var props llamaCPPProps
	if err := c.do(ctx, http.MethodGet, base, "/props", url.Values{"model": {s.Model}}, nil, &props); err != nil {
		return err
	}
	s.applyLlamaCPPDefaults(props)
	if props.ModelPath == "" {
		s.Notes = append(s.Notes, "llama.cpp did not report the model file; load the model first")
		return nil
	}
	if err := fileSpec(props.ModelPath, s); err != nil {
		return err
	}
	s.settleWeights()
	return nil
}

func (c *specClient) describeLMStudio(ctx context.Context, base *url.URL, s *modelSpec) error {
	var list struct {
		Models []struct {
			Key              string             `json:"key"`
			Path             string             `json:"path"`
			Format           string             `json:"format"`
			SizeBytes        int64              `json:"size_bytes"`
			MaxContextLength int64              `json:"max_context_length"`
			LoadedInstances  []lmStudioInstance `json:"loaded_instances"`
		} `json:"models"`
	}
	if err := c.do(ctx, http.MethodGet, base, "/api/v1/models", nil, nil, &list); err != nil {
		return err
	}
	for _, m := range list.Models {
		if m.Key != s.Model {
			continue
		}
		s.DownloadBytes, s.DownloadSource = m.SizeBytes, "lmstudio-models"
		s.MaxContextTokens = m.MaxContextLength
		s.Format = m.Format
		s.applyLMStudioDefaults(m.LoadedInstances)
		if path := c.lmStudioFile(m.Path); path != "" {
			saved := *s
			if err := fileSpec(path, s); err != nil {
				*s = saved
				s.Notes = append(s.Notes, "could not read the model file: "+err.Error())
			}
			// LM Studio's listed size covers the whole model folder (a
			// vision projector included); the GGUF read gives the tensors.
			if saved.DownloadBytes > 0 {
				s.DownloadBytes, s.DownloadSource = saved.DownloadBytes, saved.DownloadSource
			}
			if saved.MaxContextTokens > 0 && s.MaxContextTokens == 0 {
				s.MaxContextTokens = saved.MaxContextTokens
			}
		}
		s.settleWeights()
		if s.Source == "" {
			s.Notes = append(s.Notes, "LM Studio's model file could not be read; enter the context cost per token")
		}
		return nil
	}
	s.Notes = append(s.Notes, "LM Studio does not list this model")
	return nil
}

// lmStudioModelsDir is LM Studio's models folder: the one its settings name
// (users often move it to a bigger drive), else the default under home.
func lmStudioModelsDir(home string) string {
	var settings struct {
		DownloadsFolder string `json:"downloadsFolder"`
	}
	if raw, err := os.ReadFile(filepath.Join(home, ".lmstudio", "settings.json")); err == nil &&
		json.Unmarshal(raw, &settings) == nil && filepath.IsAbs(settings.DownloadsFolder) {
		return filepath.Clean(settings.DownloadsFolder)
	}
	return filepath.Join(home, ".lmstudio", "models")
}

// lmStudioFile resolves LM Studio's relative model path to a GGUF file in its
// models folder, or "" when it is not a GGUF model there.
func (c *specClient) lmStudioFile(rel string) string {
	if rel == "" {
		return ""
	}
	home, err := c.homeDir()
	if err != nil {
		return ""
	}
	root := lmStudioModelsDir(home)
	p := filepath.Join(root, filepath.FromSlash(rel))
	if !strings.HasPrefix(p, root+string(filepath.Separator)) {
		return ""
	}
	if st, err := os.Stat(p); err == nil && st.IsDir() {
		matches, _ := filepath.Glob(filepath.Join(p, "*.gguf"))
		for _, m := range matches {
			if !strings.Contains(strings.ToLower(filepath.Base(m)), "mmproj") {
				return m
			}
		}
		return ""
	}
	if strings.HasSuffix(strings.ToLower(p), ".gguf") {
		return p
	}
	return ""
}
