// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

// Model details are what an engine and a model's file say about one
// downloaded model, for the model's "About" view: its description, tags,
// capabilities, size and links to its page. Every field is copied from what
// the engine or the GGUF header reported; nothing is inferred, and a field the
// engine did not report is left empty.
//
// A model on another node is described by that node: its proxy answers
// peerModelDetailsPath on the cluster mTLS ingress, because only it can read
// its engine's model files.

const (
	// peerModelDetailsPath is served on the mTLS ingress only:
	// GET /pair/v1/peer/models/details?model=<name>. The engine is the one
	// whose ingress answers.
	peerModelDetailsPath = "/pair/v1/peer/models/details"
	// The broker gives a relayed call 5 seconds (proxyCallTimeout), so the
	// whole answer, a peer's included, has to fit inside that.
	modelDetailsTimeout = 4500 * time.Millisecond
	// peerDetailsTimeout bounds the question to another node, and
	// peerEngineTimeout that node's own engine and file reads, so a slow peer
	// still answers inside the caller's wait.
	peerDetailsTimeout = 3500 * time.Millisecond
	peerEngineTimeout  = 3 * time.Second
	// maxLicenseText keeps a full licence text (Ollama sends the whole file)
	// to a readable excerpt.
	maxLicenseText = 16 << 10
)

// modelLink is a page about the model. Source says which reported field the
// URL came from, so the UI can say why the link is there.
type modelLink struct {
	Label  string `json:"label"`
	URL    string `json:"url"`
	Source string `json:"source"`
}

type modelDetails struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`

	Name        string   `json:"name,omitempty"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	Languages   []string `json:"languages,omitempty"`
	Author      string   `json:"author,omitempty"`
	Publisher   string   `json:"publisher,omitempty"`
	// License is the licence's name (general.license, "apache-2.0"), and
	// LicenseText the licence the engine ships with the model, cut to
	// maxLicenseText.
	License     string `json:"license,omitempty"`
	LicenseText string `json:"licenseText,omitempty"`

	Type           string   `json:"type,omitempty"`
	Architecture   string   `json:"architecture,omitempty"`
	Family         string   `json:"family,omitempty"`
	ParameterSize  string   `json:"parameterSize,omitempty"`
	ParameterCount int64    `json:"parameterCount,omitempty"`
	Quantization   string   `json:"quantization,omitempty"`
	Format         string   `json:"format,omitempty"`
	Capabilities   []string `json:"capabilities,omitempty"`

	// ContextLength is the longest context the model was trained for, and
	// LoadedContext the window a running engine gave it.
	ContextLength int64  `json:"contextLength,omitempty"`
	LoadedContext int64  `json:"loadedContext,omitempty"`
	SizeBytes     int64  `json:"sizeBytes,omitempty"`
	Digest        string `json:"digest,omitempty"`
	ModifiedAt    string `json:"modifiedAt,omitempty"`
	// Parameters are the engine's default generation settings for the
	// model (Ollama's Modelfile PARAMETER lines).
	Parameters string `json:"parameters,omitempty"`

	Links []modelLink `json:"links,omitempty"`
	Notes []string    `json:"notes,omitempty"`
}

// ggufFileTypes names llama.cpp's general.file_type values (llama_ftype).
var ggufFileTypes = map[int64]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
	10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
	16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
	22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
	28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
	37: "TQ2_0", 38: "MXFP4_MOE",
}

func metaString(meta map[string]ggufValue, key string) string {
	s, _ := meta[key].(string)
	return strings.TrimSpace(s)
}

func metaStrings(meta map[string]ggufValue, key string) []string {
	switch v := meta[key].(type) {
	case []string:
		out := make([]string, 0, len(v))
		for _, s := range v {
			if s = strings.TrimSpace(s); s != "" {
				out = append(out, s)
			}
		}
		return out
	case string:
		if s := strings.TrimSpace(v); s != "" {
			return []string{s}
		}
	}
	return nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// webURL returns raw when it is an absolute http(s) URL, else "".
func webURL(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
		return ""
	}
	return u.String()
}

// huggingFaceRepo is the page of a Hugging Face repository id ("org/name").
func huggingFaceRepo(id string) string {
	parts := strings.Split(strings.Trim(strings.TrimSpace(id), "/"), "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return ""
	}
	for _, p := range parts {
		if strings.ContainsAny(p, " ?#:") {
			return ""
		}
	}
	return "https://huggingface.co/" + url.PathEscape(parts[0]) + "/" + url.PathEscape(parts[1])
}

func (d *modelDetails) addLink(label, raw, source string) {
	u := webURL(raw)
	if u == "" {
		return
	}
	for _, l := range d.Links {
		if l.URL == u {
			return
		}
	}
	d.Links = append(d.Links, modelLink{Label: label, URL: u, Source: source})
}

// applyGeneral copies the general.* metadata of a GGUF header (or Ollama's
// model_info, which carries the same keys) into d. Fields d already has are
// kept: the engine's own answer comes first.
func (d *modelDetails) applyGeneral(meta map[string]ggufValue) {
	set := func(dst *string, v string) {
		if *dst == "" {
			*dst = v
		}
	}
	set(&d.Name, metaString(meta, "general.name"))
	set(&d.Description, metaString(meta, "general.description"))
	set(&d.Author, firstNonEmpty(metaString(meta, "general.author"), metaString(meta, "general.organization")))
	set(&d.License, firstNonEmpty(metaString(meta, "general.license"), metaString(meta, "general.license.name")))
	set(&d.Architecture, metaString(meta, "general.architecture"))
	set(&d.ParameterSize, metaString(meta, "general.size_label"))
	set(&d.Type, metaString(meta, "general.type"))
	if d.Tags == nil {
		d.Tags = metaStrings(meta, "general.tags")
	}
	if d.Languages == nil {
		d.Languages = metaStrings(meta, "general.languages")
	}
	if d.ParameterCount == 0 {
		d.ParameterCount, _ = metaInt(meta, "general.parameter_count")
	}
	if d.Quantization == "" {
		if ft, ok := metaInt(meta, "general.file_type"); ok {
			d.Quantization = ggufFileTypes[ft]
		}
	}
	if d.ContextLength == 0 && d.Architecture != "" {
		d.ContextLength, _ = metaInt(meta, d.Architecture+".context_length")
	}

	d.addLink("Model page", metaString(meta, "general.url"), "general.url")
	d.addLink("Repository", metaString(meta, "general.repo_url"), "general.repo_url")
	d.addLink("Source model", metaString(meta, "general.source.url"), "general.source.url")
	d.addLink("Source repository", metaString(meta, "general.source.repo_url"), "general.source.repo_url")
	d.addLink("Hugging Face", huggingFaceRepo(metaString(meta, "general.source.huggingface.repository")), "general.source.huggingface.repository")
	d.addLink("License", metaString(meta, "general.license.link"), "general.license.link")
	if n, ok := metaInt(meta, "general.base_model.count"); ok {
		for i := int64(0); i < n && i < 8; i++ {
			k := fmt.Sprintf("general.base_model.%d.", i)
			label := "Base model"
			if name := metaString(meta, k+"name"); name != "" {
				label = "Base model: " + name
			}
			d.addLink(label, firstNonEmpty(metaString(meta, k+"url"), metaString(meta, k+"repo_url")), k+"url")
		}
	}
}

// ollamaPage is the ollama.com page of a model pulled from Ollama's registry,
// or the Hugging Face page of one pulled from hf.co. A name with any other
// registry host has no known page.
func ollamaPage(name string) (label, page string) {
	ref := name
	if i := strings.LastIndex(ref, ":"); i > strings.LastIndex(ref, "/") {
		ref = ref[:i]
	}
	parts := strings.Split(ref, "/")
	switch {
	case len(parts) == 1 && parts[0] != "":
		return "Ollama library", "https://ollama.com/library/" + url.PathEscape(parts[0])
	case len(parts) == 2 && !strings.ContainsAny(parts[0], ".:"):
		return "Ollama", "https://ollama.com/" + url.PathEscape(parts[0]) + "/" + url.PathEscape(parts[1])
	case len(parts) == 3 && (parts[0] == "hf.co" || parts[0] == "huggingface.co"):
		return "Hugging Face", huggingFaceRepo(parts[1] + "/" + parts[2])
	case len(parts) >= 2 && parts[0] == "registry.ollama.ai":
		return ollamaPage(strings.Join(parts[1:], "/"))
	}
	return "", ""
}

func (c *specClient) details(ctx context.Context, engine string, base *url.URL, model string) (modelDetails, error) {
	d := modelDetails{Engine: engine, Model: model}
	var err error
	switch engine {
	case "ollama":
		err = c.detailsOllama(ctx, base, &d)
	case "llamacpp":
		err = c.detailsLlamaCPP(ctx, base, &d)
	case "lmstudio":
		err = c.detailsLMStudio(ctx, base, &d)
	default:
		err = fmt.Errorf("unknown engine %q", engine)
	}
	sort.SliceStable(d.Capabilities, func(i, j int) bool { return d.Capabilities[i] < d.Capabilities[j] })
	return d, err
}

func (c *specClient) detailsOllama(ctx context.Context, base *url.URL, d *modelDetails) error {
	var show struct {
		License      string         `json:"license"`
		Parameters   string         `json:"parameters"`
		ModelInfo    map[string]any `json:"model_info"`
		Capabilities []string       `json:"capabilities"`
		ModifiedAt   string         `json:"modified_at"`
		Details      struct {
			Format            string `json:"format"`
			Family            string `json:"family"`
			ParameterSize     string `json:"parameter_size"`
			QuantizationLevel string `json:"quantization_level"`
		} `json:"details"`
	}
	if err := c.do(ctx, http.MethodPost, base, "/api/show", nil, map[string]any{"model": d.Model}, &show); err != nil {
		return err
	}
	d.Format = show.Details.Format
	d.Family = show.Details.Family
	d.ParameterSize = show.Details.ParameterSize
	d.Quantization = show.Details.QuantizationLevel
	d.Capabilities = show.Capabilities
	d.ModifiedAt = show.ModifiedAt
	d.Parameters = strings.TrimSpace(show.Parameters)
	if lic := strings.TrimSpace(show.License); lic != "" {
		if len(lic) > maxLicenseText {
			lic = lic[:maxLicenseText] + "\n…"
		}
		d.LicenseText = lic
	}
	d.applyGeneral(jsonToMeta(show.ModelInfo))

	var tags struct {
		Models []struct {
			Name       string `json:"name"`
			Size       int64  `json:"size"`
			Digest     string `json:"digest"`
			ModifiedAt string `json:"modified_at"`
		} `json:"models"`
	}
	if err := c.do(ctx, http.MethodGet, base, "/api/tags", nil, nil, &tags); err == nil {
		for _, m := range tags.Models {
			if m.Name == d.Model {
				d.SizeBytes, d.Digest = m.Size, m.Digest
				d.ModifiedAt = firstNonEmpty(d.ModifiedAt, m.ModifiedAt)
			}
		}
	}
	if label, page := ollamaPage(d.Model); page != "" {
		// The registry page goes first: it is where the model was pulled from.
		links := d.Links
		d.Links = nil
		d.addLink(label, page, "model name")
		for _, l := range links {
			d.addLink(l.Label, l.URL, l.Source)
		}
	}
	return nil
}

// fileDetails reads a GGUF model file's general metadata into d.
func fileDetails(path string, d *modelDetails) error {
	meta, err := readGGUFFileMetadata(path)
	if err != nil {
		return err
	}
	d.Format = firstNonEmpty(d.Format, "gguf")
	d.applyGeneral(meta)
	return nil
}

func (c *specClient) detailsLlamaCPP(ctx context.Context, base *url.URL, d *modelDetails) error {
	var props struct {
		ModelPath  string          `json:"model_path"`
		Modalities map[string]bool `json:"modalities"`
		Settings   struct {
			NCtx int64 `json:"n_ctx"`
		} `json:"default_generation_settings"`
	}
	if err := c.do(ctx, http.MethodGet, base, "/props", url.Values{"model": {d.Model}}, nil, &props); err != nil {
		return err
	}
	d.LoadedContext = props.Settings.NCtx
	for name, on := range props.Modalities {
		if on {
			d.Capabilities = append(d.Capabilities, name)
		}
	}
	if props.ModelPath == "" {
		d.Notes = append(d.Notes, "llama.cpp did not report the model file; load the model to see what its file says about it")
		return nil
	}
	if err := fileDetails(props.ModelPath, d); err != nil {
		d.Notes = append(d.Notes, "could not read the model file: "+err.Error())
	}
	return nil
}

func (c *specClient) detailsLMStudio(ctx context.Context, base *url.URL, d *modelDetails) error {
	var list struct {
		Models []struct {
			Type             string `json:"type"`
			Key              string `json:"key"`
			Publisher        string `json:"publisher"`
			DisplayName      string `json:"display_name"`
			Description      string `json:"description"`
			Architecture     string `json:"architecture"`
			Path             string `json:"path"`
			Format           string `json:"format"`
			SizeBytes        int64  `json:"size_bytes"`
			ParamsString     string `json:"params_string"`
			MaxContextLength int64  `json:"max_context_length"`
			Quantization     struct {
				Name string `json:"name"`
			} `json:"quantization"`
			Capabilities    json.RawMessage `json:"capabilities"`
			LoadedInstances []struct {
				Config struct {
					ContextLength int64 `json:"context_length"`
				} `json:"config"`
			} `json:"loaded_instances"`
		} `json:"models"`
	}
	if err := c.do(ctx, http.MethodGet, base, "/api/v1/models", nil, nil, &list); err != nil {
		return err
	}
	for _, m := range list.Models {
		if m.Key != d.Model {
			continue
		}
		d.Type = m.Type
		d.Name = m.DisplayName
		d.Description = strings.TrimSpace(m.Description)
		d.Publisher = m.Publisher
		d.Architecture = m.Architecture
		d.Format = m.Format
		d.SizeBytes = m.SizeBytes
		d.ParameterSize = m.ParamsString
		d.ContextLength = m.MaxContextLength
		d.Quantization = m.Quantization.Name
		d.Capabilities = lmStudioCapabilities(m.Capabilities)
		for _, inst := range m.LoadedInstances {
			d.LoadedContext = max64(d.LoadedContext, inst.Config.ContextLength)
		}
		if path := c.lmStudioFile(m.Path); path != "" {
			if err := fileDetails(path, d); err != nil {
				d.Notes = append(d.Notes, "could not read the model file: "+err.Error())
			}
		}
		return nil
	}
	d.Notes = append(d.Notes, "LM Studio does not list this model")
	return nil
}

// lmStudioCapabilities reads LM Studio's capabilities, an object of flags
// ({"vision":true,"trained_for_tool_use":false}) or a list of names.
func lmStudioCapabilities(raw json.RawMessage) []string {
	var flags map[string]bool
	if json.Unmarshal(raw, &flags) == nil {
		var out []string
		for name, on := range flags {
			if !on {
				continue
			}
			if name == "trained_for_tool_use" {
				name = "tools"
			}
			out = append(out, name)
		}
		return out
	}
	var names []string
	if json.Unmarshal(raw, &names) == nil {
		return names
	}
	return nil
}

type modelDetailsRequest struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
	// NodeID names the node whose engine has the model; empty for this node.
	NodeID string `json:"nodeId,omitempty"`
}

// modelDetails runs model/details for a model on this node or on another.
func (p *Proxy) modelDetails(ctx context.Context, raw json.RawMessage) (*modelDetails, error) {
	var req modelDetailsRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, errors.New(`expected {"engine","model","nodeId"?}`)
	}
	if _, ok := profileFor(req.Engine); !ok {
		return nil, fmt.Errorf("unknown engine; use one of %s", engineNames())
	}
	if req.Model == "" {
		return nil, errors.New("model is required")
	}
	var f *facade
	for _, candidate := range p.enabledFacades() {
		if candidate.profile.Name == req.Engine {
			f = candidate
		}
	}
	if f == nil {
		d := modelDetails{Engine: req.Engine, Model: req.Model}
		d.Notes = append(d.Notes, "PAIR is not relaying this engine on this node, so it cannot ask it about models")
		return &d, nil
	}
	if req.NodeID == "" {
		d := f.localModelDetails(ctx, req.Model)
		return &d, nil
	}
	return f.remoteModelDetails(ctx, req.NodeID, req.Model)
}

func (f *facade) localModelDetails(ctx context.Context, model string) modelDetails {
	base, ok := f.localBackendTarget()
	if !ok {
		d := modelDetails{Engine: f.profile.Name, Model: model}
		d.Notes = append(d.Notes, "the engine is not running on this node; start it to see the model's details")
		return d
	}
	d, err := specReader.details(ctx, f.profile.Name, base, model)
	if err != nil {
		slog.Debug("model details failed", "engine", f.profile.Name, "model", model, "err", err)
		d.Notes = append(d.Notes, "could not read the model from the engine: "+err.Error())
	}
	return d
}

// remoteModelDetails asks another node's proxy, over cluster mTLS, to describe
// one of its models.
func (f *facade) remoteModelDetails(ctx context.Context, nodeID, model string) (*modelDetails, error) {
	p := f.host
	p.mesh.Refresh()
	var node *Node
	for _, n := range f.discovery.Nodes() {
		if n.ID == nodeID {
			n := n
			node = &n
		}
	}
	if node == nil {
		return nil, fmt.Errorf("node %q is not known to this engine's relay", nodeID)
	}
	u := f.targetURL(*node)
	if u == nil {
		return nil, fmt.Errorf("node %q advertises no address", nodeID)
	}
	selfPort, aliases := f.selfAddresses()
	if isSelfTarget(u, selfPort) || isAnyAliasSelfTarget(u, aliases) {
		d := f.localModelDetails(ctx, model)
		return &d, nil
	}
	d := modelDetails{Engine: f.profile.Name, Model: model}
	if !p.mesh.HasPin(node.ClusterUUID) {
		d.Notes = append(d.Notes, "this node is not a trusted member of your cluster, so PAIR cannot ask it about its models")
		return &d, nil
	}
	u.Scheme = "https"
	u.Path = peerModelDetailsPath
	u.RawQuery = url.Values{"model": {model}}.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	resp, err := (&http.Client{Transport: p.peerHTTPTransport(node.ClusterUUID), Timeout: peerDetailsTimeout}).Do(req)
	if err != nil {
		slog.Debug("peer model details failed", "node_id", nodeID, "engine", f.profile.Name, "model", model, "err", err)
		// The error names the node's address; the note stays in plain words.
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			d.Notes = append(d.Notes, "that node did not answer in time; it may be offline or unreachable from here")
		} else {
			d.Notes = append(d.Notes, "could not reach that node; it may be offline or unreachable from here")
		}
		return &d, nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") || resp.Header.Get(peerModelDetailsHeader) != "1" {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		// A node running an older PAIR forwards the unknown path to its
		// engine, which answers 404.
		d.Notes = append(d.Notes, fmt.Sprintf("that node did not describe the model (HTTP %d); it may run an older PAIR without model details", resp.StatusCode))
		return &d, nil
	}
	var got modelDetails
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&got); err != nil {
		d.Notes = append(d.Notes, "that node's answer could not be read: "+err.Error())
		return &d, nil
	}
	got.Engine, got.Model = f.profile.Name, model
	return &got, nil
}

// peerModelDetailsHeader marks an answer from a proxy's details route, so a
// JSON answer forwarded from an engine is never mistaken for one.
const peerModelDetailsHeader = "X-PAIR-Model-Details"

// servePeerModelDetails answers another node's model/details question on the
// cluster mTLS ingress.
func (f *facade) servePeerModelDetails(w http.ResponseWriter, r *http.Request) bool {
	if r.URL.Path != peerModelDetailsPath {
		return false
	}
	model := r.URL.Query().Get("model")
	if r.Method != http.MethodGet || model == "" {
		writeIngressError(w, http.StatusBadRequest, "model-details", "expected GET "+peerModelDetailsPath+"?model={name}")
		return true
	}
	ctx, cancel := context.WithTimeout(r.Context(), peerEngineTimeout)
	defer cancel()
	d := f.localModelDetails(ctx, model)
	w.Header().Set(peerModelDetailsHeader, "1")
	writeSessionJSON(w, http.StatusOK, d)
	return true
}
