// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A 30B-class dense model: 48 layers, 8 KV heads of 128: 192 KiB per token.
func denseSpec() modelSpec {
	return modelSpec{Engine: "llamacpp", Model: "m", Source: "metadata", WeightsBytes: 11 * gib,
		MaxContextTokens: 262144, KVBytesPerToken: 48 * 8 * (128 + 128) * 2}
}

func twoCards() []planGPU {
	return []planGPU{{Name: "A", VramBytes: 32 * gib}, {Name: "B", VramBytes: 32 * gib}}
}

func TestPlanTwoCardsFixedWindow(t *testing.T) {
	res, err := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), WindowTokens: 32768}, denseSpec())
	if err != nil {
		t.Fatal(err)
	}
	// 64 GiB − 2×3.2 GiB reserve − 11 GiB = 46.6 GiB; 32K × 192 KiB = 6 GiB.
	if res.ReserveBytes != 2*(32*gib/10) {
		t.Fatalf("reserve %d", res.ReserveBytes)
	}
	if res.Sessions != 7 || res.BytesPerSession != 6*gib || !res.Fits {
		t.Fatalf("sessions %d, per session %d, fits %v", res.Sessions, res.BytesPerSession, res.Fits)
	}
	if res.PoolTokens != 7*32768 {
		t.Fatalf("pool %d", res.PoolTokens)
	}
}

func TestPlanCacheTypeHalvesCost(t *testing.T) {
	f16, _ := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), WindowTokens: 131072}, denseSpec())
	q8, _ := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), WindowTokens: 131072, CacheType: "q8_0"}, denseSpec())
	if f16.Sessions != 1 || q8.Sessions != 3 {
		t.Fatalf("f16 %d sessions, q8_0 %d", f16.Sessions, q8.Sessions)
	}
}

func TestPlanSessionsCapNeverShrinksWindow(t *testing.T) {
	free, err := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), WindowTokens: 32768}, denseSpec())
	if err != nil || free.Sessions < 2 {
		t.Fatalf("uncapped: %v %d sessions", err, free.Sessions)
	}
	capped, err := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), WindowTokens: 32768, MaxSessions: 1}, denseSpec())
	if err != nil || capped.Sessions != 1 || capped.WindowTokens != 32768 {
		t.Fatalf("capped: %v %d sessions of %d", err, capped.Sessions, capped.WindowTokens)
	}
	// A cap above what fits does not add sessions.
	over, _ := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), WindowTokens: 32768, MaxSessions: 200}, denseSpec())
	if over.Sessions != free.Sessions || !over.Fits || over.UsedBytes+over.ReserveBytes > over.TotalBytes {
		t.Fatalf("over cap: %d sessions, want %d", over.Sessions, free.Sessions)
	}
}

func TestPlanDefaultsToFullWindowPerSession(t *testing.T) {
	spec := denseSpec()
	spec.MaxContextTokens = 32768
	res, err := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), MaxSessions: 4}, spec)
	if err != nil || res.WindowTokens != 32768 || res.Sessions != 4 {
		t.Fatalf("%v: %d sessions of %d", err, res.Sessions, res.WindowTokens)
	}
}

func TestPlanCapsAtModelWindow(t *testing.T) {
	spec := denseSpec()
	spec.MaxContextTokens = 32768
	res, _ := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards()}, spec)
	if res.WindowTokens != 32768 {
		t.Fatalf("window %d, want the model's maximum", res.WindowTokens)
	}
	res, _ = computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), WindowTokens: 65536}, spec)
	if res.WindowTokens != 32768 || len(res.Warnings) == 0 {
		t.Fatalf("window %d warnings %v", res.WindowTokens, res.Warnings)
	}
}

func TestPlanUnifiedMemoryKeepsAQuarter(t *testing.T) {
	res, _ := computePlan(planRequest{Engine: "ollama", Model: "m", GPUs: []planGPU{{VramBytes: 64 * gib}}, Unified: true, WindowTokens: 8192}, denseSpec())
	if res.ReserveBytes != 16*gib {
		t.Fatalf("reserve %d, want a quarter", res.ReserveBytes)
	}
}

func TestPlanModelTooLarge(t *testing.T) {
	spec := denseSpec()
	spec.WeightsBytes = 70 * gib
	res, err := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards()}, spec)
	if err != nil || res.Sessions != 0 || res.Fits || len(res.Warnings) == 0 {
		t.Fatalf("err %v sessions %d fits %v", err, res.Sessions, res.Fits)
	}
}

func TestPlanSlidingWindowCountsWindowOnly(t *testing.T) {
	spec := modelSpec{Source: "metadata", WeightsBytes: gib, MaxContextTokens: 131072,
		KVBytesPerToken: 1000, SlidingBytesPerToken: 5000, SlidingWindow: 1024}
	if b := sessionBytes(spec, 1000, 1, 131072); b != 131072*1000+1024*5000 {
		t.Fatalf("session bytes %d", b)
	}
	res, _ := computePlan(planRequest{GPUs: twoCards(), MaxSessions: 2}, spec)
	if !res.Fits || res.UsedBytes+res.ReserveBytes > res.TotalBytes {
		t.Fatalf("sliding plan overflows: %+v", res)
	}
}

func TestPlanManualValuesAndErrors(t *testing.T) {
	res, err := computePlan(planRequest{GPUs: twoCards(), WindowTokens: 8192, WeightsBytes: gib, KVBytesPerToken: 100 << 10}, modelSpec{})
	if err != nil || res.Spec.Source != "manual" || res.Sessions == 0 {
		t.Fatalf("manual plan: %v %+v", err, res)
	}
	for _, req := range []planRequest{
		{GPUs: twoCards(), MaxSessions: maxPlannedSessions + 1},
		{GPUs: twoCards(), WindowTokens: -1},
		{GPUs: twoCards(), CacheType: "q2"},
		{GPUs: nil},
		{GPUs: twoCards(), ReservePercent: 95},
	} {
		if _, err := computePlan(req, denseSpec()); err == nil {
			t.Fatalf("accepted %+v", req)
		}
	}
	if _, err := computePlan(planRequest{GPUs: twoCards()}, modelSpec{WeightsBytes: gib}); err == nil {
		t.Fatal("planned without a context cost")
	}
}

// ggufFile builds a GGUF v3 header with the given metadata.
func ggufFile(t *testing.T, kv []any) []byte {
	t.Helper()
	var b bytes.Buffer
	w := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(0))
	w(uint64(len(kv) / 2))
	for i := 0; i < len(kv); i += 2 {
		str(kv[i].(string))
		switch v := kv[i+1].(type) {
		case string:
			w(uint32(8))
			str(v)
		case uint32:
			w(uint32(4))
			w(v)
		case []int32:
			w(uint32(9))
			w(uint32(5))
			w(uint64(len(v)))
			for _, e := range v {
				w(e)
			}
		case []string:
			w(uint32(9))
			w(uint32(8))
			w(uint64(len(v)))
			for _, e := range v {
				str(e)
			}
		case float32:
			w(uint32(6))
			w(v)
		}
	}
	return b.Bytes()
}

func TestGGUFDenseModel(t *testing.T) {
	data := ggufFile(t, []any{
		"general.architecture", "llama",
		"tokenizer.ggml.tokens", []string{"a", "b", "c"},
		"llama.block_count", uint32(32),
		"llama.context_length", uint32(131072),
		"llama.attention.head_count", uint32(32),
		"llama.attention.head_count_kv", uint32(8),
		"llama.embedding_length", uint32(4096),
		"llama.rope.freq_base", float32(500000),
	})
	meta, err := readGGUFMetadata(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	var s modelSpec
	if !s.applyArchitecture(meta) {
		t.Fatal("architecture not understood")
	}
	// 32 layers × 8 heads × (128+128) × 2 bytes = 128 KiB per token.
	if s.KVBytesPerToken != 128<<10 || s.KeyLength != 128 || s.MaxContextTokens != 131072 {
		t.Fatalf("%+v", s)
	}
}

func TestGGUFHybridAndPerLayerHeads(t *testing.T) {
	heads := make([]int32, 8)
	for i := range heads {
		if i%4 == 3 {
			heads[i] = 4
		}
	}
	data := ggufFile(t, []any{
		"general.architecture", "qwen3next",
		"qwen3next.block_count", uint32(8),
		"qwen3next.attention.head_count", uint32(16),
		"qwen3next.attention.head_count_kv", heads,
		"qwen3next.attention.key_length", uint32(256),
		"qwen3next.attention.value_length", uint32(256),
		"qwen3next.ssm.state_size", uint32(128),
		"qwen3next.ssm.inner_size", uint32(4096),
		"qwen3next.ssm.conv_kernel", uint32(4),
		"qwen3next.ssm.group_count", uint32(16),
	})
	meta, err := readGGUFMetadata(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	var s modelSpec
	if !s.applyArchitecture(meta) {
		t.Fatal("architecture not understood")
	}
	if s.AttentionLayers != 2 || s.KVBytesPerToken != 2*4*512*2 {
		t.Fatalf("attention layers %d, per token %d", s.AttentionLayers, s.KVBytesPerToken)
	}
	if want := int64(6) * (4096*128 + 3*(4096+2*16*128)) * 4; s.RecurrentBytesPerSession != want {
		t.Fatalf("recurrent %d want %d", s.RecurrentBytesPerSession, want)
	}
}

func TestGGUFSlidingWindowPattern(t *testing.T) {
	meta := map[string]ggufValue{
		"general.architecture":            "gemma3",
		"gemma3.block_count":              int64(12),
		"gemma3.attention.head_count":     int64(4),
		"gemma3.attention.head_count_kv":  int64(1),
		"gemma3.attention.key_length":     int64(256),
		"gemma3.attention.value_length":   int64(256),
		"gemma3.attention.sliding_window": int64(512),
	}
	var s modelSpec
	s.applyArchitecture(meta)
	// Two of twelve layers are global.
	if s.KVBytesPerToken != 2*1024 || s.SlidingBytesPerToken != 10*1024 || s.SlidingWindow != 512 {
		t.Fatalf("%+v", s)
	}
}

func TestGGUFRejectsGarbage(t *testing.T) {
	if _, err := readGGUFMetadata(strings.NewReader("NOPE")); err == nil {
		t.Fatal("accepted a non-GGUF file")
	}
	data := ggufFile(t, []any{"general.architecture", "llama"})
	if _, err := readGGUFMetadata(bytes.NewReader(data[:len(data)-3])); err == nil {
		t.Fatal("accepted a truncated header")
	}
}

func TestDescribeOllamaMetadataAndMeasured(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct{ Model string }
		json.NewDecoder(r.Body).Decode(&body)
		switch r.URL.Path {
		case "/api/show":
			if body.Model == "dense" {
				json.NewEncoder(w).Encode(map[string]any{"details": map[string]any{"format": "gguf"}, "model_info": map[string]any{
					"general.architecture": "qwen3", "qwen3.block_count": 28, "qwen3.context_length": 40960,
					"qwen3.attention.head_count": 16, "qwen3.attention.head_count_kv": 8,
					"qwen3.attention.key_length": 128, "qwen3.attention.value_length": 128}})
				return
			}
			json.NewEncoder(w).Encode(map[string]any{"details": map[string]any{"format": "safetensors"}, "model_info": map[string]any{
				"general.architecture": "qwen3_5", "qwen3_5.block_count": 64}})
		case "/api/tags":
			json.NewEncoder(w).Encode(map[string]any{"models": []any{
				map[string]any{"name": "dense", "size": 500 << 20},
				map[string]any{"name": "mlx", "size": 10 * gib}}})
		case "/api/ps":
			json.NewEncoder(w).Encode(map[string]any{"models": []any{
				map[string]any{"name": "mlx", "size": 10*gib + 4096*100000, "context_length": 4096}}})
		}
	}))
	defer srv.Close()
	base, _ := url.Parse(srv.URL)
	c := &specClient{client: srv.Client(), homeDir: os.UserHomeDir}

	s, err := c.describe(context.Background(), "ollama", base, "dense", 0)
	if err != nil || s.Source != "metadata" || s.KVBytesPerToken != 28*8*256*2 || s.WeightsBytes != 500<<20 {
		t.Fatalf("%v %+v", err, s)
	}
	s, err = c.describe(context.Background(), "ollama", base, "mlx", 2)
	if err != nil || s.Source != "measured" || s.KVBytesPerToken != 100000/2 {
		t.Fatalf("%v %+v", err, s)
	}
}

func TestDescribeLlamaCPPReadsShards(t *testing.T) {
	dir := t.TempDir()
	head := ggufFile(t, []any{"general.architecture", "llama", "llama.block_count", uint32(2),
		"llama.attention.head_count", uint32(4), "llama.attention.head_count_kv", uint32(2), "llama.embedding_length", uint32(512)})
	first := filepath.Join(dir, "m-00001-of-00002.gguf")
	os.WriteFile(first, append(head, make([]byte, 1000)...), 0o600)
	os.WriteFile(filepath.Join(dir, "m-00002-of-00002.gguf"), make([]byte, 5000), 0o600)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"model_path": first})
	}))
	defer srv.Close()
	base, _ := url.Parse(srv.URL)
	s, err := (&specClient{client: srv.Client()}).describe(context.Background(), "llamacpp", base, "m", 0)
	if err != nil || s.Source != "metadata" || s.WeightsBytes != int64(len(head))+1000+5000 {
		t.Fatalf("%v %+v", err, s)
	}
}

func TestPlannerStoreLifecycle(t *testing.T) {
	s := newPlannerStore()
	s.path = filepath.Join(t.TempDir(), plannerFileName)
	if !s.snapshot().AutoReload {
		t.Fatal("auto reload should default on")
	}
	plan := savedPlan{Engine: "llamacpp", Model: "m", WindowTokens: 32768, Sessions: 4}
	if err := s.put(plan); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := s.markApplied("llamacpp", "m", 65536, 4, "f16", false); ok {
		t.Fatal("applied a plan that changed meanwhile")
	}
	if p, ok, err := s.markApplied("llamacpp", "m", 32768, 4, "f16", false); !ok || err != nil || !p.Applied {
		t.Fatalf("mark applied: %v %v", ok, err)
	}
	if err := s.put(plan); err != nil || !s.snapshot().Plans[0].Applied {
		t.Fatal("re-saving the same plan cleared applied")
	}
	plan.Sessions = 2
	s.put(plan)
	if s.snapshot().Plans[0].Applied {
		t.Fatal("a changed plan stayed applied")
	}
	// Ollama's settings are engine-wide: one plan at a time.
	s.put(savedPlan{Engine: "ollama", Model: "a", WindowTokens: 8192, Sessions: 2})
	s.put(savedPlan{Engine: "ollama", Model: "b", WindowTokens: 8192, Sessions: 2})
	n := 0
	for _, p := range s.snapshot().Plans {
		if p.Engine == "ollama" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("%d ollama plans", n)
	}
	s.setAutoReload(false)

	reloaded := newPlannerStore()
	reloaded.path = s.path
	if err := reloaded.load(); err != nil {
		t.Fatal(err)
	}
	if st := reloaded.snapshot(); st.AutoReload || len(st.Plans) != 2 {
		t.Fatalf("reloaded %+v", st)
	}
	if err := s.put(savedPlan{Engine: "nope", Model: "m", WindowTokens: 8192, Sessions: 1}); err == nil {
		t.Fatal("accepted an unknown engine")
	}
}

func TestPlannerAppliedSetsCapacity(t *testing.T) {
	p := NewProxy(nil)
	p.planner.put(savedPlan{Engine: "llamacpp", Model: "m", WindowTokens: 16384, Sessions: 3})
	raw, _ := json.Marshal(map[string]any{"applied": map[string]any{"engine": "llamacpp", "model": "m", "windowTokens": 16384, "sessions": 3}})
	if _, err := p.setPlanner(context.Background(), raw); err != nil {
		t.Fatal(err)
	}
	got := p.capacity.settingsSnapshot()[capacityKey{Engine: "llamacpp", Model: "m"}]
	if got.ContextTokens != 16384*3 || got.MaxSessions != 3 {
		t.Fatalf("capacity %+v", got)
	}
	if a := p.nodeActivity(); !a.Idle {
		t.Fatalf("fresh proxy not idle: %+v", a)
	}
}

func TestPlanReplyBudgetDefaultAndChange(t *testing.T) {
	if got := (savedPlan{WindowTokens: 40960}).replyBudget(); got != 8192 {
		t.Fatalf("default reply budget %d, want 8192", got)
	}
	if got := (savedPlan{WindowTokens: 4096}).replyBudget(); got != 2048 {
		t.Fatalf("small window reply budget %d, want half the window", got)
	}
	if (savedPlan{Engine: "llamacpp", Model: "m", WindowTokens: 8192, Sessions: 1, CacheType: "f16", ReplyTokens: 8192}).validate() == nil {
		t.Fatal("a reply budget as large as the window was accepted")
	}

	p := NewProxy(nil)
	p.planner.put(savedPlan{Engine: "llamacpp", Model: "m", WindowTokens: 16384, Sessions: 2})
	raw, _ := json.Marshal(map[string]any{"applied": map[string]any{"engine": "llamacpp", "model": "m", "windowTokens": 16384, "sessions": 2}})
	if _, err := p.setPlanner(context.Background(), raw); err != nil {
		t.Fatal(err)
	}
	k := capacityKey{Engine: "llamacpp", Model: "m"}
	if got := p.capacity.settingsSnapshot()[k].ReplyTokens; got != 8192 {
		t.Fatalf("applied reply budget %d, want the 8192 default", got)
	}
	// A new reply budget alone keeps the plan applied and takes effect at once.
	raw, _ = json.Marshal(map[string]any{"save": map[string]any{"engine": "llamacpp", "model": "m", "windowTokens": 16384, "sessions": 2, "cacheType": "f16", "replyTokens": 4096}})
	if _, err := p.setPlanner(context.Background(), raw); err != nil {
		t.Fatal(err)
	}
	if plan, _ := p.planner.find("llamacpp", "m"); !plan.Applied {
		t.Fatal("changing only the reply budget made the plan pending")
	}
	if got := p.capacity.settingsSnapshot()[k].ReplyTokens; got != 4096 {
		t.Fatalf("reply budget %d after the change, want 4096", got)
	}
}
