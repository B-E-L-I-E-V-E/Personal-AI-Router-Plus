// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Coverage for model capabilities: what each engine's answer maps to, how
// placements combine, the router's capability endpoint and model list, and
// the cache behind them.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Recorded from LM Studio's /api/v1/models on a live node, trimmed to the
// fields PAIR reads. GLM reports reasoning as an object, not a boolean.
const lmStudioModelsLive = `{"models":[
	{"type":"llm","publisher":"lmstudio-community","key":"glm-4.7-flash","display_name":"GLM 4.7 Flash",
	 "architecture":"deepseek2","format":"gguf","params_string":"30B","max_context_length":202752,
	 "capabilities":{"vision":false,"trained_for_tool_use":true,"reasoning":{"allowed_options":["off","on"],"default":"on"}},
	 "loaded_instances":[{"id":"glm-4.7-flash","config":{"context_length":8192,"parallel":4}}]},
	{"type":"llm","publisher":"mistralai","key":"mistralai/devstral-small-2-2512","display_name":"Devstral Small 2 2512",
	 "architecture":"mistral3","format":"gguf","max_context_length":393216,
	 "capabilities":{"vision":true,"trained_for_tool_use":true},"loaded_instances":[]},
	{"type":"embedding","publisher":"nomic-ai","key":"text-embedding-nomic-embed-text-v1.5","display_name":"Nomic Embed Text v1.5",
	 "architecture":"nomic-bert","format":"gguf","max_context_length":2048,"loaded_instances":[]}]}`

func TestLMStudioCapabilities_ReadsObjectFlagsAsOn(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []string
	}{
		{"boolean flags", `{"vision":true,"trained_for_tool_use":false}`, []string{"vision"}},
		{"reasoning object", `{"vision":false,"trained_for_tool_use":true,"reasoning":{"allowed_options":["off","on"],"default":"on"}}`, []string{"reasoning", "tools"}},
		{"reasoning that can only be off", `{"reasoning":{"allowed_options":["off"],"default":"off"}}`, nil},
		{"null flag", `{"vision":null}`, nil},
		{"list of names", `["vision","tools"]`, []string{"vision", "tools"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := lmStudioCapabilities(json.RawMessage(tc.raw))
			slices.Sort(got)
			want := tc.want
			slices.Sort(want)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("capabilities = %v, want %v", got, want)
			}
		})
	}
}

// reportFor describes model on one placement of engine through the engine's
// details reader, and combines it into a report.
func reportFor(t *testing.T, engine string, base *url.URL, model string) modelCapabilitiesReport {
	t.Helper()
	d, err := remoteSpecReader.details(context.Background(), engine, base, model)
	if err != nil {
		t.Fatalf("details: %v", err)
	}
	return aggregateCapabilities(model, []placementDetails{{ref: testPlacement(engine, "node-a", model), details: d, ok: true}})
}

func testPlacement(engine, nodeID, model string) placementRef {
	profile, _ := profileFor(engine)
	return placementRef{cand: candidate{id: nodeID, fac: &facade{profile: profile}}, nodeName: nodeID + ".local.", model: model}
}

func allEngine(keys ...string) map[string]string {
	out := map[string]string{
		"chat": sourceEngine, "structuredOutput": sourceEstimated, "tools": sourceEngine, "vision": sourceEngine,
		"embedding": sourceEngine, "reasoning": sourceEngine, "contextWindow": sourceEngine,
		"maxContextWindow": sourceEngine, "maxOutputTokens": sourceEstimated, "recommendedOutputTokens": sourceEstimated,
		"maxInputTokens": sourceEstimated,
	}
	for _, k := range keys {
		out[k] = sourceEstimated
	}
	return out
}

func TestModelCapabilities_LMStudioMapping(t *testing.T) {
	base := engineFixture(t, map[string]string{"/api/v1/models": lmStudioModelsLive})
	cases := []struct {
		name        string
		model       string
		wantCaps    modelCapabilities
		wantLimits  modelLimits
		wantLoaded  bool
		wantSources map[string]string
	}{
		{
			name:        "loaded reasoning model",
			model:       "glm-4.7-flash",
			wantCaps:    modelCapabilities{Chat: true, StructuredOutput: true, Tools: true, Reasoning: true},
			wantLimits:  modelLimits{ContextWindow: 8192, MaxContextWindow: 202752, RecommendedOutputTokens: 4096, MaxInputTokens: 4096},
			wantLoaded:  true,
			wantSources: allEngine(),
		},
		{
			// Not loaded: the window it would get is LM Studio's default.
			name:        "unloaded vision model",
			model:       "mistralai/devstral-small-2-2512",
			wantCaps:    modelCapabilities{Chat: true, StructuredOutput: true, Tools: true, Vision: true},
			wantLimits:  modelLimits{ContextWindow: 4096, MaxContextWindow: 393216, RecommendedOutputTokens: 1024, MaxInputTokens: 3072},
			wantSources: allEngine("contextWindow"),
		},
		{
			name:        "embedding model",
			model:       "text-embedding-nomic-embed-text-v1.5",
			wantCaps:    modelCapabilities{Embedding: true},
			wantLimits:  modelLimits{ContextWindow: 2048, MaxContextWindow: 2048, RecommendedOutputTokens: 1024, MaxInputTokens: 1024},
			wantSources: allEngine("contextWindow"),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := reportFor(t, "lmstudio", base, tc.model)
			if got.Capabilities != tc.wantCaps {
				t.Errorf("capabilities = %+v, want %+v", got.Capabilities, tc.wantCaps)
			}
			if !reflect.DeepEqual(got.Limits, tc.wantLimits) {
				t.Errorf("limits = %+v, want %+v", got.Limits, tc.wantLimits)
			}
			if got.Placements[0].Loaded != tc.wantLoaded {
				t.Errorf("loaded = %v, want %v", got.Placements[0].Loaded, tc.wantLoaded)
			}
			if !reflect.DeepEqual(got.Sources, tc.wantSources) {
				t.Errorf("sources = %v, want %v", got.Sources, tc.wantSources)
			}
		})
	}
}

func TestModelCapabilities_OllamaMapping(t *testing.T) {
	int64p := func(v int64) *int64 { return &v }
	cases := []struct {
		name        string
		model       string
		show        string
		ps          string
		wantCaps    modelCapabilities
		wantLimits  modelLimits
		wantSources map[string]string
	}{
		{
			name:        "loaded thinking model",
			model:       "qwen3:0.6b",
			show:        ollamaShowQwen3,
			ps:          `{"models":[{"name":"qwen3:0.6b","model":"qwen3:0.6b","context_length":16384}]}`,
			wantCaps:    modelCapabilities{Chat: true, StructuredOutput: true, Tools: true, Reasoning: true},
			wantLimits:  modelLimits{ContextWindow: 16384, MaxContextWindow: 40960, RecommendedOutputTokens: 4096, MaxInputTokens: 12288},
			wantSources: allEngine(),
		},
		{
			// Recorded from a live Ollama: a deepseek-r1 pulled before Ollama
			// reported thinking lists only completion. It still reasons.
			name:  "reasoning model the engine is silent about",
			model: "deepseek-r1:latest",
			show: `{"parameters":"stop \"<｜end▁of▁sentence｜>\"\nnum_ctx 32768\nnum_predict 2048",
				"model_info":{"general.architecture":"qwen2","qwen2.context_length":131072},"capabilities":["completion"]}`,
			ps:          `{"models":[]}`,
			wantCaps:    modelCapabilities{Chat: true, StructuredOutput: true, Reasoning: true},
			wantLimits:  modelLimits{ContextWindow: 32768, MaxContextWindow: 131072, MaxOutputTokens: int64p(2048), RecommendedOutputTokens: 2048, MaxInputTokens: 30720},
			wantSources: map[string]string{"chat": sourceEngine, "structuredOutput": sourceEstimated, "tools": sourceEngine, "vision": sourceEngine, "embedding": sourceEngine, "reasoning": sourceEstimated, "contextWindow": sourceEngine, "maxContextWindow": sourceEngine, "maxOutputTokens": sourceEngine, "recommendedOutputTokens": sourceEstimated, "maxInputTokens": sourceEstimated},
		},
		{
			name:        "embedding model",
			model:       "nomic-embed-text:latest",
			show:        `{"model_info":{"general.architecture":"nomic-bert","nomic-bert.context_length":2048},"capabilities":["embedding"]}`,
			ps:          `{"models":[]}`,
			wantCaps:    modelCapabilities{Embedding: true},
			wantLimits:  modelLimits{ContextWindow: 2048, MaxContextWindow: 2048, RecommendedOutputTokens: 1024, MaxInputTokens: 1024},
			wantSources: allEngine("contextWindow"),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := engineFixture(t, map[string]string{"/api/show": tc.show, "/api/ps": tc.ps, "/api/tags": `{"models":[]}`})
			got := reportFor(t, "ollama", base, tc.model)
			if got.Capabilities != tc.wantCaps {
				t.Errorf("capabilities = %+v, want %+v", got.Capabilities, tc.wantCaps)
			}
			if !reflect.DeepEqual(got.Limits, tc.wantLimits) {
				t.Errorf("limits = %+v, want %+v", got.Limits, tc.wantLimits)
			}
			if !reflect.DeepEqual(got.Sources, tc.wantSources) {
				t.Errorf("sources = %v, want %v", got.Sources, tc.wantSources)
			}
		})
	}
}

func TestModelCapabilities_LlamaCPPMapping(t *testing.T) {
	props := `{"model_path":"/models/qwq.gguf","modalities":{"vision":false,"audio":false},
		"chat_template":"{{ '<think>\n' }}{% for m in messages %}{{ m.content }}{% endfor %}",
		"chat_template_caps":{"supports_tools":true},"default_generation_settings":{"n_ctx":65536}}`
	base := engineFixture(t, map[string]string{"/props?model=qwq-32b": props})

	got := reportFor(t, "llamacpp", base, "qwq-32b")

	wantCaps := modelCapabilities{Chat: true, StructuredOutput: true, Tools: true, Reasoning: true}
	if got.Capabilities != wantCaps {
		t.Errorf("capabilities = %+v, want %+v", got.Capabilities, wantCaps)
	}
	// llama.cpp reports the loaded window but not the trained one, and only
	// the template says the model thinks.
	wantLimits := modelLimits{ContextWindow: 65536, MaxContextWindow: 65536, RecommendedOutputTokens: 16384, MaxInputTokens: 49152}
	if !reflect.DeepEqual(got.Limits, wantLimits) {
		t.Errorf("limits = %+v, want %+v", got.Limits, wantLimits)
	}
	wantSources := allEngine("chat", "embedding", "reasoning", "maxContextWindow")
	if !reflect.DeepEqual(got.Sources, wantSources) {
		t.Errorf("sources = %v, want %v", got.Sources, wantSources)
	}
}

func TestAggregateCapabilities_HoldsWhereverTheRequestLands(t *testing.T) {
	loaded := modelDetails{Type: "llm", Capabilities: []string{"vision", "tools"}, CapabilitiesComplete: true, ContextLength: 131072, LoadedContext: 32768}
	smaller := modelDetails{Type: "llm", Capabilities: []string{"tools", "reasoning"}, CapabilitiesComplete: true, ContextLength: 65536, LoadedContext: 8192}

	got := aggregateCapabilities("shared", []placementDetails{
		{ref: testPlacement("lmstudio", "node-a", "shared"), details: loaded, ok: true},
		{ref: testPlacement("lmstudio", "node-b", "shared"), details: smaller, ok: true},
	})

	// Vision on only one node cannot be relied on; reasoning on one node has
	// to be budgeted for.
	wantCaps := modelCapabilities{Chat: true, StructuredOutput: true, Tools: true, Reasoning: true}
	if got.Capabilities != wantCaps {
		t.Errorf("capabilities = %+v, want %+v", got.Capabilities, wantCaps)
	}
	wantLimits := modelLimits{ContextWindow: 8192, MaxContextWindow: 131072, RecommendedOutputTokens: 4096, MaxInputTokens: 4096}
	if !reflect.DeepEqual(got.Limits, wantLimits) {
		t.Errorf("limits = %+v, want %+v", got.Limits, wantLimits)
	}
	wantPlacements := []capabilityPlacement{
		{Node: capabilityNode{ID: "node-a", Name: "node-a.local."}, Engine: "lmstudio", Loaded: true, ContextWindow: 32768, MaxContextWindow: 131072, DetailsAvailable: true},
		{Node: capabilityNode{ID: "node-b", Name: "node-b.local."}, Engine: "lmstudio", Loaded: true, ContextWindow: 8192, MaxContextWindow: 65536, DetailsAvailable: true},
	}
	if !reflect.DeepEqual(got.Placements, wantPlacements) {
		t.Errorf("placements = %+v, want %+v", got.Placements, wantPlacements)
	}
}

func TestAggregateCapabilities_UnavailablePeerMakesValuesEstimated(t *testing.T) {
	known := modelDetails{Type: "llm", Capabilities: []string{"tools", "reasoning"}, CapabilitiesComplete: true, ContextLength: 202752, LoadedContext: 16384}

	got := aggregateCapabilities("glm-4.7-flash", []placementDetails{
		{ref: testPlacement("lmstudio", "node-a", "glm-4.7-flash"), details: known, ok: true},
		{ref: testPlacement("lmstudio", "node-b", "glm-4.7-flash"), ok: false},
	})

	if got.Placements[1].DetailsAvailable || got.Placements[1].Loaded {
		t.Errorf("unavailable placement = %+v, want details unavailable and not loaded", got.Placements[1])
	}
	// node-b's window is a guess, and it is the smaller one.
	if got.Limits.ContextWindow != estimatedLoadContext || got.Sources["contextWindow"] != sourceEstimated {
		t.Errorf("contextWindow = %d (%s), want %d (estimated)", got.Limits.ContextWindow, got.Sources["contextWindow"], estimatedLoadContext)
	}
	// node-a's engine said the model reasons, which holds whatever node-b is.
	if !got.Capabilities.Reasoning || got.Sources["reasoning"] != sourceEngine {
		t.Errorf("reasoning = %v (%s), want true from the engine", got.Capabilities.Reasoning, got.Sources["reasoning"])
	}
	// node-a's engine said tools; node-b could not say, so it is estimated.
	if got.Sources["tools"] != sourceEstimated || got.Sources["chat"] != sourceEstimated {
		t.Errorf("sources = %v, want tools and chat estimated", got.Sources)
	}
	if got.Limits.MaxContextWindow != 202752 || got.Sources["maxContextWindow"] != sourceEngine {
		t.Errorf("maxContextWindow = %d (%s), want 202752 from the engine", got.Limits.MaxContextWindow, got.Sources["maxContextWindow"])
	}
}

func TestAggregateCapabilities_EstimateStaysWithinTheReportedMaximum(t *testing.T) {
	// Seen live: an embedding model whose engine reports a 2048 maximum on
	// one node, and a second node that could not be asked.
	known := modelDetails{Type: "embedding", CapabilitiesComplete: true, ContextLength: 2048}

	got := aggregateCapabilities("text-embedding-nomic-embed-text-v1.5", []placementDetails{
		{ref: testPlacement("lmstudio", "node-a", "text-embedding-nomic-embed-text-v1.5"), details: known, ok: true},
		{ref: testPlacement("lmstudio", "node-b", "text-embedding-nomic-embed-text-v1.5"), ok: false},
	})

	if got.Limits.MaxContextWindow != 2048 || got.Sources["maxContextWindow"] != sourceEngine {
		t.Errorf("maxContextWindow = %d (%s), want 2048 from the engine", got.Limits.MaxContextWindow, got.Sources["maxContextWindow"])
	}
	if got.Limits.ContextWindow != 2048 {
		t.Errorf("contextWindow = %d, want 2048", got.Limits.ContextWindow)
	}
	if p := got.Placements[1]; p.ContextWindow != 2048 || p.MaxContextWindow != 2048 {
		t.Errorf("unavailable placement = %+v, want its estimates capped at 2048", p)
	}
}

func TestAggregateCapabilities_UnavailablePeerStillReportsReasoningByName(t *testing.T) {
	got := aggregateCapabilities("glm-4.7-flash", []placementDetails{{ref: testPlacement("lmstudio", "node-b", "glm-4.7-flash"), ok: false}})

	if !got.Capabilities.Reasoning || got.Sources["reasoning"] != sourceEstimated {
		t.Fatalf("reasoning = %v (%s), want true, estimated from the model family", got.Capabilities.Reasoning, got.Sources["reasoning"])
	}
	if got.Limits.RecommendedOutputTokens < 2048 {
		t.Fatalf("recommendedOutputTokens = %d, want room for thinking", got.Limits.RecommendedOutputTokens)
	}
}

func TestRecommendedOutput_LeavesRoomForThinking(t *testing.T) {
	int64p := func(v int64) *int64 { return &v }
	cases := []struct {
		name      string
		window    int64
		reasoning bool
		maxOutput *int64
		want      int64
	}{
		{"reasoning model in a large window", 202752, true, nil, 32768},
		{"reasoning model in an 8K window", 8192, true, nil, 4096},
		{"reasoning model in a 4K window keeps half for the prompt", 4096, true, nil, 2048},
		{"plain model in a large window", 131072, false, nil, 8192},
		{"plain model in a 4K window", 4096, false, nil, 1024},
		{"engine cap wins", 131072, true, int64p(1000), 1000},
		{"unknown window, reasoning", 0, true, nil, 8192},
		{"unknown window, plain", 0, false, nil, 2048},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := recommendedOutput(tc.window, tc.reasoning, tc.maxOutput); got != tc.want {
				t.Fatalf("recommendedOutput = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestFetchPeerModelDetails_TellsAnOlderPeerFromAnAnswer(t *testing.T) {
	answer := modelDetails{Type: "llm", ContextLength: 8192, CapabilitiesComplete: true}
	cases := []struct {
		name    string
		handler http.HandlerFunc
		wantErr bool
	}{
		{
			name: "peer with the details route",
			handler: func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != peerModelDetailsPath || r.URL.Query().Get("model") != "glm-4.7-flash" {
					http.NotFound(w, r)
					return
				}
				w.Header().Set(peerModelDetailsHeader, "1")
				writeSessionJSON(w, http.StatusOK, answer)
			},
		},
		{
			// An older PAIR forwards the unknown path to its engine.
			name:    "older peer",
			handler: func(w http.ResponseWriter, r *http.Request) { http.NotFound(w, r) },
			wantErr: true,
		},
		{
			name: "engine answers JSON without the marker",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"type":"llm"}`))
			},
			wantErr: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(tc.handler)
			t.Cleanup(srv.Close)
			base, _ := url.Parse(srv.URL)

			got, err := fetchPeerModelDetails(context.Background(), srv.Client(), base, "glm-4.7-flash")

			if tc.wantErr {
				var older errPeerNoDetails
				if !errors.As(err, &older) {
					t.Fatalf("err = %v, want errPeerNoDetails", err)
				}
				return
			}
			if err != nil || got.ContextLength != 8192 || !got.CapabilitiesComplete {
				t.Fatalf("got %+v, %v; want the peer's answer", got, err)
			}
		})
	}
}

// countingFetch answers every placement with d, or fails while fail is set,
// counting the calls.
type countingFetch struct {
	calls atomic.Int32
	mu    sync.Mutex
	d     modelDetails
	fail  bool
}

func (c *countingFetch) fetch(context.Context, placementRef) (placementInfo, error) {
	c.calls.Add(1)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.fail {
		return placementInfo{}, errors.New("unreachable")
	}
	return placementInfo{details: c.d}, nil
}

// waitCalls waits for the background refresh to have asked n times.
func waitCalls(t *testing.T, f *countingFetch, n int32) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for f.calls.Load() < n {
		if time.Now().After(deadline) {
			t.Fatalf("fetch calls = %d, want %d", f.calls.Load(), n)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestCapabilityCache_AsksOncePerPlacementUntilStale(t *testing.T) {
	f := &countingFetch{d: modelDetails{LoadedContext: 8192}}
	c := newCapabilityCache(f.fetch)
	now := time.Unix(1_000_000, 0)
	var nowMu sync.Mutex
	c.now = func() time.Time { nowMu.Lock(); defer nowMu.Unlock(); return now }
	advance := func(d time.Duration) { nowMu.Lock(); now = now.Add(d); nowMu.Unlock() }
	ref := testPlacement("lmstudio", "node-a", "glm-4.7-flash")
	ref.inventory = inventoryFingerprint([]string{"glm-4.7-flash"})

	first := c.lookup(context.Background(), []placementRef{ref}, time.Second)
	second := c.lookup(context.Background(), []placementRef{ref}, time.Second)
	if !first[0].ok || !second[0].ok || f.calls.Load() != 1 {
		t.Fatalf("ok = %v/%v, calls = %d; want both answered from one fetch", first[0].ok, second[0].ok, f.calls.Load())
	}

	// Stale: the old answer is served at once and refreshed behind it.
	f.mu.Lock()
	f.d = modelDetails{LoadedContext: 32768}
	f.mu.Unlock()
	advance(capabilityTTL)
	stale := c.lookup(context.Background(), []placementRef{ref}, time.Second)
	if stale[0].details.LoadedContext != 8192 {
		t.Fatalf("stale lookup = %d, want the cached 8192", stale[0].details.LoadedContext)
	}
	waitCalls(t, f, 2)
	deadline := time.Now().Add(2 * time.Second)
	for c.lookup(context.Background(), []placementRef{ref}, time.Second)[0].details.LoadedContext != 32768 {
		if time.Now().After(deadline) {
			t.Fatal("refreshed answer never replaced the stale one")
		}
		time.Sleep(time.Millisecond)
	}

	// The node's inventory changed: its placement is asked again at once.
	moved := ref
	moved.inventory = inventoryFingerprint([]string{"glm-4.7-flash", "qwen3.8-27b"})
	c.lookup(context.Background(), []placementRef{moved}, time.Second)
	if got := f.calls.Load(); got != 3 {
		t.Fatalf("calls after an inventory change = %d, want 3", got)
	}
}

func TestCapabilityCache_RetriesAFailureAndKeepsAGoodAnswer(t *testing.T) {
	f := &countingFetch{fail: true}
	c := newCapabilityCache(f.fetch)
	now := time.Unix(1_000_000, 0)
	var nowMu sync.Mutex
	c.now = func() time.Time { nowMu.Lock(); defer nowMu.Unlock(); return now }
	advance := func(d time.Duration) { nowMu.Lock(); now = now.Add(d); nowMu.Unlock() }
	ref := testPlacement("ollama", "node-a", "deepseek-r1:latest")

	if got := c.lookup(context.Background(), []placementRef{ref}, time.Second); got[0].ok {
		t.Fatal("a failed lookup was reported as answered")
	}
	// Retried after the short failure interval, not the full TTL.
	f.mu.Lock()
	f.fail, f.d = false, modelDetails{ContextLength: 131072}
	f.mu.Unlock()
	advance(capabilityFailedTTL)
	c.lookup(context.Background(), []placementRef{ref}, time.Second)
	waitCalls(t, f, 2)
	deadline := time.Now().Add(2 * time.Second)
	for !c.lookup(context.Background(), []placementRef{ref}, time.Second)[0].ok {
		if time.Now().After(deadline) {
			t.Fatal("the retried answer never arrived")
		}
		time.Sleep(time.Millisecond)
	}

	// A refresh that fails keeps the good answer.
	f.mu.Lock()
	f.fail = true
	f.mu.Unlock()
	advance(capabilityTTL)
	c.lookup(context.Background(), []placementRef{ref}, time.Second)
	waitCalls(t, f, 3)
	if got := c.lookup(context.Background(), []placementRef{ref}, time.Second); !got[0].ok || got[0].details.ContextLength != 131072 {
		t.Fatalf("after a failed refresh got %+v, want the previous answer", got[0])
	}
}

func TestCapabilityCache_DoesNotWaitPastItsBound(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	c := newCapabilityCache(func(ctx context.Context, _ placementRef) (placementInfo, error) {
		select {
		case <-release:
		case <-ctx.Done():
		}
		return placementInfo{}, ctx.Err()
	})

	start := time.Now()
	got := c.lookup(context.Background(), []placementRef{testPlacement("lmstudio", "node-a", "slow")}, 20*time.Millisecond)

	if got[0].ok {
		t.Fatal("an unanswered placement was reported as answered")
	}
	if waited := time.Since(start); waited > time.Second {
		t.Fatalf("lookup waited %v, want about 20ms", waited)
	}
}

// capabilityRouter is a router whose engines answer details from fixtures,
// each on its own manual node.
func capabilityRouter(t *testing.T, nodes ...capabilityNodeSpec) *Proxy {
	t.Helper()
	p := NewProxy(NewCodec(rwNop{}))
	p.facades = make(map[string]*facade)
	for _, profile := range profiles {
		p.facades[profile.Name] = newFacade(p, profile, NewDiscovery(), 0)
	}
	rf := newFacade(p, routerProfile, NewDiscovery(), 0)
	rf.router = true
	p.router = rf
	for _, n := range nodes {
		list := make([]map[string]string, 0, len(n.models))
		for _, m := range n.models {
			list = append(list, map[string]string{"id": m, "object": "model"})
		}
		listed, _ := json.Marshal(map[string]any{"object": "list", "data": list})
		routes := map[string]string{"/v1/models": string(listed)}
		for k, v := range n.routes {
			routes[k] = v
		}
		base := engineFixture(t, routes)
		node := nodeFor(t, n.nodeID, base.String())
		node.Host = n.nodeID + ".local."
		node.Models = n.models
		p.facades[n.engine].discovery.AddManual(node)
	}
	return p
}

type capabilityNodeSpec struct {
	engine, nodeID string
	models         []string
	routes         map[string]string
}

func TestRouterModelCapabilities_ReportsTheModel(t *testing.T) {
	p := capabilityRouter(t,
		capabilityNodeSpec{engine: "lmstudio", nodeID: "ultron", models: []string{"glm-4.7-flash", "mistralai/devstral-small-2-2512"},
			routes: map[string]string{"/api/v1/models": lmStudioModelsLive}},
		capabilityNodeSpec{engine: "ollama", nodeID: "ultron-ollama", models: []string{"qwen3:0.6b"},
			routes: map[string]string{"/api/show": ollamaShowQwen3, "/api/tags": ollamaTagsQwen3, "/api/ps": `{"models":[]}`}},
	)

	cases := []struct {
		name, path, wantID string
		wantEngine         string
		wantWindow         int64
	}{
		{"LM Studio model", "/pair/v1/models/glm-4.7-flash", "glm-4.7-flash", "lmstudio", 8192},
		{"escaped slash in the id", "/pair/v1/models/mistralai%2Fdevstral-small-2-2512", "mistralai/devstral-small-2-2512", "lmstudio", 4096},
		{"unescaped slash in the id", "/pair/v1/models/mistralai/devstral-small-2-2512", "mistralai/devstral-small-2-2512", "lmstudio", 4096},
		// Ollama's naming implies :latest on neither side here; the tag is
		// matched as Ollama spells it.
		{"Ollama model", "/pair/v1/models/qwen3:0.6b", "qwen3:0.6b", "ollama", 4096},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := routerRequest(t, p, http.MethodGet, tc.path, "")
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body %s", rec.Code, rec.Body)
			}
			var got modelCapabilitiesReport
			if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
				t.Fatal(err)
			}
			if got.ID != tc.wantID || len(got.Placements) != 1 || got.Placements[0].Engine != tc.wantEngine || !got.Placements[0].DetailsAvailable {
				t.Fatalf("report = %+v, want %s on one %s placement with details", got, tc.wantID, tc.wantEngine)
			}
			if got.Limits.ContextWindow != tc.wantWindow {
				t.Errorf("contextWindow = %d, want %d", got.Limits.ContextWindow, tc.wantWindow)
			}
			if len(got.Sources) != 11 {
				t.Errorf("sources = %v, want one per capability and limit", got.Sources)
			}
			if rec.Header().Get(requestIDHeader) == "" {
				t.Error("response carries no X-Request-ID")
			}
		})
	}
}

func TestRouterModelCapabilities_UnknownModelIsNotFound(t *testing.T) {
	p := capabilityRouter(t, capabilityNodeSpec{engine: "lmstudio", nodeID: "ultron", models: []string{"glm-4.7-flash"},
		routes: map[string]string{"/api/v1/models": lmStudioModelsLive}})

	rec := routerRequest(t, p, http.MethodGet, "/pair/v1/models/does-not-exist", "")

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"error": routerModelNotFound, "code": "model_not_found"}
	if !reflect.DeepEqual(body, want) {
		t.Fatalf("body = %v, want %v", body, want)
	}
}

func TestRouterModelCapabilities_RejectsOtherMethods(t *testing.T) {
	p := capabilityRouter(t)

	rec := routerRequest(t, p, http.MethodPost, "/pair/v1/models/glm-4.7-flash", "")

	if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("status = %d, Allow = %q; want 405 allowing GET", rec.Code, rec.Header().Get("Allow"))
	}
}

func TestRouterModelList_CarriesCapabilitiesAndLimits(t *testing.T) {
	p := capabilityRouter(t, capabilityNodeSpec{engine: "lmstudio", nodeID: "ultron", models: []string{"glm-4.7-flash"},
		routes: map[string]string{"/api/v1/models": lmStudioModelsLive}})

	rec := routerRequest(t, p, http.MethodGet, "/v1/models", "")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body %s", rec.Code, rec.Body)
	}
	var list struct {
		Data []struct {
			ID   string          `json:"id"`
			Pair routerModelInfo `json:"pair"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatal(err)
	}
	if len(list.Data) != 1 {
		t.Fatalf("data = %+v, want one model", list.Data)
	}
	info := list.Data[0].Pair
	wantCaps := modelCapabilities{Chat: true, StructuredOutput: true, Tools: true, Reasoning: true}
	wantLimits := modelLimits{ContextWindow: 8192, MaxContextWindow: 202752, RecommendedOutputTokens: 4096, MaxInputTokens: 4096}
	if info.Capabilities != wantCaps || !reflect.DeepEqual(info.Limits, wantLimits) || info.Sources["reasoning"] != sourceEngine {
		t.Fatalf("pair = %+v, want %+v and %+v with reasoning from the engine", info, wantCaps, wantLimits)
	}
}

// plannedOnUltron is the setting from the planner: a 65,536-token window per
// session, 16,384 of it kept for the reply, 8 sessions at once.
var plannedOnUltron = &capabilityConfigured{ContextWindow: 65536, ReplyTokens: 16384, MaxInputTokens: 49152, Sessions: 8}

func TestAggregateCapabilities_UsesThePlannedWindowAndReplyRoom(t *testing.T) {
	glm := func(loaded int64) modelDetails {
		return modelDetails{Type: "llm", Capabilities: []string{"tools", "reasoning"}, CapabilitiesComplete: true,
			ContextLength: 202752, LoadedContext: loaded}
	}
	cases := []struct {
		name        string
		details     modelDetails
		set         *capabilityConfigured
		wantLimits  modelLimits
		wantSources map[string]string
	}{
		{
			// The plan is saved but the engine still runs the old load, and
			// it truncates there. The reply room does not fit in it.
			name:        "plan saved, engine not reloaded yet",
			details:     glm(8192),
			set:         plannedOnUltron,
			wantLimits:  modelLimits{ContextWindow: 8192, MaxContextWindow: 202752, RecommendedOutputTokens: 4096, MaxInputTokens: 4096},
			wantSources: map[string]string{"contextWindow": sourceEngine, "recommendedOutputTokens": sourceEstimated, "maxInputTokens": sourceEstimated},
		},
		{
			name:        "plan applied",
			details:     glm(65536),
			set:         plannedOnUltron,
			wantLimits:  modelLimits{ContextWindow: 65536, MaxContextWindow: 202752, RecommendedOutputTokens: 16384, MaxInputTokens: 49152},
			wantSources: map[string]string{"contextWindow": sourceEngine, "recommendedOutputTokens": sourceConfigured, "maxInputTokens": sourceConfigured},
		},
		{
			// LM Studio loads it with the planned window on first use.
			name:        "planned model not loaded",
			details:     glm(0),
			set:         plannedOnUltron,
			wantLimits:  modelLimits{ContextWindow: 65536, MaxContextWindow: 202752, RecommendedOutputTokens: 16384, MaxInputTokens: 49152},
			wantSources: map[string]string{"contextWindow": sourceConfigured, "recommendedOutputTokens": sourceConfigured, "maxInputTokens": sourceConfigured},
		},
		{
			// PAIR refuses a session beyond its setting even though the
			// engine loaded more.
			name:        "setting smaller than the load",
			details:     glm(131072),
			set:         &capabilityConfigured{ContextWindow: 32768, MaxInputTokens: 32768, Sessions: 4},
			wantLimits:  modelLimits{ContextWindow: 32768, MaxContextWindow: 202752, RecommendedOutputTokens: 8192, MaxInputTokens: 24576},
			wantSources: map[string]string{"contextWindow": sourceConfigured, "recommendedOutputTokens": sourceEstimated, "maxInputTokens": sourceEstimated},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := aggregateCapabilities("glm-4.7-flash", []placementDetails{
				{ref: testPlacement("lmstudio", "ultron", "glm-4.7-flash"), details: tc.details, ok: true, set: tc.set},
			})
			if !reflect.DeepEqual(got.Limits, tc.wantLimits) {
				t.Errorf("limits = %+v, want %+v", got.Limits, tc.wantLimits)
			}
			for k, want := range tc.wantSources {
				if got.Sources[k] != want {
					t.Errorf("sources[%s] = %s, want %s", k, got.Sources[k], want)
				}
			}
			if got.Placements[0].Configured != tc.set {
				t.Errorf("placement configured = %+v, want %+v", got.Placements[0].Configured, tc.set)
			}
		})
	}
}

func TestAggregateCapabilities_SmallestReplyRoomAcrossPlacements(t *testing.T) {
	loaded := modelDetails{Type: "llm", CapabilitiesComplete: true, ContextLength: 131072, LoadedContext: 65536}

	got := aggregateCapabilities("shared", []placementDetails{
		{ref: testPlacement("lmstudio", "node-a", "shared"), details: loaded, ok: true, set: plannedOnUltron},
		{ref: testPlacement("lmstudio", "node-b", "shared"), details: loaded, ok: true,
			set: &capabilityConfigured{ContextWindow: 65536, ReplyTokens: 8192, MaxInputTokens: 57344, Sessions: 8}},
	})

	if got.Limits.RecommendedOutputTokens != 8192 || got.Limits.MaxInputTokens != 57344 {
		t.Fatalf("limits = %+v, want the smaller reply room, 8192, and 57344 for the messages", got.Limits)
	}
}

func TestSettingFor_ReadsTheNodesCapacityReport(t *testing.T) {
	p := NewProxy(NewCodec(rwNop{}))
	// The planner's save: a pool for 8 sessions of 65,536 tokens, with
	// 16,384 of each kept for the reply.
	if err := p.capacity.set(capacityKey{Engine: "lmstudio", Model: "glm-4.7-flash"},
		capacitySetting{ContextTokens: 8 * 65536, MaxSessions: 8, ReplyTokens: 16384}, false); err != nil {
		t.Fatal(err)
	}
	if err := p.capacity.set(capacityKey{Engine: "ollama", Model: "deepseek-r1:latest"},
		capacitySetting{ContextTokens: 32768, MaxSessions: 2}, false); err != nil {
		t.Fatal(err)
	}
	rows := p.capacityReport(context.Background())

	cases := []struct {
		name string
		ref  placementRef
		want *capabilityConfigured
	}{
		{"planned LM Studio model", testPlacement("lmstudio", "ultron", "glm-4.7-flash"), plannedOnUltron},
		{"Ollama name without its tag", testPlacement("ollama", "ultron", "deepseek-r1"),
			&capabilityConfigured{ContextWindow: 16384, MaxInputTokens: 16384, Sessions: 2}},
		{"same model on another engine", testPlacement("ollama", "ultron", "glm-4.7-flash"), nil},
		{"model without a setting", testPlacement("lmstudio", "ultron", "qwen3.8-27b"), nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := settingFor(rows, tc.ref); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("setting = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestCapabilityCache_KeepsTheSettingWhenTheEngineCannotBeAsked(t *testing.T) {
	c := newCapabilityCache(func(context.Context, placementRef) (placementInfo, error) {
		return placementInfo{set: plannedOnUltron}, errors.New("engine unreachable")
	})

	got := c.lookup(context.Background(), []placementRef{testPlacement("lmstudio", "ultron", "glm-4.7-flash")}, time.Second)

	if got[0].ok || got[0].set != plannedOnUltron {
		t.Fatalf("lookup = %+v, want no details but the planned setting", got[0])
	}
}
