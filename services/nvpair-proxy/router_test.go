// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Coverage for the PAIR router: one port that routes a request by its model to
// whichever engine, on whichever node, advertises it.
//
// Each fake engine below stands for one engine on one machine. Two fakes with
// the same node ID are two engines on the same machine; the scheduler's load
// is per machine, so that is what the least-busy tests vary.

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/schedulerwire"
)

// fakeEngine is one engine on one node. It lists models, records what it was
// asked to serve, and answers inference with a fixed status.
type fakeEngine struct {
	engine string
	nodeID string
	srv    *httptest.Server
	served atomic.Int32
	// lastPath and lastBody are the most recent inference request it served.
	lastPath atomic.Value
	lastBody atomic.Value
}

type fakeEngineSpec struct {
	engine string
	nodeID string
	// models is what the node advertises to discovery and lists.
	models []string
	// status answers inference; zero means 200.
	status int
	// stream answers inference with these server-sent event chunks.
	stream []string
}

func startFakeEngine(t *testing.T, spec fakeEngineSpec) *fakeEngine {
	t.Helper()
	fe := &fakeEngine{engine: spec.engine, nodeID: spec.nodeID}
	fe.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && (r.URL.Path == "/v1/models" || r.URL.Path == "/models") {
			data := make([]map[string]string, 0, len(spec.models))
			for _, m := range spec.models {
				data = append(data, map[string]string{"id": m, "object": "model", "owned_by": spec.engine})
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
			return
		}
		body, _ := io.ReadAll(r.Body)
		fe.served.Add(1)
		fe.lastPath.Store(r.URL.Path)
		fe.lastBody.Store(string(body))
		if len(spec.stream) > 0 {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			for _, chunk := range spec.stream {
				_, _ = io.WriteString(w, "data: "+chunk+"\n\n")
				w.(http.Flusher).Flush()
			}
			return
		}
		status := spec.status
		if status == 0 {
			status = http.StatusOK
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = fmt.Fprintf(w, `{"served_by":%q,"node":%q}`, spec.engine, spec.nodeID)
	}))
	t.Cleanup(fe.srv.Close)
	return fe
}

// routerProxy builds a proxy with a facade per engine and the router, none of
// them bound, each engine's discovery holding the given fakes as manual nodes.
func routerProxy(t *testing.T, specs ...fakeEngineSpec) (*Proxy, map[string]*fakeEngine) {
	t.Helper()
	setForTest(t, &retryBackoff, []time.Duration{time.Millisecond})
	return routerProxyWithCodec(t, NewCodec(rwNop{}), specs...)
}

func routerProxyWithCodec(t *testing.T, codec *Codec, specs ...fakeEngineSpec) (*Proxy, map[string]*fakeEngine) {
	t.Helper()
	p := NewProxy(codec)
	p.facades = make(map[string]*facade)
	for _, profile := range profiles {
		p.facades[profile.Name] = newFacade(p, profile, NewDiscovery(), 0)
	}
	rf := newFacade(p, routerProfile, NewDiscovery(), 0)
	rf.router = true
	p.router = rf

	fakes := make(map[string]*fakeEngine, len(specs))
	for _, spec := range specs {
		fe := startFakeEngine(t, spec)
		node := nodeFor(t, spec.nodeID, fe.srv.URL)
		node.Host = spec.nodeID + ".local."
		node.Models = spec.models
		p.facades[spec.engine].discovery.AddManual(node)
		fakes[spec.engine+"@"+spec.nodeID] = fe
	}
	return p, fakes
}

// loadSnapshot ranks nodes with the given pending work, in the order given.
func loadSnapshot(p *Proxy, pending map[string]int, order ...string) {
	ranks := make([]schedulerwire.NodeRank, 0, len(order))
	for i, id := range order {
		ranks = append(ranks, schedulerwire.NodeRank{ID: id, Pending: pending[id], Rank: i})
	}
	applySnapshot(p, schedulerwire.Priority{Nodes: order, Ranks: ranks})
	p.priorityMu.Lock()
	for id, n := range pending {
		p.priorityPending[id] = n
	}
	p.priorityMu.Unlock()
}

func routerRequest(t *testing.T, p *Proxy, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.RemoteAddr = "127.0.0.1:50000"
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	rec := httptest.NewRecorder()
	p.routerFacade().handlePlain(rec, req)
	return rec
}

func chatBody(model string) string {
	return fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}]}`, model)
}

func TestRouter_SendsTheRequestToTheEngineThatAdvertisesTheModel(t *testing.T) {
	p, fakes := routerProxy(t,
		fakeEngineSpec{engine: "ollama", nodeID: "node-a", models: []string{"llama3:latest"}},
		fakeEngineSpec{engine: "lmstudio", nodeID: "node-b", models: []string{"qwen2.5-7b-instruct"}},
	)
	body := chatBody("qwen2.5-7b-instruct")

	rec := routerRequest(t, p, http.MethodPost, "/v1/chat/completions", body)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body %s", rec.Code, rec.Body)
	}
	lmstudio, ollama := fakes["lmstudio@node-b"], fakes["ollama@node-a"]
	if lmstudio.served.Load() != 1 || ollama.served.Load() != 0 {
		t.Fatalf("served lmstudio=%d ollama=%d, want only LM Studio", lmstudio.served.Load(), ollama.served.Load())
	}
	if got := lmstudio.lastPath.Load(); got != "/v1/chat/completions" {
		t.Errorf("engine saw path %v, want /v1/chat/completions unchanged", got)
	}
	if got := lmstudio.lastBody.Load(); got != body {
		t.Errorf("engine saw body %v, want the request body unchanged", got)
	}
	if rec.Header().Get(requestIDHeader) == "" {
		t.Error("response carries no X-Request-ID")
	}
}

func TestRouter_ModelInTwoEnginesGoesToTheLeastBusyNode(t *testing.T) {
	cases := []struct {
		name    string
		pending map[string]int
		want    string
	}{
		{name: "Ollama's node is busier", pending: map[string]int{"node-a": 3, "node-b": 0}, want: "lmstudio@node-b"},
		{name: "LM Studio's node is busier", pending: map[string]int{"node-a": 0, "node-b": 3}, want: "ollama@node-a"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, fakes := routerProxy(t,
				fakeEngineSpec{engine: "ollama", nodeID: "node-a", models: []string{"shared-model"}},
				fakeEngineSpec{engine: "lmstudio", nodeID: "node-b", models: []string{"shared-model"}},
			)
			// node-a ranks first either way, so only the load can move the
			// request to node-b.
			loadSnapshot(p, tc.pending, "node-a", "node-b")

			rec := routerRequest(t, p, http.MethodPost, "/v1/chat/completions", chatBody("shared-model"))

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body %s", rec.Code, rec.Body)
			}
			for key, fe := range fakes {
				want := int32(0)
				if key == tc.want {
					want = 1
				}
				if got := fe.served.Load(); got != want {
					t.Errorf("%s served %d requests, want %d", key, got, want)
				}
			}
		})
	}
}

func TestRouter_UntaggedNameMatchesOllamaLatestButNotOtherEnginesExactIDs(t *testing.T) {
	cases := []struct {
		name   string
		engine string
		code   int
	}{
		{name: "Ollama implies :latest", engine: "ollama", code: http.StatusOK},
		{name: "LM Studio matches exactly", engine: "lmstudio", code: http.StatusBadGateway},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, fakes := routerProxy(t,
				fakeEngineSpec{engine: tc.engine, nodeID: "node-a", models: []string{"llama3:latest"}},
			)

			rec := routerRequest(t, p, http.MethodPost, "/v1/chat/completions", chatBody("llama3"))

			if rec.Code != tc.code {
				t.Fatalf("status = %d, want %d; body %s", rec.Code, tc.code, rec.Body)
			}
			wantServed := int32(0)
			if tc.code == http.StatusOK {
				wantServed = 1
			}
			if got := fakes[tc.engine+"@node-a"].served.Load(); got != wantServed {
				t.Errorf("engine served %d requests, want %d", got, wantServed)
			}
		})
	}
}

func TestRouter_UnadvertisedModelIsAModelNotFound502(t *testing.T) {
	p, fakes := routerProxy(t,
		fakeEngineSpec{engine: "ollama", nodeID: "node-a", models: []string{"llama3:latest"}},
	)

	rec := routerRequest(t, p, http.MethodPost, "/v1/chat/completions", chatBody("nobody-has-this"))

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
	var got map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("body is not JSON: %s", rec.Body)
	}
	want := map[string]string{"error": "no available node advertises the requested model", "code": "model_not_found"}
	if got["error"] != want["error"] || got["code"] != want["code"] || len(got) != len(want) {
		t.Errorf("body = %v, want %v", got, want)
	}
	if fakes["ollama@node-a"].served.Load() != 0 {
		t.Error("an engine was contacted for a model nobody advertises")
	}
}

func TestRouter_FailsOverFromOneEngineToAnother(t *testing.T) {
	p, fakes := routerProxy(t,
		fakeEngineSpec{engine: "ollama", nodeID: "node-a", models: []string{"shared-model"}, status: http.StatusServiceUnavailable},
		fakeEngineSpec{engine: "lmstudio", nodeID: "node-b", models: []string{"shared-model"}},
	)
	// node-a is preferred, and its Ollama refuses.
	loadSnapshot(p, map[string]int{}, "node-a", "node-b")

	rec := routerRequest(t, p, http.MethodPost, "/v1/chat/completions", chatBody("shared-model"))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body %s", rec.Code, rec.Body)
	}
	if got := fakes["ollama@node-a"].served.Load(); got != 1 {
		t.Errorf("Ollama was tried %d times, want once", got)
	}
	if got := fakes["lmstudio@node-b"].served.Load(); got != 1 {
		t.Errorf("LM Studio served %d requests, want 1 after the failover", got)
	}
	if !strings.Contains(rec.Body.String(), `"served_by":"lmstudio"`) {
		t.Errorf("client got %s, want LM Studio's answer", rec.Body)
	}
}

func TestRouter_StreamsTheChosenEnginesResponseThrough(t *testing.T) {
	chunks := []string{`{"choices":[{"delta":{"content":"Hel"}}]}`, `{"choices":[{"delta":{"content":"lo"}}]}`, "[DONE]"}
	p, _ := routerProxy(t,
		fakeEngineSpec{engine: "lmstudio", nodeID: "node-b", models: []string{"qwen"}, stream: chunks},
	)
	body := `{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hi"}]}`

	rec := routerRequest(t, p, http.MethodPost, "/v1/chat/completions", body)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body %s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	var want strings.Builder
	for _, c := range chunks {
		want.WriteString("data: " + c + "\n\n")
	}
	if rec.Body.String() != want.String() {
		t.Errorf("streamed body = %q, want %q", rec.Body.String(), want.String())
	}
}

func TestRouter_AnthropicMessagesSkipsEnginesWithoutThatRoute(t *testing.T) {
	// llama.cpp has no /v1/messages, so its copy of the model is not eligible.
	p, fakes := routerProxy(t,
		fakeEngineSpec{engine: "llamacpp", nodeID: "node-c", models: []string{"gguf-model"}},
	)

	rec := routerRequest(t, p, http.MethodPost, "/v1/messages", `{"model":"gguf-model","max_tokens":8,"messages":[]}`)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body %s", rec.Code, rec.Body)
	}
	if fakes["llamacpp@node-c"].served.Load() != 0 {
		t.Error("llama.cpp was sent an Anthropic request it does not serve")
	}
}

func TestRouter_RequestWithoutAModelIsRefused(t *testing.T) {
	p, _ := routerProxy(t,
		fakeEngineSpec{engine: "ollama", nodeID: "node-a", models: []string{"llama3:latest"}},
	)

	rec := routerRequest(t, p, http.MethodPost, "/v1/chat/completions", `{"messages":[]}`)

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"code":"model_required"`) {
		t.Fatalf("status = %d body %s, want 400 model_required", rec.Code, rec.Body)
	}
}

func TestRouter_EngineSpecificPathsAreNotServed(t *testing.T) {
	p, _ := routerProxy(t)

	rec := routerRequest(t, p, http.MethodPost, "/api/chat", chatBody("llama3"))

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 for an engine-specific path", rec.Code)
	}
}

func TestRouter_ModelListCombinesEveryEngineAndNamesWhereEachModelRuns(t *testing.T) {
	p, _ := routerProxy(t,
		fakeEngineSpec{engine: "ollama", nodeID: "node-a", models: []string{"llama3:latest", "shared-model"}},
		fakeEngineSpec{engine: "lmstudio", nodeID: "node-b", models: []string{"shared-model", "qwen"}},
		fakeEngineSpec{engine: "llamacpp", nodeID: "node-c", models: []string{"gguf-model"}},
	)
	// An unreachable node is left out rather than failing the list.
	dead := startFakeEngine(t, fakeEngineSpec{engine: "llamacpp", nodeID: "node-d"})
	deadNode := nodeFor(t, "node-d", dead.srv.URL)
	dead.srv.Close()
	p.facadeFor("llamacpp").discovery.AddManual(deadNode)

	rec := routerRequest(t, p, http.MethodGet, "/v1/models", "")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body %s", rec.Code, rec.Body)
	}
	var list struct {
		Object string `json:"object"`
		Data   []struct {
			ID     string          `json:"id"`
			Object string          `json:"object"`
			Pair   routerModelInfo `json:"pair"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatalf("decode: %v; body %s", err, rec.Body)
	}
	if list.Object != "list" {
		t.Errorf("object = %q, want list", list.Object)
	}
	type want struct {
		engines []string
		nodes   []routerNode
	}
	expected := map[string]want{
		"llama3:latest": {engines: []string{"ollama"}, nodes: []routerNode{{ID: "node-a", Name: "node-a.local.", Engine: "ollama"}}},
		"shared-model": {engines: []string{"ollama", "lmstudio"}, nodes: []routerNode{
			{ID: "node-a", Name: "node-a.local.", Engine: "ollama"},
			{ID: "node-b", Name: "node-b.local.", Engine: "lmstudio"},
		}},
		"qwen":       {engines: []string{"lmstudio"}, nodes: []routerNode{{ID: "node-b", Name: "node-b.local.", Engine: "lmstudio"}}},
		"gguf-model": {engines: []string{"llamacpp"}, nodes: []routerNode{{ID: "node-c", Name: "node-c.local.", Engine: "llamacpp"}}},
	}
	if len(list.Data) != len(expected) {
		t.Fatalf("listed %d models, want %d (de-duplicated): %s", len(list.Data), len(expected), rec.Body)
	}
	for _, m := range list.Data {
		w, ok := expected[m.ID]
		if !ok {
			t.Errorf("unexpected model %q", m.ID)
			continue
		}
		if m.Object != "model" {
			t.Errorf("%s object = %q, want model", m.ID, m.Object)
		}
		if fmt.Sprint(m.Pair.Engines) != fmt.Sprint(w.engines) {
			t.Errorf("%s engines = %v, want %v", m.ID, m.Pair.Engines, w.engines)
		}
		if fmt.Sprint(m.Pair.Nodes) != fmt.Sprint(w.nodes) {
			t.Errorf("%s nodes = %v, want %v", m.ID, m.Pair.Nodes, w.nodes)
		}
	}
}

func TestRouter_ModelListWithNothingAnsweringIsUnavailable(t *testing.T) {
	p, _ := routerProxy(t)

	rec := routerRequest(t, p, http.MethodGet, "/v1/models", "")

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
}

func TestRouter_WorkloadIsFiledUnderTheServingEngineWithItsOwnID(t *testing.T) {
	setForTest(t, &retryBackoff, []time.Duration{time.Millisecond})
	out := &recordingWriter{}
	p, _ := routerProxyWithCodec(t, NewCodec(out),
		fakeEngineSpec{engine: "lmstudio", nodeID: "node-b", models: []string{"qwen"}},
	)

	rec := routerRequest(t, p, http.MethodPost, "/v1/chat/completions", chatBody("qwen"))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}

	var started, requestLog bool
	for _, line := range out.lines() {
		var frame struct {
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if json.Unmarshal(line, &frame) != nil {
			continue
		}
		switch frame.Method {
		case workloadStartedMethod:
			var params workloadParams
			if err := json.Unmarshal(frame.Params, &params); err != nil {
				t.Fatalf("decode workload: %v", err)
			}
			started = true
			if params.WorkloadInfo.Engine != "lmstudio" {
				t.Errorf("workload engine = %q, want lmstudio", params.WorkloadInfo.Engine)
			}
			if !strings.HasPrefix(params.WorkloadInfo.ID, routerRequestIDPrefix) {
				t.Errorf("workload id = %q, want the router's %q prefix", params.WorkloadInfo.ID, routerRequestIDPrefix)
			}
		case "lmstudio:proxy/request":
			requestLog = true
		case "router:proxy/request", "proxy/request":
			t.Errorf("request event sent as %q, want it in the serving engine's log", frame.Method)
		}
	}
	if !started {
		t.Error("no workload:started was emitted")
	}
	if !requestLog {
		t.Error("the serving engine's request log got no proxy/request")
	}
}

func TestRouter_EnableListensAndRestoresAPortMovedBySetPort(t *testing.T) {
	redirectConfigDir(t)
	first := NewProxy(NewCodec(rwNop{}))
	first.serveCtx = t.Context()
	status, err := first.enableRouter(routerParams{Port: freeTCPPort(t), IgnorePersistedPort: true})
	if err != nil {
		t.Fatalf("enable router: %v", err)
	}
	if !status.Listening || status.Port == 0 {
		t.Fatalf("enable reported %+v", status)
	}

	moved := freeTCPPort(t)
	if err := first.routerFacade().setPort(moved); err != nil {
		t.Fatalf("set-port: %v", err)
	}
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/v1/models", moved))
	if err != nil {
		t.Fatalf("router not listening on the moved port: %v", err)
	}
	resp.Body.Close()
	first.shutdown(t.Context())

	second := NewProxy(NewCodec(rwNop{}))
	second.serveCtx = t.Context()
	t.Cleanup(func() { second.shutdown(t.Context()) })
	restored, err := second.enableRouter(routerParams{})
	if err != nil {
		t.Fatalf("re-enable router: %v", err)
	}
	if restored.Port != moved {
		t.Errorf("router came back on %d, want the persisted %d", restored.Port, moved)
	}
}
