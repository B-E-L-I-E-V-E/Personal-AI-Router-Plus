// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/clustertrust"
)

// Recorded from a live Ollama 0.40.11 for qwen3:0.6b, licence text cut short.
const (
	ollamaShowQwen3 = `{"license":"                                 Apache License\n                           Version 2.0, January 2004\n",
		"parameters":"temperature                    0.6\ntop_k                          20",
		"details":{"parent_model":"","format":"gguf","family":"qwen3","families":["qwen3"],"parameter_size":"751.63M","quantization_level":"Q4_K_M"},
		"model_info":{"general.architecture":"qwen3","general.basename":"Qwen3","general.file_type":15,"general.license":"apache-2.0",
			"general.parameter_count":751632384,"general.quantization_version":2,"general.size_label":"0.6B","general.type":"model",
			"qwen3.context_length":40960,"qwen3.block_count":28},
		"capabilities":["completion","tools","thinking"],"modified_at":"2026-11-10T23:53:45.015542771-08:00"}`
	ollamaTagsQwen3 = `{"models":[{"name":"qwen3:0.6b","model":"qwen3:0.6b","modified_at":"2026-11-10T23:53:45.015542771-08:00",
		"size":522653767,"digest":"7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435"}]}`
)

func ollamaDetailsFixture(t *testing.T) *url.URL {
	t.Helper()
	return engineFixture(t, map[string]string{"/api/show": ollamaShowQwen3, "/api/tags": ollamaTagsQwen3})
}

func TestOllamaModelDetails_ReportsWhatShowAndTagsSay(t *testing.T) {
	d, err := specReader.details(context.Background(), "ollama", ollamaDetailsFixture(t), "qwen3:0.6b")
	if err != nil {
		t.Fatal(err)
	}
	want := modelDetails{
		Engine: "ollama", Model: "qwen3:0.6b",
		License:        "apache-2.0",
		LicenseText:    "Apache License\n                           Version 2.0, January 2004",
		Type:           "model",
		Architecture:   "qwen3",
		Family:         "qwen3",
		ParameterSize:  "751.63M",
		ParameterCount: 751632384,
		Quantization:   "Q4_K_M",
		Format:         "gguf",
		Capabilities:   []string{"completion", "thinking", "tools"},
		ContextLength:  40960,
		SizeBytes:      522653767,
		Digest:         "7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435",
		ModifiedAt:     "2026-11-10T23:53:45.015542771-08:00",
		Parameters:     "temperature                    0.6\ntop_k                          20",
		Links:          []modelLink{{Label: "Ollama library", URL: "https://ollama.com/library/qwen3", Source: "model name"}},
	}
	if !reflect.DeepEqual(d, want) {
		t.Fatalf("details:\n got %+v\nwant %+v", d, want)
	}
}

func TestOllamaPage_FollowsTheRegistryInTheName(t *testing.T) {
	cases := []struct {
		name, model, wantURL string
	}{
		{"library model with tag", "qwen3:0.6b", "https://ollama.com/library/qwen3"},
		{"library model without tag", "llama3.2", "https://ollama.com/library/llama3.2"},
		{"user namespace", "someone/custom-model:latest", "https://ollama.com/someone/custom-model"},
		{"explicit ollama registry", "registry.ollama.ai/library/gemma3:4b", "https://ollama.com/library/gemma3"},
		{"hugging face pull", "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M", "https://huggingface.co/Qwen/Qwen3-8B-GGUF"},
		{"another registry has no known page", "registry.example.com/team/model:1", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, got := ollamaPage(tc.model); got != tc.wantURL {
				t.Fatalf("ollamaPage(%q) = %q, want %q", tc.model, got, tc.wantURL)
			}
		})
	}
}

// describedGGUF is a model file whose header describes the model the way
// llama.cpp's convert script writes it.
func describedGGUF(t *testing.T, dir, name string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	data := ggufFile(t, []any{
		"general.architecture", "llama",
		"general.type", "model",
		"general.name", "Example Instruct",
		"general.description", "A small instruction-tuned example model.",
		"general.organization", "Example Org",
		"general.size_label", "8B",
		"general.license", "llama3.1",
		"general.license.link", "https://example.com/license",
		"general.tags", []string{"text-generation", "facebook", "meta"},
		"general.languages", []string{"en", "de"},
		"general.repo_url", "https://huggingface.co/example/example-instruct",
		"general.url", "javascript:alert(1)",
		"general.source.huggingface.repository", "example/example-instruct-base",
		"general.base_model.count", uint32(1),
		"general.base_model.0.name", "Example Base",
		"general.base_model.0.repo_url", "https://huggingface.co/example/example-base",
		"general.file_type", uint32(15),
		"general.parameter_count", uint64(8030261248),
		"llama.context_length", uint32(131072),
		"tokenizer.ggml.tokens", []string{"a", "b"},
	})
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// describedGGUFDetails is what describedGGUF's header says, before the
// engine adds its own fields.
func describedGGUFDetails() modelDetails {
	return modelDetails{
		Name:           "Example Instruct",
		Description:    "A small instruction-tuned example model.",
		Tags:           []string{"text-generation", "facebook", "meta"},
		Languages:      []string{"en", "de"},
		Author:         "Example Org",
		License:        "llama3.1",
		Type:           "model",
		Architecture:   "llama",
		ParameterSize:  "8B",
		ParameterCount: 8030261248,
		Quantization:   "Q4_K_M",
		Format:         "gguf",
		ContextLength:  131072,
		Links: []modelLink{
			{Label: "Repository", URL: "https://huggingface.co/example/example-instruct", Source: "general.repo_url"},
			{Label: "Hugging Face", URL: "https://huggingface.co/example/example-instruct-base", Source: "general.source.huggingface.repository"},
			{Label: "License", URL: "https://example.com/license", Source: "general.license.link"},
			{Label: "Base model: Example Base", URL: "https://huggingface.co/example/example-base", Source: "general.base_model.0.url"},
		},
	}
}

func TestLlamaCPPModelDetails_ReadsTheModelFile(t *testing.T) {
	path := describedGGUF(t, t.TempDir(), "example.gguf")
	props, err := json.Marshal(map[string]any{
		"model_path":                  path,
		"modalities":                  map[string]bool{"vision": false, "audio": false},
		"default_generation_settings": map[string]any{"n_ctx": 16384},
	})
	if err != nil {
		t.Fatal(err)
	}
	base := engineFixture(t, map[string]string{"/props?model=example": string(props)})

	d, err := specReader.details(context.Background(), "llamacpp", base, "example")
	if err != nil {
		t.Fatal(err)
	}
	want := describedGGUFDetails()
	want.Engine, want.Model, want.LoadedContext = "llamacpp", "example", 16384
	if !reflect.DeepEqual(d, want) {
		t.Fatalf("details:\n got %+v\nwant %+v", d, want)
	}
}

func TestLlamaCPPModelDetails_SaysWhenTheFileIsUnknown(t *testing.T) {
	base := engineFixture(t, map[string]string{"/props?model=idle": `{"model_path":""}`})
	d, err := specReader.details(context.Background(), "llamacpp", base, "idle")
	if err != nil {
		t.Fatal(err)
	}
	if d.Name != "" || len(d.Links) != 0 || len(d.Notes) != 1 || !strings.Contains(d.Notes[0], "load the model") {
		t.Fatalf("details = %+v, want only a note to load the model", d)
	}
}

func TestLMStudioModelDetails_CombinesTheListingAndTheFile(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, ".lmstudio", "models", "example", "example-instruct-GGUF")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	describedGGUF(t, dir, "example-instruct-Q4_K_M.gguf")
	base := engineFixture(t, map[string]string{"/api/v1/models": `{"models":[
		{"type":"llm","publisher":"example","key":"example-instruct","display_name":"Example Instruct 8B",
		 "architecture":"llama","quantization":{"name":"Q4_K_M","bits_per_weight":4},"size_bytes":4920000000,
		 "params_string":"8B","max_context_length":131072,"format":"gguf","path":"example/example-instruct-GGUF",
		 "capabilities":{"vision":false,"trained_for_tool_use":true},"description":null,
		 "loaded_instances":[{"id":"example-instruct","config":{"context_length":32768}}]}]}`})
	reader := &specClient{client: http.DefaultClient, homeDir: func() (string, error) { return home, nil }}

	d, err := reader.details(context.Background(), "lmstudio", base, "example-instruct")
	if err != nil {
		t.Fatal(err)
	}
	want := describedGGUFDetails()
	// LM Studio's own answer wins over the file where both say something.
	want.Engine, want.Model = "lmstudio", "example-instruct"
	want.Name = "Example Instruct 8B"
	want.Type = "llm"
	want.Publisher = "example"
	want.SizeBytes = 4920000000
	want.LoadedContext = 32768
	want.Capabilities = []string{"tools"}
	if !reflect.DeepEqual(d, want) {
		t.Fatalf("details:\n got %+v\nwant %+v", d, want)
	}
}

func TestLMStudioModelDetails_SaysWhenTheModelIsNotListed(t *testing.T) {
	base := engineFixture(t, map[string]string{"/api/v1/models": `{"models":[]}`})
	d, err := specReader.details(context.Background(), "lmstudio", base, "missing")
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Notes) != 1 || !strings.Contains(d.Notes[0], "does not list") {
		t.Fatalf("notes = %v, want one saying LM Studio does not list it", d.Notes)
	}
}

func TestModelDetailsRPC_RejectsAMissingModel(t *testing.T) {
	p := testProxy(ollamaCase(t).profile, NewDiscovery(), 0)
	if _, err := p.modelDetails(context.Background(), json.RawMessage(`{"engine":"ollama"}`)); err == nil {
		t.Fatal("want an error for a request without a model")
	}
	if _, err := p.modelDetails(context.Background(), json.RawMessage(`{"engine":"vllm","model":"m"}`)); err == nil {
		t.Fatal("want an error for an unknown engine")
	}
}

// detailsPair is two paired nodes: A asks, and B's Ollama has the model.
// bIngress serves B's cluster ingress; a test may replace its handler.
func detailsPair(t *testing.T, bHandler func(*facade) http.Handler) *facade {
	t.Helper()
	tc := ollamaCase(t)
	aDir, bDir := pairedClusterDirs(t)

	b := testProxy(tc.profile, NewDiscovery(), tc.profile.StandalonePort).soleFacade()
	b.host.mesh = clustertrust.Open(bDir)
	engine := nodeFor(t, "engine", ollamaDetailsFixture(t).String())
	if err := b.setLocalBackend(localBackend{Host: engine.Addresses[0], Port: engine.Port, Healthy: true}); err != nil {
		t.Fatalf("set local backend: %v", err)
	}
	ingress := httptest.NewUnstartedServer(bHandler(b))
	ingress.TLS = b.host.mesh.ServerTLSConfig()
	ingress.StartTLS()
	t.Cleanup(ingress.Close)

	a := testProxy(tc.profile, NewDiscovery(), tc.profile.StandalonePort).soleFacade()
	a.host.mesh = clustertrust.Open(aDir)
	n := nodeForModel(t, "node-b", ingress.URL, "qwen3:0.6b")
	n.ClusterUUID = "b"
	a.discovery.SetSubscribed([]Node{n})
	return a
}

func askNodeB(t *testing.T, a *facade) *modelDetails {
	t.Helper()
	d, err := a.host.modelDetails(context.Background(), json.RawMessage(`{"engine":"ollama","model":"qwen3:0.6b","nodeId":"node-b"}`))
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestRemoteModelDetails_AsksTheNodeThatHasTheModel(t *testing.T) {
	a := detailsPair(t, func(b *facade) http.Handler { return http.HandlerFunc(b.handleClusterIngress) })
	d := askNodeB(t, a)
	if d.Family != "qwen3" || d.SizeBytes != 522653767 || len(d.Links) != 1 || len(d.Notes) != 0 {
		t.Fatalf("details = %+v, want node B's Ollama answer", d)
	}
}

func TestRemoteModelDetails_ExplainsANodeWithoutTheRoute(t *testing.T) {
	// An older PAIR forwards the unknown path to its engine, which answers 404.
	a := detailsPair(t, func(*facade) http.Handler { return http.NotFoundHandler() })
	d := askNodeB(t, a)
	if d.Family != "" || len(d.Notes) != 1 || !strings.Contains(d.Notes[0], "older PAIR") {
		t.Fatalf("details = %+v, want only a note about an older PAIR", d)
	}
}

func TestRemoteModelDetails_RefusesAnUntrustedNode(t *testing.T) {
	a := detailsPair(t, func(b *facade) http.Handler { return http.HandlerFunc(b.handleClusterIngress) })
	nodes := a.discovery.Nodes()
	nodes[0].ClusterUUID = "stranger"
	a.discovery.SetSubscribed(nodes)
	d := askNodeB(t, a)
	if d.Family != "" || len(d.Notes) != 1 || !strings.Contains(d.Notes[0], "not a trusted member") {
		t.Fatalf("details = %+v, want only a note that the node is not trusted", d)
	}
}

func TestRemoteModelDetails_RejectsAnUnknownNode(t *testing.T) {
	a := detailsPair(t, func(b *facade) http.Handler { return http.HandlerFunc(b.handleClusterIngress) })
	if _, err := a.host.modelDetails(context.Background(), json.RawMessage(`{"engine":"ollama","model":"m","nodeId":"nowhere"}`)); err == nil {
		t.Fatal("want an error for a node discovery does not know")
	}
}

func TestRemoteModelDetails_ExplainsAnUnreachableNode(t *testing.T) {
	a := detailsPair(t, func(b *facade) http.Handler { return http.HandlerFunc(b.handleClusterIngress) })
	nodes := a.discovery.Nodes()
	// A listener that is closed at once: nothing answers on that port.
	gone := httptest.NewServer(http.NotFoundHandler())
	gone.Close()
	u, err := url.Parse(gone.URL)
	if err != nil {
		t.Fatal(err)
	}
	moved := nodeFor(t, nodes[0].ID, "http://"+u.Host)
	moved.ClusterUUID = nodes[0].ClusterUUID
	a.discovery.SetSubscribed([]Node{moved})
	d := askNodeB(t, a)
	if len(d.Notes) != 1 || !strings.Contains(d.Notes[0], "unreachable from here") || strings.Contains(d.Notes[0], u.Host) {
		t.Fatalf("notes = %v, want one plain note without the node's address", d.Notes)
	}
}
