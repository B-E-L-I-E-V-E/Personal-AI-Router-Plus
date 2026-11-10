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
	"strings"
	"testing"
)

func TestSessionShare(t *testing.T) {
	share := func(pool int64, sessions int, want int64) {
		t.Run("", func(t *testing.T) {
			if got := sessionShare(pool, sessions); got != want {
				t.Fatalf("sessionShare(%d, %d) = %d, want %d", pool, sessions, got, want)
			}
		})
	}
	share(262144, 1, 262144)
	share(262144, 4, 65536)
	share(262144, 6, 43008) // 43690 rounded down to a multiple of 1024
	share(262144, 12, 21504)
	share(0, 4, 0)
	share(262144, 0, 0)
}

func TestEffectiveCapacity(t *testing.T) {
	k := capacityKey{Engine: "ollama", Model: "qwen3:0.6b"}

	t.Run("a setting wins over detection", func(t *testing.T) {
		row := effective(k, &capacitySetting{ContextTokens: 262144, MaxSessions: 6},
			&detectedCapacity{Window: 262144, Slots: 1, Pool: 262144}, localUsage{Sessions: 2, Tokens: 90000})
		if row.Source != "configured" || row.ContextTokens != 262144 || row.MaxSessions != 6 || row.SessionShare != 43008 {
			t.Fatalf("row = %+v, want the configured 262144 ÷ 6 = 43008", row)
		}
		if row.ActiveSessions != 2 || row.UsedTokens != 90000 {
			t.Fatalf("usage = %d sessions, %d tokens, want 2 and 90000", row.ActiveSessions, row.UsedTokens)
		}
		if row.Warning != "" {
			t.Fatalf("unexpected warning %q", row.Warning)
		}
	})
	t.Run("a complete detection is used when nothing is set", func(t *testing.T) {
		row := effective(k, nil, &detectedCapacity{Window: 32768, Slots: 4, Pool: 32768}, localUsage{})
		if row.Source != "detected" || row.SessionShare != 8192 {
			t.Fatalf("row = %+v, want detected 32768 ÷ 4 = 8192", row)
		}
	})
	t.Run("a window alone gives a share but asks for a setting", func(t *testing.T) {
		row := effective(k, nil, &detectedCapacity{Window: 32768}, localUsage{})
		if row.SessionShare != 32768 || row.MaxSessions != 0 || row.Warning == "" {
			t.Fatalf("row = %+v, want share 32768, unknown sessions, and a warning", row)
		}
	})
	t.Run("nothing known stays unknown", func(t *testing.T) {
		row := effective(k, nil, nil, localUsage{Sessions: 1, Tokens: 10})
		if row.Source != "unknown" || row.SessionShare != 0 {
			t.Fatalf("row = %+v, want unknown", row)
		}
	})
	t.Run("warns when the share is too small to be useful", func(t *testing.T) {
		row := effective(k, &capacitySetting{ContextTokens: 32768, MaxSessions: 12}, nil, localUsage{})
		if row.SessionShare != 2048 || !strings.Contains(row.Warning, "below") {
			t.Fatalf("row = %+v, want share 2048 with a low-share warning", row)
		}
	})
	t.Run("warns when the share exceeds the engine's window", func(t *testing.T) {
		row := effective(k, &capacitySetting{ContextTokens: 262144, MaxSessions: 2},
			&detectedCapacity{Window: 32768}, localUsage{})
		if !strings.Contains(row.Warning, "truncate") {
			t.Fatalf("warning = %q, want a truncation warning", row.Warning)
		}
	})
}

func TestCapacitySettings_PersistAndValidate(t *testing.T) {
	path := filepath.Join(t.TempDir(), capacityFileName)
	s := newCapacityStore()
	s.path = path
	k := capacityKey{Engine: "llamacpp", Model: "bartowski/Qwen3.8-Flash-Next-GGUF:Q4_K_M"}

	reject := func(name string, v capacitySetting) {
		t.Run(name, func(t *testing.T) {
			if err := s.set(k, v, false); err == nil {
				t.Fatalf("set(%+v) succeeded, want a validation error", v)
			}
		})
	}
	reject("pool below the minimum", capacitySetting{ContextTokens: 512, MaxSessions: 1})
	reject("no sessions", capacitySetting{ContextTokens: 65536, MaxSessions: 0})
	reject("too many sessions", capacitySetting{ContextTokens: 65536, MaxSessions: maxConfiguredSessions + 1})
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("a rejected setting wrote the file (stat err %v)", err)
	}

	if err := s.set(k, capacitySetting{ContextTokens: 262144, MaxSessions: 6}, false); err != nil {
		t.Fatalf("set: %v", err)
	}
	reloaded := newCapacityStore()
	reloaded.path = path
	if err := reloaded.load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := reloaded.settingsSnapshot()[k]; got != (capacitySetting{ContextTokens: 262144, MaxSessions: 6}) {
		t.Fatalf("reloaded setting = %+v, want 262144 / 6", got)
	}

	if err := s.set(k, capacitySetting{}, true); err != nil {
		t.Fatalf("clear: %v", err)
	}
	if err := reloaded.load(); err != nil {
		t.Fatalf("reload: %v", err)
	}
	if _, ok := reloaded.settingsSnapshot()[k]; ok {
		t.Fatal("cleared setting survived a reload")
	}
}

func TestCapacitySettings_CorruptFileIsReported(t *testing.T) {
	path := filepath.Join(t.TempDir(), capacityFileName)
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	s := newCapacityStore()
	s.path = path
	if err := s.load(); err == nil {
		t.Fatal("load of a corrupt file succeeded, want an error")
	}
	if len(s.settingsSnapshot()) != 0 {
		t.Fatal("a corrupt file must leave the settings empty")
	}
}

// engineFixture serves canned JSON by path (and query, when given).
func engineFixture(t *testing.T, routes map[string]string) *url.URL {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := r.URL.Path
		if r.URL.RawQuery != "" {
			key += "?" + r.URL.RawQuery
		}
		body, ok := routes[key]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("parse fixture url: %v", err)
	}
	return u
}

func TestDetectCapacity(t *testing.T) {
	detect := func(t *testing.T, engine string, base *url.URL) map[string]detectedCapacity {
		t.Helper()
		got, err := newCapacityStore().detect(context.Background(), engine, base)
		if err != nil {
			t.Fatalf("detect %s: %v", engine, err)
		}
		return got
	}

	t.Run("ollama reports the loaded window only", func(t *testing.T) {
		base := engineFixture(t, map[string]string{
			"/api/ps": `{"models":[{"name":"qwen3:0.6b","model":"qwen3:0.6b","context_length":40960}]}`,
		})
		got := detect(t, "ollama", base)
		if got["qwen3:0.6b"] != (detectedCapacity{Window: 40960}) {
			t.Fatalf("detected = %+v, want window 40960 and nothing else", got)
		}
	})
	t.Run("ollama with nothing loaded reports nothing", func(t *testing.T) {
		// Recorded from a live Ollama 0.40.0.
		if got := detect(t, "ollama", engineFixture(t, map[string]string{"/api/ps": `{"models":[]}`})); len(got) != 0 {
			t.Fatalf("detected = %+v, want none", got)
		}
	})
	t.Run("lm studio reports window, parallel and a shared pool", func(t *testing.T) {
		base := engineFixture(t, map[string]string{"/api/v1/models": `{"models":[
			{"type":"llm","key":"qwen/qwen3-8b","max_context_length":131072,
			 "loaded_instances":[{"id":"qwen/qwen3-8b","config":{"context_length":65536,"parallel":4}}]},
			{"type":"embedding","key":"text-embedding-nomic-embed-text-v1.5","loaded_instances":[],"max_context_length":2048}]}`})
		got := detect(t, "lmstudio", base)
		want := detectedCapacity{Window: 65536, Slots: 4, Pool: 65536}
		if d := got["qwen/qwen3-8b"]; d.Window != want.Window || d.Slots != want.Slots || d.Pool != want.Pool {
			t.Fatalf("detected = %+v, want %+v", d, want)
		}
		if len(got) != 1 {
			t.Fatalf("detected %d models, want only the loaded one", len(got))
		}
	})
	t.Run("a single-model llama.cpp server reports slots", func(t *testing.T) {
		base := engineFixture(t, map[string]string{
			"/props": `{"model_alias":"qwen3-8b","total_slots":4,"default_generation_settings":{"n_ctx":16384}}`,
		})
		d := detect(t, "llamacpp", base)["qwen3-8b"]
		if d.Window != 16384 || d.Slots != 4 || d.Pool != 65536 {
			t.Fatalf("detected = %+v, want 16384 × 4 = 65536", d)
		}
	})
	t.Run("a llama.cpp router is asked per loaded model", func(t *testing.T) {
		// The top-level answer is recorded from a live llama.cpp router (b11146).
		base := engineFixture(t, map[string]string{
			"/props":                    `{"role":"router","max_instances":4,"models_autoload":true,"model_alias":"llama-server","model_path":"none","default_generation_settings":{"params":null,"n_ctx":0}}`,
			"/models":                   `{"object":"list","data":[{"id":"loaded-model","status":{"value":"loaded"}},{"id":"idle-model","status":{"value":"unloaded"}}]}`,
			"/props?model=loaded-model": `{"total_slots":2,"default_generation_settings":{"n_ctx":32768}}`,
		})
		got := detect(t, "llamacpp", base)
		if d := got["loaded-model"]; d.Window != 32768 || d.Slots != 2 {
			t.Fatalf("loaded-model = %+v, want 32768 with 2 slots", d)
		}
		if _, ok := got["idle-model"]; ok {
			t.Fatal("an unloaded model must not be reported")
		}
		if _, ok := got["llama-server"]; ok {
			t.Fatal("the router itself must not be reported as a model")
		}
	})
}

func TestCapacityAPI(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAPIFixture(t, tc, okUpstream(t))
		put := func(body string) *httptest.ResponseRecorder {
			t.Helper()
			return fx.do(http.MethodPut, capacityAPIPath, "", body)
		}

		w := put(`{"engine":"` + tc.profile.Name + `","model":"big-model","contextTokens":262144,"maxSessions":6}`)
		wantStatus(t, w, http.StatusOK)
		var row modelCapacity
		if err := json.Unmarshal(w.Body.Bytes(), &row); err != nil {
			t.Fatalf("decode row: %v", err)
		}
		if row.SessionShare != 43008 || row.Source != "configured" {
			t.Fatalf("row = %+v, want configured share 43008", row)
		}

		var report struct {
			Models []modelCapacity `json:"models"`
		}
		get := fx.do(http.MethodGet, capacityAPIPath, "", "")
		wantStatus(t, get, http.StatusOK)
		if err := json.Unmarshal(get.Body.Bytes(), &report); err != nil {
			t.Fatalf("decode report: %v", err)
		}
		if len(report.Models) != 1 || report.Models[0].Model != "big-model" {
			t.Fatalf("report = %+v, want the configured model", report.Models)
		}

		wantStatus(t, put(`{"engine":"nope","model":"m","contextTokens":65536,"maxSessions":1}`), http.StatusBadRequest)
		wantStatus(t, put(`{"engine":"`+tc.profile.Name+`","model":"m","contextTokens":65536,"maxSessions":0}`), http.StatusBadRequest)
		wantStatus(t, put(`{"engine":"`+tc.profile.Name+`","model":"big-model","clear":true}`), http.StatusNoContent)
		wantStatus(t, fx.do(http.MethodDelete, capacityAPIPath, "", ""), http.StatusMethodNotAllowed)
	})
}

// Only sessions whose last turn ran on this node count against its capacity,
// and each counts once with the context its engine last reported.
func TestLocalUsageByModel(t *testing.T) {
	s, clk := newClockedStore(t)
	record := func(id, node string, local bool, tokens int64) {
		t.Helper()
		s.end(id, beginT(t, s, id, ""))
		s.pin(id, "ollama", node, local)
		s.recordUsage(id, "ollama", "qwen3:0.6b", tokenUsage{Prompt: tokens - 100, Completion: 100})
	}
	record("here-1", "self", true, 80000)
	record("here-2", "self", true, 20000)
	record("elsewhere", "node-b", false, 50000)

	k := capacityKey{Engine: "ollama", Model: "qwen3:0.6b"}
	if got := s.localUsageByModel()[k]; got != (localUsage{Sessions: 2, Tokens: 100000}) {
		t.Fatalf("local usage = %+v, want 2 sessions holding 100000 tokens", got)
	}

	clk.advance(defaultSessionIdleTimeout)
	if got := s.localUsageByModel()[k]; got != (localUsage{}) {
		t.Fatalf("usage after expiry = %+v, want none", got)
	}
}
