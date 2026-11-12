// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// TestLiveOllama_PerformanceFiguresAndBenchmark routes real requests through
// the proxy to a real Ollama and checks the figures it records. It sends a few
// short requests and one benchmark run (four more) to the model.
func TestLiveOllama_PerformanceFiguresAndBenchmark(t *testing.T) {
	base := os.Getenv("PAIR_LIVE_OLLAMA_URL")
	if base == "" {
		t.Skip("set PAIR_LIVE_OLLAMA_URL to run against a real Ollama")
	}
	model := os.Getenv("PAIR_LIVE_OLLAMA_MODEL")
	if model == "" {
		model = "qwen3:0.6b"
	}
	tc := ollamaCase(t)
	disc := NewDiscovery()
	disc.AddManual(nodeForModel(t, "live", base, model))
	p := newTestProxy(tc.profile, NewCodec(&recRW{}), disc, tc.profile.FacadePort)
	f := p.soleFacade()

	send := func(path, body string) {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
		req.RemoteAddr = "127.0.0.1:50000"
		w := httptest.NewRecorder()
		f.handlePlain(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("%s: status %d: %s", path, w.Code, w.Body.String())
		}
	}
	latest := func() perfRecord {
		t.Helper()
		rep := p.perf.report()
		if len(rep.Recent) == 0 {
			t.Fatal("no request was recorded")
		}
		r := rep.Recent[0]
		t.Logf("%s streamed=%v in=%d out=%d think=%v/%v tokens/%v ms read=%v write=%v ttft=%v load=%v sources=%+v",
			r.Path, r.Streamed, r.PromptTokens, r.CompletionTokens, r.Thinking, deref64(r.ThinkingTokens), derefF(r.ThinkingMs),
			derefF(r.ReadTPS), derefF(r.WriteTPS), derefF(r.TTFTMs), derefF(r.LoadMs), r.Sources)
		return r
	}
	chat := func(stream, think bool) string {
		return fmt.Sprintf(`{"model":%q,"stream":%v,"think":%v,"messages":[{"role":"user","content":"Is 17 a prime number? Answer yes or no."}],"options":{"num_predict":512,"temperature":0}}`,
			model, stream, think)
	}

	t.Run("streamed chat with thinking", func(t *testing.T) {
		send("/api/chat", chat(true, true))
		r := latest()
		if !r.Thinking || r.ThinkingMs == nil || r.Sources.Thinking != perfSourceProxy {
			t.Fatalf("want a thinking reply clocked by the proxy, got %+v", r)
		}
		if r.Sources.Read != perfSourceEngine || r.Sources.Write != perfSourceEngine || r.ReadTPS == nil || r.WriteTPS == nil {
			t.Fatalf("want engine-reported speeds, got %+v", r)
		}
	})
	t.Run("non-streamed chat with thinking", func(t *testing.T) {
		send("/api/chat", chat(false, true))
		r := latest()
		if !r.Thinking || r.Sources.Thinking != perfSourceEstimated || r.TTFTMs == nil {
			t.Fatalf("want an estimated thinking share and an engine TTFT, got %+v", r)
		}
	})
	t.Run("streamed chat with thinking off", func(t *testing.T) {
		send("/api/chat", chat(true, false))
		if r := latest(); r.Thinking {
			t.Fatalf("thinking off still recorded thinking: %+v", r)
		}
	})
	t.Run("OpenAI-compatible stream with usage", func(t *testing.T) {
		send("/v1/chat/completions", fmt.Sprintf(`{"model":%q,"stream":true,"stream_options":{"include_usage":true},"max_tokens":512,"messages":[{"role":"user","content":"Is 17 prime? Answer yes or no."}]}`, model))
		r := latest()
		if r.CompletionTokens == 0 || r.Sources.Tokens != perfSourceEngine || r.WriteTPS == nil {
			t.Fatalf("want engine tokens and a measured write speed, got %+v", r)
		}
	})
	t.Run("benchmark", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(f.handlePlain))
		defer srv.Close()
		run, err := p.perf.beginRun(tc.profile.Name, model)
		if err != nil {
			t.Fatal(err)
		}
		p.runBenchmark(run, tc.profile.Name, srv.Listener.Addr().(*net.TCPAddr).Port)
		got := p.perf.benchmarkSnapshot()[0]
		for _, s := range got.Steps {
			if s.Result == nil {
				t.Errorf("step %s: %s %s", s.Name, s.State, s.Detail)
				continue
			}
			r := s.Result
			t.Logf("step %-11s think=%s in=%d out=%d read=%v write=%v ttft=%v thinking=%v ms (%v tokens)",
				s.Name, s.Thinking, r.PromptTokens, r.CompletionTokens, derefF(r.ReadTPS), derefF(r.WriteTPS), derefF(r.TTFTMs), derefF(r.ThinkingMs), deref64(r.ThinkingTokens))
			if (s.Thinking == "on") != r.Thinking {
				t.Errorf("step %s: thinking %s but recorded thinking=%v", s.Name, s.Thinking, r.Thinking)
			}
		}
		if got.State != benchDone {
			t.Fatalf("run %s: %s", got.State, got.Error)
		}
	})
}

func derefF(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}

func deref64(p *int64) any {
	if p == nil {
		return nil
	}
	return *p
}
