// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// steppingClock returns a time that moves 10 ms forward on every read, so a
// test knows exactly when each frame "arrived": the tracker reads the clock
// once per frame it measures and once when the body ends.
func steppingClock(start time.Time) func() time.Time {
	n := 0
	return func() time.Time {
		n++
		return start.Add(time.Duration(n) * 10 * time.Millisecond)
	}
}

// measure drains body through the observing reader in small, uneven reads
// and returns the record it produced, if any.
func measure(t *testing.T, contentType, body string) (perfRecord, bool) {
	t.Helper()
	start := time.Unix(1_700_000_000, 0)
	var got *perfRecord
	tr := newPerfTracker(start, perfRecord{Engine: "ollama", Model: "m"}, func(r perfRecord) { got = &r })
	tr.now = steppingClock(start)
	r := newObservedBody(io.NopCloser(&chunkedReader{data: []byte(body), sizes: []int{1, 7, 3, 64, 2}}), contentType, nil, tr)
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(out) != body {
		t.Fatalf("body changed in transit:\n got %q\nwant %q", out, body)
	}
	if got == nil {
		return perfRecord{}, false
	}
	return *got, true
}

func ndjson(lines ...string) string { return strings.Join(lines, "\n") + "\n" }

func sse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("data: " + e + "\n\n")
	}
	return b.String()
}

func wantFloat(t *testing.T, name string, got *float64, want float64) {
	t.Helper()
	if got == nil {
		t.Fatalf("%s = nil, want %v", name, want)
	}
	if *got != want {
		t.Fatalf("%s = %v, want %v", name, *got, want)
	}
}

func wantNil(t *testing.T, name string, got *float64) {
	t.Helper()
	if got != nil {
		t.Fatalf("%s = %v, want nil", name, *got)
	}
}

func TestPerfTracker_OllamaChatStreamUsesEngineTimingsAndClocksThinking(t *testing.T) {
	// Two thinking chunks, then two answer chunks, then the final object with
	// the engine's counts and durations (nanoseconds).
	rec, ok := measure(t, "application/x-ndjson", ndjson(
		`{"message":{"role":"assistant","content":"","thinking":"Let"},"done":false}`,
		`{"message":{"role":"assistant","content":"","thinking":" me think"},"done":false}`,
		`{"message":{"role":"assistant","content":"Paris"},"done":false}`,
		`{"message":{"role":"assistant","content":"."},"done":false}`,
		`{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":20,"eval_count":5,`+
			`"prompt_eval_duration":100000000,"eval_duration":250000000,"load_duration":50000000,"total_duration":420000000}`,
	))
	if !ok {
		t.Fatal("no record")
	}
	if rec.PromptTokens != 20 || rec.CompletionTokens != 5 || rec.Sources.Tokens != perfSourceEngine {
		t.Fatalf("tokens = %d in / %d out (%s), want 20 / 5 from the engine", rec.PromptTokens, rec.CompletionTokens, rec.Sources.Tokens)
	}
	wantFloat(t, "read tokens/s", rec.ReadTPS, 200)  // 20 tokens in 100 ms
	wantFloat(t, "write tokens/s", rec.WriteTPS, 20) // 5 tokens in 250 ms
	wantFloat(t, "load ms", rec.LoadMs, 50)
	if rec.Sources.Read != perfSourceEngine || rec.Sources.Write != perfSourceEngine {
		t.Fatalf("speed sources = %+v, want both from the engine", rec.Sources)
	}
	wantFloat(t, "time to first token", rec.TTFTMs, 10) // the first frame, one clock step
	if !rec.Thinking || rec.ThinkingTokens == nil || *rec.ThinkingTokens != 3 {
		t.Fatalf("thinking = %v with %v tokens, want true with 3 (5 tokens x 2 of 4 chunks, rounded)", rec.Thinking, rec.ThinkingTokens)
	}
	wantFloat(t, "thinking ms", rec.ThinkingMs, 20) // first thinking frame to first answer frame
	if rec.Sources.Thinking != perfSourceProxy || !rec.Streamed {
		t.Fatalf("thinking source = %q, streamed = %v; want proxy and true", rec.Sources.Thinking, rec.Streamed)
	}
}

func TestPerfTracker_OllamaGenerateBodyEstimatesThinkingShare(t *testing.T) {
	// One JSON body: 30 characters of thinking and 10 of answer, so a quarter
	// of the reply is the answer.
	rec, ok := measure(t, "application/json",
		`{"response":"0123456789","thinking":"012345678901234567890123456789","done":true,`+
			`"prompt_eval_count":40,"eval_count":8,"prompt_eval_duration":200000000,"eval_duration":400000000,"load_duration":0}`)
	if !ok {
		t.Fatal("no record")
	}
	if rec.Streamed {
		t.Fatal("a single body is not a stream")
	}
	wantFloat(t, "read tokens/s", rec.ReadTPS, 200)
	wantFloat(t, "write tokens/s", rec.WriteTPS, 20)
	wantFloat(t, "time to first token", rec.TTFTMs, 200) // load (0) plus prompt evaluation
	if rec.Sources.TTFT != perfSourceEngine {
		t.Fatalf("ttft source = %q, want engine", rec.Sources.TTFT)
	}
	if rec.ThinkingTokens == nil || *rec.ThinkingTokens != 6 {
		t.Fatalf("thinking tokens = %v, want 6 (three quarters of 8)", rec.ThinkingTokens)
	}
	wantFloat(t, "thinking ms", rec.ThinkingMs, 300) // three quarters of 400 ms
	if rec.Sources.Thinking != perfSourceEstimated {
		t.Fatalf("thinking source = %q, want estimated", rec.Sources.Thinking)
	}
	wantNil(t, "load ms", rec.LoadMs)
}

func TestPerfTracker_LlamaCppStreamReadsReasoningContentAndTimings(t *testing.T) {
	rec, ok := measure(t, "text/event-stream", sse(
		`{"choices":[{"delta":{"role":"assistant","content":null}}]}`,
		`{"choices":[{"delta":{"reasoning_content":"Hmm"}}]}`,
		`{"choices":[{"delta":{"content":"Four"}}]}`,
		`{"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":2},`+
			`"timings":{"prompt_n":100,"prompt_ms":50,"predicted_n":2,"predicted_ms":40}}`,
		`[DONE]`,
	))
	if !ok {
		t.Fatal("no record")
	}
	if rec.PromptTokens != 120 || rec.CompletionTokens != 2 {
		t.Fatalf("tokens = %d / %d, want the usage block's 120 / 2", rec.PromptTokens, rec.CompletionTokens)
	}
	// llama.cpp evaluated 100 of the 120 prompt tokens; the rest were cached.
	wantFloat(t, "read tokens/s", rec.ReadTPS, 2000)
	wantFloat(t, "write tokens/s", rec.WriteTPS, 50)
	wantFloat(t, "time to first token", rec.TTFTMs, 20) // role-only first chunk is not a token
	if rec.ThinkingTokens == nil || *rec.ThinkingTokens != 1 {
		t.Fatalf("thinking tokens = %v, want 1", rec.ThinkingTokens)
	}
	wantFloat(t, "thinking ms", rec.ThinkingMs, 10)
}

func TestPerfTracker_ThinkTagsInContentCountAsThinking(t *testing.T) {
	// No usage block (a stream outside a session is forwarded as sent), so
	// tokens are counted from the chunks and speeds come from the proxy clock.
	rec, ok := measure(t, "text/event-stream", sse(
		`{"choices":[{"delta":{"content":"<think>"}}]}`,
		`{"choices":[{"delta":{"content":"step one"}}]}`,
		`{"choices":[{"delta":{"content":"</think>"}}]}`,
		`{"choices":[{"delta":{"content":"\n\n"}}]}`,
		`{"choices":[{"delta":{"content":"Done"}}]}`,
		`[DONE]`,
	))
	if !ok {
		t.Fatal("no record")
	}
	if rec.CompletionTokens != 5 || rec.Sources.Tokens != perfSourceEstimated {
		t.Fatalf("completion = %d (%s), want 5 counted chunks, estimated", rec.CompletionTokens, rec.Sources.Tokens)
	}
	if rec.ThinkingTokens == nil || *rec.ThinkingTokens != 3 {
		t.Fatalf("thinking tokens = %v, want the 3 chunks from <think> to </think>", rec.ThinkingTokens)
	}
	// Thinking runs from the <think> chunk (t=10) to "Done" (t=50): the
	// whitespace between does not start the answer.
	wantFloat(t, "thinking ms", rec.ThinkingMs, 40)
	// Four tokens after the first, over the 40 ms from first to last chunk.
	wantFloat(t, "write tokens/s", rec.WriteTPS, 100)
	if rec.Sources.Write != perfSourceProxy {
		t.Fatalf("write source = %q, want proxy", rec.Sources.Write)
	}
	wantNil(t, "read tokens/s", rec.ReadTPS) // no prompt count to divide
}

func TestPerfTracker_OllamaOpenAIReasoningFieldCountsAsThinking(t *testing.T) {
	rec, ok := measure(t, "text/event-stream", sse(
		`{"choices":[{"delta":{"role":"assistant","content":"","reasoning":"ok"}}]}`,
		`{"choices":[{"delta":{"content":"Yes"}}]}`,
		`{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}`,
	))
	if !ok || !rec.Thinking {
		t.Fatalf("record = %+v (found %v), want a thinking reply", rec, ok)
	}
	// Dispatch to first token is 10 ms, so 10 prompt tokens read at 1000/s on
	// the proxy's clock.
	wantFloat(t, "read tokens/s", rec.ReadTPS, 1000)
	if rec.Sources.Read != perfSourceProxy {
		t.Fatalf("read source = %q, want proxy", rec.Sources.Read)
	}
}

func TestPerfTracker_AnthropicStreamReadsThinkingDeltas(t *testing.T) {
	rec, ok := measure(t, "text/event-stream",
		"event: message_start\n"+sse(`{"type":"message_start","message":{"content":[],"usage":{"input_tokens":30,"output_tokens":1}}}`)+
			sse(
				`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Consider"}}`,
				`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}`,
			))
	if !ok {
		t.Fatal("no record")
	}
	if rec.PromptTokens != 30 || rec.CompletionTokens != 12 {
		t.Fatalf("tokens = %d / %d, want 30 / 12", rec.PromptTokens, rec.CompletionTokens)
	}
	if rec.ThinkingTokens == nil || *rec.ThinkingTokens != 6 {
		t.Fatalf("thinking tokens = %v, want 6 (12 x 1 of 2 chunks)", rec.ThinkingTokens)
	}
}

func TestPerfTracker_NonThinkingBodyWithoutTimingsHasNoSpeeds(t *testing.T) {
	rec, ok := measure(t, "application/json",
		`{"choices":[{"message":{"role":"assistant","content":"Hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":1}}`)
	if !ok {
		t.Fatal("no record")
	}
	if rec.Thinking || rec.ThinkingTokens != nil {
		t.Fatalf("thinking = %v / %v, want none", rec.Thinking, rec.ThinkingTokens)
	}
	wantNil(t, "read tokens/s", rec.ReadTPS)
	wantNil(t, "write tokens/s", rec.WriteTPS)
	wantNil(t, "time to first token", rec.TTFTMs)
	// The body was measured at the first clock step and ended at the second.
	if rec.TotalMs != 20 {
		t.Fatalf("total = %v ms, want 20 (dispatch to end of body)", rec.TotalMs)
	}
}

func TestPerfTracker_EmptyReplyProducesNoRecord(t *testing.T) {
	if _, ok := measure(t, "application/json", `{"error":"nope"}`); ok {
		t.Fatal("a reply with no output and no counts produced a record")
	}
}

func TestPerfStore_SummaryBoundsWindowAndKeepsBenchmarksApart(t *testing.T) {
	s := newPerfStore()
	speed := func(v float64) *float64 { return &v }
	for i := 1; i <= perfWindow+20; i++ {
		s.add(perfRecord{Engine: "ollama", Model: "m", NodeID: "n1", At: int64(i), TotalMs: 1,
			PromptTokens: 10, CompletionTokens: 2, WriteTPS: speed(float64(i))}, false)
	}
	s.add(perfRecord{Engine: "ollama", Model: "m", NodeID: "n1", At: 9999, WriteTPS: speed(1e6)}, true)

	rep := s.report()
	if len(rep.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(rep.Models))
	}
	m := rep.Models[0]
	if m.Requests != perfWindow || m.PromptTokens != 10*perfWindow {
		t.Fatalf("requests = %d, prompt tokens = %d; want the last %d requests only", m.Requests, m.PromptTokens, perfWindow)
	}
	// Speeds 21..120 remain: the benchmark's huge value is not among them.
	if m.WriteTPS == nil || m.WriteTPS.Min != 21 || m.WriteTPS.Max != 120 || m.WriteTPS.Avg != 70.5 {
		t.Fatalf("write stats = %+v, want min 21, max 120, avg 70.5", m.WriteTPS)
	}
	if m.WriteTPS.P50 != 70.5 || m.WriteTPS.P95 != 115.05 {
		t.Fatalf("write percentiles = p50 %v, p95 %v; want 70.5 and 115.05", m.WriteTPS.P50, m.WriteTPS.P95)
	}
	if len(m.Trend) != perfTrendPoints || m.Trend[len(m.Trend)-1].At != perfWindow+20 {
		t.Fatalf("trend has %d points ending at %d, want %d ending at the latest request", len(m.Trend), m.Trend[len(m.Trend)-1].At, perfTrendPoints)
	}
	if len(rep.Recent) != perfReportRecent || rep.Recent[0].At != perfWindow+20 {
		t.Fatalf("recent = %d newest-first, want %d starting with the latest", len(rep.Recent), perfReportRecent)
	}

	s.clear()
	if rep := s.report(); len(rep.Models) != 0 || len(rep.Recent) != 0 {
		t.Fatalf("after clear: %d models, %d recent; want none", len(rep.Models), len(rep.Recent))
	}
}

func TestPercentile_InterpolatesBetweenRanks(t *testing.T) {
	cases := []struct {
		name   string
		sorted []float64
		q      float64
		want   float64
	}{
		{"single value", []float64{7}, 0.95, 7},
		{"median of two", []float64{10, 20}, 0.5, 15},
		{"p95 of five", []float64{1, 2, 3, 4, 5}, 0.95, 4.8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := percentile(tc.sorted, tc.q); fmt.Sprintf("%.4f", got) != fmt.Sprintf("%.4f", tc.want) {
				t.Fatalf("percentile = %v, want %v", got, tc.want)
			}
		})
	}
}

// ollamaChatStream is a short streamed /api/chat reply with thinking.
const ollamaChatStream = `{"message":{"role":"assistant","content":"","thinking":"Hmm"},"done":false}` + "\n" +
	`{"message":{"role":"assistant","content":"Paris."},"done":false}` + "\n" +
	`{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":12,"eval_count":4,` +
	`"prompt_eval_duration":60000000,"eval_duration":80000000}` + "\n"

func TestPerformanceAPI_ReportsRoutedRequests(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = io.WriteString(w, ollamaChatStream)
		})
		fx := newAPIFixture(t, tc, upstream)
		path := "/v1/chat/completions"
		if tc.profile.Name == "ollama" {
			path = "/api/chat"
		}
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(fmt.Sprintf(`{"model":%q}`, tc.requestedModel)))
		req.RemoteAddr = "127.0.0.1:50000"
		w := httptest.NewRecorder()
		fx.proxy.soleFacade().handlePlain(w, req)
		wantStatus(t, w, http.StatusOK)

		res := fx.do(http.MethodGet, performanceAPIPath, "", "")
		wantStatus(t, res, http.StatusOK)
		var rep perfReport
		if err := json.Unmarshal(res.Body.Bytes(), &rep); err != nil {
			t.Fatalf("decode report: %v", err)
		}
		if rep.Scope != perfScope || len(rep.Models) != 1 || len(rep.Prompts) != len(benchmarkPrompts) {
			t.Fatalf("report = %+v, want this node's scope, one model and the benchmark prompts", rep)
		}
		m := rep.Models[0]
		if m.Model != tc.requestedModel || m.NodeID != "node-a" || m.Requests != 1 || m.ThinkingRequests != 1 {
			t.Fatalf("model summary = %+v, want one thinking request for %s on node-a", m, tc.requestedModel)
		}
		if m.ReadTPS == nil || m.ReadTPS.Avg != 200 || m.WriteTPS == nil || m.WriteTPS.Avg != 50 {
			t.Fatalf("speeds = read %+v, write %+v; want 200 and 50 tokens/s", m.ReadTPS, m.WriteTPS)
		}

		del := fx.do(http.MethodDelete, performanceAPIPath, "", "")
		wantStatus(t, del, http.StatusNoContent)
		if rep := fx.proxy.perf.report(); len(rep.Models) != 0 {
			t.Fatalf("after DELETE: %d models, want none", len(rep.Models))
		}
	})
}

func TestPerformanceAPI_BenchmarkHeaderKeepsRequestOutOfLiveFigures(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		var forwardedMark string
		upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			forwardedMark = r.Header.Get(benchmarkHeader)
			_, _ = io.Copy(io.Discard, r.Body)
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = io.WriteString(w, ollamaChatStream)
		})
		fx := newAPIFixture(t, tc, upstream)
		ch := fx.proxy.perf.wait("bench-1")
		path := "/v1/chat/completions"
		if tc.profile.Name == "ollama" {
			path = "/api/chat"
		}
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(fmt.Sprintf(`{"model":%q}`, tc.requestedModel)))
		req.RemoteAddr = "127.0.0.1:50000"
		req.Header.Set(benchmarkHeader, "1")
		req.Header.Set(requestIDHeader, "bench-1")
		w := httptest.NewRecorder()
		fx.proxy.soleFacade().handlePlain(w, req)
		wantStatus(t, w, http.StatusOK)

		select {
		case rec := <-ch:
			if rec.RequestID != "bench-1" || rec.CompletionTokens != 4 {
				t.Fatalf("benchmark record = %+v, want bench-1 with 4 tokens", rec)
			}
		default:
			t.Fatal("the benchmark waiter received nothing")
		}
		if rep := fx.proxy.perf.report(); len(rep.Models) != 0 {
			t.Fatalf("live figures include the benchmark request: %+v", rep.Models)
		}
		if forwardedMark != "" {
			t.Fatalf("the engine received %s: %q, want PAIR's mark stripped", benchmarkHeader, forwardedMark)
		}
	})
}

func TestBenchmarkRequest_UsesEngineDialectAndThinkingSwitch(t *testing.T) {
	thinking := benchmarkPrompts[3]
	cases := []struct {
		name, engine, mode, wantPath string
		check                        func(t *testing.T, body map[string]any)
	}{
		{"ollama thinking on", "ollama", "on", "/api/chat", func(t *testing.T, b map[string]any) {
			if b["think"] != true || b["stream"] != true {
				t.Fatalf("body = %v, want think true and a stream", b)
			}
		}},
		{"ollama default sends no switch", "ollama", "default", "/api/chat", func(t *testing.T, b map[string]any) {
			if _, ok := b["think"]; ok {
				t.Fatalf("body = %v, want no think field", b)
			}
		}},
		{"llama.cpp thinking off", "llamacpp", "off", "/v1/chat/completions", func(t *testing.T, b map[string]any) {
			kw, _ := b["chat_template_kwargs"].(map[string]any)
			if kw["enable_thinking"] != false {
				t.Fatalf("body = %v, want enable_thinking false", b)
			}
			opts, _ := b["stream_options"].(map[string]any)
			if opts["include_usage"] != true {
				t.Fatalf("body = %v, want usage requested", b)
			}
		}},
		{"lm studio has no switch", "lmstudio", "default", "/v1/chat/completions", func(t *testing.T, b map[string]any) {
			if _, ok := b["chat_template_kwargs"]; ok {
				t.Fatalf("body = %v, want no template switch", b)
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path, raw := benchmarkRequest(tc.engine, "bench-x", "m", thinking, tc.mode)
			if path != tc.wantPath {
				t.Fatalf("path = %s, want %s", path, tc.wantPath)
			}
			var body map[string]any
			if err := json.Unmarshal(raw, &body); err != nil {
				t.Fatalf("decode: %v", err)
			}
			msgs, _ := body["messages"].([]any)
			first, _ := msgs[0].(map[string]any)
			if content, _ := first["content"].(string); !strings.HasPrefix(content, "Benchmark run bench-x.") {
				t.Fatalf("prompt = %q, want it to start with the run ID so no prompt cache answers it", content)
			}
			tc.check(t, body)
		})
	}
}

func TestBenchmarkReport_IsAboutAThousandTokens(t *testing.T) {
	// Roughly four characters a token; the step is documented as about a
	// thousand tokens, and must fit an engine's smallest default window.
	n := len(benchmarkReport())
	if n < 3000 || n > 6000 {
		t.Fatalf("report is %d characters, want 3000-6000 (about 750-1500 tokens)", n)
	}
}

func TestPerfStore_OneBenchmarkAtATime(t *testing.T) {
	s := newPerfStore()
	run, err := s.beginRun("ollama", "m")
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	if _, err := s.beginRun("ollama", "m"); err != errBenchmarkRunning {
		t.Fatalf("second run err = %v, want errBenchmarkRunning", err)
	}
	s.endRun(run, benchDone, "")
	if _, err := s.beginRun("ollama", "m"); err != nil {
		t.Fatalf("run after the first finished: %v", err)
	}
	runs := s.benchmarkSnapshot()
	if len(runs) != 2 || runs[1].State != benchDone || len(runs[1].Steps) != len(benchmarkPrompts) {
		t.Fatalf("runs = %+v, want two, the older done with every step", runs)
	}
}

func TestRunBenchmark_RunsEveryStepThroughTheFacade(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			// This engine has no thinking switch for the model.
			if strings.Contains(string(body), `"think":true`) || strings.Contains(string(body), `"enable_thinking":true`) {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = io.WriteString(w, `{"error":"model does not support thinking"}`)
				return
			}
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = io.WriteString(w, ollamaChatStream)
		})
		fx := newAPIFixture(t, tc, upstream)
		srv := httptest.NewServer(http.HandlerFunc(fx.proxy.soleFacade().handlePlain))
		t.Cleanup(srv.Close)
		port := srv.Listener.Addr().(*net.TCPAddr).Port

		run, err := fx.proxy.perf.beginRun(tc.profile.Name, tc.requestedModel)
		if err != nil {
			t.Fatalf("begin: %v", err)
		}
		fx.proxy.runBenchmark(run, tc.profile.Name, port)

		got := fx.proxy.perf.benchmarkSnapshot()[0]
		if got.State != benchDone {
			t.Fatalf("run state = %s (%s), want done", got.State, got.Error)
		}
		for _, step := range got.Steps {
			wantState := benchDone
			if step.Thinking == "on" {
				wantState = benchSkipped
			}
			if step.State != wantState {
				t.Fatalf("step %s = %s (%s), want %s", step.Name, step.State, step.Detail, wantState)
			}
			if wantState == benchDone && (step.Result == nil || step.Result.CompletionTokens != 4) {
				t.Fatalf("step %s result = %+v, want the engine's 4 tokens", step.Name, step.Result)
			}
		}
		if rep := fx.proxy.perf.report(); len(rep.Models) != 0 {
			t.Fatalf("live figures include benchmark requests: %+v", rep.Models)
		}
	})
}

func TestPerfTracker_ReplyEndingInAnErrorProducesNoRecord(t *testing.T) {
	cases := []struct{ name, contentType, body string }{
		{"ollama aborted stream", "application/x-ndjson", ndjson(
			`{"message":{"role":"assistant","content":"so"},"done":false}`,
			`{"error":"prediction aborted, token repeat limit reached"}`,
		)},
		{"anthropic error event", "text/event-stream", sse(
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
			`{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`,
		)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if rec, ok := measure(t, tc.contentType, tc.body); ok {
				t.Fatalf("an aborted reply produced a record: %+v", rec)
			}
		})
	}
}
