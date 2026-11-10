// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestUsageCounts(t *testing.T) {
	counts := func(name, doc string, wantPrompt, wantCompletion int64, wantOK bool) {
		t.Run(name, func(t *testing.T) {
			p, c, ok := usageCounts([]byte(doc))
			if ok != wantOK || p != wantPrompt || c != wantCompletion {
				t.Fatalf("usageCounts = (%d, %d, %v), want (%d, %d, %v)", p, c, ok, wantPrompt, wantCompletion, wantOK)
			}
		})
	}
	counts("ollama final object", `{"model":"m","done":true,"prompt_eval_count":812,"eval_count":96}`, 812, 96, true)
	counts("openai usage", `{"choices":[],"usage":{"prompt_tokens":1200,"completion_tokens":40,"total_tokens":1240}}`, 1200, 40, true)
	counts("anthropic message_start", `{"type":"message_start","message":{"usage":{"input_tokens":300,"output_tokens":1}}}`, 300, 1, true)
	counts("anthropic message_delta", `{"type":"message_delta","usage":{"output_tokens":75}}`, 0, 75, true)
	counts("anthropic counts cached input", `{"usage":{"input_tokens":10,"cache_read_input_tokens":900,"cache_creation_input_tokens":90,"output_tokens":5}}`, 1000, 5, true)
	counts("content chunk without usage", `{"choices":[{"delta":{"content":"hi"}}]}`, 0, 0, false)
	counts("negative counts read as zero", `{"prompt_eval_count":-4,"eval_count":7}`, 0, 7, true)
	counts("not json", `data: nope`, 0, 0, false)
}

// readThrough drains body through a usageReader in small, uneven reads, the
// way a stream arrives, and returns what passed through and what was found.
func readThrough(t *testing.T, contentType, body string) (string, *tokenUsage) {
	t.Helper()
	var got *tokenUsage
	r := newUsageReader(io.NopCloser(&chunkedReader{data: []byte(body), sizes: []int{1, 7, 3, 64, 2}}),
		contentType, func(u tokenUsage) { got = &u })
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	return string(out), got
}

// chunkedReader returns data in a repeating pattern of read sizes.
type chunkedReader struct {
	data  []byte
	sizes []int
	i     int
}

func (c *chunkedReader) Read(p []byte) (int, error) {
	if len(c.data) == 0 {
		return 0, io.EOF
	}
	n := c.sizes[c.i%len(c.sizes)]
	c.i++
	n = min(n, len(p), len(c.data))
	copy(p, c.data[:n])
	c.data = c.data[n:]
	return n, nil
}

func TestUsageReader(t *testing.T) {
	reads := func(name, contentType, body string, want *tokenUsage) {
		t.Run(name, func(t *testing.T) {
			out, got := readThrough(t, contentType, body)
			if out != body {
				t.Fatalf("body changed in transit:\n got %q\nwant %q", out, body)
			}
			switch {
			case want == nil && got != nil:
				t.Fatalf("usage = %+v, want none", *got)
			case want != nil && got == nil:
				t.Fatalf("usage not found, want %+v", *want)
			case want != nil && *got != *want:
				t.Fatalf("usage = %+v, want %+v", *got, *want)
			}
		})
	}

	reads("ollama ndjson stream", "application/x-ndjson",
		`{"message":{"content":"Hel"},"done":false}`+"\n"+
			`{"message":{"content":"lo"},"done":false}`+"\n"+
			`{"done":true,"prompt_eval_count":512,"eval_count":2}`+"\n",
		&tokenUsage{Prompt: 512, Completion: 2})
	reads("ndjson final line without newline", "application/x-ndjson",
		`{"done":false}`+"\n"+`{"done":true,"prompt_eval_count":5,"eval_count":1}`,
		&tokenUsage{Prompt: 5, Completion: 1})
	reads("openai sse with usage chunk", "text/event-stream; charset=utf-8",
		"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"+
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":12}}\n\n"+
			"data: [DONE]\n\n",
		&tokenUsage{Prompt: 900, Completion: 12})
	reads("anthropic sse spreads counts across events", "text/event-stream",
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":300,\"output_tokens\":1}}}\n\n"+
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"x\"}}\n\n"+
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":75}}\n\n",
		&tokenUsage{Prompt: 300, Completion: 75})
	reads("non-streamed openai json", "application/json",
		`{"id":"x","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":40,"completion_tokens":3}}`,
		&tokenUsage{Prompt: 40, Completion: 3})
	reads("stream without usage", "text/event-stream",
		"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n", nil)
	reads("unrelated content type", "text/plain", `{"prompt_eval_count":1}`, nil)

	t.Run("an oversized frame is skipped, not parsed", func(t *testing.T) {
		big := `{"usage":{"prompt_tokens":1},"pad":"` + strings.Repeat("x", maxUsageFrame) + `"}`
		_, got := readThrough(t, "application/x-ndjson", big+"\n"+`{"done":true,"prompt_eval_count":9,"eval_count":2}`+"\n")
		if got == nil || *got != (tokenUsage{Prompt: 9, Completion: 2}) {
			t.Fatalf("usage = %v, want the later frame's counts only", got)
		}
	})
	t.Run("an oversized json body reports nothing", func(t *testing.T) {
		big := `{"usage":{"prompt_tokens":1,"completion_tokens":1},"pad":"` + strings.Repeat("x", maxUsageBody) + `"}`
		if _, got := readThrough(t, "application/json", big); got != nil {
			t.Fatalf("usage = %+v, want none past the body bound", *got)
		}
	})
	t.Run("a stream cut short reports nothing", func(t *testing.T) {
		var got *tokenUsage
		r := newUsageReader(io.NopCloser(strings.NewReader(`{"done":true,"prompt_eval_count":5,"eval_count":1}`)),
			"application/x-ndjson", func(u tokenUsage) { got = &u })
		buf := make([]byte, 8)
		if _, err := r.Read(buf); err != nil {
			t.Fatalf("read: %v", err)
		}
		if err := r.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
		if got != nil {
			t.Fatalf("usage = %+v from an unfinished stream, want none", *got)
		}
	})
}

func TestRequestStreamsOpenAI(t *testing.T) {
	check := func(name, body string, want bool) {
		t.Run(name, func(t *testing.T) {
			if got := requestStreamsOpenAI([]byte(body)); got != want {
				t.Fatalf("requestStreamsOpenAI(%s) = %v, want %v", body, got, want)
			}
		})
	}
	check("stream without options", `{"model":"m","stream":true}`, true)
	check("stream with other options", `{"stream":true,"stream_options":{"continuous_usage_stats":true}}`, true)
	check("client already asked for usage", `{"stream":true,"stream_options":{"include_usage":true}}`, false)
	check("client declined usage", `{"stream":true,"stream_options":{"include_usage":false}}`, false)
	check("not streaming", `{"model":"m"}`, false)
	check("not json", `nope`, false)
}

func TestWithStreamUsage(t *testing.T) {
	in := `{"model":"m","stream":true,"messages":[{"role":"user","content":"a < b & c"}],"stream_options":{"continuous_usage_stats":true}}`
	out := withStreamUsage([]byte(in))
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("rewritten body is not JSON: %v (%s)", err, out)
	}
	var want map[string]any
	if err := json.Unmarshal([]byte(in), &want); err != nil {
		t.Fatalf("decode input: %v", err)
	}
	want["stream_options"].(map[string]any)["include_usage"] = true
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("rewritten body\n got %v\nwant %v", got, want)
	}
	if bytes.Contains(out, []byte("\\u003c")) || !bytes.Contains(out, []byte("a < b & c")) {
		t.Fatalf("rewrite HTML-escaped the client's content: %s", out)
	}
	if bad := withStreamUsage([]byte(`nope`)); string(bad) != "nope" {
		t.Fatalf("a body that is not JSON must pass through unchanged, got %q", bad)
	}
}

// The proxy asks a session's OpenAI stream for usage, leaves other requests
// exactly as sent, and records what the engine reports on the session.
func TestHandleHTTP_SessionContextFromEngineUsage(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		var mu sync.Mutex
		var lastBody []byte
		upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, err := io.ReadAll(r.Body)
			if err != nil {
				t.Errorf("upstream read: %v", err)
			}
			mu.Lock()
			lastBody = b
			mu.Unlock()
			w.Header().Set("Content-Type", "application/x-ndjson")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, `{"done":false}`+"\n"+`{"done":true,"prompt_eval_count":1500,"eval_count":250}`+"\n")
		})
		fx := newAPIFixture(t, tc, upstream)

		send := func(path, body, session string) {
			t.Helper()
			req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
			req.RemoteAddr = "127.0.0.1:50000"
			if session != "" {
				req.Header.Set(sessionIDHeader, session)
			}
			w := httptest.NewRecorder()
			fx.proxy.soleFacade().handlePlain(w, req)
			wantStatus(t, w, http.StatusOK)
		}
		body := func() string {
			mu.Lock()
			defer mu.Unlock()
			return string(lastBody)
		}

		t.Run("records the engine's counts on the session", func(t *testing.T) {
			send(tc.inferencePath, fmt.Sprintf(`{"model":%q}`, tc.requestedModel), "conv-1")
			got := decodeSession(t, fx.do(http.MethodGet, sessionsAPIPrefix+"/conv-1", "", ""))
			c, ok := got.Context[tc.profile.Name]
			if !ok {
				t.Fatalf("session context = %v, want an entry for %s", got.Context, tc.profile.Name)
			}
			if c.Tokens != 1750 || c.PromptTokens != 1500 || c.CompletionTokens != 250 || c.Model != tc.requestedModel {
				t.Fatalf("context = %+v, want 1750 tokens (1500 + 250) for %s", c, tc.requestedModel)
			}
		})

		openAIStream := fmt.Sprintf(`{"model":%q,"stream":true}`, tc.requestedModel)
		t.Run("asks a session's OpenAI stream for usage", func(t *testing.T) {
			send("/v1/chat/completions", openAIStream, "conv-2")
			var sent struct {
				StreamOptions struct {
					IncludeUsage bool `json:"include_usage"`
				} `json:"stream_options"`
			}
			if err := json.Unmarshal([]byte(body()), &sent); err != nil {
				t.Fatalf("decode forwarded body: %v", err)
			}
			if !sent.StreamOptions.IncludeUsage {
				t.Fatalf("forwarded body %s lacks stream_options.include_usage", body())
			}
		})
		t.Run("forwards a request outside a session unchanged", func(t *testing.T) {
			send("/v1/chat/completions", openAIStream, "")
			if body() != openAIStream {
				t.Fatalf("forwarded body = %s, want it exactly as sent: %s", body(), openAIStream)
			}
		})
	})
}
