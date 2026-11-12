// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

// TestLiveOllamaBehavesAsTheBudgetAssumes checks, against a real Ollama, the
// engine behaviour the reply budget is built on and fakeOllama imitates:
//
//   - a prompt over the window is accepted and cut to the last half of the
//     window;
//   - with "truncate": false it is refused, naming its exact token count;
//   - options.num_predict caps the reply.
//
// It needs a running Ollama with a small model and is skipped unless
// PAIR_LIVE_OLLAMA_URL is set, for example:
//
//	PAIR_LIVE_OLLAMA_URL=http://127.0.0.1:11435 PAIR_LIVE_OLLAMA_MODEL=qwen3:0.6b \
//	  go test -run TestLiveOllama -v .
func TestLiveOllamaBehavesAsTheBudgetAssumes(t *testing.T) {
	base := os.Getenv("PAIR_LIVE_OLLAMA_URL")
	if base == "" {
		t.Skip("set PAIR_LIVE_OLLAMA_URL to run against a real Ollama")
	}
	model := os.Getenv("PAIR_LIVE_OLLAMA_MODEL")
	if model == "" {
		model = "qwen3:0.6b"
	}
	const window = 4096
	client := &http.Client{Timeout: 5 * time.Minute}
	chat := func(content string, extra map[string]any) (int, map[string]any) {
		t.Helper()
		body := map[string]any{
			"model": model, "stream": false, "think": false, "keep_alive": 0,
			"messages": []map[string]string{{"role": "user", "content": content}},
			"options":  map[string]any{"num_ctx": window, "num_predict": 1},
		}
		for k, v := range extra {
			body[k] = v
		}
		raw, _ := json.Marshal(body)
		resp, err := client.Post(base+"/api/chat", "application/json", bytes.NewReader(raw))
		if err != nil {
			t.Fatalf("post: %v", err)
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		var out map[string]any
		_ = json.Unmarshal(b, &out)
		if out == nil {
			out = map[string]any{"raw": string(b)}
		}
		return resp.StatusCode, out
	}
	var sb strings.Builder
	for i := 0; sb.Len() < 4*window*2; i++ {
		fmt.Fprintf(&sb, "Line %d: the quick brown fox jumps over the lazy dog. ", i)
	}
	oversized := sb.String() // about twice the window

	t.Run("an oversized prompt is silently cut to half the window", func(t *testing.T) {
		status, out := chat(oversized, nil)
		if status != http.StatusOK {
			t.Fatalf("status %d: %v", status, out)
		}
		if got, _ := out["prompt_eval_count"].(float64); got > window/2+16 {
			t.Fatalf("read %v tokens of a prompt over %d; expected Ollama's half-window cut", got, window)
		}
	})
	t.Run("truncate false refuses it with the exact count", func(t *testing.T) {
		status, out := chat(oversized, map[string]any{"truncate": false})
		msg, _ := out["error"].(string)
		if status != http.StatusBadRequest || !strings.Contains(msg, "exceeds the available context size") {
			t.Fatalf("status %d: %v; want 400 naming the context size", status, out)
		}
	})
	t.Run("num_predict caps the reply", func(t *testing.T) {
		status, out := chat("Write a long essay about foxes.", map[string]any{"options": map[string]any{"num_ctx": window, "num_predict": 64}})
		if status != http.StatusOK {
			t.Fatalf("status %d: %v", status, out)
		}
		if got, _ := out["eval_count"].(float64); got > 64 {
			t.Fatalf("reply of %v tokens with num_predict 64", got)
		}
	})
}
