// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"testing"
)

func decoded(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var v map[string]any
	if err := json.Unmarshal(body, &v); err != nil {
		t.Fatalf("not JSON: %s", body)
	}
	return v
}

func TestSessionBudgetOllamaStopsSilentCutsAndCapsReply(t *testing.T) {
	in := []byte(`{"model":"m","messages":[{"role":"user","content":"hi <b>"}],"options":{"temperature":0.2}}`)
	v := decoded(t, withSessionBudget(in, "ollama", "/api/chat", 8192))
	opts := v["options"].(map[string]any)
	if v["truncate"] != false || opts["num_predict"] != float64(8192) || opts["temperature"] != 0.2 {
		t.Fatalf("got %v", v)
	}
	if v["messages"].([]any)[0].(map[string]any)["content"] != "hi <b>" {
		t.Fatal("message content changed")
	}
}

func TestSessionBudgetKeepsWhatTheClientSet(t *testing.T) {
	in := []byte(`{"model":"m","prompt":"x","truncate":true,"options":{"num_predict":50}}`)
	if out := withSessionBudget(in, "ollama", "/api/generate", 8192); string(out) != string(in) {
		t.Fatalf("rewrote a body whose budget the client set: %s", out)
	}
	in = []byte(`{"model":"m","messages":[],"max_completion_tokens":10}`)
	if out := withSessionBudget(in, "llamacpp", "/v1/chat/completions", 8192); string(out) != string(in) {
		t.Fatalf("rewrote max_completion_tokens: %s", out)
	}
}

func TestSessionBudgetOpenAICapsReplyOnly(t *testing.T) {
	v := decoded(t, withSessionBudget([]byte(`{"model":"m","messages":[]}`), "ollama", "/v1/chat/completions", 4096))
	if v["max_tokens"] != float64(4096) || v["truncate"] != nil {
		t.Fatalf("got %v", v)
	}
}

func TestSessionBudgetWithoutReplyBudget(t *testing.T) {
	// Ollama still stops silent cuts; nothing to cap elsewhere.
	v := decoded(t, withSessionBudget([]byte(`{"model":"m","prompt":"x"}`), "ollama", "/api/generate", 0))
	if v["truncate"] != false || v["options"] != nil {
		t.Fatalf("got %v", v)
	}
	in := []byte(`{"model":"m","messages":[]}`)
	if out := withSessionBudget(in, "lmstudio", "/v1/chat/completions", 0); string(out) != string(in) {
		t.Fatalf("got %s", out)
	}
	if out := withSessionBudget([]byte(`not json`), "ollama", "/api/chat", 10); string(out) != "not json" {
		t.Fatal("rewrote a body that is not JSON")
	}
}
