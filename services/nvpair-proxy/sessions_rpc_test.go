// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"testing"
)

// rpcCall sends one request to the proxy as the broker relays it (engine
// addressed) and returns the decoded result, or the error message.
func rpcCall(t *testing.T, p *Proxy, rec *recRW, method, params string) (json.RawMessage, string) {
	t.Helper()
	rec.mu.Lock()
	rec.b = nil
	rec.mu.Unlock()
	id := json.RawMessage(`1`)
	p.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: "ollama:" + method, Params: json.RawMessage(params)})
	rec.mu.Lock()
	defer rec.mu.Unlock()
	var resp struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.b, &resp); err != nil {
		t.Fatalf("%s: decode response %q: %v", method, rec.b, err)
	}
	if resp.Error != nil {
		return nil, resp.Error.Message
	}
	return resp.Result, ""
}

func TestSessionsRPC(t *testing.T) {
	rec := &recRW{}
	p := newTestProxy(anyProfile(t), NewCodec(rec), NewDiscovery(), 11435)
	beginT(t, p.sessions, "conv-1", "planner")

	t.Run("sessions/configure with no fields reads the policy", func(t *testing.T) {
		res, errMsg := rpcCall(t, p, rec, "sessions/configure", `{}`)
		var got map[string]int
		if errMsg != "" || json.Unmarshal(res, &got) != nil || got["idleTimeoutSeconds"] != 600 || got["queueTimeoutSeconds"] != 60 {
			t.Fatalf("result = %s, err %q; want the default policy", res, errMsg)
		}
	})
	t.Run("sessions/list returns this node's sessions", func(t *testing.T) {
		res, errMsg := rpcCall(t, p, rec, "sessions/list", `{}`)
		var got struct {
			Sessions []sessionJSON `json:"sessions"`
		}
		if errMsg != "" || json.Unmarshal(res, &got) != nil || len(got.Sessions) != 1 || got.Sessions[0].SessionID != "conv-1" {
			t.Fatalf("result = %s, err %q; want conv-1", res, errMsg)
		}
	})
	t.Run("capacity/set and capacity/report", func(t *testing.T) {
		if _, errMsg := rpcCall(t, p, rec, "capacity/set", `{"engine":"ollama","model":"m","contextTokens":65536,"maxSessions":4}`); errMsg != "" {
			t.Fatalf("capacity/set: %s", errMsg)
		}
		res, errMsg := rpcCall(t, p, rec, "capacity/report", `{}`)
		var got struct {
			Models []modelCapacity `json:"models"`
		}
		if errMsg != "" || json.Unmarshal(res, &got) != nil || len(got.Models) != 1 || got.Models[0].SessionShare != 16384 {
			t.Fatalf("result = %s, err %q; want m with a 16384 share", res, errMsg)
		}
		if _, errMsg := rpcCall(t, p, rec, "capacity/set", `{"engine":"nope","model":"m","contextTokens":65536,"maxSessions":4}`); errMsg == "" {
			t.Fatal("capacity/set accepted an unknown engine")
		}
	})
	t.Run("sessions/end needs the owner", func(t *testing.T) {
		if _, errMsg := rpcCall(t, p, rec, "sessions/end", `{"sessionId":"conv-1","agentId":"coder"}`); errMsg == "" {
			t.Fatal("another agent ended the session")
		}
		if _, errMsg := rpcCall(t, p, rec, "sessions/end", `{"sessionId":"conv-1","agentId":"planner"}`); errMsg != "" {
			t.Fatalf("sessions/end: %s", errMsg)
		}
		if _, err := p.sessions.get("conv-1"); err == nil {
			t.Fatal("the session is still live after sessions/end")
		}
	})
}
