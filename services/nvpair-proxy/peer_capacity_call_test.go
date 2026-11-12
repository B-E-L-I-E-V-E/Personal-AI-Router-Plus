// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"
)

// peerCapacityCall runs one capacity method on node-b from A, as the desktop
// on A asks for it.
func (fx *peerFixture) peerCapacityCall(method, params string) (json.RawMessage, error) {
	fx.t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	return fx.a.host.callCapacity(ctx, peerCapacityCallParams{
		NodeID: "node-b", Method: method, Params: json.RawMessage(params),
	})
}

// A cluster member sizes, saves, applies and clears another node's plan; the
// plan and the limit live on that node, and the caller's own are untouched.
func TestPeerCapacityCall_SaveApplyAndClearOnThePeer(t *testing.T) {
	tc := engineCases(t)[0]
	fx := newPeerFixture(t, tc)
	engine, model := tc.profile.Name, tc.requestedModel
	k := capacityKey{Engine: engine, Model: model}

	// Plan: worked out on B, for the hardware the caller reports for B.
	raw, err := fx.peerCapacityCall("capacity/plan",
		`{"engine":"`+engine+`","model":"`+model+`","gpus":[{"name":"GPU","vramBytes":25769803776}],"windowTokens":8192,"maxSessions":2,"weightsBytes":1073741824,"kvBytesPerToken":131072}`)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	var plan planResult
	if err := json.Unmarshal(raw, &plan); err != nil {
		t.Fatalf("plan result %s: %v", raw, err)
	}
	if plan.Spec.Model != model || plan.WindowTokens != 8192 || plan.TotalBytes != 25769803776 {
		t.Fatalf("plan = %+v; want %s sized for 24 GiB at 8192 tokens", plan, model)
	}

	// Save: the plan is B's, not A's.
	save := `{"save":{"engine":"` + engine + `","model":"` + model + `","windowTokens":8192,"sessions":2,"cacheType":"f16","reservePercent":10,"replyTokens":2048}}`
	if raw, err = fx.peerCapacityCall("capacity/planner-set", save); err != nil {
		t.Fatalf("save: %v", err)
	}
	if !strings.Contains(string(raw), `"plans"`) {
		t.Fatalf("save answered %s; want the planner state", raw)
	}
	if _, ok := fx.b.host.planner.find(engine, model); !ok {
		t.Fatal("the plan was not saved on B")
	}
	if _, ok := fx.a.host.planner.find(engine, model); ok {
		t.Fatal("the plan was saved on A")
	}

	// Applied: B enforces the plan's capacity.
	applied := `{"applied":{"engine":"` + engine + `","model":"` + model + `","windowTokens":8192,"sessions":2,"cacheType":"f16"}}`
	if _, err = fx.peerCapacityCall("capacity/planner-set", applied); err != nil {
		t.Fatalf("applied: %v", err)
	}
	set, ok := fx.b.host.capacity.settingsSnapshot()[k]
	if !ok || set.ContextTokens != 16384 || set.MaxSessions != 2 || set.ReplyTokens != 2048 {
		t.Fatalf("B's setting = %+v (present %v); want 2 sessions of 8192 with a 2048 reply", set, ok)
	}
	if _, ok := fx.a.host.capacity.settingsSnapshot()[k]; ok {
		t.Fatal("A's capacity changed")
	}

	// The planner and report read back from B.
	if raw, err = fx.peerCapacityCall("capacity/planner", `{}`); err != nil || !strings.Contains(string(raw), `"applied":true`) {
		t.Fatalf("planner = %s, %v; want the applied plan", raw, err)
	}
	if raw, err = fx.peerCapacityCall("capacity/report", ``); err != nil || !strings.Contains(string(raw), `"maxSessions":2`) {
		t.Fatalf("report = %s, %v; want 2 sessions", raw, err)
	}

	// Remove the plan and clear the limit, as Remove plan does.
	if _, err = fx.peerCapacityCall("capacity/planner-set", `{"remove":{"engine":"`+engine+`","model":"`+model+`"}}`); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if _, err = fx.peerCapacityCall("capacity/set", `{"engine":"`+engine+`","model":"`+model+`","clear":true}`); err != nil {
		t.Fatalf("clear: %v", err)
	}
	if _, ok := fx.b.host.planner.find(engine, model); ok {
		t.Fatal("B still has the plan")
	}
	if _, ok := fx.b.host.capacity.settingsSnapshot()[k]; ok {
		t.Fatal("B still has the limit")
	}
}

// A bad change is refused with B's own reason, and B keeps what it had.
func TestPeerCapacityCall_PeerRefusesABadChange(t *testing.T) {
	tc := engineCases(t)[0]
	fx := newPeerFixture(t, tc)
	fx.limitB(65536, 4, time.Second)
	_, err := fx.peerCapacityCall("capacity/set",
		`{"engine":"`+tc.profile.Name+`","model":"`+tc.requestedModel+`","contextTokens":8192,"maxSessions":2,"replyTokens":9000}`)
	if err == nil || !strings.Contains(err.Error(), "node-b refused") {
		t.Fatalf("err = %v; want node-b's refusal", err)
	}
	k := capacityKey{Engine: tc.profile.Name, Model: tc.requestedModel}
	if set := fx.b.host.capacity.settingsSnapshot()[k]; set.MaxSessions != 4 {
		t.Fatalf("B's setting = %+v; want it unchanged", set)
	}
}

func TestPeerCapacityCall_Rejects(t *testing.T) {
	tc := engineCases(t)[0]
	fx := newPeerFixture(t, tc)
	ctx := context.Background()

	if _, err := fx.a.host.callCapacity(ctx, peerCapacityCallParams{NodeID: "node-b", Method: "sessions/list"}); err == nil {
		t.Error("a method outside capacity was forwarded")
	}
	if _, err := fx.a.host.callCapacity(ctx, peerCapacityCallParams{NodeID: "node-z", Method: "capacity/planner"}); err == nil {
		t.Error("an unknown node answered")
	}
	// The ingress itself takes only POST, and only capacity methods.
	if resp := fx.peerRequest(http.MethodGet, peerCapacityCallPath, ""); resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("GET = %d; want 405", resp.StatusCode)
	}
	if resp := fx.peerRequest(http.MethodPost, peerCapacityCallPath, `{"method":"node/state","params":{"state":"stopped"}}`); resp.StatusCode != http.StatusBadRequest {
		t.Errorf("node/state through the capacity call = %d; want 400", resp.StatusCode)
	}
	if st := fx.b.host.node.current(); st != nodeRunning {
		t.Errorf("B's state = %s; want running", st)
	}
}

// No nodeId, or this node's own, runs here without the network.
func TestPeerCapacityCall_SelfRunsLocally(t *testing.T) {
	tc := engineCases(t)[0]
	fx := newPeerFixture(t, tc)
	raw, err := fx.a.host.callCapacity(context.Background(), peerCapacityCallParams{
		Method: "capacity/planner-set",
		Params: json.RawMessage(`{"save":{"engine":"` + tc.profile.Name + `","model":"` + tc.requestedModel + `","windowTokens":4096,"sessions":1,"cacheType":"f16","reservePercent":10}}`),
	})
	if err != nil || !strings.Contains(string(raw), `"plans"`) {
		t.Fatalf("local save = %s, %v", raw, err)
	}
	if _, ok := fx.a.host.planner.find(tc.profile.Name, tc.requestedModel); !ok {
		t.Fatal("the plan was not saved here")
	}
	if _, ok := fx.b.host.planner.find(tc.profile.Name, tc.requestedModel); ok {
		t.Fatal("the plan reached B")
	}
}

func TestPeerCapacityCallRPC(t *testing.T) {
	tc := engineCases(t)[0]
	fx := newPeerFixture(t, tc)
	fx.limitB(65536, 2, time.Second)
	rec := &recRW{}
	fx.a.host.codec = NewCodec(rec)

	if _, errMsg := rpcCall(t, fx.a.host, rec, "capacity/peer-call", `{"nodeId":"node-b"}`); errMsg == "" {
		t.Fatal("capacity/peer-call accepted no method")
	}
	rec.mu.Lock()
	rec.b = nil
	rec.mu.Unlock()
	id := json.RawMessage(`9`)
	fx.a.host.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: tc.profile.Name + ":capacity/peer-call",
		Params: json.RawMessage(`{"nodeId":"node-b","method":"capacity/report"}`)})
	deadline := time.Now().Add(10 * time.Second)
	for !rec.has(`"id":9`) {
		if time.Now().After(deadline) {
			t.Fatal("no reply to capacity/peer-call")
		}
		time.Sleep(10 * time.Millisecond)
	}
	rec.mu.Lock()
	raw := append([]byte(nil), rec.b...)
	rec.mu.Unlock()
	var resp struct {
		Result struct {
			Models []modelCapacity `json:"models"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil || resp.Error != nil {
		t.Fatalf("reply %s: decode err %v", raw, err)
	}
	if len(resp.Result.Models) == 0 || resp.Result.Models[0].MaxSessions != 2 {
		t.Fatalf("result = %+v; want B's 2 sessions", resp.Result)
	}
}
