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

// ingressURL is the address A dials for B's mTLS ingress.
func (fx *peerFixture) ingressURL() string {
	fx.t.Helper()
	for _, n := range fx.a.discovery.Nodes() {
		if n.ID == "node-b" {
			u := fx.a.targetURL(n)
			u.Scheme = "https"
			return u.String()
		}
	}
	fx.t.Fatal("node-b is not in A's discovery")
	return ""
}

// peerRequest sends one request to B's ingress with A's client certificate.
func (fx *peerFixture) peerRequest(method, path, body string) *http.Response {
	fx.t.Helper()
	req, err := http.NewRequest(method, fx.ingressURL()+path, strings.NewReader(body))
	if err != nil {
		fx.t.Fatalf("new request: %v", err)
	}
	client := &http.Client{Transport: fx.a.host.peerHTTPTransport("b"), Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		fx.t.Fatalf("%s %s: %v", method, path, err)
	}
	fx.t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

func TestPeerCapacity_ReadsWhatThePeerEnforces(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newPeerFixture(t, tc)
		fx.limitBWithReply(65536, 4, 8192, time.Second)

		rows, err := fx.a.host.peerCapacityReport(context.Background(), "node-b")
		if err != nil {
			t.Fatalf("peerCapacityReport: %v", err)
		}
		var got *modelCapacity
		for i := range rows {
			if rows[i].Engine == tc.profile.Name && rows[i].Model == tc.requestedModel {
				got = &rows[i]
			}
		}
		if got == nil {
			t.Fatalf("rows = %+v; want %s/%s", rows, tc.profile.Name, tc.requestedModel)
		}
		if got.MaxSessions != 4 || got.SessionShare != 16384 || got.ReplyTokens != 8192 || got.Source != "configured" {
			t.Fatalf("row = %+v; want 4 sessions of 16384 with an 8192 reply budget", got)
		}
	})
}

func TestPeerCapacity_IngressIsReadOnly(t *testing.T) {
	tc := engineCases(t)[0]
	fx := newPeerFixture(t, tc)
	fx.limitB(65536, 4, time.Second)
	k := capacityKey{Engine: tc.profile.Name, Model: tc.requestedModel}

	for _, method := range []string{http.MethodPut, http.MethodPost, http.MethodDelete} {
		body := `{"engine":"` + tc.profile.Name + `","model":"` + tc.requestedModel + `","clear":true}`
		if resp := fx.peerRequest(method, peerCapacityPath, body); resp.StatusCode != http.StatusMethodNotAllowed {
			t.Errorf("%s %s = %d; want 405", method, peerCapacityPath, resp.StatusCode)
		}
	}
	// The loopback-only capacity API is not answered on the ingress: there it
	// is just a path forwarded to the engine.
	body := `{"engine":"` + tc.profile.Name + `","model":"` + tc.requestedModel + `","contextTokens":8192,"maxSessions":1}`
	fx.peerRequest(http.MethodPut, capacityAPIPath, body)

	if set, ok := fx.b.host.capacity.settingsSnapshot()[k]; !ok || set.MaxSessions != 4 || set.ContextTokens != 65536 {
		t.Fatalf("B's setting = %+v (present %v); a peer must not change it", set, ok)
	}
}

func TestPeerCapacity_UnknownOrUnpinnedNode(t *testing.T) {
	tc := engineCases(t)[0]
	fx := newPeerFixture(t, tc)
	if _, err := fx.a.host.peerCapacityReport(context.Background(), "node-z"); err == nil {
		t.Fatal("an unknown node answered")
	}
	// Drop the pin: the node is still discovered but no longer a cluster member.
	n := fx.a.discovery.Nodes()[0]
	n.ClusterUUID = "not-pinned"
	fx.a.discovery.SetSubscribed([]Node{n})
	if _, err := fx.a.host.peerCapacityReport(context.Background(), "node-b"); err == nil {
		t.Fatal("an unpinned node answered")
	}
}

func TestPeerCapacityRPC(t *testing.T) {
	tc := engineCases(t)[0]
	fx := newPeerFixture(t, tc)
	fx.limitB(65536, 2, time.Second)
	rec := &recRW{}
	fx.a.host.codec = NewCodec(rec)

	if _, errMsg := rpcCall(t, fx.a.host, rec, "capacity/peer-report", `{}`); errMsg == "" {
		t.Fatal("capacity/peer-report accepted no nodeId")
	}

	rec.mu.Lock()
	rec.b = nil
	rec.mu.Unlock()
	id := json.RawMessage(`7`)
	fx.a.host.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: tc.profile.Name + ":capacity/peer-report",
		Params: json.RawMessage(`{"nodeId":"node-b"}`)})
	// Answered off the control loop, so wait for the reply.
	deadline := time.Now().Add(10 * time.Second)
	for !rec.has(`"id":7`) {
		if time.Now().After(deadline) {
			t.Fatal("no reply to capacity/peer-report")
		}
		time.Sleep(10 * time.Millisecond)
	}
	rec.mu.Lock()
	raw := append([]byte(nil), rec.b...)
	rec.mu.Unlock()
	var resp struct {
		Result struct {
			NodeID string          `json:"nodeId"`
			Models []modelCapacity `json:"models"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil || resp.Error != nil {
		t.Fatalf("reply %s: decode err %v", raw, err)
	}
	if resp.Result.NodeID != "node-b" || len(resp.Result.Models) == 0 || resp.Result.Models[0].MaxSessions != 2 {
		t.Fatalf("result = %+v; want node-b with 2 sessions", resp.Result)
	}
}
