// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// beginT admits one request and fails the test on error.
func beginT(t *testing.T, s *sessionStore, id, agentID string) uint64 {
	t.Helper()
	_, token, err := s.begin(id, agentID, func() {})
	if err != nil {
		t.Fatalf("begin(%q, %q): %v", id, agentID, err)
	}
	return token
}

func TestSessionStore_FirstAgentOwnsTheSession(t *testing.T) {
	s := newSessionStore()
	s.end("conv-1", beginT(t, s, "conv-1", "agent-a"))

	t.Run("owner may continue", func(t *testing.T) {
		s.end("conv-1", beginT(t, s, "conv-1", "agent-a"))
	})
	t.Run("another agent is refused", func(t *testing.T) {
		if _, _, err := s.begin("conv-1", "agent-b", func() {}); !errors.Is(err, errAgentMismatch) {
			t.Fatalf("begin as other agent: err = %v, want errAgentMismatch", err)
		}
	})
	t.Run("omitting the agent is refused", func(t *testing.T) {
		if _, _, err := s.begin("conv-1", "", func() {}); !errors.Is(err, errAgentMismatch) {
			t.Fatalf("begin without agent: err = %v, want errAgentMismatch", err)
		}
	})
}

func TestSessionStore_CountsInFlightRequests(t *testing.T) {
	s := newSessionStore()
	first := beginT(t, s, "conv-1", "")
	second := beginT(t, s, "conv-1", "")
	if v, err := s.get("conv-1"); err != nil || v.InFlight != 2 || v.Requests != 2 {
		t.Fatalf("after two begins: inFlight=%d requests=%d err=%v, want 2, 2, nil", v.InFlight, v.Requests, err)
	}
	s.end("conv-1", first)
	s.end("conv-1", first) // a repeated end must not release the other request
	if v, _ := s.get("conv-1"); v.InFlight != 1 {
		t.Fatalf("after one end: inFlight=%d, want 1", v.InFlight)
	}
	s.end("conv-1", second)
	if v, _ := s.get("conv-1"); v.InFlight != 0 {
		t.Fatalf("after both ends: inFlight=%d, want 0", v.InFlight)
	}
}

func TestSessionStore_PinsPerEngine(t *testing.T) {
	s := newSessionStore()
	beginT(t, s, "conv-1", "")
	s.pin("conv-1", "ollama", "node-a")
	s.pin("conv-1", "lmstudio", "node-b")
	v, err := s.get("conv-1")
	if err != nil {
		t.Fatalf("get after begin: %v", err)
	}
	if v.Nodes["ollama"] != "node-a" || v.Nodes["lmstudio"] != "node-b" {
		t.Fatalf("nodes = %v, want ollama=node-a lmstudio=node-b", v.Nodes)
	}
}

func TestSessionHeaders(t *testing.T) {
	parse := func(name string, set map[string]string, wantSession, wantAgent string, wantErr error) {
		t.Run(name, func(t *testing.T) {
			h := http.Header{}
			for k, v := range set {
				h.Set(k, v)
			}
			gotSession, gotAgent, err := sessionHeaders(h)
			if !errors.Is(err, wantErr) {
				t.Fatalf("err = %v, want %v", err, wantErr)
			}
			if gotSession != wantSession || gotAgent != wantAgent {
				t.Fatalf("got session=%q agent=%q, want session=%q agent=%q", gotSession, gotAgent, wantSession, wantAgent)
			}
		})
	}
	parse("neither header", nil, "", "", nil)
	parse("both headers", map[string]string{sessionIDHeader: "conv-1", agentIDHeader: "planner"}, "conv-1", "planner", nil)
	parse("agent without session", map[string]string{agentIDHeader: "planner"}, "", "planner", nil)
	parse("malformed session", map[string]string{sessionIDHeader: "conv 1"}, "", "", errInvalidSessionID)
	parse("empty session", map[string]string{sessionIDHeader: ""}, "", "", errInvalidSessionID)
	parse("malformed agent", map[string]string{agentIDHeader: "a/b"}, "", "", errInvalidAgentID)
}

func TestPreferNode(t *testing.T) {
	ids := func(cands []candidate) string {
		out := make([]string, len(cands))
		for i, c := range cands {
			out[i] = c.id
		}
		return strings.Join(out, ",")
	}
	mk := func() []candidate { return []candidate{{id: "a"}, {id: "b"}, {id: "c"}} }

	t.Run("moves the node to the front keeping the rest in order", func(t *testing.T) {
		c := mk()
		if !preferNode(c, "c") || ids(c) != "c,a,b" {
			t.Fatalf("got %s, want c,a,b", ids(c))
		}
	})
	t.Run("leaves order alone when the node is first", func(t *testing.T) {
		c := mk()
		if !preferNode(c, "a") || ids(c) != "a,b,c" {
			t.Fatalf("got %s, want a,b,c", ids(c))
		}
	})
	t.Run("reports an absent node", func(t *testing.T) {
		c := mk()
		if preferNode(c, "z") || ids(c) != "a,b,c" {
			t.Fatalf("got %s and found, want a,b,c and not found", ids(c))
		}
	})
}

// sessionFixture is two nodes that both advertise the model, each counting the
// inference requests it receives.
type sessionFixture struct {
	t      *testing.T
	tc     engineCase
	proxy  *Proxy
	rec    *recRW
	hits   map[string]*atomic.Int32
	status map[string]*atomic.Int32 // non-zero overrides the node's reply status
}

func newSessionFixture(t *testing.T, tc engineCase) *sessionFixture {
	t.Helper()
	fx := &sessionFixture{
		t: t, tc: tc, rec: &recRW{},
		hits:   map[string]*atomic.Int32{},
		status: map[string]*atomic.Int32{},
	}
	disc := NewDiscovery()
	for _, id := range []string{"node-a", "node-b"} {
		hits, status := &atomic.Int32{}, &atomic.Int32{}
		fx.hits[id], fx.status[id] = hits, status
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			hits.Add(1)
			if code := status.Load(); code != 0 {
				w.WriteHeader(int(code))
				return
			}
			w.Header().Set(servedByTestHeader, id)
			w.WriteHeader(http.StatusOK)
			if _, err := io.WriteString(w, `{"done":true}`); err != nil {
				t.Errorf("upstream %s write: %v", id, err)
			}
		}))
		t.Cleanup(srv.Close)
		disc.AddManual(nodeForModel(t, id, srv.URL, tc.advertisedModel))
	}
	fx.proxy = newTestProxy(tc.profile, NewCodec(fx.rec), disc, tc.profile.FacadePort)
	return fx
}

// send issues one inference request and returns the response.
func (fx *sessionFixture) send(headers map[string]string) *httptest.ResponseRecorder {
	fx.t.Helper()
	req := httptest.NewRequest(http.MethodPost, fx.tc.inferencePath,
		strings.NewReader(fmt.Sprintf(`{"model":%q}`, fx.tc.requestedModel)))
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	fx.proxy.soleFacade().handleHTTP(w, req)
	return w
}

// servedByTestHeader is set by a fixture node on the reply it served, so a
// test can tell the serving node from one that was tried and failed over.
const servedByTestHeader = "X-Test-Served-By"

// servedBy sends one request and reports which node served the reply.
func (fx *sessionFixture) servedBy(headers map[string]string) string {
	fx.t.Helper()
	w := fx.send(headers)
	if w.Code != http.StatusOK {
		fx.t.Fatalf("status = %d, want 200 (body %q)", w.Code, w.Body.String())
	}
	return w.Header().Get(servedByTestHeader)
}

func (fx *sessionFixture) prioritize(nodes ...string) {
	fx.t.Helper()
	fx.proxy.SetPriority(nodes)
}

func TestHandleHTTP_SessionReturnsToItsNode(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newSessionFixture(t, tc)
		session := map[string]string{sessionIDHeader: "conv-1"}

		fx.prioritize("node-a", "node-b")
		if got := fx.servedBy(session); got != "node-a" {
			t.Fatalf("first turn served by %q, want node-a (scheduler's choice)", got)
		}

		// The scheduler now prefers node-b. Unrelated traffic follows it, but
		// the session's next turn goes back to the node holding its cache.
		fx.prioritize("node-b", "node-a")
		if got := fx.servedBy(nil); got != "node-b" {
			t.Fatalf("request without a session served by %q, want node-b", got)
		}
		if got := fx.servedBy(session); got != "node-a" {
			t.Fatalf("second turn served by %q, want node-a (session affinity)", got)
		}
	})
}

func TestHandleHTTP_SessionRepinsAfterFailover(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newSessionFixture(t, tc)
		session := map[string]string{sessionIDHeader: "conv-1"}

		fx.prioritize("node-a", "node-b")
		if got := fx.servedBy(session); got != "node-a" {
			t.Fatalf("first turn served by %q, want node-a", got)
		}

		// node-a starts refusing work: the turn fails over to node-b, and the
		// session follows the node that actually served it.
		fx.status["node-a"].Store(http.StatusServiceUnavailable)
		if got := fx.servedBy(session); got != "node-b" {
			t.Fatalf("turn during node-a outage served by %q, want node-b", got)
		}
		fx.status["node-a"].Store(0)
		if got := fx.servedBy(session); got != "node-b" {
			t.Fatalf("turn after recovery served by %q, want node-b (re-pinned)", got)
		}
	})
}

func TestHandleHTTP_ManualSelectionOutranksSession(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newSessionFixture(t, tc)
		session := map[string]string{sessionIDHeader: "conv-1"}

		fx.prioritize("node-a", "node-b")
		if got := fx.servedBy(session); got != "node-a" {
			t.Fatalf("first turn served by %q, want node-a", got)
		}
		fx.proxy.soleFacade().SetSelected("node-b")
		if got := fx.servedBy(session); got != "node-b" {
			t.Fatalf("turn with manual selection served by %q, want node-b", got)
		}
	})
}

func TestHandleHTTP_SessionRefusals(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		refused := func(name string, setup map[string]string, headers map[string]string, wantStatus int) {
			t.Run(name, func(t *testing.T) {
				fx := newSessionFixture(t, tc)
				if setup != nil {
					if w := fx.send(setup); w.Code != http.StatusOK {
						t.Fatalf("setup request status = %d, want 200", w.Code)
					}
				}
				before := fx.hits["node-a"].Load() + fx.hits["node-b"].Load()
				w := fx.send(headers)
				if w.Code != wantStatus {
					t.Fatalf("status = %d, want %d", w.Code, wantStatus)
				}
				var body struct {
					Error string `json:"error"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil || body.Error == "" {
					t.Fatalf("body %q: want a JSON error (decode err %v)", w.Body.String(), err)
				}
				if w.Header().Get(requestIDHeader) == "" {
					t.Fatalf("refusal is missing %s", requestIDHeader)
				}
				if after := fx.hits["node-a"].Load() + fx.hits["node-b"].Load(); after != before {
					t.Fatalf("a refused request reached an engine (%d -> %d requests)", before, after)
				}
			})
		}

		refused("malformed session id", nil,
			map[string]string{sessionIDHeader: "has space"}, http.StatusBadRequest)
		refused("malformed agent id", nil,
			map[string]string{sessionIDHeader: "conv-1", agentIDHeader: "a/b"}, http.StatusBadRequest)
		refused("another agent's session",
			map[string]string{sessionIDHeader: "conv-1", agentIDHeader: "planner"},
			map[string]string{sessionIDHeader: "conv-1", agentIDHeader: "coder"}, http.StatusConflict)
	})
}

func TestHandleHTTP_SessionIDsReachEventsAndWorkloads(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newSessionFixture(t, tc)
		w := fx.send(map[string]string{sessionIDHeader: "conv-1", agentIDHeader: "planner"})
		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		if got := w.Header().Get(sessionIDHeader); got != "conv-1" {
			t.Fatalf("response %s = %q, want conv-1", sessionIDHeader, got)
		}
		if !waitFor(t, fx.rec, workloadCompletedMethod) {
			t.Fatal("workload:completed not emitted")
		}

		var sawEvent, sawWorkload bool
		for _, f := range decodeFrames(t, fx.rec) {
			switch {
			case strings.HasSuffix(f.Method, ":proxy/request"):
				var ev RequestEvent
				if err := json.Unmarshal(f.Params, &ev); err != nil {
					t.Fatalf("decode proxy/request: %v", err)
				}
				if ev.SessionID != "conv-1" || ev.AgentID != "planner" {
					t.Fatalf("proxy/request session=%q agent=%q, want conv-1 and planner", ev.SessionID, ev.AgentID)
				}
				sawEvent = true
			case f.Method == workloadSubmittedMethod:
				var p workloadParams
				if err := json.Unmarshal(f.Params, &p); err != nil {
					t.Fatalf("decode workload: %v", err)
				}
				wi := p.WorkloadInfo
				if wi.SessionID != "conv-1" || wi.RequesterID == nil || *wi.RequesterID != "planner" {
					t.Fatalf("workload sessionId=%q requesterId=%v, want conv-1 and planner", wi.SessionID, wi.RequesterID)
				}
				sawWorkload = true
			}
		}
		if !sawEvent || !sawWorkload {
			t.Fatalf("saw proxy/request=%v workload:submitted=%v, want both", sawEvent, sawWorkload)
		}
	})
}
