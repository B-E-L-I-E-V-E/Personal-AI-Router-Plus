// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// stateFixture is this node, served over real HTTP the way a client reaches
// it, with its own engine and optionally one other node. This node's engine
// streams a first chunk at once and finishes only when the test releases it,
// so a test can change the node's state while a request is mid-stream.
type stateFixture struct {
	t       *testing.T
	tc      engineCase
	proxy   *Proxy
	url     string
	started chan struct{} // one value per inference request this node's engine began
	ended   chan error    // why each of those ended: nil when finished, else the context error
	release chan struct{}
	once    sync.Once
}

func newStateFixture(t *testing.T, tc engineCase, withOtherNode bool) *stateFixture {
	t.Helper()
	fx := &stateFixture{
		t: t, tc: tc,
		started: make(chan struct{}, 8),
		ended:   make(chan error, 8),
		release: make(chan struct{}),
	}
	t.Cleanup(fx.finish)
	self := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		if r.URL.Path != tc.inferencePath {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		fx.started <- struct{}{}
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set(servedByTestHeader, "self")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "{\"done\":false}\n")
		w.(http.Flusher).Flush()
		select {
		case <-fx.release:
			_, _ = io.WriteString(w, "{\"done\":true}\n")
			fx.ended <- nil
		case <-r.Context().Done():
			fx.ended <- r.Context().Err()
		}
	}))
	t.Cleanup(self.Close)
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set(servedByTestHeader, "node-b")
		_, _ = io.WriteString(w, "{\"done\":true}\n")
	}))
	t.Cleanup(other.Close)

	const selfPort = 47124
	disc := NewDiscovery()
	disc.AddManual(Node{ID: "self", Addresses: []string{"127.0.0.1"}, Port: selfPort, Models: []string{tc.advertisedModel}})
	if withOtherNode {
		disc.AddManual(nodeForModel(t, "node-b", other.URL, tc.advertisedModel))
	}
	fx.proxy = newTestProxy(tc.profile, NewCodec(&recRW{}), disc, selfPort)
	backend := nodeFor(t, "backend", self.URL)
	if err := fx.proxy.soleFacade().setLocalBackend(localBackend{Host: backend.Addresses[0], Port: backend.Port, Healthy: true}); err != nil {
		t.Fatalf("set local backend: %v", err)
	}
	// This node first, so a request lands elsewhere only when its state
	// sends it there.
	fx.proxy.SetPriority([]string{"self", "node-b"})
	front := httptest.NewServer(http.HandlerFunc(fx.proxy.soleFacade().handlePlain))
	t.Cleanup(front.Close)
	fx.url = front.URL
	return fx
}

func (fx *stateFixture) finish() { fx.once.Do(func() { close(fx.release) }) }

// stream starts a request and returns once its first chunk has arrived, so
// the request is committed to this node's engine and running.
func (fx *stateFixture) stream() (*http.Response, *bufio.Reader) {
	fx.t.Helper()
	resp, err := http.Post(fx.url+fx.tc.inferencePath, "application/json",
		strings.NewReader(fmt.Sprintf(`{"model":%q,"stream":true}`, fx.tc.requestedModel)))
	if err != nil {
		fx.t.Fatalf("start stream: %v", err)
	}
	fx.t.Cleanup(func() { _ = resp.Body.Close() })
	if resp.StatusCode != http.StatusOK {
		fx.t.Fatalf("stream status = %d, want 200", resp.StatusCode)
	}
	rd := bufio.NewReader(resp.Body)
	if _, err := rd.ReadString('\n'); err != nil {
		fx.t.Fatalf("read first chunk: %v", err)
	}
	return resp, rd
}

// infer sends one request and returns its status, who served it and its body.
func (fx *stateFixture) infer() (int, string, string) {
	fx.t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, fx.url+fx.tc.inferencePath,
		strings.NewReader(fmt.Sprintf(`{"model":%q}`, fx.tc.requestedModel)))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fx.t.Fatalf("infer: %v", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, resp.Header.Get(servedByTestHeader), string(b)
}

// setState changes the node's state through its HTTP API.
func (fx *stateFixture) setState(st nodeState) nodeStateJSON {
	fx.t.Helper()
	req, _ := http.NewRequest(http.MethodPut, fx.url+nodeStateAPIPath, strings.NewReader(`{"state":"`+string(st)+`"}`))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fx.t.Fatalf("set state: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		fx.t.Fatalf("set state status = %d, want 200", resp.StatusCode)
	}
	var v nodeStateJSON
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil {
		fx.t.Fatalf("decode state: %v", err)
	}
	return v
}

func (fx *stateFixture) waitEnded() error {
	fx.t.Helper()
	select {
	case err := <-fx.ended:
		return err
	case <-time.After(5 * time.Second):
		fx.t.Fatal("this node's engine request never ended")
		return nil
	}
}

func TestNodeState_PauseFinishesTheRunningRequestAndSendsNewOnesElsewhere(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newStateFixture(t, tc, true)
		_, rd := fx.stream()

		if v := fx.setState(nodePaused); v.State != "paused" || v.Serving != 1 || v.Cancelled != 0 {
			t.Fatalf("pause = %+v, want paused with the one request still serving", v)
		}
		status, servedBy, body := fx.infer()
		if status != http.StatusOK || servedBy != "node-b" {
			t.Fatalf("new request while paused: status %d served by %q (%s), want 200 from node-b", status, servedBy, body)
		}

		fx.finish()
		rest, err := io.ReadAll(rd)
		if err != nil || !strings.Contains(string(rest), `"done":true`) {
			t.Fatalf("the running request did not finish normally: %q, %v", rest, err)
		}
		if err := fx.waitEnded(); err != nil {
			t.Fatalf("the running request was cut off: %v", err)
		}
	})
}

func TestNodeState_PausedWithNoOtherNodeRefusesWith503(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newStateFixture(t, tc, false)
		fx.setState(nodePaused)
		status, _, body := fx.infer()
		if status != http.StatusServiceUnavailable || !strings.Contains(body, errNodesHeld.Error()) {
			t.Fatalf("request while paused: status %d %s, want 503 naming the paused nodes", status, body)
		}
	})
}

func TestNodeState_StopCancelsTheRunningRequestAndResumeRestoresService(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newStateFixture(t, tc, false)
		_, rd := fx.stream()

		if v := fx.setState(nodeStopped); v.State != "stopped" || v.Cancelled != 1 {
			t.Fatalf("stop = %+v, want stopped with one request cancelled", v)
		}
		if err := fx.waitEnded(); err == nil {
			t.Fatal("the engine finished the request instead of seeing it cancelled")
		}
		// The client's stream ends without the final chunk.
		rest, _ := io.ReadAll(rd)
		if strings.Contains(string(rest), `"done":true`) {
			t.Fatalf("the stopped request still completed: %q", rest)
		}
		if status, _, _ := fx.infer(); status != http.StatusServiceUnavailable {
			t.Fatalf("request while stopped: status %d, want 503", status)
		}

		if v := fx.setState(nodeRunning); v.State != "running" {
			t.Fatalf("resume = %+v, want running", v)
		}
		fx.finish()
		if status, servedBy, body := fx.infer(); status != http.StatusOK || servedBy != "self" {
			t.Fatalf("request after resume: status %d served by %q (%s), want 200 from self", status, servedBy, body)
		}
	})
}

func TestNodeState_StopFreesTheSlotsSessionsHoldHere(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAdmitFixture(t, tc, false)
		fx.setCapacity(8192, 2)
		wantStatus(t, fx.turn("conv-1"), http.StatusOK)
		if held := fx.proxy.sessions.slotStatus(fx.key()).Held; held != 1 {
			t.Fatalf("held slots = %d, want 1", held)
		}
		if _, err := fx.proxy.setLocalNodeState(nodeStopped); err != nil {
			t.Fatalf("stop: %v", err)
		}
		if held := fx.proxy.sessions.slotStatus(fx.key()).Held; held != 0 {
			t.Fatalf("held slots after stop = %d, want 0", held)
		}
	})
}

func TestNodeState_PauseRefusesARequestWaitingForASlot(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAdmitFixture(t, tc, false)
		fx.setCapacity(8192, 1)
		fx.setQueue(30 * time.Second)
		wantStatus(t, fx.turn("conv-1"), http.StatusOK)

		got := make(chan *httptest.ResponseRecorder, 1)
		go func() { got <- fx.turn("conv-2") }()
		waitUntil(t, "the second session queues", func() bool {
			return fx.proxy.sessions.slotStatus(fx.key()).Queued == 1
		})
		if _, err := fx.proxy.setLocalNodeState(nodePaused); err != nil {
			t.Fatalf("pause: %v", err)
		}
		select {
		case w := <-got:
			wantStatus(t, w, http.StatusServiceUnavailable)
		case <-time.After(5 * time.Second):
			t.Fatal("the queued request kept waiting after the pause")
		}
	})
}

func TestNodeState_PausedPeerRefusesAndTheCallerStopsRoutingToIt(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		shortJobDeadline(t)
		fx := newPeerFixture(t, tc)
		wantStatus(t, fx.send("conv-1", 0), http.StatusOK)

		if _, err := fx.b.host.setLocalNodeState(nodePaused); err != nil {
			t.Fatalf("pause B: %v", err)
		}
		w := fx.send("conv-1", 0)
		wantStatus(t, w, http.StatusServiceUnavailable)
		if !strings.Contains(w.Body.String(), "paused") {
			t.Fatalf("refusal %q does not say the node is paused", w.Body.String())
		}
		if st := fx.a.host.node.peerState("b"); st != nodePaused {
			t.Fatalf("A's view of B = %q, want paused", st)
		}

		// Once B runs again and A hears so, A routes to it again.
		fx.b.host.setLocalNodeState(nodeRunning)
		fx.a.host.node.markPeer("b", nodeRunning)
		wantStatus(t, fx.send("conv-1", 0), http.StatusOK)
	})
}

func TestNodeState_AnyClusterMemberCanSetAnotherNodesState(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newPeerFixture(t, tc)
		ctx := context.Background()

		v, err := fx.a.host.nodeStateCall(ctx, nodeStateParams{NodeID: "node-b", State: "paused"})
		if err != nil {
			t.Fatalf("pause B from A: %v", err)
		}
		if v.NodeID != "node-b" || v.State != "paused" || fx.b.host.node.current() != nodePaused {
			t.Fatalf("pause B from A = %+v, B is %q; want B paused", v, fx.b.host.node.current())
		}
		if st := fx.a.host.node.peerState("b"); st != nodePaused {
			t.Fatalf("A's view of B after pausing it = %q, want paused", st)
		}
		if got := fx.a.host.nodeStates().Peers; len(got) != 1 || got[0].NodeID != "node-b" || got[0].State != "paused" {
			t.Fatalf("A's node states = %+v, want node-b paused", got)
		}

		if v, err = fx.a.host.nodeStateCall(ctx, nodeStateParams{NodeID: "node-b"}); err != nil || v.State != "paused" {
			t.Fatalf("read B from A = %+v, %v; want paused", v, err)
		}
		if _, err = fx.a.host.nodeStateCall(ctx, nodeStateParams{NodeID: "node-b", State: "running"}); err != nil {
			t.Fatalf("resume B from A: %v", err)
		}
		if fx.b.host.node.current() != nodeRunning || fx.a.host.node.peerState("b") != nodeRunning {
			t.Fatal("B did not resume, or A still holds it as paused")
		}
		if _, err = fx.a.host.nodeStateCall(ctx, nodeStateParams{NodeID: "unknown"}); err == nil {
			t.Fatal("setting the state of a node outside the cluster succeeded")
		}
	})
}

func TestNodeState_PeerNoticeMarksThePeerUntilItExpires(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		f := testProxy(tc.profile, NewDiscovery(), tc.profile.StandalonePort).soleFacade()
		now := time.Unix(1000, 0)
		f.host.node.now = func() time.Time { return now }

		req := httptest.NewRequest(http.MethodPost, peerNodeNoticePath, strings.NewReader(`{"state":"stopped"}`))
		w := httptest.NewRecorder()
		if !f.servePeerNodeState(w, req, "peer-1") {
			t.Fatal("the notice route was not served")
		}
		wantStatus(t, w, http.StatusNoContent)
		if st := f.host.node.peerState("peer-1"); st != nodeStopped {
			t.Fatalf("peer state = %q, want stopped", st)
		}
		now = now.Add(peerStateTTL)
		if st := f.host.node.peerState("peer-1"); st != nodeRunning {
			t.Fatalf("peer state after the TTL = %q, want running", st)
		}

		bad := httptest.NewRecorder()
		f.servePeerNodeState(bad, httptest.NewRequest(http.MethodPost, peerNodeNoticePath, strings.NewReader(`{"state":"asleep"}`)), "peer-1")
		wantStatus(t, bad, http.StatusBadRequest)
	})
}

func TestNodeState_SurvivesARestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), nodeStateFileName)
	s := newNodeStateStore()
	s.path = path
	if _, _, err := s.set(nodePaused); err != nil {
		t.Fatalf("pause: %v", err)
	}
	restarted := newNodeStateStore()
	restarted.path = path
	if err := restarted.load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	if restarted.current() != nodePaused || restarted.view().Since == "" {
		t.Fatalf("after restart = %+v, want paused with its start time", restarted.view())
	}

	fresh := newNodeStateStore()
	fresh.path = filepath.Join(t.TempDir(), nodeStateFileName)
	if err := fresh.load(); err != nil || fresh.current() != nodeRunning {
		t.Fatalf("with no saved state = %q, %v; want running", fresh.current(), err)
	}
}

func TestNodeState_RejectsUnknownStates(t *testing.T) {
	for _, s := range []string{"", "Paused", "off"} {
		if _, err := parseNodeState(s); err == nil {
			t.Errorf("parseNodeState(%q) succeeded", s)
		}
	}
}

func TestNodeState_ReportedInNodeActivity(t *testing.T) {
	p := NewProxy(NewCodec(rwNop{}))
	if got := p.nodeActivity().State; got != "running" {
		t.Fatalf("activity state = %q, want running", got)
	}
	p.node.set(nodePaused)
	if got := p.nodeActivity().State; got != "paused" {
		t.Fatalf("activity state = %q, want paused", got)
	}
}
