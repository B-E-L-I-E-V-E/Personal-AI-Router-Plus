// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// admitFixture is this node (its own engine behind the local backend) plus,
// optionally, one other node with the same model. Each node's engine replies
// with the prompt token count the test sets, and tags its reply so the test
// can tell which node served.
type admitFixture struct {
	t      *testing.T
	tc     engineCase
	proxy  *Proxy
	prompt atomic.Int64 // prompt_eval_count the engines report
}

func newAdmitFixture(t *testing.T, tc engineCase, withOtherNode bool) *admitFixture {
	t.Helper()
	fx := &admitFixture{t: t, tc: tc}
	fx.prompt.Store(100)
	engine := func(node string) *httptest.Server {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			w.Header().Set("Content-Type", "application/x-ndjson")
			w.Header().Set(servedByTestHeader, node)
			w.WriteHeader(http.StatusOK)
			_, _ = fmt.Fprintf(w, "{\"done\":true,\"prompt_eval_count\":%d,\"eval_count\":10}\n", fx.prompt.Load())
		}))
		t.Cleanup(srv.Close)
		return srv
	}

	const selfPort = 47123
	disc := NewDiscovery()
	self := Node{ID: "self", Addresses: []string{"127.0.0.1"}, Port: selfPort, Models: []string{tc.advertisedModel}}
	disc.AddManual(self)
	if withOtherNode {
		disc.AddManual(nodeForModel(t, "node-b", engine("node-b").URL, tc.advertisedModel))
	}
	fx.proxy = newTestProxy(tc.profile, NewCodec(&recRW{}), disc, selfPort)
	backend := nodeFor(t, "backend", engine("self").URL)
	if err := fx.proxy.soleFacade().setLocalBackend(localBackend{Host: backend.Addresses[0], Port: backend.Port, Healthy: true}); err != nil {
		t.Fatalf("set local backend: %v", err)
	}
	// Prefer this node, so a request lands elsewhere only when admission
	// sends it there.
	fx.proxy.SetPriority([]string{"self", "node-b"})
	return fx
}

func (fx *admitFixture) setCapacity(contextTokens int64, maxSessions int) {
	fx.t.Helper()
	k := capacityKey{Engine: fx.tc.profile.Name, Model: fx.tc.requestedModel}
	if err := fx.proxy.capacity.set(k, capacitySetting{ContextTokens: contextTokens, MaxSessions: maxSessions}, false); err != nil {
		fx.t.Fatalf("set capacity: %v", err)
	}
}

func (fx *admitFixture) setQueue(timeout time.Duration) {
	fx.t.Helper()
	cfg := fx.proxy.sessions.config()
	cfg.QueueTimeout = timeout
	if err := fx.proxy.sessions.configure(cfg); err != nil {
		fx.t.Fatalf("configure queue: %v", err)
	}
}

// send issues one session turn; pad lengthens the body.
func (fx *admitFixture) send(ctx context.Context, session string, pad int) *httptest.ResponseRecorder {
	fx.t.Helper()
	body := fmt.Sprintf(`{"model":%q,"pad":%q}`, fx.tc.requestedModel, strings.Repeat("x", pad))
	req := httptest.NewRequest(http.MethodPost, fx.tc.inferencePath, strings.NewReader(body)).WithContext(ctx)
	req.Header.Set(sessionIDHeader, session)
	w := httptest.NewRecorder()
	fx.proxy.soleFacade().handleHTTP(w, req)
	return w
}

func (fx *admitFixture) turn(session string) *httptest.ResponseRecorder {
	fx.t.Helper()
	return fx.send(context.Background(), session, 0)
}

func (fx *admitFixture) key() capacityKey {
	return capacityKey{Engine: fx.tc.profile.Name, Model: fx.tc.requestedModel}
}

func waitUntil(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting until %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestAdmission_SessionKeepsItsSlotAcrossTurns(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAdmitFixture(t, tc, false)
		fx.setCapacity(65536, 1)
		fx.setQueue(0)

		wantStatus(t, fx.turn("conv-1"), http.StatusOK)
		wantStatus(t, fx.turn("conv-1"), http.StatusOK) // still its slot
		if st := fx.proxy.sessions.slotStatus(fx.key()); st.Held != 1 {
			t.Fatalf("held slots = %d, want 1", st.Held)
		}
		w := fx.turn("conv-2")
		wantStatus(t, w, http.StatusServiceUnavailable)
		if w.Header().Get("Retry-After") == "" {
			t.Fatal("a full node's refusal is missing Retry-After")
		}
	})
}

func TestAdmission_FullNodeSendsNewSessionsElsewhere(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAdmitFixture(t, tc, true)
		fx.setCapacity(65536, 1)

		if w := fx.turn("conv-1"); w.Header().Get(servedByTestHeader) != "self" {
			t.Fatalf("first session served by %q, want self", w.Header().Get(servedByTestHeader))
		}
		w := fx.turn("conv-2")
		wantStatus(t, w, http.StatusOK)
		if got := w.Header().Get(servedByTestHeader); got != "node-b" {
			t.Fatalf("second session served by %q, want node-b while this node is full", got)
		}
		if w.Header().Get(queueWaitHeader) != "" {
			t.Fatal("a request that had somewhere else to go must not wait")
		}
	})
}

func TestAdmission_QueuedSessionGetsTheNextFreeSlot(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAdmitFixture(t, tc, false)
		fx.setCapacity(65536, 1)
		fx.setQueue(5 * time.Second)
		wantStatus(t, fx.turn("conv-1"), http.StatusOK)

		result := make(chan *httptest.ResponseRecorder, 1)
		go func() { result <- fx.turn("conv-2") }()
		waitUntil(t, "conv-2 is queued", func() bool { return fx.proxy.sessions.slotStatus(fx.key()).Queued == 1 })

		// The waiting session stays alive however long it waits.
		if v, err := fx.proxy.sessions.get("conv-2"); err != nil || v.InFlight != 1 {
			t.Fatalf("queued session: inFlight=%d err=%v, want 1 and live", v.InFlight, err)
		}

		if err := fx.proxy.sessions.remove("conv-1", ""); err != nil {
			t.Fatalf("end conv-1: %v", err)
		}
		select {
		case w := <-result:
			wantStatus(t, w, http.StatusOK)
			if ms, err := strconv.Atoi(w.Header().Get(queueWaitHeader)); err != nil || ms < 0 {
				t.Fatalf("%s = %q, want the wait in milliseconds", queueWaitHeader, w.Header().Get(queueWaitHeader))
			}
		case <-time.After(3 * time.Second):
			t.Fatal("the queued session was not admitted when a slot freed")
		}
	})
}

func TestAdmission_QueueTimesOut(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAdmitFixture(t, tc, false)
		fx.setCapacity(65536, 1)
		fx.setQueue(50 * time.Millisecond)
		wantStatus(t, fx.turn("conv-1"), http.StatusOK)
		w := fx.turn("conv-2")
		wantStatus(t, w, http.StatusServiceUnavailable)
		if !strings.Contains(w.Body.String(), "timed out") {
			t.Fatalf("body = %q, want a queue timeout", w.Body.String())
		}
		if st := fx.proxy.sessions.slotStatus(fx.key()); st.Queued != 0 {
			t.Fatalf("queued = %d after the timeout, want 0", st.Queued)
		}
	})
}

func TestAdmission_OutgrownSessionMustShorten(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAdmitFixture(t, tc, false)
		fx.setCapacity(8192, 1) // share 8192
		fx.prompt.Store(9000)

		// PAIR learns the size only from the engine, so the turn that crosses
		// the share is served and measured.
		w := fx.send(context.Background(), "conv-1", 200)
		wantStatus(t, w, http.StatusOK)
		if got := w.Header().Get(contextLimitHeader); got != "8192" {
			t.Fatalf("%s = %q, want 8192 on a locally served turn", contextLimitHeader, got)
		}

		t.Run("a turn that is not shorter is refused", func(t *testing.T) {
			w := fx.send(context.Background(), "conv-1", 300)
			wantStatus(t, w, http.StatusBadRequest)
			if !strings.Contains(w.Body.String(), "9010") || !strings.Contains(w.Body.String(), "8192") {
				t.Fatalf("body = %q, want the measured size and the share", w.Body.String())
			}
			if got := w.Header().Get(contextTokensHeader); got != "9010" {
				t.Fatalf("%s = %q, want 9010", contextTokensHeader, got)
			}
		})
		t.Run("a compacted, shorter turn is let through and re-measured", func(t *testing.T) {
			fx.prompt.Store(3000)
			wantStatus(t, fx.send(context.Background(), "conv-1", 10), http.StatusOK)
			v, err := fx.proxy.sessions.get("conv-1")
			if err != nil {
				t.Fatalf("get: %v", err)
			}
			if got := v.Context[tc.profile.Name].Tokens; got != 3010 {
				t.Fatalf("re-measured context = %d, want 3010", got)
			}
		})
	})
}

func TestAdmission_NoCapacityMeansNoGate(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAdmitFixture(t, tc, false)
		fx.setQueue(0)
		for i := range 5 {
			wantStatus(t, fx.turn(fmt.Sprintf("conv-%d", i)), http.StatusOK)
		}
		if st := fx.proxy.sessions.slotStatus(fx.key()); st.Held != 0 {
			t.Fatalf("held slots = %d with no capacity known, want 0", st.Held)
		}
	})
}

func TestSlots_ArrivalOrder(t *testing.T) {
	s := newSessionStore()
	cfg := s.config()
	cfg.QueueTimeout = 5 * time.Second
	if err := s.configure(cfg); err != nil {
		t.Fatalf("configure: %v", err)
	}
	k := capacityKey{Engine: "ollama", Model: "m"}
	limit := func() int { return 1 }
	for _, id := range []string{"holder", "first", "second"} {
		beginT(t, s, id, "")
	}
	if !s.takeSlotNow("holder", k, 1) {
		t.Fatal("holder could not take the free slot")
	}

	order := make(chan string, 2)
	for i, id := range []string{"first", "second"} {
		go func() {
			if err := s.waitForSlot(context.Background(), id, k, limit); err != nil {
				t.Errorf("%s: %v", id, err)
				return
			}
			order <- id
		}()
		waitUntil(t, id+" is queued", func() bool { return s.slotStatus(k).Queued == i+1 })
	}

	t.Run("a newcomer cannot jump the queue", func(t *testing.T) {
		beginT(t, s, "newcomer", "")
		if s.takeSlotNow("newcomer", k, 2) {
			t.Fatal("a session took a slot while earlier arrivals were waiting")
		}
	})

	s.mu.Lock()
	s.releaseSlotsLocked(s.sessions["holder"])
	s.mu.Unlock()
	if got := <-order; got != "first" {
		t.Fatalf("slot went to %q, want first", got)
	}
	s.mu.Lock()
	s.releaseSlotsLocked(s.sessions["first"])
	s.mu.Unlock()
	if got := <-order; got != "second" {
		t.Fatalf("slot went to %q, want second", got)
	}
}

func TestSlots_ReleasedWhenTheSessionMovesOrSwitchesModel(t *testing.T) {
	s := newSessionStore()
	beginT(t, s, "conv-1", "")
	a := capacityKey{Engine: "ollama", Model: "a"}
	b := capacityKey{Engine: "ollama", Model: "b"}

	if !s.takeSlotNow("conv-1", a, 1) {
		t.Fatal("could not take a slot for model a")
	}
	t.Run("switching model gives up the old slot", func(t *testing.T) {
		if !s.takeSlotNow("conv-1", b, 1) {
			t.Fatal("could not take a slot for model b")
		}
		if st := s.slotStatus(a); st.Held != 0 {
			t.Fatalf("model a still holds %d slots", st.Held)
		}
	})
	t.Run("moving to another node gives up the slot", func(t *testing.T) {
		s.pin("conv-1", "ollama", candidate{id: "node-b"})
		if st := s.slotStatus(b); st.Held != 0 {
			t.Fatalf("model b still holds %d slots after the session moved", st.Held)
		}
	})
}

func TestSlots_WaitEndsWhenTheSessionEnds(t *testing.T) {
	s := newSessionStore()
	cfg := s.config()
	cfg.QueueTimeout = 5 * time.Second
	if err := s.configure(cfg); err != nil {
		t.Fatalf("configure: %v", err)
	}
	k := capacityKey{Engine: "ollama", Model: "m"}
	beginT(t, s, "holder", "")
	beginT(t, s, "waiter", "")
	s.takeSlotNow("holder", k, 1)

	done := make(chan error, 1)
	go func() { done <- s.waitForSlot(context.Background(), "waiter", k, func() int { return 1 }) }()
	waitUntil(t, "waiter is queued", func() bool { return s.slotStatus(k).Queued == 1 })
	if err := s.remove("waiter", ""); err != nil {
		t.Fatalf("remove: %v", err)
	}
	select {
	case err := <-done:
		if !errors.Is(err, errSessionGone) {
			t.Fatalf("wait ended with %v, want errSessionGone", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("ending the session did not end its wait")
	}
}
