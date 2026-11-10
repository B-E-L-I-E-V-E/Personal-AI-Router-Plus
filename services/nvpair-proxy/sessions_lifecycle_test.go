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
	"sync"
	"testing"
	"time"
)

// fakeClock drives a store's notion of now.
type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func newClockedStore(t *testing.T) (*sessionStore, *fakeClock) {
	t.Helper()
	clk := &fakeClock{t: time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)}
	s := newSessionStore()
	s.now = clk.now
	return s, clk
}

func wantGone(t *testing.T, s *sessionStore, id string) {
	t.Helper()
	if _, err := s.get(id); !errors.Is(err, errSessionGone) {
		t.Fatalf("get(%q): err = %v, want errSessionGone", id, err)
	}
}

func wantLive(t *testing.T, s *sessionStore, id string) sessionView {
	t.Helper()
	v, err := s.get(id)
	if err != nil {
		t.Fatalf("get(%q): %v, want a live session", id, err)
	}
	return v
}

func TestSessionExpiry_AfterIdleTimeout(t *testing.T) {
	s, clk := newClockedStore(t)
	s.end("conv-1", beginT(t, s, "conv-1", ""))

	clk.advance(defaultSessionIdleTimeout - time.Second)
	wantLive(t, s, "conv-1")

	clk.advance(time.Second)
	wantGone(t, s, "conv-1")

	// An ended ID must not quietly start a new conversation.
	if _, _, err := s.begin("conv-1", "", func() {}); !errors.Is(err, errSessionGone) {
		t.Fatalf("begin on an expired session: err = %v, want errSessionGone", err)
	}
}

func TestSessionExpiry_ActivitySlidesTheWindow(t *testing.T) {
	s, clk := newClockedStore(t)
	s.end("conv-1", beginT(t, s, "conv-1", ""))

	clk.advance(9 * time.Minute)
	if _, err := s.keepalive("conv-1", ""); err != nil {
		t.Fatalf("keepalive: %v", err)
	}
	clk.advance(9 * time.Minute)
	s.end("conv-1", beginT(t, s, "conv-1", "")) // a request slides it too
	clk.advance(9 * time.Minute)
	v := wantLive(t, s, "conv-1")
	if want := clk.t.Add(time.Minute); !v.ExpiresAt.Equal(want) {
		t.Fatalf("expiresAt = %v, want %v", v.ExpiresAt, want)
	}

	clk.advance(time.Minute)
	wantGone(t, s, "conv-1")
}

func TestSessionExpiry_NeverWhileARequestIsInFlight(t *testing.T) {
	s, clk := newClockedStore(t)
	token := beginT(t, s, "conv-1", "")

	clk.advance(2 * time.Hour) // a very long generation, or a long queue wait
	if removed := s.sweep(); removed != 0 {
		t.Fatalf("sweep removed %d sessions while a request was in flight", removed)
	}
	wantLive(t, s, "conv-1")

	// The window restarts when the request ends.
	s.end("conv-1", token)
	clk.advance(defaultSessionIdleTimeout - time.Second)
	wantLive(t, s, "conv-1")
	clk.advance(time.Second)
	wantGone(t, s, "conv-1")
}

func TestSessionExpiry_PerSessionIdleTimeout(t *testing.T) {
	s, clk := newClockedStore(t)
	if _, err := s.create("short", "", 2*time.Minute); err != nil {
		t.Fatalf("create: %v", err)
	}
	clk.advance(2 * time.Minute)
	wantGone(t, s, "short")

	t.Run("rejects a timeout below the minimum", func(t *testing.T) {
		if _, err := s.create("too-short", "", 30*time.Second); !errors.Is(err, errIdleTimeout) {
			t.Fatalf("err = %v, want errIdleTimeout", err)
		}
	})
	t.Run("rejects a timeout above the maximum", func(t *testing.T) {
		if _, err := s.create("too-long", "", 2*time.Hour); !errors.Is(err, errIdleTimeout) {
			t.Fatalf("err = %v, want errIdleTimeout", err)
		}
	})
}

func TestSessionExpiry_StoreDefaultIsAdjustable(t *testing.T) {
	s, clk := newClockedStore(t)
	cfg := s.config()
	cfg.IdleTimeout = 5 * time.Minute
	if err := s.configure(cfg); err != nil {
		t.Fatalf("configure: %v", err)
	}
	s.end("conv-1", beginT(t, s, "conv-1", ""))
	clk.advance(5 * time.Minute)
	wantGone(t, s, "conv-1")
}

func TestSessionExpiry_MaxLifetimeOverridesKeepalives(t *testing.T) {
	s, clk := newClockedStore(t)
	cfg := s.config()
	cfg.MaxLifetime = 30 * time.Minute
	if err := s.configure(cfg); err != nil {
		t.Fatalf("configure: %v", err)
	}
	s.end("conv-1", beginT(t, s, "conv-1", ""))
	for range 3 {
		clk.advance(9 * time.Minute)
		if _, err := s.keepalive("conv-1", ""); err != nil {
			t.Fatalf("keepalive: %v", err)
		}
	}
	clk.advance(3 * time.Minute) // 30 minutes since creation
	wantGone(t, s, "conv-1")
}

func TestSessionRemove_CancelsInFlightAndTombstones(t *testing.T) {
	s, _ := newClockedStore(t)
	var cancelled int
	if _, _, err := s.begin("conv-1", "planner", func() { cancelled++ }); err != nil {
		t.Fatalf("begin: %v", err)
	}

	t.Run("only the owner may end it", func(t *testing.T) {
		if err := s.remove("conv-1", "coder"); !errors.Is(err, errAgentMismatch) {
			t.Fatalf("remove as another agent: err = %v, want errAgentMismatch", err)
		}
		if cancelled != 0 {
			t.Fatal("a refused remove cancelled the in-flight request")
		}
	})

	if err := s.remove("conv-1", "planner"); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if cancelled != 1 {
		t.Fatalf("in-flight requests cancelled = %d, want 1", cancelled)
	}
	wantGone(t, s, "conv-1")
	if err := s.remove("conv-1", "planner"); !errors.Is(err, errSessionGone) {
		t.Fatalf("second remove: err = %v, want errSessionGone", err)
	}
}

func TestSessionLimit(t *testing.T) {
	s, clk := newClockedStore(t)
	cfg := s.config()
	cfg.MaxSessions = 2
	if err := s.configure(cfg); err != nil {
		t.Fatalf("configure: %v", err)
	}
	s.end("a", beginT(t, s, "a", ""))
	s.end("b", beginT(t, s, "b", ""))
	if _, _, err := s.begin("c", "", func() {}); !errors.Is(err, errSessionLimit) {
		t.Fatalf("third session: err = %v, want errSessionLimit", err)
	}

	// An expired session no longer holds a place.
	clk.advance(defaultSessionIdleTimeout)
	beginT(t, s, "c", "")
}

func TestSessionTombstones_ExpireAfterTheirTTL(t *testing.T) {
	s, clk := newClockedStore(t)
	if _, err := s.create("conv-1", "", 0); err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := s.remove("conv-1", ""); err != nil {
		t.Fatalf("remove: %v", err)
	}
	clk.advance(sessionTombstoneTTL + time.Second)
	s.sweep()
	// Long after the tombstone is gone the ID is free again.
	beginT(t, s, "conv-1", "")
}

func TestApplySessionConfigParams(t *testing.T) {
	base := defaultSessionConfig()
	apply := func(name, params string, want sessionConfig, wantErr bool) {
		t.Run(name, func(t *testing.T) {
			got, err := applySessionConfigParams(base, json.RawMessage(params))
			if (err != nil) != wantErr {
				t.Fatalf("err = %v, wantErr %v", err, wantErr)
			}
			if !wantErr && got != want {
				t.Fatalf("got %+v, want %+v", got, want)
			}
		})
	}
	five := base
	five.IdleTimeout = 5 * time.Minute
	apply("changes only the idle timeout", `{"idleTimeoutSeconds":300}`, five, false)
	all := sessionConfig{IdleTimeout: 20 * time.Minute, MaxLifetime: 2 * time.Hour, MaxSessions: 12}
	apply("changes every field", `{"idleTimeoutSeconds":1200,"maxLifetimeSeconds":7200,"maxSessions":12}`, all, false)
	apply("empty params keep the policy", `{}`, base, false)
	apply("rejects an idle timeout out of range", `{"idleTimeoutSeconds":10}`, sessionConfig{}, true)
	apply("rejects zero sessions", `{"maxSessions":0}`, sessionConfig{}, true)
	apply("rejects an unknown field", `{"idleTimeout":300}`, sessionConfig{}, true)
}

// apiFixture fronts one blocking-capable upstream with a proxy and drives the
// loopback session API through handlePlain, the path a real client takes.
type apiFixture struct {
	t     *testing.T
	tc    engineCase
	proxy *Proxy
	rec   *recRW
}

func newAPIFixture(t *testing.T, tc engineCase, upstream http.Handler) *apiFixture {
	t.Helper()
	srv := httptest.NewServer(upstream)
	t.Cleanup(srv.Close)
	rec := &recRW{}
	disc := NewDiscovery()
	disc.AddManual(nodeForModel(t, "node-a", srv.URL, tc.advertisedModel))
	return &apiFixture{t: t, tc: tc, proxy: newTestProxy(tc.profile, NewCodec(rec), disc, tc.profile.FacadePort), rec: rec}
}

func (fx *apiFixture) do(method, path, agent, body string) *httptest.ResponseRecorder {
	fx.t.Helper()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, path, rd)
	req.RemoteAddr = "127.0.0.1:50000"
	if agent != "" {
		req.Header.Set(agentIDHeader, agent)
	}
	w := httptest.NewRecorder()
	fx.proxy.soleFacade().handlePlain(w, req)
	return w
}

func (fx *apiFixture) infer(sessionID, agent string) *httptest.ResponseRecorder {
	fx.t.Helper()
	req := httptest.NewRequest(http.MethodPost, fx.tc.inferencePath,
		strings.NewReader(fmt.Sprintf(`{"model":%q}`, fx.tc.requestedModel)))
	req.RemoteAddr = "127.0.0.1:50000"
	req.Header.Set(sessionIDHeader, sessionID)
	if agent != "" {
		req.Header.Set(agentIDHeader, agent)
	}
	w := httptest.NewRecorder()
	fx.proxy.soleFacade().handlePlain(w, req)
	return w
}

func decodeSession(t *testing.T, w *httptest.ResponseRecorder) sessionJSON {
	t.Helper()
	var out sessionJSON
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode session %q: %v", w.Body.String(), err)
	}
	return out
}

func wantStatus(t *testing.T, w *httptest.ResponseRecorder, want int) {
	t.Helper()
	if w.Code != want {
		t.Fatalf("status = %d, want %d (body %q)", w.Code, want, w.Body.String())
	}
}

func okUpstream(t *testing.T) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		if _, err := io.WriteString(w, `{"done":true}`); err != nil {
			t.Errorf("upstream write: %v", err)
		}
	})
}

func TestSessionsAPI_Lifecycle(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAPIFixture(t, tc, okUpstream(t))

		created := fx.do(http.MethodPost, sessionsAPIPrefix, "planner", `{"idleTimeoutSeconds":300}`)
		wantStatus(t, created, http.StatusCreated)
		sess := decodeSession(t, created)
		if !uuidV4Pattern.MatchString(sess.SessionID) {
			t.Fatalf("minted sessionId = %q, want a version 4 UUID", sess.SessionID)
		}
		if sess.AgentID == nil || *sess.AgentID != "planner" {
			t.Fatalf("agentId = %v, want planner", sess.AgentID)
		}
		if sess.IdleTimeoutSeconds != 300 || sess.KeepaliveSeconds != 100 {
			t.Fatalf("idle=%d keepalive=%d, want 300 and 100", sess.IdleTimeoutSeconds, sess.KeepaliveSeconds)
		}
		item := sessionsAPIPrefix + "/" + sess.SessionID

		t.Run("inference uses the session and reports its expiry", func(t *testing.T) {
			w := fx.infer(sess.SessionID, "planner")
			wantStatus(t, w, http.StatusOK)
			if _, err := time.Parse(time.RFC3339, w.Header().Get(sessionExpiresAtHeader)); err != nil {
				t.Fatalf("%s = %q: %v", sessionExpiresAtHeader, w.Header().Get(sessionExpiresAtHeader), err)
			}
			got := decodeSession(t, fx.do(http.MethodGet, item, "", ""))
			if got.Requests != 1 || got.Nodes[tc.profile.Name] != "node-a" {
				t.Fatalf("requests=%d nodes=%v, want 1 and %s=node-a", got.Requests, got.Nodes, tc.profile.Name)
			}
		})
		t.Run("keepalive needs the owner", func(t *testing.T) {
			wantStatus(t, fx.do(http.MethodPost, item+"/keepalive", "coder", ""), http.StatusConflict)
			wantStatus(t, fx.do(http.MethodPost, item+"/keepalive", "planner", ""), http.StatusOK)
		})
		t.Run("patch changes the idle timeout", func(t *testing.T) {
			w := fx.do(http.MethodPatch, item, "planner", `{"idleTimeoutSeconds":1200}`)
			wantStatus(t, w, http.StatusOK)
			if got := decodeSession(t, w).IdleTimeoutSeconds; got != 1200 {
				t.Fatalf("idleTimeoutSeconds = %d, want 1200", got)
			}
			wantStatus(t, fx.do(http.MethodPatch, item, "planner", `{"idleTimeoutSeconds":5}`), http.StatusBadRequest)
		})
		t.Run("list filters by agent", func(t *testing.T) {
			wantStatus(t, fx.do(http.MethodPost, sessionsAPIPrefix, "", `{"sessionId":"unowned"}`), http.StatusCreated)
			var list struct {
				Sessions []sessionJSON `json:"sessions"`
			}
			w := fx.do(http.MethodGet, sessionsAPIPrefix+"?agentId=planner", "", "")
			wantStatus(t, w, http.StatusOK)
			if err := json.Unmarshal(w.Body.Bytes(), &list); err != nil {
				t.Fatalf("decode list: %v", err)
			}
			if len(list.Sessions) != 1 || list.Sessions[0].SessionID != sess.SessionID {
				t.Fatalf("planner's sessions = %+v, want only %s", list.Sessions, sess.SessionID)
			}
		})
		t.Run("delete ends it for good", func(t *testing.T) {
			wantStatus(t, fx.do(http.MethodDelete, item, "planner", ""), http.StatusNoContent)
			wantStatus(t, fx.do(http.MethodGet, item, "", ""), http.StatusGone)
			wantStatus(t, fx.infer(sess.SessionID, "planner"), http.StatusGone)
		})
	})
}

func TestSessionsAPI_Refusals(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAPIFixture(t, tc, okUpstream(t))
		wantStatus(t, fx.do(http.MethodPost, sessionsAPIPrefix, "", `{"sessionId":"conv-1"}`), http.StatusCreated)

		refuse := func(name, method, path, agent, body string, want int) {
			t.Run(name, func(t *testing.T) {
				wantStatus(t, fx.do(method, path, agent, body), want)
			})
		}
		refuse("duplicate id", http.MethodPost, sessionsAPIPrefix, "", `{"sessionId":"conv-1"}`, http.StatusConflict)
		refuse("malformed id in body", http.MethodPost, sessionsAPIPrefix, "", `{"sessionId":"a b"}`, http.StatusBadRequest)
		refuse("unknown body field", http.MethodPost, sessionsAPIPrefix, "", `{"ttl":60}`, http.StatusBadRequest)
		refuse("agents disagree", http.MethodPost, sessionsAPIPrefix, "planner", `{"agentId":"coder"}`, http.StatusBadRequest)
		refuse("unknown session", http.MethodGet, sessionsAPIPrefix+"/nope", "", "", http.StatusNotFound)
		refuse("unknown sub-path", http.MethodGet, sessionsAPIPrefix+"/conv-1/other", "", "", http.StatusNotFound)
		refuse("wrong method", http.MethodPut, sessionsAPIPrefix+"/conv-1", "", "", http.StatusMethodNotAllowed)
	})
}

func TestSessionsAPI_IsLoopbackOnly(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newAPIFixture(t, tc, okUpstream(t))
		req := httptest.NewRequest(http.MethodGet, sessionsAPIPrefix, nil)
		req.RemoteAddr = "192.0.2.10:50000"
		w := httptest.NewRecorder()
		fx.proxy.soleFacade().handlePlain(w, req)
		wantStatus(t, w, http.StatusForbidden)
	})
}

func TestSessionsAPI_IsNeverForwarded(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		var hits int
		var mu sync.Mutex
		fx := newAPIFixture(t, tc, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			mu.Lock()
			hits++
			mu.Unlock()
			w.WriteHeader(http.StatusOK)
		}))
		fx.do(http.MethodGet, sessionsAPIPrefix, "", "")
		fx.do(http.MethodGet, sessionsAPIPrefix+"/missing", "", "")
		mu.Lock()
		defer mu.Unlock()
		if hits != 0 {
			t.Fatalf("session API reached the engine %d times, want 0", hits)
		}
	})
}

func TestSessionsAPI_DeleteCancelsInFlightInference(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		received := make(chan struct{}, 1)
		release := make(chan struct{})
		var once sync.Once
		fx := newAPIFixture(t, tc, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case received <- struct{}{}:
			default:
			}
			select {
			case <-release:
			case <-r.Context().Done():
			}
		}))
		defer once.Do(func() { close(release) })

		wantStatus(t, fx.do(http.MethodPost, sessionsAPIPrefix, "", `{"sessionId":"conv-1"}`), http.StatusCreated)
		done := make(chan struct{})
		go func() {
			defer close(done)
			fx.infer("conv-1", "")
		}()
		select {
		case <-received:
		case <-time.After(5 * time.Second):
			t.Fatal("upstream never received the request")
		}

		wantStatus(t, fx.do(http.MethodDelete, sessionsAPIPrefix+"/conv-1", "", ""), http.StatusNoContent)
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("ending the session did not stop its in-flight request")
		}

		if !waitFor(t, fx.rec, workloadErroredMethod) {
			t.Fatal("no terminal workload event after the session ended")
		}
		var sawCancelled bool
		for _, f := range decodeFrames(t, fx.rec) {
			if f.Method != workloadErroredMethod {
				continue
			}
			var p workloadParams
			if err := json.Unmarshal(f.Params, &p); err != nil {
				t.Fatalf("decode workload: %v", err)
			}
			if p.WorkloadInfo.State == "cancelled" {
				sawCancelled = true
			}
		}
		if !sawCancelled {
			t.Fatal("the ended session's request was not reported as cancelled")
		}
	})
}
