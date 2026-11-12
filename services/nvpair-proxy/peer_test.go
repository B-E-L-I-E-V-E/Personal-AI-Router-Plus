// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
)

// peerFixture is two paired nodes: A, where requests start and which has no
// engine of its own, and B, whose engine serves them through its mTLS
// ingress. B enforces its own context capacity.
type peerFixture struct {
	t      *testing.T
	tc     engineCase
	a, b   *facade
	prompt atomic.Int64
	body   atomic.Value // the last inference body B's engine received ([]byte)
}

func newPeerFixture(t *testing.T, tc engineCase) *peerFixture {
	t.Helper()
	fx := &peerFixture{t: t, tc: tc}
	fx.prompt.Store(100)

	aDir, bDir := pairedClusterDirs(t)

	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		if r.URL.Path == tc.inferencePath {
			fx.body.Store(b)
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, "{\"done\":true,\"prompt_eval_count\":%d,\"eval_count\":10}\n", fx.prompt.Load())
	}))
	t.Cleanup(engine.Close)

	fx.b = testProxy(tc.profile, NewDiscovery(), tc.profile.StandalonePort).soleFacade()
	fx.b.host.mesh = clustertrust.Open(bDir)
	backend := nodeFor(t, "engine", engine.URL)
	if err := fx.b.setLocalBackend(localBackend{Host: backend.Addresses[0], Port: backend.Port, Healthy: true}); err != nil {
		t.Fatalf("set local backend: %v", err)
	}
	ingress := httptest.NewUnstartedServer(http.HandlerFunc(fx.b.handleClusterIngress))
	ingress.TLS = fx.b.host.mesh.ServerTLSConfig()
	ingress.StartTLS()
	t.Cleanup(ingress.Close)

	fx.a = testProxy(tc.profile, NewDiscovery(), tc.profile.StandalonePort).soleFacade()
	fx.a.host.mesh = clustertrust.Open(aDir)
	n := nodeForModel(t, "node-b", ingress.URL, tc.advertisedModel)
	n.ClusterUUID = "b"
	fx.a.discovery.SetSubscribed([]Node{n})
	return fx
}

// pairedClusterDirs makes two cluster identities, "a" and "b", that trust
// each other, and returns their state directories.
func pairedClusterDirs(t *testing.T) (aDir, bDir string) {
	t.Helper()
	aDir, bDir = t.TempDir(), t.TempDir()
	clustertrusttest.Join(t, aDir, "cluster", "a")
	clustertrusttest.Join(t, bDir, "cluster", "b")
	pin := func(dst, src, id string) {
		t.Helper()
		pem, err := os.ReadFile(filepath.Join(src, "node.crt"))
		if err != nil {
			t.Fatalf("read cert: %v", err)
		}
		body, err := json.Marshal(map[string]string{"nodeUuid": id, "certPem": string(pem)})
		if err != nil {
			t.Fatalf("encode pin: %v", err)
		}
		if err := os.MkdirAll(filepath.Join(dst, "trusted"), 0o700); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(dst, "trusted", id+".json"), body, 0o600); err != nil {
			t.Fatalf("write pin: %v", err)
		}
	}
	pin(aDir, bDir, "b")
	pin(bDir, aDir, "a")
	return aDir, bDir
}

// limitB sets B's capacity for the model and how long B queues.
func (fx *peerFixture) limitB(contextTokens int64, sessions int, queue time.Duration) {
	fx.limitBWithReply(contextTokens, sessions, 0, queue)
}

// limitBWithReply also keeps reply tokens of each of B's session windows for
// replies.
func (fx *peerFixture) limitBWithReply(contextTokens int64, sessions int, reply int64, queue time.Duration) {
	fx.t.Helper()
	k := capacityKey{Engine: fx.tc.profile.Name, Model: fx.tc.requestedModel}
	if err := fx.b.host.capacity.set(k, capacitySetting{ContextTokens: contextTokens, MaxSessions: sessions, ReplyTokens: reply}, false); err != nil {
		fx.t.Fatalf("set capacity: %v", err)
	}
	cfg := fx.b.host.sessions.config()
	cfg.QueueTimeout = queue
	if err := fx.b.host.sessions.configure(cfg); err != nil {
		fx.t.Fatalf("configure: %v", err)
	}
}

func (fx *peerFixture) heldOnB() int {
	return fx.b.host.sessions.slotStatus(capacityKey{Engine: fx.tc.profile.Name, Model: fx.tc.requestedModel}).Held
}

// send issues one session turn on A; pad lengthens the body.
func (fx *peerFixture) send(session string, pad int) *httptest.ResponseRecorder {
	fx.t.Helper()
	body := fmt.Sprintf(`{"model":%q,"pad":%q}`, fx.tc.requestedModel, strings.Repeat("x", pad))
	req := httptest.NewRequest(http.MethodPost, fx.tc.inferencePath, strings.NewReader(body))
	req.RemoteAddr = "127.0.0.1:50000"
	req.Header.Set(sessionIDHeader, session)
	w := httptest.NewRecorder()
	fx.a.handlePlain(w, req)
	return w
}

// shortJobDeadline bounds A's retries so a refused request returns promptly.
func shortJobDeadline(t *testing.T) {
	t.Helper()
	prev := jobDeadline
	jobDeadline = 1500 * time.Millisecond
	t.Cleanup(func() { jobDeadline = prev })
}

func TestPeerAdmission_ServingNodeHoldsTheSlot(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newPeerFixture(t, tc)
		fx.limitB(65536, 1, 0)
		shortJobDeadline(t)

		wantStatus(t, fx.send("conv-1", 0), http.StatusOK)
		if got := fx.heldOnB(); got != 1 {
			t.Fatalf("slots held on B = %d, want 1 for A's session", got)
		}
		if got := len(fx.b.host.sessions.list("")); got != 0 {
			t.Fatalf("B lists %d sessions, want none: A's session is A's to list", got)
		}
		wantStatus(t, fx.send("conv-1", 0), http.StatusOK) // same session keeps the slot

		t.Run("a second session is refused while B is full", func(t *testing.T) {
			wantStatus(t, fx.send("conv-2", 0), http.StatusServiceUnavailable)
		})
		t.Run("ending the session on A frees B's slot", func(t *testing.T) {
			if err := fx.a.host.sessions.remove("conv-1", ""); err != nil {
				t.Fatalf("end conv-1: %v", err)
			}
			waitUntil(t, "B releases the slot", func() bool { return fx.heldOnB() == 0 })
			wantStatus(t, fx.send("conv-2", 0), http.StatusOK)
		})
	})
}

func TestPeerAdmission_ServingNodeEnforcesTheShare(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newPeerFixture(t, tc)
		fx.limitB(8192, 1, 0)
		fx.prompt.Store(9000)

		wantStatus(t, fx.send("conv-1", 200), http.StatusOK)
		w := fx.send("conv-1", 300)
		wantStatus(t, w, http.StatusBadRequest)
		if !strings.Contains(w.Body.String(), "serving node") {
			t.Fatalf("body = %q, want the serving node's share refusal", w.Body.String())
		}
		fx.prompt.Store(500)
		wantStatus(t, fx.send("conv-1", 10), http.StatusOK)
	})
}

func TestPeerSessions_AreScopedToTheCallingNode(t *testing.T) {
	s := newSessionStore()
	k := capacityKey{Engine: "ollama", Model: "m"}
	for _, peer := range []string{"a", "c"} {
		key := peerSessionKey(peer, "conv-1")
		if _, _, err := s.beginPeer(key, peer, func() {}); err != nil {
			t.Fatalf("beginPeer %s: %v", peer, err)
		}
		if !s.takeSlotNow(key, k, 2) {
			t.Fatalf("peer %s could not take a slot", peer)
		}
	}
	if got := s.slotStatus(k).Held; got != 2 {
		t.Fatalf("held = %d, want 2: the same session ID from two nodes is two sessions", got)
	}
	t.Run("only the owning node may release", func(t *testing.T) {
		if s.releasePeer(peerSessionKey("a", "conv-1"), "c") {
			t.Fatal("node c released node a's session")
		}
		if !s.releasePeer(peerSessionKey("a", "conv-1"), "a") {
			t.Fatal("node a could not release its own session")
		}
	})
	t.Run("a released peer session can start again", func(t *testing.T) {
		if _, _, err := s.beginPeer(peerSessionKey("a", "conv-1"), "a", func() {}); err != nil {
			t.Fatalf("beginPeer after release: %v", err)
		}
	})
}

func TestPin_ReleasesTheNodeASessionLeaves(t *testing.T) {
	s := newSessionStore()
	var mu sync.Mutex
	var released []string
	done := make(chan struct{}, 4)
	s.onPeerRelease = func(ref remoteRef, id string) {
		mu.Lock()
		released = append(released, ref.nodeID+"/"+id)
		mu.Unlock()
		done <- struct{}{}
	}
	beginT(t, s, "conv-1", "")
	peerAt := func(id, uuid string) candidate {
		return candidate{id: id, peerUUID: uuid, url: mustURL(t, "https://"+id+":11434")}
	}
	s.pin("conv-1", "ollama", peerAt("node-b", "b"))
	s.pin("conv-1", "ollama", peerAt("node-b", "b")) // staying put releases nothing
	s.pin("conv-1", "ollama", peerAt("node-c", "c"))
	<-done
	if err := s.remove("conv-1", ""); err != nil {
		t.Fatalf("remove: %v", err)
	}
	<-done
	mu.Lock()
	defer mu.Unlock()
	if strings.Join(released, ",") != "node-b/conv-1,node-c/conv-1" {
		t.Fatalf("released = %v, want node-b on the move, then node-c on end", released)
	}
}

func mustURL(t *testing.T, raw string) *url.URL {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("parse %q: %v", raw, err)
	}
	return u
}

func TestPeerAdmission_ServingNodeAppliesItsReplyBudget(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := newPeerFixture(t, tc)
		fx.limitBWithReply(8192, 1, 2048, 0) // 6144 left for the conversation on B
		fx.prompt.Store(6500)

		wantStatus(t, fx.send("conv-1", 200), http.StatusOK)
		sent, _ := fx.body.Load().([]byte)
		var v map[string]any
		if err := json.Unmarshal(sent, &v); err != nil {
			t.Fatalf("B's engine got %q", sent)
		}
		switch {
		case tc.profile.Name == "ollama" && (tc.inferencePath == "/api/chat" || tc.inferencePath == "/api/generate"):
			if opts, _ := v["options"].(map[string]any); v["truncate"] != false || opts["num_predict"] != float64(2048) {
				t.Fatalf("B's engine got %v, want truncate false and num_predict 2048", v)
			}
		case isOpenAIInferencePath(tc.inferencePath):
			if v["max_tokens"] != float64(2048) {
				t.Fatalf("B's engine got %v, want max_tokens 2048", v)
			}
		}
		// Under B's window but over what its conversation may use.
		w := fx.send("conv-1", 300)
		wantStatus(t, w, http.StatusBadRequest)
		if !strings.Contains(w.Body.String(), "serving node") || !strings.Contains(w.Body.String(), "6144") {
			t.Fatalf("body = %q, want the serving node's conversation limit", w.Body.String())
		}
	})
}
