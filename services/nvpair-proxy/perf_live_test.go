// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestPerformanceLive_ShowsARequestWhileItStreams(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		release := make(chan struct{})
		upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			w.Header().Set("Content-Type", "application/x-ndjson")
			fl, _ := w.(http.Flusher)
			for _, line := range []string{
				`{"message":{"role":"assistant","content":"","thinking":"Hmm"},"done":false}`,
				`{"message":{"role":"assistant","content":"","thinking":" so"},"done":false}`,
				`{"message":{"role":"assistant","content":"Paris"},"done":false}`,
			} {
				_, _ = io.WriteString(w, line+"\n")
				fl.Flush()
			}
			<-release
			_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":12,"eval_count":3,"prompt_eval_duration":60000000,"eval_duration":30000000}`+"\n")
		})
		fx := newAPIFixture(t, tc, upstream)
		path := "/v1/chat/completions"
		if tc.profile.Name == "ollama" {
			path = "/api/chat"
		}
		done := make(chan struct{})
		go func() {
			defer close(done)
			req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(fmt.Sprintf(`{"model":%q,"stream":true}`, tc.requestedModel)))
			req.RemoteAddr = "127.0.0.1:50000"
			req.Header.Set(sessionIDHeader, "conv-live")
			req.Header.Set(agentIDHeader, "planner")
			fx.proxy.soleFacade().handlePlain(httptest.NewRecorder(), req)
		}()

		var got liveRequest
		waitUntil(t, "the live request to reach its answer", func() bool {
			live := fx.proxy.performanceLive().Live
			if len(live) != 1 {
				return false
			}
			got = live[0]
			return got.Phase == livePhaseAnswering
		})
		if got.SessionID != "conv-live" || got.AgentID != "planner" || got.Model != tc.requestedModel || got.NodeID != "node-a" {
			t.Fatalf("live request = %+v, want session conv-live, agent planner, %s on node-a", got, tc.requestedModel)
		}
		if got.OutputTokens != 3 || got.ThinkingTokens != 2 || got.TTFTMs == nil || got.ThinkingMs == nil {
			t.Fatalf("live counts = %+v, want 3 tokens out, 2 thinking, a first-token time and a thinking time", got)
		}
		if got.PromptTokensEstimate <= 0 || got.Origin != liveOriginRouted {
			t.Fatalf("live request = %+v, want a prompt estimate from the body and origin routed", got)
		}

		close(release)
		<-done
		rep := fx.proxy.performanceLive()
		if len(rep.Live) != 0 {
			t.Fatalf("after the reply ended: %d live requests, want none", len(rep.Live))
		}
		if len(rep.Recent) != 1 || rep.Recent[0].SessionID != "conv-live" || rep.Recent[0].PromptTokens != 12 {
			t.Fatalf("recent = %+v, want the finished request with its session and the engine's 12 prompt tokens", rep.Recent)
		}

		res := fx.do(http.MethodGet, performanceAPIPath+"/live", "", "")
		wantStatus(t, res, http.StatusOK)
		var body perfLive
		if err := json.Unmarshal(res.Body.Bytes(), &body); err != nil || len(body.Recent) != 1 {
			t.Fatalf("GET live = %s (%v), want the finished request", res.Body.String(), err)
		}
	})
}

func TestLiveRegistry_DispatchRestartsCountsOnFailover(t *testing.T) {
	l := newLiveRegistry()
	e, end := l.begin(liveRequest{RequestID: "r", Model: "m"}, 400)
	defer end()
	t0 := time.Now()
	l.dispatch(e, "node-a", false, t0)
	l.progress(e, t0.Add(10*time.Millisecond), false, true, true)
	l.dispatch(e, "node-b", true, t0.Add(20*time.Millisecond))
	got := l.snapshot(t0.Add(30 * time.Millisecond))[0]
	if got.NodeID != "node-b" || !got.Local || got.OutputTokens != 0 || got.Phase != livePhaseReading || got.TTFTMs != nil {
		t.Fatalf("after failover = %+v, want node-b reading with no tokens yet", got)
	}
	if got.PromptTokensEstimate != 100 {
		t.Fatalf("prompt estimate = %d, want 100 (400 bytes / 4)", got.PromptTokensEstimate)
	}
}

func TestMeasurePeerRequest_RecordsTheReplyAsServedHere(t *testing.T) {
	tc := ollamaCase(t)
	fx := newAPIFixture(t, tc, http.NotFoundHandler())
	f := fx.proxy.soleFacade()
	fx.proxy.selfNodeID.Store("self-id")

	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"qwen3","stream":true}`))
	req.Header.Set(sessionIDHeader, "peer-conv")
	tr, end := f.measurePeerRequest(req, "peer-node")
	if tr == nil {
		t.Fatal("a peer generation request was not measured")
	}
	if b, _ := io.ReadAll(req.Body); string(b) != `{"model":"qwen3","stream":true}` {
		t.Fatalf("forwarded body = %q, want it replayed unchanged", b)
	}
	live := fx.proxy.performanceLive().Live
	if len(live) != 1 || live[0].Origin != liveOriginPeer || live[0].NodeID != "self-id" || live[0].SessionID != "peer-conv" {
		t.Fatalf("live = %+v, want one peer request served by self-id", live)
	}
	body := newObservedBody(io.NopCloser(strings.NewReader(ollamaChatStream)), "application/x-ndjson", nil, tr)
	_, _ = io.ReadAll(body)
	end()
	rep := fx.proxy.perf.report()
	if len(rep.Models) != 1 || rep.Models[0].NodeID != "self-id" || rep.Recent[0].Origin != liveOriginPeer {
		t.Fatalf("report = %+v, want the peer request recorded on self-id", rep.Models)
	}

	embed := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(`{"model":"qwen3"}`))
	if tr, end := f.measurePeerRequest(embed, "peer-node"); tr != nil {
		end()
		t.Fatal("an embedding request was measured")
	}
}
