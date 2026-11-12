// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// hitEngine is a fake engine that counts the requests it serves and records
// whether PAIR's benchmark headers reached it.
type hitEngine struct {
	mu     sync.Mutex
	hits   int
	leaked []string
}

func (e *hitEngine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	_, _ = io.Copy(io.Discard, r.Body)
	e.mu.Lock()
	e.hits++
	for _, h := range []string{benchmarkHeader, benchmarkNodeHeader} {
		if r.Header.Get(h) != "" {
			e.leaked = append(e.leaked, h)
		}
	}
	e.mu.Unlock()
	w.Header().Set("Content-Type", "application/x-ndjson")
	_, _ = io.WriteString(w, ollamaChatStream)
}

func (e *hitEngine) count() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.hits
}

// twoNodeProxy routes to two nodes, node-a and node-b, that both have the
// model.
func twoNodeProxy(t *testing.T, tc engineCase) (*Proxy, *hitEngine, *hitEngine) {
	t.Helper()
	a, b := &hitEngine{}, &hitEngine{}
	sa, sb := httptest.NewServer(a), httptest.NewServer(b)
	t.Cleanup(sa.Close)
	t.Cleanup(sb.Close)
	disc := NewDiscovery()
	disc.AddManual(nodeForModel(t, "node-a", sa.URL, tc.advertisedModel))
	disc.AddManual(nodeForModel(t, "node-b", sb.URL, tc.advertisedModel))
	return testProxy(tc.profile, disc, tc.profile.FacadePort), a, b
}

func pinnedRequest(tc engineCase, node string, benchmark bool) *http.Request {
	req := tc.inferenceRequest()
	if benchmark {
		req.Header.Set(benchmarkHeader, "1")
	}
	if node != "" {
		req.Header.Set(benchmarkNodeHeader, node)
	}
	return req
}

// A benchmark step for a chosen node is served by that node, every time.
func TestBenchmarkNode_StepRunsOnTheChosenNode(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		p, a, b := twoNodeProxy(t, tc)
		for i := 0; i < 4; i++ {
			for _, pin := range []string{"node-a", "node-b"} {
				w := httptest.NewRecorder()
				p.soleFacade().handleHTTP(w, pinnedRequest(tc, pin, true))
				wantStatus(t, w, http.StatusOK)
			}
		}
		if a.count() != 4 || b.count() != 4 {
			t.Fatalf("node-a served %d, node-b %d; want 4 each", a.count(), b.count())
		}
		if len(a.leaked)+len(b.leaked) != 0 {
			t.Fatalf("PAIR's headers reached the engine: %v %v", a.leaked, b.leaked)
		}
	})
}

// A node that cannot serve the model fails the step at once rather than
// sending it elsewhere.
func TestBenchmarkNode_UnservableNodeIsRefused(t *testing.T) {
	tc := engineCases(t)[0]
	p, a, b := twoNodeProxy(t, tc)
	w := httptest.NewRecorder()
	p.soleFacade().handleHTTP(w, pinnedRequest(tc, "node-z", true))
	wantStatus(t, w, http.StatusBadGateway)
	if !strings.Contains(w.Body.String(), "benchmark-node") {
		t.Fatalf("body = %s; want the benchmark-node refusal", w.Body.String())
	}
	if a.count()+b.count() != 0 {
		t.Fatal("a request for an unknown node reached an engine")
	}
}

// The node header means nothing on an ordinary request: it is routed as
// usual and the header is not forwarded.
func TestBenchmarkNode_IgnoredWithoutTheBenchmarkMark(t *testing.T) {
	tc := engineCases(t)[0]
	p, a, b := twoNodeProxy(t, tc)
	w := httptest.NewRecorder()
	p.soleFacade().handleHTTP(w, pinnedRequest(tc, "node-z", false))
	wantStatus(t, w, http.StatusOK)
	if a.count()+b.count() != 1 {
		t.Fatalf("served %d+%d; want the request routed as usual", a.count(), b.count())
	}
	if len(a.leaked)+len(b.leaked) != 0 {
		t.Fatalf("PAIR's headers reached the engine: %v %v", a.leaked, b.leaked)
	}
}

// A run for a chosen node sends every step there and records the node.
func TestRunBenchmark_OnAChosenNode(t *testing.T) {
	tc := ollamaCase(t)
	p, a, b := twoNodeProxy(t, tc)
	srv := httptest.NewServer(http.HandlerFunc(p.soleFacade().handlePlain))
	t.Cleanup(srv.Close)
	port := srv.Listener.Addr().(*net.TCPAddr).Port

	run, err := p.perf.beginRun(tc.profile.Name, tc.requestedModel, "node-b")
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	p.runBenchmark(run, tc.profile.Name, port)

	got := p.perf.benchmarkSnapshot()[0]
	if got.NodeID != "node-b" || got.State != benchDone {
		t.Fatalf("run = %+v; want done on node-b", got)
	}
	if a.count() != 0 || b.count() != len(benchmarkPrompts) {
		t.Fatalf("node-a served %d, node-b %d; want every step on node-b", a.count(), b.count())
	}
	for _, step := range got.Steps {
		if step.Result != nil && step.Result.NodeID != "node-b" {
			t.Fatalf("step %s ran on %s", step.Name, step.Result.NodeID)
		}
	}
}

// Starting a run for a node outside the cluster is refused before anything
// is sent.
func TestStartBenchmark_UnknownNode(t *testing.T) {
	tc := ollamaCase(t)
	p, _, _ := twoNodeProxy(t, tc)
	if _, err := p.startBenchmark(benchmarkParams{Engine: tc.profile.Name, Model: tc.requestedModel, NodeID: "node-z"}); err == nil {
		t.Fatal("a run for an unknown node started")
	}
	if len(p.perf.benchmarkSnapshot()) != 0 {
		t.Fatal("a refused run was recorded")
	}
}
