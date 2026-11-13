// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestLivePAIRBalancesAConcurrentBurstByLoad sends a burst of concurrent
// requests for one model through a running PAIR router and checks that it was
// split the way the router's rule says it should be.
//
// The rule: each request goes to the eligible node with the least load, where
// load is the scheduler's pending work plus GPU pressure plus the reservations
// this proxy holds, and each dispatch adds a reservation to its node. So an
// even split is not the goal: a burst arriving while one node is busy should
// mostly go to the other. The test reads the load from GET /pair/v1/routing
// just before the burst, works out the split that rule gives, and compares it
// with where each reply says it came from (X-PAIR-Node-ID / X-PAIR-Engine).
//
// It fails when the split differs from the prediction, or when PAIR itself
// refuses a request. A reply an engine produced is attributed to its node
// whatever its status: an engine that cannot run the model (LM Studio
// answering 400 "Failed to load model") is reported, but it is the engine's
// failure, not the router's choice.
//
// The prediction is exact only while the scheduler's figures stay put. On a
// live cluster they rarely do: every workload event triggers a new snapshot,
// often a dozen or more during one burst, and each replaces this proxy's
// reservations with the scheduler's own count, which may not yet include
// requests dispatched a moment earlier. Requests that finish within the burst
// free their node for later arrivals too. So when the routing generation
// changed during the burst, each node may be up to a quarter of the burst
// (rounded up) off the prediction; when it did not, the split must match
// exactly. Either way a burst that piles onto one node fails.
//
// A burst also first loads the model wherever it is not loaded, and an engine
// still loading can turn a request away to another node. So the test sends an
// uncounted warm-up burst first.
//
// It needs a running PAIR router on a real cluster and is skipped unless
// PAIR_LIVE_URL is set:
//
//	PAIR_LIVE_URL          the router, for example http://127.0.0.1:11400.
//	PAIR_LIVE_MODEL        a model the router lists on two or more nodes.
//	PAIR_LIVE_CONCURRENCY  requests sent at once (default 8).
//	PAIR_LIVE_MAX_TOKENS   reply budget per request (default 32).
//	PAIR_LIVE_PATH         /v1/chat/completions (default) or /v1/embeddings,
//	                       for an embedding model.
//	PAIR_LIVE_WARMUP       requests in the uncounted warm-up burst (default
//	                       the burst size; 0 skips it).
//
// Replies are attributed by where they finally came from. If one node's
// engine fails and the request fails over, it is counted where it landed, so
// use a model every listed node can actually run.
//
// For example, from services/nvpair-proxy:
//
//	PAIR_LIVE_URL=http://127.0.0.1:11400 PAIR_LIVE_MODEL=flint \
//	  go test -run TestLivePAIRBalances -v .
func TestLivePAIRBalancesAConcurrentBurstByLoad(t *testing.T) {
	base := strings.TrimRight(os.Getenv("PAIR_LIVE_URL"), "/")
	if base == "" {
		t.Skip("set PAIR_LIVE_URL to send a burst through a running PAIR router")
	}
	model := os.Getenv("PAIR_LIVE_MODEL")
	if model == "" {
		t.Fatal("set PAIR_LIVE_MODEL to a model the router lists on two or more nodes")
	}
	concurrency := liveInt(t, "PAIR_LIVE_CONCURRENCY", 8)
	maxTokens := liveInt(t, "PAIR_LIVE_MAX_TOKENS", 32)
	path := envOr("PAIR_LIVE_PATH", "/v1/chat/completions")
	if path != "/v1/chat/completions" && path != "/v1/embeddings" {
		t.Fatalf("PAIR_LIVE_PATH=%q: want /v1/chat/completions or /v1/embeddings", path)
	}
	client := &http.Client{Timeout: 10 * time.Minute}

	owners := liveModelOwners(t, client, base, model)
	names := map[string]string{}
	for _, n := range owners {
		names[n.ID] = n.Name
	}
	t.Logf("router lists %s on: %s", model, liveOwnerList(owners))

	type outcome struct {
		requestID string
		status    int
		node      string
		engine    string
		elapsed   time.Duration
		err       error
		body      string
	}
	results := make([]outcome, concurrency)
	requests := make([]*http.Request, concurrency)
	for i := range concurrency {
		id := fmt.Sprintf("pair-live-burst-%d-%02d", time.Now().UnixNano(), i)
		results[i].requestID = id
		payload := map[string]any{
			"model":      model,
			"max_tokens": maxTokens,
			"stream":     false,
			"messages": []map[string]string{{
				"role": "user", "content": fmt.Sprintf("Request %d: reply with the single word ok.", i),
			}},
		}
		if path == "/v1/embeddings" {
			payload = map[string]any{"model": model, "input": fmt.Sprintf("PAIR load-balancing check, request %d.", i)}
		}
		body, _ := json.Marshal(payload)
		req, err := http.NewRequest(http.MethodPost, base+path, bytes.NewReader(body))
		if err != nil {
			t.Fatalf("request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set(requestIDHeader, id)
		requests[i] = req
	}

	warmup := concurrency
	if v := os.Getenv("PAIR_LIVE_WARMUP"); v != "" {
		if warmup, _ = strconv.Atoi(v); warmup < 0 {
			t.Fatalf("PAIR_LIVE_WARMUP=%q: want 0 or more", v)
		}
	}
	if warmup > 0 {
		liveWarmUp(t, client, base, path, requests[0], warmup)
	}

	before := liveRouting(t, client, base)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i, req := range requests {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			began := time.Now()
			resp, err := client.Do(req)
			results[i].elapsed = time.Since(began)
			if err != nil {
				results[i].err = err
				return
			}
			reply, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
			resp.Body.Close()
			results[i].status = resp.StatusCode
			results[i].node = resp.Header.Get(servedNodeHeader)
			results[i].engine = resp.Header.Get(servedEngineHeader)
			if resp.StatusCode != http.StatusOK {
				results[i].body = strings.Join(strings.Fields(string(reply)), " ")
			}
		}()
	}
	burstStart := time.Now()
	close(start)
	wg.Wait()
	after := liveRouting(t, client, base)
	t.Logf("%d concurrent requests finished in %s", concurrency, time.Since(burstStart).Round(time.Millisecond))

	t.Logf("load before the burst (generation %d):", before.Generation)
	for _, n := range before.Nodes {
		if _, owner := names[n.ID]; owner {
			t.Logf("  %s: pending %d + GPU pressure %d + reservations %d = %d",
				liveNodeLabel(names, n.ID), n.Pending, n.GPUPressure, n.Reservations, n.Load)
		}
	}

	actual := map[string]int{}
	engineErrors := 0
	for _, o := range results {
		place := liveNodeLabel(names, o.node) + "/" + o.engine
		switch {
		case o.err != nil:
			t.Errorf("%s: %v", o.requestID, o.err)
			continue
		case o.node == "":
			// No engine produced this reply: PAIR refused it itself.
			t.Errorf("%s: PAIR refused it with %d: %.300s", o.requestID, o.status, o.body)
			continue
		case o.status != http.StatusOK:
			engineErrors++
			t.Logf("  %s → %s answered %d in %s (engine error, not a routing failure): %.200s",
				o.requestID, place, o.status, o.elapsed.Round(time.Millisecond), o.body)
		default:
			t.Logf("  %s → %s answered 200 in %s", o.requestID, place, o.elapsed.Round(time.Millisecond))
		}
		actual[o.node]++
	}

	predicted, predictable := predictBurstSplit(before, owners, concurrency)
	t.Logf("split: %s", liveSplit(names, actual))
	if engineErrors > 0 {
		t.Logf("%d of %d replies were engine errors; they still show where the router sent them", engineErrors, concurrency)
	}
	if !predictable {
		t.Logf("no prediction: not every node offering %s is in the scheduler's ranking", model)
		return
	}
	t.Logf("predicted from the load: %s", liveSplit(names, predicted))
	tolerance := 0
	if after.Generation != before.Generation {
		tolerance = (concurrency + 3) / 4
		t.Logf("the scheduler's figures changed during the burst (generation %d → %d), so each node may be %d request(s) off",
			before.Generation, after.Generation, tolerance)
	}
	for id := range liveUnion(actual, predicted) {
		if diff := actual[id] - predicted[id]; diff > tolerance || -diff > tolerance {
			t.Errorf("%s got %d requests, the load predicts %d (allowed difference %d)",
				liveNodeLabel(names, id), actual[id], predicted[id], tolerance)
		}
	}
}

// liveWarmUp sends an uncounted concurrent burst shaped like template, so
// every node offering the model has it loaded before the measured burst.
func liveWarmUp(t *testing.T, client *http.Client, base, path string, template *http.Request, n int) {
	t.Helper()
	body, err := template.GetBody()
	if err != nil {
		t.Fatalf("warm-up body: %v", err)
	}
	payload, _ := io.ReadAll(body)
	var wg sync.WaitGroup
	statuses := make([]int, n)
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req, _ := http.NewRequest(http.MethodPost, base+path, bytes.NewReader(payload))
			req.Header.Set("Content-Type", "application/json")
			resp, err := client.Do(req)
			if err != nil {
				return
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			statuses[i] = resp.StatusCode
		}()
	}
	wg.Wait()
	t.Logf("warm-up: %d requests, statuses %v", n, statuses)
}

func liveUnion(a, b map[string]int) map[string]bool {
	out := map[string]bool{}
	for k := range a {
		out[k] = true
	}
	for k := range b {
		out[k] = true
	}
	return out
}

// predictBurstSplit applies reserveCandidate's rule to a burst: each request
// takes the ranked owner with the least load, ties going to the higher rank,
// and adds one reservation there. It reports false when an owner is not
// ranked, because the rule then falls back to candidate order.
func predictBurstSplit(routing routingReport, owners []routerNode, burst int) (map[string]int, bool) {
	ranked := map[string]routingNode{}
	for _, n := range routing.Nodes {
		ranked[n.ID] = n
	}
	type slot struct {
		id         string
		rank, load int
	}
	var slots []slot
	seen := map[string]bool{}
	for _, o := range owners {
		if seen[o.ID] {
			continue
		}
		seen[o.ID] = true
		n, ok := ranked[o.ID]
		if !ok {
			return nil, false
		}
		slots = append(slots, slot{id: o.ID, rank: n.Rank, load: n.Load})
	}
	split := map[string]int{}
	for range burst {
		best := 0
		for i, s := range slots {
			if s.load < slots[best].load || (s.load == slots[best].load && s.rank < slots[best].rank) {
				best = i
			}
		}
		slots[best].load++
		split[slots[best].id]++
	}
	return split, true
}

func liveRouting(t *testing.T, client *http.Client, base string) routingReport {
	t.Helper()
	resp, err := client.Get(base + routingPath)
	if err != nil {
		t.Fatalf("GET %s%s: %v", base, routingPath, err)
	}
	defer resp.Body.Close()
	var report routingReport
	if err := json.NewDecoder(resp.Body).Decode(&report); err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s%s answered %d (%v); PAIR_LIVE_URL must be the PAIR router", base, routingPath, resp.StatusCode, err)
	}
	return report
}

// liveModelOwners reads the router's model list for the nodes offering model.
func liveModelOwners(t *testing.T, client *http.Client, base, model string) []routerNode {
	t.Helper()
	resp, err := client.Get(base + "/v1/models")
	if err != nil {
		t.Fatalf("GET %s/v1/models: %v", base, err)
	}
	defer resp.Body.Close()
	var list struct {
		Data []struct {
			ID   string           `json:"id"`
			Pair *routerModelInfo `json:"pair"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s/v1/models answered %d (%v); PAIR_LIVE_URL must be the PAIR router", base, resp.StatusCode, err)
	}
	for _, m := range list.Data {
		if m.ID == model && m.Pair != nil {
			return m.Pair.Nodes
		}
	}
	t.Fatalf("the router does not list %s", model)
	return nil
}

func liveOwnerList(owners []routerNode) string {
	parts := make([]string, 0, len(owners))
	for _, o := range owners {
		parts = append(parts, o.Name+"/"+o.Engine)
	}
	return strings.Join(parts, ", ")
}

func liveSplit(names map[string]string, split map[string]int) string {
	ids := make([]string, 0, len(split))
	for id := range split {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	parts := make([]string, 0, len(ids))
	for _, id := range ids {
		parts = append(parts, fmt.Sprintf("%s=%d", liveNodeLabel(names, id), split[id]))
	}
	return strings.Join(parts, ", ")
}

func liveNodeLabel(names map[string]string, id string) string {
	if name := names[id]; name != "" {
		return name
	}
	if id == "" {
		return "(no node)"
	}
	return id
}

func liveInt(t *testing.T, key string, fallback int) int {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		t.Fatalf("%s=%q: want a positive integer", key, v)
	}
	return n
}
