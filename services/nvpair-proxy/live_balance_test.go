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

// TestLivePAIRSpreadsConcurrentRequestsAcrossNodes sends a burst of concurrent
// requests for one model through a running PAIR router and reports, from
// PAIR's own record of each request, which node and engine served it. It
// fails if any request fails, or if every request landed on one node although
// the router lists the model on several.
//
// A burst is what the proxy's reservations exist for: each dispatch counts
// against its node until it ends, so concurrent requests should not all pick
// the node that looked idle when the burst began.
//
// It needs a running PAIR router on a real cluster and is skipped unless
// PAIR_LIVE_URL is set:
//
//	PAIR_LIVE_URL          the router, for example http://127.0.0.1:11400.
//	PAIR_LIVE_MODEL        a model the router lists on two or more nodes.
//	PAIR_LIVE_CONCURRENCY  requests sent at once (default 8).
//	PAIR_LIVE_ROUNDS       bursts to send, one after another (default 1).
//	PAIR_LIVE_MAX_TOKENS   reply budget per request (default 64).
//
// For example, from services/nvpair-proxy:
//
//	PAIR_LIVE_URL=http://127.0.0.1:11400 PAIR_LIVE_MODEL=flint \
//	  PAIR_LIVE_CONCURRENCY=12 go test -run TestLivePAIRSpreads -v .
func TestLivePAIRSpreadsConcurrentRequestsAcrossNodes(t *testing.T) {
	base := strings.TrimRight(os.Getenv("PAIR_LIVE_URL"), "/")
	if base == "" {
		t.Skip("set PAIR_LIVE_URL to send a burst through a running PAIR router")
	}
	model := os.Getenv("PAIR_LIVE_MODEL")
	if model == "" {
		t.Fatal("set PAIR_LIVE_MODEL to a model the router lists on two or more nodes")
	}
	concurrency := liveInt(t, "PAIR_LIVE_CONCURRENCY", 8)
	rounds := liveInt(t, "PAIR_LIVE_ROUNDS", 1)
	maxTokens := liveInt(t, "PAIR_LIVE_MAX_TOKENS", 64)
	client := &http.Client{Timeout: 10 * time.Minute}

	owners := liveModelOwners(t, client, base, model)
	names := map[string]string{}
	for _, n := range owners {
		names[n.ID] = n.Name
	}
	t.Logf("router lists %s on %d node/engine pair(s): %v", model, len(owners), owners)

	type outcome struct {
		requestID string
		status    int
		elapsed   time.Duration
		err       error
		body      string
	}
	var outcomes []outcome
	for round := 1; round <= rounds; round++ {
		results := make([]outcome, concurrency)
		start := make(chan struct{})
		var wg sync.WaitGroup
		for i := range concurrency {
			wg.Add(1)
			go func() {
				defer wg.Done()
				id := fmt.Sprintf("pair-live-burst-%d-r%d-%02d", time.Now().UnixNano(), round, i)
				results[i] = outcome{requestID: id}
				body, _ := json.Marshal(map[string]any{
					"model":      model,
					"max_tokens": maxTokens,
					"stream":     false,
					"messages": []map[string]string{{
						"role": "user", "content": fmt.Sprintf("Request %d: reply with the single word ok.", i),
					}},
				})
				req, _ := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set(requestIDHeader, id)
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
				if resp.StatusCode != http.StatusOK {
					results[i].body = strings.Join(strings.Fields(string(reply)), " ")
				}
			}()
		}
		burstStart := time.Now()
		close(start)
		wg.Wait()
		t.Logf("round %d: %d concurrent requests finished in %s", round, concurrency, time.Since(burstStart).Round(time.Millisecond))
		outcomes = append(outcomes, results...)
	}

	ids := make([]string, 0, len(outcomes))
	for _, o := range outcomes {
		switch {
		case o.err != nil:
			t.Errorf("request %s: %v", o.requestID, o.err)
		case o.status != http.StatusOK:
			t.Errorf("request %s: status %d after %s, want 200; body %.300s", o.requestID, o.status, o.elapsed.Round(time.Millisecond), o.body)
		default:
			ids = append(ids, o.requestID)
		}
	}
	records := waitForLivePerfRecords(t, client, base, ids)

	perPlace := map[string]int{}
	nodes := map[string]bool{}
	for _, o := range outcomes {
		rec, ok := records[o.requestID]
		if !ok {
			continue
		}
		place := fmt.Sprintf("%s/%s", liveNodeLabel(names, rec.NodeID), rec.Engine)
		if rec.Local {
			place += " (this node)"
		}
		perPlace[place]++
		nodes[rec.NodeID] = true
		t.Logf("  %s → %s in %s", o.requestID, place, o.elapsed.Round(time.Millisecond))
	}
	places := make([]string, 0, len(perPlace))
	for p := range perPlace {
		places = append(places, p)
	}
	sort.Strings(places)
	summary := make([]string, 0, len(places))
	for _, p := range places {
		summary = append(summary, fmt.Sprintf("%s=%d", p, perPlace[p]))
	}
	t.Logf("distribution of %d requests: %s", len(records), strings.Join(summary, ", "))

	ownerNodes := map[string]bool{}
	for _, n := range owners {
		ownerNodes[n.ID] = true
	}
	if len(ownerNodes) > 1 && concurrency > 1 && len(nodes) < 2 {
		t.Errorf("every request ran on one node although %d nodes offer %s", len(ownerNodes), model)
	}
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

// waitForLivePerfRecords collects PAIR's records for the given request IDs.
// Each is written when its request ends, so a short wait covers the race.
func waitForLivePerfRecords(t *testing.T, client *http.Client, base string, ids []string) map[string]perfRecord {
	t.Helper()
	want := map[string]bool{}
	for _, id := range ids {
		want[id] = true
	}
	found := map[string]perfRecord{}
	deadline := time.Now().Add(15 * time.Second)
	for {
		resp, err := client.Get(base + "/pair/v1/performance")
		if err != nil {
			t.Fatalf("GET %s/pair/v1/performance: %v", base, err)
		}
		var report struct {
			Recent []perfRecord `json:"recent"`
		}
		decodeErr := json.NewDecoder(resp.Body).Decode(&report)
		resp.Body.Close()
		if decodeErr != nil {
			t.Fatalf("decode performance report: %v", decodeErr)
		}
		for _, rec := range report.Recent {
			if want[rec.RequestID] {
				found[rec.RequestID] = rec
			}
		}
		if len(found) == len(want) {
			return found
		}
		if time.Now().After(deadline) {
			t.Errorf("PAIR recorded %d of %d requests", len(found), len(want))
			return found
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func liveNodeLabel(names map[string]string, id string) string {
	if name := names[id]; name != "" {
		return name
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
