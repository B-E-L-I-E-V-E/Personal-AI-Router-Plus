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
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestLivePAIRProxiesTheRequestToTheExpectedNode sends one real inference
// request to a running PAIR and checks, from PAIR's own record of that request,
// which node and engine served it. It is how to prove that a request made on
// this machine was proxied across the cluster to a particular peer.
//
// The request carries its own X-Request-ID, and PAIR's performance record for
// that ID (GET /pair/v1/performance on the same port) names the node that
// served it and whether it was this node. Nothing is inferred from the reply.
//
// It needs a running PAIR on a real cluster and is skipped unless PAIR_LIVE_URL
// is set:
//
//	PAIR_LIVE_URL         PAIR endpoint to send to: the router
//	                      (http://127.0.0.1:11400) or an engine port.
//	PAIR_LIVE_MODEL       model to request (default glm-4.7-flash).
//	PAIR_LIVE_NODE        node that must serve it: its name as the router's
//	                      model list shows it (for example Ultron), or its node
//	                      ID. Names are looked up on PAIR_LIVE_ROUTER_URL.
//	PAIR_LIVE_ENGINE      engine that must serve it (ollama, lmstudio,
//	                      llamacpp); optional.
//	PAIR_LIVE_ROUTER_URL  router used to look up node names (default
//	                      PAIR_LIVE_URL).
//	PAIR_LIVE_MAX_TOKENS  reply budget (default 256).
//
// For example, from services/nvpair-proxy:
//
//	PAIR_LIVE_URL=http://127.0.0.1:11400 PAIR_LIVE_MODEL=glm-4.7-flash \
//	  PAIR_LIVE_NODE=Ultron PAIR_LIVE_ENGINE=lmstudio \
//	  go test -run TestLivePAIRProxies -v .
func TestLivePAIRProxiesTheRequestToTheExpectedNode(t *testing.T) {
	base := strings.TrimRight(os.Getenv("PAIR_LIVE_URL"), "/")
	if base == "" {
		t.Skip("set PAIR_LIVE_URL to send a request through a running PAIR")
	}
	model := envOr("PAIR_LIVE_MODEL", "glm-4.7-flash")
	wantEngine := os.Getenv("PAIR_LIVE_ENGINE")
	routerURL := strings.TrimRight(envOr("PAIR_LIVE_ROUTER_URL", base), "/")
	maxTokens, err := strconv.Atoi(envOr("PAIR_LIVE_MAX_TOKENS", "256"))
	if err != nil {
		t.Fatalf("PAIR_LIVE_MAX_TOKENS: %v", err)
	}
	client := &http.Client{Timeout: 10 * time.Minute}

	wantNodeID, wantNodeName := "", os.Getenv("PAIR_LIVE_NODE")
	if wantNodeName != "" {
		wantNodeID = resolveLiveNode(t, client, routerURL, wantNodeName)
		t.Logf("expecting node %s (%s)", wantNodeName, wantNodeID)
	}

	requestID := fmt.Sprintf("pair-live-route-%d", time.Now().UnixNano())
	body, _ := json.Marshal(map[string]any{
		"model":      model,
		"max_tokens": maxTokens,
		"stream":     false,
		"messages":   []map[string]string{{"role": "user", "content": "Reply with the single word: ok"}},
	})
	req, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(requestIDHeader, requestID)
	started := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("POST %s/v1/chat/completions: %v", base, err)
	}
	reply, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	t.Logf("POST %s/v1/chat/completions model=%s → %d in %s", base, model, resp.StatusCode, time.Since(started).Round(time.Millisecond))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, want 200; body %s", resp.StatusCode, reply)
	}
	if got := resp.Header.Get(requestIDHeader); got != requestID {
		t.Errorf("X-Request-ID = %q, want %q", got, requestID)
	}
	var completion struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(reply, &completion); err != nil {
		t.Fatalf("reply is not a chat completion: %v; body %.300s", err, reply)
	}

	rec := waitForLivePerfRecord(t, client, base, requestID)
	t.Logf("PAIR recorded request %s: engine=%s node=%s local=%v", requestID, rec.Engine, rec.NodeID, rec.Local)
	if wantNodeID != "" {
		if rec.NodeID != wantNodeID {
			t.Errorf("served by node %s, want %s (%s)", rec.NodeID, wantNodeName, wantNodeID)
		}
		if rec.Local {
			t.Errorf("served by this node's own engine, want it proxied to %s", wantNodeName)
		}
	}
	if wantEngine != "" && rec.Engine != wantEngine {
		t.Errorf("served by engine %s, want %s", rec.Engine, wantEngine)
	}
}

var liveNodeIDPattern = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-`)

// resolveLiveNode turns a node name into its ID using the router's model list,
// which names every node that offers a model. An ID is returned as given.
func resolveLiveNode(t *testing.T, client *http.Client, routerURL, node string) string {
	t.Helper()
	if liveNodeIDPattern.MatchString(node) {
		return node
	}
	resp, err := client.Get(routerURL + "/v1/models")
	if err != nil {
		t.Fatalf("look up node %q on %s/v1/models: %v (give the node ID instead)", node, routerURL, err)
	}
	defer resp.Body.Close()
	var list struct {
		Data []struct {
			Pair *routerModelInfo `json:"pair"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("look up node %q: %s/v1/models answered %d (%v); give the node ID instead", node, routerURL, resp.StatusCode, err)
	}
	seen := map[string]string{}
	for _, m := range list.Data {
		if m.Pair == nil {
			continue
		}
		for _, n := range m.Pair.Nodes {
			seen[n.Name] = n.ID
			if strings.EqualFold(strings.TrimSuffix(n.Name, "."), strings.TrimSuffix(node, ".")) {
				return n.ID
			}
		}
	}
	t.Fatalf("no node named %q in %s/v1/models; nodes seen: %v", node, routerURL, seen)
	return ""
}

// waitForLivePerfRecord reads PAIR's record of one request. The record is
// written when the request finishes, so a short wait covers the race with the
// reply arriving here.
func waitForLivePerfRecord(t *testing.T, client *http.Client, base, requestID string) perfRecord {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
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
		if decodeErr != nil || resp.StatusCode != http.StatusOK {
			t.Fatalf("GET %s/pair/v1/performance answered %d (%v)", base, resp.StatusCode, decodeErr)
		}
		for _, rec := range report.Recent {
			if rec.RequestID == requestID {
				return rec
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("PAIR has no performance record for request %s", requestID)
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
