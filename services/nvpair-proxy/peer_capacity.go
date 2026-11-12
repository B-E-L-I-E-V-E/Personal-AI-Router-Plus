// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"runtime/debug"
	"time"
)

// Another node's context capacity, read over mTLS.
//
// Each node enforces its own capacity (peer.go), so only that node can say
// what it enforces. Its ingress answers a read-only copy of its capacity
// report to pinned cluster members; nothing on the ingress changes capacity.
// The desktop reaches only its own proxy, so the local proxy fetches a peer's
// report on its behalf (capacity/peer-report).
const (
	// peerCapacityPath is served on the mTLS ingress only, and only to GET.
	peerCapacityPath = "/pair/v1/peer/capacity"
	// peerCapacityTimeout covers the peer's own detection (detectionTimeout)
	// plus the round trip.
	peerCapacityTimeout = detectionTimeout + 2*time.Second
	// maxPeerCapacityBody bounds what is read from a peer's answer.
	maxPeerCapacityBody = 1 << 20
)

var errPeerNotReachable = errors.New("node is not a reachable cluster peer")

// answerPeerCapacity answers capacity/peer-report {"nodeId"} with that node's
// report. It runs on its own goroutine so a slow or unreachable peer never
// holds up the control loop; the codec serializes the reply with every other
// write.
func (p *Proxy) answerPeerCapacity(msg *Message) {
	var params struct {
		NodeID string `json:"nodeId"`
	}
	if err := json.Unmarshal(msg.Params, &params); err != nil || params.NodeID == "" {
		p.codec.RespondError(msg.ID, -32602, `invalid params: expected {"nodeId"}`)
		return
	}
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.Error("recovered from a panic while reading a peer's capacity",
					"panic", fmt.Sprint(recovered), "stack", string(debug.Stack()))
				p.codec.RespondError(msg.ID, -32603, "internal error handling capacity/peer-report")
			}
		}()
		ctx, cancel := context.WithTimeout(context.Background(), 2*peerCapacityTimeout)
		defer cancel()
		rows, err := p.peerCapacityReport(ctx, params.NodeID)
		if err != nil {
			slog.Debug("peer capacity read failed", "node_id", params.NodeID, "err", err)
			p.codec.RespondError(msg.ID, -32000, err.Error())
			return
		}
		if err := p.codec.Respond(msg.ID, map[string]any{"nodeId": params.NodeID, "models": rows}); err != nil {
			log.Printf("failed to respond to capacity/peer-report: %v", err)
		}
	}()
}

// servePeerCapacity answers a pinned peer's read of this node's capacity.
func (f *facade) servePeerCapacity(w http.ResponseWriter, r *http.Request) bool {
	if r.URL.Path != peerCapacityPath {
		return false
	}
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeIngressError(w, http.StatusMethodNotAllowed, "peer-capacity", "expected GET "+peerCapacityPath)
		return true
	}
	ctx, cancel := context.WithTimeout(r.Context(), detectionTimeout+time.Second)
	defer cancel()
	writeSessionJSON(w, http.StatusOK, map[string]any{"models": f.host.capacityReport(ctx)})
	return true
}

// peerCapacityReport fetches the capacity report of the node with this
// discovery ID. The node must be a pinned cluster peer, reached over mTLS
// like any routed request; this node's own ID answers locally. Every facade
// that knows the node is tried, since any of its ingresses holds the same
// report.
func (p *Proxy) peerCapacityReport(ctx context.Context, nodeID string) ([]modelCapacity, error) {
	p.mesh.Refresh()
	var lastErr error = errPeerNotReachable
	for _, f := range p.enabledFacades() {
		selfPort, aliasBound := f.selfAddresses()
		for _, n := range f.discovery.Nodes() {
			if n.ID != nodeID {
				continue
			}
			u := f.targetURL(n)
			if u == nil {
				continue
			}
			if isSelfTarget(u, selfPort) || isAnyAliasSelfTarget(u, aliasBound) {
				return p.capacityReport(ctx), nil
			}
			if !p.mesh.HasPin(n.ClusterUUID) {
				continue
			}
			u.Scheme = "https"
			u.Path = peerCapacityPath
			rows, err := p.fetchPeerCapacity(ctx, u.String(), n.ClusterUUID)
			if err == nil {
				return rows, nil
			}
			lastErr = err
		}
	}
	return nil, lastErr
}

func (p *Proxy) fetchPeerCapacity(ctx context.Context, target, peerUUID string) ([]modelCapacity, error) {
	ctx, cancel := context.WithTimeout(ctx, peerCapacityTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Transport: p.peerHTTPTransport(peerUUID)}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("read peer capacity: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxPeerCapacityBody))
	if err != nil {
		return nil, fmt.Errorf("read peer capacity: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("read peer capacity: peer answered %d", resp.StatusCode)
	}
	var out struct {
		Models []modelCapacity `json:"models"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("read peer capacity: %w", err)
	}
	if out.Models == nil {
		out.Models = []modelCapacity{}
	}
	return out.Models, nil
}
