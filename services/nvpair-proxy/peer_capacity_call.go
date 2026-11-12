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
	"strings"
	"time"
)

// Another node's context capacity, planned and changed over mTLS.
//
// Any cluster member may manage any node, so a member may size, save, apply
// and clear another node's per-model capacity, the same way it may pause that
// node or change its engines. The plan is worked out on the node that runs
// the model: only it can read the model from its engine, and it keeps its own
// plans and enforces its own capacity. The caller supplies that node's GPUs
// and system memory, which every member already has from node-info.
//
// The desktop reaches only its own proxy, so the local proxy forwards the call
// (capacity/peer-call) to the target's ingress, which runs the same method it
// would run for its own desktop.
const (
	// peerCapacityCallPath is served on the mTLS ingress only, and only to POST.
	peerCapacityCallPath = "/pair/v1/peer/capacity/call"
	// peerCapacityCallTimeout covers planning (which reads the model from the
	// engine) and the round trip.
	peerCapacityCallTimeout = 20 * time.Second
	// maxPeerCapacityCallBody bounds a call's body.
	maxPeerCapacityCallBody = 64 << 10
)

// peerCapacityMethods are the capacity methods another node may call here.
var peerCapacityMethods = map[string]bool{
	"capacity/report":      true,
	"capacity/plan":        true,
	"capacity/planner":     true,
	"capacity/planner-set": true,
	"capacity/set":         true,
}

var errPeerCapacityUnsupported = errors.New("it runs a PAIR Plus without remote Context & sessions (1.3.0 or older); update it to change its settings from here")

type peerCapacityCallParams struct {
	NodeID string          `json:"nodeId"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params,omitempty"`
}

// callCapacityLocal runs one capacity method on this node, as its own
// desktop would.
func (p *Proxy) callCapacityLocal(ctx context.Context, method string, raw json.RawMessage) (any, error) {
	if len(raw) == 0 {
		raw = json.RawMessage(`{}`)
	}
	switch method {
	case "capacity/report":
		return map[string]any{"models": p.capacityReport(ctx)}, nil
	case "capacity/plan":
		res, err := p.planCapacity(ctx, raw)
		if err != nil {
			return nil, err
		}
		return res, nil
	case "capacity/planner":
		return p.plannerResult(), nil
	case "capacity/planner-set":
		res, err := p.setPlanner(ctx, raw)
		if err != nil {
			return nil, err
		}
		return res, nil
	case "capacity/set":
		row, err := p.setCapacity(ctx, raw)
		if err != nil {
			return nil, err
		}
		return map[string]any{"model": row}, nil
	}
	return nil, fmt.Errorf("unknown capacity method %q", method)
}

// callCapacity runs a capacity method on the node with this discovery ID:
// here when it is this node (or none is given), otherwise on that cluster
// peer's ingress.
func (p *Proxy) callCapacity(ctx context.Context, params peerCapacityCallParams) (json.RawMessage, error) {
	if !peerCapacityMethods[params.Method] {
		return nil, fmt.Errorf("method must be one of capacity/report, capacity/plan, capacity/planner, capacity/planner-set, capacity/set")
	}
	self := params.NodeID == ""
	var peers []peerTarget
	if !self {
		peers, self = p.clusterPeers(params.NodeID)
	}
	if self {
		v, err := p.callCapacityLocal(ctx, params.Method, params.Params)
		if err != nil {
			return nil, err
		}
		return json.Marshal(v)
	}
	if len(peers) == 0 {
		return nil, fmt.Errorf("node %s is not a reachable member of this node's cluster", params.NodeID)
	}
	t := peers[0]
	var out struct {
		Result json.RawMessage `json:"result"`
	}
	body := map[string]any{"method": params.Method, "params": params.Params}
	if len(params.Params) == 0 {
		body["params"] = json.RawMessage(`{}`)
	}
	if err := p.peerCall(ctx, t, http.MethodPost, peerCapacityCallPath, body, &out); err != nil {
		// An older node forwards the unknown path to its engine, which does
		// not know it either.
		if strings.HasSuffix(err.Error(), "answered 404") || strings.HasSuffix(err.Error(), "answered 405") {
			return nil, fmt.Errorf("node %s %w", t.nodeID, errPeerCapacityUnsupported)
		}
		return nil, err
	}
	if len(out.Result) == 0 {
		return nil, fmt.Errorf("node %s %w", t.nodeID, errPeerCapacityUnsupported)
	}
	return out.Result, nil
}

// answerPeerCapacityCall answers capacity/peer-call {"nodeId","method","params"}
// off the control loop, since a peer's answer waits on the network and on its
// engine.
func (p *Proxy) answerPeerCapacityCall(msg *Message) {
	var params peerCapacityCallParams
	if err := json.Unmarshal(msg.Params, &params); err != nil || params.Method == "" {
		p.codec.RespondError(msg.ID, -32602, `invalid params: expected {"nodeId"?,"method","params"?}`)
		return
	}
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.Error("recovered from a panic in a capacity call",
					"panic", fmt.Sprint(recovered), "stack", string(debug.Stack()))
				p.codec.RespondError(msg.ID, -32603, "internal error handling capacity/peer-call")
			}
		}()
		ctx, cancel := context.WithTimeout(p.lifetime(), peerCapacityCallTimeout+5*time.Second)
		defer cancel()
		res, err := p.callCapacity(ctx, params)
		if err != nil {
			slog.Debug("capacity call failed", "node_id", params.NodeID, "method", params.Method, "err", err)
			p.codec.RespondError(msg.ID, -32000, err.Error())
			return
		}
		if err := p.codec.Respond(msg.ID, res); err != nil {
			log.Printf("failed to respond to capacity/peer-call: %v", err)
		}
	}()
}

// servePeerCapacityCall runs a cluster member's capacity call on this node.
func (f *facade) servePeerCapacityCall(w http.ResponseWriter, r *http.Request, peer string) bool {
	if r.URL.Path != peerCapacityCallPath {
		return false
	}
	if r.Method != http.MethodPost {
		methodNotAllowed(w, "POST")
		return true
	}
	var body struct {
		Method string          `json:"method"`
		Params json.RawMessage `json:"params"`
	}
	raw, err := io.ReadAll(io.LimitReader(r.Body, maxPeerCapacityCallBody))
	if err == nil {
		err = json.Unmarshal(raw, &body)
	}
	if err != nil || !peerCapacityMethods[body.Method] {
		writeIngressError(w, http.StatusBadRequest, "peer-capacity-call",
			`expected {"method":"capacity/report"|"capacity/plan"|"capacity/planner"|"capacity/planner-set"|"capacity/set","params"}`)
		return true
	}
	if body.Method != "capacity/report" && body.Method != "capacity/planner" {
		slog.Info("cluster peer capacity change", "peer", peer, "method", body.Method)
	}
	ctx, cancel := context.WithTimeout(r.Context(), peerCapacityCallTimeout)
	defer cancel()
	res, err := f.host.callCapacityLocal(ctx, body.Method, body.Params)
	if err != nil {
		writeSessionJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return true
	}
	writeSessionJSON(w, http.StatusOK, map[string]any{"result": res})
	return true
}
