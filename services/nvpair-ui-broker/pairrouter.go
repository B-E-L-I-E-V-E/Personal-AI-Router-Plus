// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// pairrouter.go brings up the PAIR router in the proxy process: one loopback
// port that routes a request by its model to whichever engine, on whichever
// node, advertises it. The proxy owns the routing and the port's persistence;
// the broker only asks for it once the engine facades it routes to are up.

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	"nvpair-shared/engines"
	"nvpair-shared/errors"
)

// pairRouterPort is the router's default port. It must stay clear of every
// engine's facade and engine ports in nvpair-shared/engines.
const pairRouterPort = 11400

// pairRouterUnavailableID is the warning raised when the router cannot bind.
const pairRouterUnavailableID = engines.ProxyComponent + ":router-unavailable"

// methodRouterReady is the proxy's announcement of the router's port. It is
// process-scoped: the router belongs to no engine.
const methodRouterReady = "router/ready"

// logRouterReady records where the router is listening. The desktop asks the
// proxy directly (router/status) when it needs the port, so nothing is relayed.
func logRouterReady(params json.RawMessage) {
	var ready struct {
		Port int `json:"port"`
	}
	if err := json.Unmarshal(params, &ready); err != nil {
		slog.Warn("invalid router/ready", "err", err)
		return
	}
	slog.Info("PAIR router listening", "port", ready.Port)
}

// enableRouterRequest mirrors the proxy's router/enable params.
type enableRouterRequest struct {
	Port                int  `json:"port,omitempty"`
	IgnorePersistedPort bool `json:"ignorePersistedPort,omitempty"`
}

// routerStatusReply mirrors the proxy's router/enable result.
type routerStatusReply struct {
	Listening bool `json:"listening"`
	Port      int  `json:"port"`
}

// enablePAIRRouter asks the proxy for the router and reports where it landed.
// A failure is a warning, never fatal: every engine's own port keeps working
// without it. The proxy restores a port the user moved it to, so the default
// here is only a request.
func (b *Broker) enablePAIRRouter(ctx context.Context, pp *proxyProcess) (int, error) {
	params, err := json.Marshal(enableRouterRequest{Port: pairRouterPort})
	if err != nil {
		return 0, err
	}
	raw, rpcErr, err := pp.Call(ctx, "router/enable", params)
	if err == nil && rpcErr != nil {
		err = fmt.Errorf("router/enable rejected: %d %s", rpcErr.Code, rpcErr.Message)
	}
	var reply routerStatusReply
	if err == nil {
		err = json.Unmarshal(raw, &reply)
	}
	if err != nil {
		slog.Warn("PAIR router unavailable; engine ports are unaffected", "port", pairRouterPort, "err", err)
		b.forwardErrorsReport(errors.ServiceError{
			ID:        pairRouterUnavailableID,
			Message:   fmt.Sprintf("NVPAIR could not open the PAIR router on port %d: %v. Each engine's own port still works.", pairRouterPort, err),
			Timestamp: nowMillis(),
			NodeID:    b.nodeID,
			Severity:  "warning",
			Action:    "none",
		})
		return 0, err
	}
	b.forwardErrorsClear(pairRouterUnavailableID)
	slog.Info("PAIR router enabled", "port", reply.Port)
	return reply.Port, nil
}
