// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net"
	"testing"

	"nvpair-shared/engines"
)

// serveRouterEnable answers router/enable on the proxy side of a pipe, binding
// the requested port unless it is in taken.
func serveRouterEnable(t *testing.T, conn net.Conn, taken map[int]bool, asked chan<- enableRouterRequest) {
	t.Helper()
	go func() {
		codec := NewCodec(conn)
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			if msg.Method != "router/enable" {
				continue
			}
			var req enableRouterRequest
			_ = json.Unmarshal(msg.Params, &req)
			asked <- req
			if taken[req.Port] {
				_ = codec.RespondError(msg.ID, codeFacadeBindFailed, "facade bind failed: address already in use")
				continue
			}
			_ = codec.Respond(msg.ID, routerStatusReply{Listening: true, Port: req.Port})
		}
	}()
}

func TestEnablePAIRRouter_AsksForTheDefaultPortAndReportsWhereItLanded(t *testing.T) {
	cases := []struct {
		name     string
		taken    map[int]bool
		wantPort int
		wantErr  bool
	}{
		{name: "port free", wantPort: pairRouterPort},
		{name: "port taken", taken: map[int]bool{pairRouterPort: true}, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			proxyClient, proxyServer := net.Pipe()
			t.Cleanup(func() {
				_ = proxyClient.Close()
				_ = proxyServer.Close()
			})
			proxy := &proxyProcess{peer: NewPeer(NewCodec(proxyClient))}
			go proxy.peer.Serve(nil, nil)
			asked := make(chan enableRouterRequest, 1)
			serveRouterEnable(t, proxyServer, tc.taken, asked)

			port, err := (&Broker{}).enablePAIRRouter(context.Background(), proxy)

			if req := <-asked; req.Port != pairRouterPort || req.IgnorePersistedPort {
				t.Errorf("asked for %+v, want port %d with the persisted port honoured", req, pairRouterPort)
			}
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, want error %v", err, tc.wantErr)
			}
			if port != tc.wantPort {
				t.Errorf("port = %d, want %d", port, tc.wantPort)
			}
		})
	}
}

func TestPAIRRouterPort_IsNoEnginesFacadeOrEnginePort(t *testing.T) {
	for _, e := range engines.All() {
		if pairRouterPort == e.FacadePort || pairRouterPort == e.EnginePortBase {
			t.Errorf("router port %d collides with %s's ports (facade %d, engine %d)",
				pairRouterPort, e.Name, e.FacadePort, e.EnginePortBase)
		}
	}
}
