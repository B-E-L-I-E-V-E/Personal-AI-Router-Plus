// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
)

const engineIdentityProbeHeader = "X-NVPAIR-Engine-Identity-Probe"

// localBackend is the explicit loopback engine the cluster mTLS ingress
// forwards to. It is supplied by the broker over node/set-local-backend and is
// deliberately NOT sourced from the discovery overlay: a request that arrived
// over the LAN mTLS ingress can only ever be dumped on this node's own local
// engine, never re-routed to a peer, so the ingress path is strictly terminal
// and cannot recurse or amplify.
type localBackend struct {
	Engine  string `json:"engine"`
	Host    string `json:"host"`
	Port    int    `json:"port"`
	Healthy bool   `json:"healthy"`
}

// currentLocalBackend snapshots the configured local engine.
func (f *facade) currentLocalBackend() localBackend {
	f.backendMu.RLock()
	defer f.backendMu.RUnlock()
	return f.backend
}

// setLocalBackend records (or, with a zero port / unhealthy flag, effectively
// clears) the local engine this facade's ingress serves.
//
// A non-loopback host is rejected rather than stored. The ingress forwards a
// pin-authenticated peer's request straight here without consulting discovery,
// so an off-box host would turn this node into a relay to an address chosen by
// whoever can reach the control channel. The broker only ever sends 127.0.0.1;
// this is the same defence-in-depth re-validation setLoopbackAlias performs on
// the alias the broker sends it.
func (f *facade) setLocalBackend(b localBackend) error {
	if b.Host != "" && !isLoopbackHost(b.Host) {
		return fmt.Errorf("local backend host %q is not loopback", b.Host)
	}
	f.backendMu.Lock()
	defer f.backendMu.Unlock()
	f.backend = b
	return nil
}

// localBackendTarget returns the loopback URL of the current local engine, and
// false when none is set/healthy (the ingress then answers 503 rather than
// forwarding). The host defaults to 127.0.0.1, and setLocalBackend refuses to
// store anything that is not loopback, so this is always a loopback target.
func (f *facade) localBackendTarget() (*url.URL, bool) {
	b := f.currentLocalBackend()
	if b.Port <= 0 || !b.Healthy {
		return nil, false
	}
	host := b.Host
	if host == "" {
		host = "127.0.0.1"
	}
	return &url.URL{Scheme: "http", Host: net.JoinHostPort(host, strconv.Itoa(b.Port))}, true
}

// handlePlain is the plaintext personality: it accepts requests only from
// loopback and hands them to the full local router (handleHTTP). A non-loopback
// caller — any LAN peer — is refused; peers must use the mTLS ingress. This is
// what closes the former open-relay exposure (the listener still binds all
// interfaces for the TLS personality, but plaintext is loopback-only).
func (f *facade) handlePlain(w http.ResponseWriter, r *http.Request) {
	if !isLoopbackRemote(r.RemoteAddr) {
		slog.Warn("rejected non-loopback plaintext request; cluster peers must use mTLS",
			"remote", r.RemoteAddr, "method", r.Method, "path", r.URL.Path)
		writeIngressError(w, http.StatusForbidden, "loopback-only",
			"plaintext requests are accepted only from loopback; cluster peers must use the mTLS ingress")
		return
	}
	// Engine-manager marks its private identity/action requests so this
	// compatibility facade can never be mistaken for the local Ollama backend.
	if r.Header.Get(engineIdentityProbeHeader) == "1" {
		writeIngressError(w, http.StatusConflict, "proxy-facade",
			"the compatibility facade is not the "+f.profile.DisplayName+" engine")
		return
	}
	// PAIR's own session API is answered here and never forwarded. Being on
	// the plaintext personality makes it loopback-only, like inference.
	if f.router && f.serveRoutingAPI(w, r) {
		return
	}
	api := f.pairAPIFacade()
	if api.serveSessionsAPI(w, r) || api.serveCapacityAPI(w, r) || api.serveNodeStateAPI(w, r) || api.servePerformanceAPI(w, r) {
		return
	}
	if f.router {
		f.handleRouter(w, r)
		return
	}
	f.handleHTTP(w, r)
}

// handleClusterIngress is the LAN mTLS personality: it authenticates the caller
// against this node's cluster pins and, once the peer is a trusted cluster
// member, forwards the request straight to the local loopback engine — exactly
// like the local plaintext path, with no route filtering. The mTLS pin is the
// sole authorization boundary (a trusted peer is treated like a local client),
// so the two personalities stay behaviorally identical toward the engine. It
// never calls resolveCandidates, so a peer request cannot be re-routed onward.
func (f *facade) handleClusterIngress(w http.ResponseWriter, r *http.Request) {
	// The router is loopback-only: peers reach this node's engines through
	// their own ports, never through it.
	if f.router {
		writeIngressError(w, http.StatusNotFound, "no-cluster-ingress",
			"the PAIR router has no cluster ingress; use the engine's own port")
		return
	}
	f.host.activeRequests.Add(1)
	defer f.host.activeRequests.Add(-1)
	// Re-derive membership and pins per request so a cluster left, or a peer
	// paired or removed, after startup is reflected immediately without a proxy
	// restart — a removed peer must stop being accepted right away, which is the
	// whole point of the gate.
	f.host.mesh.Refresh()
	peer, ok := f.host.mesh.VerifyClientPin(r)
	if !ok {
		writeIngressError(w, http.StatusForbidden, "cluster-auth",
			"client certificate is not a pinned member of this node's cluster")
		return
	}
	// A calling node releasing a session's slot here (peer.go), or reading
	// this node's capacity (peer_capacity.go).
	if f.servePeerRelease(w, r, peer) || f.servePeerCapacity(w, r) {
		return
	}
	// A calling node sizing, saving, applying or clearing this node's
	// capacity: any cluster member may manage any node (peer_capacity_call.go).
	if f.servePeerCapacityCall(w, r, peer) {
		return
	}
	// A calling node asking this node to describe one of its models
	// (modeldetails.go).
	if f.servePeerModelDetails(w, r) {
		return
	}
	// Any cluster member may read or set this node's state, and each tells
	// the others when its own changes (nodestate.go).
	if f.servePeerNodeState(w, r, peer) {
		return
	}
	// A paused or stopped node takes nothing from other nodes. The 503 and
	// its X-PAIR-Node-State header tell the calling node to try another, and
	// to leave this one out until it runs again. A request taken is recorded
	// so a stop can cancel it, including while it waits for a slot below.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	endServing, err := f.host.node.begin(cancel)
	if err != nil {
		writeNodeRefusal(w, f.host.node.current())
		return
	}
	defer endServing()
	r = r.WithContext(ctx)
	target, ok := f.localBackendTarget()
	if !ok {
		writeIngressError(w, http.StatusServiceUnavailable, "no-local-backend",
			"no local inference backend is available on this node")
		return
	}
	// This node enforces its own context capacity for sessions other nodes
	// route here, as it does for its own (peer.go).
	r, record, done, status, err := f.admitPeer(r, peer)
	if err != nil {
		slog.Debug("cluster ingress refused by context admission",
			"peer", peer, "path", r.URL.Path, "status", status, "reason", err)
		if errors.Is(err, errNodePaused) || errors.Is(err, errNodeStopped) {
			writeNodeRefusal(w, f.host.node.current())
			return
		}
		writePeerRefusal(w, status, err)
		return
	}
	defer done()
	slog.Debug("cluster ingress forwarding to local backend",
		"peer", peer, "method", r.Method, "path", r.URL.Path, "target", target.Host)
	rp := f.newLocalReverseProxy(target)
	perf, endLive := f.measurePeerRequest(r, peer)
	defer endLive()
	if record != nil || perf != nil {
		rp.ModifyResponse = func(resp *http.Response) error {
			if resp.StatusCode >= 300 {
				perf = nil
			}
			resp.Body = newObservedBody(resp.Body, resp.Header.Get("Content-Type"), record, perf)
			return nil
		}
	}
	rp.ServeHTTP(w, r)
}

// newLocalReverseProxy streams a request to the local engine, preserving
// cancellation (the request context is the proxy's root context, so a client
// disconnect or shutdown tears down the upstream call and stops generation).
func (f *facade) newLocalReverseProxy(target *url.URL) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			req.Host = target.Host
		},
		Transport: f.host.plainHTTPTransport(),
		ErrorHandler: func(ew http.ResponseWriter, _ *http.Request, err error) {
			slog.Warn("cluster ingress upstream error", "target", target.Host, "err", err)
			writeIngressError(ew, http.StatusBadGateway, "backend-error", "local inference backend error")
		},
	}
}

// isLoopbackRemote reports whether an http.Request RemoteAddr (host:port) is a
// loopback address (127.0.0.0/8 or ::1). An unparseable/empty RemoteAddr is not
// loopback, so it fails closed.
func isLoopbackRemote(remoteAddr string) bool {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// writeIngressError returns the actual failure without granting browser permissions.
func writeIngressError(w http.ResponseWriter, status int, code, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	body, err := json.Marshal(map[string]string{"error": msg, "code": code})
	if err != nil {
		body = []byte(`{"error":"ingress error"}`)
	}
	_, _ = w.Write(body)
}
