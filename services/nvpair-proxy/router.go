// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// The PAIR router is one loopback port for every engine. A client names a
// model; the router finds every (node, engine) pair whose inventory advertises
// it, lets the scheduler's ordering and this process's reservations pick the
// least busy, and forwards the request unchanged to that engine's own
// OpenAI-compatible route — on this node through the engine's local backend,
// on a peer through the peer's mTLS ingress for that engine. Failover walks
// the same set, so it may cross from one engine to another.
//
// It is a facade so it shares the listener machinery (two-personality split,
// persisted port, live rebind) and the dispatch loop in handleHTTP. What makes
// it different is confined to this file and a handful of branches: it fronts
// no engine, so its candidates come from the engine facades rather than from a
// discovery overlay of its own, and everything engine-specific about an
// attempt is read from the candidate's own facade (candidate.fac).
//
// The decision is made here, on the node the request arrived at. A peer sees
// an ordinary request on its engine's ingress, so peers need no change.

import (
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"sort"
	"strconv"
	"sync"

	"nvpair-shared/cors"
	"nvpair-shared/engines"
)

// routerStandalonePort is where the router listens unless the broker asks for
// another port or the user moved it. It is none of the engines' facade or
// engine ports.
const routerStandalonePort = 11400

// routerPortFile persists a port chosen through router/set-port, like each
// engine facade's own file.
const routerPortFile = "router-proxy-port.json"

// routerModelNotFound is the router's answer when no engine on any node
// advertises the requested model. The wording matches the engine facades'.
const routerModelNotFound = "no available node advertises the requested model"

// routerRequestIDPrefix keeps the router's workload ids apart from the engine
// facades'. A workload's store key is (origin, engine, runId, id) and the
// router files each job under the engine that serves it, so a bare counter
// would collide with that engine's own "1", "2", ….
const routerRequestIDPrefix = "router-"

// routerProfile is the router's facade profile. Its routes classify only the
// inference paths it accepts; the model list is answered by serveRouterModels.
// It is not in profiles, so facade/enable cannot create one.
var routerProfile = engineProfile{
	Engine: engines.Engine{
		Name:        "router",
		DisplayName: "PAIR router",
		PortFile:    routerPortFile,
	},
	StandalonePort: routerStandalonePort,
	Routes:         slices.Concat(openAIInferenceRoutes, anthropicInferenceRoutes),
	ModelNaming:    exactID,
}

// routerParams is router/enable's request.
type routerParams struct {
	// Port is the port the broker wants. Zero means routerStandalonePort.
	Port int `json:"port,omitempty"`
	// IgnorePersistedPort uses Port even when a port was persisted.
	IgnorePersistedPort bool `json:"ignorePersistedPort,omitempty"`
}

// routerStatus answers router/enable, router/status and router/set-port.
type routerStatus struct {
	Listening bool `json:"listening"`
	// Port is the bound port, or zero while the router is not listening.
	Port int `json:"port"`
}

func (p *Proxy) routerFacade() *facade {
	p.facadeMu.Lock()
	defer p.facadeMu.Unlock()
	return p.router
}

// engineFacadesInOrder snapshots the enabled engine facades in the engine
// table's order, so the router's tie-breaks do not depend on map iteration.
func (p *Proxy) engineFacadesInOrder() []*facade {
	p.facadeMu.Lock()
	defer p.facadeMu.Unlock()
	out := make([]*facade, 0, len(p.facades))
	for _, profile := range profiles {
		if f := p.facades[profile.Name]; f != nil {
			out = append(out, f)
		}
	}
	return out
}

// enableRouter brings the router up and reports where it landed. Like
// enableFacade, enabling it twice is an idempotent success, and a bind failure
// is tagged so the broker can tell it from a caller error.
func (p *Proxy) enableRouter(params routerParams) (routerStatus, error) {
	if params.Port != 0 && (params.Port < 1 || params.Port > 65535) {
		return routerStatus{}, fmt.Errorf("port must be between 1 and 65535 (got %d)", params.Port)
	}
	p.facadeMu.Lock()
	defer p.facadeMu.Unlock()
	if p.router != nil {
		return routerStatus{Listening: true, Port: p.router.port}, nil
	}
	requested := params.Port
	if requested == 0 {
		requested = routerStandalonePort
	}
	persisted, hasPersisted := loadPersistedPort(routerProfile)
	port := chooseStartupPort(routerProfile, requested, params.IgnorePersistedPort, persisted, hasPersisted)
	if hasPersisted && port == persisted {
		slog.Info("restored persisted router port", "port", persisted)
	}

	f := newFacade(p, routerProfile, NewDiscovery(), port)
	f.router = true
	ln, err := f.listen()
	if err != nil {
		return routerStatus{}, fmt.Errorf("%w: port %d: %w", errFacadeBindFailed, port, err)
	}
	f.serveHTTP(p.serveCtx, ln)
	p.router = f
	if err := f.notify("ready", ReadyParams{Version: Version, Port: port}); err != nil {
		slog.Warn("failed to announce the router", "err", err)
	}
	return routerStatus{Listening: true, Port: port}, nil
}

// handleRouterMessage serves the router's JSON-RPC methods. They are
// process-scoped: the broker may send them addressed to any engine, and the
// address is ignored.
func (p *Proxy) handleRouterMessage(method string, msg *Message) {
	switch method {
	case "router/enable":
		var params routerParams
		if len(msg.Params) > 0 {
			if err := json.Unmarshal(msg.Params, &params); err != nil {
				p.codec.RespondError(msg.ID, -32602, "invalid params: expected {\"port\",\"ignorePersistedPort\"}")
				return
			}
		}
		result, err := p.enableRouter(params)
		if err != nil {
			code := -32000
			if stderrors.Is(err, errFacadeBindFailed) {
				code = codeFacadeBindFailed
			}
			slog.Warn("router/enable failed", "port", params.Port, "err", err)
			p.codec.RespondError(msg.ID, code, err.Error())
			return
		}
		slog.Info("router enabled", "port", result.Port)
		_ = p.codec.Respond(msg.ID, result)

	case "router/status":
		var result routerStatus
		if f := p.routerFacade(); f != nil {
			result.Listening = true
			result.Port, _ = f.selfAddresses()
		}
		_ = p.codec.Respond(msg.ID, result)

	case "router/set-port":
		f := p.routerFacade()
		if f == nil {
			p.codec.RespondError(msg.ID, -32002, "the PAIR router is not enabled on this proxy")
			return
		}
		var params struct {
			Port int `json:"port"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			p.codec.RespondError(msg.ID, -32602, "invalid params: expected {\"port\": <int>}")
			return
		}
		if params.Port < 1 || params.Port > 65535 {
			p.codec.RespondError(msg.ID, -32602, "port must be between 1 and 65535")
			return
		}
		if err := f.setPort(params.Port); err != nil {
			p.codec.RespondError(msg.ID, -32000, err.Error())
			return
		}
		_ = p.codec.Respond(msg.ID, routerStatus{Listening: true, Port: params.Port})
	}
}

// notifyRouter is facade.notify for the router. Its port announcement goes out
// unaddressed as router/ready, because it belongs to no engine. Request events
// are not sent from here at all: handleHTTP reports a routed request through
// the engine that served it, and one that reached no engine has no engine log
// to appear in.
func (f *facade) notifyRouter(method string, params any) error {
	if method == "ready" {
		return f.host.codec.Notify("router/ready", params)
	}
	return nil
}

// mintRequestID allocates this facade's next request id.
func (f *facade) mintRequestID() string {
	id := strconv.FormatUint(f.nextRequestID.Add(1), 10)
	if f.router {
		return routerRequestIDPrefix + id
	}
	return id
}

// pairAPIFacade is the facade that answers PAIR's own /pair/v1 API on this
// port. The router has no engine of its own, so it lends the request to the
// first enabled engine facade; those APIs are process-wide apart from
// capacity detection, which then reads that engine.
func (f *facade) pairAPIFacade() *facade {
	if !f.router {
		return f
	}
	if all := f.host.engineFacadesInOrder(); len(all) > 0 {
		return all[0]
	}
	return f
}

// leadEngine names the engine a request is expected to run on: the facade's
// own on an engine port, the first candidate's on the router.
func leadEngine(cands []candidate, f *facade) string {
	if f.router && len(cands) > 0 {
		return cands[0].owner(f).profile.Name
	}
	return f.profile.Name
}

// admissionEngine names the engine whose context capacity admitSession
// applies: the engine of the first local candidate, which is the one it gates.
func admissionEngine(cands []candidate, f *facade) string {
	if !f.router {
		return f.profile.Name
	}
	for _, c := range cands {
		if c.local {
			return c.owner(f).profile.Name
		}
	}
	return leadEngine(cands, f)
}

// preferSessionNode moves the node that served the session's last turn to the
// front. On the router that is the node and the engine together, so a turn
// returns to the KV cache that actually holds it.
func preferSessionNode(cands []candidate, sess sessionView, f *facade) {
	if !f.router {
		if node := sess.Nodes[f.profile.Name]; node != "" {
			preferNode(cands, node)
		}
		return
	}
	for i, c := range cands {
		if sess.Nodes[c.owner(f).profile.Name] != c.id {
			continue
		}
		if i > 0 {
			copy(cands[1:i+1], cands[:i])
			cands[0] = c
		}
		return
	}
}

// resolveRouterCandidates is the router's candidate set: every enabled engine
// that serves the path, each contributing the owners its own resolution
// admits — the same inventory match, pause/stop and self-forward rules as on
// its own port. held sums the capable nodes left out because they are paused
// or stopped.
//
// The union is ordered by the scheduler's priority list, then node ID, then
// the engine table's order. That last key decides only between two engines on
// the same machine; which machine goes first is reserveCandidate's choice, on
// pending work plus GPU pressure plus this process's reservations, exactly as
// on an engine port. A manual node pin is per engine and does not apply here.
func (p *Proxy) resolveRouterCandidates(path, model string) (out []candidate, held int) {
	for _, f := range p.engineFacadesInOrder() {
		if route, ok := f.profile.routeFor(http.MethodPost, path); !ok || route.Role != roleInferencePOST {
			continue
		}
		cands, h := f.resolveCandidatesHeld(model)
		out = append(out, cands...)
		held += h
	}
	rank := make(map[string]int)
	for i, id := range p.PriorityList() {
		if _, seen := rank[id]; !seen {
			rank[id] = i
		}
	}
	rankOf := func(id string) int {
		if r, ok := rank[id]; ok {
			return r
		}
		return len(rank)
	}
	sort.SliceStable(out, func(i, j int) bool {
		ri, rj := rankOf(out[i].id), rankOf(out[j].id)
		if ri != rj {
			return ri < rj
		}
		return out[i].id < out[j].id
	})
	slog.Debug("router resolved", "path", path, "candidates", len(out), "held", held)
	return out, held
}

// handleRouter serves the router's own surface. It is deliberately small: the
// combined model list, and the inference paths every engine shares. Anything
// else is an engine-specific endpoint and belongs on that engine's port.
func (f *facade) handleRouter(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/v1/models" && r.Method == http.MethodGet {
		f.serveRouterModels(w, r)
		return
	}
	if _, ok := f.profile.routeFor(http.MethodPost, r.URL.Path); ok &&
		(r.Method == http.MethodPost || r.Method == http.MethodOptions) {
		f.handleHTTP(w, r)
		return
	}
	writeIngressError(w, http.StatusNotFound, "not_found",
		"the PAIR router serves GET /v1/models and POST /v1/chat/completions, /v1/completions, /v1/embeddings and /v1/messages")
}

// routerNode is one place a model can run, in the router's model list.
type routerNode struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Engine string `json:"engine"`
}

// routerModelInfo is the "pair" member the router adds to each model record.
// Capabilities, Limits and Sources are the model's capability report
// (capabilities.go) without its placements, which Nodes already names.
type routerModelInfo struct {
	Engines      []string          `json:"engines"`
	Nodes        []routerNode      `json:"nodes"`
	Capabilities modelCapabilities `json:"capabilities"`
	Limits       modelLimits       `json:"limits"`
	Sources      map[string]string `json:"sources"`
}

// routerModel accumulates one model id across every engine and node.
type routerModel struct {
	raw  json.RawMessage
	info routerModelInfo
	// placements are where the model is listed, for its capability report.
	placements []placementRef
}

// routerListSource is one (engine, node) model list to fetch.
type routerListSource struct {
	fac  *facade
	cand candidate
	// node is the discovery record the candidate came from. Its inventory
	// decides which listed models the router keeps.
	node Node
}

// serveRouterModels answers GET /v1/models with every engine's inventory on
// every routable node, as one OpenAI list de-duplicated by model id. Each
// record is the first one seen for that id, in engine-table then node order,
// with a "pair" member naming every engine and node that offers it.
//
// It aggregates partially, like the engine facades' combined lists: an engine
// or node that does not answer is left out. Only when nothing answers is the
// inventory unavailable. CORS is not merged here: the router is for local
// programs, so a browser origin gets no CORS grant.
func (f *facade) serveRouterModels(w http.ResponseWriter, r *http.Request) {
	p := f.host
	requestID := resolveRequestID(r.Header)
	w.Header().Set(requestIDHeader, requestID)

	var sources []routerListSource
	for _, ef := range p.engineFacadesInOrder() {
		byID := make(map[string]Node)
		for _, n := range ef.discovery.Nodes() {
			byID[n.ID] = n
		}
		for _, c := range ef.resolveCandidates("") {
			sources = append(sources, routerListSource{fac: ef, cand: c, node: byID[c.id]})
		}
	}

	results := make([][]json.RawMessage, len(sources))
	ok := make([]bool, len(sources))
	var wg sync.WaitGroup
	for i, src := range sources {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i], ok[i] = f.fetchRouterModelList(r, src)
		}()
	}
	wg.Wait()

	byID := make(map[string]*routerModel)
	var order []string
	answered := false
	for i, src := range sources {
		if !ok[i] {
			continue
		}
		answered = true
		engine := src.fac.profile.Name
		for _, raw := range results[i] {
			var identity struct {
				ID string `json:"id"`
			}
			if err := json.Unmarshal(raw, &identity); err != nil || identity.ID == "" {
				continue
			}
			// List only what routing would send to this node. An engine's
			// OpenAI list can name models its inventory does not: LM Studio
			// lists DFlash draft models there, which it cannot serve alone and
			// answers with whatever model is loaded instead. A model the
			// router lists must be one it routes.
			if !nodeAdvertisesModel(src.fac.profile, src.node, identity.ID) {
				continue
			}
			m := byID[identity.ID]
			if m == nil {
				m = &routerModel{raw: raw}
				byID[identity.ID] = m
				order = append(order, identity.ID)
			}
			if !slices.Contains(m.info.Engines, engine) {
				m.info.Engines = append(m.info.Engines, engine)
			}
			node := routerNode{ID: src.cand.id, Name: src.node.Host, Engine: engine}
			if !slices.Contains(m.info.Nodes, node) {
				m.info.Nodes = append(m.info.Nodes, node)
				m.placements = append(m.placements, placementForNode(src.fac, src.cand, src.node, identity.ID))
			}
		}
	}
	if !answered {
		writeIngressError(w, http.StatusServiceUnavailable, "inventory_unavailable", "model inventory unavailable")
		return
	}

	// One lookup for every placement, so cold details are fetched together
	// and the list waits for them once.
	var refs []placementRef
	for _, id := range order {
		refs = append(refs, byID[id].placements...)
	}
	looked := p.capabilities.lookup(r.Context(), refs, modelListCapabilityWait)
	for _, id := range order {
		m := byID[id]
		n := len(m.placements)
		report := aggregateCapabilities(id, looked[:n])
		looked = looked[n:]
		m.info.Capabilities, m.info.Limits, m.info.Sources = report.Capabilities, report.Limits, report.Sources
	}

	data := make([]json.RawMessage, 0, len(order))
	for _, id := range order {
		m := byID[id]
		var record map[string]json.RawMessage
		if err := json.Unmarshal(m.raw, &record); err != nil {
			continue
		}
		info, err := json.Marshal(m.info)
		if err != nil {
			continue
		}
		record["pair"] = info
		if _, has := record["object"]; !has {
			record["object"] = json.RawMessage(`"model"`)
		}
		encoded, err := json.Marshal(record)
		if err != nil {
			continue
		}
		data = append(data, encoded)
	}
	body, err := json.Marshal(struct {
		Object string            `json:"object"`
		Data   []json.RawMessage `json:"data"`
	}{Object: "list", Data: data})
	if err != nil {
		writeIngressError(w, http.StatusInternalServerError, "encode_failed", "failed to encode model inventory")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

// fetchRouterModelList reads one engine's OpenAI model list from one node,
// through the engine's own route for /v1/models (llama.cpp's is /models).
func (f *facade) fetchRouterModelList(r *http.Request, src routerListSource) ([]json.RawMessage, bool) {
	p := f.host
	route, ok := src.fac.profile.routeFor(http.MethodGet, "/v1/models")
	if !ok {
		return nil, false
	}
	target := *src.cand.url
	target.Path = route.upstreamPath()
	target.RawPath = ""
	target.RawQuery = ""
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, target.String(), nil)
	if err != nil {
		return nil, false
	}
	req.Header = cors.FanoutHeaders(r.Header)
	req.Header.Del("Origin")
	req.Header.Del("Content-Length")
	req.Header.Set("Accept", "application/json")
	client := *modelListClient
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	if src.cand.peerUUID != "" {
		client.Transport = p.candidateTransport(src.cand)
	}
	resp, err := client.Do(req)
	if err != nil {
		src.fac.targets.Forget(src.cand.id)
		slog.Debug("router model list unavailable", "engine", src.fac.profile.Name, "node_id", src.cand.id, "err", err)
		return nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		slog.Debug("router model list refused", "engine", src.fac.profile.Name, "node_id", src.cand.id, "status", resp.StatusCode)
		return nil, false
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxModelListBytes+1))
	if err != nil || len(body) > maxModelListBytes {
		return nil, false
	}
	var envelope struct {
		Data *[]json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil || envelope.Data == nil {
		return nil, false
	}
	return *envelope.Data, true
}

// servedNodeHeader and servedEngineHeader name, on a router reply, the node
// and engine that produced it. They are set on every reply an engine produced,
// and on a transport failure after the last attempt, so a failed request can
// be traced as well as a successful one. A reply the router wrote itself, such
// as model_not_found, carries neither.
const (
	servedNodeHeader   = "X-PAIR-Node-ID"
	servedEngineHeader = "X-PAIR-Engine"
)

func setServedByHeaders(h http.Header, nodeID, engine string) {
	h.Set(servedNodeHeader, nodeID)
	h.Set(servedEngineHeader, engine)
}

// routingPath is the router's read-only view of how it is choosing nodes.
const routingPath = "/pair/v1/routing"

// routingNode is one node's place in the router's choice: the scheduler's
// rank, pending work and GPU pressure, the reservations this process holds on
// it, and their sum, which is what reserveCandidate compares.
type routingNode struct {
	ID           string `json:"id"`
	Rank         int    `json:"rank"`
	Pending      int    `json:"pending"`
	GPUPressure  int    `json:"gpuPressure"`
	Reservations int    `json:"reservations"`
	Load         int    `json:"load"`
}

// routingReport answers GET /pair/v1/routing.
type routingReport struct {
	// Generation is the scheduler snapshot the figures belong to. A request
	// dispatched under one generation is counted by the next.
	Generation uint64        `json:"generation"`
	Nodes      []routingNode `json:"nodes"`
}

// routingSnapshot reads the scheduler baseline and this process's reservations
// in priority order, under the same lock reserveCandidate takes.
func (p *Proxy) routingSnapshot() routingReport {
	p.priorityMu.RLock()
	defer p.priorityMu.RUnlock()
	report := routingReport{Generation: p.appliedPriorityGeneration, Nodes: make([]routingNode, 0, len(p.priority))}
	for rank, id := range p.priority {
		n := routingNode{
			ID:           id,
			Rank:         rank,
			Pending:      p.priorityPending[id],
			GPUPressure:  p.priorityGPUPressure[id],
			Reservations: p.priorityReservations[id],
		}
		n.Load = n.Pending + n.GPUPressure + n.Reservations
		report.Nodes = append(report.Nodes, n)
	}
	return report
}

// serveRoutingAPI answers GET /pair/v1/routing on the router, reporting
// whether it handled the request.
func (f *facade) serveRoutingAPI(w http.ResponseWriter, r *http.Request) bool {
	if r.URL.Path != routingPath {
		return false
	}
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeIngressError(w, http.StatusMethodNotAllowed, "method_not_allowed", "use GET")
		return true
	}
	body, err := json.Marshal(f.host.routingSnapshot())
	if err != nil {
		writeIngressError(w, http.StatusInternalServerError, "encode_failed", "failed to encode the routing report")
		return true
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
	return true
}
