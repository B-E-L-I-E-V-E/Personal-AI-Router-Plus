// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Model capabilities tell a client of the PAIR router what a model can do and
// how much it can take, so the client can size its requests instead of asking
// its user to type the limits in: whether the model chats, embeds, sees,
// calls tools or reasons before answering, the context window a request can
// rely on, and how many output tokens to ask for.
//
// The facts come from the engines, through the model-details readers
// (modeldetails.go): this node's engines directly, a peer's over its mTLS
// details route. Each placement — one engine on one node that the router
// would send the model to — is described separately, and the report combines
// them so that it holds wherever the request lands. Anything PAIR inferred
// rather than read from an engine is marked "estimated".
//
// Answers are cached per placement. Inference never waits on them: only the
// capability endpoint and the router's model list read the cache.

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"
)

const (
	// modelCapabilitiesPrefix is GET /pair/v1/models/{id} on the router.
	modelCapabilitiesPrefix = "/pair/v1/models/"
	// capabilityTTL keeps a placement's details this long, so a loaded or
	// unloaded model is noticed within it. A failed lookup is retried sooner.
	capabilityTTL       = 30 * time.Second
	capabilityFailedTTL = 5 * time.Second
	// capabilityForget drops entries nobody has asked about for this long.
	capabilityForget = 10 * time.Minute
	// capabilityWait bounds how long the capability endpoint waits for a cold
	// lookup, and modelListCapabilityWait the router's model list.
	capabilityWait          = modelDetailsTimeout
	modelListCapabilityWait = 2500 * time.Millisecond
	// estimatedLoadContext is assumed for a model that is not loaded and has
	// no configured window: the default LM Studio and Ollama load with.
	estimatedLoadContext = 4096
)

const (
	sourceEngine    = "engine"
	sourceEstimated = "estimated"
)

// modelCapabilities are what a model can do. Every placement must have a
// capability for it to be reported, except reasoning, which any placement
// having is enough: a request must budget for thinking wherever it lands.
type modelCapabilities struct {
	Chat             bool `json:"chat"`
	StructuredOutput bool `json:"structuredOutput"`
	Tools            bool `json:"tools"`
	Vision           bool `json:"vision"`
	Embedding        bool `json:"embedding"`
	Reasoning        bool `json:"reasoning"`
}

// modelLimits are the sizes a request should keep to, in tokens.
type modelLimits struct {
	// ContextWindow is what one request can rely on: the smallest window
	// across the placements.
	ContextWindow int64 `json:"contextWindow"`
	// MaxContextWindow is the largest window the model supports.
	MaxContextWindow int64 `json:"maxContextWindow"`
	// MaxOutputTokens is an engine's own cap on a reply; nil when none sets
	// one, and the reply is then bounded by the window less the prompt.
	MaxOutputTokens *int64 `json:"maxOutputTokens"`
	// RecommendedOutputTokens is enough for an answer, and for a reasoning
	// model its thinking too.
	RecommendedOutputTokens int64 `json:"recommendedOutputTokens"`
}

type capabilityNode struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// capabilityPlacement is one engine on one node the router would use.
type capabilityPlacement struct {
	Node             capabilityNode `json:"node"`
	Engine           string         `json:"engine"`
	Loaded           bool           `json:"loaded"`
	ContextWindow    int64          `json:"contextWindow"`
	MaxContextWindow int64          `json:"maxContextWindow"`
	// DetailsAvailable is false when the placement could not be asked, so its
	// values are estimates.
	DetailsAvailable bool `json:"detailsAvailable"`
}

// modelCapabilitiesReport answers GET /pair/v1/models/{id}.
type modelCapabilitiesReport struct {
	ID           string                `json:"id"`
	Capabilities modelCapabilities     `json:"capabilities"`
	Limits       modelLimits           `json:"limits"`
	Placements   []capabilityPlacement `json:"placements"`
	// Sources says, for each capability and limit, whether every engine
	// involved stated it ("engine") or PAIR inferred it ("estimated").
	Sources map[string]string `json:"sources"`
}

// fact is one value about one placement and whether its engine stated it.
type fact[T any] struct {
	value  T
	engine bool
}

// placementFacts is what one placement's details say, with the gaps filled.
type placementFacts struct {
	chat, structured, tools, vision, embedding, reasoning fact[bool]
	loaded                                                bool
	context, maxContext                                   fact[int64]
	// maxOutput is zero when the engine sets no cap.
	maxOutput int64
}

// reasoningFamilies match model names and architectures known to think
// before answering. They are only consulted when the engine does not say.
var reasoningFamilies = regexp.MustCompile(`(?i)(deepseek-r1|deepseek-v3\.[12]|\br1\b|-r1[-:]|qwq|qwen3|glm-4\.[5-9]|glm-z1|gpt-oss|magistral|phi-4-(mini-)?reasoning|thinking|reasoning|\bthink\b|kimi-k2|minimax-m[12]|seed-oss|exaone-deep|nemotron.*(nano|super|ultra))`)

// embeddingArchitectures are architectures that only embed.
var embeddingArchitectures = []string{"bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "xlm-roberta", "gte", "t5encoder"}

func hasCapability(d modelDetails, names ...string) bool {
	for _, c := range d.Capabilities {
		for _, n := range names {
			if strings.EqualFold(c, n) {
				return true
			}
		}
	}
	return false
}

// derivePlacement turns one placement's details into facts. ok is false when
// the placement could not be asked, and d is then empty.
func derivePlacement(engine, model string, d modelDetails, ok bool) placementFacts {
	var pf placementFacts
	complete := ok && d.CapabilitiesComplete
	name := model + " " + d.Name + " " + d.Architecture

	switch {
	case ok && engine == "lmstudio" && d.Type != "":
		pf.embedding = fact[bool]{strings.HasPrefix(strings.ToLower(d.Type), "embedding"), true}
	case complete && engine == "ollama":
		pf.embedding = fact[bool]{hasCapability(d, "embedding") && !hasCapability(d, "completion"), true}
	default:
		pf.embedding = fact[bool]{strings.Contains(strings.ToLower(name), "embed") ||
			slices.Contains(embeddingArchitectures, strings.ToLower(d.Architecture)), false}
	}
	pf.chat = fact[bool]{!pf.embedding.value, pf.embedding.engine}
	// Every engine PAIR fronts constrains a chat model's output to a JSON
	// schema, but none says so per model.
	pf.structured = fact[bool]{pf.chat.value, false}

	switch {
	case hasCapability(d, "tools"):
		pf.tools = fact[bool]{true, true}
	case complete:
		pf.tools = fact[bool]{false, true}
	}
	switch {
	case hasCapability(d, "vision"):
		pf.vision = fact[bool]{true, true}
	case complete || (ok && engine == "llamacpp" && d.LoadedContext > 0):
		// llama.cpp lists every modality of a loaded model.
		pf.vision = fact[bool]{false, true}
	}
	switch {
	case hasCapability(d, "thinking", "reasoning"):
		pf.reasoning = fact[bool]{true, true}
	case d.TemplateThinks || reasoningFamilies.MatchString(name):
		// Ollama models pulled before it reported thinking say nothing, and
		// a reasoning model that is wrongly budgeted answers with nothing at
		// all, so the evidence wins over an engine's silence.
		pf.reasoning = fact[bool]{true, false}
	case complete:
		pf.reasoning = fact[bool]{false, true}
	}

	if d.ContextLength > 0 {
		pf.maxContext = fact[int64]{d.ContextLength, true}
	}
	switch {
	case d.LoadedContext > 0:
		pf.loaded = true
		pf.context = fact[int64]{d.LoadedContext, true}
	case d.ConfiguredContext > 0:
		pf.context = fact[int64]{d.ConfiguredContext, true}
	default:
		pf.context = fact[int64]{estimatedLoadContext, false}
		if pf.maxContext.value > 0 && pf.maxContext.value < estimatedLoadContext {
			pf.context.value = pf.maxContext.value
		}
	}
	if pf.maxContext.value == 0 {
		pf.maxContext = fact[int64]{pf.context.value, false}
	}
	pf.maxOutput = d.MaxOutputTokens
	return pf
}

func sourceOf(engine bool) string {
	if engine {
		return sourceEngine
	}
	return sourceEstimated
}

// recommendedOutput is how many tokens to ask for: room for an answer, and a
// reasoning model's thinking before it.
func recommendedOutput(window int64, reasoning bool, maxOutput *int64) int64 {
	clamp := func(v, lo, hi int64) int64 { return max(lo, min(v, hi)) }
	var n int64
	switch {
	case window <= 0 && reasoning:
		n = 8192
	case window <= 0:
		n = 2048
	case reasoning:
		n = clamp(window/4, 4096, 32768)
	default:
		n = clamp(window/8, 1024, 8192)
	}
	if window > 0 {
		n = min(n, window/2)
	}
	if maxOutput != nil {
		n = min(n, *maxOutput)
	}
	return n
}

// placementDetails is one placement and what was learned about it.
type placementDetails struct {
	ref     placementRef
	details modelDetails
	ok      bool
}

// aggregateCapabilities combines the placements into one report. There is at
// least one placement.
func aggregateCapabilities(id string, placements []placementDetails) modelCapabilitiesReport {
	report := modelCapabilitiesReport{ID: id, Placements: make([]capabilityPlacement, 0, len(placements))}
	type boolAcc struct {
		value  bool
		engine bool
	}
	all := func(get func(placementFacts) fact[bool]) func([]placementFacts) boolAcc {
		return func(fs []placementFacts) boolAcc {
			acc := boolAcc{value: true, engine: true}
			for _, f := range fs {
				v := get(f)
				acc.value = acc.value && v.value
				acc.engine = acc.engine && v.engine
			}
			return acc
		}
	}
	facts := make([]placementFacts, 0, len(placements))
	var reportedMax int64
	for _, pd := range placements {
		pf := derivePlacement(pd.ref.engine(), pd.ref.model, pd.details, pd.ok)
		if pf.maxContext.engine {
			reportedMax = max(reportedMax, pf.maxContext.value)
		}
		facts = append(facts, pf)
	}
	// It is one model, so an estimate never exceeds the largest context an
	// engine reported for it on another placement.
	for i := range facts {
		f := &facts[i]
		if reportedMax > 0 && !f.maxContext.engine {
			f.maxContext.value = reportedMax
		}
		if reportedMax > 0 && !f.context.engine {
			f.context.value = min(f.context.value, reportedMax)
		}
	}
	for i, pd := range placements {
		pf := facts[i]
		report.Placements = append(report.Placements, capabilityPlacement{
			Node:             capabilityNode{ID: pd.ref.cand.id, Name: pd.ref.nodeName},
			Engine:           pd.ref.engine(),
			Loaded:           pf.loaded,
			ContextWindow:    pf.context.value,
			MaxContextWindow: pf.maxContext.value,
			DetailsAvailable: pd.ok,
		})
	}
	sources := make(map[string]string)
	setBool := func(key string, dst *bool, acc boolAcc) {
		*dst = acc.value
		sources[key] = sourceOf(acc.engine)
	}
	caps := &report.Capabilities
	setBool("chat", &caps.Chat, all(func(f placementFacts) fact[bool] { return f.chat })(facts))
	setBool("structuredOutput", &caps.StructuredOutput, all(func(f placementFacts) fact[bool] { return f.structured })(facts))
	setBool("tools", &caps.Tools, all(func(f placementFacts) fact[bool] { return f.tools })(facts))
	setBool("vision", &caps.Vision, all(func(f placementFacts) fact[bool] { return f.vision })(facts))
	setBool("embedding", &caps.Embedding, all(func(f placementFacts) fact[bool] { return f.embedding })(facts))
	// Reasoning is true if any placement reasons. It is the engine's word
	// when every placement's engine spoke, or when an engine said yes.
	reasoning := boolAcc{engine: true}
	for _, f := range facts {
		reasoning.value = reasoning.value || f.reasoning.value
		reasoning.engine = reasoning.engine && f.reasoning.engine
	}
	for _, f := range facts {
		if f.reasoning.value && f.reasoning.engine {
			reasoning.engine = true
		}
	}
	setBool("reasoning", &caps.Reasoning, reasoning)

	limits := &report.Limits
	ctxEngine, maxEngine := true, false
	for i, f := range facts {
		if i == 0 || f.context.value < limits.ContextWindow {
			limits.ContextWindow = f.context.value
		}
		ctxEngine = ctxEngine && f.context.engine
		if f.maxContext.value > limits.MaxContextWindow || (f.maxContext.value == limits.MaxContextWindow && f.maxContext.engine) {
			limits.MaxContextWindow, maxEngine = f.maxContext.value, f.maxContext.engine
		}
		if f.maxOutput > 0 && (limits.MaxOutputTokens == nil || f.maxOutput < *limits.MaxOutputTokens) {
			v := f.maxOutput
			limits.MaxOutputTokens = &v
		}
	}
	if limits.MaxContextWindow < limits.ContextWindow {
		limits.MaxContextWindow, maxEngine = limits.ContextWindow, ctxEngine
	}
	sources["contextWindow"] = sourceOf(ctxEngine)
	sources["maxContextWindow"] = sourceOf(maxEngine)
	sources["maxOutputTokens"] = sourceOf(limits.MaxOutputTokens != nil)
	limits.RecommendedOutputTokens = recommendedOutput(limits.ContextWindow, caps.Reasoning, limits.MaxOutputTokens)
	sources["recommendedOutputTokens"] = sourceEstimated
	report.Sources = sources
	return report
}

// placementRef is one (engine, node) the router would send a model to, and
// the model's name as that node advertises it.
type placementRef struct {
	cand     candidate
	nodeName string
	model    string
	// inventory fingerprints the node's advertised models, so a change to
	// what the node offers makes its cached details stale.
	inventory string
}

func (r placementRef) engine() string {
	if r.cand.fac == nil {
		return ""
	}
	return r.cand.fac.profile.Name
}

type capabilityKey struct {
	engine, node, model, inventory string
}

func (r placementRef) key() capabilityKey {
	return capabilityKey{engine: r.engine(), node: r.cand.id, model: r.model, inventory: r.inventory}
}

// capabilityEntry is one placement's cached details. done is closed when the
// first lookup finishes; a refresh keeps the previous answer until it lands.
type capabilityEntry struct {
	// answered is set once the first lookup finished, and expires is when
	// the answer is next refreshed.
	answered   bool
	expires    time.Time
	used       time.Time
	details    modelDetails
	ok         bool
	done       chan struct{}
	refreshing bool
}

// capabilityCache caches placements' details. fetch asks a placement; tests
// replace it.
type capabilityCache struct {
	mu      sync.Mutex
	entries map[capabilityKey]*capabilityEntry
	now     func() time.Time
	fetch   func(ctx context.Context, ref placementRef) (modelDetails, error)
}

func newCapabilityCache(fetch func(ctx context.Context, ref placementRef) (modelDetails, error)) *capabilityCache {
	return &capabilityCache{entries: make(map[capabilityKey]*capabilityEntry), now: time.Now, fetch: fetch}
}

// lookup returns each placement's details, waiting at most wait for the ones
// never asked before. A stale answer is returned at once and refreshed in the
// background; a placement that has not answered is returned with ok false.
func (c *capabilityCache) lookup(ctx context.Context, refs []placementRef, wait time.Duration) []placementDetails {
	pending := make([]chan struct{}, 0, len(refs))
	entries := make([]*capabilityEntry, len(refs))
	c.mu.Lock()
	now := c.now()
	for k, e := range c.entries {
		if now.Sub(e.used) > capabilityForget && !e.refreshing {
			delete(c.entries, k)
		}
	}
	for i, ref := range refs {
		k := ref.key()
		e := c.entries[k]
		if e == nil {
			e = &capabilityEntry{done: make(chan struct{}), refreshing: true}
			c.entries[k] = e
			go c.refresh(k, e, ref)
		} else if !e.refreshing && !now.Before(e.expires) {
			e.refreshing = true
			go c.refresh(k, e, ref)
		}
		e.used = now
		entries[i] = e
		if !e.answered {
			pending = append(pending, e.done)
		}
	}
	c.mu.Unlock()

	if len(pending) > 0 {
		timer := time.NewTimer(wait)
		defer timer.Stop()
	waiting:
		for _, done := range pending {
			select {
			case <-done:
			case <-timer.C:
				break waiting
			case <-ctx.Done():
				break waiting
			}
		}
	}

	out := make([]placementDetails, len(refs))
	c.mu.Lock()
	defer c.mu.Unlock()
	for i, e := range entries {
		out[i] = placementDetails{ref: refs[i], details: e.details, ok: e.ok}
	}
	return out
}

// refresh asks one placement and stores the answer. It runs detached from
// the request that started it, so a client that gives up still leaves the
// answer cached for the next.
func (c *capabilityCache) refresh(k capabilityKey, e *capabilityEntry, ref placementRef) {
	ctx, cancel := context.WithTimeout(context.Background(), modelDetailsTimeout)
	defer cancel()
	d, err := c.fetch(ctx, ref)
	if err != nil {
		slog.Debug("model capabilities unavailable", "engine", k.engine, "node_id", k.node, "model", k.model, "err", err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	// A failed refresh keeps a previous good answer, and is retried sooner
	// than a good answer is refreshed.
	if err == nil {
		e.details, e.ok = d, true
		e.expires = c.now().Add(capabilityTTL)
	} else {
		e.expires = c.now().Add(capabilityFailedTTL)
	}
	e.refreshing = false
	if !e.answered {
		e.answered = true
		close(e.done)
	}
}

// remoteSpecReader reads an engine on another machine, a manual node's,
// whose model files are not on this disk.
var remoteSpecReader = &specClient{client: specReader.client, homeDir: specReader.homeDir, noFiles: true}

// describePlacement asks a placement's engine about the model: this node's
// engine directly, a cluster peer through its proxy's details route over
// mTLS, a manual node's engine at the address the user gave.
func (p *Proxy) describePlacement(ctx context.Context, ref placementRef) (modelDetails, error) {
	engine := ref.engine()
	switch {
	case ref.cand.local:
		return specReader.details(ctx, engine, ref.cand.url, ref.model)
	case ref.cand.peerUUID != "":
		client := &http.Client{Transport: p.candidateTransport(ref.cand), Timeout: peerDetailsTimeout}
		return fetchPeerModelDetails(ctx, client, ref.cand.url, ref.model)
	default:
		return remoteSpecReader.details(ctx, engine, ref.cand.url, ref.model)
	}
}

// inventoryFingerprint identifies a node's advertised inventory.
func inventoryFingerprint(models []string) string {
	sorted := slices.Clone(models)
	slices.Sort(sorted)
	return strings.Join(sorted, "\x00")
}

// modelPlacements lists where the router would send a model: every enabled
// engine's candidates for it, as routing resolves them, with the name each
// node advertises it under.
func (p *Proxy) modelPlacements(model string) []placementRef {
	var out []placementRef
	for _, ef := range p.engineFacadesInOrder() {
		byID := make(map[string]Node)
		for _, n := range ef.discovery.Nodes() {
			byID[n.ID] = n
		}
		for _, c := range ef.resolveCandidates(model) {
			out = append(out, placementForNode(ef, c, byID[c.id], model))
		}
	}
	return out
}

// placementForNode is a candidate as a placement of model, named as node
// advertises it.
func placementForNode(ef *facade, c candidate, node Node, model string) placementRef {
	advertised := model
	want := ef.profile.normalizeModel(model)
	for _, m := range node.Models {
		if ef.profile.normalizeModel(m) == want {
			advertised = m
			break
		}
	}
	return placementRef{cand: c, nodeName: node.Host, model: advertised, inventory: inventoryFingerprint(node.Models)}
}

// serveModelCapabilities answers GET /pair/v1/models/{id} on the router,
// reporting whether it handled the request.
func (f *facade) serveModelCapabilities(w http.ResponseWriter, r *http.Request) bool {
	if !strings.HasPrefix(r.URL.Path, modelCapabilitiesPrefix) {
		return false
	}
	w.Header().Set(requestIDHeader, resolveRequestID(r.Header))
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeIngressError(w, http.StatusMethodNotAllowed, "method_not_allowed", "use GET")
		return true
	}
	id := strings.TrimPrefix(r.URL.EscapedPath(), modelCapabilitiesPrefix)
	if unescaped, err := url.PathUnescape(id); err == nil {
		id = unescaped
	}
	if strings.TrimSpace(id) == "" {
		writeIngressError(w, http.StatusBadRequest, "model_required", "expected GET "+modelCapabilitiesPrefix+"{id}")
		return true
	}
	p := f.host
	refs := p.modelPlacements(id)
	if len(refs) == 0 {
		writeIngressError(w, http.StatusNotFound, "model_not_found", routerModelNotFound)
		return true
	}
	report := aggregateCapabilities(id, p.capabilities.lookup(r.Context(), refs, capabilityWait))
	body, err := json.Marshal(report)
	if err != nil {
		writeIngressError(w, http.StatusInternalServerError, "encode_failed", "failed to encode the model's capabilities")
		return true
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
	return true
}
