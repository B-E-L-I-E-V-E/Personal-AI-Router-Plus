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
	sourceEngine     = "engine"
	sourceConfigured = "configured"
	sourceEstimated  = "estimated"
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
	// model its thinking too: the room for the reply the user set in PAIR,
	// when one is set.
	RecommendedOutputTokens int64 `json:"recommendedOutputTokens"`
	// MaxInputTokens is the most a request's messages and history may hold:
	// the window less the reply budget. PAIR refuses a session conversation
	// that outgrows the part its setting gives it.
	MaxInputTokens int64 `json:"maxInputTokens"`
}

// capabilityConfigured is a node's PAIR capacity setting for the model, as
// the planner saved it: each session's window, the room in it kept for the
// reply, what is left for messages, and how many sessions run at once.
type capabilityConfigured struct {
	ContextWindow  int64 `json:"contextWindow"`
	ReplyTokens    int64 `json:"replyTokens"`
	MaxInputTokens int64 `json:"maxInputTokens"`
	Sessions       int   `json:"sessions"`
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
	// Configured is the node's PAIR capacity setting for the model, or nil.
	Configured *capabilityConfigured `json:"configured"`
}

// modelCapabilitiesReport answers GET /pair/v1/models/{id}.
type modelCapabilitiesReport struct {
	ID           string                `json:"id"`
	Capabilities modelCapabilities     `json:"capabilities"`
	Limits       modelLimits           `json:"limits"`
	Placements   []capabilityPlacement `json:"placements"`
	// Sources says, for each capability and limit, whether every engine
	// involved stated it ("engine"), the user's PAIR setting decided it
	// ("configured"), or PAIR inferred it ("estimated").
	Sources map[string]string `json:"sources"`
}

// fact is one value about one placement and whether its engine stated it.
// configured marks a value the user's PAIR setting decided instead.
type fact[T any] struct {
	value      T
	engine     bool
	configured bool
}

func (f fact[T]) source() string {
	switch {
	case f.engine:
		return sourceEngine
	case f.configured:
		return sourceConfigured
	}
	return sourceEstimated
}

// combinedSource is the source of a value several facts decided: estimated
// if any was, else configured if any was, else the engines'.
func combinedSource(sources ...string) string {
	out := sourceEngine
	for _, s := range sources {
		switch s {
		case sourceEstimated:
			return sourceEstimated
		case sourceConfigured:
			out = sourceConfigured
		}
	}
	return out
}

// placementFacts is what one placement's details say, with the gaps filled.
type placementFacts struct {
	chat, structured, tools, vision, embedding, reasoning fact[bool]
	loaded                                                bool
	context, maxContext                                   fact[int64]
	// maxOutput is zero when the engine sets no cap, and reply zero when
	// the user's PAIR setting keeps no room for the reply.
	maxOutput int64
	reply     int64
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

// derivePlacement turns one placement's details and its node's PAIR setting
// into facts. ok is false when the placement could not be asked, and d is
// then empty; set is nil when the node has no setting for the model.
func derivePlacement(engine, model string, d modelDetails, ok bool, set *capabilityConfigured) placementFacts {
	var pf placementFacts
	complete := ok && d.CapabilitiesComplete
	name := model + " " + d.Name + " " + d.Architecture

	switch {
	case ok && engine == "lmstudio" && d.Type != "":
		pf.embedding = fact[bool]{value: strings.HasPrefix(strings.ToLower(d.Type), "embedding"), engine: true}
	case complete && engine == "ollama":
		pf.embedding = fact[bool]{value: hasCapability(d, "embedding") && !hasCapability(d, "completion"), engine: true}
	default:
		pf.embedding = fact[bool]{value: strings.Contains(strings.ToLower(name), "embed") ||
			slices.Contains(embeddingArchitectures, strings.ToLower(d.Architecture))}
	}
	pf.chat = fact[bool]{value: !pf.embedding.value, engine: pf.embedding.engine}
	// Every engine PAIR fronts constrains a chat model's output to a JSON
	// schema, but none says so per model.
	pf.structured = fact[bool]{value: pf.chat.value, engine: false}

	switch {
	case hasCapability(d, "tools"):
		pf.tools = fact[bool]{value: true, engine: true}
	case complete:
		pf.tools = fact[bool]{value: false, engine: true}
	}
	switch {
	case hasCapability(d, "vision"):
		pf.vision = fact[bool]{value: true, engine: true}
	case complete || (ok && engine == "llamacpp" && d.LoadedContext > 0):
		// llama.cpp lists every modality of a loaded model.
		pf.vision = fact[bool]{value: false, engine: true}
	}
	switch {
	case hasCapability(d, "thinking", "reasoning"):
		pf.reasoning = fact[bool]{value: true, engine: true}
	case d.TemplateThinks || reasoningFamilies.MatchString(name):
		// Ollama models pulled before it reported thinking say nothing, and
		// a reasoning model that is wrongly budgeted answers with nothing at
		// all, so the evidence wins over an engine's silence.
		pf.reasoning = fact[bool]{value: true, engine: false}
	case complete:
		pf.reasoning = fact[bool]{value: false, engine: true}
	}

	if d.ContextLength > 0 {
		pf.maxContext = fact[int64]{value: d.ContextLength, engine: true}
	}
	switch {
	case d.LoadedContext > 0:
		pf.loaded = true
		pf.context = fact[int64]{value: d.LoadedContext, engine: true}
	case d.ConfiguredContext > 0:
		pf.context = fact[int64]{value: d.ConfiguredContext, engine: true}
	default:
		pf.context = fact[int64]{value: estimatedLoadContext, engine: false}
		if pf.maxContext.value > 0 && pf.maxContext.value < estimatedLoadContext {
			pf.context.value = pf.maxContext.value
		}
	}
	// The user's window per session: the engine truncates at what it loaded
	// and PAIR refuses beyond the setting, so the smaller holds. A setting
	// for a model that is not loaded is the window it will be loaded with.
	if set != nil && set.ContextWindow > 0 && (!pf.loaded || set.ContextWindow < pf.context.value) {
		pf.context = fact[int64]{value: set.ContextWindow, configured: true}
	}
	if pf.maxContext.value == 0 {
		pf.maxContext = fact[int64]{value: pf.context.value}
	}
	pf.maxOutput = d.MaxOutputTokens
	if set != nil {
		pf.reply = set.ReplyTokens
	}
	return pf
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

// placementDetails is one placement and what was learned about it: the
// engine's details, and the node's PAIR capacity setting for the model.
type placementDetails struct {
	ref     placementRef
	details modelDetails
	ok      bool
	set     *capabilityConfigured
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
		pf := derivePlacement(pd.ref.engine(), pd.ref.model, pd.details, pd.ok, pd.set)
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
		if reportedMax > 0 && !f.context.engine && !f.context.configured {
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
			Configured:       pd.set,
		})
	}
	sources := make(map[string]string)
	setBool := func(key string, dst *bool, acc boolAcc) {
		*dst = acc.value
		sources[key] = fact[bool]{engine: acc.engine}.source()
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
	ctxSources := make([]string, 0, len(facts))
	maxSource := sourceEstimated
	var reply int64
	for i, f := range facts {
		if i == 0 || f.context.value < limits.ContextWindow {
			limits.ContextWindow = f.context.value
		}
		ctxSources = append(ctxSources, f.context.source())
		if f.maxContext.value > limits.MaxContextWindow || (f.maxContext.value == limits.MaxContextWindow && f.maxContext.engine) {
			limits.MaxContextWindow, maxSource = f.maxContext.value, f.maxContext.source()
		}
		if f.maxOutput > 0 && (limits.MaxOutputTokens == nil || f.maxOutput < *limits.MaxOutputTokens) {
			v := f.maxOutput
			limits.MaxOutputTokens = &v
		}
		if f.reply > 0 && (reply == 0 || f.reply < reply) {
			reply = f.reply
		}
	}
	ctxSource := combinedSource(ctxSources...)
	if limits.MaxContextWindow < limits.ContextWindow {
		limits.MaxContextWindow, maxSource = limits.ContextWindow, ctxSource
	}
	sources["contextWindow"] = ctxSource
	sources["maxContextWindow"] = maxSource
	sources["maxOutputTokens"] = fact[bool]{engine: limits.MaxOutputTokens != nil}.source()
	// The room for the reply the user set wins over PAIR's estimate, as long
	// as it leaves something for the messages.
	if reply > 0 && reply < limits.ContextWindow {
		limits.RecommendedOutputTokens = reply
		sources["recommendedOutputTokens"] = sourceConfigured
		sources["maxInputTokens"] = sourceConfigured
	} else {
		limits.RecommendedOutputTokens = recommendedOutput(limits.ContextWindow, caps.Reasoning, limits.MaxOutputTokens)
		sources["recommendedOutputTokens"] = sourceEstimated
		sources["maxInputTokens"] = sourceEstimated
	}
	limits.MaxInputTokens = limits.ContextWindow - limits.RecommendedOutputTokens
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
	info       placementInfo
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
	fetch   func(ctx context.Context, ref placementRef) (placementInfo, error)
}

// placementInfo is what is cached for a placement: its engine's details and
// its node's PAIR capacity setting for the model (nil when there is none, or
// the node could not say).
type placementInfo struct {
	details modelDetails
	set     *capabilityConfigured
}

func newCapabilityCache(fetch func(ctx context.Context, ref placementRef) (placementInfo, error)) *capabilityCache {
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
		out[i] = placementDetails{ref: refs[i], details: e.info.details, ok: e.ok, set: e.info.set}
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
		e.info, e.ok = d, true
		e.expires = c.now().Add(capabilityTTL)
	} else {
		// With no good answer yet, the node's setting is still worth having.
		if !e.ok {
			e.info.set = d.set
		}
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

// describePlacement asks a placement's engine about the model, and the
// placement's node for its PAIR capacity setting, at the same time. The
// engine: this node's directly, a cluster peer's through its proxy's details
// route over mTLS, a manual node's at the address the user gave. The setting:
// this node's own, a cluster peer's over its mTLS capacity route; a manual
// node has no PAIR to ask. The setting is returned even when the engine could
// not be asked, and its absence is never an error.
func (p *Proxy) describePlacement(ctx context.Context, ref placementRef) (placementInfo, error) {
	engine := ref.engine()
	var info placementInfo
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		info.set = p.placementSetting(ctx, ref)
	}()
	var err error
	switch {
	case ref.cand.local:
		info.details, err = specReader.details(ctx, engine, ref.cand.url, ref.model)
	case ref.cand.peerUUID != "":
		client := &http.Client{Transport: p.candidateTransport(ref.cand), Timeout: peerDetailsTimeout}
		info.details, err = fetchPeerModelDetails(ctx, client, ref.cand.url, ref.model)
	default:
		info.details, err = remoteSpecReader.details(ctx, engine, ref.cand.url, ref.model)
	}
	wg.Wait()
	return info, err
}

// placementSetting reads the placement's node's capacity report and returns
// its setting for the model, or nil.
func (p *Proxy) placementSetting(ctx context.Context, ref placementRef) *capabilityConfigured {
	var rows []modelCapacity
	switch {
	case ref.cand.local:
		rows = p.capacityReport(ctx)
	case ref.cand.peerUUID != "":
		u := *ref.cand.url
		u.Path, u.RawPath, u.RawQuery = peerCapacityPath, "", ""
		var err error
		if rows, err = p.fetchPeerCapacity(ctx, u.String(), ref.cand.peerUUID); err != nil {
			slog.Debug("peer capacity unavailable for capabilities", "node_id", ref.cand.id, "err", err)
			return nil
		}
	default:
		return nil
	}
	return settingFor(rows, ref)
}

// settingFor picks the row for the placement's engine and model, matched under
// the engine's naming rule, and returns its setting if the user made one.
func settingFor(rows []modelCapacity, ref placementRef) *capabilityConfigured {
	if ref.cand.fac == nil {
		return nil
	}
	profile := ref.cand.fac.profile
	want := profile.normalizeModel(ref.model)
	for _, row := range rows {
		if row.Engine != profile.Name || row.Configured == nil || profile.normalizeModel(row.Model) != want {
			continue
		}
		input := row.MessageTokens
		if input == 0 {
			input = row.SessionShare
		}
		return &capabilityConfigured{
			ContextWindow:  row.SessionShare,
			ReplyTokens:    row.ReplyTokens,
			MaxInputTokens: input,
			Sessions:       row.MaxSessions,
		}
	}
	return nil
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
