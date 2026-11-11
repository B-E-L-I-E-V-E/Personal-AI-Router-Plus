// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"sort"
	"sync"
	"time"
)

// A session groups the requests of one ongoing conversation so the proxy can
// send each turn back to the node that served the last one. The engines are
// stateless at the API level — the client resends the whole conversation every
// turn — but each engine keeps the evaluated prompt in its KV cache, so a turn
// that lands on the same node reuses that work instead of recomputing it.
//
// The store holds routing metadata only: IDs, the owning agent, timestamps and
// the node per engine. It never holds prompts or responses (spec §9).
//
// Sessions live on the node a request originates from, because that is where
// the routing decision is made (architecture: a peer request is served, never
// re-routed). Nothing here is replicated to peers.
//
// A session expires after a sliding idle timeout. Any request for it, and the
// keepalive endpoint, slide the window; a session with a request in flight never
// expires, so a long generation or a long wait for a node cannot end it. Expiry
// and an explicit end both remove the session completely and leave a tombstone,
// so a later request naming the ID gets 410 Gone instead of silently starting
// a fresh conversation under the old name.
const (
	sessionIDHeader        = "X-PAIR-Session-ID"
	agentIDHeader          = "X-PAIR-Agent-ID"
	sessionExpiresAtHeader = "X-PAIR-Session-Expires-At"
)

const (
	defaultSessionIdleTimeout = 10 * time.Minute
	minSessionIdleTimeout     = time.Minute
	maxSessionIdleTimeout     = time.Hour
	defaultMaxSessions        = 4096
	// sessionSweepInterval is how often expired sessions are removed. Expiry is
	// also checked on every access, so this bounds only how long an abandoned
	// session's memory lingers, never whether a request sees it as expired.
	sessionSweepInterval = 15 * time.Second
	// A tombstone is what turns reuse of an ended ID into 410. It is kept long
	// enough to cover an agent that slept through its session's expiry, and the
	// count is bounded so a client churning IDs cannot grow it without limit.
	sessionTombstoneTTL  = 24 * time.Hour
	maxSessionTombstones = 16384
)

var (
	// errInvalidSessionID and errInvalidAgentID reject a malformed header. Unlike
	// the request ID these are not replaced: a substituted session ID would
	// silently start a new conversation, which is the opposite of what the
	// caller asked for.
	errInvalidSessionID = errors.New("invalid " + sessionIDHeader + ": use 1-128 characters from letters, digits, '.', '_', ':' and '-'")
	errInvalidAgentID   = errors.New("invalid " + agentIDHeader + ": use 1-128 characters from letters, digits, '.', '_', ':' and '-'")
	// errAgentMismatch refuses a request whose agent is not the session's owner.
	// The proxy has no authentication on loopback, so this is an integrity check
	// against two agents sharing an ID by mistake, not a security boundary.
	errAgentMismatch = errors.New("session belongs to a different agent")
	errSessionGone   = errors.New("session has ended or expired; start a new session")
	errSessionExists = errors.New("a session with this ID already exists")
	errNoSession     = errors.New("no such session")
	errSessionLimit  = errors.New("the session limit is reached; end an idle session or wait for one to expire")
	errIdleTimeout   = fmt.Errorf("idleTimeoutSeconds must be between %d and %d",
		int(minSessionIdleTimeout.Seconds()), int(maxSessionIdleTimeout.Seconds()))
)

// sessionStatus maps a session error to its HTTP status.
func sessionStatus(err error) int {
	switch {
	case errors.Is(err, errAgentMismatch), errors.Is(err, errSessionExists):
		return http.StatusConflict
	case errors.Is(err, errSessionGone):
		return http.StatusGone
	case errors.Is(err, errNoSession):
		return http.StatusNotFound
	case errors.Is(err, errSessionLimit):
		return http.StatusServiceUnavailable
	}
	return http.StatusBadRequest
}

// sessionConfig is the store-wide policy. It is adjustable at runtime through
// sessions/configure, and from the command line for a standalone proxy.
type sessionConfig struct {
	// IdleTimeout is the default sliding window for sessions that do not set
	// their own.
	IdleTimeout time.Duration
	// MaxLifetime caps a session's total length even while keepalives arrive.
	// Zero disables the cap. A request in flight is still allowed to finish.
	MaxLifetime time.Duration
	// MaxSessions bounds the number of live sessions.
	MaxSessions int
	// QueueTimeout is how long a session request may wait for a context slot
	// on this node before it is refused with 503. Zero refuses at once.
	QueueTimeout time.Duration
	// MaxQueued bounds how many session requests may wait for slots at once,
	// per model.
	MaxQueued int
}

func defaultSessionConfig() sessionConfig {
	return sessionConfig{
		IdleTimeout:  defaultSessionIdleTimeout,
		MaxSessions:  defaultMaxSessions,
		QueueTimeout: defaultQueueTimeout,
		MaxQueued:    defaultMaxQueued,
	}
}

func (c sessionConfig) validate() error {
	if c.IdleTimeout < minSessionIdleTimeout || c.IdleTimeout > maxSessionIdleTimeout {
		return errIdleTimeout
	}
	if c.MaxLifetime < 0 {
		return errors.New("maxLifetimeSeconds must not be negative")
	}
	if c.MaxSessions < 1 {
		return errors.New("maxSessions must be at least 1")
	}
	if c.QueueTimeout < 0 || c.QueueTimeout > maxQueueTimeout {
		return fmt.Errorf("queueTimeoutSeconds must be between 0 and %d", int(maxQueueTimeout.Seconds()))
	}
	if c.MaxQueued < 0 {
		return errors.New("maxQueued must not be negative")
	}
	return nil
}

func validIdleTimeout(d time.Duration) bool {
	return d >= minSessionIdleTimeout && d <= maxSessionIdleTimeout
}

type session struct {
	id      string
	agentID string
	// peer is set for a session another node routes here: its cluster ID. Such
	// a session exists only to account for this node's capacity (peer.go); it
	// is not listed, and its owner is the node that created it.
	peer string
	// remote is, per engine, the other node that served this session's last
	// turn, so that node can be told to release its slot when the session ends
	// or moves.
	remote  map[string]remoteRef
	created time.Time
	// lastActive is the last time a request for this session began or ended,
	// or a keepalive arrived.
	lastActive time.Time
	// idleTimeout overrides the store default when non-zero.
	idleTimeout time.Duration
	requests    uint64
	// nodes is the node that last served this session, per engine facade. A
	// session can use more than one engine; each engine's KV cache is its own.
	nodes map[string]string
	// local records, per engine, whether that node is this one.
	local map[string]bool
	// slots is the model whose context slot on this node the session holds,
	// per engine. See slots.go.
	slots map[string]string
	// context is the conversation's size after its last completed turn, per
	// engine, as the engine itself counted it.
	context map[string]sessionContext
	// inFlight holds a cancel function per request being served. Its size is
	// what keeps the session alive while it waits on the model, and ending the
	// session cancels every entry.
	inFlight map[uint64]context.CancelFunc
}

// sessionContext is how much of a model's context window a session occupied
// after its last completed turn on one engine. Tokens is prompt plus
// completion: the client resends the whole conversation each turn, so that is
// what the next turn starts from.
type sessionContext struct {
	Model      string
	Tokens     int64
	Prompt     int64
	Completion int64
	// BodyBytes is the size of the request that produced these counts.
	BodyBytes int
	Updated   time.Time
}

// sessionView is a copy of a session's fields, safe to read without the lock.
type sessionView struct {
	ID          string
	AgentID     string
	Created     time.Time
	LastActive  time.Time
	ExpiresAt   time.Time
	IdleTimeout time.Duration
	InFlight    int
	Requests    uint64
	Nodes       map[string]string
	Context     map[string]sessionContext
}

type sessionStore struct {
	mu         sync.Mutex
	cfg        sessionConfig
	sessions   map[string]*session
	tombstones map[string]time.Time
	nextToken  uint64
	now        func() time.Time
	// onPeerRelease tells another node a session no longer needs its slot
	// there. It is called without the lock held, on its own goroutine.
	onPeerRelease func(ref remoteRef, sessionID string)
	// queues holds the requests waiting for a context slot, per model, in
	// arrival order. changed is closed and replaced whenever a slot frees.
	queues  map[capacityKey][]*slotWaiter
	changed chan struct{}
}

func newSessionStore() *sessionStore {
	return &sessionStore{
		cfg:        defaultSessionConfig(),
		sessions:   make(map[string]*session),
		tombstones: make(map[string]time.Time),
		now:        time.Now,
		queues:     make(map[capacityKey][]*slotWaiter),
		changed:    make(chan struct{}),
	}
}

// sessionHeaders reads and validates the session and agent headers. Both are
// optional; an absent header yields "". A present but malformed one is an error.
func sessionHeaders(h http.Header) (sessionID, agentID string, err error) {
	if v, ok := h[http.CanonicalHeaderKey(sessionIDHeader)]; ok {
		if len(v) != 1 || !validRequestID(v[0]) {
			return "", "", errInvalidSessionID
		}
		sessionID = v[0]
	}
	agentID, err = agentHeader(h)
	if err != nil {
		return "", "", err
	}
	return sessionID, agentID, nil
}

// agentHeader reads and validates X-PAIR-Agent-ID alone.
func agentHeader(h http.Header) (string, error) {
	v, ok := h[http.CanonicalHeaderKey(agentIDHeader)]
	if !ok {
		return "", nil
	}
	if len(v) != 1 || !validRequestID(v[0]) {
		return "", errInvalidAgentID
	}
	return v[0], nil
}

func (s *sessionStore) config() sessionConfig {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cfg
}

// configure replaces the store policy. A lower session limit does not evict
// live sessions; it only refuses new ones until the count falls below it.
func (s *sessionStore) configure(cfg sessionConfig) error {
	if err := cfg.validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cfg = cfg
	return nil
}

// expiresAtLocked is when the session expires if nothing touches it again.
func (s *sessionStore) expiresAtLocked(sess *session) time.Time {
	idle := sess.idleTimeout
	if idle == 0 {
		idle = s.cfg.IdleTimeout
	}
	at := sess.lastActive.Add(idle)
	if s.cfg.MaxLifetime > 0 {
		if capAt := sess.created.Add(s.cfg.MaxLifetime); capAt.Before(at) {
			at = capAt
		}
	}
	return at
}

// expiredLocked reports whether a session has run out of time. A session with
// a request in flight is never expired.
func (s *sessionStore) expiredLocked(sess *session, now time.Time) bool {
	return len(sess.inFlight) == 0 && !now.Before(s.expiresAtLocked(sess))
}

// removeLocked deletes a session, cancels anything it has in flight, and
// leaves a tombstone.
func (s *sessionStore) removeLocked(sess *session, now time.Time) {
	for _, cancel := range sess.inFlight {
		cancel()
	}
	s.releaseSlotsLocked(sess)
	delete(s.sessions, sess.id)
	for _, ref := range sess.remote {
		s.notifyPeerReleaseLocked(ref, sess.id)
	}
	if sess.peer != "" {
		// A peer session is the other node's to end; tombstoning it here would
		// refuse that node's next turn after this copy merely went idle.
		return
	}
	s.tombstones[sess.id] = now
	if len(s.tombstones) > maxSessionTombstones {
		s.pruneTombstonesLocked(now)
	}
}

// pruneTombstonesLocked drops tombstones past their TTL and, if the set is
// still over its bound, the oldest of the rest.
func (s *sessionStore) pruneTombstonesLocked(now time.Time) {
	for id, at := range s.tombstones {
		if now.Sub(at) > sessionTombstoneTTL {
			delete(s.tombstones, id)
		}
	}
	if excess := len(s.tombstones) - maxSessionTombstones; excess > 0 {
		type tomb struct {
			id string
			at time.Time
		}
		all := make([]tomb, 0, len(s.tombstones))
		for id, at := range s.tombstones {
			all = append(all, tomb{id, at})
		}
		sort.Slice(all, func(i, j int) bool { return all[i].at.Before(all[j].at) })
		for _, t := range all[:excess] {
			delete(s.tombstones, t.id)
		}
	}
}

// liveLocked returns the session for id, expiring it first if its time is up.
func (s *sessionStore) liveLocked(id string, now time.Time) (*session, error) {
	if sess, ok := s.sessions[id]; ok {
		if s.expiredLocked(sess, now) {
			s.removeLocked(sess, now)
			return nil, errSessionGone
		}
		return sess, nil
	}
	if _, ok := s.tombstones[id]; ok {
		return nil, errSessionGone
	}
	return nil, errNoSession
}

// createLocked starts a session. The caller has checked the ID is free.
func (s *sessionStore) createLocked(id, agentID string, idle time.Duration, now time.Time) (*session, error) {
	if len(s.sessions) >= s.cfg.MaxSessions {
		// Expired sessions still count until swept; sweep before refusing.
		s.sweepLocked(now)
		if len(s.sessions) >= s.cfg.MaxSessions {
			return nil, errSessionLimit
		}
	}
	sess := &session{
		id: id, agentID: agentID, created: now, lastActive: now, idleTimeout: idle,
		nodes: make(map[string]string), local: make(map[string]bool), slots: make(map[string]string),
		remote:   make(map[string]remoteRef),
		context:  make(map[string]sessionContext),
		inFlight: make(map[uint64]context.CancelFunc),
	}
	s.sessions[id] = sess
	return sess, nil
}

// create starts a session explicitly. An empty idle uses the store default.
func (s *sessionStore) create(id, agentID string, idle time.Duration) (sessionView, error) {
	if idle != 0 && !validIdleTimeout(idle) {
		return sessionView{}, errIdleTimeout
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	if _, err := s.liveLocked(id, now); !errors.Is(err, errNoSession) {
		if err == nil {
			return sessionView{}, errSessionExists
		}
		return sessionView{}, err
	}
	sess, err := s.createLocked(id, agentID, idle, now)
	if err != nil {
		return sessionView{}, err
	}
	return s.viewLocked(sess), nil
}

// begin admits one request to a session, creating the session on first use.
// The agent that creates a session owns it; every later request must name the
// same agent, including naming none when the session was created without one.
// cancel stops the request if the session is ended while it runs. A successful
// begin returns a token that must be passed to end.
func (s *sessionStore) begin(id, agentID string, cancel context.CancelFunc) (sessionView, uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	sess, err := s.liveLocked(id, now)
	switch {
	case errors.Is(err, errNoSession):
		if sess, err = s.createLocked(id, agentID, 0, now); err != nil {
			return sessionView{}, 0, err
		}
	case err != nil:
		return sessionView{}, 0, err
	case sess.agentID != agentID:
		return sessionView{}, 0, errAgentMismatch
	}
	s.nextToken++
	token := s.nextToken
	sess.inFlight[token] = cancel
	sess.lastActive = now
	sess.requests++
	return s.viewLocked(sess), token, nil
}

// end releases a request admitted by begin and slides the session's window.
// It is a no-op once the session has been removed.
func (s *sessionStore) end(id string, token uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[id]
	if !ok {
		return
	}
	if _, ok := sess.inFlight[token]; !ok {
		return
	}
	delete(sess.inFlight, token)
	sess.lastActive = s.now()
}

// owned returns a live session after checking the caller is its agent.
func (s *sessionStore) ownedLocked(id, agentID string, now time.Time) (*session, error) {
	sess, err := s.liveLocked(id, now)
	if err != nil {
		return nil, err
	}
	if sess.agentID != agentID {
		return nil, errAgentMismatch
	}
	return sess, nil
}

// keepalive slides a session's window without sending a request.
func (s *sessionStore) keepalive(id, agentID string) (sessionView, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	sess, err := s.ownedLocked(id, agentID, now)
	if err != nil {
		return sessionView{}, err
	}
	sess.lastActive = now
	return s.viewLocked(sess), nil
}

// setIdleTimeout changes one session's window and slides it.
func (s *sessionStore) setIdleTimeout(id, agentID string, idle time.Duration) (sessionView, error) {
	if !validIdleTimeout(idle) {
		return sessionView{}, errIdleTimeout
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	sess, err := s.ownedLocked(id, agentID, now)
	if err != nil {
		return sessionView{}, err
	}
	sess.idleTimeout = idle
	sess.lastActive = now
	return s.viewLocked(sess), nil
}

// remove ends a session now, cancelling any request it has in flight.
func (s *sessionStore) remove(id, agentID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	sess, err := s.ownedLocked(id, agentID, now)
	if err != nil {
		return err
	}
	s.removeLocked(sess, now)
	return nil
}

// pin records the node that served a session on one engine, and whether that
// node is this one. A session that moves off a node releases what it held
// there: its slot here, or its slot on the other node it left.
func (s *sessionStore) pin(id, engine string, c candidate) {
	if c.id == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[id]
	if !ok {
		return
	}
	sess.nodes[engine] = c.id
	sess.local[engine] = c.local
	if !c.local {
		s.releaseSlotLocked(sess, engine)
	}
	if prev, had := sess.remote[engine]; had && prev.nodeID != c.id {
		s.notifyPeerReleaseLocked(prev, sess.id)
		delete(sess.remote, engine)
	}
	if !c.local && c.peerUUID != "" && c.url != nil {
		sess.remote[engine] = remoteRef{nodeID: c.id, host: c.url.Host, peerUUID: c.peerUUID}
	}
}

// localUsage is the live sessions this node serves for one engine and model,
// with the context they occupy.
type localUsage struct {
	Sessions int
	Tokens   int64
}

// localUsageByModel sums, per engine and model, the live sessions whose last
// turn on that engine ran on this node.
func (s *sessionStore) localUsageByModel() map[capacityKey]localUsage {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	out := make(map[capacityKey]localUsage)
	for _, sess := range s.sessions {
		if s.expiredLocked(sess, now) {
			continue
		}
		for engine, c := range sess.context {
			if !sess.local[engine] {
				continue
			}
			k := capacityKey{Engine: engine, Model: c.Model}
			u := out[k]
			u.Sessions++
			u.Tokens += c.Tokens
			out[k] = u
		}
	}
	return out
}

// recordUsage stores the token counts an engine reported for a session's
// completed turn. Counts are all the store keeps; never any text.
// bodyBytes is the size of the request that produced the counts, so a later
// request can be recognised as shorter (a compacted conversation).
func (s *sessionStore) recordUsage(id, engine, model string, u tokenUsage, bodyBytes int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if sess, ok := s.sessions[id]; ok {
		sess.context[engine] = sessionContext{
			Model: model, Tokens: u.total(), Prompt: u.Prompt, Completion: u.Completion,
			BodyBytes: bodyBytes, Updated: s.now(),
		}
	}
}

// get returns a copy of one live session.
func (s *sessionStore) get(id string) (sessionView, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, err := s.liveLocked(id, s.now())
	if err != nil {
		return sessionView{}, err
	}
	return s.viewLocked(sess), nil
}

// list returns the live sessions, optionally only one agent's, oldest first.
func (s *sessionStore) list(agentID string) []sessionView {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sweepLocked(s.now())
	out := make([]sessionView, 0, len(s.sessions))
	for _, sess := range s.sessions {
		if sess.peer != "" {
			continue
		}
		if agentID == "" || sess.agentID == agentID {
			out = append(out, s.viewLocked(sess))
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if !out[i].Created.Equal(out[j].Created) {
			return out[i].Created.Before(out[j].Created)
		}
		return out[i].ID < out[j].ID
	})
	return out
}

// sweep removes every expired session and stale tombstone. It returns how many
// sessions it removed.
func (s *sessionStore) sweep() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sweepLocked(s.now())
}

func (s *sessionStore) sweepLocked(now time.Time) int {
	removed := 0
	for _, sess := range s.sessions {
		if s.expiredLocked(sess, now) {
			s.removeLocked(sess, now)
			removed++
		}
	}
	s.pruneTombstonesLocked(now)
	return removed
}

// run sweeps on a timer until ctx ends.
func (s *sessionStore) run(ctx context.Context) {
	t := time.NewTicker(sessionSweepInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.sweep()
		}
	}
}

func (s *sessionStore) viewLocked(sess *session) sessionView {
	nodes := make(map[string]string, len(sess.nodes))
	for k, v := range sess.nodes {
		nodes[k] = v
	}
	ctxs := make(map[string]sessionContext, len(sess.context))
	for k, v := range sess.context {
		ctxs[k] = v
	}
	idle := sess.idleTimeout
	if idle == 0 {
		idle = s.cfg.IdleTimeout
	}
	return sessionView{
		ID:          sess.id,
		AgentID:     sess.agentID,
		Created:     sess.created,
		LastActive:  sess.lastActive,
		ExpiresAt:   s.expiresAtLocked(sess),
		IdleTimeout: idle,
		InFlight:    len(sess.inFlight),
		Requests:    sess.requests,
		Nodes:       nodes,
		Context:     ctxs,
	}
}

// preferNode moves nodeID to the front of cands when it is present, keeping
// the relative order of the rest. It reports whether the node was found. The
// capability gate has already run, so a node that no longer advertises the
// model is simply absent and the scheduler's order stands.
func preferNode(cands []candidate, nodeID string) bool {
	for i, c := range cands {
		if c.id != nodeID {
			continue
		}
		if i > 0 {
			copy(cands[1:i+1], cands[:i])
			cands[0] = c
		}
		return true
	}
	return false
}

// optionalString maps "" to nil, for the workload's nullable string fields.
func optionalString(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// sessionConfigParams is the sessions/configure payload. Every field is
// optional; an omitted one keeps its current value.
type sessionConfigParams struct {
	IdleTimeoutSeconds *int `json:"idleTimeoutSeconds"`
	MaxLifetimeSeconds *int `json:"maxLifetimeSeconds"`
	MaxSessions        *int `json:"maxSessions"`
	QueueTimeoutSecs   *int `json:"queueTimeoutSeconds"`
	MaxQueued          *int `json:"maxQueued"`
}

// applySessionConfigParams overlays a sessions/configure payload on cfg.
func applySessionConfigParams(cfg sessionConfig, raw json.RawMessage) (sessionConfig, error) {
	var params sessionConfigParams
	if len(raw) > 0 {
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&params); err != nil {
			return cfg, errors.New(`expected {"idleTimeoutSeconds","maxLifetimeSeconds","maxSessions","queueTimeoutSeconds","maxQueued"}`)
		}
	}
	if params.IdleTimeoutSeconds != nil {
		cfg.IdleTimeout = time.Duration(*params.IdleTimeoutSeconds) * time.Second
	}
	if params.MaxLifetimeSeconds != nil {
		cfg.MaxLifetime = time.Duration(*params.MaxLifetimeSeconds) * time.Second
	}
	if params.MaxSessions != nil {
		cfg.MaxSessions = *params.MaxSessions
	}
	if params.QueueTimeoutSecs != nil {
		cfg.QueueTimeout = time.Duration(*params.QueueTimeoutSecs) * time.Second
	}
	if params.MaxQueued != nil {
		cfg.MaxQueued = *params.MaxQueued
	}
	return cfg, cfg.validate()
}

// sessionConfigResult is the wire form of the policy in force.
func sessionConfigResult(cfg sessionConfig) map[string]int {
	return map[string]int{
		"idleTimeoutSeconds":  int(cfg.IdleTimeout.Seconds()),
		"maxLifetimeSeconds":  int(cfg.MaxLifetime.Seconds()),
		"maxSessions":         cfg.MaxSessions,
		"queueTimeoutSeconds": int(cfg.QueueTimeout.Seconds()),
		"maxQueued":           cfg.MaxQueued,
	}
}

// registerSessionFlags declares the session policy flags on fs. The returned
// function applies only the flags given on the command line to base, so a
// saved policy keeps every value the operator did not explicitly override,
// and a field without a flag keeps base's value rather than a zero.
func registerSessionFlags(fs *flag.FlagSet) func(base sessionConfig) sessionConfig {
	d := defaultSessionConfig()
	idle := fs.Duration("session-idle-timeout", d.IdleTimeout, "default sliding idle timeout for sessions (1m to 60m); adjustable at runtime via sessions/configure")
	lifetime := fs.Duration("session-max-lifetime", d.MaxLifetime, "maximum total length of a session even while it is kept alive; 0 disables the cap")
	maxSessions := fs.Int("max-sessions", d.MaxSessions, "maximum number of live sessions")
	queueTimeout := fs.Duration("session-queue-timeout", d.QueueTimeout, "how long a session request may wait for a context slot on this node (0 to 5m; 0 refuses at once)")
	maxQueued := fs.Int("session-max-queued", d.MaxQueued, "maximum session requests waiting for context slots, per model")
	return func(base sessionConfig) sessionConfig {
		cfg := base
		fs.Visit(func(f *flag.Flag) {
			switch f.Name {
			case "session-idle-timeout":
				cfg.IdleTimeout = *idle
			case "session-max-lifetime":
				cfg.MaxLifetime = *lifetime
			case "max-sessions":
				cfg.MaxSessions = *maxSessions
			case "session-queue-timeout":
				cfg.QueueTimeout = *queueTimeout
			case "session-max-queued":
				cfg.MaxQueued = *maxQueued
			}
		})
		return cfg
	}
}
