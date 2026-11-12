// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"io"
	"log/slog"
	"math"
	"net/http"
	"sort"
	"sync"
	"time"
)

// Live request telemetry (spec §5.16): every generation request in progress
// on this node, with its figures so far, updated as each frame of the reply
// arrives. A request appears when the proxy receives it and leaves when its
// reply ends; its final figures then go to the recent list. Like everything
// else here it holds numbers and identifiers only, never text.

// Live phases, in the order a request moves through them.
const (
	livePhaseWaiting   = "waiting"   // received; admission, queueing or choosing a node
	livePhaseReading   = "reading"   // sent to a node; the engine is reading the prompt
	livePhaseThinking  = "thinking"  // the reply's reasoning part is streaming
	livePhaseAnswering = "answering" // the answer is streaming
)

// Live request origins.
const (
	// liveOriginRouted is a request a client on this machine sent to this
	// node's proxy, served here or on another node.
	liveOriginRouted = "routed"
	// liveOriginPeer is a request another node sent here to be served by this
	// node's engine (cluster ingress).
	liveOriginPeer = "peer"
)

// liveRequest is one request in progress, as reported.
type liveRequest struct {
	RequestID string `json:"requestId"`
	SessionID string `json:"sessionId,omitempty"`
	AgentID   string `json:"agentId,omitempty"`
	Engine    string `json:"engine"`
	Model     string `json:"model"`
	// NodeID is the node serving it; empty while it is still waiting, and
	// empty for a peer request when this node has not yet learned its own ID.
	NodeID    string `json:"nodeId"`
	Local     bool   `json:"local"`
	Path      string `json:"path"`
	Origin    string `json:"origin"`
	Benchmark bool   `json:"benchmark,omitempty"`
	Phase     string `json:"phase"`

	ReceivedAt   int64 `json:"receivedAt"`
	DispatchedAt int64 `json:"dispatchedAt,omitempty"`
	// ElapsedMs is from receipt to now.
	ElapsedMs float64 `json:"elapsedMs"`

	// PromptTokensEstimate is the request body's size over four bytes a
	// token, until the engine reports the real count at the end.
	PromptTokensEstimate int64 `json:"promptTokensEstimate"`
	// OutputTokens and ThinkingTokens count streamed chunks so far (about one
	// token each); a reply that is not streamed shows them at its end only.
	OutputTokens   int64    `json:"outputTokens"`
	ThinkingTokens int64    `json:"thinkingTokens"`
	TTFTMs         *float64 `json:"timeToFirstTokenMs,omitempty"`
	ThinkingMs     *float64 `json:"thinkingMs,omitempty"`
	// WriteTPS is the writing speed so far, first to latest token.
	WriteTPS *float64 `json:"writeTokensPerSecond,omitempty"`
}

// liveEntry is the mutable state behind one liveRequest; the store's lock
// guards it.
type liveEntry struct {
	req        liveRequest
	received   time.Time
	dispatched time.Time
	firstOut   time.Time
	lastOut    time.Time
	firstThink time.Time
	firstAns   time.Time
}

type liveRegistry struct {
	mu      sync.Mutex
	entries map[*liveEntry]struct{}
}

func newLiveRegistry() *liveRegistry {
	return &liveRegistry{entries: make(map[*liveEntry]struct{})}
}

// begin registers a request as received. The returned function removes it.
func (l *liveRegistry) begin(req liveRequest, bodyLen int) (*liveEntry, func()) {
	now := time.Now()
	req.Phase = livePhaseWaiting
	req.ReceivedAt = now.UnixMilli()
	req.PromptTokensEstimate = int64(math.Ceil(float64(bodyLen) / 4))
	e := &liveEntry{req: req, received: now}
	l.mu.Lock()
	l.entries[e] = struct{}{}
	l.mu.Unlock()
	return e, func() {
		l.mu.Lock()
		delete(l.entries, e)
		l.mu.Unlock()
	}
}

// dispatch records the node an attempt was sent to. A failed-over request is
// dispatched again, so the latest node wins and the counts restart.
func (l *liveRegistry) dispatch(e *liveEntry, nodeID string, local bool, at time.Time) {
	if e == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	e.req.NodeID, e.req.Local, e.req.Phase = nodeID, local, livePhaseReading
	e.dispatched = at
	e.req.DispatchedAt = at.UnixMilli()
	e.firstOut, e.lastOut, e.firstThink, e.firstAns = time.Time{}, time.Time{}, time.Time{}, time.Time{}
	e.req.OutputTokens, e.req.ThinkingTokens = 0, 0
}

// progress applies one frame's measurement: think and answer report whether
// the frame carried reasoning or answer text, visible whether the answer has
// started in earnest.
func (l *liveRegistry) progress(e *liveEntry, at time.Time, think, answer, visible bool) {
	if e == nil || (!think && !answer) {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	e.req.OutputTokens++
	if e.firstOut.IsZero() {
		e.firstOut = at
	}
	e.lastOut = at
	if think {
		e.req.ThinkingTokens++
		if e.firstThink.IsZero() {
			e.firstThink = at
		}
		if e.firstAns.IsZero() {
			e.req.Phase = livePhaseThinking
		}
	}
	if answer && visible && e.firstAns.IsZero() {
		e.firstAns = at
	}
	if !e.firstAns.IsZero() {
		e.req.Phase = livePhaseAnswering
	}
}

// snapshot reports every request in progress, oldest first.
func (l *liveRegistry) snapshot(now time.Time) []liveRequest {
	l.mu.Lock()
	out := make([]liveRequest, 0, len(l.entries))
	for e := range l.entries {
		r := e.req
		r.ElapsedMs = *fptr(msBetween(e.received, now))
		if !e.firstOut.IsZero() {
			start := e.dispatched
			if start.IsZero() {
				start = e.received
			}
			r.TTFTMs = fptr(msBetween(start, e.firstOut))
			if e.lastOut.After(e.firstOut) && r.OutputTokens > 1 {
				r.WriteTPS = perSecond(float64(r.OutputTokens-1), msBetween(e.firstOut, e.lastOut))
			}
		}
		if !e.firstThink.IsZero() {
			end := e.firstAns
			if end.IsZero() {
				end = now
			}
			r.ThinkingMs = fptr(msBetween(e.firstThink, end))
		}
		out = append(out, r)
	}
	l.mu.Unlock()
	sort.Slice(out, func(i, j int) bool {
		if out[i].ReceivedAt != out[j].ReceivedAt {
			return out[i].ReceivedAt < out[j].ReceivedAt
		}
		return out[i].RequestID < out[j].RequestID
	})
	return out
}

// perfLive is the performance/live result and the body of
// GET /pair/v1/performance/live: requests in progress and the latest finished
// ones, for a view that refreshes several times a second.
type perfLive struct {
	Live   []liveRequest `json:"live"`
	Recent []perfRecord  `json:"recent"`
}

func (p *Proxy) performanceLive() perfLive {
	rep := perfLive{Live: p.live.snapshot(time.Now())}
	p.perf.mu.Lock()
	n := min(len(p.perf.recent), perfReportRecent)
	rep.Recent = make([]perfRecord, 0, n)
	for i := len(p.perf.recent) - 1; i >= len(p.perf.recent)-n; i-- {
		rep.Recent = append(rep.Recent, p.perf.recent[i])
	}
	p.perf.mu.Unlock()
	return rep
}

// measurePeerRequest puts a generation request another node sent here into
// the live view and returns a tracker for its reply, so requests this node
// serves for the cluster are measured like its own. It buffers the body to
// read the model, which the forward then replays.
func (f *facade) measurePeerRequest(r *http.Request, peer string) (*perfTracker, func()) {
	if !isInferenceRequest(f.profile, r.Method, r.URL.Path) || !isGenerationPath(r.URL.Path) {
		return nil, func() {}
	}
	body, model := bufferBodyAndModel(r)
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	if model == "" {
		return nil, func() {}
	}
	p := f.host
	self, _ := p.selfNodeID.Load().(string)
	requestID := resolveRequestID(r.Header)
	sessionID := r.Header.Get(sessionIDHeader)
	agentID := r.Header.Get(agentIDHeader)
	if !validRequestID(sessionID) {
		sessionID = ""
	}
	if !validRequestID(agentID) {
		agentID = ""
	}
	entry, end := p.live.begin(liveRequest{
		RequestID: requestID, SessionID: sessionID, AgentID: agentID,
		Engine: f.profile.Name, Model: model, Path: r.URL.Path, Origin: liveOriginPeer,
	}, len(body))
	start := time.Now()
	p.live.dispatch(entry, self, true, start)
	tr := newPerfTracker(start, perfRecord{
		RequestID: requestID, Engine: f.profile.Name, Model: model,
		NodeID: self, Local: true, Path: r.URL.Path,
		SessionID: sessionID, AgentID: agentID, Origin: liveOriginPeer,
	}, func(rec perfRecord) { p.perf.add(rec, false) })
	tr.live, tr.liveReg = entry, p.live
	slog.Debug("measuring a peer request", "peer", peer, "engine", f.profile.Name, "model", model, "request_id", requestID)
	return tr, end
}
