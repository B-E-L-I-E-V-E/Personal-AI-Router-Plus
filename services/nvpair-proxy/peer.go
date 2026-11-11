// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Context capacity across the cluster: each node enforces its own.
//
// The node a request starts on routes it and forwards it, headers included,
// so a session request reaching a peer's mTLS ingress still carries
// X-PAIR-Session-ID. The ingress applies the same slot and share rules as
// local admission (admit.go) to the model on its own engine, tracking the
// session under the calling node's cluster ID so two nodes' session IDs never
// collide. A full peer waits briefly for a slot and then answers 503, which
// the calling node already treats as "try the next candidate"; a session that
// has outgrown its share there is answered 400.
//
// The calling node releases a peer's slot explicitly when its session moves
// to another node or ends, so the slot does not sit idle until it expires.
const (
	// peerReleasePathPrefix is served on the mTLS ingress only.
	peerReleasePathPrefix = "/pair/v1/peer/sessions/"
	peerReleaseSuffix     = "/release"
	// maxPeerQueueWait keeps a peer's wait for a slot well inside the calling
	// node's wait for response headers (proxyResponseTimeout), so a slow
	// queue reads as "full" rather than as a dead node.
	maxPeerQueueWait   = 90 * time.Second
	peerReleaseTimeout = 5 * time.Second
)

// remoteRef identifies another node that holds a slot for a session, and how
// to reach its ingress.
type remoteRef struct {
	nodeID   string
	host     string
	peerUUID string
}

// peerSessionKey is the store key for a session another node routes here.
// The "/" cannot appear in a client's session ID, so the two never collide.
func peerSessionKey(peer, sessionID string) string {
	return "peer:" + peer + "/" + sessionID
}

// beginPeer admits one request for a peer's session, creating its record on
// first use. A peer session is never refused as ended: only the calling node
// decides that, and a copy that went idle here is simply started again.
func (s *sessionStore) beginPeer(key, peer string, cancel context.CancelFunc) (sessionView, uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	delete(s.tombstones, key)
	sess, err := s.liveLocked(key, now)
	if err != nil {
		if sess, err = s.createLocked(key, "", 0, now); err != nil {
			return sessionView{}, 0, err
		}
		sess.peer = peer
	}
	s.nextToken++
	token := s.nextToken
	sess.inFlight[token] = cancel
	sess.lastActive = now
	sess.requests++
	return s.viewLocked(sess), token, nil
}

// releasePeer ends a peer's session here at that node's request. Only the peer
// that owns the session may release it.
func (s *sessionStore) releasePeer(key, peer string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[key]
	if !ok || sess.peer != peer {
		return false
	}
	s.removeLocked(sess, s.now())
	return true
}

// notifyPeerReleaseLocked schedules a release message to another node.
func (s *sessionStore) notifyPeerReleaseLocked(ref remoteRef, sessionID string) {
	if s.onPeerRelease == nil {
		return
	}
	notify := s.onPeerRelease
	go notify(ref, sessionID)
}

// sendPeerRelease tells another node this session no longer needs its slot.
// Best effort: a node that does not answer releases the slot when the session
// goes idle there.
func (p *Proxy) sendPeerRelease(ref remoteRef, sessionID string) {
	ctx, cancel := context.WithTimeout(context.Background(), peerReleaseTimeout)
	defer cancel()
	u := url.URL{Scheme: "https", Host: ref.host, Path: peerReleasePathPrefix + url.PathEscape(sessionID) + peerReleaseSuffix}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), nil)
	if err != nil {
		return
	}
	client := &http.Client{Transport: p.peerHTTPTransport(ref.peerUUID)}
	resp, err := client.Do(req)
	if err != nil {
		slog.Debug("peer session release failed", "node_id", ref.nodeID, "session_id", sessionID, "err", err)
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	_ = resp.Body.Close()
}

// servePeerRelease answers a release from the calling node.
func (f *facade) servePeerRelease(w http.ResponseWriter, r *http.Request, peer string) bool {
	if !strings.HasPrefix(r.URL.Path, peerReleasePathPrefix) || !strings.HasSuffix(r.URL.Path, peerReleaseSuffix) {
		return false
	}
	id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, peerReleasePathPrefix), peerReleaseSuffix)
	if r.Method != http.MethodPost || !validRequestID(id) {
		writeIngressError(w, http.StatusBadRequest, "peer-release", "expected POST "+peerReleasePathPrefix+"{sessionId}"+peerReleaseSuffix)
		return true
	}
	f.host.sessions.releasePeer(peerSessionKey(peer, id), peer)
	w.WriteHeader(http.StatusNoContent)
	return true
}

// admitPeer applies this node's context capacity to a session request that
// another node routed here. It returns the request to forward (its body
// restored), a function to call when the request ends, and, when the request
// is refused, the status and message to answer with.
func (f *facade) admitPeer(r *http.Request, peer string) (*http.Request, func(tokenUsage), func(), int, error) {
	none := func() {}
	if !isInferenceRequest(f.profile, r.Method, r.URL.Path) {
		return r, nil, none, 0, nil
	}
	sessionID := r.Header.Get(sessionIDHeader)
	if sessionID == "" || !validRequestID(sessionID) {
		return r, nil, none, 0, nil
	}
	body, model := bufferBodyAndModel(r)
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	if model == "" {
		return r, nil, none, 0, nil
	}
	p := f.host
	k := capacityKey{Engine: f.profile.Name, Model: model}
	key := peerSessionKey(peer, sessionID)

	ctx, cancel := context.WithCancel(r.Context())
	sess, token, err := p.sessions.beginPeer(key, peer, cancel)
	if err != nil {
		cancel()
		return r, nil, none, sessionStatus(err), err
	}
	done := func() {
		p.sessions.end(key, token)
		cancel()
	}
	p.sessions.pin(key, k.Engine, candidate{id: "self", local: true})
	record := func(u tokenUsage) { p.sessions.recordUsage(key, k.Engine, model, u, len(body)) }

	lim := p.limitFor(f, k)
	if lim.SessionShare > 0 {
		if c, ok := sess.Context[k.Engine]; ok && c.Model == model && c.Tokens >= lim.SessionShare && len(body) >= c.BodyBytes {
			done()
			return r, nil, none, http.StatusBadRequest, fmt.Errorf("%w: it holds %d tokens against a share of %d on the serving node; shorten the conversation or end the session",
				errContextExceeded, c.Tokens, lim.SessionShare)
		}
	}
	if lim.MaxSessions > 0 && !p.sessions.takeSlotNow(key, k, lim.MaxSessions) {
		wait := min(p.sessions.config().QueueTimeout, maxPeerQueueWait)
		wctx, wcancel := context.WithTimeout(ctx, wait)
		err := p.sessions.waitForSlot(wctx, key, k, func() int {
			if n := p.limitFor(f, k).MaxSessions; n > 0 {
				return n
			}
			return int(^uint(0) >> 1)
		})
		wcancel()
		if err != nil {
			done()
			if errors.Is(err, context.DeadlineExceeded) {
				err = errQueueTimeout
			}
			return r, nil, none, http.StatusServiceUnavailable, err
		}
	}
	return r.WithContext(ctx), record, done, 0, nil
}

// writePeerRefusal answers a refused peer session request.
func writePeerRefusal(w http.ResponseWriter, status int, err error) {
	if status == http.StatusServiceUnavailable {
		w.Header().Set("Retry-After", strconv.Itoa(5))
	}
	writeIngressError(w, status, "context-admission", err.Error())
}
