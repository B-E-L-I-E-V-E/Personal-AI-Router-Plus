// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// sessionsAPIPrefix is PAIR's own session API. It is answered by the proxy and
// never forwarded to an engine, on every facade's port, because the store is
// process-wide. It is reachable only through the loopback plaintext
// personality: handlePlain dispatches here, and cluster ingress never does.
const sessionsAPIPrefix = "/pair/v1/sessions"

// maxSessionAPIBody bounds a session API request body.
const maxSessionAPIBody = 64 << 10

// sessionJSON is the wire form of a session.
type sessionJSON struct {
	SessionID          string            `json:"sessionId"`
	AgentID            *string           `json:"agentId"`
	CreatedAt          string            `json:"createdAt"`
	LastActiveAt       string            `json:"lastActiveAt"`
	ExpiresAt          string            `json:"expiresAt"`
	IdleTimeoutSeconds int               `json:"idleTimeoutSeconds"`
	KeepaliveSeconds   int               `json:"keepaliveIntervalSeconds"`
	InFlight           int               `json:"inFlight"`
	Requests           uint64            `json:"requests"`
	Nodes              map[string]string `json:"nodes"`
	// Context is the conversation's size after its last completed turn, per
	// engine, as the engine counted it. Absent until a turn reports usage.
	Context map[string]sessionContextJSON `json:"context"`
}

type sessionContextJSON struct {
	Model            string `json:"model"`
	Tokens           int64  `json:"tokens"`
	PromptTokens     int64  `json:"promptTokens"`
	CompletionTokens int64  `json:"completionTokens"`
	UpdatedAt        string `json:"updatedAt"`
	// The limits on this node, present when this node serves the session
	// and the model's window is known: the session's window, the part kept
	// for replies, what the conversation may hold (window less reply), and
	// how many tokens the next post may add before reaching it.
	WindowTokens            int64  `json:"windowTokens,omitempty"`
	ReplyTokens             int64  `json:"replyTokens,omitempty"`
	ConversationLimitTokens int64  `json:"conversationLimitTokens,omitempty"`
	PostRoomTokens          *int64 `json:"postRoomTokens,omitempty"`
}

// sessionJSON is v's wire form with this node's limits on each context it
// serves (see sessionContextJSON).
func (p *Proxy) sessionJSON(v sessionView) sessionJSON {
	out := toSessionJSON(v)
	for engine, c := range out.Context {
		if !v.Local[engine] {
			continue
		}
		var f *facade
		for _, candidate := range p.enabledFacades() {
			if candidate.profile.Name == engine {
				f = candidate
			}
		}
		if f == nil {
			continue
		}
		lim := p.limitFor(f, capacityKey{Engine: engine, Model: c.Model})
		if lim.SessionShare <= 0 {
			continue
		}
		room := lim.postRoom(c.Tokens)
		c.WindowTokens, c.ReplyTokens, c.ConversationLimitTokens, c.PostRoomTokens =
			lim.SessionShare, lim.ReplyTokens, lim.conversationLimit(), &room
		out.Context[engine] = c
	}
	return out
}

func toSessionJSON(v sessionView) sessionJSON {
	// A third of the window leaves two chances to refresh before expiry.
	keepalive := int(v.IdleTimeout.Seconds()) / 3
	if keepalive < 1 {
		keepalive = 1
	}
	ctxs := make(map[string]sessionContextJSON, len(v.Context))
	for engine, c := range v.Context {
		ctxs[engine] = sessionContextJSON{
			Model: c.Model, Tokens: c.Tokens, PromptTokens: c.Prompt, CompletionTokens: c.Completion,
			UpdatedAt: c.Updated.UTC().Format(time.RFC3339Nano),
		}
	}
	return sessionJSON{
		SessionID:          v.ID,
		AgentID:            optionalString(v.AgentID),
		CreatedAt:          v.Created.UTC().Format(time.RFC3339Nano),
		LastActiveAt:       v.LastActive.UTC().Format(time.RFC3339Nano),
		ExpiresAt:          v.ExpiresAt.UTC().Format(time.RFC3339Nano),
		IdleTimeoutSeconds: int(v.IdleTimeout.Seconds()),
		KeepaliveSeconds:   keepalive,
		InFlight:           v.InFlight,
		Requests:           v.Requests,
		Nodes:              v.Nodes,
		Context:            ctxs,
	}
}

// serveSessionsAPI answers a request under sessionsAPIPrefix. It reports
// whether the path belonged to the API, so the caller routes everything else.
//
//	POST   /pair/v1/sessions                  create
//	GET    /pair/v1/sessions[?agentId=...]    list
//	GET    /pair/v1/sessions/{id}             read
//	PATCH  /pair/v1/sessions/{id}             change idleTimeoutSeconds
//	DELETE /pair/v1/sessions/{id}             end now
//	POST   /pair/v1/sessions/{id}/keepalive   slide the idle window
//
// Every call that changes a session must name its owner in X-PAIR-Agent-ID
// (or omit it for an unowned session), the same rule inference follows.
func (f *facade) serveSessionsAPI(w http.ResponseWriter, r *http.Request) bool {
	path := r.URL.Path
	if path != sessionsAPIPrefix && !strings.HasPrefix(path, sessionsAPIPrefix+"/") {
		return false
	}
	store := f.host.sessions
	w.Header().Set(requestIDHeader, resolveRequestID(r.Header))

	rest := strings.TrimPrefix(strings.TrimPrefix(path, sessionsAPIPrefix), "/")
	parts := []string{}
	if rest != "" {
		parts = strings.Split(rest, "/")
	}

	agentID, err := agentHeader(r.Header)
	if err != nil {
		writeSessionError(w, err)
		return true
	}

	switch {
	case len(parts) == 0:
		switch r.Method {
		case http.MethodPost:
			f.createSession(w, r, agentID)
		case http.MethodGet:
			filter := r.URL.Query().Get("agentId")
			views := store.list(filter)
			out := make([]sessionJSON, len(views))
			for i, v := range views {
				out[i] = f.host.sessionJSON(v)
			}
			writeSessionJSON(w, http.StatusOK, map[string]any{"sessions": out})
		default:
			methodNotAllowed(w, "GET, POST")
		}

	case len(parts) == 1 && validRequestID(parts[0]):
		id := parts[0]
		switch r.Method {
		case http.MethodGet:
			v, err := store.get(id)
			f.respondSession(w, r, v, err, http.StatusOK)
		case http.MethodPatch:
			var body struct {
				IdleTimeoutSeconds *int `json:"idleTimeoutSeconds"`
			}
			if err := decodeSessionBody(r, &body); err != nil {
				writeSessionError(w, err)
				return true
			}
			if body.IdleTimeoutSeconds == nil {
				writeSessionError(w, errors.New("nothing to change: send idleTimeoutSeconds"))
				return true
			}
			v, err := store.setIdleTimeout(id, agentID, time.Duration(*body.IdleTimeoutSeconds)*time.Second)
			f.respondSession(w, r, v, err, http.StatusOK)
		case http.MethodDelete:
			if err := store.remove(id, agentID); err != nil {
				writeSessionError(w, err)
				return true
			}
			slog.Debug("session ended by request", "session_id", id)
			w.WriteHeader(http.StatusNoContent)
		default:
			methodNotAllowed(w, "GET, PATCH, DELETE")
		}

	case len(parts) == 2 && validRequestID(parts[0]) && parts[1] == "keepalive":
		if r.Method != http.MethodPost {
			methodNotAllowed(w, "POST")
			return true
		}
		v, err := store.keepalive(parts[0], agentID)
		f.respondSession(w, r, v, err, http.StatusOK)

	default:
		writeSessionJSON(w, http.StatusNotFound, map[string]string{"error": "unknown session API path"})
	}
	return true
}

func (f *facade) createSession(w http.ResponseWriter, r *http.Request, headerAgent string) {
	var body struct {
		SessionID          string `json:"sessionId"`
		AgentID            string `json:"agentId"`
		IdleTimeoutSeconds int    `json:"idleTimeoutSeconds"`
	}
	if err := decodeSessionBody(r, &body); err != nil {
		writeSessionError(w, err)
		return
	}
	agentID := headerAgent
	if body.AgentID != "" {
		if !validRequestID(body.AgentID) {
			writeSessionError(w, errInvalidAgentID)
			return
		}
		if headerAgent != "" && headerAgent != body.AgentID {
			writeSessionError(w, errors.New("agentId in the body and "+agentIDHeader+" disagree"))
			return
		}
		agentID = body.AgentID
	}
	id := body.SessionID
	if id == "" {
		id = newRequestID()
	} else if !validRequestID(id) {
		writeSessionError(w, errInvalidSessionID)
		return
	}
	v, err := f.host.sessions.create(id, agentID, time.Duration(body.IdleTimeoutSeconds)*time.Second)
	f.respondSession(w, r, v, err, http.StatusCreated)
}

// decodeSessionBody decodes an optional JSON body, rejecting unknown fields so
// a misspelled option fails loudly instead of being ignored.
func decodeSessionBody(r *http.Request, dst any) error {
	if r.Body == nil {
		return nil
	}
	dec := json.NewDecoder(io.LimitReader(r.Body, maxSessionAPIBody))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil && !errors.Is(err, io.EOF) {
		return errors.New("invalid JSON body: " + err.Error())
	}
	return nil
}

func (f *facade) respondSession(w http.ResponseWriter, r *http.Request, v sessionView, err error, status int) {
	if err != nil {
		writeSessionError(w, err)
		return
	}
	w.Header().Set(sessionExpiresAtHeader, v.ExpiresAt.UTC().Format(time.RFC3339))
	writeSessionJSON(w, status, f.host.sessionJSON(v))
}

func writeSessionError(w http.ResponseWriter, err error) {
	writeSessionJSON(w, sessionStatus(err), map[string]string{"error": err.Error()})
}

func methodNotAllowed(w http.ResponseWriter, allow string) {
	w.Header().Set("Allow", allow)
	writeSessionJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
}

func writeSessionJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Debug("session API response write failed", "err", err)
	}
}
