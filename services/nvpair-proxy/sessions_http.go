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
}

func toSessionJSON(v sessionView) sessionJSON {
	// A third of the window leaves two chances to refresh before expiry.
	keepalive := int(v.IdleTimeout.Seconds()) / 3
	if keepalive < 1 {
		keepalive = 1
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
				out[i] = toSessionJSON(v)
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
			respondSession(w, v, err, http.StatusOK)
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
			respondSession(w, v, err, http.StatusOK)
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
		respondSession(w, v, err, http.StatusOK)

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
	respondSession(w, v, err, http.StatusCreated)
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

func respondSession(w http.ResponseWriter, v sessionView, err error, status int) {
	if err != nil {
		writeSessionError(w, err)
		return
	}
	w.Header().Set(sessionExpiresAtHeader, v.ExpiresAt.UTC().Format(time.RFC3339))
	writeSessionJSON(w, status, toSessionJSON(v))
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
