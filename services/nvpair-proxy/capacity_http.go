// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"sort"
)

// capacityAPIPath is PAIR's context-capacity report and setting for this node.
// Like the session API it is answered by the proxy, never forwarded, and
// reachable from loopback only.
//
//	GET /pair/v1/capacity   every known model: setting, detection, usage
//	PUT /pair/v1/capacity   {"engine","model","contextTokens","maxSessions","replyTokens"?}
//	                        sets a model's pool, session count and reply
//	                        budget; send {"engine","model","clear":true} to
//	                        remove it
const capacityAPIPath = "/pair/v1/capacity"

func (f *facade) serveCapacityAPI(w http.ResponseWriter, r *http.Request) bool {
	if r.URL.Path != capacityAPIPath {
		return false
	}
	w.Header().Set(requestIDHeader, resolveRequestID(r.Header))
	switch r.Method {
	case http.MethodGet:
		writeSessionJSON(w, http.StatusOK, map[string]any{"models": f.host.capacityReport(r.Context())})
	case http.MethodPut:
		raw, err := io.ReadAll(io.LimitReader(r.Body, maxSessionAPIBody))
		if err != nil {
			writeSessionJSON(w, http.StatusBadRequest, map[string]string{"error": "read body: " + err.Error()})
			return true
		}
		row, err := f.host.setCapacity(r.Context(), raw)
		if err != nil {
			writeSessionJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return true
		}
		if row == nil {
			w.WriteHeader(http.StatusNoContent)
			return true
		}
		slog.Info("context capacity changed", "engine", row.Engine, "model", row.Model,
			"context_tokens", row.ContextTokens, "max_sessions", row.MaxSessions, "reply_tokens", row.ReplyTokens)
		writeSessionJSON(w, http.StatusOK, row)
	default:
		methodNotAllowed(w, "GET, PUT")
	}
	return true
}

// capacityReport lists every model this node knows something about: those the
// user configured, those a local engine reports loaded, and those with live
// sessions here. Detection failures are not fatal; the model is simply
// reported without a detection.
func (p *Proxy) capacityReport(ctx context.Context) []modelCapacity {
	settings := p.capacity.settingsSnapshot()
	usage := p.sessions.localUsageByModel()
	detected := make(map[capacityKey]detectedCapacity)
	for _, f := range p.enabledFacades() {
		base, ok := f.localBackendTarget()
		if !ok {
			continue
		}
		models, err := p.capacity.detect(ctx, f.profile.Name, base)
		if err != nil {
			slog.Debug("context detection failed", "engine", f.profile.Name, "err", err)
		}
		for model, d := range models {
			detected[capacityKey{Engine: f.profile.Name, Model: model}] = d
		}
	}

	keys := make(map[capacityKey]bool)
	for k := range settings {
		keys[k] = true
	}
	for k := range detected {
		keys[k] = true
	}
	for k := range usage {
		keys[k] = true
	}
	for k := range p.sessions.slotKeys() {
		keys[k] = true
	}
	rows := make([]modelCapacity, 0, len(keys))
	for k := range keys {
		var set *capacitySetting
		if v, ok := settings[k]; ok {
			set = &v
		}
		var det *detectedCapacity
		if v, ok := detected[k]; ok {
			det = &v
		}
		row := effective(k, set, det, usage[k])
		st := p.sessions.slotStatus(k)
		row.HeldSlots, row.Queued = st.Held, st.Queued
		rows = append(rows, row)
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Engine != rows[j].Engine {
			return rows[i].Engine < rows[j].Engine
		}
		return rows[i].Model < rows[j].Model
	})
	return rows
}
