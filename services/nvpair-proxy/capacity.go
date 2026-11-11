// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"nvpair-shared/engines"
)

// Context capacity is how many tokens of context a model on this node can hold
// across all of its sessions at once (the pool), how many sessions may share
// it, and so how much each session may use (the share, pool ÷ sessions).
//
// The user's setting is authoritative. Detection from the engine only fills in
// what the user has not set, because the engines report it unevenly: Ollama
// reports the per-request window but not its parallel slot count, LM Studio
// reports both for a loaded instance, and a llama.cpp router reports neither at
// the top level. A value PAIR cannot establish is reported as unknown rather
// than guessed.
//
// This file reports capacity and usage; it does not yet refuse or queue work.
const (
	// shareGranularity rounds a session's share down to a whole number of
	// these, so a share is never a fraction of an engine's allocation unit.
	shareGranularity = 1024
	// minUsefulShare is the smallest share worth running a session in. A
	// setting below it is accepted but reported with a warning.
	minUsefulShare        = 4096
	maxConfiguredSessions = 1024
	minConfiguredPool     = shareGranularity
	// detectionTTL caches an engine's answer so a busy status page does not
	// poll the engine on every request.
	detectionTTL     = 15 * time.Second
	detectionTimeout = 3 * time.Second
	maxDetectionBody = 4 << 20
	capacityFileName = "context-capacity.json"
)

type capacityKey struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
}

// capacitySetting is the user's choice for one model on this node.
type capacitySetting struct {
	ContextTokens int64 `json:"contextTokens"`
	MaxSessions   int   `json:"maxSessions"`
}

// detectedCapacity is what an engine reported for one loaded model. Zero means
// the engine did not say.
type detectedCapacity struct {
	// Window is the most context one request can use.
	Window int64
	// Slots is how many requests the engine serves at once.
	Slots int
	// Pool is the total context across slots, when the engine's allocation
	// model makes it knowable.
	Pool int64
	// Basis says how Pool was derived, for the person reading the report.
	Basis string
}

type capacityStore struct {
	mu       sync.Mutex
	settings map[capacityKey]capacitySetting
	// path persists settings; empty keeps them in memory only (tests).
	path string

	detectMu sync.Mutex
	detected map[string]detectionEntry // by engine
	// refreshing marks engines with a background detection in progress.
	refreshing map[string]bool
	client     *http.Client
	now        func() time.Time
}

type detectionEntry struct {
	at     time.Time
	models map[string]detectedCapacity
	err    error
}

func newCapacityStore() *capacityStore {
	return &capacityStore{
		settings: make(map[capacityKey]capacitySetting),
		detected: make(map[string]detectionEntry),
		client:   &http.Client{Timeout: detectionTimeout},
		now:      time.Now,
	}
}

func (c capacitySetting) validate() error {
	if c.ContextTokens < minConfiguredPool {
		return fmt.Errorf("contextTokens must be at least %d", minConfiguredPool)
	}
	if c.MaxSessions < 1 || c.MaxSessions > maxConfiguredSessions {
		return fmt.Errorf("maxSessions must be between 1 and %d", maxConfiguredSessions)
	}
	return nil
}

// sessionShare is a pool divided among sessions, rounded down to the
// allocation granularity.
func sessionShare(pool int64, sessions int) int64 {
	if pool <= 0 || sessions <= 0 {
		return 0
	}
	share := pool / int64(sessions)
	return share - share%shareGranularity
}

// load reads persisted settings. A missing file is an empty configuration; a
// corrupt one is reported and ignored rather than half-applied.
func (s *capacityStore) load() error {
	if s.path == "" {
		return nil
	}
	data, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var file struct {
		Models []struct {
			capacityKey
			capacitySetting
		} `json:"models"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		return fmt.Errorf("parse %s: %w", s.path, err)
	}
	settings := make(map[capacityKey]capacitySetting, len(file.Models))
	for _, m := range file.Models {
		if m.Engine == "" || m.Model == "" || m.validate() != nil {
			continue
		}
		settings[m.capacityKey] = m.capacitySetting
	}
	s.mu.Lock()
	s.settings = settings
	s.mu.Unlock()
	return nil
}

// saveLocked writes settings atomically.
func (s *capacityStore) saveLocked() error {
	if s.path == "" {
		return nil
	}
	type entry struct {
		capacityKey
		capacitySetting
	}
	out := struct {
		Models []entry `json:"models"`
	}{Models: make([]entry, 0, len(s.settings))}
	for k, v := range s.settings {
		out.Models = append(out.Models, entry{k, v})
	}
	sort.Slice(out.Models, func(i, j int) bool {
		if out.Models[i].Engine != out.Models[j].Engine {
			return out.Models[i].Engine < out.Models[j].Engine
		}
		return out.Models[i].Model < out.Models[j].Model
	})
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, s.path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// set stores, or with clear removes, the setting for one model.
func (s *capacityStore) set(k capacityKey, v capacitySetting, clear bool) error {
	if k.Engine == "" || k.Model == "" {
		return errors.New("engine and model are required")
	}
	if !clear {
		if err := v.validate(); err != nil {
			return err
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	prev, had := s.settings[k]
	if clear {
		delete(s.settings, k)
	} else {
		s.settings[k] = v
	}
	if err := s.saveLocked(); err != nil {
		if had {
			s.settings[k] = prev
		} else {
			delete(s.settings, k)
		}
		return fmt.Errorf("save capacity settings: %w", err)
	}
	return nil
}

func (s *capacityStore) settingsSnapshot() map[capacityKey]capacitySetting {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[capacityKey]capacitySetting, len(s.settings))
	for k, v := range s.settings {
		out[k] = v
	}
	return out
}

// detect returns what the local engine reports for its loaded models, cached
// for detectionTTL. base is the engine's own loopback address.
func (s *capacityStore) detect(ctx context.Context, engine string, base *url.URL) (map[string]detectedCapacity, error) {
	s.detectMu.Lock()
	defer s.detectMu.Unlock()
	now := s.now()
	if e, ok := s.detected[engine]; ok && now.Sub(e.at) < detectionTTL {
		return e.models, e.err
	}
	var (
		models map[string]detectedCapacity
		err    error
	)
	switch engine {
	case engines.NameOllama:
		models, err = s.detectOllama(ctx, base)
	case engines.NameLMStudio:
		models, err = s.detectLMStudio(ctx, base)
	case engines.NameLlamaCPP:
		models, err = s.detectLlamaCPP(ctx, base)
	default:
		err = fmt.Errorf("no context detection for engine %q", engine)
	}
	s.detected[engine] = detectionEntry{at: now, models: models, err: err}
	return models, err
}

func (s *capacityStore) getJSON(ctx context.Context, base *url.URL, path string, query url.Values, dst any) error {
	u := *base
	u.Path = path
	u.RawQuery = query.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: HTTP %d", path, resp.StatusCode)
	}
	return json.NewDecoder(io.LimitReader(resp.Body, maxDetectionBody)).Decode(dst)
}

// detectOllama reads loaded models from /api/ps. context_length is the
// per-request window (num_ctx). Ollama allocates that for each of its
// OLLAMA_NUM_PARALLEL slots, but does not report the slot count, so the pool
// stays unknown unless the user sets it.
func (s *capacityStore) detectOllama(ctx context.Context, base *url.URL) (map[string]detectedCapacity, error) {
	var ps struct {
		Models []struct {
			Name          string `json:"name"`
			Model         string `json:"model"`
			ContextLength int64  `json:"context_length"`
		} `json:"models"`
	}
	if err := s.getJSON(ctx, base, "/api/ps", nil, &ps); err != nil {
		return nil, err
	}
	out := make(map[string]detectedCapacity, len(ps.Models))
	for _, m := range ps.Models {
		name := m.Name
		if name == "" {
			name = m.Model
		}
		if name == "" || m.ContextLength <= 0 {
			continue
		}
		out[name] = detectedCapacity{Window: m.ContextLength}
	}
	return out, nil
}

// detectLMStudio reads loaded instances from /api/v1/models. LM Studio's
// default unified KV cache shares one context_length-sized pool among its
// parallel predictions, so the pool is the context length.
func (s *capacityStore) detectLMStudio(ctx context.Context, base *url.URL) (map[string]detectedCapacity, error) {
	var list struct {
		Models []struct {
			Key             string `json:"key"`
			LoadedInstances []struct {
				ID     string `json:"id"`
				Config struct {
					ContextLength int64 `json:"context_length"`
					Parallel      int   `json:"parallel"`
				} `json:"config"`
			} `json:"loaded_instances"`
		} `json:"models"`
	}
	if err := s.getJSON(ctx, base, "/api/v1/models", nil, &list); err != nil {
		return nil, err
	}
	out := make(map[string]detectedCapacity)
	for _, m := range list.Models {
		for _, inst := range m.LoadedInstances {
			if inst.Config.ContextLength <= 0 {
				continue
			}
			name := inst.ID
			if name == "" {
				name = m.Key
			}
			out[name] = detectedCapacity{
				Window: inst.Config.ContextLength,
				Slots:  inst.Config.Parallel,
				Pool:   inst.Config.ContextLength,
				Basis:  "unified KV cache: parallel requests share one context-length pool",
			}
		}
	}
	return out, nil
}

// detectLlamaCPP reads /props. A single-model server reports its per-slot
// n_ctx and total_slots; the pool is reported as their product, which is exact
// for a split cache and an upper bound for a unified one. A router reports no
// model of its own, so each loaded model is asked for separately, and a model
// that does not answer stays unknown.
func (s *capacityStore) detectLlamaCPP(ctx context.Context, base *url.URL) (map[string]detectedCapacity, error) {
	type props struct {
		Role                      string `json:"role"`
		ModelAlias                string `json:"model_alias"`
		TotalSlots                int    `json:"total_slots"`
		DefaultGenerationSettings struct {
			NCtx int64 `json:"n_ctx"`
		} `json:"default_generation_settings"`
	}
	toCapacity := func(p props) (detectedCapacity, bool) {
		if p.DefaultGenerationSettings.NCtx <= 0 {
			return detectedCapacity{}, false
		}
		d := detectedCapacity{Window: p.DefaultGenerationSettings.NCtx, Slots: p.TotalSlots}
		if p.TotalSlots > 0 {
			d.Pool = d.Window * int64(p.TotalSlots)
			d.Basis = "n_ctx per slot × total_slots; an upper bound if the KV cache is unified"
		}
		return d, true
	}
	var top props
	if err := s.getJSON(ctx, base, "/props", nil, &top); err != nil {
		return nil, err
	}
	out := make(map[string]detectedCapacity)
	if top.Role != "router" {
		if d, ok := toCapacity(top); ok && top.ModelAlias != "" {
			out[top.ModelAlias] = d
		}
		return out, nil
	}
	var models struct {
		Data []struct {
			ID     string `json:"id"`
			Status struct {
				Value string `json:"value"`
			} `json:"status"`
		} `json:"data"`
	}
	if err := s.getJSON(ctx, base, "/models", nil, &models); err != nil {
		return out, err
	}
	for _, m := range models.Data {
		if m.Status.Value != "loaded" {
			continue
		}
		var p props
		if err := s.getJSON(ctx, base, "/props", url.Values{"model": {m.ID}}, &p); err != nil {
			continue
		}
		if d, ok := toCapacity(p); ok {
			out[m.ID] = d
		}
	}
	return out, nil
}

// modelCapacity is one row of the capacity report.
type modelCapacity struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
	// Configured is the user's setting, if any.
	Configured *capacitySetting `json:"configured"`
	// Detected is what the engine reported, if it is loaded and said.
	Detected *detectedJSON `json:"detected"`
	// The effective values: the setting where there is one, else detection.
	// Zero means unknown.
	ContextTokens int64  `json:"contextTokens"`
	MaxSessions   int    `json:"maxSessions"`
	SessionShare  int64  `json:"sessionContextTokens"`
	Source        string `json:"source"`
	// Live sessions this node serves for the model, and the context they hold.
	ActiveSessions int   `json:"activeSessions"`
	UsedTokens     int64 `json:"usedContextTokens"`
	// HeldSlots is how many sessions hold one of the model's context slots
	// here; Queued is how many requests are waiting for one.
	HeldSlots int    `json:"heldSlots"`
	Queued    int    `json:"queued"`
	Warning   string `json:"warning,omitempty"`
}

type detectedJSON struct {
	WindowTokens int64  `json:"windowTokens,omitempty"`
	Slots        int    `json:"slots,omitempty"`
	PoolTokens   int64  `json:"poolTokens,omitempty"`
	Basis        string `json:"basis,omitempty"`
}

// effective combines a setting and a detection into the values in force.
func effective(k capacityKey, set *capacitySetting, det *detectedCapacity, use localUsage) modelCapacity {
	row := modelCapacity{Engine: k.Engine, Model: k.Model, Configured: set, Source: "unknown",
		ActiveSessions: use.Sessions, UsedTokens: use.Tokens}
	if det != nil {
		row.Detected = &detectedJSON{WindowTokens: det.Window, Slots: det.Slots, PoolTokens: det.Pool, Basis: det.Basis}
	}
	switch {
	case set != nil:
		row.ContextTokens, row.MaxSessions, row.Source = set.ContextTokens, set.MaxSessions, "configured"
	case det != nil && det.Pool > 0 && det.Slots > 0:
		row.ContextTokens, row.MaxSessions, row.Source = det.Pool, det.Slots, "detected"
	case det != nil && det.Window > 0:
		// The window alone bounds one session but says nothing about how many
		// fit, so only the share is known.
		row.SessionShare, row.Source = det.Window, "detected"
		row.Warning = "the engine reports its per-request window but not how many requests share the pool; set contextTokens and maxSessions to control it"
		return row
	}
	row.SessionShare = sessionShare(row.ContextTokens, row.MaxSessions)
	if row.Source == "configured" && det != nil && det.Window > 0 && row.SessionShare > det.Window {
		row.Warning = fmt.Sprintf("each session's share (%d) exceeds the engine's per-request window (%d); the engine will truncate before PAIR's limit", row.SessionShare, det.Window)
	}
	if row.SessionShare > 0 && row.SessionShare < minUsefulShare {
		row.Warning = fmt.Sprintf("each session gets %d tokens, below the %d useful minimum; allow fewer sessions or a larger pool", row.SessionShare, minUsefulShare)
	}
	return row
}

// cachedDetection returns the last detection for an engine without waiting
// for the engine. When it is stale or missing, a refresh starts in the
// background, so the request path never blocks on an engine query.
func (s *capacityStore) cachedDetection(engine string, base *url.URL) map[string]detectedCapacity {
	s.detectMu.Lock()
	e, ok := s.detected[engine]
	stale := !ok || s.now().Sub(e.at) >= detectionTTL
	refreshing := s.refreshing[engine]
	if stale && !refreshing && base != nil {
		if s.refreshing == nil {
			s.refreshing = make(map[string]bool)
		}
		s.refreshing[engine] = true
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), detectionTimeout)
			defer cancel()
			_, _ = s.detect(ctx, engine, base)
			s.detectMu.Lock()
			delete(s.refreshing, engine)
			s.detectMu.Unlock()
		}()
	}
	s.detectMu.Unlock()
	return e.models
}

// limitFor returns the effective capacity for one model on this node, from
// the setting and the cached detection.
func (p *Proxy) limitFor(f *facade, k capacityKey) modelCapacity {
	var base *url.URL
	if b, ok := f.localBackendTarget(); ok {
		base = b
	}
	var set *capacitySetting
	if v, ok := p.capacity.settingsSnapshot()[k]; ok {
		set = &v
	}
	var det *detectedCapacity
	if v, ok := p.capacity.cachedDetection(k.Engine, base)[k.Model]; ok {
		det = &v
	}
	return effective(k, set, det, localUsage{})
}
