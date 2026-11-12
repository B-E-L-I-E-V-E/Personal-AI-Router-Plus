// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// The node keeps the capacity plans the user chose for its models, so any
// cluster member can read and change them (later phases) and so the desktop's
// idle check survives restarts. A plan is "pending" until its engine has been
// reloaded with it; applying is the desktop's job, because only the broker can
// change an engine's launch settings.

const plannerFileName = "capacity-plans.json"

// defaultReplyTokens is the reply budget a plan keeps when none is chosen:
// room for a long answer, and for a thinking model's reasoning before it.
const defaultReplyTokens = 8192

// replyBudget is the plan's reply budget, the default when unset, never more
// than half its window.
func (p savedPlan) replyBudget() int64 {
	r := p.ReplyTokens
	if r == 0 {
		r = defaultReplyTokens
	}
	if half := p.WindowTokens / 2; r > half {
		r = half
	}
	return r
}

type savedPlan struct {
	Engine         string  `json:"engine"`
	Model          string  `json:"model"`
	WindowTokens   int64   `json:"windowTokens"`
	Sessions       int     `json:"sessions"`
	CacheType      string  `json:"cacheType"`
	ReservePercent float64 `json:"reservePercent,omitempty"`
	// OffloadExperts keeps a mixture-of-experts model's expert weights in
	// system memory (llama.cpp --cpu-moe).
	OffloadExperts bool `json:"offloadExperts,omitempty"`
	// ReplyTokens is kept from each session's window for replies; zero means
	// the default (defaultReplyTokens, at most half the window).
	ReplyTokens int64 `json:"replyTokens,omitempty"`
	// Applied is set once the engine runs with this plan; changing the plan
	// clears it.
	Applied   bool  `json:"applied"`
	AppliedAt int64 `json:"appliedAt,omitempty"`
	SavedAt   int64 `json:"savedAt"`
}

type plannerState struct {
	// AutoReload lets PAIR apply a pending plan by itself once the node has
	// no sessions and no requests. When off, only the user's Apply does.
	AutoReload bool        `json:"autoReload"`
	Plans      []savedPlan `json:"plans"`
}

type plannerStore struct {
	mu    sync.Mutex
	path  string
	state plannerState
}

func newPlannerStore() *plannerStore {
	return &plannerStore{state: plannerState{AutoReload: true}}
}

func (s *plannerStore) load() error {
	data, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var st plannerState
	st.AutoReload = true
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	kept := st.Plans[:0]
	for _, p := range st.Plans {
		if p.validate() == nil {
			kept = append(kept, p)
		}
	}
	st.Plans = kept
	s.mu.Lock()
	s.state = st
	s.mu.Unlock()
	return nil
}

func (s *plannerStore) saveLocked() error {
	if s.path == "" {
		return nil
	}
	data, err := json.MarshalIndent(s.state, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

func (p savedPlan) validate() error {
	if _, ok := profileFor(p.Engine); !ok {
		return fmt.Errorf("unknown engine; use one of %s", engineNames())
	}
	if p.Model == "" || len(p.Model) > 512 {
		return errors.New("model is required")
	}
	if p.WindowTokens < minPlannedWindowTokens || p.WindowTokens > 1<<24 {
		return fmt.Errorf("windowTokens must be between %d and %d", minPlannedWindowTokens, 1<<24)
	}
	if p.Sessions < 1 || p.Sessions > maxPlannedSessions {
		return fmt.Errorf("sessions must be between 1 and %d", maxPlannedSessions)
	}
	if _, ok := cacheTypeFactor[p.CacheType]; !ok {
		return errors.New("cacheType must be f16, q8_0 or q4_0")
	}
	if p.ReplyTokens < 0 || (p.ReplyTokens > 0 && p.ReplyTokens >= p.WindowTokens) {
		return errors.New("replyTokens must be less than windowTokens")
	}
	if p.OffloadExperts && p.Engine == "ollama" {
		return errors.New("Ollama cannot keep expert weights in system memory")
	}
	return nil
}

func (s *plannerStore) snapshot() plannerState {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := plannerState{AutoReload: s.state.AutoReload, Plans: append([]savedPlan{}, s.state.Plans...)}
	sort.Slice(out.Plans, func(i, j int) bool {
		if out.Plans[i].Engine != out.Plans[j].Engine {
			return out.Plans[i].Engine < out.Plans[j].Engine
		}
		return out.Plans[i].Model < out.Plans[j].Model
	})
	return out
}

func (s *plannerStore) find(engine, model string) (savedPlan, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, p := range s.state.Plans {
		if p.Engine == engine && p.Model == model {
			return p, true
		}
	}
	return savedPlan{}, false
}

// put saves a plan. Ollama's window and slot count are engine-wide, so one
// Ollama plan replaces any other.
func (s *plannerStore) put(plan savedPlan) error {
	if plan.CacheType == "" {
		plan.CacheType = "f16"
	}
	if err := plan.validate(); err != nil {
		return err
	}
	plan.SavedAt = time.Now().UnixMilli()
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := s.state.Plans[:0]
	for _, p := range s.state.Plans {
		same := p.Engine == plan.Engine && p.Model == plan.Model
		if same || (plan.Engine == "ollama" && p.Engine == "ollama") {
			if same && p.sameSettings(plan.WindowTokens, plan.Sessions, plan.CacheType, plan.OffloadExperts) {
				plan.Applied, plan.AppliedAt = p.Applied, p.AppliedAt
			}
			continue
		}
		kept = append(kept, p)
	}
	s.state.Plans = append(kept, plan)
	return s.saveLocked()
}

func (s *plannerStore) remove(engine, model string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := s.state.Plans[:0]
	for _, p := range s.state.Plans {
		if !(p.Engine == engine && p.Model == model) {
			kept = append(kept, p)
		}
	}
	s.state.Plans = kept
	return s.saveLocked()
}

func (s *plannerStore) setAutoReload(on bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.AutoReload = on
	return s.saveLocked()
}

// markApplied records that the engine now runs plan's numbers. A plan changed
// since the apply started stays pending.
func (p savedPlan) sameSettings(window int64, sessions int, cache string, offload bool) bool {
	return p.WindowTokens == window && p.Sessions == sessions && p.CacheType == cache && p.OffloadExperts == offload
}

func (s *plannerStore) markApplied(engine, model string, window int64, sessions int, cache string, offload bool) (savedPlan, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, p := range s.state.Plans {
		if p.Engine == engine && p.Model == model {
			if !p.sameSettings(window, sessions, cache, offload) {
				return p, false, nil
			}
			s.state.Plans[i].Applied = true
			s.state.Plans[i].AppliedAt = time.Now().UnixMilli()
			return s.state.Plans[i], true, s.saveLocked()
		}
	}
	return savedPlan{}, false, errors.New("no saved plan for that model")
}

// nodeActivity is what the idle check needs: anything here blocks a reload.
type nodeActivity struct {
	Sessions       int   `json:"sessions"`
	ActiveRequests int64 `json:"activeRequests"`
	Queued         int   `json:"queued"`
	Idle           bool  `json:"idle"`
}

func (p *Proxy) nodeActivity() nodeActivity {
	a := nodeActivity{ActiveRequests: p.activeRequests.Load()}
	a.Sessions, a.Queued = p.sessions.activityCounts()
	a.Idle = a.Sessions == 0 && a.ActiveRequests == 0 && a.Queued == 0
	return a
}

// plannerResult is the capacity/planner result.
func (p *Proxy) plannerResult() map[string]any {
	st := p.planner.snapshot()
	return map[string]any{"autoReload": st.AutoReload, "plans": st.Plans, "activity": p.nodeActivity()}
}

var specReader = &specClient{
	client:  &http.Client{Timeout: 10 * time.Second},
	homeDir: os.UserHomeDir,
}

// planCapacity runs capacity/plan: describe the model on its local engine,
// then size it for the GPUs the caller reports.
func (p *Proxy) planCapacity(ctx context.Context, raw json.RawMessage) (*planResult, error) {
	var req planRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, errors.New(`expected {"engine","model","gpus":[{"vramBytes"}],"windowTokens"|"sessions"?}`)
	}
	if _, ok := profileFor(req.Engine); !ok {
		return nil, fmt.Errorf("unknown engine; use one of %s", engineNames())
	}
	if req.Model == "" {
		return nil, errors.New("model is required")
	}
	spec := modelSpec{Engine: req.Engine, Model: req.Model}
	var f *facade
	for _, candidate := range p.enabledFacades() {
		if candidate.profile.Name == req.Engine {
			f = candidate
		}
	}
	if f != nil {
		if base, ok := f.localBackendTarget(); ok {
			parallel := 0
			if saved, ok := p.planner.find(req.Engine, req.Model); ok && saved.Applied {
				parallel = saved.Sessions
			}
			described, err := specReader.describe(ctx, req.Engine, base, req.Model, parallel)
			if err != nil {
				described.Notes = append(described.Notes, "could not read the model from the engine: "+err.Error())
			}
			spec = described
		} else {
			spec.Notes = append(spec.Notes, "this engine is not running on this node")
		}
	} else {
		spec.Notes = append(spec.Notes, "this engine is not running on this node")
	}
	applyWebFacts(&spec, p.webFacts.result(webFactsKey{Engine: req.Engine, Model: req.Model}))
	res, err := computePlan(req, spec)
	if err != nil {
		res.Error = err.Error()
	}
	return &res, nil
}

type plannerSetParams struct {
	AutoReload *bool      `json:"autoReload,omitempty"`
	Save       *savedPlan `json:"save,omitempty"`
	Remove     *struct {
		Engine string `json:"engine"`
		Model  string `json:"model"`
	} `json:"remove,omitempty"`
	// Applied reports that the engine was reloaded with these numbers; the
	// proxy then sets the model's context capacity to match.
	Applied *savedPlan `json:"applied,omitempty"`
}

func (p *Proxy) setPlanner(ctx context.Context, raw json.RawMessage) (map[string]any, error) {
	var params plannerSetParams
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, errors.New(`expected {"autoReload"?,"save"?,"remove"?,"applied"?}`)
	}
	if params.AutoReload != nil {
		if err := p.planner.setAutoReload(*params.AutoReload); err != nil {
			return nil, err
		}
	}
	if params.Save != nil {
		if err := p.planner.put(*params.Save); err != nil {
			return nil, err
		}
		// The reply budget is PAIR's own limit, not the engine's, so a plan
		// still applied after the save takes a new one at once.
		if plan, ok := p.planner.find(params.Save.Engine, params.Save.Model); ok && plan.Applied {
			if err := p.applyPlanCapacity(plan); err != nil {
				return nil, err
			}
		}
	}
	if params.Remove != nil {
		if err := p.planner.remove(params.Remove.Engine, params.Remove.Model); err != nil {
			return nil, err
		}
	}
	if a := params.Applied; a != nil {
		cache := a.CacheType
		if cache == "" {
			cache = "f16"
		}
		plan, ok, err := p.planner.markApplied(a.Engine, a.Model, a.WindowTokens, a.Sessions, cache, a.OffloadExperts)
		if err != nil {
			return nil, err
		}
		if ok {
			if err := p.applyPlanCapacity(plan); err != nil {
				return nil, err
			}
		}
	}
	return p.plannerResult(), nil
}

// applyPlanCapacity sets the model's context capacity to an applied plan's:
// window × sessions, that many sessions, and the plan's reply budget.
func (p *Proxy) applyPlanCapacity(plan savedPlan) error {
	k := capacityKey{Engine: plan.Engine, Model: plan.Model}
	return p.capacity.set(k, capacitySetting{
		ContextTokens: plan.WindowTokens * int64(plan.Sessions),
		MaxSessions:   plan.Sessions,
		ReplyTokens:   plan.replyBudget(),
	}, false)
}
