// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// The proxy owns its session policy and context capacity settings and saves
// them itself, beside its port files, so the desktop and terminal interface
// only send commands and display what the proxy reports.
const sessionPolicyFileName = "session-policy.json"

// sessionPolicyFile is the saved form of the session policy, in seconds.
type sessionPolicyFile struct {
	IdleTimeoutSeconds  int `json:"idleTimeoutSeconds"`
	MaxLifetimeSeconds  int `json:"maxLifetimeSeconds"`
	MaxSessions         int `json:"maxSessions"`
	QueueTimeoutSeconds int `json:"queueTimeoutSeconds"`
	MaxQueued           int `json:"maxQueued"`
}

func policyToFile(c sessionConfig) sessionPolicyFile {
	return sessionPolicyFile{
		IdleTimeoutSeconds:  int(c.IdleTimeout / time.Second),
		MaxLifetimeSeconds:  int(c.MaxLifetime / time.Second),
		MaxSessions:         c.MaxSessions,
		QueueTimeoutSeconds: int(c.QueueTimeout / time.Second),
		MaxQueued:           c.MaxQueued,
	}
}

func (f sessionPolicyFile) config() sessionConfig {
	return sessionConfig{
		IdleTimeout:  time.Duration(f.IdleTimeoutSeconds) * time.Second,
		MaxLifetime:  time.Duration(f.MaxLifetimeSeconds) * time.Second,
		MaxSessions:  f.MaxSessions,
		QueueTimeout: time.Duration(f.QueueTimeoutSeconds) * time.Second,
		MaxQueued:    f.MaxQueued,
	}
}

// loadSessionPolicy reads a saved policy. A missing file reports false; an
// unreadable or invalid one is an error the caller logs and ignores.
func loadSessionPolicy(path string) (sessionConfig, bool, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return sessionConfig{}, false, nil
	}
	if err != nil {
		return sessionConfig{}, false, err
	}
	var f sessionPolicyFile
	if err := json.Unmarshal(data, &f); err != nil {
		return sessionConfig{}, false, fmt.Errorf("parse %s: %w", path, err)
	}
	cfg := f.config()
	if err := cfg.validate(); err != nil {
		return sessionConfig{}, false, fmt.Errorf("%s: %w", path, err)
	}
	return cfg, true, nil
}

// saveSessionPolicy writes the policy atomically.
func saveSessionPolicy(path string, c sessionConfig) error {
	data, err := json.MarshalIndent(policyToFile(c), "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// configureSessions applies a sessions/configure payload and, when it changed
// anything, saves the result. It returns the policy in force.
func (p *Proxy) configureSessions(raw json.RawMessage) (sessionConfig, error) {
	prev := p.sessions.config()
	cfg, err := applySessionConfigParams(prev, raw)
	if err != nil {
		return prev, err
	}
	if cfg == prev {
		return cfg, nil
	}
	if p.sessionPolicyPath != "" {
		if err := saveSessionPolicy(p.sessionPolicyPath, cfg); err != nil {
			return prev, fmt.Errorf("save session policy: %w", err)
		}
	}
	if err := p.sessions.configure(cfg); err != nil {
		return prev, err
	}
	return cfg, nil
}

// sessionsListResult is the sessions/list result: this node's own sessions.
func (p *Proxy) sessionsListResult(raw json.RawMessage) (map[string]any, error) {
	var params struct {
		AgentID string `json:"agentId"`
	}
	if len(raw) > 0 && string(raw) != "null" {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, errors.New(`expected {"agentId"?}`)
		}
	}
	views := p.sessions.list(params.AgentID)
	out := make([]sessionJSON, len(views))
	for i, v := range views {
		out[i] = p.sessionJSON(v)
	}
	return map[string]any{"sessions": out}, nil
}

// capacitySetParams is the capacity/set payload.
type capacitySetParams struct {
	Engine        string `json:"engine"`
	Model         string `json:"model"`
	ContextTokens int64  `json:"contextTokens"`
	MaxSessions   int    `json:"maxSessions"`
	ReplyTokens   int64  `json:"replyTokens"`
	Clear         bool   `json:"clear"`
}

// setCapacity applies a capacity/set payload and returns the model's row, or
// nil when clearing left nothing known about it.
func (p *Proxy) setCapacity(ctx context.Context, raw json.RawMessage) (*modelCapacity, error) {
	var params capacitySetParams
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, errors.New(`expected {"engine","model","contextTokens","maxSessions","replyTokens"?,"clear"?}`)
	}
	if _, ok := profileFor(params.Engine); !ok {
		return nil, fmt.Errorf("unknown engine; use one of %s", engineNames())
	}
	k := capacityKey{Engine: params.Engine, Model: params.Model}
	if err := p.capacity.set(k, capacitySetting{ContextTokens: params.ContextTokens, MaxSessions: params.MaxSessions, ReplyTokens: params.ReplyTokens}, params.Clear); err != nil {
		return nil, err
	}
	for _, row := range p.capacityReport(ctx) {
		if row.Engine == k.Engine && row.Model == k.Model {
			return &row, nil
		}
	}
	return nil, nil
}
