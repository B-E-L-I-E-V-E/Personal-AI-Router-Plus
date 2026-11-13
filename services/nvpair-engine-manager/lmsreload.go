// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// lmsreload.go reloads one LM Studio model with a chosen context length and
// number of parallel predictions. LM Studio sets both per model load, not in
// the server's launch settings, so applying a capacity plan means unloading
// the model and loading it again with the plan's numbers. It drives the lms
// CLI, which LM Studio's other manifest actions (load, unload, pull) use too.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

const (
	// reloadOutcomeReloaded: the model now runs with the requested numbers.
	reloadOutcomeReloaded = "reloaded"
	// reloadOutcomeUnsupported: PAIR cannot drive this node's LM Studio (no
	// lms, an lms without the needed load options, or a peer without this
	// method). Nothing was changed; the user sets the numbers in LM Studio.
	reloadOutcomeUnsupported = "unsupported"

	maxReloadParallel      = 64
	maxReloadContextLength = 1 << 24
)

// modelReloadRequest is the body of engine:reload-model and of the ec
// /v1/models/reload endpoint.
type modelReloadRequest struct {
	Engine        string `json:"engine"`
	Model         string `json:"model"`
	ContextLength int64  `json:"contextLength"`
	Parallel      int    `json:"parallel"`
}

// modelReloadResult reports what the reload did. Identifier is the name the
// loaded instance answers to, which stays the one PAIR routes by.
type modelReloadResult struct {
	Outcome       string `json:"outcome"`
	Reason        string `json:"reason,omitempty"`
	Model         string `json:"model"`
	Identifier    string `json:"identifier,omitempty"`
	ContextLength int64  `json:"contextLength,omitempty"`
	Parallel      int    `json:"parallel,omitempty"`
}

func (r modelReloadRequest) validate() error {
	if r.Engine != "lmstudio" {
		return fmt.Errorf("reload-model sets per-load context only for lmstudio, not %q", r.Engine)
	}
	if r.Model == "" {
		return errors.New("model is required")
	}
	if r.ContextLength < 1 || r.ContextLength > maxReloadContextLength {
		return fmt.Errorf("contextLength must be between 1 and %d", maxReloadContextLength)
	}
	if r.Parallel < 1 || r.Parallel > maxReloadParallel {
		return fmt.Errorf("parallel must be between 1 and %d", maxReloadParallel)
	}
	return nil
}

// lmsLoadedInstance is one row of `lms ps --json`.
type lmsLoadedInstance struct {
	ModelKey      string `json:"modelKey"`
	Identifier    string `json:"identifier"`
	ContextLength int64  `json:"contextLength"`
	Parallel      int    `json:"parallel"`
}

// commandRunner runs argv and returns its stdout (Executor.runCommandOutput).
type commandRunner func(ctx context.Context, argv []string) (string, error)

// ModelReload reloads an LM Studio model with req's context length and
// parallel predictions.
func (e *Executor) ModelReload(ctx context.Context, req modelReloadRequest) (modelReloadResult, error) {
	if err := req.validate(); err != nil {
		return modelReloadResult{}, err
	}
	st, err := e.state(req.Engine)
	if err != nil {
		return modelReloadResult{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, e.actionTimeout)
	defer cancel()
	return lmsReload(ctx, e.runCommandOutput, expandPath(strings.TrimSpace(st.plat.Runtime.CLI)), req)
}

func lmsReload(ctx context.Context, run commandRunner, cli string, req modelReloadRequest) (modelReloadResult, error) {
	unsupported := func(reason string) (modelReloadResult, error) {
		return modelReloadResult{Outcome: reloadOutcomeUnsupported, Reason: reason, Model: req.Model}, nil
	}
	if cli == "" {
		return unsupported("this node's LM Studio has no lms command configured")
	}
	// Check the options before unloading anything, so an older lms leaves
	// the model as it was.
	help, err := run(ctx, []string{cli, "load", "--help"})
	if err != nil {
		return unsupported("lms could not run: " + err.Error())
	}
	for _, option := range []string{"--context-length", "--parallel", "--identifier"} {
		if !strings.Contains(help, option) {
			return unsupported("this LM Studio's lms load has no " + option + " option; update LM Studio")
		}
	}

	loaded, err := lmsPS(ctx, run, cli)
	if err != nil {
		return modelReloadResult{}, err
	}
	matches := matchingInstances(loaded, req.Model)
	// The reloaded instance answers to the name PAIR plans and routes by.
	key, identifier := req.Model, req.Model
	if len(matches) > 0 && matches[0].ModelKey != "" {
		key = matches[0].ModelKey
	}
	done := modelReloadResult{
		Outcome:       reloadOutcomeReloaded,
		Model:         req.Model,
		Identifier:    identifier,
		ContextLength: req.ContextLength,
		Parallel:      req.Parallel,
	}
	if len(matches) == 1 && matches[0].Identifier == identifier && matches[0].ContextLength == req.ContextLength && matches[0].Parallel == req.Parallel {
		return done, nil
	}

	for _, m := range matches {
		if _, err := run(ctx, []string{cli, "unload", m.Identifier}); err != nil {
			return modelReloadResult{}, fmt.Errorf("unload %s: %w", m.Identifier, err)
		}
	}
	if _, err := run(ctx, lmsLoadArgs(cli, key, identifier, req.ContextLength, req.Parallel)); err != nil {
		loadErr := fmt.Errorf("load %s with context length %d and %d parallel predictions: %w",
			key, req.ContextLength, req.Parallel, err)
		if len(matches) == 0 {
			return modelReloadResult{}, loadErr
		}
		prev := matches[0]
		if _, restoreErr := run(ctx, lmsLoadArgs(cli, key, identifier, prev.ContextLength, prev.Parallel)); restoreErr != nil {
			return modelReloadResult{}, fmt.Errorf("%w; loading it again with its previous settings also failed: %v", loadErr, restoreErr)
		}
		return modelReloadResult{}, fmt.Errorf("%w; it was loaded again with its previous context length %d and %d parallel predictions",
			loadErr, prev.ContextLength, prev.Parallel)
	}

	after, err := lmsPS(ctx, run, cli)
	if err != nil {
		return modelReloadResult{}, err
	}
	for _, m := range after {
		if m.Identifier != identifier {
			continue
		}
		if m.ContextLength != req.ContextLength || m.Parallel != req.Parallel {
			return modelReloadResult{}, fmt.Errorf("LM Studio loaded %s with context length %d and %d parallel predictions, not %d and %d",
				identifier, m.ContextLength, m.Parallel, req.ContextLength, req.Parallel)
		}
		return done, nil
	}
	return modelReloadResult{}, fmt.Errorf("LM Studio does not list %s as loaded after loading it", identifier)
}

func lmsLoadArgs(cli, key, identifier string, contextLength int64, parallel int) []string {
	return []string{cli, "load", key,
		"--context-length", strconv.FormatInt(contextLength, 10),
		"--parallel", strconv.Itoa(parallel),
		"--identifier", identifier,
		"--yes"}
}

func lmsPS(ctx context.Context, run commandRunner, cli string) ([]lmsLoadedInstance, error) {
	out, err := run(ctx, []string{cli, "ps", "--json"})
	if err != nil {
		return nil, fmt.Errorf("lms ps --json: %w", err)
	}
	out = strings.TrimSpace(out)
	if out == "" {
		return nil, nil
	}
	var loaded []lmsLoadedInstance
	if err := json.Unmarshal([]byte(out), &loaded); err != nil {
		return nil, fmt.Errorf("lms ps --json: parse: %w", err)
	}
	return loaded, nil
}

// matchingInstances returns the loaded instances of model: the instance
// named model (the name PAIR routes by), else every instance of the model key
// model.
func matchingInstances(loaded []lmsLoadedInstance, model string) []lmsLoadedInstance {
	var byID, byKey []lmsLoadedInstance
	for _, m := range loaded {
		if m.Identifier == model {
			byID = append(byID, m)
		} else if m.ModelKey == model {
			byKey = append(byKey, m)
		}
	}
	if len(byID) > 0 {
		return byID
	}
	return byKey
}
