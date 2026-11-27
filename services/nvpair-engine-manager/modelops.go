// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
)

// modelActionRequest is the shared body for ec model load/unload/delete endpoints.
type modelActionRequest struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
	// Load, on a load request, is the context window and parallel count to
	// load the model with. Only LM Studio takes it: it fixes both per load.
	Load *loadSettings `json:"load,omitempty"`
}

// loadSettings are the per-load settings an engine fixes when a model loads.
type loadSettings struct {
	ContextLength int64 `json:"contextLength"`
	Parallel      int   `json:"parallel"`
}

func (l loadSettings) validate() error {
	if l.ContextLength < 1 || l.Parallel < 1 {
		return fmt.Errorf("load settings need a context length and a parallel count of at least 1")
	}
	return nil
}

// ModelLoad warms a model into engine memory using each engine's manifest action.
//
// With settings, the model loads with that context window and parallel count
// instead of the engine's per-model default. The caller unloads a loaded copy
// first: LM Studio would otherwise keep the old instance and add a second.
func (e *Executor) ModelLoad(ctx context.Context, engine, model string, settings *loadSettings) (json.RawMessage, error) {
	action, params, err := modelActionWire(engine, "load", model)
	if settings != nil {
		action, params, err = loadWithSettingsWire(engine, model, *settings)
	}
	if err != nil {
		return nil, err
	}
	return e.Action(ctx, engine, action, params)
}

func loadWithSettingsWire(engine, model string, settings loadSettings) (string, json.RawMessage, error) {
	if engine != "lmstudio" {
		return "", nil, fmt.Errorf("%s does not take load settings; only LM Studio fixes them per model load", engine)
	}
	if model == "" {
		return "", nil, fmt.Errorf("model is required")
	}
	if err := settings.validate(); err != nil {
		return "", nil, err
	}
	return marshalModelAction("load_model_with_settings", map[string]any{
		"model":          model,
		"context_length": settings.ContextLength,
		"parallel":       settings.Parallel,
	})
}

// ModelUnload frees a model from engine memory.
func (e *Executor) ModelUnload(ctx context.Context, engine, model string) (json.RawMessage, error) {
	action, params, err := modelActionWire(engine, "unload", model)
	if err != nil {
		return nil, err
	}
	return e.Action(ctx, engine, action, params)
}

// ModelDelete removes a downloaded model from disk when the manifest exposes it.
func (e *Executor) ModelDelete(ctx context.Context, engine, model string) (json.RawMessage, error) {
	action, params, err := modelActionWire(engine, "delete", model)
	if err != nil {
		return nil, err
	}
	return e.Action(ctx, engine, action, params)
}

func modelActionWire(engine, op, model string) (string, json.RawMessage, error) {
	if model == "" {
		return "", nil, fmt.Errorf("model is required")
	}
	switch op {
	case "load":
		switch engine {
		case "ollama":
			return marshalModelAction("run_model", map[string]any{"model": model, "stream": false})
		default:
			return marshalModelAction("load_model", map[string]string{"model": model})
		}
	case "unload":
		switch engine {
		case "ollama":
			return marshalModelAction("unload_model", map[string]any{"model": model, "keep_alive": 0})
		default:
			return marshalModelAction("unload_model", map[string]string{"model": model})
		}
	case "delete":
		switch engine {
		case "ollama":
			return marshalModelAction("delete_model", map[string]string{"name": model})
		default:
			return marshalModelAction("delete_model", map[string]string{"model": model})
		}
	default:
		return "", nil, fmt.Errorf("unknown model op %q", op)
	}
}

func marshalModelAction(action string, params any) (string, json.RawMessage, error) {
	b, err := json.Marshal(params)
	if err != nil {
		return "", nil, err
	}
	return action, b, nil
}
