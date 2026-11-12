// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"strconv"
	"strings"
)

// startValue is one setting the engine or the model reports, for the planner
// to start from before the user chooses. Source says who reported it:
// "engine" for what the running engine uses for this model now, "model" for
// the model's own default parameter.
type startValue struct {
	Value  int64  `json:"value"`
	Source string `json:"source"`
}

// modelDefaults are the starting values the engine and the model report. A
// value nothing reports is left out; nothing here is guessed.
type modelDefaults struct {
	// Window is each session's context window.
	Window *startValue `json:"window,omitempty"`
	// Sessions is how many requests the engine runs at once (its slots).
	Sessions *startValue `json:"sessions,omitempty"`
	// Reply is the model's default reply length.
	Reply *startValue `json:"reply,omitempty"`
}

// setDefault records v from source unless a value is already known: callers
// set the value in force first.
func (s *modelSpec) setDefault(field func(*modelDefaults) **startValue, v int64, source string) {
	if v <= 0 {
		return
	}
	if s.Defaults == nil {
		s.Defaults = &modelDefaults{}
	}
	if p := field(s.Defaults); *p == nil {
		*p = &startValue{Value: v, Source: source}
	}
}

func defaultWindow(d *modelDefaults) **startValue   { return &d.Window }
func defaultSessions(d *modelDefaults) **startValue { return &d.Sessions }
func defaultReply(d *modelDefaults) **startValue    { return &d.Reply }

// ollamaParameters reads the numeric parameters of Ollama's /api/show
// "parameters" text: one "name value" pair per line. Repeated names (stop)
// and non-numeric values are skipped.
func ollamaParameters(text string) map[string]int64 {
	out := map[string]int64{}
	for _, line := range strings.Split(text, "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		if n, err := strconv.ParseInt(fields[1], 10, 64); err == nil {
			out[fields[0]] = n
		}
	}
	return out
}

// llamaCPPProps is the part of llama.cpp's /props the planner reads.
type llamaCPPProps struct {
	ModelPath  string `json:"model_path"`
	TotalSlots int    `json:"total_slots"`
	// Older servers report the window at the top level.
	NCtx     int64 `json:"n_ctx"`
	Settings struct {
		NCtx     int64 `json:"n_ctx"`
		NPredict int64 `json:"n_predict"`
		Params   struct {
			NPredict int64 `json:"n_predict"`
		} `json:"params"`
	} `json:"default_generation_settings"`
}

// applyLlamaCPPDefaults: the server's per-slot window, its slot count, and
// its default reply length (-1, unlimited, is not a budget).
func (s *modelSpec) applyLlamaCPPDefaults(p llamaCPPProps) {
	window := p.Settings.NCtx
	if window <= 0 {
		window = p.NCtx
	}
	s.setDefault(defaultWindow, window, "engine")
	s.setDefault(defaultSessions, int64(p.TotalSlots), "engine")
	reply := p.Settings.Params.NPredict
	if reply <= 0 {
		reply = p.Settings.NPredict
	}
	s.setDefault(defaultReply, reply, "engine")
}

// lmStudioInstance is one loaded copy of a model in LM Studio's
// /api/v1/models.
type lmStudioInstance struct {
	Config struct {
		ContextLength int64 `json:"context_length"`
		Parallel      int64 `json:"parallel"`
	} `json:"config"`
}

// applyLMStudioDefaults takes the first loaded instance's context and
// parallel slots.
func (s *modelSpec) applyLMStudioDefaults(instances []lmStudioInstance) {
	for _, inst := range instances {
		s.setDefault(defaultWindow, inst.Config.ContextLength, "engine")
		s.setDefault(defaultSessions, inst.Config.Parallel, "engine")
	}
}
