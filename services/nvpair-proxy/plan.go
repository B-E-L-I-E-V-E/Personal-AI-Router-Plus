// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"fmt"
	"strings"
)

// The capacity planner answers: with this node's GPU memory and this model,
// how many sessions fit without running out of memory? The context window
// belongs to one conversation, so every session gets the whole window — the
// model's maximum unless the user lowers it — and the memory left decides how
// many such sessions run at once. Sessions never share or split a window.
//
//	usable    = total VRAM − reserve − model weights − other resident models
//	per token = key/value cache bytes for one token of context
//	session   = window × per token (+ fixed sliding-window and recurrent state)
//	sessions  = usable ÷ session, at most the user's cap
//
// The reserve keeps headroom for the driver, the engine's working buffers and
// everything the formula does not count. On Apple silicon the GPU shares
// system memory and macOS lets it use only about three quarters of it, so the
// reserve is at least a quarter of the total there.

const (
	gib                    = int64(1) << 30
	defaultReservePercent  = 10
	minReservePerGPU       = gib
	unifiedMemoryGPUShare  = 0.75
	maxPlannedSessions     = 256
	minPlannedWindowTokens = 2048
)

// cacheTypeFactor is the key/value cache size relative to f16. q8_0 stores 32
// values in 34 bytes and q4_0 in 18, against 64 for f16.
var cacheTypeFactor = map[string]float64{
	"f16":  1,
	"q8_0": 34.0 / 64,
	"q4_0": 18.0 / 64,
}

type planGPU struct {
	Name      string `json:"name"`
	VramBytes int64  `json:"vramBytes"`
}

type planRequest struct {
	Engine string    `json:"engine"`
	Model  string    `json:"model"`
	GPUs   []planGPU `json:"gpus"`
	// Unified marks GPUs that share system memory (Apple silicon).
	Unified bool `json:"unified,omitempty"`

	// WindowTokens is each session's window; zero means the model's maximum.
	WindowTokens int64 `json:"windowTokens,omitempty"`
	// MaxSessions caps how many sessions are planned; zero means as many as
	// fit. It never shrinks a session's window.
	MaxSessions int `json:"maxSessions,omitempty"`

	CacheType      string  `json:"cacheType,omitempty"`
	ReservePercent float64 `json:"reservePercent,omitempty"`
	// OtherModelsBytes is memory taken by other models kept loaded beside
	// this one.
	OtherModelsBytes int64 `json:"otherModelsBytes,omitempty"`

	// Manual values for what the engine cannot describe.
	WeightsBytes    int64 `json:"weightsBytes,omitempty"`
	KVBytesPerToken int64 `json:"kvBytesPerToken,omitempty"`
}

type planResult struct {
	Spec modelSpec `json:"spec"`

	TotalBytes      int64 `json:"totalBytes"`
	ReserveBytes    int64 `json:"reserveBytes"`
	WeightsBytes    int64 `json:"weightsBytes"`
	OtherBytes      int64 `json:"otherModelsBytes"`
	ContextBytes    int64 `json:"contextBytes"`
	BytesPerToken   int64 `json:"bytesPerToken"`
	BytesPerSession int64 `json:"bytesPerSession"`

	CacheType    string   `json:"cacheType"`
	WindowTokens int64    `json:"windowTokens"`
	Sessions     int      `json:"sessions"`
	PoolTokens   int64    `json:"poolTokens"`
	UsedBytes    int64    `json:"usedBytes"`
	Fits         bool     `json:"fits"`
	Warnings     []string `json:"warnings,omitempty"`
	// Error is why the model could not be sized; Spec still says what is
	// known, so the caller can ask for the missing values.
	Error string `json:"error,omitempty"`
}

// sessionBytes is the memory one session of window tokens needs.
func sessionBytes(spec modelSpec, perToken int64, factor float64, window int64) int64 {
	b := window * perToken
	if spec.SlidingBytesPerToken > 0 {
		w := window
		if spec.SlidingWindow > 0 && spec.SlidingWindow < w {
			w = spec.SlidingWindow
		}
		b += int64(float64(spec.SlidingBytesPerToken)*factor) * w
	}
	return b + spec.RecurrentBytesPerSession
}

// computePlan is pure: it applies the request to a model description.
func computePlan(req planRequest, spec modelSpec) (planResult, error) {
	res := planResult{Spec: spec}
	if req.WindowTokens < 0 || req.MaxSessions < 0 || req.OtherModelsBytes < 0 || req.WeightsBytes < 0 || req.KVBytesPerToken < 0 {
		return res, errors.New("values must not be negative")
	}
	if req.MaxSessions > maxPlannedSessions {
		return res, fmt.Errorf("maxSessions must be at most %d", maxPlannedSessions)
	}
	cache := req.CacheType
	if cache == "" {
		cache = "f16"
	}
	factor, ok := cacheTypeFactor[cache]
	if !ok {
		return res, errors.New("cacheType must be f16, q8_0 or q4_0")
	}
	res.CacheType = cache
	pct := req.ReservePercent
	if pct == 0 {
		pct = defaultReservePercent
	}
	if pct < 0 || pct > 90 {
		return res, errors.New("reservePercent must be between 0 and 90")
	}
	if len(req.GPUs) == 0 {
		return res, errors.New("no GPU memory reported for this node")
	}

	for _, g := range req.GPUs {
		if g.VramBytes <= 0 {
			continue
		}
		res.TotalBytes += g.VramBytes
		r := int64(float64(g.VramBytes) * pct / 100)
		if r < minReservePerGPU {
			r = minReservePerGPU
		}
		res.ReserveBytes += r
	}
	if res.TotalBytes == 0 {
		return res, errors.New("no GPU memory reported for this node")
	}
	if req.Unified {
		if floor := int64(float64(res.TotalBytes) * (1 - unifiedMemoryGPUShare)); res.ReserveBytes < floor {
			res.ReserveBytes = floor
		}
	}

	res.WeightsBytes = spec.WeightsBytes
	if req.WeightsBytes > 0 {
		res.WeightsBytes = req.WeightsBytes
	}
	perF16 := spec.KVBytesPerToken
	if req.KVBytesPerToken > 0 {
		perF16 = req.KVBytesPerToken
		res.Spec.Source = "manual"
	}
	if res.WeightsBytes <= 0 {
		return res, errors.New("the model's size is unknown; enter it (weightsBytes)")
	}
	if perF16 <= 0 {
		return res, errors.New("the model's context cost is unknown; enter it per token (kvBytesPerToken)")
	}
	// A measured cost was taken at the engine's own cache type and already
	// includes it; only scale what was computed from the architecture.
	perToken := perF16
	if res.Spec.Source != "measured" {
		perToken = int64(float64(perF16) * factor)
	}
	if perToken < 1 {
		perToken = 1
	}
	res.BytesPerToken = perToken
	res.OtherBytes = req.OtherModelsBytes
	res.ContextBytes = res.TotalBytes - res.ReserveBytes - res.WeightsBytes - res.OtherBytes
	if res.ContextBytes <= 0 {
		res.Warnings = append(res.Warnings, "the model does not fit in this node's GPU memory with the reserve kept free")
		res.ContextBytes = 0
		return res, nil
	}

	maxWindow := spec.MaxContextTokens
	window := req.WindowTokens
	if window == 0 {
		window = maxWindow
		if window == 0 {
			return res, errors.New("the model's maximum context is unknown; enter a window (windowTokens)")
		}
	}
	if maxWindow > 0 && window > maxWindow {
		res.Warnings = append(res.Warnings, fmt.Sprintf("the model supports at most %d tokens; using that", maxWindow))
		window = maxWindow
	}
	res.WindowTokens = window
	n := res.ContextBytes / sessionBytes(spec, perToken, factor, window)
	limit := int64(maxPlannedSessions)
	if req.MaxSessions > 0 {
		limit = int64(req.MaxSessions)
	}
	if n > limit {
		n = limit
	}
	res.Sessions = int(n)
	if n == 0 {
		res.Warnings = append(res.Warnings, "not even one session of this window fits; choose a smaller window or a smaller cache type")
	}
	res.BytesPerSession = sessionBytes(spec, perToken, factor, res.WindowTokens)
	res.PoolTokens = res.WindowTokens * int64(res.Sessions)
	res.UsedBytes = res.WeightsBytes + res.OtherBytes + res.BytesPerSession*int64(res.Sessions)
	res.Fits = res.Sessions > 0 && res.WindowTokens > 0 && res.UsedBytes+res.ReserveBytes <= res.TotalBytes
	if res.Spec.Source == "measured" && cache != "f16" {
		res.Warnings = append(res.Warnings, "the cost was measured at the engine's current cache type; re-measure after applying "+cache)
	}
	if strings.EqualFold(req.Engine, "ollama") {
		res.Warnings = append(res.Warnings, "Ollama applies the window and slot count to every model it runs, and restarts to change them")
	}
	return res, nil
}
