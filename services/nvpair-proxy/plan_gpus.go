// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// A node with several GPUs runs a model split by layers: each GPU holds whole
// layers, and each layer's part of the context cache sits on the GPU that holds
// the layer (llama.cpp --split-mode layer, Ollama's spread, LM Studio's split).
// The memory adds up, but every GPU keeps its own reserve for its driver and
// compute buffers, and a session fits only when its part fits on every GPU, so
// the planner sizes each GPU on its own and takes the smallest.
//
// Layers are split in proportion to the memory each GPU has after its reserve,
// as llama.cpp does by default with free memory, so GPUs of different sizes
// fill evenly. A model whose layer count is unknown is split by the same ratio
// without rounding to whole layers.

// planGPUUse is one GPU's part of a plan.
type planGPUUse struct {
	Name         string `json:"name"`
	TotalBytes   int64  `json:"totalBytes"`
	ReserveBytes int64  `json:"reserveBytes"`
	// Layers is how many of the model's layers this GPU holds; zero when
	// the model's layer count is unknown.
	Layers int `json:"layers"`
	// Share is this GPU's fraction of the model and of each session's cache.
	Share            float64 `json:"share"`
	WeightsBytes     int64   `json:"weightsBytes"`
	OtherModelsBytes int64   `json:"otherModelsBytes"`
	// BytesPerSession is this GPU's part of one session's context.
	BytesPerSession int64 `json:"bytesPerSession"`
	// Sessions is how many sessions this GPU's part has room for.
	Sessions int `json:"sessions"`
}

// gpuSplit is how a model and its sessions spread over a node's GPUs.
type gpuSplit struct {
	GPUs     []planGPUUse
	Sessions int64
	// TensorSplit is the llama.cpp --tensor-split value for this split.
	TensorSplit string
	// OneGPUIndex is the GPU with the most room once the whole model sits on
	// it, and OneGPUSessions how many sessions fit there.
	OneGPUIndex    int
	OneGPUSessions int64
}

// splitLayers divides layers among GPUs in proportion to free, by largest
// remainder, so the counts add up to layers exactly.
func splitLayers(free []int64, layers int) []int {
	var sum int64
	for _, f := range free {
		sum += f
	}
	counts := make([]int, len(free))
	if sum <= 0 || layers <= 0 {
		return counts
	}
	rem := make([]float64, len(free))
	given := 0
	for i, f := range free {
		exact := float64(layers) * float64(f) / float64(sum)
		counts[i] = int(math.Floor(exact))
		rem[i] = exact - float64(counts[i])
		given += counts[i]
	}
	for ; given < layers; given++ {
		best := 0
		for i := range rem {
			if rem[i] > rem[best] {
				best = i
			}
		}
		counts[best]++
		rem[best] = -1
	}
	return counts
}

// splitLayersByBytes divides layers among GPUs in order, as llama.cpp
// assigns them: GPU 0 takes the first ones. Each GPU takes layers until its
// share of the bytes matches its share of free memory, so GPUs fill evenly
// when layers differ in size, as they do when the first layers' experts are
// kept in system memory (--n-cpu-moe).
func splitLayersByBytes(free []int64, layerBytes []int64) []int {
	counts := make([]int, len(free))
	var freeSum, total int64
	for _, f := range free {
		freeSum += f
	}
	for _, b := range layerBytes {
		total += b
	}
	if freeSum <= 0 || len(free) == 0 {
		return counts
	}
	var cumFree, cum int64
	next := 0
	for i := 0; i < len(free)-1; i++ {
		cumFree += free[i]
		target := float64(total) * float64(cumFree) / float64(freeSum)
		start := next
		// A layer goes here when its middle falls within this GPU's part.
		for next < len(layerBytes) && float64(cum)+float64(layerBytes[next])/2 <= target {
			cum += layerBytes[next]
			next++
		}
		counts[i] = next - start
	}
	counts[len(free)-1] = len(layerBytes) - next
	return counts
}

// fitSessions is how many sessions of perSession fit in room, at most limit.
func fitSessions(room, perSession, limit int64) int64 {
	if room <= 0 {
		return 0
	}
	if perSession <= 0 {
		return limit
	}
	return min(room/perSession, limit)
}

// splitAcrossGPUs places weights (the part of the model on the GPUs) and
// sessions of perSession bytes on the GPUs. other is memory taken by other
// models, spread by the same ratio. reserves are each GPU's reserve.
// layerBytes, when known (one per layer, summing to weights), is what each
// layer keeps on the GPU; the layers are then split so that each GPU fills
// evenly with target sessions' context. Without it every layer counts the
// same.
func splitAcrossGPUs(gpus []planGPU, reserves []int64, layers int, layerBytes []int64, weights, other, perSession, limit, target int64) gpuSplit {
	free := make([]int64, len(gpus))
	var freeSum int64
	for i, g := range gpus {
		free[i] = max(0, g.VramBytes-reserves[i])
		freeSum += free[i]
	}
	byBytes := layers > 0 && len(layerBytes) == layers
	var counts []int
	if byBytes {
		// A layer holds its weights and its part of every session's cache.
		cost := make([]int64, layers)
		for i, b := range layerBytes {
			cost[i] = b + perSession*target/int64(layers)
		}
		counts = splitLayersByBytes(free, cost)
	} else {
		counts = splitLayers(free, layers)
	}
	first := 0
	split := gpuSplit{Sessions: limit, OneGPUIndex: -1}
	parts := make([]string, len(gpus))
	for i, g := range gpus {
		share := 0.0
		switch {
		case layers > 0:
			share = float64(counts[i]) / float64(layers)
		case freeSum > 0:
			share = float64(free[i]) / float64(freeSum)
		}
		otherHere := int64(0)
		if freeSum > 0 {
			otherHere = int64(float64(other) * float64(free[i]) / float64(freeSum))
		}
		weightsHere := int64(math.Ceil(float64(weights) * share))
		if byBytes {
			weightsHere = 0
			for _, b := range layerBytes[first : first+counts[i]] {
				weightsHere += b
			}
			first += counts[i]
		}
		use := planGPUUse{
			Name:             g.Name,
			TotalBytes:       g.VramBytes,
			ReserveBytes:     reserves[i],
			Layers:           counts[i],
			Share:            share,
			WeightsBytes:     weightsHere,
			OtherModelsBytes: otherHere,
			BytesPerSession:  int64(math.Ceil(float64(perSession) * share)),
		}
		room := free[i] - otherHere - use.WeightsBytes
		if share == 0 {
			// Holds nothing of this model, so it limits nothing.
			use.Sessions = int(limit)
		} else {
			use.Sessions = int(fitSessions(room, use.BytesPerSession, limit))
		}
		split.Sessions = min(split.Sessions, int64(use.Sessions))
		split.GPUs = append(split.GPUs, use)

		if one := fitSessions(free[i]-otherHere-weights, perSession, limit); free[i]-otherHere-weights > 0 &&
			(split.OneGPUIndex < 0 || one > split.OneGPUSessions) {
			split.OneGPUIndex, split.OneGPUSessions = i, one
		}
		if layers > 0 {
			parts[i] = strconv.Itoa(counts[i])
		} else {
			// GiB to a tenth: llama.cpp reads the values as proportions.
			parts[i] = strconv.FormatFloat(math.Round(float64(free[i])/float64(gib)*10)/10, 'f', -1, 64)
		}
	}
	split.TensorSplit = strings.Join(parts, ",")
	return split
}

// applyGPUSplit records each GPU's part of the plan and returns how many
// sessions fit on every GPU.
func (res *planResult) applyGPUSplit(req planRequest, gpus []planGPU, reserves []int64, perSession, limit int64) int64 {
	// Balance the layers for the sessions the GPUs' memory holds in all.
	target := fitSessions(res.ContextBytes, perSession, limit)
	split := splitAcrossGPUs(gpus, reserves, res.Spec.Layers, res.gpuLayerBytes(), res.GPUWeightsBytes, res.OtherBytes, perSession, limit, target)
	res.GPUs = split.GPUs
	if len(gpus) < 2 {
		return split.Sessions
	}
	res.TensorSplit = split.TensorSplit
	if strings.EqualFold(req.Engine, "ollama") && split.OneGPUIndex >= 0 {
		res.OneGPUIndex = split.OneGPUIndex
		res.OneGPUSessions = int(split.OneGPUSessions)
	}
	for i, g := range split.GPUs {
		if g.Share > 0 && g.WeightsBytes+g.OtherModelsBytes+g.ReserveBytes > g.TotalBytes {
			res.Warnings = append(res.Warnings, fmt.Sprintf("GPU %d's part of the model does not fit in it with the reserve kept free", i))
		}
	}
	return split.Sessions
}

// gpuLayerBytes is what each layer of a mixture-of-experts model keeps on the
// GPU: its share of the weights other than the routed experts, plus its
// experts unless --n-cpu-moe keeps them in system memory. nil when the
// per-layer expert sizes are unknown, so layers count the same.
func (res *planResult) gpuLayerBytes() []int64 {
	spec := res.Spec
	if spec.Layers <= 0 || len(spec.ExpertLayerBytes) == 0 || len(spec.ExpertLayerBytes) > spec.Layers {
		return nil
	}
	onGPU := spec.ExpertBytes - res.RAMUsedBytes
	dense := res.GPUWeightsBytes - onGPU
	if dense < 0 {
		return nil
	}
	out := make([]int64, spec.Layers)
	for i := range out {
		out[i] = dense / int64(spec.Layers)
		if i < len(spec.ExpertLayerBytes) && i >= res.CPUExpertLayers {
			out[i] += spec.ExpertLayerBytes[i]
		}
	}
	// The remainder of the even division goes to the last layer, so the
	// layers add up to the weights on the GPUs exactly.
	out[len(out)-1] += dense % int64(spec.Layers)
	return out
}
