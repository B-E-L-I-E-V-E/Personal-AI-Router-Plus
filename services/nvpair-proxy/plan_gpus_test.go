// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"reflect"
	"testing"
)

// What an RTX 4090 reports as its memory in node-info.
const rtx4090Bytes = int64(23706206208)

func twoRTX4090s() []planGPU {
	return []planGPU{{Name: "NVIDIA GeForce RTX 4090", VramBytes: rtx4090Bytes}, {Name: "NVIDIA GeForce RTX 4090", VramBytes: rtx4090Bytes}}
}

// A model of 64 layers with a 1 MiB per-token cache (16 KiB per layer).
func layeredSpec(weights int64) modelSpec {
	return modelSpec{Engine: "llamacpp", Model: "m", Source: "metadata", WeightsBytes: weights,
		MaxContextTokens: 131072, KVBytesPerToken: 1 << 20, Layers: 64}
}

func TestPlanGPUSplit_TwoEqualGPUsSplitLayersEvenly(t *testing.T) {
	res, err := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoRTX4090s(), WindowTokens: 8192}, layeredSpec(20*gib))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.GPUs) != 2 || res.GPUs[0].Layers != 32 || res.GPUs[1].Layers != 32 {
		t.Fatalf("layers per GPU: %+v", res.GPUs)
	}
	if res.TensorSplit != "32,32" {
		t.Fatalf("tensor split %q", res.TensorSplit)
	}
	// Each GPU: 22.08 GiB − 2.21 GiB reserve − 10 GiB of weights = 9.87 GiB,
	// and holds half of each 8 GiB session: 4 GiB. Two sessions fit.
	for i, g := range res.GPUs {
		if g.WeightsBytes != 10*gib || g.BytesPerSession != 4*gib || g.Sessions != 2 {
			t.Fatalf("GPU %d: %+v", i, g)
		}
	}
	if res.Sessions != 2 || !res.Fits {
		t.Fatalf("sessions %d, fits %v", res.Sessions, res.Fits)
	}
}

func TestPlanGPUSplit_UnevenGPUsSplitByMemoryRatio(t *testing.T) {
	gpus := []planGPU{{Name: "big", VramBytes: 24 * gib}, {Name: "small", VramBytes: 12 * gib}}
	spec := layeredSpec(12 * gib)
	spec.Layers = 48
	res, err := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: gpus, WindowTokens: 4096}, spec)
	if err != nil {
		t.Fatal(err)
	}
	// 21.6 GiB and 10.8 GiB after the reserve: two to one.
	if res.GPUs[0].Layers != 32 || res.GPUs[1].Layers != 16 || res.TensorSplit != "32,16" {
		t.Fatalf("split %q: %+v", res.TensorSplit, res.GPUs)
	}
	// The big GPU: 21.6 − 8 = 13.6 GiB for 2.67 GiB parts; the small one:
	// 10.8 − 4 = 6.8 GiB for 1.33 GiB parts. Both have room for five.
	if res.GPUs[0].Sessions != 5 || res.GPUs[1].Sessions != 5 || res.Sessions != 5 {
		t.Fatalf("sessions %d: %+v", res.Sessions, res.GPUs)
	}
}

func TestPlanGPUSplit_WholeLayersLimitSessionsBelowThePooledSum(t *testing.T) {
	spec := layeredSpec(3 * gib)
	spec.Layers = 3 // two layers on one GPU, one on the other
	res, err := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoRTX4090s(), WindowTokens: 3072}, spec)
	if err != nil {
		t.Fatal(err)
	}
	pooled := res.ContextBytes / res.BytesPerSession
	if res.GPUs[0].Layers != 2 || res.GPUs[1].Layers != 1 {
		t.Fatalf("layers: %+v", res.GPUs)
	}
	// The GPU with two layers holds two thirds of every session and runs
	// out first.
	if int64(res.Sessions) >= pooled || res.Sessions != res.GPUs[0].Sessions {
		t.Fatalf("sessions %d, pooled %d, per GPU %+v", res.Sessions, pooled, res.GPUs)
	}
}

func TestPlanGPUSplit_OllamaReportsWhenTheModelFitsOnOneGPU(t *testing.T) {
	tests := []struct {
		name        string
		weights     int64
		wantOneGPU  bool
		minSessions int
	}{
		{name: "small model fits on one GPU", weights: 4 * gib, wantOneGPU: true, minSessions: 2},
		{name: "large model needs both GPUs", weights: 30 * gib, wantOneGPU: false, minSessions: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec := layeredSpec(tt.weights)
			spec.Engine = "ollama"
			res, err := computePlan(planRequest{Engine: "ollama", Model: "m", GPUs: twoRTX4090s(), WindowTokens: 4096}, spec)
			if err != nil {
				t.Fatal(err)
			}
			if (res.OneGPUSessions > 0) != tt.wantOneGPU {
				t.Fatalf("one GPU sessions %d", res.OneGPUSessions)
			}
			if res.Sessions < tt.minSessions || res.Sessions < res.OneGPUSessions {
				t.Fatalf("sessions %d, one GPU %d", res.Sessions, res.OneGPUSessions)
			}
		})
	}
}

func TestPlanGPUSplit_OneGPUHasNoSplit(t *testing.T) {
	res, err := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: twoRTX4090s()[:1], WindowTokens: 8192}, layeredSpec(10*gib))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.GPUs) != 1 || res.GPUs[0].Layers != 64 || res.TensorSplit != "" || res.OneGPUSessions != 0 {
		t.Fatalf("%q %+v", res.TensorSplit, res.GPUs)
	}
}

func TestPlanGPUSplit_UnknownLayerCountSplitsByMemory(t *testing.T) {
	gpus := []planGPU{{Name: "big", VramBytes: 24 * gib}, {Name: "small", VramBytes: 12 * gib}}
	spec := layeredSpec(6 * gib)
	spec.Layers = 0
	res, err := computePlan(planRequest{Engine: "llamacpp", Model: "m", GPUs: gpus, WindowTokens: 4096}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if res.TensorSplit != "21.6,10.8" || res.GPUs[0].WeightsBytes != 4*gib {
		t.Fatalf("%q %+v", res.TensorSplit, res.GPUs)
	}
}

func TestSplitLayers_CountsAlwaysAddUp(t *testing.T) {
	tests := []struct {
		name   string
		free   []int64
		layers int
		want   []int
	}{
		{name: "equal", free: []int64{10, 10}, layers: 33, want: []int{17, 16}},
		{name: "three to one", free: []int64{30, 10}, layers: 40, want: []int{30, 10}},
		{name: "three GPUs", free: []int64{24, 24, 12}, layers: 61, want: []int{25, 24, 12}},
		{name: "no memory", free: []int64{0, 0}, layers: 8, want: []int{0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := splitLayers(tt.free, tt.layers); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}
