// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// moeGGUF builds a GGUF header for a small mixture-of-experts model: a dense
// first layer, then two layers of 8 experts (2 used per token), each expert
// tensor 4096 × 1024 in Q4_K, the down projection in Q6_K.
func moeGGUF(t *testing.T) []byte {
	t.Helper()
	var b bytes.Buffer
	w := func(v any) {
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	type tensor struct {
		name string
		dims []uint64
		typ  uint32
	}
	tensors := []tensor{
		{"blk.0.attn_q.weight", []uint64{1024, 1024}, 12},
		{"blk.0.ffn_up.weight", []uint64{1024, 4096}, 12},
	}
	for l := 1; l < 3; l++ {
		p := "blk." + string(rune('0'+l)) + "."
		tensors = append(tensors,
			tensor{p + "attn_q.weight", []uint64{1024, 1024}, 12},
			tensor{p + "ffn_gate_exps.weight", []uint64{1024, 4096, 8}, 12},
			tensor{p + "ffn_up_exps.weight", []uint64{1024, 4096, 8}, 12},
			tensor{p + "ffn_down_exps.weight", []uint64{4096, 1024, 8}, 14},
			tensor{p + "ffn_up_shexp.weight", []uint64{1024, 4096}, 12},
		)
	}
	kv := [][2]any{
		{"general.architecture", "qwen3moe"},
		{"qwen3moe.block_count", uint32(3)},
		{"qwen3moe.attention.head_count", uint32(8)},
		{"qwen3moe.attention.head_count_kv", uint32(4)},
		{"qwen3moe.attention.key_length", uint32(128)},
		{"qwen3moe.context_length", uint32(32768)},
		{"qwen3moe.expert_count", uint32(8)},
		{"qwen3moe.expert_used_count", uint32(2)},
	}
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(len(tensors)))
	w(uint64(len(kv)))
	for _, e := range kv {
		str(e[0].(string))
		switch v := e[1].(type) {
		case string:
			w(uint32(8))
			str(v)
		case uint32:
			w(uint32(4))
			w(v)
		}
	}
	for _, tn := range tensors {
		str(tn.name)
		w(uint32(len(tn.dims)))
		for _, d := range tn.dims {
			w(d)
		}
		w(tn.typ)
		w(uint64(0))
	}
	return b.Bytes()
}

// One expert layer of moeGGUF: gate and up in Q4_K (144 bytes per 256
// elements), down in Q6_K (210).
const moeLayerBytes = 2*(1024*4096*8/256*144) + 1024*4096*8/256*210

func TestGGUFTensorTableSizesExpertsPerLayer(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "moe.gguf")
	if err := os.WriteFile(path, moeGGUF(t), 0o600); err != nil {
		t.Fatal(err)
	}
	var s modelSpec
	if err := fileSpec(path, &s); err != nil {
		t.Fatal(err)
	}
	if s.ExpertCount != 8 || s.ExpertUsedCount != 2 || s.ExpertBytes != 2*moeLayerBytes {
		t.Fatalf("experts %d used %d, bytes %d want %d", s.ExpertCount, s.ExpertUsedCount, s.ExpertBytes, 2*moeLayerBytes)
	}
	want := []int64{0, moeLayerBytes, moeLayerBytes}
	if len(s.ExpertLayerBytes) != 3 || s.ExpertLayerBytes[0] != want[0] || s.ExpertLayerBytes[1] != want[1] || s.ExpertLayerBytes[2] != want[2] {
		t.Fatalf("per-layer expert bytes %v, want %v (the dense first layer has none)", s.ExpertLayerBytes, want)
	}
}

func TestGGUFTensorTableCountsTotalAndActiveParameters(t *testing.T) {
	path := filepath.Join(t.TempDir(), "moe.gguf")
	if err := os.WriteFile(path, moeGGUF(t), 0o600); err != nil {
		t.Fatal(err)
	}
	var s modelSpec
	if err := fileSpec(path, &s); err != nil {
		t.Fatal(err)
	}
	const attention = 1024 * 1024
	const denseFFN = 1024 * 4096
	const sharedExpert = 1024 * 4096
	const experts = 3 * 1024 * 4096 * 8 // gate, up and down, eight experts
	wantTotal := int64(3*attention + denseFFN + 2*(experts+sharedExpert))
	// Each token runs 2 of the 8 experts in each expert layer.
	wantActive := int64(3*attention + denseFFN + 2*(experts/8*2+sharedExpert))
	if s.TotalParams != wantTotal || s.ActiveParams != wantActive {
		t.Fatalf("params total %d active %d, want %d and %d", s.TotalParams, s.ActiveParams, wantTotal, wantActive)
	}
}

func TestOllamaTensorListSizesExperts(t *testing.T) {
	tensors, ok := ollamaTensors([]ollamaTensor{
		{Name: "blk.0.ffn_gate_exps.weight", Type: "Q4_K", Shape: []uint64{1024, 4096, 8}},
		{Name: "blk.0.ffn_down_exps.weight", Type: "MXFP4", Shape: []uint64{4096, 1024, 8}},
		{Name: "blk.0.attn_q.weight", Type: "SOMETHING_NEW", Shape: []uint64{1024, 1024}},
	})
	if !ok {
		t.Fatal("an unknown non-expert type must not block the sum")
	}
	layers, ok := expertLayerBytes(tensors)
	if want := int64(1024*4096*8/256*144 + 1024*4096*8/32*17); !ok || len(layers) != 1 || layers[0] != want {
		t.Fatalf("expert bytes %v want [%d]", layers, want)
	}
	if total, _ := parameterCounts(tensors, 8, 2); total != 2*1024*4096*8+1024*1024 {
		t.Fatalf("total params %d: a tensor of unknown type still counts", total)
	}
	if _, ok := ollamaTensors([]ollamaTensor{{Name: "blk.0.ffn_up_exps.weight", Type: "Q9_X", Shape: []uint64{8}}}); ok {
		t.Fatal("an expert tensor of unknown type must make the size unknown")
	}
}

// A 30B-A3B-class model: 18 GiB of weights, 16 GiB of them experts in 16
// layers of 1 GiB.
func moeSpec() modelSpec {
	s := denseSpec()
	s.WeightsBytes = 18 * gib
	s.ExpertCount = 128
	s.ExpertUsedCount = 8
	s.ExpertBytes = 16 * gib
	s.ExpertLayerBytes = make([]int64, 16)
	for i := range s.ExpertLayerBytes {
		s.ExpertLayerBytes[i] = gib
	}
	return s
}

func TestPlanExpertOffloadFreesGPUMemory(t *testing.T) {
	card := []planGPU{{Name: "A", VramBytes: 24 * gib}}
	base := planRequest{Engine: "llamacpp", Model: "m", GPUs: card, WindowTokens: 32768, SystemRAMBytes: 64 * gib}
	onGPU, err := computePlan(base, moeSpec())
	if err != nil || !onGPU.CanOffloadExperts || onGPU.RAMUsedBytes != 0 {
		t.Fatalf("on GPU: %v %+v", err, onGPU)
	}
	base.CPUExpertLayers = 16
	off, err := computePlan(base, moeSpec())
	if err != nil {
		t.Fatal(err)
	}
	if off.GPUWeightsBytes != 2*gib || off.RAMUsedBytes != 16*gib || !off.RAMFits || off.Sessions <= onGPU.Sessions {
		t.Fatalf("offloaded: gpu weights %d ram %d sessions %d (on GPU %d)", off.GPUWeightsBytes, off.RAMUsedBytes, off.Sessions, onGPU.Sessions)
	}
	if off.RAMReserveBytes != 16*gib {
		t.Fatalf("ram reserve %d, want a quarter", off.RAMReserveBytes)
	}
}

func TestPlanPartialExpertSplitCountsOnlyTheFirstLayersInRAM(t *testing.T) {
	card := []planGPU{{Name: "A", VramBytes: 24 * gib}}
	req := planRequest{Engine: "llamacpp", Model: "m", GPUs: card, WindowTokens: 32768, SystemRAMBytes: 64 * gib}
	sessions := map[int]int{}
	for _, layers := range []int{0, 4, 10, 16} {
		req.CPUExpertLayers = layers
		res, err := computePlan(req, moeSpec())
		if err != nil {
			t.Fatal(err)
		}
		inRAM := int64(layers) * gib
		if res.CPUExpertLayers != layers || res.RAMUsedBytes != inRAM || res.GPUWeightsBytes != 18*gib-inRAM {
			t.Fatalf("%d layers on the CPU: ram %d gpu weights %d", layers, res.RAMUsedBytes, res.GPUWeightsBytes)
		}
		sessions[layers] = res.Sessions
	}
	// 24 GiB − 2.4 GiB reserve − GPU weights, in sessions of 32K × 192 KiB = 6 GiB.
	want := map[int]int{0: 0, 4: 1, 10: 2, 16: 3}
	for layers, n := range want {
		if sessions[layers] != n {
			t.Errorf("%d expert layers on the CPU: %d sessions fit, want %d", layers, sessions[layers], n)
		}
	}
}

func TestPlanExpertSplitSkipsDenseLeadingLayers(t *testing.T) {
	spec := moeSpec()
	spec.ExpertLayerBytes = append([]int64{0, 0}, spec.ExpertLayerBytes[:14]...)
	spec.ExpertBytes = 14 * gib
	req := planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), WindowTokens: 8192, SystemRAMBytes: 64 * gib, CPUExpertLayers: 3}
	res, err := computePlan(req, spec)
	if err != nil {
		t.Fatal(err)
	}
	if res.RAMUsedBytes != gib {
		t.Fatalf("--n-cpu-moe 3 over two dense layers moves one expert layer, got %d bytes", res.RAMUsedBytes)
	}
}

func TestPlanExpertLayersAboveTheModelMeanAllOfThem(t *testing.T) {
	req := planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), WindowTokens: 8192, SystemRAMBytes: 64 * gib, CPUExpertLayers: maxCPUExpertLayers}
	res, err := computePlan(req, moeSpec())
	if err != nil {
		t.Fatal(err)
	}
	if res.CPUExpertLayers != 16 || res.RAMUsedBytes != 16*gib {
		t.Fatalf("layers %d ram %d, want all 16 layers", res.CPUExpertLayers, res.RAMUsedBytes)
	}
	req.CPUExpertLayers = maxCPUExpertLayers + 1
	if _, err := computePlan(req, moeSpec()); err == nil {
		t.Fatal("a layer count past the limit must be refused")
	}
}

func TestPlanExpertOffloadReportsRAMOverflow(t *testing.T) {
	req := planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), WindowTokens: 8192, SystemRAMBytes: 16 * gib, CPUExpertLayers: 16}
	res, err := computePlan(req, moeSpec())
	if err != nil {
		t.Fatal(err)
	}
	if res.RAMFits || res.Fits {
		t.Fatalf("16 GiB of experts in 16 GiB of RAM must not fit: %+v", res)
	}
}

func TestPlanExpertOffloadRefusedWhereUseless(t *testing.T) {
	for name, req := range map[string]planRequest{
		"ollama":  {Engine: "ollama", GPUs: twoCards(), SystemRAMBytes: 64 * gib, CPUExpertLayers: 4},
		"unified": {Engine: "llamacpp", GPUs: twoCards(), Unified: true, SystemRAMBytes: 64 * gib, CPUExpertLayers: 4},
		"no ram":  {Engine: "llamacpp", GPUs: twoCards(), CPUExpertLayers: 4},
	} {
		res, err := computePlan(req, moeSpec())
		if err == nil || res.CanOffloadExperts || !strings.Contains(err.Error(), "expert") {
			t.Fatalf("%s: %v %+v", name, err, res)
		}
	}
	if _, err := computePlan(planRequest{Engine: "llamacpp", GPUs: twoCards(), SystemRAMBytes: 64 * gib, CPUExpertLayers: 4}, denseSpec()); err == nil {
		t.Fatal("a dense model has no experts to move")
	}
}

func TestPlanReportsSessionCostWhenModelDoesNotFit(t *testing.T) {
	spec := denseSpec()
	spec.WeightsBytes = 70 * gib
	res, err := computePlan(planRequest{Engine: "llamacpp", GPUs: twoCards(), WindowTokens: 8192}, spec)
	if err != nil || res.Sessions != 0 || res.BytesPerSession == 0 {
		t.Fatalf("%v sessions %d per session %d", err, res.Sessions, res.BytesPerSession)
	}
}
