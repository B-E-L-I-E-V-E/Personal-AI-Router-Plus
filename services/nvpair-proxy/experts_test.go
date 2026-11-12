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

// moeGGUF builds a GGUF header for a small mixture-of-experts model: two
// layers of 8 experts, each expert tensor 4096 × 1024 in Q4_K.
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
	var tensors []tensor
	for l := 0; l < 2; l++ {
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
		{"qwen3moe.block_count", uint32(2)},
		{"qwen3moe.attention.head_count", uint32(8)},
		{"qwen3moe.attention.head_count_kv", uint32(4)},
		{"qwen3moe.attention.key_length", uint32(128)},
		{"qwen3moe.context_length", uint32(32768)},
		{"qwen3moe.expert_count", uint32(8)},
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

// Q4_K: 144 bytes per 256 elements; Q6_K: 210.
const moeExpertBytes = 2 * (2*(1024*4096*8/256*144) + 1024*4096*8/256*210)

func TestGGUFTensorTableSizesExperts(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "moe.gguf")
	if err := os.WriteFile(path, moeGGUF(t), 0o600); err != nil {
		t.Fatal(err)
	}
	var s modelSpec
	if err := fileSpec(path, &s); err != nil {
		t.Fatal(err)
	}
	if s.ExpertCount != 8 || s.ExpertBytes != moeExpertBytes {
		t.Fatalf("experts %d, bytes %d want %d", s.ExpertCount, s.ExpertBytes, moeExpertBytes)
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
	b, ok := expertBytes(tensors)
	if want := int64(1024*4096*8/256*144 + 1024*4096*8/32*17); !ok || b != want {
		t.Fatalf("expert bytes %d want %d", b, want)
	}
	if _, ok := ollamaTensors([]ollamaTensor{{Name: "blk.0.ffn_up_exps.weight", Type: "Q9_X", Shape: []uint64{8}}}); ok {
		t.Fatal("an expert tensor of unknown type must make the size unknown")
	}
}

// A 30B-A3B-class model: 18 GiB of weights, 16 GiB of them experts.
func moeSpec() modelSpec {
	s := denseSpec()
	s.WeightsBytes = 18 * gib
	s.ExpertCount = 128
	s.ExpertBytes = 16 * gib
	return s
}

func TestPlanExpertOffloadFreesGPUMemory(t *testing.T) {
	card := []planGPU{{Name: "A", VramBytes: 24 * gib}}
	base := planRequest{Engine: "llamacpp", Model: "m", GPUs: card, WindowTokens: 32768, SystemRAMBytes: 64 * gib}
	onGPU, err := computePlan(base, moeSpec())
	if err != nil || !onGPU.CanOffloadExperts || onGPU.RAMUsedBytes != 0 {
		t.Fatalf("on GPU: %v %+v", err, onGPU)
	}
	base.OffloadExperts = true
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

func TestPlanExpertOffloadReportsRAMOverflow(t *testing.T) {
	req := planRequest{Engine: "llamacpp", Model: "m", GPUs: twoCards(), WindowTokens: 8192, SystemRAMBytes: 16 * gib, OffloadExperts: true}
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
		"ollama":  {Engine: "ollama", GPUs: twoCards(), SystemRAMBytes: 64 * gib, OffloadExperts: true},
		"unified": {Engine: "llamacpp", GPUs: twoCards(), Unified: true, SystemRAMBytes: 64 * gib, OffloadExperts: true},
		"no ram":  {Engine: "llamacpp", GPUs: twoCards(), OffloadExperts: true},
	} {
		res, err := computePlan(req, moeSpec())
		if err == nil || res.CanOffloadExperts || !strings.Contains(err.Error(), "expert") {
			t.Fatalf("%s: %v %+v", name, err, res)
		}
	}
	if _, err := computePlan(planRequest{Engine: "llamacpp", GPUs: twoCards(), SystemRAMBytes: 64 * gib, OffloadExperts: true}, denseSpec()); err == nil {
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
