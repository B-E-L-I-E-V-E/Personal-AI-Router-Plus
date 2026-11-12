// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
)

func TestSettleWeights_CountsInMemorySizeAndNamesItsSource(t *testing.T) {
	cases := []struct {
		name         string
		download     int64
		tensors      int64
		wantBytes    int64
		wantSource   string
		wantNoteLike string
	}{
		{name: "tensor sum known: in-memory size from the tensors", download: 1050, tensors: 1000, wantBytes: 1000, wantSource: "tensors"},
		{name: "no tensor sum (MLX): download size as an estimate", download: 4000, wantBytes: 4000, wantSource: "download"},
		{name: "download far larger (projector): download size, with a note", download: 1500, tensors: 1000, wantBytes: 1500, wantSource: "download", wantNoteLike: "vision projector"},
		{name: "nothing known", wantBytes: 0, wantSource: ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := modelSpec{DownloadBytes: tc.download, TensorBytes: tc.tensors}
			s.settleWeights()
			if s.WeightsBytes != tc.wantBytes || s.WeightsSource != tc.wantSource {
				t.Fatalf("got %d from %q, want %d from %q", s.WeightsBytes, s.WeightsSource, tc.wantBytes, tc.wantSource)
			}
			gotNote := strings.Join(s.Notes, " ")
			if tc.wantNoteLike == "" && gotNote != "" || !strings.Contains(gotNote, tc.wantNoteLike) {
				t.Fatalf("notes %q, want one mentioning %q", gotNote, tc.wantNoteLike)
			}
		})
	}
}

func TestOllamaTensorBytes_SumsGGMLTypesAndRefusesUnknownOnes(t *testing.T) {
	ggml := []ollamaTensor{
		{Name: "token_embd.weight", Type: "Q4_K", Shape: []uint64{256, 4}},
		{Name: "output_norm.weight", Type: "F32", Shape: []uint64{256}},
	}
	if b, ok := ollamaTensorBytes(ggml); !ok || b != 4*144+256*4 {
		t.Fatalf("ggml tensors: %d %v", b, ok)
	}
	mlx := append(ggml, ollamaTensor{Name: "blk.0.attn_q.weight", Type: "nvfp4", Shape: []uint64{256}})
	if b, ok := ollamaTensorBytes(mlx); ok {
		t.Fatalf("an MLX tensor type was summed: %d", b)
	}
	if _, ok := ollamaTensorBytes(nil); ok {
		t.Fatal("an empty tensor list gave a size")
	}
}

func TestDescribeOllama_ReportsDownloadInMemoryAndLoadedSizesApart(t *testing.T) {
	const download = 2000 + 48 // the tensors and some metadata
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			json.NewEncoder(w).Encode(map[string]any{
				"details": map[string]any{"format": "gguf"},
				"model_info": map[string]any{
					"general.architecture": "qwen3", "qwen3.block_count": 2, "qwen3.context_length": 40960,
					"qwen3.attention.head_count": 2, "qwen3.attention.head_count_kv": 1,
					"qwen3.attention.key_length": 64, "qwen3.attention.value_length": 64},
				"tensors": []any{map[string]any{"name": "token_embd.weight", "type": "F16", "shape": []int{1000}}},
			})
		case "/api/tags":
			json.NewEncoder(w).Encode(map[string]any{"models": []any{map[string]any{"name": "m", "size": download}}})
		case "/api/ps":
			json.NewEncoder(w).Encode(map[string]any{"models": []any{
				map[string]any{"name": "m", "size": 9000, "size_vram": 8000, "context_length": 4096}}})
		}
	}))
	defer srv.Close()
	base, _ := url.Parse(srv.URL)

	s, err := (&specClient{client: srv.Client(), homeDir: os.UserHomeDir}).describe(context.Background(), "ollama", base, "m", 0)
	if err != nil {
		t.Fatal(err)
	}
	if s.DownloadBytes != download || s.DownloadSource != "ollama-tags" {
		t.Errorf("download %d from %q", s.DownloadBytes, s.DownloadSource)
	}
	if s.TensorBytes != 2000 || s.WeightsBytes != 2000 || s.WeightsSource != "tensors" {
		t.Errorf("in memory %d, counted %d from %q; want the 2000-byte tensor sum", s.TensorBytes, s.WeightsBytes, s.WeightsSource)
	}
	if s.LoadedBytes != 9000 || s.LoadedVRAMBytes != 8000 || s.LoadedContextTokens != 4096 {
		t.Errorf("loaded now %d (%d on GPU) at %d tokens", s.LoadedBytes, s.LoadedVRAMBytes, s.LoadedContextTokens)
	}
	if s.Source != "metadata" {
		t.Errorf("a described model was measured instead: source %q", s.Source)
	}
}

func TestPlanSources_NameTheModelOrTheOverride(t *testing.T) {
	spec := modelSpec{Source: "metadata", WeightsBytes: gib, WeightsSource: "tensors", MaxContextTokens: 8192, KVBytesPerToken: 1024}
	res, err := computePlan(planRequest{GPUs: twoCards()}, spec)
	if err != nil || res.WeightsSource != "tensors" || res.KVSource != "metadata" {
		t.Fatalf("model values: %v weights %q kv %q", err, res.WeightsSource, res.KVSource)
	}
	res, err = computePlan(planRequest{GPUs: twoCards(), WeightsBytes: 2 * gib, KVBytesPerToken: 2048}, spec)
	if err != nil || res.WeightsBytes != 2*gib || res.WeightsSource != "manual" || res.KVSource != "manual" || res.BytesPerToken != 2048 {
		t.Fatalf("overrides: %v %+v", err, res)
	}
	if res.Spec.WeightsBytes != gib || res.Spec.KVBytesPerToken != 1024 || res.Spec.Source != "metadata" {
		t.Fatalf("the override replaced the model's own values in spec: %+v", res.Spec)
	}
}
