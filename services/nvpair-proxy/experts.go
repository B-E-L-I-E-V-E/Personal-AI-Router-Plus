// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"regexp"
	"strings"
)

// Mixture-of-experts models keep most of their weights in expert tensors, of
// which only a few are used per token. llama.cpp can keep those in system
// memory (--cpu-moe) while attention and the context cache stay on the GPU,
// which frees GPU memory for more sessions at the cost of speed. The planner
// needs the exact size of those tensors, summed from the model's tensor table.

// expertTensor matches the routed-expert weights llama.cpp's --cpu-moe moves:
// blk.N.ffn_{up,down,gate,gate_up}_exps. Shared experts (_shexp) stay on the
// GPU.
var expertTensor = regexp.MustCompile(`\.ffn_(up|down|gate|gate_up)_exps(\.|$)`)

// ggmlTypeSize is (elements per block, bytes per block) for ggml tensor types.
var ggmlTypeSize = map[uint32][2]uint64{
	0: {1, 4}, 1: {1, 2}, 2: {32, 18}, 3: {32, 20}, 6: {32, 22}, 7: {32, 24},
	8: {32, 34}, 9: {32, 36}, 10: {256, 84}, 11: {256, 110}, 12: {256, 144},
	13: {256, 176}, 14: {256, 210}, 15: {256, 292}, 16: {256, 66}, 17: {256, 74},
	18: {256, 98}, 19: {256, 50}, 20: {32, 18}, 21: {256, 110}, 22: {256, 82},
	23: {256, 136}, 24: {1, 1}, 25: {1, 2}, 26: {1, 4}, 27: {1, 8}, 28: {1, 8},
	29: {256, 56}, 30: {1, 2}, 34: {256, 54}, 35: {256, 66}, 39: {32, 17},
}

// ggmlTypeID maps the type names Ollama reports to ggml type numbers.
var ggmlTypeID = map[string]uint32{
	"F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8,
	"Q8_1": 9, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14,
	"Q8_K": 15, "IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ1_S": 19,
	"IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22, "IQ4_XS": 23, "I8": 24, "I16": 25,
	"I32": 26, "I64": 27, "F64": 28, "IQ1_M": 29, "BF16": 30, "TQ1_0": 34,
	"TQ2_0": 35, "MXFP4": 39,
}

// tensorSize is the stored size of one tensor, and false for an unknown type.
func tensorSize(t ggufTensor) (int64, bool) {
	size, known := ggmlTypeSize[t.Type]
	if !known {
		return 0, false
	}
	return int64((t.Elements + size[0] - 1) / size[0] * size[1]), true
}

// tensorBytes sums every tensor: the weights' size once loaded. ok is false
// for an empty table or an unknown type, so a partial sum is never reported.
func tensorBytes(tensors []ggufTensor) (total int64, ok bool) {
	for _, t := range tensors {
		b, known := tensorSize(t)
		if !known {
			return 0, false
		}
		total += b
	}
	return total, len(tensors) > 0
}

// ollamaTensorBytes sums Ollama's /api/show tensor list; ok is false when a
// type is not a ggml one (MLX models report types such as nvfp4).
func ollamaTensorBytes(list []ollamaTensor) (int64, bool) {
	tensors := make([]ggufTensor, 0, len(list))
	for _, t := range list {
		id, ok := ggmlTypeID[strings.ToUpper(t.Type)]
		if !ok {
			return 0, false
		}
		n := uint64(1)
		for _, d := range t.Shape {
			n *= d
		}
		tensors = append(tensors, ggufTensor{Name: t.Name, Type: id, Elements: n})
	}
	return tensorBytes(tensors)
}

// expertBytes sums the routed-expert tensors. ok is false when a tensor's
// type is unknown, so a partial sum is never mistaken for the whole.
func expertBytes(tensors []ggufTensor) (total int64, ok bool) {
	for _, t := range tensors {
		if !expertTensor.MatchString(t.Name) {
			continue
		}
		b, known := tensorSize(t)
		if !known {
			return 0, false
		}
		total += b
	}
	return total, true
}

// ollamaTensors converts Ollama's /api/show tensor list.
func ollamaTensors(list []ollamaTensor) ([]ggufTensor, bool) {
	out := make([]ggufTensor, 0, len(list))
	for _, t := range list {
		id, ok := ggmlTypeID[strings.ToUpper(t.Type)]
		if !ok {
			if expertTensor.MatchString(t.Name) {
				return nil, false
			}
			continue
		}
		n := uint64(1)
		for _, d := range t.Shape {
			n *= d
		}
		out = append(out, ggufTensor{Name: t.Name, Type: id, Elements: n})
	}
	return out, true
}

type ollamaTensor struct {
	Name  string   `json:"name"`
	Type  string   `json:"type"`
	Shape []uint64 `json:"shape"`
}

// applyExperts records the expert count and expert weight size on s.
func (s *modelSpec) applyExperts(meta map[string]ggufValue, tensors []ggufTensor) {
	if s.Architecture != "" {
		if n, ok := metaInt(meta, s.Architecture+".expert_count"); ok && n > 1 {
			s.ExpertCount = int(n)
		}
	}
	if s.ExpertCount == 0 {
		return
	}
	if b, ok := expertBytes(tensors); ok && b > 0 {
		s.ExpertBytes += b
	} else if len(tensors) > 0 {
		s.Notes = append(s.Notes, "the size of this model's expert weights could not be read, so they cannot be planned in system memory")
	}
}
