// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"testing"
)

func TestLoadWithSettingsWire(t *testing.T) {
	action, params, err := loadWithSettingsWire("lmstudio", "qwen/qwen3.8-27b", loadSettings{ContextLength: 262144, Parallel: 4})
	if err != nil {
		t.Fatalf("wire: %v", err)
	}
	if action != "load_model_with_settings" {
		t.Fatalf("action = %q", action)
	}
	var got map[string]any
	if err := json.Unmarshal(params, &got); err != nil {
		t.Fatal(err)
	}
	if got["model"] != "qwen/qwen3.8-27b" || got["context_length"] != float64(262144) || got["parallel"] != float64(4) {
		t.Fatalf("params = %v", got)
	}
}

func TestLoadWithSettingsRejects(t *testing.T) {
	if _, _, err := loadWithSettingsWire("ollama", "m", loadSettings{ContextLength: 4096, Parallel: 1}); err == nil {
		t.Fatal("ollama must not take load settings")
	}
	if _, _, err := loadWithSettingsWire("lmstudio", "m", loadSettings{ContextLength: 0, Parallel: 1}); err == nil {
		t.Fatal("a zero context length must be rejected")
	}
	if _, _, err := loadWithSettingsWire("lmstudio", "", loadSettings{ContextLength: 4096, Parallel: 1}); err == nil {
		t.Fatal("a missing model must be rejected")
	}
}

// The manifest's command must expand with the numbers the wire carries.
func TestLMStudioLoadWithSettingsCommand(t *testing.T) {
	m, found := buildBundledRegistry().Get("lmstudio")
	if !found {
		t.Fatal("no bundled lmstudio manifest")
	}
	act, ok := m.Actions["load_model_with_settings"]
	if !ok {
		t.Fatal("lmstudio manifest has no load_model_with_settings")
	}
	argv, err := resolveArgs(act.Cmd, map[string]string{
		"cli": "lms", "model": "qwen/qwen3.8-27b", "context_length": "262144", "parallel": "4",
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"lms", "load", "qwen/qwen3.8-27b", "--context-length", "262144", "--parallel", "4", "--yes"}
	if len(argv) != len(want) {
		t.Fatalf("argv = %v", argv)
	}
	for i := range want {
		if argv[i] != want[i] {
			t.Fatalf("argv = %v, want %v", argv, want)
		}
	}
}

func TestLoadWithSettingsWaitsForTheModel(t *testing.T) {
	with := modelActionRequest{Engine: "lmstudio", Model: "m", Load: &loadSettings{ContextLength: 4096, Parallel: 1}}
	if !loadsWithSettings(controlLoadPath, with) {
		t.Error("a load with settings must use the readiness budget")
	}
	if loadsWithSettings(controlLoadPath, modelActionRequest{Engine: "lmstudio", Model: "m"}) {
		t.Error("a plain load keeps the ordinary budget")
	}
	if loadsWithSettings(controlUnloadPath, with) {
		t.Error("only a load waits for the model")
	}
}
