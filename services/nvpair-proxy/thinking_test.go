// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const thinkingModelYAML = `model: qwen/qwen3.8-27b
config:
  operation:
    fields:
      - key: llm.prediction.temperature
        value: 1.0
customFields:
  - key: reasoningEffort
    type: select
    defaultValue: xhigh
    effects:
      - type: setJinjaVariable
        variable: reasoning_effort
  - key: enableThinking
    displayName: Enable Thinking
    type: boolean
    defaultValue: true
    effects:
      - type: setJinjaVariable
        variable: enable_thinking
`

func thinkingFixture(t *testing.T, existing string) (thinkingStore, string) {
	t.Helper()
	home := t.TempDir()
	dir := filepath.Join(home, "hub", "models", "qwen", "qwen3.8-27b")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.yaml"), []byte(thinkingModelYAML), 0o644); err != nil {
		t.Fatal(err)
	}
	store := thinkingStore{home: home}
	path := store.configPath("qwen/qwen3.8-27b")
	if existing != "" {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(existing), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return store, path
}

func TestThinkingFieldIsFoundByItsTemplateVariable(t *testing.T) {
	field, def, ok := thinkingFieldOf([]byte(thinkingModelYAML))
	if !ok || field != "enableThinking" || !def {
		t.Fatalf("field = %q default %v ok %v", field, def, ok)
	}
	if _, _, ok := thinkingFieldOf([]byte("model: x\ncustomFields:\n  - key: a\n    effects:\n      - variable: other\n")); ok {
		t.Fatal("a model without enable_thinking must have no thinking field")
	}
	_, def, _ = thinkingFieldOf([]byte("customFields:\n  - key: t\n    defaultValue: false\n    effects:\n      - variable: enable_thinking\n"))
	if def {
		t.Fatal("default false must be read")
	}
}

func TestThinkingConfigKey(t *testing.T) {
	if got := thinkingConfigKey("qwen/qwen3.8-27b", "enableThinking"); got != "ext.virtualModel.customField.qwen.qwen3.827b.enableThinking" {
		t.Fatalf("key = %q", got)
	}
}

func TestThinkingWriteKeepsEveryOtherField(t *testing.T) {
	existing := `{
  "preset": "mine",
  "extra": {"keep": [1, 2]},
  "operation": {
    "other": true,
    "fields": [
      { "key": "llm.prediction.temperature", "value": 0.4 },
      { "key": "ext.virtualModel.customField.qwen.qwen3.827b.reasoningEffort", "value": "low" }
    ]
  },
  "load": { "fields": [ { "key": "llm.load.contextLength", "value": 262144 } ] }
}`
	store, path := thinkingFixture(t, existing)
	st, err := store.read("qwen/qwen3.8-27b")
	if err != nil || !st.Supported || !st.Value || st.Explicit {
		t.Fatalf("before = %+v err %v", st, err)
	}
	st, err = store.write("qwen/qwen3.8-27b", ptrBool(false))
	if err != nil || st.Value || !st.Explicit {
		t.Fatalf("after set = %+v err %v", st, err)
	}
	raw, _ := os.ReadFile(path)
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got["preset"] != "mine" || got["extra"] == nil || got["load"] == nil {
		t.Fatalf("top-level entries lost: %s", raw)
	}
	for _, want := range []string{`"llm.prediction.temperature"`, `reasoningEffort`, `"other": true`, `llm.load.contextLength`, `enableThinking`} {
		if !strings.Contains(string(raw), want) {
			t.Fatalf("%s missing from %s", want, raw)
		}
	}
	// Setting again replaces, never duplicates.
	if _, err := store.write("qwen/qwen3.8-27b", ptrBool(true)); err != nil {
		t.Fatal(err)
	}
	raw, _ = os.ReadFile(path)
	if strings.Count(string(raw), "enableThinking") != 1 {
		t.Fatalf("thinking field duplicated: %s", raw)
	}
	// Clearing returns to the model default and keeps the rest.
	st, err = store.write("qwen/qwen3.8-27b", nil)
	if err != nil || st.Explicit || !st.Value {
		t.Fatalf("after clear = %+v err %v", st, err)
	}
	raw, _ = os.ReadFile(path)
	if strings.Contains(string(raw), "enableThinking") || !strings.Contains(string(raw), "llm.prediction.temperature") {
		t.Fatalf("clear changed the wrong entries: %s", raw)
	}
}

func TestThinkingCreatesAMissingConfig(t *testing.T) {
	store, path := thinkingFixture(t, "")
	if _, err := store.write("qwen/qwen3.8-27b", ptrBool(false)); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(path)
	if !strings.Contains(string(raw), `"value": false`) {
		t.Fatalf("config = %s", raw)
	}
}

func TestThinkingLeavesAnUnreadableConfigAlone(t *testing.T) {
	store, path := thinkingFixture(t, "{not json")
	if _, err := store.write("qwen/qwen3.8-27b", ptrBool(false)); err == nil {
		t.Fatal("an invalid config must be refused")
	}
	if raw, _ := os.ReadFile(path); string(raw) != "{not json" {
		t.Fatalf("config was rewritten: %s", raw)
	}
}

func TestThinkingUnsupportedAndUnsafeModels(t *testing.T) {
	store, _ := thinkingFixture(t, "")
	st, err := store.read("glm-4.7-flash")
	if err != nil || st.Supported || st.Reason == "" {
		t.Fatalf("model without a hub definition = %+v err %v", st, err)
	}
	for _, bad := range []string{"../etc/passwd", "a/../../b", "a/b/c", "", "a b"} {
		if _, err := store.read(bad); err == nil {
			t.Errorf("%q must be refused", bad)
		}
		if _, err := store.write(bad, ptrBool(false)); err == nil {
			t.Errorf("write %q must be refused", bad)
		}
	}
}

func ptrBool(b bool) *bool { return &b }
