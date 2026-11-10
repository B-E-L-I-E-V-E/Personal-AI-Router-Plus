// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"testing"
)

func TestParseLifecycleValid(t *testing.T) {
	params := json.RawMessage(`{"workloadInfo":{"id":"wl-1","model":"llama-3","engine":"trt-llm","state":"queued","originatedFrom":"node-A","createdAt":1,"startedAt":null,"completedAt":null,"error":null,"requesterId":null}}`)
	w, err := parseLifecycle(params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if w.ID != "wl-1" || w.Model != "llama-3" || w.Engine != "trt-llm" {
		t.Fatalf("unexpected workload: %+v", w)
	}
}

// The caller-facing request ID must survive the workload-manager's typed
// decode and re-encode, or peers and the desktop would never see it.
func TestParseLifecyclePreservesRequestID(t *testing.T) {
	params := json.RawMessage(`{"workloadInfo":{"id":"wl-1","model":"m","engine":"ollama","state":"queued","originatedFrom":"node-A","createdAt":1,"requestId":"agent-7:turn-12"}}`)
	w, err := parseLifecycle(params)
	if err != nil {
		t.Fatalf("parseLifecycle: %v", err)
	}
	if w.RequestID != "agent-7:turn-12" {
		t.Fatalf("RequestID = %q, want %q", w.RequestID, "agent-7:turn-12")
	}
	out, err := json.Marshal(w)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round struct {
		RequestID string `json:"requestId"`
	}
	if err := json.Unmarshal(out, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round.RequestID != "agent-7:turn-12" {
		t.Fatalf("re-encoded requestId = %q, want %q", round.RequestID, "agent-7:turn-12")
	}
}

func TestParseLifecycleRejectsMissingFields(t *testing.T) {
	cases := map[string]string{
		"no workloadInfo":      `{}`,
		"empty id":             `{"workloadInfo":{"id":"","model":"m","engine":"e","state":"queued","originatedFrom":"n"}}`,
		"empty engine":         `{"workloadInfo":{"id":"x","model":"m","engine":"","state":"queued","originatedFrom":"n"}}`,
		"empty state":          `{"workloadInfo":{"id":"x","model":"m","engine":"e","state":"","originatedFrom":"n"}}`,
		"empty originatedFrom": `{"workloadInfo":{"id":"x","model":"m","engine":"e","state":"queued","originatedFrom":""}}`,
	}
	for name, body := range cases {
		if _, err := parseLifecycle(json.RawMessage(body)); err == nil {
			t.Errorf("%s: expected error, got nil", name)
		}
	}
}

func TestParseRemove(t *testing.T) {
	id, node, err := parseRemove(json.RawMessage(`{"workloadId":"wl-9","originatedFrom":"node-C"}`))
	if err != nil || id != "wl-9" || node != "node-C" {
		t.Fatalf("expected wl-9/node-C, got %q/%q err=%v", id, node, err)
	}
	// originatedFrom is optional (backward compatible): a legacy payload
	// without it still parses, with an empty originatedFrom.
	id, node, err = parseRemove(json.RawMessage(`{"workloadId":"wl-9"}`))
	if err != nil || id != "wl-9" || node != "" {
		t.Fatalf("expected wl-9/empty, got %q/%q err=%v", id, node, err)
	}
	if _, _, err := parseRemove(json.RawMessage(`{"workloadId":""}`)); err == nil {
		t.Fatal("expected error for empty workloadId")
	}
}

func TestLifecycleMethodMapping(t *testing.T) {
	for method, want := range map[string]WorkloadState{
		MethodSubmitted: StateQueued,
		MethodStarted:   StateRunning,
		MethodCompleted: StateCompleted,
		MethodErrored:   StateFailed,
	} {
		if !isLifecycleMethod(method) {
			t.Errorf("%s should be a lifecycle method", method)
		}
		if lifecycleMethods[method] != want {
			t.Errorf("%s mapped to %q, want %q", method, lifecycleMethods[method], want)
		}
	}
	if isLifecycleMethod(MethodRemove) {
		t.Error("workloads:remove is not a lifecycle method")
	}
	if isLifecycleMethod("bogus:method") {
		t.Error("unknown method should not be a lifecycle method")
	}
}
