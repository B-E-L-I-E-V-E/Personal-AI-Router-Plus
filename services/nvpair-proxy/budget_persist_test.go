// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"testing"
)

// What is configured survives a restart: the reply budget in the capacity
// settings and in saved plans, with the rest of each plan.

func TestCapacitySettingsKeepTheReplyBudgetAcrossRestarts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "context-capacity.json")
	k := capacityKey{Engine: "ollama", Model: "m"}

	first := newCapacityStore()
	first.path = path
	if err := first.set(k, capacitySetting{ContextTokens: 16384, MaxSessions: 2, ReplyTokens: 2048}, false); err != nil {
		t.Fatal(err)
	}

	second := newCapacityStore()
	second.path = path
	if err := second.load(); err != nil {
		t.Fatal(err)
	}
	got := second.settingsSnapshot()[k]
	if got.ContextTokens != 16384 || got.MaxSessions != 2 || got.ReplyTokens != 2048 {
		t.Fatalf("after restart = %+v, want 16384 / 2 / reply 2048", got)
	}
	if row := effective(k, &got, nil, localUsage{}); row.MessageTokens != 8192-2048 {
		t.Fatalf("message tokens after restart = %d, want %d", row.MessageTokens, 8192-2048)
	}
}

func TestSavedPlansKeepEverySettingAcrossRestarts(t *testing.T) {
	path := filepath.Join(t.TempDir(), plannerFileName)
	first := newPlannerStore()
	first.path = path
	if err := first.setAutoReload(false); err != nil {
		t.Fatal(err)
	}
	want := savedPlan{Engine: "llamacpp", Model: "moe", WindowTokens: 32768, Sessions: 3, CacheType: "q8_0",
		ReservePercent: 15, OffloadExperts: true, ReplyTokens: 4096}
	if err := first.put(want); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := first.markApplied("llamacpp", "moe", 32768, 3, "q8_0", true); !ok || err != nil {
		t.Fatalf("markApplied: %v %v", ok, err)
	}

	second := newPlannerStore()
	second.path = path
	if err := second.load(); err != nil {
		t.Fatal(err)
	}
	st := second.snapshot()
	if st.AutoReload || len(st.Plans) != 1 {
		t.Fatalf("after restart = %+v, want auto reload off and one plan", st)
	}
	got := st.Plans[0]
	if got.WindowTokens != 32768 || got.Sessions != 3 || got.CacheType != "q8_0" || got.ReservePercent != 15 ||
		!got.OffloadExperts || got.ReplyTokens != 4096 || !got.Applied {
		t.Fatalf("plan after restart = %+v, want every setting kept and still applied", got)
	}
}

func TestSavedPlansDropInvalidEntriesOnLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), plannerFileName)
	// A reply budget as large as the window cannot be honoured; the plan is
	// dropped rather than applied half right.
	body := `{"autoReload":true,"plans":[
		{"engine":"llamacpp","model":"ok","windowTokens":8192,"sessions":1,"cacheType":"f16","replyTokens":2048},
		{"engine":"llamacpp","model":"bad","windowTokens":8192,"sessions":1,"cacheType":"f16","replyTokens":8192}]}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	s := newPlannerStore()
	s.path = path
	if err := s.load(); err != nil {
		t.Fatal(err)
	}
	if plans := s.snapshot().Plans; len(plans) != 1 || plans[0].Model != "ok" {
		t.Fatalf("plans = %+v, want only the valid one", plans)
	}
}
