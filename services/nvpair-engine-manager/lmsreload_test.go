// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

const lmsLoadHelp = `Usage: lms load [options] [model-key]
   -c, --context-length <length>
   --parallel <count>
   --identifier <identifier>
   -y, --yes`

// fakeLMS answers the lms commands lmsReload runs and records every command.
// Loading a model replaces its listing with the requested numbers, unless
// failLoad says that load fails.
type fakeLMS struct {
	help     string
	loaded   []lmsLoadedInstance
	failLoad func(contextLength int64) bool
	commands []string
}

func (f *fakeLMS) run(_ context.Context, argv []string) (string, error) {
	f.commands = append(f.commands, strings.Join(argv[1:], " "))
	switch {
	case argv[1] == "load" && argv[2] == "--help":
		return f.help, nil
	case argv[1] == "ps":
		data, err := json.Marshal(f.loaded)
		return string(data), err
	case argv[1] == "unload":
		kept := f.loaded[:0]
		for _, m := range f.loaded {
			if m.Identifier != argv[2] {
				kept = append(kept, m)
			}
		}
		f.loaded = kept
		return "", nil
	case argv[1] == "load":
		ctxLen, err := strconv.ParseInt(argv[4], 10, 64)
		if err != nil {
			return "", err
		}
		parallel, err := strconv.Atoi(argv[6])
		if err != nil {
			return "", err
		}
		if f.failLoad != nil && f.failLoad(ctxLen) {
			return "", errors.New("not enough memory")
		}
		f.loaded = append(f.loaded, lmsLoadedInstance{ModelKey: argv[2], Identifier: argv[8], ContextLength: ctxLen, Parallel: parallel})
		return "", nil
	}
	return "", errors.New("unexpected command " + strings.Join(argv, " "))
}

func planRequest(contextLength int64, parallel int) modelReloadRequest {
	return modelReloadRequest{Engine: "lmstudio", Model: "glm-4.7-flash", ContextLength: contextLength, Parallel: parallel}
}

func TestLMSReload_UnloadsAndLoadsWithPlannedNumbers(t *testing.T) {
	lms := &fakeLMS{help: lmsLoadHelp, loaded: []lmsLoadedInstance{
		{ModelKey: "glm-4.7-flash", Identifier: "glm-4.7-flash", ContextLength: 8192, Parallel: 4},
	}}

	got, err := lmsReload(context.Background(), lms.run, "lms", planRequest(131072, 5))
	if err != nil {
		t.Fatalf("lmsReload: %v", err)
	}

	want := modelReloadResult{Outcome: reloadOutcomeReloaded, Model: "glm-4.7-flash", Identifier: "glm-4.7-flash", ContextLength: 131072, Parallel: 5}
	if got != want {
		t.Fatalf("result = %+v, want %+v", got, want)
	}
	wantCommands := []string{
		"load --help",
		"ps --json",
		"unload glm-4.7-flash",
		"load glm-4.7-flash --context-length 131072 --parallel 5 --identifier glm-4.7-flash --yes",
		"ps --json",
	}
	if !reflect.DeepEqual(lms.commands, wantCommands) {
		t.Fatalf("commands = %q, want %q", lms.commands, wantCommands)
	}
}

func TestLMSReload_LoadsAModelThatIsNotLoaded(t *testing.T) {
	lms := &fakeLMS{help: lmsLoadHelp}

	got, err := lmsReload(context.Background(), lms.run, "lms", planRequest(32768, 2))
	if err != nil {
		t.Fatalf("lmsReload: %v", err)
	}

	if got.Outcome != reloadOutcomeReloaded || got.ContextLength != 32768 || got.Parallel != 2 {
		t.Fatalf("result = %+v, want reloaded with 32768 and 2", got)
	}
	for _, c := range lms.commands {
		if strings.HasPrefix(c, "unload") {
			t.Fatalf("unloaded although nothing was loaded: %q", lms.commands)
		}
	}
}

func TestLMSReload_LeavesAMatchingLoadAlone(t *testing.T) {
	lms := &fakeLMS{help: lmsLoadHelp, loaded: []lmsLoadedInstance{
		{ModelKey: "glm-4.7-flash", Identifier: "glm-4.7-flash", ContextLength: 131072, Parallel: 5},
	}}

	got, err := lmsReload(context.Background(), lms.run, "lms", planRequest(131072, 5))
	if err != nil {
		t.Fatalf("lmsReload: %v", err)
	}

	if got.Outcome != reloadOutcomeReloaded {
		t.Fatalf("outcome = %q, want %q", got.Outcome, reloadOutcomeReloaded)
	}
	wantCommands := []string{"load --help", "ps --json"}
	if !reflect.DeepEqual(lms.commands, wantCommands) {
		t.Fatalf("commands = %q, want %q", lms.commands, wantCommands)
	}
}

func TestLMSReload_UnloadsEveryInstanceOfTheModelKey(t *testing.T) {
	lms := &fakeLMS{help: lmsLoadHelp, loaded: []lmsLoadedInstance{
		{ModelKey: "glm-4.7-flash", Identifier: "glm-4.7-flash:2", ContextLength: 4096, Parallel: 1},
		{ModelKey: "glm-4.7-flash", Identifier: "glm-4.7-flash:3", ContextLength: 4096, Parallel: 1},
		{ModelKey: "qwen/qwen3.8-27b", Identifier: "qwen/qwen3.8-27b", ContextLength: 8192, Parallel: 4},
	}}

	if _, err := lmsReload(context.Background(), lms.run, "lms", planRequest(65536, 3)); err != nil {
		t.Fatalf("lmsReload: %v", err)
	}

	wantLoaded := []lmsLoadedInstance{
		{ModelKey: "qwen/qwen3.8-27b", Identifier: "qwen/qwen3.8-27b", ContextLength: 8192, Parallel: 4},
		{ModelKey: "glm-4.7-flash", Identifier: "glm-4.7-flash", ContextLength: 65536, Parallel: 3},
	}
	if !reflect.DeepEqual(lms.loaded, wantLoaded) {
		t.Fatalf("loaded = %+v, want %+v", lms.loaded, wantLoaded)
	}
}

func TestLMSReload_ReportsUnsupportedWithoutTouchingTheModel(t *testing.T) {
	test := func(name, cli, help, wantReason string) {
		t.Run(name, func(t *testing.T) {
			lms := &fakeLMS{help: help, loaded: []lmsLoadedInstance{
				{ModelKey: "glm-4.7-flash", Identifier: "glm-4.7-flash", ContextLength: 8192, Parallel: 4},
			}}

			got, err := lmsReload(context.Background(), lms.run, cli, planRequest(131072, 5))
			if err != nil {
				t.Fatalf("lmsReload: %v", err)
			}

			if got.Outcome != reloadOutcomeUnsupported {
				t.Fatalf("outcome = %q, want %q", got.Outcome, reloadOutcomeUnsupported)
			}
			if !strings.Contains(got.Reason, wantReason) {
				t.Fatalf("reason = %q, want it to mention %q", got.Reason, wantReason)
			}
			for _, c := range lms.commands {
				if c != "load --help" {
					t.Fatalf("ran %q although the reload is unsupported", c)
				}
			}
		})
	}

	test("no lms command", "", lmsLoadHelp, "no lms command")
	test("lms without parallel", "lms", "Usage: lms load\n  -c, --context-length <length>\n  --identifier <identifier>", "--parallel")
}

func TestLMSReload_RestoresThePreviousLoadWhenThePlanFailsToLoad(t *testing.T) {
	lms := &fakeLMS{
		help:     lmsLoadHelp,
		loaded:   []lmsLoadedInstance{{ModelKey: "glm-4.7-flash", Identifier: "glm-4.7-flash", ContextLength: 8192, Parallel: 4}},
		failLoad: func(contextLength int64) bool { return contextLength == 131072 },
	}

	_, err := lmsReload(context.Background(), lms.run, "lms", planRequest(131072, 5))

	if err == nil || !strings.Contains(err.Error(), "loaded again with its previous context length 8192") {
		t.Fatalf("err = %v, want the failed load and the restore reported", err)
	}
	wantLoaded := []lmsLoadedInstance{{ModelKey: "glm-4.7-flash", Identifier: "glm-4.7-flash", ContextLength: 8192, Parallel: 4}}
	if !reflect.DeepEqual(lms.loaded, wantLoaded) {
		t.Fatalf("loaded = %+v, want the previous load back", lms.loaded)
	}
}

func TestModelReloadRequest_RejectsBadNumbers(t *testing.T) {
	test := func(name string, req modelReloadRequest, wantErr string) {
		t.Run(name, func(t *testing.T) {
			err := req.validate()
			if err == nil || !strings.Contains(err.Error(), wantErr) {
				t.Fatalf("validate() = %v, want an error mentioning %q", err, wantErr)
			}
		})
	}

	test("other engine", modelReloadRequest{Engine: "ollama", Model: "m", ContextLength: 4096, Parallel: 1}, "only for lmstudio")
	test("no model", modelReloadRequest{Engine: "lmstudio", ContextLength: 4096, Parallel: 1}, "model is required")
	test("no context length", modelReloadRequest{Engine: "lmstudio", Model: "m", Parallel: 1}, "contextLength")
	test("too many predictions", modelReloadRequest{Engine: "lmstudio", Model: "m", ContextLength: 4096, Parallel: 65}, "parallel")
}
