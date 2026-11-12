// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

// engineServer answers each path with the JSON value given for it.
func engineServer(t *testing.T, answers map[string]any) (*specClient, *url.URL) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		answer, ok := answers[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(answer)
	}))
	t.Cleanup(srv.Close)
	base, _ := url.Parse(srv.URL)
	return &specClient{client: srv.Client()}, base
}

func wantStart(t *testing.T, name string, got *startValue, value int64, source string) {
	t.Helper()
	if value == 0 {
		if got != nil {
			t.Fatalf("%s: want none, got %+v", name, *got)
		}
		return
	}
	if got == nil || got.Value != value || got.Source != source {
		t.Fatalf("%s: want %d from %q, got %+v", name, value, source, got)
	}
}

const qwen3Info = `{"general.architecture":"qwen3","qwen3.block_count":28,"qwen3.context_length":40960,
	"qwen3.attention.head_count":16,"qwen3.attention.head_count_kv":8,
	"qwen3.attention.key_length":128,"qwen3.attention.value_length":128}`

func ollamaShow(t *testing.T, parameters string) map[string]any {
	t.Helper()
	var info map[string]any
	if err := json.Unmarshal([]byte(qwen3Info), &info); err != nil {
		t.Fatal(err)
	}
	return map[string]any{"parameters": parameters, "model_info": info}
}

func TestOllamaParameters_ReadsNumbersAndSkipsText(t *testing.T) {
	got := ollamaParameters("num_ctx                        8192\nstop                           \"<|im_end|>\"\ntemperature                    0.6\nnum_predict                    -1\n")
	if len(got) != 2 || got["num_ctx"] != 8192 || got["num_predict"] != -1 {
		t.Fatalf("got %v", got)
	}
}

func TestDescribeOllama_StartsFromLoadedWindowThenModelParameters(t *testing.T) {
	cases := []struct {
		name       string
		parameters string
		ps         []any
		window     int64
		windowFrom string
		reply      int64
	}{
		{
			name:       "loaded copy wins over num_ctx",
			parameters: "num_ctx 8192\nnum_predict 2048",
			ps:         []any{map[string]any{"name": "qwen3:0.6b", "size": 1, "context_length": 16384}},
			window:     16384, windowFrom: "engine", reply: 2048,
		},
		{
			name:       "num_ctx when the model is not loaded",
			parameters: "num_ctx 8192",
			ps:         []any{map[string]any{"name": "other:1b", "size": 1, "context_length": 4096}},
			window:     8192, windowFrom: "model",
		},
		{
			name:       "nothing reported leaves no defaults",
			parameters: "temperature 0.6\nnum_predict -1",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, base := engineServer(t, map[string]any{
				"/api/show": ollamaShow(t, tc.parameters),
				"/api/tags": map[string]any{"models": []any{map[string]any{"name": "qwen3:0.6b", "size": 500 << 20}}},
				"/api/ps":   map[string]any{"models": tc.ps},
			})
			s, err := c.describe(context.Background(), "ollama", base, "qwen3:0.6b", 0)
			if err != nil || s.Source != "metadata" {
				t.Fatalf("%v %+v", err, s)
			}
			if tc.window == 0 && tc.reply == 0 {
				if s.Defaults != nil {
					t.Fatalf("want no defaults, got %+v", *s.Defaults)
				}
				return
			}
			wantStart(t, "window", s.Defaults.Window, tc.window, tc.windowFrom)
			wantStart(t, "sessions", s.Defaults.Sessions, 0, "")
			wantStart(t, "reply", s.Defaults.Reply, tc.reply, "model")
		})
	}
}

func TestDescribeLlamaCPP_StartsFromServerSlotsAndWindow(t *testing.T) {
	cases := []struct {
		name  string
		props map[string]any
		reply int64
	}{
		{
			name: "current server",
			props: map[string]any{"total_slots": 4, "default_generation_settings": map[string]any{
				"n_ctx": 8192, "params": map[string]any{"n_predict": 1024}}},
			reply: 1024,
		},
		{
			name: "older server with top-level n_ctx and unlimited replies",
			props: map[string]any{"total_slots": 4, "n_ctx": 8192, "default_generation_settings": map[string]any{
				"n_predict": -1}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, base := engineServer(t, map[string]any{"/props": tc.props})
			s, err := c.describe(context.Background(), "llamacpp", base, "m", 0)
			if err != nil || s.Defaults == nil {
				t.Fatalf("%v %+v", err, s)
			}
			wantStart(t, "window", s.Defaults.Window, 8192, "engine")
			wantStart(t, "sessions", s.Defaults.Sessions, 4, "engine")
			wantStart(t, "reply", s.Defaults.Reply, tc.reply, "engine")
		})
	}
}

func TestDescribeLMStudio_StartsFromLoadedInstance(t *testing.T) {
	c, base := engineServer(t, map[string]any{"/api/v1/models": map[string]any{"models": []any{
		map[string]any{"key": "loaded", "size_bytes": 1 << 30, "max_context_length": 131072,
			"loaded_instances": []any{map[string]any{"id": "loaded", "config": map[string]any{"context_length": 32768, "parallel": 4}}}},
		map[string]any{"key": "idle", "size_bytes": 1 << 30, "max_context_length": 131072, "loaded_instances": []any{}},
	}}})
	c.homeDir = func() (string, error) { return t.TempDir(), nil }

	s, err := c.describe(context.Background(), "lmstudio", base, "loaded", 0)
	if err != nil || s.Defaults == nil {
		t.Fatalf("%v %+v", err, s)
	}
	wantStart(t, "window", s.Defaults.Window, 32768, "engine")
	wantStart(t, "sessions", s.Defaults.Sessions, 4, "engine")
	wantStart(t, "reply", s.Defaults.Reply, 0, "")

	s, err = c.describe(context.Background(), "lmstudio", base, "idle", 0)
	if err != nil || s.Defaults != nil {
		t.Fatalf("an unloaded model reports no defaults: %v %+v", err, s.Defaults)
	}
}
