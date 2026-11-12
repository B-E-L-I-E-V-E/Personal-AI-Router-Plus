// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"nvpair-shared/engines"
)

// The benchmark (spec §5.16) sends a fixed set of prompts for one model
// through this node's own facade, so it is routed, admitted and measured
// exactly like real traffic, and records the same figures. Runs one at a time,
// one request at a time, and waits for the node to be idle before each
// request, so it never competes with a real session for the model.
const (
	// benchmarkHeader marks a request as the benchmark's, so its record goes
	// to the run instead of the live figures.
	benchmarkHeader = "X-PAIR-Benchmark"
	// benchmarkIdleWait bounds how long a step waits for other requests to
	// finish before it is skipped.
	benchmarkIdleWait = 2 * time.Minute
	// benchmarkStepTimeout bounds one benchmark request, model load included.
	benchmarkStepTimeout = 5 * time.Minute
	// benchmarkIdlePoll is how often the idle check looks again.
	benchmarkIdlePoll = 250 * time.Millisecond
)

// benchmarkPrompt is one step of the benchmark. The set is fixed so runs on
// different models and nodes compare like for like.
type benchmarkPrompt struct {
	Name        string `json:"name"`
	Label       string `json:"label"`
	Description string `json:"description"`
	// Thinking is "off", "on", or "default" where the engine has no switch.
	Thinking  string `json:"thinking"`
	MaxTokens int    `json:"maxTokens"`
	prompt    func() string
}

// benchmarkReport is the long-input step's document: about a thousand tokens
// of fixed, plain text.
func benchmarkReport() string {
	sections := []string{
		"The harbor authority recorded steady traffic through the spring, with most vessels arriving on schedule and leaving within a day.",
		"Cargo volumes rose slightly compared with the previous quarter, driven mainly by grain and timber shipments heading north.",
		"Two of the four cranes on the east quay were serviced in turn, so loading continued at three quarters of normal capacity.",
		"Weather delays were limited to a single storm that closed the outer channel for most of an afternoon.",
		"The pilot service added a second boat, which shortened waiting times at the anchorage by roughly a third.",
		"Staff completed safety training ahead of the summer season, and no injuries were reported during the period.",
		"A survey of the inner basin found silt building up near the old ferry ramp, and dredging is planned for the autumn.",
		"Fuel sales to fishing boats held level, while sales to visiting yachts grew as the marina extension opened.",
	}
	var b strings.Builder
	for round := 1; round <= 5; round++ {
		for i, s := range sections {
			fmt.Fprintf(&b, "Section %d.%d. %s\n", round, i+1, s)
		}
	}
	return b.String()
}

var benchmarkPrompts = []benchmarkPrompt{
	{
		Name: "short", Label: "Short prompt", Thinking: "off", MaxTokens: 64,
		Description: "A one-line question with a one-sentence answer: time to first token and overhead.",
		prompt:      func() string { return "In one sentence, what is the capital of France?" },
	},
	{
		Name: "long-input", Label: "Long input", Thinking: "off", MaxTokens: 128,
		Description: "About a thousand tokens of report to summarize in two sentences: reading speed.",
		prompt: func() string {
			return benchmarkReport() + "\nSummarize the report above in two sentences."
		},
	},
	{
		Name: "long-output", Label: "Long output", Thinking: "off", MaxTokens: 400,
		Description: "A short request for a long reply of up to 400 tokens: writing speed.",
		prompt: func() string {
			return "Write a story of about 300 words about a lighthouse keeper who finds a message in a bottle."
		},
	},
	{
		Name: "thinking", Label: "Thinking", Thinking: "on", MaxTokens: 1024,
		Description: "A short reasoning puzzle with thinking on, where the model supports it: time spent thinking.",
		prompt: func() string {
			return "A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball. How much does the ball cost? Answer with the amount."
		},
	},
}

func benchmarkPromptList() []benchmarkPrompt {
	return append([]benchmarkPrompt(nil), benchmarkPrompts...)
}

// Benchmark run and step states.
const (
	benchPending = "pending"
	benchRunning = "running"
	benchDone    = "done"
	benchFailed  = "failed"
	benchSkipped = "skipped"
)

type benchmarkStep struct {
	Name     string      `json:"name"`
	Label    string      `json:"label"`
	Thinking string      `json:"thinking"`
	State    string      `json:"state"`
	Detail   string      `json:"detail,omitempty"`
	Result   *perfRecord `json:"result,omitempty"`
}

type benchmarkRun struct {
	ID         string          `json:"id"`
	Engine     string          `json:"engine"`
	Model      string          `json:"model"`
	State      string          `json:"state"`
	StartedAt  int64           `json:"startedAt"`
	FinishedAt int64           `json:"finishedAt,omitempty"`
	Error      string          `json:"error,omitempty"`
	Steps      []benchmarkStep `json:"steps"`
}

var errBenchmarkRunning = errors.New("a benchmark is already running; wait for it to finish")

func (s *perfStore) benchmarkSnapshot() []benchmarkRun {
	s.benchMu.Lock()
	defer s.benchMu.Unlock()
	out := make([]benchmarkRun, 0, len(s.runs))
	for i := len(s.runs) - 1; i >= 0; i-- {
		r := *s.runs[i]
		r.Steps = append([]benchmarkStep(nil), r.Steps...)
		out = append(out, r)
	}
	return out
}

// updateRun applies a change to a run under the store's lock.
func (s *perfStore) updateRun(run *benchmarkRun, change func(*benchmarkRun)) {
	s.benchMu.Lock()
	change(run)
	s.benchMu.Unlock()
}

// beginRun registers a new run, refusing while another is in progress.
func (s *perfStore) beginRun(engine, model string) (*benchmarkRun, error) {
	s.benchMu.Lock()
	defer s.benchMu.Unlock()
	if s.running != nil {
		return nil, errBenchmarkRunning
	}
	run := &benchmarkRun{
		ID:        "bench-" + newRequestID()[:8],
		Engine:    engine,
		Model:     model,
		State:     benchRunning,
		StartedAt: time.Now().UnixMilli(),
	}
	for _, p := range benchmarkPrompts {
		thinking := p.Thinking
		if engine != engines.NameOllama && engine != engines.NameLlamaCPP {
			thinking = "default"
		}
		run.Steps = append(run.Steps, benchmarkStep{Name: p.Name, Label: p.Label, Thinking: thinking, State: benchPending})
	}
	s.running = run
	s.runs = append(s.runs, run)
	if len(s.runs) > benchmarkHistory {
		s.runs = append([]*benchmarkRun(nil), s.runs[len(s.runs)-benchmarkHistory:]...)
	}
	return run, nil
}

func (s *perfStore) endRun(run *benchmarkRun, state, errText string) {
	s.benchMu.Lock()
	run.State, run.Error, run.FinishedAt = state, errText, time.Now().UnixMilli()
	if s.running == run {
		s.running = nil
	}
	s.benchMu.Unlock()
}

func (s *perfStore) wait(requestID string) chan perfRecord {
	ch := make(chan perfRecord, 1)
	s.benchMu.Lock()
	s.waiters[requestID] = ch
	s.benchMu.Unlock()
	return ch
}

func (s *perfStore) unwait(requestID string) {
	s.benchMu.Lock()
	delete(s.waiters, requestID)
	s.benchMu.Unlock()
}

type benchmarkParams struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
}

// startBenchmark validates a request and starts the run in the background.
func (p *Proxy) startBenchmark(params benchmarkParams) (benchmarkRun, error) {
	params.Model = strings.TrimSpace(params.Model)
	if params.Model == "" {
		return benchmarkRun{}, errors.New("model is required")
	}
	f := p.facadeFor(params.Engine)
	if f == nil {
		return benchmarkRun{}, fmt.Errorf("engine %q has no facade on this node", params.Engine)
	}
	port, _ := f.selfAddresses()
	if port <= 0 {
		return benchmarkRun{}, fmt.Errorf("engine %q is not listening yet", params.Engine)
	}
	run, err := p.perf.beginRun(params.Engine, params.Model)
	if err != nil {
		return benchmarkRun{}, err
	}
	slog.Info("benchmark started", "run", run.ID, "engine", params.Engine, "model", params.Model)
	snapshot := *run
	snapshot.Steps = append([]benchmarkStep(nil), run.Steps...)
	go p.runBenchmark(run, f.profile.Name, port)
	return snapshot, nil
}

func (p *Proxy) runBenchmark(run *benchmarkRun, engine string, port int) {
	base := "http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(port))
	client := &http.Client{Timeout: benchmarkStepTimeout}
	failed := 0
	for i, prompt := range benchmarkPrompts {
		p.perf.updateRun(run, func(r *benchmarkRun) { r.Steps[i].State = benchRunning })
		if !p.waitIdle(benchmarkIdleWait) {
			p.perf.updateRun(run, func(r *benchmarkRun) {
				r.Steps[i].State, r.Steps[i].Detail = benchSkipped, "the node stayed busy with other requests"
			})
			continue
		}
		rec, state, detail := p.benchmarkStep(client, base, engine, run, i, prompt)
		if state == benchFailed {
			failed++
		}
		p.perf.updateRun(run, func(r *benchmarkRun) {
			r.Steps[i].State, r.Steps[i].Detail, r.Steps[i].Result = state, detail, rec
		})
	}
	state, errText := benchDone, ""
	if failed == len(benchmarkPrompts) {
		state, errText = benchFailed, "every step failed"
	}
	p.perf.endRun(run, state, errText)
	slog.Info("benchmark finished", "run", run.ID, "engine", run.Engine, "model", run.Model, "state", state)
}

// waitIdle waits until no request is in progress on this node.
func (p *Proxy) waitIdle(limit time.Duration) bool {
	deadline := time.Now().Add(limit)
	for p.activeRequests.Load() > 0 {
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(benchmarkIdlePoll)
	}
	return true
}

// benchmarkStep sends one prompt and returns its record. A model that refuses
// the thinking switch is retried without it for an "off" step, and an "on"
// step it refuses is skipped as unsupported.
func (p *Proxy) benchmarkStep(client *http.Client, base, engine string, run *benchmarkRun, i int, prompt benchmarkPrompt) (*perfRecord, string, string) {
	thinking := run.Steps[i].Thinking
	requestID := fmt.Sprintf("%s-%d-%s", run.ID, i+1, prompt.Name)
	ch := p.perf.wait(requestID)
	defer p.perf.unwait(requestID)

	path, body := benchmarkRequest(engine, run.ID, run.Model, prompt, thinking)
	status, errText := postBenchmark(client, base+path, requestID, body)
	if status >= 400 && status < 500 && thinking != "default" {
		if thinking == "on" {
			return nil, benchSkipped, "the model does not support thinking"
		}
		_, body = benchmarkRequest(engine, run.ID, run.Model, prompt, "default")
		status, errText = postBenchmark(client, base+path, requestID, body)
	}
	if errText != "" {
		return nil, benchFailed, errText
	}
	select {
	case rec := <-ch:
		return &rec, benchDone, ""
	case <-time.After(5 * time.Second):
		return nil, benchFailed, "the engine reported no figures for the reply"
	}
}

func postBenchmark(client *http.Client, url, requestID string, body []byte) (int, string) {
	ctx, cancel := context.WithTimeout(context.Background(), benchmarkStepTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return 0, err.Error()
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(requestIDHeader, requestID)
	req.Header.Set(benchmarkHeader, "1")
	resp, err := client.Do(req)
	if err != nil {
		return 0, err.Error()
	}
	defer resp.Body.Close()
	// The reply is read to the end so the proxy measures all of it, and
	// discarded: the benchmark keeps the figures, never the text.
	_, copyErr := io.Copy(io.Discard, resp.Body)
	if resp.StatusCode >= 400 {
		return resp.StatusCode, fmt.Sprintf("the engine answered %d", resp.StatusCode)
	}
	if copyErr != nil {
		return resp.StatusCode, copyErr.Error()
	}
	return resp.StatusCode, ""
}

// benchmarkRequest builds one step's streamed request in the engine's own
// dialect. Every prompt starts with the run ID, so no engine can answer it
// from a prompt cache left by an earlier run.
func benchmarkRequest(engine, runID, model string, prompt benchmarkPrompt, thinking string) (string, []byte) {
	content := "Benchmark run " + runID + ".\n" + prompt.prompt()
	messages := []map[string]string{{"role": "user", "content": content}}
	if engine == engines.NameOllama {
		body := map[string]any{
			"model":    model,
			"messages": messages,
			"stream":   true,
			"options":  map[string]any{"temperature": 0, "seed": 7, "num_predict": prompt.MaxTokens},
		}
		if thinking != "default" {
			body["think"] = thinking == "on"
		}
		raw, _ := json.Marshal(body)
		return "/api/chat", raw
	}
	body := map[string]any{
		"model":          model,
		"messages":       messages,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
		"max_tokens":     prompt.MaxTokens,
		"temperature":    0,
		"seed":           7,
	}
	if engine == engines.NameLlamaCPP && thinking != "default" {
		body["chat_template_kwargs"] = map[string]any{"enable_thinking": thinking == "on"}
	}
	raw, _ := json.Marshal(body)
	return "/v1/chat/completions", raw
}

func (f *facade) serveBenchmarkStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, "POST")
		return
	}
	raw, err := io.ReadAll(io.LimitReader(r.Body, maxSessionAPIBody))
	if err != nil {
		writeSessionJSON(w, http.StatusBadRequest, map[string]string{"error": "read body: " + err.Error()})
		return
	}
	var params benchmarkParams
	if err := json.Unmarshal(raw, &params); err != nil {
		writeSessionJSON(w, http.StatusBadRequest, map[string]string{"error": `expected {"model"}`})
		return
	}
	params.Engine = f.profile.Name
	run, err := f.host.startBenchmark(params)
	if errors.Is(err, errBenchmarkRunning) {
		writeSessionJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
		return
	}
	if err != nil {
		writeSessionJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	writeSessionJSON(w, http.StatusAccepted, run)
}
