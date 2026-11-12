// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
)

// The performance store keeps measurements in memory only: a restart starts
// it empty. Each bound below caps what one node can hold however much traffic
// it routes.
const (
	// perfWindow is how many recent requests per model and node the averages
	// and percentiles cover.
	perfWindow = 100
	// perfRecentLimit bounds the recent-request list across all models.
	perfRecentLimit = 200
	// perfTrendPoints is how many of a model's latest requests its trend shows.
	perfTrendPoints = 30
	// perfReportRecent is how many recent requests a report returns.
	perfReportRecent = 50
	// benchmarkHistory bounds the benchmark runs kept.
	benchmarkHistory = 20
)

// perfScope tells a reader which requests the report covers: those this
// node's proxy routed, whichever node served them. Requests other nodes route
// are measured by their own proxies.
const perfScope = "routed-by-this-node"

type perfKey struct {
	Engine string
	Model  string
	NodeID string
}

type perfStore struct {
	mu     sync.Mutex
	byKey  map[perfKey][]perfRecord
	local  map[perfKey]bool
	recent []perfRecord

	benchMu sync.Mutex
	runs    []*benchmarkRun
	running *benchmarkRun
	// waiters receive the record of a benchmark request, keyed by the request
	// ID the runner chose.
	waiters map[string]chan perfRecord
}

func newPerfStore() *perfStore {
	return &perfStore{
		byKey:   make(map[perfKey][]perfRecord),
		local:   make(map[perfKey]bool),
		waiters: make(map[string]chan perfRecord),
	}
}

// add files a completed request. A benchmark request goes to the run waiting
// for it and stays out of the live figures, so a benchmark never skews what
// real traffic looks like.
func (s *perfStore) add(rec perfRecord, benchmark bool) {
	if benchmark {
		s.benchMu.Lock()
		ch := s.waiters[rec.RequestID]
		s.benchMu.Unlock()
		if ch != nil {
			select {
			case ch <- rec:
			default:
			}
		}
		return
	}
	k := perfKey{Engine: rec.Engine, Model: rec.Model, NodeID: rec.NodeID}
	s.mu.Lock()
	defer s.mu.Unlock()
	list := append(s.byKey[k], rec)
	if len(list) > perfWindow {
		list = append([]perfRecord(nil), list[len(list)-perfWindow:]...)
	}
	s.byKey[k] = list
	s.local[k] = rec.Local
	s.recent = append(s.recent, rec)
	if len(s.recent) > perfRecentLimit {
		s.recent = append([]perfRecord(nil), s.recent[len(s.recent)-perfRecentLimit:]...)
	}
}

func (s *perfStore) clear() {
	s.mu.Lock()
	s.byKey = make(map[perfKey][]perfRecord)
	s.local = make(map[perfKey]bool)
	s.recent = nil
	s.mu.Unlock()
	s.benchMu.Lock()
	kept := s.runs[:0]
	for _, r := range s.runs {
		if r == s.running {
			kept = append(kept, r)
		}
	}
	s.runs = kept
	s.benchMu.Unlock()
}

// perfStat summarizes one figure across a model's window.
type perfStat struct {
	Count int     `json:"count"`
	Avg   float64 `json:"avg"`
	P50   float64 `json:"p50"`
	P95   float64 `json:"p95"`
	Min   float64 `json:"min"`
	Max   float64 `json:"max"`
}

func summarize(values []float64) *perfStat {
	if len(values) == 0 {
		return nil
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	sum := 0.0
	for _, v := range sorted {
		sum += v
	}
	round := func(v float64) float64 { return math.Round(v*100) / 100 }
	return &perfStat{
		Count: len(sorted),
		Avg:   round(sum / float64(len(sorted))),
		P50:   round(percentile(sorted, 0.50)),
		P95:   round(percentile(sorted, 0.95)),
		Min:   round(sorted[0]),
		Max:   round(sorted[len(sorted)-1]),
	}
}

// percentile reads a sorted slice by linear interpolation between the
// nearest ranks.
func percentile(sorted []float64, q float64) float64 {
	if len(sorted) == 1 {
		return sorted[0]
	}
	pos := q * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	return sorted[lo] + (sorted[hi]-sorted[lo])*(pos-float64(lo))
}

// perfTrendPoint is one request in a model's trend.
type perfTrendPoint struct {
	At         int64    `json:"at"`
	ReadTPS    *float64 `json:"readTokensPerSecond,omitempty"`
	WriteTPS   *float64 `json:"writeTokensPerSecond,omitempty"`
	TTFTMs     *float64 `json:"timeToFirstTokenMs,omitempty"`
	ThinkingMs *float64 `json:"thinkingMs,omitempty"`
}

// perfModelSummary is one model on one node, over its last perfWindow
// requests.
type perfModelSummary struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
	NodeID string `json:"nodeId"`
	Local  bool   `json:"local"`
	// Requests is how many requests the figures cover (at most perfWindow).
	Requests         int   `json:"requests"`
	PromptTokens     int64 `json:"promptTokens"`
	CompletionTokens int64 `json:"completionTokens"`
	ThinkingRequests int   `json:"thinkingRequests"`
	ThinkingTokens   int64 `json:"thinkingTokens"`
	LastAt           int64 `json:"lastAt"`

	ReadTPS    *perfStat `json:"readTokensPerSecond,omitempty"`
	WriteTPS   *perfStat `json:"writeTokensPerSecond,omitempty"`
	TTFTMs     *perfStat `json:"timeToFirstTokenMs,omitempty"`
	ThinkingMs *perfStat `json:"thinkingMs,omitempty"`
	TotalMs    *perfStat `json:"totalMs,omitempty"`
	LoadMs     *perfStat `json:"loadMs,omitempty"`

	// Sources says where the latest request's figures came from.
	Sources perfSources      `json:"sources"`
	Trend   []perfTrendPoint `json:"trend"`
}

func summarizeModel(k perfKey, local bool, list []perfRecord) perfModelSummary {
	m := perfModelSummary{Engine: k.Engine, Model: k.Model, NodeID: k.NodeID, Local: local, Requests: len(list)}
	var read, write, ttft, think, total, load []float64
	collect := func(dst *[]float64, v *float64) {
		if v != nil {
			*dst = append(*dst, *v)
		}
	}
	for _, r := range list {
		m.PromptTokens += r.PromptTokens
		m.CompletionTokens += r.CompletionTokens
		if r.Thinking {
			m.ThinkingRequests++
			if r.ThinkingTokens != nil {
				m.ThinkingTokens += *r.ThinkingTokens
			}
			collect(&think, r.ThinkingMs)
		}
		collect(&read, r.ReadTPS)
		collect(&write, r.WriteTPS)
		collect(&ttft, r.TTFTMs)
		collect(&load, r.LoadMs)
		total = append(total, r.TotalMs)
		if r.At > m.LastAt {
			m.LastAt = r.At
			m.Sources = r.Sources
		}
	}
	m.ReadTPS, m.WriteTPS, m.TTFTMs = summarize(read), summarize(write), summarize(ttft)
	m.ThinkingMs, m.TotalMs, m.LoadMs = summarize(think), summarize(total), summarize(load)
	start := max(len(list)-perfTrendPoints, 0)
	m.Trend = make([]perfTrendPoint, 0, len(list)-start)
	for _, r := range list[start:] {
		m.Trend = append(m.Trend, perfTrendPoint{At: r.At, ReadTPS: r.ReadTPS, WriteTPS: r.WriteTPS, TTFTMs: r.TTFTMs, ThinkingMs: r.ThinkingMs})
	}
	return m
}

// perfReport is the performance/report result and the body of
// GET /pair/v1/performance.
type perfReport struct {
	Scope      string             `json:"scope"`
	Window     int                `json:"window"`
	Models     []perfModelSummary `json:"models"`
	Recent     []perfRecord       `json:"recent"`
	Benchmarks []benchmarkRun     `json:"benchmarks"`
	Prompts    []benchmarkPrompt  `json:"benchmarkPrompts"`
}

func (s *perfStore) report() perfReport {
	rep := perfReport{Scope: perfScope, Window: perfWindow, Prompts: benchmarkPromptList()}
	s.mu.Lock()
	rep.Models = make([]perfModelSummary, 0, len(s.byKey))
	for k, list := range s.byKey {
		rep.Models = append(rep.Models, summarizeModel(k, s.local[k], list))
	}
	n := min(len(s.recent), perfReportRecent)
	rep.Recent = make([]perfRecord, 0, n)
	for i := len(s.recent) - 1; i >= len(s.recent)-n; i-- {
		rep.Recent = append(rep.Recent, s.recent[i])
	}
	s.mu.Unlock()
	sort.Slice(rep.Models, func(i, j int) bool {
		a, b := rep.Models[i], rep.Models[j]
		if a.Model != b.Model {
			return a.Model < b.Model
		}
		if a.Engine != b.Engine {
			return a.Engine < b.Engine
		}
		return a.NodeID < b.NodeID
	})
	rep.Benchmarks = s.benchmarkSnapshot()
	return rep
}

// performanceAPIPath is PAIR's performance report for the requests this node
// routes. Like the session API it is answered by the proxy, never forwarded,
// and reachable from loopback only.
//
//	GET    /pair/v1/performance            per-model figures, recent requests, benchmarks
//	DELETE /pair/v1/performance            forget the figures and finished benchmarks
//	POST   /pair/v1/performance/benchmark  {"model"} runs the benchmark on this engine
const performanceAPIPath = "/pair/v1/performance"

func (f *facade) servePerformanceAPI(w http.ResponseWriter, r *http.Request) bool {
	path := strings.TrimSuffix(r.URL.Path, "/")
	if path != performanceAPIPath && path != performanceAPIPath+"/benchmark" {
		return false
	}
	w.Header().Set(requestIDHeader, resolveRequestID(r.Header))
	if path == performanceAPIPath+"/benchmark" {
		f.serveBenchmarkStart(w, r)
		return true
	}
	switch r.Method {
	case http.MethodGet:
		writeSessionJSON(w, http.StatusOK, f.host.perf.report())
	case http.MethodDelete:
		f.host.perf.clear()
		w.WriteHeader(http.StatusNoContent)
	default:
		methodNotAllowed(w, "GET, DELETE")
	}
	return true
}
