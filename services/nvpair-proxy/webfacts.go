// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Web facts fill the gaps model/details leaves: when the engine and the model
// file do not say a model's context, size or capabilities, the user can ask
// PAIR to read the model's own pages (the http(s) links model/details
// returned, and nothing else) with a small model on this node's Ollama.
//
// It runs only when asked, as a job: model/web-facts-start starts it and
// model/web-facts reports its progress and the last result, because reading a
// page takes longer than the broker waits for one call. Results are kept
// apart from what the engine reported, with the page each value came from and
// the model that read it, and the engine's answer always wins. The capacity
// planner uses a web-derived maximum context, and the mixture-of-experts
// flag, only when the engine and the GGUF reported none.

const (
	webFactsFileName     = "model-web-facts.json"
	defaultReaderModel   = "qwen3.5:4b"
	webFactsJobTimeout   = 4 * time.Minute
	maxWebFactsPages     = 3
	maxStoredWebFacts    = 64
	readerModelMaxLength = 200
)

// webFact is one value read from a page. Fills names the modelDetails field it
// fills because the engine left that field empty; it is empty when the engine
// already reported the field, whose answer wins.
type webFact struct {
	Field  string    `json:"field"`
	Value  factValue `json:"value"`
	Source string    `json:"source"`
	Fills  string    `json:"fills,omitempty"`
}

type webFactConflict struct {
	Field  string          `json:"field"`
	Values []factCandidate `json:"values"`
}

type webFactsPage struct {
	URL    string `json:"url"`
	Chunks int    `json:"chunks"`
	Note   string `json:"note,omitempty"`
}

// webFacts is one lookup's result.
type webFacts struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
	NodeID string `json:"nodeId,omitempty"`
	// Reader is the model that read the pages.
	Reader    string            `json:"reader"`
	ReadAt    int64             `json:"readAt"`
	Pages     []webFactsPage    `json:"pages"`
	Facts     []webFact         `json:"facts"`
	Conflicts []webFactConflict `json:"conflicts,omitempty"`
	Notes     []string          `json:"notes,omitempty"`
}

// fact returns the value read for a field.
func (w *webFacts) fact(field string) (webFact, bool) {
	if w == nil {
		return webFact{}, false
	}
	for _, f := range w.Facts {
		if f.Field == field {
			return f, true
		}
	}
	return webFact{}, false
}

type webFactsKey struct {
	Engine, NodeID, Model string
}

// webFactsJob is a lookup in progress or just finished.
type webFactsJob struct {
	State       string `json:"state"` // running, done or failed
	Phase       string `json:"phase,omitempty"`
	Reader      string `json:"reader"`
	StartedAt   int64  `json:"startedAt"`
	FinishedAt  int64  `json:"finishedAt,omitempty"`
	PagesTotal  int    `json:"pagesTotal"`
	PagesRead   int    `json:"pagesRead"`
	ChunksTotal int    `json:"chunksTotal"`
	ChunksRead  int    `json:"chunksRead"`
	Error       string `json:"error,omitempty"`
}

type webFactsFile struct {
	ReaderModel string     `json:"readerModel,omitempty"`
	Results     []webFacts `json:"results"`
}

type webFactsStore struct {
	mu      sync.Mutex
	path    string
	file    webFactsFile
	jobs    map[webFactsKey]*webFactsJob
	running bool
}

func newWebFactsStore() *webFactsStore {
	return &webFactsStore{jobs: map[webFactsKey]*webFactsJob{}}
}

func (s *webFactsStore) load() error {
	data, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var f webFactsFile
	if err := json.Unmarshal(data, &f); err != nil {
		return err
	}
	s.mu.Lock()
	s.file = f
	s.mu.Unlock()
	return nil
}

func (s *webFactsStore) saveLocked() {
	if s.path == "" {
		return
	}
	data, err := json.MarshalIndent(s.file, "", "  ")
	if err == nil {
		err = os.MkdirAll(filepath.Dir(s.path), 0o755)
	}
	if err == nil {
		tmp := s.path + ".tmp"
		if err = os.WriteFile(tmp, data, 0o600); err == nil {
			err = os.Rename(tmp, s.path)
		}
	}
	if err != nil {
		slog.Warn("could not save web facts", "err", err)
	}
}

func (s *webFactsStore) readerModel() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return firstNonEmpty(s.file.ReaderModel, defaultReaderModel)
}

func (s *webFactsStore) result(k webFactsKey) *webFacts {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.file.Results {
		r := s.file.Results[i]
		if r.Engine == k.Engine && r.NodeID == k.NodeID && r.Model == k.Model {
			return &r
		}
	}
	return nil
}

func (s *webFactsStore) putResult(r webFacts) {
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := []webFacts{r}
	for _, old := range s.file.Results {
		if (old.Engine != r.Engine || old.NodeID != r.NodeID || old.Model != r.Model) && len(kept) < maxStoredWebFacts {
			kept = append(kept, old)
		}
	}
	s.file.Results = kept
	s.saveLocked()
}

// begin registers a job, refusing a second one while any is running: the
// reader model is large and one lookup at a time keeps the node usable.
// started is false when the same lookup is already running.
func (s *webFactsStore) begin(k webFactsKey, reader string) (job *webFactsJob, started bool, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running {
		if j := s.jobs[k]; j != nil && j.State == "running" {
			return j, false, nil
		}
		return nil, false, errors.New("another web lookup is running; try again when it finishes")
	}
	if reader != s.file.ReaderModel && reader != "" {
		s.file.ReaderModel = reader
		s.saveLocked()
	}
	j := &webFactsJob{State: "running", Phase: "asking the engine for the model's pages", Reader: reader, StartedAt: time.Now().UnixMilli()}
	s.jobs[k] = j
	s.running = true
	return j, true, nil
}

// update changes a job under the store's lock.
func (s *webFactsStore) update(j *webFactsJob, change func(*webFactsJob)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	change(j)
	if j.State != "running" {
		s.running = false
		j.FinishedAt = time.Now().UnixMilli()
	}
}

func (s *webFactsStore) job(k webFactsKey) *webFactsJob {
	s.mu.Lock()
	defer s.mu.Unlock()
	if j := s.jobs[k]; j != nil {
		c := *j
		return &c
	}
	return nil
}

type webFactsRequest struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
	NodeID string `json:"nodeId,omitempty"`
	// ReaderModel is the Ollama model that reads the pages; empty keeps the
	// saved setting (default qwen3.5:4b). A new value is saved.
	ReaderModel string `json:"readerModel,omitempty"`
}

type webFactsStatus struct {
	ReaderModel string       `json:"readerModel"`
	Job         *webFactsJob `json:"job,omitempty"`
	Result      *webFacts    `json:"result,omitempty"`
}

func parseWebFactsRequest(raw json.RawMessage) (webFactsRequest, error) {
	var req webFactsRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return req, errors.New(`expected {"engine","model","nodeId"?,"readerModel"?}`)
	}
	if _, ok := profileFor(req.Engine); !ok {
		return req, fmt.Errorf("unknown engine; use one of %s", engineNames())
	}
	if req.Model == "" {
		return req, errors.New("model is required")
	}
	req.ReaderModel = strings.TrimSpace(req.ReaderModel)
	if len(req.ReaderModel) > readerModelMaxLength || strings.ContainsAny(req.ReaderModel, " \t\n") {
		return req, errors.New("readerModel must be one Ollama model name")
	}
	return req, nil
}

func (p *Proxy) webFactsStatus(raw json.RawMessage) (*webFactsStatus, error) {
	req, err := parseWebFactsRequest(raw)
	if err != nil {
		return nil, err
	}
	k := webFactsKey{req.Engine, req.NodeID, req.Model}
	return &webFactsStatus{ReaderModel: p.webFacts.readerModel(), Job: p.webFacts.job(k), Result: p.webFacts.result(k)}, nil
}

// startWebFacts starts a lookup and answers at once; the caller polls
// model/web-facts.
func (p *Proxy) startWebFacts(raw json.RawMessage) (*webFactsStatus, error) {
	req, err := parseWebFactsRequest(raw)
	if err != nil {
		return nil, err
	}
	reader := firstNonEmpty(req.ReaderModel, p.webFacts.readerModel())
	k := webFactsKey{req.Engine, req.NodeID, req.Model}
	job, started, err := p.webFacts.begin(k, reader)
	if err != nil {
		return nil, err
	}
	if started {
		go p.runWebFacts(k, job, reader)
	}
	return p.webFactsStatus(raw)
}

// ollamaReaderFor is the reader on this node's Ollama, or why there is none.
func (p *Proxy) ollamaReaderFor(model string) (factReader, error) {
	f := p.facadeFor("ollama")
	if f == nil {
		return nil, errors.New("PAIR is not relaying Ollama on this node, and the web reader runs on this node's Ollama")
	}
	base, ok := f.localBackendTarget()
	if !ok {
		return nil, errors.New("Ollama is not running on this node; start it to look models up on the web")
	}
	return &ollamaReader{base: base, model: model, client: &http.Client{}}, nil
}

// newFactReader is replaced in tests.
var newFactReader = func(p *Proxy, model string) (factReader, error) { return p.ollamaReaderFor(model) }

func (p *Proxy) runWebFacts(k webFactsKey, job *webFactsJob, readerModel string) {
	ctx, cancel := context.WithTimeout(context.Background(), webFactsJobTimeout)
	defer cancel()
	result, err := p.lookUpWebFacts(ctx, k, readerModel, func(change func(*webFactsJob)) { p.webFacts.update(job, change) })
	if err != nil {
		slog.Info("web facts lookup failed", "engine", k.Engine, "model", k.Model, "reader", readerModel)
		p.webFacts.update(job, func(j *webFactsJob) { j.State, j.Error, j.Phase = "failed", err.Error(), "" })
		return
	}
	p.webFacts.putResult(*result)
	slog.Info("web facts lookup finished", "engine", k.Engine, "model", k.Model, "reader", readerModel, "facts", len(result.Facts), "conflicts", len(result.Conflicts))
	p.webFacts.update(job, func(j *webFactsJob) { j.State, j.Phase = "done", "" })
}

// detailsFor is the model's details, from this node or from another.
func (p *Proxy) detailsFor(ctx context.Context, k webFactsKey) (*modelDetails, error) {
	ctx, cancel := context.WithTimeout(ctx, modelDetailsTimeout)
	defer cancel()
	params, _ := json.Marshal(modelDetailsRequest{Engine: k.Engine, Model: k.Model, NodeID: k.NodeID})
	return p.modelDetails(ctx, params)
}

// pagesToRead picks the http(s) links model/details returned, without the
// licence page, at most maxWebFactsPages of them.
func pagesToRead(links []modelLink) []string {
	var out []string
	for _, l := range links {
		if l.Source == "general.license.link" || webURL(l.URL) == "" {
			continue
		}
		dup := false
		for _, u := range out {
			dup = dup || u == l.URL
		}
		if !dup && len(out) < maxWebFactsPages {
			out = append(out, l.URL)
		}
	}
	return out
}

type pageChunks struct {
	page   fetchedPage
	chunks []string
}

func (p *Proxy) lookUpWebFacts(ctx context.Context, k webFactsKey, readerModel string, progress func(func(*webFactsJob))) (*webFacts, error) {
	reader, err := newFactReader(p, readerModel)
	if err != nil {
		return nil, err
	}
	if err := reader.Ready(ctx); err != nil {
		return nil, err
	}
	details, err := p.detailsFor(ctx, k)
	if err != nil {
		return nil, err
	}
	pages := pagesToRead(details.Links)
	if len(pages) == 0 {
		return nil, errors.New("the engine reported no web page for this model, so there is nothing to read")
	}
	result := &webFacts{Engine: k.Engine, Model: k.Model, NodeID: k.NodeID, Reader: reader.Model()}
	progress(func(j *webFactsJob) { j.Phase, j.PagesTotal = "loading the model's pages", len(pages) })

	var loaded []pageChunks
	total := 0
	for _, u := range pages {
		page, err := fetchPage(ctx, u)
		info := webFactsPage{URL: u}
		if err != nil {
			info.Note = err.Error()
		} else {
			chunks := chunkText(page.Text, chunkChars)
			if room := maxChunks - total; len(chunks) > room {
				chunks = chunks[:max(room, 0)]
				page.Truncated = true
			}
			if page.Truncated {
				info.Note = "the page is long; PAIR read the start of it"
			}
			info.Chunks = len(chunks)
			total += len(chunks)
			if len(chunks) > 0 {
				loaded = append(loaded, pageChunks{page, chunks})
			}
		}
		result.Pages = append(result.Pages, info)
		progress(func(j *webFactsJob) { j.PagesRead++ })
	}
	if total == 0 {
		return nil, errors.New("none of the model's pages could be read: " + result.Pages[0].Note)
	}

	acc := newFactAccumulator(modelFactSchema)
	schema := modelFactSchema.jsonSchema()
	progress(func(j *webFactsJob) { j.ChunksTotal = total })
	done, failed := 0, 0
	for _, pc := range loaded {
		for i, chunk := range pc.chunks {
			done++
			n := done
			progress(func(j *webFactsJob) { j.Phase = fmt.Sprintf("%s is reading part %d of %d", reader.Model(), n, total) })
			cctx, cancel := context.WithTimeout(ctx, chunkReadTimeout)
			patch, err := reader.Read(cctx, readerSystem, readerPrompt(k.Model, k.Engine, pc.page.URL, i+1, len(pc.chunks), chunk), schema)
			cancel()
			progress(func(j *webFactsJob) { j.ChunksRead++ })
			if err != nil {
				if ctx.Err() != nil {
					return nil, errors.New("the lookup took longer than PAIR allows; try a smaller reader model")
				}
				failed++
				result.Notes = appendOnce(result.Notes, "part of a page could not be read: "+err.Error())
				continue
			}
			acc.merge(patch, chunk, pc.page.URL)
		}
	}
	if failed == total {
		return nil, errors.New(strings.TrimPrefix(result.Notes[0], "part of a page could not be read: "))
	}
	finishWebFacts(result, acc, details)
	result.ReadAt = time.Now().UnixMilli()
	return result, nil
}

func appendOnce(list []string, s string) []string {
	for _, v := range list {
		if v == s {
			return list
		}
	}
	return append(list, s)
}

// finishWebFacts lays the merged values out in schema order and marks the
// ones that fill a field the engine left empty.
func finishWebFacts(result *webFacts, acc *factAccumulator, d *modelDetails) {
	for _, f := range modelFactSchema {
		if c, ok := acc.values[f.Name]; ok {
			result.Facts = append(result.Facts, webFact{Field: f.Name, Value: c.Value, Source: c.Source, Fills: fillsField(f.Name, c.Value, d)})
		}
		if cs, ok := acc.conflicts[f.Name]; ok {
			result.Conflicts = append(result.Conflicts, webFactConflict{Field: f.Name, Values: cs})
		}
		if n := acc.invalid[f.Name]; n > 0 {
			result.Notes = append(result.Notes, fmt.Sprintf("the reader gave %d value(s) for %s that are not allowed; they were left out", n, f.Name))
		}
	}
}

// fillsField names the modelDetails field a web value fills, or "" when the
// engine already reported it. Fields modelDetails does not have (version,
// moe) always fill.
func fillsField(field string, v factValue, d *modelDetails) string {
	switch field {
	case "name":
		if d.Name == "" {
			return "name"
		}
	case "description":
		if d.Description == "" {
			return "description"
		}
	case "version":
		return "version"
	case "parameterSize":
		if d.ParameterSize == "" && d.ParameterCount == 0 {
			return "parameterSize"
		}
	case "dataSize":
		if d.SizeBytes == 0 {
			return "sizeBytes"
		}
	case "context":
		if d.ContextLength == 0 {
			return "contextLength"
		}
	case "vision", "tools", "thinking":
		// An engine that lists capabilities lists all it has.
		if len(d.Capabilities) == 0 {
			return "capabilities"
		}
	case "moe":
		return "moe"
	case "url":
		for _, l := range d.Links {
			if l.URL == v {
				return ""
			}
		}
		return "links"
	}
	return ""
}

// applyWebFacts fills the planner's model description from a web lookup where
// the engine and the model file said nothing: the maximum context, and a note
// when the page calls the model a mixture of experts the engine did not
// describe. A page never gives the per-token cache cost, so none is derived.
func applyWebFacts(spec *modelSpec, w *webFacts) {
	if w == nil {
		return
	}
	if c, ok := w.fact("context"); ok && spec.MaxContextTokens == 0 {
		if n, ok := c.Value.(int64); ok && n > 0 {
			spec.MaxContextTokens = n
			spec.ContextSource = "web"
			spec.ContextPage = c.Source
			spec.ContextReader = w.Reader
			spec.Notes = append(spec.Notes, fmt.Sprintf("The maximum context (%d tokens) is from the model's page, read by %s; the engine and the model file did not report one.", n, w.Reader))
		}
	}
	if m, ok := w.fact("moe"); ok && spec.ExpertCount == 0 && m.Value == true {
		spec.WebMoE = true
		spec.Notes = append(spec.Notes, fmt.Sprintf("The model's page, read by %s, says this is a mixture-of-experts model, but the engine and the model file do not describe its experts, so PAIR cannot size keeping them in system memory.", w.Reader))
	}
}
