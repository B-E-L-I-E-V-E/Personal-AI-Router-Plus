// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"time"
	"unicode"
)

// Performance telemetry (spec §5.16): for every generation request this node
// routes, the proxy measures how fast the engine read the prompt and wrote the
// reply, and how long a thinking model spent reasoning. It reads the same
// response stream as the session usage reader (usage.go), frame by frame, and
// keeps only numbers: a frame's text is measured for its length and dropped,
// never stored, logged or emitted.
//
// Every figure says where it came from:
//
//   - "engine": the engine reported it (Ollama's *_duration fields,
//     llama.cpp's timings block, the usage counts).
//   - "proxy": the proxy measured it on its own clock, from when the attempt
//     was dispatched to when frames arrived.
//   - "estimated": apportioned from other figures, such as the thinking share
//     of a non-streamed reply's generation time by its share of the text.
const (
	perfSourceEngine    = "engine"
	perfSourceProxy     = "proxy"
	perfSourceEstimated = "estimated"
)

// perfSources records where each of a record's figures came from.
type perfSources struct {
	Tokens   string `json:"tokens,omitempty"`
	Read     string `json:"read,omitempty"`
	Write    string `json:"write,omitempty"`
	TTFT     string `json:"ttft,omitempty"`
	Thinking string `json:"thinking,omitempty"`
}

// perfRecord is one completed generation request's measurements. Optional
// figures are nil when neither the engine nor the proxy could measure them.
type perfRecord struct {
	RequestID string `json:"requestId"`
	// At is when the request finished, in Unix milliseconds.
	At     int64  `json:"at"`
	Engine string `json:"engine"`
	Model  string `json:"model"`
	// NodeID is the node that served the request; Local is true when that is
	// this node.
	NodeID   string `json:"nodeId"`
	Local    bool   `json:"local"`
	Path     string `json:"path"`
	Streamed bool   `json:"streamed"`

	PromptTokens     int64 `json:"promptTokens"`
	CompletionTokens int64 `json:"completionTokens"`
	// Thinking is true when the reply had a reasoning part.
	Thinking       bool   `json:"thinking"`
	ThinkingTokens *int64 `json:"thinkingTokens,omitempty"`

	// ReadTPS is prompt tokens processed per second; WriteTPS is generated
	// tokens per second.
	ReadTPS  *float64 `json:"readTokensPerSecond,omitempty"`
	WriteTPS *float64 `json:"writeTokensPerSecond,omitempty"`
	// TTFTMs is from dispatch to the first generated token (thinking or
	// answer).
	TTFTMs       *float64 `json:"timeToFirstTokenMs,omitempty"`
	PromptMs     *float64 `json:"promptMs,omitempty"`
	GenerationMs *float64 `json:"generationMs,omitempty"`
	LoadMs       *float64 `json:"loadMs,omitempty"`
	ThinkingMs   *float64 `json:"thinkingMs,omitempty"`
	// TotalMs is from dispatch to the end of the body, on the proxy's clock.
	TotalMs float64 `json:"totalMs"`

	Sources perfSources `json:"sources"`
}

// perfTracker accumulates one response's measurements as its frames pass.
// It is driven by usageReader under that reader's lock, so it needs none of
// its own.
type perfTracker struct {
	start    time.Time
	now      func() time.Time
	streamed bool
	done     func(perfRecord)
	// base carries the identity fields the caller already knows.
	base perfRecord

	inThinkTag bool
	// failed is set when the engine ended the reply with an error frame
	// (Ollama's "prediction aborted", an OpenAI or Anthropic error event):
	// an aborted reply is not a measurement.
	failed bool

	firstOut, lastOut     time.Time
	firstThink, lastThink time.Time
	firstAnswer           time.Time
	thinkChunks           int64
	answerChunks          int64
	thinkChars            int64
	answerChars           int64

	usage      tokenUsage
	usageFound bool

	// Ollama native durations, in nanoseconds.
	ollama struct {
		found                                 bool
		prompt, eval, load, total             int64
		promptSet, evalSet, loadSet, totalSet bool
	}
	// llama.cpp timings.
	llama struct {
		found                                    bool
		promptN, promptMs, predictedN, predicted float64
	}
}

func newPerfTracker(start time.Time, base perfRecord, done func(perfRecord)) *perfTracker {
	return &perfTracker{start: start, now: time.Now, base: base, done: done}
}

// perfMessage is the part of a chat message the tracker measures.
type perfMessage struct {
	Content          json.RawMessage `json:"content"`
	Thinking         string          `json:"thinking"`
	ReasoningContent string          `json:"reasoning_content"`
	Reasoning        string          `json:"reasoning"`
}

type perfBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Thinking string `json:"thinking"`
}

// perfFrame decodes every dialect PAIR routes at once; each field is absent in
// the dialects that do not use it.
type perfFrame struct {
	// Ollama /api/generate.
	Response *string `json:"response"`
	Thinking *string `json:"thinking"`
	// Ollama /api/chat, and Anthropic message_start (whose content is a block
	// array, hence the raw message).
	Message *perfMessage `json:"message"`
	// OpenAI-compatible.
	Choices []struct {
		Delta   *perfMessage `json:"delta"`
		Message *perfMessage `json:"message"`
		Text    *string      `json:"text"`
	} `json:"choices"`
	// Anthropic Messages: streamed deltas and blocks, and a non-streamed
	// reply's content array.
	Delta        *perfBlock      `json:"delta"`
	ContentBlock *perfBlock      `json:"content_block"`
	Content      json.RawMessage `json:"content"`
	// Error is any dialect's error field or event.
	Error json.RawMessage `json:"error"`

	// Ollama timings, in nanoseconds.
	PromptEvalDuration *int64 `json:"prompt_eval_duration"`
	EvalDuration       *int64 `json:"eval_duration"`
	LoadDuration       *int64 `json:"load_duration"`
	TotalDuration      *int64 `json:"total_duration"`
	// llama.cpp timings, on its native and OpenAI-compatible routes.
	Timings *struct {
		PromptN     *float64 `json:"prompt_n"`
		PromptMs    *float64 `json:"prompt_ms"`
		PredictedN  *float64 `json:"predicted_n"`
		PredictedMs *float64 `json:"predicted_ms"`
	} `json:"timings"`
}

// observe measures one frame: a streamed line or event's JSON, or a whole
// non-streamed body.
func (t *perfTracker) observe(doc []byte) {
	var f perfFrame
	if json.Unmarshal(doc, &f) != nil {
		return
	}
	if e := bytes.TrimSpace(f.Error); len(e) > 0 && !bytes.Equal(e, []byte("null")) {
		t.failed = true
		return
	}
	at := t.now()
	var think, answer int64
	var answerVisible bool
	addThink := func(s string) { think += int64(len(s)) }
	addAnswer := func(s string) {
		th, an, vis := t.splitThinkTags(s)
		think += th
		answer += an
		answerVisible = answerVisible || vis
	}
	addMessage := func(m *perfMessage) {
		if m == nil {
			return
		}
		addThink(m.Thinking)
		addThink(m.ReasoningContent)
		addThink(m.Reasoning)
		t.measureContent(m.Content, addThink, addAnswer)
	}
	if f.Response != nil {
		addAnswer(*f.Response)
	}
	if f.Thinking != nil {
		addThink(*f.Thinking)
	}
	addMessage(f.Message)
	for _, c := range f.Choices {
		addMessage(c.Delta)
		addMessage(c.Message)
		if c.Text != nil {
			addAnswer(*c.Text)
		}
	}
	for _, b := range []*perfBlock{f.Delta, f.ContentBlock} {
		if b == nil {
			continue
		}
		addThink(b.Thinking)
		addAnswer(b.Text)
	}
	t.measureContent(f.Content, addThink, addAnswer)

	if think > 0 {
		t.thinkChunks++
		t.thinkChars += think
		if t.firstThink.IsZero() {
			t.firstThink = at
		}
		t.lastThink = at
	}
	if answer > 0 {
		t.answerChunks++
		t.answerChars += answer
		if answerVisible && t.firstAnswer.IsZero() {
			t.firstAnswer = at
		}
	}
	if think+answer > 0 {
		if t.firstOut.IsZero() {
			t.firstOut = at
		}
		t.lastOut = at
	}

	if bytes.Contains(doc, []byte(`"usage"`)) || bytes.Contains(doc, []byte(`"eval_count"`)) ||
		bytes.Contains(doc, []byte(`"prompt_eval_count"`)) {
		if prompt, completion, ok := usageCounts(doc); ok {
			if prompt > 0 {
				t.usage.Prompt = prompt
			}
			if completion > 0 {
				t.usage.Completion = completion
			}
			t.usageFound = true
		}
	}
	set := func(dst *int64, isSet *bool, v *int64) {
		if v != nil && *v >= 0 {
			*dst, *isSet, t.ollama.found = *v, true, true
		}
	}
	set(&t.ollama.prompt, &t.ollama.promptSet, f.PromptEvalDuration)
	set(&t.ollama.eval, &t.ollama.evalSet, f.EvalDuration)
	set(&t.ollama.load, &t.ollama.loadSet, f.LoadDuration)
	set(&t.ollama.total, &t.ollama.totalSet, f.TotalDuration)
	if tm := f.Timings; tm != nil && tm.PredictedMs != nil {
		t.llama.found = true
		t.llama.promptN, t.llama.promptMs = deref(tm.PromptN), deref(tm.PromptMs)
		t.llama.predictedN, t.llama.predicted = deref(tm.PredictedN), *tm.PredictedMs
	}
}

func deref(p *float64) float64 {
	if p == nil {
		return 0
	}
	return *p
}

// measureContent measures a content field that is either a string (Ollama,
// OpenAI) or an array of typed blocks (Anthropic).
func (t *perfTracker) measureContent(raw json.RawMessage, addThink, addAnswer func(string)) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return
	}
	switch raw[0] {
	case '"':
		var s string
		if json.Unmarshal(raw, &s) == nil {
			addAnswer(s)
		}
	case '[':
		var blocks []perfBlock
		if json.Unmarshal(raw, &blocks) == nil {
			for _, b := range blocks {
				addThink(b.Thinking)
				addAnswer(b.Text)
			}
		}
	}
}

const (
	thinkOpenTag  = "<think>"
	thinkCloseTag = "</think>"
)

// splitThinkTags divides answer text into the part inside <think>…</think>
// (tags included, so a chunk that is only a tag still marks the boundary) and
// the part outside. The open state carries across frames. visible reports
// whether the outside part has anything but whitespace.
func (t *perfTracker) splitThinkTags(s string) (think, answer int64, visible bool) {
	for len(s) > 0 {
		if t.inThinkTag {
			i := strings.Index(s, thinkCloseTag)
			if i < 0 {
				return think + int64(len(s)), answer, visible
			}
			think += int64(i + len(thinkCloseTag))
			s = s[i+len(thinkCloseTag):]
			t.inThinkTag = false
			continue
		}
		i := strings.Index(s, thinkOpenTag)
		out := s
		if i >= 0 {
			out = s[:i]
		}
		answer += int64(len(out))
		visible = visible || strings.IndexFunc(out, func(r rune) bool { return !unicode.IsSpace(r) }) >= 0
		if i < 0 {
			return think, answer, visible
		}
		think += int64(len(thinkOpenTag))
		s = s[i+len(thinkOpenTag):]
		t.inThinkTag = true
	}
	return think, answer, visible
}

func msBetween(a, b time.Time) float64 {
	return float64(b.Sub(a)) / float64(time.Millisecond)
}

func fptr(v float64) *float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return nil
	}
	v = math.Round(v*100) / 100
	return &v
}

func perSecond(tokens float64, ms float64) *float64 {
	if tokens <= 0 || ms <= 0 {
		return nil
	}
	return fptr(tokens / ms * 1000)
}

// finish turns the accumulated frames into a record and hands it on. A reply
// with no generated output and no token counts produces nothing: there is
// nothing to measure.
func (t *perfTracker) finish() {
	rec, ok := t.record(t.now())
	if ok && t.done != nil {
		t.done(rec)
	}
}

func (t *perfTracker) record(end time.Time) (perfRecord, bool) {
	r := t.base
	r.Streamed = t.streamed
	r.At = end.UnixMilli()
	r.TotalMs = *fptr(msBetween(t.start, end))
	outChunks := t.thinkChunks + t.answerChunks
	if t.failed {
		return perfRecord{}, false
	}
	if !t.usageFound && !t.llama.found && outChunks == 0 {
		return perfRecord{}, false
	}

	// Token counts: the engine's usage, else llama.cpp's timings, else the
	// streamed chunks counted (one chunk is about one token).
	promptProcessed := float64(0)
	switch {
	case t.usageFound:
		r.PromptTokens, r.CompletionTokens = t.usage.Prompt, t.usage.Completion
		r.Sources.Tokens = perfSourceEngine
	case t.llama.found:
		r.PromptTokens, r.CompletionTokens = int64(t.llama.promptN), int64(t.llama.predictedN)
		r.Sources.Tokens = perfSourceEngine
	default:
		r.CompletionTokens = outChunks
		r.Sources.Tokens = perfSourceEstimated
	}
	promptProcessed = float64(r.PromptTokens)

	// Reading: the engine's prompt-evaluation time where it reports one.
	switch {
	case t.ollama.promptSet && t.ollama.prompt > 0:
		r.PromptMs = fptr(float64(t.ollama.prompt) / 1e6)
		r.Sources.Read = perfSourceEngine
	case t.llama.found && t.llama.promptMs > 0:
		r.PromptMs = fptr(t.llama.promptMs)
		// llama.cpp counts only the prompt tokens it evaluated, not the cached
		// prefix, so its rate uses its own count.
		promptProcessed = t.llama.promptN
		r.Sources.Read = perfSourceEngine
	case t.streamed && !t.firstOut.IsZero():
		// Dispatch to first token: prompt evaluation plus queueing, model load
		// and network, so this rate is a lower bound.
		r.PromptMs = fptr(msBetween(t.start, t.firstOut))
		r.Sources.Read = perfSourceProxy
	}
	if r.PromptMs != nil {
		r.ReadTPS = perSecond(promptProcessed, *r.PromptMs)
	}
	if r.ReadTPS == nil {
		r.Sources.Read = ""
	}

	// Writing: the engine's generation time where it reports one, else the
	// span between the first and last streamed tokens.
	writeTokens := float64(r.CompletionTokens)
	switch {
	case t.ollama.evalSet && t.ollama.eval > 0:
		r.GenerationMs = fptr(float64(t.ollama.eval) / 1e6)
		r.Sources.Write = perfSourceEngine
	case t.llama.found && t.llama.predicted > 0:
		r.GenerationMs = fptr(t.llama.predicted)
		writeTokens = t.llama.predictedN
		r.Sources.Write = perfSourceEngine
	case t.streamed && outChunks > 1 && t.lastOut.After(t.firstOut):
		r.GenerationMs = fptr(msBetween(t.firstOut, t.lastOut))
		// The first token arrived at the window's start, so it is not part of
		// the time measured.
		writeTokens--
		r.Sources.Write = perfSourceProxy
	}
	if r.GenerationMs != nil {
		r.WriteTPS = perSecond(writeTokens, *r.GenerationMs)
	}
	if r.WriteTPS == nil {
		r.Sources.Write = ""
	}

	if t.ollama.loadSet && t.ollama.load > 0 {
		r.LoadMs = fptr(float64(t.ollama.load) / 1e6)
	}

	// Time to first token: measured directly on a stream; for a single body,
	// what the engine spent before generating (load plus prompt).
	switch {
	case t.streamed && !t.firstOut.IsZero():
		r.TTFTMs = fptr(msBetween(t.start, t.firstOut))
		r.Sources.TTFT = perfSourceProxy
	case t.ollama.promptSet:
		r.TTFTMs = fptr(float64(t.ollama.load+t.ollama.prompt) / 1e6)
		r.Sources.TTFT = perfSourceEngine
	case t.llama.found && t.llama.promptMs > 0:
		r.TTFTMs = fptr(t.llama.promptMs)
		r.Sources.TTFT = perfSourceEngine
	}

	// Thinking.
	if t.thinkChars > 0 {
		r.Thinking = true
		if t.streamed {
			tokens := t.thinkChunks
			if r.CompletionTokens > 0 && outChunks > 0 && t.usageFound {
				tokens = int64(math.Round(float64(r.CompletionTokens) * float64(t.thinkChunks) / float64(outChunks)))
			}
			r.ThinkingTokens = &tokens
			end := t.firstAnswer
			if end.IsZero() {
				end = t.lastThink
			}
			if end.After(t.firstThink) {
				r.ThinkingMs = fptr(msBetween(t.firstThink, end))
			} else {
				zero := 0.0
				r.ThinkingMs = &zero
			}
			r.Sources.Thinking = perfSourceProxy
		} else {
			share := float64(t.thinkChars) / float64(t.thinkChars+t.answerChars)
			tokens := int64(math.Round(float64(r.CompletionTokens) * share))
			r.ThinkingTokens = &tokens
			if r.GenerationMs != nil {
				r.ThinkingMs = fptr(*r.GenerationMs * share)
			}
			r.Sources.Thinking = perfSourceEstimated
		}
	}
	return r, true
}

// isGenerationPath reports whether a route generates text, so it has reading
// and writing speeds to measure. Embedding routes are inference but generate
// nothing.
func isGenerationPath(path string) bool {
	return generationPaths[path]
}

var generationPaths = map[string]bool{
	"/api/generate":        true,
	"/api/chat":            true,
	"/v1/chat/completions": true,
	"/v1/completions":      true,
	"/v1/messages":         true,
}
