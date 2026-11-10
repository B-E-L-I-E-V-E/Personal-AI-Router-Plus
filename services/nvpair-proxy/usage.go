// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"mime"
	"sync"
)

// tokenUsage is the token count an engine reported for one completed request.
// Prompt is the whole prompt the engine held for the turn, cached prefix
// included; Completion is what it generated. Together they are how much of the
// context window the conversation occupies after this turn, because the client
// resends the full conversation every time.
type tokenUsage struct {
	Prompt     int64
	Completion int64
}

func (u tokenUsage) total() int64 { return u.Prompt + u.Completion }

// Usage is read from the response stream as it passes through, and only the
// numeric counts are kept. No text is retained, logged, or emitted: the reader
// holds at most one bounded frame at a time and discards it once inspected
// (spec §9).
const (
	// maxUsageFrame bounds one streamed line or event. Usage frames are tiny;
	// a larger frame is content and is skipped without being parsed.
	maxUsageFrame = 64 << 10
	// maxUsageBody bounds a non-streamed JSON body held for its final parse.
	// A larger reply simply reports no usage.
	maxUsageBody = 4 << 20
)

// usageReader wraps an upstream response body, passing every byte through
// unchanged while watching for the engine's token counts. done is called once,
// when the body ends cleanly with usage found; a body that is cut short, or
// that never reports usage, calls nothing.
type usageReader struct {
	src  io.ReadCloser
	mode usageMode
	done func(tokenUsage)

	mu       sync.Mutex
	frame    []byte // current streamed line, or the whole JSON body
	overflow bool   // the current frame exceeded its bound and is being skipped
	usage    tokenUsage
	found    bool
	finished bool
}

type usageMode int

const (
	usageNone   usageMode = iota
	usageLines            // NDJSON: Ollama's native streaming
	usageEvents           // SSE: OpenAI and Anthropic streaming
	usageJSON             // a single JSON document
)

// usageModeFor picks how to read a response from its content type.
func usageModeFor(contentType string) usageMode {
	mt, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return usageNone
	}
	// A table rather than a switch: the desktop contract checker reads
	// `case "a/b":` string cases as JSON-RPC method names.
	return usageModes[mt]
}

var usageModes = map[string]usageMode{
	"application/x-ndjson": usageLines,
	"application/jsonl":    usageLines,
	"application/json-seq": usageLines,
	"text/event-stream":    usageEvents,
	"application/json":     usageJSON,
}

func newUsageReader(src io.ReadCloser, contentType string, done func(tokenUsage)) io.ReadCloser {
	mode := usageModeFor(contentType)
	if mode == usageNone || done == nil {
		return src
	}
	return &usageReader{src: src, mode: mode, done: done}
}

func (u *usageReader) Read(p []byte) (int, error) {
	n, err := u.src.Read(p)
	if n > 0 {
		u.observe(p[:n])
	}
	if err == io.EOF {
		u.finish()
	}
	return n, err
}

func (u *usageReader) Close() error { return u.src.Close() }

func (u *usageReader) observe(b []byte) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.mode == usageJSON {
		if u.overflow {
			return
		}
		if len(u.frame)+len(b) > maxUsageBody {
			u.overflow, u.frame = true, nil
			return
		}
		u.frame = append(u.frame, b...)
		return
	}
	for len(b) > 0 {
		i := bytes.IndexByte(b, '\n')
		if i < 0 {
			u.appendFrame(b)
			return
		}
		u.appendFrame(b[:i])
		u.endLine()
		b = b[i+1:]
	}
}

func (u *usageReader) appendFrame(b []byte) {
	if u.overflow {
		return
	}
	if len(u.frame)+len(b) > maxUsageFrame {
		u.overflow, u.frame = true, u.frame[:0]
		return
	}
	u.frame = append(u.frame, b...)
}

// endLine inspects one complete line and forgets it.
func (u *usageReader) endLine() {
	line := bytes.TrimSpace(u.frame)
	skip := u.overflow
	u.frame, u.overflow = u.frame[:0], false
	if skip || len(line) == 0 {
		return
	}
	if u.mode == usageEvents {
		// Only data lines carry JSON; "event:", "id:" and comments do not.
		var ok bool
		if line, ok = bytes.CutPrefix(line, []byte("data:")); !ok {
			return
		}
		line = bytes.TrimSpace(line)
	}
	// Cheap pre-check so content frames are never decoded.
	if !bytes.Contains(line, []byte(`"usage"`)) && !bytes.Contains(line, []byte(`"eval_count"`)) &&
		!bytes.Contains(line, []byte(`"prompt_eval_count"`)) {
		return
	}
	u.merge(line)
}

func (u *usageReader) finish() {
	u.mu.Lock()
	if u.finished {
		u.mu.Unlock()
		return
	}
	u.finished = true
	if u.mode == usageJSON {
		if !u.overflow && len(u.frame) > 0 {
			u.merge(u.frame)
		}
	} else if len(u.frame) > 0 {
		u.endLine() // a final line without a trailing newline
	}
	u.frame = nil
	usage, found := u.usage, u.found
	u.mu.Unlock()
	if found {
		u.done(usage)
	}
}

// merge folds one JSON document's counts into the running usage. Streams can
// spread them across frames (Anthropic reports input tokens at message_start
// and output tokens at message_delta), so each field keeps its latest value.
func (u *usageReader) merge(doc []byte) {
	prompt, completion, ok := usageCounts(doc)
	if !ok {
		return
	}
	if prompt > 0 {
		u.usage.Prompt = prompt
	}
	if completion > 0 {
		u.usage.Completion = completion
	}
	u.found = true
}

// usageCounts reads token counts from any dialect PAIR routes:
//
//   - Ollama native: prompt_eval_count and eval_count on the final object.
//   - OpenAI-compatible: usage.prompt_tokens and usage.completion_tokens.
//   - Anthropic Messages: usage.input_tokens (plus any cache_read_input_tokens
//     and cache_creation_input_tokens, which Anthropic counts separately) and
//     usage.output_tokens, at the top level or under message.usage.
func usageCounts(doc []byte) (prompt, completion int64, ok bool) {
	type usageBlock struct {
		PromptTokens       *int64 `json:"prompt_tokens"`
		CompletionTokens   *int64 `json:"completion_tokens"`
		InputTokens        *int64 `json:"input_tokens"`
		OutputTokens       *int64 `json:"output_tokens"`
		CacheReadInput     *int64 `json:"cache_read_input_tokens"`
		CacheCreationInput *int64 `json:"cache_creation_input_tokens"`
	}
	var v struct {
		PromptEvalCount *int64      `json:"prompt_eval_count"`
		EvalCount       *int64      `json:"eval_count"`
		Usage           *usageBlock `json:"usage"`
		Message         *struct {
			Usage *usageBlock `json:"usage"`
		} `json:"message"`
	}
	if err := json.Unmarshal(doc, &v); err != nil {
		return 0, 0, false
	}
	val := func(p *int64) int64 {
		if p == nil || *p < 0 {
			return 0
		}
		return *p
	}
	if v.PromptEvalCount != nil || v.EvalCount != nil {
		return val(v.PromptEvalCount), val(v.EvalCount), true
	}
	block := v.Usage
	if block == nil && v.Message != nil {
		block = v.Message.Usage
	}
	if block == nil {
		return 0, 0, false
	}
	switch {
	case block.PromptTokens != nil || block.CompletionTokens != nil:
		return val(block.PromptTokens), val(block.CompletionTokens), true
	case block.InputTokens != nil || block.OutputTokens != nil:
		in := val(block.InputTokens) + val(block.CacheReadInput) + val(block.CacheCreationInput)
		return in, val(block.OutputTokens), true
	}
	return 0, 0, false
}

// requestStreamsOpenAI reports whether an OpenAI-compatible request body asks
// for a stream without asking for usage. An OpenAI-compatible stream reports
// token counts only when stream_options.include_usage is set.
func requestStreamsOpenAI(body []byte) bool {
	var v struct {
		Stream        bool `json:"stream"`
		StreamOptions *struct {
			IncludeUsage *bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	if json.Unmarshal(body, &v) != nil || !v.Stream {
		return false
	}
	return v.StreamOptions == nil || v.StreamOptions.IncludeUsage == nil
}

// withStreamUsage returns body with stream_options.include_usage set to true.
// Every other field keeps its value; the re-encoding may reorder keys and
// compact whitespace, which no JSON consumer depends on. The engine
// then appends one final chunk carrying usage and an empty choices array,
// which is standard OpenAI streaming behavior. It returns body unchanged when
// it cannot rewrite it safely.
func withStreamUsage(body []byte) []byte {
	var fields map[string]json.RawMessage
	if json.Unmarshal(body, &fields) != nil {
		return body
	}
	opts := map[string]json.RawMessage{}
	if raw, ok := fields["stream_options"]; ok && len(raw) > 0 && !bytes.Equal(raw, []byte("null")) {
		if json.Unmarshal(raw, &opts) != nil {
			return body
		}
	}
	opts["include_usage"] = json.RawMessage("true")
	encoded, err := json.Marshal(opts)
	if err != nil {
		return body
	}
	fields["stream_options"] = encoded
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(fields); err != nil {
		return body
	}
	return bytes.TrimRight(out.Bytes(), "\n")
}

// isOpenAIInferencePath reports whether a path is an OpenAI-compatible
// generation route, the dialect whose streams need include_usage.
func isOpenAIInferencePath(path string) bool {
	return path == "/v1/chat/completions" || path == "/v1/completions"
}
