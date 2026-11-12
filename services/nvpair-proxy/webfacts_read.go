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
	"math"
	"mime"
	"net"
	"net/http"
	"net/url"
	"strings"
	"syscall"
	"time"
	"unicode/utf8"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

// Reading a model's web page into facts is a fixed pipeline, driven by a
// schema the caller supplies:
//
//	fetch (http(s) only, capped) → clean HTML to text → split into chunks
//	→ a small local model answers each chunk with partial JSON
//	→ ordinary code merges the parts, flags conflicts, validates the schema
//
// The model may interpret the text ("mixture-of-experts" supports moe: true)
// but every value must be supported by a chunk: a field no chunk supports
// stays empty, and two chunks that disagree leave it empty too, with both
// values reported. Page bodies and prompts are never logged.

const (
	maxPageBytes     = 2 << 20
	maxRedirects     = 5
	pageFetchTimeout = 10 * time.Second
	// maxCleanChars caps the text kept from one page after cleaning, and
	// chunkChars the text in one chunk: about 3,000 tokens, so a chunk, the
	// instructions and the answer fit a 16K window with room to spare.
	maxCleanChars = 48_000
	chunkChars    = 12_000
	maxChunks     = 8
	// readerContextTokens is the window asked of the reader model.
	readerContextTokens = 16384
	chunkReadTimeout    = 90 * time.Second
	maxReaderAnswer     = 64 << 10
)

// factField is one field of the JSON the reader fills.
type factField struct {
	Name        string
	Type        string // "string", "integer" or "boolean"
	Description string
	MaxLen      int
	// FreeText fields are prose: differently worded values from two chunks
	// are not a conflict, and the first one is kept.
	FreeText bool
	// URL fields must be an http(s) address that appears in the text read.
	URL bool
	// Min and Max bound an integer.
	Min, Max int64
}

type factSchema []factField

func (s factSchema) field(name string) (factField, bool) {
	for _, f := range s {
		if f.Name == name {
			return f, true
		}
	}
	return factField{}, false
}

// jsonSchema is the schema handed to the reader as its output format. No field
// is required, so the reader can leave out what a chunk does not support.
func (s factSchema) jsonSchema() map[string]any {
	props := map[string]any{}
	for _, f := range s {
		p := map[string]any{"type": f.Type, "description": f.Description}
		props[f.Name] = p
	}
	return map[string]any{"type": "object", "properties": props}
}

// modelFactSchema is what PAIR asks of a model's page.
var modelFactSchema = factSchema{
	{Name: "name", Type: "string", MaxLen: 120, Description: "The model's name as the page writes it"},
	{Name: "description", Type: "string", MaxLen: 600, FreeText: true, Description: "One or two sentences on what the model is and what it is for"},
	{Name: "version", Type: "string", MaxLen: 40, Description: "The model's version or release, such as 3 or 2.5"},
	{Name: "parameterSize", Type: "string", MaxLen: 40, Description: "The parameter count of the downloaded size, such as 0.6B or 8B"},
	{Name: "dataSize", Type: "string", MaxLen: 40, Description: "The download or file size of the downloaded size, such as 523MB"},
	{Name: "context", Type: "integer", Min: 512, Max: 1 << 24, Description: "The longest context window of the downloaded size, in tokens (write 40K as 40960 only if the page says 40,960; otherwise 40000)"},
	{Name: "vision", Type: "boolean", Description: "Whether the model accepts images as input"},
	{Name: "tools", Type: "boolean", Description: "Whether the model supports tool or function calling"},
	{Name: "thinking", Type: "boolean", Description: "Whether the model has a thinking or reasoning mode"},
	{Name: "moe", Type: "boolean", Description: "Whether the model is a mixture-of-experts model"},
	{Name: "url", Type: "string", MaxLen: 300, URL: true, Description: "The address of the model's own page"},
}

// ---------------------------------------------------------------------------
// Fetching

// errPrivateAddress refuses pages on this machine or the local network: a
// link comes from a model file's metadata, and a model file is not trusted to
// point PAIR at the user's router or another node.
var errPrivateAddress = errors.New("the address is on this machine or the local network")

// allowPrivatePages lets tests serve pages from httptest's loopback server.
var allowPrivatePages = false

func publicAddressOnly(_, address string, _ syscall.RawConn) error {
	if allowPrivatePages {
		return nil
	}
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return err
	}
	ip := net.ParseIP(host)
	if ip == nil || ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified() || ip.IsMulticast() || !ip.IsGlobalUnicast() {
		return errPrivateAddress
	}
	return nil
}

// pageClient fetches pages: http(s) only, at most maxRedirects redirects,
// each one to http(s), and never to a private address.
var pageClient = &http.Client{
	Timeout: pageFetchTimeout,
	Transport: &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 5 * time.Second, Control: publicAddressOnly}).DialContext,
		TLSHandshakeTimeout:   5 * time.Second,
		ResponseHeaderTimeout: pageFetchTimeout,
		MaxIdleConns:          4,
		IdleConnTimeout:       30 * time.Second,
	},
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= maxRedirects {
			return fmt.Errorf("more than %d redirects", maxRedirects)
		}
		if req.URL.Scheme != "http" && req.URL.Scheme != "https" {
			return fmt.Errorf("redirect to a %s address refused", req.URL.Scheme)
		}
		return nil
	},
}

// fetchedPage is one page's text, cleaned. Truncated says the page was longer
// than PAIR reads.
type fetchedPage struct {
	URL       string
	Text      string
	Truncated bool
}

// fetchPage downloads an http(s) page and cleans it to text.
func fetchPage(ctx context.Context, raw string) (fetchedPage, error) {
	page := fetchedPage{URL: raw}
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return page, errors.New("only http and https pages are read")
	}
	ctx, cancel := context.WithTimeout(ctx, pageFetchTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return page, err
	}
	req.Header.Set("Accept", "text/html, text/plain;q=0.8")
	req.Header.Set("User-Agent", "NVIDIA-PAIR-model-facts/1")
	resp, err := pageClient.Do(req)
	if err != nil {
		if errors.Is(err, errPrivateAddress) {
			return page, errPrivateAddress
		}
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			return page, errors.New("the page did not load in time")
		}
		return page, errors.New("the page could not be loaded")
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		return page, fmt.Errorf("the page answered HTTP %d", resp.StatusCode)
	}
	media, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if media != "text/html" && media != "text/plain" && media != "application/xhtml+xml" {
		return page, fmt.Errorf("the page is %s, not text", firstNonEmpty(media, "of an unknown type"))
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxPageBytes+1))
	if err != nil {
		return page, errors.New("the page could not be read in time")
	}
	if len(body) > maxPageBytes {
		body = body[:maxPageBytes]
		page.Truncated = true
	}
	if media == "text/plain" {
		page.Text = normalizeText(string(body))
	} else {
		page.Text = cleanHTML(body)
	}
	if len(page.Text) > maxCleanChars {
		page.Text = truncateRunes(page.Text, maxCleanChars)
		page.Truncated = true
	}
	return page, nil
}

// ---------------------------------------------------------------------------
// Cleaning

// skipped elements carry no readable text.
var skipped = map[atom.Atom]bool{
	atom.Script: true, atom.Style: true, atom.Noscript: true, atom.Svg: true,
	atom.Template: true, atom.Iframe: true, atom.Canvas: true, atom.Button: true,
	atom.Select: true, atom.Form: true, atom.Nav: true, atom.Footer: true,
}

// blocks end a line.
var blocks = map[atom.Atom]bool{
	atom.P: true, atom.Div: true, atom.Section: true, atom.Article: true, atom.Main: true,
	atom.Header: true, atom.Aside: true, atom.Br: true, atom.Hr: true, atom.Li: true,
	atom.Ul: true, atom.Ol: true, atom.Tr: true, atom.Table: true, atom.Pre: true,
	atom.Blockquote: true, atom.Dt: true, atom.Dd: true, atom.Dl: true, atom.Figcaption: true,
	atom.Title: true,
}

var headings = map[atom.Atom]bool{
	atom.H1: true, atom.H2: true, atom.H3: true, atom.H4: true, atom.H5: true, atom.H6: true,
}

// cleanHTML keeps a page's readable text: the title, the description in its
// meta tags, headings (as "## " lines, which the chunker splits at), list
// items, table cells and paragraphs. Scripts, styles, navigation and forms
// are dropped.
func cleanHTML(body []byte) string {
	z := html.NewTokenizer(bytes.NewReader(body))
	var out strings.Builder
	skipDepth := 0
	var skipAtom atom.Atom
	inHead, inTitle := false, false
	for {
		tt := z.Next()
		switch tt {
		case html.ErrorToken:
			return normalizeText(out.String())
		case html.StartTagToken, html.SelfClosingTagToken:
			name, hasAttr := z.TagName()
			a := atom.Lookup(name)
			if skipDepth > 0 {
				if a == skipAtom && tt == html.StartTagToken {
					skipDepth++
				}
				continue
			}
			switch {
			case a == atom.Head:
				inHead = true
			case a == atom.Title:
				inTitle = true
				out.WriteString("\n")
			case a == atom.Meta && hasAttr:
				if text := metaDescription(z); text != "" {
					out.WriteString("\n" + text + "\n")
				}
			case skipped[a] && tt == html.StartTagToken:
				skipDepth, skipAtom = 1, a
			case headings[a]:
				out.WriteString("\n\n## ")
			case a == atom.Td || a == atom.Th:
				out.WriteString(" | ")
			case a == atom.Li:
				out.WriteString("\n- ")
			case blocks[a]:
				out.WriteString("\n")
			}
		case html.EndTagToken:
			name, _ := z.TagName()
			a := atom.Lookup(name)
			if skipDepth > 0 {
				if a == skipAtom {
					skipDepth--
				}
				continue
			}
			switch a {
			case atom.Head:
				inHead = false
			case atom.Title:
				inTitle = false
			}
			if headings[a] || blocks[a] {
				out.WriteString("\n")
			}
		case html.TextToken:
			if skipDepth > 0 {
				continue
			}
			// In <head> only the title is text worth reading.
			if inHead && !inTitle {
				continue
			}
			// Line breaks in the source are not breaks on the page.
			out.WriteString(strings.NewReplacer("\n", " ", "\r", " ", "\t", " ").Replace(string(z.Text())))
		}
	}
}

// metaDescription returns the content of a description meta tag.
func metaDescription(z *html.Tokenizer) string {
	var key, content string
	for {
		k, v, more := z.TagAttr()
		switch string(k) {
		case "name", "property":
			key = strings.ToLower(string(v))
		case "content":
			content = string(v)
		}
		if !more {
			break
		}
	}
	// name="description", or the Open Graph og:description and og:title.
	if key == "description" || strings.TrimPrefix(key, "og:") != key && (strings.HasSuffix(key, "description") || strings.HasSuffix(key, "title")) {
		return strings.TrimSpace(html.UnescapeString(content))
	}
	return ""
}

// normalizeText collapses runs of spaces, trims lines, drops empty lines and
// lines repeated verbatim (menus and cards that repeat on a page).
func normalizeText(s string) string {
	seen := map[string]bool{}
	var out []string
	blank := false
	for _, line := range strings.Split(strings.ReplaceAll(s, "\r", ""), "\n") {
		line = strings.Join(strings.Fields(line), " ")
		line = strings.TrimPrefix(line, "| ")
		if line == "" || line == "-" || line == "##" || line == "|" {
			blank = true
			continue
		}
		if seen[line] && len(line) > 24 {
			continue
		}
		seen[line] = true
		if blank && len(out) > 0 && strings.HasPrefix(line, "## ") {
			out = append(out, "")
		}
		blank = false
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

func truncateRunes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// ---------------------------------------------------------------------------
// Chunking

// chunkText splits text at its headings and packs whole sections into chunks
// of at most size bytes. A section longer than size is split at paragraph
// (line) boundaries, and a line longer than size at a space.
func chunkText(text string, size int) []string {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	if len(text) <= size {
		return []string{text}
	}
	var sections []string
	var cur []string
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(line, "## ") && len(cur) > 0 {
			sections = append(sections, strings.Join(cur, "\n"))
			cur = nil
		}
		cur = append(cur, line)
	}
	if len(cur) > 0 {
		sections = append(sections, strings.Join(cur, "\n"))
	}

	var pieces []string
	for _, sec := range sections {
		if len(sec) <= size {
			pieces = append(pieces, sec)
			continue
		}
		for _, line := range strings.Split(sec, "\n") {
			for len(line) > size {
				cut := strings.LastIndex(line[:size], " ")
				if cut <= size/2 {
					cut = len(truncateRunes(line, size))
				}
				pieces = append(pieces, line[:cut])
				line = strings.TrimSpace(line[cut:])
			}
			if line != "" {
				pieces = append(pieces, line)
			}
		}
	}

	var chunks []string
	var b strings.Builder
	for _, p := range pieces {
		if b.Len() > 0 && b.Len()+1+len(p) > size {
			chunks = append(chunks, b.String())
			b.Reset()
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(p)
	}
	if b.Len() > 0 {
		chunks = append(chunks, b.String())
	}
	return chunks
}

// ---------------------------------------------------------------------------
// The reader

// factReader turns one chunk of text into a partial JSON object. It is an
// interface so another inference engine can stand in for Ollama.
type factReader interface {
	// Model names the model that reads, for the result's label.
	Model() string
	// Ready says whether the reader can run, in words for the user if not.
	Ready(ctx context.Context) error
	Read(ctx context.Context, system, prompt string, schema map[string]any) (map[string]json.RawMessage, error)
}

// ollamaReader asks a model on this node's Ollama, with its answer held to the
// schema by Ollama's structured output.
type ollamaReader struct {
	base   *url.URL
	model  string
	client *http.Client
}

func (r *ollamaReader) Model() string { return r.model }

func (r *ollamaReader) Ready(ctx context.Context) error {
	var tags struct {
		Models []struct {
			Name  string `json:"name"`
			Model string `json:"model"`
		} `json:"models"`
	}
	if err := specReader.do(ctx, http.MethodGet, r.base, "/api/tags", nil, nil, &tags); err != nil {
		return errors.New("could not ask this node's Ollama which models it has")
	}
	want := r.model
	if !strings.Contains(want, ":") {
		want += ":latest"
	}
	for _, m := range tags.Models {
		if m.Name == r.model || m.Name == want || m.Model == r.model || m.Model == want {
			return nil
		}
	}
	return fmt.Errorf("%s is not downloaded in this node's Ollama; download it to look models up on the web", r.model)
}

func (r *ollamaReader) Read(ctx context.Context, system, prompt string, schema map[string]any) (map[string]json.RawMessage, error) {
	body := map[string]any{
		"model": r.model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": prompt},
		},
		"stream":     false,
		"think":      false,
		"format":     schema,
		"keep_alive": "5m",
		"options":    map[string]any{"temperature": 0, "num_ctx": readerContextTokens},
	}
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	u := *r.base
	u.Path = "/api/chat"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, errors.New("the reader model did not answer in time")
		}
		return nil, errors.New("could not reach this node's Ollama")
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error string `json:"error"`
		}
		_ = json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&e)
		return nil, fmt.Errorf("Ollama refused the request (HTTP %d%s)", resp.StatusCode, prefixed(": ", e.Error))
	}
	var chat struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxReaderAnswer)).Decode(&chat); err != nil {
		return nil, errors.New("Ollama's answer could not be read")
	}
	return parsePatch(chat.Message.Content)
}

func prefixed(prefix, s string) string {
	if s == "" {
		return ""
	}
	return prefix + s
}

// parsePatch reads the reader's answer: one JSON object.
func parsePatch(content string) (map[string]json.RawMessage, error) {
	content = strings.TrimSpace(content)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	var patch map[string]json.RawMessage
	if err := json.Unmarshal([]byte(strings.TrimSpace(content)), &patch); err != nil {
		return nil, errors.New("the reader model did not answer with a JSON object")
	}
	return patch, nil
}

const readerSystem = `You read text from web pages about AI models and answer with one JSON object.
Fill only the fields the text supports for the model you are asked about. You may interpret the text: a "mixture-of-experts" design supports "moe": true, and "supports function calling" supports "tools": true. Do not guess. Leave out every field the text does not support, and leave a field out when the text gives different values for different sizes and does not say which one applies to the model asked about.`

// readerPrompt is the question for one chunk.
func readerPrompt(model, engine, page string, part, parts int, chunk string) string {
	return fmt.Sprintf("The downloaded model is %q (engine: %s).\nText from %s (part %d of %d):\n\"\"\"\n%s\n\"\"\"\nAnswer with the JSON fields this text supports for %q.",
		model, engine, page, part, parts, chunk, model)
}

// ---------------------------------------------------------------------------
// Validation and merging

// factValue is a validated value: a string, an int64 or a bool.
type factValue = any

// validateFact checks one field of a reader's answer against the schema. It
// returns ok=false for a null or absent value, and an error for a value the
// schema does not allow. text is the chunk the value came from, page its URL.
func validateFact(f factField, raw json.RawMessage, text, page string) (factValue, bool, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, false, nil
	}
	switch f.Type {
	case "string":
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, false, fmt.Errorf("%s must be text", f.Name)
		}
		s = strings.Join(strings.Fields(s), " ")
		if s == "" || strings.EqualFold(s, "unknown") || strings.EqualFold(s, "n/a") || strings.EqualFold(s, "null") {
			return nil, false, nil
		}
		if f.MaxLen > 0 && len(s) > f.MaxLen {
			if f.FreeText {
				s = truncateRunes(s, f.MaxLen) + "…"
			} else {
				return nil, false, fmt.Errorf("%s is longer than %d characters", f.Name, f.MaxLen)
			}
		}
		if f.URL {
			u := webURL(s)
			if u == "" {
				return nil, false, fmt.Errorf("%s is not an http(s) address", f.Name)
			}
			if u != page && !strings.Contains(text, u) {
				return nil, false, fmt.Errorf("%s is not an address the page gives", f.Name)
			}
			s = u
		}
		return s, true, nil
	case "integer":
		var n float64
		if err := json.Unmarshal(raw, &n); err != nil {
			return nil, false, fmt.Errorf("%s must be a number", f.Name)
		}
		if n != math.Trunc(n) {
			return nil, false, fmt.Errorf("%s must be a whole number", f.Name)
		}
		v := int64(n)
		if (f.Min != 0 && v < f.Min) || (f.Max != 0 && v > f.Max) {
			return nil, false, fmt.Errorf("%s is outside %d–%d", f.Name, f.Min, f.Max)
		}
		return v, true, nil
	case "boolean":
		var b bool
		if err := json.Unmarshal(raw, &b); err != nil {
			return nil, false, fmt.Errorf("%s must be true or false", f.Name)
		}
		return b, true, nil
	}
	return nil, false, fmt.Errorf("%s has an unknown type", f.Name)
}

// factCandidate is one value a page gave for a field, and where.
type factCandidate struct {
	Value  factValue `json:"value"`
	Source string    `json:"source"`
}

// factAccumulator merges partial answers into one object.
type factAccumulator struct {
	schema    factSchema
	values    map[string]factCandidate
	conflicts map[string][]factCandidate
	// invalid counts values the schema refused, by field.
	invalid map[string]int
}

func newFactAccumulator(schema factSchema) *factAccumulator {
	return &factAccumulator{schema: schema, values: map[string]factCandidate{}, conflicts: map[string][]factCandidate{}, invalid: map[string]int{}}
}

// sameFact says whether two values for f agree: equal text ignoring case and
// spacing, or integers within 2.5% of each other (40K and 40,960 tokens).
func sameFact(f factField, a, b factValue) bool {
	switch av := a.(type) {
	case string:
		bv, _ := b.(string)
		norm := func(s string) string {
			return strings.NewReplacer(" ", "", "-", "", "_", "").Replace(strings.ToLower(s))
		}
		return f.FreeText || norm(av) == norm(bv)
	case int64:
		bv, _ := b.(int64)
		hi, lo := max64(av, bv), min64(av, bv)
		return hi > 0 && float64(hi-lo)/float64(hi) <= 0.025
	}
	return a == b
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// merge adds one chunk's answer. Values outside the schema are dropped; a
// value that disagrees with an earlier one turns the field into a conflict,
// which leaves it empty.
func (acc *factAccumulator) merge(patch map[string]json.RawMessage, text, source string) {
	for name, raw := range patch {
		f, ok := acc.schema.field(name)
		if !ok {
			continue
		}
		v, ok, err := validateFact(f, raw, text, source)
		if err != nil {
			acc.invalid[name]++
			continue
		}
		if !ok {
			continue
		}
		c := factCandidate{Value: v, Source: source}
		if prev, conflicted := acc.conflicts[name]; conflicted {
			if !containsFact(f, prev, v) {
				acc.conflicts[name] = append(prev, c)
			}
			continue
		}
		have, ok := acc.values[name]
		if !ok {
			acc.values[name] = c
			continue
		}
		if sameFact(f, have.Value, v) {
			// Keep the more precise of two agreeing numbers: 40,960 over 40,000.
			if hv, isInt := have.Value.(int64); isInt && hv%1024 != 0 && v.(int64)%1024 == 0 {
				acc.values[name] = c
			}
			continue
		}
		acc.conflicts[name] = []factCandidate{have, c}
		delete(acc.values, name)
	}
}

func containsFact(f factField, list []factCandidate, v factValue) bool {
	for _, c := range list {
		if sameFact(f, c.Value, v) {
			return true
		}
	}
	return false
}
