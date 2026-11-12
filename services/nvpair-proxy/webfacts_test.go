// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// allowLoopbackPages lets fetchPage read httptest's loopback servers.
func allowLoopbackPages(t *testing.T) {
	t.Helper()
	allowPrivatePages = true
	t.Cleanup(func() { allowPrivatePages = false })
}

func rawPatch(t *testing.T, s string) map[string]json.RawMessage {
	t.Helper()
	p, err := parsePatch(s)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestCleanHTML_KeepsReadableTextAndDropsScripts(t *testing.T) {
	page := `<html><head><title>Example Model</title><meta name="description" content="Example &amp; friends."><meta property="og:title" content="OG Title"><meta property="og:image" content="https://x.example/i.png">
		<script>var leaked = "SCRIPT"</script><style>.x{}</style></head>
		<body><nav>Home Models Pricing</nav><h1>Example</h1><p>A   small
		model.</p><ul><li>Fast</li><li>Small</li></ul><table><tr><td>Context</td><td>128K</td></tr></table>
		<footer>Copyright</footer></body></html>`
	got := cleanHTML([]byte(page))
	for _, want := range []string{"Example Model", "Example & friends.", "OG Title", "## Example", "A small model.", "- Fast", "Context | 128K"} {
		if !strings.Contains(got, want) {
			t.Errorf("cleaned text lacks %q:\n%s", want, got)
		}
	}
	for _, unwanted := range []string{"i.png", "SCRIPT", ".x{}", "Pricing", "Copyright"} {
		if strings.Contains(got, unwanted) {
			t.Errorf("cleaned text keeps %q:\n%s", unwanted, got)
		}
	}
}

func TestChunkText_KeepsShortTextInOneChunk(t *testing.T) {
	if got := chunkText("## A\nshort", 100); len(got) != 1 || got[0] != "## A\nshort" {
		t.Fatalf("chunks = %q, want the text as one chunk", got)
	}
	if got := chunkText("  \n ", 100); got != nil {
		t.Fatalf("chunks of blank text = %q, want none", got)
	}
}

func TestChunkText_SplitsAtHeadingsWithinTheSize(t *testing.T) {
	text := "## One\n" + strings.Repeat("a ", 20) + "\n## Two\n" + strings.Repeat("b ", 20) + "\n## Three\n" + strings.Repeat("c ", 20)
	chunks := chunkText(text, 60)
	if len(chunks) != 3 {
		t.Fatalf("got %d chunks, want one per section: %q", len(chunks), chunks)
	}
	for i, want := range []string{"## One", "## Two", "## Three"} {
		if !strings.HasPrefix(chunks[i], want) {
			t.Errorf("chunk %d = %q, want it to start at heading %q", i, chunks[i], want)
		}
	}
}

func TestChunkText_SplitsAnOversizedSectionAtSpaces(t *testing.T) {
	long := strings.Repeat("word ", 100) // 500 bytes, one line
	chunks := chunkText("## Big\n"+long, 120)
	if len(chunks) < 4 {
		t.Fatalf("got %d chunks, want the long line split", len(chunks))
	}
	var rebuilt []string
	for _, c := range chunks {
		if len(c) > 120 {
			t.Errorf("chunk is %d bytes, over the 120-byte size", len(c))
		}
		rebuilt = append(rebuilt, strings.Fields(c)...)
	}
	if got := strings.Count(strings.Join(rebuilt, " "), "word"); got != 100 {
		t.Errorf("chunks hold %d words, want all 100", got)
	}
}

func TestValidateFact_RefusesValuesTheSchemaDoesNotAllow(t *testing.T) {
	field := func(name string) factField {
		f, _ := modelFactSchema.field(name)
		return f
	}
	cases := []struct {
		name  string
		field string
		raw   string
	}{
		{"text for a number", "context", `"128K"`},
		{"fraction for a whole number", "context", `4096.5`},
		{"context below the minimum", "context", `12`},
		{"number for a flag", "moe", `1`},
		{"number for text", "name", `7`},
		{"too long a name", "name", `"` + strings.Repeat("n", 200) + `"`},
		{"address that is not http", "url", `"ftp://example.com/model"`},
		{"download size as parameter count", "parameterSize", `"523MB"`},
		{"parameter count as download size", "dataSize", `"0.6B"`},
		{"size tag as version", "version", `"0.6b"`},
		{"tagged name", "name", `"qwen3:0.6b"`},
		{"address the page never gives", "url", `"https://elsewhere.example/model"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := validateFact(field(tc.field), json.RawMessage(tc.raw), "the page text", "https://page.example/m"); err == nil {
				t.Fatalf("%s = %s was accepted", tc.field, tc.raw)
			}
		})
	}
}

func TestValidateFact_TreatsNullAndUnknownAsNoValue(t *testing.T) {
	f, _ := modelFactSchema.field("version")
	for _, raw := range []string{`null`, `""`, `"unknown"`, `"N/A"`} {
		if _, ok, err := validateFact(f, json.RawMessage(raw), "", ""); ok || err != nil {
			t.Errorf("version = %s gave ok=%v err=%v, want no value and no error", raw, ok, err)
		}
	}
}

func TestValidateFact_AcceptsAnAddressThePageGives(t *testing.T) {
	f, _ := modelFactSchema.field("url")
	v, ok, err := validateFact(f, json.RawMessage(`"https://example.com/models/m"`), "see https://example.com/models/m for more", "https://page.example/")
	if err != nil || !ok || v != "https://example.com/models/m" {
		t.Fatalf("got %v ok=%v err=%v", v, ok, err)
	}
}

func TestFactSchema_KeepsTheFieldOrderForTheReader(t *testing.T) {
	b, err := json.Marshal(modelFactSchema.jsonSchema())
	if err != nil {
		t.Fatal(err)
	}
	text := string(b)
	last := -1
	for _, f := range modelFactSchema {
		if f.Pointed {
			if strings.Contains(text, `"`+f.Name+`"`) {
				t.Errorf("pointed field %s is in the whole reading's schema", f.Name)
			}
			continue
		}
		at := strings.Index(text, `"`+f.Name+`":`)
		if at < last {
			t.Errorf("field %s is out of the schema's order in %s", f.Name, text)
		}
		last = at
	}
}

func TestFactAccumulator_KeepsAgreeingValuesAndFalse(t *testing.T) {
	acc := newFactAccumulator(modelFactSchema)
	acc.merge(rawPatch(t, `{"name":"Qwen3","context":40000,"vision":false,"tools":null}`), "", "https://a.example/")
	acc.merge(rawPatch(t, `{"name":"qwen 3","context":40960,"vision":false,"unlisted":"x"}`), "", "https://b.example/")
	if got := acc.values["name"].Value; got != "Qwen3" {
		t.Errorf("name = %v, want the first of two agreeing spellings", got)
	}
	if got := acc.values["context"].Value; got != int64(40960) {
		t.Errorf("context = %v, want 40960, the precise one of 40K and 40,960", got)
	}
	if got, ok := acc.values["vision"]; !ok || got.Value != false {
		t.Errorf("vision = %+v, want false kept as a value", got)
	}
	if _, ok := acc.values["tools"]; ok {
		t.Error("tools was null and must stay empty")
	}
	if _, ok := acc.values["unlisted"]; ok {
		t.Error("a field outside the schema was kept")
	}
	if len(acc.conflicts) != 0 {
		t.Errorf("conflicts = %v, want none", acc.conflicts)
	}
}

func TestFactAccumulator_LeavesAConflictingFieldEmpty(t *testing.T) {
	acc := newFactAccumulator(modelFactSchema)
	acc.merge(rawPatch(t, `{"context":131072,"moe":true,"description":"A model."}`), "", "https://a.example/")
	acc.merge(rawPatch(t, `{"context":32768,"moe":true,"description":"Another wording."}`), "", "https://b.example/")
	acc.merge(rawPatch(t, `{"context":32768}`), "", "https://c.example/")
	if _, ok := acc.values["context"]; ok {
		t.Fatal("context was kept although two chunks disagree")
	}
	got := acc.conflicts["context"]
	if len(got) != 2 || got[0].Value != int64(131072) || got[1].Value != int64(32768) {
		t.Fatalf("context conflict = %+v, want both values once", got)
	}
	if acc.values["moe"].Value != true {
		t.Error("moe agreed and should be kept")
	}
	if acc.values["description"].Value != "A model." {
		t.Error("two wordings of the description are not a conflict; the first should be kept")
	}
}

func TestFactAccumulator_CountsInvalidValues(t *testing.T) {
	acc := newFactAccumulator(modelFactSchema)
	acc.merge(rawPatch(t, `{"context":"lots"}`), "", "https://a.example/")
	if acc.invalid["context"] != 1 || len(acc.values) != 0 {
		t.Fatalf("invalid = %v values = %v", acc.invalid, acc.values)
	}
}

func TestFetchPage_RefusesNonHTTPAddresses(t *testing.T) {
	for _, raw := range []string{"file:///etc/passwd", "ftp://example.com/m", "javascript:alert(1)", "mailto:a@example.com", "not a url"} {
		if _, err := fetchPage(context.Background(), raw); err == nil {
			t.Errorf("fetchPage(%q) succeeded", raw)
		}
	}
}

func TestPagesToRead_KeepsOnlyWebLinksWithoutTheLicence(t *testing.T) {
	got := pagesToRead([]modelLink{
		{URL: "https://ollama.com/library/qwen3", Source: "model name"},
		{URL: "javascript:alert(1)", Source: "general.url"},
		{URL: "https://www.apache.org/licenses/LICENSE-2.0", Source: "general.license.link"},
		{URL: "https://ollama.com/library/qwen3", Source: "general.url"},
		{URL: "http://a.example/", Source: "general.repo_url"},
		{URL: "http://b.example/", Source: "general.source.url"},
		{URL: "http://c.example/", Source: "general.source.repo_url"},
	})
	want := []string{"https://ollama.com/library/qwen3", "http://a.example/", "http://b.example/"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("pages = %v, want %v", got, want)
	}
}

func TestFetchPage_RefusesALocalNetworkAddress(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("hello"))
	}))
	defer srv.Close()
	if _, err := fetchPage(context.Background(), srv.URL); !errors.Is(err, errPrivateAddress) {
		t.Fatalf("err = %v, want the loopback page refused", err)
	}
}

func TestFetchPage_ReadsAtMostTheSizeCap(t *testing.T) {
	allowLoopbackPages(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte("<html><body><p>HEAD</p><script>"))
		_, _ = w.Write([]byte(strings.Repeat("x", 3<<20)))
		_, _ = w.Write([]byte("</script><p>TAIL</p></body></html>"))
	}))
	defer srv.Close()
	page, err := fetchPage(context.Background(), srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if !page.Truncated || !strings.Contains(page.Text, "HEAD") || strings.Contains(page.Text, "TAIL") {
		t.Fatalf("truncated=%v text=%q, want the start read and the part past 2 MiB left out", page.Truncated, page.Text)
	}
}

func TestFetchPage_StopsAtTheDeadline(t *testing.T) {
	allowLoopbackPages(t)
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	}))
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := fetchPage(ctx, srv.URL); err == nil {
		t.Fatal("a page that never answers was read")
	}
	if took := time.Since(start); took > 2*time.Second {
		t.Fatalf("fetchPage took %v, want it stopped at the deadline", took)
	}
}

func TestFetchPage_RefusesRedirectsAwayFromHTTP(t *testing.T) {
	allowLoopbackPages(t)
	hops := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/ftp":
			http.Redirect(w, r, "ftp://example.com/model", http.StatusFound)
		case "/loop":
			hops++
			http.Redirect(w, r, "/loop", http.StatusFound)
		case "/binary":
			w.Header().Set("Content-Type", "application/octet-stream")
			_, _ = w.Write([]byte{0, 1, 2})
		}
	}))
	defer srv.Close()
	if _, err := fetchPage(context.Background(), srv.URL+"/ftp"); err == nil {
		t.Error("a redirect to ftp was followed")
	}
	if _, err := fetchPage(context.Background(), srv.URL+"/loop"); err == nil || hops > maxRedirects+1 {
		t.Errorf("redirect loop: err=%v after %d hops, want a refusal after %d", err, hops, maxRedirects)
	}
	if _, err := fetchPage(context.Background(), srv.URL+"/binary"); err == nil || !strings.Contains(err.Error(), "not text") {
		t.Errorf("binary page: err=%v, want it refused as not text", err)
	}
}

// webFactsFixture is this node's Ollama, which has the model being looked up
// (its file names the model's page) and the reader model, and the page.
type webFactsFixture struct {
	proxy   *Proxy
	page    *httptest.Server
	mu      sync.Mutex
	prompts []string
	formats []map[string]any
}

const webFactsModel = "example.test/acme/example:7b"

func newWebFactsFixture(t *testing.T, readerTags string) *webFactsFixture {
	t.Helper()
	allowLoopbackPages(t)
	fx := &webFactsFixture{}
	filler := strings.Repeat("This paragraph is about nothing in particular. ", 280)
	fx.page = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprintf(w, `<html><head><title>Example</title><script>var leaked="SCRIPT"</script></head><body>
			<h1>Example</h1><p>Example 7B reads long documents.</p>
			<h2>Specs</h2><table><tr><td>Context</td><td>128K</td></tr></table><p>It is a mixture-of-experts model.</p>
			<h2>Background</h2><p>%s</p>
			<h2>Other sizes</h2><p>The 70B size has a 32K context.</p></body></html>`, filler)
	}))
	t.Cleanup(fx.page.Close)

	show := fmt.Sprintf(`{"details":{"format":"gguf","family":"example","parameter_size":"7.2B"},
		"model_info":{"general.architecture":"example","general.url":%q,"general.license.link":"https://licence.example/"},
		"capabilities":["completion"]}`, fx.page.URL+"/m")
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/show":
			_, _ = w.Write([]byte(show))
		case "/api/tags":
			_, _ = w.Write([]byte(readerTags))
		case "/api/chat":
			var req struct {
				Model    string `json:"model"`
				Format   map[string]any
				Messages []struct{ Content string }
			}
			_ = json.NewDecoder(r.Body).Decode(&req)
			prompt := req.Messages[len(req.Messages)-1].Content
			fx.mu.Lock()
			fx.prompts = append(fx.prompts, prompt)
			fx.formats = append(fx.formats, req.Format)
			fx.mu.Unlock()
			// A pointed question asks for one field and the evidence.
			field := ""
			if props, _ := req.Format["properties"].(map[string]any); props["evidence"] != nil {
				for name := range props {
					if name != "evidence" {
						field = name
					}
				}
			}
			specs, others := strings.Contains(prompt, "## Specs"), strings.Contains(prompt, "## Other sizes")
			answer := `{}`
			switch {
			case field == "context" && specs:
				answer = `{"context":131072,"evidence":"Context | 128K"}`
			case field == "context" && others:
				answer = `{"context":32768,"evidence":"32K context"}`
			case field == "moe" && specs:
				answer = `{"moe":true,"evidence":"mixture-of-experts"}`
			case field == "parameterSize" && specs:
				answer = `{"parameterSize":"523MB","evidence":"wrong kind of size"}`
			case field != "":
				answer = `{"evidence":"nothing"}`
			case specs:
				answer = `{"name":"Example","moe":false,"context":1,"vision":null,"url":"https://elsewhere.example/x"}`
			case others:
				answer = `{"tools":false}`
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"message": map[string]string{"role": "assistant", "content": answer}})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(engine.Close)

	tc := ollamaCase(t)
	f := testProxy(tc.profile, NewDiscovery(), tc.profile.StandalonePort).soleFacade()
	n := nodeFor(t, "engine", engine.URL)
	if err := f.setLocalBackend(localBackend{Host: n.Addresses[0], Port: n.Port, Healthy: true}); err != nil {
		t.Fatal(err)
	}
	fx.proxy = f.host
	fx.proxy.webFacts.path = filepath.Join(t.TempDir(), webFactsFileName)
	return fx
}

const readerPulled = `{"models":[{"name":"qwen3.5:4b","model":"qwen3.5:4b"}]}`

// lookUp starts a lookup over RPC and polls it to the end.
func (fx *webFactsFixture) lookUp(t *testing.T) *webFactsStatus {
	t.Helper()
	params := json.RawMessage(fmt.Sprintf(`{"engine":"ollama","model":%q}`, webFactsModel))
	st, err := fx.proxy.startWebFacts(params)
	if err != nil {
		t.Fatal(err)
	}
	if st.Job == nil || st.ReaderModel != defaultReaderModel {
		t.Fatalf("start answered %+v, want a job read by %s", st, defaultReaderModel)
	}
	return fx.waitForLookup(t)
}

// waitForLookup polls model/web-facts until the lookup ends.
func (fx *webFactsFixture) waitForLookup(t *testing.T) *webFactsStatus {
	t.Helper()
	params := json.RawMessage(fmt.Sprintf(`{"engine":"ollama","model":%q}`, webFactsModel))
	deadline := time.Now().Add(10 * time.Second)
	for {
		st, err := fx.proxy.webFactsStatus(params)
		if err != nil {
			t.Fatal(err)
		}
		if st.Job.State != "running" {
			return st
		}
		if time.Now().After(deadline) {
			t.Fatalf("the lookup is still running: %+v", st.Job)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestWebFacts_FillsWhatTheEngineLeftEmptyFromThePage(t *testing.T) {
	fx := newWebFactsFixture(t, readerPulled)
	st := fx.lookUp(t)
	if st.Job.State != "done" {
		t.Fatalf("job = %+v, want done", st.Job)
	}
	r := st.Result
	if r == nil || r.Reader != defaultReaderModel || len(r.Pages) != 1 || r.Pages[0].URL != fx.page.URL+"/m" {
		t.Fatalf("result = %+v, want the one page named by general.url read by %s", r, defaultReaderModel)
	}
	if r.Pages[0].Chunks < 2 || st.Job.ChunksRead != r.Pages[0].Chunks {
		t.Fatalf("page read in %d chunks (%d read), want it split in several", r.Pages[0].Chunks, st.Job.ChunksRead)
	}
	got := map[string]webFact{}
	for _, f := range r.Facts {
		got[f.Field] = f
	}
	if f := got["name"]; f.Value != "Example" || f.Fills != "name" || f.Source != fx.page.URL+"/m" {
		t.Errorf("name = %+v, want it to fill the empty name, sourced to the page", f)
	}
	if f := got["moe"]; f.Value != true || f.Fills != "moe" {
		t.Errorf("moe = %+v, want true", f)
	}
	if f := got["tools"]; f.Value != false || f.Fills != "" {
		t.Errorf("tools = %+v, want false kept, filling nothing: the engine listed its capabilities", f)
	}
	for _, missing := range []string{"context", "url", "vision", "version"} {
		if f, ok := got[missing]; ok {
			t.Errorf("%s = %+v, want it empty", missing, f)
		}
	}
	if len(r.Conflicts) != 1 || r.Conflicts[0].Field != "context" || len(r.Conflicts[0].Values) != 2 {
		t.Errorf("conflicts = %+v, want the two contexts reported", r.Conflicts)
	}
	for _, refused := range []string{"url", "parameterSize"} {
		if !strings.Contains(strings.Join(r.Notes, "\n"), refused) {
			t.Errorf("notes = %v, want the invalid %s reported as left out", r.Notes, refused)
		}
	}
	if pointed := len(fx.prompts) - r.Pages[0].Chunks; pointed != 5*r.Pages[0].Chunks {
		t.Errorf("%d pointed questions for %d chunks, want 5 per chunk", pointed, r.Pages[0].Chunks)
	}
	for i, prompt := range fx.prompts {
		if strings.Contains(prompt, "SCRIPT") {
			t.Error("a page script reached the reader")
		}
		if !strings.Contains(prompt, webFactsModel) {
			t.Errorf("prompt %d does not name the model asked about", i)
		}
		props, _ := fx.formats[i]["properties"].(map[string]any)
		if props["evidence"] == nil {
			for _, f := range modelFactSchema {
				if _, asked := props[f.Name]; asked == f.Pointed {
					t.Errorf("format %d: field %s asked=%v, want only fields that are not pointed in the whole reading", i, f.Name, asked)
				}
			}
		} else if len(props) != 2 {
			t.Errorf("format %d = %v, want one field and the evidence", i, fx.formats[i])
		}
	}

	reloaded := newWebFactsStore()
	reloaded.path = fx.proxy.webFacts.path
	if err := reloaded.load(); err != nil {
		t.Fatal(err)
	}
	if kept := reloaded.result(webFactsKey{Engine: "ollama", Model: webFactsModel}); kept == nil || len(kept.Facts) != len(r.Facts) {
		t.Fatalf("saved result = %+v, want the lookup kept across restarts", kept)
	}
}

func TestWebFacts_SaysWhenTheReaderModelIsNotDownloaded(t *testing.T) {
	fx := newWebFactsFixture(t, `{"models":[{"name":"qwen3:0.6b","model":"qwen3:0.6b"}]}`)
	st := fx.lookUp(t)
	if st.Job.State != "failed" || !strings.Contains(st.Job.Error, "qwen3.5:4b is not downloaded") {
		t.Fatalf("job = %+v, want it to say the reader model is missing", st.Job)
	}
	if len(fx.prompts) != 0 {
		t.Fatal("another model was asked instead of the missing reader")
	}
	if st.Result != nil {
		t.Fatalf("result = %+v, want none", st.Result)
	}
}

func TestWebFacts_SavesOnlyAReaderModelOllamaHas(t *testing.T) {
	fx := newWebFactsFixture(t, readerPulled)
	params := json.RawMessage(fmt.Sprintf(`{"engine":"ollama","model":%q,"readerModel":"qwen3.5:4bb"}`, webFactsModel))
	if _, err := fx.proxy.startWebFacts(params); err != nil {
		t.Fatal(err)
	}
	st := fx.waitForLookup(t)
	if st.Job.State != "failed" || st.ReaderModel != defaultReaderModel {
		t.Fatalf("job = %+v, setting = %q; want the misspelt reader refused and the setting kept", st.Job, st.ReaderModel)
	}
}

func TestWebFacts_RefusesASecondLookupWhileOneRuns(t *testing.T) {
	s := newWebFactsStore()
	a := webFactsKey{Engine: "ollama", Model: "a"}
	if _, started, err := s.begin(a, "r"); err != nil || !started {
		t.Fatalf("first lookup: started=%v err=%v", started, err)
	}
	if _, started, err := s.begin(a, "r"); err != nil || started {
		t.Fatalf("same lookup again: started=%v err=%v, want the running one returned", started, err)
	}
	if _, _, err := s.begin(webFactsKey{Engine: "ollama", Model: "b"}, "r"); err == nil {
		t.Fatal("a second lookup started while one runs")
	}
}

func TestWebFactsRPC_RejectsBadRequests(t *testing.T) {
	p := testProxy(ollamaCase(t).profile, NewDiscovery(), 0)
	for _, raw := range []string{`{"engine":"ollama"}`, `{"engine":"vllm","model":"m"}`, `{"engine":"ollama","model":"m","readerModel":"two words"}`} {
		if _, err := p.startWebFacts(json.RawMessage(raw)); err == nil {
			t.Errorf("start %s was accepted", raw)
		}
	}
}

func TestApplyWebFacts_FillsOnlyWhatTheEngineDidNotReport(t *testing.T) {
	w := &webFacts{Reader: "qwen3.5:4b", Facts: []webFact{
		{Field: "context", Value: int64(131072), Source: "https://page.example/m"},
		{Field: "moe", Value: true, Source: "https://page.example/m"},
	}}

	empty := modelSpec{}
	applyWebFacts(&empty, w)
	if empty.MaxContextTokens != 131072 || empty.ContextSource != "web" || empty.ContextPage != "https://page.example/m" || empty.ContextReader != "qwen3.5:4b" {
		t.Errorf("spec = %+v, want the page's context, labelled", empty)
	}
	if !empty.WebMoE || len(empty.Notes) != 2 {
		t.Errorf("spec = %+v, want the moe flag and two notes", empty)
	}
	if empty.KVBytesPerToken != 0 || empty.ExpertCount != 0 || empty.ExpertBytes != 0 {
		t.Errorf("spec = %+v, want no cost or expert size derived from a page", empty)
	}

	reported := modelSpec{MaxContextTokens: 40960, ExpertCount: 8, ExpertBytes: 1 << 30}
	applyWebFacts(&reported, w)
	if reported.MaxContextTokens != 40960 || reported.ContextSource != "" || reported.WebMoE || len(reported.Notes) != 0 {
		t.Errorf("spec = %+v, want the engine's values kept untouched", reported)
	}
}

func TestPlanCapacity_UsesAWebContextWhenTheEngineGivesNone(t *testing.T) {
	p := testProxy(ollamaCase(t).profile, NewDiscovery(), 0)
	p.webFacts.putResult(webFacts{Engine: "llamacpp", Model: "m", Reader: "qwen3.5:4b", Facts: []webFact{
		{Field: "context", Value: int64(65536), Source: "https://page.example/m"},
	}})
	res, err := p.planCapacity(context.Background(), json.RawMessage(`{"engine":"llamacpp","model":"m","gpus":[{"vramBytes":17179869184}],"weightsBytes":1073741824,"kvBytesPerToken":1024}`))
	if err != nil {
		t.Fatal(err)
	}
	if res.Spec.MaxContextTokens != 65536 || res.Spec.ContextSource != "web" || res.WindowTokens != 65536 {
		t.Fatalf("plan = window %d, spec %+v; want the page's 65,536-token context used and labelled", res.WindowTokens, res.Spec)
	}
}
