// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// These tests drive PAIR only through its public HTTP surface — the capacity
// API, the session API and inference on the plaintext facade — against a fake
// Ollama that behaves the way the real one was measured to:
//
//   - a prompt over its window is silently cut to the last half of the window,
//     unless the request says "truncate": false, when it is refused with the
//     exact token count;
//   - a reply with no length of its own runs long (fakeLongReply tokens);
//   - token counts are reported as prompt_eval_count / eval_count (or OpenAI
//     usage), at four characters a token.
//
// They prove that what is configured through the API is what requests,
// responses, refusals and session reports obey, and that a change to the
// configuration takes effect at once.

const (
	fakeWindow    = 8192
	fakeLongReply = 3000
	charsPerToken = 4
)

type fakeOllama struct {
	mu   sync.Mutex
	last map[string]any // the last inference body received
}

func (e *fakeOllama) lastBody() map[string]any {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.last
}

func (e *fakeOllama) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	if json.Unmarshal(raw, &body) != nil {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"models":[]}`)
		return
	}
	e.mu.Lock()
	e.last = body
	e.mu.Unlock()

	chars := 0
	if p, ok := body["prompt"].(string); ok {
		chars += len(p)
	}
	if msgs, ok := body["messages"].([]any); ok {
		for _, m := range msgs {
			if c, ok := m.(map[string]any)["content"].(string); ok {
				chars += len(c)
			}
		}
	}
	prompt := chars / charsPerToken
	if prompt > fakeWindow {
		if body["truncate"] == false {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = fmt.Fprintf(w, `{"error":"request (%d tokens) exceeds the available context size (%d tokens), try increasing it"}`, prompt, fakeWindow)
			return
		}
		prompt = fakeWindow / 2 // Ollama's silent cut
	}
	reply := fakeLongReply
	if opts, ok := body["options"].(map[string]any); ok {
		if n, ok := opts["num_predict"].(float64); ok {
			reply = int(n)
		}
	}
	if n, ok := body["max_tokens"].(float64); ok {
		reply = int(n)
	}
	if strings.HasPrefix(r.URL.Path, "/v1/") {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":%d}}`, prompt, reply)
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	_, _ = fmt.Fprintf(w, "{\"done\":true,\"prompt_eval_count\":%d,\"eval_count\":%d}\n", prompt, reply)
}

type budgetFixture struct {
	t      *testing.T
	tc     engineCase
	proxy  *Proxy
	engine *fakeOllama
}

func newBudgetFixture(t *testing.T) *budgetFixture {
	t.Helper()
	tc := ollamaCase(t)
	fx := &budgetFixture{t: t, tc: tc, engine: &fakeOllama{}}
	srv := httptest.NewServer(fx.engine)
	t.Cleanup(srv.Close)

	const selfPort = 47124
	disc := NewDiscovery()
	disc.AddManual(Node{ID: "self", Addresses: []string{"127.0.0.1"}, Port: selfPort, Models: []string{tc.advertisedModel}})
	fx.proxy = newTestProxy(tc.profile, NewCodec(&recRW{}), disc, selfPort)
	backend := nodeFor(t, "backend", srv.URL)
	if err := fx.proxy.soleFacade().setLocalBackend(localBackend{Host: backend.Addresses[0], Port: backend.Port, Healthy: true}); err != nil {
		t.Fatalf("set local backend: %v", err)
	}
	return fx
}

// call sends a loopback request through the plaintext facade, the same
// entry point a local client uses.
func (fx *budgetFixture) call(method, path, session, body string) *httptest.ResponseRecorder {
	fx.t.Helper()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, path, rd)
	req.RemoteAddr = "127.0.0.1:50000"
	if session != "" {
		req.Header.Set(sessionIDHeader, session)
	}
	w := httptest.NewRecorder()
	fx.proxy.soleFacade().handlePlain(w, req)
	return w
}

// chat sends one /api/chat turn whose user message is about tokens long,
// with extra JSON fields merged in.
func (fx *budgetFixture) chat(session string, tokens int, extra string) *httptest.ResponseRecorder {
	fx.t.Helper()
	content := strings.Repeat("abcd", tokens)
	body := fmt.Sprintf(`{"model":%q,"stream":false,"messages":[{"role":"user","content":%q}]%s}`, fx.tc.requestedModel, content, extra)
	return fx.call(http.MethodPost, "/api/chat", session, body)
}

func (fx *budgetFixture) putCapacity(body string) *httptest.ResponseRecorder {
	fx.t.Helper()
	return fx.call(http.MethodPut, capacityAPIPath, "", body)
}

func (fx *budgetFixture) capacityRow() modelCapacity {
	fx.t.Helper()
	w := fx.call(http.MethodGet, capacityAPIPath, "", "")
	wantStatus(fx.t, w, http.StatusOK)
	var report struct {
		Models []modelCapacity `json:"models"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &report); err != nil {
		fx.t.Fatalf("decode capacity: %v", err)
	}
	for _, m := range report.Models {
		if m.Model == fx.tc.requestedModel {
			return m
		}
	}
	fx.t.Fatalf("capacity report has no %s: %s", fx.tc.requestedModel, w.Body.String())
	return modelCapacity{}
}

func (fx *budgetFixture) sessionContext(id string) sessionContextJSON {
	fx.t.Helper()
	w := fx.call(http.MethodGet, sessionsAPIPrefix+"/"+id, "", "")
	wantStatus(fx.t, w, http.StatusOK)
	var s sessionJSON
	if err := json.Unmarshal(w.Body.Bytes(), &s); err != nil {
		fx.t.Fatalf("decode session: %v", err)
	}
	return s.Context[fx.tc.profile.Name]
}

func (fx *budgetFixture) capacityBody(contextTokens int64, sessions int, reply int64) string {
	return fmt.Sprintf(`{"engine":%q,"model":%q,"contextTokens":%d,"maxSessions":%d,"replyTokens":%d}`,
		fx.tc.profile.Name, fx.tc.requestedModel, contextTokens, sessions, reply)
}

func TestBudgetAPI_ConfigurationIsReadBack(t *testing.T) {
	fx := newBudgetFixture(t)
	w := fx.putCapacity(fx.capacityBody(2*fakeWindow, 2, 2048))
	wantStatus(t, w, http.StatusOK)
	var row modelCapacity
	if err := json.Unmarshal(w.Body.Bytes(), &row); err != nil {
		t.Fatal(err)
	}
	for _, got := range []modelCapacity{row, fx.capacityRow()} {
		if got.SessionShare != fakeWindow || got.ReplyTokens != 2048 || got.MessageTokens != fakeWindow-2048 ||
			got.Configured == nil || got.Configured.ReplyTokens != 2048 {
			t.Fatalf("capacity = %+v, want window %d, reply 2048, messages %d", got, fakeWindow, fakeWindow-2048)
		}
	}
}

func TestBudgetAPI_RejectsAnImpossibleReplyBudget(t *testing.T) {
	fx := newBudgetFixture(t)
	for _, reply := range []int64{fakeWindow, fakeWindow + 1, -1} {
		w := fx.putCapacity(fx.capacityBody(2*fakeWindow, 2, reply))
		wantStatus(t, w, http.StatusBadRequest)
	}
	// Nothing was stored by the refused calls.
	w := fx.call(http.MethodGet, capacityAPIPath, "", "")
	if strings.Contains(w.Body.String(), `"replyTokens":8192`) {
		t.Fatalf("a refused reply budget was stored: %s", w.Body.String())
	}
}

func TestBudgetAPI_SessionTurnsObeyTheConfiguration(t *testing.T) {
	fx := newBudgetFixture(t)
	wantStatus(t, fx.putCapacity(fx.capacityBody(2*fakeWindow, 2, 2048)), http.StatusOK)

	// A turn with no reply length of its own: the engine is told to refuse
	// rather than cut, and to stop at the reply budget.
	w := fx.chat("s1", 3000, "")
	wantStatus(t, w, http.StatusOK)
	sent := fx.engine.lastBody()
	opts, _ := sent["options"].(map[string]any)
	if sent["truncate"] != false || opts["num_predict"] != float64(2048) {
		t.Fatalf("engine got %v, want truncate false and num_predict 2048", sent)
	}
	for header, want := range map[string]string{
		contextLimitHeader:   fmt.Sprint(fakeWindow),
		contextReplyHeader:   "2048",
		contextMessageHeader: fmt.Sprint(fakeWindow - 2048),
	} {
		if got := w.Header().Get(header); got != want {
			t.Fatalf("%s = %q, want %q", header, got, want)
		}
	}

	// The session API reports the same limits and the room left: 6144 less
	// the 3000 prompt and 2048 reply just measured.
	c := fx.sessionContext("s1")
	if c.WindowTokens != fakeWindow || c.ReplyTokens != 2048 || c.ConversationLimitTokens != fakeWindow-2048 ||
		c.PostRoomTokens == nil || *c.PostRoomTokens != fakeWindow-2048-3000-2048 {
		t.Fatalf("session context = %+v", c)
	}
	if c.Tokens != 3000+2048 || c.PromptTokens != 3000 || c.CompletionTokens != 2048 {
		t.Fatalf("measured %+v, want 3000 prompt and 2048 completion", c)
	}
}

func TestBudgetAPI_ClientSettingsAreKept(t *testing.T) {
	fx := newBudgetFixture(t)
	wantStatus(t, fx.putCapacity(fx.capacityBody(2*fakeWindow, 2, 2048)), http.StatusOK)

	wantStatus(t, fx.chat("s1", 100, `,"truncate":true,"options":{"num_predict":50,"temperature":0.1}`), http.StatusOK)
	sent := fx.engine.lastBody()
	opts, _ := sent["options"].(map[string]any)
	if sent["truncate"] != true || opts["num_predict"] != float64(50) || opts["temperature"] != 0.1 {
		t.Fatalf("engine got %v, want the client's own truncate, num_predict and temperature", sent)
	}
}

func TestBudgetAPI_OversizedMessageIsRefusedNotCut(t *testing.T) {
	fx := newBudgetFixture(t)
	wantStatus(t, fx.putCapacity(fx.capacityBody(2*fakeWindow, 2, 2048)), http.StatusOK)

	// In a session: the engine's exact refusal reaches the client.
	w := fx.chat("s1", 12000, "")
	wantStatus(t, w, http.StatusBadRequest)
	if !strings.Contains(w.Body.String(), "12000 tokens") || !strings.Contains(w.Body.String(), fmt.Sprint(fakeWindow)) {
		t.Fatalf("body = %q, want the engine's count and window", w.Body.String())
	}

	// Outside a session the request goes through exactly as sent, so the
	// engine's own behaviour (a silent cut) applies.
	w = fx.chat("", 12000, "")
	wantStatus(t, w, http.StatusOK)
	if _, set := fx.engine.lastBody()["truncate"]; set {
		t.Fatalf("a request outside a session was rewritten: %v", fx.engine.lastBody())
	}
	if !strings.Contains(w.Body.String(), fmt.Sprintf(`"prompt_eval_count":%d`, fakeWindow/2)) {
		t.Fatalf("body = %q, want the engine's silent cut", w.Body.String())
	}
}

func TestBudgetAPI_ConversationLimitAndLiveChanges(t *testing.T) {
	fx := newBudgetFixture(t)
	wantStatus(t, fx.putCapacity(fx.capacityBody(2*fakeWindow, 2, 2048)), http.StatusOK)

	// 4500 prompt + 2048 reply = 6548, past the 6144 the conversation may use.
	wantStatus(t, fx.chat("s1", 4500, ""), http.StatusOK)
	if c := fx.sessionContext("s1"); c.PostRoomTokens == nil || *c.PostRoomTokens != 0 {
		t.Fatalf("post room = %v, want 0 once the conversation is past its limit", c.PostRoomTokens)
	}
	w := fx.chat("s1", 4600, "")
	wantStatus(t, w, http.StatusBadRequest)
	for _, want := range []string{"6548", fmt.Sprint(fakeWindow), "2048", fmt.Sprint(fakeWindow - 2048)} {
		if !strings.Contains(w.Body.String(), want) {
			t.Fatalf("refusal %q does not name %s", w.Body.String(), want)
		}
	}
	// A shorter, compacted conversation is let through.
	wantStatus(t, fx.chat("s1", 1000, ""), http.StatusOK)
	wantStatus(t, fx.chat("s1", 4500, ""), http.StatusOK) // measured 6548 again

	// A smaller reply budget leaves the conversation more room at once: the
	// same turn that was refused now fits under 8192 - 1024 = 7168.
	wantStatus(t, fx.putCapacity(fx.capacityBody(2*fakeWindow, 2, 1024)), http.StatusOK)
	w = fx.chat("s1", 4600, "")
	wantStatus(t, w, http.StatusOK)
	if got := w.Header().Get(contextMessageHeader); got != "7168" {
		t.Fatalf("%s = %q after the change, want 7168", contextMessageHeader, got)
	}
	if opts, _ := fx.engine.lastBody()["options"].(map[string]any); opts["num_predict"] != float64(1024) {
		t.Fatalf("engine got num_predict %v, want the new 1024 budget", opts["num_predict"])
	}

	// With the setting cleared there is no budget to enforce or report, but
	// a session's conversation is still never silently cut.
	// 200 with the row while live sessions still make the model known, 204
	// when nothing is.
	if w := fx.call(http.MethodPut, capacityAPIPath, "", fmt.Sprintf(`{"engine":%q,"model":%q,"clear":true}`, fx.tc.profile.Name, fx.tc.requestedModel)); w.Code != http.StatusOK && w.Code != http.StatusNoContent {
		t.Fatalf("clear: status %d, body %q", w.Code, w.Body.String())
	}
	if row := fx.capacityRow(); row.Configured != nil || row.ReplyTokens != 0 || row.MessageTokens != 0 {
		t.Fatalf("after clearing, capacity = %+v, want no setting and no budget", row)
	}
	w = fx.chat("s2", 100, "")
	wantStatus(t, w, http.StatusOK)
	if w.Header().Get(contextReplyHeader) != "" || w.Header().Get(contextMessageHeader) != "" {
		t.Fatalf("budget headers %q / %q with no budget set", w.Header().Get(contextReplyHeader), w.Header().Get(contextMessageHeader))
	}
	sent := fx.engine.lastBody()
	if opts, _ := sent["options"].(map[string]any); sent["truncate"] != false || opts != nil {
		t.Fatalf("engine got %v, want truncate false and no reply cap", sent)
	}
}

func TestBudgetAPI_OpenAIRouteCapsTheReply(t *testing.T) {
	fx := newBudgetFixture(t)
	wantStatus(t, fx.putCapacity(fx.capacityBody(2*fakeWindow, 2, 2048)), http.StatusOK)

	body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}]}`, fx.tc.requestedModel)
	w := fx.call(http.MethodPost, "/v1/chat/completions", "s1", body)
	wantStatus(t, w, http.StatusOK)
	sent := fx.engine.lastBody()
	if sent["max_tokens"] != float64(2048) {
		t.Fatalf("engine got %v, want max_tokens 2048", sent)
	}
	if _, set := sent["truncate"]; set {
		t.Fatal("the OpenAI route has no truncate switch; PAIR must not invent one")
	}
}
