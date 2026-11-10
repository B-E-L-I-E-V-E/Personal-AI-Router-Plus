// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"
)

var uuidV4Pattern = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)

func TestValidRequestID(t *testing.T) {
	accepts := func(name, id string) {
		t.Run(name, func(t *testing.T) {
			if !validRequestID(id) {
				t.Fatalf("validRequestID(%q) = false, want true", id)
			}
		})
	}
	rejects := func(name, id string) {
		t.Run(name, func(t *testing.T) {
			if validRequestID(id) {
				t.Fatalf("validRequestID(%q) = true, want false", id)
			}
		})
	}

	accepts("uuid", "3f2b8c1e-7d4a-4f6b-9c2d-1a2b3c4d5e6f")
	accepts("ulid", "01JC3ZK8Q9V7X2M4N6P8R0T2W4")
	accepts("prefixed trace id", "agent-7:turn_12.retry-1")
	accepts("maximum length", strings.Repeat("a", maxRequestIDLen))

	rejects("empty", "")
	rejects("too long", strings.Repeat("a", maxRequestIDLen+1))
	rejects("contains space", "abc def")
	rejects("contains newline", "abc\ndef")
	rejects("contains slash", "abc/def")
	rejects("non-ascii", "réquest")
}

func TestNewRequestID_IsUniqueUUIDv4(t *testing.T) {
	seen := make(map[string]bool)
	for range 1000 {
		id := newRequestID()
		if !uuidV4Pattern.MatchString(id) {
			t.Fatalf("newRequestID() = %q, want a lowercase version 4 UUID", id)
		}
		if seen[id] {
			t.Fatalf("newRequestID() repeated %q", id)
		}
		seen[id] = true
	}
}

// decodeFrames parses the newline-delimited JSON-RPC frames a recRW captured.
func decodeFrames(t *testing.T, rec *recRW) []struct {
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
} {
	t.Helper()
	rec.mu.Lock()
	raw := append([]byte(nil), rec.b...)
	rec.mu.Unlock()

	var frames []struct {
		Method string          `json:"method"`
		Params json.RawMessage `json:"params"`
	}
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var f struct {
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.Unmarshal(line, &f); err != nil {
			t.Fatalf("decode frame %q: %v", line, err)
		}
		frames = append(frames, f)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan frames: %v", err)
	}
	return frames
}

// requestIDsIn returns the request IDs carried by every frame of one method.
// Facade notifications are engine-addressed ("ollama-proxy:proxy/request"), so
// the method matches either bare or after an address prefix. Request events use
// request_id; workload events carry workloadInfo.requestId.
func requestIDsIn(t *testing.T, rec *recRW, method string) []string {
	t.Helper()
	var ids []string
	for _, f := range decodeFrames(t, rec) {
		if f.Method != method && !strings.HasSuffix(f.Method, ":"+method) {
			continue
		}
		var p struct {
			RequestID    string `json:"request_id"`
			WorkloadInfo struct {
				RequestID string `json:"requestId"`
			} `json:"workloadInfo"`
		}
		if err := json.Unmarshal(f.Params, &p); err != nil {
			t.Fatalf("decode %s params: %v", method, err)
		}
		ids = append(ids, p.RequestID+p.WorkloadInfo.RequestID)
	}
	return ids
}

func TestHandleHTTP_RequestID(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		// serve sends one inference request through a proxy fronting a single
		// upstream and returns what the client and the upstream each saw.
		serve := func(t *testing.T, clientID string) (resp *http.Response, upstreamGot string, rec *recRW) {
			t.Helper()
			var mu sync.Mutex
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				upstreamGot = r.Header.Get(requestIDHeader)
				mu.Unlock()
				// An upstream that sets its own X-Request-ID must not produce a
				// second value on the client response.
				w.Header().Set(requestIDHeader, "upstream-own-id")
				w.WriteHeader(http.StatusOK)
				if _, err := io.WriteString(w, `{"done":true}`); err != nil {
					t.Errorf("upstream write: %v", err)
				}
			}))
			t.Cleanup(upstream.Close)

			rec = &recRW{}
			disc := NewDiscovery()
			disc.AddManual(nodeForModel(t, "node-a", upstream.URL, tc.advertisedModel))
			p := newTestProxy(tc.profile, NewCodec(rec), disc, tc.profile.FacadePort)

			req := httptest.NewRequest(http.MethodPost, tc.inferencePath,
				strings.NewReader(fmt.Sprintf(`{"model":%q}`, tc.requestedModel)))
			if clientID != "" {
				req.Header.Set(requestIDHeader, clientID)
			}
			w := httptest.NewRecorder()
			p.soleFacade().handleHTTP(w, req)
			if !waitFor(t, rec, workloadCompletedMethod) {
				t.Fatal("workload:completed not emitted")
			}
			mu.Lock()
			defer mu.Unlock()
			return w.Result(), upstreamGot, rec
		}

		t.Run("echoes a valid client id", func(t *testing.T) {
			const id = "agent-7:turn-12"
			resp, upstreamGot, rec := serve(t, id)
			if got := resp.Header.Values(requestIDHeader); len(got) != 1 || got[0] != id {
				t.Fatalf("response %s = %q, want exactly [%q]", requestIDHeader, got, id)
			}
			if upstreamGot != id {
				t.Fatalf("upstream saw %s %q, want %q", requestIDHeader, upstreamGot, id)
			}
			for _, method := range []string{"proxy/request-started", "proxy/request", workloadSubmittedMethod, workloadCompletedMethod} {
				ids := requestIDsIn(t, rec, method)
				if len(ids) == 0 {
					t.Fatalf("no %s frame emitted", method)
				}
				for _, got := range ids {
					if got != id {
						t.Fatalf("%s carried request id %q, want %q", method, got, id)
					}
				}
			}
		})

		t.Run("mints an id when the client sends none", func(t *testing.T) {
			resp, upstreamGot, _ := serve(t, "")
			got := resp.Header.Get(requestIDHeader)
			if !uuidV4Pattern.MatchString(got) {
				t.Fatalf("response %s = %q, want a minted version 4 UUID", requestIDHeader, got)
			}
			if upstreamGot != got {
				t.Fatalf("upstream saw %q, client saw %q; want the same id", upstreamGot, got)
			}
		})

		t.Run("replaces an invalid client id", func(t *testing.T) {
			resp, _, _ := serve(t, "has spaces in it")
			got := resp.Header.Get(requestIDHeader)
			if !uuidV4Pattern.MatchString(got) {
				t.Fatalf("response %s = %q, want a minted version 4 UUID", requestIDHeader, got)
			}
		})
	})
}

func TestHandleHTTP_RequestIDOnRejection(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		rec := &recRW{}
		p := newTestProxy(tc.profile, NewCodec(rec), NewDiscovery(), tc.profile.FacadePort)

		const id = "rejected-request-1"
		req := httptest.NewRequest(http.MethodPost, tc.inferencePath,
			strings.NewReader(fmt.Sprintf(`{"model":%q}`, tc.requestedModel)))
		req.Header.Set(requestIDHeader, id)
		w := httptest.NewRecorder()
		p.soleFacade().handleHTTP(w, req)

		if w.Code != http.StatusBadGateway {
			t.Fatalf("status = %d, want %d with no eligible node", w.Code, http.StatusBadGateway)
		}
		if got := w.Header().Get(requestIDHeader); got != id {
			t.Fatalf("rejection %s = %q, want %q", requestIDHeader, got, id)
		}
		if !waitFor(t, rec, "proxy/request") {
			t.Fatal("proxy/request not emitted for the rejection")
		}
		ids := requestIDsIn(t, rec, "proxy/request")
		if len(ids) != 1 || ids[0] != id {
			t.Fatalf("proxy/request request ids = %q, want [%q]", ids, id)
		}
	})
}
