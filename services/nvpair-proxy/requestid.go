// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/rand"
	"fmt"
	"net/http"
)

// requestIDHeader carries the caller-visible request ID. A client may supply
// one; otherwise the proxy mints it. Either way the same value is echoed on the
// response, forwarded upstream (so a peer's ingress and the engine see it), and
// stamped on the proxy's request events and workload records, which lets a
// client match a response, a log line, and a job card to one request.
//
// It is distinct from the facade's internal counter (reqID): that counter is a
// per-process store key for workload identity and restarts at 1, so it cannot
// serve as a caller-facing correlation handle.
const requestIDHeader = "X-Request-ID"

// maxRequestIDLen bounds a client-supplied ID. Long enough for a UUID, a ULID,
// or a prefixed trace ID; short enough that it cannot bloat logs and events.
const maxRequestIDLen = 128

// resolveRequestID returns the client's request ID when it is acceptable and a
// freshly minted one otherwise. A rejected value is replaced rather than
// refused: the request ID is a correlation aid, never a reason to fail
// inference, and the response header tells the caller which value was used.
func resolveRequestID(h http.Header) string {
	if id := h.Get(requestIDHeader); validRequestID(id) {
		return id
	}
	return newRequestID()
}

// validRequestID accepts 1–128 characters from a conservative set: letters,
// digits, and . _ : -. That covers UUIDs, ULIDs and common trace formats while
// keeping header injection, whitespace and log-forging characters out.
func validRequestID(id string) bool {
	if id == "" || len(id) > maxRequestIDLen {
		return false
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case c == '.', c == '_', c == ':', c == '-':
		default:
			return false
		}
	}
	return true
}

// newRequestID mints a random RFC 9562 version 4 UUID.
func newRequestID() string {
	var b [16]byte
	// crypto/rand.Read never returns an error on supported platforms.
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
