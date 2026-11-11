// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"strconv"
)

// A session's window is one budget shared by its messages, its history and
// the model's replies. Two engine behaviours break that budget silently, so
// PAIR adjusts a session request before this node's engine sees it:
//
//   - Ollama accepts a conversation longer than its window and keeps only
//     the last half of the window, answering as if nothing were lost. Its
//     own endpoints take "truncate": false, which makes it refuse instead,
//     naming the request's exact token count. Its OpenAI-compatible
//     endpoints have no such switch.
//   - A reply with no length of its own runs until the model stops, past the
//     window if need be, by forgetting the start of the conversation. A
//     reply budget caps it: options.num_predict on Ollama's endpoints,
//     max_tokens on the OpenAI-compatible ones.
//
// Values the client set are kept. Only sessions are adjusted; other requests
// are forwarded exactly as sent.

// withSessionBudget returns body with the engine's truncation turned off and
// the reply capped at replyTokens (when positive), wherever the client left
// them unset. It returns body unchanged when there is nothing to set or it
// cannot rewrite the body safely.
func withSessionBudget(body []byte, engine, path string, replyTokens int64) []byte {
	var fields map[string]json.RawMessage
	if json.Unmarshal(body, &fields) != nil || fields == nil {
		return body
	}
	changed := false
	switch {
	case engine == "ollama" && (path == "/api/chat" || path == "/api/generate"):
		if _, ok := fields["truncate"]; !ok {
			fields["truncate"] = json.RawMessage("false")
			changed = true
		}
		if replyTokens > 0 {
			opts := map[string]json.RawMessage{}
			if raw, ok := fields["options"]; ok && len(raw) > 0 && !bytes.Equal(raw, []byte("null")) {
				if json.Unmarshal(raw, &opts) != nil {
					return body
				}
			}
			if _, ok := opts["num_predict"]; !ok {
				opts["num_predict"] = json.RawMessage(strconv.FormatInt(replyTokens, 10))
				encoded, err := json.Marshal(opts)
				if err != nil {
					return body
				}
				fields["options"] = encoded
				changed = true
			}
		}
	case isOpenAIInferencePath(path):
		_, max := fields["max_tokens"]
		_, maxCompletion := fields["max_completion_tokens"]
		if replyTokens > 0 && !max && !maxCompletion {
			fields["max_tokens"] = json.RawMessage(strconv.FormatInt(replyTokens, 10))
			changed = true
		}
	}
	if !changed {
		return body
	}
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(fields); err != nil {
		return body
	}
	return bytes.TrimRight(out.Bytes(), "\n")
}
