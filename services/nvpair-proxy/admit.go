// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"time"
)

const (
	// queueWaitHeader reports how long a request waited for a context slot.
	queueWaitHeader = "X-PAIR-Queue-Wait-Ms"
	// contextTokensHeader is the session's context as last measured, before
	// this turn. contextLimitHeader is its share on the node that served it,
	// sent only when that node is this one and the share is known.
	contextTokensHeader = "X-PAIR-Context-Session-Tokens"
	contextLimitHeader  = "X-PAIR-Context-Session-Limit"
	// contextReplyHeader is the part of that window kept for replies, and
	// contextMessageHeader what remains for the conversation: the most its
	// messages and history may hold. Sent with the limit when a reply budget
	// is set.
	contextReplyHeader   = "X-PAIR-Context-Reply-Tokens"
	contextMessageHeader = "X-PAIR-Context-Message-Limit"
)

// errContextExceeded refuses a session that has outgrown its share and is not
// sending a shorter conversation. Waiting would not help, so it is not queued.
var errContextExceeded = errors.New("the session's conversation has reached its limit on this node")

// admissionStatus maps an admission error to its HTTP status.
func admissionStatus(err error) int {
	switch {
	case errors.Is(err, errContextExceeded):
		return http.StatusBadRequest
	case errors.Is(err, errSessionGone):
		return http.StatusGone
	}
	return http.StatusServiceUnavailable
}

// admitSession applies this node's context capacity to one session request.
// Only this node's own candidate is gated: capacity on other nodes is theirs
// to enforce. It returns the candidates the request may use, the limit that
// applied, and how long it waited for a slot.
//
//   - Outgrown: the session's last measured context is at or above its share
//     and this request is not shorter than the one that produced it. A shorter
//     request is a compacted conversation, so it is let through and measured
//     again. Estimating tokens from request bytes would refuse image-heavy
//     requests that fit, so PAIR decides only on what the engine measured.
//   - Slot: the session holds a slot, or takes a free one; otherwise this node
//     is left out while another node can serve, and the request waits for a
//     slot only when none can.
func (f *facade) admitSession(ctx context.Context, id string, sess sessionView, model string, bodyLen int, cands []candidate) ([]candidate, modelCapacity, time.Duration, error) {
	p := f.host
	local := -1
	for i, c := range cands {
		if c.local {
			local = i
			break
		}
	}
	if local < 0 {
		return cands, modelCapacity{}, 0, nil
	}
	k := capacityKey{Engine: f.profile.Name, Model: model}
	lim := p.limitFor(f, k)
	if lim.MaxSessions == 0 && lim.SessionShare == 0 {
		return cands, lim, 0, nil
	}
	others := make([]candidate, 0, len(cands)-1)
	others = append(others, cands[:local]...)
	others = append(others, cands[local+1:]...)

	if lim.SessionShare > 0 {
		// With a reply budget, the conversation may use only the rest of the
		// window, so the model always has room to answer.
		limit := lim.SessionShare
		if lim.MessageTokens > 0 {
			limit = lim.MessageTokens
		}
		if c, ok := sess.Context[k.Engine]; ok && c.Model == model && c.Tokens >= limit && bodyLen >= c.BodyBytes {
			if len(others) > 0 {
				return others, lim, 0, nil
			}
			if lim.MessageTokens > 0 {
				return nil, lim, 0, fmt.Errorf("%w: it holds %d tokens, and each session's %d-token window keeps %d for the reply, leaving %d for the conversation; shorten the conversation or end the session",
					errContextExceeded, c.Tokens, lim.SessionShare, lim.ReplyTokens, lim.MessageTokens)
			}
			return nil, lim, 0, fmt.Errorf("%w: it holds %d tokens against a share of %d; shorten the conversation or end the session",
				errContextExceeded, c.Tokens, lim.SessionShare)
		}
	}
	if lim.MaxSessions == 0 {
		return cands, lim, 0, nil
	}
	if p.sessions.takeSlotNow(id, k, lim.MaxSessions) {
		return cands, lim, 0, nil
	}
	if len(others) > 0 {
		return others, lim, 0, nil
	}

	began := time.Now()
	err := p.sessions.waitForSlot(ctx, id, k, func() int {
		// A setting removed while requests wait lifts the limit.
		if n := p.limitFor(f, k).MaxSessions; n > 0 {
			return n
		}
		return math.MaxInt
	})
	waited := time.Since(began)
	if err != nil {
		if ctx.Err() != nil {
			if _, gone := p.sessions.get(id); gone != nil {
				return nil, lim, waited, errSessionGone
			}
		}
		return nil, lim, waited, err
	}
	return cands, lim, waited, nil
}
