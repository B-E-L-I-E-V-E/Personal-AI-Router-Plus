// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"time"
)

// Context slots are how a model's capacity on this node is shared among
// sessions. A model configured for N sessions (capacity.go) has N slots. A
// session takes one when it is first admitted to this node for that model and
// keeps it, across turns, until it ends, expires, or moves to another node.
// While every slot is held, a new session's request goes to another node that
// has the model, or waits here in arrival order for a slot to free.
const (
	defaultQueueTimeout = time.Minute
	// maxQueueTimeout leaves room inside jobDeadline for the dispatch itself.
	maxQueueTimeout  = 5 * time.Minute
	defaultMaxQueued = 64
)

var (
	errQueueTimeout = errors.New("every context slot for this model is in use; timed out waiting for one")
	errQueueFull    = errors.New("every context slot for this model is in use and the wait queue is full")
)

// slotWaiter is one request waiting for a slot, identified by its session.
type slotWaiter struct{ sessionID string }

// heldSlotsLocked counts the sessions holding a slot for one model.
func (s *sessionStore) heldSlotsLocked(k capacityKey) int {
	n := 0
	for _, sess := range s.sessions {
		if sess.slots[k.Engine] == k.Model {
			n++
		}
	}
	return n
}

// slotStatus is a model's slot use, for the capacity report.
type slotStatus struct {
	Held   int
	Queued int
}

func (s *sessionStore) slotStatus(k capacityKey) slotStatus {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slotStatus{Held: s.heldSlotsLocked(k), Queued: len(s.queues[k])}
}

// signalLocked wakes every waiter so each re-checks for a free slot.
func (s *sessionStore) signalLocked() {
	close(s.changed)
	s.changed = make(chan struct{})
}

func (s *sessionStore) releaseSlotLocked(sess *session, engine string) {
	if _, ok := sess.slots[engine]; ok {
		delete(sess.slots, engine)
		s.signalLocked()
	}
}

func (s *sessionStore) releaseSlotsLocked(sess *session) {
	clear(sess.slots)
	// Always wake waiters: besides freeing slots, this runs when a session
	// ends, and that session may itself be one of the waiters.
	s.signalLocked()
}

// holdsSlot reports whether a session already holds a slot for the model.
func (s *sessionStore) holdsSlot(id string, k capacityKey) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[id]
	return ok && sess.slots[k.Engine] == k.Model
}

// tryTakeSlotLocked gives the session a slot if one is free and no earlier
// arrival is waiting for it. A session switching models on an engine gives up
// its old slot first.
func (s *sessionStore) tryTakeSlotLocked(sess *session, k capacityKey, limit int) bool {
	if sess.slots[k.Engine] == k.Model {
		return true
	}
	if q := s.queues[k]; len(q) > 0 && q[0].sessionID != sess.id {
		return false
	}
	held := s.heldSlotsLocked(k)
	if _, other := sess.slots[k.Engine]; other {
		// The old slot is for a different model, so it does not count here,
		// but it must not be kept alongside the new one.
		s.releaseSlotLocked(sess, k.Engine)
	}
	if held >= limit {
		return false
	}
	sess.slots[k.Engine] = k.Model
	return true
}

// takeSlotNow takes a free slot without waiting. It reports false when every
// slot is held or earlier arrivals are waiting.
func (s *sessionStore) takeSlotNow(id string, k capacityKey, limit int) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[id]
	if !ok {
		return false
	}
	return s.tryTakeSlotLocked(sess, k, limit)
}

// waitForSlot queues the session for a slot and waits until it gets one, ctx
// ends, or the configured queue timeout passes. limit is re-read on every
// wake, so changing the setting takes effect for requests already waiting.
func (s *sessionStore) waitForSlot(ctx context.Context, id string, k capacityKey, limit func() int) error {
	s.mu.Lock()
	sess, ok := s.sessions[id]
	if !ok {
		s.mu.Unlock()
		return errSessionGone
	}
	if err := s.refusedLocked(); err != nil {
		s.mu.Unlock()
		return err
	}
	if s.tryTakeSlotLocked(sess, k, limit()) {
		s.mu.Unlock()
		return nil
	}
	timeout, maxQueued := s.cfg.QueueTimeout, s.cfg.MaxQueued
	if timeout <= 0 {
		s.mu.Unlock()
		return errQueueTimeout
	}
	if len(s.queues[k]) >= maxQueued {
		s.mu.Unlock()
		return errQueueFull
	}
	w := &slotWaiter{sessionID: id}
	s.queues[k] = append(s.queues[k], w)
	s.mu.Unlock()

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	defer s.dequeue(k, w)
	for {
		s.mu.Lock()
		sess, ok := s.sessions[id]
		if !ok {
			s.mu.Unlock()
			return errSessionGone
		}
		if err := s.refusedLocked(); err != nil {
			s.mu.Unlock()
			return err
		}
		// Only the head of the queue may take a slot, so arrival order holds.
		if q := s.queues[k]; len(q) > 0 && q[0] == w && s.tryTakeSlotLocked(sess, k, limit()) {
			s.mu.Unlock()
			return nil
		}
		changed := s.changed
		s.mu.Unlock()
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
			return errQueueTimeout
		case <-changed:
		}
	}
}

// dequeue removes a waiter, and wakes the rest: the next in line may now be at
// the head.
func (s *sessionStore) dequeue(k capacityKey, w *slotWaiter) {
	s.mu.Lock()
	defer s.mu.Unlock()
	q := s.queues[k]
	for i, x := range q {
		if x == w {
			q = append(q[:i], q[i+1:]...)
			break
		}
	}
	if len(q) == 0 {
		delete(s.queues, k)
	} else {
		s.queues[k] = q
	}
	s.signalLocked()
}

// slotKeys lists every model with a held slot or a waiting request.
func (s *sessionStore) slotKeys() map[capacityKey]bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[capacityKey]bool)
	for _, sess := range s.sessions {
		for engine, model := range sess.slots {
			out[capacityKey{Engine: engine, Model: model}] = true
		}
	}
	for k := range s.queues {
		out[k] = true
	}
	return out
}

// activityCounts is every live session on this node (its own and those peers
// hold here) and every request waiting for a slot, for the idle check.
func (s *sessionStore) activityCounts() (sessions, queued int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, q := range s.queues {
		queued += len(q)
	}
	return len(s.sessions), queued
}

// refusedLocked is why this node takes no slot requests now, if it does not:
// a paused or stopped node (nodestate.go) lets no waiter through.
func (s *sessionStore) refusedLocked() error {
	if s.refuse == nil {
		return nil
	}
	return s.refuse()
}

// refuseWaiters wakes every request waiting for a slot so each re-checks
// whether this node still takes requests.
func (s *sessionStore) refuseWaiters() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.signalLocked()
}

// releaseAllSlots frees every slot sessions hold on this node. The sessions
// stay: their next turn takes a slot again wherever it runs.
func (s *sessionStore) releaseAllSlots() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, sess := range s.sessions {
		clear(sess.slots)
	}
	s.signalLocked()
}
