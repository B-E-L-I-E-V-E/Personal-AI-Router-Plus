// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { ModelCapacity, SavedCapacityPlan } from '@/shared/types/sessions'
import {
    capacitySummary,
    findCapacity,
    modelConfigLines
} from '@/ui/components/ModelManager/model-config-summary'

function capacity(overrides: Partial<ModelCapacity> = {}): ModelCapacity {
    return {
        engine: 'ollama',
        model: 'qwen3:0.6b',
        configured: { contextTokens: 163840, maxSessions: 4, replyTokens: 8192 },
        detectedWindowTokens: 0,
        contextTokens: 163840,
        maxSessions: 4,
        sessionContextTokens: 40960,
        replyTokens: 8192,
        messageTokens: 32768,
        source: 'configured',
        heldSlots: 0,
        queued: 0,
        usedContextTokens: 0,
        warning: '',
        ...overrides
    }
}

function plan(overrides: Partial<SavedCapacityPlan> = {}): SavedCapacityPlan {
    return {
        engine: 'ollama',
        model: 'qwen3:0.6b',
        windowTokens: 40960,
        sessions: 4,
        cacheType: 'f16',
        reservePercent: 10,
        offloadExperts: false,
        replyTokens: 0,
        applied: true,
        appliedAt: 1,
        savedAt: 1,
        ...overrides
    }
}

describe('model capacity summary', () => {
    it('lists sessions, per-session window, response and post context', () => {
        expect(capacitySummary(capacity())).toBe(
            '4 sessions · 40,960 per session · 8,192 response · 32,768 post'
        )
    })

    it('says how many sessions are in use and waiting', () => {
        expect(capacitySummary(capacity({ heldSlots: 1 }))).toBe(
            '4 sessions (1 in use) · 40,960 per session · 8,192 response · 32,768 post'
        )
        expect(capacitySummary(capacity({ maxSessions: 1, heldSlots: 1, queued: 2 }))).toMatch(
            /^1 session \(1 in use, 2 waiting\) · /
        )
    })

    it('leaves out response and post context when no reply budget is set', () => {
        expect(capacitySummary(capacity({ replyTokens: 0, messageTokens: 0 }))).toBe(
            '4 sessions · 40,960 per session'
        )
    })

    it('shows only the window an engine reports, marked as reported', () => {
        const detected = capacity({
            configured: null,
            source: 'detected',
            contextTokens: 0,
            maxSessions: 0,
            replyTokens: 0,
            messageTokens: 0
        })
        expect(capacitySummary(detected)).toBe('40,960 per session · as the engine reports')
    })

    it('shows nothing when the proxy knows nothing', () => {
        expect(capacitySummary(undefined)).toBeNull()
        expect(
            capacitySummary(
                capacity({ source: 'unknown', sessionContextTokens: 0, maxSessions: 0 })
            )
        ).toBeNull()
    })

    it('finds a model’s row by engine and model', () => {
        const report = [capacity(), capacity({ engine: 'llamacpp', maxSessions: 2 })]
        expect(findCapacity(report, 'llamacpp', 'qwen3:0.6b')?.maxSessions).toBe(2)
        expect(findCapacity(report, 'lmstudio', 'qwen3:0.6b')).toBeUndefined()
        expect(findCapacity(null, 'ollama', 'qwen3:0.6b')).toBeUndefined()
    })
})

describe('model row config lines', () => {
    it('shows only the summary for an applied plan', () => {
        expect(modelConfigLines(capacity(), plan(), true)).toEqual({
            summary: '4 sessions · 40,960 per session · 8,192 response · 32,768 post'
        })
    })

    it('marks what is in force apart from a pending plan', () => {
        const lines = modelConfigLines(
            capacity(),
            plan({ applied: false, sessions: 2, windowTokens: 65536 }),
            false
        )
        expect(lines.summary).toBe(
            'Now: 4 sessions · 40,960 per session · 8,192 response · 32,768 post'
        )
        expect(lines.planLine).toBe(
            'Plan: 2 × 65,536 tokens, 8,192 for replies — not applied; press Apply'
        )
    })

    it('keeps the plan line when the proxy reports nothing', () => {
        expect(modelConfigLines(undefined, plan(), true)).toEqual({
            planLine: 'Plan: 4 × 40,960 tokens, 8,192 for replies — applied'
        })
    })

    it('keeps the plan line when an applied plan’s limit was lifted', () => {
        const detected = capacity({ configured: null, source: 'detected', maxSessions: 0 })
        expect(modelConfigLines(detected, plan(), true).planLine).toMatch(/^Plan: /)
    })

    it('shows nothing without a plan or report', () => {
        expect(modelConfigLines(undefined, undefined, true)).toEqual({})
    })
})
