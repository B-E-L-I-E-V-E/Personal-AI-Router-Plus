// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { SavedCapacityPlan } from '@/shared/types/sessions'
import {
    loadDiffersFromPlan,
    planNeedsApply,
    planStatus
} from '@/ui/components/SessionsSettings/plan-status'

function plan(overrides: Partial<SavedCapacityPlan> = {}): SavedCapacityPlan {
    return {
        engine: 'ollama',
        model: 'm',
        windowTokens: 40960,
        sessions: 21,
        cacheType: 'f16',
        reservePercent: 10,
        cpuExpertLayers: 0,
        replyTokens: 0,
        applied: false,
        appliedAt: 0,
        savedAt: 0,
        ...overrides
    }
}

describe('plan status line', () => {
    it('names sessions, window and the reply budget the proxy applies', () => {
        // No reply budget saved: the 8,192 default.
        expect(planStatus(plan({ applied: true }), true, 0)).toBe(
            'Plan: 21 × 40,960 tokens, 8,192 for replies — applied'
        )
        // At most half the window, as the proxy computes it.
        expect(
            planStatus(plan({ windowTokens: 8192, sessions: 2, applied: true }), true, 0)
        ).toContain('4,096 for replies')
        expect(planStatus(plan({ replyTokens: 2048, applied: true }), true, 0)).toContain(
            '2,048 for replies'
        )
    })

    it('says when and how a pending plan will apply', () => {
        expect(planStatus(plan(), true, 0)).toContain('will apply when this node is idle')
        expect(planStatus(plan(), false, 0)).toContain('not applied')
        expect(planStatus(plan({ cacheType: 'q8_0' }), true, 0)).toContain('8-bit cache')
    })

    it('reports what LM Studio has loaded, not the saved flag', () => {
        const lm = plan({ engine: 'lmstudio', windowTokens: 262144, sessions: 4, applied: true })
        // Saved as applied, but the model runs with the 8,192 it was loaded with.
        expect(planStatus(lm, true, 8192)).toContain('not applied: loaded with 8,192 tokens')
        expect(planStatus(lm, false, 8192)).toContain('Press Apply now to reload')
        expect(planStatus(lm, true, 0)).toContain('the model is not loaded')
        expect(planStatus(lm, true, 262144)).toContain('— applied')
        expect(loadDiffersFromPlan(lm, 8192)).toBe(true)
        expect(loadDiffersFromPlan(lm, 262144)).toBe(false)
        // Other engines are restarted with the plan, so they have no live check.
        expect(loadDiffersFromPlan(plan({ applied: true }), 8192)).toBe(false)
    })

    it('offers Apply when the load differs, or the plan is pending', () => {
        const lm = plan({ engine: 'lmstudio', windowTokens: 262144, applied: true })
        expect(planNeedsApply(lm, 8192)).toBe(true)
        expect(planNeedsApply(lm, 262144)).toBe(false)
        expect(planNeedsApply(plan({ applied: false }), 0)).toBe(true)
        expect(planNeedsApply(plan({ applied: true }), 0)).toBe(false)
    })
})
