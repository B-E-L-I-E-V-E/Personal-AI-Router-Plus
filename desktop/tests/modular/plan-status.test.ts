// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { SavedCapacityPlan } from '@/shared/types/sessions'
import { planStatus } from '@/ui/components/SessionsSettings/plan-status'

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
        expect(planStatus(plan({ applied: true }), true)).toBe(
            'Plan: 21 × 40,960 tokens, 8,192 for replies — applied'
        )
        // At most half the window, as the proxy computes it.
        expect(
            planStatus(plan({ windowTokens: 8192, sessions: 2, applied: true }), true)
        ).toContain('4,096 for replies')
        expect(planStatus(plan({ replyTokens: 2048, applied: true }), true)).toContain(
            '2,048 for replies'
        )
    })

    it('says when and how a pending plan will apply', () => {
        expect(planStatus(plan(), true)).toContain('will apply when this node is idle')
        expect(planStatus(plan(), false)).toContain('not applied')
        expect(planStatus(plan({ engine: 'lmstudio' }), true)).toContain('LM Studio')
        expect(planStatus(plan({ cacheType: 'q8_0' }), true)).toContain('8-bit cache')
    })
})
