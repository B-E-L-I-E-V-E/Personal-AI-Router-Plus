// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { PlannerState } from '@/shared/types/sessions'
import { findPlan, useCapacityPlannerStore } from '@/ui/stores/capacity-planner.store'

const state: PlannerState = {
    autoReload: true,
    plans: [
        {
            engine: 'ollama',
            model: 'qwen3:0.6b',
            windowTokens: 40960,
            sessions: 21,
            cacheType: 'f16',
            reservePercent: 10,
            offloadExperts: false,
            replyTokens: 0,
            applied: true,
            appliedAt: 1,
            savedAt: 1
        }
    ],
    activity: { sessions: 0, activeRequests: 0, queued: 0, idle: true }
}

const getPlanner = vi.fn()

beforeEach(() => {
    getPlanner.mockReset()
    useCapacityPlannerStore.setState({ planner: null })
    vi.stubGlobal('window', { pairApi: { sessions: { getPlanner } } })
})

describe('capacity planner store', () => {
    it('finds a model’s plan by engine and model', () => {
        expect(findPlan(state, 'ollama', 'qwen3:0.6b')?.sessions).toBe(21)
        expect(findPlan(state, 'llamacpp', 'qwen3:0.6b')).toBeUndefined()
        expect(findPlan(null, 'ollama', 'qwen3:0.6b')).toBeUndefined()
    })

    it('refreshes from the router and keeps the last answer when it is unreachable', async () => {
        getPlanner.mockResolvedValueOnce(state)
        await useCapacityPlannerStore.getState().refresh()
        expect(useCapacityPlannerStore.getState().planner).toBe(state)

        getPlanner.mockRejectedValueOnce(new Error('not reachable'))
        await useCapacityPlannerStore.getState().refresh()
        expect(useCapacityPlannerStore.getState().planner).toBe(state)
    })
})
