// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { PlannerState } from '@/shared/types/sessions'
import {
    findPlan,
    removePlanAndLimit,
    useCapacityPlannerStore
} from '@/ui/stores/capacity-planner.store'

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
const setPlanner = vi.fn()
const setCapacity = vi.fn()

beforeEach(() => {
    getPlanner.mockReset()
    setPlanner.mockReset()
    setCapacity.mockReset()
    useCapacityPlannerStore.setState({ planner: null })
    vi.stubGlobal('window', { pairApi: { sessions: { getPlanner, setPlanner, setCapacity } } })
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

    it('removing a plan also lifts the limit it set', async () => {
        const after = { ...state, plans: [] }
        setPlanner.mockResolvedValueOnce(after)
        setCapacity.mockResolvedValueOnce(null)
        await expect(removePlanAndLimit('ollama', 'qwen3:0.6b')).resolves.toBe(after)
        expect(setPlanner).toHaveBeenCalledWith({
            remove: { engine: 'ollama', model: 'qwen3:0.6b' }
        })
        expect(setCapacity).toHaveBeenCalledWith({
            engine: 'ollama',
            model: 'qwen3:0.6b',
            clear: true
        })
    })
})
