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
            cpuExpertLayers: 0,
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
const getCapacity = vi.fn()
const getPeerCapacity = vi.fn()

beforeEach(() => {
    getPlanner.mockReset()
    // A peer's planner, unless a test says otherwise, is not answered.
    getPlanner.mockRejectedValue(new Error('not reachable'))
    setPlanner.mockReset()
    setCapacity.mockReset()
    getCapacity.mockReset()
    getCapacity.mockResolvedValue([])
    getPeerCapacity.mockReset()
    useCapacityPlannerStore.setState({
        planner: null,
        capacity: null,
        peerCapacity: {},
        peerPlanner: {},
        peerPlannerError: {}
    })
    vi.stubGlobal('window', {
        pairApi: {
            sessions: { getPlanner, setPlanner, setCapacity, getCapacity, getPeerCapacity }
        }
    })
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

    it('refreshes the capacity report alongside, each answer standing alone', async () => {
        const report = [
            {
                engine: 'ollama',
                model: 'qwen3:0.6b',
                configured: null,
                detectedWindowTokens: 40960,
                contextTokens: 0,
                maxSessions: 0,
                sessionContextTokens: 40960,
                replyTokens: 0,
                messageTokens: 0,
                source: 'detected',
                heldSlots: 0,
                queued: 0,
                usedContextTokens: 0,
                warning: ''
            }
        ]
        getPlanner.mockRejectedValueOnce(new Error('not reachable'))
        getCapacity.mockResolvedValueOnce(report)
        await useCapacityPlannerStore.getState().refresh()
        expect(useCapacityPlannerStore.getState().capacity).toBe(report)
        expect(useCapacityPlannerStore.getState().planner).toBeNull()

        getPlanner.mockResolvedValueOnce(state)
        getCapacity.mockRejectedValueOnce(new Error('not reachable'))
        await useCapacityPlannerStore.getState().refresh()
        expect(useCapacityPlannerStore.getState().planner).toBe(state)
        expect(useCapacityPlannerStore.getState().capacity).toBe(report)
    })

    it('keeps each peer’s report by node ID and drops it when the peer is unreachable', async () => {
        const report = [
            {
                engine: 'ollama',
                model: 'qwen3:0.6b',
                configured: null,
                detectedWindowTokens: 40960,
                contextTokens: 0,
                maxSessions: 0,
                sessionContextTokens: 40960,
                replyTokens: 0,
                messageTokens: 0,
                source: 'detected',
                heldSlots: 0,
                queued: 0,
                usedContextTokens: 0,
                warning: ''
            }
        ]
        getPeerCapacity.mockResolvedValueOnce(report).mockResolvedValueOnce([])
        await Promise.all([
            useCapacityPlannerStore.getState().refreshPeer('node-b'),
            useCapacityPlannerStore.getState().refreshPeer('node-b'),
            useCapacityPlannerStore.getState().refreshPeer('node-c')
        ])
        // Two reads for node-b at once are one read.
        expect(getPeerCapacity.mock.calls.map(c => c[0])).toEqual(['node-b', 'node-c'])
        expect(useCapacityPlannerStore.getState().peerCapacity).toEqual({
            'node-b': report,
            'node-c': []
        })
        expect(useCapacityPlannerStore.getState().capacity).toBeNull()

        getPeerCapacity.mockRejectedValueOnce(new Error('node is not a reachable cluster peer'))
        await useCapacityPlannerStore.getState().refreshPeer('node-b')
        expect(useCapacityPlannerStore.getState().peerCapacity).toEqual({ 'node-c': [] })
    })

    it('keeps a peer’s planner when the peer answers it, so its models can be planned from here', async () => {
        getPeerCapacity.mockResolvedValue([])
        getPlanner.mockResolvedValueOnce(state)
        await useCapacityPlannerStore.getState().refreshPeer('node-b')
        expect(getPlanner).toHaveBeenCalledWith('node-b')
        expect(useCapacityPlannerStore.getState().peerPlanner).toEqual({ 'node-b': state })
        // This node's own planner is not touched by a peer's.
        expect(useCapacityPlannerStore.getState().planner).toBeNull()

        // An older or unreachable peer can no longer be planned from here.
        getPlanner.mockRejectedValueOnce(
            new Error('node node-b runs a PAIR Plus without remote Context & sessions')
        )
        await useCapacityPlannerStore.getState().refreshPeer('node-b')
        expect(useCapacityPlannerStore.getState().peerPlanner).toEqual({})
        expect(useCapacityPlannerStore.getState().peerCapacity).toEqual({ 'node-b': [] })
        // The reason is kept for the dialog, and cleared once the peer answers.
        expect(useCapacityPlannerStore.getState().peerPlannerError['node-b']).toContain(
            'without remote Context & sessions'
        )
        getPlanner.mockResolvedValueOnce(state)
        await useCapacityPlannerStore.getState().refreshPeer('node-b')
        expect(useCapacityPlannerStore.getState().peerPlannerError).toEqual({})
        expect(useCapacityPlannerStore.getState().peerPlanner).toEqual({ 'node-b': state })
        useCapacityPlannerStore.setState({ peerPlanner: {} })

        useCapacityPlannerStore.getState().setPeerPlanner('node-c', state)
        expect(useCapacityPlannerStore.getState().peerPlanner).toEqual({ 'node-c': state })
    })

    it('removing a peer’s plan lifts the limit on that peer', async () => {
        const after = { ...state, plans: [] }
        setPlanner.mockResolvedValueOnce(after)
        setCapacity.mockResolvedValueOnce(null)
        await expect(removePlanAndLimit('ollama', 'qwen3:0.6b', 'node-b')).resolves.toBe(after)
        expect(setPlanner).toHaveBeenCalledWith({
            nodeId: 'node-b',
            remove: { engine: 'ollama', model: 'qwen3:0.6b' }
        })
        expect(setCapacity).toHaveBeenCalledWith({
            nodeId: 'node-b',
            engine: 'ollama',
            model: 'qwen3:0.6b',
            clear: true
        })
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
