// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'

const callProxy = vi.fn()

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => ({ callProxy })
}))

import {
    getNodeStates,
    parseNodeStates,
    setNodeState
} from '@/electron/service-bridge/node-state-bridge'
import { nodeServingState, useNodeStateStore } from '@/ui/stores/node-state.store'
import { servingActions } from '@/ui/utils/node-serving-actions'

const getAll = vi.fn()
const set = vi.fn()

beforeEach(() => {
    callProxy.mockReset()
    getAll.mockReset()
    set.mockReset()
    useNodeStateStore.setState({ self: 'running', peers: {}, pending: {}, errors: {} })
    vi.stubGlobal('window', { pairApi: { nodeState: { getAll, set } } })
})

describe('node state bridge', () => {
    it('sets this node without a node ID and another node with one', async () => {
        callProxy.mockResolvedValue({ state: 'paused', serving: 1 })
        await setNodeState({ nodeId: null, state: 'paused' })
        expect(callProxy.mock.calls[0]?.slice(1)).toEqual(['node/state', { state: 'paused' }])

        callProxy.mockResolvedValue({ nodeId: 'jarvis', state: 'stopped', cancelled: 2 })
        const result = await setNodeState({ nodeId: 'jarvis', state: 'stopped' })
        expect(callProxy.mock.calls[1]?.slice(1)).toEqual([
            'node/state',
            { state: 'stopped', nodeId: 'jarvis' }
        ])
        expect(result).toEqual({
            nodeId: 'jarvis',
            state: 'stopped',
            since: null,
            serving: 0,
            cancelled: 2
        })
    })

    it('reads every known state and treats anything unknown as running', async () => {
        callProxy.mockResolvedValue({
            self: { state: 'paused', since: '2026-11-11T10:00:00Z', serving: 1 },
            peers: [{ nodeId: 'ultron', state: 'stopped' }, { state: 'paused' }]
        })
        const snapshot = await getNodeStates()
        expect(callProxy.mock.calls[0]?.[1]).toBe('node/states')
        expect(snapshot.self.state).toBe('paused')
        expect(snapshot.self.since).toBe('2026-11-11T10:00:00Z')
        expect(snapshot.peers.map(p => [p.nodeId, p.state])).toEqual([['ultron', 'stopped']])
        expect(parseNodeStates({ self: { state: 'asleep' } }).self.state).toBe('running')
    })
})

describe('node state store', () => {
    it('keeps a peer only while it is paused or stopped', async () => {
        getAll.mockResolvedValueOnce({
            self: { nodeId: null, state: 'running', since: null, serving: 0, cancelled: 0 },
            peers: [{ nodeId: 'jarvis', state: 'paused', since: null, serving: 0, cancelled: 0 }]
        })
        await useNodeStateStore.getState().refresh()
        expect(nodeServingState(useNodeStateStore.getState(), 'jarvis', false)).toBe('paused')
        expect(nodeServingState(useNodeStateStore.getState(), 'ultron', false)).toBe('running')

        set.mockResolvedValueOnce({ nodeId: 'jarvis', state: 'running' })
        await useNodeStateStore.getState().change('jarvis', false, 'running')
        expect(set).toHaveBeenCalledWith('jarvis', 'running')
        expect(useNodeStateStore.getState().peers).toEqual({})
    })

    it('changes this node by sending no node ID', async () => {
        set.mockResolvedValueOnce({ nodeId: null, state: 'stopped' })
        await useNodeStateStore.getState().change('self-uuid', true, 'stopped')
        expect(set).toHaveBeenCalledWith(null, 'stopped')
        expect(useNodeStateStore.getState().self).toBe('stopped')
        expect(useNodeStateStore.getState().pending).toEqual({})
    })

    it('records why a change failed and clears it on the next attempt', async () => {
        set.mockRejectedValueOnce(new Error('node jarvis is not a reachable member'))
        await useNodeStateStore.getState().change('jarvis', false, 'paused')
        expect(useNodeStateStore.getState().errors.jarvis).toContain('not a reachable member')
        expect(useNodeStateStore.getState().pending).toEqual({})

        set.mockResolvedValueOnce({ nodeId: 'jarvis', state: 'paused' })
        await useNodeStateStore.getState().change('jarvis', false, 'paused')
        expect(useNodeStateStore.getState().errors).toEqual({})
    })

    it('keeps the last answer when the router is unreachable', async () => {
        useNodeStateStore.setState({ self: 'paused' })
        getAll.mockRejectedValueOnce(new Error('no engine proxy is running'))
        await useNodeStateStore.getState().refresh()
        expect(useNodeStateStore.getState().self).toBe('paused')
    })
})

describe('node card controls', () => {
    it('offer pause and stop while running, resume and stop while paused, resume while stopped', () => {
        expect(servingActions('running')).toEqual(['paused', 'stopped'])
        expect(servingActions('paused')).toEqual(['running', 'stopped'])
        expect(servingActions('stopped')).toEqual(['running'])
    })
})
