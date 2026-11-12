// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { create } from 'zustand'
import type { NodeServingState, NodeStatesSnapshot } from '@/shared/types/node-state'
import getErrorString from '@/shared/utils/get-error-string'

/**
 * Which nodes are paused or stopped, as this node's router knows it. The proxy
 * owns the state; this caches its last answer for the node cards. A node with
 * no entry is running.
 */
interface NodeStateStore {
    /** This node's state. */
    self: NodeServingState
    /** Other nodes' states, by node ID; only paused or stopped nodes appear. */
    peers: Record<string, NodeServingState>
    /** Node IDs ('self' for this node) with a change in flight. */
    pending: Record<string, boolean>
    /** The last failed change per node key, until the next attempt. */
    errors: Record<string, string>
    apply: (snapshot: NodeStatesSnapshot) => void
    refresh: () => Promise<void>
    /** Changes a node's state; isSelf picks this node over nodeId. */
    change: (nodeId: string, isSelf: boolean, state: NodeServingState) => Promise<void>
}

export const useNodeStateStore = create<NodeStateStore>((set, get) => ({
    self: 'running',
    peers: {},
    pending: {},
    errors: {},
    apply: snapshot => {
        const peers: Record<string, NodeServingState> = {}
        for (const p of snapshot.peers) {
            if (p.nodeId && p.state !== 'running') peers[p.nodeId] = p.state
        }
        set({ self: snapshot.self.state, peers })
    },
    refresh: async () => {
        try {
            get().apply(await window.pairApi.nodeState.getAll())
        } catch {
            // The router is not reachable yet; cards keep the last answer.
        }
    },
    change: async (nodeId, isSelf, state) => {
        const key = isSelf ? 'self' : nodeId
        set(s => {
            const errors = { ...s.errors }
            delete errors[key]
            return { pending: { ...s.pending, [key]: true }, errors }
        })
        try {
            const result = await window.pairApi.nodeState.set(isSelf ? null : nodeId, state)
            if (isSelf) {
                set({ self: result.state })
            } else {
                set(s => {
                    const peers = { ...s.peers }
                    if (result.state === 'running') delete peers[nodeId]
                    else peers[nodeId] = result.state
                    return { peers }
                })
            }
        } catch (err) {
            set(s => ({
                errors: { ...s.errors, [key]: getErrorString(err) ?? 'The node did not answer' }
            }))
        } finally {
            set(s => {
                const pending = { ...s.pending }
                delete pending[key]
                return { pending }
            })
        }
    }
}))

/** One node's state from the store. */
export function nodeServingState(
    store: Pick<NodeStateStore, 'self' | 'peers'>,
    nodeId: string,
    isSelf: boolean
): NodeServingState {
    return isSelf ? store.self : (store.peers[nodeId] ?? 'running')
}
