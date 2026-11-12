// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    NodeServingState,
    NodeStateChange,
    NodeStateInfo,
    NodeStatesSnapshot
} from '@/shared/types/node-state'
import type { JsonObject, JsonValue } from './json-rpc-subprocess'
import { callSessionProxy } from './sessions-bridge'

/**
 * Node state lives in nvpair-proxy, which also reaches the other nodes of the
 * cluster over its mTLS ingress, so any member can pause, stop or resume any
 * node. The desktop only relays the call and renders the answer.
 */

function objectValue(value: JsonValue | undefined): JsonObject | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

function servingState(value: JsonValue | undefined): NodeServingState {
    return value === 'paused' || value === 'stopped' ? value : 'running'
}

function count(value: JsonValue | undefined): number {
    return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function parseNodeState(value: JsonValue | undefined): NodeStateInfo {
    const obj = objectValue(value)
    const nodeId = obj?.nodeId
    const since = obj?.since
    return {
        nodeId: typeof nodeId === 'string' && nodeId ? nodeId : null,
        state: servingState(obj?.state),
        since: typeof since === 'string' && since ? since : null,
        serving: count(obj?.serving),
        cancelled: count(obj?.cancelled)
    }
}

export function parseNodeStates(value: JsonValue | undefined): NodeStatesSnapshot {
    const obj = objectValue(value)
    const peers = Array.isArray(obj?.peers) ? obj.peers : []
    return {
        self: parseNodeState(obj?.self),
        peers: peers.map(parseNodeState).filter(p => p.nodeId !== null)
    }
}

export async function getNodeStates(): Promise<NodeStatesSnapshot> {
    return parseNodeStates(await callSessionProxy('node/states', {}))
}

export async function setNodeState(change?: NodeStateChange): Promise<NodeStateInfo> {
    if (!change) throw new Error('no node state change given')
    const params: JsonObject = { state: change.state }
    if (change.nodeId) params.nodeId = change.nodeId
    return parseNodeState(await callSessionProxy('node/state', params))
}
