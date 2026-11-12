// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Whether a node takes requests (nvpair-proxy `node/state`). Paused finishes
 * what the node is serving and takes nothing new; stopped also cancels what
 * it is serving.
 */
export type NodeServingState = 'running' | 'paused' | 'stopped'

export interface NodeStateInfo {
    /** Another node's ID; null for this node. */
    nodeId: string | null
    state: NodeServingState
    /** When the node entered this state (RFC 3339), when known. */
    since: string | null
    /** Requests the node's engine is serving now. */
    serving: number
    /** Requests a stop cancelled. */
    cancelled: number
}

export interface NodeStatesSnapshot {
    self: NodeStateInfo
    /** Other nodes known to be paused or stopped; a node not listed is running. */
    peers: NodeStateInfo[]
}

export interface NodeStateChange {
    /** The node to change; null for this node. */
    nodeId: string | null
    state: NodeServingState
}
