// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Session policy and context capacity, owned and persisted by nvpair-proxy.
 * The desktop reads and changes them through the broker and renders what the
 * proxy reports; it keeps no copy of its own.
 */

/** The proxy's session policy, in seconds. */
export interface SessionPolicy {
    idleTimeoutSeconds: number
    maxLifetimeSeconds: number
    maxSessions: number
    queueTimeoutSeconds: number
    maxQueued: number
}

/** One model's context use by a session, as its engine counted it. */
export interface SessionContextUse {
    engine: string
    model: string
    tokens: number
}

/** A live session this node routes. */
export interface SessionSummary {
    sessionId: string
    agentId: string | null
    expiresAt: string
    inFlight: number
    requests: number
    context: SessionContextUse[]
}

/** One model's context capacity on this node. */
export interface ModelCapacity {
    engine: string
    model: string
    configured: { contextTokens: number; maxSessions: number } | null
    detectedWindowTokens: number
    contextTokens: number
    maxSessions: number
    sessionContextTokens: number
    source: string
    heldSlots: number
    queued: number
    usedContextTokens: number
    warning: string
}

/** A change to one model's capacity setting. */
export interface CapacityChange {
    engine: string
    model: string
    contextTokens?: number
    maxSessions?: number
    clear?: boolean
}
