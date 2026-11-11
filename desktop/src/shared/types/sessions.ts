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

/** How much the capacity planner knows about a model, and from where. */
export interface ModelSpec {
    engine: string
    model: string
    /** 'metadata', 'measured', 'manual', or '' when the context cost is unknown. */
    source: string
    architecture: string
    format: string
    weightsBytes: number
    maxContextTokens: number
    kvBytesPerToken: number
    notes: string[]
}

export type KvCacheType = 'f16' | 'q8_0' | 'q4_0'

/** A capacity/plan question. Set windowTokens or sessions, not both. */
export interface CapacityPlanRequest {
    engine: string
    model: string
    windowTokens?: number
    sessions?: number
    cacheType?: KvCacheType
    reservePercent?: number
    otherModelsBytes?: number
    weightsBytes?: number
    kvBytesPerToken?: number
}

export interface CapacityPlan {
    spec: ModelSpec
    totalBytes: number
    reserveBytes: number
    weightsBytes: number
    otherModelsBytes: number
    contextBytes: number
    bytesPerToken: number
    bytesPerSession: number
    cacheType: KvCacheType
    windowTokens: number
    sessions: number
    poolTokens: number
    usedBytes: number
    fits: boolean
    warnings: string[]
    /** The planner's refusal, when it could not size the model. */
    error: string
}

export interface SavedCapacityPlan {
    engine: string
    model: string
    windowTokens: number
    sessions: number
    cacheType: KvCacheType
    reservePercent: number
    applied: boolean
    appliedAt: number
    savedAt: number
}

export interface PlannerState {
    /** Apply a pending plan by itself once the node is idle. */
    autoReload: boolean
    plans: SavedCapacityPlan[]
    activity: { sessions: number; activeRequests: number; queued: number; idle: boolean }
}

export interface PlannerChange {
    autoReload?: boolean
    save?: Omit<SavedCapacityPlan, 'applied' | 'appliedAt' | 'savedAt'>
    remove?: { engine: string; model: string }
}

export interface ApplyPlanResult {
    applied: boolean
    /** What PAIR changed in the engine's launch settings, or why it could not. */
    message: string
}
