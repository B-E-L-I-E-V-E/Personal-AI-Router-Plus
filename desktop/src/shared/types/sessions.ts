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
    /**
     * Tokens the next post may add before the conversation reaches its limit
     * on this node (the window less the reply budget); null when unknown, as
     * for a session another node serves.
     */
    postRoomTokens: number | null
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
    configured: { contextTokens: number; maxSessions: number; replyTokens: number } | null
    detectedWindowTokens: number
    contextTokens: number
    maxSessions: number
    sessionContextTokens: number
    /** Kept from each session's window for replies (0: no reply budget). */
    replyTokens: number
    /** What remains for a session's messages and history. */
    messageTokens: number
    source: string
    heldSlots: number
    queued: number
    usedContextTokens: number
    warning: string
}

/** A change to one model's capacity setting. */
export interface CapacityChange {
    /** The node to change; omitted means this one. */
    nodeId?: string
    engine: string
    model: string
    contextTokens?: number
    maxSessions?: number
    replyTokens?: number
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
    /**
     * The weights' size the planner counts: in memory once loaded when the
     * tensor table gives it, else the download size as an estimate.
     */
    weightsBytes: number
    /** 'tensors', 'download', or '' when the size is unknown. */
    weightsSource: string
    /** Size on disk, and where it was read ('ollama-tags', 'lmstudio-models', 'gguf-files'). */
    downloadBytes: number
    downloadSource: string
    /** Sum of the weight tensors: what the weights take in memory once loaded (0 when unknown). */
    tensorBytes: number
    /** A copy loaded now (Ollama /api/ps): its whole footprint, the GPU part, its context. */
    loadedBytes: number
    loadedVramBytes: number
    loadedContextTokens: number
    maxContextTokens: number
    /**
     * 'web' when maxContextTokens was read from the model's page because the
     * engine and the model file reported none; '' otherwise.
     */
    contextSource: string
    /** The page the web context came from, and the model that read it. */
    contextPage: string
    contextReader: string
    /** The model's page calls it a mixture of experts the engine did not describe. */
    webMoe: boolean
    kvBytesPerToken: number
    /** Routed experts in a mixture-of-experts model (0 for a dense model). */
    expertCount: number
    /** Experts each token runs (0 when unknown). */
    expertUsedCount: number
    /** Size of the expert weights, which llama.cpp can keep in system memory. */
    expertBytes: number
    /**
     * The expert weights of each layer (blk.i), in the order llama.cpp's
     * --n-cpu-moe takes them; 0 for a layer without experts.
     */
    expertLayerBytes: number[]
    /** Every weight in the tensor table, and those one token runs (0 when unknown). */
    totalParams: number
    activeParams: number
    /** The model's layer count; 0 when unknown. */
    layers: number
    /** Starting values the engine, its launch settings or the model report. */
    defaults: ModelDefaults
    notes: string[]
}

/**
 * Where a starting value came from: what the running engine uses for this
 * model now ('engine'), the engine's launch settings ('launch'), or the
 * model's own default parameter ('model').
 */
export type StartSource = 'engine' | 'launch' | 'model'

export interface StartValue {
    value: number
    source: StartSource
}

/** A value nothing reports is left out. */
export interface ModelDefaults {
    /** Each session's context window. */
    window?: StartValue
    /** Requests the engine runs at once (its slots). */
    sessions?: StartValue
    /** The model's default reply length. */
    reply?: StartValue
}

export type KvCacheType = 'f16' | 'q8_0' | 'q4_0'

/**
 * A capacity/plan question. Every session gets the whole window; the planner
 * counts how many such sessions fit.
 */
export interface CapacityPlanRequest {
    /** The node the model runs on; omitted means this one. */
    nodeId?: string
    engine: string
    model: string
    /** Each session's window; omitted means the model's maximum. */
    windowTokens?: number
    /** At most this many sessions; omitted means as many as fit. */
    maxSessions?: number
    /** Kept from each session's window for replies; omitted means the default. */
    replyTokens?: number
    /**
     * Keep the expert weights of a mixture-of-experts model's first N layers
     * in system memory (llama.cpp --n-cpu-moe N); omitted means none.
     */
    cpuExpertLayers?: number
    cacheType?: KvCacheType
    reservePercent?: number
    otherModelsBytes?: number
    weightsBytes?: number
    kvBytesPerToken?: number
    /** Also read the engine's launch settings into spec.defaults. */
    withLaunchDefaults?: boolean
}

/** One GPU's part of a plan on a node with several GPUs. */
export interface CapacityPlanGpu {
    name: string
    totalBytes: number
    reserveBytes: number
    /** The model's layers on this GPU; 0 when the layer count is unknown. */
    layers: number
    /** This GPU's fraction of the model and of each session's context. */
    share: number
    weightsBytes: number
    otherModelsBytes: number
    /** This GPU's part of one session's context. */
    bytesPerSession: number
    /** Sessions this GPU's part has room for. */
    sessions: number
}

export interface CapacityPlan {
    spec: ModelSpec
    totalBytes: number
    reserveBytes: number
    weightsBytes: number
    /** Where weightsBytes came from: the spec's weightsSource, or 'manual'. */
    weightsSource: string
    /** Where the per-token cost came from: the spec's source, or 'manual'. */
    kvSource: string
    /** The weights on the GPU: all less the experts kept in system memory. */
    gpuWeightsBytes: number
    otherModelsBytes: number
    contextBytes: number
    bytesPerToken: number
    bytesPerSession: number
    cacheType: KvCacheType
    windowTokens: number
    sessions: number
    /** Kept from each session's window for replies. */
    replyTokens: number
    /** What a session's posts and history may use: window less reply. */
    postTokens: number
    poolTokens: number
    usedBytes: number
    fits: boolean
    /** Whether this model, engine and node can keep experts in system memory. */
    canOffloadExperts: boolean
    /** Layers whose experts are in system memory (llama.cpp --n-cpu-moe). */
    cpuExpertLayers: number
    expertBytes: number
    /** Why expert offload is unavailable, for a mixture-of-experts model. */
    offloadNote: string
    /** The GPU shares system memory (Apple silicon). */
    unified: boolean
    ramTotalBytes: number
    ramReserveBytes: number
    ramUsedBytes: number
    ramFits: boolean
    /** Each GPU's part: the model's layers split by memory after the reserve. */
    gpus: CapacityPlanGpu[]
    /** llama.cpp --tensor-split for the split; empty with one GPU. */
    tensorSplit: string
    /**
     * Ollama loads a model on one GPU when it fits there: this many sessions
     * fit on GPU oneGpuIndex alone (0 when the model needs more than one).
     */
    oneGpuIndex: number
    oneGpuSessions: number
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
    /** Layers whose experts are kept in system memory (llama.cpp --n-cpu-moe). */
    cpuExpertLayers: number
    /** Kept from each session's window for replies; 0 means the default. */
    replyTokens: number
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
    /** The node whose planner to change; omitted means this one. */
    nodeId?: string
    autoReload?: boolean
    save?: Omit<SavedCapacityPlan, 'applied' | 'appliedAt' | 'savedAt'>
    remove?: { engine: string; model: string }
}

export interface ApplyPlanResult {
    applied: boolean
    /** What PAIR changed in the engine's launch settings, or why it could not. */
    message: string
}
