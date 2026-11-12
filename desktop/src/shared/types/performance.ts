// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-model performance figures and benchmark runs, measured and held by
 * nvpair-proxy for the requests this node routes. The desktop renders what
 * the proxy reports and keeps no copy of its own. Only numbers cross this
 * boundary: no prompt or reply text is measured into, or carried by, these
 * types.
 */

/** Where a figure came from. */
export type PerfSource = 'engine' | 'proxy' | 'estimated' | ''

export interface PerfSources {
    tokens: PerfSource
    read: PerfSource
    write: PerfSource
    ttft: PerfSource
    thinking: PerfSource
}

/** One figure summarized over a model's recent requests. */
export interface PerfStat {
    count: number
    avg: number
    p50: number
    p95: number
    min: number
    max: number
}

/** One request's figures; null where nothing could measure it. */
export interface PerfRecord {
    requestId: string
    at: number
    engine: string
    model: string
    nodeId: string
    local: boolean
    path: string
    streamed: boolean
    promptTokens: number
    completionTokens: number
    thinking: boolean
    thinkingTokens: number | null
    readTokensPerSecond: number | null
    writeTokensPerSecond: number | null
    timeToFirstTokenMs: number | null
    promptMs: number | null
    generationMs: number | null
    loadMs: number | null
    thinkingMs: number | null
    totalMs: number
    sources: PerfSources
}

export interface PerfTrendPoint {
    at: number
    readTokensPerSecond: number | null
    writeTokensPerSecond: number | null
    timeToFirstTokenMs: number | null
    thinkingMs: number | null
}

/** A model on one node, over its most recent requests. */
export interface PerfModelSummary {
    engine: string
    model: string
    nodeId: string
    local: boolean
    requests: number
    promptTokens: number
    completionTokens: number
    thinkingRequests: number
    thinkingTokens: number
    lastAt: number
    readTokensPerSecond: PerfStat | null
    writeTokensPerSecond: PerfStat | null
    timeToFirstTokenMs: PerfStat | null
    thinkingMs: PerfStat | null
    totalMs: PerfStat | null
    loadMs: PerfStat | null
    sources: PerfSources
    trend: PerfTrendPoint[]
}

export type BenchmarkState = 'pending' | 'running' | 'done' | 'failed' | 'skipped'

export interface BenchmarkStep {
    name: string
    label: string
    thinking: string
    state: BenchmarkState
    detail: string
    result: PerfRecord | null
}

export interface BenchmarkRun {
    id: string
    engine: string
    model: string
    state: BenchmarkState
    startedAt: number
    finishedAt: number | null
    error: string
    steps: BenchmarkStep[]
}

/** One step of the fixed benchmark, as the proxy documents it. */
export interface BenchmarkPrompt {
    name: string
    label: string
    description: string
    thinking: string
    maxTokens: number
}

export interface PerformanceReport {
    /** Which requests the figures cover; the proxy reports this node's routing. */
    scope: string
    /** How many recent requests per model and node the figures cover. */
    window: number
    models: PerfModelSummary[]
    recent: PerfRecord[]
    benchmarks: BenchmarkRun[]
    benchmarkPrompts: BenchmarkPrompt[]
}
