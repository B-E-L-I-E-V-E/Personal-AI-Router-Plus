// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineType } from '@/shared/types/engines'
import type { PerfModelSummary } from '@/shared/types/performance'
import type { WorkloadState } from '@/shared/types/workloads'
import { proxyEngineName } from './performance-format'

/** A request in the cluster's job list, as far as this page needs it. */
interface LiveJob {
    nodeId: string | null
    model: string
    engine: EngineType
    state: WorkloadState
    createdAt: number
    startedAt: number | null
}

/** A model a node's engine holds in memory. */
interface LoadedModel {
    nodeId: string
    engine: EngineType
    model: string
}

/** A model a node is working on right now. */
export interface LiveModel {
    engine: string
    model: string
    running: number
    queued: number
    /** When the oldest of its requests started (or was queued), Unix ms. */
    since: number
    /**
     * Time to first token of its most recently started request: from when
     * PAIR received it to when the engine's first token arrived (queueing
     * included). Null while none has produced a token yet.
     */
    firstTokenMs: number | null
    /** Start of the request firstTokenMs belongs to, to pick the latest. */
    lastStartedAt: number
}

export interface NodePerformance {
    nodeId: string
    name: string
    local: boolean
    /** Models with figures on this node, busiest first. */
    models: PerfModelSummary[]
    /** Models with requests running or queued on this node now. */
    live: LiveModel[]
    /** Models loaded in memory on this node now, by engine name. */
    loaded: { engine: string; model: string }[]
    requests: number
    promptTokens: number
    completionTokens: number
}

export function liveKey(engine: string, model: string): string {
    return `${engine}:${model}`
}

/**
 * Groups everything the page shows by node: the proxy's per-model figures,
 * the live job list (cluster-wide, pushed as it changes) and each engine's
 * loaded models. Every known node appears, so an idle node still shows that
 * it is idle; a node seen only in the figures (it has since left) is kept.
 */
export function groupByNode(input: {
    nodes: { id: string; name: string }[]
    selfId: string | null
    summaries: PerfModelSummary[]
    jobs: LiveJob[]
    loaded: LoadedModel[]
}): NodePerformance[] {
    const byId = new Map<string, NodePerformance>()
    const node = (nodeId: string, name?: string): NodePerformance => {
        let n = byId.get(nodeId)
        if (!n) {
            n = {
                nodeId,
                name: name ?? '',
                local: nodeId === input.selfId,
                models: [],
                live: [],
                loaded: [],
                requests: 0,
                promptTokens: 0,
                completionTokens: 0
            }
            byId.set(nodeId, n)
        }
        if (name && !n.name) n.name = name
        return n
    }
    for (const n of input.nodes) node(n.id, n.name)

    for (const s of input.summaries) {
        const n = node(s.nodeId)
        n.local = n.local || s.local
        n.models.push(s)
        n.requests += s.requests
        n.promptTokens += s.promptTokens
        n.completionTokens += s.completionTokens
    }

    for (const job of input.jobs) {
        if (!job.nodeId || (job.state !== 'running' && job.state !== 'queued')) continue
        const engine = proxyEngineName(job.engine) ?? job.engine
        const n = node(job.nodeId)
        let live = n.live.find(l => l.engine === engine && l.model === job.model)
        const since = job.startedAt ?? job.createdAt
        if (!live) {
            live = {
                engine,
                model: job.model,
                running: 0,
                queued: 0,
                since,
                firstTokenMs: null,
                lastStartedAt: 0
            }
            n.live.push(live)
        }
        if (job.state === 'running') live.running++
        else live.queued++
        // A job turns running when the engine's first token arrives.
        if (
            job.state === 'running' &&
            job.startedAt !== null &&
            job.startedAt >= live.lastStartedAt
        ) {
            live.lastStartedAt = job.startedAt
            live.firstTokenMs = Math.max(0, job.startedAt - job.createdAt)
        }
        live.since = Math.min(live.since, since)
    }

    for (const m of input.loaded) {
        const engine = proxyEngineName(m.engine)
        const n = byId.get(m.nodeId)
        if (!engine || !n) continue
        n.loaded.push({ engine, model: m.model })
    }

    const out = [...byId.values()]
    for (const n of out) {
        n.models.sort((a, b) => b.requests - a.requests || a.model.localeCompare(b.model))
        n.live.sort((a, b) => a.since - b.since)
        n.loaded.sort((a, b) => a.model.localeCompare(b.model))
    }
    // This node first, then busy nodes, then by name.
    return out.sort(
        (a, b) =>
            Number(b.local) - Number(a.local) ||
            Number(b.live.length > 0) - Number(a.live.length > 0) ||
            (a.name || a.nodeId).localeCompare(b.name || b.nodeId)
    )
}
