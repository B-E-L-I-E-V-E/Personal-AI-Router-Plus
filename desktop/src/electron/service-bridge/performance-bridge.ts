// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    LivePhase,
    LiveRequest,
    PerfOrigin,
    PerformanceLive,
    BenchmarkPrompt,
    BenchmarkRun,
    BenchmarkState,
    BenchmarkStep,
    PerfModelSummary,
    PerformanceReport,
    PerfRecord,
    PerfSource,
    PerfSources,
    PerfStat,
    PerfTrendPoint
} from '@/shared/types/performance'
import type { JsonObject, JsonValue } from './json-rpc-subprocess'
import { getModularSupervisor } from './modular-supervisor'
import { proxyEngineFromManagerId } from './proxy-engines'
import { callSessionProxy } from './sessions-bridge'

/**
 * Performance figures and benchmarks live in nvpair-proxy, process-wide like
 * sessions, so any running engine's relay reaches them. A benchmark is started
 * on the relay of the engine that serves the model.
 */

function objectValue(value: JsonValue | undefined): JsonObject | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

function arrayValue(value: JsonValue | undefined): JsonValue[] {
    return Array.isArray(value) ? value : []
}

function stringValue(value: JsonValue | undefined): string {
    return typeof value === 'string' ? value : ''
}

function numberValue(value: JsonValue | undefined): number {
    return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function optionalNumber(value: JsonValue | undefined): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null
}

const SOURCES: readonly PerfSource[] = ['engine', 'proxy', 'estimated']

function sourceValue(value: JsonValue | undefined): PerfSource {
    return SOURCES.find(s => s === value) ?? ''
}

function parseSources(value: JsonValue | undefined): PerfSources {
    const obj = objectValue(value)
    return {
        tokens: sourceValue(obj?.tokens),
        read: sourceValue(obj?.read),
        write: sourceValue(obj?.write),
        ttft: sourceValue(obj?.ttft),
        thinking: sourceValue(obj?.thinking)
    }
}

function parseStat(value: JsonValue | undefined): PerfStat | null {
    const obj = objectValue(value)
    if (!obj || numberValue(obj.count) <= 0) return null
    return {
        count: numberValue(obj.count),
        avg: numberValue(obj.avg),
        p50: numberValue(obj.p50),
        p95: numberValue(obj.p95),
        min: numberValue(obj.min),
        max: numberValue(obj.max)
    }
}

function parseRecord(value: JsonValue | undefined): PerfRecord | null {
    const obj = objectValue(value)
    const model = stringValue(obj?.model)
    if (!obj || !model) return null
    return {
        requestId: stringValue(obj.requestId),
        at: numberValue(obj.at),
        engine: stringValue(obj.engine),
        model,
        nodeId: stringValue(obj.nodeId),
        local: obj.local === true,
        path: stringValue(obj.path),
        streamed: obj.streamed === true,
        sessionId: stringValue(obj.sessionId) || null,
        agentId: stringValue(obj.agentId) || null,
        origin: originValue(obj.origin),
        promptTokens: numberValue(obj.promptTokens),
        completionTokens: numberValue(obj.completionTokens),
        thinking: obj.thinking === true,
        thinkingTokens: optionalNumber(obj.thinkingTokens),
        readTokensPerSecond: optionalNumber(obj.readTokensPerSecond),
        writeTokensPerSecond: optionalNumber(obj.writeTokensPerSecond),
        timeToFirstTokenMs: optionalNumber(obj.timeToFirstTokenMs),
        promptMs: optionalNumber(obj.promptMs),
        generationMs: optionalNumber(obj.generationMs),
        loadMs: optionalNumber(obj.loadMs),
        thinkingMs: optionalNumber(obj.thinkingMs),
        totalMs: numberValue(obj.totalMs),
        sources: parseSources(obj.sources)
    }
}

function parseTrend(value: JsonValue | undefined): PerfTrendPoint[] {
    const out: PerfTrendPoint[] = []
    for (const entry of arrayValue(value)) {
        const obj = objectValue(entry)
        if (!obj) continue
        out.push({
            at: numberValue(obj.at),
            readTokensPerSecond: optionalNumber(obj.readTokensPerSecond),
            writeTokensPerSecond: optionalNumber(obj.writeTokensPerSecond),
            timeToFirstTokenMs: optionalNumber(obj.timeToFirstTokenMs),
            thinkingMs: optionalNumber(obj.thinkingMs)
        })
    }
    return out
}

function parseModel(value: JsonValue | undefined): PerfModelSummary | null {
    const obj = objectValue(value)
    const model = stringValue(obj?.model)
    if (!obj || !model) return null
    return {
        engine: stringValue(obj.engine),
        model,
        nodeId: stringValue(obj.nodeId),
        local: obj.local === true,
        requests: numberValue(obj.requests),
        promptTokens: numberValue(obj.promptTokens),
        completionTokens: numberValue(obj.completionTokens),
        thinkingRequests: numberValue(obj.thinkingRequests),
        thinkingTokens: numberValue(obj.thinkingTokens),
        lastAt: numberValue(obj.lastAt),
        readTokensPerSecond: parseStat(obj.readTokensPerSecond),
        writeTokensPerSecond: parseStat(obj.writeTokensPerSecond),
        timeToFirstTokenMs: parseStat(obj.timeToFirstTokenMs),
        thinkingMs: parseStat(obj.thinkingMs),
        totalMs: parseStat(obj.totalMs),
        loadMs: parseStat(obj.loadMs),
        sources: parseSources(obj.sources),
        trend: parseTrend(obj.trend)
    }
}

function originValue(value: JsonValue | undefined): PerfOrigin {
    return value === 'peer' ? 'peer' : 'routed'
}

const PHASES: readonly LivePhase[] = ['waiting', 'reading', 'thinking', 'answering']

function parseLive(value: JsonValue | undefined): LiveRequest | null {
    const obj = objectValue(value)
    const model = stringValue(obj?.model)
    if (!obj || !model) return null
    return {
        requestId: stringValue(obj.requestId),
        sessionId: stringValue(obj.sessionId) || null,
        agentId: stringValue(obj.agentId) || null,
        engine: stringValue(obj.engine),
        model,
        nodeId: stringValue(obj.nodeId),
        local: obj.local === true,
        path: stringValue(obj.path),
        origin: originValue(obj.origin),
        benchmark: obj.benchmark === true,
        phase: PHASES.find(p => p === obj.phase) ?? 'waiting',
        receivedAt: numberValue(obj.receivedAt),
        elapsedMs: numberValue(obj.elapsedMs),
        promptTokensEstimate: numberValue(obj.promptTokensEstimate),
        outputTokens: numberValue(obj.outputTokens),
        thinkingTokens: numberValue(obj.thinkingTokens),
        timeToFirstTokenMs: optionalNumber(obj.timeToFirstTokenMs),
        thinkingMs: optionalNumber(obj.thinkingMs),
        writeTokensPerSecond: optionalNumber(obj.writeTokensPerSecond)
    }
}

export async function getPerformanceLive(): Promise<PerformanceLive> {
    const obj = objectValue(await callSessionProxy('performance/live', {}))
    return {
        live: arrayValue(obj?.live).flatMap(r => parseLive(r) ?? []),
        recent: arrayValue(obj?.recent).flatMap(r => parseRecord(r) ?? [])
    }
}

const STATES: readonly BenchmarkState[] = ['pending', 'running', 'done', 'failed', 'skipped']

function stateValue(value: JsonValue | undefined): BenchmarkState {
    return STATES.find(s => s === value) ?? 'pending'
}

function parseStep(value: JsonValue | undefined): BenchmarkStep | null {
    const obj = objectValue(value)
    const name = stringValue(obj?.name)
    if (!obj || !name) return null
    return {
        name,
        label: stringValue(obj.label) || name,
        thinking: stringValue(obj.thinking),
        state: stateValue(obj.state),
        detail: stringValue(obj.detail),
        result: parseRecord(obj.result)
    }
}

function parseRun(value: JsonValue | undefined): BenchmarkRun | null {
    const obj = objectValue(value)
    const id = stringValue(obj?.id)
    if (!obj || !id) return null
    return {
        id,
        engine: stringValue(obj.engine),
        model: stringValue(obj.model),
        nodeId: stringValue(obj.nodeId),
        state: stateValue(obj.state),
        startedAt: numberValue(obj.startedAt),
        finishedAt: optionalNumber(obj.finishedAt),
        error: stringValue(obj.error),
        steps: arrayValue(obj.steps).flatMap(s => parseStep(s) ?? [])
    }
}

function parsePrompt(value: JsonValue | undefined): BenchmarkPrompt | null {
    const obj = objectValue(value)
    const name = stringValue(obj?.name)
    if (!obj || !name) return null
    return {
        name,
        label: stringValue(obj.label) || name,
        description: stringValue(obj.description),
        thinking: stringValue(obj.thinking),
        maxTokens: numberValue(obj.maxTokens)
    }
}

function parsePerformanceReport(value: JsonValue | undefined): PerformanceReport {
    const obj = objectValue(value)
    return {
        scope: stringValue(obj?.scope),
        window: numberValue(obj?.window),
        models: arrayValue(obj?.models).flatMap(m => parseModel(m) ?? []),
        recent: arrayValue(obj?.recent).flatMap(r => parseRecord(r) ?? []),
        benchmarks: arrayValue(obj?.benchmarks).flatMap(r => parseRun(r) ?? []),
        benchmarkPrompts: arrayValue(obj?.benchmarkPrompts).flatMap(p => parsePrompt(p) ?? [])
    }
}

export async function getPerformanceReport(): Promise<PerformanceReport> {
    return parsePerformanceReport(await callSessionProxy('performance/report', {}))
}

export async function clearPerformance(): Promise<null> {
    await callSessionProxy('performance/clear', {})
    return null
}

/**
 * Start the benchmark for a model on the engine (proxy name) that serves it.
 * With a node ID every step runs on that node, this one or a cluster peer.
 */
export async function runBenchmark(request?: {
    engine: string
    model: string
    nodeId?: string
}): Promise<BenchmarkRun> {
    const engine = proxyEngineFromManagerId(request?.engine ?? '')
    if (!engine || !request?.model) throw new Error('Choose a model to benchmark')
    const params: JsonObject = { engine: request.engine, model: request.model }
    if (request.nodeId) params.nodeId = request.nodeId
    const result = await getModularSupervisor().callProxy(engine, 'performance/benchmark', params)
    const run = parseRun(objectValue(result)?.run)
    if (!run) throw new Error('The router did not start the benchmark')
    return run
}
