// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineType } from '@/shared/types/engines'
import type { BenchmarkRun, PerfSource } from '@/shared/types/performance'

/** Tokens per second, to a precision that reads at a glance. */
export function formatRate(value: number | null | undefined): string {
    if (value === null || value === undefined || !Number.isFinite(value)) return '—'
    if (value >= 1000) return `${(value / 1000).toFixed(1)}k`
    if (value >= 100) return value.toFixed(0)
    return value.toFixed(1)
}

/** A duration in milliseconds, switching to seconds past one second. */
export function formatMs(value: number | null | undefined): string {
    if (value === null || value === undefined || !Number.isFinite(value)) return '—'
    if (value >= 60_000) return `${(value / 60_000).toFixed(1)} min`
    if (value >= 1000) return `${(value / 1000).toFixed(value >= 10_000 ? 0 : 1)} s`
    return `${Math.round(value)} ms`
}

export function formatCount(value: number): string {
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
    if (value >= 10_000) return `${(value / 1000).toFixed(0)}k`
    return value.toLocaleString()
}

const SOURCE_LABELS: Record<PerfSource, string> = {
    engine: 'reported by the engine',
    proxy: 'measured by PAIR',
    estimated: 'estimated',
    '': ''
}

export function sourceLabel(source: PerfSource): string {
    return SOURCE_LABELS[source]
}

/** The proxy's engine name for a desktop engine type. */
const PROXY_ENGINE_NAMES: Partial<Record<EngineType, string>> = {
    ollama: 'ollama',
    'lm-studio': 'lmstudio',
    'llama-cpp': 'llamacpp'
}

export function proxyEngineName(engineType: EngineType): string | null {
    return PROXY_ENGINE_NAMES[engineType] ?? null
}

const ENGINE_LABELS: Record<string, string> = {
    ollama: 'Ollama',
    lmstudio: 'LM Studio',
    llamacpp: 'llama.cpp'
}

export function engineLabel(engine: string): string {
    return ENGINE_LABELS[engine] ?? engine
}

/** Averages over a run's completed steps, for comparing runs side by side. */
interface BenchmarkSummary {
    readTokensPerSecond: number | null
    writeTokensPerSecond: number | null
    timeToFirstTokenMs: number | null
    thinkingMs: number | null
    nodeId: string | null
}

function average(values: number[]): number | null {
    if (values.length === 0) return null
    return values.reduce((sum, v) => sum + v, 0) / values.length
}

export function summarizeBenchmark(run: BenchmarkRun): BenchmarkSummary {
    const results = run.steps.flatMap(step => (step.result ? [step.result] : []))
    const pick = (f: (r: (typeof results)[number]) => number | null) =>
        average(
            results.flatMap(r => {
                const v = f(r)
                return v === null ? [] : [v]
            })
        )
    // Reading speed is the long-input step's: short prompts mostly measure
    // overhead. Writing speed is the long-output step's for the same reason.
    const step = (name: string) => run.steps.find(s => s.name === name)?.result ?? null
    return {
        readTokensPerSecond:
            step('long-input')?.readTokensPerSecond ?? pick(r => r.readTokensPerSecond),
        writeTokensPerSecond:
            step('long-output')?.writeTokensPerSecond ?? pick(r => r.writeTokensPerSecond),
        timeToFirstTokenMs: step('short')?.timeToFirstTokenMs ?? pick(r => r.timeToFirstTokenMs),
        thinkingMs: step('thinking')?.thinkingMs ?? null,
        nodeId: results[0]?.nodeId ?? null
    }
}

/** Points for an SVG polyline drawing values across a width and height. */
export function sparklinePoints(values: number[], width: number, height: number): string {
    if (values.length === 0) return ''
    const max = Math.max(...values)
    const min = Math.min(...values)
    const span = max - min || 1
    const step = values.length > 1 ? width / (values.length - 1) : 0
    return values
        .map((v, i) => {
            const x = values.length > 1 ? i * step : width / 2
            const y = height - ((v - min) / span) * (height - 2) - 1
            return `${x.toFixed(1)},${y.toFixed(1)}`
        })
        .join(' ')
}
