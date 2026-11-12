// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { BenchmarkRun, BenchmarkStep, PerfRecord } from '@/shared/types/performance'
import {
    formatMs,
    formatRate,
    proxyEngineName,
    sparklinePoints,
    summarizeBenchmark
} from '@/ui/components/Performance/performance-format'

function record(fields: Partial<PerfRecord>): PerfRecord {
    return {
        requestId: 'r',
        at: 0,
        engine: 'ollama',
        model: 'm',
        nodeId: 'node-a',
        local: true,
        path: '/api/chat',
        streamed: true,
        sessionId: null,
        agentId: null,
        origin: 'routed',
        promptTokens: 0,
        completionTokens: 0,
        thinking: false,
        thinkingTokens: null,
        readTokensPerSecond: null,
        writeTokensPerSecond: null,
        timeToFirstTokenMs: null,
        promptMs: null,
        generationMs: null,
        loadMs: null,
        thinkingMs: null,
        totalMs: 0,
        sources: { tokens: '', read: '', write: '', ttft: '', thinking: '' },
        ...fields
    }
}

describe('performance formatting', () => {
    it('formats rates and durations for reading at a glance', () => {
        expect(formatRate(null)).toBe('—')
        expect(formatRate(12.345)).toBe('12.3')
        expect(formatRate(345.6)).toBe('346')
        expect(formatRate(2500)).toBe('2.5k')
        expect(formatMs(420.4)).toBe('420 ms')
        expect(formatMs(1500)).toBe('1.5 s')
        expect(formatMs(90_000)).toBe('1.5 min')
    })

    it('maps desktop engine types to the proxy engine names', () => {
        expect(proxyEngineName('lm-studio')).toBe('lmstudio')
        expect(proxyEngineName('llama-cpp')).toBe('llamacpp')
        expect(proxyEngineName('ollama')).toBe('ollama')
    })

    it('summarizes a run from the step that measures each figure', () => {
        const step = (name: string, result: PerfRecord | null): BenchmarkStep => ({
            name,
            label: name,
            thinking: 'off',
            state: result ? 'done' : 'skipped',
            detail: '',
            result
        })
        const run: BenchmarkRun = {
            id: 'b',
            engine: 'ollama',
            model: 'm',
            nodeId: '',
            state: 'done',
            startedAt: 0,
            finishedAt: 1,
            error: '',
            steps: [
                step('short', record({ timeToFirstTokenMs: 90, readTokensPerSecond: 50 })),
                step('long-input', record({ readTokensPerSecond: 900 })),
                step('long-output', record({ writeTokensPerSecond: 120 })),
                step('thinking', null)
            ]
        }
        expect(summarizeBenchmark(run)).toEqual({
            readTokensPerSecond: 900,
            writeTokensPerSecond: 120,
            timeToFirstTokenMs: 90,
            thinkingMs: null,
            nodeId: 'node-a'
        })
    })

    it('draws a sparkline across the full width and height', () => {
        expect(sparklinePoints([1, 3], 100, 10)).toBe('0.0,9.0 100.0,1.0')
        expect(sparklinePoints([], 100, 10)).toBe('')
    })
})
