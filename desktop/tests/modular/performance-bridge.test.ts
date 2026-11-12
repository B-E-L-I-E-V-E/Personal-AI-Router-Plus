// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'

const callProxy = vi.fn()

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => ({ callProxy })
}))

import {
    clearPerformance,
    getPerformanceLive,
    getPerformanceReport,
    runBenchmark
} from '@/electron/service-bridge/performance-bridge'

beforeEach(() => {
    callProxy.mockReset()
})

describe('performance bridge', () => {
    it('parses model summaries, keeping absent figures as null', async () => {
        callProxy.mockResolvedValueOnce({
            scope: 'routed-by-this-node',
            window: 100,
            models: [
                {
                    engine: 'ollama',
                    model: 'qwen3:0.6b',
                    nodeId: 'node-a',
                    local: true,
                    requests: 3,
                    promptTokens: 60,
                    completionTokens: 90,
                    thinkingRequests: 1,
                    thinkingTokens: 40,
                    lastAt: 1700000000000,
                    writeTokensPerSecond: { count: 3, avg: 80, p50: 79, p95: 90, min: 70, max: 91 },
                    readTokensPerSecond: { count: 0 },
                    sources: { tokens: 'engine', write: 'engine', read: 'bogus' },
                    trend: [{ at: 1, writeTokensPerSecond: 80 }]
                },
                { engine: 'ollama' }
            ],
            recent: [],
            benchmarks: [],
            benchmarkPrompts: [{ name: 'short', label: 'Short prompt', maxTokens: 64 }]
        })
        const report = await getPerformanceReport()
        expect(callProxy.mock.calls[0]?.[1]).toBe('performance/report')
        expect(report.models).toHaveLength(1)
        const m = report.models[0]
        expect(m?.writeTokensPerSecond?.p95).toBe(90)
        expect(m?.readTokensPerSecond).toBeNull()
        expect(m?.timeToFirstTokenMs).toBeNull()
        expect(m?.sources.read).toBe('')
        expect(m?.trend[0]?.readTokensPerSecond).toBeNull()
        expect(report.benchmarkPrompts[0]?.label).toBe('Short prompt')
    })

    it('parses benchmark runs and their step results', async () => {
        callProxy.mockResolvedValueOnce({
            benchmarks: [
                {
                    id: 'bench-1',
                    engine: 'ollama',
                    model: 'm',
                    state: 'done',
                    startedAt: 5,
                    steps: [
                        {
                            name: 'thinking',
                            label: 'Thinking',
                            thinking: 'on',
                            state: 'done',
                            result: {
                                model: 'm',
                                thinking: true,
                                thinkingTokens: 120,
                                thinkingMs: 900,
                                totalMs: 1500
                            }
                        },
                        { name: 'short', state: 'skipped', detail: 'busy' }
                    ]
                }
            ]
        })
        const report = await getPerformanceReport()
        const run = report.benchmarks[0]
        expect(run?.finishedAt).toBeNull()
        expect(run?.steps[0]?.result?.thinkingMs).toBe(900)
        expect(run?.steps[1]?.state).toBe('skipped')
        expect(run?.steps[1]?.result).toBeNull()
    })

    it('starts a benchmark on the relay of the engine that serves the model', async () => {
        callProxy.mockResolvedValueOnce({
            run: { id: 'bench-2', engine: 'lmstudio', model: 'x', state: 'running', steps: [] }
        })
        const run = await runBenchmark({ engine: 'lmstudio', model: 'x' })
        expect(callProxy).toHaveBeenCalledWith('lm-studio', 'performance/benchmark', {
            engine: 'lmstudio',
            model: 'x'
        })
        expect(run.state).toBe('running')
    })

    it('refuses an unknown engine without calling the proxy', async () => {
        await expect(runBenchmark({ engine: 'other', model: 'x' })).rejects.toThrow()
        expect(callProxy).not.toHaveBeenCalled()
    })

    it('clears through the first running engine', async () => {
        callProxy.mockResolvedValueOnce({ cleared: true })
        await expect(clearPerformance()).resolves.toBeNull()
        expect(callProxy.mock.calls[0]?.[1]).toBe('performance/clear')
    })

    it('parses live requests with their figures so far', async () => {
        callProxy.mockResolvedValueOnce({
            live: [
                {
                    requestId: 'r1',
                    sessionId: 'conv-1',
                    engine: 'ollama',
                    model: 'qwen3:0.6b',
                    nodeId: 'node-a',
                    local: true,
                    origin: 'peer',
                    phase: 'thinking',
                    elapsedMs: 900,
                    promptTokensEstimate: 30,
                    outputTokens: 40,
                    thinkingTokens: 40,
                    timeToFirstTokenMs: 120,
                    thinkingMs: 700
                },
                { requestId: 'no-model' }
            ],
            recent: [{ model: 'm', sessionId: '', origin: 'routed' }]
        })
        const live = await getPerformanceLive()
        expect(callProxy.mock.calls[0]?.[1]).toBe('performance/live')
        expect(live.live).toHaveLength(1)
        expect(live.live[0]?.phase).toBe('thinking')
        expect(live.live[0]?.origin).toBe('peer')
        expect(live.live[0]?.agentId).toBeNull()
        expect(live.live[0]?.writeTokensPerSecond).toBeNull()
        expect(live.recent[0]?.sessionId).toBeNull()
    })
})
