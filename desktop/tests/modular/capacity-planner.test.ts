// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'

const callProxy = vi.fn()
const callProcess = vi.fn()

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => ({ callProxy, callProcess })
}))
vi.mock('@/electron/service-bridge/modular-state', () => ({
    getModularBridgeState: () => ({
        getSelfId: () => 'node-1',
        getSelfGpus: () => [
            { name: 'NVIDIA RTX 5090', vramBytes: 32 * 1024 ** 3 },
            { name: 'NVIDIA RTX 5090', vramBytes: 32 * 1024 ** 3 }
        ],
        getNodeHardware: (nodeId: string) =>
            nodeId === 'node-2'
                ? {
                      gpus: [{ name: 'NVIDIA GeForce RTX 4090', vramBytes: 24 * 1024 ** 3 }],
                      ramBytes: 64 * 1024 ** 3
                  }
                : { gpus: [], ramBytes: 0 }
    })
}))

import {
    applyPlan,
    getPlanner,
    llamaCppLaunchText,
    ollamaLaunchText,
    parsePlanner,
    planCapacity,
    runAutoReloadCheckForTest,
    splitLaunchTokens
} from '@/electron/service-bridge/capacity-planner'

const plan = { windowTokens: 32768, sessions: 4, cacheType: 'f16' as const }

function settingsSnapshot(launchText: string, extra: Record<string, unknown> = {}) {
    return {
        nodeId: 'node-1',
        engine: 'llamacpp',
        revision: 7,
        appliedRevision: 7,
        sequence: 1,
        epoch: 'e',
        settings: { serverPort: 8081, proxyPort: 8080, launchText },
        effectiveServerPort: 8081,
        effectiveProxyPort: 8080,
        running: true,
        adopted: false,
        editable: true,
        reason: '',
        format: 'pair-arguments-v1',
        phase: 'idle',
        error: '',
        requestId: '',
        ...extra
    }
}

function plannerState(overrides: Record<string, unknown> = {}) {
    return {
        autoReload: true,
        plans: [
            {
                engine: 'llamacpp',
                model: 'm',
                windowTokens: 32768,
                sessions: 4,
                cacheType: 'f16',
                applied: false
            }
        ],
        activity: { sessions: 0, activeRequests: 0, queued: 0, idle: true },
        ...overrides
    }
}

beforeEach(() => {
    callProxy.mockReset()
    callProcess.mockReset()
})

describe('launch text', () => {
    it('keeps quoted tokens intact', () => {
        expect(splitLaunchTokens(`A="two words" --x 'y z'  --flag`)).toEqual([
            'A="two words"',
            '--x',
            "'y z'",
            '--flag'
        ])
    })

    it('replaces Ollama window and slot variables and keeps the rest', () => {
        const out = ollamaLaunchText(
            'OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=10m',
            plan
        )
        expect(out).toBe(
            'OLLAMA_HOST=127.0.0.1:11435 OLLAMA_KEEP_ALIVE=10m OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=4'
        )
    })

    it('turns on flash attention for a quantized Ollama cache unless set', () => {
        expect(ollamaLaunchText('', { ...plan, cacheType: 'q8_0' })).toBe(
            'OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=4 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_FLASH_ATTENTION=1'
        )
        expect(
            ollamaLaunchText('OLLAMA_FLASH_ATTENTION=0', { ...plan, cacheType: 'q8_0' })
        ).not.toContain('OLLAMA_FLASH_ATTENTION=1')
        // Going back to f16 removes the cache type PAIR set.
        expect(ollamaLaunchText('OLLAMA_KV_CACHE_TYPE=q8_0', plan)).not.toContain('KV_CACHE')
    })

    it('replaces every spelling of the llama.cpp options it owns', () => {
        const out = llamaCppLaunchText(
            'LLAMA_ARG_X=1 -c 4096 --parallel=2 --temp 0.7 -ctk q4_0 --ctx-size=8192',
            plan
        )
        expect(out).toBe('LLAMA_ARG_X=1 --temp 0.7 --ctx-size 131072 --parallel 4')
    })

    it('adds the llama.cpp cache type and flash attention once', () => {
        expect(llamaCppLaunchText('', { ...plan, cacheType: 'q4_0' })).toBe(
            '--ctx-size 131072 --parallel 4 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on'
        )
        expect(llamaCppLaunchText('-fa on', { ...plan, cacheType: 'q4_0' })).toBe(
            '-fa on --ctx-size 131072 --parallel 4 --cache-type-k q4_0 --cache-type-v q4_0'
        )
    })
})

describe('expert offload', () => {
    it('keeps llama.cpp experts on the CPU only when the plan says so', () => {
        expect(llamaCppLaunchText('--temp 0.7', { ...plan, offloadExperts: true })).toBe(
            '--temp 0.7 --ctx-size 131072 --parallel 4 --cpu-moe'
        )
        // Turning it off removes every spelling PAIR owns.
        expect(llamaCppLaunchText('-cmoe --n-cpu-moe 12 -ncmoe=3 --temp 0.7', plan)).toBe(
            '--temp 0.7 --ctx-size 131072 --parallel 4'
        )
    })

    it('sends system memory and the offload choice with a plan request', async () => {
        callProxy.mockResolvedValueOnce({ spec: {}, sessions: 3, canOffloadExperts: true })
        const result = await planCapacity({
            engine: 'llamacpp',
            model: 'm',
            windowTokens: 32768,
            offloadExperts: true
        })
        const params = callProxy.mock.calls[0][2]
        expect(params.systemRamBytes).toBeGreaterThan(0)
        expect(params.offloadExperts).toBe(true)
        expect(result.canOffloadExperts).toBe(true)
        expect(result.ramFits).toBe(true)
    })
})

describe('web-derived context', () => {
    it('reads a context the planner took from the model’s page, with its label', async () => {
        callProxy.mockResolvedValueOnce({
            spec: {
                maxContextTokens: 65536,
                contextSource: 'web',
                contextPage: 'https://ollama.com/library/example',
                contextReader: 'qwen3.5:4b',
                webMoe: true
            },
            windowTokens: 65536
        })
        const result = await planCapacity({ engine: 'llamacpp', model: 'm' })
        expect(result.spec).toMatchObject({
            maxContextTokens: 65536,
            contextSource: 'web',
            contextPage: 'https://ollama.com/library/example',
            contextReader: 'qwen3.5:4b',
            webMoe: true
        })
    })

    it('treats an unknown context source as the engine’s own', async () => {
        callProxy.mockResolvedValueOnce({
            spec: { maxContextTokens: 40960, contextSource: 'guess' }
        })
        const result = await planCapacity({ engine: 'ollama', model: 'm' })
        expect(result.spec.contextSource).toBe('')
        expect(result.spec.webMoe).toBe(false)
    })
})

describe('post and reply budgets', () => {
    it('asks the planner for a reply budget and reads the split back', async () => {
        callProxy.mockResolvedValueOnce({
            spec: {},
            windowTokens: 40960,
            sessions: 21,
            replyTokens: 4096,
            postTokens: 36864
        })
        const result = await planCapacity({
            engine: 'ollama',
            model: 'm',
            windowTokens: 40960,
            replyTokens: 4096
        })
        expect(callProxy.mock.calls[0][2].replyTokens).toBe(4096)
        expect(result.replyTokens).toBe(4096)
        expect(result.postTokens).toBe(36864)
    })

    it('leaves the reply budget to the planner’s default when unset', async () => {
        callProxy.mockResolvedValueOnce({ spec: {}, replyTokens: 8192, postTokens: 32768 })
        await planCapacity({ engine: 'ollama', model: 'm' })
        expect(callProxy.mock.calls[0][2]).not.toHaveProperty('replyTokens')
    })

    it('reads a saved plan’s reply budget', () => {
        const state = parsePlanner({
            plans: [
                {
                    engine: 'ollama',
                    model: 'm',
                    windowTokens: 8192,
                    sessions: 2,
                    replyTokens: 2048
                },
                { engine: 'ollama', model: 'n', windowTokens: 8192, sessions: 2 }
            ]
        })
        expect(state.plans.map(p => p.replyTokens)).toEqual([2048, 0])
    })
})

describe('capacity planner bridge', () => {
    it('sends this node’s GPUs with the plan request', async () => {
        callProxy.mockResolvedValueOnce({
            spec: { source: 'metadata', notes: [] },
            sessions: 7,
            windowTokens: 32768,
            fits: true
        })
        const result = await planCapacity({ engine: 'llamacpp', model: 'm', windowTokens: 32768 })
        const params = callProxy.mock.calls[0][2]
        expect(params.gpus).toHaveLength(2)
        expect(params.unified).toBe(false)
        expect(result.sessions).toBe(7)
        expect(result.error).toBe('')
    })

    it('reports the planner’s refusal as an error, not a throw', async () => {
        callProxy.mockRejectedValueOnce(
            new Error(
                'invalid params: the model’s context cost is unknown; enter it per token (kvBytesPerToken)'
            )
        )
        const result = await planCapacity({ engine: 'ollama', model: 'm' })
        expect(result.error).toBe(
            'the model’s context cost is unknown; enter it per token (kvBytesPerToken)'
        )
    })

    it('defaults auto reload to on', () => {
        expect(parsePlanner({ plans: [] }).autoReload).toBe(true)
        expect(parsePlanner({ autoReload: false }).autoReload).toBe(false)
    })

    it('splits llama.cpp across GPUs only when the plan does', () => {
        const split = { ...plan, tensorSplit: '32,32' }
        expect(llamaCppLaunchText('-sm row -ts 3,1 --temp 0.7', split)).toBe(
            '--temp 0.7 --ctx-size 131072 --parallel 4 --split-mode layer --tensor-split 32,32'
        )
        // Without a split the user's own GPU options stay.
        expect(llamaCppLaunchText('-ts 3,1', plan)).toBe('-ts 3,1 --ctx-size 131072 --parallel 4')
    })

    it('tells Ollama to spread across GPUs once, and never removes it', () => {
        expect(ollamaLaunchText('', { ...plan, spreadGpus: true })).toBe(
            'OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=4 OLLAMA_SCHED_SPREAD=1'
        )
        expect(ollamaLaunchText('OLLAMA_SCHED_SPREAD=1', { ...plan, spreadGpus: true })).toBe(
            'OLLAMA_SCHED_SPREAD=1 OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=4'
        )
        expect(ollamaLaunchText('OLLAMA_SCHED_SPREAD=1', plan)).toContain('OLLAMA_SCHED_SPREAD=1')
    })

    it('writes the GPU split the node needs now when applying', async () => {
        const gpu = { name: 'NVIDIA RTX 5090', totalBytes: 32 * 1024 ** 3, share: 0.5, layers: 32 }
        callProxy.mockImplementation(async (_engine: string, method: string) => {
            if (method === 'capacity/planner') return plannerState()
            if (method === 'capacity/plan')
                return { spec: { notes: [] }, gpus: [gpu, gpu], tensorSplit: '32,32' }
            return {}
        })
        callProcess.mockImplementation(async (_target: string, method: string) => {
            if (method === 'engine:get-settings')
                return settingsSnapshot('--temp 0.7', { appliedRevision: 8 })
            if (method === 'engine:apply-settings') return { revision: 8, phase: 'succeeded' }
            return null
        })
        const result = await applyPlan({ engine: 'llamacpp', model: 'm' })
        expect(result.message).toContain('--tensor-split 32,32')
        const sized = callProxy.mock.calls.find(c => c[1] === 'capacity/plan')
        expect(sized?.[2]).toMatchObject({ engine: 'llamacpp', model: 'm', windowTokens: 32768 })
        expect(sized?.[2].gpus).toHaveLength(2)
        const apply = callProcess.mock.calls.find(c => c[1] === 'engine:apply-settings')
        expect(apply?.[2].settings.launchText).toBe(
            '--temp 0.7 --ctx-size 131072 --parallel 4 --split-mode layer --tensor-split 32,32'
        )
    })

    it('applies a plan by rewriting launch settings and recording it', async () => {
        callProxy.mockImplementation(async (_engine: string, method: string) =>
            method === 'capacity/planner' ? plannerState() : {}
        )
        callProcess.mockImplementation(async (_target: string, method: string) => {
            if (method === 'engine:get-settings')
                return settingsSnapshot('--temp 0.7', { appliedRevision: 8 })
            if (method === 'engine:apply-settings') return { revision: 8, phase: 'succeeded' }
            return null
        })
        const result = await applyPlan({ engine: 'llamacpp', model: 'm' })
        expect(result.applied).toBe(true)
        const apply = callProcess.mock.calls.find(c => c[1] === 'engine:apply-settings')
        expect(apply?.[2]).toMatchObject({
            nodeId: 'node-1',
            engine: 'llamacpp',
            expectedRevision: 7,
            settings: { launchText: '--temp 0.7 --ctx-size 131072 --parallel 4' }
        })
        expect(apply?.[2].requestId).toMatch(/^[0-9a-f-]{36}$/)
        const applied = callProxy.mock.calls.find(c => c[1] === 'capacity/planner-set')
        expect(applied?.[2]).toEqual({
            applied: {
                engine: 'llamacpp',
                model: 'm',
                windowTokens: 32768,
                sessions: 4,
                cacheType: 'f16',
                offloadExperts: false
            }
        })
    })

    it('plans another node’s model on that node, for its GPUs and memory', async () => {
        callProxy.mockResolvedValueOnce({ spec: { notes: [] }, sessions: 3, windowTokens: 8192 })
        const result = await planCapacity({
            nodeId: 'node-2',
            engine: 'ollama',
            model: 'm',
            windowTokens: 8192
        })
        expect(result.sessions).toBe(3)
        const [, method, params] = callProxy.mock.calls[0]
        expect(method).toBe('capacity/peer-call')
        expect(params).toEqual({
            nodeId: 'node-2',
            method: 'capacity/plan',
            params: {
                engine: 'ollama',
                model: 'm',
                gpus: [{ name: 'NVIDIA GeForce RTX 4090', vramBytes: 24 * 1024 ** 3 }],
                unified: false,
                systemRamBytes: 64 * 1024 ** 3,
                windowTokens: 8192
            }
        })
    })

    it('reads this node’s planner directly and another node’s through the cluster', async () => {
        callProxy.mockResolvedValue(plannerState())
        await getPlanner()
        await getPlanner('node-1')
        await getPlanner('node-2')
        expect(callProxy.mock.calls.map(c => [c[1], c[2]])).toEqual([
            ['capacity/planner', {}],
            ['capacity/planner', {}],
            ['capacity/peer-call', { nodeId: 'node-2', method: 'capacity/planner', params: {} }]
        ])
    })

    it('applies another node’s plan to that node’s engine and records it there', async () => {
        callProxy.mockImplementation(
            async (_engine: string, method: string, params: Record<string, unknown>) =>
                method === 'capacity/peer-call' && params.method === 'capacity/planner'
                    ? plannerState()
                    : {}
        )
        callProcess.mockImplementation(async (_target: string, method: string) => {
            if (method === 'engine:get-settings')
                return settingsSnapshot('--temp 0.7', { nodeId: 'node-2', appliedRevision: 8 })
            if (method === 'engine:apply-settings') return { revision: 8, phase: 'succeeded' }
            return null
        })
        const result = await applyPlan({ engine: 'llamacpp', model: 'm', nodeId: 'node-2' })
        expect(result.applied).toBe(true)
        const settings = callProcess.mock.calls.filter(c => String(c[1]).startsWith('engine:'))
        expect(settings.every(c => c[2].nodeId === 'node-2')).toBe(true)
        // Nothing went to this node's own planner.
        expect(callProxy.mock.calls.every(c => c[1] === 'capacity/peer-call')).toBe(true)
        const applied = callProxy.mock.calls.find(c => c[2].method === 'capacity/planner-set')
        expect(applied?.[2]).toEqual({
            nodeId: 'node-2',
            method: 'capacity/planner-set',
            params: {
                applied: {
                    engine: 'llamacpp',
                    model: 'm',
                    windowTokens: 32768,
                    sessions: 4,
                    cacheType: 'f16',
                    offloadExperts: false
                }
            }
        })
    })

    it('limits LM Studio traffic with the plan’s reply budget', async () => {
        callProxy.mockImplementation(async (_e: string, method: string) =>
            method === 'capacity/planner'
                ? plannerState({
                      plans: [
                          {
                              engine: 'lmstudio',
                              model: 'm',
                              windowTokens: 8192,
                              sessions: 2,
                              cacheType: 'f16',
                              applied: false
                          }
                      ]
                  })
                : {}
        )
        await applyPlan({ engine: 'lmstudio', model: 'm' })
        const set = callProxy.mock.calls.find(c => c[1] === 'capacity/set')
        // The default 8,192 is more than half this window, so half is kept.
        expect(set?.[2]).toEqual({
            engine: 'lmstudio',
            model: 'm',
            contextTokens: 16384,
            maxSessions: 2,
            replyTokens: 4096
        })
    })

    it('does not restart an engine PAIR does not manage', async () => {
        callProxy.mockResolvedValue(plannerState())
        callProcess.mockResolvedValue(settingsSnapshot('', { editable: false, reason: 'adopted' }))
        const result = await applyPlan({ engine: 'llamacpp', model: 'm' })
        expect(result.applied).toBe(false)
        expect(result.message).toContain('adopted')
        expect(callProcess.mock.calls.some(c => c[1] === 'engine:apply-settings')).toBe(false)
    })

    it('waits for two idle checks before auto-reloading, and never while busy', async () => {
        callProcess.mockImplementation(async (_target: string, method: string) => {
            if (method === 'engine:get-settings')
                return settingsSnapshot('', { appliedRevision: 8 })
            if (method === 'engine:apply-settings') return { revision: 8, phase: 'succeeded' }
            return null
        })
        const busy = plannerState({
            activity: { sessions: 1, activeRequests: 0, queued: 0, idle: false }
        })
        callProxy.mockImplementation(async (_e: string, method: string) =>
            method === 'capacity/planner' ? busy : {}
        )
        await runAutoReloadCheckForTest()
        await runAutoReloadCheckForTest()
        expect(callProcess).not.toHaveBeenCalled()

        callProxy.mockImplementation(async (_e: string, method: string) =>
            method === 'capacity/planner' ? plannerState() : {}
        )
        await runAutoReloadCheckForTest()
        expect(callProcess).not.toHaveBeenCalled()
        await runAutoReloadCheckForTest()
        expect(callProcess.mock.calls.some(c => c[1] === 'engine:apply-settings')).toBe(true)
    })

    it('leaves pending plans alone when auto reload is off', async () => {
        callProxy.mockImplementation(async (_e: string, method: string) =>
            method === 'capacity/planner' ? plannerState({ autoReload: false }) : {}
        )
        await runAutoReloadCheckForTest()
        await runAutoReloadCheckForTest()
        expect(callProcess).not.toHaveBeenCalled()
    })
})
