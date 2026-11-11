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
        ]
    })
}))

import {
    applyPlan,
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
            new Error('invalid params: the model’s context cost is unknown; enter it per token (kvBytesPerToken)')
        )
        const result = await planCapacity({ engine: 'ollama', model: 'm' })
        expect(result.error).toBe('the model’s context cost is unknown; enter it per token (kvBytesPerToken)')
    })

    it('defaults auto reload to on', () => {
        expect(parsePlanner({ plans: [] }).autoReload).toBe(true)
        expect(parsePlanner({ autoReload: false }).autoReload).toBe(false)
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
        const applied = callProxy.mock.calls.find(c => c[1] === 'capacity/planner-set')
        expect(applied?.[2]).toEqual({
            applied: {
                engine: 'llamacpp',
                model: 'm',
                windowTokens: 32768,
                sessions: 4,
                cacheType: 'f16'
            }
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
