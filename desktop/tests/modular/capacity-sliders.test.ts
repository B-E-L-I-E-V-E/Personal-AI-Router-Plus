// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { CapacityPlan, CapacityPlanGpu } from '@/shared/types/sessions'
import { multiGpuLaunch, multiGpuLaunchText, oneGpuIndex } from '@/shared/utils/capacity-gpus'
import {
    gpuBars,
    memoryUse,
    nearestStop,
    replyBudget,
    replyStops,
    sessionsSliderMax,
    shortGpuName,
    shortTokens,
    textKindFits,
    windowStops
} from '@/ui/components/SessionsSettings/capacity-sliders'

const GiB = 1024 ** 3

function plan(overrides: Partial<CapacityPlan> = {}): CapacityPlan {
    return {
        spec: {
            engine: 'llamacpp',
            model: 'm',
            source: 'metadata',
            architecture: '',
            format: '',
            weightsBytes: 18 * GiB,
            maxContextTokens: 40960,
            contextSource: '',
            contextPage: '',
            contextReader: '',
            webMoe: false,
            kvBytesPerToken: 0,
            expertCount: 0,
            expertBytes: 0,
            layers: 48,
            notes: []
        },
        totalBytes: 24 * GiB,
        reserveBytes: 2 * GiB,
        weightsBytes: 18 * GiB,
        gpuWeightsBytes: 18 * GiB,
        otherModelsBytes: 0,
        contextBytes: 4 * GiB,
        bytesPerToken: 0,
        bytesPerSession: 1 * GiB,
        cacheType: 'f16',
        windowTokens: 32768,
        sessions: 4,
        replyTokens: 8192,
        postTokens: 32768,
        poolTokens: 0,
        usedBytes: 0,
        fits: true,
        canOffloadExperts: false,
        offloadExperts: false,
        expertBytes: 0,
        offloadNote: '',
        unified: false,
        ramTotalBytes: 64 * GiB,
        ramReserveBytes: 16 * GiB,
        ramUsedBytes: 0,
        ramFits: true,
        gpus: [],
        tensorSplit: '',
        oneGpuIndex: 0,
        oneGpuSessions: 0,
        warnings: [],
        error: '',
        ...overrides
    }
}

describe('window slider stops', () => {
    it('stops at powers of two and the model maximum', () => {
        expect(windowStops(40960)).toEqual([2048, 4096, 8192, 16384, 32768, 40960])
        expect(windowStops(131072).at(-1)).toBe(131072)
        expect(windowStops(0).at(-1)).toBe(1 << 20)
    })

    it('snaps to the nearest stop', () => {
        const stops = windowStops(40960)
        expect(stops[nearestStop(stops, 30000)]).toBe(32768)
        expect(stops[nearestStop(stops, 40960)]).toBe(40960)
    })

    it('labels stops briefly', () => {
        expect(shortTokens(32768)).toBe('32K')
        expect(shortTokens(40960)).toBe('40K')
        expect(shortTokens(1 << 20)).toBe('1M')
    })
})

describe('sessions slider', () => {
    it('reaches well past what fits, within the proxy limit', () => {
        expect(sessionsSliderMax(0)).toBe(8)
        expect(sessionsSliderMax(3)).toBe(11)
        expect(sessionsSliderMax(21)).toBe(42)
        expect(sessionsSliderMax(250)).toBe(256)
    })
})

describe('memory calculator', () => {
    it('adds model, sessions and reserve on the GPU and turns over past the total', () => {
        expect(memoryUse(plan(), 4)).toMatchObject({ gpuNeeded: 24 * GiB, gpuOver: false })
        expect(memoryUse(plan(), 5).gpuOver).toBe(true)
    })

    it('counts offloaded experts against system memory', () => {
        const off = plan({ offloadExperts: true, gpuWeightsBytes: 2 * GiB, ramUsedBytes: 16 * GiB })
        expect(memoryUse(off, 20)).toMatchObject({
            gpuNeeded: 24 * GiB,
            gpuOver: false,
            ramNeeded: 32 * GiB,
            ramOver: false
        })
        expect(memoryUse({ ...off, ramTotalBytes: 24 * GiB }, 1).ramOver).toBe(true)
    })

    it('treats Apple silicon memory as one shared pool', () => {
        expect(memoryUse(plan({ unified: true }), 1).shared).toBe(true)
    })
})

describe('text kinds', () => {
    it('lists measured kinds from most to least text per token, rounded down', () => {
        const fits = textKindFits(40960)
        expect(fits.map(f => f.label)[0]).toBe('English prose')
        expect(fits.find(f => f.label === 'English prose')?.characters).toBe(160000)
        expect(fits.find(f => f.label === 'English prose')).toMatchObject({
            words: 27000,
            pages: 54
        })
        expect(fits.find(f => f.label === 'Chinese')).toMatchObject({
            characters: 69000,
            words: 46000,
            pages: 92
        })
        // Every kind has words and pages.
        for (const f of fits) expect(f.words > 0 && f.pages > 0).toBe(true)
        expect(fits.at(-1)?.characters).toBe(36000)
        for (let i = 1; i < fits.length; i++)
            expect(fits[i].charsPerToken).toBeLessThanOrEqual(fits[i - 1].charsPerToken)
    })
})

describe('reply budget', () => {
    it('stops at powers of two up to half the window', () => {
        expect(replyStops(40960)).toEqual([512, 1024, 2048, 4096, 8192, 16384])
        expect(replyStops(2048)).toEqual([512, 1024])
    })

    it('defaults to 8,192 and never takes more than half the window', () => {
        expect(replyBudget(40960)).toBe(8192)
        expect(replyBudget(40960, 4096)).toBe(4096)
        expect(replyBudget(8192)).toBe(4096)
    })
})

/** Two 24 GB GPUs holding half the 18 GiB model and half of each session. */
function twoGpus(overrides: Partial<CapacityPlan> = {}): CapacityPlan {
    const half: CapacityPlanGpu = {
        name: 'NVIDIA GeForce RTX 4090',
        totalBytes: 24 * GiB,
        reserveBytes: 2 * GiB,
        layers: 24,
        share: 0.5,
        weightsBytes: 9 * GiB,
        otherModelsBytes: 0,
        bytesPerSession: 0.5 * GiB,
        sessions: 26
    }
    return plan({
        totalBytes: 48 * GiB,
        reserveBytes: 4 * GiB,
        gpus: [half, { ...half }],
        tensorSplit: '24,24',
        ...overrides
    })
}

describe('GPU bars', () => {
    it('shows nothing extra for one GPU', () => {
        expect(gpuBars(plan(), 4, 'llamacpp')).toEqual([])
    })

    it('splits the model and each session across two GPUs', () => {
        const bars = gpuBars(twoGpus(), 10, 'llamacpp')
        expect(bars.map(b => b.label)).toEqual(['GPU 0 · RTX 4090', 'GPU 1 · RTX 4090'])
        // 9 GiB of weights + 10 × 0.5 GiB of context + 2 GiB kept free.
        expect(bars[0]).toMatchObject({
            needed: 16 * GiB,
            total: 24 * GiB,
            over: false,
            layers: 24
        })
    })

    it('turns only the GPU that runs out red', () => {
        const p = twoGpus()
        p.gpus[1] = { ...p.gpus[1], totalBytes: 12 * GiB }
        const bars = gpuBars(p, 10, 'llamacpp')
        expect(bars.map(b => b.over)).toEqual([false, true])
    })

    it('puts the whole model on one GPU when Ollama keeps it there', () => {
        const p = twoGpus({ oneGpuIndex: 1, oneGpuSessions: 3, gpuWeightsBytes: 4 * GiB })
        const bars = gpuBars(p, 3, 'ollama')
        expect(bars[0]).toMatchObject({ idle: true, needed: 2 * GiB, layers: 0 })
        // 4 GiB of weights + 3 × 1 GiB of context + 2 GiB kept free.
        expect(bars[1]).toMatchObject({ idle: false, needed: 9 * GiB, layers: 48, share: 1 })
        // More sessions than one GPU holds: Ollama spreads the model.
        expect(gpuBars(p, 4, 'ollama').every(b => !b.idle)).toBe(true)
    })

    it('shortens GPU names to the model', () => {
        expect(shortGpuName('NVIDIA GeForce RTX 4090')).toBe('RTX 4090')
        expect(shortGpuName('Apple M5 Max')).toBe('Apple M5 Max')
    })
})

describe('multi-GPU launch settings', () => {
    it('gives llama.cpp the layer split', () => {
        const launch = multiGpuLaunch('llamacpp', twoGpus(), 4)
        expect(launch).toEqual({ tensorSplit: '24,24', spreadGpus: false })
        expect(multiGpuLaunchText(launch)).toBe('--split-mode layer --tensor-split 24,24')
    })

    it('tells Ollama to spread only when one GPU is not enough', () => {
        const p = twoGpus({ oneGpuIndex: 0, oneGpuSessions: 3 })
        expect(oneGpuIndex('ollama', p, 3)).toBe(0)
        expect(multiGpuLaunch('ollama', p, 3).spreadGpus).toBe(false)
        expect(multiGpuLaunch('ollama', p, 4).spreadGpus).toBe(true)
        expect(multiGpuLaunchText(multiGpuLaunch('ollama', p, 4))).toBe('OLLAMA_SCHED_SPREAD=1')
    })

    it('sets nothing for one GPU, LM Studio, or a plan that failed', () => {
        expect(multiGpuLaunch('llamacpp', plan(), 4)).toEqual({
            tensorSplit: '',
            spreadGpus: false
        })
        expect(multiGpuLaunch('lmstudio', twoGpus(), 4).tensorSplit).toBe('')
        expect(multiGpuLaunch('llamacpp', twoGpus({ error: 'x' }), 4).tensorSplit).toBe('')
        expect(multiGpuLaunchText({ tensorSplit: '', spreadGpus: false })).toBe('')
    })
})
