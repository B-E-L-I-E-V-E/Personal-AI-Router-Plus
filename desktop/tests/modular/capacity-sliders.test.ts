// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { CapacityPlan } from '@/shared/types/sessions'
import {
    cpuExpertBytes,
    expertLayersOnCpu,
    expertSplitStops,
    memoryUse,
    nearestStop,
    parameterLabel,
    replyBudget,
    replyStops,
    sessionsSliderMax,
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
            expertUsedCount: 0,
            expertBytes: 0,
            expertLayerBytes: [],
            totalParams: 0,
            activeParams: 0,
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
        cpuExpertLayers: 0,
        expertBytes: 0,
        offloadNote: '',
        unified: false,
        ramTotalBytes: 64 * GiB,
        ramReserveBytes: 16 * GiB,
        ramUsedBytes: 0,
        ramFits: true,
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

    it('counts the experts of the layers moved to system memory there, not on the GPU', () => {
        const moe = plan()
        // 16 expert layers of 1 GiB.
        moe.spec.expertLayerBytes = Array.from({ length: 16 }, () => GiB)
        expect(memoryUse(moe, 20, 16)).toMatchObject({
            gpuNeeded: 24 * GiB,
            gpuOver: false,
            ramNeeded: 32 * GiB,
            ramOver: false
        })
        // Half the experts in system memory: 8 GiB more on the GPU.
        expect(memoryUse(moe, 4, 8)).toMatchObject({ gpuNeeded: 16 * GiB, ramNeeded: 24 * GiB })
        expect(memoryUse(moe, 4, 0)).toMatchObject({ gpuNeeded: 24 * GiB, ramOver: false })
        expect(memoryUse({ ...moe, ramTotalBytes: 24 * GiB }, 1, 16).ramOver).toBe(true)
    })

    it('treats Apple silicon memory as one shared pool', () => {
        expect(memoryUse(plan({ unified: true }), 1).shared).toBe(true)
    })
})

describe('expert split slider', () => {
    // Two dense layers, then three expert layers of different sizes.
    const layers = [0, 0, 3 * GiB, 2 * GiB, 2 * GiB]

    it('stops once per expert layer, skipping dense ones', () => {
        expect(expertSplitStops(layers)).toEqual([0, 3, 4, 5])
        expect(expertSplitStops([])).toEqual([0])
    })

    it('sums the experts of the first layers, as --n-cpu-moe does', () => {
        expect(cpuExpertBytes(layers, 0)).toBe(0)
        expect(cpuExpertBytes(layers, 3)).toBe(3 * GiB)
        expect(cpuExpertBytes(layers, 5)).toBe(7 * GiB)
        expect(cpuExpertBytes(layers, 1024)).toBe(7 * GiB)
    })

    it('reads a saved --n-cpu-moe count as expert layers', () => {
        const stops = expertSplitStops(layers)
        expect(expertLayersOnCpu(stops, 0)).toBe(0)
        expect(expertLayersOnCpu(stops, 2)).toBe(0)
        expect(expertLayersOnCpu(stops, 4)).toBe(2)
        // A plan saved before the split kept every expert on the CPU.
        expect(expertLayersOnCpu(stops, 1024)).toBe(3)
    })

    it('labels parameter counts as models are named', () => {
        expect(parameterLabel(284_334_567_511)).toBe('284B')
        expect(parameterLabel(30_532_122_624)).toBe('30.5B')
        expect(parameterLabel(13_801_959_511)).toBe('13.8B')
        expect(parameterLabel(600_000_000)).toBe('600M')
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
