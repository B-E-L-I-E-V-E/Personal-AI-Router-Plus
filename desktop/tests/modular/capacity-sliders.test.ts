// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { CapacityPlan } from '@/shared/types/sessions'
import {
    memoryUse,
    nearestStop,
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
            kvBytesPerToken: 0,
            expertCount: 0,
            expertBytes: 0,
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
        expect(fits.find(f => f.label === 'English prose')?.words).toBe(27000)
        expect(fits.find(f => f.label === 'Chinese')?.words).toBe(0)
        expect(fits.at(-1)?.characters).toBe(36000)
        for (let i = 1; i < fits.length; i++)
            expect(fits[i].charsPerToken).toBeLessThanOrEqual(fits[i - 1].charsPerToken)
    })
})
