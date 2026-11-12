// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { CapacityPlan } from '@/shared/types/sessions'
import {
    memoryUse,
    nearestStop,
    replyBudget,
    replyStops,
    sessionsFit,
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
    it('reaches past what a large unified-memory plan fits', () => {
        // qwen3:0.6b on a 128 GiB Mac: 77 sessions of 40,960 tokens at q4_0.
        const fit = sessionsFit(plan({ sessions: 77, unified: true }))
        expect(fit).toBe(77)
        expect(sessionsSliderMax(fit ?? 0)).toBe(154)
    })

    it('offers no count when the plan could not size the model', () => {
        // A model whose context cost is unknown used to leave a 1–8 slider.
        const unsized = plan({
            sessions: 0,
            error: 'the model’s context cost is unknown; enter it per token (kvBytesPerToken)'
        })
        expect(sessionsFit(unsized)).toBeNull()
        expect(sessionsFit(null)).toBeNull()
    })

    it('keeps a fit of zero, so a too-large window shows every count as over', () => {
        expect(sessionsFit(plan({ sessions: 0, fits: false }))).toBe(0)
    })

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
