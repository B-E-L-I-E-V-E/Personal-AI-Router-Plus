// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { CapacityPlan } from '@/shared/types/sessions'
import {
    kvStops,
    labelledStops,
    memoryUse,
    nearestStop,
    parseAmount,
    perTokenLabel,
    replyBudget,
    replyStops,
    sessionsFit,
    sessionsSliderMax,
    shortTokens,
    sizeLabel,
    textKindFits,
    weightStops,
    windowStops,
    withStop
} from '@/ui/components/SessionsSettings/capacity-sliders'
import {
    kvFact,
    kvUsed,
    loadedFact,
    sizeFacts,
    weightsUsed
} from '@/ui/components/SessionsSettings/model-size'

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
            weightsSource: 'tensors',
            downloadBytes: 18 * GiB,
            downloadSource: 'gguf-files',
            tensorBytes: 18 * GiB,
            loadedBytes: 0,
            loadedVramBytes: 0,
            loadedContextTokens: 0,
            maxContextTokens: 40960,
            contextSource: '',
            contextPage: '',
            contextReader: '',
            webMoe: false,
            kvBytesPerToken: 0,
            expertCount: 0,
            expertBytes: 0,
            defaults: {},
            notes: []
        },
        totalBytes: 24 * GiB,
        reserveBytes: 2 * GiB,
        weightsBytes: 18 * GiB,
        weightsSource: 'tensors',
        kvSource: 'metadata',
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

describe('model weights slider', () => {
    it('stops at round sizes past the node memory and at the model size itself', () => {
        const stops = weightStops(3.7 * GiB, 16 * GiB)
        expect(stops[0]).toBe(0.25 * GiB)
        expect(stops).toContain(3.7 * GiB)
        expect(stops.at(-1)).toBe(24 * GiB)
        for (let i = 1; i < stops.length; i++) expect(stops[i]).toBeGreaterThan(stops[i - 1])
    })

    it('reaches twice a model larger than the node', () => {
        expect(weightStops(400 * GiB, 128 * GiB).at(-1)).toBe(1024 * GiB)
        expect(weightStops(0, 0).at(-1)).toBe(8 * GiB)
    })

    it('does not add a value already within 1% of a stop', () => {
        expect(withStop([GiB, 2 * GiB], 2.01 * GiB)).toEqual([GiB, 2 * GiB])
        expect(withStop([GiB, 2 * GiB], 0)).toEqual([GiB, 2 * GiB])
        expect(withStop([GiB, 2 * GiB], 1.5 * GiB)).toEqual([GiB, 1.5 * GiB, 2 * GiB])
    })
})

describe('context cost slider', () => {
    it('stops from 2 KB to 2 MB per token and at the model cost', () => {
        const stops = kvStops(112 * 1024)
        expect(stops[0]).toBe(2048)
        expect(stops.at(-1)).toBe(2048 * 1024)
        expect(stops).toContain(112 * 1024)
        expect(kvStops(128 * 1024)).toHaveLength(20)
    })

    it('labels about six stops, the ends and the model value', () => {
        const stops = kvStops(112 * 1024)
        const labelled = labelledStops(stops, 112 * 1024)
        expect(labelled[0]).toBe(0)
        expect(labelled.at(-1)).toBe(stops.length - 1)
        expect(labelled).toContain(stops.indexOf(112 * 1024))
        expect(labelled.length).toBeLessThanOrEqual(9)
    })
})

describe('amount formatting and parsing', () => {
    it('shows sizes in MB below 1 GB and trims trailing zeros', () => {
        expect(sizeLabel(512 * 1024 ** 2)).toBe('512 MB')
        expect(sizeLabel(4 * GiB)).toBe('4 GB')
        expect(sizeLabel(3.7 * GiB)).toBe('3.7 GB')
        expect(sizeLabel(42.5 * GiB)).toBe('42.5 GB')
    })

    it('shows per-token costs in KB, and MB from 1 MB', () => {
        expect(perTokenLabel(112 * 1024)).toBe('112 KB')
        expect(perTokenLabel(59.5 * 1024)).toBe('59.5 KB')
        expect(perTokenLabel(1536 * 1024)).toBe('1.5 MB')
    })

    it('parses positive decimals only', () => {
        expect(parseAmount(' 4.25 ', GiB)).toBe(4.25 * GiB)
        expect(parseAmount('112', 1024)).toBe(112 * 1024)
        expect(parseAmount('0', GiB)).toBeNull()
        expect(parseAmount('4,2', GiB)).toBeNull()
        expect(parseAmount('-1', GiB)).toBeNull()
        expect(parseAmount('', GiB)).toBeNull()
    })
})

describe('model size sources', () => {
    const mlx = plan({
        weightsBytes: 3.7 * GiB,
        weightsSource: 'download',
        spec: {
            ...plan().spec,
            format: 'safetensors',
            weightsBytes: 3.7 * GiB,
            weightsSource: 'download',
            downloadBytes: 3.7 * GiB,
            downloadSource: 'ollama-tags',
            tensorBytes: 0,
            loadedBytes: 3.8 * GiB,
            loadedVramBytes: 3.8 * GiB,
            loadedContextTokens: 4096
        }
    })

    it('names the download size and its source apart from the in-memory size', () => {
        expect(sizeFacts(plan().spec)).toEqual([
            'Download size 18 GB (the GGUF files)',
            'In memory when loaded ≈ 18 GB (its weight tensors added up)'
        ])
        expect(sizeFacts(mlx.spec)).toEqual([
            'Download size 3.7 GB (Ollama’s model list)',
            'In memory when loaded: not reported for safetensors models'
        ])
    })

    it('says which size the plan uses', () => {
        expect(weightsUsed(plan())).toBe('The plan uses the in-memory size.')
        expect(weightsUsed(mlx)).toMatch(/download size as an estimate/)
        expect(
            weightsUsed(
                plan({
                    weightsSource: 'download',
                    spec: { ...plan().spec, downloadBytes: 20 * GiB }
                })
            )
        ).toMatch(/holds more than the weight tensors/)
        expect(weightsUsed(plan({ weightsSource: 'manual', weightsBytes: 20 * GiB }))).toBe(
            'The plan uses the 20 GB you set (the model’s: 18 GB).'
        )
        expect(weightsUsed(plan({ weightsSource: '' }))).toMatch(/unknown/)
    })

    it('describes a loaded copy as more than its weights', () => {
        expect(loadedFact(mlx.spec)).toBe(
            'Loaded now: 3.8 GB with a 4,096-token context. That is the weights, their context cache and the engine’s working buffers together.'
        )
        expect(
            loadedFact({ ...mlx.spec, loadedVramBytes: 2 * GiB, loadedContextTokens: 0 })
        ).toMatch(/^Loaded now: 3\.8 GB, 2 GB of it on the GPU\./)
        expect(loadedFact(plan().spec)).toBe('')
    })

    it('says where the per-token cost came from and its precision', () => {
        expect(kvFact(plan())).toMatch(/architecture.*f16/)
        expect(kvFact(plan({ kvSource: 'measured' }))).toMatch(/current cache precision/)
        expect(
            kvFact(
                plan({ kvSource: 'manual', spec: { ...plan().spec, kvBytesPerToken: 112 * 1024 } })
            )
        ).toBe('The value you set, at full precision (f16); the model’s: 112 KB.')
        expect(kvFact(plan({ kvSource: '' }))).toMatch(/does not describe/)
        expect(
            kvUsed(plan({ bytesPerToken: 59.5 * 1024, cacheType: 'q8_0', windowTokens: 32768 }))
        ).toBe(
            'This plan counts 59.5 KB per token at 8-bit, so a session of 32,768 tokens needs 1 GB for its context.'
        )
        expect(kvUsed(plan())).toBe('')
    })
})
