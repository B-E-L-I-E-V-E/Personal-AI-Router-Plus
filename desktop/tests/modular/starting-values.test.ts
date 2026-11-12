// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { ModelDefaults, ModelSpec } from '@/shared/types/sessions'
import { startingValues, withStop } from '@/ui/components/SessionsSettings/starting-values'

function spec(defaults: ModelDefaults, extra: Partial<ModelSpec> = {}): ModelSpec {
    return {
        engine: 'ollama',
        model: 'qwen3.5:4b',
        source: 'metadata',
        architecture: 'qwen3_5',
        format: 'gguf',
        weightsBytes: 3 * 1024 ** 3,
        maxContextTokens: 262144,
        contextSource: '',
        contextPage: '',
        contextReader: '',
        webMoe: false,
        kvBytesPerToken: 32768,
        expertCount: 0,
        expertBytes: 0,
        defaults,
        notes: [],
        ...extra
    }
}

describe('the planner’s starting values for a model with no saved plan', () => {
    it('starts from what the engine runs with now, marked with its source', () => {
        expect(
            startingValues(
                spec({
                    window: { value: 40960, source: 'launch' },
                    sessions: { value: 4, source: 'launch' },
                    reply: { value: 2048, source: 'model' }
                })
            )
        ).toEqual({
            window: { value: 40960, source: 'launch' },
            sessions: { value: 4, source: 'launch' },
            reply: { value: 2048, source: 'model' }
        })
    })

    it('falls back to the trained context when nothing else is reported', () => {
        expect(startingValues(spec({}))).toEqual({
            window: { value: 262144, source: 'maximum' }
        })
    })

    it('keeps today’s behaviour when the window came only from the model’s page', () => {
        expect(startingValues(spec({}, { contextSource: 'web' }))).toEqual({})
    })

    it.each([
        {
            name: 'a window past the model’s maximum is capped at it',
            defaults: { window: { value: 1 << 20, source: 'engine' as const } },
            want: { window: { value: 262144, source: 'engine' } }
        },
        {
            name: 'a window below the planner’s smallest falls back to the maximum',
            defaults: { window: { value: 1024, source: 'model' as const } },
            want: { window: { value: 262144, source: 'maximum' } }
        },
        {
            name: 'more slots than the planner plans are capped',
            defaults: { sessions: { value: 512, source: 'launch' as const } },
            want: {
                window: { value: 262144, source: 'maximum' },
                sessions: { value: 256, source: 'launch' }
            }
        }
    ])('$name', ({ defaults, want }) => {
        expect(startingValues(spec(defaults))).toEqual(want)
    })
})

describe('withStop', () => {
    it('adds a starting value between the slider’s stops', () => {
        expect(withStop([2048, 32768, 65536], 40960)).toEqual([2048, 32768, 40960, 65536])
    })

    it('leaves the stops alone for a value already there, past the top, or none', () => {
        const stops = [2048, 4096]
        expect(withStop(stops, 4096)).toBe(stops)
        expect(withStop(stops, 8192)).toBe(stops)
        expect(withStop(stops, undefined)).toBe(stops)
    })
})
