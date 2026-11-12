// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'

const callProxy = vi.fn()

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => ({ callProxy })
}))
vi.mock('@/electron/service-bridge/modular-state', () => ({
    getModularBridgeState: () => ({ getSelfId: () => 'node-1' })
}))

import {
    getModelDetails,
    getModelWebFacts,
    startModelWebFacts
} from '@/electron/service-bridge/model-details'
import { detailRows, engineGaps, webFillRows } from '@/ui/utils/model-details'

// The proxy's answer for qwen3:0.6b on a live Ollama 0.40.11.
const qwen3 = {
    engine: 'ollama',
    model: 'qwen3:0.6b',
    license: 'apache-2.0',
    type: 'model',
    architecture: 'qwen3',
    family: 'qwen3',
    parameterSize: '751.63M',
    parameterCount: 751632384,
    quantization: 'Q4_K_M',
    format: 'gguf',
    capabilities: ['completion', 'thinking', 'tools'],
    contextLength: 40960,
    sizeBytes: 522653767,
    digest: '7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435',
    modifiedAt: '2026-11-10T23:53:45.015542771-08:00',
    links: [
        { label: 'Ollama library', url: 'https://ollama.com/library/qwen3', source: 'model name' },
        { label: 'Bad', url: 'javascript:alert(1)', source: 'general.url' }
    ]
}

describe('model details bridge', () => {
    beforeEach(() => callProxy.mockReset())

    it('asks this node’s proxy without a node id for its own models', async () => {
        callProxy.mockResolvedValue(qwen3)
        const d = await getModelDetails({
            engineType: 'ollama',
            nodeId: 'node-1',
            model: 'qwen3:0.6b'
        })
        expect(callProxy).toHaveBeenCalledWith('ollama', 'model/details', {
            engine: 'ollama',
            model: 'qwen3:0.6b'
        })
        expect(d.family).toBe('qwen3')
        expect(d.capabilities).toEqual(['completion', 'thinking', 'tools'])
        expect(d.description).toBe('')
        expect(d.tags).toEqual([])
    })

    it('names the node that has the model when it is another node', async () => {
        callProxy.mockResolvedValue({ engine: 'lmstudio', model: 'm' })
        await getModelDetails({ engineType: 'lm-studio', nodeId: 'node-2', model: 'm' })
        expect(callProxy).toHaveBeenCalledWith('lm-studio', 'model/details', {
            engine: 'lmstudio',
            model: 'm',
            nodeId: 'node-2'
        })
    })

    it('keeps only web links', async () => {
        callProxy.mockResolvedValue(qwen3)
        const d = await getModelDetails({
            engineType: 'ollama',
            nodeId: 'node-1',
            model: 'qwen3:0.6b'
        })
        expect(d.links).toEqual([
            {
                label: 'Ollama library',
                url: 'https://ollama.com/library/qwen3',
                source: 'model name'
            }
        ])
    })

    it('turns a refusal into a note instead of throwing', async () => {
        callProxy.mockRejectedValueOnce(new Error('invalid params: node "x" is not known'))
        const d = await getModelDetails({ engineType: 'llama-cpp', nodeId: 'x', model: 'm' })
        expect(d.engine).toBe('llama-cpp')
        expect(d.notes).toEqual(['Could not read the model\'s details: node "x" is not known'])
    })
})

describe('model detail rows', () => {
    it('lists only what the engine reported, in reading order', async () => {
        callProxy.mockResolvedValue(qwen3)
        const d = await getModelDetails({
            engineType: 'ollama',
            nodeId: 'node-1',
            model: 'qwen3:0.6b'
        })
        const labels = detailRows(d).map(([label]) => label)
        expect(labels).toEqual([
            'Type',
            'Family',
            'Architecture',
            'Parameters',
            'Quantization',
            'Format',
            'Trained context',
            'Size on disk',
            'License',
            'Modified',
            'Digest'
        ])
        expect(Object.fromEntries(detailRows(d))).toMatchObject({
            Parameters: '751.63M · 752M',
            'Trained context': '40,960 tokens',
            Digest: '7df6b6e09427'
        })
    })
})

// A lookup's status as the proxy reports it after reading ollama.com.
const webStatus = {
    readerModel: 'qwen3.5:4b',
    job: { state: 'done', reader: 'qwen3.5:4b', chunksTotal: 1, chunksRead: 1, startedAt: 1 },
    result: {
        engine: 'ollama',
        model: 'qwen3:0.6b',
        reader: 'qwen3.5:4b',
        readAt: 1762900000000,
        pages: [{ url: 'https://ollama.com/library/qwen3', chunks: 1 }, { url: 'file:///x' }],
        facts: [
            {
                field: 'description',
                value: 'Qwen3 is the latest generation of large language models in Qwen series.',
                source: 'https://ollama.com/library/qwen3',
                fills: 'description'
            },
            {
                field: 'context',
                value: 40960,
                source: 'https://ollama.com/library/qwen3'
            },
            {
                field: 'moe',
                value: false,
                source: 'https://ollama.com/library/qwen3',
                fills: 'moe'
            },
            { field: 'tools', value: { nested: true }, source: 'https://ollama.com/library/qwen3' },
            { field: 'vision', value: true, source: 'javascript:alert(1)', fills: 'capabilities' }
        ],
        conflicts: [
            {
                field: 'dataSize',
                values: [
                    { value: '523MB', source: 'https://ollama.com/library/qwen3' },
                    { value: '5.2GB', source: 'https://ollama.com/library/qwen3' }
                ]
            },
            { field: 'name', values: [{ value: 'Qwen3', source: 'x' }] }
        ]
    }
}

describe('web facts bridge', () => {
    beforeEach(() => callProxy.mockReset())

    it('asks for a model’s status without a node id for this node', async () => {
        callProxy.mockResolvedValue(webStatus)
        await getModelWebFacts({ engineType: 'ollama', nodeId: 'node-1', model: 'qwen3:0.6b' })
        expect(callProxy).toHaveBeenCalledWith('ollama', 'model/web-facts', {
            engine: 'ollama',
            model: 'qwen3:0.6b'
        })
    })

    it('starts a lookup with the reader model and the other node it names', async () => {
        callProxy.mockResolvedValue({ readerModel: 'qwen3.5:4b', job: { state: 'running' } })
        const st = await startModelWebFacts({
            engineType: 'llama-cpp',
            nodeId: 'node-2',
            model: 'm',
            readerModel: ' qwen3.5:4b '
        })
        expect(callProxy).toHaveBeenCalledWith('llama-cpp', 'model/web-facts-start', {
            engine: 'llamacpp',
            model: 'm',
            nodeId: 'node-2',
            readerModel: 'qwen3.5:4b'
        })
        expect(st.job?.state).toBe('running')
        expect(st.result).toBeNull()
    })

    it('never sends a reader model when only reading the status', async () => {
        callProxy.mockResolvedValue({})
        await getModelWebFacts({
            engineType: 'ollama',
            nodeId: 'node-1',
            model: 'm',
            readerModel: 'x'
        })
        expect(callProxy.mock.calls[0][2]).not.toHaveProperty('readerModel')
    })

    it('keeps only valid values with a web source, and real conflicts', async () => {
        callProxy.mockResolvedValue(webStatus)
        const st = await getModelWebFacts({
            engineType: 'ollama',
            nodeId: 'node-1',
            model: 'qwen3:0.6b'
        })
        expect(st.result?.reader).toBe('qwen3.5:4b')
        expect(st.result?.pages).toEqual([
            { url: 'https://ollama.com/library/qwen3', chunks: 1, note: '' }
        ])
        expect(st.result?.facts.map(f => [f.field, f.value, f.fills])).toEqual([
            ['description', webStatus.result.facts[0].value, 'description'],
            ['context', 40960, ''],
            ['moe', false, 'moe']
        ])
        expect(st.result?.conflicts.map(c => c.field)).toEqual(['dataSize'])
    })

    it('shows only the values that fill a gap the engine left', async () => {
        callProxy.mockResolvedValue(webStatus)
        const st = await getModelWebFacts({
            engineType: 'ollama',
            nodeId: 'node-1',
            model: 'qwen3:0.6b'
        })
        expect(webFillRows(st.result?.facts ?? [])).toEqual([
            {
                label: 'Description',
                text: webStatus.result.facts[0].value,
                source: 'https://ollama.com/library/qwen3'
            },
            { label: 'Mixture of experts', text: 'No', source: 'https://ollama.com/library/qwen3' }
        ])
    })

    it('ignores an unknown job state and defaults the reader model', async () => {
        callProxy.mockResolvedValue({ job: { state: 'exploded' } })
        const st = await getModelWebFacts({ engineType: 'ollama', nodeId: 'node-1', model: 'm' })
        expect(st.job).toBeNull()
        expect(st.readerModel).toBe('qwen3.5:4b')
    })

    it('turns a refusal into an error instead of throwing', async () => {
        callProxy.mockRejectedValueOnce(
            new Error('invalid params: another web lookup is running; try again when it finishes')
        )
        const st = await startModelWebFacts({ engineType: 'ollama', nodeId: 'node-1', model: 'm' })
        expect(st.error).toBe('another web lookup is running; try again when it finishes')
    })
})

describe('engine gaps', () => {
    it('lists what qwen3:0.6b’s engine left empty', async () => {
        callProxy.mockResolvedValue(qwen3)
        const d = await getModelDetails({
            engineType: 'ollama',
            nodeId: 'node-1',
            model: 'qwen3:0.6b'
        })
        expect(engineGaps(d)).toEqual(['Name', 'Description'])
    })
})
