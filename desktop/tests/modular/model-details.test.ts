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

import { getModelDetails } from '@/electron/service-bridge/model-details'
import { detailRows } from '@/ui/utils/model-details'

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
