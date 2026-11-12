// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { PerfModelSummary } from '@/shared/types/performance'
import { groupByNode } from '@/ui/components/Performance/performance-nodes'

function summary(fields: Partial<PerfModelSummary>): PerfModelSummary {
    return {
        engine: 'ollama',
        model: 'm',
        nodeId: 'node-a',
        local: false,
        requests: 1,
        promptTokens: 10,
        completionTokens: 5,
        thinkingRequests: 0,
        thinkingTokens: 0,
        lastAt: 0,
        readTokensPerSecond: null,
        writeTokensPerSecond: null,
        timeToFirstTokenMs: null,
        thinkingMs: null,
        totalMs: null,
        loadMs: null,
        sources: { tokens: '', read: '', write: '', ttft: '', thinking: '' },
        trend: [],
        ...fields
    }
}

describe('performance grouped by node', () => {
    it('puts each model under the node that served it, with live and loaded models', () => {
        const nodes = groupByNode({
            nodes: [
                { id: 'self', name: 'Desk' },
                { id: 'node-b', name: 'Lab' },
                { id: 'node-c', name: 'Attic' }
            ],
            selfId: 'self',
            summaries: [
                summary({ nodeId: 'node-b', model: 'small', requests: 2 }),
                summary({ nodeId: 'node-b', model: 'big', requests: 9 }),
                summary({ nodeId: 'self', model: 'small', requests: 1 })
            ],
            jobs: [
                {
                    nodeId: 'node-b',
                    model: 'big',
                    engine: 'ollama',
                    state: 'running',
                    createdAt: 100,
                    startedAt: 200
                },
                {
                    nodeId: 'node-b',
                    model: 'big',
                    engine: 'ollama',
                    state: 'queued',
                    createdAt: 150,
                    startedAt: null
                },
                {
                    nodeId: 'node-b',
                    model: 'old',
                    engine: 'ollama',
                    state: 'completed',
                    createdAt: 1,
                    startedAt: 2
                },
                {
                    nodeId: null,
                    model: 'unplaced',
                    engine: 'ollama',
                    state: 'queued',
                    createdAt: 1,
                    startedAt: null
                }
            ],
            loaded: [{ nodeId: 'node-c', engine: 'lm-studio', model: 'x' }]
        })

        expect(nodes.map(n => n.nodeId)).toEqual(['self', 'node-b', 'node-c'])
        const lab = nodes[1]
        expect(lab?.models.map(m => m.model)).toEqual(['big', 'small'])
        expect(lab?.requests).toBe(11)
        expect(lab?.live).toEqual([
            {
                engine: 'ollama',
                model: 'big',
                running: 1,
                queued: 1,
                since: 150,
                firstTokenMs: 100,
                lastStartedAt: 200
            }
        ])
        expect(nodes[0]?.local).toBe(true)
        expect(nodes[2]?.loaded).toEqual([{ engine: 'lmstudio', model: 'x' }])
        expect(nodes[2]?.models).toEqual([])
    })

    it('keeps a node seen only in the figures', () => {
        const nodes = groupByNode({
            nodes: [],
            selfId: null,
            summaries: [summary({ nodeId: 'gone', local: false })],
            jobs: [],
            loaded: []
        })
        expect(nodes).toHaveLength(1)
        expect(nodes[0]?.name).toBe('')
    })
})
