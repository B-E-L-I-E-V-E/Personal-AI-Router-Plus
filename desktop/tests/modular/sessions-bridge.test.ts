// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'

const callProxy = vi.fn()

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => ({ callProxy })
}))

import {
    endSession,
    getCapacityReport,
    getSessionPolicy,
    listSessions,
    setCapacity,
    setSessionPolicy
} from '@/electron/service-bridge/sessions-bridge'

beforeEach(() => {
    callProxy.mockReset()
})

describe('sessions bridge', () => {
    it('uses the next engine when one engine proxy is not running', async () => {
        callProxy
            .mockRejectedValueOnce(new Error('ollama-proxy not available'))
            .mockResolvedValueOnce({ idleTimeoutSeconds: 600, queueTimeoutSeconds: 60 })
        const policy = await getSessionPolicy()
        expect(policy.idleTimeoutSeconds).toBe(600)
        expect(callProxy.mock.calls.map(c => c[0])).toEqual(['ollama', 'lm-studio'])
        expect(callProxy.mock.calls[1]?.[1]).toBe('sessions/configure')
    })

    it('reports a refusal from the proxy instead of trying another engine', async () => {
        callProxy.mockRejectedValueOnce(
            new Error('invalid params: idleTimeoutSeconds must be between 60 and 3600')
        )
        await expect(setSessionPolicy({ idleTimeoutSeconds: 5 })).rejects.toThrow(
            'idleTimeoutSeconds'
        )
        expect(callProxy).toHaveBeenCalledTimes(1)
    })

    it('sends only the policy fields that changed', async () => {
        callProxy.mockResolvedValueOnce({ idleTimeoutSeconds: 300 })
        await setSessionPolicy({ idleTimeoutSeconds: 300 })
        expect(callProxy.mock.calls[0]?.[2]).toEqual({ idleTimeoutSeconds: 300 })
    })

    it('parses sessions and their context use', async () => {
        callProxy.mockResolvedValueOnce({
            sessions: [
                {
                    sessionId: 'conv-1',
                    agentId: 'planner',
                    expiresAt: '2026-11-11T00:04:03Z',
                    inFlight: 0,
                    requests: 2,
                    context: {
                        ollama: { model: 'qwen3:0.6b', tokens: 39, postRoomTokens: 6105 },
                        llamacpp: { model: 'other', tokens: 5 }
                    }
                },
                { sessionId: '', agentId: null }
            ]
        })
        const sessions = await listSessions()
        expect(sessions).toHaveLength(1)
        expect(sessions[0]).toMatchObject({
            sessionId: 'conv-1',
            agentId: 'planner',
            requests: 2,
            context: expect.arrayContaining([
                { engine: 'ollama', model: 'qwen3:0.6b', tokens: 39, postRoomTokens: 6105 },
                // Served by another node: its room is unknown here.
                { engine: 'llamacpp', model: 'other', tokens: 5, postRoomTokens: null }
            ])
        })
    })

    it('parses capacity rows, configured or detected', async () => {
        callProxy.mockResolvedValueOnce({
            models: [
                {
                    engine: 'ollama',
                    model: 'qwen3:0.6b',
                    configured: { contextTokens: 262144, maxSessions: 6 },
                    detected: { windowTokens: 40960 },
                    contextTokens: 262144,
                    maxSessions: 6,
                    sessionContextTokens: 43008,
                    source: 'configured',
                    heldSlots: 2,
                    queued: 1,
                    usedContextTokens: 90000
                },
                {
                    engine: 'lmstudio',
                    model: 'qwen/qwen3-8b',
                    configured: null,
                    detected: null,
                    source: 'unknown'
                }
            ]
        })
        const rows = await getCapacityReport()
        expect(rows[0]).toMatchObject({
            configured: { contextTokens: 262144, maxSessions: 6 },
            detectedWindowTokens: 40960,
            sessionContextTokens: 43008,
            heldSlots: 2,
            queued: 1
        })
        expect(rows[1]).toMatchObject({
            configured: null,
            detectedWindowTokens: 0,
            source: 'unknown'
        })
    })

    it('passes capacity changes and session ends through', async () => {
        callProxy.mockResolvedValue({})
        await setCapacity({ engine: 'ollama', model: 'm', contextTokens: 65536, maxSessions: 4 })
        await setCapacity({ engine: 'ollama', model: 'm', clear: true })
        await endSession({ sessionId: 'conv-1', agentId: null })
        expect(callProxy.mock.calls.map(c => [c[1], c[2]])).toEqual([
            [
                'capacity/set',
                { engine: 'ollama', model: 'm', contextTokens: 65536, maxSessions: 4 }
            ],
            ['capacity/set', { engine: 'ollama', model: 'm', clear: true }],
            ['sessions/end', { sessionId: 'conv-1', agentId: '' }]
        ])
    })

    it('carries the reply budget both ways', async () => {
        callProxy.mockResolvedValueOnce({
            models: [
                {
                    engine: 'ollama',
                    model: 'm',
                    configured: { contextTokens: 16384, maxSessions: 2, replyTokens: 2048 },
                    sessionContextTokens: 8192,
                    replyTokens: 2048,
                    messageTokens: 6144,
                    source: 'configured'
                }
            ]
        })
        const [row] = await getCapacityReport()
        expect(row).toMatchObject({
            configured: { contextTokens: 16384, maxSessions: 2, replyTokens: 2048 },
            replyTokens: 2048,
            messageTokens: 6144
        })

        callProxy.mockResolvedValue({})
        await setCapacity({
            engine: 'ollama',
            model: 'm',
            contextTokens: 16384,
            maxSessions: 2,
            replyTokens: 2048
        })
        expect(callProxy.mock.calls.at(-1)?.[2]).toEqual({
            engine: 'ollama',
            model: 'm',
            contextTokens: 16384,
            maxSessions: 2,
            replyTokens: 2048
        })
    })

    it('reads a missing reply budget as none', async () => {
        callProxy.mockResolvedValueOnce({
            models: [{ engine: 'ollama', model: 'm', configured: null, source: 'detected' }]
        })
        const [row] = await getCapacityReport()
        expect(row.replyTokens).toBe(0)
        expect(row.messageTokens).toBe(0)
    })
})
