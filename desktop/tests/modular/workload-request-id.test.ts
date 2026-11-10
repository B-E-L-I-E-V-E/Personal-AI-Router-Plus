// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))

import { parseWorkloadsInitial } from '@/electron/service-bridge/modular-state'

// The proxy stamps each workload with the caller-facing X-Request-ID so a job
// card can be matched to the client's request. The field is additive, so a
// workload from an older producer must still parse without it.
describe('workload requestId', () => {
    const baseRow = {
        id: 'job-1',
        engine: 'ollama',
        state: 'running',
        model: 'llama3.1',
        originatedFrom: 'uuid-node-a',
        createdAt: 100
    }

    it('keeps the request id the proxy stamped', () => {
        const [workload] = parseWorkloadsInitial({
            workloads: [{ ...baseRow, requestId: 'agent-7:turn-12' }]
        })
        expect(workload?.requestId).toBe('agent-7:turn-12')
    })

    it('omits the request id when the producer sent none', () => {
        const [workload] = parseWorkloadsInitial({ workloads: [baseRow] })
        expect(workload).toBeDefined()
        expect(workload).not.toHaveProperty('requestId')
    })
})
