// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelCapacity, SavedCapacityPlan } from '@/shared/types/sessions'
import { planStatus } from '@/ui/components/SessionsSettings/plan-status'

/** The proxy's capacity row for one model, if it reports one. */
export function findCapacity(
    report: ModelCapacity[] | null,
    engine: string,
    model: string
): ModelCapacity | undefined {
    return report?.find(c => c.engine === engine && c.model === model)
}

function sessionsText(c: ModelCapacity): string {
    const count = `${c.maxSessions} ${c.maxSessions === 1 ? 'session' : 'sessions'}`
    const live: string[] = []
    if (c.heldSlots > 0) live.push(`${c.heldSlots} in use`)
    if (c.queued > 0) live.push(`${c.queued} waiting`)
    return live.length > 0 ? `${count} (${live.join(', ')})` : count
}

/**
 * What the proxy enforces for a model, in one line:
 * "4 sessions (1 in use) · 40,960 per session · 8,192 response · 32,768 post".
 * Only values the proxy reports appear; null when it knows none.
 */
export function capacitySummary(c: ModelCapacity | undefined): string | null {
    if (!c || c.source === 'unknown' || c.sessionContextTokens <= 0) return null
    const parts: string[] = []
    if (c.maxSessions > 0) parts.push(sessionsText(c))
    parts.push(`${c.sessionContextTokens.toLocaleString()} per session`)
    if (c.replyTokens > 0 && c.messageTokens > 0) {
        parts.push(`${c.replyTokens.toLocaleString()} response`)
        parts.push(`${c.messageTokens.toLocaleString()} post`)
    }
    if (c.source === 'detected') parts.push('as the engine reports')
    return parts.join(' · ')
}

interface ModelConfigLines {
    /** What is in force now, from the capacity report. */
    summary?: string
    /** The saved plan, when it is not what is in force yet. */
    planLine?: string
}

/**
 * The lines under a model row. An applied plan is what the summary shows, so
 * it needs no line of its own; a pending plan keeps its "Plan: …" line and the
 * summary is marked "Now:" so planned and applied numbers read apart.
 */
export function modelConfigLines(
    capacity: ModelCapacity | undefined,
    plan: SavedCapacityPlan | undefined,
    autoReload: boolean
): ModelConfigLines {
    const summary = capacitySummary(capacity)
    if (!plan) return summary ? { summary } : {}
    const planLine = planStatus(plan, autoReload)
    if (!summary) return { planLine }
    if (plan.applied && capacity?.source === 'configured') return { summary }
    return { summary: `Now: ${summary}`, planLine }
}
