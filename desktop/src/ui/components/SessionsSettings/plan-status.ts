// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SavedCapacityPlan } from '@/shared/types/sessions'

function tokens(n: number): string {
    return n > 0 ? n.toLocaleString() : '—'
}

/** What a saved plan's state means, in one line. */
export function planStatus(plan: SavedCapacityPlan, autoReload: boolean): string {
    const what = `${plan.sessions} × ${tokens(plan.windowTokens)} tokens${
        plan.cacheType === 'f16' ? '' : ` (${plan.cacheType === 'q8_0' ? '8-bit' : '4-bit'} cache)`
    }`
    if (plan.applied) return `Plan: ${what} — applied`
    if (plan.engine === 'lmstudio') return `Plan: ${what} — set LM Studio’s load settings to match`
    return autoReload
        ? `Plan: ${what} — will apply when this node is idle`
        : `Plan: ${what} — not applied; press Apply`
}
