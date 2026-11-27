// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SavedCapacityPlan } from '@/shared/types/sessions'

function tokens(n: number): string {
    return n > 0 ? n.toLocaleString() : '—'
}

/**
 * Whether the engine runs a model with other settings than its plan. Only LM
 * Studio can differ after an apply: it fixes the context when a model loads,
 * and a model loaded or reloaded by hand or on demand keeps whatever it got.
 * `loadedWindow` is the context the engine reports for the loaded model; zero
 * means it reports none, as for a model that is not loaded.
 */
export function loadDiffersFromPlan(plan: SavedCapacityPlan, loadedWindow: number): boolean {
    return plan.engine === 'lmstudio' && loadedWindow !== plan.windowTokens
}

/** Whether pressing Apply would change anything. */
export function planNeedsApply(plan: SavedCapacityPlan, loadedWindow: number): boolean {
    return plan.engine === 'lmstudio' ? loadDiffersFromPlan(plan, loadedWindow) : !plan.applied
}

/** What a saved plan's state means, in one line. */
export function planStatus(
    plan: SavedCapacityPlan,
    autoReload: boolean,
    loadedWindow: number
): string {
    const reply = Math.min(plan.replyTokens || 8192, Math.floor(plan.windowTokens / 2))
    const what = `${plan.sessions} × ${tokens(plan.windowTokens)} tokens, ${tokens(reply)} for replies${
        plan.cacheType === 'f16' ? '' : ` (${plan.cacheType === 'q8_0' ? '8-bit' : '4-bit'} cache)`
    }`
    if (plan.engine === 'lmstudio' && loadDiffersFromPlan(plan, loadedWindow)) {
        if (loadedWindow <= 0)
            return `Plan: ${what} — not applied: the model is not loaded. Apply now loads it with these settings`
        return autoReload
            ? `Plan: ${what} — not applied: loaded with ${tokens(loadedWindow)} tokens. Reloads when this node is idle`
            : `Plan: ${what} — not applied: loaded with ${tokens(loadedWindow)} tokens. Press Apply now to reload`
    }
    // For LM Studio the loaded model is the truth, not the saved flag.
    if (plan.applied || plan.engine === 'lmstudio') return `Plan: ${what} — applied`
    return autoReload
        ? `Plan: ${what} — will apply when this node is idle`
        : `Plan: ${what} — not applied; press Apply`
}
