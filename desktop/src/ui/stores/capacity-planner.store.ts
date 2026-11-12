// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { create } from 'zustand'
import type { ModelCapacity, PlannerState, SavedCapacityPlan } from '@/shared/types/sessions'

/**
 * This node's saved capacity plans and the capacity its proxy enforces, shared
 * by the model rows (which show each model's settings and plan) and Settings →
 * Sessions. The proxy owns both; this only caches its last answers.
 */
interface CapacityPlannerStore {
    planner: PlannerState | null
    capacity: ModelCapacity[] | null
    setPlanner: (state: PlannerState) => void
    refresh: () => Promise<void>
}

export const useCapacityPlannerStore = create<CapacityPlannerStore>(set => ({
    planner: null,
    capacity: null,
    setPlanner: planner => set({ planner }),
    refresh: async () => {
        // Each answer stands alone: the router not being reachable yet leaves
        // the last one in place, and rows simply show what is known.
        await Promise.all([
            window.pairApi.sessions
                .getPlanner()
                .then(planner => set({ planner }))
                .catch(() => undefined),
            window.pairApi.sessions
                .getCapacity()
                .then(capacity => set({ capacity }))
                .catch(() => undefined)
        ])
    }
}))

/** The saved plan for one model, if any. */
export function findPlan(
    planner: PlannerState | null,
    engine: string,
    model: string
): SavedCapacityPlan | undefined {
    return planner?.plans.find(p => p.engine === engine && p.model === model)
}

/**
 * Removes a model's saved plan and lifts the limit the plan set, so the model
 * goes back to what its engine reports. Returns the planner state after.
 */
export async function removePlanAndLimit(engine: string, model: string): Promise<PlannerState> {
    const state = await window.pairApi.sessions.setPlanner({ remove: { engine, model } })
    await window.pairApi.sessions.setCapacity({ engine, model, clear: true })
    return state
}
