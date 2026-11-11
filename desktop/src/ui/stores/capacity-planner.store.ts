// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { create } from 'zustand'
import type { PlannerState, SavedCapacityPlan } from '@/shared/types/sessions'

/**
 * This node's saved capacity plans, shared by the model rows (which show each
 * model's plan) and Settings → Sessions. The proxy owns the plans; this only
 * caches its last answer.
 */
interface CapacityPlannerStore {
    planner: PlannerState | null
    setPlanner: (state: PlannerState) => void
    refresh: () => Promise<void>
}

export const useCapacityPlannerStore = create<CapacityPlannerStore>(set => ({
    planner: null,
    setPlanner: planner => set({ planner }),
    refresh: async () => {
        try {
            set({ planner: await window.pairApi.sessions.getPlanner() })
        } catch {
            // The router is not reachable yet; rows simply show no plan.
        }
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
