// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { create } from 'zustand'
import type { ModelCapacity, PlannerState, SavedCapacityPlan } from '@/shared/types/sessions'

/**
 * This node's saved capacity plans and the capacity its proxy enforces, shared
 * by the model rows (which show each model's settings and plan) and Settings →
 * Sessions, plus each peer's capacity and planner, keyed by node ID, for the
 * peers' model rows and for planning a peer's models from here. The proxies
 * own all of it; this only caches their last answers.
 */
interface CapacityPlannerStore {
    planner: PlannerState | null
    capacity: ModelCapacity[] | null
    peerCapacity: Record<string, ModelCapacity[]>
    /**
     * Each peer's planner. A peer missing here cannot be planned from this
     * node: it runs an older PAIR Plus, or did not answer.
     */
    peerPlanner: Record<string, PlannerState>
    setPlanner: (state: PlannerState) => void
    setPeerPlanner: (nodeId: string, state: PlannerState) => void
    refresh: () => Promise<void>
    refreshPeer: (nodeId: string) => Promise<void>
}

/** How often open model rows re-read the capacity in force. */
export const CAPACITY_REFRESH_MS = 15_000

// One read per peer at a time, however many of its rows ask.
const peerReads = new Set<string>()

export const useCapacityPlannerStore = create<CapacityPlannerStore>(set => ({
    planner: null,
    capacity: null,
    peerCapacity: {},
    peerPlanner: {},
    setPlanner: planner => set({ planner }),
    setPeerPlanner: (nodeId, state) =>
        set(s => ({ peerPlanner: { ...s.peerPlanner, [nodeId]: state } })),
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
    },
    refreshPeer: async nodeId => {
        if (peerReads.has(nodeId)) return
        peerReads.add(nodeId)
        try {
            await Promise.all([
                window.pairApi.sessions
                    .getPeerCapacity(nodeId)
                    .then(report =>
                        set(s => ({ peerCapacity: { ...s.peerCapacity, [nodeId]: report } }))
                    )
                    .catch(() =>
                        // An unreachable peer shows no summary rather than old numbers.
                        set(s => {
                            const { [nodeId]: _gone, ...rest } = s.peerCapacity
                            return { peerCapacity: rest }
                        })
                    ),
                window.pairApi.sessions
                    .getPlanner(nodeId)
                    .then(planner =>
                        set(s => ({ peerPlanner: { ...s.peerPlanner, [nodeId]: planner } }))
                    )
                    .catch(() =>
                        // An older or unreachable peer cannot be planned from here.
                        set(s => {
                            const { [nodeId]: _gone, ...rest } = s.peerPlanner
                            return { peerPlanner: rest }
                        })
                    )
            ])
        } finally {
            peerReads.delete(nodeId)
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

/**
 * Removes a model's saved plan and lifts the limit the plan set, so the model
 * goes back to what its engine reports. Returns the planner state after. A
 * node ID other than this node's changes that node.
 */
export async function removePlanAndLimit(
    engine: string,
    model: string,
    nodeId?: string
): Promise<PlannerState> {
    const state = await window.pairApi.sessions.setPlanner({ nodeId, remove: { engine, model } })
    await window.pairApi.sessions.setCapacity({ nodeId, engine, model, clear: true })
    return state
}
