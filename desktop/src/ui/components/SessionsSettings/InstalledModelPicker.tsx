// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react'
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import { EnabledEngineTypes } from '@/shared/constants/engines'
import type { EngineType } from '@/shared/types/engines'
import type { PlannerState } from '@/shared/types/sessions'
import { engineManagerName } from '@/shared/utils/engines'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import CapacityPlanner from './CapacityPlanner'
import { planStatus } from './plan-status'

const ENGINE_LABEL: Record<EngineType, string> = {
    ollama: 'Ollama',
    'lm-studio': 'LM Studio',
    'llama-cpp': 'llama.cpp'
}

interface Candidate {
    /** The proxy's engine name, as plans and capacity use it. */
    engine: string
    label: string
    model: string
    sizeBytes: number
}

/**
 * Plan a model that is installed on this node but not loaded, so it can be
 * sized before it is first used — the plan decides how it is loaded. Models
 * already listed under context capacity are left out; they carry their own
 * planner.
 */
export default function InstalledModelPicker({
    listed,
    planner,
    onPlanner,
    onError
}: {
    /** "engine/model" keys already shown as capacity rows. */
    listed: Set<string>
    planner: PlannerState | null
    onPlanner: (state: PlannerState | null, message?: string) => void
    onError: (message: string) => void
}) {
    const selfId = useConnectionStore(s => s.selfId)
    const models = useEngineModelsStore(s => s.models)
    const [open, setOpen] = useState(false)
    const [planning, setPlanning] = useState<string | null>(null)

    const candidates = useMemo(() => {
        const out: Candidate[] = []
        if (!selfId) return out
        for (const type of EnabledEngineTypes) {
            const engine = engineManagerName(type)
            for (const m of models.get(`${selfId}:${type}`)?.models ?? []) {
                if (!m.downloaded || listed.has(`${engine}/${m.name}`)) continue
                out.push({ engine, label: ENGINE_LABEL[type], model: m.name, sizeBytes: m.size })
            }
        }
        // Planned-but-unloaded models first, then by name.
        const planned = new Set(planner?.plans.map(p => `${p.engine}/${p.model}`))
        return out.sort((a, b) => {
            const pa = planned.has(`${a.engine}/${a.model}`) ? 0 : 1
            const pb = planned.has(`${b.engine}/${b.model}`) ? 0 : 1
            return pa - pb || a.model.localeCompare(b.model)
        })
    }, [selfId, models, listed, planner])

    if (candidates.length === 0) {
        return listed.size === 0 ? (
            <Text kind="body/regular/sm" className="text-subtle-color">
                Start an engine on this node to plan its installed models.
            </Text>
        ) : null
    }

    return (
        <Stack gap="2">
            <div>
                <Button size="small" kind="secondary" onClick={() => setOpen(o => !o)}>
                    {open
                        ? 'Hide installed models'
                        : `Plan an installed model (${candidates.length})`}
                </Button>
            </div>
            {open &&
                candidates.map(c => {
                    const key = `${c.engine}/${c.model}`
                    const saved = planner?.plans.find(
                        p => p.engine === c.engine && p.model === c.model
                    )
                    return (
                        <Stack key={key} gap="2" className="pair-paper p-3">
                            <Flex justify="between" align="center" gap="2" wrap="wrap">
                                <Text kind="body/semibold/sm">
                                    {c.model}{' '}
                                    <span className="text-subtle-color">
                                        · {c.label}
                                        {c.sizeBytes > 0
                                            ? ` · ${(c.sizeBytes / 1024 ** 3).toFixed(1)} GB`
                                            : ''}
                                    </span>
                                </Text>
                                {planning !== key && (
                                    <Button
                                        size="small"
                                        kind="secondary"
                                        onClick={() => setPlanning(key)}
                                    >
                                        Plan
                                    </Button>
                                )}
                            </Flex>
                            {saved && (
                                <Text kind="body/regular/sm">
                                    {planStatus(saved, planner?.autoReload ?? true)}
                                </Text>
                            )}
                            {planning === key && (
                                <CapacityPlanner
                                    engine={c.engine}
                                    model={c.model}
                                    saved={saved}
                                    onDone={(state, message) => {
                                        setPlanning(null)
                                        onPlanner(state, message)
                                    }}
                                    onError={onError}
                                />
                            )}
                        </Stack>
                    )
                })}
        </Stack>
    )
}
