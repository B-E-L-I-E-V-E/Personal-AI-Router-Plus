// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react'
import {
    Button,
    Flex,
    ModalContent,
    ModalDialog,
    ModalHeading,
    ModalRoot,
    Stack,
    Text
} from '@nvidia/foundations-react-core'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import CapacityPlanner from '@/ui/components/SessionsSettings/CapacityPlanner'
import { planStatus } from '@/ui/components/SessionsSettings/plan-status'
import { findPlan, useCapacityPlannerStore } from '@/ui/stores/capacity-planner.store'

/**
 * One model's context window and session count, sized from this node's
 * memory: the capacity planner in a dialog, opened from the model's menu.
 */
export function CapacityPlannerDialog({
    engine,
    model,
    displayName,
    onClose
}: {
    /** The proxy's engine name (ollama, llamacpp, lmstudio). */
    engine: string
    model: string
    displayName: string
    onClose: () => void
}) {
    const planner = useCapacityPlannerStore(s => s.planner)
    const setPlanner = useCapacityPlannerStore(s => s.setPlanner)
    const refresh = useCapacityPlannerStore(s => s.refresh)
    const saved = findPlan(planner, engine, model)
    const [error, setError] = useState<string | null>(null)
    const [result, setResult] = useState<string | null>(null)

    return (
        <ModalRoot open onOpenChange={open => !open && onClose()}>
            <ModalDialog>
                <ModalContent
                    className="no-drag-elements"
                    style={{ width: 'min(880px, 94vw)', maxHeight: '88vh', overflowY: 'auto' }}
                >
                    <ModalHeading>Context &amp; sessions · {displayName}</ModalHeading>
                    <Stack gap="3" className="-mt-2">
                        {error && (
                            <InlineErrorBanner message={error} onClose={() => setError(null)} />
                        )}
                        {saved && !result && (
                            <Text kind="body/regular/sm">
                                {planStatus(saved, planner?.autoReload ?? true)}
                            </Text>
                        )}
                        {engine === 'ollama' && !result && (
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                Ollama uses one window and session count for every model it runs, so
                                this plan applies to all Ollama models on this node.
                            </Text>
                        )}
                        {result ? (
                            <Stack gap="3">
                                <Text kind="body/regular/sm">{result}</Text>
                                <Flex justify="end">
                                    <Button
                                        size="small"
                                        kind="primary"
                                        color="brand"
                                        onClick={onClose}
                                    >
                                        Done
                                    </Button>
                                </Flex>
                            </Stack>
                        ) : (
                            <CapacityPlanner
                                engine={engine}
                                model={model}
                                saved={saved}
                                onDone={(state, message) => {
                                    if (state) setPlanner(state)
                                    void refresh()
                                    if (message) setResult(message)
                                    else onClose()
                                }}
                                onError={setError}
                            />
                        )}
                    </Stack>
                </ModalContent>
            </ModalDialog>
        </ModalRoot>
    )
}
