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
    Switch,
    Text
} from '@nvidia/foundations-react-core'
import getErrorString from '@/shared/utils/get-error-string'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import CapacityPlanner from '@/ui/components/SessionsSettings/CapacityPlanner'
import { planStatus } from '@/ui/components/SessionsSettings/plan-status'
import { useConnectionStore } from '@/ui/stores/connection.store'
import {
    findPlan,
    removePlanAndLimit,
    useCapacityPlannerStore
} from '@/ui/stores/capacity-planner.store'

/**
 * One model's context window and session count, sized from its node's
 * memory: the capacity planner in a dialog, opened from the model's menu. A
 * model on another node is planned, saved and applied on that node.
 */
export function CapacityPlannerDialog({
    nodeId,
    nodeName,
    engine,
    model,
    displayName,
    onClose
}: {
    /** The node the model runs on. */
    nodeId: string
    /** That node's name, shown when it is not this node. */
    nodeName: string
    /** The proxy's engine name (ollama, llamacpp, lmstudio). */
    engine: string
    model: string
    displayName: string
    onClose: () => void
}) {
    const selfId = useConnectionStore(s => s.selfId)
    // Another node's model goes to that node; this node's needs no ID.
    const remote = nodeId !== selfId ? nodeId : undefined
    const where = remote ? nodeName || 'that node' : 'this node'
    const ownPlanner = useCapacityPlannerStore(s => s.planner)
    const peerPlanner = useCapacityPlannerStore(s => (remote ? s.peerPlanner[remote] : undefined))
    const planner = remote ? (peerPlanner ?? null) : ownPlanner
    const setOwnPlanner = useCapacityPlannerStore(s => s.setPlanner)
    const setPeerPlanner = useCapacityPlannerStore(s => s.setPeerPlanner)
    const refreshOwn = useCapacityPlannerStore(s => s.refresh)
    const refreshPeer = useCapacityPlannerStore(s => s.refreshPeer)
    const setPlanner = (state: Parameters<typeof setOwnPlanner>[0]) =>
        remote ? setPeerPlanner(remote, state) : setOwnPlanner(state)
    const refresh = () => (remote ? refreshPeer(remote) : refreshOwn())
    const saved = findPlan(planner, engine, model)
    const [error, setError] = useState<string | null>(null)
    const [result, setResult] = useState<string | null>(null)
    const [busy, setBusy] = useState(false)
    const autoReload = planner?.autoReload ?? true

    const run = (work: () => Promise<void>, failure: string) => {
        setBusy(true)
        work()
            .catch(err => setError(getErrorString(err) ?? failure))
            .finally(() => setBusy(false))
    }
    const applyNow = () =>
        run(async () => {
            const applied = await window.pairApi.sessions.applyPlan(engine, model, remote)
            setPlanner(await window.pairApi.sessions.getPlanner(remote))
            setResult(applied.message)
        }, 'Could not apply the plan')
    // Removing a plan also lifts the limit it set, so the model goes back to
    // what its engine reports.
    const removePlan = () =>
        run(async () => {
            setPlanner(await removePlanAndLimit(engine, model, remote))
            void refresh()
            onClose()
        }, 'Could not remove the plan')
    const setAutoReload = (on: boolean) =>
        run(async () => {
            setPlanner(await window.pairApi.sessions.setPlanner({ nodeId: remote, autoReload: on }))
        }, 'Could not change the setting')

    return (
        <ModalRoot open onOpenChange={open => !open && onClose()}>
            <ModalDialog>
                <ModalContent
                    className="no-drag-elements"
                    style={{ width: 'min(880px, 94vw)', maxHeight: '88vh', overflowY: 'auto' }}
                >
                    <ModalHeading>
                        Context &amp; sessions · {displayName}
                        {remote ? ` on ${where}` : ''}
                    </ModalHeading>
                    <Stack gap="3" className="-mt-2">
                        {remote && !result && (
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                Planned for {where}’s GPUs and memory, and saved on {where}.
                                Applying restarts the engine on {where}. Reloading when idle is done
                                by PAIR on {where}, so it needs PAIR running there.
                            </Text>
                        )}
                        {error && (
                            <InlineErrorBanner message={error} onClose={() => setError(null)} />
                        )}
                        {saved && !result && (
                            <Flex justify="between" align="center" gap="2" wrap="wrap">
                                <Text kind="body/regular/sm">{planStatus(saved, autoReload)}</Text>
                                <Flex gap="2">
                                    {!saved.applied && (
                                        <Button
                                            size="small"
                                            kind="primary"
                                            color="brand"
                                            disabled={busy}
                                            onClick={applyNow}
                                        >
                                            Apply now
                                        </Button>
                                    )}
                                    <Button
                                        size="small"
                                        kind="secondary"
                                        disabled={busy}
                                        onClick={removePlan}
                                    >
                                        Remove plan
                                    </Button>
                                </Flex>
                            </Flex>
                        )}
                        {!result && (
                            <Flex align="center" gap="3">
                                <Switch
                                    checked={autoReload}
                                    size="small"
                                    disabled={busy}
                                    aria-label="Reload automatically when idle"
                                    onCheckedChange={checked => setAutoReload(checked === true)}
                                />
                                <Stack gap="0">
                                    <Text kind="body/semibold/sm">
                                        Reload automatically when idle
                                    </Text>
                                    <Text kind="body/regular/sm" className="text-subtle-color">
                                        For every model on {where}: apply a saved plan once the node
                                        has no sessions and no requests. When off, plans apply only
                                        with Save and apply now, or Apply now.
                                        {planner && !planner.activity.idle
                                            ? ` Busy now: ${planner.activity.sessions} sessions, ${planner.activity.activeRequests} requests.`
                                            : ''}
                                    </Text>
                                </Stack>
                            </Flex>
                        )}
                        {engine === 'ollama' && !result && (
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                Ollama uses one window and session count for every model it runs, so
                                this plan applies to all Ollama models on {where}.
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
                                nodeId={remote}
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
