// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react'
import { Button, Text } from '@nvidia/foundations-react-core'
import { DismissibleTooltip } from '@/ui/components/DismissibleTooltip/DismissibleTooltip'
import { ConfirmModal } from '@/ui/components/ConfirmModal'
import { Pause, Play, Square } from '@/ui/components/icons'
import type { NodeServingState } from '@/shared/types/node-state'
import { nodeServingState, useNodeStateStore } from '@/ui/stores/node-state.store'
import { servingActions } from '@/ui/utils/node-serving-actions'

const STATE_LABEL: Record<NodeServingState, string> = {
    running: 'Running',
    paused: 'Paused',
    stopped: 'Stopped'
}

function useServingState(nodeId: string, isSelf: boolean) {
    return useNodeStateStore(s => nodeServingState(s, nodeId, isSelf))
}

/** Why the last pause, stop or resume of a node failed, if it did. */
export function NodeStateError({ nodeId, isSelf }: { nodeId: string; isSelf: boolean }) {
    const error = useNodeStateStore(s => s.errors[isSelf ? 'self' : nodeId])
    if (!error) return null
    return (
        <Text kind="body/regular/xs" style={{ color: 'var(--color-danger, #e5484d)' }}>
            {error}
        </Text>
    )
}

/** Pill on a node card naming a paused or stopped node; nothing while running. */
export function NodeStateBadge({ nodeId, isSelf }: { nodeId: string; isSelf: boolean }) {
    const state = useServingState(nodeId, isSelf)
    if (state === 'running') return null
    const color =
        state === 'stopped' ? 'var(--color-danger, #e5484d)' : 'var(--color-warning, #f5a524)'
    return (
        <Text
            kind="body/regular/xs"
            className="shrink-0 rounded-full px-1.5 py-0.5 uppercase"
            data-node-serving-state={state}
            style={{
                color,
                backgroundColor: `color-mix(in srgb, ${color} 18%, transparent)`,
                letterSpacing: '0.04em',
                lineHeight: 1
            }}
        >
            {STATE_LABEL[state]}
        </Text>
    )
}

const ACTION: Record<NodeServingState, { label: string; tooltip: string }> = {
    running: { label: 'Resume', tooltip: 'Resume: take requests again' },
    paused: {
        label: 'Pause',
        tooltip: 'Pause: finish what is running, take no new requests'
    },
    stopped: { label: 'Stop', tooltip: 'Stop: cancel every running request, take none' }
}

/**
 * Pause, Resume and Stop for one node. Any member of the cluster may manage
 * any node; the proxy reaches the others over the cluster's mTLS ingress.
 */
export function NodeServingControls({
    nodeId,
    nodeName,
    isSelf,
    placement
}: {
    nodeId: string
    nodeName: string
    isSelf: boolean
    placement: 'left' | 'top'
}) {
    const state = useServingState(nodeId, isSelf)
    const busy = useNodeStateStore(s => s.pending[isSelf ? 'self' : nodeId] === true)
    const change = useNodeStateStore(s => s.change)
    const [confirmStop, setConfirmStop] = useState(false)
    // change() records its own failure in the store, shown under the card's name.
    const run = (next: NodeServingState) => void change(nodeId, isSelf, next)

    return (
        <>
            {servingActions(state).map(next => {
                const Icon = next === 'running' ? Play : next === 'paused' ? Pause : Square
                return (
                    <DismissibleTooltip
                        key={next}
                        slotContent={ACTION[next].tooltip}
                        placement={placement}
                    >
                        <Button
                            kind="tertiary"
                            color={next === 'stopped' ? 'danger' : 'neutral'}
                            size="tiny"
                            className="px-2"
                            disabled={busy}
                            aria-label={`${ACTION[next].label} ${nodeName}`}
                            onClick={() => (next === 'stopped' ? setConfirmStop(true) : run(next))}
                        >
                            <Icon style={{ fontSize: 14 }} />
                        </Button>
                    </DismissibleTooltip>
                )
            })}
            <ConfirmModal
                open={confirmStop}
                onOpenChange={setConfirmStop}
                title={`Stop ${nodeName}?`}
                message="Every request this node is running is cancelled now, and it takes no new ones until you resume it. Requests go to other nodes that can serve them."
                confirmLabel="Stop node"
                confirmColor="danger"
                onConfirm={() => run('stopped')}
            />
        </>
    )
}
