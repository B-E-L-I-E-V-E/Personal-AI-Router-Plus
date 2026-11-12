// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import type { LivePhase, LiveRequest, PerfRecord } from '@/shared/types/performance'
import getErrorString from '@/shared/utils/get-error-string'
import {
    engineLabel,
    formatCount,
    formatMs,
    formatRate
} from '@/ui/components/Performance/performance-format'

// The proxy updates a request's figures as each frame of its reply arrives;
// this panel polls them while it is open, a few times a second.
const LIVE_REFRESH_MS = 500
const RECENT_SHOWN = 8

const PHASE_LABELS: Record<LivePhase, string> = {
    waiting: 'Waiting for a node',
    reading: 'Reading the prompt',
    thinking: 'Thinking',
    answering: 'Answering'
}

const PHASE_COLORS: Record<LivePhase, 'gray' | 'yellow' | 'blue' | 'green'> = {
    waiting: 'gray',
    reading: 'yellow',
    thinking: 'blue',
    answering: 'green'
}

/** Whether a request belongs on this node's card. */
function onNode(
    r: { nodeId: string; local: boolean },
    nodeId: string,
    isLocal: boolean,
    waiting: boolean
): boolean {
    if (r.nodeId === nodeId) return true
    // A peer request served here before this node learned its own ID, and a
    // request still waiting for a node, show on this node's own card.
    return isLocal && (r.nodeId === '' ? r.local || waiting : false)
}

function Identity({
    model,
    engine,
    sessionId,
    agentId,
    origin,
    benchmark
}: {
    model: string
    engine: string
    sessionId: string | null
    agentId: string | null
    origin: 'routed' | 'peer'
    benchmark?: boolean
}) {
    return (
        <td>
            <span className="perf-figure">{model}</span>
            <span className="perf-figure-detail">
                {[
                    engineLabel(engine),
                    origin === 'peer' ? 'from another node' : 'from this machine',
                    benchmark ? 'benchmark' : ''
                ]
                    .filter(Boolean)
                    .join(' · ')}
            </span>
            {(sessionId || agentId) && (
                <span className="perf-figure-detail" title="Session and agent">
                    {sessionId ? `session ${sessionId}` : ''}
                    {sessionId && agentId ? ' · ' : ''}
                    {agentId ? `agent ${agentId}` : ''}
                </span>
            )}
        </td>
    )
}

function LiveRow({ r }: { r: LiveRequest }) {
    return (
        <tr>
            <Identity
                model={r.model}
                engine={r.engine}
                sessionId={r.sessionId}
                agentId={r.agentId}
                origin={r.origin}
                benchmark={r.benchmark}
            />
            <td>
                <Badge color={PHASE_COLORS[r.phase]} kind="solid">
                    {PHASE_LABELS[r.phase]}
                </Badge>
                <span className="perf-figure-detail">{formatMs(r.elapsedMs)}</span>
            </td>
            <td title="Estimated from the request's size until the engine reports its count">
                <span className="perf-figure">~{formatCount(r.promptTokensEstimate)}</span>
            </td>
            <td>
                <span className="perf-figure">{formatCount(r.outputTokens)}</span>
                {r.thinkingTokens > 0 && (
                    <span className="perf-figure-detail">
                        {formatCount(r.thinkingTokens)} thinking
                    </span>
                )}
            </td>
            <td>
                <span className="perf-figure">{formatMs(r.timeToFirstTokenMs)}</span>
            </td>
            <td>
                <span className="perf-figure">
                    {r.writeTokensPerSecond === null
                        ? '—'
                        : `${formatRate(r.writeTokensPerSecond)} tok/s`}
                </span>
            </td>
            <td>
                <span className="perf-figure">
                    {r.thinkingMs === null ? '—' : formatMs(r.thinkingMs)}
                </span>
            </td>
        </tr>
    )
}

function RecentRow({ r }: { r: PerfRecord }) {
    return (
        <tr>
            <Identity
                model={r.model}
                engine={r.engine}
                sessionId={r.sessionId}
                agentId={r.agentId}
                origin={r.origin}
            />
            <td>
                <span className="perf-figure">{new Date(r.at).toLocaleTimeString()}</span>
                <span className="perf-figure-detail">{formatMs(r.totalMs)} total</span>
            </td>
            <td>
                <span className="perf-figure">{formatCount(r.promptTokens)}</span>
                <span className="perf-figure-detail">
                    {r.readTokensPerSecond === null
                        ? ''
                        : `read ${formatRate(r.readTokensPerSecond)} tok/s`}
                </span>
            </td>
            <td>
                <span className="perf-figure">{formatCount(r.completionTokens)}</span>
                {r.thinkingTokens !== null && r.thinkingTokens > 0 && (
                    <span className="perf-figure-detail">
                        {formatCount(r.thinkingTokens)} thinking
                    </span>
                )}
            </td>
            <td>
                <span className="perf-figure">{formatMs(r.timeToFirstTokenMs)}</span>
            </td>
            <td>
                <span className="perf-figure">
                    {r.writeTokensPerSecond === null
                        ? '—'
                        : `${formatRate(r.writeTokensPerSecond)} tok/s`}
                </span>
            </td>
            <td>
                <span className="perf-figure">
                    {r.thinkingMs === null ? '—' : formatMs(r.thinkingMs)}
                </span>
            </td>
        </tr>
    )
}

function Header({ first, second }: { first: string; second: string }) {
    return (
        <thead>
            <tr>
                <th>Model · session</th>
                <th>{first}</th>
                <th>Tokens in</th>
                <th>Tokens out</th>
                <th>Time to first token</th>
                <th>{second}</th>
                <th>Thinking</th>
            </tr>
        </thead>
    )
}

/**
 * Real-time token telemetry for one node: every request in progress on it,
 * updated as its reply streams, and the latest finished ones. Shown under the
 * node's card when its stopwatch is on.
 */
export default function NodeRequestTelemetry({
    nodeId,
    isLocal,
    active
}: {
    nodeId: string
    isLocal: boolean
    active: boolean
}) {
    const [live, setLive] = useState<LiveRequest[]>([])
    const [recent, setRecent] = useState<PerfRecord[]>([])
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        if (!active) return
        let cancelled = false
        const refresh = () => {
            window.pairApi.performance
                .getLive()
                .then(rep => {
                    if (cancelled) return
                    setError(null)
                    setLive(rep.live.filter(r => onNode(r, nodeId, isLocal, r.phase === 'waiting')))
                    setRecent(
                        rep.recent
                            .filter(r => onNode(r, nodeId, isLocal, false))
                            .slice(0, RECENT_SHOWN)
                    )
                })
                .catch(err => {
                    if (!cancelled) setError(getErrorString(err) ?? 'The router is not reachable')
                })
        }
        refresh()
        const timer = setInterval(refresh, LIVE_REFRESH_MS)
        return () => {
            cancelled = true
            clearInterval(timer)
        }
    }, [active, nodeId, isLocal])

    return (
        <Stack gap="3" className="perf-node-live">
            <Flex align="center" justify="between" gap="3" wrap="wrap">
                <Text kind="body/semibold/sm">Requests in real time</Text>
                <Text kind="body/regular/xs" className="text-subtle-color">
                    {isLocal
                        ? 'Every request on this node: from this machine and from other nodes.'
                        : 'Requests this machine sends to this node. Requests from other nodes show in their own app.'}
                </Text>
            </Flex>
            {error && (
                <Text kind="body/regular/sm" className="text-feedback-danger">
                    {error}
                </Text>
            )}
            <div className="perf-table-wrap">
                <table className="perf-table perf-model-table">
                    <Header first="Now" second="Writing so far" />
                    <tbody>
                        {live.length === 0 ? (
                            <tr>
                                <td colSpan={7} className="text-subtle-color">
                                    No request is running on this node.
                                </td>
                            </tr>
                        ) : (
                            live.map(r => <LiveRow key={r.requestId} r={r} />)
                        )}
                    </tbody>
                </table>
            </div>
            {recent.length > 0 && (
                <>
                    <Text kind="body/semibold/sm">Just finished</Text>
                    <div className="perf-table-wrap">
                        <table className="perf-table perf-model-table">
                            <Header first="Finished" second="Writing" />
                            <tbody>
                                {recent.map(r => (
                                    <RecentRow key={r.requestId} r={r} />
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            )}
            <Text kind="body/regular/xs" className="text-subtle-color">
                Tokens out and thinking count streamed tokens as they arrive; tokens in is estimated
                from the request’s size until the engine reports it at the end. Only counts, timings
                and IDs are shown — never the text.
            </Text>
        </Stack>
    )
}
