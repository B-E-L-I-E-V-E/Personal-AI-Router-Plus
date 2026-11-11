// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
    Button,
    Flex,
    FormField,
    Stack,
    Switch,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import type {
    ModelCapacity,
    PlannerState,
    SavedCapacityPlan,
    SessionPolicy,
    SessionSummary
} from '@/shared/types/sessions'
import getErrorString from '@/shared/utils/get-error-string'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import CapacityPlanner from './CapacityPlanner'
import InstalledModelPicker from './InstalledModelPicker'
import { planStatus } from './plan-status'

// The proxy owns these settings and their saved state; this panel only reads
// what it reports and sends changes. It polls while open because sessions and
// slots change with every request and the proxy pushes no events for them.
const REFRESH_MS = 5000
const SHARE_GRANULARITY = 1024
const MIN_USEFUL_SHARE = 4096

// The proxy stores a model's capacity as window × sessions; the page shows
// the window each session gets, since a window belongs to one conversation.
function sessionWindow(pool: number, sessions: number): number {
    if (pool <= 0 || sessions <= 0) return 0
    const share = Math.floor(pool / sessions)
    return share - (share % SHARE_GRANULARITY)
}

function formatTokens(n: number): string {
    return n > 0 ? n.toLocaleString() : '—'
}

function wholeNumber(text: string): number | null {
    const trimmed = text.trim()
    if (!/^\d+$/.test(trimmed)) return null
    return Number(trimmed)
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
    return (
        <div className="settings-card settings-card-stacked pair-paper p-4">
            <Stack gap="3">
                <Text kind="body/semibold/md">{title}</Text>
                {children}
            </Stack>
        </div>
    )
}

function PolicyCard({
    policy,
    onSaved,
    onError
}: {
    policy: SessionPolicy
    onSaved: (p: SessionPolicy) => void
    onError: (message: string) => void
}) {
    const [idleMinutes, setIdleMinutes] = useState(String(policy.idleTimeoutSeconds / 60))
    const [queueSeconds, setQueueSeconds] = useState(String(policy.queueTimeoutSeconds))
    const [saving, setSaving] = useState(false)

    const idle = wholeNumber(idleMinutes)
    const queue = wholeNumber(queueSeconds)
    const valid = idle !== null && idle >= 1 && idle <= 60 && queue !== null && queue <= 300
    const changed =
        valid && (idle * 60 !== policy.idleTimeoutSeconds || queue !== policy.queueTimeoutSeconds)

    const save = () => {
        if (!valid) return
        setSaving(true)
        window.pairApi.sessions
            .setPolicy({ idleTimeoutSeconds: idle * 60, queueTimeoutSeconds: queue })
            .then(onSaved)
            .catch(err => onError(getErrorString(err) ?? 'Could not save the session policy'))
            .finally(() => setSaving(false))
    }

    return (
        <Card title="Session policy">
            <Text kind="body/regular/sm" className="text-subtle-color">
                A session ends after it has been idle this long. Any request, a keepalive, or
                waiting on the model keeps it open.
            </Text>
            <Flex align="end" gap="3" wrap="wrap">
                <FormField slotLabel="Idle timeout (minutes, 1–60)">
                    <TextInput value={idleMinutes} onValueChange={setIdleMinutes} />
                </FormField>
                <FormField slotLabel="Wait for a slot (seconds, 0–300)">
                    <TextInput value={queueSeconds} onValueChange={setQueueSeconds} />
                </FormField>
                <Button kind="primary" color="brand" onClick={save} disabled={!changed || saving}>
                    Save
                </Button>
            </Flex>
            {!valid && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Enter whole numbers within the ranges shown.
                </Text>
            )}
        </Card>
    )
}

function CapacityRow({
    row,
    saved,
    autoReload,
    onChanged,
    onPlanner,
    onError
}: {
    row: ModelCapacity
    saved: SavedCapacityPlan | undefined
    autoReload: boolean
    onChanged: () => void
    onPlanner: (state: PlannerState | null, message?: string) => void
    onError: (message: string) => void
}) {
    const [planning, setPlanning] = useState(false)
    const initialPool = row.configured?.contextTokens ?? row.contextTokens
    const initialSessions = row.configured?.maxSessions ?? row.maxSessions
    const initialWindow = sessionWindow(initialPool, initialSessions)
    const [windowText, setWindowText] = useState(initialWindow > 0 ? String(initialWindow) : '')
    const [sessions, setSessions] = useState(initialSessions > 0 ? String(initialSessions) : '')
    const [saving, setSaving] = useState(false)

    const windowN = wholeNumber(windowText) ?? 0
    const sessionsN = wholeNumber(sessions) ?? 0
    const valid =
        windowN >= SHARE_GRANULARITY &&
        windowN % SHARE_GRANULARITY === 0 &&
        sessionsN >= 1 &&
        sessionsN <= 1024

    const run = (change: Promise<null>) => {
        setSaving(true)
        change
            .then(onChanged)
            .catch(err => onError(getErrorString(err) ?? 'Could not change the capacity'))
            .finally(() => setSaving(false))
    }

    return (
        <Stack gap="2" className="pair-paper p-3">
            <Flex justify="between" align="center" gap="2" wrap="wrap">
                <Text kind="body/semibold/sm">
                    {row.model} <span className="text-subtle-color">· {row.engine}</span>
                </Text>
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {row.heldSlots} of {row.maxSessions || '—'} sessions · {row.queued} waiting ·{' '}
                    {formatTokens(row.usedContextTokens)} tokens in use
                </Text>
            </Flex>
            <Flex align="end" gap="3" wrap="wrap">
                <FormField slotLabel="Context window per session (tokens)">
                    <TextInput value={windowText} onValueChange={setWindowText} />
                </FormField>
                <FormField slotLabel="Sessions at once">
                    <TextInput value={sessions} onValueChange={setSessions} />
                </FormField>
                <Stack gap="1">
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        Context held in memory
                    </Text>
                    <Text kind="body/semibold/sm">
                        {formatTokens(valid ? windowN * sessionsN : 0)} tokens
                    </Text>
                </Stack>
                <Button
                    kind="primary"
                    color="brand"
                    disabled={!valid || saving}
                    onClick={() =>
                        run(
                            window.pairApi.sessions.setCapacity({
                                engine: row.engine,
                                model: row.model,
                                contextTokens: windowN * sessionsN,
                                maxSessions: sessionsN
                            })
                        )
                    }
                >
                    Save
                </Button>
                {row.configured && (
                    <Button
                        kind="secondary"
                        disabled={saving}
                        onClick={() =>
                            run(
                                window.pairApi.sessions.setCapacity({
                                    engine: row.engine,
                                    model: row.model,
                                    clear: true
                                })
                            )
                        }
                    >
                        Clear
                    </Button>
                )}
            </Flex>
            {row.detectedWindowTokens > 0 && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    The engine reports a {formatTokens(row.detectedWindowTokens)}-token window per
                    request.
                </Text>
            )}
            {valid && windowN < MIN_USEFUL_SHARE && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Below {formatTokens(MIN_USEFUL_SHARE)} tokens per session is too little to be
                    useful.
                </Text>
            )}
            {row.warning && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {row.warning}
                </Text>
            )}
            {saved && (
                <Flex justify="between" align="center" gap="2" wrap="wrap">
                    <Text kind="body/regular/sm">{planStatus(saved, autoReload)}</Text>
                    <Flex gap="2">
                        {!saved.applied && (
                            <Button
                                size="small"
                                kind="primary"
                                color="brand"
                                disabled={saving}
                                onClick={() => {
                                    setSaving(true)
                                    window.pairApi.sessions
                                        .applyPlan(row.engine, row.model)
                                        .then(async result =>
                                            onPlanner(
                                                await window.pairApi.sessions.getPlanner(),
                                                result.message
                                            )
                                        )
                                        .catch(err =>
                                            onError(
                                                getErrorString(err) ?? 'Could not apply the plan'
                                            )
                                        )
                                        .finally(() => setSaving(false))
                                }}
                            >
                                Apply
                            </Button>
                        )}
                        <Button
                            size="small"
                            kind="secondary"
                            disabled={saving}
                            onClick={() =>
                                window.pairApi.sessions
                                    .setPlanner({
                                        remove: { engine: row.engine, model: row.model }
                                    })
                                    .then(state => onPlanner(state))
                                    .catch(err =>
                                        onError(getErrorString(err) ?? 'Could not remove the plan')
                                    )
                            }
                        >
                            Remove plan
                        </Button>
                    </Flex>
                </Flex>
            )}
            {planning ? (
                <CapacityPlanner
                    engine={row.engine}
                    model={row.model}
                    saved={saved}
                    onDone={(state, message) => {
                        setPlanning(false)
                        onPlanner(state, message)
                    }}
                    onError={onError}
                />
            ) : (
                <div>
                    <Button size="small" kind="secondary" onClick={() => setPlanning(true)}>
                        Plan from GPU memory
                    </Button>
                </div>
            )}
        </Stack>
    )
}

function SessionRow({
    session,
    onEnded,
    onError
}: {
    session: SessionSummary
    onEnded: () => void
    onError: (message: string) => void
}) {
    const [ending, setEnding] = useState(false)
    const tokens = session.context.reduce((sum, c) => sum + c.tokens, 0)
    const expires = new Date(session.expiresAt)
    return (
        <Flex justify="between" align="center" gap="3" wrap="wrap" className="pair-paper p-3">
            <Stack gap="1">
                <Text kind="body/semibold/sm">{session.sessionId}</Text>
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {session.agentId ? `Agent ${session.agentId} · ` : ''}
                    {session.requests} requests · {formatTokens(tokens)} tokens
                    {session.inFlight > 0
                        ? ' · running'
                        : Number.isNaN(expires.getTime())
                          ? ''
                          : ` · expires ${expires.toLocaleTimeString()}`}
                </Text>
            </Stack>
            <Button
                kind="secondary"
                disabled={ending}
                onClick={() => {
                    setEnding(true)
                    window.pairApi.sessions
                        .end(session.sessionId, session.agentId)
                        .then(onEnded)
                        .catch(err => onError(getErrorString(err) ?? 'Could not end the session'))
                        .finally(() => setEnding(false))
                }}
            >
                End
            </Button>
        </Flex>
    )
}

export default function SessionsSettings() {
    const [policy, setPolicy] = useState<SessionPolicy | null>(null)
    const [capacity, setCapacity] = useState<ModelCapacity[]>([])
    const [sessions, setSessions] = useState<SessionSummary[]>([])
    const [planner, setPlanner] = useState<PlannerState | null>(null)
    const [notice, setNotice] = useState<string | null>(null)
    const [error, setError] = useState<string | null>(null)

    const refresh = useCallback(() => {
        Promise.all([
            window.pairApi.sessions.getPolicy(),
            window.pairApi.sessions.getCapacity(),
            window.pairApi.sessions.list(),
            window.pairApi.sessions.getPlanner()
        ])
            .then(([p, c, s, pl]) => {
                setPolicy(p)
                setCapacity(c)
                setSessions(s)
                setPlanner(pl)
            })
            .catch(err => setError(getErrorString(err) ?? 'The router is not reachable'))
    }, [])

    useEffect(() => {
        refresh()
        const timer = setInterval(refresh, REFRESH_MS)
        return () => clearInterval(timer)
    }, [refresh])

    const onPlanner = useCallback(
        (state: PlannerState | null, message?: string) => {
            if (state) setPlanner(state)
            if (message) setNotice(message)
            refresh()
        },
        [refresh]
    )
    const autoReload = planner?.autoReload ?? true
    const listed = useMemo(
        () => new Set(capacity.map(row => `${row.engine}/${row.model}`)),
        [capacity]
    )
    const savedFor = (row: ModelCapacity) =>
        planner?.plans.find(p => p.engine === row.engine && p.model === row.model)

    return (
        <Stack gap="4" className="py-4">
            {error && <InlineErrorBanner message={error} onClose={() => setError(null)} />}
            {notice && (
                <Flex justify="between" align="center" gap="3" className="pair-paper p-3">
                    <Text kind="body/regular/sm">{notice}</Text>
                    <Button size="small" kind="tertiary" onClick={() => setNotice(null)}>
                        Dismiss
                    </Button>
                </Flex>
            )}
            {policy && (
                <PolicyCard
                    key={`${policy.idleTimeoutSeconds}-${policy.queueTimeoutSeconds}`}
                    policy={policy}
                    onSaved={setPolicy}
                    onError={setError}
                />
            )}
            <Card title="Context capacity on this node">
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Each session gets a model’s full context window; this sets how many sessions it
                    runs at once. When every session slot is held, new sessions go to another node
                    or wait.
                </Text>
                <Flex align="center" gap="3" wrap="wrap">
                    <Switch
                        checked={autoReload}
                        size="small"
                        aria-label="Reload automatically when idle"
                        onCheckedChange={checked =>
                            window.pairApi.sessions
                                .setPlanner({ autoReload: checked === true })
                                .then(state => setPlanner(state))
                                .catch(err =>
                                    setError(getErrorString(err) ?? 'Could not change the setting')
                                )
                        }
                    />
                    <Stack gap="0">
                        <Text kind="body/semibold/sm">Reload automatically when idle</Text>
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            Apply a saved plan once this node has no sessions and no requests. When
                            off, plans apply only when you press Apply.
                            {planner && !planner.activity.idle
                                ? ` Busy now: ${planner.activity.sessions} sessions, ${planner.activity.activeRequests} requests.`
                                : ''}
                        </Text>
                    </Stack>
                </Flex>
                {capacity.length === 0 ? (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        No models are loaded on this node.
                    </Text>
                ) : (
                    capacity.map(row => (
                        <CapacityRow
                            key={`${row.engine}/${row.model}/${row.configured?.contextTokens ?? 0}/${row.configured?.maxSessions ?? 0}`}
                            row={row}
                            saved={savedFor(row)}
                            autoReload={autoReload}
                            onChanged={refresh}
                            onPlanner={onPlanner}
                            onError={setError}
                        />
                    ))
                )}
                <InstalledModelPicker
                    listed={listed}
                    planner={planner}
                    onPlanner={onPlanner}
                    onError={setError}
                />
            </Card>
            <Card title="Active sessions">
                {sessions.length === 0 ? (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        No sessions are open.
                    </Text>
                ) : (
                    sessions.map(s => (
                        <SessionRow
                            key={s.sessionId}
                            session={s}
                            onEnded={refresh}
                            onError={setError}
                        />
                    ))
                )}
            </Card>
        </Stack>
    )
}
