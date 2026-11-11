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
import { CapacityPlannerDialog } from '@/ui/components/ModelManager/CapacityPlannerDialog'
import { useCapacityPlannerStore } from '@/ui/stores/capacity-planner.store'
import { planStatus } from './plan-status'

// The proxy owns these settings and their saved state; this panel only reads
// what it reports and sends changes. It polls while open because sessions and
// slots change with every request and the proxy pushes no events for them.
const REFRESH_MS = 5000
const SHARE_GRANULARITY = 1024

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

/** One model in the capacity summary: what PAIR enforces now and its plan. */
interface SummaryEntry {
    engine: string
    model: string
    capacity?: ModelCapacity
    plan?: SavedCapacityPlan
}

function CapacitySummaryRow({
    entry,
    autoReload,
    onChange,
    onChanged,
    onPlanner,
    onError
}: {
    entry: SummaryEntry
    autoReload: boolean
    onChange: () => void
    onChanged: () => void
    onPlanner: (state: PlannerState | null, message?: string) => void
    onError: (message: string) => void
}) {
    const [busy, setBusy] = useState(false)
    const { capacity: row, plan } = entry
    const run = (work: Promise<unknown>, failure: string) => {
        setBusy(true)
        work.catch(err => onError(getErrorString(err) ?? failure)).finally(() => setBusy(false))
    }
    const perSession = row ? sessionWindow(row.contextTokens, row.maxSessions) : 0
    return (
        <Stack gap="2" className="pair-paper p-3">
            <Flex justify="between" align="center" gap="2" wrap="wrap">
                <Text kind="body/semibold/sm">
                    {entry.model} <span className="text-subtle-color">· {entry.engine}</span>
                </Text>
                {row && (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {row.heldSlots} of {row.maxSessions || '—'} sessions · {row.queued} waiting
                        · {formatTokens(row.usedContextTokens)} tokens in use
                    </Text>
                )}
            </Flex>
            {row && row.maxSessions > 0 && (
                <Text kind="body/regular/sm">
                    Now: {row.maxSessions} session{row.maxSessions === 1 ? '' : 's'} of{' '}
                    {formatTokens(perSession)} tokens
                    {row.configured ? '' : ' (reported by the engine)'}
                </Text>
            )}
            {plan && <Text kind="body/regular/sm">{planStatus(plan, autoReload)}</Text>}
            {row?.warning && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {row.warning}
                </Text>
            )}
            <Flex gap="2" wrap="wrap">
                <Button size="small" kind="secondary" disabled={busy} onClick={onChange}>
                    Change…
                </Button>
                {plan && !plan.applied && (
                    <Button
                        size="small"
                        kind="primary"
                        color="brand"
                        disabled={busy}
                        onClick={() =>
                            run(applySavedPlan(entry, onPlanner), 'Could not apply the plan')
                        }
                    >
                        Apply
                    </Button>
                )}
                {plan && (
                    <Button
                        size="small"
                        kind="secondary"
                        disabled={busy}
                        onClick={() =>
                            run(
                                window.pairApi.sessions
                                    .setPlanner({
                                        remove: { engine: entry.engine, model: entry.model }
                                    })
                                    .then(state => onPlanner(state)),
                                'Could not remove the plan'
                            )
                        }
                    >
                        Remove plan
                    </Button>
                )}
                {row?.configured && (
                    <Button
                        size="small"
                        kind="secondary"
                        disabled={busy}
                        onClick={() =>
                            run(
                                window.pairApi.sessions
                                    .setCapacity({
                                        engine: entry.engine,
                                        model: entry.model,
                                        clear: true
                                    })
                                    .then(onChanged),
                                'Could not clear the limit'
                            )
                        }
                    >
                        Clear limit
                    </Button>
                )}
            </Flex>
        </Stack>
    )
}

async function applySavedPlan(
    entry: SummaryEntry,
    onPlanner: (state: PlannerState | null, message?: string) => void
) {
    const result = await window.pairApi.sessions.applyPlan(entry.engine, entry.model)
    onPlanner(await window.pairApi.sessions.getPlanner(), result.message)
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
    const planner = useCapacityPlannerStore(st => st.planner)
    const setPlanner = useCapacityPlannerStore(st => st.setPlanner)
    const [changing, setChanging] = useState<SummaryEntry | null>(null)
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
    }, [setPlanner])

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
        [refresh, setPlanner]
    )
    const autoReload = planner?.autoReload ?? true
    // Every model PAIR limits or has a plan for, loaded or not.
    const entries = useMemo(() => {
        const byKey = new Map<string, SummaryEntry>()
        for (const row of capacity)
            byKey.set(`${row.engine}/${row.model}`, {
                engine: row.engine,
                model: row.model,
                capacity: row
            })
        for (const plan of planner?.plans ?? []) {
            const key = `${plan.engine}/${plan.model}`
            const found = byKey.get(key)
            byKey.set(key, { engine: plan.engine, model: plan.model, ...found, plan })
        }
        return [...byKey.values()].sort((a, b) => a.model.localeCompare(b.model))
    }, [capacity, planner])

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
                    Each session gets a model’s full context window, and each model runs a set
                    number of sessions at once. Set them per model with Context &amp; sessions in
                    the model’s menu on the Overview, or with Change… here. When every session slot
                    is held, new sessions go to another node or wait.
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
                {entries.length === 0 ? (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        No models are loaded or planned on this node.
                    </Text>
                ) : (
                    entries.map(entry => (
                        <CapacitySummaryRow
                            key={`${entry.engine}/${entry.model}`}
                            entry={entry}
                            autoReload={autoReload}
                            onChange={() => setChanging(entry)}
                            onChanged={refresh}
                            onPlanner={onPlanner}
                            onError={setError}
                        />
                    ))
                )}
                {changing && (
                    <CapacityPlannerDialog
                        engine={changing.engine}
                        model={changing.model}
                        displayName={changing.model}
                        onClose={() => {
                            setChanging(null)
                            refresh()
                        }}
                    />
                )}
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
