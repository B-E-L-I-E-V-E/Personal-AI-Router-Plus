// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from 'react'
import { Button, Flex, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { SessionPolicy, SessionSummary } from '@/shared/types/sessions'
import getErrorString from '@/shared/utils/get-error-string'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'

// The proxy owns these settings and their saved state; this panel only reads
// what it reports and sends changes. It polls while open because sessions
// change with every request and the proxy pushes no events for them. Each
// model's context capacity is set in its own menu (CapacityPlannerDialog).
const REFRESH_MS = 5000

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
    const rooms = session.context.flatMap(c =>
        c.postRoomTokens === null ? [] : [c.postRoomTokens]
    )
    const room = rooms.length > 0 ? Math.min(...rooms) : null
    const expires = new Date(session.expiresAt)
    return (
        <Flex justify="between" align="center" gap="3" wrap="wrap" className="pair-paper p-3">
            <Stack gap="1">
                <Text kind="body/semibold/sm">{session.sessionId}</Text>
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {session.agentId ? `Agent ${session.agentId} · ` : ''}
                    {session.requests} requests · {formatTokens(tokens)} tokens
                    {room === null
                        ? ''
                        : room > 0
                          ? ` · room for ${room.toLocaleString()} more`
                          : ' · full: shorten the conversation'}
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
    const [sessions, setSessions] = useState<SessionSummary[]>([])
    const [error, setError] = useState<string | null>(null)

    const refresh = useCallback(() => {
        Promise.all([window.pairApi.sessions.getPolicy(), window.pairApi.sessions.list()])
            .then(([p, s]) => {
                setPolicy(p)
                setSessions(s)
            })
            .catch(err => setError(getErrorString(err) ?? 'The router is not reachable'))
    }, [])

    useEffect(() => {
        refresh()
        const timer = setInterval(refresh, REFRESH_MS)
        return () => clearInterval(timer)
    }, [refresh])

    return (
        <Stack gap="4" className="py-4">
            {error && <InlineErrorBanner message={error} onClose={() => setError(null)} />}
            {policy && (
                <PolicyCard
                    key={`${policy.idleTimeoutSeconds}-${policy.queueTimeoutSeconds}`}
                    policy={policy}
                    onSaved={setPolicy}
                    onError={setError}
                />
            )}
            <Card title="Active sessions">
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Each model’s context window, room for the reply and sessions at once are set in
                    its menu on the Overview: … → Context &amp; sessions.
                </Text>
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
