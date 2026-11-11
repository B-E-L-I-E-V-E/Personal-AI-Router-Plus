// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Button, Flex, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type {
    CapacityPlan,
    KvCacheType,
    PlannerState,
    SavedCapacityPlan
} from '@/shared/types/sessions'
import getErrorString from '@/shared/utils/get-error-string'

const GB = 1024 ** 3

function gb(bytes: number): string {
    return `${(bytes / GB).toFixed(1)} GB`
}

function tokens(n: number): string {
    return n > 0 ? n.toLocaleString() : '—'
}

function wholeNumber(text: string): number | null {
    const trimmed = text.trim()
    if (!/^\d+$/.test(trimmed)) return null
    return Number(trimmed)
}

function decimal(text: string): number | null {
    const trimmed = text.trim()
    if (!/^\d+(\.\d+)?$/.test(trimmed)) return null
    return Number(trimmed)
}

const CACHE_TYPES: { id: KvCacheType; label: string }[] = [
    { id: 'f16', label: 'Full (f16)' },
    { id: 'q8_0', label: '8-bit' },
    { id: 'q4_0', label: '4-bit' }
]

const SOURCE_TEXT: Record<string, string> = {
    metadata: 'Context cost computed from the model’s architecture.',
    measured: 'Context cost measured from the loaded model.',
    manual: 'Using the values you entered.'
}

function Choice<T extends string>({
    options,
    value,
    onChange
}: {
    options: { id: T; label: string }[]
    value: T
    onChange: (v: T) => void
}) {
    return (
        <Flex gap="1" wrap="wrap">
            {options.map(o => (
                <Button
                    key={o.id}
                    size="small"
                    kind={o.id === value ? 'primary' : 'secondary'}
                    color={o.id === value ? 'brand' : undefined}
                    onClick={() => onChange(o.id)}
                >
                    {o.label}
                </Button>
            ))}
        </Flex>
    )
}

function Line({ label, value }: { label: string; value: string }) {
    return (
        <Flex justify="between" gap="3">
            <Text kind="body/regular/sm" className="text-subtle-color">
                {label}
            </Text>
            <Text kind="body/regular/sm">{value}</Text>
        </Flex>
    )
}

export default function CapacityPlanner({
    engine,
    model,
    saved,
    onDone,
    onError
}: {
    engine: string
    model: string
    saved: SavedCapacityPlan | undefined
    onDone: (state: PlannerState | null, message?: string) => void
    onError: (message: string) => void
}) {
    const [windowText, setWindowText] = useState(saved ? String(saved.windowTokens) : '')
    const [capText, setCapText] = useState('')
    const [cacheType, setCacheType] = useState<KvCacheType>(saved?.cacheType ?? 'f16')
    const [reserve, setReserve] = useState(String(saved?.reservePercent || 10))
    const [weightsGb, setWeightsGb] = useState('')
    const [kbPerToken, setKbPerToken] = useState('')
    const [plan, setPlan] = useState<CapacityPlan | null>(null)
    const [busy, setBusy] = useState(false)

    const windowN = windowText.trim() === '' ? 0 : wholeNumber(windowText)
    const capN = capText.trim() === '' ? 0 : wholeNumber(capText)
    const reserveN = decimal(reserve)
    const weightsN = weightsGb.trim() === '' ? 0 : decimal(weightsGb)
    const kbN = kbPerToken.trim() === '' ? 0 : decimal(kbPerToken)
    const inputsValid =
        windowN !== null &&
        capN !== null &&
        capN <= 256 &&
        reserveN !== null &&
        reserveN <= 90 &&
        weightsN !== null &&
        kbN !== null

    useEffect(() => {
        if (!inputsValid) return
        const timer = setTimeout(() => {
            window.pairApi.sessions
                .plan({
                    engine,
                    model,
                    cacheType,
                    reservePercent: reserveN || undefined,
                    windowTokens: windowN || undefined,
                    maxSessions: capN || undefined,
                    weightsBytes: weightsN ? Math.round(weightsN * GB) : undefined,
                    kvBytesPerToken: kbN ? Math.round(kbN * 1024) : undefined
                })
                .then(setPlan)
                .catch(err => onError(getErrorString(err) ?? 'Could not size this model'))
        }, 300)
        return () => clearTimeout(timer)
    }, [engine, model, windowN, capN, cacheType, reserveN, weightsN, kbN, inputsValid, onError])

    const usable = plan && !plan.error && plan.sessions > 0 && plan.windowTokens >= 2048
    const needsManual = plan?.error && /unknown/.test(plan.error)

    const save = (apply: boolean) => {
        if (!plan || !usable) return
        setBusy(true)
        window.pairApi.sessions
            .setPlanner({
                save: {
                    engine,
                    model,
                    windowTokens: plan.windowTokens,
                    sessions: plan.sessions,
                    cacheType: plan.cacheType,
                    reservePercent: reserveN ?? 10
                }
            })
            .then(async state => {
                if (!apply) return onDone(state)
                const result = await window.pairApi.sessions.applyPlan(engine, model)
                onDone(await window.pairApi.sessions.getPlanner(), result.message)
            })
            .catch(err => onError(getErrorString(err) ?? 'Could not save the plan'))
            .finally(() => setBusy(false))
    }

    return (
        <Stack gap="3" className="pair-paper p-3">
            <Text kind="body/regular/sm" className="text-subtle-color">
                Each session gets this model’s full context window. PAIR runs as many sessions at
                once as fit in the GPU memory left after the model and a safety reserve.
            </Text>
            <Flex align="end" gap="3" wrap="wrap">
                <FormField slotLabel="Context window per session (blank: model maximum)">
                    <TextInput value={windowText} onValueChange={setWindowText} />
                </FormField>
                <FormField slotLabel="At most sessions (blank: as many as fit)">
                    <TextInput value={capText} onValueChange={setCapText} />
                </FormField>
                <FormField slotLabel="Keep free (% of each GPU)">
                    <TextInput value={reserve} onValueChange={setReserve} />
                </FormField>
            </Flex>
            <Stack gap="1">
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Context cache precision (lower fits more sessions; slightly less accurate)
                </Text>
                <Choice options={CACHE_TYPES} value={cacheType} onChange={setCacheType} />
            </Stack>
            {(needsManual || weightsGb || kbPerToken) && (
                <Flex align="end" gap="3" wrap="wrap">
                    <FormField slotLabel="Model size (GB)">
                        <TextInput value={weightsGb} onValueChange={setWeightsGb} />
                    </FormField>
                    <FormField slotLabel="Context cost (KB per token)">
                        <TextInput value={kbPerToken} onValueChange={setKbPerToken} />
                    </FormField>
                </Flex>
            )}
            {!inputsValid && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Enter numbers only; at most 256 sessions and 90% kept free.
                </Text>
            )}

            {plan?.error && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {plan.error}
                </Text>
            )}
            {plan && !plan.error && (
                <Stack gap="1">
                    <Text kind="body/semibold/md">
                        {plan.sessions > 0
                            ? `${plan.sessions} session${plan.sessions === 1 ? '' : 's'} of ${tokens(plan.windowTokens)} tokens`
                            : 'Does not fit'}
                    </Text>
                    <Line label="GPU memory" value={gb(plan.totalBytes)} />
                    <Line label="Kept free" value={gb(plan.reserveBytes)} />
                    <Line label="Model" value={gb(plan.weightsBytes)} />
                    {plan.otherModelsBytes > 0 && (
                        <Line label="Other models" value={gb(plan.otherModelsBytes)} />
                    )}
                    <Line label="Left for context" value={gb(plan.contextBytes)} />
                    <Line
                        label="Each session"
                        value={`${gb(plan.bytesPerSession)} (${(plan.bytesPerToken / 1024).toFixed(0)} KB per token)`}
                    />
                    <Line
                        label="Planned use"
                        value={`${gb(plan.usedBytes)} of ${gb(plan.totalBytes)}`}
                    />
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {SOURCE_TEXT[plan.spec.source] ?? ''}
                        {plan.spec.maxContextTokens > 0
                            ? ` The model supports up to ${tokens(plan.spec.maxContextTokens)} tokens.`
                            : ''}
                    </Text>
                    {[...plan.spec.notes, ...plan.warnings].map(w => (
                        <Text key={w} kind="body/regular/sm" className="text-subtle-color">
                            {w}
                        </Text>
                    ))}
                </Stack>
            )}
            <Flex gap="2" wrap="wrap">
                <Button
                    kind="primary"
                    color="brand"
                    disabled={!usable || busy}
                    onClick={() => save(true)}
                >
                    Save and apply now
                </Button>
                <Button kind="secondary" disabled={!usable || busy} onClick={() => save(false)}>
                    Save plan
                </Button>
                <Button kind="tertiary" disabled={busy} onClick={() => onDone(null)}>
                    Close
                </Button>
            </Flex>
            {engine !== 'lmstudio' && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Applying restarts{' '}
                    {engine === 'ollama' ? 'Ollama (every model it runs)' : 'llama.cpp'} with these
                    settings; requests in progress on it are interrupted.
                </Text>
            )}
        </Stack>
    )
}
