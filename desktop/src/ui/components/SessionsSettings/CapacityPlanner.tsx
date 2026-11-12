// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from 'react'
import {
    Button,
    Flex,
    FormField,
    Slider,
    Stack,
    Switch,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import type {
    CapacityPlan,
    KvCacheType,
    PlannerState,
    SavedCapacityPlan
} from '@/shared/types/sessions'
import getErrorString from '@/shared/utils/get-error-string'
import {
    MAX_SESSIONS,
    memoryUse,
    MIN_WINDOW_TOKENS,
    nearestStop,
    replyBudget,
    replyStops,
    sessionsSliderMax,
    shortTokens,
    windowStops
} from './capacity-sliders'
import TextFitDetails from './TextFitDetails'

const GB = 1024 ** 3

function gb(bytes: number): string {
    return `${(bytes / GB).toFixed(1)} GB`
}

function tokens(n: number): string {
    return n > 0 ? n.toLocaleString() : '—'
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

/** Window used when the model does not report its maximum. */
const FALLBACK_WINDOW = 32768

const DANGER_TEXT = { color: 'var(--text-color-feedback-danger)' }

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

/** Turns the sliders inside red when the settings need more than the node has. */
function SliderTone({ over, children }: { over: boolean; children: React.ReactNode }) {
    return <div className={over ? 'pair-slider-over' : undefined}>{children}</div>
}

/** One labelled memory bar: what the settings need against what there is. */
function MemoryBar({
    label,
    needed,
    total,
    over,
    detail
}: {
    label: string
    needed: number
    total: number
    over: boolean
    detail: string
}) {
    const pct = total > 0 ? Math.min(100, (needed / total) * 100) : 0
    return (
        <Stack gap="1">
            <Flex justify="between" gap="3">
                <Text kind="body/semibold/sm">{label}</Text>
                <Text kind="body/semibold/sm" style={over ? DANGER_TEXT : undefined}>
                    {gb(needed)} of {gb(total)}
                    {over ? ' — too much' : ''}
                </Text>
            </Flex>
            <div
                role="meter"
                aria-label={label}
                aria-valuemin={0}
                aria-valuemax={total}
                aria-valuenow={needed}
                style={{
                    height: 8,
                    borderRadius: 4,
                    overflow: 'hidden',
                    background: 'color-mix(in srgb, currentColor 15%, transparent)'
                }}
            >
                <div
                    style={{
                        width: `${pct}%`,
                        height: '100%',
                        background: over
                            ? 'var(--text-color-feedback-danger)'
                            : 'var(--color-brand)'
                    }}
                />
            </div>
            <Text kind="body/regular/sm" className="text-subtle-color">
                {detail}
            </Text>
        </Stack>
    )
}

export default function CapacityPlanner({
    nodeId,
    engine,
    model,
    saved,
    onDone,
    onError
}: {
    /** The node the model runs on; omitted means this one. */
    nodeId?: string
    engine: string
    model: string
    saved: SavedCapacityPlan | undefined
    onDone: (state: PlannerState | null, message?: string) => void
    onError: (message: string) => void
}) {
    // null until the model's maximum is known; then the chosen window.
    const [windowTokens, setWindowTokens] = useState<number | null>(saved?.windowTokens ?? null)
    const [sessions, setSessions] = useState<number | null>(saved?.sessions ?? null)
    const [sessionsTouched, setSessionsTouched] = useState(saved !== undefined)
    const [cacheType, setCacheType] = useState<KvCacheType>(saved?.cacheType ?? 'f16')
    const [reserve, setReserve] = useState(saved?.reservePercent || 10)
    const [offload, setOffload] = useState(saved?.offloadExperts ?? false)
    // 0 until chosen: the default budget for the window.
    const [reply, setReply] = useState(saved?.replyTokens ?? 0)
    const [weightsGb, setWeightsGb] = useState('')
    const [kbPerToken, setKbPerToken] = useState('')
    const [plan, setPlan] = useState<CapacityPlan | null>(null)
    const [busy, setBusy] = useState(false)

    const weightsN = weightsGb.trim() === '' ? 0 : decimal(weightsGb)
    const kbN = kbPerToken.trim() === '' ? 0 : decimal(kbPerToken)
    const manualValid = weightsN !== null && kbN !== null

    useEffect(() => {
        if (!manualValid) return
        const timer = setTimeout(() => {
            window.pairApi.sessions
                .plan({
                    nodeId,
                    engine,
                    model,
                    cacheType,
                    reservePercent: reserve || undefined,
                    windowTokens: windowTokens ?? undefined,
                    offloadExperts: offload || undefined,
                    weightsBytes: weightsN ? Math.round(weightsN * GB) : undefined,
                    kvBytesPerToken: kbN ? Math.round(kbN * 1024) : undefined
                })
                .then(result => {
                    // A model that does not report its maximum needs a window
                    // to be sized at all.
                    if (windowTokens === null && /maximum context is unknown/.test(result.error)) {
                        setWindowTokens(FALLBACK_WINDOW)
                        return
                    }
                    setPlan(result)
                    if (!result.error) {
                        if (windowTokens === null) setWindowTokens(result.windowTokens)
                        if (!sessionsTouched) setSessions(Math.max(1, result.sessions))
                    }
                })
                .catch(err => onError(getErrorString(err) ?? 'Could not size this model'))
        }, 200)
        return () => clearTimeout(timer)
    }, [
        nodeId,
        engine,
        model,
        windowTokens,
        cacheType,
        reserve,
        offload,
        weightsN,
        kbN,
        manualValid,
        sessionsTouched,
        onError
    ])

    const stops = useMemo(
        () => windowStops(plan?.spec.maxContextTokens ?? 0),
        [plan?.spec.maxContextTokens]
    )
    const windowIndex = nearestStop(stops, windowTokens ?? stops[stops.length - 1])
    const windowNow = stops[windowIndex]
    const replyOptions = useMemo(() => replyStops(windowNow), [windowNow])
    const replyIndex = nearestStop(replyOptions, replyBudget(windowNow, reply))
    const replyNow = replyOptions[replyIndex]
    const messageTokens = windowNow - replyNow
    const fit = plan && !plan.error ? plan.sessions : 0
    // Reaches past what fits, and never below a saved count.
    const sessionsMax = Math.max(sessionsSliderMax(fit), Math.min(sessions ?? 0, MAX_SESSIONS))
    const chosen = Math.min(sessions ?? 1, sessionsMax)
    const use = plan && !plan.error ? memoryUse(plan, chosen) : null
    const over = use ? use.gpuOver || use.ramOver : false
    // Settings past the hardware can still be saved; the sliders and bars
    // turn red to say so.
    const usable =
        plan !== null &&
        !plan.error &&
        use !== null &&
        chosen >= 1 &&
        plan.windowTokens >= MIN_WINDOW_TOKENS
    const needsManual = plan?.error && /unknown/.test(plan.error)

    const save = (apply: boolean) => {
        if (!plan || !usable) return
        setBusy(true)
        window.pairApi.sessions
            .setPlanner({
                nodeId,
                save: {
                    engine,
                    model,
                    windowTokens: plan.windowTokens,
                    sessions: chosen,
                    cacheType: plan.cacheType,
                    reservePercent: reserve,
                    offloadExperts: plan.offloadExperts,
                    replyTokens: replyNow
                }
            })
            .then(async state => {
                if (!apply) return onDone(state)
                const result = await window.pairApi.sessions.applyPlan(engine, model, nodeId)
                onDone(await window.pairApi.sessions.getPlanner(nodeId), result.message)
            })
            .catch(err => onError(getErrorString(err) ?? 'Could not save the plan'))
            .finally(() => setBusy(false))
    }

    const gpuLabel = use?.shared ? 'Memory (shared by GPU and system)' : 'GPU memory'

    return (
        <Stack gap="3" className="pair-paper p-3">
            <Text kind="body/regular/sm" className="text-subtle-color">
                Each session gets the whole context window. Choose the window, then how many
                sessions run at once; the bars show the memory that needs.
            </Text>

            <Stack gap="1">
                <Flex justify="between" gap="3">
                    <Text kind="body/semibold/sm">Context window per session</Text>
                    <Text kind="body/semibold/sm">{tokens(stops[windowIndex])} tokens</Text>
                </Flex>
                <SliderTone over={over}>
                    <Slider
                        min={0}
                        max={stops.length - 1}
                        step={1}
                        value={windowIndex}
                        onValueChange={i => setWindowTokens(stops[i])}
                        customSteps={stops.map((_, i) => i)}
                        stepFormatFn={i => shortTokens(stops[i] ?? 0)}
                        attributes={{ SliderInput: { 'aria-label': 'Context window per session' } }}
                    />
                </SliderTone>
            </Stack>

            <Stack gap="1">
                <Flex justify="between" gap="3">
                    <Text kind="body/semibold/sm">Room for the reply</Text>
                    <Text kind="body/semibold/sm">{tokens(replyNow)} tokens</Text>
                </Flex>
                <Slider
                    min={0}
                    max={replyOptions.length - 1}
                    step={1}
                    value={replyIndex}
                    onValueChange={i => setReply(replyOptions[i])}
                    customSteps={replyOptions.map((_, i) => i)}
                    stepFormatFn={i => shortTokens(replyOptions[i] ?? 0)}
                    attributes={{ SliderInput: { 'aria-label': 'Room for the reply' } }}
                />
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Each session’s window is shared: {tokens(messageTokens)} tokens for its messages
                    and history, {tokens(replyNow)} for the model’s reply, a thinking model’s
                    reasoning included. PAIR refuses a conversation that outgrows its part, and caps
                    replies that set no length of their own.
                </Text>
                <TextFitDetails windowTokens={windowNow} replyTokens={replyNow} />
            </Stack>

            <Stack gap="1">
                <Flex justify="between" gap="3">
                    <Text kind="body/semibold/sm">Sessions at once</Text>
                    <Text
                        kind="body/semibold/sm"
                        style={chosen > fit && plan && !plan.error ? DANGER_TEXT : undefined}
                    >
                        {chosen} {plan && !plan.error ? `(${fit} fit)` : ''}
                    </Text>
                </Flex>
                <SliderTone over={over}>
                    <Slider
                        min={1}
                        max={sessionsMax}
                        step={1}
                        value={chosen}
                        onValueChange={n => {
                            setSessions(n)
                            setSessionsTouched(true)
                        }}
                        customSteps={[
                            1,
                            ...(fit > 1 && fit < sessionsMax ? [fit] : []),
                            sessionsMax
                        ]}
                        attributes={{ SliderInput: { 'aria-label': 'Sessions at once' } }}
                    />
                </SliderTone>
            </Stack>

            <Stack gap="1">
                <Flex justify="between" gap="3">
                    <Text kind="body/semibold/sm">Keep free</Text>
                    <Text kind="body/semibold/sm">{reserve}% of each GPU</Text>
                </Flex>
                <SliderTone over={over}>
                    <Slider
                        min={0}
                        max={50}
                        step={5}
                        value={reserve}
                        onValueChange={setReserve}
                        customSteps={[0, 10, 25, 50]}
                        stepFormatFn={v => `${v}%`}
                        attributes={{ SliderInput: { 'aria-label': 'Keep free' } }}
                    />
                </SliderTone>
            </Stack>

            <Stack gap="1">
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Context cache precision (lower fits more sessions; slightly less accurate)
                </Text>
                <Choice options={CACHE_TYPES} value={cacheType} onChange={setCacheType} />
            </Stack>

            {plan && (plan.canOffloadExperts || plan.offloadExperts) && (
                <Flex align="center" gap="3">
                    <Switch
                        checked={offload}
                        size="small"
                        aria-label="Keep expert weights in system memory"
                        onCheckedChange={checked => setOffload(checked === true)}
                    />
                    <Stack gap="0">
                        <Text kind="body/semibold/sm">
                            Keep expert weights in system memory ({gb(plan.expertBytes)})
                        </Text>
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            Frees GPU memory for more sessions; tokens come out slower because the
                            experts run on the CPU.
                        </Text>
                    </Stack>
                </Flex>
            )}
            {plan?.offloadNote && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    This is a mixture-of-experts model, but {plan.offloadNote}.
                </Text>
            )}

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
            {!manualValid && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Enter numbers only.
                </Text>
            )}

            {plan?.error && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {plan.error}
                </Text>
            )}
            {plan && !plan.error && use && (
                <Stack gap="2">
                    <Text kind="body/semibold/md" style={over ? DANGER_TEXT : undefined}>
                        {over
                            ? `${chosen} session${chosen === 1 ? '' : 's'} of ${tokens(plan.windowTokens)} tokens need more memory than this node has`
                            : `${chosen} session${chosen === 1 ? '' : 's'} of ${tokens(plan.windowTokens)} tokens`}
                    </Text>
                    {over && (
                        <Text kind="body/regular/sm" style={DANGER_TEXT}>
                            You can still save this plan. The engine may then spill onto the CPU and
                            run slowly, or fail to load the model.
                        </Text>
                    )}
                    <MemoryBar
                        label={gpuLabel}
                        needed={use.gpuNeeded}
                        total={use.gpuTotal}
                        over={use.gpuOver}
                        detail={[
                            `model ${gb(plan.gpuWeightsBytes)}`,
                            plan.otherModelsBytes > 0
                                ? `other models ${gb(plan.otherModelsBytes)}`
                                : '',
                            `${chosen} × ${gb(plan.bytesPerSession)} context (${(plan.bytesPerToken / 1024).toFixed(0)} KB per token)`,
                            `kept free ${gb(plan.reserveBytes)}`
                        ]
                            .filter(Boolean)
                            .join(' · ')}
                    />
                    {!use.shared && use.ramTotal > 0 && (
                        <MemoryBar
                            label="System memory"
                            needed={use.ramNeeded}
                            total={use.ramTotal}
                            over={use.ramOver}
                            detail={[
                                plan.offloadExperts
                                    ? `expert weights ${gb(plan.ramUsedBytes)}`
                                    : 'no model weights',
                                `kept free for the system ${gb(plan.ramReserveBytes)}`
                            ].join(' · ')}
                        />
                    )}
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {SOURCE_TEXT[plan.spec.source] ?? ''}
                        {plan.spec.maxContextTokens > 0
                            ? ` The model supports up to ${tokens(plan.spec.maxContextTokens)} tokens${
                                  plan.spec.contextSource === 'web'
                                      ? `, according to its page (read by ${plan.spec.contextReader})`
                                      : ''
                              }.`
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
