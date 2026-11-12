// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from 'react'
import {
    Button,
    Flex,
    Slider,
    Stack,
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
    cpuExpertBytes,
    expertLayersOnCpu,
    expertSplitStops,
    kvStops,
    labelledStops,
    MAX_SESSIONS,
    memoryUse,
    MIN_WINDOW_TOKENS,
    nearestStop,
    parameterLabel,
    parseAmount,
    perTokenLabel,
    replyBudget,
    replyStops,
    sessionsFit,
    sessionsSliderMax,
    shortTokens,
    sizeLabel,
    weightStops,
    windowStops,
    withStop
} from './capacity-sliders'
import { kvFact, kvUsed, loadedFact, sizeFacts, weightsUsed } from './model-size'
import TextFitDetails from './TextFitDetails'

const GB = 1024 ** 3

function gb(bytes: number): string {
    return `${(bytes / GB).toFixed(1)} GB`
}

function tokens(n: number): string {
    return n > 0 ? n.toLocaleString() : '—'
}

const CACHE_TYPES: { id: KvCacheType; label: string }[] = [
    { id: 'f16', label: 'Full (f16)' },
    { id: 'q8_0', label: '8-bit' },
    { id: 'q4_0', label: '4-bit' }
]

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

/** The exact amount typed beside a slider, in its unit: 4.25, 112. */
function amountText(bytes: number, unitBytes: number): string {
    return String(Number((bytes / unitBytes).toFixed(2)))
}

/**
 * A snapping slider for one of the model's figures, with the exact value
 * editable beside it. Empty means the model's own value; anything set
 * overrides it.
 */
function AmountSlider({
    label,
    modelValue,
    value,
    stops,
    unit,
    unitBytes,
    format,
    over,
    onChange,
    children
}: {
    label: string
    /** What the model reports; 0 when unknown. */
    modelValue: number
    /** The override; null for the model's value. */
    value: number | null
    stops: number[]
    unit: string
    unitBytes: number
    format: (bytes: number) => string
    over: boolean
    onChange: (bytes: number | null) => void
    children: React.ReactNode
}) {
    const [text, setText] = useState(value === null ? '' : amountText(value, unitBytes))
    const current = value ?? modelValue
    const all = useMemo(() => withStop(stops, current), [stops, current])
    const index = nearestStop(all, current > 0 ? current : all[Math.floor(all.length / 2)])
    const ticks = useMemo(() => labelledStops(all, modelValue), [all, modelValue])
    const typedInvalid = text.trim() !== '' && parseAmount(text, unitBytes) === null
    return (
        <Stack gap="1">
            <Flex justify="between" gap="3">
                <Text kind="body/semibold/sm">{label}</Text>
                <Text kind="body/semibold/sm">
                    {current > 0 ? format(current) : 'not reported — set it'}
                    {value !== null && modelValue > 0 ? ' (yours)' : ''}
                </Text>
            </Flex>
            <Flex align="center" gap="3">
                <div style={{ flex: 1, minWidth: 0 }}>
                    <SliderTone over={over}>
                        <Slider
                            min={0}
                            max={all.length - 1}
                            step={1}
                            value={index}
                            onValueChange={i => {
                                setText(amountText(all[i], unitBytes))
                                onChange(all[i])
                            }}
                            customSteps={ticks}
                            stepFormatFn={i => format(all[i] ?? 0)}
                            attributes={{ SliderInput: { 'aria-label': label } }}
                        />
                    </SliderTone>
                </div>
                <div style={{ width: 112 }}>
                    <TextInput
                        value={text}
                        placeholder={modelValue > 0 ? amountText(modelValue, unitBytes) : ''}
                        slotEnd={unit}
                        aria-label={`${label}, exact (${unit})`}
                        onValueChange={t => {
                            setText(t)
                            if (t.trim() === '') return onChange(null)
                            const bytes = parseAmount(t, unitBytes)
                            if (bytes !== null) onChange(bytes)
                        }}
                    />
                </div>
            </Flex>
            {typedInvalid && (
                <Text kind="body/regular/sm" style={DANGER_TEXT}>
                    Enter a number of {unit}.
                </Text>
            )}
            {value !== null && modelValue > 0 && (
                <div>
                    <Button
                        size="small"
                        kind="tertiary"
                        onClick={() => {
                            setText('')
                            onChange(null)
                        }}
                    >
                        Use the model’s {format(modelValue)}
                    </Button>
                </div>
            )}
            {children}
        </Stack>
    )
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

/**
 * What a mixture-of-experts model holds and runs: every expert stays loaded,
 * since any token may pick any of them, but each token runs only a few.
 */
function moeSummary(plan: CapacityPlan): string {
    const { expertCount, expertUsedCount, totalParams, activeParams } = plan.spec
    const counts =
        totalParams > 0 && activeParams > 0
            ? `${parameterLabel(totalParams)} total · ${parameterLabel(activeParams)} active per token. `
            : ''
    const used = expertUsedCount > 0 ? `${expertUsedCount} of its ${expertCount}` : 'a few of its'
    return `${counts}A mixture-of-experts model: each token runs ${used} experts per layer, but every expert must stay loaded, in GPU or system memory, because any token may pick any of them.`
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
    // llama.cpp's --n-cpu-moe: the first N layers keep their experts in
    // system memory.
    const [cpuLayers, setCpuLayers] = useState(saved?.cpuExpertLayers ?? 0)
    // 0 until chosen: the default budget for the window.
    const [reply, setReply] = useState(saved?.replyTokens ?? 0)
    // Overrides of what the model reports, in bytes; null uses the model's.
    const [weightsOverride, setWeightsOverride] = useState<number | null>(null)
    const [kvOverride, setKvOverride] = useState<number | null>(null)
    const [plan, setPlan] = useState<CapacityPlan | null>(null)
    const [busy, setBusy] = useState(false)

    useEffect(() => {
        const timer = setTimeout(() => {
            window.pairApi.sessions
                .plan({
                    nodeId,
                    engine,
                    model,
                    cacheType,
                    reservePercent: reserve || undefined,
                    windowTokens: windowTokens ?? undefined,
                    cpuExpertLayers: cpuLayers || undefined,
                    weightsBytes: weightsOverride ?? undefined,
                    kvBytesPerToken: kvOverride ?? undefined
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
        cpuLayers,
        weightsOverride,
        kvOverride,
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
    const known = sessionsFit(plan)
    // Reaches past what fits, and never below a saved count.
    const sessionsMax = Math.max(
        sessionsSliderMax(known ?? 0),
        Math.min(sessions ?? 0, MAX_SESSIONS)
    )
    const chosen = Math.min(sessions ?? 1, sessionsMax)
    const expertStops = useMemo(
        () => expertSplitStops(plan?.spec.expertLayerBytes ?? []),
        [plan?.spec.expertLayerBytes]
    )
    const expertLayers = expertStops.length - 1
    const layersOnCpu = expertLayersOnCpu(expertStops, cpuLayers)
    const cpuLayersNow = expertStops[layersOnCpu]
    const use = plan && !plan.error ? memoryUse(plan, chosen, cpuLayersNow) : null
    const expertsInRam = cpuExpertBytes(plan?.spec.expertLayerBytes ?? [], cpuLayersNow)
    const over = use ? use.gpuOver || use.ramOver : false
    // Settings past the hardware can still be saved; the sliders and bars
    // turn red to say so.
    const usable =
        plan !== null &&
        !plan.error &&
        use !== null &&
        chosen >= 1 &&
        plan.windowTokens >= MIN_WINDOW_TOKENS
    const modelWeights = plan?.spec.weightsBytes ?? 0
    const modelKv = plan?.spec.kvBytesPerToken ?? 0
    const sizeStops = useMemo(
        () => weightStops(modelWeights, plan?.totalBytes ?? 0),
        [modelWeights, plan?.totalBytes]
    )
    const costStops = useMemo(() => kvStops(modelKv), [modelKv])

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
                    cpuExpertLayers: plan.canOffloadExperts ? cpuLayersNow : 0,
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
                    {known !== null && (
                        <Text
                            kind="body/semibold/sm"
                            style={chosen > known ? DANGER_TEXT : undefined}
                        >
                            {chosen} ({known} fit)
                        </Text>
                    )}
                </Flex>
                {known === null ? (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {plan
                            ? 'Chosen once PAIR can size this model; see below for what it needs.'
                            : 'Sizing this model…'}
                    </Text>
                ) : (
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
                                ...(known > 1 && known < sessionsMax ? [known] : []),
                                sessionsMax
                            ]}
                            attributes={{ SliderInput: { 'aria-label': 'Sessions at once' } }}
                        />
                    </SliderTone>
                )}
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

            {plan && (
                <AmountSlider
                    label="Model weights in memory"
                    modelValue={modelWeights}
                    value={weightsOverride}
                    stops={sizeStops}
                    unit="GB"
                    unitBytes={GB}
                    format={sizeLabel}
                    over={over}
                    onChange={setWeightsOverride}
                >
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        The memory the weights take once loaded (in GPU memory, and in system memory
                        for any part kept there), not the download.{' '}
                        {sizeFacts(plan.spec).join(' · ')}.
                    </Text>
                    {loadedFact(plan.spec) && (
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            {loadedFact(plan.spec)}
                        </Text>
                    )}
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {weightsUsed(plan)}
                    </Text>
                </AmountSlider>
            )}

            {plan && (
                <AmountSlider
                    label="Context cache per token"
                    modelValue={modelKv}
                    value={kvOverride}
                    stops={costStops}
                    unit="KB"
                    unitBytes={1024}
                    format={perTokenLabel}
                    over={over}
                    onChange={setKvOverride}
                >
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        Every token a session holds keeps its keys and values in memory; this is
                        that cost for one token. {kvFact(plan)}
                    </Text>
                    {kvUsed(plan) && (
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            {kvUsed(plan)}
                        </Text>
                    )}
                </AmountSlider>
            )}

            {plan && plan.spec.expertCount > 0 && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {moeSummary(plan)}
                </Text>
            )}
            {plan && plan.canOffloadExperts && expertLayers > 0 && (
                <Stack gap="1">
                    <Flex justify="between" gap="3">
                        <Text kind="body/semibold/sm">Expert layers kept on the GPU</Text>
                        <Text kind="body/semibold/sm">
                            {expertLayers - layersOnCpu} of {expertLayers}
                        </Text>
                    </Flex>
                    <SliderTone over={over}>
                        <Slider
                            min={0}
                            max={expertLayers}
                            step={1}
                            value={expertLayers - layersOnCpu}
                            onValueChange={g => setCpuLayers(expertStops[expertLayers - g])}
                            customSteps={[0, expertLayers]}
                            attributes={{
                                SliderInput: { 'aria-label': 'Expert layers kept on the GPU' }
                            }}
                        />
                    </SliderTone>
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {layersOnCpu === 0
                            ? `All expert weights on the GPU (${gb(plan.expertBytes)}).`
                            : `${gb(plan.expertBytes - expertsInRam)} of expert weights on the GPU, ${gb(expertsInRam)} in system memory, where the CPU runs them.`}{' '}
                        {engine === 'llamacpp' && layersOnCpu > 0
                            ? `Applied as --n-cpu-moe ${cpuLayersNow}.`
                            : ''}
                        {engine === 'lmstudio' && layersOnCpu > 0
                            ? `In LM Studio: ${cpuLayersNow} layers with MoE weights forced onto the CPU.`
                            : ''}
                    </Text>
                </Stack>
            )}
            {plan?.offloadNote && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    This is a mixture-of-experts model, but {plan.offloadNote}.
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
                            `model weights ${gb(plan.weightsBytes - expertsInRam)}`,
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
                                expertsInRam > 0
                                    ? `expert weights of ${layersOnCpu} layers ${gb(expertsInRam)}`
                                    : 'no model weights',
                                `kept free for the system ${gb(plan.ramReserveBytes)}`
                            ].join(' · ')}
                        />
                    )}
                    {plan.spec.maxContextTokens > 0 && (
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            The model supports up to {tokens(plan.spec.maxContextTokens)} tokens
                            {plan.spec.contextSource === 'web'
                                ? `, according to its page (read by ${plan.spec.contextReader})`
                                : ''}
                            .
                        </Text>
                    )}
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
