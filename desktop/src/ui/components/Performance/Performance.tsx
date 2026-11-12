// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
    Badge,
    Button,
    Dropdown,
    Flex,
    Stack,
    Text,
    type DropdownEntry
} from '@nvidia/foundations-react-core'
import type {
    BenchmarkPrompt,
    BenchmarkRun,
    BenchmarkState,
    PerfModelSummary,
    PerformanceReport,
    PerfSource,
    PerfStat
} from '@/shared/types/performance'
import getErrorString from '@/shared/utils/get-error-string'
import { workloadExecutionNodeId } from '@/shared/utils/workloads'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { Stopwatch } from '@/ui/components/icons'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useWorkloadsStore } from '@/ui/stores/workloads.store'
import {
    engineLabel,
    formatCount,
    formatMs,
    formatRate,
    proxyEngineName,
    sourceLabel,
    sparklinePoints,
    summarizeBenchmark
} from './performance-format'
import { groupByNode, liveKey, type LiveModel, type NodePerformance } from './performance-nodes'

// The proxy measures every request it routes and keeps the figures; this page
// reads its report, polling because the proxy pushes no events for it (faster
// while a benchmark runs). What each node is running right now comes from the
// cluster's job list, which is pushed as it changes, so it is live.
const REFRESH_MS = 4000
const REFRESH_RUNNING_MS = 1000

function Card({
    title,
    action,
    children
}: {
    title: React.ReactNode
    action?: React.ReactNode
    children: React.ReactNode
}) {
    return (
        <div className="pair-paper p-4">
            <Stack gap="3">
                <Flex align="center" justify="between" gap="3" wrap="wrap">
                    <Text kind="body/semibold/md">{title}</Text>
                    {action}
                </Flex>
                {children}
            </Stack>
        </div>
    )
}

function Sparkline({ values, label }: { values: number[]; label: string }) {
    const width = 96
    const height = 22
    if (values.length < 2) return null
    return (
        <svg
            className="perf-sparkline"
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label={label}
        >
            <polyline
                points={sparklinePoints(values, width, height)}
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinejoin="round"
            />
        </svg>
    )
}

function useNodeName(): (nodeId: string, local: boolean) => string {
    const nodes = useNodesStore(state => state.nodes)
    return useCallback(
        (nodeId: string, local: boolean) => {
            const name = nodes.get(nodeId)?.name
            if (name) return local ? `${name} (this node)` : name
            return local ? 'This node' : `Node ${nodeId.slice(0, 8)}`
        },
        [nodes]
    )
}

/** Re-renders every second while something is running, so elapsed times tick. */
function useNow(active: boolean): number {
    const [now, setNow] = useState(() => Date.now())
    useEffect(() => {
        if (!active) return
        const timer = setInterval(() => setNow(Date.now()), 1000)
        return () => clearInterval(timer)
    }, [active])
    return now
}

function Figure({
    value,
    detail,
    source
}: {
    value: string
    detail?: string
    source?: PerfSource
}) {
    const hint = source ? sourceLabel(source) : ''
    return (
        <td title={hint || undefined}>
            <span className="perf-figure">{value}</span>
            {detail && <span className="perf-figure-detail">{detail}</span>}
        </td>
    )
}

function p95(stat: PerfStat | null, format: (v: number) => string): string {
    return stat ? `p95 ${format(stat.p95)}` : ''
}

function LiveChip({ live, now }: { live: LiveModel; now: number }) {
    const parts = [
        live.running > 0 ? `${live.running} running` : '',
        live.queued > 0 ? `${live.queued} queued` : ''
    ].filter(Boolean)
    return (
        <span className="perf-live-chip">
            <span className="perf-live-dot" aria-hidden />
            <strong>{live.model}</strong>
            <span className="text-subtle-color">
                {engineLabel(live.engine)} · {parts.join(', ')} ·{' '}
                {formatMs(Math.max(0, now - live.since))} ·{' '}
                {live.firstTokenMs === null
                    ? 'waiting for the first token'
                    : `time to first token ${formatMs(live.firstTokenMs)}`}
            </span>
        </span>
    )
}

function ModelTable({
    models,
    liveModels,
    benchmarkDisabled,
    onBenchmark
}: {
    models: PerfModelSummary[]
    liveModels: Set<string>
    benchmarkDisabled: boolean
    onBenchmark: (engine: string, model: string) => void
}) {
    return (
        <div className="perf-table-wrap">
            <table className="perf-table perf-model-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Requests</th>
                        <th>Tokens in / out</th>
                        <th>Reading</th>
                        <th>Writing</th>
                        <th>Time to first token</th>
                        <th>Thinking / request</th>
                        <th>Writing trend</th>
                        <th aria-label="Actions" />
                    </tr>
                </thead>
                <tbody>
                    {models.map(m => {
                        const thinkingTokens =
                            m.thinkingRequests > 0
                                ? Math.round(m.thinkingTokens / m.thinkingRequests)
                                : null
                        const trend = m.trend.flatMap(p =>
                            p.writeTokensPerSecond === null ? [] : [p.writeTokensPerSecond]
                        )
                        return (
                            <tr key={liveKey(m.engine, m.model)}>
                                <td>
                                    <Flex align="center" gap="2">
                                        {liveModels.has(liveKey(m.engine, m.model)) && (
                                            <span
                                                className="perf-live-dot"
                                                title="Running now"
                                                aria-label="Running now"
                                            />
                                        )}
                                        <span className="perf-figure">{m.model}</span>
                                    </Flex>
                                    <span className="perf-figure-detail">
                                        {engineLabel(m.engine)}
                                    </span>
                                </td>
                                <Figure
                                    value={formatCount(m.requests)}
                                    detail={`last ${new Date(m.lastAt).toLocaleTimeString()}`}
                                />
                                <Figure
                                    value={`${formatCount(m.promptTokens)} / ${formatCount(m.completionTokens)}`}
                                    source={m.sources.tokens}
                                />
                                <Figure
                                    value={`${formatRate(m.readTokensPerSecond?.avg)} tok/s`}
                                    detail={p95(m.readTokensPerSecond, formatRate)}
                                    source={m.sources.read}
                                />
                                <Figure
                                    value={`${formatRate(m.writeTokensPerSecond?.avg)} tok/s`}
                                    detail={p95(m.writeTokensPerSecond, formatRate)}
                                    source={m.sources.write}
                                />
                                <Figure
                                    value={formatMs(m.timeToFirstTokenMs?.p50)}
                                    detail={p95(m.timeToFirstTokenMs, formatMs)}
                                    source={m.sources.ttft}
                                />
                                <Figure
                                    value={m.thinkingMs ? formatMs(m.thinkingMs.avg) : 'None'}
                                    detail={
                                        m.thinkingMs
                                            ? `${m.thinkingRequests} of ${m.requests}` +
                                              (thinkingTokens === null
                                                  ? ''
                                                  : ` · ~${formatCount(thinkingTokens)} tokens`)
                                            : ''
                                    }
                                    source={m.thinkingMs ? m.sources.thinking : undefined}
                                />
                                <td>
                                    <Sparkline
                                        values={trend}
                                        label={`${m.model} writing speed, recent requests`}
                                    />
                                </td>
                                <td>
                                    <Button
                                        kind="tertiary"
                                        size="small"
                                        disabled={benchmarkDisabled}
                                        onClick={() => onBenchmark(m.engine, m.model)}
                                    >
                                        Benchmark
                                    </Button>
                                </td>
                            </tr>
                        )
                    })}
                </tbody>
            </table>
        </div>
    )
}

function NodeCard({
    node,
    displayName,
    now,
    benchmarkDisabled,
    onBenchmark
}: {
    node: NodePerformance
    displayName: string
    now: number
    benchmarkDisabled: boolean
    onBenchmark: (engine: string, model: string) => void
}) {
    const liveModels = new Set(node.live.map(l => liveKey(l.engine, l.model)))
    return (
        <Card
            title={
                <Flex align="center" gap="2" wrap="wrap">
                    <span>{displayName}</span>
                    <Badge color={node.live.length > 0 ? 'green' : 'gray'} kind="solid">
                        {node.live.length > 0 ? 'Busy' : 'Idle'}
                    </Badge>
                </Flex>
            }
            action={
                node.requests > 0 ? (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {formatCount(node.requests)} requests · {formatCount(node.promptTokens)}{' '}
                        tokens in · {formatCount(node.completionTokens)} out
                    </Text>
                ) : undefined
            }
        >
            <Stack gap="2">
                <Flex align="center" gap="2" wrap="wrap">
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        Running now:
                    </Text>
                    {node.live.length === 0 ? (
                        <Text kind="body/regular/sm">nothing</Text>
                    ) : (
                        node.live.map(l => (
                            <LiveChip key={liveKey(l.engine, l.model)} live={l} now={now} />
                        ))
                    )}
                </Flex>
                {node.loaded.length > 0 && (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        Loaded in memory:{' '}
                        {node.loaded.map(m => `${m.model} (${engineLabel(m.engine)})`).join(', ')}
                    </Text>
                )}
            </Stack>
            {node.models.length > 0 ? (
                <ModelTable
                    models={node.models}
                    liveModels={liveModels}
                    benchmarkDisabled={benchmarkDisabled}
                    onBenchmark={onBenchmark}
                />
            ) : (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    No requests routed through this node have run here yet.
                </Text>
            )}
        </Card>
    )
}

const STATE_COLORS: Record<BenchmarkState, 'green' | 'yellow' | 'red' | 'gray'> = {
    pending: 'gray',
    running: 'yellow',
    done: 'green',
    failed: 'red',
    skipped: 'gray'
}

const STATE_LABELS: Record<BenchmarkState, string> = {
    pending: 'Waiting',
    running: 'Running',
    done: 'Done',
    failed: 'Failed',
    skipped: 'Skipped'
}

function BenchmarkRunCard({
    run,
    nodeName
}: {
    run: BenchmarkRun
    nodeName: (nodeId: string, local: boolean) => string
}) {
    const summary = summarizeBenchmark(run)
    const firstResult = run.steps.find(s => s.result)?.result ?? null
    return (
        <div className="pair-paper p-3">
            <Stack gap="2">
                <Flex align="center" justify="between" gap="3" wrap="wrap">
                    <Flex align="center" gap="2" wrap="wrap">
                        <Text kind="body/semibold/sm">{run.model}</Text>
                        <Badge color="gray" kind="outline">
                            {engineLabel(run.engine)}
                        </Badge>
                        {run.nodeId ? (
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                on {nodeName(run.nodeId, firstResult?.local ?? false)}
                            </Text>
                        ) : (
                            firstResult && (
                                <Text kind="body/regular/sm" className="text-subtle-color">
                                    {nodeName(firstResult.nodeId, firstResult.local)}
                                </Text>
                            )
                        )}
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            {new Date(run.startedAt).toLocaleString()}
                        </Text>
                    </Flex>
                    <Badge color={STATE_COLORS[run.state]} kind="solid">
                        {STATE_LABELS[run.state]}
                    </Badge>
                </Flex>
                {run.state !== 'running' && (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        Reading {formatRate(summary.readTokensPerSecond)} tokens/s · writing{' '}
                        {formatRate(summary.writeTokensPerSecond)} tokens/s · time to first token{' '}
                        {formatMs(summary.timeToFirstTokenMs)}
                        {summary.thinkingMs !== null
                            ? ` · thinking ${formatMs(summary.thinkingMs)}`
                            : ''}
                    </Text>
                )}
                {run.error && (
                    <Text kind="body/regular/sm" className="text-feedback-danger">
                        {run.error}
                    </Text>
                )}
                <div className="perf-table-wrap">
                    <table className="perf-table">
                        <thead>
                            <tr>
                                <th>Step</th>
                                <th>Thinking</th>
                                <th>Tokens in / out</th>
                                <th>Time to first token</th>
                                <th>Reading</th>
                                <th>Writing</th>
                                <th>Thought for</th>
                                <th>Total</th>
                            </tr>
                        </thead>
                        <tbody>
                            {run.steps.map(step => {
                                const r = step.result
                                return (
                                    <tr key={step.name}>
                                        <td>{step.label}</td>
                                        <td>{step.thinking}</td>
                                        {r ? (
                                            <>
                                                <td>
                                                    {formatCount(r.promptTokens)} /{' '}
                                                    {formatCount(r.completionTokens)}
                                                </td>
                                                <td>{formatMs(r.timeToFirstTokenMs)}</td>
                                                <td>{formatRate(r.readTokensPerSecond)}/s</td>
                                                <td>{formatRate(r.writeTokensPerSecond)}/s</td>
                                                <td>
                                                    {r.thinking
                                                        ? `${formatMs(r.thinkingMs)} (~${formatCount(r.thinkingTokens ?? 0)} tokens)`
                                                        : '—'}
                                                </td>
                                                <td>{formatMs(r.totalMs)}</td>
                                            </>
                                        ) : (
                                            <td colSpan={6} className="text-subtle-color">
                                                {STATE_LABELS[step.state]}
                                                {step.detail ? `: ${step.detail}` : ''}
                                            </td>
                                        )}
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            </Stack>
        </div>
    )
}

interface BenchmarkTarget {
    engine: string
    model: string
    /** The node every step runs on. */
    nodeId: string
}

function targetKey(t: BenchmarkTarget): string {
    return `${t.nodeId}:${t.engine}:${t.model}`
}

function BenchmarkCard({
    report,
    targets,
    selfId,
    running,
    starting,
    onRun,
    nodeName
}: {
    report: PerformanceReport
    targets: BenchmarkTarget[]
    selfId: string | null
    running: boolean
    starting: boolean
    onRun: (target: BenchmarkTarget) => void
    nodeName: (nodeId: string, local: boolean) => string
}) {
    const [selected, setSelected] = useState<BenchmarkTarget | null>(null)
    const target =
        (selected && targets.find(t => targetKey(t) === targetKey(selected))) ?? targets[0] ?? null
    const label = (t: BenchmarkTarget) =>
        `${t.model} · ${engineLabel(t.engine)} · ${nodeName(t.nodeId, t.nodeId === selfId)}`
    const items: DropdownEntry[] = targets.map(t => ({
        children: label(t),
        onSelect: () => setSelected(t)
    }))
    return (
        <Card
            title="Model benchmark"
            action={
                <Flex align="center" gap="2">
                    {targets.length > 0 && (
                        <Dropdown
                            items={items}
                            size="small"
                            aria-label="Choose a model and the node to run it on"
                        >
                            {target ? label(target) : ''}
                        </Dropdown>
                    )}
                    <Button
                        kind="primary"
                        color="brand"
                        size="small"
                        disabled={!target || running || starting}
                        onClick={() => target && onRun(target)}
                    >
                        {running ? 'Running…' : 'Run benchmark'}
                    </Button>
                </Flex>
            }
        >
            <Text kind="body/regular/sm" className="text-subtle-color">
                A per-model test: sends the same {report.benchmarkPrompts.length || 'fixed'}{' '}
                requests to the chosen model on the chosen node, one at a time, so models and nodes
                compare like for like. Every step runs on that node, this one or another in the
                cluster, or fails if it cannot serve the model. It uses the model — loading it if
                needed — and waits for other requests routed from here to finish before each step.
                Its figures are kept apart from the live ones above.
            </Text>
            <PromptList prompts={report.benchmarkPrompts} />
            {report.benchmarks.length === 0 ? (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    No benchmark has run yet.
                </Text>
            ) : (
                <Stack gap="2">
                    {report.benchmarks.map(run => (
                        <BenchmarkRunCard key={run.id} run={run} nodeName={nodeName} />
                    ))}
                </Stack>
            )}
        </Card>
    )
}

function PromptList({ prompts }: { prompts: BenchmarkPrompt[] }) {
    if (prompts.length === 0) return null
    return (
        <ul className="perf-prompt-list">
            {prompts.map(p => (
                <li key={p.name}>
                    <Text kind="body/regular/sm">
                        <strong>{p.label}</strong>{' '}
                        <span className="text-subtle-color">
                            — {p.description} Thinking {p.thinking}, up to {p.maxTokens} tokens.
                        </span>
                    </Text>
                </li>
            ))}
        </ul>
    )
}

export default function Performance() {
    const [report, setReport] = useState<PerformanceReport | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [starting, setStarting] = useState(false)
    const nodeName = useNodeName()
    const nodes = useNodesStore(state => state.nodes)
    const selfId = useConnectionStore(state => state.selfId)
    const engineModels = useEngineModelsStore(state => state.models)
    const workloads = useWorkloadsStore(state => state.workloads)

    const running = report?.benchmarks.some(r => r.state === 'running') ?? false

    const refresh = useCallback(() => {
        window.pairApi.performance
            .getReport()
            .then(setReport)
            .catch(err => setError(getErrorString(err) ?? 'The router is not reachable'))
    }, [])

    useEffect(() => {
        refresh()
        const timer = setInterval(refresh, running ? REFRESH_RUNNING_MS : REFRESH_MS)
        return () => clearInterval(timer)
    }, [refresh, running])

    const grouped = useMemo(
        () =>
            groupByNode({
                nodes: [...nodes.values()].map(n => ({ id: n.id, name: n.name })),
                selfId,
                summaries: report?.models ?? [],
                jobs: [...workloads.values()].map(w => ({
                    nodeId: workloadExecutionNodeId(w),
                    model: w.model,
                    engine: w.engine,
                    state: w.state,
                    createdAt: w.createdAt,
                    startedAt: w.startedAt
                })),
                loaded: [...engineModels.values()].flatMap(list =>
                    list.models
                        .filter(m => m.status === 'loaded')
                        .map(m => ({ nodeId: list.nodeId, engine: list.engineType, model: m.name }))
                )
            }),
        [nodes, selfId, report, workloads, engineModels]
    )
    const now = useNow(grouped.some(n => n.live.length > 0))

    // The benchmark is per model and node: each model a node has downloaded,
    // or has served traffic routed from here, can be run on that node.
    const targets = useMemo(() => {
        const seen = new Map<string, BenchmarkTarget>()
        for (const list of engineModels.values()) {
            const engine = proxyEngineName(list.engineType)
            if (!engine) continue
            for (const m of list.models) {
                if (!m.downloaded) continue
                const t = { engine, model: m.name, nodeId: list.nodeId }
                seen.set(targetKey(t), t)
            }
        }
        for (const m of report?.models ?? []) {
            const t = { engine: m.engine, model: m.model, nodeId: m.nodeId }
            seen.set(targetKey(t), t)
        }
        // Models that have served traffic here come first, then models loaded
        // somewhere, then this node's, so the default is one already in use
        // rather than the first name in the list.
        const used = new Set(
            (report?.models ?? []).map(m =>
                targetKey({ engine: m.engine, model: m.model, nodeId: m.nodeId })
            )
        )
        const loaded = new Set(
            [...engineModels.values()].flatMap(list => {
                const engine = proxyEngineName(list.engineType)
                return engine
                    ? list.models
                          .filter(m => m.status === 'loaded')
                          .map(m => targetKey({ engine, model: m.name, nodeId: list.nodeId }))
                    : []
            })
        )
        const rank = (t: BenchmarkTarget) =>
            used.has(targetKey(t)) ? 0 : loaded.has(targetKey(t)) ? 1 : t.nodeId === selfId ? 2 : 3
        return [...seen.values()].sort(
            (a, b) =>
                rank(a) - rank(b) ||
                a.model.localeCompare(b.model) ||
                a.nodeId.localeCompare(b.nodeId)
        )
    }, [engineModels, report, selfId])

    const runBenchmark = (target: BenchmarkTarget) => {
        setStarting(true)
        window.pairApi.performance
            .runBenchmark(target.engine, target.model, target.nodeId)
            .then(refresh)
            .catch(err => setError(getErrorString(err) ?? 'Could not start the benchmark'))
            .finally(() => setStarting(false))
    }

    const reset = () => {
        window.pairApi.performance
            .clear()
            .then(refresh)
            .catch(err => setError(getErrorString(err) ?? 'Could not reset the figures'))
    }

    return (
        <Stack className="grow overflow-y-auto w-full px-4">
            <Stack gap="4" className="py-4 perf-page">
                {error && <InlineErrorBanner message={error} onClose={() => setError(null)} />}
                <Card
                    title={
                        <Flex align="center" gap="2">
                            <Stopwatch style={{ fontSize: 18 }} />
                            <span>Performance by node</span>
                        </Flex>
                    }
                    action={
                        <Button
                            kind="tertiary"
                            size="small"
                            onClick={reset}
                            disabled={!report || report.models.length === 0}
                        >
                            Reset figures
                        </Button>
                    }
                >
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        Each node shows what it is running right now and how fast each of its models
                        has worked, over the model’s last {report?.window ?? 100} requests that this
                        node routed. Reading is prompt tokens processed per second, writing is
                        tokens generated per second, and thinking is the time a reasoning model
                        spent before its answer. Hover a figure to see whether the engine reported
                        it or PAIR measured it. Only counts and timings are kept — never prompt or
                        reply text — and they reset when the service restarts.
                    </Text>
                </Card>
                {grouped.map(n => (
                    <NodeCard
                        key={n.nodeId}
                        node={n}
                        displayName={
                            n.name
                                ? n.local
                                    ? `${n.name} (this node)`
                                    : n.name
                                : nodeName(n.nodeId, n.local)
                        }
                        now={now}
                        benchmarkDisabled={running || starting}
                        onBenchmark={(engine, model) =>
                            runBenchmark({ engine, model, nodeId: n.nodeId })
                        }
                    />
                ))}
                {report && (
                    <BenchmarkCard
                        report={report}
                        targets={targets}
                        selfId={selfId}
                        running={running}
                        starting={starting}
                        onRun={runBenchmark}
                        nodeName={nodeName}
                    />
                )}
            </Stack>
        </Stack>
    )
}
