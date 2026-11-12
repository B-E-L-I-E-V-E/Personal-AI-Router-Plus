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
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { Stopwatch } from '@/ui/components/icons'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
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

// The proxy measures every request it routes and keeps the figures; this page
// only reads its report. It polls because the proxy pushes no events for
// them, faster while a benchmark is running so its steps appear as they end.
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

function Metric({
    label,
    value,
    detail,
    source
}: {
    label: string
    value: string
    detail?: string
    source?: PerfSource
}) {
    const hint = source ? sourceLabel(source) : ''
    return (
        <Stack gap="1" className="perf-metric" title={hint || undefined}>
            <Text kind="body/regular/xs" className="text-subtle-color">
                {label}
            </Text>
            <Text kind="body/semibold/lg">{value}</Text>
            {(detail || hint) && (
                <Text kind="body/regular/xs" className="text-subtle-color">
                    {[detail, hint].filter(Boolean).join(' · ')}
                </Text>
            )}
        </Stack>
    )
}

function statDetail(stat: PerfStat | null, format: (v: number) => string): string {
    if (!stat) return ''
    return `p50 ${format(stat.p50)} · p95 ${format(stat.p95)}`
}

function Sparkline({ values, label }: { values: number[]; label: string }) {
    const width = 160
    const height = 32
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

function ModelCard({
    summary,
    nodeName,
    benchmarkDisabled,
    onBenchmark
}: {
    summary: PerfModelSummary
    nodeName: string
    benchmarkDisabled: boolean
    onBenchmark: () => void
}) {
    const write = summary.writeTokensPerSecond
    const read = summary.readTokensPerSecond
    const ttft = summary.timeToFirstTokenMs
    const thinking = summary.thinkingMs
    const writeTrend = summary.trend.flatMap(p =>
        p.writeTokensPerSecond === null ? [] : [p.writeTokensPerSecond]
    )
    const thinkingPerRequest =
        summary.thinkingRequests > 0
            ? Math.round(summary.thinkingTokens / summary.thinkingRequests)
            : null
    return (
        <Card
            title={
                <Flex align="center" gap="2" wrap="wrap">
                    <span>{summary.model}</span>
                    <Badge color="gray" kind="outline">
                        {engineLabel(summary.engine)}
                    </Badge>
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {nodeName}
                    </Text>
                </Flex>
            }
            action={
                <Button
                    kind="secondary"
                    size="small"
                    disabled={benchmarkDisabled}
                    onClick={onBenchmark}
                >
                    Run benchmark
                </Button>
            }
        >
            <div className="perf-grid">
                <Metric
                    label="Requests"
                    value={formatCount(summary.requests)}
                    detail={`last ${new Date(summary.lastAt).toLocaleTimeString()}`}
                />
                <Metric
                    label="Tokens in / out"
                    value={`${formatCount(summary.promptTokens)} / ${formatCount(summary.completionTokens)}`}
                    source={summary.sources.tokens}
                />
                <Metric
                    label="Reading (tokens/s)"
                    value={formatRate(read?.avg)}
                    detail={statDetail(read, formatRate)}
                    source={summary.sources.read}
                />
                <Metric
                    label="Writing (tokens/s)"
                    value={formatRate(write?.avg)}
                    detail={statDetail(write, formatRate)}
                    source={summary.sources.write}
                />
                <Metric
                    label="Time to first token"
                    value={formatMs(ttft?.p50)}
                    detail={ttft ? `p95 ${formatMs(ttft.p95)}` : ''}
                    source={summary.sources.ttft}
                />
                <Metric
                    label="Thinking per request"
                    value={thinking ? formatMs(thinking.avg) : 'None'}
                    detail={
                        thinking
                            ? `${summary.thinkingRequests} of ${summary.requests} thought` +
                              (thinkingPerRequest === null
                                  ? ''
                                  : ` · ~${formatCount(thinkingPerRequest)} tokens`)
                            : ''
                    }
                    source={thinking ? summary.sources.thinking : undefined}
                />
            </div>
            {writeTrend.length > 1 && (
                <Flex align="center" gap="3" className="text-subtle-color">
                    <Sparkline values={writeTrend} label="Writing speed, recent requests" />
                    <Text kind="body/regular/xs" className="text-subtle-color">
                        Writing speed over the last {writeTrend.length} requests
                    </Text>
                </Flex>
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
                        {firstResult && (
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                {nodeName(firstResult.nodeId, firstResult.local)}
                            </Text>
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
                        {formatRate(summary.writeTokensPerSecond)} tokens/s · first token{' '}
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
                                <th>First token</th>
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
}

function targetKey(t: BenchmarkTarget): string {
    return `${t.engine}:${t.model}`
}

function BenchmarkCard({
    report,
    targets,
    running,
    starting,
    onRun,
    nodeName
}: {
    report: PerformanceReport
    targets: BenchmarkTarget[]
    running: boolean
    starting: boolean
    onRun: (target: BenchmarkTarget) => void
    nodeName: (nodeId: string, local: boolean) => string
}) {
    const [selected, setSelected] = useState<BenchmarkTarget | null>(null)
    const target = selected ?? targets[0] ?? null
    const items: DropdownEntry[] = targets.map(t => ({
        children: `${t.model} · ${engineLabel(t.engine)}`,
        onSelect: () => setSelected(t)
    }))
    return (
        <Card
            title="Benchmark"
            action={
                <Flex align="center" gap="2">
                    {targets.length > 0 && (
                        <Dropdown items={items} size="small" aria-label="Choose a model">
                            {target ? `${target.model} · ${engineLabel(target.engine)}` : ''}
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
                Sends {report.benchmarkPrompts.length || 'a fixed set of'} requests to the model
                through PAIR, one at a time, and records the same figures as real traffic. It uses
                the model — loading it if needed — and waits for other requests on this node to
                finish before each step. Its figures are kept apart from the live ones above.
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
    const engineModels = useEngineModelsStore(state => state.models)

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

    // Anything the cluster has downloaded can be benchmarked, as can any model
    // that has served traffic here.
    const targets = useMemo(() => {
        const seen = new Map<string, BenchmarkTarget>()
        for (const list of engineModels.values()) {
            const engine = proxyEngineName(list.engineType)
            if (!engine) continue
            for (const m of list.models) {
                if (!m.downloaded) continue
                const t = { engine, model: m.name }
                seen.set(targetKey(t), t)
            }
        }
        for (const m of report?.models ?? []) {
            const t = { engine: m.engine, model: m.model }
            seen.set(targetKey(t), t)
        }
        return [...seen.values()].sort((a, b) => a.model.localeCompare(b.model))
    }, [engineModels, report])

    const runBenchmark = (target: BenchmarkTarget) => {
        setStarting(true)
        window.pairApi.performance
            .runBenchmark(target.engine, target.model)
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
                            <span>Performance per model</span>
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
                        Speeds for the requests this node routes, on any node that serves them, over
                        each model’s last {report?.window ?? 100} requests. Reading is prompt tokens
                        processed per second, writing is tokens generated per second, and thinking
                        is the time a reasoning model spent before its answer. Hover a figure to see
                        whether the engine reported it or PAIR measured it. Requests other nodes
                        route are measured on those nodes. Only counts and timings are kept — never
                        prompt or reply text — and they reset when the service restarts.
                    </Text>
                    {report && report.models.length === 0 && (
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            No requests yet. Send one through PAIR, or run the benchmark below.
                        </Text>
                    )}
                </Card>
                {report?.models.map(m => (
                    <ModelCard
                        key={`${m.engine}:${m.model}:${m.nodeId}`}
                        summary={m}
                        nodeName={nodeName(m.nodeId, m.local)}
                        benchmarkDisabled={running || starting}
                        onBenchmark={() => runBenchmark({ engine: m.engine, model: m.model })}
                    />
                ))}
                {report && (
                    <BenchmarkCard
                        report={report}
                        targets={targets}
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
