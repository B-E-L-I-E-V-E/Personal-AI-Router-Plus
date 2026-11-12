// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Button, Flex, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { EngineType } from '@/shared/types/engines'
import type { ModelDetails, WebFactsStatus } from '@/shared/types/model-details'
import {
    engineGaps,
    pageHost,
    webFactLabel,
    webFactText,
    webFillRows
} from '@/ui/utils/model-details'

const POLL_MS = 1000

/**
 * "Look up on the web": a small model on this node's Ollama reads the model's
 * own pages (the links above) and fills what the engine left empty. The values
 * stay in their own section, labelled with the page and the reading model;
 * the engine's answer always wins.
 */
export function ModelWebFacts({
    engineType,
    nodeId,
    model,
    details
}: {
    engineType: EngineType
    nodeId: string
    model: string
    details: ModelDetails
}) {
    const [status, setStatus] = useState<WebFactsStatus | null>(null)
    const [reader, setReader] = useState('')
    const running = status?.job?.state === 'running'

    useEffect(() => {
        let live = true
        void window.pairApi.engines.getModelWebFacts(engineType, nodeId, model).then(s => {
            if (!live) return
            setStatus(s)
            setReader(r => r || s.readerModel)
        })
        return () => {
            live = false
        }
    }, [engineType, nodeId, model])

    useEffect(() => {
        if (!running) return
        const timer = setInterval(() => {
            void window.pairApi.engines.getModelWebFacts(engineType, nodeId, model).then(setStatus)
        }, POLL_MS)
        return () => clearInterval(timer)
    }, [running, engineType, nodeId, model])

    const gaps = engineGaps(details)
    if (details.links.length === 0 || (gaps.length === 0 && !status?.result)) return null

    const start = () => {
        void window.pairApi.engines
            .startModelWebFacts(engineType, nodeId, model, reader)
            .then(setStatus)
    }
    const result = status?.result ?? null
    const fills = result ? webFillRows(result.facts) : []
    const alreadyReported = result ? result.facts.length - fills.length : 0
    const job = status?.job ?? null

    return (
        <Stack gap="2" className="rounded p-3" style={{ border: '1px dashed #ffffff40' }}>
            <Text kind="body/semibold/sm">From the model’s page</Text>
            <Text kind="body/regular/sm" className="text-subtle-color">
                {gaps.length > 0
                    ? `The engine did not report: ${gaps.join(', ').toLowerCase()}. `
                    : ''}
                PAIR can read the model’s page with a model on this node’s Ollama and fill in what
                the page says. Values from the page are labelled and never replace what the engine
                reported.
            </Text>
            <Flex gap="2" align="end" wrap="wrap">
                <FormField slotLabel="Reader model (Ollama)">
                    <TextInput value={reader} onValueChange={setReader} disabled={running} />
                </FormField>
                <Button
                    size="small"
                    kind="secondary"
                    disabled={running || reader.trim() === ''}
                    onClick={start}
                >
                    {result ? 'Look up again' : 'Look up on the web'}
                </Button>
            </Flex>
            {running && job && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {job.phase || 'Reading…'}
                </Text>
            )}
            {job?.state === 'failed' && (
                <Text kind="body/regular/sm" style={{ color: 'var(--text-color-feedback-danger)' }}>
                    {job.error}
                </Text>
            )}
            {status?.error && (
                <Text kind="body/regular/sm" style={{ color: 'var(--text-color-feedback-danger)' }}>
                    {status.error}
                </Text>
            )}
            {result && (
                <Stack gap="2">
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        Read by {result.reader} from{' '}
                        {result.pages.map(p => pageHost(p.url)).join(', ')}
                        {result.readAt > 0 ? `, ${new Date(result.readAt).toLocaleString()}` : ''}.
                        Check these on the page before relying on them.
                    </Text>
                    {fills.length > 0 ? (
                        <dl
                            className="grid gap-x-4 gap-y-1 m-0"
                            style={{ gridTemplateColumns: 'max-content 1fr' }}
                        >
                            {fills.map(row => (
                                <div key={row.label} className="contents">
                                    <dt>
                                        <Text kind="body/regular/sm" className="text-subtle-color">
                                            {row.label}
                                        </Text>
                                    </dt>
                                    <dd className="m-0 min-w-0 break-words">
                                        <Text kind="body/regular/sm" title={`from ${row.source}`}>
                                            {row.text}
                                        </Text>
                                    </dd>
                                </div>
                            ))}
                        </dl>
                    ) : (
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            The page did not say anything the engine left out.
                        </Text>
                    )}
                    {alreadyReported > 0 && (
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            {alreadyReported === 1
                                ? 'One more value on the page is'
                                : `${alreadyReported} more values on the page are`}{' '}
                            already reported by the engine, whose answer is shown above.
                        </Text>
                    )}
                    {result.conflicts.map(c => (
                        <Text key={c.field} kind="body/regular/sm" className="text-subtle-color">
                            {webFactLabel(c.field)} is left empty: the page gives{' '}
                            {c.values.map(v => webFactText(c.field, v.value)).join(' and ')}.
                        </Text>
                    ))}
                    {[
                        ...result.pages
                            .filter(p => p.note)
                            .map(p => `${pageHost(p.url)}: ${p.note}`),
                        ...result.notes
                    ].map(note => (
                        <Text key={note} kind="body/regular/sm" className="text-subtle-color">
                            {note}
                        </Text>
                    ))}
                </Stack>
            )}
        </Stack>
    )
}
