// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import {
    Button,
    Flex,
    ModalContent,
    ModalDialog,
    ModalHeading,
    ModalRoot,
    Stack,
    Text
} from '@nvidia/foundations-react-core'
import { OpenInNew } from '@/ui/components/icons'
import type { EngineType } from '@/shared/types/engines'
import type { ModelDetails } from '@/shared/types/model-details'
import { detailRows } from '@/ui/utils/model-details'

function Chips({ items }: { items: string[] }) {
    return (
        <Flex gap="1" wrap="wrap">
            {items.map(item => (
                <span
                    key={item}
                    className="rounded px-2 py-0.5 text-xs"
                    style={{ border: '1px solid #ffffff33' }}
                >
                    {item}
                </span>
            ))}
        </Flex>
    )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
    return (
        <Stack gap="1">
            <Text kind="body/semibold/sm">{title}</Text>
            {children}
        </Stack>
    )
}

function Preformatted({ text }: { text: string }) {
    return (
        <pre
            className="text-xs whitespace-pre-wrap rounded p-2 m-0"
            style={{ maxHeight: 200, overflowY: 'auto', background: '#ffffff0d' }}
        >
            {text}
        </pre>
    )
}

/**
 * "About this model": what the engine that has the model, and the model's
 * file, say about it — description, tags, capabilities, size, licence and
 * links to its pages. Only reported facts are shown.
 */
export function ModelDetailsDialog({
    engineType,
    nodeId,
    model,
    displayName,
    onClose
}: {
    engineType: EngineType
    nodeId: string
    model: string
    displayName: string
    onClose: () => void
}) {
    const [details, setDetails] = useState<ModelDetails | null>(null)

    useEffect(() => {
        let live = true
        void window.pairApi.engines.getModelDetails(engineType, nodeId, model).then(d => {
            if (live) setDetails(d)
        })
        return () => {
            live = false
        }
    }, [engineType, nodeId, model])

    const rows = details ? detailRows(details) : []
    const reported =
        details !== null &&
        (rows.length > 0 ||
            details.description !== '' ||
            details.tags.length > 0 ||
            details.capabilities.length > 0 ||
            details.links.length > 0)

    return (
        <ModalRoot open onOpenChange={open => !open && onClose()}>
            <ModalDialog>
                <ModalContent
                    className="no-drag-elements"
                    style={{ width: 'min(720px, 94vw)', maxHeight: '88vh', overflowY: 'auto' }}
                >
                    <ModalHeading>About {displayName}</ModalHeading>
                    <Stack gap="4" className="-mt-2">
                        {!details && (
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                Asking the engine about this model…
                            </Text>
                        )}
                        {details && details.description && (
                            <Text kind="body/regular/sm">{details.description}</Text>
                        )}
                        {details && details.links.length > 0 && (
                            <Flex gap="2" wrap="wrap">
                                {details.links.map(link => (
                                    <Button
                                        key={link.url}
                                        size="small"
                                        kind="secondary"
                                        title={`${link.url} (from ${link.source})`}
                                        onClick={() =>
                                            void window.windowApi.window.openExternal(link.url)
                                        }
                                    >
                                        {link.label}
                                        <OpenInNew style={{ fontSize: 12 }} className="ml-1" />
                                    </Button>
                                ))}
                            </Flex>
                        )}
                        {details && details.capabilities.length > 0 && (
                            <Section title="Capabilities">
                                <Chips items={details.capabilities} />
                            </Section>
                        )}
                        {details && details.tags.length > 0 && (
                            <Section title="Tags">
                                <Chips items={details.tags} />
                            </Section>
                        )}
                        {rows.length > 0 && (
                            <dl
                                className="grid gap-x-4 gap-y-1 m-0"
                                style={{ gridTemplateColumns: 'max-content 1fr' }}
                            >
                                {rows.map(([label, value]) => (
                                    <div key={label} className="contents">
                                        <dt>
                                            <Text
                                                kind="body/regular/sm"
                                                className="text-subtle-color"
                                            >
                                                {label}
                                            </Text>
                                        </dt>
                                        <dd className="m-0 min-w-0 break-words">
                                            <Text kind="body/regular/sm">{value}</Text>
                                        </dd>
                                    </div>
                                ))}
                            </dl>
                        )}
                        {details && details.parameters && (
                            <Section title="Default generation settings">
                                <Preformatted text={details.parameters} />
                            </Section>
                        )}
                        {details && details.licenseText && (
                            <Section title="License text">
                                <Preformatted text={details.licenseText} />
                            </Section>
                        )}
                        {details && !reported && details.notes.length === 0 && (
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                The engine reported nothing more about this model.
                            </Text>
                        )}
                        {details &&
                            details.notes.map(note => (
                                <Text
                                    key={note}
                                    kind="body/regular/sm"
                                    className="text-subtle-color"
                                >
                                    {note}
                                </Text>
                            ))}
                        <Flex justify="end">
                            <Button size="small" kind="primary" color="brand" onClick={onClose}>
                                Close
                            </Button>
                        </Flex>
                    </Stack>
                </ModalContent>
            </ModalDialog>
        </ModalRoot>
    )
}
