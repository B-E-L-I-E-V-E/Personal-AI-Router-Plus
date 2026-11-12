// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelDetails, WebFact, WebFactValue } from '@/shared/types/model-details'
import { formatBytes } from '@/ui/utils/formatters'

/** "751,632,384" as "752M", the way model cards write parameter counts. */
function formatCount(n: number): string {
    if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`
    if (n >= 1e6) return `${Math.round(n / 1e6)}M`
    return n.toLocaleString()
}

function formatDate(raw: string): string {
    const t = Date.parse(raw)
    return Number.isNaN(t) ? raw : new Date(t).toLocaleString()
}

/** The facts the engine reported, in reading order; empty ones are left out. */
export function detailRows(d: ModelDetails): [string, string][] {
    const rows: [string, string | false][] = [
        ['Name', d.name],
        ['Author', d.author],
        ['Publisher', d.publisher],
        ['Type', d.type],
        ['Family', d.family],
        ['Architecture', d.architecture],
        [
            'Parameters',
            [d.parameterSize, d.parameterCount > 0 && formatCount(d.parameterCount)]
                .filter(Boolean)
                .filter((v, i, all) => all.indexOf(v) === i)
                .join(' · ')
        ],
        ['Quantization', d.quantization],
        ['Format', d.format],
        ['Trained context', d.contextLength > 0 && `${d.contextLength.toLocaleString()} tokens`],
        ['Loaded context', d.loadedContext > 0 && `${d.loadedContext.toLocaleString()} tokens`],
        ['Size on disk', d.sizeBytes > 0 && formatBytes(d.sizeBytes)],
        ['Languages', d.languages.join(', ')],
        ['License', d.license],
        ['Modified', d.modifiedAt && formatDate(d.modifiedAt)],
        ['Digest', d.digest && d.digest.slice(0, 12)]
    ]
    return rows.filter((r): r is [string, string] => typeof r[1] === 'string' && r[1] !== '')
}

/** Engine fields a model's page can fill, and their labels. */
const FILLABLE: [string, (d: ModelDetails) => boolean][] = [
    ['Name', d => d.name === ''],
    ['Description', d => d.description === ''],
    ['Parameters', d => d.parameterSize === '' && d.parameterCount === 0],
    ['Size', d => d.sizeBytes === 0],
    ['Context', d => d.contextLength === 0],
    ['Capabilities', d => d.capabilities.length === 0]
]

/** What the engine left empty that a model's page may say. */
export function engineGaps(d: ModelDetails): string[] {
    return FILLABLE.filter(([, empty]) => empty(d)).map(([label]) => label)
}

const WEB_LABELS: Record<string, string> = {
    name: 'Name',
    description: 'Description',
    version: 'Version',
    parameterSize: 'Parameters',
    dataSize: 'Download size',
    context: 'Context',
    vision: 'Image input',
    tools: 'Tool calling',
    thinking: 'Thinking mode',
    moe: 'Mixture of experts',
    url: 'Model page'
}

export function webFactLabel(field: string): string {
    return WEB_LABELS[field] ?? field
}

export function webFactText(field: string, value: WebFactValue): string {
    if (typeof value === 'boolean') return value ? 'Yes' : 'No'
    if (typeof value === 'number') {
        return field === 'context' ? `${value.toLocaleString()} tokens` : value.toLocaleString()
    }
    return value
}

/** The web values that fill a gap the engine left, as label, text and page. */
export function webFillRows(facts: WebFact[]): { label: string; text: string; source: string }[] {
    return facts
        .filter(f => f.fills !== '')
        .map(f => ({
            label: webFactLabel(f.field),
            text: webFactText(f.field, f.value),
            source: f.source
        }))
}

/** "ollama.com" for a page address, to name where a value came from. */
export function pageHost(url: string): string {
    try {
        return new URL(url).host
    } catch {
        return url
    }
}
