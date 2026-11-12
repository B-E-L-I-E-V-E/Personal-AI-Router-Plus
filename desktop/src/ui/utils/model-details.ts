// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelDetails } from '@/shared/types/model-details'
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
