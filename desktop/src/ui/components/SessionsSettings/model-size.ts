// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CapacityPlan, KvCacheType, ModelSpec } from '@/shared/types/sessions'
import { perTokenLabel, sizeLabel } from './capacity-sliders'

/**
 * What the planner's two model figures mean and where they came from, in
 * words. The planner counts the memory the weights take once loaded, which
 * is not always the download size, and the context cache each token of a
 * session keeps.
 */

const DOWNLOAD_SOURCE: Record<string, string> = {
    'ollama-tags': 'Ollama’s model list',
    'lmstudio-models': 'LM Studio’s model list',
    'gguf-files': 'the GGUF files'
}

const CACHE_LABEL: Record<KvCacheType, string> = {
    f16: 'full precision (f16)',
    q8_0: '8-bit',
    q4_0: '4-bit'
}

/** One line per size the model reports: download, in memory, loaded now. */
export function sizeFacts(spec: ModelSpec): string[] {
    const lines: string[] = []
    if (spec.downloadBytes > 0) {
        const from = DOWNLOAD_SOURCE[spec.downloadSource]
        lines.push(`Download size ${sizeLabel(spec.downloadBytes)}${from ? ` (${from})` : ''}`)
    }
    lines.push(
        spec.tensorBytes > 0
            ? `In memory when loaded ≈ ${sizeLabel(spec.tensorBytes)} (its weight tensors added up)`
            : `In memory when loaded: not reported${spec.format ? ` for ${spec.format} models` : ''}`
    )
    return lines
}

/** What a copy loaded now takes, which is more than its weights. */
export function loadedFact(spec: ModelSpec): string {
    if (spec.loadedBytes <= 0) return ''
    const gpu =
        spec.loadedVramBytes > 0 && spec.loadedVramBytes < spec.loadedBytes
            ? `, ${sizeLabel(spec.loadedVramBytes)} of it on the GPU`
            : ''
    const context =
        spec.loadedContextTokens > 0
            ? ` with a ${spec.loadedContextTokens.toLocaleString()}-token context`
            : ''
    return `Loaded now: ${sizeLabel(spec.loadedBytes)}${gpu}${context}. That is the weights, their context cache and the engine’s working buffers together.`
}

/** Which size the plan counts, and why. */
export function weightsUsed(plan: CapacityPlan): string {
    const spec = plan.spec
    switch (plan.weightsSource) {
        case 'manual':
            return `The plan uses the ${sizeLabel(plan.weightsBytes)} you set${
                spec.weightsBytes > 0 ? ` (the model’s: ${sizeLabel(spec.weightsBytes)})` : ''
            }.`
        case 'tensors':
            return 'The plan uses the in-memory size.'
        case 'download':
            return spec.tensorBytes > 0
                ? 'The plan uses the download size, which holds more than the weight tensors and may all load.'
                : 'The plan uses the download size as an estimate: quantized weights load at about their download size, but the engine does not report it exactly for this model.'
        default:
            return 'The model’s size is unknown; set the memory its weights take once loaded.'
    }
}

/** Where the per-token cost came from, and the precision it is at. */
export function kvFact(plan: CapacityPlan): string {
    switch (plan.kvSource) {
        case 'metadata':
            return 'Computed from the model’s architecture (attention layers × key/value heads × head size), at full precision (f16).'
        case 'measured':
            return 'Measured from the loaded copy (its memory less the weights, per token of its context), at the engine’s current cache precision.'
        case 'manual':
            return `The value you set, at full precision (f16)${
                plan.spec.kvBytesPerToken > 0
                    ? `; the model’s: ${perTokenLabel(plan.spec.kvBytesPerToken)}`
                    : ''
            }.`
        default:
            return 'The engine does not describe this model’s layers; set the cost, or load the model once so PAIR can measure it.'
    }
}

/** What the plan counts per token at the chosen cache precision. */
export function kvUsed(plan: CapacityPlan): string {
    if (plan.bytesPerToken <= 0) return ''
    const precision =
        plan.kvSource === 'measured' ? 'as measured' : `at ${CACHE_LABEL[plan.cacheType]}`
    return `This plan counts ${perTokenLabel(plan.bytesPerToken)} per token ${precision}, so a session of ${plan.windowTokens.toLocaleString()} tokens needs ${sizeLabel(plan.bytesPerSession)} for its context.`
}
