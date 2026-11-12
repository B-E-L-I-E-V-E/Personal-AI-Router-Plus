// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CapacityPlan } from '@/shared/types/sessions'

/** The smallest window the planner saves. */
export const MIN_WINDOW_TOKENS = 2048
/** Window stops offered when the model does not report its maximum. */
const UNKNOWN_MAX_WINDOW = 1 << 20
/** The proxy plans at most this many sessions per model. */
export const MAX_SESSIONS = 256

/**
 * The context windows the window slider stops at: powers of two from 2K up to
 * the model's maximum, and the maximum itself (often not a power of two, such
 * as 40,960).
 */
export function windowStops(maxContextTokens: number): number[] {
    const top = maxContextTokens > 0 ? maxContextTokens : UNKNOWN_MAX_WINDOW
    const stops: number[] = []
    for (let w = MIN_WINDOW_TOKENS; w < top; w *= 2) stops.push(w)
    stops.push(top)
    return stops
}

/** The index of the stop closest to tokens. */
export function nearestStop(stops: number[], tokens: number): number {
    let best = 0
    for (let i = 1; i < stops.length; i++) {
        if (Math.abs(stops[i] - tokens) < Math.abs(stops[best] - tokens)) best = i
    }
    return best
}

/**
 * How many sessions of the plan's window fit, or null when the plan could not
 * size the model (its size or context cost is unknown): then no count can be
 * judged, so the dialog offers none rather than a slider of arbitrary length.
 */
export function sessionsFit(plan: CapacityPlan | null): number | null {
    return plan && !plan.error ? plan.sessions : null
}

/**
 * The sessions slider's top: well past what fits, since plans may exceed the
 * hardware (the sliders turn red), but within the proxy's limit.
 */
export function sessionsSliderMax(fit: number): number {
    return Math.min(MAX_SESSIONS, Math.max(8, fit + 8, fit * 2))
}

interface MemoryUse {
    /** GPU memory the settings need, the reserve included. */
    gpuNeeded: number
    gpuTotal: number
    gpuOver: boolean
    /** System memory: expert weights in it plus the reserve kept for the OS. */
    ramNeeded: number
    ramTotal: number
    ramOver: boolean
    /** The GPU uses system memory (Apple silicon): one pool, shown once. */
    shared: boolean
}

/**
 * What sessions sessions of plan's window need, against what the node has,
 * with the experts of the first cpuExpertLayers layers in system memory.
 */
export function memoryUse(
    plan: CapacityPlan,
    sessions: number,
    cpuExpertLayers = plan.cpuExpertLayers
): MemoryUse {
    const inRam = cpuExpertBytes(plan.spec.expertLayerBytes, cpuExpertLayers)
    const gpuNeeded =
        plan.weightsBytes -
        inRam +
        plan.otherModelsBytes +
        plan.bytesPerSession * sessions +
        plan.reserveBytes
    const ramNeeded = inRam + plan.ramReserveBytes
    return {
        gpuNeeded,
        gpuTotal: plan.totalBytes,
        gpuOver: gpuNeeded > plan.totalBytes,
        ramNeeded,
        ramTotal: plan.ramTotalBytes,
        ramOver: inRam > 0 && plan.ramTotalBytes > 0 && ramNeeded > plan.ramTotalBytes,
        shared: plan.unified
    }
}

/**
 * The expert weights llama.cpp's --n-cpu-moe n keeps in system memory: those
 * of the first n layers.
 */
export function cpuExpertBytes(layerBytes: number[], n: number): number {
    return layerBytes.slice(0, Math.max(0, n)).reduce((sum, b) => sum + b, 0)
}

/**
 * The --n-cpu-moe values that move one more expert layer to system memory:
 * element k keeps the experts of k layers there. Layers without experts (some
 * models start with dense ones) are skipped, since moving them changes
 * nothing.
 */
export function expertSplitStops(layerBytes: number[]): number[] {
    const stops = [0]
    layerBytes.forEach((b, i) => {
        if (b > 0) stops.push(i + 1)
    })
    return stops
}

/** How many expert layers --n-cpu-moe n moves to system memory. */
export function expertLayersOnCpu(stops: number[], n: number): number {
    let k = 0
    while (k + 1 < stops.length && stops[k + 1] <= n) k++
    return k
}

/** Parameter counts as models are named: 284B, 30.5B, 3.3B, 600M. */
export function parameterLabel(n: number): string {
    if (n >= 100e9) return `${Math.round(n / 1e9)}B`
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
    return `${Math.round(n / 1e6)}M`
}

/** Short token counts for slider labels: 4K, 32K, 40,960 → 40K, 1M. */
export function shortTokens(n: number): string {
    if (n >= 1 << 20 && n % (1 << 20) === 0) return `${n / (1 << 20)}M`
    if (n >= 1024) return `${Math.round(n / 1024)}K`
    return String(n)
}

/** Two significant figures, rounded down, so estimates never overstate. */
function roughlyDown(n: number): number {
    if (n < 100) return Math.floor(n)
    const scale = 10 ** (Math.floor(Math.log10(n)) - 1)
    return Math.floor(n / scale) * scale
}

/** The typing standard's word: five characters and a space. */
const CHARS_PER_STANDARD_WORD = 6
/** Chinese words average about one and a half characters. */
const CHARS_PER_CHINESE_WORD = 1.5
/** A single-spaced page. */
const WORDS_PER_PAGE = 500

/**
 * Text per token for kinds of text, measured with qwen3's tokenizer on
 * samples of each. Other modern tokenizers (Llama 3, Gemma, GPT-style BPE)
 * give similar numbers for English and code and differ more for other
 * languages. English and code words were counted in the samples; text
 * without words is counted in standard words of six characters, and Chinese
 * in words of one and a half characters.
 */
const TEXT_KINDS: { label: string; charsPerToken: number; charsPerWord: number }[] = [
    // Measured: 0.67 and 0.47 words per token.
    { label: 'English prose', charsPerToken: 4.1, charsPerWord: 4.1 / 0.67 },
    { label: 'Code', charsPerToken: 3.6, charsPerWord: 3.6 / 0.47 },
    { label: 'Minified JSON', charsPerToken: 3.4, charsPerWord: CHARS_PER_STANDARD_WORD },
    { label: 'Chinese', charsPerToken: 1.7, charsPerWord: CHARS_PER_CHINESE_WORD },
    { label: 'Base64', charsPerToken: 1.3, charsPerWord: CHARS_PER_STANDARD_WORD },
    { label: 'UUIDs and hashes', charsPerToken: 1.2, charsPerWord: CHARS_PER_STANDARD_WORD },
    { label: 'Emoji', charsPerToken: 0.9, charsPerWord: CHARS_PER_STANDARD_WORD }
]

interface TextKindFit {
    label: string
    charsPerToken: number
    /** Characters, words and single-spaced pages that fit, rounded down. */
    characters: number
    words: number
    pages: number
}

/** How much of each kind of text fits in a window of tokens. */
export function textKindFits(tokens: number): TextKindFit[] {
    return TEXT_KINDS.map(k => {
        const characters = tokens * k.charsPerToken
        const words = characters / k.charsPerWord
        return {
            label: k.label,
            charsPerToken: k.charsPerToken,
            characters: roughlyDown(characters),
            words: roughlyDown(words),
            pages: roughlyDown(words / WORDS_PER_PAGE)
        }
    })
}

/** The reply budget a plan keeps when none is chosen (the proxy's default). */
const DEFAULT_REPLY_TOKENS = 8192
const MIN_REPLY_TOKENS = 512

/**
 * The reply budget slider's stops: powers of two from 512 up to half the
 * window, so the conversation always keeps at least half.
 */
export function replyStops(windowTokens: number): number[] {
    const stops: number[] = []
    for (let r = MIN_REPLY_TOKENS; r <= windowTokens / 2; r *= 2) stops.push(r)
    return stops.length > 0 ? stops : [Math.max(1, Math.floor(windowTokens / 2))]
}

/** The reply budget for a window: the saved one, else the default, at most half the window. */
export function replyBudget(windowTokens: number, saved = 0): number {
    return Math.min(saved || DEFAULT_REPLY_TOKENS, Math.floor(windowTokens / 2))
}

const GiB = 1024 ** 3
const KiB = 1024

/** Model weight sizes the size slider stops at, in GB. */
const WEIGHT_STOPS_GB = [
    0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024
]
/** Context cache costs per token the cost slider stops at, in KB. */
const KV_STOPS_KB = [
    2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048
]

/** stops with value added in order, unless it is already one (within 1%). */
export function withStop(stops: number[], value: number): number[] {
    if (value <= 0 || stops.some(s => Math.abs(s - value) <= value * 0.01)) return stops
    return [...stops, value].sort((a, b) => a - b)
}

/**
 * The weight-size slider's stops, in bytes: round sizes up to well past both
 * the model's own size and the node's memory, and the model's size itself.
 */
export function weightStops(modelBytes: number, memoryBytes: number): number[] {
    const top = Math.max(2 * modelBytes, 1.5 * memoryBytes, 8 * GiB)
    const stops: number[] = []
    for (const gb of WEIGHT_STOPS_GB) {
        stops.push(gb * GiB)
        if (gb * GiB >= top) break
    }
    return withStop(stops, modelBytes)
}

/** The context-cost slider's stops, in bytes per token, with the model's own cost. */
export function kvStops(modelBytesPerToken: number): number[] {
    const stops = KV_STOPS_KB.map(kb => kb * KiB)
    return withStop(stops, modelBytesPerToken)
}

/**
 * Which stops get a label under the slider: about six spread evenly, the
 * ends, and the model's own value.
 */
export function labelledStops(stops: number[], modelValue: number): number[] {
    const every = Math.max(1, Math.ceil(stops.length / 6))
    const keep = new Set<number>([0, stops.length - 1])
    for (let i = 0; i < stops.length; i += every) keep.add(i)
    if (modelValue > 0) keep.add(nearestStop(stops, modelValue))
    return [...keep].sort((a, b) => a - b)
}

/** Trims a fixed-point number: 4.0 → 4, 1.50 → 1.5. */
function trimmed(n: number, digits: number): string {
    return String(Number(n.toFixed(digits)))
}

/** A weight size for slider labels and the exact field: 512 MB, 1.5 GB, 24 GB. */
export function sizeLabel(bytes: number): string {
    if (bytes < GiB) return `${Math.round(bytes / 1024 ** 2)} MB`
    return `${trimmed(bytes / GiB, bytes < 10 * GiB ? 2 : 1)} GB`
}

/** A per-token context cost: 112 KB, 59.5 KB, 1.5 MB. */
export function perTokenLabel(bytes: number): string {
    if (bytes >= 1024 * KiB) return `${trimmed(bytes / (1024 * KiB), 2)} MB`
    return `${trimmed(bytes / KiB, bytes < 10 * KiB ? 2 : 1)} KB`
}

/** Parses a typed amount ("4.2") in unit bytes; null unless a positive number. */
export function parseAmount(text: string, unitBytes: number): number | null {
    const trimmedText = text.trim()
    if (!/^\d+(\.\d+)?$/.test(trimmedText)) return null
    const bytes = Math.round(Number(trimmedText) * unitBytes)
    return bytes > 0 ? bytes : null
}
