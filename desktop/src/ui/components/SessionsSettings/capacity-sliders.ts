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

/** What sessions sessions of plan's window need, against what the node has. */
export function memoryUse(plan: CapacityPlan, sessions: number): MemoryUse {
    const gpuNeeded =
        plan.gpuWeightsBytes +
        plan.otherModelsBytes +
        plan.bytesPerSession * sessions +
        plan.reserveBytes
    const ramNeeded = plan.ramUsedBytes + plan.ramReserveBytes
    return {
        gpuNeeded,
        gpuTotal: plan.totalBytes,
        gpuOver: gpuNeeded > plan.totalBytes,
        ramNeeded,
        ramTotal: plan.ramTotalBytes,
        ramOver: plan.offloadExperts && plan.ramTotalBytes > 0 && ramNeeded > plan.ramTotalBytes,
        shared: plan.unified
    }
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
