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

// Rough text sizes per token for modern tokenizers (Llama 3, Qwen, Gemma,
// GPT-style BPE): English prose averages about 4 characters or 0.75 words a
// token, source code about 3 characters. A single-spaced page is about 500
// words.
const CHARS_PER_TOKEN = 4
const CODE_CHARS_PER_TOKEN = 3
const WORDS_PER_TOKEN = 0.75
const WORDS_PER_PAGE = 500

/** Round to two significant figures, so estimates do not look exact. */
function roughly(n: number): number {
    if (n < 100) return Math.round(n)
    const scale = 10 ** (Math.floor(Math.log10(n)) - 1)
    return Math.round(n / scale) * scale
}

/** About how much text a window of tokens holds. */
export function approxText(tokens: number): {
    characters: number
    words: number
    pages: number
    codeCharacters: number
} {
    return {
        characters: roughly(tokens * CHARS_PER_TOKEN),
        words: roughly(tokens * WORDS_PER_TOKEN),
        pages: roughly((tokens * WORDS_PER_TOKEN) / WORDS_PER_PAGE),
        codeCharacters: roughly(tokens * CODE_CHARS_PER_TOKEN)
    }
}

/** One line describing approxText for people. */
export function approxTextLine(tokens: number): string {
    if (tokens <= 0) return ''
    const t = approxText(tokens)
    const pages = t.pages >= 1 ? `, ${t.pages.toLocaleString()} pages` : ''
    return `About ${t.characters.toLocaleString()} characters of English (${t.words.toLocaleString()} words${pages}) or ${t.codeCharacters.toLocaleString()} characters of code per conversation, counting the model's replies.`
}
