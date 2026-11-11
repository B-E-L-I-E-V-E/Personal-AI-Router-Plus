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

/** Two significant figures, rounded down, so estimates never overstate. */
function roughlyDown(n: number): number {
    if (n < 100) return Math.floor(n)
    const scale = 10 ** (Math.floor(Math.log10(n)) - 1)
    return Math.floor(n / scale) * scale
}

/**
 * Text per token for kinds of text, measured with qwen3's tokenizer on
 * samples of each. Other modern tokenizers (Llama 3, Gemma, GPT-style BPE)
 * give similar numbers for English and code and differ more for other
 * languages. Words are counted only where text has words.
 */
const TEXT_KINDS: { label: string; charsPerToken: number; wordsPerToken?: number }[] = [
    { label: 'English prose', charsPerToken: 4.1, wordsPerToken: 0.67 },
    { label: 'Code', charsPerToken: 3.6, wordsPerToken: 0.47 },
    { label: 'Minified JSON', charsPerToken: 3.4 },
    { label: 'Chinese', charsPerToken: 1.7 },
    { label: 'Base64', charsPerToken: 1.3 },
    { label: 'UUIDs and hashes', charsPerToken: 1.2 },
    { label: 'Emoji', charsPerToken: 0.9 }
]

interface TextKindFit {
    label: string
    charsPerToken: number
    /** Characters that fit, rounded down. */
    characters: number
    /** Words that fit, rounded down; 0 where the text has no words. */
    words: number
}

/** How much of each kind of text fits in a window of tokens. */
export function textKindFits(tokens: number): TextKindFit[] {
    return TEXT_KINDS.map(k => ({
        label: k.label,
        charsPerToken: k.charsPerToken,
        characters: roughlyDown(tokens * k.charsPerToken),
        words: k.wordsPerToken ? roughlyDown(tokens * k.wordsPerToken) : 0
    }))
}
