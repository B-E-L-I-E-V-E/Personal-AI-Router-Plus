// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelSpec, StartSource } from '@/shared/types/sessions'
import { MAX_SESSIONS, MIN_WINDOW_TOKENS } from './capacity-sliders'

/** A starting value's origin, as the planner shows it. */
type StartingSource = StartSource | 'maximum'

interface Starting {
    value: number
    source: StartingSource
}

export interface StartingValues {
    window?: Starting
    sessions?: Starting
    reply?: Starting
}

/** The marker shown beside a slider that still sits on its starting value. */
export const STARTING_LABEL: Record<StartingSource, string> = {
    engine: 'engine now',
    launch: 'engine settings',
    model: 'model default',
    maximum: 'model maximum'
}

/**
 * Where a model with no saved plan starts: the window, slots and reply length
 * the engine runs it with or the model sets, in that order, else the window
 * the model was trained for. Nothing reported means no starting value, and
 * the planner keeps its own default (as many sessions as fit, the default
 * reply budget).
 */
export function startingValues(spec: ModelSpec): StartingValues {
    const out: StartingValues = {}
    const max = spec.maxContextTokens
    const reported = spec.defaults.window
    if (reported && reported.value >= MIN_WINDOW_TOKENS) {
        out.window = {
            value: max > 0 ? Math.min(reported.value, max) : reported.value,
            source: reported.source
        }
    } else if (max > 0 && spec.contextSource !== 'web') {
        out.window = { value: max, source: 'maximum' }
    }
    const sessions = spec.defaults.sessions
    if (sessions)
        out.sessions = { value: Math.min(sessions.value, MAX_SESSIONS), source: sessions.source }
    if (spec.defaults.reply) out.reply = { ...spec.defaults.reply }
    return out
}

/** Slider stops with value added in order, so a starting value is a stop. */
export function withStop(stops: number[], value: number | undefined): number[] {
    if (!value || stops.includes(value) || value > stops[stops.length - 1]) return stops
    return [...stops, value].sort((a, b) => a - b)
}
