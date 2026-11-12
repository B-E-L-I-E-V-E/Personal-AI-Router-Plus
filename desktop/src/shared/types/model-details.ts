// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineType } from '@/shared/types/engines'

/** A page about a model, and the reported field its address came from. */
export interface ModelLink {
    label: string
    url: string
    source: string
}

/**
 * What a model's engine and file say about one downloaded model
 * (nvpair-proxy `model/details`). Every field is what the engine or the GGUF
 * header reported; empty strings, zeros and empty lists mean it said nothing.
 */
export interface ModelDetails {
    engine: string
    model: string
    name: string
    description: string
    tags: string[]
    languages: string[]
    author: string
    publisher: string
    license: string
    licenseText: string
    type: string
    architecture: string
    family: string
    parameterSize: string
    parameterCount: number
    quantization: string
    format: string
    capabilities: string[]
    /** The longest context the model was trained for. */
    contextLength: number
    /** The window a running engine gave it. */
    loadedContext: number
    sizeBytes: number
    digest: string
    modifiedAt: string
    /** The engine's default generation settings (Ollama PARAMETER lines). */
    parameters: string
    links: ModelLink[]
    /** Why something could not be read, in plain words. */
    notes: string[]
}

export interface ModelDetailsRequest {
    engineType: EngineType
    nodeId: string
    model: string
}

/** A value read from a model's page: text, a number or a flag. */
export type WebFactValue = string | number | boolean

/**
 * One value a small local model read from the model's web page
 * (nvpair-proxy `model/web-facts`). `fills` names the model-details field it
 * fills because the engine left that field empty; it is '' when the engine
 * already reported the field, whose answer wins.
 */
export interface WebFact {
    field: string
    value: WebFactValue
    /** The page the value was read from. */
    source: string
    fills: string
}

/** Two or more pages, or parts of one, that gave different values. */
export interface WebFactConflict {
    field: string
    values: { value: WebFactValue; source: string }[]
}

export interface WebFactsPage {
    url: string
    chunks: number
    note: string
}

/** One web lookup's result, kept apart from what the engine reported. */
export interface WebFacts {
    /** The model that read the pages. */
    reader: string
    readAt: number
    pages: WebFactsPage[]
    facts: WebFact[]
    conflicts: WebFactConflict[]
    notes: string[]
}

export type WebFactsJobState = 'running' | 'done' | 'failed'

export interface WebFactsJob {
    state: WebFactsJobState
    phase: string
    reader: string
    chunksTotal: number
    chunksRead: number
    error: string
}

export interface WebFactsStatus {
    /** The Ollama model this node reads pages with (a setting). */
    readerModel: string
    job: WebFactsJob | null
    result: WebFacts | null
    /** Why the status could not be read, in plain words. */
    error: string
}

export interface WebFactsRequest extends ModelDetailsRequest {
    /** Starting only: the reader model to use and save; '' keeps the setting. */
    readerModel?: string
}
