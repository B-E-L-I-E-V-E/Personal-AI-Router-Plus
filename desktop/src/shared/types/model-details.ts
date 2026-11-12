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
