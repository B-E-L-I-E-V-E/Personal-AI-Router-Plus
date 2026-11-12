// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelDetails, ModelDetailsRequest, ModelLink } from '@/shared/types/model-details'
import getErrorString from '@/shared/utils/get-error-string'
import { engineManagerName } from '@/shared/utils/engines'
import type { JsonObject, JsonValue } from './json-rpc-subprocess'
import { getModularBridgeState } from './modular-state'
import { getModularSupervisor } from './modular-supervisor'
import { isProxyEngine } from './proxy-engines'

/**
 * A downloaded model's description, tags, capabilities and links, read by
 * this node's nvpair-proxy from the engine that has the model. A model on
 * another node is described by that node's proxy over cluster mTLS.
 */

function objectValue(value: JsonValue | undefined): JsonObject | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

function stringValue(value: JsonValue | undefined): string {
    return typeof value === 'string' ? value : ''
}

function numberValue(value: JsonValue | undefined): number {
    return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function stringList(value: JsonValue | undefined): string[] {
    return Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : []
}

/** Only absolute http(s) addresses are kept; anything else is dropped. */
function parseLink(value: JsonValue): ModelLink | null {
    const obj = objectValue(value)
    const url = stringValue(obj?.url)
    if (!/^https?:\/\//i.test(url)) return null
    return { label: stringValue(obj?.label) || url, url, source: stringValue(obj?.source) }
}

function parseModelDetails(
    value: JsonValue | undefined,
    request: { engine: string; model: string },
    note?: string
): ModelDetails {
    const obj = objectValue(value)
    const notes = stringList(obj?.notes)
    if (note) notes.push(note)
    return {
        engine: stringValue(obj?.engine) || request.engine,
        model: stringValue(obj?.model) || request.model,
        name: stringValue(obj?.name),
        description: stringValue(obj?.description),
        tags: stringList(obj?.tags),
        languages: stringList(obj?.languages),
        author: stringValue(obj?.author),
        publisher: stringValue(obj?.publisher),
        license: stringValue(obj?.license),
        licenseText: stringValue(obj?.licenseText),
        type: stringValue(obj?.type),
        architecture: stringValue(obj?.architecture),
        family: stringValue(obj?.family),
        parameterSize: stringValue(obj?.parameterSize),
        parameterCount: numberValue(obj?.parameterCount),
        quantization: stringValue(obj?.quantization),
        format: stringValue(obj?.format),
        capabilities: stringList(obj?.capabilities),
        contextLength: numberValue(obj?.contextLength),
        loadedContext: numberValue(obj?.loadedContext),
        sizeBytes: numberValue(obj?.sizeBytes),
        digest: stringValue(obj?.digest),
        modifiedAt: stringValue(obj?.modifiedAt),
        parameters: stringValue(obj?.parameters),
        links: (Array.isArray(obj?.links) ? obj.links : [])
            .map(parseLink)
            .filter((l): l is ModelLink => l !== null),
        notes
    }
}

export async function getModelDetails(request?: ModelDetailsRequest): Promise<ModelDetails> {
    const asked = { engine: request?.engineType ?? '', model: request?.model ?? '' }
    if (!request) return parseModelDetails(undefined, asked, 'no request')
    const { engineType, nodeId, model } = request
    if (!isProxyEngine(engineType)) {
        return parseModelDetails(undefined, asked, 'PAIR cannot describe models of this engine')
    }
    const params: JsonObject = { engine: engineManagerName(engineType), model }
    const selfId = getModularBridgeState().getSelfId()
    if (nodeId && nodeId !== selfId) params.nodeId = nodeId
    try {
        const result = await getModularSupervisor().callProxy(engineType, 'model/details', params)
        return parseModelDetails(result, asked)
    } catch (err) {
        const reason = (getErrorString(err) ?? 'the request failed').replace(
            /^invalid params: /,
            ''
        )
        return parseModelDetails(undefined, asked, `Could not read the model's details: ${reason}`)
    }
}
