// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    ModelDetails,
    ModelDetailsRequest,
    ModelLink,
    WebFact,
    WebFactConflict,
    WebFacts,
    WebFactsJob,
    WebFactsPage,
    WebFactsRequest,
    WebFactsStatus,
    WebFactValue
} from '@/shared/types/model-details'
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

// ---------------------------------------------------------------------------
// Web facts: what a small local model read from the model's own pages, where
// the engine reported nothing. The proxy runs the lookup as a job, since it
// takes longer than one relayed call may; the dialog starts it and polls.

/** The reader model's name when the proxy does not report one. */
const DEFAULT_READER_MODEL = 'qwen3.5:4b'

function factValue(value: JsonValue | undefined): WebFactValue | null {
    if (typeof value === 'string' || typeof value === 'boolean') return value
    if (typeof value === 'number' && Number.isFinite(value)) return value
    return null
}

/** A web value is kept only with a value, a field and an http(s) source. */
function parseWebFact(value: JsonValue): WebFact | null {
    const obj = objectValue(value)
    const v = factValue(obj?.value)
    const field = stringValue(obj?.field)
    const source = stringValue(obj?.source)
    if (v === null || !field || !/^https?:\/\//i.test(source)) return null
    return { field, value: v, source, fills: stringValue(obj?.fills) }
}

function parseConflict(value: JsonValue): WebFactConflict | null {
    const obj = objectValue(value)
    const field = stringValue(obj?.field)
    const values = (Array.isArray(obj?.values) ? obj.values : []).flatMap(item => {
        const c = objectValue(item)
        const v = factValue(c?.value)
        return v === null ? [] : [{ value: v, source: stringValue(c?.source) }]
    })
    return field && values.length > 1 ? { field, values } : null
}

function parsePage(value: JsonValue): WebFactsPage | null {
    const obj = objectValue(value)
    const url = stringValue(obj?.url)
    if (!/^https?:\/\//i.test(url)) return null
    return { url, chunks: numberValue(obj?.chunks), note: stringValue(obj?.note) }
}

function nonNull<T>(v: T | null): v is T {
    return v !== null
}

function parseWebFacts(value: JsonValue | undefined): WebFacts | null {
    const obj = objectValue(value)
    if (!obj) return null
    return {
        reader: stringValue(obj.reader),
        readAt: numberValue(obj.readAt),
        pages: (Array.isArray(obj.pages) ? obj.pages : []).map(parsePage).filter(nonNull),
        facts: (Array.isArray(obj.facts) ? obj.facts : []).map(parseWebFact).filter(nonNull),
        conflicts: (Array.isArray(obj.conflicts) ? obj.conflicts : [])
            .map(parseConflict)
            .filter(nonNull),
        notes: stringList(obj.notes)
    }
}

function parseJob(value: JsonValue | undefined): WebFactsJob | null {
    const obj = objectValue(value)
    const state = obj?.state
    if (state !== 'running' && state !== 'done' && state !== 'failed') return null
    return {
        state,
        phase: stringValue(obj?.phase),
        reader: stringValue(obj?.reader),
        chunksTotal: numberValue(obj?.chunksTotal),
        chunksRead: numberValue(obj?.chunksRead),
        error: stringValue(obj?.error)
    }
}

function parseWebFactsStatus(value: JsonValue | undefined, error = ''): WebFactsStatus {
    const obj = objectValue(value)
    return {
        readerModel: stringValue(obj?.readerModel) || DEFAULT_READER_MODEL,
        job: parseJob(obj?.job),
        result: parseWebFacts(obj?.result),
        error
    }
}

async function callWebFacts(
    method: 'model/web-facts' | 'model/web-facts-start',
    request?: WebFactsRequest
): Promise<WebFactsStatus> {
    if (!request) return parseWebFactsStatus(undefined, 'no request')
    const { engineType, nodeId, model, readerModel } = request
    if (!isProxyEngine(engineType)) {
        return parseWebFactsStatus(undefined, 'PAIR cannot look up models of this engine')
    }
    const params: JsonObject = { engine: engineManagerName(engineType), model }
    const selfId = getModularBridgeState().getSelfId()
    if (nodeId && nodeId !== selfId) params.nodeId = nodeId
    if (method === 'model/web-facts-start' && readerModel?.trim()) {
        params.readerModel = readerModel.trim()
    }
    try {
        return parseWebFactsStatus(
            await getModularSupervisor().callProxy(engineType, method, params)
        )
    } catch (err) {
        const reason = (getErrorString(err) ?? 'the request failed').replace(
            /^invalid params: /,
            ''
        )
        return parseWebFactsStatus(undefined, reason)
    }
}

export function getModelWebFacts(request?: WebFactsRequest): Promise<WebFactsStatus> {
    return callWebFacts('model/web-facts', request)
}

export function startModelWebFacts(request?: WebFactsRequest): Promise<WebFactsStatus> {
    return callWebFacts('model/web-facts-start', request)
}
