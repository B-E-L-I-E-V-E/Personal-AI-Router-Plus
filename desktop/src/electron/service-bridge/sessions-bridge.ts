// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    CapacityChange,
    ModelCapacity,
    SessionContextUse,
    SessionPolicy,
    SessionSummary
} from '@/shared/types/sessions'
import getErrorString from '@/shared/utils/get-error-string'
import type { JsonObject, JsonValue } from './json-rpc-subprocess'
import { getModularSupervisor } from './modular-supervisor'
import { PROXY_ENGINES } from './proxy-engines'

/**
 * Session policy, live sessions and context capacity live in nvpair-proxy.
 * They are process-wide, so any enabled engine's relay reaches the same
 * store; the first engine whose proxy answers is used.
 */
export async function callSessionProxy(
    method: string,
    params?: JsonObject
): Promise<JsonValue | undefined> {
    let lastError: unknown = new Error('no engine proxy is running')
    for (const engine of PROXY_ENGINES) {
        try {
            return await getModularSupervisor().callProxy(engine, method, params)
        } catch (err) {
            lastError = err
            // Only a missing facade moves on; a refusal from the proxy itself is
            // the answer and is reported as such.
            if (!(getErrorString(err) ?? '').includes('not available')) throw err
        }
    }
    throw lastError
}

function objectValue(value: JsonValue | undefined): JsonObject | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

function arrayValue(value: JsonValue | undefined): JsonValue[] {
    return Array.isArray(value) ? value : []
}

function stringValue(value: JsonValue | undefined): string {
    return typeof value === 'string' ? value : ''
}

function numberValue(value: JsonValue | undefined): number {
    return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function parsePolicy(value: JsonValue | undefined): SessionPolicy {
    const obj = objectValue(value)
    return {
        idleTimeoutSeconds: numberValue(obj?.idleTimeoutSeconds),
        maxLifetimeSeconds: numberValue(obj?.maxLifetimeSeconds),
        maxSessions: numberValue(obj?.maxSessions),
        queueTimeoutSeconds: numberValue(obj?.queueTimeoutSeconds),
        maxQueued: numberValue(obj?.maxQueued)
    }
}

function parseSessions(value: JsonValue | undefined): SessionSummary[] {
    const out: SessionSummary[] = []
    for (const entry of arrayValue(objectValue(value)?.sessions)) {
        const obj = objectValue(entry)
        const sessionId = stringValue(obj?.sessionId)
        if (!obj || !sessionId) continue
        const context: SessionContextUse[] = []
        const ctx = objectValue(obj.context)
        for (const engine of Object.keys(ctx ?? {})) {
            const use = objectValue(ctx?.[engine])
            if (!use) continue
            context.push({ engine, model: stringValue(use.model), tokens: numberValue(use.tokens) })
        }
        out.push({
            sessionId,
            agentId: stringValue(obj.agentId) || null,
            expiresAt: stringValue(obj.expiresAt),
            inFlight: numberValue(obj.inFlight),
            requests: numberValue(obj.requests),
            context
        })
    }
    return out
}

function parseCapacity(value: JsonValue | undefined): ModelCapacity[] {
    const out: ModelCapacity[] = []
    for (const entry of arrayValue(objectValue(value)?.models)) {
        const obj = objectValue(entry)
        const engine = stringValue(obj?.engine)
        const model = stringValue(obj?.model)
        if (!obj || !engine || !model) continue
        const configured = objectValue(obj.configured)
        out.push({
            engine,
            model,
            configured: configured
                ? {
                      contextTokens: numberValue(configured.contextTokens),
                      maxSessions: numberValue(configured.maxSessions),
                      replyTokens: numberValue(configured.replyTokens)
                  }
                : null,
            detectedWindowTokens: numberValue(objectValue(obj.detected)?.windowTokens),
            contextTokens: numberValue(obj.contextTokens),
            maxSessions: numberValue(obj.maxSessions),
            sessionContextTokens: numberValue(obj.sessionContextTokens),
            replyTokens: numberValue(obj.replyTokens),
            messageTokens: numberValue(obj.messageTokens),
            source: stringValue(obj.source),
            heldSlots: numberValue(obj.heldSlots),
            queued: numberValue(obj.queued),
            usedContextTokens: numberValue(obj.usedContextTokens),
            warning: stringValue(obj.warning)
        })
    }
    return out
}

export async function getSessionPolicy(): Promise<SessionPolicy> {
    return parsePolicy(await callSessionProxy('sessions/configure', {}))
}

export async function setSessionPolicy(change?: Partial<SessionPolicy>): Promise<SessionPolicy> {
    const params: JsonObject = {}
    for (const [key, value] of Object.entries(change ?? {})) {
        if (typeof value === 'number') params[key] = value
    }
    return parsePolicy(await callSessionProxy('sessions/configure', params))
}

export async function listSessions(): Promise<SessionSummary[]> {
    return parseSessions(await callSessionProxy('sessions/list', {}))
}

export async function endSession(request?: {
    sessionId: string
    agentId: string | null
}): Promise<null> {
    if (!request) return null
    await callSessionProxy('sessions/end', {
        sessionId: request.sessionId,
        agentId: request.agentId ?? ''
    })
    return null
}

export async function getCapacityReport(): Promise<ModelCapacity[]> {
    return parseCapacity(await callSessionProxy('capacity/report', {}))
}

export async function setCapacity(change?: CapacityChange): Promise<null> {
    if (!change) return null
    const params: JsonObject = { engine: change.engine, model: change.model }
    if (change.clear) params.clear = true
    if (typeof change.contextTokens === 'number') params.contextTokens = change.contextTokens
    if (typeof change.maxSessions === 'number') params.maxSessions = change.maxSessions
    if (typeof change.replyTokens === 'number') params.replyTokens = change.replyTokens
    await callSessionProxy('capacity/set', params)
    return null
}
