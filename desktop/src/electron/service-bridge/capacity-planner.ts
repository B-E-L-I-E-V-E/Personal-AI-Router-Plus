// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from 'node:crypto'
import type {
    ApplyPlanResult,
    CapacityPlan,
    CapacityPlanRequest,
    KvCacheType,
    ModelSpec,
    PlannerChange,
    PlannerState,
    SavedCapacityPlan
} from '@/shared/types/sessions'
import type { EngineManagerName } from '@/shared/types/engines'
import type { EngineSettingsReceipt, EngineSettingsSnapshot } from '@/shared/types/engine-settings'
import { MODULAR_ENGINE_LIFECYCLE_CALL_TIMEOUT_MS } from '@/shared/constants/modular-runtime'
import getErrorString from '@/shared/utils/get-error-string'
import { createStructuredLogger } from '@/shared/utils/log'
import type { JsonObject, JsonValue } from './json-rpc-subprocess'
import { parseEngineSettings, parseEngineSettingsReceipt } from './engine-settings'
import { getModularBridgeState } from './modular-state'
import { getModularSupervisor } from './modular-supervisor'
import { callSessionProxy } from './sessions-bridge'

const log = createStructuredLogger('capacity-planner')

/**
 * The capacity planner sizes a model's sessions from this node's GPU memory
 * (nvpair-proxy does the math and reads the model), and applies a saved plan
 * by rewriting the engine's launch settings through the broker, which
 * restarts the engine. Plans and the auto-reload switch live in the proxy.
 */

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

function cacheTypeValue(value: JsonValue | undefined): KvCacheType {
    return value === 'q8_0' || value === 'q4_0' ? value : 'f16'
}

function stringList(value: JsonValue | undefined): string[] {
    return arrayValue(value).filter((v): v is string => typeof v === 'string')
}

function parseSpec(value: JsonValue | undefined): ModelSpec {
    const obj = objectValue(value)
    return {
        engine: stringValue(obj?.engine),
        model: stringValue(obj?.model),
        source: stringValue(obj?.source),
        architecture: stringValue(obj?.architecture),
        format: stringValue(obj?.format),
        weightsBytes: numberValue(obj?.weightsBytes),
        maxContextTokens: numberValue(obj?.maxContextTokens),
        kvBytesPerToken: numberValue(obj?.kvBytesPerToken),
        notes: stringList(obj?.notes)
    }
}

function parsePlan(value: JsonValue | undefined, error = ''): CapacityPlan {
    const obj = objectValue(value)
    return {
        spec: parseSpec(obj?.spec),
        totalBytes: numberValue(obj?.totalBytes),
        reserveBytes: numberValue(obj?.reserveBytes),
        weightsBytes: numberValue(obj?.weightsBytes),
        otherModelsBytes: numberValue(obj?.otherModelsBytes),
        contextBytes: numberValue(obj?.contextBytes),
        bytesPerToken: numberValue(obj?.bytesPerToken),
        bytesPerSession: numberValue(obj?.bytesPerSession),
        cacheType: cacheTypeValue(obj?.cacheType),
        windowTokens: numberValue(obj?.windowTokens),
        sessions: numberValue(obj?.sessions),
        poolTokens: numberValue(obj?.poolTokens),
        usedBytes: numberValue(obj?.usedBytes),
        fits: obj?.fits === true,
        warnings: stringList(obj?.warnings),
        error: error || stringValue(obj?.error)
    }
}

function parseSavedPlan(value: JsonValue): SavedCapacityPlan | null {
    const obj = objectValue(value)
    const engine = stringValue(obj?.engine)
    const model = stringValue(obj?.model)
    if (!obj || !engine || !model) return null
    return {
        engine,
        model,
        windowTokens: numberValue(obj.windowTokens),
        sessions: numberValue(obj.sessions),
        cacheType: cacheTypeValue(obj.cacheType),
        reservePercent: numberValue(obj.reservePercent),
        applied: obj.applied === true,
        appliedAt: numberValue(obj.appliedAt),
        savedAt: numberValue(obj.savedAt)
    }
}

export function parsePlanner(value: JsonValue | undefined): PlannerState {
    const obj = objectValue(value)
    const activity = objectValue(obj?.activity)
    return {
        autoReload: obj?.autoReload !== false,
        plans: arrayValue(obj?.plans)
            .map(parseSavedPlan)
            .filter((p): p is SavedCapacityPlan => p !== null),
        activity: {
            sessions: numberValue(activity?.sessions),
            activeRequests: numberValue(activity?.activeRequests),
            queued: numberValue(activity?.queued),
            idle: activity?.idle === true
        }
    }
}

/** Apple silicon GPUs share system memory with the CPU. */
function isUnifiedMemory(gpus: { name: string }[]): boolean {
    return gpus.some(gpu => /\bapple\b/i.test(gpu.name))
}

export async function planCapacity(request?: CapacityPlanRequest): Promise<CapacityPlan> {
    if (!request) return parsePlan(undefined, 'no request')
    const gpus = getModularBridgeState()
        .getSelfGpus()
        .filter(gpu => gpu.vramBytes > 0)
    const params: JsonObject = {
        engine: request.engine,
        model: request.model,
        gpus: gpus.map(gpu => ({ name: gpu.name, vramBytes: gpu.vramBytes })),
        unified: isUnifiedMemory(gpus)
    }
    for (const key of [
        'windowTokens',
        'sessions',
        'reservePercent',
        'otherModelsBytes',
        'weightsBytes',
        'kvBytesPerToken'
    ] as const) {
        const v = request[key]
        if (typeof v === 'number' && v > 0) params[key] = v
    }
    if (request.cacheType) params.cacheType = request.cacheType
    try {
        return parsePlan(await callSessionProxy('capacity/plan', params))
    } catch (err) {
        return parsePlan(
            undefined,
            (getErrorString(err) ?? 'planning failed').replace(/^invalid params: /, '')
        )
    }
}

export async function getPlanner(): Promise<PlannerState> {
    return parsePlanner(await callSessionProxy('capacity/planner', {}))
}

export async function setPlanner(change?: PlannerChange): Promise<PlannerState> {
    const params: JsonObject = {}
    if (typeof change?.autoReload === 'boolean') params.autoReload = change.autoReload
    if (change?.save) params.save = { ...change.save }
    if (change?.remove) params.remove = { ...change.remove }
    return parsePlanner(await callSessionProxy('capacity/planner-set', params))
}

// ---------------------------------------------------------------------------
// Launch settings

/**
 * Splits pair-arguments-v1 text into tokens without unquoting them, so tokens
 * this module does not touch are written back exactly as the user wrote them.
 */
export function splitLaunchTokens(text: string): string[] {
    const tokens: string[] = []
    let current = ''
    let inToken = false
    let quote: '"' | "'" | null = null
    for (let i = 0; i < text.length; i++) {
        const ch = text[i]
        if (quote) {
            current += ch
            if (quote === '"' && ch === '\\' && (text[i + 1] === '"' || text[i + 1] === '\\')) {
                current += text[++i]
            } else if (ch === quote) {
                quote = null
            }
            continue
        }
        if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r') {
            if (inToken) tokens.push(current)
            current = ''
            inToken = false
            continue
        }
        if (ch === '"' || ch === "'") quote = ch
        current += ch
        inToken = true
    }
    if (inToken) tokens.push(current)
    return tokens
}

const ENV_ASSIGNMENT = /^[A-Za-z_][A-Za-z0-9_]*=/

function envName(token: string): string {
    return token.slice(0, token.indexOf('=')).toUpperCase()
}

/** Splits leading environment assignments from the engine arguments. */
function splitEnv(tokens: string[]): { env: string[]; args: string[] } {
    let i = 0
    while (i < tokens.length && ENV_ASSIGNMENT.test(tokens[i])) i++
    return { env: tokens.slice(0, i), args: tokens.slice(i) }
}

interface PlanNumbers {
    windowTokens: number
    sessions: number
    cacheType: KvCacheType
}

/**
 * Ollama takes the window and slot count from its environment, for every model
 * it runs: OLLAMA_CONTEXT_LENGTH per request, OLLAMA_NUM_PARALLEL slots. A
 * quantized cache needs flash attention.
 */
export function ollamaLaunchText(text: string, plan: PlanNumbers): string {
    const { env, args } = splitEnv(splitLaunchTokens(text))
    const managed = new Set([
        'OLLAMA_CONTEXT_LENGTH',
        'OLLAMA_NUM_PARALLEL',
        'OLLAMA_KV_CACHE_TYPE'
    ])
    const kept = env.filter(token => !managed.has(envName(token)))
    const added = [
        `OLLAMA_CONTEXT_LENGTH=${plan.windowTokens}`,
        `OLLAMA_NUM_PARALLEL=${plan.sessions}`
    ]
    if (plan.cacheType !== 'f16') {
        added.push(`OLLAMA_KV_CACHE_TYPE=${plan.cacheType}`)
        if (!kept.some(token => envName(token) === 'OLLAMA_FLASH_ATTENTION')) {
            added.push('OLLAMA_FLASH_ATTENTION=1')
        }
    }
    return [...kept, ...added, ...args].join(' ')
}

// llama.cpp options this module owns, with their spellings. Each takes a value.
const LLAMA_VALUE_OPTIONS = [
    ['-c', '--ctx-size'],
    ['-np', '--parallel'],
    ['-ctk', '--cache-type-k'],
    ['-ctv', '--cache-type-v']
]

/**
 * llama.cpp splits --ctx-size evenly across --parallel slots, so the total is
 * window × sessions. A quantized V cache needs flash attention.
 */
export function llamaCppLaunchText(text: string, plan: PlanNumbers): string {
    const { env, args } = splitEnv(splitLaunchTokens(text))
    const owned = LLAMA_VALUE_OPTIONS.flat()
    const kept: string[] = []
    for (let i = 0; i < args.length; i++) {
        const token = args[i]
        const name = token.includes('=') ? token.slice(0, token.indexOf('=')) : token
        if (owned.includes(name)) {
            if (!token.includes('=')) i++ // skip the value
            continue
        }
        kept.push(token)
    }
    const added = [
        '--ctx-size',
        String(plan.windowTokens * plan.sessions),
        '--parallel',
        String(plan.sessions)
    ]
    if (plan.cacheType !== 'f16') {
        added.push('--cache-type-k', plan.cacheType, '--cache-type-v', plan.cacheType)
        if (!kept.some(token => token === '-fa' || token.startsWith('--flash-attn'))) {
            added.push('--flash-attn', 'on')
        }
    }
    return [...env, ...kept, ...added].join(' ')
}

const SETTINGS_SETTLE_TIMEOUT_MS = 180_000

async function readSettings(
    nodeId: string,
    engine: EngineManagerName
): Promise<EngineSettingsSnapshot> {
    return parseEngineSettings(
        await getModularSupervisor().callProcess('broker', 'engine:get-settings', {
            nodeId,
            engine
        })
    )
}

async function waitForApplied(
    nodeId: string,
    engine: EngineManagerName,
    receipt: EngineSettingsReceipt
): Promise<EngineSettingsSnapshot> {
    const deadline = Date.now() + SETTINGS_SETTLE_TIMEOUT_MS
    for (;;) {
        const snap = await readSettings(nodeId, engine)
        if (snap.phase === 'failed')
            throw new Error(snap.error || 'the engine did not restart with the new settings')
        if (snap.appliedRevision >= receipt.revision && snap.phase !== 'applying') return snap
        if (Date.now() > deadline) throw new Error('timed out waiting for the engine to restart')
        await new Promise(resolve => setTimeout(resolve, 1000))
    }
}

let applying: Promise<ApplyPlanResult> | null = null

/** Applies the saved plan for one model. Only one apply runs at a time. */
export async function applyPlan(request?: {
    engine: string
    model: string
}): Promise<ApplyPlanResult> {
    if (!request) return { applied: false, message: 'no model given' }
    if (applying)
        return { applied: false, message: 'another plan is being applied; try again in a moment' }
    applying = applyPlanNow(request.engine, request.model)
    try {
        return await applying
    } finally {
        applying = null
    }
}

async function applyPlanNow(engine: string, model: string): Promise<ApplyPlanResult> {
    const state = await getPlanner()
    const plan = state.plans.find(p => p.engine === engine && p.model === model)
    if (!plan) return { applied: false, message: 'save a plan for this model first' }
    const numbers = {
        windowTokens: plan.windowTokens,
        sessions: plan.sessions,
        cacheType: plan.cacheType
    }
    const markApplied = async () => {
        await callSessionProxy('capacity/planner-set', {
            applied: {
                engine,
                model,
                windowTokens: plan.windowTokens,
                sessions: plan.sessions,
                cacheType: plan.cacheType
            }
        })
    }

    if (engine === 'lmstudio') {
        // LM Studio sets context per model load, outside PAIR's launch settings;
        // PAIR still limits traffic to the plan.
        await callSessionProxy('capacity/set', {
            engine,
            model,
            contextTokens: plan.windowTokens * plan.sessions,
            maxSessions: plan.sessions
        })
        return {
            applied: false,
            message: `PAIR now limits ${model} to ${plan.sessions} sessions of ${plan.windowTokens} tokens. LM Studio sets context when it loads a model: load it with a ${plan.windowTokens}-token context and ${plan.sessions} parallel slots to match.`
        }
    }
    if (engine !== 'ollama' && engine !== 'llamacpp')
        return { applied: false, message: `unknown engine ${engine}` }

    const nodeId = getModularBridgeState().getSelfId()
    if (!nodeId) return { applied: false, message: 'this node is not ready yet' }
    const managerName = engine as EngineManagerName
    const snap = await readSettings(nodeId, managerName)
    if (!snap.editable) {
        return {
            applied: false,
            message: `PAIR does not manage this ${engine} installation's launch settings${snap.reason ? ` (${snap.reason})` : ''}, so it cannot reload it.`
        }
    }
    const launchText =
        engine === 'ollama'
            ? ollamaLaunchText(snap.settings.launchText, numbers)
            : llamaCppLaunchText(snap.settings.launchText, numbers)
    if (launchText === snap.settings.launchText && snap.running) {
        await markApplied()
        return { applied: true, message: 'the engine already runs with these settings' }
    }
    const receipt = parseEngineSettingsReceipt(
        await getModularSupervisor().callProcess(
            'broker',
            'engine:apply-settings',
            {
                nodeId,
                engine: managerName,
                expectedRevision: snap.revision,
                // The broker requires one to correlate the operation's progress.
                requestId: randomUUID(),
                settings: { ...snap.settings, launchText }
            },
            MODULAR_ENGINE_LIFECYCLE_CALL_TIMEOUT_MS
        )
    )
    if (receipt.phase === 'failed') throw new Error('the engine rejected the new settings')
    if (receipt.phase === 'applying') await waitForApplied(nodeId, managerName, receipt)
    await markApplied()
    log.info({
        sublevel: 'apply',
        message: 'Applied capacity plan',
        data: {
            engine,
            model,
            windowTokens: plan.windowTokens,
            sessions: plan.sessions,
            cacheType: plan.cacheType
        }
    })
    return {
        applied: true,
        message: `${engine} restarted with ${plan.sessions} sessions of ${plan.windowTokens} tokens`
    }
}

// ---------------------------------------------------------------------------
// Idle auto-reload

const AUTO_RELOAD_INTERVAL_MS = 30_000
// The node must look idle on this many checks in a row, so a pause between
// two requests of a busy agent does not trigger a restart.
const IDLE_CHECKS_REQUIRED = 2
const FAILED_RETRY_MS = 10 * 60_000

let timer: ReturnType<typeof setInterval> | null = null
let idleChecks = 0
const failedAt = new Map<string, number>()

async function autoReloadTick(): Promise<void> {
    if (applying) return
    let state: PlannerState
    try {
        state = await getPlanner()
    } catch {
        return // services not up
    }
    const pending = state.plans.filter(
        p =>
            !p.applied &&
            p.engine !== 'lmstudio' &&
            Date.now() - (failedAt.get(`${p.engine}/${p.model}`) ?? 0) > FAILED_RETRY_MS
    )
    if (!state.autoReload || pending.length === 0 || !state.activity.idle) {
        idleChecks = 0
        return
    }
    if (++idleChecks < IDLE_CHECKS_REQUIRED) return
    idleChecks = 0
    const plan = pending[0]
    try {
        const result = await applyPlan({ engine: plan.engine, model: plan.model })
        if (!result.applied) failedAt.set(`${plan.engine}/${plan.model}`, Date.now())
        log.info({ sublevel: 'auto-reload', message: 'Idle auto-reload', data: { ...result } })
    } catch (err) {
        failedAt.set(`${plan.engine}/${plan.model}`, Date.now())
        log.warn({
            sublevel: 'apply',
            message: 'Idle auto-reload failed',
            data: {
                engine: plan.engine,
                model: plan.model,
                error: getErrorString(err) ?? 'unknown'
            }
        })
    }
}

export function startCapacityAutoReload(): void {
    if (timer) return
    timer = setInterval(() => void autoReloadTick(), AUTO_RELOAD_INTERVAL_MS)
    timer.unref?.()
}

export function stopCapacityAutoReload(): void {
    if (timer) clearInterval(timer)
    timer = null
    idleChecks = 0
}

/** Test hook: run one auto-reload check now. */
export const runAutoReloadCheckForTest = autoReloadTick
