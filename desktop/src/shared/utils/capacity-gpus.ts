// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CapacityPlan } from '@/shared/types/sessions'

/**
 * How a plan uses a node's GPUs, read from the split nvpair-proxy computed.
 * Shared by the dialog, which shows it, and the bridge, which writes it into
 * the engine's launch settings.
 */

/**
 * The GPU Ollama loads the model on alone, or -1 when the plan spreads it.
 * Ollama keeps a model on one GPU when the model and every session fit there,
 * and spreads it by layers only when they do not.
 */
export function oneGpuIndex(engine: string, plan: CapacityPlan, sessions: number): number {
    if (engine !== 'ollama' || plan.gpus.length < 2) return -1
    return plan.oneGpuSessions >= sessions ? plan.oneGpuIndex : -1
}

interface MultiGpuLaunch {
    /** llama.cpp: --split-mode layer --tensor-split with this value. */
    tensorSplit: string
    /** Ollama: OLLAMA_SCHED_SPREAD=1. */
    spreadGpus: boolean
}

/**
 * The GPU settings a plan needs in the engine's launch settings. llama.cpp
 * gets the layer split the plan was sized for; Ollama is told to spread when
 * the sessions do not fit on one GPU, so it does not try one GPU and spill the
 * rest onto the CPU. One GPU needs nothing.
 */
export function multiGpuLaunch(
    engine: string,
    plan: CapacityPlan,
    sessions: number
): MultiGpuLaunch {
    const none = { tensorSplit: '', spreadGpus: false }
    if (plan.error || plan.gpus.length < 2) return none
    if (engine === 'llamacpp') return { ...none, tensorSplit: plan.tensorSplit }
    if (engine === 'ollama') return { ...none, spreadGpus: oneGpuIndex(engine, plan, sessions) < 0 }
    return none
}

/** What applying writes for the GPUs, as it appears in the launch settings. */
export function multiGpuLaunchText(launch: MultiGpuLaunch): string {
    if (launch.tensorSplit) return `--split-mode layer --tensor-split ${launch.tensorSplit}`
    return launch.spreadGpus ? 'OLLAMA_SCHED_SPREAD=1' : ''
}
