// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RouterStatus } from '@/shared/types/router'
import type { JsonValue } from './json-rpc-subprocess'
import { callSessionProxy } from './sessions-bridge'

/**
 * The PAIR router lives in nvpair-proxy, which owns its port and its routing.
 * The desktop only asks where it is listening, to show it in Endpoints.
 */
function parseRouterStatus(value: JsonValue | undefined): RouterStatus {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        return { listening: false, port: null }
    }
    const port = value.port
    const listening = value.listening === true && typeof port === 'number' && port > 0
    return { listening, port: listening && typeof port === 'number' ? port : null }
}

export async function getRouterStatus(): Promise<RouterStatus> {
    return parseRouterStatus(await callSessionProxy('router/status', {}))
}
