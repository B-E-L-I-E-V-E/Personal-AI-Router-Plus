// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The PAIR router (nvpair-proxy `router/status`): one loopback port that routes
 * a request by its model to whichever engine, on whichever node, has it.
 */
export interface RouterStatus {
    listening: boolean
    /** The bound port; null while the router is not listening. */
    port: number | null
}
