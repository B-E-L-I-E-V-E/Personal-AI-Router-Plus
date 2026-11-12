// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NodeServingState } from '@/shared/types/node-state'

/** The states a node card offers to move to from each state. */
export function servingActions(state: NodeServingState): NodeServingState[] {
    switch (state) {
        case 'running':
            return ['paused', 'stopped']
        case 'paused':
            return ['running', 'stopped']
        case 'stopped':
            return ['running']
    }
}
