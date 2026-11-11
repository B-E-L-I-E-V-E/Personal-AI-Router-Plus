// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type SettingsWindowTab = 'cluster' | 'sessions' | 'service'

export function isSettingsWindowTab(value: string): value is SettingsWindowTab {
    return value === 'cluster' || value === 'sessions' || value === 'service'
}
