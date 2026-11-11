// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'

const hasFeed = { value: false }
const checkForUpdates = vi.fn(async () => undefined)

vi.mock('node:fs', async importOriginal => {
    const actual = await importOriginal<typeof import('node:fs')>()
    return { ...actual, existsSync: () => hasFeed.value }
})
vi.mock('electron', () => ({
    app: { isPackaged: true },
    BrowserWindow: { getAllWindows: () => [] }
}))
vi.mock('electron-log', () => ({ default: { info: vi.fn(), error: vi.fn() } }))
vi.mock('electron-updater', () => ({
    default: { autoUpdater: { on: vi.fn(), checkForUpdates } }
}))
vi.mock('@/electron/connector', () => ({ destroyConnector: vi.fn() }))
vi.mock('@/electron/updater/update-notification', () => ({ notifyUpdateAvailable: vi.fn() }))
vi.mock('@/electron/updater/pending-update-store', () => ({
    clearPendingUpdate: vi.fn(),
    readPendingUpdateVersion: () => null,
    writePendingUpdateVersion: vi.fn()
}))

// A source build carries no app-update.yml, so it has nothing to check
// against. It must say so instead of failing every check with an error.
describe('updater without an update feed', () => {
    beforeEach(() => {
        vi.resetModules()
        checkForUpdates.mockClear()
        Object.defineProperty(process, 'resourcesPath', {
            value: '/app/Resources',
            configurable: true
        })
    })

    it('reports updates as unavailable and never checks', async () => {
        hasFeed.value = false
        const updater = await import('@/electron/updater')
        updater.initializeUpdater()
        await updater.checkForUpdates()
        expect(updater.getUpdateStatus()).toMatchObject({ phase: 'unavailable', error: null })
        expect(checkForUpdates).not.toHaveBeenCalled()
    })

    it('still checks a build that has a feed', async () => {
        hasFeed.value = true
        const updater = await import('@/electron/updater')
        updater.initializeUpdater()
        await updater.checkForUpdates()
        expect(updater.getUpdateStatus().phase).not.toBe('unavailable')
        expect(checkForUpdates).toHaveBeenCalledTimes(1)
    })
})
