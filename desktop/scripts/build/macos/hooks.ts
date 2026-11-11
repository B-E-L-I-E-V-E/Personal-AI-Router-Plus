// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from 'node:child_process'
import { chmodSync, closeSync, existsSync, openSync, readSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import type { AfterPackContext, BuildResult } from 'electron-builder'

function findAppBundle(appOutDir: string): string {
    const entries = readdirSync(appOutDir, { withFileTypes: true })
    const app = entries.find(e => e.isDirectory() && e.name.endsWith('.app'))
    if (!app) {
        throw new Error(`No .app bundle found in ${appOutDir}`)
    }
    return join(appOutDir, app.name)
}

/**
 * extraResources directories holding loose Go binaries. One: `cli-bin` carries
 * the services workers, `nvpair-tui`, and the Inference Demo's
 * `inference-dispatcher`.
 */
const GO_BINARY_RESOURCE_DIRS = ['cli-bin']

/**
 * The `.dmg` has no installer step (unlike the old `.pkg` postinstall that ran
 * `chmod a+rx` as root), so the bundled Go subprocess binaries must carry exec
 * bits in the shipped bundle. The build scripts already chmod them, but we
 * re-assert it here so a copy that dropped the mode bit can never produce a
 * non-executable binary in a release.
 */
function ensureGoBinaryExecBits(appPath: string): void {
    for (const dirName of GO_BINARY_RESOURCE_DIRS) {
        const dir = join(appPath, 'Contents/Resources', dirName)
        if (!existsSync(dir)) continue
        for (const entry of readdirSync(dir, { withFileTypes: true })) {
            if (!entry.isFile() || entry.name.includes('.')) continue
            chmodSync(join(dir, entry.name), 0o755)
        }
    }
}

/** Mach-O magic numbers, thin and universal, in either byte order. */
const MACH_O_MAGICS = new Set([
    0xfeedface, 0xfeedfacf, 0xcefaedfe, 0xcffaedfe, 0xcafebabe, 0xbebafeca
])

function isMachO(path: string): boolean {
    const fd = openSync(path, 'r')
    try {
        const head = Buffer.alloc(4)
        return readSync(fd, head, 0, 4, 0) === 4 && MACH_O_MAGICS.has(head.readUInt32BE(0))
    } finally {
        closeSync(fd)
    }
}

/**
 * Executables @electron/osx-sign does not sign: the privileged-helper pair
 * placed beside the main executable in Contents/MacOS, and the Go services in
 * Resources/cli-bin. codesign refuses to seal the app around an unsigned
 * subcomponent, and notarization rejects any unsigned or runtime-less binary,
 * so each is signed here first with the hardened runtime and a secure
 * timestamp. Mirrors the per-binary loop of NVIDIA's internal pipeline.
 */
function signNestedExecutables(appPath: string, identity: string): void {
    const mainExecutable = readdirSync(join(appPath, 'Contents/MacOS')).find(name =>
        appPath.endsWith(`/${name}.app`)
    )
    const dirs = ['Contents/MacOS', ...GO_BINARY_RESOURCE_DIRS.map(d => `Contents/Resources/${d}`)]
    for (const rel of dirs) {
        const dir = join(appPath, rel)
        if (!existsSync(dir)) continue
        for (const entry of readdirSync(dir, { withFileTypes: true })) {
            if (!entry.isFile() || (rel === 'Contents/MacOS' && entry.name === mainExecutable))
                continue
            const file = join(dir, entry.name)
            if (!isMachO(file)) continue
            execFileSync(
                'codesign',
                ['--sign', identity, '--force', '--timestamp', '--options', 'runtime', file],
                { stdio: 'inherit' }
            )
        }
    }
}

// Unsigned unless PAIR_MAC_SIGN=1, when the nested executables are signed here
// and electron-builder then signs and notarizes the bundle around them (see the
// mac block in electron-builder.config.ts).
export async function macAfterPack(context: AfterPackContext): Promise<void> {
    if (context.electronPlatformName !== 'darwin') return
    const appPath = findAppBundle(context.appOutDir)
    ensureGoBinaryExecBits(appPath)
    if (process.env.PAIR_MAC_SIGN === '1') {
        signNestedExecutables(
            appPath,
            process.env.PAIR_MAC_SIGN_IDENTITY || 'Developer ID Application'
        )
    }
}

export async function macAfterAllArtifactBuild(buildResult: BuildResult): Promise<string[]> {
    return buildResult.artifactPaths
}
