// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Fragment, useState } from 'react'
import { Button, Stack, Text } from '@nvidia/foundations-react-core'
import { textKindFits } from './capacity-sliders'

const RIGHT = { textAlign: 'right' as const }
const GROUP = { textAlign: 'center' as const, borderBottom: '1px solid currentColor' }

/**
 * How much text one session's window holds, by kind of text, in three
 * groups: the largest post (the window less the reply budget), the longest
 * reply, and the whole window. It follows the sliders; More info says where
 * the numbers come from.
 */
export default function TextFitDetails({
    windowTokens,
    replyTokens
}: {
    windowTokens: number
    replyTokens: number
}) {
    const [open, setOpen] = useState(false)
    const postTokens = windowTokens - replyTokens
    if (windowTokens <= 0 || postTokens <= 0) return null
    const posts = textKindFits(postTokens)
    const replies = textKindFits(replyTokens)
    const windows = textKindFits(windowTokens)
    const cell = (value: number) => (
        <Text kind="body/regular/sm" role="cell" style={RIGHT}>
            {value.toLocaleString()}
        </Text>
    )
    const head = (label: string) => (
        <Text kind="body/semibold/sm" role="columnheader" style={RIGHT}>
            {label}
        </Text>
    )
    const group = (label: string, span: number) => (
        <Text
            kind="body/semibold/sm"
            className="text-subtle-color"
            style={{ ...GROUP, gridColumn: `span ${span}` }}
        >
            {label}
        </Text>
    )
    return (
        <Stack gap="1">
            <Text kind="body/regular/sm" className="text-subtle-color">
                A post can use {postTokens.toLocaleString()} tokens of the{' '}
                {windowTokens.toLocaleString()}-token window, and the response up to{' '}
                {replyTokens.toLocaleString()}. That is about:
            </Text>
            <div
                role="table"
                aria-label="Text that fits in a post, a reply and the whole window"
                style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr repeat(8, auto)',
                    columnGap: '0.75rem',
                    rowGap: '0.25rem',
                    alignItems: 'baseline'
                }}
            >
                <span />
                <span />
                {group('One post', 2)}
                {group('The response', 2)}
                {group('Whole window', 3)}
                <Text kind="body/semibold/sm" role="columnheader">
                    Text
                </Text>
                {head('Chars/token')}
                {head('Post words')}
                {head('Post pages')}
                {head('Response words')}
                {head('Response pages')}
                {head('Total characters')}
                {head('Total words')}
                {head('Total pages')}
                {posts.map((k, i) => (
                    <Fragment key={k.label}>
                        <Text kind="body/regular/sm" role="cell">
                            {k.label}
                        </Text>
                        <Text kind="body/regular/sm" role="cell" style={RIGHT}>
                            {k.charsPerToken}
                        </Text>
                        {cell(k.words)}
                        {cell(k.pages)}
                        {cell(replies[i].words)}
                        {cell(replies[i].pages)}
                        {cell(windows[i].characters)}
                        {cell(windows[i].words)}
                        {cell(windows[i].pages)}
                    </Fragment>
                ))}
            </div>
            <div>
                <Button
                    size="small"
                    kind="tertiary"
                    aria-expanded={open}
                    onClick={() => setOpen(o => !o)}
                >
                    {open ? 'Less info' : 'More info'}
                </Button>
            </div>
            {open && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    A session’s window is one budget for everything the model reads and writes. A
                    post is the most a session’s messages and history may hold: the window less the
                    room kept for the reply, so in a longer conversation each new post has less
                    room. The reply columns are the longest reply that room allows; a thinking model
                    spends part of it reasoning before it answers, so its visible answer is shorter.
                    The whole window is post and reply together. Models read text as tokens, and how
                    much text a token holds depends on the text: ordinary English packs about four
                    characters into a token, while hashes, IDs, encoded data and Chinese or Japanese
                    take about one token per character, and an emoji takes more than one. The
                    figures were measured with qwen3’s tokenizer on samples of each kind; other
                    models give similar numbers for English and code and differ more for other
                    languages. Words for English and code were counted in the samples; Chinese words
                    average about one and a half characters; text without words (JSON, base64, IDs,
                    emoji) is counted in standard words of five characters and a space. A page is
                    500 words, single spaced. Counts are rounded down.
                </Text>
            )}
        </Stack>
    )
}
