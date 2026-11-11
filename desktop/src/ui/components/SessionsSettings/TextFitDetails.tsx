// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Fragment, useState } from 'react'
import { Button, Stack, Text } from '@nvidia/foundations-react-core'
import { textKindFits } from './capacity-sliders'

const RIGHT = { textAlign: 'right' as const }

/**
 * How much text fits in one session's context window, by kind of text, and
 * when a reply budget is given, how long a reply can be. It follows the
 * sliders; More info says where the numbers come from.
 */
export default function TextFitDetails({
    tokens,
    replyTokens = 0,
    label
}: {
    tokens: number
    /** The reply budget; adds reply words and pages columns when positive. */
    replyTokens?: number
    label?: string
}) {
    const [open, setOpen] = useState(false)
    if (tokens <= 0) return null
    const replies = replyTokens > 0 ? textKindFits(replyTokens) : null
    return (
        <Stack gap="1">
            <Text kind="body/regular/sm" className="text-subtle-color">
                {label ?? `One conversation in ${tokens.toLocaleString()} tokens holds about:`}
            </Text>
            <div
                role="table"
                aria-label="Text that fits in the context window"
                style={{
                    display: 'grid',
                    gridTemplateColumns: replies
                        ? '1fr auto auto auto auto auto auto'
                        : '1fr auto auto auto auto',
                    columnGap: '0.75rem',
                    rowGap: '0.25rem',
                    alignItems: 'baseline'
                }}
            >
                <Text kind="body/semibold/sm" role="columnheader">
                    Text
                </Text>
                <Text kind="body/semibold/sm" role="columnheader" style={RIGHT}>
                    Chars/token
                </Text>
                <Text kind="body/semibold/sm" role="columnheader" style={RIGHT}>
                    Characters
                </Text>
                <Text kind="body/semibold/sm" role="columnheader" style={RIGHT}>
                    Words
                </Text>
                <Text kind="body/semibold/sm" role="columnheader" style={RIGHT}>
                    Pages
                </Text>
                {replies && (
                    <>
                        <Text kind="body/semibold/sm" role="columnheader" style={RIGHT}>
                            Reply words
                        </Text>
                        <Text kind="body/semibold/sm" role="columnheader" style={RIGHT}>
                            Reply pages
                        </Text>
                    </>
                )}
                {textKindFits(tokens).map((k, i) => (
                    <Fragment key={k.label}>
                        <Text kind="body/regular/sm" role="cell">
                            {k.label}
                        </Text>
                        <Text kind="body/regular/sm" role="cell" style={RIGHT}>
                            {k.charsPerToken}
                        </Text>
                        <Text kind="body/regular/sm" role="cell" style={RIGHT}>
                            {k.characters.toLocaleString()}
                        </Text>
                        <Text kind="body/regular/sm" role="cell" style={RIGHT}>
                            {k.words.toLocaleString()}
                        </Text>
                        <Text kind="body/regular/sm" role="cell" style={RIGHT}>
                            {k.pages.toLocaleString()}
                        </Text>
                        {replies && (
                            <>
                                <Text kind="body/regular/sm" role="cell" style={RIGHT}>
                                    {replies[i].words.toLocaleString()}
                                </Text>
                                <Text kind="body/regular/sm" role="cell" style={RIGHT}>
                                    {replies[i].pages.toLocaleString()}
                                </Text>
                            </>
                        )}
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
                    Models read text as tokens, and how much text a token holds depends on the text:
                    ordinary English packs about four characters into a token, while hashes, IDs,
                    encoded data and Chinese or Japanese take about one token per character, and an
                    emoji takes more than one. The figures were measured with qwen3’s tokenizer on
                    samples of each kind; other models give similar numbers for English and code and
                    differ more for other languages. Words for English and code were counted in the
                    samples; Chinese words average about one and a half characters; text without
                    words (JSON, base64, IDs, emoji) is counted in standard words of five characters
                    and a space. A page is 500 words, single spaced. The window holds the whole
                    conversation: your messages, the model’s replies and the earlier turns it
                    re-reads. The reply columns are the longest reply the reply budget allows; a
                    thinking model spends part of that budget reasoning before it answers, so its
                    visible answer is shorter. Counts are rounded down.
                </Text>
            )}
        </Stack>
    )
}
