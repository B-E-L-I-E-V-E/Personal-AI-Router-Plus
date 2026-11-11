<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Microservice: Engine Proxy (`nvpair-proxy`)

> **Status: implemented.** House spec style (`nvpair-cluster-manager`,
> `nvpair-workload-manager`, `nvpair-engine-manager`, `nvpair-job-scheduler`).
> Where this file and `README.md` disagree, this file wins.

## 1. Purpose

Carries inference **to** the node that should serve it. One process hosts a
**facade** per enabled engine: each facade is an HTTP reverse proxy speaking its
engine's own dialect, filtering to nodes that advertise the requested model,
choosing among them, and failing over when one refuses.

It is a data plane, not a policy component. Ranking belongs to
`nvpair-job-scheduler`, and cluster membership and trust to
`nvpair-cluster-manager`. `nvpair-engine-manager` owns an engine's *install and
process* lifecycle; the per-request *workload* lifecycle events in §5.5 are this
process's own. What it owns beyond those events is the choice of *which
advertised owner receives this request*, and the local accounting that makes
that choice good under bursts.

### 1.1 What a caller is promised

The proxy aims to get every inference request **started** on some node that can
serve it, and spends a bounded amount of effort doing so. Stated precisely:

> While budget remains (§5.1), a failure that happens *before* the engine begins
> producing output is retried rather than returned. Once output begins, the
> request is committed to that node and is the caller's to finish with.

Failures absorbed that way include a retryable status, a transport error, an
engine that answers its headers and then produces nothing, and a target that
leaves discovery mid-request. Waiting for an owner to *exist* costs no attempt
at all, so a node blinking out does not spend the budget.

Four things bound that aim, and a caller should design for them.

**Effort is bounded, and which bound ran out changes what the caller gets.**
Absorption holds only while budget remains. Past that there are two shapes, both
detailed in §5.4:

- **The dispatch budget ran out.** The final permitted attempt is never
  retried, so its own failure is the answer: the upstream's status and body
  passed through unchanged, or a `502` if it failed at the transport and a
  `504` if it answered and produced nothing.
- **The deadline passed between attempts, or no owner came back.** Nothing is
  in flight to answer, so the proxy generates a `503` with `Retry-After` that
  names which of those it was.

Either way this is a persistent, bounded attempt to start — not an assurance
that every submission eventually runs.

**"Started" is the guarantee, not "finished".** Commitment ends at the first
byte of response body (§5.2), so `completed` means the engine began generating,
not that a whole answer arrived.

**A truncated stream still reports `completed`**, for two reasons that compound.
To the caller it is the same situation either way — it holds partial output and
is the only party that can decide what to do with it, which is what the
retrying harness above this layer is for. And the proxy genuinely cannot tell a
truncated stream from a complete one: a streaming response carries no
`Content-Length`, and the end-of-turn marker is engine-specific, so recognising
it would mean interpreting response content, which §9 forbids. (Reading a
session's token counts, §5.12, is narrower: it reads only the numeric usage
fields and never decides anything about the content.)

**A non-streaming request is the weakest case.** Its headers arrive only when
the response is complete, so "queued inside the engine" and "generating right
now" look identical from outside. The attempt cap therefore has to cover whole
generation, and when it expires the proxy cannot tell whether it abandoned a
stalled request or a working one — so retrying may discard real progress. That
is a genuine limitation, not a safety margin (§5.6).

## 2. Scope

**In scope**

- One listener per enabled engine, serving loopback plaintext for local clients
  and pin-gated cluster mTLS for peers on the same port.
- Model-owner filtering, target precedence, and retryable failover.
- Per-request workload lifecycle events.
- The process-wide reservation map that makes concurrent dispatch spread.
- The persisted per-engine port a user chose through `set-port`.

**Out of scope**

- Deciding node rank (scheduler), membership or pinning (cluster-manager),
  engine install/start/stop (engine-manager).
- Any cryptography of its own: it consumes the trust fabric, never builds it.
- Model search or catalog browsing.

## 3. Process model

The process starts with **no engine and no listener**. The broker then sends one
`facade/enable` per engine, carrying that engine's port and any alias addresses.

A flag cannot express this. The broker plans a different port for each engine —
Ollama's managed facade wants `:11434`, LM Studio's wants `:1234`, and
llama.cpp's managed facade wants `:8080`; any may be absent so the child keeps
its own persisted port — and a single-valued flag carries only one plan.

### 3.1 Why one process

Between scheduler snapshots a facade takes short-lived **reservations** for work
it has dispatched but that the scheduler has not yet observed. Those live in the
process, and every facade shares them. If each engine had its own proxy
process, none could see the others' reservations, so simultaneous bursts across
Ollama, LM Studio, and llama.cpp could select the same node, each incorrectly
believing it was idle.
Sharing the map is the reason the engines share a process.

### 3.2 What sharing costs

**Shared fate on the process.** A crash takes every facade down; the supervisor
restarts them together and the crash is reported once, against the process.

Inside the process the boundaries are finer, and deliberately so:

| Failure | Blast radius |
| --- | --- |
| Bind race lost on enable | That engine only; reported so the broker can retry elsewhere |
| Port cannot be re-listened | That engine only; its ownership gate is released |
| Panic handling a request | That request only; logged with stack, answered as an internal error |
| Process crash or exit | Every facade |

The first three failures are contained within their listed boundaries. Only a
process crash or exit takes down every facade. Containment matters here in a way
it would not for a single-engine process: one engine's bind race must not take
another engine's working listener with it.

## 4. Addressing

Every facade-scoped message carries the engine it concerns, in both directions.
Without it the engine would be implied by which process a message travelled
through, which is precisely the property one process destroys.

The address is a bare engine id (`ollama:ready`), deliberately **not** the
component name (`ollama-proxy:ready`). The component form is how a client
addresses the component through the broker; the two meet in the broker's relay,
which takes one off and puts the other on.

| Class | Addressed? | Examples |
| --- | --- | --- |
| Facade-scoped | yes | `ready`, `error`, `node/*`, `errors:*`, `set-port`, `discovery:subscribe`, `discovery:nodes` |
| Process-scoped | no | `log/set-level`, `node/set-priority`, `workload:*`, `discovery:node-activity` |
| Bootstrap | names its engine in the payload | `facade/enable` |

`facade/enable` is unaddressed because it runs *before* the facade it names
exists. Only a known engine id counts as an address: plenty of unaddressed
methods contain a colon, so treating `errors:report` as engine `errors` would
strip a real method down to `report`.

An unaddressed facade-scoped message resolves to **no** facade. Falling back to
"the only facade" would work with one engine enabled and misroute silently with
two.

## 5. Request path

For a model-bearing inference request:

1. Filter a request-local discovery snapshot to nodes whose per-engine inventory
   advertises the requested model. Ollama normalizes the implicit `:latest` tag;
   LM Studio and llama.cpp ids match exactly. An empty owner set returns a local
   `502` without contacting an engine.
2. Order the eligible owners: explicit `node/select` pin, then the scheduler's
   priority list, then deterministic default ordering.
3. Reserve the least estimated-loaded scheduler-listed candidate and move it to
   the front (§6).
4. Forward, failing over through the remaining candidates on a retryable status
   or transport error. An upstream model `404` counts as retryable, because
   positive inventory can be stale.
5. When a round's candidates are exhausted, back off and re-resolve (§5.1),
   until the dispatch budget or the deadline runs out.

An ineligible manual selection cannot override the capability gate, and failover
never broadens to an excluded node.

The llama.cpp facade exposes `GET /models` and `GET /v1/models` as aliases for
the fleet's llama.cpp inventory. Both query every candidate's `GET /models`
concurrently and merge duplicate model ids into an OpenAI list envelope
(`{"object":"list","data":[...]}`). A selected node affects candidate ordering,
not inventory scope. Its inference routes remain `/v1/*`.

### 5.1 Retry bounds

| Bound | Value | Governs |
| --- | --- | --- |
| `maxDispatchAttempts` | 5 | how many dispatches may fail |
| `jobDeadline` | 10 minutes from request creation | how long new attempts may keep being started, including while waiting for an owner to exist |
| `firstBodyTimeout` | `proxyResponseTimeout` (120s) | one attempt's wait for first content, measured from the arrival of headers |

The retry budget applies only to inference requests. Routes not classified in
the engine's route table are still forwarded verbatim, and some perform
state-changing operations; for example, `POST /api/pull` starts a model
download. Since any `5xx` is retryable regardless of route, re-resolving and
redispatching such a request could start the operation several times, which is
both non-obvious and wasteful. For a non-inference request the proxy therefore
resolves candidates once and tries each at most once. It does not re-resolve
candidates, back off between rounds, or apply the inference deadline.

**An attempt is consumed by a dispatch, never by a resolution.** Finding no
eligible owner sends nothing, so a job whose only owner briefly drops out does
not burn its budget while the node comes back.

**Elapsed time per attempt is the sum of three budgets, not one.** They run in
sequence: the dial (`proxyDialTimeout`, 10s), then headers
(`proxyResponseTimeout`, 120s, enforced by the transport), then first content
(`firstBodyTimeout`, 120s, which starts only once headers arrive). One attempt
can therefore occupy around 240s — the two 120s caps back to back, since a slow
dial is a LAN rarity — and nothing caps their sum.

Both bounds are checked before every dispatch and **never truncate an attempt
already in flight**. Killing one seconds from its first token and then failing
the job for being out of time would spend the entire wait and discard the
result. `jobDeadline` is therefore a "no new attempts after" line rather than a
ceiling on elapsed time, and the worst case is the deadline plus one whole
attempt: a last dispatch starting just under 600s and running its full 240s
puts time to first content near fourteen minutes.

Which bound ends a job depends on how its attempts fail, and all three cases
are real:

- **Fast failures** — a refused connection, a prompt `503` — cost about a
  second, so all five attempts fit well inside the deadline and the dispatch
  budget is what ends the job. This is the common case.
- **Slow failures** — an engine that accepts and then withholds everything —
  cost close to a whole attempt each, so the deadline arrives first. At 240s
  per attempt the fourth dispatch would start past 600s, so the job ends after
  three attempts with budget unspent.
- **No owner at all** costs no attempt, so the deadline is the only thing that
  can end the wait.

The last case is why the deadline cannot be dropped as redundant with the
attempt count, and the middle one is why five attempts is a ceiling rather than
a promise. **Do not remove either bound as unreachable.**

#### Waiting for an owner

There is no memory of nodes seen in the past: every resolution runs against the
live discovery snapshot, so "eligible owner" always means "advertising this
model right now". What differs is the position in the request's life.

- **No owner at admission** — nothing has been dispatched, so there is nothing
  to wait for being interrupted. Return `502` immediately (§5.4). A model
  nobody advertises will not become servable by waiting.
- **No owner after at least one dispatch** — an owner existed a moment ago and
  went away, so the proxy waits for one to reappear, re-resolving on the backoff
  schedule. This costs no attempt, so `jobDeadline` is the only thing that ends
  it: the deadline *is* the grace period for a node coming back.

The asymmetry is deliberate. Waiting is justified by evidence that a capable
owner existed within this request's lifetime, and nothing weaker.

Candidates are re-resolved every round, so a node that recovered or a model that
finished pulling becomes eligible mid-retry. The same node may be retried — with
one owner, all five attempts go to it.

A requester that has gone away ends the retry immediately: nobody is left to
receive the answer, so the remaining attempts belong to work someone is waiting
for. Its end is reported as `cancelled` and is never attributed to a missing
node.

### 5.2 The commit point

**Commit is the first byte of response body, not the response headers.**

Headers are not a valid commit signal: they may show only that the engine
accepted the request. Treating them as commitment would disable the header
timeout and prevent failover while the request remains queued inside the
engine. The first content byte is the first engine-independent evidence that
generation has begun.

The difference is engine-specific and large. Measured against LM Studio 0.4.x
with a model loaded at `--parallel 1`, probing while a long generation held the
single slot:

| Request | Headers | First body byte |
| --- | --- | --- |
| streaming | 12ms, on accept | 35.5s, when the slot freed |
| non-streaming | 38.9s, withheld until complete | with the headers |

Nothing arrives during that wait — no SSE keepalive, no empty role delta — so a
byte is real content rather than a heartbeat.

**Only the existence of the byte matters, never its value.** The proxy does not
look at it: inspecting content would make the gate engine-specific, and it
would mean reading response data §9 forbids. Whatever the byte is, it is
spliced back onto the front of the stream so the client receives the response
intact.

`EOF` with no bytes counts as a commit. This is defensive rather than expected
— an inference route answering `200` with an empty body is an engine
misbehaving — but treating it as a stall would spend the whole budget waiting
for content that is never coming and then fail a request the engine considered
answered.

A first-content timeout **never** commits, including on the final permitted
attempt, where it is terminal with `504`. Committing there would leave the peek
reader blocked on the same body, racing two readers on one stream, and against
a silent upstream the copy would block with nothing left to interrupt it.

The gate is scoped to inference, and costs a non-streaming response nothing,
since its headers and body arrive together. That is why the request's own
`stream` flag never has to be consulted and no per-dialect default has to be
guessed, and it is what makes the boundary engine-independent — correct whether
an engine withholds headers or sends them on accept.

**After commit there is no retry.** A stream truncated from there is the
caller's to finish with (§1.1).

#### Known gap: a silent upstream after commit

There is no upstream read or idle deadline. If a committed engine keeps the
connection open and stops producing, nothing in this process ends the request:
the client write deadline needs a write to trip, and a silent upstream produces
nothing to write. A *crashed* engine is handled — the copy fails and §5.4's
deferred reporter emits the terminal — but a *frozen* one holds the request
until the client gives up.

Closing it needs a read deadline refreshed on every chunk, and a value chosen
so a slow-but-healthy generation is not cut off mid-answer. That judgement is
not made here.

### 5.3 Per-attempt cancellation

A watcher cancels an attempt promptly when its target leaves discovery.
Otherwise the attempt could stay blocked until the first-content timeout while
the broker's node-loss sweep had already marked the workload failed, so the UI
would report failure while the client was still waiting.

Each attempt runs on a context derived from the request's, and that child
relationship is load-bearing: the disconnect watcher and the outcome
classification both read the *parent*, so cancelling a child cannot be mistaken
for the requester leaving. Confusing the two would either report a job that went
on to succeed as `cancelled`, or let a real disconnect keep burning attempts.

**Commit and cancellation are mutually exclusive, decided by one atomic
claim.** Both can become true at once — the watcher's discovery check can pass
while the commit path is already running, and the work between the arrival of
the first byte and the end of the commit includes a synchronous notification
write and two mutex acquisitions. A pair of signals cannot express which
happened first, so the attempt has a single state that each side claims by
compare-and-swap; the loser becomes a no-op.

**A committed first byte wins that claim.** A delivered byte is direct evidence
that this node is serving this request, whereas an absence from discovery can be
a transient announcement gap. If the watcher claims first the attempt is
abandoned without committing, so the request retries rather than being served a
stream the proxy then truncates itself.

### 5.4 Status handling and outcome classification

Whether a status is retried depends on *when* it arrives, not only on its class.
Before commit the proxy may try elsewhere; at commit the status becomes the
client's answer.

| Upstream status | Pre-commit, budget remaining | Workload state if it is the answer |
| --- | --- | --- |
| `2xx` | commits (§5.2) | `completed` |
| `3xx` | not retried; not expected from an engine, and forwarded verbatim if one sends it | `failed` |
| `4xx` except `404`/`408`/`429` | not retried — the same request fails identically everywhere, so retrying only spends budget | `failed` |
| `404` | retried on an inference route only: positive model inventory can be stale | `failed` |
| `408`, `429` | retried | `failed` |
| `5xx` | retried | `failed` |

Anything not retried commits and becomes the client's answer immediately, with
no further dispatch, whatever budget is left.

These are the statuses the proxy itself returns:

| Condition | Status |
| --- | --- |
| No owner advertises the model, nothing dispatched | `502` |
| Final permitted attempt returned a status | that status and body, unchanged |
| Final permitted attempt failed at the transport | `502` |
| Final permitted attempt answered but produced no content | `504` |
| The loop ended between attempts, nothing in flight | `503` with `Retry-After` |

The final permitted attempt is never retried, so it answers for itself. That is
why exhausting the dispatch budget does not produce a `503`: the fifth attempt
is the answer. The `503` is for a loop that ended with no attempt to answer,
which means time ran out.

Its body names which, because the two call for different responses from the
caller:

- **the deadline passed while attempts were still failing** — an owner was
  present throughout, and slow attempts reached 10 minutes before they reached
  five dispatches (§5.1);
- **no owner became available** — the candidate set was empty and the wait for
  a node to come back timed out.

A third reason, **every dispatch attempt failed**, guards the case where the
loop ends on the count with nothing having answered. The paths above make that
unreachable today; it is cheap to keep and wrong to rely on.

The resulting workload state:

| Outcome | State |
| --- | --- |
| committed 2xx, streamed to the end | `completed` |
| committed 2xx, truncated by the node dying | `completed` |
| committed non-2xx | `failed` |
| client disconnected, or dead-client write deadline tripped | `cancelled` |
| cancelled by our own shutdown | `cancelled` |
| budget or deadline exhausted pre-commit | `failed` |

`cancelled` distinguishes requester-initiated termination from an actionable
engine, node, or routing failure. Client disconnects and shutdown cancellation
therefore do not appear in the failed bucket.

**Reporting is deferred.** A mid-copy error is never returned to the handler:
`httputil.ReverseProxy` converts it into `panic(http.ErrAbortHandler)` when
running under a real server. That panic bypasses ordinary code following the
proxy call, so terminal workload and request reporting must run in a `defer`.
The deferred reporter records the abort as operational detail, emits the
terminal event, and re-panics so `http.Server` still closes the connection.
Reservation release is also deferred so it survives the same unwind.

That panic is not the failure mode §3.2 describes, despite the shared word.
`http.ErrAbortHandler` is a sentinel `net/http` defines to mean "abandon this
response without logging a stack", and `http.Server` recovers it silently by
contract; re-panicking is how the connection gets closed, and swallowing it
would leave the client waiting on a response that will never be finished.
§3.2's entry covers an *unexpected* panic — a bug — which is logged with its
stack and answered as an internal error.

### 5.5 Workload lifecycle

| Event | When | State | `scheduledOn` |
| --- | --- | --- | --- |
| `workload:submitted` | admitted | `queued` | empty |
| `workload:submitted` | each dispatch | `queued` | the target |
| `workload:submitted` | between attempts | `queued` | empty |
| `workload:started` | commit | `running` | the node that served |
| `workload:completed` / `workload:errored` | terminal | per §5.4 | unchanged |

`submitted` covers the whole pre-commit life; `started` means the engine is
producing content. That split is forced as well as honest: the broker's store
merges by state rank and rejects a lower one, so a job that claimed `running` at
dispatch time could not return to `queued` for its next attempt and the retry
would be invisible. Keeping the loop inside one state, with `scheduledOn`
carrying the detail, is the only shape that store can represent.

Every event carries `seq`, the producer's event counter from 1. The
workload-manager's inter-node dedup is a permanent set, so a retry returning to
a placement it already used — `queued` on A, cleared, `queued` on A again, which
this loop produces routinely — would otherwise be indistinguishable from a
redelivery and dropped by every peer.

A job admitted but not yet dispatched has no execution node. Consumers must
treat an absent `scheduledOn` as "not placed" rather than assuming a node.

### 5.6 Coverage gap: non-streaming requests

A non-streaming request is the case the retry policy serves worst, and the
weakness is the opposite of too little retrying.

Its headers arrive only when the response is complete, so from outside the
engine "queued behind other work" and "generating right now" are
indistinguishable. The 120s header cap therefore has to cover whole generation
— the first-content wait adds nothing here, since the body arrives with the
headers — and when it expires the proxy retries without being able to tell
whether it just abandoned a stalled request or a working one. A generation
legitimately slower than that cap is restarted elsewhere, discarding real
progress, and the restart faces the same cap.

Nothing in the current engine surface fixes this: there is no progress signal
before the response is complete. A caller that cares should prefer a streaming
request, where the first content byte gives the proxy the evidence it lacks
here.

### 5.7 Accepted behavior

**An abandoned attempt may still run inside the engine.** None of the engines
PAIR supports today exposes a way to withdraw a request it has accepted, and all
of them queue internally, so a retried job can have more than one copy generating while
the abandoned copy keeps its node busy.

Whether closing the connection cancels anything upstream is **unverified** —
plausible, engine-specific, and never measured. The behavior above is stated as
the pessimistic case for that reason.

This is accepted rather than overlooked: those internal queues are invisible and
un-retargetable, which is the problem the retry policy works around rather than
a mechanism it can build on. Do not reach for engine environment bounds or
occupancy probes to "fix" it — LM Studio exposes no such signal at all, and an
adopted engine keeps its own environment regardless. An engine that did offer a
real cancellation API would change this calculus, and the claim should be
revisited per engine rather than assumed to hold forever.

### 5.8 Browser policy

Each facade preserves its engine's CORS policy, including missing headers and
permission denials. Ingress gates apply before preflight forwarding. Proxy errors
grant no CORS permissions. Multi-target preflights intersect permissions from all
responding engines without forwarding caller credentials; unavailable engines
are skipped, and no responders yields 502. Preflights take no reservations.

Model inventories with an Origin require agreement from every responding engine:
a denial yields 403, an invalid inventory yields 502 without a partial list. An
invalid-inventory error retains CORS headers only when all responding engines
approve. See README.md for the complete forwarding and intersection rules.

### 5.9 Request ID

Every request handled by a facade carries one caller-facing request ID in the
`X-Request-ID` header.

- **Accepted** from the client when it is 1–128 characters drawn from letters,
  digits, `.`, `_`, `:` and `-`. That covers UUIDs, ULIDs and common trace
  formats while keeping whitespace, header-injection and log-forging characters
  out.
- **Minted** as a random version 4 UUID when the client sent none or sent one
  outside that set. An unacceptable value is replaced, never refused: the ID is
  a correlation aid and must not fail inference.
- **Echoed** on every response, including rejections, preflights, model lists
  and the proxy's own error bodies. Exactly one value is returned; an upstream
  that sets its own `X-Request-ID` does not add a second.
- **Forwarded** on every dispatch attempt, so a peer's ingress and the engine
  see the same value.
- **Recorded** as `request_id` on `proxy/request-started` and `proxy/request`,
  as `requestId` on every `workload:*` event, and as `request_id` on
  per-request log lines.

The request ID is distinct from the event `id` and `Workload.id`. Those are a
per-facade counter that restarts with the process and form part of the workload
store key; the request ID is opaque, not a key, and need not be unique across
nodes or clients. A client that reuses an ID gets the same ID back and nothing
else changes.

### 5.10 Sessions and agents

An inference request may name a conversation with `X-PAIR-Session-ID` and the
agent that owns it with `X-PAIR-Agent-ID`. Both use the request ID's character
set (§5.9). On any non-inference route both headers are forwarded untouched and
otherwise ignored.

- **A malformed header is refused with `400`**, not replaced. A substituted
  session ID would silently start a different conversation.
- **The first request creates the session**, and the agent it names (or none)
  owns it. Every later request for that session must name the same agent,
  including naming none; otherwise it is refused with `409`. Loopback has no
  authentication, so this guards against two agents colliding on one ID, not
  against a hostile local process.
- **Refusals happen before candidate selection**, so they take no reservation,
  count no attempt, and never reach an engine.
- **Affinity.** For each engine, the session remembers the node that committed
  its last turn (§5.2) and puts that node first in the next turn's first round.
  The engines keep the evaluated prompt in their KV cache, so landing on the
  same node reuses that work. Affinity ranks below an explicit manual selection
  and, like it, cannot override the capability gate: a node that no longer
  advertises the model is absent from the candidates and the scheduler's order
  stands. Later retry rounds use the ordinary order, and the session re-pins to
  whichever node commits.
- **Echo and records.** The response carries `X-PAIR-Session-ID`. Request
  events carry `session_id` and `agent_id`; workload events carry `sessionId`,
  with the agent in `requesterId`.

Sessions are held in memory by the proxy on the node where requests originate,
because that node makes the routing decision; nothing is replicated to peers.
The store holds IDs, the owning agent, timestamps, an in-flight count and the
node per engine. It never holds prompts or responses (§9).

### 5.11 Session lifetime and API

A session expires after a **sliding idle timeout**: 10 minutes by default,
adjustable from 1 to 60 minutes for the whole process (`sessions/configure`, or
`--session-idle-timeout`) and per session at creation or by `PATCH`.

- **What slides the window:** a request for the session starting or ending,
  and `POST /pair/v1/sessions/{id}/keepalive`.
- **Waiting is activity.** A session with a request in flight never expires,
  however long the engine takes or the request waits for an owner (§5.1). The
  window restarts when the last such request ends.
- **Optional maximum lifetime** (`maxLifetimeSeconds`, off by default) caps the
  total length even under keepalives. It too waits for in-flight work.
- **Expiry and ending are complete.** The session's record is removed, any
  request still running for it is cancelled (reported `cancelled`, exactly as a
  client disconnect), and the ID is tombstoned for 24 hours: reusing it returns
  `410 Gone` rather than silently opening a new conversation under the old
  name. Expiry is checked on every access, and a sweep every 15 seconds
  reclaims abandoned sessions.
- **Limit.** At most `maxSessions` (default 4096) live sessions. A new one past
  the limit is refused with `503` until one ends or expires.

The proxy itself answers `/pair/v1/sessions` on every facade port; it is never
forwarded to an engine and, being on the plaintext personality (§7), is
loopback-only. Create (`POST`, optional `sessionId`, `agentId`,
`idleTimeoutSeconds`), list (`GET`, optional `agentId` filter), read, keepalive,
`PATCH idleTimeoutSeconds`, and `DELETE`. Every call that changes a session
follows the ownership rule of §5.10. A session object reports `expiresAt` and a
suggested `keepaliveIntervalSeconds` of one third of its window.

`sessions/configure` is a process-wide JSON-RPC request with optional
`idleTimeoutSeconds`, `maxLifetimeSeconds`, `maxSessions`,
`queueTimeoutSeconds` and `maxQueued`; omitted fields keep their value, an empty
payload only reads, and the result is the policy in force. A change is saved to
`session-policy.json` beside the port files and restored at startup; a flag
given explicitly on the command line overrides the saved value. A lower limit
refuses new sessions without evicting live ones.

The desktop's **Settings → Sessions** tab drives the proxy through the
broker's engine relay with the process-wide requests `sessions/configure`,
`sessions/list` (this node's own sessions, optional `agentId` filter),
`sessions/end` (`sessionId`, plus the owner's `agentId`), `capacity/report` and
`capacity/set` (the `PUT /pair/v1/capacity` body). Any enabled engine's relay
reaches the same store. The desktop keeps no copy of these settings.

### 5.12 Context accounting and capacity

PAIR tracks how much of a model's context each session occupies, and how much
each model on this node can hold, so it can later decide how many sessions fit.
Admission on these numbers is §5.13.

**A session's context** is the token count its engine reported for the last
completed turn: prompt plus completion. The client resends the whole
conversation each turn, so that is what the next turn starts from. Counts are
read from the reply as it streams through, in every routed dialect:

| Dialect | Fields |
| --- | --- |
| Ollama native | `prompt_eval_count`, `eval_count` on the final object |
| OpenAI-compatible | `usage.prompt_tokens`, `usage.completion_tokens` |
| Anthropic Messages | `usage.input_tokens` (plus any `cache_read_input_tokens` and `cache_creation_input_tokens`), `usage.output_tokens`, possibly spread across `message_start` and `message_delta` |

All three engines count the full prompt, cached prefix included. An
OpenAI-compatible stream reports usage only when asked, so for a **session**
request that streams and sets neither way, the proxy adds
`stream_options.include_usage: true`. The engine then appends one standard
usage chunk with empty `choices`. Requests outside a session, and session
requests that chose either way, are forwarded as sent. A reply cut short, one
that reports no usage, or a non-streamed body over 4 MiB leaves the previous
count in place. The count is per engine, and exposed as `context` on the
session object.

**Capacity** is configured per engine and model on this node, through
`PUT /pair/v1/capacity`, and saved in `context-capacity.json` beside the port
files:

- `contextTokens` — the pool: total context the model can hold across all of
  its sessions at once.
- `maxSessions` — how many sessions may share it (1–1024).
- The **session share** is `contextTokens ÷ maxSessions`, rounded down to a
  multiple of 1024. For example 262144 ÷ 6 = 43008.

The setting is authoritative. Where it is absent, the local engine's report
fills in what it reliably can, cached for 15 seconds:

| Engine | Source | What is known |
| --- | --- | --- |
| Ollama | `/api/ps` `context_length` | The per-request window. Ollama allocates it once per `OLLAMA_NUM_PARALLEL` slot but does not report the slot count, so the pool stays unknown. |
| LM Studio | `/api/v1/models` `loaded_instances[].config` | `context_length` and `parallel`. The default unified KV cache shares one `context_length` pool. |
| llama.cpp | `/props` `default_generation_settings.n_ctx`, `total_slots` | Per-slot window and slot count; the pool as their product, an upper bound when the cache is unified. A router reports no model of its own, so each loaded model is asked with `?model=`. |

A value the engine does not report is reported as unknown, never guessed.
Detection covers this node's own engines only; capacity on other nodes is
theirs to report. `GET /pair/v1/capacity` lists every model with a setting, a
detection, or live sessions here, with the effective pool, sessions and share,
the live sessions this node serves for it and the context they hold, and a
warning when the share is below 4096 tokens, exceeds the engine's own window,
or cannot be computed. Each row also reports `heldSlots` and `queued` (§5.13).

### 5.13 Context admission

Admission applies to a **session** inference request whose model has a known
capacity on the node that serves it: a session count, a share, or both (§5.12).
Every node enforces its own capacity, for its own sessions and for sessions
other nodes route to it (§5.14). Requests outside a session are never gated.

- **Slots.** A model configured for N sessions has N context slots here. A
  session takes one when first admitted for that model and keeps it across
  turns until it ends, expires (§5.11), moves to another node at a commit
  (§5.10), or switches model on that engine.
- **Full.** When every slot is held, this node drops out of the request's
  candidates and the request goes to another node that has the model. Only
  when no other node can serve does it wait here, in arrival order, for a slot.
  A newcomer cannot take a slot while earlier arrivals wait. The wait is bounded
  by `queueTimeoutSeconds` (default 60, at most 300, 0 refuses at once) and
  `maxQueued` per model (default 64), both set with `sessions/configure`; past
  either the request gets `503` with `Retry-After`. A waiting request is in
  flight, so its session cannot expire; ending the session ends the wait.
- **Outgrown.** When the session's last measured context is at or above its
  share and this request is not shorter than the one that produced that
  measurement, this node drops out. With no other node left the request gets
  `400` naming the measured size and the share, and is not queued, because
  waiting would not shrink it. A shorter request is a compacted conversation:
  it is admitted and measured again. The turn that first crosses the share is
  admitted, since PAIR learns a turn's size only from the engine. Estimating
  tokens from request bytes is avoided because it would refuse image-heavy
  requests that fit.
- **Headers.** `X-PAIR-Context-Session-Tokens` carries the session's measured
  context before this turn, on every reply including refusals.
  `X-PAIR-Context-Session-Limit` carries the share when this node served the
  turn. `X-PAIR-Queue-Wait-Ms` reports a wait for a slot.

Admission runs before a reservation is taken or a workload is submitted, so a
request waiting for a slot does not yet appear as a job. A slot taken when this
node was a candidate is released at the commit if another node served instead.

### 5.14 Capacity across nodes

The originating node forwards a session request to a peer with its headers,
`X-PAIR-Session-ID` included. The peer's mTLS ingress (§7) applies §5.13 to
its own engine before forwarding: it tracks the session under the caller's
cluster ID, so the same session ID from two nodes is two sessions, and reads
the engine's token counts from the reply. A peer session is never listed by
the peer and never refused as ended there; only the originating node decides
that.

- **Full peer.** The peer waits for a slot for at most the lesser of its
  `queueTimeoutSeconds` and 90 seconds (inside the caller's 120-second wait for
  response headers), then answers `503` with `Retry-After`. The originating
  node treats that like any `503` (§5.4): it tries the next candidate, and
  between rounds backs off and re-resolves until its job deadline.
- **Outgrown on the peer.** The peer answers `400`, which is not retried.
- **Release.** When a session's turn commits on a different node, or the
  session ends or expires, the originating node sends
  `POST /pair/v1/peer/sessions/{id}/release` over mTLS to the node it left, so
  that node frees the slot at once. Only the originating node may release its
  session. Release is best effort; an unreleased slot frees when the peer's copy
  of the session goes idle.

## 6. Reservations

A reservation is one in-flight dispatch this process has made since the last
scheduler snapshot. Estimated load for a node is
`pending + gpuPressure + reservations`, the first two from the scheduler.

- **Taken** when a candidate is chosen, unless an eligible manual pin applies,
  and re-taken through the same path on each retry round (§5.1).
- **Moved** to the node an attempt actually targets, because the node that
  refused the request is not doing the work. An attempt can hold a node for the
  whole first-content budget, so the claim follows each dispatch rather than
  waiting for the commit.
- **Released** when the request ends, so a node stops counting as loaded as soon
  as it stops working. Without release a trickle of short requests makes an idle
  node read as busy for the rest of the snapshot interval.
- **Released before every retry backoff**, for the same reason applied to a
  waiting job: between attempts nothing is running, and because the map is
  process-wide a stale claim would push the *other* engine's traffic away from a
  node that is in fact free. This mirrors clearing `scheduledOn`, one being the
  proxy's own estimate and the other the scheduler's.

Each is stamped with the snapshot **generation** it was counted against. A
snapshot supersedes every reservation taken before it — its pending counts
already include that work — so a release naming an older generation is dropped.
Applying it would double-count the completion and drive the node's estimated
load below zero, which reads as permanently idle and attracts every subsequent
dispatch.

`node/set-priority` therefore carries a generation and is applied at most once;
a redelivered snapshot must not clear live reservations.

## 7. Listener personalities

One port per facade, demultiplexed on the connection's first byte:

- **Loopback plaintext** for local clients. A LAN caller is refused.
- **Cluster mTLS** when `--cluster-dir` shows this node is a member: a peer
  whose client certificate matches a local pin is forwarded straight to the
  local engine reported by `node/set-local-backend`, never re-routed onward.

Membership and pins are re-derived per request and on a watch, so joining or
leaving a cluster needs no restart.

An engine with an inherited host variable (today Ollama alone) may also be given
loopback-only **alias** addresses, so clients already using that variable enter
the same routing path. An occupied alias is non-fatal: the existing owner is
untouched, the primary listener stays up, and a warning is reported.

## 8. Ports

| | Ollama | LM Studio | llama.cpp |
| --- | --- | --- | --- |
| Engine's own client-facing port | 11434 | 1234 | 8080 |
| Where PAIR relocates the engine | 11435 | 1235 | 8081 |
| Standalone port, when `port` is omitted | 11435 | 1234 | 8080 |
| Persisted-port file | `proxy-port.json` | `lmstudio-proxy-port.json` | `llamacpp-proxy-port.json` |

A port chosen at runtime via `set-port` is persisted per engine and restored
when that facade is enabled, taking precedence over the requested port, so the
facade returns where the user left it. `ignorePersistedPort` bypasses that for a
broker-coordinated start.

`port` 0 means "the engine's standalone default". Any other out-of-range value
is rejected rather than honoured as ephemeral: the facade announces the
requested port in `ready`, so binding an ephemeral one would leave the broker
unable to locate it.

## 9. Logging constraints

Never log prompts, chat messages, request bodies, stream chunks, full responses,
credentials, or pairing data. Operational metadata only — engine, model, job id,
request id, node id, path, stream flag, normalized error text — and keep per-request routing
detail below info level.

The one place the proxy reads response bytes is session token accounting
(§5.12). It holds at most one bounded frame at a time, extracts only the
engine's numeric token counts, and discards the frame; no text is stored,
logged, emitted, or used to make a decision.

The log component is the **process** (`nvpair-proxy`), because at process init
there is no engine to name. Facade-scoped records carry an `engine` field, which
is what keeps a line attributable when several facades share the sink.

## 10. Adding an engine

Everything engine-specific is one entry in `engines.go` plus the shared identity
in `nvpair-shared/engines`. A new engine needs: its identity and discovery
service key, its ports and persisted-port filename, its route table and model
normalization, and its empty model-list envelope. It needs no new process, no
new supervisor, and no new relay wiring.

## 11. Failure modes

| Mode | Behavior |
| --- | --- |
| Requested port taken at enable | Tagged bind failure; broker retries on a fallback port |
| No advertised owner for the model | Local `502`, no engine contacted |
| All owners refuse retryably | Last upstream status surfaced |
| Transport error with candidates left | Forget the node's confirmed address, fail over |
| Client disconnects mid-stream | Terminal workload event emitted at once; upstream cancelled |
| Broker link closed | Facades stop serving, then the shared transport pool closes |
