<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

Release history for Personal AI Router, newest first. Entries match the
published releases on GitHub.

PAIR Plus releases are published on GitHub by the B-E-L-I-E-V-E/Personal-AI-Router-Plus
release workflow, with macOS builds signed and notarized; the app updates from
those releases. A build made locally from source is unsigned unless built with
`PAIR_MAC_SIGN=1`, and updates only once a release exists.

## 1.3.6 — Plan nodes with several GPUs per GPU (#18)

- **Context & sessions…** on a node with more than one GPU now sizes each GPU separately. The model is split by whole layers in proportion to each GPU's memory after its own reserve, so mismatched GPUs split by ratio. Each layer's part of the context stays on its GPU, so a session fits only when its part fits on every GPU.
- The dialog shows a bar per GPU (for example **GPU 0 · RTX 4090**) next to the total, turns a GPU red when it runs out even if the total still has room, and says how the layers are split. For Ollama it shows the model kept on one GPU until you plan more sessions than that GPU holds.
- Applying a multi-GPU plan also writes `--split-mode layer --tensor-split` (layers per GPU) for llama.cpp, and `OLLAMA_SCHED_SPREAD=1` for Ollama when the sessions need more than one GPU. The dialog lists what will be set.
- `capacity/plan` adds `gpus[]`, `tensorSplit`, `oneGpuIndex` and `oneGpuSessions`. `spec` now includes `layers` in the desktop's parsing.

## 1.3.5 — Context & sessions starts from the model's and engine's own defaults (#16)

- A model with no saved plan opens **Context & sessions…** on what is in force for it: the window, session count and reply length its engine runs it with now (the loaded model or the engine's launch settings), or the model's own defaults, instead of generic values.
- A small marker beside each slider says where its value came from (*engine now*, *engine settings*, *model default*, *model maximum*) while the slider still sits there. Nothing reported keeps the old starting point.
- Works for another node's model too: that node reads its own engine.
- `capacity/plan` returns the new `spec.defaults` (`window`, `sessions`, `reply`, each `{value, source}`).

## 1.3.4 — Sliders for the planner's model size and context cost, with exact size labels (#19)

- **Context & sessions…** has two new sliders, **Model weights in memory** and **Context cache per token**. They snap to round values, have an exact field beside them, and are always shown. Both start at what the model reports, and moving one overrides it for the plan. **Use the model's** resets it. They replace the two text fields that only appeared when a value was unknown.
- The dialog says what the size means. The plan counts the memory the weights take once loaded, not the download. It shows the download size and where it was read (Ollama's or LM Studio's model list, or the GGUF files). It shows the in-memory size (the weight tensors added up) and, for a model Ollama has loaded, what that copy takes now (weights, context cache and buffers together). It also says which size the plan uses. For MLX and safetensors models the download size is used as an estimate, and the dialog says so.
- The per-token figure says where it came from (the architecture at f16, a measured loaded copy, or your value) and what the plan counts at the chosen cache precision.
- `capacity/plan` adds `spec.downloadBytes`, `downloadSource`, `tensorBytes`, `weightsSource`, `loadedBytes`, `loadedVramBytes` and `loadedContextTokens`, plus the result's `weightsSource` and `kvSource`. An override no longer changes `spec.source`.

## 1.3.3 — Sessions at once no longer stops at 8 for models PAIR cannot size (#20)

- In **Context & sessions…**, a model whose size or context cost PAIR does not know no longer gets a **Sessions at once** slider that stops at 8. The slider appears once the model can be sized (enter its context cost, or load it so PAIR can measure it), and then reaches past what fits, in red, as before.
- PAIR now reads LM Studio models from the folder LM Studio's settings name (`downloadsFolder` in `~/.lmstudio/settings.json`), not only `~/.lmstudio/models`, so models kept on another drive are sized without entering their cost by hand.

## 1.3.2 — Run the benchmark on a chosen node (#15)

- **Run benchmark** runs on the node you choose, this one or any other in the cluster. The model list shows each model on each node that has it, and **Benchmark** on a node card's model row runs on that node. Every step runs on that node, or fails if the node can't serve the model.
- **Context & sessions…** for another node's model says why it can't plan that node: the PAIR version and firewall ports it needs. It also offers **Try again**, instead of leaving the menu item out.
- `POST /pair/v1/performance/benchmark` and `performance/benchmark` take an optional `nodeId`, and a run reports it.

## 1.3.1 — Plan another node's Context & sessions from any cluster member (#14)

- **Context & sessions…** is offered for models on other nodes, not only this one. From a laptop you can size, save, apply and remove a workstation's plan.
- The plan is worked out on the node that runs the model, for that node's GPUs and system memory, and is saved and enforced there. Applying restarts that node's engine. Its model rows show its plan line too.
- A node on PAIR Plus 1.3.0 or older shows no Context & sessions for its models from other nodes until it is updated.
- New API: `POST /pair/v1/peer/capacity/call` on the cluster's mTLS link, and the JSON-RPC method `capacity/peer-call`.

## 1.3.0 — Performance, model details, node pause and stop, and an API guide

- A new **Performance** tab, with a stopwatch icon next to Settings, shows each model's speed per node: tokens in and out, reading and writing speed in tokens per second, time to first token and, for thinking models, the time spent reasoning. A live view shows every request in progress. **Run benchmark** sends four fixed prompts through PAIR, one at a time and only when the node is idle. Each figure says whether the engine reported it or PAIR measured it, and only counts and timings are kept, never prompt or reply text. New APIs: `GET`/`DELETE /pair/v1/performance`, `GET /pair/v1/performance/live` and `POST /pair/v1/performance/benchmark` (#9).
- Each model row on the Overview shows what its node enforces: sessions at once (and how many are in use), the context window per session, and the response and post context, for this node's models and other nodes' (#6, #8).
- Each downloaded model's menu has **About this model…**: the description, tags, capabilities, family, parameters, quantization, context length, size and licence its engine and model file report, with buttons for its Ollama library or Hugging Face pages, for models on any node in the cluster (#7). **Look up on the web** fills facts the engine left out from the model's own page, read by a small local model; nothing is guessed, and the engine's values always win (#10).
- Each node card has **Pause**, **Resume** and **Stop**. A paused node finishes its running requests and takes no new ones; a stopped node also cancels them. New requests go to other nodes that can serve them, or get `503` when none can. Any cluster member can manage any other, and the state survives a restart; `GET`/`PUT /pair/v1/node/state` (#11).
- A new **HTTP API Guide** shows every call with the JSON a node sends back, recorded from a real node, and every error. The OpenAPI error object now matches what PAIR sends (`error`, with `code` where it helps), and the `X-PAIR-Served-By` header, which PAIR never sent, is gone from it.
- The repository is now [B-E-L-I-E-V-E/Personal-AI-Router-Plus](https://github.com/B-E-L-I-E-V-E/Personal-AI-Router-Plus); the old address redirects, and updates come from the new one.

## 1.2.0 — Context & sessions per model (#3)

- Each model's menu has Context & sessions…: choose the context window, the room kept for the reply, and how many sessions run at once, sized from this node's GPU and system memory. Every session gets the whole window.
- Memory bars show what the settings need and turn red past the hardware; such a plan can still be saved.
- A live table shows how much text fits in one post, the response and the whole window, in words, pages and characters, for English, code, JSON, Chinese, base64, IDs and emoji.
- PAIR keeps room for the reply in every session: it refuses a conversation that outgrows the rest, stops Ollama from silently cutting long conversations, and caps replies that set no length.
- Mixture-of-experts models can keep their expert weights in system memory on llama.cpp.
- The capacity and session APIs report each model's reply budget and the room a session's next post has.

## 1.1.0 — PAIR Plus: sessions, agents and context capacity

- Every request carries an `X-Request-ID`: yours if you send one, otherwise PAIR makes one, and it appears on the response, job and logs.
- Conversations can be sessions (`X-PAIR-Session-ID`) owned by an agent (`X-PAIR-Agent-ID`). Each turn goes back to the node that served the last one, so the engine reuses its cached prompt. Another agent using the session is refused.
- Sessions end after an idle timeout (10 minutes by default, 1 to 60), kept open by any request, a keepalive, or a request still running. Ending a session cancels its running requests, and reusing its ID is refused.
- PAIR tracks how much context each session uses from the engine's own token counts, and reports each model's context capacity.
- Set a model's context pool and number of sessions; each session gets an equal share. When every slot is taken, new sessions go to another node or wait in line, and a session that outgrows its share is asked to shorten its conversation. Every node enforces its own capacity.
- New **Settings → Sessions** tab for the session policy, per-model capacity, and live sessions.
- The window title reads NVIDIA PAIR Plus, and updates come from PAIR Plus releases. macOS builds are signed with a Developer ID and notarized.

## 1.0.0 — Personal AI Router 1.0.0

- Cuts release 1.0.0, which collects every change since 0.1.1 listed below. See the [1.0.0 release notes](docs/release-notes.mdx).

## 0.1.15 — Safer engine removal and data resets (#153)

- Removing an engine never follows a symlink or Windows junction into another folder.
- An LM Studio models folder linked from `~/.lmstudio/models` is kept when LM Studio is removed.
- Resetting app data, from Settings or the terminal interface, stops before deleting anything if an engine PAIR installed cannot be removed, and says which one.

## 0.1.14 — Engine install no longer shows a fake percentage (#145)

- Installing an engine no longer sits at a made-up 75%. Download progress still shows a real percentage; the install step itself now shows "Installing" with no number.

## 0.1.13 — Uninstalling keeps your downloaded models (#150)

- Uninstalling an engine keeps the models it downloaded. Uninstalling LM Studio no longer deletes `~/.lmstudio/models`.
- llama.cpp keeps its downloaded models in `~/.llamacpp`, outside PAIR's data folder, so removing PAIR's data no longer deletes them.
- Removing all data — when uninstalling PAIR, from Settings > Reset app data, or from the terminal interface's reset — now also uninstalls the engines PAIR installed, including LM Studio on Windows, and leaves engines you installed yourself alone. Downloaded models are kept.

## 0.1.12 — A rebuilt terminal interface, and one model catalogue for both front ends (#117)

- The terminal interface is organised around machines: Nodes, Jobs, Service, Errors, and Logs, with a per-machine drill-down for engines, models, ports, and hardware.
- Engines can be started with your own arguments and environment variables, edited from the terminal interface on this machine or on a paired one. Changes are validated before they are saved.
- Downloadable models are served by the engine manager, so the terminal interface and the desktop application browse the same catalogue.
- The Inference Demo runs from the terminal interface's Jobs tab, so a headless machine can demonstrate routing.
- The terminal interface says when a newer release of PAIR is published, on every tab until dismissed. It installs nothing.
- Pairing keys follow the words on screen: `p` pairs, `a` accepts, `f` finds a machine by address.

## 0.1.11 — llama.cpp is a PAIR-managed engine (#148)

- llama.cpp can be installed, started, stopped, and updated from PAIR, alongside Ollama and LM Studio.
- Browse and pull llama.cpp models from the model hub, sourced live from approved Hugging Face publishers, with a bounded search across public publishers.
- Delete a downloaded llama.cpp model from the engine's model list.
- Inference routed to llama.cpp goes through the local proxy, so the engine participates in cluster routing like the others.
- On Windows, PAIR installs the Visual C++ runtime llama.cpp requires.

## 0.1.10 — The desktop app is called NVIDIA PAIR (#147)

- The application now appears as **NVIDIA PAIR** in Finder, the Dock, and the macOS menu bar, instead of the abbreviated `PAIR`.

## 0.1.9 — Clearer in-flight jobs (#141)

- Jobs that have not started yet are labeled In flight, and they and their connection lines use the same yellow as the GPU chart.
- A job's card names the node it was sent to with "Sent to" until the job starts, including a job that fails or is cancelled before it starts, instead of "Ran on".

## 0.1.8 — Retry failed scheduler priority notifications (#122)

- The scheduler retries changed priorities after a failed local notification write and keeps its status aligned with the last successful delivery.

## 0.1.7 — Workload broadcast reliability: ordered broadcasts, dedup after emit (#80)

- Inter-node workload broadcasts are now serialized in origin order, so a lifecycle remove can no longer overtake its own upsert on a peer and resurrect a ghost workload.
- Dedup keys are recorded only after the broker emit succeeds; a failed emit's retry is no longer swallowed as a duplicate.

## 0.1.6 — The app is now called NVIDIA PAIR (#132)

- The desktop app, tray, shortcuts, and installer now show the name NVIDIA PAIR.
- Installed programs lists show NVIDIA Corporation as the publisher.
- Existing installs keep their settings, engines, cluster membership, and firewall rules when they update.

## 0.1.5 — Avoid unnecessary cluster trust notifications (#62)

- Cluster trust notifications now follow successfully saved peer endorsements. Duplicate endorsements and failed writes no longer cause unnecessary refreshes.

## 0.1.4 — Fix HTTP connection reuse in engine health probes (#37)

- Periodic engine monitoring now reuses HTTP/1 connections instead of opening a new connection for every health probe.

## 0.1.3 — Route Anthropic Messages API requests (#27)

- The Ollama and LM Studio endpoints offered by PAIR now accept requests for the Anthropic Messages API via POST to the /v1/messages endpoint and direct them to the owner of the requested model, just as happens with the other inference methods.

## 0.1.2 — fix(mdns): send responses from UDP 5353 (GitHub Issue #1) (#102)

- Discovery traffic now originates from UDP 5353, allowing standards-compliant mDNS peers and network reflectors to accept PAIR node records.

## 0.1.1

### Fixed

- Building from source for an Intel/AMD (`x64`) target failed while compiling
  the bundled log sanitizer. The packaging script passed Electron's `x64`
  architecture name straight to the Go toolchain, which expects `amd64`.
  Arm64 targets were unaffected.

This release contains no application changes. The fix is to the build tooling
only, and the published 0.1.0 installers were not affected by it, so 0.1.1 is
functionally identical to 0.1.0 for anyone installing it.

## 0.1.0

### Fixed

- **Windows on Arm installations were missing every executable.** The
  installer's compressed payload used a compression filter its extractor could
  not decode, so an install reported success with the application tree present
  but all `.exe` and `.dll` files silently skipped. Arm64 installs now
  complete correctly.
- **Windows on Arm now installs under 64-bit Program Files** instead of the
  32-bit location.
- **Silent installs no longer abort** on the installer's payload check.

### Changed

- Raised the Electron floor and updated the Go toolchain and service
  dependencies to pick up security fixes.
- Trimmed the README, and added an `AGENTS.md` so an agent working from a fork
  has an entry point.
