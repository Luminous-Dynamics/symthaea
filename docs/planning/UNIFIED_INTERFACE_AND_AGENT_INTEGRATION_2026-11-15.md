# Unified Symthaea Interface and Agent Integration

**Date:** 2026-11-15  
**Status:** Research/design proposal  
**Branch:** `research/unified-interface-and-agent-boundaries-2026-11`

## Executive decision

Build one coherent Symthaea experience from the interface and service surfaces that already exist. Do not replace the Leptos UI, start another dashboard, or turn Symthaea into a wrapper around a third-party chat product.

The design principle is:

> **One cognitive core, one versioned service contract, many replaceable clients and capability adapters.**

This is a unification of protocols, navigation, evidence, identity and visual language—not a requirement to combine every system into one binary or one frontend.

The current repository already contains the core ingredients:

- `symthaea-service`: Unix-socket/TCP service and optional HTTP gateway.
- `POST /v1/service`: the existing versioned JSON request/response protocol.
- `GET /v1/ws/live`: cycle-synchronous telemetry, when the experience bridge is enabled.
- `crates/bridges/symthaea-ui`: Leptos 0.8 CSR browser client with Converse and Observe.
- Terminal surfaces: `symthaea-repl` and `symthaea-shell`.
- REST benchmark/observability routes in `src/api`.
- Existing visualization exits: `canvas_svg` and `mental_movie`.

The Leptos UI should remain the canonical first-party visual interface. The terminal REPL remains a first-class alternative. Open WebUI, Aider and Codex should be external clients/agents connected through explicit adapters.

## 1. Clarify the products before integrating them

These are different categories and should not be treated as interchangeable bots.

| Product | What it is | Recommended relationship to Symthaea |
|---|---|---|
| Open WebUI (assuming this is what “OpenChat” meant) | Multi-model chat client with external tool integrations | Optional compatibility client and integration test |
| Aider | AI pair-programming agent | Optional coding agent; optionally consume Symthaea through an OpenAI-compatible model endpoint |
| Codex CLI / app-server | Coding agent with its own thread, turn, approval and event protocol | Separate agent adapter; use the local app-server protocol for a deep integration |
| Groq | Hosted inference provider with an OpenAI-compatible API surface | Optional model-provider adapter, not a bot or UI |
| Ollama | Local inference service | Local provider; remains independently startable |
| MCP | Tool/capability protocol | Narrow capability boundary for approved Symthaea operations |

If “OpenChat” meant a different named project, the same architecture still applies: first identify whether it is a UI, agent, provider or protocol. Avoid building an integration around a name alone.

### Recommended order

1. Finish and harden the existing Symthaea UI and service contract.
2. Add a narrow, versioned OpenAI-compatible chat surface so clients such as Aider and Open WebUI can use Symthaea as a model/backend where the semantics fit.
3. Expose a small read-only MCP tool set for Open WebUI and other clients.
4. Integrate Codex through its local app-server protocol only when we need to display Codex threads, approvals and streamed agent events inside the Symthaea workbench.
5. Add Groq as an optional inference provider behind the existing provider-selection boundary.

Do not make external agents responsible for Symthaea's memory, ethics policy, trust decisions or service lifecycle.

## 2. Unified product vision

The interface should feel like a coherent instrument for exploring, understanding and collaborating with a cognitive system—not a wall of unrelated gauges.

The core experience has three layers:

- **Presence:** Is the service alive, what subsystem is active, and what is actually being observed?
- **Conversation and work:** What is being discussed, reasoned about, executed, approved or changed?
- **Evidence and stewardship:** What happened, what supports the claim, what changed, and who authorized it?

### Proposed navigation

| Surface | Purpose | Initial scope |
|---|---|---|
| Home / Presence | Service health and current activity | Daemon status, active bridges, resource/load state, recent events |
| Converse | Human ↔ Symthaea conversation | Existing chat, transcript, context/source references, response confidence and safety outcome |
| Observe | Inspect live cognitive state | Existing vitality orb, telemetry, self-portrait and imagination projection; clear stale/no-data states |
| Steward | Introspection and lifecycle | Status, introspect, save, sleep, audit events; explicit confirmations for state-changing actions |
| Workbench | Human-supervised coding | Aider/Codex sessions, diffs, tool calls, approvals, test/build receipts |
| Lab | Experiments and simulation | Symtropy experiment runs, seeds, replay hashes, measurements, comparisons |
| Commons | Network and trust | Mycelix peers, identity, signed provenance and governance receipts; disabled unless explicitly connected |
| Providers | Model and tool routing | Ollama, Groq and other providers; model capability, locality, egress, latency and cost |
| Evidence | Cross-system receipts | Event timeline, source revision, configuration digest, benchmark and test provenance |

Build the navigation shell and shared state model before building every pane. The first useful increment is to complete the existing Steward operations and improve Observe/Converse reliability.

### Visual and interaction principles

- Organic, calm, tactile visual language; avoid a generic operations dashboard.
- Use the orb as a summary visualization, not as a substitute for precise telemetry.
- Keep raw measurements, computed indicators, predictions and interpretations visibly distinct.
- Every chart or number must expose units, last-updated time, data availability and provenance.
- Show uncertainty and missing data honestly. Do not animate a value merely because telemetry is absent.
- Use progressive disclosure: simple conversation first, detailed instrumentation when requested.
- Keyboard access, accessible contrast, screen-reader labels and responsive desktop/mobile layouts are core requirements.
- Keep the terminal REPL and shell as supported clients of the same service protocol, not deprecated legacy interfaces.

### Important scientific boundary

Values such as Phi, consciousness level, moral score, coherence, valence and arousal are model outputs or experimental proxies. The UI must label their definitions and limitations. They are not, by themselves, proof of phenomenal consciousness, moral status or clinical condition. Never infer a human user's psychological or medical state from these readings.

## 3. The service contract and adapter architecture

The service already has a useful JSON protocol, HTTP gateway, bearer-token support, request versioning, audit logging and cycle telemetry. Preserve this as the foundation.

### A. Typed wire contract

The browser client currently uses `serde_json::Value` to avoid depending on the full Symthaea engine crate. Keep that dependency separation, but consider introducing a small `symthaea-wire` or `symthaea-view-types` crate containing only:

- Versioned request/response DTOs.
- Stable status, health, audit and telemetry view types.
- Event envelopes and error types.
- Schema/compatibility tests.

The crate must not depend on the cognitive engine, model runtimes, Holochain, Bevy, audio or GPU subsystems. It should be cheap to build natively and for WASM.

Use additive changes where possible. Breaking changes require an explicit protocol version or a compatibility window.

### B. OpenAI-compatible chat adapter

For Aider/Open WebUI interoperability, implement an adapter rather than teaching the cognitive core multiple UI protocols.

Initial target endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`

Then add streaming, tool calls and other features only when they can be represented faithfully.

Required compatibility tests should cover:

- System/user/assistant message ordering and multiple turns.
- Streaming and non-streaming response equivalence.
- Finish reasons and token-usage semantics (do not fabricate unavailable usage).
- Temperature, max-token, stop and seed handling where supported.
- Tool definitions, tool choice and structured arguments only where the backend can enforce them.
- Errors and status codes for unsupported parameters.
- Model discovery and explicit model identity.
- Cancellation and client disconnect behavior.

Aider can use arbitrary OpenAI-compatible model endpoints via `OPENAI_API_BASE`, `OPENAI_API_KEY` and a model name prefixed with `openai/`. This makes it a good compatibility client once the endpoint is complete. It does not mean Aider should be embedded inside the Symthaea service process.

Groq's API is also OpenAI-compatible at `https://api.groq.com/openai/v1`, with feature differences. That is a provider configuration, not a new Symthaea core dependency.

### C. MCP capability server

MCP should expose deliberately narrow capabilities, not the whole operating system.

Start with read-only tools:

- `symthaea.status`
- `symthaea.introspect`
- `symthaea.audit_events`
- `symthaea.protocol_info`
- `symthaea.benchmark_results`
- `symtropy.experiment_catalog`
- `symtropy.replay_inspect`
- `mycelix.peer_status` (only when the relevant connector is enabled)

State-changing operations such as sleep, save, starting a service, changing a configuration or running an experiment must be separate capabilities with bounded schemas, explicit policy and approval receipts.

Never expose a generic unrestricted shell command, host filesystem browser or arbitrary process execution tool through MCP.

Open WebUI currently supports MCP over Streamable HTTP, and deliberately admin-gates MCP connections because they are privileged. Use it as an optional client and compatibility test, not as a reason to make the Symthaea capability server public.

### D. Codex integration

Codex app-server is not an MCP server. It has its own JSON-RPC lifecycle and agent event model. Use its documented local stdio or Unix-socket transport for a native workbench integration. Do not base production architecture on its experimental network WebSocket transport.

The Codex adapter should initially be an observer and approval surface:

- Start/resume a thread only on explicit user request.
- Display agent messages, command execution, file diffs and test results.
- Forward approval requests to the human.
- Preserve Codex's sandbox and approval policy.
- Do not let Symthaea silently approve an external agent's actions.

For automated batch jobs, use the Codex SDK rather than trying to turn the app-server into a generic job runner.

### E. Mycelix and Symtropy boundaries

Mycelix should provide optional identity, trust, provenance and governance connections. A trust score is evidence to consider, not unilateral authorization.

Symtropy should provide experiment execution and physical/digital-twin evidence. Symthaea may propose hypotheses or plans; Symtropy executes only within an explicit scenario, simulation or actuator boundary. Record seed, model/physics revision, scenario, controls, result digest and uncertainty.

Sol Atlas should be a spatial visualization/client, not a dependency of the Symthaea cognitive service.

## 4. Authentication, browser security and reliability gaps

The existing UI is not yet ready for external browser clients with authentication enabled.

### Current issues identified

1. `crates/bridges/symthaea-ui/src/api.rs` sends HTTP requests without an Authorization header or browser session mechanism.
2. The service's `/v1/ws/live` handler enforces bearer authentication on the WebSocket handshake. The browser WebSocket API does not allow arbitrary Authorization headers.
3. The UI currently opens the WebSocket without an authentication mechanism.
4. The UI marks the telemetry connection as connected before the WebSocket handshake succeeds.
5. Changing the gateway URL does not currently rebuild the active telemetry subscription.
6. The UI runs on a different port from the example HTTP gateway. The HTTP router's CORS behavior and allowed origins need explicit verification before browser-to-gateway calls can work reliably.

### Recommended design

Prefer a same-origin local gateway/reverse-proxy arrangement:

- The browser talks only to the UI's origin.
- A local proxy or session broker forwards API requests to the service and attaches the server-side bearer credential.
- The browser never receives the long-lived service bearer token.
- WebSocket upgrades use the same authenticated session boundary.
- Use secure, HttpOnly, SameSite cookies when the local deployment supports them.
- If a short-lived WebSocket ticket is used, mint it only after authenticated authorization; make it short-lived, single-use, scoped to one connection and unusable as a general API token.

Do not put long-lived bearer tokens in query strings, URLs, local storage or telemetry payloads. Do not open wildcard CORS to solve a local development problem.

### Reliability requirements

- Set connected=true only after the handshake completes.
- Show separate states for service reachable, authentication valid, telemetry connected and telemetry fresh.
- Reconnect with capped exponential backoff and jitter.
- Reconnect when gateway settings change.
- Handle sequence gaps, lagged broadcasts and stale snapshots visibly.
- Bound payload sizes and decode budgets.
- Reject malformed, non-finite or dimensionally invalid telemetry.
- Keep the interface useful when the experience bridge is disabled; display “no live cycle telemetry” rather than zeros that look like measurements.

## 5. Local test programme

Run these in stages, after the current NixOS dry-activation ends and the workstation has recovered memory. Do not start all workspace features or multiple heavy builds concurrently.

### Symthaea: service and interface

| Priority | Test | Acceptance evidence |
|---|---|---|
| P0 | Existing service unit/protocol tests | Exact package and test output, no ignored failures |
| P0 | HTTP auth negative/positive tests | Missing and wrong tokens rejected; correct token accepted |
| P0 | Protocol version and request-size boundaries | Old/new version behavior explicit; oversize request rejected |
| P0 | UI DTO/parser tests | Missing, null, non-finite, malformed and future additive fields handled |
| P0 | Gateway/WS reconnect tests | Handshake state accurate; reconnect and sequence gaps surfaced |
| P0 | Telemetry disabled path | UI remains useful with experience bridge off |
| P1 | HTTP + WebSocket end-to-end | Authenticated query, status, telemetry and audit round-trip |
| P1 | Projection safety | SVG and mental-movie size/dimension limits; malformed base64 rejected |
| P1 | Browser origin/security tests | Same-origin deployment works; unauthorized origins denied |
| P1 | Steward flows | Introspect, save and sleep have correct permissions, audit events and visible outcomes |
| P2 | Accessibility | Keyboard, screen-reader labels, contrast, mobile breakpoints |
| P2 | Latency/load | p50/p95/p99 query latency, time-to-first-token, streaming gap, memory and CPU |
| P2 | Long-session reliability | Multi-hour session, daemon restart, browser refresh, event backlog/lag and recovery |

Do not run a heavy browser/wasm build during the active workstation build. Once idle, a first isolated compile/test target is the `symthaea-ui` package for `wasm32-unknown-unknown`, not every workspace member.

### Symtropy: physics and embodied integration

Start with the standalone core packages:

```bash
cd /srv/luminous-dynamics/symtropy
cargo test -p symtropy-math -p symtropy-physics -p symtropy-consciousness-physics --lib
```

Then, separately:

- Run deterministic replay tests across independent process runs and compare state/replay digests.
- Run property tests for collision, constraint and integration invariants.
- Assert finite state, non-negative energy quantities where required, and declared conservation/error bounds.
- Re-run the controller/environment falsification studies with recorded seeds and the published statistical correction procedure.
- Benchmark 10/100/500/1,000 agents with fixed hardware and compiler; measure p50/p95 step time, allocations, memory and replay drift.
- Test the Symthaea embodiment adapter separately from the Bevy renderer: observation → thought representation → bridge → motor intent, including malformed observations, timeouts and safe fallback.
- Only after core tests pass should the full Bevy/GPU/audio stack be built.

Never use a single headline “cooperation” or “Phi” result without its scenario, seed, comparator, confidence interval and falsification boundary.

### Mycelix: trust and distributed-system integration

Use disposable test DNA/conductors and isolated sandboxes. Keep the existing production Holochain/Conductor services inactive.

- Run the Rust integrity/coordinator tests in the selected app's DNA workspace.
- Build integrity and coordinator zomes for `wasm32-unknown-unknown` independently.
- Pack/install DNA in a disposable sandbox and exercise real zome calls.
- Test two or more conductors for DHT publication, eventual visibility, offline/reconnect and duplicate delivery.
- Exercise invalid signatures, stale/replayed messages, duplicate actions, unauthorized callers, malformed payloads and incorrect epistemic tiers.
- Test the trust layer against honest, noisy, non-IID, Sybil-like, colluding and adaptive adversaries.
- Measure detection, false positives, false negatives, calibration, convergence, communication overhead and latency at multiple adversarial ratios.
- Test bridge contracts with stale trust evidence, disconnected peers and failed verification. Unknown outcome must not authorize a new mutation.

Repository-specific Mycelix products have different maturity. Do not treat an older prototype's README roadmap or a landing-page benchmark claim as verification of the current protocol.

## 6. Dependency isolation and independent qualification

The NixOS configuration change that removes Aider from `environment.systemPackages` while keeping it as an independently tested flake package is the correct pattern.

Apply the same principle selectively:

- Keep the bootable host profile small and boring.
- Expose optional AI agents, experimental compilers, GPU research tools and large simulation stacks as separate flake packages or development shells.
- Keep Codex installed for now because its current package smoke test and configuration checks pass; moving it is a separate choice, not necessary to finish Aider isolation.
- Consider moving heavyweight Ollama/model runtimes out of the base system path if the service is normally off. Keep the provider package easily launchable as an optional app or managed user service.
- Use per-project toolchain pins and lockfiles. Avoid promoting a project-specific compiler/toolchain into the host default without a compatibility reason.
- Preserve separate evidence for evaluation, package tests, host closure, activation, runtime service health, GPU driver/kernel alignment and physical boot.

Do not equate:
- flake evaluation with package build,
- package build with tests,
- host dry-run with a real toplevel build,
- dry-activation with successful boot,
- service unit availability with service health,
- an API response with correct cognitive behavior.

## 7. Measured resource scheduling

Keep the global hard ceiling conservative. The existing profile `max-jobs = 2; cores = 4;` permits up to eight requested build cores, but that is a limit, not proof that the host can safely sustain it under a heavy desktop workload.

Use an explicit build wrapper rather than a daemon configuration that continuously changes under multiple concurrent clients.

### Inputs

- `MemAvailable` from `/proc/meminfo`.
- Memory PSI `some` and `full` from `/proc/pressure/memory`.
- Swap/ZRAM pressure and swap-in/out deltas.
- Nix daemon cgroup current/peak memory, when accessible.
- Foreground desktop activity and user-selected “interactive / balanced / batch” mode.
- A per-build memory estimate from historical derivation receipts, where available.

### Policy sketch

| Condition | Policy |
|---|---|
| Memory pressure high or available RAM below the safety floor | Do not launch new local builds; allow cache substitution or a remote builder |
| Workstation in interactive mode | One derivation, one or two builder cores |
| Balanced mode with healthy RAM/PSI | One derivation, two to four cores |
| Batch mode with measured headroom | At most two derivations, within the configured hard core ceiling |

The thresholds must be calibrated against actual workstation measurements. Never assume 2 GiB available is a universally safe floor for every derivation; it is an initial responsiveness target to test.

### Control-loop safety

- Sample before build launch and between independent build commands.
- Use hysteresis and a cooldown so the policy does not oscillate.
- Never exceed the global hard cap.
- Do not kill a running derivation just because memory briefly dips; stop scheduling additional work first.
- If full-memory PSI remains elevated or OOM counters increase, stop starting local work and recommend a remote builder/cache.
- Record every decision with timestamp, inputs, selected `max-jobs`/`cores`, command, flake.lock digest and result.
- Provide a fixed conservative mode and a dry-run explanation mode.
- Keep adaptive scheduling out of the Nix daemon itself until the wrapper is measured and proven.

Nix exposes `max-jobs` and `cores` as separate controls and supports per-invocation overrides. The maximum requested build-core count is their product; builders may use fewer cores.

## 8. Flake update policy

No blanket `nix flake update` is recommended now.

The current NixOS lockfile has already been deliberately refreshed and the exact PR head passed its hosted evaluation, package and configuration jobs. A full update would change the dependency experiment and potentially refresh the previously held TurboQuant and NUR inputs again.

For future changes:

1. Identify the specific input that needs an update.
2. Update only that input (for example, `nix flake update nixpkgs`).
3. Review the lockfile diff and source revision.
4. Run flake evaluation and affected package checks.
5. Run the host dry-build.
6. Record the exact lockfile digest in the receipt.

Keep Nixpkgs, Home Manager, sops-nix, custom CUDA/TurboQuant sources and project toolchains separate in the review. Do not bump `system.stateVersion` as part of an ordinary input refresh.

## 9. Delivery plan

### Phase 0 — finish the current workstation qualification

- Let the existing local `nixos-rebuild dry-activate` finish.
- Capture final exit code and exact head/lock digest.
- Capture before/after memory, PSI and swap.
- Do not run boot or reboot in this phase.

### Phase 1 — harden the existing UI

- Correct auth/session and WebSocket handshake handling.
- Fix connected/fresh status semantics.
- Add reconnect and gateway-change behavior.
- Add parser and HTTP/WebSocket integration tests.
- Add Steward status/introspect/audit/sleep/save surfaces with approval where required.

### Phase 2 — stable contracts and event model

- Extract small versioned wire/view DTOs.
- Add typed event envelope, sequence numbers, timestamps, source/provenance and bounded payload schema.
- Make terminal and browser clients consume the same protocol.

### Phase 3 — optional ecosystem adapters

- OpenAI-compatible chat endpoint for Aider/Open WebUI.
- Narrow MCP server for read-only tools first.
- Codex app-server local Unix-socket adapter if workbench integration is still valuable.
- Provider adapters for Ollama and Groq with explicit egress and cost visibility.

### Phase 4 — Lab and Commons

- Symtropy scenario runner and replay/evidence view.
- Mycelix test-network and signed provenance view.
- Keep both optional and independently qualified.

### Phase 5 — longitudinal evaluation

Measure task completion, tool-call correctness, approval compliance, provenance coverage, user-perceived latency, memory pressure, recovery and regression rates. Compare the integrated system with the same individual tools used directly. A unified interface is only better if it improves useful outcomes without weakening safety or increasing unnecessary resource use.

## 10. Acceptance criteria

The unified interface programme is ready for a production pilot only when:

- All UI and backend routes have explicit auth/origin policy and negative tests.
- No long-lived bearer token is exposed to browser storage or URLs.
- The UI distinguishes absent, stale, invalid and fresh telemetry.
- Event sequence gaps and reconnection are tested.
- Read-only tools are separate from mutation capabilities.
- Mutating actions produce authorization and audit receipts.
- Aider, Codex, Open WebUI and provider integrations are optional and independently removable.
- Symtropy experiments have deterministic replay evidence.
- Mycelix operations are tested on disposable conductors before any real network enablement.
- The base NixOS generation can build without optional agent and research-package dependencies.
- Every performance claim is tied to an exact revision, scenario, command, repeated measurements and raw logs.

## References

- Existing UI: `crates/bridges/symthaea-ui/`
- Service daemon and gateway: `src/bin/symthaea.rs`
- REST/benchmark API: `src/api/`
- Shared control-plane and audit helpers: `src/control_plane.rs`
- Existing benchmarking strategy: `docs/planning/BENCHMARKING_STRATEGY.md`
- Existing memory/build benchmark protocol in NixOS configuration: `docs/WORKSTATION_BUILD_BENCHMARK.md`
- [Aider OpenAI-compatible API documentation](https://aider.chat/docs/llms/openai-compat.html)
- [Open WebUI OpenAI-compatible provider requirements](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/)
- [Open WebUI MCP integration and security model](https://docs.openwebui.com/features/extensibility/mcp/)
- [Groq OpenAI compatibility](https://console.groq.com/docs/openai)
- [Codex app-server documentation](https://learn.chatgpt.com/docs/app-server)
- [Nix cores/jobs guide](https://nix.dev/manual/nix/2.34/advanced-topics/cores-vs-jobs.html)
- [Nix flake update reference](https://nix.dev/manual/nix/2.23/command-ref/new-cli/nix3-flake-update)
