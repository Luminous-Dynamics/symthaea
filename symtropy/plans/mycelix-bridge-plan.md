# Plan: `symtropy-mycelix-bridge`

**Status:** ✅ **Milestone 1 shipped 2026-04-17** · Author: Claude + Tristan · Roadmap: Phase 1, Track B

## Status board

| Milestone | Status | Commit |
|---|---|---|
| M1 — Spike | ✅ Complete | `91408552cd` |
| M2 — Event API + 3 zome calls | Pending | — |
| M3 — Scenario harness | Pending | — |
| M4 — 50-NPC visual demo | Pending | — |

## Architectural pivot landed in M1

The original plan called for wrapping `symthaea-mycelix-holochain` as a Bevy Resource. That hit a hard dependency wall: `holochain_client 0.6.0` exact-pins `serde = 1.0.203`, Bevy 0.18 requires `serde_core >= 1.0.221` via `hashbrown 0.16`. Not resolvable in one compilation unit.

**Resolution:** run the Holochain client as a **separate process** and pipe JSON over stdin/stdout. The crate `mycelix-conductor-bridge` already implements this protocol — so the bridge spawns it as a child process rather than linking it.

**Why this is better than the original plan:**

1. Zero Holochain dependencies in the Bevy crate — no serde conflict possible.
2. Process isolation: a crash or hang in the Holochain client doesn't crash the game.
3. The `mycelix-conductor-bridge` subprocess is reusable by other hosts (e.g., test runners, CLI tools) with no changes.
4. If either Bevy or Holochain bumps serde again, nothing in this crate breaks.
5. License surface shrinks: this crate could be permissively licensed since it has no AGPL deps in its graph. (We're keeping it AGPL because it's Mycelix-specific glue; that's a policy choice, not a technical requirement.)

**Cost:** subprocess IPC has per-request latency (~1 ms for JSON parse + syscall) and a startup cost (~1 s for conductor auth). Acceptable for the game-loop budget (5–10 Hz per NPC) — negligible next to the 4 ms of an actual zome call.

**What the plan below should be read as:** M1's "Design" and "Threading model" sections were written for the in-process architecture. The shipped implementation is subprocess-based. M2–M4 sections are still accurate except where they assume typed Rust methods on `GovernanceDispatcher`; those translate to JSON variants in the subprocess protocol.

---


## Goal

A Bevy `Resource` that lets Bevy systems and NPCs call real Mycelix zomes over the shared Holochain conductor. Every Bevy agent becomes a first-class Mycelix user, turning Bevy scenes into visual, reproducible integration tests for the 16-cluster Mycelix architecture at scales `headless_test` can't reach.

**Concrete success criterion:** A Bevy scene spawns 50 NPC-Agents. Each independently reads Praxis credentials, proposes and votes on governance items, and queries its TEND balance — all through real Holochain RPC against a running conductor. The scenario runs green in CI and exposes at least one invariant (e.g., "a Byzantine quorum of Orange-tier agents cannot pass a constitutional proposal") that the existing `headless_test` doesn't already cover.

## Non-goals

1. **Not** a generic Holochain-for-Bevy crate. This is Mycelix-specific, AGPL-licensed. A permissive `bevy-holochain-client` is conceivable but out of scope.
2. **Not** browser/WASM support. `holochain_client` is tokio-based; browser path is Track A's domain (iframe overlay, Phase 4).
3. **Not** a UI crate. That's Track C. This is pure backend integration.
4. **Not** a WebSocket reimplementation. Reuse `holochain_client` through the existing `symthaea-mycelix-holochain` wrapper.

## Key research findings

1. **Serde version conflict is load-bearing.** The monorepo already has two Holochain integrations (`mycelix-conductor-bridge`, `symthaea-mycelix-holochain`), both living **outside** the symtropy workspace because `holochain_client` pins `serde` incompatibly with the broader workspace graph. `symtropy-mycelix-bridge` must do the same — top-level sibling with its own `Cargo.lock`.

2. **Upstream client: `holochain_client` tagged `holochain-0.6.0` via git.** The crates.io 0.6.0 has a versioning inconsistency (`holochain_conductor_api` 0.4.0 uses `serde tag="data"` differently); the existing monorepo pattern is the git tag. Latest upstream is 0.7.1 but switching isn't required and would be churn.

3. **`symthaea-mycelix-holochain` is the foundation we build on.** Public API already exposes `HolochainConductor::new(url, app_id).connect().await`, `GovernanceDispatcher::{submit_proposal, get_active_proposals, vote}`, `FinanceDispatcher::query_tend_balance`. We wrap these, not `AppWebsocket` directly.

4. **`bevy-tokio-tasks 0.18.0` solves the async/ECS bridge.** Owns the tokio runtime, exposes `TokioTasksRuntime` resource with `spawn_background_task(|ctx| async { ... })` and `ctx.run_on_main_thread(|world| ...)` for safe sync-back to ECS state. Single crate, matches Bevy 0.18, the de-facto 2026 pattern.

5. **Performance envelope (from Holochain Wind Tunnel, 2025):**
   - Read-only zome calls: ~2,400/sec peer capacity; ~4 ms latency.
   - Read+write cycles (with DHT validation): ~54/sec capacity.
   - **Design budget: 5–10 Hz reads per NPC, 1 Hz writes per 10 NPCs.** At 50 NPCs: ~500 reads/frame, ~5 writes/frame. Comfortable, not tight.

6. **Auth flow is two-hop:** admin websocket (33800) issues app-auth token → app websocket (8888) opens with token → zome calls. `symthaea-mycelix-holochain::HolochainConductor::connect()` already abstracts this.

## Design

### Crate layout

```
/srv/luminous-dynamics/symtropy-mycelix-bridge/
├── Cargo.toml                  # NOT in symtropy workspace (serde conflict)
├── Cargo.lock                  # own lockfile
├── README.md
└── src/
    ├── lib.rs                  # public API re-exports
    ├── resource.rs             # MycelixClient Bevy Resource
    ├── plugin.rs               # BevyMycelixPlugin<Cfg>
    ├── events.rs               # MycelixRequest, MycelixResponse (Bevy events)
    ├── systems.rs              # pump_requests, pump_responses systems
    ├── governance.rs           # Bevy-flavored helpers on GovernanceDispatcher
    ├── finance.rs              # Bevy-flavored helpers on FinanceDispatcher
    └── scenarios.rs            # test-scenario harness for CI
```

Crate name: `symtropy-mycelix-bridge`. License: **AGPL-3.0-or-later** (pulls symthaea-mycelix-holochain transitively).

### Public API sketch

```rust
// Resource — the handle Bevy systems hold
#[derive(Resource)]
pub struct MycelixClient {
    runtime: Arc<HolochainConductor>,
    tx: Sender<MycelixRequest>,
    rx: Receiver<MycelixResponse>,
}

// Config
pub struct MycelixConfig {
    pub admin_url: String,       // "ws://localhost:33800"
    pub app_url: String,         // "ws://localhost:8888"
    pub app_id: String,          // "mycelix-unified"
    pub inflight_budget: usize,  // default: 128
}

// Plugin
pub struct BevyMycelixPlugin {
    pub config: MycelixConfig,
}

impl Plugin for BevyMycelixPlugin {
    fn build(&self, app: &mut App) {
        // 1. Add bevy-tokio-tasks
        app.add_plugins(TokioTasksPlugin::default());
        // 2. Spawn background task that owns HolochainConductor
        //    + bounded request/response channels.
        // 3. Insert MycelixClient resource.
        // 4. Register pump_requests (Update) + pump_responses (Update) systems.
        // 5. Register MycelixRequest / MycelixResponse events.
    }
}

// Event-based API (recommended for NPCs)
#[derive(Event)]
pub enum MycelixRequest {
    GetActiveProposals { requester: Entity },
    SubmitProposal { requester: Entity, text: String, did: String },
    Vote { requester: Entity, proposal: String, cast: bool, did: String },
    QueryTendBalance { requester: Entity, did: String },
    // ... one variant per zome call we support
}

#[derive(Event)]
pub enum MycelixResponse {
    ActiveProposals { requester: Entity, proposals: Vec<Proposal> },
    ProposalSubmitted { requester: Entity, action_hash: String },
    VoteCast { requester: Entity, proposal: String },
    TendBalance { requester: Entity, balance: BalanceResponse },
    Error { requester: Entity, reason: String },
}

// Direct-call API (for scenarios + tests, not per-frame)
impl MycelixClient {
    pub fn submit_proposal_async(&self, /* ... */) -> Future<Result<String>>;
}
```

### Threading model

```
  Bevy main thread (ECS, 64Hz)               Tokio runtime thread
  ─────────────────────────────              ─────────────────────────────
  pump_requests system                       HolochainConductor
   ↓ MycelixRequest event                     ↑ call_zome async
   ↓ send via channel ────────────────────→  ↓ response
   ↑ recv from channel ←──────────────────  ↑ send via channel
   ↓ MycelixResponse event
  pump_responses system
   ↓ user systems handle it
```

Channel crate: **`flume`** (MPMC, async+sync, no tokio-dependency). Bounded to `inflight_budget` to prevent queue blowup.

Both channels are `Send + Sync`. `MycelixClient` is `Clone` via `Arc<Inner>`.

### Error model

`MycelixResponse::Error { requester, reason }` is the only way errors surface to Bevy systems — keeps systems infallible. The tokio task logs with `tracing` and attempts a single reconnect before propagating. Connection-drop is an `Error` variant; systems that care subscribe.

### Test scenarios

Dedicated `scenarios` module ships a set of declarative runners:

```rust
pub fn scenario_proposal_vote_invariant(app: &mut App, n_agents: usize);
pub fn scenario_orange_tier_byzantine(app: &mut App, n_orange: usize, n_green: usize);
pub fn scenario_tend_velocity_under_demurrage(app: &mut App, n_agents: usize);
```

Each scenario:
1. Spawns `n` `McNpc` entities with consciousness tier + DID components.
2. Registers systems that drive each NPC to make Mycelix calls at random phase.
3. Asserts invariants after `N` ticks.
4. Returns a `ScenarioReport` for CI consumption.

These are **the real Phase 1 deliverable** — the Bevy Resource is plumbing; the scenarios are the product.

## Phased implementation

### Milestone 1 — Spike (½ day)

- Create crate dir outside workspace. Copy `mycelix-conductor-bridge/Cargo.toml` pattern for serde pinning.
- Wire `bevy-tokio-tasks` + `flume` + `symthaea-mycelix-holochain`.
- `fn hello_world_system(...)` sends `GetActiveProposals { requester: Entity::PLACEHOLDER }`; background task answers; system logs count.
- Manual test against a running conductor.

**Gate:** one round-trip zome call succeeds from a Bevy system.

### Milestone 2 — Event API + 3 zome calls (1 day)

- Add `MycelixRequest::SubmitProposal { requester, text, did }` and `MycelixRequest::Vote { requester, proposal_hash, approve, did }` variants. (`GetActiveProposals` already shipped in M1.)
- Add corresponding `MycelixResponse` variants (`ProposalSubmitted { action_hash }`, `VoteCast { proposal_hash }`).
- Extend `WireCommand` + `WireResponse` in `systems.rs` to map these to the JSON protocol already spoken by `mycelix-conductor-bridge`. The subprocess side already supports `SubmitProposal` and `CastVote` — no subprocess changes needed.
- Add `MycelixRequest::QueryTendBalance { requester, did }`. This requires a new subprocess command — extend `mycelix-conductor-bridge::Command` enum with `QueryTendBalance`, plumb through to `FinanceDispatcher::query_tend_balance`.
- Correlation: upgrade from FIFO to request-ID. Each request carries a `u64` correlation id; responses include it.
- Integration test: spawn 5 entities, each submits one proposal, all proposals appear in the next tick's `GetActiveProposals` response.

**Gate:** 5-entity test passes reliably against a running conductor.

### Milestone 3 — Scenario harness (1 day)

- `ScenarioConfig`, `ScenarioReport`, `run_scenario(app, config, ticks)`.
- First three scenarios: proposal_vote_invariant, orange_tier_byzantine, tend_velocity_under_demurrage.
- Wire into monorepo CI as `symtropy-mycelix-verify` (peer to `symtropy-governance-verify`).

**Gate:** CI runs all three scenarios green on PRs.

### Milestone 4 — Visual NPC example (½ day)

- Example `examples/npc_village.rs`: 50 entities in a Bevy scene, each has a visible consciousness tier (coloured sphere), each makes 1 Hz zome calls, colour changes when a proposal they voted on passes.
- Not a CI test — a demo. This is the "show, don't tell" artefact for funders and the community.

**Gate:** demo runs at 60fps with 50 NPCs + visible activity.

Total: **~3 days** of focused work. Each milestone has a clear end state.

## Unknowns / open questions

1. **Conductor availability in CI.** Do we run a Holochain conductor as part of CI, or mock at the transport layer? Running real Holochain in CI is expensive (startup ~10s) but mocking loses the integration-test value. **Recommendation:** real conductor, only run on nightly CI; PR CI runs scenarios against a mock.

2. **`inflight_budget` default.** 128 is a guess. Want to measure actual max-in-flight from a 50-NPC scenario and tune. Probably fine for milestone 1–3.

3. **Reconnect strategy.** Single-retry in the current plan. If conductor restarts mid-scenario, the scenario fails. Acceptable for now; revisit if it bites.

4. **DID lifecycle.** Each NPC needs a DID. Do we pre-mint them at scenario start (real registration call), or use pre-seeded test DIDs from a fixture? **Recommendation:** fixture pool for unit tests, pre-mint for long-running scenarios.

5. **Which zomes to cover first.** Current plan: governance (proposal, vote), finance (TEND balance). Should we also include commons (resource claims) and civic (justice, emergency)? **Recommendation:** defer to milestone 2 gate — user decides after proposal/vote works end-to-end.

## Deployment-target question (from review)

External reviewer asked: *"native desktop vs WASM browser vs both?"*

**For Track B specifically: native only.** `holochain_client` and `symthaea-mycelix-holochain` are tokio-based; WASM would require a `web_sys::WebSocket` re-implementation (Track A's Phase 4 work). This plan assumes native.

**For the broader three-track question:** the tracks are deliberately split on deployment target:
- Track B: native (integration tests, CI, desktop binaries).
- Track C: native (in-game UI screens via bevy_egui).
- Track A: browser-only (WASM + CSS-3D iframe overlay for hero demo).

Players of a native Symtropy game see Track C screens reading real Mycelix state via Track B. Players of a browser Symtropy demo see Track A iframes with the real Leptos UI. No deployment target is left uncovered; none of the tracks force WebView embedding.

## Licensing

`symtropy-mycelix-bridge` will be **AGPL-3.0-or-later**. Reasoning:
- Transitively depends on `symthaea-mycelix-holochain` (AGPL) which depends on `holochain_client` (CAL-1.0, Cryptographic Autonomy License — copyleft).
- Mycelix governance / economy logic is the IP moat; permissive licensing here would undermine the commercial licensing path.

Users of the permissive core crates (`symtropy-math`, `-physics`, `-bevy-core`) are unaffected — they simply don't take this dep.

## Decision points

All four M1 decisions confirmed 2026-04-17:

- [x] **Crate location:** top-level sibling `/srv/luminous-dynamics/symtropy-mycelix-bridge/`. (Post-pivot: the serde-conflict rationale went away, but keeping it as a top-level sibling for cohesion with `mycelix-conductor-bridge` and `symthaea-mycelix-holochain` is still the right call.)
- [x] **Zome scope for M2:** governance + finance only (no commons/civic in M2; expand once the pipeline is proven).
- [x] **CI strategy:** mock transport for PRs (fast), real conductor for nightlies.
- [x] **Kickoff:** M1 completed in worktree `session-mycelix-bridge-m1` (branch `worktree-session-mycelix-bridge-m1`). Same pattern for M2+.

## Pending decisions for M2

- [ ] **Worktree re-use.** Continue in `session-mycelix-bridge-m1` or spin up `session-mycelix-bridge-m2`? (Recommendation: new worktree, so the M1 delta stays reviewable independently.)
- [ ] **Subprocess command extension.** `mycelix-conductor-bridge` needs `QueryTendBalance`, `QueryActiveProposals` (and possibly others). Contribute to that crate directly, or fork into a symtropy-specific variant? (Recommendation: direct contribution — it's a small additive change and benefits other consumers.)
- [ ] **Correlation id protocol.** Add `id` field to `WireCommand` + `WireResponse`? (Recommendation: yes. FIFO correlation is too fragile once requests can fail or arrive out of order.)

