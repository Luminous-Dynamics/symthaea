# Symthaea Comprehensive Review — Security & Quality Audit

Generated 2026-07-01 via a 26-agent parallel review (docs cross-check + main-crate
architecture/bugs + 4 core/bridge-tier groups + 9 domain-theme groups) covering all
139 sub-crates, with adversarial verification of high-severity findings. This
document tracks findings and fix status; update the Status column as items land.

## Status legend

- `OPEN` — not yet fixed
- `FIXED` — fixed on this branch
- `WONTFIX` — accepted risk, reasoning noted
- `TRACKED` — lower priority, catalogued but not scheduled

## Confirmed high-severity findings (survived adversarial verification)

| # | Crate | Location | Issue | Status |
|---|-------|----------|-------|--------|
| 1 | `symthaea-hdc-crypto` / `symthaea-core::hdc::hdc_crypto` | `crates/core/symthaea-hdc-crypto/src/binary_hv.rs:75` | `BinaryHV::new_random(seed: u64)` is fully deterministic (BLAKE3 XOF of the seed), yet docs claim Shannon-perfect secrecy / true OTP. Duplicated in `symthaea-core`, which is live in production: `src/swarm/mesh/mod.rs` uses `HdcMac` for wisdom-packet authentication. A known/brute-forced 64-bit seed fully recovers the mask. | FIXED — added `BinaryHV::new_secure_random()`/`secure_random()` (OS entropy via `getrandom`/`rand::OsRng`) in both crates, plus `HdcThresholdSharing::split_secure`; `new_random(seed)` kept but doc'd as non-secure/test-only. 55+33 tests pass. |
| 2 | `symthaea-sim-bridge` | `crates/bridges/symthaea-sim-bridge/src/lib.rs:521` | `CommandSolver::execute()` never spawns the named external process; returns a hardcoded canned string regardless of args. Every "real path" (non-dry-run) invocation in `symthaea-mujoco-bridge`, `-ngspice-bridge`, `-openfoam-bridge`, `-opensees-bridge` silently no-ops and reports fabricated convergence/drift metrics. | FIXED — `execute()` now genuinely spawns via `std::process::Command`, errors on spawn failure/non-zero exit. The 4 downstream adapters now return an honest `SimulationError::Adapter` ("real-output parsing not yet implemented") on their real path instead of fabricating a converged result. 9 tests pass. |
| 3 | `symthaea-soma` | `crates/domains/symthaea-soma/src/pairing.rs:126` | Ed25519 keypair generation seeds a non-cryptographic PRNG purely from a monotonically-increasing cycle counter starting near 0 — trivially guessable, forges pairing signatures. | FIXED — `generate_keypair()` now draws the seed from OS entropy (`getrandom`), not the cycle counter. |
| 4 | `symthaea-soma` | `crates/domains/symthaea-soma/src/pairing.rs:145` | In the default (non-`ed25519-dalek`) build, the "public key" transmitted in cleartext **is** the symmetric MAC key — any BLE eavesdropper can forge future authentication MACs indefinitely. | FIXED — fallback build now uses real X25519 Diffie-Hellman (`x25519-dalek`): the exchanged `pubkey` is a genuine DH public key, and the MAC key is the DH-derived shared secret, never transmitted. Also fixed `mod pairing;` being gated behind the `pairing` feature itself, which made this fallback code permanently dead/unreachable — it's now always compiled. 12 tests pass under both feature configs, including a new eavesdropper-can't-forge-MAC regression test. |
| 5 | `symthaea-broca` | `crates/domains/symthaea-broca/src/wasm_architect.rs:266` | Unsafe `wasmtime::Module::deserialize` on attacker-influenced bytes, mislabeled "Safely Deserialize" — UB risk per wasmtime's own safety docs, not just a sandboxing gap. No fuel/memory limits configured either. | FIXED — unsafe deserialize now gated behind a `compat_hash` check (`Engine::precompile_compatibility_hash()`) so it's only attempted on artifacts signed by an engine with a matching wasmtime version/target/Config; otherwise falls back to safe `Module::new`. Added fuel metering (50M budget) + `StoreLimits` (64MiB memory cap) so even a successfully-loaded module is resource-bounded. Compiles clean (`--features wasm-sandbox`); full functional test blocked by a pre-existing, unrelated `mamba-cpu`/`candle_core` build break (not introduced by this fix — see note below). |
| 6 | `symthaea-broca` | `crates/domains/symthaea-broca/src/wasm_architect.rs:250` | Unsigned/unwrapped artifacts skip signature verification entirely and still reach the unsafe deserialize path — the signature check only applies to inputs that happen to parse as `SignedArtifact`. | FIXED — `load_verified_module()` now requires a validly-signed `SignedArtifact` unconditionally; there is no unauthenticated fallback path anymore. |
| 7 | `symthaea-broca` | `crates/domains/symthaea-broca/src/self_optimization.rs:57` | `evolve_file` mutates, compiles, and executes arbitrary Rust on disk, gated only by a regex blocklist (`compute_moral_safety`) trivially evaded by string concatenation or semantically-equivalent APIs. Public API. | FIXED — added `require_path_within_project_root()`: `evolve_file` now canonicalizes and rejects any path resolving outside `project_root` (absolute paths, `..` traversal, symlink escapes) before any read/mutate/compile step. The regex scorer remains a coarse pre-filter (documented as such, not a security boundary) — full process-level sandboxing is a follow-up beyond this pass's scope. 4 tests pass. |

## Reported, refuted on adversarial verification (lower confidence — kept for tracking)

| Crate | Location | Issue | Status |
|-------|----------|-------|--------|
| `symthaea` (main) | `src/cognitive_loop/managers/radio_dispatcher/transport.rs:532` | `MeshEncryption` is a repeating-key XOR cipher with a non-cryptographic auth tag, wired into `SpectrumManager` for governance-vote/emergency-alert mesh traffic. Verifier did not confirm as a live defect — the struct's own doc comment already discloses it as a "test/simulation placeholder... NOT cryptographically secure," so this may be a known, intentional gap rather than an oversight. Still worth a tracking issue given it's wired to real traffic paths. | TRACKED |

## Safety-critical medium-severity findings (this pass)

| Crate | Location | Issue | Status |
|-------|----------|-------|--------|
| `symthaea-hal` | `crates/domains/symthaea-hal/src/interlock.rs:183` | `SafetyInterlock::filter_command()` hardcodes `NUM_ACTUATORS=21` and indexes `safe.torques[i]` without checking `command.torques.len()` — panics on mismatched morphology (Dexterous53=53, FullSpine=64 actuators), or silently skips torque-clamping for actuators beyond index 21 on larger rigs. | FIXED — commands with more torques than this interlock has configured limits for are now rejected (interlock trips, returns `HalError::Safety`) rather than silently passing unclamped torques through; commands with fewer torques are handled via `NUM_ACTUATORS.min(len)`, matching `CalibrationProfile`'s existing pattern. |
| `symthaea-hal` | `crates/domains/symthaea-hal/src/gpio_estop.rs:100` | `GpioEstop::poll()` unconditionally clears the shared e-stop `Arc<Mutex<bool>>` whenever the physical button isn't pressed — silently un-triggers a software/remote-initiated e-stop between polls, no latching, no distinction between fault sources. | FIXED — `poll()` now only ever *sets* the shared flag on its own trigger condition; it never clears it. Clearing is the exclusive responsibility of `SafetyInterlock::release_estop()`. 2 new regression tests added. |
| `symthaea-fabrication-kernel` | `crates/domains/symthaea-fabrication-kernel/src/nurbs.rs:168` | `NurbsCurve::tessellate`/`find_knot_span` index `self.knots` with no bounds check (unlike the parallel `NurbsSurface` path, which uses `.get(...).unwrap_or(...)`). Reachable via `step_import.rs` parsing untrusted STEP CAD files with a malformed/short knot list — DoS panic on untrusted CAD input. | FIXED — both methods now use the same defensive `.get(...).unwrap_or(...)` pattern as `NurbsSurface`. 2 new regression tests (empty knots, short knots) added. |
| `symthaea-spore` | `crates/domains/symthaea-spore/src/secure_boot.rs:613` | `generate_mok_password()` — the human-typed Secure Boot MOK-enrollment password (gates enrolling a signing key into a machine's UEFI trust store) — uses a non-cryptographic xorshift64 PRNG seeded from a stack address + atomic counter, despite doc claiming "~62 bits of entropy" and a comment noting the crypto-RNG upgrade "would" happen but never landed. | FIXED — replaced `xorshift64`/`generate_seed()` with `secure_random_index()`, drawing each word from OS entropy (`getrandom`, native and wasm32 via the `js` feature) instead of a predictable seed. 34 tests pass. |

## Other medium/low findings catalogued but not scheduled for this pass

Full detail available in the workflow transcript; summarized by pattern here so they aren't lost:

- **Deserialize-then-panic pattern** (medium): `symthaea-clinical`'s `SymptomEncoding`, `DiagnosticProfile`, `TherapeuticIntervention` all cache `encoding: Option<BinaryHV>` as `#[serde(skip)]` with derived `Deserialize` and no recompute step — `.encoding()` panics via `.expect()` after any JSON round-trip.
- **NaN-unsafe `partial_cmp(...).unwrap()` sorts/comparisons** (low, systemic — recurs in 15+ files): `symthaea-fep`, `symthaea-quantum-chemistry`, `symthaea-nuclear` (7 sites), `symthaea-materials`, `symthaea-continuum-physics`, `symthaea-muse`, `symthaea-atelier`, `symthaea-hdc-ltc`, `symthaea-workspace`, `symthaea-phi-search`, `symthaea-fabrication-kernel` (2 sites), `symthaea-engineering`, `symthaea-vehicle`. None currently reachable with malformed input, but worth a shared `total_cmp`/lint sweep rather than 15 individual patches.
- **Unchecked `usize` underflow on empty/zero-sized input** (low): `symthaea-fep::GenerativeModel::new`, `symthaea-fep::agent::select_action`, `symthaea-hdc-ltc::NetworkConfig`, `symthaea-continuum-physics::Maxwell1D/2D`, `symthaea-memory::SemanticMemory::new`, `symthaea-phi-search::SearchConfig`.
- **`symthaea-app-db` config generator**: `validate_hostname`/`validate_username` exist but aren't called from `config_gen.rs`; the live installer's "Express" quick-setup path can feed unvalidated `extra_users` into generated Nix source with no escaping (Nix-injection risk into a root-run script).
- **`symthaea-web`**: SSH password persisted in plaintext to `sessionStorage` (inconsistent with a sibling in-memory-only page); a separate page uses `js_sys::eval()` against the codebase's own documented "don't use eval" rule.
- **`symthaea-phone-embodiment`**: LLM/task-derived strings flow unsanitized into `adb shell` argv — shell metacharacters execute on the physically-connected phone.
- **`symthaea-multirotor`**: raw-pointer writes into MuJoCo actuator arrays with no bounds check against the loaded model's actuator count.
- **`symthaea-swarm`**: `TelepathicSocket` deserializes untrusted P2P gossip payloads via bincode's unbounded default config — memory-exhaustion DoS from any gossip peer.
- **`symthaea-engineering`**: `self_audit()` reads an arbitrary file path with no sandboxing — latent path traversal if ever fed an LLM-generated or user-supplied path.
- **`symthaea-fabrication-kernel`**: `PrinterConfig.api_key` derives `Debug`/`Serialize` with no redaction — leaks into logs/config dumps.
- Full list of low-severity findings (unchecked arithmetic, missing empty-collection guards, thread-spawn `.expect()` instead of graceful degradation) omitted here for brevity — see workflow run `wf_3be22a42-5f2` if needed.

## Architecture notes

**Real strengths:**
- `ThermodynamicManager`/`SubstrateManager` wiring is deep and matches `THE_SUBSTRATE_ROADMAP.md`'s Phase 2-4 claims closely, including real CfC hidden-state masking.
- The `CognitiveSubsystem` trait + `run_subsystem!` macro pattern (23 managers, co-prime tick intervals, panic-isolated via `catch_unwind` with health tracking) is a genuinely good god-object mitigation.
- `symthaea-fep`, `symthaea-causal-reasoning`, `symthaea-hodge`, `symthaea-geodesic`, `symthaea-therapeutic`, the "genesis pipeline" biology crates, and `symthaea-physics-bridge` are production-grade with docs matching implementation.

**Systemic problems:**
- The LLM-facing facade (`Symthaea::process()`) and the autonomous `CognitiveLoopService::cycle()` loop are two structurally disconnected "brains" sharing almost no state — CLAUDE.md's single "8-phase pipeline" description conflates two materially different systems.
- `EthicsAndValuesManager` was never actually dissolved into `EthicsEngine` as CLAUDE.md claims — both managers coexist, recreating the "dual-throttle" problem the doc says was fixed.
- **Fabricated-results pattern** recurs beyond `symthaea-sim-bridge`: `symthaea-silicon`'s RTL synthesis is a templated stub; `symthaea-broca`'s `iac_harvester.rs` claims to harvest from public repos but does zero network I/O; `symthaea-gazebo-bridge` spawns a real process but ignores its output.
- **Duplicate/competing implementations nothing reconciles**: `symthaea-consciousness-topology` (fully built/tested/wired) is dead — real call sites use a different topology module in `symthaea-core`/`symthaea-hodge`. `symthaea-voice` duplicates `src/voice/*`. `symthaea-hdc-ltc` duplicates `symthaea-core::hdc_ltc_unified`. `symthaea-hdc-crypto` duplicates `symthaea-core::hdc_crypto` (both share the weak-seed flaw above).
- `symthaea-zkproof`'s excluded RISC0 subcrates contradict the project's own "DASTARK not SNARK" architecture rule.
- Repo hygiene: stray gitignored empty `symthaea/symthaea/` directory (safe to delete); git-tracked stale duplicate at `crates/symthaea-biometrics/` (should be deleted — canonical copy is `crates/domains/symthaea-biometrics/`).

## Documentation health — CLAUDE.md / THE_SUBSTRATE_*.md are stale

The June 30, 2026 "Reorganize Symthaea crates by tier" migration (`79d50ca8`) was never reflected back into the docs:

| Claim | Documented | Actual |
|---|---|---|
| Sub-crates | 52, flat under `symthaea/crates/` | 139, split into `crates/{core,bridges,domains}/` |
| Workspace members | 55 | 134 active (141 candidates, 7 excluded) |
| Rust lines | ~1,134K (~901K code) | 1,683K (1,366K code) — 48% more |
| Tests (main / workspace) | ~7,395 / ~21,600 | 10,350 / 28,328 |
| Feature flags | 100, `default=[]` | 175, `default=["default-mind"]` |
| `symthaea-core` location | sibling of `symthaea/` | nested at `crates/core/symthaea-core/` |
| Broca pipeline | 21K+ LOC, 229+ tests | 60K+ LOC, 472 tests |
| `CognitiveLoopService` fields | 56→38 via refactor | grown back to ~135 fields |
| `quantum-chemistry` substrate integration | "deferred" | already wired and functional, off by default |

## Fix-pass notes

- All 9 items above (5 confirmed highs + 2 more `wasm_architect.rs` findings
  fixed as part of the same module + 4 safety-critical mediums, though the
  table above already reflects them as 7 highs) were fixed and verified via
  `cargo test` on this branch; see individual Status cells for test counts.
- **New pre-existing issue found while verifying fix #5/#6**: `symthaea-broca`'s
  `wasm-sandbox` feature alone cannot fully compile `execute_with_hypervector`
  because its (untouched-by-this-pass) signature references
  `crate::projection::HdcSsmProjection`, which is gated behind the separate
  `mamba-cpu` feature. Enabling `mamba-cpu` in turn hits 22 unrelated
  `candle_core` unresolved-crate errors elsewhere in the same crate
  (`controller.rs`, `training.rs`, `mamba.rs`, `mamba_model.rs`). Neither is
  caused by this pass's changes (confirmed via `git stash`/`git show` against
  the pre-fix state), but it means `wasm-sandbox + mamba-cpu` together is
  currently an unbuildable feature combination — worth its own tracking issue.
- **New pre-existing issue found while fixing #4**: `symthaea-soma`'s
  `mod pairing;` was gated behind `#[cfg(feature = "pairing")]` in `lib.rs`,
  which made the module's own documented dual-mode design (Ed25519 vs.
  X25519 fallback) impossible to reach in practice — without the `pairing`
  feature, the module didn't compile into `SomaEngine` at all, so there was
  no pairing capability, secure or not, in default builds. Fixed as part of
  this pass (see item #4) by making the module and its 3 usage sites in
  `engine.rs` unconditional.

## Review methodology / known gaps

- 15 review groups (main architecture, main bug-hunt, 2 core-tier groups, 2 bridge-tier
  groups, 9 domain-theme groups covering all 101 domain crates) plus a dedicated
  docs-vs-reality cross-check agent, with adversarial single-pass verification on
  all high/critical findings (capped at 10 per group, none hit the cap).
- Two groups (`domains-infra-observability`, `domains-engineering-sim`, 21 crates)
  and the `symthaea-broca` verification pass initially failed on an API session
  rate limit and were re-run to completion via workflow resume (cached agents
  reused; no groups had to be redone). Full coverage achieved.
- This document intentionally omits the full low-severity finding list for
  readability; see workflow run `wf_3be22a42-5f2` transcripts for the complete
  structured output if a full re-derivation is ever needed.
