# Symthaea Improvement Plan — July 2026

Synthesized from a 5-dimension parallel review (architecture, integration gaps, Broca,
CI/build health, robotics/embodiment) run 2026-07-03. All file:line references verified
against the working tree that day. Kept at monorepo root deliberately — `symthaea/docs/`
syncs to the public standalone repo; this is internal.

## Verdict

The system is deep and mostly honest about itself, but it is **split-brained**: the
LLM-facing product path (`Symthaea::process()`) and the autonomous ~31Hz loop
(`CognitiveLoopService::cycle()`) are two separate cognitions sharing almost nothing —
different Phi, different ethics coverage, different memory, and two independent
"Broca" text engines. The highest-value work is **unification and closing safety
gaps**, not new capability. Second theme: a large amount of **built-but-dark** code
(orphaned modules, disabled quality switches, untested features) that inflates the
claimed capability surface without running.

## Safety findings (fix first — all are cheap relative to impact)

1. **The product path ships un-gated by the ethics engine.** `EthicsEngine` + verdict
   gating exists only in the cognitive loop (`src/cognitive_loop/mod.rs:734`, gated at
   motor output `cycle.rs:307`). `Symthaea::process()` output has no ethics-engine
   check — `ContinuousMind` only has mesh `moral_topology` gossip (`src/mind/tick.rs:62`).
2. **The ahimsa fast-path is dead code in production.** `cycle.rs:315` hardcodes
   `ahimsa_violated: false`. Every platform's ahimsa→Red logic passes unit tests but is
   unreachable live; only `VERDICT_BLOCKED` can reach Red.
3. **`enable_moral_anomaly_response: false`** by default (`config/mod.rs:901`) — the
   anomaly detector is constructed but its response is disabled.
4. **`SafeFallback` docs overpromise.** Core doc table (`embodiment.rs:301-311`) promises
   fallback behavior for Manipulator/AUV/Quadruped/Surgical; only humanoid, multirotor,
   vehicle, helicopter actually implement it (4/10).
5. **(Found 2026-07-03, Phi audit) The one Phi implementation actually gating motor
   safety in production has no enforced accuracy validation, and a formal
   "consciousness verdict" generator sitting in compiled, exported, one-call-away
   production plumbing weights a self-debunked method at its highest confidence.**
   See "The Phi/Psi sprawl" in Phase 1.5 below — this is the single most concerning
   finding of the whole review and belongs here, not buried in a phase note.

(Non-issue, resolved during review: the "plaintext crates.io token" in
`symthaea/CLAUDE.md:10` was the BWS secret *ID*, not a token; wording fixed 2026-07-03.
CLAUDE.md is not in the standalone sync file list — nothing leaked.)

## Phase 0 — Broken things & dead weight (S effort each, ~1-2 days total) — DONE 2026-07-03

- [x] **Ahimsa dead-path**: derive `ahimsa_violated` in `cycle.rs:315` from a real ethics
      signal instead of the hardcoded `false`. Commit `28d6a83581`. Wired from
      `EthicsEngineOutput::ahimsa_violated` (deontological `ahimsa_*`/`prevent_suffering`/
      `minimize_collateral` violations net of restorative-justice credit) via a new
      `CognitiveLoopService::last_ahimsa_violated` field, set in `cycle_strategy.rs`
      alongside `last_ethics_verdict`. Verified: `cargo check -p symthaea --lib --locked`
      clean.
- [x] **CI: remove the always-failing matrix leg / pin sbom / drop redundant job / tighten
      timeout**. Commit `a9b22a6c7d`.
- [x] **Broca half-split breakage** (`code-sheaf-eval` + two broken bins). Commit
      `92144a1578`. Root cause: `code_analysis.rs` was misclassified as toolkit in the
      2026-07-02 split — `evaluation.rs` (core) needs it, and broca-tools depends on
      broca (not the reverse), so the import was structurally impossible after the move.
      Moved it back; relocated `broca_dreamer.rs`/`fused_cognitive_node.rs` to
      broca-tools (they need `foraging_bridge`/`invariant_guard`/`memory_ring`, all
      toolkit). Verified: both `-p symthaea-broca --features code-sheaf-eval` and
      `-p symthaea-broca-tools --bins --features mamba-cpu` clean.
- [x] **Delete orphans** (`proof_state.rs`, `curriculum_generator.rs`). Verified via main
      crate check.
- [x] **Actuator-count table divergence**. Commit `ba090577b6`. Fixed Humanoid 64→21
      (Dmc21 default, not FullSpine). No automated parity test added — would pull AGPL
      humanoid physics + bevy into a deliberately thin/permissive-licensed bridge crate
      for one integer assertion; documented the manual-recheck expectation instead.
- [x] **Workspace `exclude` bug (found during Phase 0, not in original scope)**. Commit
      `ced92660ec`. Real Cargo behavior: an explicit `"."` in a non-virtual workspace's
      `members` list silently disables `exclude` for every glob-matched member. Had let
      `spark-engine`, `symthaea-lab`, and (post-reorg) `symthaea-muse-wasm` compile as
      full workspace members despite being listed as excluded. Root-caused via isolated
      repro, confirmed via `cargo metadata`/`cargo tree` before and after (146→143
      workspace members). Removed `"."` from `members` (root stays included automatically
      via its own `[package]` table); `default-members` may still reference it.

Process notes from this pass, saved to memory: (1) `git mv` done but not yet committed
can get scooped into an unrelated concurrent session's commit in this shared-tree
monorepo — happened here (`code_analysis.rs`'s move landed via `b015f4aceb`, someone
else's clippy-debt commit); content was fine, just misattributed. (2) `run_in_background`
Bash calls do not reliably inherit `CARGO_TARGET_DIR` (the harness's `VAR=value :` wrapper
only scopes the assignment to the no-op `:`), silently falling back to the shared,
contended default target dir — use `env CARGO_TARGET_DIR=... cargo ...` or run foreground.

## Phase 1 — Safety & truth on the product path (M, ~1-2 weeks)

- [ ] **Seam B — ethics-gate `Symthaea::process()`**: give the facade an `EthicsEngine`
      (reuse `ethics_engine.rs`) and gate `ProcessResponse` before return. ~3-4 files,
      300-500 LOC.
- [ ] **Seam A — one canonical Phi (scope corrected/expanded 2026-07-03, see Phi/Psi
      audit below)**: extract `ConsciousnessCore::phi()` and make
      `ContinuousMind::update_consciousness` (`src/mind/tick.rs:323`, currently a naive
      mean-pairwise-dissimilarity with a hardcoded 0.1 empty-memory fallback) delegate to
      it. The facade's Phi and the loop's Phi are currently incomparable numbers. This
      item originally undersold the problem — it's not two Phis, see below.
- [ ] **Decide `enable_moral_anomaly_response`**: benchmark ON, then default ON or delete
      the subsystem. Same for `enable_validation_overlay` (`config/mod.rs:904`).
- [ ] **Close the SafeFallback gap**: implement for manipulator, quadruped, surgical,
      orbital, auv, exoskeleton per the behaviors the core doc already specifies — or
      amend the doc table to stop promising them. Also unify orbital's 0.3 hard cliff
      (`embodiment.rs:71`) and surgical's parallel `SurgicalSafetyLevel` ladder under the
      `MotorSafetyLevel` contract so per-tier enforcement is auditable.

## Phase 1.5 — The Phi/Psi sprawl (audited 2026-07-03, direct investigation)

Prompted by a direct question ("we may have many phi implementations") after Phase 0
landed. The original review's "two Phis" (facade vs loop) undersold this badly. Full
audit below; four parallel research agents were launched but all failed mid-run to a
session-wide API rate limit (unrelated to the codebase) — findings below are from
continuing solo with direct grep/read, so treat as thorough-but-not-exhaustive.

**At least four live, independent, consequential "how integrated/conscious is the
system" measures exist, plus several dormant ones:**

1. **`SpectralMIPFinder`** (`crates/core/symthaea-core/src/consciousness_metrics/
   spectral_mip.rs`) — O(n³) Fiedler-ordering MIP search over a mutual-information graph
   Laplacian. This is what the **cognitive loop's motor-safety chain actually runs**:
   `ConsciousnessEngine::measure()` (`cognitive_loop/consciousness_engine/measure.rs`)
   pushes every 2 cycles, computes every 47, adapts every 94, writes into
   `self.carryover.history.consciousness_level`. Every robotics platform's
   `MotorSafetyLevel::from_phi()` call (10+ platforms + `motor_bridge.rs`) receives this
   same value uniformly, since it's threaded through the single `EmbodimentBridge::step()`
   trait parameter — **this one consumer chain is internally consistent**.
2. **`ConsciousnessUnificationEngine.psi`** (`src/consciousness/dynamics/
   consciousness_unification.rs`) — a *different* formula entirely: a weighted sum of
   baseline CfC temporal-coherence + voice + flow + relational + body + embodied
   contributions (`cognitive_loop/helpers/cycle_extracted.rs:544-554`), explicitly
   Tononi-1994-inspired in its comments but structurally unrelated to SpectralMIPFinder.
   Feeds **`ethics_engine.rs:927`** (`consciousness_level: input.unified_psi`) and
   **Broca's generation gate** (`training.rs`, `broca_psi > 0.4`). So within the *same
   cognitive cycle*, motor safety and ethics/language-generation gate off two unrelated
   numbers with no cross-validation between them — this is the real headline finding,
   more precise than the original "facade vs loop" framing.
3. **`TieredPhi`** (`hdc/tiered_phi/`) — a well-documented, deliberately multi-tier system
   (RandomBaseline/SampledPartition/SpectralConnectivity/ExhaustivePartition) with an
   honest warning that its own SpectralConnectivity tier (algebraic connectivity λ₂) is
   **"deprecated: r = -0.62 with true Φ"** (`tiered_phi/core.rs:106`). Real production
   consumers are narrow and non-safety-critical: `symthaea-quadruped`'s curiosity drive,
   `symthaea-physics`'s plasma encoder, and — via the `phi_engine` caching wrapper —
   **`symthaea-mycelix-bridge`'s federated-learning gradient quality scoring**
   (`assess_update()`/`ConsciousnessBackend`, mycelix-core FL research, NOT the civic
   governance system — corrected below).
4. **`ContinuousMind::update_consciousness`** (facade, `src/mind/tick.rs:323`) — bespoke
   pairwise-HV-dissimilarity mean, doesn't call into any of the above, plus ad hoc
   "relational psi boost" and "swarm phi boost" (`mesh_peers.average_phi()`) multipliers
   layered on top. Two *separate* `average_phi()` implementations exist across
   `src/swarm/holochain.rs:196` and `src/swarm/mesh/mod.rs:937`.

**Three "let's unify this" layers exist and none of them is what the hot path calls:**

- **`PhiOrchestrator`** (`hdc/phi_orchestrator.rs`) — adaptively selects between
  `ConnectivityCalculator` (algebraic connectivity — the *same* underlying method
  TieredPhi calls deprecated), `ResonantPhiCalculator` (a fifth algorithm: coupled-
  oscillator dynamics), and `TieredPhi`. Its own doc comment claims algebraic
  connectivity is **"Most accurate"** — a direct, unresolved contradiction with
  TieredPhi's own docs one file over calling it deprecated at r=-0.62. The doc comment
  also names types (`RealPhiCalculator`, `ResonatorPhiCalculator`) that don't match what
  it actually imports (`ConnectivityCalculator`, `ResonantPhiCalculator`) — stale
  documentation compounding the contradiction.
- **`phi_engine/`** — a caching wrapper, itself built on `TieredPhi`.
- **`research/phi-lab/`** — an entire parallel tree with its own copies of
  `tiered_phi.rs`, `phi_orchestrator.rs`, `consciousness_equation_v2.rs`, and 8+
  topology-validation examples. Appears to be a research sandbox, not wired into
  production, but is one more place "Phi" gets independently defined.

None of `PhiOrchestrator`, `phi_engine`, or `TieredPhi` is what
`ConsciousnessEngine::measure()` actually calls for the production consciousness_level —
it goes straight to `SpectralMIPFinder`.

**`TruePhiCalculator`** (`consciousness_metrics/calculator.rs`, Shannon-entropy-based
"TruePhi") is used in physics/neuroscience modules and directly inside
`symthaea-mycelix-bridge` (separately from its `phi_engine`/TieredPhi usage above) — same
FL-research scope, not governance.

**Correction on Mycelix governance** (the original synthesis overclaimed this): the civic
voting-tier system does **not** currently consume any symthaea Phi/Psi value.
`mycelix-bridge-common::consciousness_profile.rs`'s 4D `ConsciousnessProfile`/
`evaluate_governance()` — which does have a `from_symthaea()` adapter constructor — is
explicitly `#[deprecated(note = "Use sovereign_gate::SovereignCredential (8D) instead")]`
at line 346. **The live system is the 8-axis Sovereign Profile**
(`crates/sovereign-profile`, published as `sovereign-profile` on crates.io; consumed via
`mycelix-bridge-common::sovereign_gate.rs`): Epistemic Integrity, Thermodynamic Yield,
Network Resilience, Economic Velocity, Civic Participation, Stewardship & Care, Semantic
Resonance, Domain Competence. Checked `sovereign_gate.rs` and `sovereign-profile/
collectors.rs` directly — zero references to Phi/Psi/symthaea. All 8 axes are externally
measurable civic signals (smart meter, node uptime, ledger, jury records) by design, so
this is not a live inconsistency, just confirms governance and cognition are — correctly,
deliberately — orthogonal systems today.

### Follow-up sweep (2026-07-03, two agents, both completed — rate limit had cleared)

**Finding A — the documented "4-component weighted blend" is fiction in the shipped
binary.** `ConsciousnessEngine::measure()`'s doc comment describes a weighted consensus
of SpectralMIPFinder (0.35) + ConsciousnessEquationV2 (0.25) + UnifiedConsciousnessPipeline
(0.25) + MultiModalIntegrator (0.15), each at its own co-prime interval. But
`ConsciousnessEngine::new()` has exactly one production call site
(`cognitive_loop/constructor.rs:989-999`), constructed as
`(SpectralMIPFinder::with_defaults(), None, None, None)`. The other three fields have
**no setter anywhere in production code** (only `tests.rs` sets them). So every
`if let Some(...) = self.x { ... } else { 0.0 }` branch for the other three always takes
the `0.0` arm, and `compute_unified()` collapses to
`max(0.35 * sigmoid(spectral_mip_phi), 0.05)`. This has been true since at least the
2026-06-10 archive snapshot — not a recent regression. Consequence: `episodic_consolidation_boost`
is always `None` and `subsystem_lr_factor` is always a no-op 1.0 in production, both only
ever set inside the dead branches.

The three dormant components are real, distinct algorithms, not stubs:
- **`ConsciousnessEquationV2`** (`src/consciousness/measurement/consciousness_equation_v2.rs`) —
  a genuine 7-component soft-min blend: `C(t) = σ(softmin(Φ,B,W,A,R,E,K;τ)) ×
  weighted_coherent_sum(state) × substrate × ρ(t)`. When live it would read
  `sigmoid(spectral_mip_phi)*0.7 + unified_psi*0.3` as its `Integration` input — i.e. it's
  *designed* to already reconcile Φ and Ψ, which undercuts the case for building a
  separate reconciliation (see updated recommendation #2 below).
- **`UnifiedConsciousnessPipeline`** (`src/consciousness/dynamics/
  unified_consciousness_pipeline.rs`) — contains its own **second, separately-instantiated
  `ConsciousnessEquationV2`**, whose output is computed then *discarded* (only
  `limiting_factor` is kept) in favor of a hand-tuned bypass formula
  `sigmoid(ltc.estimate_phi()) * (0.5 + 0.25*binding + 0.25*workspace)`, because the "real"
  equation returns near-zero on cold start. `ltc.estimate_phi()`
  (`consciousness/integration/hierarchical_ltc.rs:683-709`) is a **fourth** distinct
  Phi-like heuristic (normalized pairwise correlation between LTC circuit outputs × global
  coherence) — so this one struct alone contains three different "Phi" computations, two
  of which it throws away.
- **`MultiModalIntegrator`** (`consciousness/integration/multi_modal_integration.rs`) —
  its own doc comment calls `compute_integrated_phi()` a "simplified heuristic": a
  hand-weighted sum over a static convergence-zone hierarchy
  (Primary/Secondary/Tertiary/Amodal, weights 0.5/1.0/2.0/3.0), not IIT math, despite
  being branded "Φ-Guided Binding" throughout the module.

Confirmed test/research-only, unreachable from live `src/` (zero construction sites
outside their own module tree, only re-exported): `SelfConsciousnessAssessment::
compute_phi_self()` (averages hand-set, never-computed "integration" attributes),
`FractalConsciousness`, `PhiGradientTopology`, `MinimalPhiValidation`. Both
`FractalConsciousness` and `PhiGradientTopology` build on `ConnectivityCalculator`
(`hdc/spectral_connectivity.rs`) — meaning `PhiGradientTopology` gradient-ascends network
topology to *maximize* a metric the codebase itself has already shown (r=-0.62) doesn't
track integration. Harmless since unreachable, but a landmine if anyone wires it up.

**Finding B — the one live Phi has no enforced validation, and one dormant module
actively inverts the known-bad-method finding.** This is the most serious result of the
whole review.

1. **`SpectralMIPFinder` — what actually drives `consciousness_level` in production —
   has no validation of its own.** Its module and test file contain zero
   correlation/ground-truth claims, only self-consistency checks.
2. **A real correlation test exists and has never run.** `tests/test_spectral_mip_validation.rs`
   computes SpectralMIPFinder vs. exhaustive search over synthetic covariance matrices,
   asserting ρ > 0.50. The root `Cargo.toml` sets `autotests = false` and enumerates 18
   explicit `[[test]]` targets — this file isn't among them. It has never executed under
   `cargo test`, isn't in CI, isn't checked by anything, ever.
3. **Even the best-documented number doesn't validate what it sounds like it validates.**
   `docs/PHI_VALIDATION_RESULTS.md` claims r≈0.99 for SpectralMIPFinder — but its own
   author caveat: *"This validates the MIP search strategy (Fiedler vs exhaustive). It
   does NOT validate the Gaussian MI framework against true IIT Φ (which requires TPMs,
   not covariance)."* I.e. it shows the fast approximation agrees with an exhaustive
   search over the *same simplified proxy* — not that the proxy tracks canonical Φ.
4. **The one wired-and-running test only checks liveness, not accuracy** —
   `tests/consciousness_ablation.rs::test_spectral_phi_produces_real_values` asserts
   non-zero output over 100 cycles, nothing about correctness — and an adjacent comment
   misattributes SampledPartition's r=0.9998 (a different method) to SpectralMIPFinder.
5. **`consciousness_verifier.rs`** computes a formal `ConsciousnessVerdict`
   (StronglyConscious/LikelyConscious/…/NotConscious) from three methods, and its own doc
   comment (line 249) weights the deprecated `SpectralConnectivity`/λ₂ method **3.0 —
   the highest of the three — calling it "empirically most reliable,"** the exact opposite
   of `tiered_phi/core.rs`'s own r=-0.62 finding one crate over, with zero cross-reference.
   It's wired into `ConsciousnessPipeline`'s optional `verifier` field
   (`hdc/consciousness_integration/pipeline.rs:246-252`, enabled via
   `enable_verification()`) — and per Finding A's sibling investigation,
   `ConsciousnessPipeline`/the `consciousness_integration` tree has zero construction
   sites in live `src/` outside its own module. So it isn't issuing verdicts to anyone
   today, but it's a fully-built, exported, one-call-away attractive nuisance that would
   confidently label a system's consciousness using a method already proven not to work.
   `unified_consciousness_engine.rs` (`ConnectivityCalculator` used unqualified as "the"
   integrated-information measure, `:273`) and `phi_engine`'s `ResonantPhiCalculator`
   doc table (cites its own r=-0.62 with no warning column) repeat the same pattern in
   miniature. Several other modules (`adaptive_topology.rs`, `causal_emergence.rs`,
   `phi_guided_search.rs`, `topology_synergy.rs`) store `ConnectivityCalculator` output as
   `phi`/`local_phi` with the deprecation caveat living only in `spectral_connectivity.rs`'s
   own doc comment, never propagated to callers.

**Finding C — symtropy's "Φ" and the flagship paper's validation table.**
Symtropy (the game-engine sibling) never imports any symthaea Phi engine
(`SpectralMIPFinder`/`TieredPhi`/`ConnectivityCalculator`). It imports
`symthaea-consciousness-equation` (the Master Consciousness Equation) and fills its Φ
input slot with locally-invented heuristics at every call site: inverse danger level
(`symtropy-robotics-bridge/src/agent.rs:316`), oscillator coherence
(`symtropy-bevy/examples/pendulum_swarm*.rs`), or a hardcoded `0.5` constant
(`symtropy-consciousness-physics/src/{biometrics,phase_transition,wasm_bindings}.rs`).
The `jphi.md` paper's "Joules per bit of integrated information" framing and the
`phi_trace_sim_driven_*.rs` / `phi_ablation`/`phi_causal` examples all consume this same
self-contained heuristic pipeline — none of it is IIT integration. Separately, the
flagship `papers/book/symthaea_book.tex` validation table (line 991-1013) lists Sampled
Partition (r=0.9998), Spectral MIP (r=0.99), and λ2 (r=-0.14) as if comparably rigorous
"against exact exhaustive computation" — per Finding B point 3, the Spectral MIP row is a
meaningfully weaker claim than the table implies. Psych-bench's Phi-adjacent benchmarks
(`qualia_confidence/`) are self-disclaiming ("do NOT measure qualia directly," module
doc) and mostly don't call any Phi calculator at all (one uses its own ad hoc
`phi_proxy`); the Butlin framework's IIT indicators only wire to real SpectralMIPFinder
output behind a feature+`#[ignore]`-gated test that doesn't run under default settings.

**Recommendation, in order (supersedes the original three):**
1. (S) **Remove or fix `consciousness_verifier.rs`'s 3x weighting of the deprecated λ₂
   method**, and add the same warning to `unified_consciousness_engine.rs` and
   `phi_engine`'s method table. This is dormant today but the single highest-severity
   landmine in the whole audit — fix before it ever gets wired to `enable_verification()`.
2. (S) Fix the `PhiOrchestrator` vs `TieredPhi` doc contradiction about algebraic
   connectivity (already noted above).
3. (M) **Wire `test_spectral_mip_validation.rs` into the actual test suite** (add it to
   `Cargo.toml`'s explicit `[[test]]` list) and re-run it. Right now nobody knows whether
   SpectralMIPFinder — the thing actually gating robot motor safety — clears even the
   weak ρ > 0.50 bar, because the test that would tell you has never executed.
4. (M) Correct `docs/PHI_VALIDATION_RESULTS.md` and `papers/book/symthaea_book.tex`'s
   validation table to state plainly that the Spectral MIP r≈0.99 number validates search
   strategy against a Gaussian-MI proxy, not agreement with true TPM-based IIT Φ — the
   current phrasing invites exactly the conflation `consciousness_ablation.rs`'s own
   comment already made (misattributing SampledPartition's number to it).
5. (M) Decide, in writing, whether Φ (SpectralMIPFinder) and Ψ (UnificationEngine) are
   *meant* to be different measures (plausible — Ψ's inputs read closer to
   "engagement/flow" than "integration") or whether ethics/Broca should gate off Φ too.
   Note `ConsciousnessEquationV2` already has a designed-in Φ/Ψ blend
   (`sigmoid(Φ)*0.7 + Ψ*0.3`) sitting dormant — turning that on (Finding A) may be a
   cheaper fix than building a new reconciliation from scratch, once its own cold-start
   near-zero-output problem (which is why `UnifiedConsciousnessPipeline` bypasses it) is
   addressed.
6. (L) Fold into Seam A: whatever becomes canonical for the facade should be the *same*
   canonical, *validated* source the loop uses — not a third independent formula.
7. (S, low priority) Add one sentence to `jphi.md` and the phi-gated-safety paper
   clarifying that symtropy's "Φ" is a locally-defined heuristic input to the Master
   Consciousness Equation, not a measured IIT quantity — avoids the paper being read as
   claiming more than the code does.

## Phase 2 — Finish what was started (M, ~2-3 weeks)

- [ ] **Finish the Broca split** (it's half-done; `symthaea-broca-tools` already exists
      with 39 files): move the ~26 remaining toolkit modules + 5 toolkit bins out of
      `symthaea-broca`, rewrite `crate::` → `symthaea_broca::` (toolkit→core coupling is
      one-directional, ~8 files). One real blocker: sever
      `emotional_gating_integration.rs:6` → `compiler_trainer::CompilerVerdict` so the
      core gating trio (codegate / language_gates / emotional_gating_integration) stays
      core. Then prune now-dead heavy deps from broca's Cargo.toml (`rnix`, `rowan`,
      `tree-sitter-*`, `pqcrypto-*`, `sled`, `syn`).
- [ ] **Automate the Broca quality gate**: `evaluate_quality_suite` +
      `check_quality_suite` with `CanonicalQualityThresholds` (`evaluation.rs:449-460`)
      over `tests/fixtures/eval-canonical-v1.jsonl` already exist but only run via the
      manual `broca-eval` bin. Wrap in a `#[test]` so generation-quality regressions
      gate CI. Note: no BLEU/exact-match exists; `avg_coherence` (cosine vs thought HV)
      is the fidelity metric.
- [ ] **NSM A/B — measure then enable or delete**: `enable_nsm_semantic` /
      `enable_nsm_gate` are fully plumbed (main loop already composes the semantic HV,
      `training.rs:670-700`) but never measured end-to-end (`generator.rs:209-215`
      comment admits this). One eval run over the canonical set decides it; no
      retraining needed. Note the main-crate config defaults these ON
      (`config/mod.rs:928-932`) while the crate defaults them OFF — reconcile.
- [ ] **Know the decoder you ship**: `BrocaConfig::default()` uses
      `BrocaDecoderKind::Structured` (deterministic readout, `generator.rs:196`) — the
      trained CfC (`Direct`) and Liquid-Mamba paths are opt-in. Decide/document which is
      the intended production path; the semantic-veto lever is unusable without
      retraining (CfC output and thought-HV live in different spaces, veto baseline ≈ 0).
- [ ] **Wire (or explicitly defer) the robotics telemetry edge**: a conductor call path
      exists (`symthaea-mycelix-conductor/src/lib.rs:619` → `robotics_dispatch::
      submit_telemetry`) but nothing drains `sensorimotor.embodiment_telemetry` from the
      loop into it — mock-test-only today.

## Phase 3 — Structural debt (L, ~1 month, parallelizable)

- [ ] **Reverse the CLS field regression** (~121 fields, back from post-refactor ~59):
      extract `PredictiveCore` (~12 loose training/prediction fields), `ChannelHub`
      (~10 IO/channel fields), fold `ethics_engine` + verdicts into the existing
      `ethics_values` manager. Target ~80 without behavior change.
- [ ] **De-monolith**: move inline glue out of `cycle()` (`cycle.rs:38-728`, ~690 lines:
      integrity sweep, USI/spectrum→neuromod coupling, embodiment step, distress
      emission) into managers; split the monolithic phase files
      (`cycle_phase_dynamics/mod.rs` 176KB, `cycle_phase_feedback.rs` 111KB) by cohesion.
- [ ] **Byzantine island decision**: `byzantine_collective` + `causal_byzantine` +
      `meta_learning_byzantine` + `unified_intelligence` + `asset_evaluator`
      (~3,590 LOC) only reference each other and never activate under any default/profile
      bundle. Wire `multi_agent` into a real profile or archive the island.
- [ ] **Candle unification**: main crate pins candle 0.8.4, broca pins 0.10.2 — two full
      candle stacks compiled, and the vendored `vendor/cudarc-0.13.9-cuda129` patch
      exists only because candle 0.8 needs cudarc 0.13.x. Migrate `neural-bridge` to
      candle 0.10 → delete the vendor patch and the duplicate stack. Also: bevy 0.15+0.18,
      wgpu ×3, rand ×4 — worth a dedup pass.
- [ ] **CI depth**: add `[lints] workspace = true` to the 47/151 member crates missing it;
      add a workspace (or subcrate-set) clippy gate (currently only the main crate is
      linted); add `quantum-consciousness` and `sentinel` singleton legs (both are real
      cfg-gated features with zero CI coverage); batch more of the ~90 never-directly-
      tested crates into `test-subcrates`; add a per-PR criterion smoke on the hottest
      benches (zero perf gating exists today).
- [ ] **`enable_*` config audit**: ~14 subsystems built-but-disabled in
      `config/mod.rs:846-974` (online learning, metabolic conductor, coherence field,
      semantic encoder, physics bridge, federation, FHE wisdom, …). For each: benchmark →
      default ON, or delete. Dark code that's never measured is liability.
- [ ] **Robotics contract uniformity**: add explicit state-dim/cmd-dim metadata to
      `EmbodimentTelemetry`; make all Phase-1 platforms populate `platform_specific`
      (only manipulator does) and use `GroundingEstimator` (only manipulator does; the
      rest hardcode `GROUNDING_SENSORIMOTOR`); source `num_actuators` from consts, not
      literals. Add deterministic replay harnesses for quadruped/surgical/orbital/
      exoskeleton (multirotor/humanoid/vehicle/manipulator already have benchmark or
      scenario harnesses).

## Phase 4 — Unification endgame (L, do only after Phases 1-3 prove the seams)

- [ ] **Seam C**: make `Symthaea` own a `CognitiveLoopService` and delegate
      perception+dynamics to `cycle()`, demoting `ContinuousMind` to a thin
      input/memory adapter. High risk (121 fields, 49 cfg gates) — last.
- [ ] **One Broca contract**: the facade Phase-5 translator is an LLM
      (`llm_organ::translate_thought`, `symthaea/mod.rs:1535`) while the loop uses the
      SSM `symthaea-broca` generator via `broca_bridge.rs` — two engines, two
      independent liquid-mamba stacks, no shared eval. Decide primary/humanizer roles
      and route both through one interface.
- [ ] **Consolidate memory**: three independent coordinators (`Symthaea`,
      `ContinuousMind`, CLS `memory_execution`) → one owner.

## Documentation corrections (stale claims found during review)

- Broca cadence is NOT "static interval 61" — it's dynamic spacing (base 5/7 modulated
  by fatigue/governance/quality, gated on `broca_psi > 0.4`, `training.rs:625-639`).
- Broca `#[test]` count is 422, not ~895. Largest test unlock: `mamba-cpu` (compiles
  clean, gates ~104 exclusive tests + 70 cfg blocks).
- `symthaea-surgical` and `symthaea-orbital` are NOT stubs — they're minimal Phase-2
  bridges (1,215 / 1,017 LOC, real safety ladders, 15/11 tests) lacking SafeFallback,
  replay, and telemetry payloads. But surgical's lib.rs doc quotes physical limits
  (5N/50mm/s) that are not what the code enforces (abstract torque_gain), and orbital's
  "dual-body dynamics" claim is unverified against its 224-LOC simulator.
- `src/mycelix/` (12K LOC) is live by default; `src/consciousness/mycelix_bridge.rs`
  (2.8K LOC) is dormant under default features. Both are called "mycelix" — rename or
  merge to stop the confusion.
