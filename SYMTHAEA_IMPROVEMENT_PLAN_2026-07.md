# Symthaea Improvement Plan — July 2026

Synthesized from a 5-dimension parallel review (architecture, integration gaps, Broca,
CI/build health, robotics/embodiment) run 2026-07-03. All file:line references verified
against the working tree that day. Kept at monorepo root deliberately — `symthaea/docs/`
syncs to the public standalone repo; this is internal.

**Extended 2026-07-03 (fourth pass, separate session)**: a fresh architecture review
re-verified the facade/loop split with three targeted scans and folded in five
previously-missing workstreams: a CI field-count ratchet (Phase 3), Broca hard-mask
gating + a confabulation benchmark (Phase 2), feature-flag profile consolidation
(Phase 3), the automated continuous-learning loop (new Phase 5), and an external
leakage-proof capability ladder (new Phase 6). Also updated with exact current facts:
CLS = 131 fields / 42 manager files; `consciousness_verifier.rs` now deprecated-marked;
a second orphaned Phi validation test found in `symthaea-phi-oracle`.

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
1. [x] (S) **Remove or fix `consciousness_verifier.rs`'s 3x weighting of the deprecated
   λ₂ method** — **DONE 2026-07-04, commit `a15b07b47b`.** Deprecated the whole module
   (`#[deprecated]` on the struct, not just a doc banner) with a full explanation;
   deliberately left the numeric weights (3.0/2.0/1.0) unchanged rather than
   "rebalancing" them, since none of the three legs is valid — reweighting invalid
   methods doesn't produce a meaningful result, only replacement does. Added
   `#[allow(deprecated)]` at all 5 consumer sites (2 tests, 1 field, 1 constructor call,
   1 re-export), each annotated with why. Also fixed the same un-warned repeat in
   `unified_consciousness_engine.rs:273` (`compute_unified_psi()` doc comment) and
   `phi_engine`'s method table (previously rated the two invalid methods
   "High"/"Medium" accuracy). Verified: `cargo check -p symthaea-core --lib` clean, 0
   errors, 0 warnings from any touched file.
2. [x] (S) Fix the `PhiOrchestrator` vs `TieredPhi` doc contradiction — **DONE
   2026-07-04, commit `a15b07b47b`.** Corrected the module doc, the adaptive-selection
   ASCII diagram, and both `PhiMode`/`CalculatorType` enum variant docs. Also fixed a
   second bug found while there: the doc-comment type names (`RealPhiCalculator`,
   `ResonatorPhiCalculator`) didn't match what the module actually imports
   (`ConnectivityCalculator`, `ResonantPhiCalculator`) — stale documentation compounding
   the original contradiction. Kept variant names for API stability (renaming
   `PhiMode::Accurate`/`Fast` would be a breaking change out of scope here); added
   caveats instead.
3. [x] (M) **Wire `test_spectral_mip_validation.rs` into the actual test suite** —
   **DONE 2026-07-04, commit `8ee930380f`.** Added the `[[test]]` entry; this
   immediately surfaced real bitrot (a match pattern invalid under the current
   edition's match-ergonomics rules, fixed per the compiler's own suggestion). **Result,
   run for the first time ever: all 6 tests pass. Pearson r = 0.9866, Spearman ρ =
   0.9264 (N=62)** — clears not just the weak ρ>0.50 bar the test asserts, but the
   test's own "STRONG VALIDATION" bar (r>0.70 and ρ>0.70). Confirms the r≈0.99 figure in
   `docs/PHI_VALIDATION_RESULTS.md` is real and reproducible; it just needed to actually
   run to be trustworthy rather than merely written down. Caveat from the audit still
   holds — this validates the Fiedler-ordering search strategy against the same
   Gaussian-MI proxy used for both exact and spectral computation, not agreement with
   true TPM-based IIT Φ.
   *Second instance — DONE 2026-07-04, commit `41c20edccd`.* `crates/symthaea-phi-oracle/`
   sat directly under `crates/` instead of `crates/domains/`, so it was never matched by
   the workspace's glob patterns and its `tests/integration.rs` (`test_signal_loss_fixed`
   included) had also never run. Moved it into `crates/domains/`, fixed the 4 relative
   paths this broke (its own `symthaea-core` dep, plus 3 consumers:
   `symthaea-telemetry-sink`, `symthaea-telemetry-grpc`, `symthaea-bevy-dash`). **Result:
   all 42 tests pass** (27 unit + 14 integration + 1 doctest), including
   `test_signal_loss_fixed`.
4. [x] (M) Correct `docs/PHI_VALIDATION_RESULTS.md` and `papers/book/symthaea_book.tex`'s
   validation table to state plainly that the Spectral MIP r≈0.99 number validates search
   strategy against a Gaussian-MI proxy, not agreement with true TPM-based IIT Φ — **DONE
   2026-07-04.** `docs/PHI_VALIDATION_RESULTS.md`: replaced the flat "✅ MIP search
   validated (r=0.99)" cell (sitting in an "IIT-Valid?" column next to genuinely
   IIT-validated rows) with the exact re-verified numbers (r=0.9866, ρ=0.9264) plus a
   caveat paragraph explaining the different ground truth. `papers/book/
   symthaea_book.tex`: updated Table `tab:phi-validation`'s Spectral MIP row to the same
   exact numbers, added a `$^\dagger$` footnote spelling out the methodological
   difference, and reworded the caption so it no longer implies all four rows were
   validated against the same exhaustive-computation ground truth.
5. [x] (M) Decide, in writing, whether Φ (SpectralMIPFinder) and Ψ (UnificationEngine) are
   *meant* to be different measures or whether ethics/Broca should gate off Φ too —
   **decision written 2026-07-04, no code change** (see below). The actual rewiring, if
   this decision is later revisited, remains item 6's job.

   **Decision: treat Φ and Ψ as intentionally distinct for now, and say so in code —
   don't unify them by default.** Reasoning: Φ (SpectralMIPFinder) answers "is this
   system doing real structural integration right now" — a computational/topological
   property, appropriate for gating something as consequential and safety-critical as
   robot motor authority. Ψ (`ConsciousnessUnificationEngine`, fed by CfC temporal
   coherence + voice + flow + relational + body + embodied contributions) answers a
   different question — "is this system in a good state for social/communicative
   output" — which is what ethics evaluation and Broca's generation trigger actually
   need. A structural-integration measure isn't obviously the *right* gate for "should I
   speak now"; Ψ's inputs are a more direct fit for that decision. So the two-measure
   split is defensible on its face — the actual problem this audit found isn't that two
   measures exist, it's that **nothing in the code says this is deliberate**, so a future
   maintainer (or this review, initially) reads it as accidental drift rather than
   design. That's a documentation gap, not necessarily an architecture bug.
   - This is explicitly a judgment call with a real counter-argument (two ungated,
     uncross-validated "consciousness" numbers is inherently confusing and risks a
     scenario where ethics greenlights something motor-safety would have blocked, or
     vice versa) — flagging that counter-argument rather than hiding it.
   - Chose not to unilaterally rewire ethics/Broca to gate off Φ instead of Ψ, because
     that's a real behavioral change to safety-critical gating logic (what permits
     speech, what permits an ethical pass), not a documentation fix — it needs explicit
     sign-off, not a judgment call made mid-audit.
   - `ConsciousnessEquationV2`'s dormant `sigmoid(Φ)*0.7 + Ψ*0.3` blend (still gated off
     by Finding A — `ConsciousnessEngine::new()`'s only production call site passes
     `None` for it) is worth noting as a *third* option nobody has evaluated: if it were
     turned on (after fixing its cold-start near-zero-output problem), ethics/Broca could
     gate off that blend instead of raw Ψ, getting some Φ-awareness without fully
     replacing Ψ. Not evaluated further here — flagging it as the natural next
     experiment if this decision is revisited.
   - **Follow-up action — DONE 2026-07-04**: added doc comments at the Ψ computation
     site (`compute_unified_psi` in `cognitive_loop/helpers/cycle_extracted.rs`) and the
     Φ site (`ConsciousnessEngine::measure` in `consciousness_engine/measure.rs`)
     cross-referencing each other and this reasoning, so the next reader sees "these are
     deliberately different, here's why" instead of two unexplained numbers. The Φ-side
     comment also now states plainly that only `SpectralMIPFinder` is live in the shipped
     binary (Finding A) rather than repeating the four-subsystem description as if all
     four run. Verified: `cargo check -p symthaea --lib` clean, 0 errors, `Finished` in
     1m02s.
6. [ ] (L) Fold into Seam A: whatever becomes canonical for the facade should be the *same*
   canonical, *validated* source the loop uses as its Φ — not a third independent
   formula. Ψ is unaffected by Seam A per the item-5 decision above (it's not meant to be
   "the" consciousness number the facade adopts; it's ethics/Broca's own signal).
7. [x] (S, low priority) Add one sentence to `jphi.md` and the phi-gated-safety paper
   clarifying that symtropy's "Φ" is a locally-defined heuristic input to the Master
   Consciousness Equation, not a measured IIT quantity — **DONE 2026-07-04, commit
   `b53ae1ea2e`.** `phi-gated-safety` already had an exemplary version of this caveat
   (right in its abstract); only `jphi.md` needed it.

## Phase 2 — Finish what was started (M, ~2-3 weeks)

- [ ] **Finish the Broca split** (it's half-done; `symthaea-broca-tools` already exists
      with 39 files): move the ~26 remaining toolkit modules + 5 toolkit bins out of
      `symthaea-broca`, rewrite `crate::` → `symthaea_broca::` (toolkit→core coupling is
      one-directional, ~8 files). One real blocker: sever
      `emotional_gating_integration.rs:6` → `compiler_trainer::CompilerVerdict` so the
      core gating trio (codegate / language_gates / emotional_gating_integration) stays
      core. Then prune now-dead heavy deps from broca's Cargo.toml (`rnix`, `rowan`,
      `tree-sitter-*`, `pqcrypto-*`, `sled`, `syn`).
- [ ] **The split already broke a real feature — fix before finishing the rest.** Found
      2026-07-04 by an independent audit pass, confirmed live: `symthaea-broca/src/
      liquid_mamba.rs` (the crate's flagship `LiquidMambaGenerator`, public, gated by
      `#[cfg(feature = "mamba-cpu")]`, `lib.rs:76-77,150`) still references 15 module
      paths that were physically moved to `symthaea-broca-tools` and never existed back
      in `symthaea-broca` since
      (`crate::morphological_bridge`, `foraging_bridge`, `sovereign_law`, `swarm_bridge`,
      `wasm_architect`, `memory_kernel`, `formal_bridge`, `sovereignty_bridge`,
      `geodesic_bridge`, `somatic_bridge`, `simulation_bridge`, `codebase_bridge`,
      `cognitive_ledger`, `compiler_feedback_bridge`, `substrate_rewriter` — 31 references
      total). `symthaea-broca`'s `Cargo.toml` has no dependency on `symthaea-broca-tools`
      at all, so there's no qualified-path fix available without adding one — this is the
      wrong-direction coupling (core→toolkit) the split was supposed to avoid, unlike the
      `emotional_gating_integration.rs` case above which is toolkit→core and fine.
      **Confirmed broken**: `cargo check -p symthaea-broca --features mamba-cpu` → 29
      compile errors. Root cause it slipped through: the split's own verification only
      ran `cargo check --features code-sheaf-eval`, never `mamba-cpu`/`liquid-mamba`, so
      the one feature it broke was never exercised. `mamba-cpu` gates 12+ real `[[bin]]`/
      `[[example]]` targets and the top-level `liquid-mamba` feature — not experimental
      scaffolding, an advertised generation backend per root CLAUDE.md.
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
      (`config/mod.rs:928-932`) while the crate defaults them OFF
      (`generator.rs:216,218`) — reconcile.
- [ ] **Broca hard-mask gating mode (added fourth pass)**: the `EpistemicCubeGate` is
      penalty/temperature-only — additive logit penalties (`gating.rs:1080,1103`) plus
      uncertainty-scaled temperature (1.3/1.5/1.8 for uncertain/unknown/OOD,
      `gating.rs:351-353`); there is **no `NEG_INFINITY` path anywhere in gating.rs**.
      Hard suppression exists in the codebase (`evaluation.rs:516-531`
      `suppress_collapse_forbidden_logits`, `controller.rs:935` top-k masking) but is not
      part of the epistemic gate. Add an opt-in hard-mask mode for E-axis extremes
      (Unknown/OOD certainty → assertion-token logits = `-inf`), reusing the
      `suppress_collapse_forbidden_logits` machinery. A gate that can only discourage is
      documentation, not a gate.
- [ ] **Confabulation benchmark (added fourth pass)**: no eval measures whether epistemic
      gating actually suppresses confident-assertion-under-ignorance. Build a small
      canonical set of prompts whose thought-HVs carry Unknown/OOD epistemic status,
      score assertion-rate in the generated text with gating off / soft (current default)
      / hard-mask, and add it beside `eval-canonical-v1.jsonl` in the quality-gate CI
      test above. This is the measurement that decides whether the soft gate was ever
      enough — do it before shipping any "epistemic honesty" claim about Broca output.
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

- [ ] **Reverse the CLS field regression** (exact count 2026-07-03: **131 fields**,
      `cognitive_loop/mod.rs:381-1029`, 51 `#[cfg]` attrs inside the struct body; back
      from post-refactor ~59 despite 42 manager files already existing across
      `cognitive_loop/` and `cognitive_loop/managers/`): extract `PredictiveCore`
      (~12 loose training/prediction fields), `ChannelHub` (~10 IO/channel fields), fold
      `ethics_engine` + verdicts into the existing `ethics_values` manager. Target ~80
      without behavior change.
- [ ] **Field-count ratchet in CI (added fourth pass)** — the refactor regressed once
      (59 → 131) because nothing enforces it; without a guard it will regress again.
      Add a cheap CI step (awk/grep count of fields in the `CognitiveLoopService` struct
      body, compared against a checked-in `MAX_CLS_FIELDS` number) that fails on increase.
      Ratchet the number down as the extraction above lands. Rule going forward: no new
      field on the service struct, ever — new state goes in a manager.
- [ ] **De-monolith**: move inline glue out of `cycle()` (`cycle.rs:38-728`, ~690 lines:
      integrity sweep, USI/spectrum→neuromod coupling, embodiment step, distress
      emission) into managers; split the monolithic phase files
      (`cycle_phase_dynamics/mod.rs` 176KB, `cycle_phase_feedback.rs` 111KB) by cohesion.
- [ ] **Byzantine island decision**: `byzantine_collective` + `causal_byzantine` +
      `meta_learning_byzantine` + `unified_intelligence` + `asset_evaluator`
      (~3,590 LOC) only reference each other and never activate under any default/profile
      bundle. Wire `multi_agent` into a real profile or archive the island.
- [ ] **More orphan candidates (added fourth pass, from a reference-count sweep of the
      55 top-level `src/` modules)**: `src/integration/` (`conscious_pipeline.rs`,
      `nix_integration.rs` — zero references anywhere, including tests and bins; the
      strongest candidate), `src/meta/` (zero external refs even with its
      `code_generation` feature on), `gui_bridge` (2 refs, both bins/tests),
      `benchmarks` (0 external refs), and thinly-referenced `resonant_speech` (7 refs).
      Same rule as the enable_* audit: wire it, or delete it — compiled-but-dark code
      inflates the claimed capability surface.
- [ ] **Feature-flag profile consolidation (added fourth pass)**: ~177 `[features]`
      entries in `symthaea/Cargo.toml`; CI samples 7 feature-group bundles + 22
      single-feature legs + 2 critical combos — effectively 0% of the combination space,
      and at least one flag (`neural-vocoder-gpu`) is marked "UNTESTED: not in CI
      matrix" in Cargo.toml itself. Define 4-5 **blessed profiles** (e.g. `minimal`,
      `default-mind`, `embodied`, `service`, `full-research`) as feature bundles, test
      those exhaustively in CI, document that anything outside a blessed profile is
      unsupported, and start folding single-purpose flags into the profiles. Goal: the
      supported configuration space becomes enumerable.
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
      input/memory adapter. High risk (131 fields, 51 cfg gates) — last.
      *Scoping intel (fourth pass, 2026-07-03)*: the blast radius is smaller than the
      struct sizes suggest. `Symthaea::process()` has only two real production consumers
      — the main CLI (`src/bin/symthaea.rs:887,1126`) and the phi-lab service
      (`research/phi-lab/src/bin/phi-lab-service.rs:336,508`); the web eval-api already
      talks to CLS directly (`src/api/demo_runner.rs`, `src/api/ws.rs:517`), and the
      REPL's `process()` is its own method. The two stacks share NO state — the only
      links are two mpsc channels (Mind→CLS swarm events via `swarm_event_tx`,
      CLS→Mind mesh outbound), and the existing `Symthaea::wire_swarm_channel()`
      (`src/symthaea/mod.rs:2578`) is called by **no production bin** — the dual-holder
      bins wire the raw sender manually. The real cost is deduplicating the parallel
      subsystems both stacks own independently (neuromodulator bath, memory
      coordinator/episodic memory, swarm/mesh handling — the memory consolidation item
      below is the same work). Suggested order: first make the facade *read* loop state
      (inject `CycleMetadata`/grounded facts into phases 3.5/4), then move perception
      into `cycle_with_hv()`, then delete `ContinuousMind`'s duplicated subsystems one
      at a time.
- [ ] **One Broca contract**: the facade Phase-5 translator is an LLM
      (`llm_organ::translate_thought`, `symthaea/mod.rs:1535`) while the loop uses the
      SSM `symthaea-broca` generator via `broca_bridge.rs` — two engines, two
      independent liquid-mamba stacks, no shared eval. Decide primary/humanizer roles
      and route both through one interface.
- [ ] **Consolidate memory**: three independent coordinators (`Symthaea`,
      `ContinuousMind`, CLS `memory_execution`) → one owner.

## Sequencing note (fourth pass)

Phases are severity-ordered, not strictly serial. Independent early wins that don't
wait on Phases 1-4: the Phase 5 systemd timer (scripts + lockfile already done), the
Phase 6 ladder freeze (protocol docs, no code), the Phase 3 field-count ratchet
(one CI step — land it *before* the extraction so the number only moves down), and
finishing Phase 1.5 rec #1 (the 3.0 weight fix, one-line). The long poles are Seam C
(Phase 4) and the CLS extraction (Phase 3); everything else can proceed around them.

## Phase 5 — Close the learning loop (M, added fourth pass)

Today all durable learning is offline and human-triggered: the Broca curriculum bridge
(cycle → gate → promote) is scripts a person runs. An architecture whose thesis is
continuous existence should own its consolidation cycle. Governed throughout by the
self-mod pipeline safety rules (autonomous-except-promotion, provenance sidecars, atomic
writes, lockfiles — see `memory/feedback_selfmod_pipeline_safety.md`).

- [ ] **Curriculum bridge Phase 4 — the timer**: `broca_curriculum_cycle.sh` already has
      a lockfile (`target/broca-curriculum-cycle.lock`) put there explicitly to make
      scheduled runs safe, and never touches the production checkpoint
      (`PROMOTION_READY.json` + manual `broca_promote_candidate.sh` is the human gate).
      No `.timer`/service unit exists yet (only `symthaea/systemd/symthaea.service`,
      unrelated). Write the NixOS systemd timer (weekly, per the design doc's
      "fast-follow" note) + a small status surface (last run, last gate verdict,
      pending-promotion flag). Promotion stays human — that boundary is a feature.
- [ ] **Loop-owned consolidation**: generalize the pattern that already exists for
      threat memory (dream consolidation in `ThreatMemory`) into a sleep/dream-scheduled
      episodic → semantic → training-corpus pipeline driven by the cognitive loop itself
      (biorhythm/cantor_dream managers are the natural owners), emitting curriculum
      objectives the Phase-4 timer consumes. End state: experience during waking cycles
      measurably changes next-week weights, with every step provenance-tracked and the
      promotion gate still human.
- [ ] **Prerequisite**: `enable_online_learning` is one of the ~14 built-but-disabled
      config switches in the Phase 3 enable_* audit — benchmark and decide it as part of
      that audit before layering the consolidation pipeline on top.

## Phase 6 — Honest capability ladder (S to start, ongoing; added fourth pass)

The ETHICS data-leakage incident (claimed 94.5%, real ~50%) is the cautionary tale:
internal benchmarks drift toward flattery. Pick a small set of **external,
leakage-proof** benchmarks and publish the honest curve per release — after the Phi
audit, the credibility of measured claims is the project's main asset.

- [ ] **Pick 3 ladder benchmarks** and freeze protocols in-tree (exact dataset hashes,
      splits, no training on eval): suggested — HumanEval pass@1 (coding agent; baseline
      already recorded: 9/40), a re-run held-out ETHICS protocol (post-leakage
      methodology, documented), and one perception/temporal benchmark that exercises the
      loop rather than the facade (Sleep-EDF or ARC-AGI 2-AFC, both already have
      harnesses in psych-bench/examples).
- [ ] **Loop-driven, not facade-scripted**: each ladder run must go through the live
      pipeline being claimed (`cycle_with_hv()` via psych-bench's `live_runner.rs` where
      applicable), not a bespoke eval path — otherwise the number describes the harness.
- [ ] **Track per release**: a `CAPABILITY_LADDER.md` table (date, commit, score,
      protocol hash) appended on each release/tag. Regressions are findings, not
      embarrassments — the honest curve is the point.

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
