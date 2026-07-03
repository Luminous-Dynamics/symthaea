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

(Non-issue, resolved during review: the "plaintext crates.io token" in
`symthaea/CLAUDE.md:10` was the BWS secret *ID*, not a token; wording fixed 2026-07-03.
CLAUDE.md is not in the standalone sync file list — nothing leaked.)

## Phase 0 — Broken things & dead weight (S effort each, ~1-2 days total)

- [ ] **Ahimsa dead-path**: derive `ahimsa_violated` in `cycle.rs:315` from a real ethics
      signal instead of the hardcoded `false`.
- [ ] **CI: remove the always-failing matrix leg** — `feature-interactions` line ~643
      tests `consciousness_full,school_learning` but `consciousness_full` was removed
      from Cargo.toml (feature no longer exists). `fail-fast:false` hides it as one
      permanently red cell.
- [ ] **CI: pin `sbom` job** to `@1.95.0` (currently `@stable`, ci.yml:827); drop the
      dangling workspace exclude `crates/bridges/symthaea-mycelix-holochain` (directory
      no longer exists); drop the redundant `test-feature-matrix-critical: core` leg
      (identical to `test`); lower `test-feature-matrix` timeout 420 → ~90 min.
- [ ] **Broca half-split breakage**: `code-sheaf-eval` feature does not compile
      (`evaluation.rs:16` imports `crate::code_analysis`, which moved to
      `symthaea-broca-tools`) — kills 33 tests + `broca-exercism-bench` silently.
      Also broken bins with refs to moved modules: `bin/fused_cognitive_node.rs:10-11`
      (`invariant_guard`, `memory_ring`), `bin/broca_dreamer.rs:16` (`foraging_bridge`).
- [ ] **Delete orphans**: `src/proof_state.rs` (560 LOC) and
      `src/curriculum_generator.rs` (731 LOC) have zero references repo-wide.
- [ ] **Actuator-count table divergence**: `symtropy-robotics-bridge-core/src/platform.rs:47`
      says Humanoid=64; the actual crate reports 21 (Dmc21). Single-source or add a
      cross-crate assertion test.

## Phase 1 — Safety & truth on the product path (M, ~1-2 weeks)

- [ ] **Seam B — ethics-gate `Symthaea::process()`**: give the facade an `EthicsEngine`
      (reuse `ethics_engine.rs`) and gate `ProcessResponse` before return. ~3-4 files,
      300-500 LOC.
- [ ] **Seam A — one canonical Phi**: extract `ConsciousnessCore::phi()` and make
      `ContinuousMind::update_consciousness` (`src/mind/tick.rs:323`, currently a naive
      mean-pairwise-dissimilarity with a hardcoded 0.1 empty-memory fallback) delegate to
      it. The facade's Phi and the loop's Phi are currently incomparable numbers.
- [ ] **Decide `enable_moral_anomaly_response`**: benchmark ON, then default ON or delete
      the subsystem. Same for `enable_validation_overlay` (`config/mod.rs:904`).
- [ ] **Close the SafeFallback gap**: implement for manipulator, quadruped, surgical,
      orbital, auv, exoskeleton per the behaviors the core doc already specifies — or
      amend the doc table to stop promising them. Also unify orbital's 0.3 hard cliff
      (`embodiment.rs:71`) and surgical's parallel `SurgicalSafetyLevel` ladder under the
      `MotorSafetyLevel` contract so per-tier enforcement is auditable.

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
