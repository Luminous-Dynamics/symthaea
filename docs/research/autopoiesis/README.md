# Autopoiesis and Organizational Autonomy Research

This directory contains AOBench's benchmark contract, operational model, runtime evidence implementation, finite formal-model track and experimental programme.

## Start here

1. [AOBench specification](AOBENCH_SPEC_2026-11-13.md) — benchmark tracks, metrics, baselines, protocol and staged implementation.
2. [Organizational closure model](ORGANIZATIONAL_CLOSURE_MODEL_2026-11-13.md) — operational definitions, system boundaries, runtime evidence, causal interventions and claim limits.
3. [Runtime evidence layer](RUNTIME_EVIDENCE_LAYER_2026-11-14.md) — first Rust implementation, API contract, verification scope and outstanding trust boundaries.
4. [Assay adequacy and model-relative identifiability](EXPERIMENTAL_ASSAY_ADEQUACY_2026-11-14.md) — deterministic finite-model distinguishability and its current limits.
5. [Finite viability-localized model](FINITE_VIABILITY_MODEL_2026-11-14.md) — AP1–AP4-style finite witnesses and local-to-global obstruction experiments.
6. [Boundary, substrate and authority research PR #12032](https://github.com/Luminous-Dynamics/symthaea/pull/12032) and [tracking issue #12031](https://github.com/Luminous-Dynamics/symthaea/issues/12031) — separates productive organization, boundary enforcement, exogenous resource flows and delegated authority.
7. [Stochastic assay qualification issue #12048](https://github.com/Luminous-Dynamics/symthaea/issues/12048) — repeated response distributions, effect thresholds, uncertainty and multiplicity control.
8. [AOBENCH-004 synthetic-organism adapter](https://github.com/Luminous-Dynamics/symthaea/issues/12023) — project the existing synthetic-organism viability environment into AOBench without duplicating its simulator.
9. [AOBENCH-001 tracking issue](https://github.com/Luminous-Dynamics/symthaea/issues/11980).
10. [MEL-AUTO-001 runtime regeneration qualification](https://github.com/Luminous-Dynamics/symthaea/issues/10098).
11. [AOBENCH-006 intervention validity](https://github.com/Luminous-Dynamics/symthaea/issues/12067) — delivery, isolation and causal attribution.
12. [AOBENCH-007 organizational reconfiguration](https://github.com/Luminous-Dynamics/symthaea/issues/12113) — structural evolution and capability preservation.
13. [Operational continuity specification](AOBENCH_OPERATIONAL_CONTINUITY_2026-11-14.md) — simulation/bisimulation and separate capability, governance and evidence continuity.
14. [AOBENCH-008 tracking issue](https://github.com/Luminous-Dynamics/symthaea/issues/12121) — operational identity and behavioral continuity across reconfiguration.
15. [Mutation adequacy and cross-plane adversarial qualification](AOBENCH_MUTATION_ADEQUACY_2026-11-14.md) — mutation-tested assurance and paired positive twins; tracked by [AOBENCH-009 #12140](https://github.com/Luminous-Dynamics/symthaea/issues/12140).
16. [Typed semantic continuity](AOBENCH_SEMANTIC_CONTINUITY_2026-11-14.md) — explicit authority, consent, side-effect, provenance and substrate transition envelopes; tracked by [AOBENCH-010 #12167](https://github.com/Luminous-Dynamics/symthaea/issues/12167).
17. [Experimental design and cross-family generalization](AOBENCH_EXPERIMENTAL_DESIGN_AND_GENERALIZATION_2026-11-14.md) — sequential probe choice, power, stopping, classical baselines and sealed mechanism-family holdouts; tracked by [AOBENCH-011 #12173](https://github.com/Luminous-Dynamics/symthaea/issues/12173).
18. [Sequential experiment design evaluator](AOBENCH_SEQUENTIAL_EXPERIMENT_EVALUATOR_2026-11-14.md) — adaptive probe selection, information gain, delivery/isolation evidence, budgeted stopping and oracle-efficiency regret.
19. [Model-class adequacy and held-out refinement](AOBENCH_MODEL_ADEQUACY_AND_REFINEMENT_2026-11-14.md) — detect when every current hypothesis is falsified and require new models to predict sealed probes; tracked by [AOBENCH-012 #12200](https://github.com/Luminous-Dynamics/symthaea/issues/12200).
20. [Decision-focused partial identification and max-potency experiments](AOBENCH_PARTIAL_IDENTIFICATION_AND_MAX_POTENCY_2026-11-14.md) — select interventions that guarantee causal-effect bound contraction and authorize robust decisions without requiring full graph identification; tracked by [AOBENCH-013 #12206](https://github.com/Luminous-Dynamics/symthaea/issues/12206).
21. [Evidence-backed residual gains beyond algorithm portfolios](AOBENCH_ALGORITHMIC_RESIDUAL_GAINS_2026-11-15.md) — measure auditable agent edge corrections beyond a calibration-frozen causal algorithm, with evaluator-only graph truth and receipt-qualified edits; tracked by [AOBENCH-016 #12238](https://github.com/Luminous-Dynamics/symthaea/issues/12238). The first Rust evaluator authenticates the baseline-selection receipt and checks separate calibration/evaluation split digests.
22. [Anytime-valid causal inference under adaptive stopping](AOBENCH_ANYTIME_VALID_CAUSAL_INFERENCE_2026-11-15.md) — time-uniform bounded-outcome effect intervals, family-wise alpha allocation across queries and evidence-qualified randomized observations; tracked by [AOBENCH-017 #12242](https://github.com/Luminous-Dynamics/symthaea/issues/12242). The v0 Rust evaluator uses a Hoeffding alpha-spending confidence sequence and requires a reported interval to contain the evaluator-recomputed sequence at the chosen stopping prefix.

## Current implementation

The first executable package is a standalone nested Cargo workspace:

- `crates/domains/symthaea-organizational-closure`
  - `src/lib.rs`: runtime component/digest/lineage verifier.
  - `src/assay.rs`: deterministic finite-model assay adequacy evaluator.
  - `src/viability.rs`: bounded finite viability, boundary-production, admissibility and global-constraint witnesses.
  - `src/reconfiguration.rs`: allowlist- and budget-constrained structural diff admission with protected-root/constraint checks. This admits a proposal structurally; it does not authorize, prove viability or establish behavioral continuity.
  - `src/identity.rs`: finite simulation/bisimulation-style operational-continuity checker over declared state, capability, invariant and action/observation models.
  - `src/semantic_continuity.rs`: typed authority, consent, effect, provenance and substrate continuity evaluator; evidence dispositions are supplied by independent adapters and are not authenticated by this module.
  - `src/experiment_design.rs`: sequential deterministic experiment evaluator with evaluator-only hidden truth, target-class information gain, intervention/cost/risk budgets and stop/abstain qualification; no stochastic power claim.
  - `src/model_refinement.rs`: model-class adequacy and held-out hypothesis refinement evaluator. A proposed model must fit the complete qualified discovery trace and sealed validation probes against an evaluator-only oracle.
  - `src/algorithmic_residual.rs`: evidence-backed residual graph-edit evaluator. It authenticates the frozen baseline-selection receipt and split digests, checks every edit against evaluator-only receipt adjudication, and reports residual gain, algorithm-suite ceiling and novel true-edge recovery.
  - `src/anytime_valid.rs`: bounded-outcome Hoeffding confidence sequence with summable alpha spending, family-wise query/arm allocation, adaptive stopping-prefix checks and independent randomization/delivery/isolation/measurement/fixed-mean evidence gates.
  - `src/delayed_outcomes.rs`: design-based arm-wise IPW normal-mixture confidence sequences for staggered entry and event-time outcomes, with family-wise query/arm alpha allocation, verified follow-up watermarks, separate event/report time, and evaluator-only potential-outcome coverage audit.
  - `src/anytime_valid.rs`: bounded-outcome Hoeffding confidence sequence with summable alpha spending, family-wise query/arm allocation, adaptive stopping-prefix checks and separate randomization/delivery/isolation/measurement/fixed-mean evidence gates.

It has exact serde / sha2 dependency pins and its own lockfile, and is explicitly excluded from the root workspace's crates/domains glob. Its focused CI builds the nested workspace independently.

The runtime verifier recomputes SHA-256 digests from stored artifact bytes and verifies process contracts, executable-identity consistency, exact component-version lineage, required/optional external observations, and producer/consumer coverage for required regenerative components. It qualifies bounded functional self-maintenance only.

The deterministic assay module compares response digests over a frozen finite model class. It does not yet provide stochastic response distributions, confidence intervals, power analysis or multiplicity correction; these are tracked in #12048.

The finite viability module checks explicit finite witnesses inspired by the 2026 viability-localized formalization: realization on viability classes, boundary production, admissibility and existence of a global assignment. It is not a categorical implementation of arbitrary topoi and does not establish biological autopoiesis. Its result is separate from runtime execution evidence and assay adequacy.

## Evidence status

- The runtime verifier previously passed exact-package format, locked all-target compile, 10 deterministic tests and strict Clippy on the exact commit recorded in [PR #12010](https://github.com/Luminous-Dynamics/symthaea/pull/12010).
- PR #12010's newer tip had a queued AOBench Runtime Qualification workflow when last checked; previous green results must not be attributed to that newer SHA.
- The finite viability module and deterministic controls are committed on the AOBench finite-model branch and opened for review in [draft PR #12047](https://github.com/Luminous-Dynamics/symthaea/pull/12047).
- The current finite-model PR head's focused AOBench workflow is queued; broad CI and governance runs were skipped. No new compiler pass is claimed.
- This environment has no Rust compiler, so local format/check/test/Clippy execution was unavailable.
- Benchmark targets remain proposals until pilot calibration, statistical qualification and confirmatory measurement.

## Contribution rule

Keep theoretical claims, executable behavior, finite formal-model witnesses, structural checks, runtime evidence, benchmark measurements and external replication distinct. Null and adverse outcomes are publishable results.
