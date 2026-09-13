# symthaea-matbench-gap-baseline-screening

Truth-blind baseline screening for the frozen Matbench composition universe produced by `symthaea-matbench-gap-exposure-plan`.

## Why this exists

The existing learned `BandgapPredictor` is not directly appropriate for `matbench_expt_gap`: it consumes crystal system as one of its Random Forest features, while Matbench supplies composition only. Passing `CrystalSystem::Unknown` only at inference would create a train/inference mismatch and would not establish a clean composition-only predictor.

This crate therefore does **not** use that learned predictor. It establishes the first honest Benchmark Zero screening controls from the information that is actually available before truth evaluation.

## Input boundary

The crate consumes only an `ExposedTrainingExclusionPlan`, which contains the truth-free source-row composition partition from #2653. It never accepts `BandgapTruthSet` and never reads an experimental gap value.

Before screening, the plan must:

1. validate structurally; and
2. match the exact currently exposed Symthaea band-gap training snapshot.

The screening receipt binds the plan's composition-universe, partition, training-snapshot, and retained-universe digests.

The full source-plan digest is retained separately as provenance. It is deliberately **not** part of the truth-free screening-subject digest because the full plan also binds exact official-artifact identities that can change when benchmark truth values change.

## Baseline prediction

Both V0 policies use the existing composition-only `electronegativity_bandgap` baseline prediction.

This is a legacy empirical/physics-inspired heuristic, **not a no-prior-knowledge model**. The implementation comments state that its parameters were chosen/fitted against common semiconductor behavior. The receipt therefore carries a fixed disclosure that an empty machine-readable `training_slices` set does not mean the method had no historical empirical tuning.

## Two baseline policies

### `TargetDistance`

Ranks every retained composition by absolute distance between the baseline-predicted gap and the preregistered target-window midpoint. Ties are broken by canonical candidate id.

This is the first target-aware composition-only heuristic baseline.

### `DeterministicRandomOrder { seed }`

Uses the **same baseline predictions** but orders candidates by a domain-separated SHA-256 key of `(seed, candidate_id)`.

This gives Benchmark Zero a null ranking control while holding the prediction surface constant. Differences in top-k selection/regret between the two policies therefore reflect ordering policy rather than a different predictor.

No runtime RNG is used.

## Separate identities

The receipt keeps distinct:

- full source-plan SHA-256 — provenance only;
- source composition-order SHA-256;
- source partition SHA-256;
- exact Symthaea training-snapshot SHA-256;
- retained screening-universe SHA-256;
- policy + target;
- truth-free screening-subject SHA-256;
- exact ordered `ScreeningRun` ranking digest;
- complete receipt SHA-256.

This avoids making truth-bearing source identity part of the screening subject while still preserving traceability to the exact frozen plan.

## Replay

`verify_baseline_screening_receipt(plan, receipt)` revalidates the current training snapshot, checks all bound plan identities, reruns the selected policy, and requires exact receipt equality.

Structural receipt validation and replay remain separate theorems.

## Learned-model boundary

A later learned benchmark model should be trained in an explicitly composition-only mode, with crystal-system information masked or removed for **both training and inference**. The current RF predictor must not be silently reused by supplying `Unknown` only at inference.

## Authority boundary

A successful baseline screening receipt proves deterministic truth-blind ranking under the recorded plan, target and policy. It does not prove:

- a globally clean holdout;
- absence of historical/manual prior knowledge;
- predictive quality;
- statistical significance;
- scientific novelty;
- material feasibility;
- candidate promotion or experiment authority.

Exact-head compile/test/rustfmt/strict-Clippy evidence is still required before qualification.
