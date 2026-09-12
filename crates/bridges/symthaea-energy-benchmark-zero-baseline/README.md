# symthaea-energy-benchmark-zero-baseline

Leakage-qualified, composition-only baseline runner for Energy Discovery Benchmark Zero.

This crate is intentionally narrow. It measures the existing Symthaea electronegativity band-gap baseline on the leakage-clean positional Matbench truth slices from `symthaea-matbench-folds`. It does not train or modify a model.

## Why this baseline comes before the current RF

`BandgapPredictor` uses an 18-feature vector that includes crystal-system ordinal. Matbench `matbench_expt_gap` supplies composition, not crystal system. Supplying `CrystalSystem::Unknown` for every candidate would therefore be an additional modeling intervention rather than a neutral missing-value operation.

The existing electronegativity baseline accepts composition alone, so it is the first method that can be measured on this truth source without inventing structural information.

A later structure-aware benchmark should use a truth/candidate source that actually supplies qualified structure or crystal-system information.

## Truth separation

The pure ranking API is:

`build_baseline_screening_run(candidates, target)`

Its candidate type contains only:

- candidate id;
- normalized composition.

There is no experimental gap field or truth argument.

The outer `run_baseline_benchmark_from_official_bytes(...)` path:

1. verifies the exact pinned Matbench source artifact;
2. obtains one leakage-qualified positional fold;
3. reconstructs only the retained candidate ids/compositions;
4. freezes the composition-only ranking;
5. only then passes the frozen ranking and experimental truth to the existing measurement-only Benchmark Zero evaluator.

## Ranking rule

For a caller-supplied target interval, candidates are ordered by:

1. minimum predicted distance to the target interval;
2. minimum predicted distance to the target midpoint;
3. lexical candidate id as a deterministic final tie-break.

The crate supplies no default photovoltaic target window. The caller must choose the target explicitly.

## Model identity and prior-knowledge disclosure

Method id:

`symthaea-bandgap/electronegativity-composition-baseline`

The method version binds the exact Git blob of:

`crates/domains/symthaea-bandgap/src/bandgap_baseline.rs`

used by this candidate branch.

The benchmark method declares no exact machine-readable training slice, because the baseline is not fitted at runtime from a machine-readable corpus. That does **not** mean the model is historically blind: its source comments state that coefficients were chosen/fitted to roughly reproduce familiar semiconductor behavior.

This prior-knowledge limitation remains explicit in every combined receipt.

## Uncertainty

The baseline emits `uncertainty_ev = None` for every candidate. It does not reinterpret model error, heuristic distance, tree dispersion, or any other quantity as calibrated uncertainty.

## Combined evidence receipt

A successful measurement returns:

- leakage-qualification SHA-256;
- exact Benchmark Zero ranking digest;
- Benchmark Zero receipt BLAKE3;
- exact baseline method id/version/source blob;
- inherited fold/index/leakage disclosures;
- one combined SHA-256 binding qualification + method + benchmark receipt;
- measurement-only benchmark metrics.

The combined identity does not upgrade the evidence class. It is a reproducible measurement receipt only.

## CLI

The companion binary requires every policy choice explicitly:

`energy-benchmark-zero-baseline <matbench_expt_gap.json.gz> <fold:0..4> <target-min-eV> <target-max-eV> <top-k>`

It performs no network fetch and omits the local artifact path from emitted JSON so identical source bytes produce machine-independent receipts.

## Deliberate non-claims

This crate does not establish:

- an official Matbench leaderboard score;
- historical independence from semiconductor knowledge;
- calibrated predictive uncertainty;
- a structure-aware model result;
- material novelty;
- synthesizability;
- device performance;
- experimental validation;
- deployment superiority;
- material certification or physical authority.

The stacked workspace remains unqualified until its lockfile is reconciled under the pinned toolchain and exact-head check/test/strict-Clippy execute successfully.
