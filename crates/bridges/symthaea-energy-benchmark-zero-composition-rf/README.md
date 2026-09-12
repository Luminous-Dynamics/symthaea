# symthaea-energy-benchmark-zero-composition-rf

Leakage-qualified adapter for evaluating Symthaea's crystal-ablated composition-only band-gap random forest under Energy Discovery Benchmark Zero.

This crate is downstream of the pinned Matbench truth parser and leakage-clean fold qualification. It does not fetch benchmark data and it does not fit on Matbench labels.

## Model boundary

The evaluated model is `CompositionBandgapPredictor` from `symthaea-bandgap`.

Its crystal feature is deliberately ablated by supplying the same `CrystalSystem::Unknown` sentinel for every curated training row and every inference row. Because the feature is constant, it carries no discriminative crystal-system information.

The model still trains on Symthaea's curated experimental band-gap table. It is therefore not historically blind or independent of public semiconductor knowledge.

## Exact training identity

Every benchmark method version binds:

- the exact `ml_bandgap.rs` source blob containing the composition-only predictor;
- the complete content-addressed curated training-table SHA-256;
- the constant crystal-ablation policy.

The training table is also declared through `ScreeningMethodProvenance.training_slices` rather than hidden behind a generic model label.

The upstream fold bridge independently content-addresses the same training table and removes normalized-composition overlaps from Matbench evaluation truth. Runtime evaluation fails if those training-table identities differ.

## Truth separation

The pure ranking function accepts only:

- candidate ID;
- composition;
- explicit target interval;
- training-table identity.

It receives no Matbench experimental gap values.

The exact-artifact runner first obtains the leakage-qualified fold, reconstructs only retained candidate identities/compositions, freezes the RF ranking, and only then passes that ranking and truth to the measurement-only Benchmark Zero evaluator.

## Ranking rule

To make comparisons fair, ranking semantics intentionally match the fixed physics baseline:

1. minimum predicted distance outside the target interval;
2. minimum predicted distance to the target midpoint;
3. lexical candidate ID as deterministic final tie-break.

No default photovoltaic target interval is embedded here.

## Uncertainty boundary

The underlying random forest exposes inter-tree prediction spread. This crate does **not** call that quantity calibrated predictive uncertainty.

Every `ScreeningRecord` therefore sets:

`uncertainty_ev = None`

until a separate calibration theorem establishes a justified interpretation.

## Receipt

A successful measurement binds:

- fold identity;
- exact learned-model ID/version/source blob;
- exact training-table SHA-256 and entry count;
- crystal-ablation policy;
- leakage-qualification SHA-256;
- exact ordered-ranking identity from Benchmark Zero;
- Benchmark Zero receipt BLAKE3;
- one combined SHA-256.

The receipt remains measurement-only evidence. It grants no experimental, material-certification, novelty, manufacturing, procurement, deployment, or physical authority.

## CLI

Run one leakage-qualified fold from a local pinned Matbench artifact:

`energy-benchmark-zero-composition-rf <matbench_expt_gap.json.gz> <fold:0..4> <target-min-eV> <target-max-eV> <top-k>`

The CLI performs no network fetch and omits the host-local artifact path from its JSON output.

## Deliberate non-claims

This crate does not establish:

- calibrated predictive uncertainty;
- historical independence from public semiconductor literature;
- official Matbench leaderboard comparability;
- material novelty or synthesizability;
- device performance;
- experimental validation;
- deployment superiority;
- material certification or physical authority.

The stacked branch remains unqualified until the inherited workspace lock is resolved by Cargo under the pinned toolchain and exact-head check/test/strict-Clippy execute successfully.
