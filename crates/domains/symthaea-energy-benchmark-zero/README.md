# symthaea-energy-benchmark-zero

Measurement-only protocol for the first energy-discovery benchmark.

The protocol is deliberately separated from any particular model or dataset. Its job is to make a benchmark receipt impossible to mint without explicit truth-data provenance, exact split identity, a content digest, method provenance, an exact ordered screening-output digest, and a declared top-k screening target.

## Leakage rule

A benchmark truth slice cannot also appear in the screening method's declared training slices. Exact training/truth overlap fails closed before metrics are computed.

This does not prove that a caller has disclosed every source of prior knowledge, nor does it prove that differently named slices share no rows. It does remove the most common accidental failure mode: evaluating against the same declared data slice used to train or tune the method. Stronger row/content overlap checks belong in dataset-specific adapters.

## Benchmark shape

For a supplied band-gap target window and ranked screening output, the protocol computes:

- top-k precision;
- top-k recall;
- full-ranking mean absolute prediction error;
- best-selected target error;
- globally best achievable target error in the truth set;
- target regret.

The exact ordered ranking is hashed separately with a domain-separated BLAKE3 identity and that digest is bound into the benchmark receipt. Two rankings that happen to produce identical summary metrics therefore remain different benchmark artifacts.

The target window is caller-supplied. This crate does not embed a claim about the universally optimal photovoltaic band gap.

## Why no result is committed yet

The existing `symthaea-bandgap` curated table is also used to train its current predictor, and its physics baseline was fitted against common semiconductor data. Using that same table as Benchmark Zero truth would not establish external generalization.

The intended first qualifying truth source is an exact, content-addressed external experimental dataset/split such as Matbench `matbench_expt_gap`. Dataset acquisition and parsing belong in a follow-up adapter so this core protocol stays deterministic and network-free.

## Authority boundary

A benchmark receipt is a measurement record only. It does not certify a material, authorize an experiment, select a deployment winner, or prove scientific novelty.
