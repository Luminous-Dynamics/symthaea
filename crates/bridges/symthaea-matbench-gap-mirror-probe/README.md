# symthaea-matbench-gap-mirror-probe

Exploratory, network-free overlap analysis for third-party CSV mirrors of Matbench `matbench_expt_gap`.

## Why this exists

The official adapter in `symthaea-matbench-gap` requires the exact compressed Matbench artifact SHA-256 before it can produce Benchmark Zero truth. When that artifact is temporarily unavailable, a third-party mirror can still answer a narrower planning question: **does Symthaea's exposed band-gap training table appear to overlap the benchmark composition universe?**

That question is useful for designing a leakage-clean holdout, but it is not benchmark evidence.

## Known mirror

The convenience entry point records the currently observed GitHub locator:

- repository: `Zhang-NJ-Lab/Datasets`
- path: `matbench_expt_gap.csv`
- observed Git blob SHA-1: `35943edef66ae36412d3f184c71869941eb57087`
- expected data rows: `4604`

The Git blob SHA-1 is **reported locator metadata**, not recomputed by this crate. The receipt derives its own SHA-256 directly from the supplied CSV bytes.

## Formula grammar

The CSV mirror contains formulas that cannot be parsed by Symthaea's current integer-only process-discovery formula parser, including decimal stoichiometries such as `Ag0.5Ge1Pb1.75S4`.

This crate therefore owns a deliberately quarantined mirror grammar supporting:

- standard element symbols;
- positive integer or decimal amounts;
- parenthesized groups with integer or decimal multipliers;
- nested parenthesized groups.

It is not a replacement for the production chemistry parsers.

## Leakage identity

Mirror and Symthaea training compositions are normalized to atomic fractions and quantized at `1e-9`, matching the conservative composition identity used by `symthaea-matbench-gap`.

Polymorph labels therefore collapse intentionally. For example, mirror `SiC` overlaps the exposed `SiC-3C`, `SiC-4H`, and `SiC-6H` training entries.

The receipt reports both:

- overlapping mirror rows; and
- overlapping unique normalized compositions.

Repeated mirror rows therefore do not become independent leakage claims.

## Network-free CLI

```text
cargo run -p symthaea-matbench-gap-mirror-probe --bin matbench-gap-mirror-probe -- matbench_expt_gap.csv
```

The JSON output contains a domain-separated receipt SHA-256 and the full overlap receipt. No host-local input path is included in the receipt.

## Authority boundary

A successful receipt proves only that the **supplied CSV bytes**:

1. matched the expected simple CSV shape and row count;
2. had contiguous canonical indices;
3. contained parseable finite non-negative gap values and supported formulas;
4. were content-addressed locally by SHA-256; and
5. were compared deterministically with Symthaea's currently exposed training compositions.

It does **not** establish:

- official Matbench artifact identity or official compressed SHA-256;
- equality with Matminer's canonical JSON row order;
- dataset licensing;
- clean holdout status;
- absence of historical/manual prior knowledge;
- Benchmark Zero truth authority;
- model quality or scientific validation.

Only the official `symthaea-matbench-gap` SHA-gated path may produce `BandgapTruthSet` for this benchmark line.
