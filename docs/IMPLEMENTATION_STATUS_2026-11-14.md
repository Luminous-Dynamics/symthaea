# Luminous Atelier implementation status — 2026-11-14

## Public research work

- Symthaea issue #12093: Personal Garment Studio, user-controlled garment design, presets, hard constraints, objectives and agency.
- Symthaea issue #12094: wardrobe-first optimization, repair, alteration and no-new-item baseline.
- Symthaea issue #11731 and PR #11732: cross-system apparel engineering digital thread and benchmark corpus.
- Symthaea issue #11753: human-AI fashion co-design and designer agency.
- Symthaea issue #11746: inverse textile design and experiment selection.
- Symtropy issue #456: specialized cloth/deformable physics profile.
- Symtropy issue #4865: cloth robotics and seam execution.
- Symtropy issue #4870: RGBench reproduction.
- Symtropy PRs #1679/#1699/#1716/#1724: textile process, Pareto, scheduling and empirical-trial draft work.
- Sol-Atlas issue #738: read-only apparel spatial projection.
- Mycelix issue #6092: evidence-qualified textile product-passport projection.

## User-facing preset catalogue

The starter library contains 17 JSON presets: five functional/hybrid starting points plus 12 style directions. Each preset is an editable starting configuration and not a claim of fit, material performance, environmental advantage, protective function or universal desirability. Heritage references require user-supplied or rights-cleared source material; modest and adaptive preferences remain user-defined.

## Rust implementation

Two isolated, dependency-free Rust crates have been added to the Symthaea public research branch:

- `tools/atelier-brief-core/`: validates decoded design-brief identity, revision and digest shape; IDs; objective bounds; selected hard-constraint contradictions; suggestion decision provenance; and fail-closed handoff status.
- `tools/atelier-wardrobe-core/`: exact subset-search oracle up to 18 optional choices, protected-item preservation, explicit no-additions/made-to-order policies, and single-currency cost arithmetic (mixed currencies fail), and a Pareto frontier over acquisition, new manufacture, cost, effort and choice count.

The crates author 23 unit tests in total (11 brief-core and 12 wardrobe-core). They do not parse JSON, generate patterns, evaluate aesthetics, simulate material behavior, verify external evidence, reimplement Symtropy's Pareto engine, or authorize production.

## Validation and CI

The local kit has a fresh JSON-only validation receipt: 20 authored JSON inputs parsed, two Draft 2020-12 schemas meta-validated, 17 presets validated against the preset schema, and one synthetic garment brief validated. The receipt is packaged in the local starter kit.

The Rust unit tests have not been compiled or executed locally because the authoring environment has no `rustc`, `cargo`, `rustfmt` or Clippy. The exact-head GitHub Actions workflow was expanded to run the existing apparel fixture harness plus both Atelier crates on Rust 1.96.0. A current-head pass has not been observed and is not claimed.

## Git status

Work is on the existing public research branch `research/apparel-engineering-thread-v0-1` and draft PR #11732. No default branch was modified or merged. Review and exact-head qualification remain open.
