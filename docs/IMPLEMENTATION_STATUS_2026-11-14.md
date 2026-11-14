# Luminous Atelier implementation status — 2026-11-14

## Public research work

- Symthaea #12093: Personal Garment Studio, user-controlled garment design, presets, hard constraints, objectives and agency.
- Symthaea #12094: wardrobe-first optimization, repair, alteration and no-new-item baseline.
- Symthaea #11731 and draft PR #11732: cross-system apparel engineering digital thread and benchmark corpus.
- Symthaea #11753: human-AI fashion co-design and designer agency.
- Symthaea #11746: inverse textile design and experiment selection.
- Symthaea #11743/#11747/#11750: textile sorting, design-for-disassembly and semantic apparel interoperability.
- Symtropy #456: specialized cloth/deformable physics profile.
- Symtropy #4865: cloth robotics and seam execution.
- Symtropy #4870: RGBench reproduction.
- Symtropy PRs #1679/#1699/#1716/#1724: textile process, Pareto, scheduling and empirical-trial draft work.
- Sol-Atlas #738: read-only apparel spatial projection.
- Mycelix #6092: evidence-qualified textile product-passport projection.

## User-facing preset catalogue

The public research branch contains 17 JSON presets: five functional/hybrid starting points plus 12 style directions. It also contains the design-preset and garment-design-brief JSON Schemas and a synthetic overshirt brief.

The new Rust preset composer in `tools/atelier-brief-core/src/preset_stack.rs` supports style/function layering, preserves source ID/revision/digest, rejects hard constraints injected by presets, and returns conflicts for user resolution rather than silently using last-write-wins or summing objective weights. All presets remain editable starting points, not claims of fit, material performance, environmental advantage, protective function or universal desirability. Heritage references require user-supplied or rights-cleared source material; modest and adaptive preferences remain user-defined.

## Rust implementation

Three Rust source components are present in two isolated, dependency-free crates:

- `tools/atelier-brief-core/`: validates decoded design-brief identity, revision and digest shape; IDs; objective bounds; selected hard-constraint contradictions; suggestion decision provenance; and fail-closed handoff status.
- `tools/atelier-brief-core/src/preset_stack.rs`: conflict-aware preset composition with explicit user override semantics and 10 authored tests.
- `tools/atelier-wardrobe-core/`: exact subset-search oracle up to 18 optional choices, protected-item preservation, explicit no-additions/made-to-order policies, common-currency cost arithmetic (mixed currencies fail), and a Pareto frontier over acquisition, new manufacture, cost, effort and choice count.

The three test modules author 33 unit tests in total: 11 brief-core, 10 preset-stack and 12 wardrobe-core. They do not parse JSON, generate patterns, evaluate aesthetics, simulate material behavior, verify external evidence, reimplement Symtropy's Pareto engine, or authorize production.

## Validation and CI

The local kit JSON validation receipt records 20 authored JSON inputs parsed, two Draft 2020-12 schemas meta-validated, 17 presets validated against the preset schema, and one synthetic garment brief validated. A public-branch fetch/parse check also confirmed all 17 preset files, both schemas and the brief example parse as JSON.

The Rust unit tests have not been compiled or executed locally because the authoring environment has no `rustc`, `cargo`, `rustfmt` or Clippy. The exact-head GitHub Actions workflow is configured to run the existing apparel fixture harness plus both Atelier crates on Rust 1.96.0. A current-head pass has not been observed and is not claimed; the latest focused run checked at the previous head was queued.

## Git status

Work is on the existing public research branch `research/apparel-engineering-thread-v0-1` and draft PR #11732. No default branch was modified or merged. Review and exact-head qualification remain open.
