# Luminous Atelier implementation status — 2026-11-14

## Public research work

- Symthaea #12093: Personal Garment Studio, user-controlled garment design, presets, hard constraints, objectives and agency.
- Symthaea #12094: wardrobe-first optimization, repair, alteration and independent acquisition policies.
- Symthaea #12134: SewingPlan Compiler — typed, verifiable garment assembly-operation graphs.
- Symthaea #11731 and draft PR #11732: cross-system apparel engineering digital thread and benchmark corpus.
- Symthaea #11753: human-AI fashion co-design and designer agency.
- Symthaea #11746: inverse textile design and experiment selection.
- Symthaea #11743/#11747/#11750: textile sorting, design-for-disassembly and semantic apparel interoperability.
- Symtropy #456: specialized cloth/deformable physics profile.
- Symtropy #4865: cloth robotics and seam execution.
- Symtropy #4870: RGBench reproduction.
- Symtropy #4875/#4878: multi-stage factory scheduling and Knitout capability preflight.
- Symtropy PRs #1679/#1699/#1716/#1724: textile process, Pareto, scheduling and empirical-trial draft work.
- Sol-Atlas #738: read-only apparel spatial projection.
- Mycelix #6092: evidence-qualified textile product-passport projection.

## User-facing preset catalogue

The public research branch contains 17 JSON presets: five functional/hybrid starting points plus 12 style directions. It also contains the design-preset and garment-design-brief JSON Schemas and a synthetic overshirt brief.

The Rust preset composer in `tools/atelier-brief-core/src/preset_stack.rs` supports style/function layering, preserves source ID/revision/digest, rejects hard constraints injected by presets, and returns conflicts for user resolution rather than silently applying last-write-wins or summing objective weights. All presets remain editable starting points, not claims of fit, material performance, environmental advantage, protective function or universal desirability.

## Rust implementation

Three independent dependency-free Rust crates are present:
- `tools/atelier-brief-core/`: brief identity/revision/digest shape, unique IDs, objective bounds, selected hard-constraint contradictions, suggestion-decision provenance and fail-closed handoff.
- `tools/atelier-brief-core/src/preset_stack.rs`: conflict-aware preset composition with explicit user override semantics.
- `tools/atelier-wardrobe-core/`: exact subset search up to 18 optional choices, protected-item preservation, independent borrowing/rental/second-hand/purchase/made-to-order policy flags, single-currency cost arithmetic and Pareto outputs.
- `tools/atelier-sewing-plan-core/`: validates sewing-plan identities, source digests, part/edge/interface references, operation dependencies and acyclic ordering; enforces machine-profile operation support, explicit join interfaces and part membership, and human-review/evidence gates for irreversible operations and unqualified declarations.

Authored unit tests: 11 brief-core + 10 preset-composer + 13 wardrobe-core + 18 sewing-plan-core = **52 tests** across the three Atelier crates. The existing apparel schema/semantic harness adds 22 more, for **74 authored Rust tests total**. They have not been compiled or executed locally; the authoring environment has no Rust toolchain.

## JSON contracts and exact-head harness

Public branch contains 17 editable presets, design-preset and garment-design-brief schemas, a synthetic overshirt brief, a sewing-assembly-plan schema and a synthetic overshirt assembly plan. The existing apparel fixture harness now also meta-validates the sewing-plan schema and validates the synthetic sewing-plan example, with adversarial schema tests. These tests are authored, not yet passed on the current head.

## Validation and CI

The exact-head GitHub Actions workflow is configured to run the existing apparel fixture harness plus all three Atelier crates on pinned Rust 1.96.0. The current authoring environment has no `rustc`, `cargo`, `rustfmt` or Clippy. A current-head pass has not been observed and is not claimed. Skipped general CI, governance, workflow-syntax, showroom and benchmark jobs are not passes.

## Git status

Work is on public research branch `research/apparel-engineering-thread-v0-1` and draft PR #11732. No default branch was modified or merged. Review and exact-head qualification remain open.
