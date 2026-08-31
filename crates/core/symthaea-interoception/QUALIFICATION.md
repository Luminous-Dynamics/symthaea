# Native Interoception v0.1 — Qualification Contract

This file defines the minimum gate for treating the v0.1 substrate as a qualified
baseline for later regulatory-affect experiments. Passing these gates does not
establish any higher-level interpretation; it only qualifies the mechanical
self-regulation substrate.

## Local crate gates

Run from the repository root in the pinned development environment:

```bash
cargo fmt --all --check
cargo test -p symthaea-interoception
cargo clippy -p symthaea-interoception --all-targets -- -D warnings
```

All commands must exit zero from the same source tree and toolchain environment.

## Mechanical gates

The test suite must demonstrate all of the following:

1. the default state is inside preferred and viable ranges;
2. normalized deviation is zero inside preferred ranges and normalized correctly at lower viability boundaries;
3. zero drive does not manufacture movement inside a preferred range;
4. undriven out-of-band state moves monotonically toward its preferred range;
5. extreme finite drives remain bounded and finite;
6. kinematic and dynamics-aware forecasts remain explicitly distinguishable;
7. dynamics-aware forecasts replay deterministically;
8. direct interventions are recorded separately from endogenous dynamics and reset measured velocity;
9. snapshot, intervention, qualification, and capsule evidence survives serialization round trips;
10. stable channel identifiers are unique;
11. named higher-level state categories remain absent from core source;
12. passive, restorative, driven, and clamped evidence-plane arms satisfy their declared mechanism expectations;
13. property-based tests preserve boundedness, determinism, and passive-recovery monotonicity across a declared generated region;
14. structural sensitivity monotonicities hold for preferred/viable widths, weights, forecast load, recovery, horizon, discount, and drive persistence;
15. zero aggregate weight cannot erase raw per-channel breach evidence.

## Workspace gates

The repository's ordinary pull-request CI must remain green for the exact PR head.
Showroom Integrity must also pass for that head. A skipped benchmark workflow is not
evidence of benchmark success and must not be reported as such.

## Machine-readable qualification receipt

`QualificationReceipt` records the exact source commit and one explicit status for
each fixed required gate. The v0.1 required gate identifiers are:

- `local_fmt`
- `local_test`
- `local_clippy`
- `workspace_ci`
- `showroom_integrity`

Each gate is one of `Passed`, `Failed`, `Skipped`, or `Pending`. `is_qualified()` is
true only when the receipt is structurally valid and every required gate is
explicitly `Passed`. `Skipped` never counts as `Passed`. Optional observations such
as `benchmark_suite` may be recorded without altering the required-gate set.

## Evidence capsule

Any result promoted beyond exploratory status should be accompanied by a valid
`EvidenceCapsuleManifest` recording at minimum:

- exact source commit;
- `Cargo.lock` SHA-256;
- `flake.lock` SHA-256 when present;
- `rust-toolchain.toml` SHA-256 when present;
- Rust toolchain identity (`rustc -vV` and `cargo -Vv`);
- target triple and architecture;
- exact experiment identifier and configuration digest;
- forecast basis;
- input drive/intervention sequence digest;
- snapshot schema version;
- evidence-plane artifact digest;
- raw result artifact hashes.

The crate validates caller-supplied provenance but does not discover or synthesize
Git state, toolchain identity, or artifact hashes itself.

A change to source, locked dependencies, toolchain, or experimental semantics starts
a new evidence lineage rather than being mixed into an existing one.

## Parameter gate

The defaults documented in `CALIBRATION.md` remain hypothesis-class values. Before
higher-level interpretation, primary qualitative findings must survive sensitivity
analysis over a declared parameter region rather than a single hand-selected point.
The executable sensitivity and property gates in the test suite are minimum
structural checks, not substitutes for a later preregistered scientific parameter
sweep.

## Stop rule

Do not wire this crate into the cognitive loop or derive higher-level regulatory
state from it until the local crate gates and the required workspace gates pass for
the exact head intended as the v0.1 baseline, and the corresponding qualification
receipt and evidence capsule validate.
