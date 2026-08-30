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
9. snapshot and intervention evidence survives serialization round trips;
10. stable channel identifiers are unique;
11. named higher-level state categories remain absent from core source;
12. passive, restorative, driven, and clamped evidence-plane arms satisfy their declared mechanism expectations.

## Workspace gates

The repository's ordinary pull-request CI must remain green for the exact PR head.
A skipped benchmark workflow is not evidence of benchmark success and must not be
reported as such.

## Evidence capsule

Any result promoted beyond exploratory status should record at minimum:

- exact source commit;
- `Cargo.lock` identity;
- Rust toolchain identity (`rustc -vV` and `cargo -Vv`);
- architecture/platform;
- exact experimental configuration;
- forecast basis;
- input drive/intervention sequence;
- snapshot schema version;
- evidence-plane declarations and measured counters;
- raw result artifact hashes.

A change to source, locked dependencies, toolchain, or experimental semantics starts
a new evidence lineage rather than being mixed into an existing one.

## Parameter gate

The defaults documented in `CALIBRATION.md` remain hypothesis-class values. Before
higher-level interpretation, primary qualitative findings must survive sensitivity
analysis over a declared parameter region rather than a single hand-selected point.

## Stop rule

Do not wire this crate into the cognitive loop or derive higher-level regulatory
state from it until the local crate gates and the required workspace gates pass for
the exact head intended as the v0.1 baseline.
