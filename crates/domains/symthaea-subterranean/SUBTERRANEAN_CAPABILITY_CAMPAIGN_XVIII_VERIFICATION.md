# Campaign XVIII Verification Record

**Campaign:** Temporal and causal assurance  
**Date:** 2026-07-20  
**Baseline commit:** `69854575c0da6e1ae7d325953067acae92daea5d`  
**Baseline tree:** `8e66e49528f73cb78c8e83876126aec77fbdc3fe`  
**Verified code-and-protocol tree before this record:** `11ee170ff490c010f8cd16274a6f0ef24a196e1a`

## Scope

Campaign XVIII adds replay-resistant clock discipline, delayed-observation authority, bounded causal event ordering, conservative command-response attribution, revision-bound plan freshness, same-frame temporal command restriction, operational evidence, checkpoint continuity, canonical release requirements, and a self-consistent reviewer bundle.

There are 26 campaign commits before this verification record. Including this record, the incremental delivery is patches 261–287 and the complete history is patches 1–287 after the initial import commit.

## Offline build and test environment

The uploaded standalone crate omits the real workspace `symthaea-core`, `symthaea-fep`, and production dependency configuration. Validation therefore used:

- Rust 1.85 toolchain binaries available in the sandbox.
- API-compatible stand-ins for omitted workspace dependencies.
- The complete Campaign XVIII source copied into the offline compatibility workspace.

This verifies Rust typing, exhaustive matches, serialization structure, deterministic behavior, authority integration, and the crate's test suite. It does not replace the authoritative Rust 1.94 full-workspace build or physical qualification.

## Executed gates

### Warning-denied compilation

```text
cargo check -p symthaea-subterranean --all-targets
RUSTFLAGS=-D warnings
PASS
```

### Complete library test suite

```text
389 passed
0 failed
1 intentionally ignored controlled-hardware timing benchmark
```

### Formatting and patch hygiene

```text
rustfmt --edition 2024 --check: PASS
git diff --check: PASS
```

### Production-source audit

Before each `#[cfg(test)]` boundary, no production occurrence was found for:

- `unsafe {`
- `panic!`
- `todo!`
- `unimplemented!`
- `.unwrap()`
- `.expect()`

The audit found one `.unwrap()` in the deterministic temporal release validator before packaging. It was replaced with a structured error and the complete gate suite was rerun successfully.

## Cross-campaign defect found

A restart test exposed a real causal-continuity defect. Formal replay history was checkpointed, but the embodiment runtime step was not. A restored machine therefore presented step zero to a nonempty replay ledger, correctly triggering a formal replay hold and obscuring the more precise temporal hold attribution.

The correction adds the causal runtime step to operational checkpoint schema version 11, validates conversion before mutating live state, and restores the step only after every checkpoint domain validates. The targeted restart test and the complete suite pass after the correction.

## Release-contract results

All eight temporal contracts passed:

1. Clock replay rejected.
2. Stale immediate-control observation removes productive work.
3. Impossible causal dependency latches review.
4. Runtime revision change invalidates a plan.
5. Overlapping effects remain causally ambiguous.
6. Temporal restriction changes the same-frame command.
7. Review hold requires safe clean dwell.
8. Supervisor checkpoint state validates.

The top-level certifiable-autonomy validator also passed with the five new `SUB-TMP-*` requirements included in the canonical registry and traceability matrix.

## Source metrics before this record

```text
143 Rust source files
49,169 Rust source lines
391 test/ignore annotations
178 tracked files
```

## Remaining authoritative gates

Production acceptance still requires:

- Rust 1.94 full-workspace build and Clippy.
- Real `symthaea-core` and `symthaea-fep` integration.
- Calibrated and authenticated clock sources.
- Hardware-in-the-loop delay, reordering, replay, and timestamp-corruption campaigns.
- Power-loss and clock-rollover testing.
- Controlled 200 Hz timing measurements.
- Independent temporal and causal assurance review.
- Cryptographic evidence and reviewer provenance adapters.
