# Capability Campaign XV Verification

**Campaign:** Human–Machine Operational Accountability  
**Date:** 2026-07-20  
**Patch range:** 187–205  
**Implementation tree before this packaging-only record:** `7e25c185a0aa05111839b556ddb360bb80f73ad5`

## Validation environment

The uploaded crate does not include its real workspace path dependencies or the requested production Rust 1.94 toolchain. Validation therefore used:

- Rust 1.85 binaries available in the execution environment;
- API-compatible local stand-ins for `symthaea-core` and `symthaea-fep`;
- the crate’s actual Campaign XV source;
- offline Cargo operation.

This validates Rust typing, exhaustive matching, deterministic behavior, serialization shape against the stand-ins, and the complete local test suite. It does not replace the authoritative full-workspace Rust 1.94 build, Clippy, production HDC/FEP integration, or hardware qualification.

## Executed gates

### Warning-denied type check

```text
RUSTFLAGS='-D warnings' cargo check \
  -p symthaea-subterranean \
  --all-targets \
  --offline
```

Result: **passed**.

### Complete deterministic library suite

```text
cargo test -p symthaea-subterranean --lib --offline
```

Result:

```text
335 passed
0 failed
1 ignored
```

The ignored test is the pre-existing controlled-hardware 200 Hz wall-clock benchmark. It remains intentionally excluded from ordinary deterministic unit-test execution.

### Formatting and whitespace

- Rust 1.85 `rustfmt --edition 2024 --check`: **passed** for every Rust source file.
- `git diff --check`: **passed**.

### Production panic-marker audit

Before each file’s `#[cfg(test)]` section, the source was scanned for:

- unsafe blocks or unsafe items;
- `panic!`;
- `todo!`;
- `unimplemented!`;
- `.unwrap()`;
- `.expect()`.

Result: **zero production construct findings**.

The audit intentionally distinguishes Rust `unsafe` constructs from identifiers such as `unsafe_cutter_frames` and comments discussing unsafe behavior.

## Campaign-scale counts

At the implementation tree recorded above:

- 117 Rust source files;
- 41,334 Rust source lines;
- 336 test or ignore annotations;
- 18 implementation/documentation commits after patch 186;
- the nineteenth campaign commit is this packaging-only verification record.

## Accountability acceptance results

The deterministic accountability validator exercises:

1. deployed decision-trace completeness;
2. observational counterfactual explanations;
3. challenge replay resistance;
4. near-miss corrective retention;
5. appeal safety monotonicity;
6. bounded challenge-deadline escalation;
7. append-only challenge-bound evidence correction;
8. operational-checkpoint continuity.

The validator is also included in the crate-wide certification validator as a release-blocking contract.

## Evidence-bundle checks

The accountability evidence bundle tests verify that:

- its cached snapshot matches the serialized supervisor state;
- at least one complete decision trace is retained;
- unresolved or overdue review state blocks release clearance;
- validation covers every canonical accountability contract;
- Operator Representative, Safety Reviewer, and Independent Auditor identities are distinct;
- every canonical accountability requirement is included exactly once;
- deterministic digest generation is stable for reproducibility testing.

The deterministic digest is not a cryptographic signature. Production must inject cryptographic digest and signing providers.

## Clean-room reconstruction procedure

Two independent reconstruction paths are required for packaging:

1. Apply patches 187–205 over the canonical patch-186 baseline.
2. Import the original uploaded crate and apply patches 1–205 in order.

The package-level verification record, produced after this source record is committed, contains the resulting final Git tree and exact application transcripts. The source record cannot contain its own final tree hash without creating a self-referential tree identity; it therefore records the implementation tree immediately before the packaging-only verification commit.

## Remaining authoritative gates

The following remain required in the complete workspace and physical qualification environment:

```text
cargo fmt --check -p symthaea-subterranean
cargo clippy -p symthaea-subterranean --all-targets -- -D warnings
cargo test -p symthaea-subterranean
cargo test -p symthaea-subterranean \
  runtime_budget::tests::reference_200_hz_control_loop_budget \
  -- --ignored --nocapture
```

Additional non-software qualification remains necessary for:

- cryptographic identity and amendment provenance;
- secure monotonic clocks and anti-rollback storage;
- independent human-factors testing;
- explanation-fidelity review against the deployed executable graph;
- incident and appeal exercises;
- jurisdiction-specific records, labor, community, and regulatory review.
