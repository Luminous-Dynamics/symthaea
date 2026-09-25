# CIV-BOOT-002D Differential Qualification Contract

Status: **source-implemented / not yet execution-qualified**.

This document freezes the source-level qualification boundary for CIV-BOOT-002D. It does not claim that the Rust candidate has compiled, that the differential harness has executed, or that any real productive capability exists.

## Exact reference oracle

The independent reference implementation is PIE PR #1619:

- PR head: `58499378af3fd3777b916daac1aa5bd573f2c680`
- source path: `scripts/pie-dependency-closure-oracle.py`
- Git blob SHA: `0f7bf960ed4a0575e8b4e2d6d50fe6676e295b80`

CIV-BOOT-002D vendors that exact blob under the capability-closure test/reference tree. The qualifier recomputes the Git blob hash before loading it and refuses to run if the bytes differ.

The vendored oracle remains an independent Python implementation. It does not call the Rust candidate, and the Rust candidate does not call it.

## Exact corpus

Fixture corpus:

`crates/domains/symthaea-capability-closure/fixtures/differential-v1.json`

Git blob SHA:

`b15dec817d22620ebeddcf0def6e0ef52a5e7268`

The qualifier hard-binds this corpus blob as well as the oracle blob. Corpus mutation therefore fails before comparison instead of silently redefining the theorem.

The corpus records the exact reference-oracle PR/head/blob identities and contains 20 fixtures:

- 14 valid structural/equivalence cases;
- 6 preregistered `InvalidInput` cases.

The valid corpus covers local closure, import dependence, downstream inheritance, local alternative routes, unsupported cycles, imported finished equipment, missing prerequisites, robotics repair/reproduction distinctions, 3D-printer frame/reproduction separation, optical-metrology vs laser-source separation, cold-vs-thermal-plasma non-substitution, and declaration-order invariance.

The malformed corpus covers empty IDs, duplicate role IDs, empty route output, empty route requirements, duplicate route IDs, and duplicate route requirements.

## Candidate adapter

Qualification-only Rust adapter:

`crates/domains/symthaea-capability-closure/examples/reference_compat.rs`

Source blob at preregistration:

`d18a6951b7c50468dd5056e61c62c0318a0bd30d`

It translates the PIE #1619 JSON dialect into the typed CIV-BOOT-002A contract and emits a normalized report containing exactly:

- `locally_reproducible`;
- `operationally_reachable`;
- target `{target, status}` entries;
- import `{import_id, targets_lost_if_removed, capability_count_lost}` entries.

The adapter is qualification tooling. Process use in the harness/example does not add filesystem/process authority to the production closure library.

## Differential harness

Harness:

`scripts/civ_boot_capability_closure_differential.py`

Source blob at preregistration:

`f25c7dd738c3f1840ffac08b558cb371c8f97666`

The harness:

1. verifies the frozen corpus Git blob SHA before loading it;
2. verifies corpus schema plus bound reference-oracle PR/head/blob identities;
3. verifies the vendored oracle Git blob SHA before loading it;
4. loads the Python oracle as an independent module;
5. builds the Rust compatibility example with `cargo build --locked` unless an explicit binary is supplied;
6. resolves Cargo's actual target directory through `cargo metadata` rather than assuming `target/`;
7. evaluates the same payload with both implementations;
8. requires each implementation to match the fixture's preregistered `Valid`/`InvalidInput` disposition;
9. requires exact normalized result equality for every valid fixture;
10. requires declaration-order equivalence-group members to produce the same canonical candidate output;
11. prints the exact oracle and corpus blob identities on success.

## Intended execution

From the repository root, in the admitted project environment:

```bash
cargo fmt --check -- \
  crates/domains/symthaea-capability-closure/Cargo.toml \
  crates/domains/symthaea-capability-closure/src/lib.rs \
  crates/domains/symthaea-capability-closure/src/root.rs \
  crates/domains/symthaea-capability-closure/src/reachability.rs \
  crates/domains/symthaea-capability-closure/src/import_leverage.rs \
  crates/domains/symthaea-capability-closure/examples/reference_compat.rs

cargo test -p symthaea-capability-closure
cargo clippy -p symthaea-capability-closure --all-targets -- -D warnings
python3 scripts/civ_boot_capability_closure_differential.py
```

A later executable receipt should additionally bind the exact integrated candidate SHA, Cargo/Rust toolchain, Python interpreter, lockfile/environment identity, command line, oracle blob, corpus blob, and outputs.

## Validation-semantic boundary

The Rust 002A contract intentionally hardens identifier validation beyond PIE #1619 in some areas, including leading/trailing whitespace and control-character rejection.

Those stricter candidate-only hardening cases are **not** included in the V1 equivalence corpus because #1619 does not reject every such form. Therefore:

```text
002D PASS
!= every candidate validation rule is byte-for-byte identical to #1619
```

Instead, V1 differential qualification establishes exact agreement on:

- the shared accepted-input domain in the frozen corpus;
- the shared malformed-input subset explicitly preregistered in the corpus;
- local closure;
- import-enabled operational reachability;
- target classification;
- structural import leverage;
- declaration-order invariance.

Any later attempt to broaden "semantic equivalence" to all validation behavior requires a new version/root or an explicit reviewed mapping.

## Core theorem

```text
same frozen fixture payload
+ independent PIE Python implementation
+ independent Rust implementation
+ exact normalized output agreement
-> evidence of V1 semantic equivalence on that corpus
```

It does not imply physical correctness of a declared route.

## Nonclaims

A successful differential run would **not** establish:

- real manufacturing feasibility;
- quantity or throughput sufficiency;
- material/process qualification;
- maintenance or lifetime adequacy;
- metrology/calibration sufficiency;
- economics;
- safety;
- self-sufficiency;
- authority to manufacture, procure, allocate resources, or operate machinery.
