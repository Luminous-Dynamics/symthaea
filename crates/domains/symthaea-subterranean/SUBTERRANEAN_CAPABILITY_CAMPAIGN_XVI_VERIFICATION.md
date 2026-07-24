# Capability Campaign XVI Verification Record

**Campaign:** Formal Runtime Assurance and Adversarial Validation  
**Date:** 2026-07-20  
**Incremental range:** patches 206–233  
**Cumulative range:** patches 1–233

## 1. Scope

This record covers the Campaign XVI source changes, executable formal-assurance
contracts, static quality gates, offline compatibility-workspace compilation,
and clean-room patch reconstruction.

The uploaded standalone crate omits the real workspace path dependencies. The
executable Rust results therefore use the established API-compatible offline
workspace and Rust 1.85. The complete Rust 1.94 workspace remains the
authoritative integration environment.

## 2. Executable results

The following commands completed successfully in the offline compatibility
workspace:

```text
RUSTFLAGS="-D warnings" cargo check \
  -p symthaea-subterranean --all-targets --offline

cargo test -p symthaea-subterranean --lib --offline -- --test-threads=1
```

Result:

```text
350 passed
0 failed
1 ignored
```

The ignored test is the existing controlled-hardware 200 Hz wall-clock
benchmark.

## 3. Formal-assurance release evidence

The deterministic Campaign XVI release validator passed all seven contracts:

1. transition-violation detection;
2. safety-monotonic formal hold;
3. replay integrity;
4. bounded-state exploration;
5. adversarial mutation detection;
6. checkpoint continuity;
7. live replay continuity.

The bounded model checker passed all six properties over 32 explored cases with
zero undetected counterexamples.

The mutation validator detected all eight preregistered mutations:

- replayed sequence;
- replaced previous digest;
- modified final command;
- modified state step;
- modified trace-completeness flag;
- replaced chain head;
- unauthorized candidate authority;
- reversed decommission state.

## 4. Defects found during verification

Campaign verification found and corrected three real implementation defects:

### 4.1 Debug-overflow panic in deterministic replay mixing

Ordinary integer multiplication could overflow in debug builds during normal
replay append. The deterministic non-cryptographic mixer now uses explicit
wrapping arithmetic.

### 4.2 Incorrect bounded sequence-edge expectation

The model checker initially treated one saturating `u64` edge inconsistently.
The bounded terminal edge is now represented without manufacturing a false
sequence violation.

### 4.3 Panic-capable mutation fixture

The production-marker audit found one `.expect()` while constructing the
adversarial replay fixture. Fixture construction now returns a structured replay
error and produces a failed validation result instead of panicking.

## 5. Static checks

The prepared source passed:

- Rust 1.85 `rustfmt --edition 2024 --check` over every Rust source file;
- `git diff --check` for the final campaign tree;
- warning-denied type checking for all targets;
- a production-source audit before `#[cfg(test)]` sections.

The production audit found zero occurrences of:

- `unsafe {`;
- `panic!`;
- `todo!`;
- `unimplemented!`;
- `.unwrap()`;
- `.expect()`.

Source metrics after Campaign XVI:

```text
126 Rust source files
43,676 Rust source lines
352 #[test] / #[ignore] annotations
```

## 6. Reconstruction method

Two independent application paths are used for delivery verification.

### Incremental path

1. Start from the canonical Campaign XV patch-205 commit.
2. Apply patches 206–233 with ordinary `git am`.
3. Compare the resulting Git tree with the prepared Campaign XVI source tree.

### Full-history path

1. Extract the original uploaded `symthaea-subterranean.tar.gz` snapshot.
2. Create the same imported-baseline Git commit used by the prior campaigns.
3. Apply patches 1–233 in order with ordinary `git am`.
4. Compare the resulting Git tree with the prepared Campaign XVI source tree.

The packaged `VERIFICATION.md` records the exact final Git tree and artifact
hashes after the final verification-record commit is included.

## 7. Trust boundaries and non-claims

The successful checks establish internal consistency of the Rust crate and the
finite formal abstraction. They do not establish:

- proof of the full physical plant;
- completeness of the formal abstraction;
- cryptographic replay authenticity;
- correctness of missing real workspace dependencies;
- production deadline compliance;
- immunity to arbitrary memory, compiler, firmware, or hardware corruption;
- regulatory certification.

Required production follow-up includes the real Rust 1.94 workspace, Clippy,
cryptographic evidence adapters, independent formal review, HIL corruption and
power-loss campaigns, sequence/clock rollover qualification, physical timing,
and validation that the abstraction is conservative for the deployed machine.
