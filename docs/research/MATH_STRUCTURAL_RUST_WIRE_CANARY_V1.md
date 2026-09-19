# MATH-RET-CONVERGENCE-001C — Independent Rust Wire Canary

Status: qualification support only  
Authority: `MeasurementOnly`

## Purpose

`MATH-RET-CONVERGENCE-001B` freezes `math-structural-compat-wire-v1` and a
stdlib-only Python reference oracle. This child adds an independent Rust
implementation of the **byte construction only**.

The goal is cross-language agreement before the wire is used to judge a future
representation relocation.

## Deliberate scope

This tranche does not import, compile, call, or copy:

- `StructuralMathEncoderV1`;
- `CanonicalAstEncoder`;
- the Q0 development harness;
- the preregistered Q0 holdout.

It changes no `symthaea-core` source or Cargo manifest.

The Rust program lives under `.github/scripts/` and is compiled directly with
pinned Rust 1.96.0 using `rustc`. It uses only the standard library.

## Why the Rust program does not implement SHA-256

The compatibility theorem has two separable pieces:

```text
semantic value -> canonical bytes -> SHA-256 commitment
```

001C independently checks the first boundary in Rust. CI hashes the resulting
files with the platform `sha256sum` and compares them with the exact known-answer
commitments frozen by 001B.

This avoids:

- adding a new dependency to `symthaea-core` merely for qualification support;
- coupling `symthaea-core` to `symthaea-muse` evidence helpers;
- maintaining a hand-written cryptographic hash implementation;
- letting the Python oracle and Rust canary share serialization code.

## Cross-language canaries

The Rust generator emits canonical bytes for the same semantic vectors frozen
by 001B:

- all-zero 2,048-byte HDC vector;
- all-`ff` 2,048-byte HDC vector;
- two-feature sparse map in two opposite input orders;
- candidate ranking `[a1, a2, a3]`;
- alternate ranking `[a2, a1, a3]`;
- exact two-record `f32` HDC similarity transcript;
- exact two-record `f64` canonical-AST cosine transcript;
- label-free blind holdout representation aggregate.

CI requires the Rust bytes to hash to the exact 001B SHA-256 commitments.

It also requires:

```text
sparse_wire(order A) == sparse_wire(order B)
ranking_wire(a1,a2,a3) != ranking_wire(a2,a1,a3)
```

Therefore map insertion order is normalized while ranking order remains part of
the evidence identity.

## Independence

The Python oracle and Rust canary share only the **published protocol and frozen
known-answer commitments**. They do not call each other for byte construction.

A future extraction should implement the same profile in the reusable Rust
representation module or qualification support and must reproduce these frozen
known answers before its compatibility receipt can be trusted.

## Holdout firewall

The Rust canary constructs only the artificial label-free holdout known-answer
vector from 001B. It does not reference the real holdout file, labels, case IDs,
candidate IDs, similarity scores, or rankings.

Thus 001C creates no new channel for observing the preregistered holdout.

## Nonclaims

Cross-language wire agreement does not qualify either mathematical
representation, does not qualify #4528, does not prove that an extraction has
occurred, does not establish HDC advantage, and has no theorem/proof authority.
