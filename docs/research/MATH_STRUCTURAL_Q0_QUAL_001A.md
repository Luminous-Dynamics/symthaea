# MATH-EXP-Q0-QUAL-001A — Frozen Structural Retrieval Development Qualification

Status: draft qualification protocol

Authority: `MeasurementOnly`

## Purpose

This tranche adds execution evidence for the already-frozen Q0 structural retrieval development lineage without changing either representation, its development fixture, or the preregistered holdout.

It is an execution-only child of exact head:

`b7619f80440d12d57bef5e24c3377b158ec5714d`

The inherited lineage is:

```text
MATH-EXP-001A structural HDC Q0
    d52e08164ce44201dea3fa56b78dfb93dd35cfd3
            ↓
MATH-EXP-001B canonical AST control
    1f38f8d0a217df7fd01715602058234c85cfe58c
            ↓
MATH-EXP-001C preregistered holdout
    b7619f80440d12d57bef5e24c3377b158ec5714d
            ↓
MATH-EXP-Q0-QUAL-001A
    execution only
```

## Frozen inherited objects

The qualification workflow must verify these Git blob identities before any Rust execution:

| Artifact | Frozen Git blob |
| --- | --- |
| `crates/core/symthaea-core/examples/math_structural_hdc_q0.rs` | `35ab3f0d51dc7298d80836f5efec2878a73bb75d` |
| `crates/core/symthaea-core/examples/math_structural_retrieval_q0.rs` | `1b1253b976a0a918d149b0acbb0a85e65f9f1f39` |
| `data/benchmarks/math_structural_q0_v1.json` | `491ffb02d0540974345534f266b9976a6597e548` |
| `crates/core/symthaea-core/examples/support/math_structural_q0_holdout_v1.rs` | `ccd0715b1030b88b7ed1d82f52bfbb8ffd70eaac` |

A mismatch is a qualification failure. The workflow must not repair or regenerate an inherited artifact.

## Development execution

The lane uses Rust `1.96.0` and runs exactly the development harnesses:

```bash
cargo test --locked -p symthaea-core --example math_structural_hdc_q0
cargo run  --locked -p symthaea-core --example math_structural_hdc_q0
cargo test --locked -p symthaea-core --example math_structural_retrieval_q0
cargo run  --locked -p symthaea-core --example math_structural_retrieval_q0
```

Their stdout/stderr are retained as qualification artifacts.

No repository-wide format/check/test result is required by this protocol. The purpose of this lane is to determine whether the exact frozen development subjects compile, test, and execute under the pinned toolchain rather than to reinterpret unrelated workspace failures.

## Holdout firewall

The preregistered holdout source is integrity-checked but **must not be built, imported, evaluated, scored, printed, or otherwise consumed** by this workflow.

In particular, this protocol forbids any cargo command naming:

`math_structural_q0_holdout_v1`

and forbids adding an evaluator for the holdout in this tranche.

The holdout remains available only for a later explicit evidence event after the development representation and measurement procedure have qualified.

## What a PASS establishes

A PASS establishes only that, for the exact child qualification subject and the exact frozen inherited blobs:

- the structural HDC development example compiles/tests under Rust 1.96.0;
- the structural HDC development example executes successfully;
- the canonical AST development example compiles/tests under Rust 1.96.0;
- the canonical AST development example executes successfully;
- the development fixture was the frozen exact Git object;
- the holdout source remained the frozen exact Git object and was not executed by the qualification lane.

## What a PASS does not establish

A PASS does **not** establish:

- HDC retrieval advantage;
- canonical-AST superiority or inferiority;
- holdout performance;
- cross-domain transfer;
- mathematical equivalence;
- proof success;
- theorem truth;
- formal authority;
- runtime integration correctness;
- production readiness;
- novelty.

Development measurements remain `MeasurementOnly`.

## Promotion rule

The exact GitHub Actions run for the exact qualification head must complete successfully before this lineage is called execution-qualified.

`queued`, `in_progress`, `skipped`, workflow creation, local source review, or prior PR governance success are not substitutes for execution evidence.

A failure is preserved and diagnosed from the exact runner output. The frozen predecessor heads are not rewritten to make the lane pass.

## Convergence consequence

After this execution-only lineage qualifies, the integration/convergence work in MATH-RET-INTEGRATION-001 may extract the exact HDC and canonical sparse implementations into one reusable module **only** under compatibility canaries requiring the extracted implementation to reproduce the frozen behavior.

Any representation change discovered during extraction opens a new encoder/baseline identity and a new evidence lineage; it is not folded into this qualification result.
