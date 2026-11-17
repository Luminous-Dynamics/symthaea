# Finite symmetry query semantics reference v0.1

This standalone Rust crate implements the first executable slice of [SYMM-QUERY-SEMANTICS-001 #12601](https://github.com/Luminous-Dynamics/symthaea/issues/12601). It composes with, and depends on, the finite group/action/orbit verifier in [symmetry quotient PR #11653](https://github.com/Luminous-Dynamics/symthaea/pull/11653). It does not implement another group-action or orbit-construction engine.

## Core rule

A valid orbit partition does not define how a query must be aggregated. The query algebra is part of the problem semantics:

- Existence: Boolean OR over included orbit representatives.
- Minimum / maximum: evaluate one invariant objective per included orbit, then lift every original tied candidate.
- Enumeration: lift every included representative to every original member.
- Counting: add each included orbit's cardinality.
- Weighted sum: multiply each invariant signed weight by the orbit cardinality and sum with checked arithmetic.

Never use group order as a substitute for orbit cardinality. The D4 fixture has 16 ordered vertex pairs, 3 orbits, and orbit sizes 4, 8 and 4.

## Query-specific invariance

The module checks only fields that affect the selected query:
- Every query checks predicate invariance.
- Minimum and maximum additionally check objective invariance.
- Weighted sum additionally checks weight invariance.
- Count and enumeration do not reject irrelevant changes in an unused objective or weight field.

This avoids both unsafe pruning and unnecessary over-rejection.

## Independent replay

The query receipt binds the exact versioned query contract bytes, problem and action identities, aggregation law, per-orbit contributions, output, and candidate-evaluation counts. The verifier replays the base quotient certificate and recomputes the entire query receipt from the exact original query. This v0.1 implementation carries canonical contract bytes, not a cryptographic hash or signature.

An independent exhaustive query evaluator is included. Frozen D4 tests compare all six query kinds against this unquotiented baseline and include wrong-predicate, wrong-objective, wrong-weight, wrong-multiplicity, tie-loss, receipt-tampering and arithmetic-overflow controls.

## Reproduction

    cargo fmt --manifest-path research/symmetry-query-semantics-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symmetry-query-semantics-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symmetry-query-semantics-001/Cargo.toml --all-targets -- -D warnings

The D4 test suite measures 3 representative inspections against 16 exhaustive candidate inspections. This is a candidate-query evaluation count on the frozen fixture, not a wall-clock speedup.

## Claim boundary

Finite, discrete, exact integer predicates/objectives/weights only. No arbitrary semiring, continuous-group integral, Haar measure, orbit-volume correction, coarea/Jacobian factor, approximate invariance, cryptographic certificate, or production search authority is implemented. Unknown query fields must be declared and qualified before adding new query kinds.
