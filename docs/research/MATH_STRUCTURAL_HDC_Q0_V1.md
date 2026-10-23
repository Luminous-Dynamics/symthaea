# MATH-EXP-001A — Structural HDC Q0 Encoder v1

Status: **experimental / non-authoritative**

Encoder identity:

`symthaea-math-structural-hdc-v1`

This tranche turns the structural-HDC hypothesis from #4021 into an executable Q0 intervention without changing runtime retrieval, mathematical memory, proof authority, or theorem verification.

## Why this exists

The April 2026 math arc preserved a clean negative result for the earlier surface-token HDC cascade selector: 22/31 = 71.0%, exactly equal to the majority-class baseline. The same report explicitly identified a tree-structured encoder over `FolFormulaExt` as the appropriate successor experiment.

The purpose of this harness is therefore narrow:

> Before testing whether HDC improves theorem search, establish that a frozen encoder actually represents mathematical structure in the intended way.

A successful Q0 manipulation check is **not** evidence that HDC improves proof search.

## Encoding laws

The v1 encoder intentionally separates structural identity from surface spelling.

It preserves:

- AST node kind;
- directional roles for implication, inequalities, subtraction, division and power;
- commutative symmetry for equality, conjunction, disjunction, addition, multiplication and iff;
- quantifier kind and numeric type;
- bound-variable binding distance rather than bound-variable spelling;
- free-variable co-reference structure through canonical first-occurrence IDs;
- repeated-variable structure;
- coarse literal classes and exponent classes.

It intentionally does **not** encode exact variable names or exact ordinary literal magnitudes in the primary structural channel.

## Why role binding + bundling is required

`BinaryHV::bind` is XOR and is therefore commutative and self-inverse. A naive tree encoder that only chains binds can erase repeated children and lose directional roles. For example, binding the same child twice cancels that child under XOR.

The Q0 encoder instead:

1. binds each child to an explicit role vector;
2. applies role-specific permutations for directional children;
3. bundles the role-bound children with a node-kind vector.

This prevents the encoder from treating `A -> B` as structurally identical to `B -> A`, while also avoiding repeated-child cancellation.

## Q0 invariants

The harness includes executable checks that:

- identical ASTs encode deterministically;
- alpha-renamed quantified formulas retain the same structural embedding;
- formulas with the same coarse literal classes retain the same structural embedding;
- commutative addition is order-invariant;
- implication direction changes the embedding;
- repeated-variable structure is not erased;
- quantifier numeric type changes the embedding;
- a structural analogue outranks both an operator-changed distractor and a shuffled-vector negative control.

## Reproduction

From the repository root:

```bash
cargo test -p symthaea-core --example math_structural_hdc_q0
cargo run -p symthaea-core --example math_structural_hdc_q0
```

The executable prints the frozen encoder ID and the Q0 similarity measurements as CSV-like output.

## Authority boundary

This encoder may propose retrieval neighborhoods only.

It cannot establish:

- theorem truth;
- proof validity;
- novelty;
- formal authority;
- epistemic confidence;
- HDC search advantage.

Any later proof result still requires the existing MATH-SPEC / MATH-EVID / MATH-VERIFY authority chain.

## Relationship to #4069

#4069 freezes the experiment manifest. This Q0 harness supplies the first executable structural encoder that a later frozen manifest can hash and bind as its exact intervention identity.

Changing any encoding rule after held-out evaluation begins requires a new encoder ID and a new experiment lineage.

## Next gate

Do **not** proceed directly to research-level conjectures.

The next evidence tranche should freeze a synthetic structural-neighbor fixture set and compare at least:

- structural HDC retrieval;
- lexical/token retrieval;
- random retrieval;
- shuffled HDC vectors;
- permuted challenge associations.

Only after the manipulation checks succeed should the system move to blind solved-problem transfer under equal proof/search budgets.
