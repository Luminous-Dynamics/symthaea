# MATH-REP-001A — Mathematical Normalization Receipt v1

Status: **contract only / retrieval-only / non-authoritative**

This contract defines the boundary for future equivalence-aware mathematical retrieval.

Its central purpose is to prevent a normalizer from quietly becoming a theorem prover, truth oracle, or replacement for source provenance.

## Core law

```text
SyntaxSimilarity != AlgebraicEquivalence != FormalAuthority

normalization receipt != theorem proof
same normal form != permission to erase source AST
unsupported normalization != non-equivalent
failed normalization != false
```

Every receipt has:

```text
authority = RetrievalOnly
source_preserved = true
```

The closed grammar contains no `truth_value`, `formal_authority`, `proof_valid`, or epistemic-confidence field.

## Outcomes

### `Normalized`

A supported source fragment was mapped to a deterministic canonical representation.

A normalized receipt requires:

- source SHA-256;
- source provenance references;
- numeric domain;
- normalizer identity;
- canonical serialization;
- normal-form SHA-256;
- explicit transformation list;
- validation state;
- side conditions, if any.

This does **not** establish the mathematical claim represented by the source formula. It establishes only the declared normalization transformation within the declared fragment and conditions.

### `Unsupported`

The normalizer does not support the source safely.

Examples include:

- variable denominators;
- non-polynomial division;
- negative/fractional powers;
- transcendentals;
- order-sensitive rewrites;
- missing side conditions;
- ambiguous numeric domains.

Unsupported records carry no normal form.

### `Rejected`

The request was ineligible for normalization under the frozen policy or resource boundary.

Rejected records likewise carry no normal form and are never interpreted as mathematical counterexamples or negative results.

## Initial fragment

The v1 contract is designed for a deliberately narrow exact-polynomial adapter:

```text
Int / exact Rat literals
variables
+  -  *
non-negative integer powers
polynomial terms/equalities
```

Numeric domain (`Int`, `Nat`, `Real`) remains part of the identity.

The contract does not authorize domain-changing rewrites.

## Side conditions

Side conditions are explicit typed records:

```text
NonZero
NonNegative
Positive
Negative
DomainConstraint
Other
```

Each condition carries a digest and one of:

```text
Assumed
Derived
FormallyVerified
```

`FormallyVerified` requires evidence references.

A transformation that cancels a factor (`CancelNonzeroFactor`) is rejected by the semantic validator unless an explicit `NonZero` condition is retained.

This encodes the difference between:

```text
x * x       <-> x^2          exact polynomial normalization
x + x       <-> 2*x          exact polynomial normalization

x / x       <-> 1            requires x != 0
sqrt(x^2)   <-> x            not in v1 safe fragment
1/x < 1/y   <-> y < x        not in v1 safe fragment
```

## Validation states

```text
NotApplicable
StructuralRewriteChecked
ExactPolynomialCanonicalization
SolverCrossChecked
FormalEquivalenceReceipt
```

These states describe the normalization mechanism only.

`SolverCrossChecked` and `FormalEquivalenceReceipt` require evidence references, but references do not authenticate themselves. Formal evidence authority remains in the qualified mathematical evidence/verification plane.

## Source preservation

A normal form is an additional retrieval projection.

It must never replace:

- the original `FolFormulaExt` source;
- source digest;
- ResearchGraph provenance;
- theorem/evidence identity;
- proof/counterexample receipts.

The intended future shape is:

```text
                     source mathematical object
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
      syntax representation            normalization receipt
             │                                 │
      syntax retrieval                  normal-form retrieval
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                       candidate proposal
                              │
                              ▼
                       proof / falsification
                              │
                              ▼
                     evidence authority plane
```

## Adversarial validator checks

The dependency-free validator includes self-tests rejecting:

1. theorem/truth-authority fields smuggled into the receipt;
2. `source_preserved = false`;
3. factor cancellation without a `NonZero` side condition;
4. a formal-equivalence validation state without evidence references;
5. an unsupported record that still carries a normal form.

It also accepts:

- safe cancellation when the explicit nonzero condition is retained;
- unsupported variable-denominator input as a non-conclusive receipt.

## Reproduction

```bash
python3 .github/scripts/validate-math-normalization-receipt.py --self-test
python3 .github/scripts/validate-math-normalization-receipt.py path/to/receipt.json
```

The semantic validator is intentionally Python-stdlib-only.

A JSON Schema is also provided for closed interchange validation:

`.github/schemas/math-normalization-receipt-v1.schema.json`

## Relationship to existing algebra code

Future implementation should adapt supported `FolFormulaExt::Term` fragments into the existing exact algebra machinery rather than creating another symbolic system.

Relevant existing modules include:

- `hdc/polynomial_algebra.rs` for exact rational/polynomial machinery;
- `hdc/tactics.rs` for existing polynomial-normalization logic.

The adapter should be conservative: inability to translate safely produces `Unsupported`, not a guessed normal form.

## Required next tranche

`MATH-REP-001B` should implement the exact polynomial adapter and prove the following canaries before retrieval integration:

```text
x*x and x^2 -> same normal form, distinct source ASTs
x+x and 2*x -> same normal form
(x+1)^2 and x^2+2*x+1 -> same normal form

x/x without x != 0 -> Unsupported or MissingSideCondition
Real/Int/Nat source domains remain distinct
rational coefficients are exact/reduced
deterministic serialization is byte-stable
```

No retrieval policy change belongs in 001B.

## Nonclaims

A schema-valid or validator-valid receipt does not establish:

- theorem truth;
- proof validity;
- novelty;
- global algebraic equivalence outside the declared fragment/conditions;
- HDC advantage;
- search improvement;
- source-provenance authenticity.

Those require their respective evidence chains.
