# MATH-REP-001B2 — Exact polynomial adapter v2

Status: **static repair candidate; unqualified**

Normalizer identity:

`symthaea-exact-polynomial-term-v2`

Predecessors:

- MATH-REP-001A receipt contract;
- MATH-REP-001B v1 exact head `21e0efb8f7adb6f75771c36fad05156e5001ac6c`;
- MATH-REP-001C preregistered fixtures exact head `c01c035fa6a14846885bfd62d0ba2555b8e35136`.

## Why v2 exists

Static review found that v1 placed **all source free variables** into canonical serialization before algebraic reduction.

That makes irrelevant/cancelled source variables part of normal-form identity. For example, `x + y - y` reduces to polynomial `x`, but v1 retains source variable list `[x,y]` while literal `x` retains `[x]`. Likewise `0 * (x + y)` reduces to the zero polynomial but v1 retains `[x,y]`, while literal zero has no variables.

This is a canonical-representation defect for an equivalence channel. It is not evidence that the mathematical identities are false.

## Why v1 is not silently patched

The v1 tranche froze this rule: changing canonical serialization requires a new normalizer identity / evidence lineage.

Therefore exact v1 head remains unchanged and v2 receives a new identity.

## Narrow v2 change

v2 reuses v1 for:

- accepted syntax;
- checked `i128` coefficient arithmetic;
- rational reduction;
- exponent checks;
- resource limits;
- refusal classification;
- existing `Rat`/`Poly` storage boundary.

Only canonical variable projection changes.

After v1 polynomial reduction, v2 computes the variable dimensions that have a non-zero exponent in at least one surviving monomial, then projects source dimensions onto active dimensions only before canonical serialization.

Thus:

```text
x + y - y  ==NF-v2 x
x - x      ==NF-v2 0
0*(x+y)    ==NF-v2 0
```

while still preserving:

```text
x           !=NF-v2 y
x - y       !=NF-v2 y - x
Int(term)   !=NF-v2 Real(term)
```

## Variable identity law

v2 uses:

```text
variable_identity = PreserveActiveFreeVariableNames
```

It does **not** alpha-normalize surviving free variables. Only variables proven irrelevant by exact zero coefficients are removed from normal-form dimensions.

This distinguishes source provenance variables from variables present in the canonical polynomial value. The original source AST/provenance remains outside the normal-form serialization and must remain attached through the normalization receipt/research graph.

## Safety of projection

A dimension is removed only if every surviving monomial has exponent zero in that dimension.

Therefore projection cannot merge two distinct surviving monomials: removed dimensions are identical (`0`) in every term. No coefficient arithmetic is needed during projection.

## v2 regression canaries

The focused harness freezes:

1. reproduction of the v1 source-variable identity defect;
2. `x+y-y == x` under v2;
3. `0*(x+y) == 0` under v2;
4. `x-x == 0` under v2;
5. surviving free names remain distinct;
6. subtraction direction remains distinct;
7. domain identity remains distinct;
8. `x*x == x^2` still holds;
9. v1 refusal behavior such as division remains unchanged;
10. projected variable list contains only active names.

Intended focused commands:

```bash
cargo test -p symthaea-core --example math_exact_polynomial_v2_q0
cargo run -p symthaea-core --example math_exact_polynomial_v2_q0
```

No execution qualification is claimed until those commands actually run on the exact candidate head.

## Consequence for the first v1 holdout

MATH-REP-001C was frozen before measured v1 output, but its visible cases helped expose the variable-projection defect during static review.

Therefore that fixture set remains valuable as:

- a frozen regression corpus;
- evidence of what exposed the defect;
- a guard against reintroducing it.

It should **not** be presented as independent confirmatory evidence for v2.

Before making a v2 retrieval-performance claim, freeze a fresh evaluation lineage that is not designed around this exact defect. Prefer a deterministic property-generated corpus and/or independently fixed public theorem corpus over another handful of hand-selected examples.

## Next stronger evaluation

The next confirmatory layer should generate algebraically equivalent/non-equivalent pairs from a frozen grammar and transformation oracle, with:

- fixed seed(s);
- fixed generation grammar;
- fixed rewrite families;
- fixed domain distribution;
- fixed size/degree envelope;
- deduplication before evaluation;
- train/dev/evaluation split committed before scores;
- explicit invalid/unsupported generation accounting.

This tests whether v2 generalizes across a broad expression distribution rather than merely passing the human-readable examples that motivated the repair.

## Deliberate nonclaims

MATH-REP-001B2 does not establish that:

- v2 compiles or executes yet;
- every polynomial equivalence is represented;
- source-variable elimination is a proof of arbitrary theorem equivalence;
- the v1 or v2 fixture sets are blinded;
- equivalence-aware retrieval improves theorem search;
- HDC provides an advantage;
- any theorem is true or novel.
