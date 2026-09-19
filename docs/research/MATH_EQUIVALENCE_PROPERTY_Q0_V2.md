# MATH-REP-001D0 — Deterministic equivalence property generator v2

Status: **generator/configuration freeze; no normalizer measurements yet**

Generator identity:

`math-equivalence-property-q0-v2`

Authority:

`MeasurementOnly`

Parent representation candidate:

`symthaea-exact-polynomial-term-v2`

## Purpose

Replace small, human-selected equivalence examples with a broad deterministic distribution whose expected labels are fixed **without consulting the normalizer or retriever under test**.

This tranche freezes only:

- generation algorithm;
- RNG algorithm;
- seeds;
- expression grammar;
- transformation/perturbation families;
- domain policy;
- pair/refusal counts;
- interpretation rules.

It deliberately does **not** bind an evaluator to the v2 normalizer yet.

## Why this follows the visible v1 regression set

The first MATH-REP-001C holdout was frozen before v1 scores, but inspection of those visible cases helped reveal the source-variable projection defect that led to v2.

Therefore MATH-REP-001C remains a valuable regression corpus but is not independent confirmation for v2.

MATH-REP-001D0 creates a fresh lineage whose breadth and case selection are determined mechanically from frozen seeds rather than from the exact defect that motivated v2.

This is still open source and therefore **preregistered, not blinded**. A later independently fixed theorem corpus remains necessary for stronger search/proof claims.

## Frozen split sizes

Each split contains four seeds.

Per seed:

```text
128 pair cases
 24 refusal cases
152 total cases
```

Per split:

```text
512 pair cases
 96 refusal cases
608 total cases
```

Because the pair generator cycles through 24 families containing 18 equivalence laws and 6 hard-negative families, each seed contains exactly:

```text
98 expected SameNormalForm
30 expected DifferentNormalForm
```

and each four-seed split therefore contains:

```text
392 expected SameNormalForm
120 expected DifferentNormalForm
 96 expected refusals
```

Every one of the eight refusal families appears exactly 12 times per split.

## Seed separation

Development seeds and evaluation seeds are disjoint and frozen in:

`data/benchmarks/math_equivalence_property_q0_v2.json`

Development seeds may be used to diagnose implementation defects.

Evaluation seeds must not be used to tune `symthaea-exact-polynomial-term-v2` and then presented as independent evaluation of the tuned implementation.

Changing a seed, count, family order, grammar range, domain rule, RNG rule, or label rule requires a new generator identity.

## Independent construction oracle

Generated labels never come from:

- v1;
- v2;
- any normal-form digest;
- HDC similarity;
- a retriever;
- observed evaluation output.

Equivalent pairs are produced by applying a frozen local algebraic law to a generated source expression.

Different-normal-form pairs are produced by a construction guaranteed to alter the formal polynomial/domain identity used by this experiment.

This is intentionally a construction oracle rather than a second copy of the normalizer.

## Equivalent-by-construction families

The 18 frozen families are:

1. addition commutativity;
2. addition associativity;
3. multiplication commutativity;
4. multiplication associativity;
5. distributivity;
6. additive identity;
7. multiplicative identity;
8. double negation;
9. square as repeated product;
10. additive cancellation;
11. subtraction as addition of a negative;
12. negation of a sum;
13. integer coefficient splitting/collection;
14. exact rational coefficient splitting/collection;
15. positive-integer power composition;
16. multiplication by zero;
17. first power identity;
18. difference of squares.

No division/cancellation requiring nonzero assumptions is used.

`0^0` is deliberately not used as an identity family.

## Hard-negative families

Six families produce expected `DifferentNormalForm` cases without querying the implementation under test:

### FreshTermInjection

Adds a unique fresh surviving variable term whose name cannot appear in the base grammar.

### CoefficientPerturbation

Changes a nonzero integer coefficient on a fixed surviving variable.

### ExponentPerturbation

Changes a positive exponent on a fixed surviving variable.

### OperatorPerturbation

Compares `v0 + v1` with `v0 * v1`.

### SurvivingVariableChange

Changes a surviving variable identity (`v0` vs `v4`).

### DomainIdentityChange

Uses the same source polynomial under different frozen numeric-domain identities. This is a **representation distinction**, not a claim that the printed expressions cannot agree on values in an overlapping mathematical domain.

That distinction matters: the normalization contract deliberately includes `Int`/`Nat`/`Real` in normal-form identity.

## Generated atom grammar

Ordinary generated atoms have the form:

```text
c * v_i^e + k
```

with frozen ranges:

```text
v_i in {v0, v1, v2}
c   in 1..=4
e   in 1..=3
k   in 0..=3
```

All ordinary coefficients/offsets are nonnegative. This avoids accidentally making Nat fixtures depend on negative-literal conventions.

Special construction variables are explicit:

- `v3` for coefficient-split families;
- `v4` for surviving-variable perturbation;
- `__delta_<seed>_<index>` for guaranteed fresh-term injection.

## Domain policy

Base domains are deterministically pseudorandom across:

- `Int`;
- `Real`;
- `Nat`.

Exceptions:

- exact rational coefficient splitting is `Real` only;
- families requiring additive inverses remap a generated `Nat` case to `Int` or `Real` deterministically;
- domain-identity negatives explicitly compare different domains.

This keeps Nat cases away from ambiguous truncated-subtraction interpretations while still testing semiring-valid Nat identities.

## Refusal families

The eight frozen refusal families are:

1. inexact `RealLit(f64)`;
2. division by a constant;
3. division by a variable expression;
4. fractional exact rational under `Int`;
5. fractional exact rational under `Nat`;
6. malformed zero denominator;
7. coefficient beyond the v2 exact `i64` envelope;
8. exponent overflow.

Expected refusal labels come from the already-frozen MATH-REP-001A/001B fragment contract, not from observed behavior.

A refusal remains non-conclusive mathematical evidence.

## RNG freeze

Generation uses an in-file xorshift64* implementation with frozen multiplier:

`0x2545F4914F6CDD1D`

and frozen zero-seed replacement:

`0x9E3779B97F4A7C15`

Range selection uses modulo sampling. Any RNG, sampling, or seed change creates a new generator identity.

The generator intentionally has no external RNG dependency.

## Machine-readable consistency gate

The companion stdlib validator:

`.github/scripts/validate-math-equivalence-property-manifest.py`

checks that the JSON manifest and Rust generator agree on:

- generator/authority identity;
- dev/evaluation seed lists;
- seed validity and disjointness;
- pair/refusal counts;
- total cases per split;
- pair family order;
- refusal family order;
- exact 18/6 same/different family partition;
- expected 98/30 pair balance per seed;
- exact refusal-family balance;
- prohibition on normalizer/retriever-generated labels;
- preregistered-not-blinded interpretation.

Intended command:

```bash
python3 .github/scripts/validate-math-equivalence-property-manifest.py
```

This validator checks configuration drift only. It is not a mathematical oracle and does not qualify v2.

## Next child tranche: bind evaluator

Only after this generator/configuration head is frozen should a child evaluator import both:

- this exact generator;
- the exact v2 candidate.

Recommended staged execution:

```text
1. generator manifest consistency
2. v2 focused regression harness
3. generated development seeds
4. classify any development failures
5. if v2 semantics are unchanged, run frozen evaluation seeds explicitly
6. preserve all evaluation failures
```

The evaluation-seed test should be ignored/opt-in by default so ordinary development commands do not silently consume the confirmatory split.

## Metrics for the generated mechanics gate

Report separately:

- SameNormalForm accuracy;
- DifferentNormalForm accuracy;
- refusal disposition accuracy;
- refusal-reason accuracy;
- per-family accuracy;
- normalization acceptance/rejection counts;
- unexpected panics/errors;
- runtime per case / total runtime.

Do not collapse everything into one headline accuracy because a system that accepts equivalences while failing all hard negatives would be unsafe for retrieval.

## After mechanics: retrieval, not truth

Passing this generated gate would establish only that v2 behaves consistently with the frozen exact-polynomial representation contract over this generated distribution.

It would **not** show that equivalence-aware retrieval improves theorem search.

The later retrieval experiment must still compare equal-budget:

```text
S    syntax-only
N    exact-normal-form-only
S+N  syntax + normal-form fusion
```

against lexical, conventional structural, HDC structural, random, shuffled, and permuted controls.

And the later proof-search experiment must use independently fixed problems/premises and verifier-backed endpoints.

## Deliberate nonclaims

MATH-REP-001D0 does not establish that:

- the generator Rust source compiles yet;
- v2 compiles or passes;
- construction-oracle labels cover all algebraic equivalence;
- generated evaluation is blinded;
- normalization improves retrieval;
- HDC improves retrieval;
- proof search improves;
- any theorem is true, valid, or novel.
