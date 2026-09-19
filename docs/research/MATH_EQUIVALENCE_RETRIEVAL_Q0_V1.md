# MATH-REP-001C — Preregistered equivalence-retrieval Q0 v1

Status: **frozen before observing MATH-REP-001B normalization outputs**

Fixture-family identity:

`math-equivalence-retrieval-q0-v1`

Development set:

`math-equivalence-retrieval-q0-dev-v1`

Preregistered holdout:

`math-equivalence-retrieval-q0-holdout-v1`

Authority:

`MeasurementOnly`

## Purpose

Freeze the first controlled equivalence-aware retrieval fixtures before using MATH-REP-001B output to tune the representation.

This tranche contains **fixtures only**. It adds no evaluator, retrieval fusion, scoring threshold, runtime memory path, proof search, or theorem authority.

## Important limitation: preregistered, not blinded

The holdout is committed in source form, so it is not secret from developers or models.

Its purpose is narrower:

- prevent changing labels/cases after observing v1 scores;
- prevent deleting awkward failures after the fact;
- preserve an exact evidence lineage;
- distinguish development diagnostics from the first untouched evaluation pass.

It is **not** a substitute for later blind/public-corpus evaluation.

For stronger Q1 evidence, use a corpus split whose challenge instances are fixed independently of the representation and whose accessible-premise boundary is frozen before retrieval/search runs.

## Frozen source files

- `examples/support/math_equivalence_fixture_types_v1.rs`
- `examples/support/math_equivalence_dev_v1.rs`
- `examples/support/math_equivalence_holdout_v1.rs`

No file in this tranche imports the MATH-REP-001B adapter. That is intentional: fixture identity is independent of implementation behavior.

## Expected relation vocabulary

Pair cases have only:

```text
SameNormalForm
DifferentNormalForm
```

Refusal cases freeze only:

```text
expected disposition
expected receipt rejection class
```

No fixture asserts theorem novelty, proof authority, epistemic confidence, or HDC advantage.

## Development set

The development set contains 12 pair cases plus 6 refusal cases.

Positive families include:

- square versus repeated product;
- repeated sum versus scalar multiplication;
- binomial expansion;
- difference of squares;
- distributivity;
- negation distribution;
- exact rational coefficient collection;
- nested powers;
- additive cancellation.

Required distinctions include:

- subtraction direction;
- free-variable identity;
- numeric-domain identity.

Refusal families include:

- inexact floating literal;
- constant division;
- variable denominator;
- malformed zero denominator;
- non-integral rational under `Int`;
- coefficient overflow.

Development cases may diagnose implementation defects. They may **not** be relabeled after observed results to make v1 pass.

## Holdout set

The preregistered holdout contains 13 pair cases plus 6 refusal cases using different surface forms from the development set.

Positive families include:

- scalar distribution;
- even power of negation;
- three-variable commutative reorder;
- rational common-denominator collection;
- power composition;
- product commutation;
- zero multiplication;
- zero exponent;
- multi-term coefficient collection.

Required distinctions include:

- product versus sum;
- degree change;
- `Nat` versus `Real` domain identity;
- free-parameter identity.

Refusal families include:

- a different inexact real literal;
- compound constant division;
- compound variable denominator;
- non-integral rational under `Nat`;
- negation that exceeds the v1 `i64` exact coefficient envelope;
- exponent overflow.

## Failure law

After any MATH-REP-001B score/output is observed:

Forbidden:

```text
change SameNormalForm <-> DifferentNormalForm
change expected refusal class to match implementation behavior
remove a failing case without a preregistered invalid-fixture reason
change normalizer semantics but retain its v1 identity
edit holdout expressions to improve scores
```

Allowed:

```text
preserve v1 failure
classify the failure
repair implementation under a new normalizer identity if semantics change
freeze a new holdout lineage before evaluating the changed representation
```

A pure implementation bug that violates the already-frozen v1 semantics may be repaired only with an exact defect demonstration and should preserve an audit trail explaining why the semantic contract did not change.

## Evaluation sequence

The intended order is:

```text
MATH-REP-001A receipt contract
            ↓
MATH-REP-001B exact term normalizer
            ↓
MATH-REP-001C fixtures frozen      <-- this tranche
            ↓
focused mechanical execution
            ↓
development diagnostics
            ↓
preregistered holdout evaluation
            ↓
retrieval comparison
```

Do not tune on the holdout and then call the same holdout independent evidence.

## Retrieval arms after normalization qualifies

The first retrieval comparison should separate representation effects:

```text
S    syntax-only
N    exact-normal-form-only
S+N  syntax + normal-form fusion
```

For each arm freeze equal:

- candidate corpus;
- query set;
- accessible-knowledge boundary;
- top-k/item budget;
- serialized byte/context budget;
- compute budget;
- tie-breaking rule;
- downstream proof/search budget.

`S+N` must not receive twice as many retrieved items merely because it has two channels.

## Required controls

At Q0/Q1 add at least:

- lexical/token retrieval;
- canonical structural non-HDC retrieval;
- structural HDC retrieval;
- exact-normal-form conventional retrieval;
- exact-normal-form HDC retrieval if tested;
- shuffled-vector control for HDC;
- permuted query/candidate associations;
- random retrieval.

This yields a causal ladder capable of separating:

```text
surface similarity
syntax structure
exact algebraic normalization
HDC representation
fusion
```

## Primary retrieval measurements

Before proof search, measure:

- Recall@k for same-normal-form families;
- false-equivalence neighborhood rate;
- MRR / reciprocal rank of intended equivalent forms;
- rank margin over named hard negatives;
- refusal correctness rate;
- harmful-transfer candidates retrieved from `DifferentNormalForm` cases;
- normalization coverage/rejection rate;
- normalized-form compute cost.

No single similarity threshold should be tuned on the holdout.

## Transition to proof-search Q1

Even a perfect Q0 normal-form result establishes only representation mechanics.

The research claim becomes interesting only when equal-budget theorem search asks whether retrieved reformulations/analogies cause measurable improvement in:

- verified useful lemma rate;
- formal solve rate;
- time/compute to first useful lemma;
- search-node/prover-call efficiency;
- repeated-failure rate;
- false-pruning/harmful-transfer rate.

Proof authority remains in the verifier/evidence plane.

## Deliberate nonclaims

MATH-REP-001C does not establish that:

- MATH-REP-001B compiles or passes;
- the fixture labels have been mechanically validated yet;
- exact normal-form equality captures all mathematical equivalence;
- equivalence-aware retrieval improves theorem proving;
- HDC improves equivalence retrieval;
- any theorem is true, novel, or formally proved.
