# MATH-EXP-001C — Freeze Q0 Holdout Before Measurement

Status: **preregistered / not executed / MeasurementOnly**

This tranche freezes the first held-out structural-neighbor set **before reading measured similarities from #4087 or #4112**.

That ordering is the entire point.

If the development Q0 cases reveal a representation defect, the defect may be repaired only under a new encoder/baseline identity and evaluated against a new untouched holdout lineage. The held-out set must not be edited to fit the result.

## Frozen parent representations

Structural HDC:

`symthaea-math-structural-hdc-v1`

Conventional structural baseline:

`canonical-ast-sparse-v1`

Parent exact head:

`1f38f8d0a217df7fd01715602058234c85cfe58c`

## Holdout identity

`math-structural-q0-holdout-v1`

Exact executable source:

`crates/core/symthaea-core/examples/support/math_structural_q0_holdout_v1.rs`

The file contains exact `FolFormulaExt` queries/candidates but intentionally contains **no evaluator** and changes neither representation.

## Why an executable source fixture

A prose fixture list leaves room for later translation choices. An exact Rust constructor freezes:

- operator tree;
- child direction;
- quantifier order/type;
- variable binding/co-reference;
- literal class;
- candidate identities;
- positive labels.

A later evaluator may import this source, but it may not rewrite the formulas after seeing representation scores.

## Nine held-out families

The holdout adds structural axes not fully covered by the development set:

1. ordered subtraction and operand direction;
2. ordered division and reciprocal structure;
3. power nodes versus expanded multiplication;
4. negation scope/depth;
5. commutative formula reordering with variable binding;
6. quantifier shadowing and binding distance;
7. nested propositional structure;
8. rational sign and denominator class;
9. ordered difference inside inequality.

Several cases are intentionally capable of exposing design defects rather than merely confirming intended behavior.

For example:

- conjunction-child reordering may reveal that first-occurrence variable canonicalization is not permutation-equivariant;
- rational denominator-class tests may reveal that a conventional baseline collapsed distinctions that the HDC encoder preserved;
- power-vs-product checks establish that v1 is currently a **syntactic structural representation**, not an algebraic-equivalence representation.

A failure on one of those cases is useful evidence. It must not be relabeled as a bad fixture after inspection.

## Frozen interpretation rule

For each case, evaluate the positive against all named distractors.

Development-set success is not sufficient. The first unbiased representation claim requires the same frozen representation identity to retrieve the intended positive on this holdout.

Do not require HDC to beat the conventional baseline as a Q0 pass condition.

Instead record independently:

```text
HDC Recall@1
HDC positive-vs-best-negative margin
HDC shuffled/permuted controls

Canonical-structural Recall@1
Canonical-structural margin

per-case failure taxonomy
latency / normalized compute
```

Only later Q1/Q2 experiments may test whether one representation yields better proof/search outcomes.

## What to do on failure

If either representation fails a held-out law:

```text
preserve result
classify failure mechanism
freeze v1 as failed/partial
change representation only under v2 identity
create a fresh holdout-v2 before evaluating v2
```

Forbidden:

```text
edit positive label after seeing scores
remove a failed case without preregistered exclusion grounds
change encoder/baseline but keep v1 identity
claim the holdout was only illustrative after it fails
```

## Stronger conventional controls after Q0

Even if both v1 representations pass this holdout, `canonical-ast-sparse-v1` is only a minimum structural comparator.

Before an HDC-specific advantage claim, add stronger conventional controls such as:

- typed root-to-leaf path kernels;
- Weisfeiler-Lehman/subtree features over formula graphs;
- tree-edit-distance diagnostics;
- at Q1+, strong dense premise retrieval with accessible-premise filtering and hard negatives;
- dependency/formula-graph retrieval.

## Authority boundary

This tranche changes no theorem truth, proof validity, evidence state, confidence, memory policy, or formal authority.

Its only authority is to freeze what will count as an unseen Q0 structural test.
