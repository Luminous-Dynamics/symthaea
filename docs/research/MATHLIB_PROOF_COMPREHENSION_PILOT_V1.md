# MATHLIB proof-comprehension pilot seed v1

**Dataset:** `SYTH-MATHLIB-PROOF-COMPREHENSION-SEED-V1`  
**Research issue:** [#13293](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Draft PR:** [#13294](https://github.com/Luminous-Dynamics/symthaea/pull/13294)  
**Machine dataset:** `MATHLIB_PROOF_COMPREHENSION_PILOT_V1.json`  
**Status:** exact source-pinned seed; not expert-reviewed gold data and not a Lean-extracted dependency graph.

## Snapshot

- Repository: `leanprover-community/mathlib4`
- Commit: `6ebcce37335bcb6bca64596e5d57bb3035648030`
- Commit date: 2026-11-22 12:53:55 UTC
- Mathlib commit message: `doc: fix doc of RiemannHypothesis (#46338)`

This snapshot contains four theorem examples chosen to expose different reasoning mechanisms.

## Tasks

| Declaration | Mechanism | Source | Key explanation challenge |
|---|---|---|---|
| `Nat.choose_pos` | Recursive case analysis and order | `Data/Nat/Choose/Basic.lean:119–122` | Understand the decreasing recursive argument and positivity transfer |
| `Set.countable_union` | Countable indexed union and simplification | `Data/Set/Countable.lean:235–237` | Recover the mathematical step hidden by `simp` without pretending source syntax is a complete proof graph |
| `Multiset.map_id` | Quotient induction | `Data/Multiset/MapFold.lean:152–153` | Explain why proving the list representative case suffices |
| `Finset.card_map` | Embeddings and cardinality | `Data/Finset/Card.lean:271–272` | Explain why injectivity prevents collisions and why multiset cardinality is enough |

## 1. Recursive combinatorial positivity

`Nat.choose_pos` states that if (k\le n), then (\binom nk>0).

Its source proof uses three branches: a zero-boundary case, a zero lower-index case, and a successor/successor recursive case. The latter transfers positivity from a smaller pair using `Nat.add_pos_left`.

**Counterfactual:** remove (k\le n). The instance (n=0,k=1) has (inom01=0), so strict positivity fails.

This tests whether a model can explain not just the theorem but why the domain condition is essential.

## 2. Countable unions and hidden simplification

`Set.countable_union` states:

[
(s\cup t).\mathrm{Countable}\iff
s.\mathrm{Countable}\land t.\mathrm{Countable}.
]

The source is a one-line `simp [union_eq_iUnion, and_comm]`. The surrounding file contains the more general `countable_iUnion` theorem at lines 212–216.

This task must be scored carefully. A good system may explain the conceptual mechanism—encode a two-set union as a countable indexed union—but it must mark the exact simp-generated dependency path as unresolved until a trace or elaborated dependency extraction has been run.

**Counterfactual:** the corresponding unrestricted statement for arbitrary uncountable index sets is false: the union of singleton sets indexed by all reals is the uncountable set of reals.

## 3. Quotient induction over multisets

`Multiset.map_id` is a strong test of conceptual proof understanding.

The visible source proof is:

`Quot.inductionOn s fun _l => congr_arg _ <| List.map_id _`.

The key idea is that a multiset is a quotient of lists by permutation. A representative list can be selected, the identity-map fact proved for that list, and the equality transported back through the quotient.

**Counterfactual:** replace identity by arbitrary `f`. For `Bool.not` and the singleton multiset `{false}`, mapping gives `{true}`, not the original multiset.

## 4. Embeddings preserve finite cardinality

`Finset.card_map` is:

`theorem card_map (f : α ↪ β) : #(s.map f) = #s := Multiset.card_map _ _`.

The type `α ↪ β` supplies injectivity. The proof delegates cardinality preservation to the multiset map theorem.

**Counterfactual:** remove injectivity and use an ordinary image. The constant map from `Bool` to `Bool` takes a two-element universal set to a one-element image.

This example tests understanding of why the embedding is part of the statement rather than decorative typing.

## What this adds

The Lean-core seed tests the basic proof vocabulary. This Mathlib seed introduces more interesting mechanisms:
- recursion and well-founded structure;
- a mathematically meaningful theorem whose source proof hides work behind `simp`;
- quotient induction and representation independence;
- a theorem whose guarantee depends on an embedding invariant.

The pair gives us a small progression from elementary proposition reasoning to library-level mathematical abstraction.

## Status and limitations

**Done:** exact commit and blob identities pinned; declaration snippets fetched; source lines and visible proof expressions inspected; four provisional concepts and counterfactuals recorded; an explicit uncertainty marker added to the `simp`-based example.

**Not done:** Lean elaborated-environment extraction, proof-kernel replay, simp trace, axiom closure, expert annotation, human comprehension study, HDC/CfC ablation, or benchmark comparison. Do not promote these annotations to benchmark gold labels.

## Next execution

1. Build a Lean metaprogram that uses the pinned environment and the dependency traversal semantics in `MATH_PROOF_COMPREHENSION_LEAN_EXTRACTION_V1.md`.
2. Extract type constants separately from proof-value constants; preserve inductive / recursor edges.
3. Trace the `simp` expansion for `Set.countable_union` rather than inferring it from the tactic text.
4. Have two independent reviewers annotate the four conceptual explanations and counterfactuals.
5. Add the resulting tasks to a frozen validation split, not a training set.
