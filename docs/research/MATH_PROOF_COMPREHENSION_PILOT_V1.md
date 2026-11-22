# MATH-COMPREHENSION-001 — Core theorem pilot seed v1

**Issue:** [#13293](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**PR:** [#13294](https://github.com/Luminous-Dynamics/symthaea/pull/13294)  
**Fixture:** `MATH_PROOF_COMPREHENSION_PILOT_V1.json`  
**Status:** source-pinned smoke-test seed; not gold-standard benchmark data.

## Why start with Lean core?

Before attempting a proof archive of millions of lines, we need to validate the smallest meaningful chain:

1. exact source identity;
2. theorem statement and assumptions;
3. proof expression;
4. conceptual explanation;
5. typed dependency edges;
6. counterfactual mutation;
7. clear uncertainty when the graph or explanation is not yet independently checked.

The first seed uses six compact proposition theorems from the Lean 4 core repository at one pinned commit. This deliberately tests the evidence format and explanation rubric without relying on Mathlib's larger dependency closure.

Pinned source revision:
- Lean 4 commit: `a9b5b8a25c4b932be8029426c2abd34685b078f0`
- `src/Init/Core.lean`, blob `2a44fea5380ec12bd026a1d8d62a2acb3fa79736`
- `src/Init/SimpLemmas.lean`, blob `f713e6cea737f7ead410c99d349f41ecdf9feff6`

## Included theorem families

| Family | Declaration | Mathematical mechanism |
|---|---|---|
| Commutativity | `And.comm` | Reorders paired evidence in both directions |
| Commutativity | `Or.comm` | Swaps the two disjunctive witness cases |
| Associativity | `and_assoc` | Changes grouping while preserving all evidence |
| Conditional redundancy | `and_iff_left_of_imp` | Uses an implication to reconstruct a redundant conjunct |
| Compositional reuse | `and_iff_right_of_imp` | Reorders a conjunction and reuses an earlier theorem |
| Case analysis | `or_iff_right_of_imp` | Uses an implication in the left branch of disjunction elimination |

## Source expressions

`And.comm` at line 1060 is defined from `Iff.intro And.symm And.symm`.

`Or.comm` at line 1064 is defined from `Iff.intro Or.symm Or.symm`.

`and_assoc` at lines 312–314 reconstructs the nested conjunction in both directions.

`and_iff_left_of_imp` at line 325 uses `And.left` in the forward direction and the hypothesis `h : a → b` to reconstruct the second conjunct in the reverse direction.

`and_iff_right_of_imp` at line 326 composes `And.comm` with the preceding theorem.

`or_iff_right_of_imp` at line 348 uses `Or.rec ha id` to handle both cases and `.inr` for the reverse map.

The fixture distinguishes *source-expression edges* from a true elaborated dependency graph. A Lean extractor must produce the latter before the graph can be treated as formal dependency ground truth.

## Initial counterfactual controls

Two assumption-removal tasks have simple finite counterexamples:

- Remove (a \to b) from `and_iff_left_of_imp`. Take (a=\mathrm{True}), (b=\mathrm{False}); then (a\land b) and (a) differ.
- Remove (a \to b) from `or_iff_right_of_imp`. The same valuation makes (a\lor b) true and (b) false.
- Remove (b \to a) from `and_iff_right_of_imp). Take (b=\mathrm{True}), (a=\mathrm{False}).

The JSON fixture records the exact mutation and expected refutation. These finite truth assignments validate these specific propositional counterexamples; they do not validate arbitrary Lean theorem mutations.

## Scoring rubric for the first blind pilot

Score each response independently on a 0–4 scale:

| Criterion | 0 | 2 | 4 |
|---|---|---|---|
| Statement fidelity | Wrong theorem | Mostly right, omissions | Exact claim and assumptions |
| Proof mechanism | Incorrect | Main idea, missing direction | Explains each proof direction |
| Dependency role | Fabricated | Partial links | Exact source-supported role |
| Counterfactual reasoning | Misses falsity | Finds an issue, weak witness | Correct minimal witness and explanation |
| Explanation discipline | Unsupported facts | Some unmarked analogy | Facts, interpretations and unknowns separated |
| Transfer | Cannot solve near task | Solves with hints | Solves unseen near task and explains why |

A response that asserts an invalid theorem as proved is a hard failure even if it scores well elsewhere. `unknown` should be rewarded over fabricated certainty.

Use two annotators, blind to the system arm; record disagreement and adjudicate. Do not use model-generated answers as the only gold labels.

## What has and has not been validated

**Done:** source files fetched at a pinned commit; blob hashes recorded; declaration line spans inspected; six source-expression annotations drafted; three hypothesis-removal counterexamples evaluated by direct Boolean reasoning.

**Not done:** Lean build, kernel proof replay, elaborated-type extraction, actual dependency extraction, independent annotation, Mathlib-scale tasks, human subject study, HDC/CfC ablation, or SOTA score.

This pilot is an evidence-format and research-method seed, not a theorem-prover result.
