# SYM-ARCH-002A5 — Differential / Reference Validation v1

## Purpose

SYM-ARCH-002A3 establishes internal structural/oracle integrity. SYM-ARCH-002A4 attacks the benchmark with low-complexity shortcut learners.

002A5 addresses a different failure mode:

> **What if the generated benchmark and its primary oracle share the same implementation mistake?**

A5 introduces a second RuleExpr evaluator implemented independently from A3, a finite factor-domain truth table, explicit coverage accounting, and an optional independently frozen train/evaluation partition.

A disagreement does not decide which implementation is correct. It makes the benchmark `INCONCLUSIVE_BENCHMARK` until the discrepancy is resolved.

This is scientific instrumentation only. It produces no architecture capability claim.

## Stack relationship

A5 is a sibling of A4 and is stacked directly on A3 / PR #59.

- A3: internal structure, provenance, oracle consistency, mutation detection;
- A4: trivial learner/shortcut resistance;
- A5: differential semantics, finite-domain coverage, and split reference checking.

A future claim-bearing benchmark should satisfy both A4 and A5 where their assumptions apply.

## Independent RuleExpr evaluator

`reference_evaluate_rule` is implemented separately from A3's `evaluate_rule` and does not call it.

It independently handles:

- equality;
- inequality;
- parity/modular predicates;
- NOT;
- AND;
- OR;
- XOR.

For every assignment in the frozen finite reference universe, A5 runs both evaluators and records any semantic disagreement or execution error.

Agreement reduces the risk of a single implementation defect. It is **not** a formal proof that both implementations are correct, because two independent implementations can still share a conceptual mistake.

## Finite factor universe

`DifferentialValidityPolicy.factor_domains` freezes a finite value set for each reference factor.

A5:

1. validates every factor domain is non-empty and duplicate-free;
2. verifies every feature referenced by the RuleExpr appears in the reference domain;
3. computes the Cartesian-product size with overflow checking;
4. refuses to enumerate beyond `max_reference_assignments`;
5. enumerates the complete finite universe deterministically;
6. computes an independently labeled truth table and domain-separated digest.

The assignment cap is a safety/resource boundary, not a statistical tuning parameter.

## Factor projection and nuisance variables

By default, `require_exact_feature_schema = true`: generated examples must contain exactly the declared reference factors.

When a preregistered task genuinely includes nuisance features, exact schema may be relaxed. In that mode:

- every declared reference factor is still mandatory;
- every reference factor value must lie inside its frozen domain;
- extra nuisance features may exist;
- nuisance features are **projected out** of reference assignment identity;
- nuisance features cannot inflate reference coverage;
- nuisance features cannot make an otherwise identical reference assignment appear novel.

This projection behavior is covered by a regression test because incorrectly hashing the full learner-visible example would create false coverage and false out-of-universe failures.

## Generated-dataset checks

For every generated train/evaluation example, A5 checks:

- exact feature schema when required;
- mandatory reference-factor presence;
- reference-factor domain membership;
- uniqueness of the projected reference assignment;
- agreement between `expected_label` and the independent reference truth table.

A projected assignment appearing more than once is a validity violation even if nuisance variables or example ids differ.

## Coverage

A5 reports:

`unique projected generated assignments / reference universe size`.

`minimum_coverage_fraction` is frozen before claim-bearing use.

Coverage below the frozen minimum yields `INCONCLUSIVE_BENCHMARK`.

A low-coverage result is not automatically called a generator bug: sampled/subset benchmarks may be intentional. The correct interpretation is that the benchmark did not satisfy the reference-coverage contract that was frozen for that experiment version.

## Optional reference partition

`ReferencePartition` can freeze the expected train and evaluation feature assignments independently of the generated dataset.

The partition must:

- contain non-empty train and eval sets;
- use valid in-domain reference assignments;
- contain no duplicate projected assignments;
- contain no train/eval overlap.

A5 then compares the generated train/eval projected assignment sets against the frozen reference sets exactly.

This catches a particularly dangerous failure mode: labels remain correct, but the generator silently moves an example across the train/held-out boundary.

For the partition to provide meaningful independence, it should be produced by a separately reviewed pure partition specification or other reference path—not copied from the generated output after the fact.

## Fail-closed result

A5 returns only:

- `PASSED`;
- `INCONCLUSIVE_BENCHMARK`.

It becomes inconclusive when any of the following occurs:

1. A3 structural validity fails;
2. the reference-domain specification is invalid;
3. A3 and the independent evaluator disagree or error on the finite universe;
4. generated feature schema violates the frozen policy;
5. a reference factor has an out-of-domain value;
6. a projected reference assignment is duplicated;
7. a generated label disagrees with the independent truth table;
8. coverage is below the frozen minimum;
9. a frozen reference partition disagrees with generated split membership.

No architecture result is interpreted through an inconclusive benchmark.

## Provenance outputs

The report records:

- reference-universe size;
- unique generated projected assignments;
- coverage fraction;
- A3/reference oracle disagreement count;
- independent-reference label disagreement count;
- a domain-separated digest of the sorted reference truth table;
- explicit violation categories.

A claim-bearing experiment should bind the A5 policy/configuration and truth-table digest into its evidence manifest/ClaimEvidenceBinding rather than treating this check as an informal preflight.

## Acceptance tests

The exact PR head must demonstrate:

1. exhaustive agreement between A3 and reference evaluators on a nested-rule finite universe;
2. detection of an injected independent-evaluator disagreement;
3. detection of wrong train/eval reference partition despite correct labels;
4. rejection of extra learner features in exact-schema mode;
5. correct nuisance projection in relaxed-schema mode;
6. rejection of out-of-domain factor values;
7. rejection of insufficient coverage;
8. rejection of missing rule factors in the reference domain;
9. rejection of a reference universe above the frozen cap;
10. no architecture score or capability wording emitted by this tranche.

## Scope / wording ceiling

Passing A5 supports only:

> **The generated benchmark agreed with an independently implemented v1 RuleExpr reference evaluator over the frozen finite domain and satisfied the frozen coverage/partition contract.**

It does not establish formal correctness of the task semantics, universal absence of generator bugs, or superiority of any Symthaea architecture.

A later procedural-world tranche can extend this pattern to independently implemented transition dynamics, temporal event schedules, interventions, and causal SCM generators.
