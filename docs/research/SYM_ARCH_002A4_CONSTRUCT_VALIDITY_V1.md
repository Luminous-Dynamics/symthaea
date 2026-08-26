# SYM-ARCH-002A4 — Adversarial Construct Validity v1

## Purpose

SYM-ARCH-002A3 checks that a generated benchmark is internally coherent: its executable oracle agrees with labels, provenance is bound, train/evaluation examples do not leak structurally, task identity is not exposed under a strict task-free policy, and known corruptions are rejected.

002A4 asks the next question:

> **Can a scientifically trivial train-only shortcut solve the benchmark anyway?**

If yes, the benchmark is not allowed to support an architecture claim, even if the candidate architecture performs well.

This tranche is measurement infrastructure only. It produces no Symthaea capability claim.

## Dependency

002A4 is stacked on SYM-ARCH-002A3 / PR #59 because it consumes `TaskProgram`, `GeneratedTaskDataset`, the executable symbolic oracle, and the A3 structural validity report.

A3 must pass before A4 scores count.

## Learner-visible boundary

Shortcut models receive only `ExampleRecord.features` and training labels.

They do **not** receive:

- `example_id` as a fitted predictor input;
- `support_tags`;
- TaskProgram support metadata;
- oracle outputs;
- evaluation labels during fitting/model selection.

The deterministic chance sanity control hashes `example_id` only to generate a reproducible fair-coin prediction. That id is not exposed to fitted shortcut models.

A unit test changes support tags to make them perfectly label-correlated and verifies that fitted shortcut scores do not change.

## Positive and negative controls

### 1. Executable symbolic oracle — positive control

The A3 `RuleExpr` interpreter predicts the evaluation labels.

The preregistration freezes `oracle_accuracy_floor`. For a deterministic synthetic task, this should normally be `1.0`.

If the oracle misses the floor, the benchmark is `INCONCLUSIVE_BENCHMARK`; architecture scores are not interpreted.

### 2. Deterministic fair-chance sanity control

A domain-separated BLAKE3 hash over a frozen `chance_seed` and evaluation example id generates a deterministic 50/50 prediction.

The observed chance accuracy is diagnostic only. One realized chance draw is **not** itself a pass/fail hypothesis test.

### 3. Training-majority predictor

The predictor chooses the majority label from the training split only, with a frozen negative-class tie break, then applies it to evaluation examples.

This exposes evaluation imbalance or a task that can be solved by ignoring all input features.

### 4. Single-feature marginal predictor

For each learner-visible feature separately:

1. fit a value-to-majority-label table on training data;
2. score that feature on training data;
3. select the feature with highest **training** accuracy only;
4. break feature-selection ties lexicographically;
5. evaluate the selected model on the held-out split;
6. fall back to the training-majority label for unseen feature values.

Evaluation labels are not used to select the feature.

This attacks label channels, marginal correlations, and high-cardinality single-feature shortcuts.

### 5. Exact lookup

The control hashes the complete learner-visible feature assignment for each training example and memorizes its label. Unseen evaluation feature assignments fall back to the training-majority label.

Under A3's strict feature-disjoint policy, exact lookup should not obtain an advantage. It remains an explicit sanity control so a relaxed future split policy cannot silently turn memorization into “generalization.”

### 6. Nearest neighbor

A deterministic categorical nearest-neighbor control uses Hamming-style distance over the union of learner-visible feature keys. Missing or unequal values count as mismatches.

All training examples at the minimum distance vote; ties fall back to the training-majority label.

This attacks local interpolation and near-duplicate structure that exact lookup does not catch.

### 7. Shuffled-relation negative controls

Training labels are deterministically permuted while preserving class prevalence. For every frozen shuffle seed, A4 refits:

- the single-feature marginal predictor;
- nearest neighbor.

It reports both mean and maximum evaluation accuracy across shuffles. The **mean** is the v1 gate to avoid making the verdict depend on one lucky permutation; maxima remain visible diagnostics.

At least four unique shuffle seeds are required. Confirmatory use should freeze a larger seed set when compute is cheap.

## Finite-sample resolution gate

A shortcut threshold is meaningless if the evaluation split is so small that ordinary chance fluctuation can reach it.

A4 therefore computes the 95% Wilson upper bound for a fair Bernoulli classifier at the actual evaluation-set size.

Let:

- `C_shortcut` = preregistered train-only shortcut accuracy ceiling;
- `C_shuffle` = preregistered shuffled-label mean accuracy ceiling;
- `U_chance95` = fair-chance 95% Wilson upper accuracy bound.

The benchmark is `INCONCLUSIVE_BENCHMARK` when:

`U_chance95 >= min(C_shortcut, C_shuffle)`.

This is a benchmark-resolution check, not a null-hypothesis significance test.

One immediate consequence is intentional: very small held-outs such as the four-item SYM-ARCH-001 compositional split cannot support a fine-grained shortcut ceiling. A harder follow-up benchmark must increase evaluation support rather than infer precision from a few Bernoulli outcomes.

## Frozen policy inputs

Before a benchmark becomes claim-bearing, freeze all of the following outside the result path:

- `shortcut_accuracy_ceiling`;
- `shuffled_mean_accuracy_ceiling`;
- `oracle_accuracy_floor`;
- `chance_seed`;
- the complete unique `shuffle_seeds` set;
- A3 structural validity policy;
- benchmark generator/version and seed manifest.

Do not weaken a ceiling, swap shuffle seeds, enlarge evaluation support selectively, or remove a shortcut control after observing architecture or construct-validity results under the same experiment version.

The `0.80` ceilings used in unit tests are synthetic test fixtures only. They are **not** recommended scientific defaults and are not preregistered thresholds for SYM-ARCH-002.

## Fail-closed verdict

A4 has only two top-level states:

- `PASSED`
- `INCONCLUSIVE_BENCHMARK`

It does not return `NEGATIVE` for an architecture when a benchmark fails.

`INCONCLUSIVE_BENCHMARK` is produced if any of the following occurs:

1. A3 structural/oracle validity fails;
2. finite evaluation support cannot resolve the frozen ceiling;
3. executable oracle accuracy is below its frozen floor;
4. majority, single-feature, exact-lookup, or nearest-neighbor evaluation accuracy reaches/exceeds `shortcut_accuracy_ceiling`;
5. the mean shuffled-label single-feature or nearest-neighbor accuracy reaches/exceeds `shuffled_mean_accuracy_ceiling`.

A benchmark failure means **fix or redesign the instrument before interpreting architecture performance**.

## What A4 does not establish

Passing A4 v1 does not prove the absence of every possible shortcut. It only rules out the specific low-complexity alternatives implemented here under the frozen policy.

Later construct-validity work may add, where justified:

- two-feature interaction controls;
- regularized logistic/linear controls;
- decision-tree controls;
- support/serialization leakage audits at the runtime boundary;
- temporal/order-only predictors;
- relation-grammar equivalence checks;
- counterfactual feature interventions;
- learned representation probes.

Those additions must be named as stronger controls rather than retroactively changing the meaning of an A4 v1 pass.

## Acceptance criteria

The exact PR head must demonstrate:

1. A3 structural validity is a hard prerequisite;
2. oracle positive control is explicit;
3. majority/chance controls are reported;
4. single-feature selection uses training data only;
5. exact lookup cannot silently use metadata;
6. nearest-neighbor distance is deterministic and feature-only;
7. shuffled labels preserve training prevalence;
8. at least four unique shuffle seeds are required;
9. tiny evaluation splits fail the finite-resolution gate;
10. an injected single-feature label channel makes the benchmark inconclusive;
11. label-correlated `support_tags` do not change shortcut scores;
12. no architecture score or capability claim is produced by this tranche.

## Wording ceiling

A passing result supports only:

> **The benchmark passed the implemented v1 structural/oracle and low-complexity shortcut controls under the frozen policy.**

It does not support:

> **The benchmark is shortcut-free.**

and it does not by itself support any claim that Symthaea is superior to a baseline architecture.
