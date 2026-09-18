# SYM-RSI-001D — Grounded Dream v7 Structural-Support Amendment

**Status:** preregistered design amendment only. No 301-304, 401-404, or 1201-1204 outcome has been consumed under this amendment.

## Purpose

D-v6 introduced local-support shrinkage, but its nearest-state support used the same 16-dimensional dream-state representation that also contains eight replicated task-quality channels. That makes locality partially endogenous to the outcome variable being predicted.

D-v7 removes that coupling before any sealed SYM-RSI-001D measurement is executed.

This amendment supersedes D-v6 for the actual SYM-RSI-001D verification/fresh/OOD measurement if implemented and qualified before those sealed partitions are consumed. D-v6 remains an intermediate evidence lineage and is not retroactively relabeled.

## Frozen structural-support representation

Local state support is computed from a separate 7-dimensional structural vector:

1. branching-search domain one-hot,
2. delayed-navigation domain one-hot,
3. rugged-optimization domain one-hot,
4. normalized step: `step / horizon(split)`,
5. signed state coordinate `a / (1 + |a|)`,
6. signed state coordinate `b / (1 + |b|)`,
7. auxiliary state `aux / (aux + 10)`.

The structural support vector explicitly excludes:
- task quality,
- all repeated quality channels,
- predicted outcome quality,
- terminal status.

Terminal status is excluded because D chooses actions only in nonterminal states; terminal states have no legal candidate action.

## Frozen support index

The v7 model builds a deterministic support index exclusively from observed TrainingReplay parent states.

Each support entry contains:
- fixture domain identity,
- observed action ID,
- the 7-D structural parent-state vector.

An entry exists only when that domain/action transition was actually observed in the frozen training replay corpus. Generated dream predictions never add entries.

The support index is sorted deterministically and bound into the dream-model evidence digest. Any change in structural-support encoding, entry set, or observation corpus therefore changes model identity.

## Frozen locality rule

For a legal, training-supported candidate action, v7 finds the maximum cosine similarity between the current 7-D structural state and recorded support entries with the same domain and action class.

Similarity is clamped to `[0, 1]` before use. Missing support yields zero support and makes the candidate non-actionable under the existing support gate.

No post-hoc similarity threshold is introduced.

For each model-predicted quality `q_predicted`, locality controls only how much predictive leverage the model receives:

`q_grounded = q_current + structural_support × (q_predicted - q_current)`

Thus:
- support `0.0` gives no quality leverage,
- support `0.5` grants half of the predicted quality change,
- support `1.0` grants the full predicted quality change.

Task-regression probability is computed from `q_grounded`, not raw `q_predicted`.

Model confidence is:

`(1 - task_regression_probability) × mean_structural_support`

Epistemic support distance is:

`1 - mean_structural_support`.

## Frozen transition-predictor scope

D-v7 changes the **epistemic support/trust calculation**, not the internal transition-prediction retrieval rule of `symthaea-dream`.

The learned transition predictor continues to receive the existing 16-dimensional dream-state representation and, for a domain-conditioned action fingerprint, chooses the recorded transition-memory observation with greatest cosine similarity in that model space before blending its observed outcome with the action heuristic.

Therefore:
- task quality may remain an input feature to the transition predictor;
- repeated quality channels may influence which recorded transition supplies the model-generated outcome;
- that 16-D similarity is **not** used as v7 structural support, epistemic confidence, support distance, or the amount of predictive leverage granted to D;
- v7 makes no claim that transition retrieval itself is quality-independent;
- v7 makes no claim that the transition selected by the predictive model is necessarily the same training observation that maximizes 7-D structural support.

This distinction is intentional and frozen before measurement. A future protocol may align predictive retrieval and structural support to the same observed transition, but that would be a new model/protocol version and may not be introduced after seeing v7 sealed outcomes.

## Frozen decision rule

D-v7 retains the existing boundaries:
- domain-conditioned action identity,
- observed-training action support gate,
- legality-aware perturbations,
- at most five model simulations per actionable candidate,
- exact per-prediction model-call accounting,
- risk penalty `0.10`,
- override margin `0.01`,
- generated outcomes remain non-empirical,
- generated outcomes cannot enter `ExperienceTree`,
- generated outcomes cannot independently promote confidence.

The D-v7 score is:

`mean structurally-grounded predicted task quality - 0.10 × task-regression probability`.

D may override C only when the candidate and base action classes are both supported by recorded training evidence and the D-v7 candidate score exceeds the base score by more than `0.01`.

## Evaluation lineage

This amendment does not change the already-frozen evaluation partitions:
- training replay: seeds 1-8,
- dream verification replay: 301-304,
- fresh D-vs-C: 401-404,
- OOD follow-up: 1201-1204.

No v7 implementation test may execute 301-304, 401-404, or 1201-1204.

The implementation must qualify before any sealed partition is consumed.

## Required implementation evidence

Before v7 may be used for SYM-RSI-001D measurement, qualification must show:

1. structural-support vectors are exactly seven dimensions;
2. changing only task quality leaves structural support vectors unchanged;
3. changing structural coordinates can change structural support;
4. support entries derive only from observed TrainingReplay parent states;
5. support index identity is included in model evidence hashing;
6. generated predictions cannot create support entries;
7. candidates with zero structural/action support cannot override C;
8. support-adjusted quality is bounded to `[0, 1]`;
9. prediction provenance binds raw quality, grounded quality, structural support, action support, actionability, and exact model-call count;
10. ordinary CI/unit tests do not consume sealed evaluation seeds;
11. no implementation or report describes the 16-D transition-retrieval similarity as v7 structural support.

## Claim boundary

A later positive D-vs-C result under v7 could support only the claim that structurally grounded learned counterfactual predictions improved the frozen fixture tasks under the preregistered protocol.

It would not establish that the transition predictor itself is quality-independent, that structural support and transition retrieval identify the same training observation, that cosine similarity is an optimal notion of semantic locality, that the result generalizes beyond the tested domains, or that Symthaea has achieved open-ended recursive self-improvement.
