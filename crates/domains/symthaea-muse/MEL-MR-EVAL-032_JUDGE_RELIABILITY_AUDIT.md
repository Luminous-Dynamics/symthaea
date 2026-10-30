# MEL-MR-EVAL-032 — automated judge reliability and drift audit

**Status:** default-feature Rust audit module and adversarial source tests authored; no model API calls, human labels, benchmark evaluator, compile or test run claimed.

## Research objective

Automated music-edit judges are measurement instruments. Their model version, prompt, decoding configuration, answer order, retry policy and evaluator implementation can change the result. A repeated call to the same model is a retry, not an independent judge.

This protocol preserves MMAE's official frozen evaluation path and adds a separate audit against blinded human references and genuinely separate judge families. It must not change or silently replace the official leaderboard score.

## Official baseline contract

For each rubric, the official MMAE path uses three responses from its frozen Qwen3-Omni evaluator, with answer order shuffled and a majority decision. The public baseline artifacts retain per-rubric response attempts, making a post-hoc reliability audit possible.

For every official attempt, retain:
- sample ID, rubric ID and source-lineage fingerprint;
- exact judge specification identity (model ID/revision, prompt digest, evaluator digest, decoding-config digest);
- one-based attempt index (1, 2, 3);
- answer-order digest;
- raw response digest and binary rubric decision.

The official three-attempt majority is emitted separately from individual-attempt accuracy. Do not change the official evaluator's prompt, model, decoding settings, candidate order policy or aggregation semantics when producing an official-compatible score.

Official source: https://audio-editing-challenge.github.io/

## Human reference subset

The manifest includes one adjudicated human reference per sample/rubric pair, plus:
- source-lineage digest;
- human verdict;
- digest of the adjudication record.

The collection protocol for these references remains EVAL-021 (#9184): blinded identity, randomized playback, trained listeners, unsure response, pre-registered exclusions and source-lineage-clustered uncertainty. An adjudicated label is not self-authenticating; retain the underlying anonymized rating record, adjudicator protocol and consent/ethics metadata in controlled storage.

## Implemented Rust audit

Module: `src/judge_audit.rs`

Types and functions:
- `JudgeSpec`: stable family/model/revision/prompt/evaluator/decoding identity.
- `JudgeAttempt`: preserves the per-attempt output and answer-order commitments.
- `HumanRubricReference`: adjudicated reference bound to a source lineage.
- `JudgeEvaluationManifest`: immutable input bundle.
- `audit_judge_evaluation`: validates coverage and calculates separate official/secondary summaries.
- `verify_judge_evaluation_audit`: recomputes from source inputs to reject tampered receipts.

The auditor:
- requires exactly one official judge spec and at least one declared secondary family;
- rejects a secondary judge that declares the same family as the official judge;
- requires exactly attempts 1, 2 and 3 for each official sample/rubric;
- permits one secondary attempt per sample/rubric and flags retries rather than treating them as another independent judge;
- requires identical human-reference sample/rubric coverage for each judge;
- binds judge outputs and human references to the same source-lineage digest;
- records abstentions explicitly for secondary judges; an official abstention invalidates the official binary triple instead of being coerced;
- computes attempt-level false-pass/false-fail and abstention counts per judge;
- computes official majority agreement with humans separately;
- records retry-pair error contingency counts and a descriptive phi coefficient;
- uses canonical, domain-separated manifest/audit commitments.

A declared family ID is only metadata. It does not prove statistical or architectural independence between models; the evaluator registry and model lineage need external verification.

## Interpretation

Report at least these separate quantities:
1. Official three-attempt majority agreement with human reference.
2. Official individual-attempt agreement with human reference.
3. Same-model retry error association, including both-wrong / one-wrong / neither-wrong pair counts.
4. Each secondary model's agreement, false-pass, false-fail and abstention rates.
5. Human-human agreement as the reference ceiling/measurement context.
6. Operation, modality, source-density and complexity slices.
7. Rank changes between the frozen official score and any secondary calibrated panel.

The retry-pair phi estimate is descriptive over paired retry errors. Do not present the number of retry pairs as independent sample size, and do not infer independence from answer-order shuffling.

## Adversarial controls

- identity/no-op output;
- requested edit omitted;
- wrong edit region or instrument;
- wrong operation;
- under-edit and over-edit;
- correct edit plus collateral change;
- loudness-only or renderer-colour shift;
- correct alternative realization not identical to the paired target;
- ambiguous or unobservable rubric that should receive an explicit human/secondary abstention.

The secondary panel must not reward no-op preservation while missing instruction execution, or reward a successful edit that damages protected content.

## Qualification gates

- Reproduce official aggregate MMAE scores under the exact official evaluator before making any model comparison.
- The official decision must cover every item/rubric in the human calibration subset.
- Every secondary model is scored on the same sample/rubric subset; abstention is shown separately rather than discarded.
- Report retry error correlation and exact joint error counts; do not count retries as independent judges.
- Use genuinely separate model families with externally verified model lineage.
- Freeze panel weights/calibration on the calibration split and evaluate on held-out source lineages.
- Never tune on the official hidden leaderboard set or rewrite historical scores after a judge upgrade.

## Relation to other Muse work

- EVAL-021 (#9184): metric-to-human calibration and unseen-renderer generalization.
- EVAL-030 (#9284): external SMEBench/MMAE benchmark adapter.
- EVAL-031 (#9300 / PR #9303): source/composition leakage preflight.
- EVAL-033 (#9307): valid alternative realizations and contract-based correctness.

## Limitations and claim boundary

This module audits saved responses and labels; it does not run Qwen, another judge, or a human study. It does not prove judge-family independence, human consensus correctness, or official evaluator conformity unless the corresponding model/evaluator artifacts and raw records are independently pinned. No evidence currently shows that the MMAE judge is biased or invalid. The audit tests those questions; it does not assume their answers.

The source module and ten adversarial tests are authored but not compiled or run in the current environment.