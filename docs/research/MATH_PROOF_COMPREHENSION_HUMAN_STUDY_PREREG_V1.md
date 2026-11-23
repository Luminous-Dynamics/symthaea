# MATH-COMPREHENSION-001 — Human comprehension and transfer pilot preregistration v1

**Parent:** [#13293](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Design PR:** [#13294](https://github.com/Luminous-Dynamics/symthaea/pull/13294)  
**Status:** preregistration template only; no participants recruited and no data collected.

## 1. Research question

Does a source-linked proof-to-concept explanation improve accurate understanding and transfer of formal mathematical reasoning compared with raw Lean/source material and a generic LLM summary?

Secondary question: can a source-linked concept explanation be shortened while preserving human comprehension and transfer?

This study does not test whether the underlying theorem is true; the Lean formalization and source-claim fidelity must be independently established before a task enters the reviewed set.

## 2. Hypotheses

- H1 (comprehension): readers given an explanation linked to exact theorem declarations and proof dependencies will score higher on exact statement restatement and lemma-role questions than readers given raw Lean/source material or a generic summary.
- H2 (transfer): readers given the evidence-linked concept explanation will perform better on a near-transfer problem that changes surface notation or context while retaining the underlying mechanism.
- H3 (faithfulness): the evidence-linked arm will have fewer unsupported mathematical claims and more correct detection of altered assumptions than the generic summary arm.
- H4 (compression): a minimum-sufficient explanation can reduce token length while remaining non-inferior on comprehension and transfer within a preregistered margin.

The hypotheses are prospective. They are not results.

## 3. Task eligibility

Do not recruit participants or run the confirmatory study until:
1. the task's exact source revision, elaborated theorem type, dependency closure and axiom profile are pinned;
2. two independent mathematical annotators review the statement, proof skeleton and counterfactual answer key;
3. a third adjudicator records disagreements;
4. all formal-fact explanation claims link to exact evidence;
5. the task is not contaminated by being present in a model's training / public benchmark exposure, or is labelled as a development-only task.

The current Lean-core and Mathlib pilot seeds are development material. They are not yet expert-reviewed gold tasks and must not be used as confirmatory test items until reviewed.

## 4. Study arms

Use the same theorem, source snapshot, model where applicable, frozen task data, tool access and maximum time budget across arms:

- A — raw source: Lean declaration(s), source and task statement; no generated explanation.
- B — generic summary: ordinary LLM explanation using the source but no typed concept graph or claim-level evidence links.
- C — evidence-linked concept explanation: exact theorem statement, dependency provenance, reviewed mathematical-role annotations, and explicit formal-fact / interpretation / analogy / unknown labels.
- D — evidence-linked proof-plan explanation: arm C plus a `ProofPlanV1` reconstruction of goal transformations, justifications, side conditions, subgoals, invariants and alternative routes. Tactic names remain provenance, not the plan itself.
- E — minimum-sufficient proof plan: arm D compressed under a token budget while preserving theorem identity, all necessary assumptions, plan-action evidence and unresolved-step labels.

The primary comparison for the new capability is D versus C: does an explicit proof plan improve validation, plan reconstruction and transfer beyond a source-linked dependency/concept explanation? E tests explanation efficiency. Compression must never remove a required assumption, side condition or evidence link. A feasibility pilot may use fewer arms if recruitment is constrained; any reduction must be fixed before confirmatory collection and reported as a protocol amendment rather than silently changing the design.

## 5. Randomization and blinding

- Use a preregistered, reproducible random assignment procedure with seed committed before data collection.
- Counterbalance task order and arm exposure. Do not let the same participant see both the source version and its explanation for the same theorem before the primary transfer test; that causes carry-over.
- Assign reviewers independently of the system arm. Reviewers should not know which system produced an explanation when practical.
- Score participant responses with anonymized IDs; do not collect unnecessary identity data.
- Preserve reader-background strata (e.g. prior Lean experience, undergraduate / graduate math exposure) using broad self-reported categories only.
- The statistician / primary analyst should use masked arm labels until the preregistered exclusion and scoring rules are frozen.

## 6. Outcomes

### Primary outcome selection

The confirmatory protocol must designate one primary endpoint before recruitment, after the feasibility pilot and power analysis. **Recommended candidate:** delayed near-transfer accuracy on a structurally related but not previously shown problem. This is the most direct test of retained, usable understanding; the final choice and smallest effect of interest must be frozen before confirmatory data collection.

### Core outcomes (reported separately)

1. **Statement fidelity (0–4):** correct theorem, quantifiers, assumptions, domain and conclusion.
2. **Proof validation (0–4):** ability to identify an invalid inference, missing warrant, unsupported analogy or changed assumption.
3. **Proof-plan reconstruction / presentation (0–4):** ability to explain why each major transformation is used, which side conditions it consumes, what subgoals it creates, and how they discharge the original goal.
4. **Mechanism comprehension (0–4):** correct account of the role of the designated construction, invariant and key lemma.
5. **Immediate near-transfer accuracy (0–4):** solve a held-out isomorphic problem and explain which mechanism transfers.
6. **Delayed near-transfer accuracy (0–4):** solve a distinct but structurally related problem after a preregistered delay, without seeing the answer key or original explanation.

Report every dimension separately. Do not collapse an incorrect theorem statement, invalid proof validation or missing side condition into a high aggregate score because of fluent presentation.

### Secondary outcomes

- counterfactual mutation diagnosis (proved / refuted-by-checked-witness / unknown / ill-formed);
- minimal counterexample validity;
- detection and localization of planted missing assumptions;
- recognition of alternative valid proof plans;
- confidence calibration (Brier score or calibration curve);
- time-to-correct-answer and time-to-plan-reconstruction;
- explanation and plan token count;
- reviewer time;
- participant-reported clarity (secondary, not a substitute for correctness);
- attrition, completion rate and delayed-test retention.

### Hard gates

- No acceptance of an altered or weakened theorem as the original theorem.
- No unsupported formal fact claims.
- No fabricated dependency or invented plan warrant.
- A correct abstention scores better than a false mathematical explanation.
- A proof-presentation score is not a proxy for delayed transfer.
- Any task with an ambiguous answer key is removed or marked inconclusive before unblinding.

## 7. Sample size and analysis

Run a small feasibility pilot first (for example, 8–12 participants per arm only if recruitment and ethics approval allow). The pilot estimates variance, task difficulty, completion time and inter-rater agreement; it is not confirmatory and should not be used to claim a statistically reliable improvement.

Before confirmatory collection:
- define one primary outcome and the multiplicity plan;
- estimate sample size from the pilot variance and a stated smallest effect of interest;
- use a power analysis for the selected design (do not adopt a fixed N without it);
- preregister exclusion rules, missing-data policy, treatment of incomplete responses and stopping rule;
- specify mixed-effects analysis with reader and task effects if the design and number of tasks support it;
- preserve confidence intervals and raw anonymized task-level outcomes.

For H4, define non-inferiority as no more than a 5 percentage-point reduction in the preregistered primary comprehension/transfer outcome at a lower explanation-token budget. This margin is a proposed design choice and must be reviewed and frozen before data collection.

## 8. Quality assurance

- Two independent annotators per task, with adjudication.
- Separate mathematical validity from pedagogical clarity.
- Include known-true, known-false and under-specified counterfactuals.
- Include at least one case where a plausible analogy fails outside its stated boundary.
- Include one theorem family absent from development/tuning.
- Freeze task identities, prompts, source bytes, model versions, tool access, token limits and scoring rubrics.
- Publish negative results, disagreements, attrition and all exclusions.

## 9. Ethical / privacy requirements

Obtain the appropriate institutional ethics review or determination before recruiting human participants. Use an informed-consent process, minimal data collection, de-identification, secure storage and an explicit retention schedule. No participant name, email, precise location or other direct identifier is required in the research dataset. Do not begin data collection until consent and oversight requirements are satisfied.

## 10. Stop conditions

Stop or mark inconclusive if:
- task statement identity cannot be established;
- annotators disagree on theorem meaning and adjudication cannot resolve it;
- explanation fact claims lack source links;
- task contamination makes the holdout invalid;
- missingness or attrition breaks the preregistered analysis;
- a hard trust-gate failure occurs.

## 11. Reporting template

Report by arm and theorem family:
- N randomized / completed / excluded, attrition and missing delayed outcomes;
- participant background stratum;
- statement-fidelity, proof-validation, proof-plan-reconstruction/presentation and mechanism-comprehension scores;
- immediate and delayed near-transfer scores, plus held-out cross-family transfer where available;
- counterfactual disposition accuracy and counterexample validity;
- alternative-plan recognition;
- false-fact / unsupported-claim / invalid-warrant rate;
- confidence calibration;
- median / p95 time-to-answer and time-to-plan-reconstruction;
- explanation and plan tokens;
- reviewer minutes and inter-rater agreement;
- effect sizes, confidence intervals and the preregistered primary-endpoint result;
- protocol version, randomization seed, source/task hashes, model/tool versions and budget receipts.

**No human study has been run.** This document is a preregistration template awaiting expert and ethics review.


## 16. Proof-plan reconstruction and the distinction between explanation and learning (2026-11-23)

A dependency graph shows which declarations are referenced; it does not by itself recover the reasoning plan. Hamami & Morris (2024) characterize proof understanding as rational reconstruction of the proof's underlying plan ([DOI](https://doi.org/10.1111/nous.12489)). Gao & Evans (2026-11-06), in a study of 143 undergraduates, report that proof-presentation performance was only moderately associated with later proof construction on delayed near transfer ([DOI](https://doi.org/10.1007/s40753-026-00319-4)).

The programme therefore distinguishes:
- formal dependency graph (typed, source/toolchain-bound);
- proof-plan graph (goal transformations, justifications, side conditions, subgoals, invariants and alternatives);
- reviewed mathematical-role graph (key estimate, encoding, reduction, construction, proof infrastructure);
- explanation claims (formal facts, reviewed interpretations, bounded analogies and unknowns);
- reader outcomes (statement comprehension, proof validation, plan reconstruction, immediate transfer, delayed transfer and cross-family transfer).

Do not mechanically infer plan actions from a graph edge or tactic trace. Do not force all valid proofs into one canonical plan. Keep alternative proof routes visible. New work is specified in [#13326 — MATH-PROOF-PLAN-001](https://github.com/Luminous-Dynamics/symthaea/issues/13326).
