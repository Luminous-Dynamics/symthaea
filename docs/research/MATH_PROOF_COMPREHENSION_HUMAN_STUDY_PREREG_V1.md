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

Use the same theorem, model, frozen task data and maximum time budget across arms:

- A — raw source: Lean declaration(s), source and task statement; no generated explanation.
- B — generic summary: ordinary LLM explanation using the source but no typed concept graph or claim-level evidence links.
- C — evidence-linked explanation: concept graph, exact statement and dependency provenance, explicit formal-fact / interpretation / analogy / unknown labels.
- D — minimum sufficient explanation: the evidence-linked explanation compressed under a token budget while preserving the same source references and proof assumptions.

Arm D is an efficiency experiment; never allow compression to remove a required assumption or evidence link.

## 5. Randomization and blinding

- Use a preregistered, reproducible random assignment procedure with seed committed before data collection.
- Counterbalance task order and arm exposure. Do not let the same participant see both the source version and its explanation for the same theorem before the primary transfer test; that causes carry-over.
- Assign reviewers independently of the system arm. Reviewers should not know which system produced an explanation when practical.
- Score participant responses with anonymized IDs; do not collect unnecessary identity data.
- Preserve reader-background strata (e.g. prior Lean experience, undergraduate / graduate math exposure) using broad self-reported categories only.
- The statistician / primary analyst should use masked arm labels until the preregistered exclusion and scoring rules are frozen.

## 6. Outcomes

### Primary outcomes

1. **Statement fidelity (0–4):** correct theorem, quantifiers, assumptions, domain and conclusion.
2. **Mechanism comprehension (0–4):** correct account of the role of the designated construction and key lemma.
3. **Near-transfer accuracy (0–4):** solve a held-out isomorphic problem and explain which mechanism transfers.

Report each separately. Do not collapse an incorrect theorem statement into a high total score because of a good explanation style.

### Secondary outcomes

- counterfactual mutation diagnosis (true / false / unknown / proof gap);
- minimal counterexample validity;
- detection of planted missing assumptions;
- confidence calibration (Brier score or calibration curve);
- time-to-correct-answer;
- explanation token count;
- reviewer time;
- participant-reported clarity (secondary, not a substitute for correctness).

### Hard gates

- No acceptance of altered or weakened theorem as the original theorem.
- No unsupported formal fact claims.
- No fabricated dependency.
- A correct abstention scores better than a false mathematical explanation.
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
- N randomized / completed / excluded;
- participant background stratum;
- statement-fidelity score;
- mechanism-comprehension score;
- near-transfer score;
- counterfactual accuracy and counterexample validity;
- false-fact / unsupported-claim rate;
- confidence calibration;
- median / p95 time;
- explanation tokens;
- reviewer minutes;
- effect sizes and confidence intervals;
- protocol version, randomization seed and source/task hashes.

**No human study has been run.** This document is a preregistration template awaiting expert and ethics review.
