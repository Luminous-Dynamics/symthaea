# Capability Amplification Protocol

**Status:** research extension proposal, 15 November 2026  
**Primary question:** does collaborating with an AI make a participant more capable of doing related work independently afterward?

## 1. The paradigm shift

Most human-AI evaluations measure the quality or speed of the artifact produced during the interaction. That is necessary but insufficient.

A system can produce a better result by replacing the participant's work. It can also help a participant learn a reusable method, develop a better mental model, improve future performance, and eventually need less assistance.

These outcomes are different:

- **Task substitution:** the system performs the work for the participant.
- **Task assistance:** the system helps complete the current task.
- **Capability amplification:** the participant becomes better able to perform related tasks independently.
- **Capability transfer:** the participant applies the learned method in a new but structurally related context.
- **Retention:** the participant still has the capability after an interval without the AI.

SympoeticBench should measure these separately. The research objective is not to maximize dependence on the collaborator; it is to expand human and joint capability while preserving agency.

## 2. Research questions

- **RQ1 — Immediate learning:** does reciprocal co-creation improve independent post-task performance beyond matched human-only and conventional-assistant conditions?
- **RQ2 — Retention:** does the participant retain the capability after a preregistered delay?
- **RQ3 — Transfer:** can the participant apply the method to a new task with a different surface form, context, or fault pattern?
- **RQ4 — Mechanism:** do explanation, graduated hints, reflection, and participant-led reasoning predict independent capability gain better than answer delivery?
- **RQ5 — Agency:** does the participant experience greater capability and control, rather than merely greater reliance on the AI?
- **RQ6 — Distribution:** which participants and task types benefit, and where does AI scaffolding fail or widen capability gaps?

## 3. Experimental conditions

Use randomized assignment or a counterbalanced within-participant design where domain and carryover effects permit:

1. **Human-only:** baseline and training tasks completed without generative AI.
2. **AI-only:** AI completes the task; this is a task-output comparator, not a human-learning condition.
3. **Conventional assistant:** the AI answers direct requests with limited scaffolding.
4. **Model-led solution:** the AI demonstrates a complete solution before the participant attempts the next task.
5. **Reflective human-led:** the AI asks questions, gives bounded hints, and keeps the participant responsible for the reasoning.
6. **Reciprocal co-creation:** participant and AI jointly formulate the problem, test hypotheses, explain reasoning, and revise the method.

Match time, information, tools, task difficulty, and practice opportunities. Record the amount and type of assistance. If conditions differ in practice dose, include it as a covariate or do not claim a clean causal comparison.

## 4. Assessment design

Each participant completes:

- **Pretest:** an independent baseline task using a frozen rubric.
- **Collaboration/training phase:** one or more practice tasks. The condition determines the interaction style.
- **Immediate post-test:** a non-identical but difficulty-matched task, completed without AI assistance.
- **Delayed retention test:** a new non-identical task after a preregistered delay.
- **Transfer test:** a task in a new context that requires the same underlying principles but not the same surface pattern.

The confirmatory study must freeze task forms and rubrics before data collection. Pilot forms should not appear in the confirmatory set. Avoid verbatim reuse of demonstration examples, and blind raters to condition and system identity where possible.

The independent tests must be genuinely independent: no AI answer, hints, tool use, hidden prompting, or agent-generated feedback may be delivered during an assessment phase.

## 5. Primary estimand: capability amplification

Let each score be on a bounded, common scale such as [0, 1]. For participant (i), define gain as post-test score minus pretest score.

The primary estimand is a difference-in-differences:

[
Delta_{mathrm{cap}} =
(overline{Q}_{mathrm{collab,post}}-overline{Q}_{mathrm{collab,pre}})
-
(overline{Q}_{mathrm{control,post}}-overline{Q}_{mathrm{control,pre}})
]

A positive value means the collaboration condition produced greater independent capability gain than the matched control. Use a preregistered primary comparator, and report all other condition contrasts separately.

The Rust implementation `participant_capability_amplification` performs a seeded independent-arm participant-cluster bootstrap over within-participant pre/post gains. The bootstrap resamples participants within each randomized arm; it does not treat turns, hints, or assessment questions as independent participants.

### Secondary estimands

- **Delayed retention gain:** the same difference-in-differences using delayed independent scores.
- **Transfer gain:** the same difference-in-differences using the held-out transfer task.
- **Retention fraction:** retained gain divided by immediate gain, only where the immediate gain is positive and the denominator is stable. Report undefined/negative cases explicitly.
- **Metacognitive calibration:** agreement between participant confidence and independently scored correctness.
- **Assistance efficiency:** capability gain per unit of human time and assistance exposure, reported alongside—not instead of—absolute gain.
- **Agency and self-efficacy:** participant-reported control and confidence, kept distinct from measured task capability.

Do not infer durable learning from a good joint artifact or one immediate post-test.

## 6. Measurement safeguards

- Use equivalent but non-identical task forms and counterbalance form order.
- Preregister the smallest effect of interest, primary outcome, delay interval, exclusions, missing-data handling, and statistical model.
- Use independent raters and report inter-rater agreement.
- Track time, hints, explanations, examples, tool use, and participant practice exposure.
- Score reasoning process as well as the final answer where the domain warrants it.
- Distinguish recall of a demonstrated answer from transfer of a generalizable method.
- Report subgroup results only when sample sizes and consent support privacy-preserving analysis.
- Do not use the evaluated AI as the sole author, tutor, judge, or source of ground truth.
- Give participants a right to stop, correct, or withdraw, and do not silently retain learning profiles beyond the approved scope.

## 7. Provisional research targets

These are hypotheses and design targets, not validated universal standards or achieved results:

| Measure | Initial target |
|---|---|
| Immediate capability amplification | Positive treatment-control gain; 95% interval excludes zero |
| Delayed retention | Positive treatment-control gain at the preregistered delay |
| Transfer | Positive treatment-control gain on a held-out context |
| Assessment integrity | Zero AI assistance during independent test phases |
| Agency | Non-inferior to reflective human-led condition |
| Assistance efficiency | Report cost and time with absolute capability gain; no efficiency-only victory |
| Reproducibility | Independent replication with new participants and held-out task forms |

The study must be powered from pilot variance and the declared smallest effect of interest. Do not select sample size or thresholds after observing confirmatory results.

## 8. Coalition attribution: who contributed to a joint capability?

For multi-person or multi-agent collaboration, exact Shapley attribution can describe how a coalition's value is allocated across contributors when every coalition outcome is identifiable.

For (n) contributors and coalition value function (v(S)), the Shapley value for contributor (i) is:

[
phi_i =
sum_{S subseteq N setminus {i}}
rac{|S|!(n-|S|-1)!}{n!}
left[v(S cup {i})-v(S)ight]
]

The Rust implementation `exact_shapley_values` accepts the complete table of (2^n) coalition values for up to 12 contributors. It rejects missing, non-identifiable, non-finite, or out-of-range coalition values rather than imputing them. This is an attribution-allocation measure for a bounded task outcome, not proof of legal authorship or a measure of intrinsic human worth.

For larger teams, a sampled Shapley estimator may be added as a separate versioned metric with its sampling error and replay budget reported. Coalition effects are not assumed additive.

## 9. Seed scenario

`SP-LEA-001` in `SEED_CORPUS.v0.1.jsonl` is a development-only scenario using fictional water-distribution diagnosis. It contains a baseline, a scaffolded collaboration phase, an independent post-test, a delayed retention check, and a transfer task. Its purpose is to test the harness and rubric—not to claim that any system has already improved human capability.

## 10. Reporting

A capability-amplification report should include:

- Pretest, immediate, delayed, and transfer distributions by condition.
- Difference-in-differences point estimates and uncertainty intervals.
- Participant counts by arm, attrition, missing data, and exclusions.
- Test-form equivalence and rater agreement.
- Assistance exposure, time, cost, and interruption burden.
- Agency, self-efficacy, and cognitive workload as separate outcomes.
- Task and scoring versions, prompt/model snapshots, environment digest, and analysis code.

The benchmark's distinctive claim should be modest and testable: it evaluates whether collaborative AI increases human independent capability, not just whether the AI makes better artifacts.
