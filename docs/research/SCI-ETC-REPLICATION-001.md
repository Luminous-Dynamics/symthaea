# SCI-ETC-REPLICATION-001 — Budget-Matched Replication of Explore-Then-Commit

**Status:** research protocol and manifest shape; no Symthaea trial results.

**Tracking issue:** [#8804](https://github.com/Luminous-Dynamics/symthaea/issues/8804).

## 1. Motivation and prior result

NewtonBench evaluates interactive scientific-law discovery across 324 tasks and 12 physics domains. Its original study found that performance is sensitive to law/system complexity and observation noise, and that an interpreter can sometimes encourage premature exploitation: https://arxiv.org/abs/2510.07172.

The October 2026 preprint *Explore, Then Commit: Measurement-Efficient Scientific Law Discovery with Language Models* reports 576 NewtonBench trials and separates the acquisition phase from a fresh final law-synthesis prompt: https://arxiv.org/abs/2610.07620.

Reported medium-task results for interpreter-enabled planner configurations:

| Model alias in paper | Baseline measurements/trial | Planner-family measurements/trial | Baseline mean magnitude-RMSLE | Planner-family mean magnitude-RMSLE |
|---|---:|---:|---:|---:|
| GPT-4.1-mini | 22.5 | 8.6 | 2.514 | 0.202 |
| GPT-4.1 | 43.0 | 8.9 | 0.626 | 0.149 |

These are results for the *bundled protocol configurations*, not clean causal estimates for variance/disagreement acquisition. The paper notes unequal baseline/planner batch constraints, inconsistent module-level symbolic recovery, random acquisition competitive with scored selectors, and a narrow primary task slice: vanilla direct-equation tier, law version v0, zero observation noise. Model aliases are not fully resolved to immutable hosted snapshots. The reported gains motivate a matched replication; they do not establish a Symthaea result or a state-of-the-art win.

## 2. Questions

1. Can we reproduce the native measurements and accuracy/error outcomes on the same pinned task cells?
2. Does the complete explore-then-commit protocol improve the budget–quality frontier when each arm has the same measurement and end-to-end resource limits?
3. How much does each component contribute: structured bootstrap, automatic diagnostics, hypothesis pool, acquisition, batch diversity, draft retention, interpreter access, and fresh synthesis?
4. Does random acquisition remain competitive with variance/pairwise scoring after all policy arms share the same candidate pool, filters, input domain, budget, and number of rollouts?
5. Do gains persist for complete predictions, symbolic recovery, noisy observations and embedded systems—or only for numerical fit on the narrow original slice?

## 3. Phase 0 — reproducibility and evaluator audit

Before launching model trials:

- Pin the NewtonBench repository revision, task module code, evaluator source, judge model/prompt/temperature, adapter revision, dependency environment, and all prompts/templates.
- Freeze model provider, immutable model snapshot where available, decoding settings, tool permissions, and policy code. If the provider only supplies a mutable alias, record the exact alias and access date and downgrade reproducibility claims.
- Reconstruct the native output schema and reproduce a small set of published raw examples without changing them. Preserve native RMSLE and Symbolic Accuracy semantics. The judge-derived Symbolic Accuracy field is not a formal equivalence proof.
- Add evaluator controls for non-finite predictions, partial-domain outputs that nevertheless yield finite native RMSLE, exceptions/timeouts, wrong equation dependencies, and missing result artifacts.
- Keep all expected task/arm/rollout cells in the denominator. Do not drop invalid, incomplete, failed, timed-out, or missing submissions.

Deliver a baseline-only compatibility receipt before any policy comparison. If the native evaluator or expected denominator cannot be reproduced, stop and resolve that issue first.

## 4. Phase 1 — equal-budget protocol comparison

### Required reference arms

1. **Native baseline, no interpreter.** Same task visibility and final submission contract as the benchmark's native unassisted arm.
2. **Native baseline, interpreter.** Preserve the original tool-enabled comparison as a reference, but not the sole comparator.
3. **Explore-then-commit, variance acquisition, no interpreter.**
4. **Explore-then-commit, pairwise acquisition, no interpreter.**
5. **Explore-then-commit, random acquisition, no interpreter.**
6. **Explore-then-commit, variance acquisition, interpreter.**
7. **Explore-then-commit, pairwise acquisition, interpreter.**
8. **Explore-then-commit, random acquisition, interpreter.**

The six planner arms share the bootstrap rules, numerical diagnostics, hypothesis pool, batch constraints, candidate input pool and finalization opportunity. Only the acquisition rule and interpreter availability vary as declared. Random acquisition is a required control, not an assumed weak baseline.

After the eight-arm replication, run separate preregistered ablations: diagnostics-only, structured-probe removal, fresh-synthesis removal, draft-retention policy, and batch-diversity removal. Each ablation must be budget-matched and must not replace the native reference arms.

### Two distinct matching profiles

Published results cannot isolate acquisition causality when measurement batch constraints differ. Run and report two separate profiles:

**A. Measurement-matched.** Use a fixed per-trial simulator-call budget grid (initial protocol proposal: 8, 16, 32, and 48 measurements where a task supports that many). All arms share the same feasible input domain, bootstrap/minimum-observation rules, hard total-call cap, and finalization allowance. If a task requires more than the declared budget for a valid minimum design, mark it `budget_infeasible` before unblinding, not as an agent failure.

**B. End-to-end interaction-matched.** Hold maximum model-action rounds, experiment batch size, total tokens when measurable, wall-clock cap, and finalization opportunities equal. Also report simulator calls used. This separates measurement efficiency from policies that simply receive more interaction opportunities.

Do not merge these two profiles into one average. A method may win under a simulator-call budget and lose under a token or latency budget.

### Enforce the batch limit in the adapter

All experiment proposals pass through a validated adapter with a hard maximum number of measurements per action and per trial. Reject malformed or over-budget requests and record the violation; do not silently truncate the action, because that changes the policy's observation and stopping process. Apply the same maximum to every arm. Preserve requested, accepted, and executed counts separately.

## 5. Randomization, pairing, and analysis

- Use the same task module, difficulty, system type, law version, initial information, and rollout seed across all arms in a paired block.
- Balance arms across module/order and execution time; randomize the order in which an arm is run to avoid provider/time confounds.
- Freeze the primary endpoint, practical minimum effect, sample-size/power plan, exclusions, missing-data rules, multiplicity correction and uncertainty method before confirmatory runs. Because the eight-arm grid yields multiple comparisons, preregister one or a small number of primary contrasts; report the remaining arm comparisons as secondary or control family-wise error.
- Begin with a small baseline-only pilot to estimate per-module variation and execution failure rates. Use the pilot to power the confirmatory comparison; do not tune the policy on a sealed confirmatory generation.
- Keep task-level/rollout-level data. Report paired differences and module-stratified effects; do not rely only on pooled means or two unpaired variants averaged into a family.
- Use independent rollout seeds and equal completed-attempt counts per arm where feasible. A run that never submits an equation remains an outcome, not an omission.

## 6. Metrics — separate the objectives

### Discovery and prediction

- **Native RMSLE:** preserve the official metric name and exact implementation. NewtonBench's magnitude-based form uses absolute values and can hide sign disagreements; audit sign separately.
- **Coverage-complete error:** recompute on a fixed shared evaluation grid and require finite predictions for every input with a finite target. Keep this distinct from native RMSLE.
- **Prediction completeness:** numerator/denominator of finite predictions over expected finite-target points; include finite-but-incomplete submissions as a separate failure class.
- **Symbolic Accuracy:** report the original judge label separately from deterministic checks and equation dependency diagnostics. Do not describe judge acceptance as formal equivalence.
- **Validated recovery:** report exact symbolic recovery where a deterministic equivalence check is available, plus judge-based recovery separately.

### Efficiency and operational reliability

- Simulator evaluations / measurements per trial, per accepted batch, and to first validated candidate.
- Model-action rounds, tool calls, input/output tokens, wall-clock latency, and compute/API cost where reliably measurable. Missing cost fields are not silently treated as zero.
- Failed/invalid/timeout trials, prediction completeness, and submission coverage.
- Cost to first complete, independently validated law, including failed experiments and invalid candidates.

### Comparative view

Plot a budget–quality Pareto frontier over simulator-call count, end-to-end cost, native RMSLE, coverage-complete error, prediction completeness, and symbolic recovery. No single composite score should allow fewer measurements to compensate for more confidently wrong laws or incomplete predictions.

## 7. Adversarial and negative controls

- **Random acquisition control:** always retain random feasible selection. Do not assume variance or pairwise disagreement must win.
- **Equal-batch control:** the adapter rejects any action that exceeds the declared batch size; report rejection count and zero unauthorized simulator executions.
- **Missing-output control:** submit finite predictions for only a subset of test inputs; native and coverage-complete scores must expose the difference.
- **Non-finite/error control:** exercise NaN, infinity, exception, timeout, and zero-output submissions.
- **Shortcut/control contamination:** rotate held-out law shifts and task instances; keep development examples separate from sealed refresh generations under SCI-BENCH-REFRESH-001.
- **Hypothesis dependence:** disagreement scores are heuristics unless their candidate predictive distributions are calibrated; report them as heuristic disagreement, not mutual information or expected information gain.
- **Tool leakage:** evaluator-only truth, expected equations, task family labels and test outcomes must never enter the model-visible context.
- **Budget abuse:** verify that repeated, duplicate, malformed, or rejected experiments consume/record the same resources as the declared adapter semantics require.

## 8. Extension tracks

Only after the narrow replication is qualified, extend as separate predeclared benchmark families:

1. Noisy observations and heteroscedastic/correlated noise.
2. Target laws embedded in simple and complex systems.
3. Held-out structural/operator shifts and out-of-domain prediction.
4. Explicitly unidentifiable worlds requiring calibrated uncertainty or abstention.
5. Sealed adaptive refresh generations to test benchmark saturation and generalization.

Results across these tracks are not pooled into one scalar. Synthetic and NewtonBench results do not establish real-world scientific discovery; real scientific workflows and independent experimental validation remain separate evidence requirements.

## 9. Prospective success criteria

These are gates for future preregistration, not results:

- Native scores and failure coverage replay correctly on a pinned reference fixture.
- All primary arms obey the same declared cap in the relevant matched-budget profile; no arm receives a larger batch or more oracle calls by accident.
- Any acquisition-policy superiority claim has a preregistered paired analysis and is not contradicted by module-level regressions hidden in the pooled mean.
- A measurement-saving claim is reported separately from symbolic-recovery and completeness claims.
- The candidate improves the budget–quality Pareto frontier against the strongest eligible baseline or provides a defensible trade-off; do not require variance/pairwise acquisition to beat random on every cell.
- Model identity, source/config revisions, prompts, evaluator/judge, seeds and all outputs are recorded sufficiently for independent replay.

## 10. Claim ceiling

This plan is designed to test the published protocol and its limitations under fairer budget controls. It does not show that Symthaea has reproduced the paper, that any component causes the reported gains, or that the method generalizes beyond the original zero-noise direct-output setting. Those claims require a completed, independently reviewed trial set.