# EGDDM ShiftBench D6.5 — preregistered statistical analysis and decision rules

**Version:** 0.1  
**Date:** 2026-11-15  
**Status:** statistical design template authored. This is not a signed preregistration, powered sample-size determination, executed study or benchmark result.

## 1. Purpose

D6.5 prevents a common failure in agent evaluations: declaring a method better because it spends less while silently losing more shifted tasks, accepting stale evidence or leaving critical obligations unverified.

The primary question is:

> Does full EGDDM reduce end-to-end engineering cost relative to the strongest preselected baseline while remaining non-inferior on shifted recovery and critical unverified exposure?

This is a constrained efficiency claim, not a single leaderboard score. A method is not superior if it saves compute by taking greater critical risk.

## 2. Why current software benchmarks are not sufficient comparators

Static issue-resolution results are important capability context, but they cannot answer D6's longitudinal question. The current Scale SWE-Bench Pro V2 full leaderboard reports Opus 5 at 99.40±0.40 and Gemini 4 Argon at 99.22±0.45; this is near-ceiling performance on that benchmark's issue-resolution metric. It is not a D6 comparator score because its tasks, evaluation protocol and outcome are different. See [Scale's current leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full).

SWE-Shift is much closer to D6's dynamic-intervention question: its 120 paired scenarios cover 12 change families at nine positions, and it reports substantial clean-to-shifted degradation across 17 systems. ChainSWE adds longitudinal multi-bug chains; SWE-Gate measures review constraints beyond functional tests. D6 must compare against these as related research, not claim a head-to-head win unless the same tasks, models, harnesses and budgets are rerun.

References:
- [SWE-Shift](https://arxiv.org/abs/2610.32196)
- [ChainSWE, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.371/)
- [SWE-Gate](https://arxiv.org/abs/2609.04167)
- [SWE-Bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full)

## 3. Study design and units

The paired experimental unit is \`scenario_pair_id\`: a clean and shifted continuation from the same frozen initial task/workspace state. The paired unit is evaluated within the same pinned model, agent harness, tool configuration, repository block and trial.

The proposed final research corpus is:
- 12 intervention families;
- nine planned-work positions: p10, p20, p30, p40, p50, p60, p70, p80 and p90;
- five independently authored seeds per family/position cell;
- 540 matched clean/shifted scenario pairs;
- 1,080 trajectories per model per trial;
- three independent trials where compute permits;
- at least three model/harness families.

Early (10–30%), middle (40–60%) and late (70–90%) are reporting aggregates, not the experiment's actual position resolution. The 108 family/position cells are first a validity pilot: one candidate task seed per cell plus controls. The full target would imply 9,720 trajectories across three model/harness families and three trials, before baseline and expert runs. The pilot is a validity stage, not a powered final evaluation. D6.4 task-admission gates must pass before agent scores are collected.

### Blocking and holdouts

Block and report by:
- repository family;
- intervention family;
- exact planned-work position percentile and its early/middle/late reporting aggregate;
- task seed;
- model/harness family;
- repeated trial.

Hold out entire repository families and a forward-time slice. The candidate task bundle, gold impact labels, acceptance contracts and grader images must be frozen before model runs. Do not use post-cutoff commits, tests, issue text or evidence as candidate-visible inputs.

The baseline set should include full rerun, reset/restart, conversation-only memory, static dependency traversal, D2 temporal replay, D2+D3 evidence acquisition and expert-maintainer predictions. Select the strongest eligible primary comparator using the validation pilot, then freeze that comparator before held-out evaluation. Keep ablations visible; do not select a different comparator after seeing final outcomes.

## 4. Primary estimand: efficiency under risk and recovery constraints

The primary claim is a three-part constrained comparison of full EGDDM against the frozen primary comparator:

1. **Shifted recovery non-inferiority:** shifted task success must not fall beyond a predeclared absolute margin.
2. **Critical-risk non-inferiority:** critical unverified obligation-seconds must not increase beyond a predeclared margin. Report worst critical gap as a co-equal safety guardrail.
3. **Cost superiority:** when both preceding gates pass, total end-to-end cost must be lower at the predeclared confidence level.

Non-inferiority margins are intentionally null in the template. They must be selected from stakeholder-approved risk tolerances and the validation pilot, then frozen and signed **before held-out data are unblinded**. Do not invent the margins after observing the candidate result. The template is not ready for unblinding until both are set.

The cost denominator includes more than tokens: model input/output tokens, agent wall time, tool calls, retries, runner compute and wall time, external service cost, reviewer time and adjudicator time. Report each cost component, not only the sum.

## 5. Outcomes

### Primary outcomes

- Shifted recovery success.
- Critical unverified obligation-seconds.
- Total end-to-end cost.

### Co-equal hard gates

The following cannot be traded against throughput or cost:
- critical unauthorized action;
- candidate access to gold labels, hidden tests or evaluator credentials;
- future-evidence leakage;
- evaluator tampering;
- stale receipt accepted for a critical obligation;
- unknown execution outcome followed by a new mutation without reconciliation.

A candidate triggering a hard gate is reported as a security/failure event regardless of its aggregate task success.

### Secondary outcomes

Report:
- clean success and paired clean/shifted success;
- retained functionality and terminal active-requirement conformance;
- time-weighted conformance and maximum critical unverified gap;
- stale receipt acceptance and future evidence leakage;
- impact precision/recall and confidence calibration (Brier score);
- sham false-positive invalidation and positive-control detection;
- unaffected evidence reuse at matched critical coverage;
- revalidation runner cost, recovery latency and pass-all-trials reliability;
- risk/coverage curve for abstention or requests for independent evidence.

Never hide a weak intervention family or late-position failure inside a single overall average. Publish all 12 family × 9 exact-position strata, plus early/middle/late aggregates, exact denominators, excluded tasks and exclusion reasons.

## 6. Statistical methods

- Use a paired exact McNemar test for binary shifted-success differences.
- Estimate paired continuous differences and ratios with a hierarchical cluster bootstrap. Resample task-seed clusters while preserving repository-family and intervention strata. The target is 10,000 bootstrap replicates.
- Preserve task-level pairing and repeated-trial structure. Three runs of the same task are not three independent tasks.
- Use 95% confidence intervals and familywise alpha 0.05 for the preregistered primary analysis.
- Target at least 90% power for the primary estimand. Determine the final sample size by simulation using pilot estimates of discordance, variance, clustering and missingness. If the 540-pair target is not sufficient, disclose the power limit and do not claim superiority; expand only under the frozen rules.
- Apply Holm-Bonferroni correction to declared secondary comparisons.
- Report pass@1, each independent trial, and pass-all-three reliability. Do not publish only best-of-N.
- Missing or timed-out candidate runs count as shifted failure; retain incurred cost and report censoring/missingness separately. A missing grader result is not a pass.
- No post-hoc subgroup selection, threshold tuning or benchmark task deletion after unblinding. Exclusions follow the frozen task-admission rules and are published.

## 7. Cost and reliability: compare on a Pareto frontier

Do not invent a weighted scalar where high task success can compensate for a critical evidence failure. Report at minimum the frontier across:
- shifted recovery success;
- critical unverified exposure;
- end-to-end cost;
- maximum critical gap;
- stale-evidence and control failure rates.

EGDDM can claim an efficiency advantage only if its cost point is lower while its recovery and critical-risk constraints are satisfied. If it is cheaper but less safe, the result is a trade-off, not superiority.

For impact prediction, add calibration and selective-prediction evaluation. When uncertain, the system should abstain or seek evidence rather than confidently declare an obligation unaffected. Report risk versus coverage so the benchmark rewards calibrated refusal to reuse stale evidence.

## 8. Preregistration lock

Before unblinding, archive and sign:
- the primary comparator;
- non-inferiority margins and the risk rationale;
- model/harness versions and tool scaffolding;
- final sample size and simulation-based power report;
- task exclusion rules and stopping rule;
- grader/runner/network and hidden-test digests;
- analysis-code digest and randomization seed;
- treatment of missing runs and protocol deviations.

Use two-person approval for changes after the lock. Protocol deviations must remain visible and trigger sensitivity analysis; do not silently rewrite the analysis.

The machine-readable template and schema are:
- \`egddm-shiftbench-analysis-plan-v0.1.template.json\`
- \`egddm-shiftbench-analysis-plan-v0.1.schema.json\`

The current template has null margins, no signatories, no analysis-code digest and no final sample-size/power calculation. It is an incomplete design artifact, not a preregistration.

## 9. Acceptance criteria for a credible improvement claim

A claim of superiority requires:
1. D6.4 task-admission controls pass, with no unresolved task-validity or custody defect.
2. Gold labels and evaluation remain blinded until lock.
3. Candidate and comparator are run at matched model/harness/tool/resource settings.
4. Paired analysis passes shifted-recovery and critical-risk non-inferiority.
5. Cost superiority is supported by its confidence interval, including reviewer/adjudicator effort.
6. Hard-failure rates are zero in the accepted run set; every planted negative control is rejected.
7. Results reproduce on held-out repository families and a forward-time slice.
8. Independent replication confirms the result before public SOTA language.

If one condition fails, report the observed result without a SOTA/superiority claim.

## 10. Limits and current status

The analysis plan is authored as a machine-readable template and research protocol. It is not signed or locked; non-inferiority margins and sample size remain unset. No tasks have been admitted or collected, no agent trials have run, and there is no empirical EGDDM performance or cost result.

**Claim ceiling:** this work specifies how to evaluate a claim. It does not establish that EGDDM is better than SWE-Shift baselines, industry systems, or SOTA.
