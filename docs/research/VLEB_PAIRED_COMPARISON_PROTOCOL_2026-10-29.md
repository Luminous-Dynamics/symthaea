# VLEB paired comparison and uncertainty protocol — 2026-10-29

## Why this increment exists

A leaderboard's scalar pass rate does not tell us whether two systems solve the same tasks, whether a small gap is distinguishable from sampling noise, or whether an apparent gain comes from the model versus its scaffold. The public September 2026 SWE-bench audit found no statistically significant separation for any of the 29 adjacent top-thirty Verified pairs under its exact paired McNemar tests; it also reported within-model scaffold ranges as large as 29.8 percentage points against an 8.8-point spread across the top thirty. Those are benchmark-specific observational findings, not a universal law, but they are a strong warning against declaring a winner from small unpaired score differences ([paper](https://arxiv.org/abs/2609.17394)).

Task validity is a separate issue. SWE-Bench Pro V2's September 22, 2026 refresh removed 89 invalid tasks, corrected 69 instruction/test conflicts, restricted network access to the model endpoint, re-graded patches on pristine images, and adopted a two-sided reference-patch/empty-patch gate ([Scale Labs release](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2)). OpenAI's July 2026 audit of the earlier SWE-Bench Pro revision reported 200 tasks flagged by an automated pipeline and 249 identified by human review; this refers to the older revision, not V2 ([audit](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)). The ICML 2026 COBA work likewise treats benchmark validity as a multi-component problem spanning user, environment, ground truth and evaluation ([PMLR paper](https://proceedings.mlr.press/v306/suh26a.html)).

VLEB already has draft task admission, per-test baseline expectations, exact execution identity, cohort integrity and auditor-mutant checks. This increment adds a distinct layer: **paired statistical analysis of per-task agent outcomes**. It does not replace those validity controls.

## Analysis contract

The new tool, tools/vleb/compare_paired_runs.py, accepts one evaluator-pinned JSON manifest that freezes:
- the exact task inventory and each task-contract digest;
- a reference arm plus comparison arms, with model/version and scaffold digest;
- the seed set, with one primary attempt for every task × seed × arm cell;
- the protocol digest, candidate patch digest, receipt digest, independent audit-report digest, expected source revision, immutable image, network mode, run ID, cost and duration;
- a preregistered 95% confidence level, bootstrap replicate count and deterministic bootstrap seed.

The program rejects a missing or duplicate cell, a replayed run/receipt/audit digest, an unpinned task contract, a mismatched protocol, retry substitution, or a difference in task contract, revision, image or network mode among paired arms. It fails closed; it does not silently drop a task or run.

## Estimands and uncertainty

The headline success rate is successful scheduled task-seed trials divided by the complete locked task × seed denominator. Fail and unresolved are both non-successes for this rate; unresolved trials are also reported explicitly. This avoids turning missing observations into an artificial improvement.

For every arm, the analyzer reports success/failure/unresolved counts, success rate, task-cluster bootstrap interval, total cost, cost per successful trial, and p50/p90 duration. For each non-reference arm it reports the paired score difference in percentage points, how many task clusters favored/lost/tied the comparison arm, and a 95% paired task-cluster bootstrap interval. It additionally reports an exact two-sided task-win sign test with Holm-adjusted p-values across all non-reference arms in the frozen manifest, separate from the effect-size interval. It also publishes an outcome-overlap table: both-success, reference-only-success, comparison-only-success and both-non-success trial cells. `resolved_disagreement_cells` counts only pass/fail disagreements where neither side is unresolved, while `unresolved_pair_cells` counts every pair containing an unresolved observation. The success-disagreement total is broader and may include unresolved-side cells, so the resolved and unresolved counts must be read separately. `effective_comparison_task_cluster_count` counts task clusters whose mean success rate differs between the two arms; it is not an effective sample size in the statistical-design-effect sense.

The resampling unit is task_id, not the individual run. Each bootstrap draw carries all preregistered seeds and all paired arms for a selected task. That design avoids treating repeated seeds on the same task as independent task evidence. The interval reflects variation across sampled task clusters conditional on the declared seed set; v1 does not separately estimate seed-to-seed or model-serving variance, nor does it stratify clusters by task family or difficulty. The fixed seed and replicate count make the analysis reproducible. With fewer than 30 task clusters, the report emits a small-cluster warning; a serious confirmatory claim should normally use a substantially larger task cohort and a preregistered precision/power analysis.

Intervals are descriptive percentile-bootstrap intervals and are not simultaneous across multiple arms. The analyzer also reports an exact two-sided task-level sign test on task clusters where the comparison arm's mean success rate differs from the reference: it tests whether wins and losses are equally likely among non-tied tasks. This is a directional-consistency test, not a test of the magnitude of the mean score difference. Its raw p-values are Holm-adjusted across every non-reference arm frozen in the manifest; an arm with no non-tied tasks has a null raw p-value and is conservatively assigned p=1 for that family adjustment. The sign test assumes independent task clusters. Effect-size intervals remain unadjusted and multi-arm effect-size claims need a preregistered simultaneous interval or other justified inference plan.

## Integrity boundary

The CLI requires evaluator-supplied pins for the raw manifest SHA-256, protocol SHA-256 and analyzer source SHA-256. Its synthetic test suite also exercises the positive CLI path, persisted report equality and fail-closed behavior for a wrong external manifest pin. This binds the run to named inputs and the intended analyzer, but the analysis program does not authenticate the receipt or audit-report bytes referenced by those digests, run the candidate, verify signatures, or prove that the declared outcomes were honestly measured. The trusted evaluator must create and pin the manifest from independently verified receipts and audit reports. Hashes alone are not signatures or execution attestations.

Output permanently declares analysis_only=true and score_eligible=false. Synthetic tests validate math and rejection behavior only; they are not benchmark measurements.

## Proposed research targets

1. **100% pair coverage:** every locked task × seed × arm is represented exactly once; 0 missing or duplicated cells.
2. **0 identity drift:** 100% agreement on task-contract digest, protocol, revision, immutable image and network mode within each paired task-seed cell.
3. **0 denominator attrition:** unresolved trials remain visible and count as non-successes in the headline denominator.
4. **Reproducibility:** identical pinned inputs and analyzer source reproduce identical point estimates and intervals.
5. **Resolution, not rank theater:** publish paired differences, interval widths and per-task overlap. Size confirmatory cohorts against a preregistered effect size and desired precision; do not invent a universal task-count threshold.
6. **Matched cost-quality frontier:** compare cost per verified successful trial and p50/p90 runtime only under matched budgets, scaffold/tool policy, environment and task distribution.
7. **External SOTA discipline:** public Legacy-Bench, SWE-Bench Pro V2, security-specific benchmarks and VLEB are different instruments. Do not compare raw percentages across them or claim superiority without a protocol-matched run and independently audited evidence.

## Command shape

The evaluator should obtain the three expected digests from its trusted run configuration, not trust values declared only by the manifest. Invoke the tool with the input path, evaluator-pinned input SHA-256, protocol SHA-256 and analyzer source SHA-256; an optional output path writes the report. The test fixture in test_compare_paired_runs.py is synthetic and is intended only to validate the code path.

No VLEB agent score or industry-relative performance is claimed by this increment.
