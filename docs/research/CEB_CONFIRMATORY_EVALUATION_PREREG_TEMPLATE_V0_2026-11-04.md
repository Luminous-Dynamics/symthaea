# CEB Confirmatory Evaluation Preregistration Template V0

**Status:** design-only template; not an approved or executed preregistration  
**Date:** 2026-11-04  
**Related:** #10366, #10385, #10115, #10173, #10314, #10369, #10370  
**Claim ceiling:** a completed form freezes an experiment; it does not itself qualify tasks, graders, execution isolation or evidence authenticity.

## 1. Why this document exists

The initial 24–30 CEB task specifications are a pilot for task validity, evidence contracts, rubric clarity and effect-size estimation. They are not automatically a statistically powered comparison.

The unit hierarchy must be explicit:

- Repository / environment family.
- Root-cause or task-generation template.
- Task instance / paired variant.
- Stochastic run seed.
- Tool / model trajectory.
- Graded outcome and evidence receipt.

Near-identical variants generated from the same template are not independent samples. Repeated seeds are not new task instances. One repository containing many injected variants must not dominate the effective sample size.

## 2. Required registration fields

Before confirmatory evaluation, record:

### Identity and provenance
- Protocol ID and version.
- Benchmark lane(s) and task-set digest.
- Repository/template family IDs and inclusion rationale.
- Task, source snapshot, environment image, toolchain, dependency and grader digests.
- Model identifier/revision, provider configuration, harness/scaffold commit, system prompt digest and tool registry digest.
- Network policy, execution isolation tier and trust-policy / signer assurance tier.
- Budget for wall time, turns, tool calls, tokens, monetary cost, CPU, memory and disk.

### Research question
- Primary hypothesis and estimand.
- Strongest matched baseline, selected before confirmatory data is observed.
- Smallest effect size of practical interest.
- Safety/non-inferiority bounds for regression, false acceptance and harmful side effects.
- Primary metric definition, numerator, denominator and treatment of missing / unresolved cases.

### Design
- Task inclusion/exclusion criteria.
- Randomization unit and assignment procedure.
- Pairing variables and blocked factors.
- Number of independent task-template / repository clusters.
- Runs per task/configuration and seed policy.
- Model × scaffold factorial terms to be estimated.
- Stopping, retry, timeout and censored-run policies.

### Analysis
- Primary estimator and confidence interval.
- Cluster level for bootstrap / robust variance.
- Handling of multiple primary outcomes.
- Predeclared subgroup analysis.
- Calibration method and held-out calibration split.
- Missingness / exclusion ledger.
- Exact report artifact and independent regrade procedure.

## 3. Pilot versus confirmatory stages

### Stage A — design pilot

Initial proposal: 24–30 unique task specifications across at least six fault families. Use this phase to:
- verify task solvability and negative-control validity;
- find ambiguities and grader edge cases;
- estimate outcome variance, task difficulty and between-template heterogeneity;
- test receipt completeness and analysis code;
- revise the protocol before registration.

Pilot results are descriptive only. Do not claim SOTA, statistically significant improvement or generalization from this sample.

### Stage B — confirmatory evaluation

Build a fresh sealed set from independently authored repository and root-cause templates. Determine sample size from the pilot's paired discordance and cluster variance, using the smallest practical effect and desired power selected in the preregistration.

Do not set task count by copying the number of task rows or generating many near-identical perturbations. Ensure the effective cluster count is adequate. A confirmatory benchmark should cover multiple independent repositories and root-cause families; report the actual number of independent clusters and sensitivity to cluster removal.

If the available cluster count cannot support the intended claim, narrow the claim or report the study as exploratory.

## 4. Matched treatment design

For each task and seed, compare as appropriate:

1. deterministic tool / classical algorithm baseline;
2. foundation model with a minimal fixed harness;
3. same model with a strong coding-agent harness;
4. same model plus Symthaea's repository twin, causal experiment selection and evidence workflow;
5. Symthaea-native generation path where genuinely implemented.

Hold fixed everything except the named factor. Use identical task instances and resource caps. Randomize run order; where practical, use paired seeds and identical initial state. Do not compare one system's best run against another system's single run.

For model × scaffold questions, use a factorial or balanced fractional-factorial design. Do not attribute a gain to Symthaea when the model, prompt, budget, tools or environment changed simultaneously.

## 5. Statistical methods

### Binary task outcomes

- Define the primary outcome at task-run level (for example, independently accepted repair).
- For matched binary outcomes, report paired discordance and a paired interval/test such as McNemar's test where its assumptions fit.
- For nested task/repository/run data, use a mixed-effects model or cluster-robust inference with repository/template and task structure represented.
- Bootstrap at the highest independent unit that matches the generalization claim (typically repository or root-cause template), not individual near-duplicate variants.
- Report absolute percentage-point difference, relative effect, interval, sample size and per-cluster distribution.

### Continuous and cost outcomes

- Report median, mean, spread, quantiles and confidence intervals for wall time, cost, experiments, tokens, memory and performance.
- Use paired, blocked measurement for runtime optimization; report baseline noise and candidate measurement noise separately.
- For cost per accepted outcome, include all eligible attempts and failed/abstained outcomes in the denominator or explicitly report conditional and unconditional versions.

### Multiple outcomes

Predeclare one primary endpoint per confirmatory claim. Use a hierarchy or familywise correction (for example Holm) for multiple confirmatory tests. Secondary and exploratory metrics remain labeled as such.

### Calibration and selective risk

- Report Brier score and log loss alongside reliability plots.
- ECE is secondary and must state binning, sample count and uncertainty; small samples can make it unstable.
- For abstention, report risk/coverage curves, paired act/abstain accuracy, false-action and false-abstention rates, and each scenario stratum separately.

### Pass@k and reliability

Pass@k asks whether at least one of k attempts succeeds. It does not establish repeatability. If reliability matters, report pass-all-k / repeated-success probability separately, with exact run counts and uncertainty.

## 6. Slice-specific estimands

### Precise debugging
- Primary: independently accepted repair rate on natural repository tasks.
- Controlled seeded-bug secondary outcomes: bug recall, edit precision, mutation score and protected-behavior regression rate.
- The admissible repair set must be defined independently and allow semantically equivalent fixes.
- Do not use exact patch overlap as the natural-task correctness criterion.

### Causal diagnosis
- Primary: independently accepted repair rate versus the strongest matched non-Symthaea baseline.
- Secondary: root-cause top-1/top-3, calibrated hypothesis score, pre-registered predicted-outcome score, discriminating experiment yield, cost per accepted repair and correct abstention.
- A causal claim requires intervention evidence or a justified identification argument; temporal sequence and correlation alone are insufficient.

### Performance engineering
- Primary: semantic-equivalent performance improvement under the pinned workload.
- Compare candidate speedup with validated expert speedup on the same environment.
- Report absolute runtime, candidate/expert speedup ratio, memory, allocations, throughput and tail latency; energy only where measurement quality is adequate.
- Reject candidates with correctness/security regressions even if faster.

### Selective escalation / abstention
- Use minimal-difference task pairs where one version should act and its twin should ask/abstain.
- Primary: paired act/abstain accuracy; secondary: question precision, blocker recall, ASK-F1, task pass after answer, false-action rate and coverage.
- Count silent non-action and post-hoc apology after a side effect as incorrect abstention.

## 7. Missingness and task accounting

No outcome may silently disappear. Retain rows for:
- pass;
- fail;
- refusal;
- abstention;
- unsupported capability;
- tool failure;
- timeout;
- ungraded;
- not attempted;
- invalidated task;
- censored or interrupted run.

An invalid task is removed only under a predeclared, independently adjudicated rule and remains visible in the exclusion ledger. Keep both eligible-universe and attempted-task denominators. Do not substitute a smaller denominator without disclosure.

## 8. Benchmark integrity gates

- Reference solution must pass and empty patch must fail before task admission.
- Known-bad and regression-inducing patches must fail.
- Hidden answer keys and grader authority stay outside agent access.
- Task and grader versions are pinned before the run.
- Score counters are recomputed from raw task rows.
- The primary result is independently regraded from a fresh immutable snapshot.
- Trust and isolation claims are restricted to their qualified assurance tiers.
- Any verifier defect discovered after evaluation triggers impact analysis and score invalidation/reissue, not a quiet patch to historical numbers.

## 9. Stop and decision rules

Stop an investigation when:
- the task contract is satisfied with sufficient independent evidence;
- remaining experiments cannot change the decision within the preregistered budget;
- no safe identifying experiment is available;
- the evidence is non-identifying or contradictory;
- the time/resource budget is exhausted.

Do not stop a benchmark early merely because one arm is winning. Any sequential looks, alpha-spending or early-stopping boundaries must be preregistered.

## 10. Release checklist

Before publishing any confirmatory result, attach:
- completed preregistration;
- task and environment manifests;
- exact model/harness/tool/configuration digest;
- raw per-task ledger and all dispositions;
- analysis code and randomization record;
- paired and cluster-level statistical report;
- positive and negative verifier controls;
- primary and pristine-regrade evidence;
- explicit claim ceiling and assurance tier;
- benchmark source revision and retrieval date.

Until #10064 and #10369 satisfy their respective gates, the CEB programme remains in the non-executable design tranche. This template cannot authorize hostile-code execution.
