# Informative missing outcomes: partial-identification protocol

**Status:** research proposal / test contract; no measured results  
**Scope:** extends the cognitive-loop adaptation benchmark and issue [#8793](https://github.com/Luminous-Dynamics/symthaea/issues/8793)  
**Related work:** [#8495](https://github.com/Luminous-Dynamics/symthaea/pull/8495), [#8766](https://github.com/Luminous-Dynamics/symthaea/issues/8766), [#8764](https://github.com/Luminous-Dynamics/symthaea/issues/8764)  
**Research motivation:** delayed and partially labelled streams are a recognized evaluation problem; see the 2026 experimental survey, [*Concept drift detection in delayed and partially labeled data streams*](https://doi.org/10.1016/j.dsp.2026.106325), and the 2026 latency-aware adaptation paper [record](https://pubmed.ncbi.nlm.nih.gov/42862614/).

## The paradigm shift

Do not treat the rows whose labels arrived as a random sample by default.

When outcome availability depends on the outcome, prediction error, intervention, risk, or a hidden operational process, the observed-label average can rank policies incorrectly. More drift alarms, more revealed labels, and lower error among observed labels are not individually evidence of higher real task value.

The benchmark must separate three things:

1. **What the policy was permitted to know at decision time.**
2. **What the evaluator can know after the registered horizon.**
3. **What a real deployment's observations identify without untestable assumptions.**

A simulation can keep hidden truth for evaluator-only validation. A real observational dataset cannot quietly use hidden truth or pretend that a missing outcome is zero. When the data do not identify a point estimate, the correct result is an interval, a sensitivity analysis, or *unidentified*—not a convenient point estimate.

## 1. Keep outcome states distinct

Each event must be assigned exactly one current outcome-availability state under the registered event ledger:

- **Observed:** a valid outcome has been revealed, with its value and reveal tick.
- **Delayed / pending:** no outcome is available yet; the declared deadline or evaluation horizon has not established permanent missingness.
- **Right-censored at horizon:** the event's outcome has not been observed by a named horizon. This does not assert that the outcome will never arrive.
- **Declared missing:** a registered source explicitly declares that the outcome will not be provided, at its declaration tick.
- **Unknown / unnotified:** no valid outcome and no valid missingness declaration is available. This must not be reinterpreted as a negative outcome or silent permission to infer the hidden target.

The policy receives only the event identity and permitted availability notice, at or after its recorded declaration time. It never receives the hidden target, future availability schedule, or the mechanism that generated missingness unless that information is itself an explicitly registered policy input. The evaluator may use simulator-only hidden truth after the run, but that truth must be provenance-separated from policy inputs.

Do not collapse delayed, right-censored, declared-missing, and unknown observations into one null value. They represent different operational and statistical conditions.

## 2. Preregister separate estimands

Every experiment must state its target before running.

### A. Decision-time operational value

What task loss, cost, or reward was actually available to the policy by the decision horizon? This measures deployable feedback and operational performance. It is not an estimate of eventual full-horizon quality unless the protocol justifies that interpretation.

### B. Full-horizon task loss

For each registered policy arm, what is the mean task loss over all assigned episodes, including outcomes not observed by the evaluation horizon? In a simulator, the independent evaluator may score the full hidden outcome after the policy run. For real data, this estimand may not be point-identified.

### C. Candidate-minus-reference difference

Use the declared direction consistently. For task loss, define

\[
\Delta = \mathbb{E}[L_{\text{candidate}}] -
          \mathbb{E}[L_{\text{reference}}],
\]

so negative values favor the candidate. Do not report "improvement" without naming the reference, outcome horizon, information-access track, and missingness estimand.

Do not merge these estimands into a single composite metric. An operational system might improve by collecting outcomes faster while its eventual task loss remains uncertain; that should be visible, not washed out.

## 3. Report nonparametric bounds for bounded outcomes

For a normalized loss \(L\in[0,1]\), let \(R=1\) mean the outcome is observed by the registered horizon. For each arm \(a\), record:

- \(p_a=P(R=1\mid a)\), the observed-outcome fraction;
- \(\mu_{a,obs}=\mathbb{E}[L\mid R=1,a]\), the observed-case mean;
- the number observed, pending/right-censored, and declared missing separately.

Without assumptions about the unobserved losses, the full-arm mean is bounded by

\[
p_a\mu_{a,obs}\ \leq\ \mu_a\ \leq\
p_a\mu_{a,obs}+(1-p_a).
\]

These are worst-case bounds, not confidence intervals. Sampling uncertainty in \(p_a\) and \(\mu_{a,obs}\) must be accounted for separately. The interval becomes uninformative as observation coverage falls; that is a meaningful result, not a reason to replace missing outcomes with observed-case averages.

For two arms with independently bounded means, a conservative difference interval is

\[
\Delta_{low}=\mu_{c,low}-\mu_{r,high},\qquad
\Delta_{high}=\mu_{c,high}-\mu_{r,low}.
\]

This interval can be wide. Do not narrow it merely because simulator arms share a seed: shared random numbers can support paired precision, but cannot reveal unobserved real-world outcomes or establish causal identification by themselves. Where a valid paired design supplies stronger bounds, state its assumptions and derive those bounds explicitly.

These formulas assume the scored outcome really is bounded in \([0,1]\). For an unbounded or differently scaled outcome, declare and justify a valid range or use a suitable alternative; do not clip data just to obtain a favorable bound.

## 4. Add a preregistered MNAR sensitivity analysis

Worst-case bounds are the assumption-light baseline. Add a transparent sensitivity analysis to show which assumptions would be needed for a conclusion.

For each arm, define

\[
\delta_a=\mathbb{E}[L\mid R=0,a]-
          \mathbb{E}[L\mid R=1,a].
\]

For a bounded loss, the feasible range is
\(-\mu_{a,obs}\leq\delta_a\leq1-\mu_{a,obs}\). Under a selected sensitivity value,

\[
\mu_a(\delta_a)=\mu_{a,obs}+(1-p_a)\delta_a.
\]

The corresponding candidate-minus-reference difference is

\[
\Delta(\delta_c,\delta_r)=
\mu_{c,obs}-\mu_{r,obs}
+(1-p_c)\delta_c-(1-p_r)\delta_r.
\]

The analysis must:

1. Preregister a feasible sensitivity grid or range for each arm, including no selection difference, adverse-to-candidate, adverse-to-reference, and extreme feasible settings.
2. Show the whole sensitivity surface or a clearly specified summary, not only a chosen favorable point.
3. Calculate and report the **tipping point**: the smallest registered departure from the no-selection assumption at which the sign of the candidate-reference comparison changes.
4. Keep sensitivity assumptions distinct from evidence. A sensitivity scenario does not establish the actual MNAR mechanism.
5. Treat different observation rates, outcome-dependent missingness, action-dependent missingness, and delayed-but-eventually-observed feedback as separate factors.

The specific grid and its rationale must be frozen before confirmatory evaluation. A universal numeric delta grid is not appropriate for every task; use the outcome scale and a justified domain-specific range.

## 5. Required stress scenarios

The pilot must include deterministic fixtures where the evaluator knows every hidden outcome and can test whether the score contract behaves honestly:

- MCAR-like outcome revelation, as a control.
- MAR-like revelation based only on recorded, permitted covariates.
- Outcome-dependent MNAR: high-loss events are less likely to be revealed.
- Reverse MNAR: high-loss events are more likely to be revealed.
- Prediction-error-dependent revelation, including a control where the error is hidden from the policy.
- Action-dependent revelation, where the intervention changes the probability or latency of outcome observation.
- Heterogeneous and out-of-order delays with right-censoring at a fixed horizon.
- Explicit missingness declarations versus silent pending/censored outcomes.

Include a **ranking-reversal negative control**: construct a fixture where the observed-case mean favors one arm while the evaluator's full hidden-outcome mean favors the other. The observed-case comparison must not pass the robust-superiority gate. The full-information simulator result is for auditor validation only and is never supplied to a policy.

## 6. Scorecard: report, do not hide

For every arm and dataset, report separately:

- Full eligible count and assignment mechanism.
- Observed outcomes and observed-outcome coverage.
- Delayed/pending, right-censored, declared-missing, and unknown counts.
- Reveal-delay distribution (at minimum median, p90, and maximum or registered tail statistic), by arm and relevant observed strata.
- Observed-case loss, explicitly labeled as selected data.
- Worst-case full-mean and candidate-reference bounds.
- The preregistered MNAR sensitivity surface and tipping point.
- In simulator tests only: full hidden-outcome loss, bias of each estimator/interval, and whether the proposed bounds contain oracle truth.
- Resource cost and policy actions, so reducing coverage or changing who gets labeled cannot masquerade as free improvement.
- Paired per-dataset effects and uncertainty, with seeds/scenario blocks as the resampling units; distinguish confidence intervals from identification bounds.

If missingness is associated with observed covariates, stratified or inverse-probability-adjusted estimates may be reported as additional estimators only when the assumptions, propensity model, positivity/overlap diagnostics, and uncertainty method are declared. Do not present inverse-probability weighting as a generic solution to MNAR: dependence on unobserved outcomes is not made ignorable by fitting a more complicated model.

For real, nonrandomized policy comparisons, missing-outcome bounds alone do not solve confounding or lack of action support. State the identification strategy separately. If no valid randomized design, simulator, or defensible adjustment identifies the causal contrast, report it as unidentified.

## 7. Robust-superiority rule

For the confirmatory task-loss comparison, the candidate may be called robustly superior only if all of the following hold:

- All evidence-access, timestamp, receipt, and hard safety gates pass.
- The candidate and reference share the preregistered information, task, and cost contract.
- The comparison uses the held-out panel and the registered uncertainty procedure.
- The upper uncertainty bound for \(\Delta\) remains below zero across the entire preregistered sensitivity set used for the claim.
- No plausible registered selection scenario reverses the candidate's ranking, and all required negative controls behave correctly.

If any plausible registered scenario changes the sign, label the result **sensitivity-dependent**. If the identification interval or uncertainty region includes zero, label superiority **not established**. If the data cannot support the estimand under the stated assumptions, label it **unidentified**. These labels are not failures of the benchmark; they are essential findings.

A zero-violation evidence-integrity gate is independent of this statistical rule. No favorable task score can offset evidence leakage, and no statistical interval can excuse a future outcome reaching the policy.

## 8. Acceptance criteria for the implementation

- Zero hidden outcome, future reveal-time, or missingness-mechanism leakage into policy-visible records.
- Availability states remain distinct through replay, score, and audit artifacts.
- Deterministic property tests verify the bounded-outcome formulas, feasible sensitivity domain, monotonicity in observation coverage, and candidate-reference interval arithmetic.
- The ranking-reversal fixture demonstrates that a favorable observed-case mean alone cannot pass robust superiority.
- On fully observed synthetic controls, bounds collapse to the observed full mean (up to declared numeric tolerance).
- In simulator stress tests, reported identification intervals contain the exact full-data result for every registered missingness scenario; failure is a hard test failure.
- Confidence intervals and identification bounds are represented as distinct fields and are never interchanged.
- Every output pins input-manifest digest, scenario ID, seed, arm/configuration digest, evaluation horizon, reveal schedule digest, and raw receipt digest.
- All planned, validated, and measured states remain separate. Passing these protocol tests is not a performance result.

## Claim ceiling

This protocol proposes how to evaluate policy value under informative missingness. It does not establish that any Symthaea policy is superior, that any missingness mechanism has been correctly modeled, or that a real-world causal effect is identifiable. Empirical claims require pinned data and code, exact-head test evidence, raw run artifacts, held-out comparisons, and uncertainty under the registered assumptions.
