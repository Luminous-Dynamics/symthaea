# VIS-DVPC-008: Calibrated Decision-Value Router v1

**Status:** Draft reference policy and synthetic tests.  
**Issue:** [#8531](https://github.com/Luminous-Dynamics/symthaea/issues/8531)  
**Vector-budget parent:** [PR #8518](https://github.com/Luminous-Dynamics/symthaea/pull/8518)  
**Paired-trace evaluator:** [PR #8506](https://github.com/Luminous-Dynamics/symthaea/pull/8506)  
**Research comparator:** [DEEP, arXiv:2609.35910](https://arxiv.org/abs/2609.35910)

## Gap being addressed

VIS-DVPC-007 accepts a numeric pre-escalation `net_benefit_lower_bound`, but a finite number is not evidence that the estimate has the advertised coverage. A model can become overconfident after its estimator artifact, task-loss profile or operating domain changes.

This prototype adds a split-conformal one-sided lower bound over the standardized overprediction residual, binds it to a frozen estimator/domain/loss/calibration profile, and blocks escalation for suspected or unknown shift. It composes with the independent latency/energy allocator rather than inventing a new scalar budget.

## Calibration mathematics

For each held-out calibration example (i), use the already-frozen estimator's predicted mean (\hat{b}_i), strictly positive predicted scale (s_i), and observed downstream benefit (b_i):

[
r_i = (\hat{b}_i - b_i) / s_i
]

Sort the standardized residuals (r_{(1)} \le \ldots \le r_{(n)}). For requested one-sided miscoverage (α), the finite-sample one-based rank is

[
k = \lceil (n+1)(1-\alpha) \rceil
]

When (k > n), the requested nominal coverage cannot be supported by this finite calibration set and profile creation is rejected. The bound for a new in-domain prediction is

[
b_{\mathrm{lower}} = \hat{b} - r_{(k)} s
]

The score scale must be positive and the computed result finite. The code does not clip a negative residual quantile or the lower bound; such clipping would change the method. A downstream benefit estimate must use the same predeclared utility/loss definition in calibration and evaluation.

### Exact limitations of this guarantee

This is a marginal split-conformal construction. Its coverage argument relies on exchangeability between calibration examples and target examples under the frozen estimator, score construction, utility spec and declared domain. It does **not** provide conditional per-scenario coverage, solve arbitrary distribution shift, prove the domain detector is correct, or guarantee safety of any action.

Therefore `OodSuspected` and `Unknown` are hard gates. Even a very positive raw prediction cannot bypass them. A shifted input falls back to the cheap path only when the caller says that path is safe; otherwise it abstains.

## Provenance and stale-profile rejection

The fitted profile retains:
- unique sorted calibration sample IDs;
- calibration split ID and externally supplied calibration-manifest digest;
- estimator ID and artifact digest;
- domain-assessor ID and artifact digest, so a changed shift detector invalidates the profile;
- loss-profile ID and domain/condition ID;
- alpha, quantile, quantile rank, and calibration count.

Every online input must match the frozen estimator, artifact digest, loss profile and declared domain. A change in any one invalidates use of the old calibrated bound. Calibration rows must agree with the profile metadata; non-finite values, non-positive scales and duplicate IDs are rejected.

The profile fitter receives calibration outcomes; the online `decide` method does not. Both are implemented in a single source file for this small reference experiment, so this is an API/type boundary, **not** a secure OS process boundary. A production adapter must separate permissions, verify the calibration manifest digest against the actual frozen artifact/rows, and prevent the online router from reading test labels or high-fidelity outputs.

This implementation uses a supplied manifest digest and retains the sample IDs but does not calculate SHA-256 itself. A trusted caller must verify that the digest names the actual bytes before admitting the profile.

## Standalone qualification

```sh
rustc --edition=2024 --test tools/vision-decision-value-calibrated-router.rs -o /tmp/vis-dvpc-008-tests
/tmp/vis-dvpc-008-tests
```

The test build includes VIS-DVPC-007's vector-ledger unit tests as a nested module, so both calibration-gate and resource-accounting tests execute together. The dedicated workflow checks Rust formatting, compiles, runs the full focused test binary, and records the exact-head SHA-256 evidence.

## Adversarial fixtures

The authored tests cover:
- known-answer finite-sample quantile rank and input-order invariance;
- unsupported finite-sample rank, duplicate calibration samples, wrong split/model binding and invalid scale;
- a positive lower bound successfully reserving both latency and energy, then settling actual costs;
- a lower bound below threshold taking the cheap path without reservation, including a zero-mean benefit prediction;
- shift detection and unknown-domain abstention, including a deliberately huge raw point estimate;
- stale domain-assessor identity/artifact and mismatched domain-assessment input digest blocked before escalation;
- stale estimator artifact, mismatched domain/loss profile, non-finite predictions and non-positive inference scale;
- deterministic receipts carrying the gate reason, profile, calibration count, nominal marginal coverage and bound.

These are synthetic contract tests, not evidence of actual coverage, calibrated benefit, real-trace performance or deployment safety.

## Benchmark path to external comparison

Using the same frozen paired traces and utility spec as VIS-DVPC-006, compare:
1. all-cheap and all-full;
2. budget-matched random escalation;
3. a lightweight point-score threshold;
4. the raw point-estimate router;
5. this calibrated lower-bound router;
6. a conservative LCB baseline with a separately chosen, validation-frozen threshold;
7. the evaluator-only two-resource hindsight oracle as a non-deployable upper reference.

Use validation for calibration/profile selection and policy tuning, then freeze the model, profile and thresholds before evaluating on the untouched test split. Do not fit or revise the conformal quantile on the test set.

Report separate metrics:
- empirical marginal interval coverage and calibration error;
- decision-value improvement over all-cheap and matched-budget random;
- beneficial/harmful escalation rate;
- utility mean and equal-group-weighted utility;
- CVaR95 decision loss, safety deltas, abstentions and blocked-shift rate;
- actual latency, energy, allocator overhead, reservation overruns and run-budget violations;
- worst-group and held-out-domain results with paired uncertainty intervals.

A profile that produces wider bounds and abstains more often is not automatically better. The claim requires improved downstream decision value at comparable measured budget with pre-registered safety/tail-risk constraints intact. If the domain shifts, the primary v1 target is fail-closed behavior, not a claim of graceful extrapolation.

For the surrounding driving evaluation, keep open-loop and closed-loop results separate; nuPlan describes both modes: https://www.nuplan.org/. Public dataset availability does not in itself confer commercial rights; any adapter must review applicable terms before use.

## Claim ceiling and next step

This PR tests a version-bound calibration mechanism and shift gate only. It does not establish empirical interval coverage, a real-data decision gain, safe downstream behavior, a complete OOD detector, SOTA, or superiority over DEEP. After exact-head tests execute successfully, the next substantive result should be a frozen paired-trace experiment with cost overhead charged and the evaluator labels inaccessible to the online policy.
