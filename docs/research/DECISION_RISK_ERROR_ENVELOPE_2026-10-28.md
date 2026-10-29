# Decision-risk error envelope: finite-sample audit benchmark

**Date:** 2026-10-28  
**Status:** Synthetic research experiment, not a deployment certificate  
**Scope:** 8 policies × 16 latent worlds (128 policy-world cells); 200 Monte Carlo repeats.

## Why this work matters

The previous certified-regret pilot used the simulator's exact maximum risk-matrix error as epsilon. It checked the regret-bound logic, but not how a real system could obtain that error bound from evidence. This experiment replaces that oracle input with held-out bounded audit observations, simultaneous cellwise confidence envelopes, and a near-threshold falsification case.

For a fixed correctly bound policy family, the conditional gate is:

    true_regret <= 2 * (r + epsilon)

where r is the calibrated radius in estimated decision-risk space, and epsilon is a valid upper bound on the maximum policy/world cell error. This experiment estimates epsilon using (1) the bounded empirical-Bernstein construction of Maurer–Pontil and (2) a familywise Hoeffding bound. The plug-in residual is retained only as a negative control, not a certificate.

## Protocol

- The joint-risk matrix comes from the existing synthetic finite game via build_risk_tensor.
- Each independent audit observation is true cell risk + Uniform(-0.05, 0.05). Outcomes are bounded in the predeclared [0, 4] bits range. This is a deliberately low-variance synthetic measurement model, not a claim about how real SCM latent worlds can be observed.
- Per-cell sample sizes: 32, 128, 512, 2,048, 4,096 and 8,192. Confidence uses familywise delta = 0.05 across all 128 cells.
- The admission gate uses the earlier synthetic pooled radius r = 0.0666424883 bits and a declared 0.20-bit regret budget. Its remaining error allowance is epsilon <= 0.0333575117 bits.
- Test conditions: correct model; 1%-scale misspecification; a near-budget perturbation with true maximum error set 0.0015 bits beyond the admissible epsilon; 5%- and 20%-scale misspecifications. Policy-row identity mismatch is intentionally excluded: that is an integrity failure and must be rejected by the signed manifest check before statistics or optimization.

## Findings

Across 200 repetitions at every model/sample-size combination, both simultaneous empirical-Bernstein and Hoeffding envelopes covered the simulator's true maximum matrix error in 200/200 observed repetitions. At 200/200, the Wilson 95% lower bound is about 98.1%. This is Monte Carlo evidence only under the declared synthetic assumptions.

| Condition | True max error (bits; scoring only) | Mean empirical-Bernstein epsilon UCB at 4,096/cell | Mean regret bound 2(r + epsilon_UCB) | Admission at 4,096/cell | Plug-in false admission at 128/cell |
|---|---:|---:|---:|---:|---:|
| Correct model | 0.00000 | 0.0243 | 0.1818 | 100% | 0% |
| 1%-scale misspecification | 0.00946 | 0.0325 | 0.1983 | 98% | 0% |
| Near-budget misspecification | 0.03486 | 0.0578 | 0.2489 | 0% | 9.5% |
| 5%-scale misspecification | 0.05604 | 0.0790 | 0.2914 | 0% | 0% |
| 20%-scale misspecification | 0.16462 | 0.1876 | 0.5085 | 0% | 0% |

The correct and 1%-scale models are admitted at 8,192 observations/cell with mean empirical-Bernstein regret bounds of 0.1589 and 0.1760 bits respectively. The near-budget model remains rejected at that sample size (mean epsilon upper bound 0.0467 bits), as do the larger shifts.

The point estimate is not a confidence certificate. In the near-budget condition, the true error is 0.03486 bits—above the 0.03336-bit allowance—but the plug-in estimator falsely admitted it in 3.0% of repeats at 32/cell, 9.5% at 128/cell, 5.0% at 512/cell, and 0.5% at 2,048/cell. The confidence-envelope methods had no observed unsafe admissions. They were conservative at low sample counts; empirical Bernstein first admitted the correct synthetic model at 4,096/cell.

The plain Hoeffding confidence radius alone requires **61,406 independent observations per cell** to fall below the remaining 0.03336-bit error budget. That is roughly 7.86 million measurements across 128 cells, before accounting for the nonzero measured residual. Empirical Bernstein admitted the correct matrix at 4,096/cell because this synthetic measurement process has low variance. This is a variance-sensitive sample-efficiency result for this setup—not a universal 15× performance claim.

## Research and industry comparison

No published algorithm was executed on this exact finite-game benchmark; **there is no SOTA claim**. A matched-baseline study should implement:

- [Conformal Policy Control (ICML 2026)](https://proceedings.mlr.press/v306/prinster26a.html), which uses a safe reference policy to regulate an optimized but untested policy.
- [CREDO / Conformalized Decision Risk Assessment (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/11e0fdc1f9ec9303e5de60670e884e53-Abstract-Conference.html), which bounds decision suboptimality using optimization geometry and conformal generative prediction.
- [Robust Decision-Focused Learning via Worst-Case Regret Minimization (UAI 2026)](https://proceedings.mlr.press/v337/yamao26a.html), which addresses measurement error and deployment distribution shift.
- [Multi-Distribution Robust Conformal Prediction (ICML 2026)](https://proceedings.mlr.press/v306/yang26ad.html), to test coverage across heterogeneous source distributions.
- [Empirical Bernstein Bounds and Sample Variance Penalization](https://arxiv.org/abs/0907.3740), the bounded variance-sensitive confidence method used here.

For governance, [NIST's AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) and [ISO/IEC 23894:2023](https://www.iso.org/standard/77304.html) are risk-process references, not algorithmic baselines.

## Paradigm shifts to pursue next

1. **Benchmark the certificate itself.** Track familywise coverage, false-admission rate, fallback rate, measurement cost per cell, and worst-group performance—not only planner regret.
2. **Move from global matrix accuracy to proven decision-relevant error.** A max error over every cell can be overly conservative. Derive an upper bound over the calibrated decision-relevant set with a proof; do not replace it with mean residuals or unqualified bootstrap estimates.
3. **Cut audit cost using paired evaluations.** Explore common random numbers, paired policy rollouts, control variates, and structural simulator bounds while stating dependence assumptions.
4. **Make auditability a prerequisite.** A production system may not be able to independently sample every latent-world cell. If the support, loss range, independence, or mapping is not evidenced, epsilon remains unverified and the gate must fail closed.
5. **Keep identity separate from uncertainty.** A row-to-action mapping change is an integrity failure. A signed policy-risk manifest must be checked before optimization and again at dispatch; statistical intervals cannot repair a changed mapping.

## Reproduction

The reproducibility bundle includes source, test, raw JSON, report and dependency scripts. Run:

    PYTHONPATH=.:dependencies python decision_risk_error_envelope_benchmark.py --output DECISION_RISK_ERROR_ENVELOPE_RESULTS_2026-10-28.json --repeats 200
    PYTHONPATH=.:dependencies python -m unittest -v test_decision_risk_error_envelope_benchmark.py

The broader local Python suite passed **23/23 tests** in 19.629 seconds. Raw JSON SHA-256: 457c60993a037eada7d372fe9901711ec311422024763f0b2085b84fa81a9a3b. Local source SHA-256: 8fd7a2eaa39d9f034de00e9f1c4f1d681d2c5c5f54f23029791c877944e43507; local test SHA-256: 7d0c1beb0ae3a366f02b13af21113254517a9fd681615763f747500ff3dd5711.

## Limitations

This is a synthetic 8-policy × 16-world game. The true simulator error is used only to score the envelope, not by the estimator. Audit samples are iid low-noise bounded draws with full cell support; real latent worlds may be unobservable, and dependence, hidden confounding, adaptive policy creation and arbitrary distribution shift are not covered. The prior calibration radius is carried forward, not jointly recalibrated here. The fallback from the earlier experiment remains a simulator-oracle reference, not a production controller. These Python tests neither compile the native Rust module nor integrate the gate into the production dispatch path.
