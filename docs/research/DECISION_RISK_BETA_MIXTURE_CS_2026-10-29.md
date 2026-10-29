# Beta-mixture e-process confidence sequence for sequential decision-risk audits

**Date:** 2026-10-29  
**Status:** Synthetic benchmark; not production certification  
**Question:** Can we retain anytime-valid admission while reducing audit cost relative to a conservative infinite-horizon empirical-Bernstein envelope?

## Executive result

I added a matched four-method sequential audit benchmark and a beta-binomial mixture likelihood-ratio e-process confidence sequence for Bernoulli risk cells. The implementation inverts the mixture e-value using a log-domain calculation and bisection, then bounds the maximum planning-matrix error by the largest distance from each planning cell to its confidence interval. Familywise delta is allocated as delta/K over the predeclared cell family.

In 1,000 fixed-seed repeats, the beta-mixture confidence sequence admitted the correct model in all runs with a mean **9,756.7 observations per cell**. The prior infinite-horizon empirical-Bernstein alpha-spending method needed **32,178.2 per cell** on average. Thus, on this particular Bernoulli toy protocol, the beta-mixture method used **69.7% fewer observations**. It used 2.3% more observations than the five-look finite-look alpha-spending method, and 17.2% more than the uncorrected repeated-peek control.

The uncorrected repeated-peek control falsely admitted the just-over-budget planning matrix in 4/1,000 runs (0.4%; Wilson 95% interval approximately 0.16–1.02%). The finite-look alpha-spending, infinite-horizon empirical-Bernstein, and beta-mixture e-process methods had zero observed false admissions in this experiment. With 0/1,000 events, the Wilson upper 95% limit is about 0.38%; this does not prove a zero population rate.

**Interpretation:** the beta-mixture e-process is a compelling next comparator for this setup: it retains an anytime-valid property under the Bernoulli model while getting much closer to finite-look efficiency than the conservative sum-over-all-times empirical-Bernstein method. This is not a universal speedup claim or a SOTA result.

## Method

For each candidate mean mu in [0,1], the likelihood ratio of the beta-binomial mixture to the Bernoulli-mu sequence is

    E_n(mu) = [ B(S_n + a, n - S_n + b) / B(a,b) ]
              / [ mu^S_n * (1-mu)^(n-S_n) ]

with a=b=1/2. Under iid Bernoulli(mu), this is a nonnegative martingale. Ville's inequality gives

    P_mu( sup_n E_n(mu) >= 1 / alpha_cell ) <= alpha_cell.

We invert the e-process to a confidence sequence and allocate alpha_cell = delta/K over the K=4 declared cells. At every permitted check, the maximum model-error upper bound is the maximum absolute distance from each planning-matrix cell to either endpoint of that cell's confidence sequence. The decision gate admits only if 2(r + epsilon_upper) <= 0.20 bits, where the reused synthetic radius is r=0.0666424883 bits and therefore the available error allowance is 0.0333575117 bits.

The comparison uses one nested audit stream per repeat, three planning matrices (correct, 0.00001 bits over the error allowance, and 0.01 bits over the allowance), and five checkpoints at 4,096, 8,192, 16,384, 32,768 and 65,536 observations per cell. The fixed seed is 20261029 and there are 1,000 repeats. Risk cells are Bernoulli with means [[0.50, 0.52], [0.48, 0.55]]; this is intentionally a narrow statistical benchmark, not a model of the full Symthaea SCM problem.

## Results

### Correct planning matrix

| Method | Admission rate | Mean observations per cell to admission |
|---|---:|---:|
| Fixed-time empirical Bernstein reused at each look (negative control) | 100% | 8,327.2 |
| Five-look alpha spending + empirical Bernstein | 100% | 9,535.5 |
| Infinite-horizon alpha spending + empirical Bernstein | 100% | 32,178.2 |
| **Beta-mixture e-process confidence sequence** | **100%** | **9,756.7** |

### Near-threshold unsafe matrix

The true max-cell model error is 0.0333675117 bits, just above the allowed epsilon=0.0333575117 bits.

| Method | False admissions | Rate | Wilson 95% interval |
|---|---:|---:|---:|
| Fixed-time repeated peek without look correction | 4 / 1,000 | 0.4% | 0.16–1.02% |
| Five-look alpha spending | 0 / 1,000 | 0% | Upper endpoint ≈ 0.38% |
| Infinite-horizon empirical-Bernstein alpha spending | 0 / 1,000 | 0% | Upper endpoint ≈ 0.38% |
| Beta-mixture e-process confidence sequence | 0 / 1,000 | 0% | Upper endpoint ≈ 0.38% |

All three corrected methods refused the near-threshold model through the 65,536-observation horizon. The clearly over-budget model was also refused by all methods. These are finite simulation results, not proof of global coverage.

## Literature and industry comparison

The mixture e-process is a baseline derived from established sequential inference, not a new theorem or invention. Waudby-Smith and Ramdas develop betting and mixture confidence sequences for bounded means, including methods valid for optional stopping and sampling without replacement ([JRSS B 2024](https://academic.oup.com/jrsssb/article/86/1/1/7043257)). Howard et al. develop time-uniform confidence sequences based on nonnegative supermartingales and empirical-Bernstein bounds ([Annals of Statistics 2021](https://arxiv.org/abs/1810.08240)).

Two newer lines merit the next comparison:
- Martinez-Taboada and Ramdas, [Gaussian-efficient testing by betting on the mean of bounded data](https://arxiv.org/abs/2608.21694), submitted 2026-08-22. It targets Gaussian-efficient fixed-horizon intervals via a novel betting strategy; to use it in a sequential gate, respect its stated horizon and add principled look correction if checks repeat.
- Mathis and Waudby-Smith, [Confidence Horizons](https://arxiv.org/abs/2608.03889), submitted 2026-08-04. This directly studies sharper validity under a known finite maximum horizon and is relevant to audit budgets with predeclared limits.

No external package or published implementation was run on the same Symthaea protocol. I have not established SOTA. The next matched study should implement or reproduce the published betting baseline and confidence-horizon method, then compare on identical randomized streams, coverage targets, cell counts, boundary gaps, model shifts, stopping rules, CPU cost, and worst-case audit count.

## Paradigm shift

**Treat audit scheduling as part of the decision algorithm, not as a reporting detail.** The gate should receive an explicit stopping contract: fixed horizon, allowed checkpoint list, familywise error budget, and audit method. A certificate records whether the horizon is fixed, which looks are permitted, the familywise alpha allocation, the cell support, the planning-matrix digest, and the actual stopping rule. Unscheduled looks invalidate a finite-look certificate. Anytime methods still require correct sampling assumptions and complete support; they do not repair systematic audit bias, omitted worlds, changed policy identity, or an unverified loss range.

For the larger Symthaea program, the target is to minimize audit cost at the same false-admission ceiling, not just minimize average interval width. The benchmark scorecard should include familywise coverage, false-admission rate with binomial intervals, admission probability for valid models, observations per cell until admit/refuse, tail regret, worst-regime coverage, and runtime. Any method that appears more efficient by breaking the evidence assumptions must fail a separate bias/support/identity gate.

## Reproduction

Files committed in this PR:

- validation/decision_risk_beta_mixture_cs_benchmark.py
- validation/test_decision_risk_beta_mixture_cs_benchmark.py
- validation/DECISION_RISK_BETA_MIXTURE_CS_RESULTS_2026-10-29.json
- validation/DECISION_RISK_BETA_MIXTURE_CS_SUMMARY_2026-10-29.json

Focused tests: **4/4 passed**. Full local 1,000-repeat comparison completed in 16.742 seconds. The exact executed source and test were reconciled with the branch versions:

- Source SHA-256: 968398e5fd04c9380f963f3567ea20e5a661e89b750419ee6a9882cd63effa19; Git blob 728592d120a77ed2e23529a4364d086443b6cce8.
- Test SHA-256: d11fba1da9b7cc988d4cea75f84e480a2207da04065855183ea417c1fb31022f; Git blob ca9db30a4e26c5e8fa572df581d11023d0151ee0.
- Raw results SHA-256: aabec07b5383df9a1c7415a5c9295c6810360fe49137d8f1c83a6e449be7156a; result blob c5c268ab3e927d32fbb642353f5c7c3c33a40aaa.

## Limits

The data are Bernoulli and iid. This sequence is not automatically valid for an arbitrary bounded-valued empirical loss, adaptively changing cells, clustered/dependent audits, hidden distribution shift, or incomplete latent-world support. The earlier signed-manifest and audit-assumption preflight remain necessary and are not integrated into Rust dispatch. Rust compilation, production wiring, external baseline execution, and remote CI are not claimed.
