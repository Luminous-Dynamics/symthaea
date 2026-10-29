# Sequential decision-risk audit: fixed looks versus anytime-valid admission

**Date:** 2026-10-29  
**Status:** Synthetic research benchmark, not a production certificate  
**Protocol:** 1,000 repeat runs, four fixed risk cells, five predeclared checkpoints.

## Why this is the next research problem

The previous error-envelope benchmark exposed that the decision gate is an adaptive procedure: it may recalculate a bound as more audit observations arrive and stop as soon as the result looks admissible. A fixed-sample confidence bound is justified at a predeclared sample size; repeatedly peeking and stopping based on it does not inherit the same guarantee.

This benchmark compares three variants on the same nested audit stream:

1. **Fixed-time repeated peek (negative control):** reuses a single-look empirical-Bernstein bound with familywise delta=0.10 at every checkpoint without correcting for the looks.
2. **Finite-look alpha spending:** the five checkpoints are predeclared; each receives delta/L, with familywise adjustment across four cells and both tails.
3. **Infinite-horizon alpha spending:** uses delta_n = delta * 6/(pi^2 n^2) over every integer n, with sum_n delta_n=delta, and cellwise empirical-Bernstein bounds.

The last two are valid only under the bounded independent audit observations and fixed support/model assumptions used here. Finite-look spending applies only to the exact listed checkpoints; adding an unscheduled look needs a new allocation. The infinite-horizon method is a deliberately conservative union bound, not an optimized confidence sequence.

## Protocol

- Fixed seed: 20261029; 1,000 simulations per condition.
- Four predeclared policy/world cells; independent Bernoulli audit outcomes with means [[0.50, 0.52], [0.48, 0.55]], all in the declared [0,1] range.
- Checkpoints: 4,096, 8,192, 16,384, 32,768 and 65,536 observations per cell.
- Per-look nominal familywise delta: 0.10.
- Carried-over decision radius: r=0.0666424883 bits; regret budget=0.20 bits; admissible model-error allowance epsilon=0.0333575117 bits.
- Tested planning matrices: exact, just over budget (epsilon allowance + 0.00001 bits), and clearly over budget (allowance + 0.01 bits).
- The true synthetic matrix is used to score false admissions, not to calculate audit bounds.

## Results

### Correct model: admission cost

| Method | Admitted by final horizon | Mean observations per cell to admission | Admission checkpoint split |
|---|---:|---:|---|
| Fixed-time repeated peek | 100% | 8,327 | 6.1% at 4,096; 89.2% at 8,192; 4.7% at 16,384 |
| Finite-look alpha spending | 100% | 9,535 | 83.6% at 8,192; 16.4% at 16,384 |
| Infinite-horizon alpha spending | 100% | 32,178 | 3.6% at 16,384; 96.4% at 32,768 |

For this predeclared five-look protocol, finite-look alpha spending required about **14.5% more audit observations** than the unadjusted repeated-peek control, while preserving an explicit alpha allocation across the planned looks. The infinite-horizon union bound required about **3.38×** as many observations as finite-look spending and **3.86×** as many as the unadjusted control. These ratios describe this toy protocol, not a universal efficiency claim.

### False admissions

| True model error | Fixed-time repeated peek | Finite-look alpha spending | Infinite-horizon alpha spending |
|---|---:|---:|---:|
| Just over allowance: 0.0333675 bits | 4/1,000 (0.4%; Wilson 95% CI 0.16–1.02%) | 0/1,000 (upper Wilson 95% limit 0.38%) | 0/1,000 (upper Wilson 95% limit 0.38%) |
| Clearly over allowance: 0.0433575 bits | 0/1,000 (upper Wilson 95% limit 0.38%) | 0/1,000 (upper Wilson 95% limit 0.38%) | 0/1,000 (upper Wilson 95% limit 0.38%) |

The four observed unadjusted false admissions illustrate the concern, but the intervals are not strong evidence that the false-admission rates differ. The central contribution is the validity distinction: finite-look or anytime alpha allocation provides a formal across-look error budget; the repeated-peek control does not. Zero failures in 1,000 synthetic repetitions does not establish zero risk.

Neither adjusted method admitted a model over budget in this run. Both conservatively refused the near-threshold condition at the 65,536-observation horizon.

## Research frontier and baselines

This prototype makes no SOTA claim and does not reimplement a published betting confidence sequence. The key literature-backed next comparators are:

- Howard et al., [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://arxiv.org/abs/1810.08240), Annals of Statistics 2021: time-uniform inference that remains valid at arbitrary stopping times.
- Waudby-Smith & Ramdas, [Estimating means of bounded random variables by betting](https://doi.org/10.1093/jrsssb/qkad009), JRSS B 2024: variance-adaptive confidence intervals and sequences based on betting.
- Martinez-Taboada & Ramdas, [Gaussian-efficient testing by betting on the mean of bounded data](https://arxiv.org/abs/2608.21694), submitted 2026-08-22: a newer, fixed-horizon bounded-mean interval designed to attain Gaussian efficiency while retaining martingale validity. It should not be reused across unscheduled looks without a valid multiple-look correction.
- Mathis & Waudby-Smith, [Confidence Horizons](https://arxiv.org/abs/2608.03889), submitted 2026-08-04: finite-horizon anytime-valid inference, a particularly relevant direction when an audit budget has a known terminal horizon.
- For decision-level comparison, [Conformal Policy Control (ICML 2026)](https://proceedings.mlr.press/v306/prinster26a.html) and [CREDO (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/11e0fdc1f9ec9303e5de60670e884e53-Abstract-Conference.html).

No external baseline implementation has been run on the Symthaea synthetic protocol. The next matched benchmark should compare finite-look alpha spending, betting-based confidence sequences, Gaussian-efficient fixed-horizon intervals with principled alpha spending, and conformal policy control under identical loss cells, audit budgets, near-boundary errors, shifts, and numerical tolerances.

## Paradigm shift

**Treat audit scheduling as part of the decision algorithm, not as a reporting detail.** The gate should receive an explicit stopping contract: fixed horizon, allowed checkpoint list, familywise error budget, and audit method. The certificate must become invalid if the caller adds an unscheduled look, changes the policy/world support, or changes the risk matrix digest. Choose a bounded-horizon confidence design when the real operational horizon is finite; use a genuinely anytime-valid confidence sequence when stopping times cannot be predeclared.

The next performance target is not merely smaller intervals. It is the minimum audit cost that meets a calibrated false-admission target under a predeclared adversarial suite, while keeping exact provenance and rejecting unsupported audit assumptions. The earlier systematic-bias and missing-support guards remain necessary: alpha spending does not defend against biased or incomplete evidence.

## Reproduction and evidence

- Source path: validation/decision_risk_sequential_audit_benchmark.py
- Test path: validation/test_decision_risk_sequential_audit_benchmark.py
- Raw JSON path: validation/DECISION_RISK_SEQUENTIAL_AUDIT_RESULTS_2026-10-29.json
- Local focused tests: **3/3 passed** in under one second.
- Exact source local SHA-256: 0de48cfc7d3a710afdf81c0c050d0e5ddaa274fe5970777bf581eb67c7e75e7d; Git blob 7176d08730d247cd8ccbb9d135b35cd1ff66fd0c.
- Exact test local SHA-256: 66f211a1d17dab4ade3cc9a98ac66e2fed2719a7127f0ed0f70c3f6908a44b1a; Git blob 9abced07dc404017e9447862394a8f3ec4f061a7.
- Raw result SHA-256: 181f819cb411ed4e18c58fbbdfb3f3880e0c1730c094fbd3e721bc50776b6d56.

The focused Python suite validates the synthetic benchmark. Rust compilation, production integration, external baseline execution, and remote CI are not claimed.
