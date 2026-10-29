# Sensitivity-adjusted anytime risk certificates under Huber contamination

**Date:** 2026-10-29. **Status:** exploratory synthetic experiment; not a production certificate or direct SOTA comparison.

## Research question

Can decision-risk certification remain sound when the observed audit stream is an IID mixture of clean target outcomes and a bounded fraction of arbitrary contamination? The previous allocation benchmarks assume IID Bernoulli observations directly from the target risk distribution. If an audit process is contaminated, a nominal confidence sequence may be statistically valid for the *observed mixture* while being invalid for the *clean target* the policy is meant to certify.

## Method

Assume each observed Bernoulli audit value is drawn independently from a Huber mixture:

Q = (1 - rho) P + rho R,

where P is the clean target distribution, R is arbitrary, and the contamination fraction is known to be at most rho_upper < 1. For Bernoulli means q, p, and r, the observed mean is q = (1-rho)p + rho r. This yields the sensitivity set

p in [max(0, (q - rho_upper)/(1-rho_upper)), min(1, q/(1-rho_upper))].

We first calculate an anytime-valid beta-mixture confidence sequence for the observed mixture mean q. We then map the entire confidence interval through the monotone Huber sensitivity transformation to obtain a confidence sequence for the clean mean p. For every risk cell, the same familywise delta=0.10 is split across four cells. The nominal comparator ignores contamination and compares the confidence sequence for q directly against the clean model risk. Both use the same confidence-sequence stopping logic; only the robust method expands the uncertainty set.

This is a transparent Bernoulli sensitivity inversion; it is **not** the general Huber-robust confidence-sequence algorithm of Wang & Ramdas (AISTATS 2023).

## Matched protocol

- Four risk cells; one contaminated cell. IID Bernoulli observations from the exact mixture distribution.
- Error allowance epsilon=0.05, familywise delta=0.10, per-cell alpha 0.025.
- Pilot of 256 observations per cell, uniform batches of 256, cap 8,192 observations per cell.
- 500 independent repetitions for each scenario × contamination fraction × method; each pair uses the same generated streams.
- Contamination fractions: 0%, 5%, 10%, 12%, 18%; adversarial contaminant outcome is R=0 in the target cell.
- Safe case: model [.15,.30,.60,.80], clean target [.15,.32,.60,.80], true max error .02.
- Unsafe case: same model, clean target [.15,.37,.60,.80], true max error .07.

## Core results: adversarially masked unsafe model

| Contamination rate | Observed target-cell mean | Nominal unsafe admissions | Huber-adjusted unsafe admissions | Nominal mean observations | Robust mean observations |
|---:|---:|---:|---:|---:|---:|
| 0% | 0.3700 | 0/500 | 0/500 | 27,021 | 27,021 |
| 5% | 0.3515 | 0/500 | 0/500 | 32,768 | 32,768 |
| 10% | 0.3330 | 120/500 (24%) | 0/500 | 30,573 | 32,768 |
| 12% | 0.3256 | 390/500 (78%) | 0/500 | 22,850 | 32,768 |
| 18% | 0.3034 | 500/500 (100%) | 0/500 | 10,041 | 32,768 |

At 10% contamination, the nominal method falsely admitted in 120 of 500 trials; its two-sided 95% Wilson interval for the empirical false-admission frequency is approximately 20.5%–27.9%. At 12% the false-admission frequency was 78% (Wilson interval approximately 74.2%–81.4%). At 18% all 500 nominal trials admitted the truly out-of-tolerance model (Wilson lower limit approximately 99.2%). The contamination-adjusted method made no unsafe admissions in these trials, with a nonzero Wilson 95% upper limit of approximately 0.76% for 0/500.

This is not a statistical paradox: the nominal confidence sequence can be calibrated for the contaminated mixture mean and still answer the wrong target-population question.

## Availability cost: safe model

The contamination adjustment is deliberately conservative. In the safe .02-error scenario, the adjusted method admitted 475/500 with no contamination, but it refused unresolved in all 500 trials at contamination rates of 5%, 10%, 12%, and 18% under the 8,192-per-cell cap. At 5%, the theoretical sensitivity set for the clean risk can still fit within the .05 error allowance in the infinite-sample limit, but this cap was insufficient given the anytime confidence-sequence width. At 10% and above, the clean-risk compatibility set itself can exceed the allowed error, so more observations cannot necessarily resolve it.

This is the important negative result: robust certification trades false admission risk for abstention when the requested conclusion is not identifiable from the contaminated observations alone.

## Identifiability frontier

For a clean risk p=.32, model risk .30, and contamination R=0, the exact sensitivity lower endpoint is p_lower = p - rho/(1-rho). For worst-case model error to remain below .05, the contamination-only offset must satisfy

.02 + rho/(1-rho) <= .05,

or rho <= 0.029126... (about 2.91%), before adding any sampling uncertainty. Above that rate, this particular safe model cannot be certified against the .05 tolerance under the stated arbitrary-contamination model without stronger assumptions or additional evidence about the contamination mechanism. This is an identifiability limit, not a scheduler failure.

## Integrity negative controls

The preflight returns fail-closed when any of these are missing: support coverage, policy/model identity match, or a defensible contamination upper bound. Unit tests assert all three negative controls. The robust method cannot repair unknown target support or a wrongly bound policy identity by adding a statistical interval.

## Literature and comparison discipline

- Wang & Ramdas, *Huber-robust confidence sequences*, AISTATS 2023: develops sequential confidence sequences robust to a bounded fraction of arbitrary distribution corruption with a moment bound. https://proceedings.mlr.press/v206/wang23p.html
- Cai & Kandasamy, *Constrained Best Arm Identification with Tests for Feasibility*, AAAI 2026: chooses which feasibility/objective measurement to take and provides sample-complexity guarantees. https://ojs.aaai.org/index.php/AAAI/article/view/39063
- Jourdan et al., *An Anytime Algorithm for Good Arm Identification* (APGAI), JMLR 2026: anytime threshold identification. https://www.jmlr.org/papers/v27/24-0680.html

No published baseline implementation was run. This report does not claim SOTA superiority; it evaluates a target-population sensitivity correction absent from the earlier IID-only allocation experiments.

## Next gates

1. Compare this sensitivity inversion against a reproduction of the Wang–Ramdas robust confidence-sequence construction under matching bounded-moment assumptions.
2. Sweep cell count, tolerance, delta, sample cap and contamination fraction; identify the robust-certification feasibility boundary analytically and empirically.
3. Add cluster/dependence stress tests and time-varying corruption. The current guarantee does not extend to dependent samples or arbitrary changing contamination.
4. Combine robustness with frontier-deficit allocation only after retaining the same robust confidence gate; measure whether allocation reduces cost without changing admission rates.
5. Keep integrity preflight independent: missing support, a digest/policy mismatch, or unknown contamination bound must refuse before inference.

## Reproduction

    python -m unittest -v test_decision_risk_huber_sensitivity.py
    python decision_risk_huber_sensitivity.py --repeats 500 --output DECISION_RISK_HUBER_SENSITIVITY_RESULTS_2026-10-29_500.json

All results are local synthetic research evidence. No production Rust integration or remote CI pass is claimed.
