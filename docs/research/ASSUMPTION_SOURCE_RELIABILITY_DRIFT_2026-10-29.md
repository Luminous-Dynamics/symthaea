# Reliability of assumption-bound issuers: calibration and drift stress

**Date:** 2026-10-29  
**Status:** local synthetic research experiment; no production guarantee.

## Research question

The identifiability witness in #9012 shows why contaminated observations cannot generally validate their own contamination cap. The next failure mode is upstream: even if an assumption issuer has a good historical audit record, can that calibration support a certificate after the issuer's reliability changes?

The answer is no without a justified stability/transport assumption. This report makes that limitation an executable regression, rather than implicitly assuming an issuer's past error rate remains constant.

## Method

The prototype calibrates the probability that a source's declared upper contamination bound fails to cover audited truth. It computes a one-sided exact Clopper–Pearson upper confidence bound on the issuer-violation probability. For a fixed horizon of M decisions, its conservative union-bound budget is

`delta_familywise <= alpha_cal + M * (delta_issuer_upper + alpha_stat)`

where alpha_cal is the calibration confidence failure probability, delta_issuer_upper is the upper confidence bound on source violations, and alpha_stat is the sequential statistical error probability per decision. The calibration event is shared across the horizon; this is not M independent calibrations. This bound still requires representative, independent/exchangeable audits and valid conditional statistical guarantees.

The gate refuses before inference when the evidence scope is missing or mismatched, the evidence is expired, the audit process is unqualified, or no independent audits are available.

## Executed results

### Exact finite-audit illustration

- Zero observed issuer violations in 200 independent audits gives a one-sided 95% upper confidence bound of **1.4867%**, not zero.
- Zero observed violations in 1,000 audits gives a one-sided 99.5% upper bound of **0.5284%**.
- A wrong epoch/scope receipt is refused before inference even with a hypothetical history of 1,000 clean audits.

### Reliability-drift stress

Reproduction: 1,000 independent historical calibration panels, each with 1,000 audits generated at historical source-violation rate 1%. For each panel, use calibration alpha 0.5%, sequential alpha 0.5%, a per-decision target budget 5%, and then evaluate future rates 1% (stationary control) and 10% (unannounced shift). Fixed seed: 20261029.

| Future issuer-violation rate | Gate acceptance rate | Calibrated upper bound below future truth |
|---:|---:|---:|
| 1% (stationary) | 100% | 0.3% of calibration panels |
| 10% (shifted) | 100% | 100% of calibration panels |

Under the stationary panel, 3/1,000 historical calibrations produced an upper bound below the fixed true 1% rate, consistent with the intended 0.5% calibration error only at this finite simulation resolution (the observed rate is 0.3%). After the 10% shift, every historical upper bound was below future truth, while every gate still accepted. This is a decisive warning about reusing an old reliability calibration outside its validity domain—not a failure of the Clopper–Pearson bound under its stated stationary model.

## Paradigm shift: certify the assurance mechanism

The assurance source is itself a statistical object. Give it an identity, scope, epoch, audit protocol, independent labels, calibrated violation rate and expiry. Re-certification should be triggered by changed scope or epoch, failed audit, material drift evidence or an exceeded error budget.

A signature verifies who made the claim and whether it changed; it does not prove the claim is true. An audit confidence bound is meaningful only when the audit labels are independent and its calibration assumptions hold. A drift detector can trigger refusal but cannot guarantee detection of every arbitrary shift.

Avoid conflating:
1. authenticity/scope/expiry checks;
2. reliability of the process supplying assumptions;
3. statistical uncertainty conditional on the assumption being valid;
4. the final per-decision and familywise risk budgets.

## Literature and comparison discipline

- Almeida et al., **High Probability Risk Control Under Covariate Shift** (PMLR 266, 2025), extends learn-then-test risk control to covariate shift using importance-weighted calibration losses. It is a direct comparator for the shift-aware risk-control aspect, but its validity relies on the chosen shift/weight assumptions: https://proceedings.mlr.press/v266/almeida25a.html
- Huang, Ma and Michailidis, **Model-Agnostic Online Certificate-Driven Calibration for Time Series Forecasting Under Distribution Shift** (UAI 2026, PMLR 337), studies online finite-sample certificates with temporal dependence and shifting dynamics, using a predict-then-update protocol: https://proceedings.mlr.press/v337/huang26b.html
- Xie, Gao and Lafferty, **Confidence Intervals for Linear Models with Arbitrary Noise Contamination** (Biometrika, online 28 Sep 2026), emphasizes non-identifiability of contamination fraction and provides a structured regression-specific inference method: https://academic.oup.com/biomet/advance-article-abstract/doi/10.1093/biomet/asag058/8842600

These are not interchangeable guarantees. They motivate separate tracks for independent assumption calibration, covariate shift, temporal dependence and arbitrary contamination.

## Reproducibility

- Source: `validation/assumption_reliability_budget.py`
- Tests: `validation/test_assumption_reliability_budget.py` — 9/9 passed locally
- Exact worked example: `validation/ASSUMPTION_RELIABILITY_WITNESS_2026-10-29.json`
- Raw drift stress: `validation/ASSUMPTION_RELIABILITY_DRIFT_STRESS_2026-10-29.json`
- Source SHA-256: `ffbbcbe162dacc6b5ec386a2583edf060609b5067880e2f8778d369b35977674`
- Test SHA-256: `f939e25a0c0db4a12a99085f4e7c4ba623c7dd39914deff44f081b36ba46a509`
- Witness SHA-256: `c391f21d5693c31d16f59865bcb173b58d752195a7fff6783c9854a07d69fa4f`
- Stress SHA-256: `ff11f416008e2dac988a647d2eb1a56be2da3246a701ddfea24dfd6690bd34df`

Run:

```sh
PYTHONPATH=validation python -m unittest -v test_assumption_reliability_budget.py
PYTHONPATH=validation python assumption_reliability_budget.py \
  --output /tmp/assumption-witness.json \
  --stress-output /tmp/assumption-drift-stress.json \
  --repeats 1000 --audits 1000
```

## Limits and next gates

This prototype does not establish that real audit labels are independent, that the bound issuer has stable future reliability, or that historical audits transfer to a new epoch. It does not integrate into production Rust dispatch. The next experiment must introduce predeclared scope-specific validity windows and a predictable/online monitoring rule; compare stationary calibration, abrupt drift, gradual drift, correlated failures and wrong-scope reuse; and measure both false acceptance and false refusal. Industry/SOTA comparisons should use matched guarantees, not raw average sample cost.
