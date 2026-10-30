# Corrected dependent-normal multi-stream benchmark (200 hypotheses)

**Date:** 2026-10-30  
**Status:** 250-repetition local pilot; 9/9 local unit tests passed; not the full 10,000-repetition paper reproduction and not a SOTA claim.

## Research objective

Reproduce the dependent-normal simulation in Tavyrikov, Goeman & de Heide, *Carefree multiple testing with e-processes*, Electronic Journal of Statistics 20 (2026), 3178–3189, DOI: https://doi.org/10.1214/26-EJS2546.

The paper specifies 200 hypotheses (100 null, 100 alternatives), null mean 0, alternative mean 0.1, unit marginal variance, cross-hypothesis dependence from a fixed random correlation matrix, 2,000 independent-over-time observations per hypothesis, and 10,000 Monte Carlo repetitions. The paper's Figure 1 compares standard/current e-BH, running-maximum e-BH and adjusted running-maximum e-BH (A1 and A2).

Paper full text: https://ris.utwente.nl/ws/portalfiles/portal/499132886/2501.19360v1.pdf

## Reference implementation audit

The authors' linked code is https://github.com/tavyrikov/runmax_eBH/.

A source audit found three reproduction details which are recorded rather than silently inherited:

1. In `simulation.py`, standard e-BH receives `E[:, t]`, where `E` is the single-observation likelihood-ratio increment, while running-maximum methods receive cumulative likelihood ratios. The paper's mathematical definition of the current e-process is the cumulative product. The benchmark therefore reports both `reference_script_single_step_ebh` (linked script behavior) and `current_cumulative_ebh` (mathematically intended current e-process). They are not interchangeable.
2. The linked script forms `cmax_Ep=np.maximum.accumulate(E_cumulative, axis=1)` without explicitly including the baseline value 1. The admissible adjusters are defined on `M>=1`. This benchmark uses `M_t=max(1,sup_{s<=t}E_s)`, A1(1)=1/2 and A2(1)=0, with stable log-domain evaluation.
3. The linked proof-counterexample script uses ±1 as log increments for factors 1/2 and 2, commenting that log(2) is approximately 1. That changes the marginal multiplicative expectation from 1 to (2e^-1+e)/3 = 1.1513469... . The independent two-stream negative control in PR #9110 uses exact factors instead.

The paper describes 10,000 repetitions, while the linked `simulation.py` sets M=100,000. This pilot uses 250; any final reproduction must clearly report 10,000 and optionally a separate 100,000 profile. These are reference-code/protocol reconciliation issues, not objections to the mathematical theorem.

## Corrected pilot protocol

- 200 hypotheses; first 100 are true nulls, remaining 100 alternatives.
- Each observation vector is multivariate normal with a fixed correlation matrix and independent time increments.
- Null mean 0; alternative mean 0.1; marginal variance 1.
- 2,000 observations per hypothesis.
- alpha=0.05.
- Correlation eigenvalues are independently generated as Exp(1), normalized to sum 200; SciPy's `random_correlation.rvs` generates the matrix.
- Correlation seed 20261030; data seed 20261031.
- Correlation matrix SHA-256 (little-endian float64 row-major): `e46b0a40b58c59cb39e301a0fe487e4a82c581219018a17aeec99854ac2a2fa2`.
- 250 Monte Carlo repetitions, processed in batches of 8 in the executed pilot.
- FDR(t) is the Monte Carlo mean FDP(t). FDR-sup(t) is the Monte Carlo mean of each replicate's maximum FDP over all times up to t. Power(t) is the mean fraction of alternatives rejected.
- Confidence intervals are normal Monte Carlo intervals based on replicate-level sample variance (mean ± 1.96 SE); they quantify simulation error, not statistical coverage of the testing procedures.

## Results

| Method | Final FDR | Final FDR-sup | Final power |
|---|---:|---:|---:|
| Linked script single-step e-BH | 0.0000% | 0.0000% | 0.00% |
| Current cumulative e-BH | 0.0944% | 1.8674% | 92.076% |
| Naive running-max e-BH | 2.0604% | 3.1118% | 96.236% |
| Adjusted running-max A1 | 0.0430% | 0.0610% | 73.808% |
| Adjusted running-max A2 | 0.0418% | 0.0595% | 75.596% |

Final FDR-sup and power Monte Carlo 95% intervals:

| Method | FDR-sup MC 95% interval | Power MC 95% interval |
|---|---:|---:|
| Linked script single-step e-BH | [0, 0] | [0, 0] |
| Current cumulative e-BH | [1.505%, 2.230%] | [91.711%, 92.441%] |
| Naive running-max e-BH | [2.659%, 3.564%] | [95.973%, 96.499%] |
| Adjusted running-max A1 | [0.018%, 0.104%] | [73.119%, 74.497%] |
| Adjusted running-max A2 | [0.018%, 0.101%] | [74.867%, 76.325%] |

Interpretation:
- In this single correlation matrix, naive running-max e-BH remains below alpha, which does not contradict the paper's arbitrary-dependence counterexample.
- Adjusted A1/A2 have very low observed FDR-sup, but pay a sizable power penalty versus the unadjusted maximum.
- Current cumulative e-BH has higher power than adjusted methods in this particular experiment, but does not provide the monotone/carefree rejection property that motivated the running-max criterion.
- The linked script's single-step e-BH row has no power in this setup and should not be confused with applying e-BH to the current cumulative e-process.

These results are preliminary because 250 repetitions are far below the paper's 10,000, and only one random correlation matrix is used. They reproduce the qualitative figure shape for the running-max/adjusted methods, not the paper's precise Monte Carlo curves.

## Artifacts and reproduction

- `validation/multistream_dependent_normal.py`: independent implementation; includes both the linked-script single-step row and the mathematically intended cumulative-current e-BH row.
- `validation/test_multistream_dependent_normal.py`: 9 tests for adjuster boundaries/domain, monotonicity, deterministic PSD correlation generation, e-BH counts, FDR-sup accounting, comparison separation and small-run reproducibility.
- `validation/DEPENDENT_NORMAL_200H_250R_2026-10-30_SUMMARY.json`: compact per-method summary and protocol.
- Full aggregate curves are provided as a separate attached JSON artifact; the repository summary contains their digest.

SHA-256:
- source: `9c7bd2b580633b5579fe90f1139b4e9cefa7d8186e4658e7e055de9fc7289a14`
- tests: `a8dd1bc7a5a4f42c6e16b2348009c4a61c3c94442604f99312f119d980011641`
- full aggregate JSON: `2278cf8d7b5dd1c27ce0cb0c1c2e7ee76e1c8ec80ca98d6e1ad97cce360b3368`
- compact summary JSON: `80286c2a4cc5564d6bd30039cb8a3034971587b0d7ee5143f3c0ec48f396d708`

Reproduce the local pilot:

```sh
PYTHONPATH=validation python -m unittest -v test_multistream_dependent_normal.py
PYTHONPATH=validation python validation/multistream_dependent_normal.py \
  --output /tmp/dependent-normal-results.json --repetitions 250 \
  --timepoints 2000 --batch-size 8 --seed 20261031
```

## Next gates

1. 10,000 repetitions matching the paper's stated protocol.
2. Multiple independently generated correlation matrices with committed matrix digests.
3. Mixed-null proportions (10%, 50%, 90%, 100%), alternative effect sizes and marginal variances.
4. Independent, block-correlated, equicorrelated and common-factor dependence.
5. Compare power-preserving conditional calibration and stochastic rounding, the two directions highlighted by the paper.
6. Report FDR, FDR-sup, FWER, power, mean rejections and Monte Carlo intervals; do not infer FWER from FDR-sup outside all-null cases.
7. Compare against exact author implementation after correcting the identified source/protocol mismatches; no SOTA claim until this direct gate is complete.
