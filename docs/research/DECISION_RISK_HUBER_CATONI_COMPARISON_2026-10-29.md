# Matched Huber–Catoni confidence-sequence benchmark

**Date:** 2026-10-29. **Status:** exploratory synthetic research; independent equation reimplementation; not production integration or direct official-code reproduction.

## Question

Can a general robust sequential confidence sequence improve upon our Bernoulli-specific sensitivity inversion? This is a comparison with the published method, not a claim of novelty. Wang & Ramdas (AISTATS 2023) present Huber-robust confidence sequences for a mean under a known upper bound on a p-th central moment, allowing a bounded fraction of arbitrary distribution corruption. Their intervals are valid at data-dependent stopping times under their stated assumptions. The authors' reference repository is GPL-3.0. This bundle independently implements the paper equations; it does not copy or execute the reference code. Full validation against the reference implementation remains open.

Paper: https://proceedings.mlr.press/v206/wang23p.html
Reference repository: https://github.com/HJ-Wg/catoni-confseq

## Methods

1. **Nominal beta-mixture CS:** estimates the observed mixture mean and incorrectly treats it as the clean target when contamination is present.
2. **Sensitivity beta CS:** builds an anytime beta-mixture interval for the observed mean, then maps it to a clean-mean interval using the declared contamination upper bound and known support [0,1].
3. **Paper-equation Huber–Catoni CS:** independently implements the robust Catoni-CS influence function and root equations described in the paper, using a clean-distribution variance bound of 0.25 for Bernoulli outcomes. This construction is designed for a broader moment-bounded model class, so it can be more conservative in this bounded-support benchmark.

Each method uses a familywise delta of 0.10 split across four cells; paired methods reuse the same stream across declared-bound conditions. Clean target risks are [0.15, 0.37, 0.60, 0.80], model predictions are [0.15, 0.30, 0.60, 0.80], true maximum error is 0.07, and tolerance is 0.05. One cell is contaminated with R=0. The sequential panel uses 50 repeats per actual-rate and declared-bound condition, evaluating at per-cell sample counts 512, 1024, 2048 and 4096.

## Matched sequential results (50 repeats per condition)

### Actual contamination 12%

| Declared bound | Method | Unsafe admissions | Unresolved | Mean observations |
|---:|---|---:|---:|---:|
| 0% | Nominal beta CS | 12/50 | 38/50 | 16,220 |
| 0% | Sensitivity beta CS | 12/50 | 38/50 | 16,220 |
| 0% | Huber–Catoni CS | 4/50 | 46/50 | 16,384 |
| 2.5% | Sensitivity beta CS | 0/50 | 50/50 | 16,384 |
| 2.5% | Huber–Catoni CS | 0/50 | 50/50 | 16,384 |
| 12% (covers actual) | Sensitivity beta CS | 0/50 | 50/50 | 16,384 |
| 12% (covers actual) | Huber–Catoni CS | 0/50 | 50/50 | 16,384 |

### Actual contamination 18%

| Declared bound | Method | Unsafe admissions | Unresolved | Mean observations |
|---:|---|---:|---:|---:|
| 0% | Nominal beta CS | 48/50 | 2/50 | 13,435 |
| 0% | Sensitivity beta CS | 48/50 | 2/50 | 13,435 |
| 0% | Huber–Catoni CS | 38/50 | 12/50 | 16,384 |
| 2.5% | Sensitivity beta CS | 14/50 | 36/50 | 16,384 |
| 2.5% | Huber–Catoni CS | 0/50 | 50/50 | 16,384 |
| 5% | Sensitivity beta CS | 0/50 | 50/50 | 16,384 |
| 5% | Huber–Catoni CS | 0/50 | 50/50 | 16,384 |
| 18% (covers actual) | Sensitivity beta CS | 0/50 | 50/50 | 16,384 |
| 18% (covers actual) | Huber–Catoni CS | 0/50 | 50/50 | 16,384 |

Every sequential-panel outcome was either admit or unresolved; none rejected. The model is unsafe in every row. Underestimating contamination can still produce false admissions even for the sensitivity method. With a bound that covers actual contamination, neither robust method produced an unsafe admission in this 50-repeat panel, but every run remained unresolved at the cap. Zero observed failures in 50 trials leaves a two-sided Wilson 95% upper limit of approximately 7.1%; this is weak evidence, not a population guarantee.

## Fixed-time coverage / interval width (300 repeats at n=1,024)

| Condition | Method | Clean-target coverage | Mean interval width |
|---|---|---:|---:|
| No contamination | Nominal beta CS | 300/300 (100%) | 0.116 |
| No contamination | Huber–Catoni, epsilon=0 | 300/300 (100%) | 0.123 |
| 12% contamination, declared epsilon=0 | Nominal beta CS | 248/300 (82.7%) | 0.112 |
| 12% contamination, declared epsilon=0 | Huber–Catoni, epsilon=0 | 237/300 (79.0%) | 0.123 |
| 12% contamination, declared epsilon=0.12 | Sensitivity beta CS | 300/300 (100%) | 0.264 |
| 12% contamination, declared epsilon=0.12 | Huber–Catoni, epsilon=0.12 | 300/300 (100%) | 0.864 |

Nominal intervals target the observed mixture mean, not the clean target, so coverage drops under contamination. In this bounded Bernoulli setting the sensitivity interval is much narrower than the general moment-robust Catoni interval. This is a reason to exploit known support when justified; it is not evidence that Catoni is inferior for unbounded or heavy-tailed outcomes. Observed 300/300 coverage is not a validity proof.

## Field-level interpretation: an assumption-indexed robustness portfolio

The promising architecture is not one universal robust interval, but a portfolio selected from a predeclared data contract:

- bounded [0,1] risks plus a defensible contamination cap: use a support-aware sensitivity set;
- broader moment-bounded measurements: use a separately validated Huber–Catoni method;
- unknown contamination/moment bounds, unsupported subgroups or identity mismatch: refuse to certify;
- sample-allocation optimization is downstream of method validity; never select assumptions based on which method admits the model.

This aligns with Wang & Ramdas (2023). Related current directions include robust bandit multiple testing under Huber contamination (AAAI 2026, https://ojs.aaai.org/index.php/AAAI/article/view/40148) and confidence intervals for linear models with arbitrary noise contamination that do not require a prior contamination level (Biometrika, online publication 28 Sep 2026, https://academic.oup.com/biomet/advance-article-abstract/doi/10.1093/biomet/asag058/8842600). The latter depends on a regression/Z-estimation structure and cannot be assumed to solve general multi-cell Bernoulli risk certification.

## Next benchmark gates

1. Numerically cross-check this independent equation implementation against the official reference implementation on a small, fixed set of identical samples; formally review the derivation with the paper.
2. Run at least 1,000 independent repeats on a broader grid including safe clean cases and measure false rejection as well as unsafe admission and abstention.
3. Extend beyond Bernoulli to bounded real losses, then heavy-tailed finite-moment data; sweep skew and variance.
4. Sweep actual versus claimed contamination, confidence level, cap, support and moment bound. Report confidence limits, worst-group coverage, P95/P99 cost and unresolved rate.
5. Evaluate the 2026 robust bandit multiple-testing method on a matched task when code/data are available.
6. Audit license compatibility before any incorporation of GPL reference code.

## Reproducibility and limitations

The local reproducibility bundle includes the two raw JSON outputs, both source versions needed to reconcile run-source provenance, tests, report, summary and checksums. The sequential benchmark is 50 repeats per condition; the fixed-time coverage panel is 300. Data are IID Bernoulli with one contaminated cell and contaminant R=0. The independent Catoni equation implementation is not the official authors' code and needs further cross-validation. No SOTA superiority, production Rust integration, or remote CI pass is claimed.
