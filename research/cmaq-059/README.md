# CMAQ-059 — Finite-sample calibration and benchmark integrity

Status: implementation and metric primitives; no Monte Carlo qualification result is claimed.

## Purpose

CMAQ-057 and CMAQ-058 have deterministic correctness fixtures and written conditional-supermartingale arguments. Those artifacts do not estimate operating characteristics. Conversely, a Monte Carlo sweep cannot prove that the required conditional-expectation assumptions hold.

CMAQ-059 supplies finite-sample reporting primitives and a paired-stream seed schedule. It is not a test statistic, scenario generator, faithful reproduction of an external baseline, or SOTA claim.

## Calibration metrics

For x false alarms among n independent full-horizon null replicates:

- Report x/n and a two-sided exact Clopper–Pearson interval.
- For the complete preregistered family of J null cells, report a one-sided Clopper–Pearson upper bound with per-cell tail probability (1-confidence)/J.
- Keep pointwise intervals and familywise upper bounds separate.
- The null family includes all preregistered arm × scenario × horizon × nominal-alpha cells. Do not omit weak cells after observing results.

The Bonferroni construction gives simultaneous coverage without assuming independence between cells. Independence is still required at the replicate level for each cell's binomial interpretation. A full generated horizon is one replicate; overlapping windows and repeated time steps are not independent trials.

The tolerance comparison is an empirical screen, not a theorem and not a deployment qualification. A theorem-backed anytime guarantee continues to depend on the conditional null, predictability, norm bounds, and implementation assumptions.

## Detection accounting

A `DetectionTrial` defines `change_point` as the number of initial pre-change observations. An alarm at sequence less than or equal to that value is an early alarm, not a detection. A post-change alarm at sequence t has delay t-change_point. A miss or early alarm is assigned restricted delay H+1.

Unconditional power uses all full-horizon replicates as its denominator. Restricted-delay quantiles include both misses and early alarms at H+1. Complete-case-only delay is prohibited.

## Reproducible paired streams

The seed key is (experiment_id, scenario_id, replicate_id). Algorithm-arm identity is deliberately absent: each arm must receive the same underlying stream for a given replicate. The SHA-256 digest is a deterministic seed schedule, not a PRNG, source of entropy, or proof of independence.

The future runner must record:
- PRNG algorithm and version;
- derived seed digest;
- generated stream digest and generator configuration;
- source/common-cause block identity;
- monitor code SHA and benchmark-manifest SHA.

When multiple sequences share a common cause, statistical resampling and independence claims must use the source/common-cause block as the unit.

## Primary statistical problem

CMAQ's primary null is direct conditional paired equality:

```
Law(X_t | Z_t, F_(t-1)) = Law(Y_t | Z_t, F_(t-1))
```

where Z_t is available before the current pair and F_(t-1) contains all previous observations and decisions. Valid null scenarios may include changing predictable covariates, serial dependence, and predictable heteroskedasticity, provided conditional paired equality continues to hold.

Do not conflate this with:
- conditional independence testing (for example SKCI);
- unconditional online two-sample change detection (for example Online RFF-MMD);
- scalar conditional-conformal restart portfolios.

Those are adjacent protocol tracks, not direct primary head-to-head results.

## Null families

1. Independent paired equality with stationary covariates.
2. Conditional paired equality under deterministic and stochastic covariate drift.
3. Conditional paired equality under predictable heteroskedasticity.
4. Serially dependent martingale innovations preserving conditional equality.
5. Pair-order/common-cause constructions with a clearly stated filtration.
6. Replay and duplicate-ID controls, treated as integrity negatives.

## Alternative families

1. Low-, medium-, and high-frequency conditional response differences.
2. Equal marginal X/Y distributions but changed conditional pairing.
3. Sparse covariate support and local regime changes.
4. Covariate-dependent scale, tail, or multimodality changes.
5. Oscillating signs and context-specific effects.
6. Late change points and repeated on/off regimes.
7. Finite Fourier-bank blind spots, including alternatives nearly orthogonal to all registered features.

Report every family separately. A pooled average must not hide a feature-bank blind spot.

## Invalid-assumption controls

Current-outcome feature leakage, changed conditional means in a reference-calibrated mode, reference/test mismatch, norm-bound violations, and hidden duplicate/common-cause streams are invalid-assumption controls. Report these as contract violations or negative controls, not ordinary valid-null false-alarm cells.

## Comparator matrix

| Arm | Role | Conditions |
|---|---|---|
| CMAQ-055 | Full policy and capital restart; correctness oracle | Same conditional-pair streams and alpha |
| CMAQ-058 | Capital-only exact renewal with shared policy state | Never describe as full policy restart |
| CMAQ-054 | Fixed kernel-family mixture | Same fixed feature inventory and priors |
| CMAQ-057 | Predictable adaptive factor stream | No current-observation selection |
| SCPT | Direct conditional paired baseline | Faithful historical KNN and betting-fraction adaptation |
| ICML 2026 kernel-family mixture | Anytime-valid family comparator | Verify conditional paired assumptions before including in primary table |
| Best fixed expert chosen in hindsight | Diagnostic oracle only | Not an alpha-valid competitor |

C055 and C058 are distinct procedures. C055 resets both policy state and capital at each start. C058 restarts capital while retaining one shared predictable policy state. Report their difference directly.

## Faithful SCPT reproduction contract

The SCPT arm is implemented in `src/scpt.rs` using the procedure in Zheng & Zhang (arXiv:2611.22886, 2026), not a static-kernel/static-bet simplification:

- Gaussian response kernel, (K(x,y)=\exp(-\|x-y\|^2/(2\sigma^2))), with a fixed, recorded bandwidth.
- Euclidean KNN neighborhoods over prior covariates only, (k_a(t)=\lceil a\sqrt t\rceil), capped by the eligible historical count, for (a\in\{2,4,6\}).
- The sequential KNN RKHS witness payoff (~ S_t) uses both the current pair and the historical KNN witness norm with the published √2 and η truncation.
- Each (a)-configuration estimates its betting fraction from past payoffs in the current covariate's neighborhood, using Equation (16) with (w(Z_t)=1).
- The published equal-weight configuration mixture averages the three predictable factors at each time; it does not select the current maximum.
- The SCPT implementation records KNN sizes, payoffs, betting fractions, factors and a chained trace.

The paper sets the continuous-response Gaussian bandwidth from the first 50 observations and then holds it fixed. For a clean online comparison, C059 exposes a pre-monitoring warm-up constructor: the first 50 observations calibrate σ and initialize the previsible payoff history, but no e-capital is accumulated during warm-up. This is a validity-safe reproduction variant, not an assertion that the paper's reported protocol used a separately declared calibration split. The benchmark must label it `SCPT-first-50-warmup` and report the 50-observation warm-up explicitly.

For the primary apples-to-apples table, use a second `SCPT-fixed-bandwidth` arm with σ estimated from a separate, preregistered calibration stream and held fixed for all test arms. Record the calibration-stream seed, response pool, median convention, resulting σ and stream digest. This prevents one comparator from silently consuming test observations for bandwidth fitting while the other arms use them as scored observations.

SCPT's Theorem 1 states its time-uniform Type-I guarantee under the full conditional paired null; its main construction uses an additional conditional independence assumption for simplifying the method and power analysis. The paper's finite-history dependence extension is a separate variant and must be labelled separately. Do not generalize the SCPT baseline beyond the exact filtration and dependence treatment reproduced.

## Counts and horizons

Initial primary grid:
- H in {128, 256, 512, 1024};
- nominal alpha in {0.01, 0.05, 0.10};
- at least 10,000 independent full-horizon null replicates per primary cell;
- at least 5,000 independent alternatives per primary cell;
- change points registered before execution, each strictly less than H.

A confirmatory precision tranche may expand null cells to 50,000 replicates according to a predeclared, arm-independent precision rule. Do not increase sample size because a particular arm's result looks favorable.

## Reporting

Every run should preserve:
1. Pointwise false-alarm rate and exact two-sided interval for each null cell.
2. Simultaneous one-sided upper bounds over the complete frozen null-cell family.
3. Unconditional power and exact interval, with early alarms reported separately.
4. Restricted mean delay and p50/p90/p95; misses and early alarms count as H+1.
5. Worst-family power and delay, not only pooled means.
6. Paired arm differences on common streams, using source-block-aware uncertainty.
7. p50/p99 factor-production latency separately from restart bookkeeping latency.
8. Peak memory, feature dimensions/count, number of map evaluations, and expert count.
9. Exact code commit, configuration/manifest digest, raw per-replicate outcomes, and aggregate-artifact digests.

## Verified research frontier snapshot (2026-11-28)

The primary comparator is SCPT because it addresses the same online paired conditional-distribution null:

- Zheng & Zhang, [Sequential Conditional Paired-Sample Testing with Applications to Online Monitoring](https://arxiv.org/abs/2611.22886), arXiv:2611.22886 (2026). Its null conditions on the current shared covariate and full past. It uses past-only sequential KNN conditional-mean-embedding estimates and covariate-indexed predictable betting fractions. The paper's reported simulations use 500 repetitions per configuration, maximum horizon 4,000, and alpha 0.05; its dependent/nonstationary section reports maximum empirical null rejection of 3.8% for the mixture and 5.4% for individual SCPT procedures. Reproduce the method faithfully; do not substitute a static-kernel/static-bet baseline.
- This SCPT paper explicitly identifies the curse of dimensionality in nearest-neighbor conditioning and points to dimensionality reduction, learned representations, and adaptive distance metrics as future directions. This is a high-value CMAQ research target: improve conditional representation quality while preserving the prequential conditional-mean-zero proof.

Adjacent, not directly interchangeable, state-of-the-art results:

- [Anytime-Valid Two-Sample Testing over Kernel Families](https://github.com/ds9-code/anytime-valid-kernel-testing), ICML 2026 Workshop on Hypothesis Testing. The published repository composes per-kernel betting e-processes into a fixed-prior mixture and reweights family members online; it reports an additive log(K) sample cost and matching lower bound. It is a two-sample kernel-family comparator, not automatically a valid conditional paired baseline. Verify the precise null and adaptation assumptions before including it in the primary table.
- Kalinke & Gavioli-Akilagun, [Optimal Online Change Detection via Random Fourier Features](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e3b37f830edb6972bb1206324cb566f-Abstract-Conference.html), NeurIPS 2025. It reports logarithmic time per observation, logarithmic space, and minimax-optimal delay guarantees for multivariate online two-sample change detection. This is an important computational baseline, but its unconditional two-sample target must remain a separate protocol table.
- He & Sutherland, [Sequential Kernel-based Conditional Independence Testing via Adaptive Betting](https://proceedings.mlr.press/v306/he26ah.html), ICML 2026, PMLR 306. This is conditional-independence testing under a Model-X conditional-distribution premise, with adaptive kernel optimization and calibration corrections. It is not the direct paired conditional-distribution equality null.

The comparison rule is therefore: same-null and same-filtration competitors belong in the primary rank table; adjacent problems belong in separate tables with their own metrics. Do not combine their scores into a single leaderboard.

## Prospective performance targets — not achieved results

- At matched alpha, seek at least 20% lower restricted mean detection delay on three preregistered shift families versus the designated comparator.
- No more than 2 percentage points of worst-family power regression.
- No material null-calibration exceedance under simultaneous familywise intervals.
- At H=1024, investigate a target of at least 10× lower p99 restart-bookkeeping time for C058 versus C055, measured on the same host and separately from factor generation.

These are research targets. No SOTA, statistical qualification, physical safety, or deployment claim is permitted until faithful baselines, exact-head code, raw artifacts, and confidence intervals are independently reviewed.
