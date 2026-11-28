# CMAQ-055 — Restart × kernel-family benchmark contract v0.1

Status: preregistration draft. No benchmark result or superiority claim is implied.

## 1. Research question

Does a fixed-prior mixture over both conditional-kernel experts and possible change starts improve worst-family detection delay, relative to:
- a fixed conditional Fourier expert;
- fixed mixtures over registered conditional representations;
- an anytime-valid kernel-family mixture without restarts;
- a faithful sequential conditional paired-sample baseline?

The experiment must distinguish representation adaptation, onset adaptation and the interaction. The exact all-start CMAQ-055 oracle is capped at 1,024 observations; it cannot be used to claim parity with experiments whose change points occur at 2,000 or 8,000 observations.

## 2. Primary task: conditional paired monitoring

Each observation is (Z_t, X_t, Y_t), where Z_t is available before the response pair. The null is the common conditional paired-distribution null:

    Law(X_t | Z_t, F_(t-1)) = Law(Y_t | Z_t, F_(t-1)).

For martingale-null cells, the exact stochastic contract must be stated separately. CMAQ-052's common-conditional-mean calibration is not a blanket claim for arbitrary conditional-distribution nulls; only use the CMAQ-052 alpha split where its fixed-mean and bounded-increment assumptions are satisfied by the child monitor.

All methods receive the same ordered pairs, covariates, source IDs, reference/calibration data, target, alpha and observation horizon. No method may use future covariates, select features after outcomes, drop misses, or obtain a larger calibration bank.

## 3. Registered primary arms

1. CMAQ-050: single fixed continuous-covariate product-Fourier monitor.
2. CMAQ-051: fixed context/product representation mixture.
3. CMAQ-054: fixed-prior multiscale product-Fourier kernel family.
4. CMAQ-055: all-start × kernel-family mixture, exact dyadic start prior.
5. SCPT (Zheng & Zhang, arXiv:2611.22886): faithful sequential conditional paired-sample baseline, using the published version and its exact assumptions.
6. Sreedhar & Li (ICML 2026 Workshop): faithful anytime-valid two-sample kernel-family mixture. Use their repository and protocol as the implementation reference; adapt to paired conditional data only if the required assumptions are preserved and the adaptation is documented.

Also report:
- every fixed CMAQ-054 expert as a diagnostic arm, not as an alpha-free selection baseline;
- a registered single-kernel conditional discrepancy baseline;
- a pooled paired-mean baseline, explicitly labeled as first-moment-only.

All sequential arms use the same outer alpha_test. The CMAQ-055 prior mixture does not Bonferroni-split alpha across starts or kernel members; each component stopping threshold is scaled by its exact prior mass.

## 4. Related but non-poolable protocols

These are important external frontier comparisons, but they are not the same primary task and must be kept in separate result tables.

### PF-CCTM / CCTM

Reproduce the published fixed-reference scalar score experiment faithfully before comparison: predictive-entropy score, ImageNet-C severity-5 setup, corruption families, calibration sizes, pre-change delay, change points, horizon, seeds and missed-detection treatment. The paper's reported 50.5 versus 244.5 median delay at n=1,000 and t0=8,000 is a paper result, not a CMAQ measurement. Do not compare it numerically to a 1,024-step synthetic paired experiment.

### Online RFF-MMD change detection

Reproduce the NeurIPS 2025 multivariate two-sample online change-detection protocol, including its logarithmic-time/logarithmic-space claims, threshold and window aggregation. It is a useful efficiency baseline for nonparametric distribution shift, not a direct replacement for conditional paired inference.

## 5. Synthetic paired-data matrix

### Null families

A. Exact paired equality: X_t=Y_t, with changing predictable covariates.

B. Stationary conditional equality with serial dependence that preserves the stated conditional null.

C. Predictable heteroskedastic martingale increments with one fixed conditional mean and a valid hard norm bound, for CMAQ-052-compatible cells only.

D. Covariate-marginal shift with unchanged conditional response law. This must not be counted as a conditional-distribution alternative when the conditional null remains true.

E. Common-cause and replay controls: duplicated source-pair IDs, overlapping clips, clock resets and identity remapping. Structural guards should reject invalid identities; do not count those rejections as statistical detections.

F. Invalid-assumption controls: conditional-mean drift, reference/test mismatch and bound violations. Report these as contract violations, not as valid-null false-alarm cells.

### Registered alternatives

1. Smooth low-frequency conditional changes.
2. Narrow/high-frequency conditional changes.
3. Sparse covariate-region changes.
4. Oscillating sign changes across predictable contexts.
5. Equal X and Y marginals with changed X/Y association conditional on Z.
6. Conditional variance, tail and skew changes with unchanged mean.
7. Multi-scale mixtures and alternatives lying between registered Fourier frequencies.
8. Changes with altered response law but unchanged covariate marginal.
9. Adversarial alternatives approximately orthogonal to the frozen finite feature inventory.

For each alternative freeze the generator, effect size, support, frequency and seed list before evaluating the methods. Include at least three effect strengths and a weak-signal boundary near each method's estimated detection limit.

## 6. Horizon and change-point grid

For the current CMAQ-055 oracle, use horizons H in {128, 256, 512, 1,024} and change points t0 in {0, 16, 64, 128, 256, 512}. A cell is eligible only when t0 < H.

Do not run or report a CMAQ-055 result beyond its implemented horizon cap. An optimized restart implementation may later preregister the larger grid:

    H = 10,000
    t0 in {0, 125, 500, 2,000, 8,000}.

The larger grid is also required for a faithful comparison to late-onset PF-CCTM settings, but only under its own task-specific protocol.

## 7. Sample sizes, alpha and inference

Primary statistical cells:
- at least 10,000 independent full-horizon null paths per method × null family × horizon × alpha cell;
- at least 5,000 alternative paths per method × alternative family × effect × change point;
- alpha in {0.01, 0.05, 0.10};
- paired common random numbers across methods where valid;
- source-level independence and effective-block inventory explicitly recorded.

Use exact binomial confidence intervals for full-horizon ever-alarm and unconditional detection probabilities. Report both point estimates and intervals. Paired method differences use a block bootstrap over independent source/common-cause blocks, never overlapping clips as independent samples.

At alpha=0.05, 10,000 null paths give a binomial standard error of about 0.22 percentage points; at alpha=0.01, about 0.10 percentage points. Do not describe a noisy estimate as proof of exact finite-sample control.

## 8. Metrics

Primary:
- full-horizon ever-false-alarm probability and confidence interval;
- unconditional detection probability, including misses;
- restricted delay with a miss assigned H+1;
- p50/p90/p95 delay among all trials using the preregistered restricted convention;
- worst scenario-family result and family-wise regression;
- calibration-envelope miss probability reported separately from conditional e-process false alarms.

Secondary:
- per-expert and joint start/kernel posterior trajectories;
- median/p95/p99 update latency on the target deployment CPU;
- peak resident memory and steady-state bytes per active expert/start;
- feature count, pair-ledger size and allocation count;
- time to warning, time to quarantine and fallback completion, when connected to an intervention controller.

Never report median delay only among detected paths without the unconditional miss rate and restricted-delay statistic.

## 9. Acceptance targets (prospective, not achieved)

- Null upper confidence bound must not materially exceed the registered alpha; any excess is investigated by family and not hidden in pooled totals.
- At least 20% improvement in paired restricted mean delay on three preregistered alternative families, while maintaining null control and no critical-family miss/power regression greater than 2 percentage points.
- No more than 1% of the actual control-cycle budget at p99 for production deployment targets; the oracle is expected to fail this target at larger horizons.
- For intervention-connected evaluations, at least 99.9% of eligible fallbacks complete before independently measured recoverability exhaustion, with denominator and exclusions reported.
- No SOTA claim without an exact reproduction of the strongest eligible baseline and the same data, alpha, horizon, miss rule and compute budget.

## 10. Evidence and reproducibility receipts

Every run must preserve:
- commit SHA and dirty-tree state;
- compiler/toolchain, dependency lock and build flags;
- generator and method config digests;
- seed manifest and independent source/block inventory;
- reference/calibration manifest and target digest;
- exact alpha and threshold formula;
- per-path stop time, change point, detected/missed flag and intervention times;
- full output CSV/Parquet plus summary JSON and confidence intervals;
- runtime/memory hardware description;
- all rejected, missing and invalid runs with explicit reason.

No skipped, queued or stale-head workflow is a pass. A smoke artifact is a wiring check, not benchmark evidence.
