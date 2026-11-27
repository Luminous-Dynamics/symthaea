# SDCB block-conformal open-world adequacy calibration v1

**Status:** exact finite-rank diagnostic. Eleven Rust tests are authored; the focused Rust 1.96 workflow is the execution gate. A false-alarm guarantee applies only under the declared exchangeability assumptions; no real-data calibration or time-series validity claim is made.

## 1. Why this improves the fixed-threshold gate

The open-world diagnostic in the prior increment (#13942) correctly identified that normalized Bayesian model weights do not prove any model is adequate. However, its caller-supplied log-density threshold was illustrative and had no calibration procedure.

This increment replaces the arbitrary universal-cutoff idea with a **rank-based, block-level nonconformity test**. It does not change Bayesian model weights or infer a posterior probability that the model set is wrong.

## 2. Nonconformity score and block aggregation

Freeze the model set and scoring protocol before calibration. For observation \(y_t\), define a larger-is-worse score such as:

\[
a_t=-\max_{s:w_s>0}\log p(y_t\mid \mathcal D_{<t},s).
\]

A calibration unit \(b\) is an independent episode or an explicitly exchangeable block. To control for within-unit extremes, use the maximum score within each block:

\[
A_b=\max_{t\in b}a_t.
\]

For a future test episode/block, compute the same block maximum \(A_*\). With \(n\) calibration units, the conservative upper-tail conformal p-value is:

\[
p_*=\frac{1+\sum_{b=1}^n\mathbf 1\{A_b\ge A_*\}}{n+1}.
\]

Flag the model set as out-of-envelope only when \(p_*\le\alpha\). Ties are counted conservatively. The minimum attainable non-randomized p-value is \(1/(n+1)\); the implementation reports whether the available calibration sample can resolve the requested \(\alpha\).

The calibration and test unit must use exactly the same canonical lowercase 64-character SHA-256 model-set digest, scoring-protocol digest, observation units/transformation, and block construction. Malformed or merely descriptive strings are rejected; otherwise the rank comparison is rejected.

## 3. Guarantee and its limits

If calibration units and the new test unit are exchangeable, and the model/scoring rule is frozen independently of those units, then the rank p-value is super-uniform:

\[
\Pr(p_*\le\alpha)\le\alpha.
\]

This gives a finite-sample false-alarm bound for the null that the test unit is exchangeable with the calibration-unit population under the fixed scoring protocol. It is not a probability that the model set is physically true or false.

**Crucial temporal caveat:** splitting one time series into non-overlapping blocks does not automatically make those blocks exchangeable or independent. The block unit must be defensible (e.g. independently generated campaigns/episodes or a justified exchangeable cluster design). Under residual temporal dependence or distribution shift, ordinary split conformal may lose its guarantee. Use a dependence-aware time-series method, explicit stationarity/dependence assumptions and uncertainty analysis, or label the result a diagnostic rather than a calibrated false-alarm test.

Recent time-series conformal theory explicitly studies dependence-adjusted coverage rather than assuming ordinary exchangeability (Zhai, Cheng & Wu, 2026, arXiv:2609.33868). Recent sequential conformal work also shows why fresh calibration folds are needed when earlier decisions recursively select later calibration populations (Yang et al., 2026, arXiv:2610.12937).

## 4. Adaptive stages and calibration custody

At each adaptive stage, the scoring function and model-set digest are frozen. This reference requires:
- a distinct calibration epoch ID at every stage;
- no reuse of any previously consumed calibration or test unit as either calibration or test data at a later stage;
- no overlap between the current calibration fold and current test unit;
- no use of the test score to fit or tune the model/scoring protocol;
- explicit propagation of the consumed epoch/unit IDs to the next stage.

The helper is pure: it reports the union of calibration-unit IDs and the evaluated test-unit ID as `consumed_unit_ids`, plus the consumed calibration epoch. The caller must preserve the complete ledger. This prevents a test episode from being recycled into a later adaptive calibration fold. A future integration should bind fold receipts to the SDCB/EGBT evidence version and replay.

## 5. Frozen fixture

Nineteen calibration units have block maxima \(0.1,0.2,\ldots,1.9\).

- Test block maximum 2.0: p-value \(1/20=0.05\), so flag at \(\alpha=0.05\).
- Test block maximum 1.75: two calibration scores are at least as large, p-value \(3/20=0.15\), so do not flag.
- With four calibration units, the minimum p-value is 0.2, so \(\alpha=0.05\) is not resolvable.

This is a constructed rank fixture, not a sensor-data result.

## 6. Benchmark gates

- Exact agreement with independent rank enumeration.
- Conservative tie handling.
- Zero calibration/test ID overlap.
- Zero cross-stage epoch/unit reuse, including test-to-calibration recycling.
- Zero scoring-protocol or model-set digest mismatch acceptance.
- Empirical false-alarm rate at or below nominal alpha on independent exchangeable held-out episodes, with binomial confidence intervals.
- Under dependent time-series tasks, do not claim the exchangeable-block guarantee; compare against dependence-aware methods and report coverage/error bounds, block size, and drift sensitivity.
- Measure detection delay and false alarms after held-out covariance regime changes, model omissions and common-cause shifts.

No universal threshold, distribution-free guarantee under arbitrary serial dependence, physical causal diagnosis, or SOTA result is claimed.
