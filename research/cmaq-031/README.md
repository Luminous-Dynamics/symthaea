# CMAQ-031 — Anytime-Valid Calibration Drift Monitoring

**Status:** dependency-free Rust evaluator and deterministic tests authored; exact-head formatting, tests and strict Clippy pending.  
**Relationship:** post-deployment lifecycle for CMAQ-029 trajectory envelopes and CMAQ-030 recoverability-margin envelopes; protects CMAQ-027/028 commitment.  
**Claim ceiling:** marginal anytime type-I control under the fixed-reference exchangeability null, using independent cluster-block scores and independently authenticated tie-breaking. No arbitrary shift guarantee or proof that no shift exists after a non-alarm.

## Why the monitor is necessary

A calibration receipt can remain internally consistent after its statistical assumptions stop matching deployment. CMAQ-029/030 bind calibrators to a source/checkpoint/contract/scope/selection protocol, but even a deployment that appears within scope can experience a score-distribution change.

The system needs a post-deployment monitor that can raise an alarm at any stopping time, not merely after a fixed monitoring horizon.

## Prior art

Kuang and Xia (2026), *Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales*, derive the exact conditional distribution of the next rank when a fixed calibration set is reused. The predictive law is:

`pi[t,j] = (1 + N[t-1,j]) / (n + t)`

where n is the fixed reference size and N counts prior online ranks in bucket j. This explicitly accounts for the dependence induced by repeatedly comparing observations against one fixed reference set.

For a pre-registered bounded feature h_j on the rank grid:

`Z_t = h(R_t) - sum_j pi[t,j] * h_j`

is predictively centered under the exchangeability null. For a predictable fixed bet lambda in [-1,1]:

`E_t = 1 + lambda * Z_t`

is a nonnegative martingale factor. A mixture of component wealth processes remains an e-process; Ville's inequality controls the probability that the mixture ever exceeds 1/alpha.

This v0.1 intentionally uses fixed bets, not online Newton-step adaptation. That keeps the monitor small and its update rule auditable while preserving the anytime null guarantee under the stated assumptions.

Fang (2026), *Anytime Valid Conformal Prediction on Streaming Time Series*, develops time-uniform confidence sequences for true running and rolling average miscoverage under distribution shifts. It also makes clear that per-instance true-miscoverage guarantees cannot generally be achieved under arbitrary shifts; running/rolling average targets and explicit assumptions are required.

WATCH (ICML 2025) uses weighted conformal test martingales for online adaptation to mild covariate shift and diagnosis of concept vs out-of-support shifts. Timans et al. (UAI 2025) use testing-by-betting to monitor bounded risk violations under unknown shift. These are direct comparators.

## Fixed-reference rank monitor

The frozen reference contains one aggregate one-sided optimism score per independent calibration cluster, plus:
- cluster identity;
- tie-breaking key and RNG receipt;
- complete block-manifest digest;
- aggregate-score receipt;
- calibration profile, manifest and scope digests.

The reference is sorted by (score, tie_break_key) and never updated during monitoring. Each reconciled deployment cluster contributes one new aggregate score, one independently generated tie key and one authenticated score receipt.

For the new score, rank it in the fixed reference to obtain R in 1..=n+1. The monitor maintains rank counts and computes the exact predictive null mean before betting. It uses four pre-registered rank features:
- increasing order: detect a shift toward high optimism scores;
- decreasing order: detect a shift toward low scores;
- tail dispersion: detect heavy-tail / extreme-rank behavior;
- center concentration: detect central compression.

Each feature uses a fixed lambda = 1/2 and an equal-weight mixture. The threshold is 1/alpha. If the mixture crosses it, the calibration profile is permanently quarantined.

## Operational behavior

- Unknown previous execution outcome returns BlockedUntilReconciliation without changing monitor state.
- ConfirmedNotExecuted returns SkippedNotExecuted; it is not a zero score.
- A reconciled cluster block is accepted only if calibration profile, scope, block and score receipt identities match.
- Duplicate deployment clusters, reused calibration cluster IDs, or duplicate tie keys are rejected.
- A shift alarm freezes the old calibration profile. It is never silently updated with post-change observations.
- A new, separately sealed calibration profile with disjoint train/validation/calibration/evaluation clusters is required to restore calibrated commitments.
- A non-alarm is not evidence that every possible shift is absent. Hidden shifts that leave candidate-visible scores exchangeable are not identifiable by this monitor.

The PRM's false-alarm claim is marginal under the fixed-reference exchangeability null. It is not a theorem for arbitrary temporal dependence, conditional shifts, arbitrary hidden mechanisms, or non-random tie handling. Independent cluster-block construction and the tie-key RNG receipt are essential parts of the protocol.

## Numerical note

The rank predictive-centering calculation uses integer numerators for feature values and predictive weights. The final bounded payoff and log wealth are represented in f64 for this benchmark implementation. A production safety decision must keep the monitor alarm conservative with respect to numerical rounding and should receive independent numerical validation; this crate is not a formal floating-point proof.

## Deterministic witnesses

1. Predictive-centered feature payoff has zero expectation under the exact finite-rank predictive law.
2. Repeated reuse of the same calibration sample updates rank counts and changes the next predictive law.
3. High ranks grow the upper-order component.
4. Low ranks grow the lower-order component.
5. Alternating extreme ranks grow the tail-dispersion component.
6. A persistent high-rank stream crosses the anytime threshold and permanently quarantines the profile.
7. Unknown and not-executed outcomes do not update the rank stream.
8. Changed scope or calibration profile fails closed.
9. Duplicate cluster IDs, duplicate tie keys and reused calibration clusters are rejected.
10. The calibration reference remains unchanged after deployment observations.
11. Invalid rank inventory and malformed receipts are rejected.

## Matched baselines

- fixed threshold on one-step residual;
- rolling-window breach-rate alarm;
- independent conformal p-value product (negative control: invalid because fixed-reference ranks are dependent);
- standard conformal test martingale;
- WATCH weighted conformal test martingale;
- predictive-rank martingale order feature;
- predictive-rank martingale order+dispersion portfolio (this work);
- anytime confidence sequence for running/rolling average miscoverage;
- no monitor / keep using stale calibration (negative control).

Report anytime false-alarm probability under null, detection delay in independent cluster blocks, detection power for upward/downward/dispersion shifts, false quarantine, stale commitment attempts blocked, recovery option availability after quarantine, and monitoring overhead.

## References

- Kuang and Xia, [Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales](https://arxiv.org/abs/2609.00536), 2026.
- Fang, [Anytime Valid Conformal Prediction on Streaming Time Series](https://arxiv.org/abs/2611.20905), 2026.
- Prinster et al., [WATCH: Adaptive Monitoring for AI Deployments via Weighted-Conformal Martingales](https://proceedings.mlr.press/v267/prinster25a.html), ICML 2025.
- Timans et al., [On Continuous Monitoring of Risk Violations under Unknown Shift](https://proceedings.mlr.press/v286/timans25a.html), UAI 2025.
