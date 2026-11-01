# HARDSEC-023 — Fixed-reference Predictive Rank Martingale

**Status:** Reference implementation and synthetic contract fixture are authored; exact small-state qualification is included in this revision. No Rust/CI pass, real PMU run, or detector superiority is claimed until exact-head Actions and physical acquisition complete.  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/9611  
**Stacked on:** HARDSEC-022 / PR #9605 → HARDSEC-021 / PR #9593 → HARDSEC-020/019/018.

## 1. The distinction this work makes

There are two different dependence problems in fixed-reference conformal monitoring:

1. **Reference-reuse dependence.** Even when calibration observations and future observations are iid from the same continuous null distribution, repeatedly ranking future observations against one fixed finite calibration sample creates dependent ranks. Treating those ranks as iid conformal p-values is incorrect.
2. **Source time dependence.** Hardware telemetry windows can themselves be serially dependent because of overlapping windows, persistent microarchitectural state, scheduler/frequency effects, repeated workload phases, or a shared latent state.

The predictive-rank martingale (PRM) addresses the first problem exactly under its iid source-observation assumptions. It does **not** solve the second problem. HARDSEC-022's functional-dependence and block-coverage auditor gives model-bound marginal and post-hoc diagnostics; it does not establish the sequential conditional-rank law needed for a PMU e-process.

This makes PRM a valuable additional detector and a necessary comparator, not an unconditional replacement for CCTM and not a certificate of valid PMU false-alarm control.

## 2. Research basis and null law

Kuang and Xia (2026), https://arxiv.org/abs/2609.00536, derive the conditional null distribution of the next rank when a fixed calibration sample of size n is reused.

Let R_t be the rank of online observation t among the n calibration observations and itself, in {1,...,n+1}. Let N_(t-1,j) count prior online ranks in bin j. Under iid calibration and online observations from the same continuous distribution:

\[
\pi_{t,j}
=\Pr(R_t=j\mid R_1,\ldots,R_{t-1})
=\frac{1+N_{t-1,j}}{n+t}.
\]

The ranks are not independent. The Pólya-urn predictive law accounts for the dependence induced by repeatedly reusing the fixed reference.

For any predictable probability vector q_t, the factor q_(t,R_t)/pi_(t,R_t) has conditional expectation one under that null law. A nonnegative wealth process built from those factors is a martingale; thresholding at 1/alpha gives marginal finite-sample anytime type-I control by Ville's inequality.

For a predeclared feature h, the implementation uses the predictively centered payoff:

\[
Z_t=h((R_t-1)/n)-\sum_{j=1}^{n+1}\pi_{t,j}h((j-1)/n),
\qquad
M_t=M_{t-1}(1+\lambda_t Z_t),
\]

where lambda_t is selected using only past rank information. The equal-weight portfolio combines the fixed Order and Dispersion components. Order targets directional/stochastic-order departures; Dispersion targets center-versus-tail departures. The portfolio weights are fixed before evaluation and sum to one.

For discrete/quantized PMU scores, the protocol uses lexicographic ranks on (score, independent uniform tie breaker). The independence and uniformity of those tie breakers remain assumptions; a digest binds the declared randomization artifact but does not authenticate its entropy source.

## 3. Rust implementation and strengthened contract

The standalone auditor is examples/hardsec_predictive_rank_martingale.rs, behind the hardsec-predictive-rank-martingale Cargo feature.

It records, per online observation:

- the exact rank and predictive probability numerator/denominator;
- the complete shared rank-history counts before and after the observation;
- each feature's conditional mean, centered payoff, predictable bet, betting factor, ONS gradient/accumulator, updated bet, and component log wealth;
- the fixed-weight portfolio wealth and first threshold crossing.

The shared Pólya history advances **once per observation**, not once per feature. This matters: Order and Dispersion see the same observed rank, so advancing the counts separately would corrupt the conditional null distribution for the second feature.

The receipt binds the plan, data, and HARDSEC-020 runtime assignment receipt bytes. It refuses to overwrite an existing output. It explicitly records that group IDs, score-transform digests, context receipts, and randomization digests are caller-supplied integrity metadata rather than authenticated proof of independence or execution.

The runtime parser now validates the ONS contract rather than merely checking that an ons object exists. Its update formula, initialization, projection-radius rule, and minimum betting factor must match the implemented constants. This prevents a plan from declaring a different optimizer while the executable silently runs the hard-coded one.

An exact rank-tree unit test enumerates every Pólya-urn rank path for n=2 and horizon 8, including the adaptive ONS updates and Order/Dispersion mixture. It verifies total null path mass is one and the probability of ever crossing the 1/alpha boundary is no greater than alpha. The conditional-centering unit test also uses a reachable history whose rank counts sum to t−1; this is important because arbitrary count vectors can make the nominal predictive probabilities fail to normalize. This is a finite-state arithmetic qualification of the implementation under the specified predictive rank law—not evidence that real PMU observations follow that law.

## 4. Assumptions and claim ceiling

The paper's theorem is **marginal over both calibration sample and online stream**. It is not a guarantee conditional on every realized fixed calibration sample. The paper also establishes a power ceiling at fixed calibration size; a longer monitoring stream cannot manufacture additional independent calibration information.

The PRM theorem requires, within the declared context:

- calibration observations and online null observations are iid from the same distribution;
- the scalar score transform is fixed before calibration and monitoring;
- tie-breaking uniforms are independent and uniform when score atoms occur;
- the context assignment/admission rule does not select a non-iid subsequence in a way that violates the rank law.

The HARDSEC-020 receipt verifies mapping and declared assignment consistency only. It does not prove context exogeneity, calibration/stream iid, independence of tie randomization, or physical PMU event semantics. The current PRM v1 run is single-context and rejects invalid/unqualified or selectively admitted records. The emitted receipt must continue to report:

- calibration iid, online iid, common-null, fixed-transform, tie-randomization, context-exogeneity, and admission-exogeneity statuses as assumption_not_verified;
- theorem_guarantee_status = not_established_from_input_receipts;
- marginal_not_calibration_conditional = true;
- sequential_pfa_valid_for_arbitrary_serial_dependence = false;
- effective sample size as null.

Do not use a block standard error, empirical ACF, distinct window IDs, or an FDM marginal-coverage statement to promote this process to a sequential e-process for autocorrelated PMU telemetry.

## 5. Qualification and benchmark plan

The machine-readable plan is tests/fixtures/hardsec/prm-v1/benchmark-plan.json. It keeps synthetic, simulator, and physical domains separate and compares:

- HARDSEC-019 fixed-reference DKW-CCTM (paper equations and source-pinned official implementation);
- HARDSEC-021 restarted fixed-reference CCTM;
- standard growing-reference CTM;
- static threshold, EWMA, and Page-Hinkley/CUSUM;
- Order PRM, Dispersion PRM, and their fixed-weight portfolio.

Required scenarios include iid continuous nulls, iid discrete scores with randomized ties, location/gradual/delayed changes, symmetric scale and tail changes, multimodal/skew changes, late changes, calibration-size power ceilings, and leakage/context/sensor negative controls. AR(1), block-dependent, overlapping-window, long-memory, and shared-latent streams are **negative controls** for the iid PRM theorem. Results on those streams may characterize empirical behavior; they must not be reported as theorem-valid PFA.

The benchmark now requires an exact small-n Pólya rank-tree false-alarm calculation and records predictive-probability normalization error. Physical confirmation must additionally pass HARDSEC-018 PMU acquisition qualification and use HARDSEC-014/016 trace lineage, separate calibration/evaluation executions, and independent device/boot support.

Primary metrics are anytime false-alarm probability over independent null episodes, detection probability by time, median/p95 delay, effect-conditioned attack recall, worst-family/context performance, tie-rank integrity, score/resource cost, CPU cycles, p95 latency, and measured energy per monitored device-hour. PFA and average run length (ARL) must remain separate endpoints. Each candidate is compared at a validation-frozen, matched false-alert operating point; thresholds cannot be tuned on confirmatory holdouts.

## 6. Research targets (aspirational, not results)

- At matched false-alert probability, reduce p95 delay by at least 20% versus the strongest independently reproduced baseline.
- At a declared operating point, reach at least 90% pooled attack-effect recall while keeping the worst family/context gap within 3 percentage points.
- Keep benign false alerts at or below 0.1 per physical device-hour.
- Zero silent clears for invalid, stale, multiplexed, misaligned, untrusted, or unseen-context telemetry.
- Keep the incremental decision layer's p95 overhead at or below 2%; report acquisition and scoring overhead separately and end-to-end.
- Establish any cross-platform claim on at least three physical microarchitectures across at least two ISA families.

These are internal research gates, not standards and not achieved measurements. A competitive claim requires reproduced baselines, matched thresholds, predeclared split/alpha plans, independent support, and exact-head executable evidence.

## 7. Next scientific step: dependence-aware conditional ranks

The highest-value open problem is to derive a valid sequential rank law when source telemetry is dependent. A promising direction is a **predictable upper envelope** u_(t,j) for each conditional null rank probability, with a predeclared probability budget for the event that the envelope ever fails. If a time-uniform construction can justify Pr(R_t=j | F_(t-1)) <= u_(t,j) simultaneously, then a predictable distribution q_t can use the conservative factor q_(t,R_t)/u_(t,R_t); on the envelope event its conditional expected factor is at most one. The hard part is constructing and independently qualifying the simultaneous conditional-probability envelope for PMU streams without assuming the desired answer.

The proof programme is tracked in [HARDSEC-024 / issue #9679](https://github.com/Luminous-Dynamics/symthaea/issues/9679). Until that proof and its hardware assumptions exist, keep iid PRM and dependent-PMU diagnostics as separate claim classes. A robust monitor should abstain or fall back to a non-confirmatory diagnostic when those conditions are not established.

## 8. Published detector comparisons and interpretation

The 2026 HARPY-V-GEM5 study reports very high within-workload results (above 99% on selected known-workload configurations), but its leave-one-program-out results are substantially more variable and attack-dependent. The paper describes near-zero results for some attack/program combinations, showing why random-window accuracy and pooled means are insufficient for zero-day generalization. It is a simulator benchmark and must not be described as a physical-board result. See [Pujol et al. (2026), DOI 10.1007/s00521-026-12467-4](https://doi.org/10.1007/s00521-026-12467-4).

The separate physical HARPY-V evaluation on a XuanTie C910 reports high precision/classification in its tested setting but also reports substantial interference when multiple counters are collected concurrently, requiring isolated collection conditions for affected measurements. It reports roughly 0.2 ms perf collection overhead per 10 ms sampling interval in its stated setup. This makes HARDSEC-018 metrology a necessary part of any detector comparison, not an optional profiling appendix. See [Pou et al. (2026), Cybersecurity 9, 222](https://doi.org/10.1186/s42400-026-00662-8).

T-SMADE reports 95.42% average detection for its Evasive Spectre-NOP setting and 100% for Evasive Spectre-memory in its evaluated dataset/configuration. These are useful comparator operating points, not universal bounds on adaptive attackers. See [T-SMADE (2024)](https://www.mdpi.com/2079-9292/13/20/4090).

Do not rank these headline figures against PRM/CCTM or a Symthaea detector unless attack effects, sampling units, platforms, holdouts, thresholds, false-alert budgets, and instrumentation costs are reproduced on a common test protocol. The benchmark remains planned, not executed.

## References

- Kuang, Q. and Xia, Y. (2026), https://arxiv.org/abs/2609.00536.
- Shaer et al. (2026), https://proceedings.mlr.press/v306/shaer26a.html.
- Bhattacharyya and Ramdas (2026), https://arxiv.org/abs/2609.27179.
- Zhai, Cheng and Wu (2026), https://arxiv.org/abs/2609.33868.
