# SEMI-DELAYED-ACI-001 — fixed-delay conformal calibration v0.1

**Issue:** #13479  
**Implementation:** `symthaea-semiconductor::benchmark::delayed_aci`  
**Status:** implementation + authored known-answer tests; exact-head validation pending.

## Research anchor and exact recursion

El Halabi & Brandt, *Adaptive Conformal Inference Under Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic*, arXiv:2609.07251 (7 September 2026), study a fixed forecast/feedback delay `tau`.

Their delayed ACI update is:

[
\alpha_{t+\tau}=\alpha_t+\gamma(\alpha-e_t),\qquad
e_t=\mathbf{1}\{Y_t\notin\widehat{\mathcal C}^{\tau}_t(\alpha_t)\}.
]

Here `alpha` is the target miscoverage rate, `gamma > 0` is the adaptation step, and the error bit is computed against the **interval frozen when the original prediction was issued**.

The delayed recursion decomposes into `tau` interleaved phase sequences. For phase `theta`, global rounds are:

[
t_\theta(m)=t_0+\theta+(m-1)\tau,
]

with update:

[
\alpha^{(\theta)}_{m+1}
=\alpha^{(\theta)}_m+\gamma(\alpha-e^{(\theta)}_m).
]

Each phase is updated once every `tau` rounds. The paper establishes the iterate bound `[-gamma, 1+gamma]`. Under its signed-residual interval construction, `alpha <= 0` gives the full real line and `alpha > 1` gives the empty set; those outcomes force error bits zero and one, respectively.

For its fixed-delay assumptions, the paper gives the finite-horizon long-run empirical miscoverage bound:

[
\left|\frac1T\sum_{t=1}^{T}e_t-\alpha\right|
\leq
\frac{1}{\gamma T}\sum_{\theta=1}^{\tau}
\max\{\alpha^{(\theta)}_1,1-\alpha^{(\theta)}_1\}
+\frac{\tau}{T}.
]

The implementation does not re-prove this theorem. It encodes the declared fixed-delay phase schedule so that an external evaluator can check the recursion and its assumptions.

Primary source: https://arxiv.org/abs/2609.07251

## Implemented contract

### Phase-aligned prediction

- The caller opens sequential discrete rounds beginning at zero.
- At most one prediction is issued per round in v0.1.
- The engine chooses `phase = round mod tau` and obtains that phase's current alpha.
- The caller supplies the interval built from that exact alpha and frozen model output.
- The prediction receipt records sample/event identity, issue round, due round, phase, alpha, interval, model-artifact SHA, policy SHA, point forecast, timestamps and state digest.
- The prediction interval and alpha never change after the outcome arrives.

### Qualified delayed outcome

A label can update the relevant phase only when:
- it is delivered in the exact declared due round `issued_round + tau`;
- label and report availability timestamps precede delivery and report availability is not earlier than label availability;
- the report is K1 / CurrentAtK1 and EligibleUnderDeclaredPolicy;
- downstream consumption is permitted;
- label ID, report-attested revision, event ID, full payload SHA and corrected numeric value match;
- the report digest independently replays;
- uncertainty is present, finite and non-negative;
- the label revision has not already been accepted.

The miss bit is calculated from the interval frozen at issue time. The engine updates only that phase:

[
\alpha_{new}=\alpha_{issue}+\gamma(\alpha_{target}-e).
]

UNKNOWN is held; it may be explicitly re-presented as qualified during the same due round. BLOCKED, stale, misaligned, late, mismatched or tampered feedback never updates alpha.

### Missing feedback and claim status

At the next round, an outcome that was not qualified and delivered in its due round is recorded as missing. It is **not** imputed as covered or uncovered, and the engine changes its guarantee state to `UncertifiedMissingOrInvalidFeedback`. End-of-stream closure marks all remaining pending feedback missing. This is intentionally stricter than silently omitting a lost label from the denominator.

The K1 gate is software evidence qualification, not physical source authentication. A valid digest and declared K1 status do not prove that a physical source issuer or reference chain is authentic.

## Delay-to-memory diagnostic

The source introduces:

[
r=\frac{\tau}{L},
]

where `L` is a memory/persistence scale of the residual process. The helper in this module computes this ratio only; it does not estimate `L`.

The paper reports that this ratio strongly organizes performance under AR(1) residual dependence, is less predictive for overall interval performance under GARCH(1,1) and Markov switching, and better characterizes the effect of scale normalization in those latter settings. Abrupt mean/variance shifts do not have a stationary memory scale.

Our benchmark must therefore estimate and report multiple memory proxies, not treat `r` as a universal safety score:
- AR(1) autocorrelation decay / e-folding time;
- integrated autocorrelation time;
- GARCH dependence in squared residuals;
- Markov-state persistence where a regime model is justified;
- rolling estimates with uncertainty and reset/change-point flags.

## Explicit scope boundary

This v0.1 is fixed-delay ACI only:
- it does not implement online split-conformal quantile estimation; the interval constructor is external;
- it does not implement variable or out-of-order delay correction;
- it does not implement finite-capacity tracking or inverse-probability weighting;
- it does not implement CA-OCRC;
- it does not establish conditional coverage for each wafer or subgroup.

Do not use the term CA-OCRC for this implementation.

## Next stage: capacity-aware OCRC

Wang & Simeone, *Capacity-Constrained Delayed Feedback for Online Conformal Risk Control*, arXiv:2611.17059 (12 November 2026), is a separate algorithm. It introduces:
- a finite pending-feedback tracking capacity `K`;
- per-round randomized proxy delay sampled from a declared distribution `F`;
- admission and eviction of tracked pending outcomes;
- two-level inverse-probability weighting to compensate for labels that were never retained.

The paper assumes either arbitrary delays bounded by known `d_max`, or IID delays from a declared distribution. It uses the delay survival distribution and the proxy admission probability in its weighted update. Its experiments cover medical imaging, NLP, computer vision and wireless communications, not semiconductor manufacturing.

The semiconductor port must first reproduce the paper's complete randomized tracking/admission/eviction and IPW equations, delay assumptions and bounded-loss requirements in an independent oracle. A simple queue cap or naive inverse probability correction is not CA-OCRC.

Primary source: https://arxiv.org/abs/2611.17059

## Required benchmark matrix

### Delay and residual dynamics
- `tau = 1, 2, 5, 10, 20, 40` rounds;
- AR(1) residuals with several autocorrelation regimes;
- GARCH(1,1) conditional scale dependence;
- Markov switching mean/scale;
- abrupt mean shift;
- abrupt variance shift;
- gradual drift and regime switching;
- residual skew, heavy tails and outliers.

### Feedback integrity
- exact fixed-delay K1 label;
- UNKNOWN then qualified re-presentation in the same due round;
- BLOCKED label;
- missing due label;
- late label;
- wrong report availability time;
- K0/stale requalification;
- revision mismatch;
- measurement event/payload/value mismatch;
- duplicate delivery;
- retroactive calibration revocation.

### Baselines
- static split conformal with frozen calibration;
- rolling signed-residual quantile;
- ACI with immediate labels (`tau = 1`);
- fixed-delay tau-DACI with unlimited pending storage;
- delayed raw ACI that ignores the K1 qualification gate (adversarial comparator only);
- online conformal prediction via online optimization (O2CP, ICML/PMLR 2025);
- FACI / strongly adaptive conformal experts;
- CA-OCRC only after a faithful implementation of arXiv:2611.17059;
- no calibration / fixed width.

### Metrics
- empirical miscoverage and coverage at nominal 90%, 95%, 99%;
- the paper's finite-horizon bound evaluated against the realized stream;
- interval width median/p95 and Winkler/interval score;
- worst rolling-window, tool and chamber miscoverage;
- recovery rounds after mean/scale change;
- alpha phase trajectories, oscillation and degenerate-set frequency;
- delay-to-memory ratio with memory-estimator uncertainty;
- qualified, unknown, blocked, late, lost and duplicate labels;
- guarantee-state degradation events;
- update latency and memory/pending-receipt count.

Do not interpret long-run marginal coverage as per-wafer conditional coverage.

## Proposed targets (hypotheses, not achieved results)

- 100% zero alpha updates from UNKNOWN/BLOCKED, wrong-round, late, stale, mismatched or tampered labels.
- 0 historical interval or alpha-at-issue mutation after delivery.
- exact recurrence and phase assignment on all independent known-answer streams.
- byte-identical alpha-state replay for a fixed schedule.
- 100% guarantee-state degradation when any due outcome is missing or invalid.
- observed coverage and bound comparison reported separately under the fixed-delay assumptions.
- benchmark delay-to-memory performance across AR(1), GARCH, Markov switching and abrupt shifts; do not demand a single curve collapse across all processes.

These are software/research acceptance targets, not physical process guarantees.

## Claim ceiling

This is an online calibration controller over supplied intervals and a declared fixed-delay stream. It does not authenticate measurement source issuers, estimate instrument traceability, prove physical causation, establish conditional per-wafer guarantees, demonstrate real-fab gains or authorize wafer disposition, APC/R2R, recipe changes or equipment control.
