# SEMI-DELAYED-ACI-001 — delayed-feedback conformal risk controller v0.1

**Issue:** #13479  
**Implementation:** `symthaea-semiconductor::benchmark::delayed_aci`  
**Status:** Rust controller with 14 authored tests; exact-head validation pending.

## Research anchor

El Halabi & Brandt, *Adaptive Conformal Inference Under Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic*, arXiv:2609.07251 (2026-09-07), studies the delayed ACI recursion and shows that fixed delay τ decomposes into τ interleaved ACI-like sequences. It also introduces the delay-to-memory ratio (r=τ/L), and reports that the ratio is especially informative for AR(1) residual dependence but less predictive for GARCH(1,1) and Markov switching.

Related baselines:
- Areces et al., *Online Conformal Prediction via Online Optimization*, ICML/PMLR 2025.
- CAP, *A General Algorithm for Online Selective Conformal Prediction with FCR Control*, JMLR 2025.
- Wang & Simeone, *Capacity-Constrained Delayed Feedback for Online Conformal Risk Control*, arXiv:2611.17059 (2026-11-12). It introduces randomized feedback tracking and inverse-probability weighting for finite feedback capacity; that exact algorithm is not implemented here.

## Two update semantics — do not conflate them

### Fixed-horizon delayed ACI

For declared integer horizon τ >= 1, the controller implements the recursion:

[
\alpha_{t+\tau} = \Pi_{[\alpha_{min},\alpha_{max}]}
\left(\alpha_t + \gamma(\alpha^* - \mathbb{1}\{y_t\notin C_t\})\right),
]

where (C_t) is the immutable interval actually issued at sequence (t), (alpha^*) is the target miscoverage, (gamma) is the learning rate, and (Pi) is the configured projection. The update is scheduled for sequence (t+	au); intervening predictions retain their own frozen alpha/interval. A label arriving after its effective sequence has already been issued is rejected from this fixed-horizon path rather than silently applied as an arrival-order update.

This is a projected implementation variant. The full paper's theorem/assumptions must be reproduced before claiming its exact finite-sample bound for this bounded projection and the fab's label process.

### Irregular arrival-order ACI

For variable physical-metrology delays and out-of-order labels, a separate mode updates the current alpha state when a qualified label arrives. It uses the frozen interval at prediction time. This mode gives deterministic arrival-order replay, but it must not be presented as the fixed-τ delayed recursion or inherit its theorem automatically.

## Evidence gate

Every prediction interval is immutable. Its receipt records:
- sample and measurement-event IDs;
- prediction sequence and event/prediction times;
- global model and interval-policy digests;
- alpha state used to construct the interval;
- exact lower and upper bounds;
- upstream prediction receipt digest.

A physical outcome updates ACI only when:
- K1 eligibility report schema/digest replays;
- report frontier is CurrentAtK1;
- eligibility is EligibleUnderDeclaredPolicy and downstream consumption is allowed;
- exact label ID, revision, measurement-event ID, full label payload digest and corrected numeric value match;
- label availability precedes report availability, which precedes delivery;
- delivery follows prediction and uses the same clock domain;
- label revision has not already been accepted.

UNKNOWN is held for later explicit re-presentation with a current K1 report. BLOCKED never updates alpha. A denial never changes current alpha, scheduled alpha or pending prediction records.

## Delay-to-memory diagnostic

The helper records (r=\tau/L) from a caller-supplied positive memory estimate (L), together with the declared dependence family:
- AR(1): treat (L) as residual correlation memory;
- GARCH(1,1): interpret it as volatility memory;
- Markov switching: interpret it as regime persistence;
- unspecified: report the ratio as descriptive only.

The software does not estimate (L) or infer it from a single stream by default. It does not claim that one threshold on (r) predicts delay performance across all dependence families.

## Benchmarks

Compare:
- split conformal;
- rolling residual quantile;
- immediate-feedback ACI;
- fixed-horizon delayed ACI;
- irregular arrival-order ACI;
- Online Conformal Prediction via Online Optimization (ICML 2025);
- CAP with selection/FCR control;
- CA-OCRC after exact independent reproduction of the 2026 randomized tracking and inverse-probability estimator;
- panel conformal after separate protocol reproduction.

Measure:
- empirical coverage at 90%, 95%, 99%;
- worst rolling-window and worst tool/chamber coverage;
- interval width and interval score;
- coverage recovery delay and alpha oscillation;
- delay-to-memory ratio by residual family;
- physical label delay, K1/report delay and accepted/held/blocked/rejected counts;
- interval mutations (target zero);
- false confidence under unqualified feedback;
- p50/p95 update latency and memory.

Do not claim per-wafer conditional coverage from long-run marginal coverage results. Report D1/D2 or target/unit-specific results separately.

## Authored tests

Fourteen Rust tests cover:
1. immutable prediction interval/alpha after outcome;
2. covered outcome moves alpha upward for future intervals;
3. miss moves alpha downward to widen future intervals;
4. fixed-horizon update is scheduled at (t+	au);
5. fixed-horizon label arriving after its slot is rejected;
6. UNKNOWN held and later K1-qualified report explicitly accepted;
7. BLOCKED no-update;
8. duplicate accepted label revision is idempotent;
9. revision/identity/value/payload mismatch is rejected;
10. report availability chronology is enforced;
11. out-of-order arrival-order updates remain explicit and deterministic;
12. multiple fixed-delay phases are interleaved at the correct sequence;
13. delay-to-memory diagnostics are process-family-specific;
14. deterministic arrival-order replay;
15. prediction-sequence overflow does not mutate the calibrator.

The test module currently contains 15 cases; each is an authored test, pending exact-head execution.

## Not implemented yet

- finite-capacity randomized tracking, proxy-delay distributions or inverse-probability weighting from CA-OCRC;
- adaptive conformal threshold construction from an online residual quantile queue;
- per-tool/chamber panel calibration;
- historical requalification/replay after a later calibration revocation (the current receipts are immutable; a new calibrator chain will be required);
- physical fab data or shadow deployment.

## Claim ceiling

This is a risk-state update controller over caller-supplied immutable intervals. It does not authenticate physical labels, guarantee conditional coverage for each wafer, optimize measurement capacity, validate an instrument or authorize APC/R2R/recipe/equipment control.
