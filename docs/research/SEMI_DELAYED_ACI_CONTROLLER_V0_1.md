# SEMI-DELAYED-ACI-001 — fixed-delay interleaved ACI controller v0.1

**Research owner:** #13479  
**Implementation:** `symthaea-semiconductor::benchmark::delayed_aci_controller`  
**Stack:** #13478 evidence-gated streaming adapter → delayed ACI controller  
**Status:** implementation and tests authored; exact-head validation pending.

## Research basis

- El Halabi & Brandt, *Adaptive Conformal Inference Under Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic*, arXiv:2609.07251 (2026-09-07). For a fixed delay, the delayed recursion decomposes into interleaved ACI-like sequences; delay relative to residual-memory time affects the useful adaptation regime. The paper cautions that the delay-to-memory ratio is not equally predictive under AR, GARCH and Markov-switching residual dynamics.
- Wang & Simeone, *Capacity-Constrained Delayed Feedback for Online Conformal Risk Control*, arXiv:2611.17059 (2026-11-12). This extends OCRC to delayed feedback with finite tracking capacity using randomized proxy-delay tracking and inverse-probability weighting. This implementation does **not** claim to implement CA-OCRC.
- Areces et al., *Online Conformal Prediction via Online Optimization*, ICML 2025, PMLR 267:1604-1649. Baseline for adversarial/stochastic online coverage through online optimization.
- Liu, Dobriban & Orabona, *Online Conformal Prediction via Universal Portfolio Algorithms*, ICML 2026, PMLR 306:75891-75952. Parameter-free online conformal method using a regret-to-coverage reduction and universal portfolio algorithms; required comparator to manually tuned ACI.
- Gibbs & Candes, *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts*, JMLR 2024. Include as a non-exchangeable baseline.

## Algorithm boundary

This v0.1 implements a fixed-delay, interleaved ACI *miscoverage-state controller*. For target miscoverage `alpha`, adaptation rate `gamma`, and miss indicator `m_t = 1{y_t not in C_t}`, the per-lane update is:

`alpha_next = clip(alpha_current + gamma * (alpha_target - m_t), alpha_min, alpha_max)`

The fixed feedback delay `tau >= 1` creates `tau` interleaved alpha lanes, indexed by prediction-step modulo `tau`. Feedback is accepted only at its exact declared step `prediction_step + tau`, before issuing the prediction at that step. This models the fixed-delay interleaving result; it deliberately rejects variable-delay, out-of-order and capacity-tracked feedback rather than approximating their theorem.

The caller supplies an interval constructed for the active alpha lane. The controller freezes lower/upper bounds and the alpha used. The interval construction method is separately content-addressed and supplied as an external component. The controller does not construct conformal quantiles or prove that the interval constructor is calibrated.

## Evidence requirements

A feedback outcome updates alpha only when:
- a pending prediction exists and its immutable receipt digest validates;
- the fixed-delay delivery step is exact and stream-ordered;
- clock domain and measurement-event identity match;
- label ID and report-attested revision match the delivery;
- full label payload digest and K1 eligibility-report digest validate;
- report frontier is CurrentAtK1;
- eligibility is EligibleUnderDeclaredPolicy and downstream consumption is allowed;
- report availability follows physical-label availability and precedes delivery;
- label value equals the qualified corrected value and the qualified uncertainty bound is finite/present.

UNKNOWN/BLOCKED, stale, late, early, unavailable, mismatched or duplicated feedback never changes alpha. The run marks fixed-delay assumptions as no longer intact when feedback is denied or missed. The alpha state can remain available for diagnostic replay, but no theorem-backed coverage claim should be carried across that broken run.

## State and receipts

Each prediction receipt freezes:
- prediction step and ACI lane;
- alpha used to construct the interval;
- lower/upper interval bounds;
- interval-method and policy digests;
- sample and measurement-event identities;
- event/prediction times and clock domain;
- whether fixed-delay assumptions were intact at issue.

Each feedback receipt binds:
- prediction receipt;
- label/revision and full payload digest;
- K1 eligibility-report digest;
- lane and alpha before/after;
- whether the interval was missed;
- feedback step;
- fixed-delay-assumption status and whether alpha mutated.

The controller is an online state mechanism, not a substitute for an immutable calibration dataset or a physical reference standard.

## Test coverage

The initial tests include:
1. fixed-delay interleaved lanes;
2. an in-interval result increases alpha toward narrower future intervals;
3. a miss decreases alpha to widen future intervals;
4. exact report-attested label revision;
5. UNKNOWN feedback leaves alpha unchanged and invalidates the assumption flag;
6. BLOCKED feedback leaves alpha unchanged;
7. off-schedule feedback is rejected without alpha update;
8. invalid delay/alpha bounds are rejected;
9. deterministic replay of a frozen event schedule;
10. accepted revision is idempotent;
11. label/report availability chronology;
12. tampered pending prediction receipt rejection.

Additional testing is required for the alpha bounds under long adversarial streams, missing-feedback expiry, common-cause shifts, and numerical edge cases before considering the module complete.

## Benchmark matrix

Baselines:
- fixed split conformal;
- rolling residual quantile;
- immediate-feedback ACI;
- fixed-delay interleaved ACI (this implementation);
- online conformal optimization (ICML 2025);
- UP-OCP universal-portfolio method (ICML 2026);
- FACI / strongly-adaptive online conformal experts;
- CA-OCRC only after exact paper-equation and randomized-tracking reproduction;
- panel/non-exchangeable conformal methods for tool/chamber contexts.

Adversarial streams:
- mean, variance and gradual drift;
- AR(1), GARCH(1,1), Markov regime switching;
- abrupt tool/chamber bias and common-reference shifts;
- heavy-tailed residuals and outliers;
- fixed delay sweep `tau in {1,2,4,8,16,32}`;
- delay-to-memory sweep under multiple residual persistence mechanisms;
- missing/unknown/revoked K1 labels;
- interval-method or policy digest mutation;
- tool/chamber subgroup shift and unseen contexts.

Metrics:
- empirical coverage at nominal 90%, 95%, 99%;
- worst rolling-window and worst tool/chamber coverage;
- interval width and Winkler/interval score;
- coverage recovery lag after mean/scale shifts;
- alpha trajectory, clipping/saturation and oscillation;
- delay-to-memory ratio and feedback loss/missingness rate;
- qualified/unknown/blocked/late/mismatched label counts;
- update latency, pending-ledger size and deterministic replay rate.

## Targets (hypotheses)

- exact reference-recursion parity on frozen known-answer sequences;
- 100% no alpha update for ineligible, stale, late or mismatched feedback;
- zero historical interval mutation;
- 100% deterministic replay for a fixed schedule;
- report empirical coverage/width and theorem assumptions separately;
- no claim of conditional per-wafer coverage from a long-run marginal bound;
- no arbitrary-delay or capacity-constrained guarantee until those algorithms are independently implemented and verified.

## Future capacity-aware extension

CA-OCRC uses a randomized tracking policy based on a proxy delay and two-level inverse-probability weighting. Before implementing it, transcribe the paper's exact tracking probability, delay distribution, truncation/support assumptions, capacity recursion and estimator from the full paper; add a separate mathematical reference oracle. Do not substitute a generic reservoir sampler or ordinary inverse-probability weights and call it CA-OCRC.

## Claim ceiling

This is a fixed-delay statistical controller with a caller-supplied interval method. It is not yet a full conformal prediction system, not a CA-OCRC implementation, and not a physical guarantee for every wafer. It does not authenticate physical outcomes, prove measurement traceability or authorize wafer disposition or equipment control.
