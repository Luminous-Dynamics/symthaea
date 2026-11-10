# VRCO-001 — Robust bounded inverse control for vocal corrections

**Date:** 10 November 2026  
**Status:** deterministic projected-gradient solver and synthetic controls; no audio experiment or closed-loop capability claim  
**Owner:** https://github.com/Luminous-Dynamics/symthaea/issues/11533

## Research question

Can a calibrated acoustic control-response matrix be turned into a bounded control proposal that robustly meets the requested target and every collateral/no-op guard, even when each Jacobian coefficient is uncertain?

## Formulation

Let the local response be:

`Δy = J Δu`

For metric i and control j, the measured local sensitivity is an interval `J_ij ∈ [l_ij, u_ij]`, not a falsely exact scalar. The solver uses the midpoint matrix for the nominal optimization:

`min_{Δu} Σ_i w_i (J_mid,i Δu - g_i)^2 + λ ||Δu||²`

where `g_i` is the midpoint of the accepted target band for a Target metric, and zero for a Collateral metric.

Actuator constraints are converted to delta bounds:
- lower = max(control minimum − current value, −maximum step);
- upper = min(control maximum − current value, maximum step).

V1 uses deterministic projected gradient descent for this convex box-constrained quadratic. It calculates a conservative Lipschitz upper bound from the weighted Jacobian Frobenius norm and regularization. Non-convergence within the frozen iteration count returns no proposal.

## Robust postcondition

After nominal optimization, every coefficient interval is propagated through the proposed control delta. For a negative delta, interval endpoints are reversed before multiplication. Metric contribution intervals are summed conservatively.

A proposal is returned only when:
- every Target metric's entire predicted interval is inside its requested non-zero response band;
- every Collateral metric's entire predicted interval is inside its no-op-calibrated acceptance band, which must contain zero.

A nominally successful result with an uncertain interval crossing a guard is rejected as NOT_ROBUSTLY_FEASIBLE and returns diagnostic predictions only.

## Implementation

The standalone crate tools/robust-vocal-control-solver:
- aligns metric and control dimensions by stable IDs and exact units;
- validates response-model and no-op calibration receipt digests;
- rejects malformed/non-finite Jacobian intervals, dimensions, bounds and configs;
- solves a weighted regularized quadratic under actuator/trust-region box constraints;
- computes robust interval predictions for all metrics;
- returns a proposal only when every target/collateral band is robustly met;
- does not execute controls or authorize runtime mutation.

## Synthetic controls

Test cases cover:
- target response with collateral preserved;
- uncertain collateral coupling that passes nominally but fails robustly;
- wide target-Jacobian uncertainty;
- actuator bounds and trust-region limits;
- zero/unobservable response;
- coupled controls that can cancel collateral response;
- response-model mismatch and missing no-op receipt;
- malformed intervals and units;
- missing/extra metric/control coverage;
- non-zero target and zero-containing collateral acceptance;
- deterministic repeated proposals;
- no iteration budget and invalid regularization.

## Integration

- VCRB-001 (#11401) supplies the measured local response matrix.
- VNOC-001 (#11420) supplies no-op tolerance calibration receipts.
- VIA-001 (#11456) provides conservative interval decision semantics.
- VRCO-001 returns candidate control deltas and robust predicted metric intervals.
- CQEP-001 compiles the control application into a capability-qualified plan.
- Runtime authority verifies current state, authorization and parent digest.
- Independent acoustic observations measure actual post-render behavior.
- VCCR-001 checks the correction trajectory; VMEA-001 checks final instruction and preservation rubrics.

The Jacobian's applicability must be scoped to the qualified renderer/voicebank, analyzer, preprocessing, register, technique, phrase context and control operating point. Re-identify when these conditions change materially.

## Claim boundary

Synthetic tests qualify only the solver and interval arithmetic contracts. They do not prove that a measured Jacobian predicts real singing, that no-op calibration is representative, or that an authorized control step produces an audible improvement. A real licensed vocal pilot with held-out references and independent measurement is required.
