# Vocal control candidate allocator (VCCA-001)

A dependency-free Rust library for selecting a minimum-effort candidate from a finite set of bounded vocal control interventions.

## Contract

The allocator consumes:
- an independently calibrated local response matrix;
- explicit coefficient-uncertainty bounds and per-metric residual envelopes;
- control capability/bounds/trust radii and effort costs;
- target response bands and protected no-op tolerances;
- a finite, predeclared list of candidate control vectors.

For candidate vector u, metric m is predicted with:

~~~text
center_m = Σ_i coefficient[m,i] × u_i
radius_m = residual_m + Σ_i uncertainty[m,i] × |u_i|
interval_m = [center_m − radius_m, center_m + radius_m]
~~~

A target passes only when the *entire* interval lies inside its requested signed band. A protected metric passes only when its entire interval lies inside the symmetric no-op guard.

The selected candidate is the feasible candidate with minimum weighted normalized control effort. Ties are resolved by candidate ID. If no candidate passes every hard condition, the allocator abstains and retains all rejection reasons. The report retains every candidate's canonical control vector and predicted metric intervals so the decision can be replayed from the returned record.

## Authority boundary

The output disposition is PROPOSAL_ONLY. This crate:
- does not execute a renderer;
- does not grant a capability or change renderer declarations;
- does not create or authorize external actions;
- does not prove the response model is valid outside its calibration profile;
- does not prove the eventual render meets the predicted intervals.

Every proposal must pass the independent renderer capability gate and then be remeasured. Feed the resulting audio/measurement receipts into VCRB/VIA/VCCR/VMEA as appropriate. An indeterminate or protected-violating observation stops the correction loop.

## Calibration and study boundary

The response matrix, residual envelope and coefficient-uncertainty bounds must be derived from a distinct calibration split and bind the exact backend, voicebank, analyzer and source/profile context. The held-out confirmatory data must not tune the profile.

This is a finite candidate allocator, not a continuous QP solver or a proof of global optimality. Its matched research baselines should include fixed-step correction, greedy one-control-at-a-time, damped pseudoinverse and regularized least squares with bounds.

## Run

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/vocal-control-candidate-allocator/Cargo.toml
cargo +1.96.0 test --manifest-path tools/vocal-control-candidate-allocator/Cargo.toml
~~~

## Claim boundary

Synthetic tests validate matrix completeness, uncertainty propagation, candidate ranking and fail-closed constraints only. They do not demonstrate acoustic-model calibration, actual renderer response, target convergence, naturalness or SOTA.
