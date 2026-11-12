# Marine robust operating-policy audit

Standalone Rust audit for selecting propulsion operating points within one frozen vessel/service/environment scenario. Constraints are robust and non-compensable; feasible candidates are compared using a Pareto frontier, never a weighted sum.

## Run

~~~sh
cargo test --manifest-path tools/marine-robust-operating-policy-audit/Cargo.toml
cargo run --manifest-path tools/marine-robust-operating-policy-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-robust-operating-policy-v0.1.0.json
~~~

The synthetic fixture must report `NO_ROBUST_FEASIBLE_CANDIDATE` and an empty Pareto frontier. Exit codes:
- 0: a Pareto set was computed among robust-feasible, independently reviewed evidence candidates;
- 2: invalid input;
- 3: no robust-feasible eligible candidate;
- 64: invocation/JSON decoding failure.

## Robust non-compensable gates

Each candidate must be evaluated for minimum delivered cargo service, maximum delivery time, maximum primary energy intensity, maximum well-to-wake GHG intensity, maximum lifecycle cost intensity and every declared frequency-resolved source-level cap. Upper bounds must remain below maxima; lower bounds must remain above minimum service. If an uncertainty interval crosses a limit, the candidate is INCOMPLETE. If the interval definitely violates a limit and the evidence is eligible, it is INFEASIBLE.

A fuel or cost saving cannot compensate for an acoustic or schedule breach. Unknown, synthetic, prediction-only or unreviewed evidence cannot become FEASIBLE. Zero uncertainty is not accepted as a substitute for unknown uncertainty.

## Acoustic meaning

V0 requires `source_level_db_re_1upa_at_1m` and a declared per-band project ceiling. Received SPL, propagation and ecological exposure are different quantities; they cannot be substituted. The ceilings are project input limits, not universal biological safety thresholds. No ecological or marine-mammal benefit claim is made.

## Pareto interpretation

Only FEASIBLE candidates enter the nondominated frontier. Conservative upper bounds are used for minimization objectives: primary energy, well-to-wake GHG, voyage duration, lifecycle cost and each declared source-level band. Incomparable trade-offs remain together. Equal candidates remain tied; no arbitrary weighted score or input-order tie-break is applied.

V0 only compares explicitly declared operating points inside one exact scenario/configuration fingerprint. It performs no interpolation or extrapolation and does not authenticate source bytes, calibration or solver claims.

## Benchmark ladder

- B0: synthetic known-answer constraints, hard-gate negative controls, exact Pareto dominance and input-order determinism.
- B1: independently reviewed model-scale or full-scale measured operating-point data in one controlled scenario.
- B2: leave-one-facility, vessel, propeller-revision, campaign and operating-condition-family holdouts using #11800.
- B3: repeated full-scale route trials with matched cargo, draft, hull state, weather and synchronized shaft/hydrophone observations.
- B4: compare fuel-only, schedule-constrained, weighted-sum, nominal-point and robust Pareto policies on cost-to-validated-improvement, held-out violations, decision regret and uncertainty coverage.

The v0 audit computes eligibility and a Pareto set; it does not itself optimize a ship, validate a CFD solver, predict propagation, prove causality, establish an ecological response or certify seaworthiness.
