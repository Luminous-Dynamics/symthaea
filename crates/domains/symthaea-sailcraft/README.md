# Symthaea Sailcraft P0 — point-of-sail and robust route candidates

**Owner:** MARINE-SAIL-AUTONOMY-001 #12153  
**Status:** synthetic software contract only  
**Version:** 0.1.0

## Scope

This crate is a small, deterministic route-candidate kernel. It consumes a declared craft-specific polar surface and qualified wind/current/waypoint-bearing inputs. It calculates:

- whether the direct waypoint bearing lies inside the supplied craft-specific no-go sector (the boundary is inclusive);
- the best port- and starboard-tack course candidate under the supplied polar surface;
- predicted boat speed and ground velocity made good (VMG), including the supplied current vector.

It searches integer-degree headings and uses bilinear interpolation only inside the declared polar domain. If inputs are missing, stale, unverified, conflicted, malformed or outside the polar domain, the evaluator refuses the unsupported result.

## Coordinate and model semantics

- Bearings are degrees clockwise from true north.
- Wind direction is the direction the wind comes **from**.
- Current direction is the direction the current flows **toward**.
- Polar angles are absolute true-wind angles (TWA), not apparent-wind angles.
- Boat speed is in metres per second.
- The polar surface carries an exact source-artifact digest and craft-configuration digest. Wind, current and waypoint-bearing evidence carries a source digest, freshness window and common-timebase identity. These metadata checks do not authenticate a publisher, sensor or physical measurement. The no-go half-angle and polar must belong to the exact craft + sail-plan + loading condition.

The 40-degree no-go half-angle in the unit-test fixture is a synthetic example only. It is not a universal sailing value or a physical vessel limit.

## Robust interval route evaluation

The `robust` module adds a deterministic interval evaluator alongside the original point-valued P0 planner. It accepts:

- true-wind speed bounds and circular true-wind-from direction bounds;
- current speed and flow-direction bounds;
- a source-bound absolute polar-speed error bound for the exact craft configuration;
- the same freshness, source-digest and shared-timebase requirements for wind/current/waypoint observations.

For each heading it computes the full true-wind-angle interval. A candidate is admitted only when the complete interval remains outside the configured no-go sector. Polar interpolation bounds are evaluated over every intersected wind-speed/TWA grid cell; bilinear extrema are taken from each clipped cell's corners. Unknown polar error is rejected rather than assumed to be zero, and all polar extrapolation is prohibited.

The ground-VMG envelope is propagated with signed projection. If the heading points away from the waypoint, the upper boat-speed bound determines the most negative boat-VMG bound. Current speed/direction uncertainty is bounded separately. Non-finite interval arithmetic is rejected.

The evaluator retains at most one worst-case-VMG candidate per tack. It only declares a unique-best tack when both tack sides are represented and one side's best guaranteed VMG strictly exceeds the maximum possible VMG of every candidate on the other side. If the envelopes overlap, a tack side is absent, or the uncertainty reaches the dead-downwind side boundary, it preserves ambiguity instead of forcing a winner. It is an interval envelope under declared input bounds, not a probabilistic forecast or physical validation.

The robust module does **not** yet model vertical wind profile, apparent-wind conversion, leeway, heel/roll, waves, tack/gybe duration or loss, actuator response, weather evolution, COLREG encounter resolution or a closed-loop sailcraft dynamics model. Those are separate research increments.

## Tack and gybe transition-cost envelopes

The `transition` module adjusts a robust steady-course VMG interval over a fixed scoring horizon using a source-bound interval for:
- elapsed tack or gybe time;
- signed ground progress toward the active waypoint during the manoeuvre;
- exact craft/configuration, polar-artifact and actuator-model identities;
- common timebase, source digest and freshness.

For horizon (H), transition duration interval (T_m), signed transition progress (D_m), and post-transition steady VMG interval (V_s):

`D_total = D_m + V_s × (H - T_m)`

`V_effective = D_total / H`

All endpoint products are evaluated. Negative transition progress is preserved. The duration upper bound must be strictly shorter than the horizon. Missing, stale, conflicted, mismatched or malformed transition evidence is not scored.

This is an interval arithmetic contract, not a physical tack/gybe model. The input bounds must be justified by a vessel-specific model or instrumented manoeuvre campaign before use in a physical claim. The module has no helm, rudder, sail-sheet, reef or furl actuator output.

## Explicit exclusions

This P0 kernel does not implement or validate:
- sail aerodynamics, hull/appendage hydrodynamics, leeway, heel/roll, wave response or capsize stability;
- apparent-wind estimation, sail trim, rudder control, actuator dynamics, reefing or furling;
- weather routing, obstacle avoidance, COLREG interpretation, or a vessel ODD;
- an autonomous driving/sailing policy or any live control connection.

The AUV crate's REMUS-class underwater hydrodynamics and WAPS wind-assist cargo-ship research are not treated as sailboat physics.

## Reproduction

```sh
cargo test --locked -p symthaea-sailcraft
cargo clippy --locked -p symthaea-sailcraft --all-targets -- -D warnings
```

## Claim ceiling

A pass establishes only deterministic software semantics for the supplied synthetic polar/wind/current inputs. It does not establish measured sailing performance, actual vessel dynamics, seaworthiness, COLREG compliance, autonomous navigation competence, safety, class/flag approval or permission to operate.
