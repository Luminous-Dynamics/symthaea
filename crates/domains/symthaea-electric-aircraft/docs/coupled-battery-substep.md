# Coupled battery electro-thermal substep

This increment introduces a production-side, bounded feedback solve for one mission substep.

## Contract

For a frozen battery-terminal demand over a declared interval, the solver:

1. resolves conservative pack OCV and resistance from the existing SOC/temperature characterization surfaces;
2. solves the low-current terminal-power root without silently reducing demand;
3. checks pack continuous-power and current limits;
4. computes pack I²R heat and adds it once to the named battery node;
5. advances the existing multi-node implicit thermal network from the unchanged substep-start state;
6. updates the candidate battery-node temperature using deterministic under-relaxation and repeats;
7. commits a new SOC and thermal state only when the fixed-point temperature residual, battery energy identity, thermal-network energy balance, reserve and limits pass.

The return value preserves the battery-characterization evidence records. It does not claim that the caller-supplied evidence bytes, publisher, calibration authority or configuration have been independently authenticated.

## Deliberate first-slice boundary

The solver couples pack temperature, OCV, resistance, current, voltage, I²R heat and the existing thermal network. Propulsion demand, propulsion efficiency and non-battery heat sources are frozen during this substep. It does not yet solve motor/inverter thermal feedback, dynamic coolant flow, pump/fan operating points, TMS architecture, cell-level electrochemistry, ageing evolution, aircraft aerodynamics, thermal runaway or airworthiness.

`non_battery_heat_generation_w` is an explicit per-node vector. It must not include pack I²R heat, which this function owns. Electrical thermal-management demand belongs in the supplied terminal-power demand; heat deposited by pumps, fans or other auxiliaries belongs in the heat vector only to the extent physically measured or justified. The same watts must not be counted twice.

## Numerical semantics

The iteration is deterministic and bounded. It replays each candidate from the same substep-start thermal state, samples the conservative characterization point at the candidate end temperature, and advances the state only after convergence. The absolute-plus-relative temperature tolerance and relaxation factor are caller-declared. Exhausting the iteration bound, leaving a characterization domain, losing the electrical root, exceeding limits, or violating an energy balance returns a typed error and no partial report.

The terminal-power equation uses the cancellation-safe low-current root:

`I = 2P / (V_oc + sqrt(V_oc^2 - 4RP))`

The battery energy identity is:

`V_oc I Δt = P_terminal Δt + I²R Δt`

This is a reduced-order pack model. The numerical fixed point is not proof that the constitutive model or its calibration is physically valid.

## Verification status

- Eight Rust tests are authored in this module, including a frozen analytical electrical reference, temperature/resistance feedback, no-load behavior, out-of-domain rejection, typed power/current/reserve failures, thermal-limit and iteration-limit failures, and an impossible electrical root.
- Tests have not been executed locally in the authoring environment; a dedicated exact-path GitHub Actions workflow runs format, test and Clippy checks.
- Synthetic fixtures are not physical battery measurements. No aircraft performance, pack safety, thermal-runaway, flight-test or certification claim is made.