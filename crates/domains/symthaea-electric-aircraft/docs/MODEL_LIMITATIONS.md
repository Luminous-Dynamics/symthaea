# Model limitations and promotion gates

## Model identity

electric-fixed-wing-energy-v0.2.0 is a deterministic reduced-order estimator for steady mission segments. Its output is useful for screening architecture choices and exposing energy-budget assumptions. It is not validated against a flight-test article.

## Characterized propulsion extension (v0.2)

An optional characterized evaluation accepts three one-dimensional curves: motor efficiency versus shaft-load fraction, inverter efficiency versus inverter input-load fraction, and propeller efficiency versus advance ratio. Values and absolute uncertainty bounds are linearly interpolated only inside each supplied domain. The mission evaluator uses nominal efficiency minus the absolute bound and records dataset IDs. No extrapolation is permitted.

This is an improvement over a single scalar efficiency, but it remains limited: the motor and inverter maps are one-dimensional projections, not full speed/torque/voltage/temperature maps; the propeller curve is not a full coefficient map over advance ratio, blade pitch, Reynolds number, Mach number and blade condition. The current code checks the SHA-256 string's syntax but does not obtain the dataset or calculate its digest. Evidence verification must happen in a separate trusted pipeline. The absolute uncertainty bound is not a statistical confidence interval.

## Current equations

For the supplied segment, the kernel uses:

- W = m g
- D = W / (L/D)
- P_aero = D V
- P_climb = m g max(ROC, 0)
- P_battery = (P_aero + P_climb)/(eta_prop eta_motor eta_inverter) + P_aux
- P_terminal = I (V_oc - I R)
- Q_dot = I² R

It solves the low-current root of the equivalent pack equation and advances a lumped pack temperature using a first-order heat-capacity/conductance model. Chemical energy draw is approximated by I * V_oc; the difference between chemical draw and terminal output represents equivalent internal resistive loss.

## Known limitations

1. Aerodynamic L/D is an input, not derived from geometry or a validated polar. It must be tied to a source, flight condition, and uncertainty interval by a higher-level caller.
2. The model assumes lift approximately equals weight and omits manoeuvre load factors, gust loads, induced/parasite drag decomposition, compressibility, ground effect, propwash interactions, Reynolds effects, and aeroelastic coupling.
3. It is not a takeoff, landing, stall, climb-envelope, or flight-dynamics model. Every segment is a steady positive-airspeed condition.
4. Motor, inverter, and propeller efficiencies are fixed per mission, not maps over torque, rpm, voltage, temperature, and airspeed.
5. Battery OCV and resistance are constant. There is no cell-level electrochemistry, SOC/temperature dependence, imbalance, contactor, fuse, arc, insulation, propagation, venting, or thermal-runaway model.
6. The pack thermal network is one lumped node and cannot establish local hot spots or cell-to-cell propagation safety.
7. Auxiliary power is a scalar load. Electrical buses, redundancy, transient loads, charging, and fault isolation are not simulated here.
8. The mission uncertainty fraction is a user-supplied screening margin, not a statistical confidence interval or regulatory reserve.
9. The model does not claim any battery chemistry, specific energy, range, certification status, or safety performance.

## Promotion gates

Do not promote this kernel to aircraft design authority until all of the following exist:

- versioned, traceable aerodynamic polars and uncertainty bounds;
- measured motor/inverter/propeller maps and calibration datasets;
- independently measured pack OCV/resistance/capacity/thermal maps over SOC, temperature, age, and rate;
- cell/pack abuse and propagation evidence for the selected battery design;
- comparison against an independently implemented higher-fidelity model;
- test-correlation evidence from dynamometer, thermal-chamber, propeller, and instrumented ground tests;
- a hazard analysis, failure-mode analysis, and reviewed safety architecture;
- an agreed certification basis and compliance plan with the competent authority;
- independent engineering review and configuration-controlled evidence.

Unknown or missing evidence must remain unknown/incomplete, not be silently replaced by nominal values.
