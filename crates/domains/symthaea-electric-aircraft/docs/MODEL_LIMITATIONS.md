# Model limitations and promotion gates

## Model identity

The core mission estimator is model version electric-fixed-wing-energy-v0.3.0. It is a deterministic reduced-order estimator for steady mission segments. Its output is useful for screening architecture choices and exposing energy-budget assumptions. It is not validated against a flight-test article.

The adverse-corner extension is separately versioned electric-fixed-wing-adverse-corner-v0.1.0. It is deterministic one-sided uncertainty screening, not a statistical confidence interval, Monte Carlo analysis, correlated worst-case proof, safety assessment or certification method.

## Scalar and characterized propulsion paths

The scalar evaluator uses fixed motor, inverter and propeller efficiencies. An optional characterized evaluator accepts three one-dimensional curves:
- motor efficiency versus shaft-load fraction;
- inverter efficiency versus inverter input-load fraction;
- propeller efficiency versus advance ratio.

Values and absolute uncertainty bounds are linearly interpolated only inside each supplied domain. The mission evaluator uses nominal efficiency minus the absolute bound and records dataset IDs. No extrapolation is permitted.

The characterized curves remain projections, not complete performance maps. Motor and inverter maps omit independent RPM/torque/voltage/temperature dimensions; the propeller curve is not a full coefficient map over advance ratio, power coefficient, blade pitch, Reynolds number, Mach number, density and blade condition. A 1D curve must be limited to its stated fixed conditions.

## Battery characterization path

The scalar battery evaluator uses constant open-circuit voltage and internal resistance. An optional pack-level characterization accepts bilinear grids for open-circuit voltage and DC resistance over SOC and pack temperature. Both grids must have identical axes and units. The solver subtracts absolute OCV uncertainty and adds absolute resistance uncertainty; it rejects points outside the characterized domain. Selected values are recorded per segment. Mission reports preserve the full source evidence records (ID, URI, evidence class, description and SHA-256 digest) as well as dataset ID lists. The evidence_content_digests_verified flag is set only by the fully verified evaluator after matching caller-supplied bytes; it does not verify signatures, publisher/authority trust, URI resolution or configuration approval.

This is still an equivalent-circuit screening model. It does not model cell-level behaviour, charge/discharge hysteresis, current-rate dependence, SOH-dependent OCV/resistance, transient polarization branches, cell imbalance, contactors/fuses, propagation, venting, or thermal runaway. SOC capacity remains nominal capacity scaled by SOH; no temperature/rate-dependent usable-capacity surface is implemented. The thermal state remains a single lumped pack node.

## Adverse-corner uncertainty path

The uncertainty module applies one-sided fractional bounds to mass, airspeed, lift-to-drag ratio, climb rate, auxiliary demand, battery capacity, voltage, resistance, power/current limits, thermal capacitance/conductance/headroom, and scalar propulsion efficiencies. It returns both nominal and adverse-corner results and retains the exact adjusted input and budget.

The current adverse-corner evaluator wraps the scalar mission evaluator. It does not yet propagate uncertainty through the characterized propulsion curves or battery grids. Its simultaneous corner assumes only the declared one-sided parameter directions; it does not model covariance, correlation, distributions, likelihood, or every nonlinear parameter interaction. It must not be called a probabilistic interval or certified worst-case proof. See ROBUST_UNCERTAINTY.md.

## Current reduced-order equations

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

1. Aerodynamic L/D is supplied, not derived from geometry or a validated polar. It must be tied to a source, flight condition, configuration and uncertainty interval by a higher-level caller.
2. The model assumes lift approximately equals weight and omits manoeuvre load factors, gust loads, induced/parasite drag decomposition, compressibility, ground effect, propwash interactions, Reynolds effects and aeroelastic coupling.
3. It is not a takeoff, landing, stall, climb-envelope or flight-dynamics model. Every segment is a steady positive-airspeed condition.
4. Scalar propulsion efficiency is fixed per mission. Characterized propulsion uses one-dimensional projections, not full motor/inverter/propeller performance surfaces.
5. Scalar battery OCV and resistance are constant. Characterized battery OCV/resistance surfaces cover only SOC and pack temperature; no cell-level electrochemistry, charge/discharge hysteresis, rate dependence, SOH-dependent OCV/resistance or dynamic polarization is modelled.
6. The pack thermal network is one lumped node and cannot establish local hot spots, cell-to-cell propagation, cooling-loop behaviour or aircraft installation safety.
7. Auxiliary power is a scalar load. Electrical buses, redundancy, transient loads, charging infrastructure, fault isolation and common-cause electrical failures are not simulated.
8. Battery specific energy is an input. There is no cell/module/pack/installed-system mass breakdown or life-cycle battery degradation model beyond SOH scaling.
9. Energy use is based on supplied true airspeed and segment duration. Wind, density altitude, atmospheric variation, route feasibility, terrain, runway, reserves required by a regulator, and alternate-airport policy are not modelled.
10. The scalar mission uncertainty fraction is a user-supplied screening margin. The adverse-corner budget is a deterministic one-sided screening scenario. Neither is a statistical confidence interval or regulatory reserve.
11. The adverse-corner method does not yet evaluate characterized map paths and does not capture parameter correlation or probability distributions.
12. Evidence records validate metadata and digest syntax. The verified-map API recomputes SHA-256 over caller-supplied bytes and rejects missing/mismatched payloads, but it does not fetch evidence URIs, verify signatures, authenticate source authority, or establish provenance/configuration approval.
13. The model does not claim any battery chemistry, specific energy, range, certification status, airworthiness or safety performance.

## Promotion gates

Do not promote this kernel to aircraft design authority until all of the following exist:

- versioned, traceable aerodynamic polars and uncertainty bounds;
- measured multi-dimensional motor/inverter/propeller maps and calibration datasets;
- independently measured pack OCV/resistance/capacity/thermal maps over SOC, temperature, age, current rate and charge/discharge direction;
- cell/pack abuse, isolation, propagation and containment evidence for the selected battery design;
- an evidence resolver that obtains immutable artifact bytes, verifies hashes/signatures and binds evidence to exact configurations;
- comparison against independently implemented higher-fidelity models;
- test-correlation evidence from dynamometer, thermal-chamber, propeller, electrical fault-injection and instrumented ground tests;
- an uncertainty method that is calibrated on independent data and handles material parameter correlations;
- a hazard analysis, failure-mode analysis and reviewed safety architecture;
- an agreed certification basis and compliance plan with the competent authority;
- independent engineering review and configuration-controlled evidence.

Unknown or missing evidence must remain unknown/incomplete, not be silently replaced by nominal values.
