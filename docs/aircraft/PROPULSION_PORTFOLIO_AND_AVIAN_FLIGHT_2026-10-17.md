# Propulsion Portfolio and Avian Flight Research Program

**Date:** 2026-10-17  
**Status:** research program proposal; no system-level performance or flight capability is claimed.  
**Tracking:** [PROP-PORTFOLIO-001 #7897](https://github.com/Luminous-Dynamics/symthaea/issues/7897), [AVIAN-001 #7898](https://github.com/Luminous-Dynamics/symthaea/issues/7898), [PROP-REGIME-001 #7896](https://github.com/Luminous-Dynamics/symthaea/issues/7896).

## Executive recommendation

Yes: Symthaea should research more than conventional propellers. But it should not try to design all propulsion systems with one universal optimizer. Establish a common, evidence-preserving architecture layer, then use mission-specific models and validation gates for each physical regime.

The highest-priority near-term path remains electric propulsors and integrated fixed-wing propulsion, because the existing aircraft-energy research already provides bounded maps, cross-channel audits, dimensionless coefficients and a signed regime vocabulary. The strongest new parallel research program is a **small, unmanned bird-inspired flight demonstrator** combining flapping, compliant/morphing wings, glide, reflexive control, and potentially jump-assisted launch. Its goal is functional capabilities with measured value, not visual imitation alone.

Rocket engines and electric space thrusters deserve architecture-level research as separate mission families. They should not be compared against atmospheric propellers with a raw thrust/efficiency score: their working-fluid assumptions, mission objective, mass ratio, energy source and operating environment are fundamentally different.

## 1. Separate the three layers of propulsion architecture

A common model should factor a system into three separable decisions:

1. **Energy source or store:** battery, fuel cell, hydrocarbon fuel, chemical propellant, solar/electric supply, or another source with explicitly accounted limitations.
2. **Energy conversion chain:** electric motor/inverter, turbine-generator, combustion and nozzle, electrostatic/electromagnetic acceleration, mechanical transmission, or a flapping drivetrain with elastic elements.
3. **Momentum-transfer device and installation:** propeller, rotor, ducted fan, jet/nozzle, electric in-space thruster, oscillating wing, or a coupled propulsor/airframe arrangement.

These layers interact but must not be collapsed into a single label. “Electric” describes part of an energy/conversion chain; it does not specify whether the vehicle uses a propeller, ducted fan, flapping wings, or an in-space electric thruster.

## 2. Which propulsion families should Symthaea investigate?

| Family | Where it is worth studying | Primary metrics | Why not a universal winner |
|---|---|---|---|
| Open propeller / rotor | Low-to-moderate speed atmospheric flight, hover/VTOL with suitable architecture | Mission energy, thrust/power, diameter, tip Mach, acoustic signature, installed mass | Diameter, ground clearance, inflow and airframe interaction constrain it |
| Ducted fan | Where diameter, guarding, integration, speed or a particular installation justifies the duct | Installed thrust/power, duct mass/drag, inlet distortion, thermal limits, noise | The duct is not free; its mass and losses must be charged to the system |
| Distributed electric propulsion | Wing-blown lift, configuration-specific interaction, redundancy or installation benefits | Whole-mission energy, installed power/mass, interaction forces, reliability, noise | Benefits depend on integration and can be reversed by motors, wiring, cooling and structural mass |
| Turbine / turbofan / turboelectric | Higher-speed atmospheric missions or long-energy-duration use cases where the system's cost/complexity fits | Net installed thrust, fuel/energy consumption, thermal limits, noise/emissions, maintenance | Different speed and mission envelope; compare whole system and life cycle |
| Flapping wing / ornithopter | Small UAVs where take-off mode, maneuverability, acoustic profile, morphing or low-Reynolds-number performance matters | Electrical energy per mission/distance, effective lift, payload, control authority, mechanism mass/life, repeatability | Flexible-wing fluid-structure interaction and drivetrain losses make the system difficult to model |
| Chemical rocket | Launch and high-thrust vacuum/space stages; selected specialized applications | Thrust, specific impulse, propellant mass, total impulse, structural/thermal mass, mission delta-v | Carries reaction mass/oxidizer; not a general replacement for aircraft propulsors |
| Electric in-space thruster | Spacecraft that can trade low thrust for high propellant efficiency over long durations | Specific impulse, thrust per electrical power, propellant, power-system/radiator mass, lifetime, delivered mission delta-v | Not an atmospheric flight thruster; energy and acceleration constraints are different |

NASA's propulsion overview explains the different physical basis of propellers, turbines/jets, ramjets and rockets. NASA's electrified-aircraft architecture taxonomy further separates all-electric, hybrid-electric and turboelectric arrangements:
- https://www.grc.nasa.gov/WWW/K-12/BGP/
- https://www.nasa.gov/eap-aircraft-concepts/aircraft-configurations/

**Priority order for Symthaea:**
1. Continue the existing bounded propeller-map, power-chain and dimensionless-comparison work.
2. Add mission-matched ducted/distributed electric configurations only when evidence and coupling models cover the conditions.
3. Start the bird-inspired ornithopter as a separate, small-UAV research track sharing common mass, mission, energy and evidence contracts.
4. Create a rocket/space-thruster *system-level* benchmark later as a separate problem domain, beginning with published analytical/reference cases rather than a hardware design.
5. Add turbine/jet performance maps as another architecture family only when a clear mission and a rights-cleared, traceable reference dataset are selected.

This ordering is a research priority, not a claim that one technology is universally superior.

## 3. Could Symthaea design a flying machine that flies like a bird?

**Yes, as a realistic research objective for a small unmanned aerial robot.** It is substantially more realistic to first achieve bird-inspired take-off, morphing, gliding, gust response and landing than to attempt a bird-like human-carrying aircraft.

The field has already demonstrated several important ingredients:

- A 2025 review of flapping-wing robots describes the coupled challenges of actuation, flexible-wing dynamics, flight control, navigation, sensing and the potential energy benefit of combining flapping with gliding: https://doi.org/10.1177/02783649251343638
- PigeonBot II demonstrated rudderless flight control using bio-inspired reflexive wing and tail morphing: https://pubmed.ncbi.nlm.nih.gov/39565868/
- RAVEN demonstrated an avian-inspired robot that walks, hops and jump-launches into flight. In that study, jump-assisted take-off increased initial flight speed and was more energy-efficient than the comparison without a jump: https://www.nature.com/articles/s41586-024-08228-9
- Research has also demonstrated avian-inspired embodied sensing for flapping-wing robots: https://www.nature.com/articles/s41467-024-53517-6

These results show that bird-inspired flight is not speculative fiction. They do **not** establish that efficient, reliable, long-endurance bird-like flight is solved.

### Proposed first demonstrator: a modular robotic bird

Begin with a small unmanned vehicle whose structure and tests are modular, and compare every major bio-inspired mechanism against a conventional, mass-matched fixed-wing electric baseline.

**A. Compliant flapping wings**

Use a lightweight wing structure with passive compliance and a replaceable drivetrain. Model crank/gear losses, wing inertia, actuator power, structural loads and cyclic durability. Avoid a design with an independently actuated motor at each feather: mechanism complexity and mass are likely to defeat the purpose.

**B. Morphing wing and tail**

Start with a small number of controlled degrees of freedom—wrist/finger-like wing-shape changes and a morphing tail—plus passive elements that coordinate the surface. Test whether morphing materially improves turn control, gust recovery and low-speed stability against a fixed-geometry baseline.

**C. Flapping plus glide**

Use active flapping for propulsion and control authority when required, then glide when the energy state, airspeed and atmospheric conditions permit. Glide energy comes from previously acquired altitude and/or atmospheric energy; it is not free energy. Estimate benefit by energy per complete mission, including the costs of climbing and transitions.

**D. Bird-inspired ground-to-air transition**

Evaluate compliant legs and spring-assisted jump launch as an optional module. Charge the leg mechanism's mass, actuator energy and failure modes to the candidate. It is useful only if it enables a launch profile or mission that the baseline cannot meet, or reduces mission energy on the same mission.

**E. Embodied sensing and adaptive control**

Combine motor current, wing phase, inertial state and appropriate airflow/shape sensing to estimate flight state. A controller should adapt wingbeat and morphology to the current mode, report when its model is outside the evidence domain, and recover safely from perturbations. Use onboard autonomy as a testable control capability, not as a substitute for aerodynamic validity.

### Research hypotheses—not assumptions

- Flapping plus glide can reduce energy per mission or unlock launch/landing/maneuver capabilities versus a mass-matched propeller baseline.
- Passive compliance and elastic energy return reduce net electrical energy per wingbeat without reducing control performance or mechanism life.
- Morphing wing/tail control improves disturbance rejection compared with a fixed morphology.
- Jump-assisted launch improves take-off capability after its extra mass and mechanical losses are included.

Any of these can fail. A useful research program should make failure visible early rather than optimize only the mechanism that looks most bird-like.

## 4. Bird-like flight benchmark ladder

All numerical values below are proposed engineering gates, not measured achievements. Set exact tolerances after selecting sensor specifications and prototype scale.

### M0 — simulation contract

- Independent known-answer checks for rigid-body dynamics, wingbeat cycle energy, force/moment sign conventions and mass accounting.
- Explicit model validity domains for unsteady aerodynamics, wing compliance and actuator behavior.
- No fabricated aerodynamic coefficients or hidden defaults.
- Repeatable simulation results for pinned input manifests and solver versions.

### M1 — bench measurements

Measure actuator electrical input, shaft/mechanical output where available, wing kinematics, force/moment, mechanism temperature and cyclic fatigue. Keep electrical power, mechanical shaft power and energy stored/returned by elastic elements distinct. Report sensor resolution, calibration, sample rate and uncertainty.

### M2 — controlled, unmanned flight

In a professionally reviewed, contained test environment, demonstrate repeatable transition from launch to powered flight, controlled turns, glide where applicable, and safe landing. Publish all attempted runs including failures and weather/environment conditions. Simulation success does not qualify physical flight.

### M3 — matched baseline

Compare the ornithopter against a conventional fixed-wing propeller UAV at matched gross mass, payload, mission route, reserve policy and environmental conditions. Report:
- total mission joules and joules per ground-track or air-track distance, with wind treatment explicit;
- payload fraction and installed mechanism mass;
- flight duration and altitude loss during glide;
- trajectory/attitude tracking error;
- gust/disturbance recovery;
- launch and landing success rates;
- acoustic spectra under a specified measurement protocol;
- energy, reliability and durability uncertainty.

Do not compare a lightweight glider's glide segment with a powered ornithopter's complete mission, or compare endurance at unmatched payloads.

### M4 — repeatability and robustness

As an initial target, achieve 20 consecutive documented mission attempts under a frozen test matrix, while publishing every success, failure, aborted run and environmental condition. Add separate tests for gust disturbances, battery state, mechanism wear and sensor dropouts. Twenty successes are an early repeatability gate, not a sufficient reliability demonstration.

### M5 — independent replication

Require separate operators or an independent lab to reproduce a subset of the flight and energy results using the pinned design, firmware/model version and test protocol before publishing general performance claims.

No human-carrying or operational-use claim follows from these gates. A small robotic demonstration is the appropriate first milestone.

## 5. The potential new paradigm: mission-adaptive morphology and propulsion

Instead of treating propulsion as a fixed engine attached to a fixed airframe, make **morphology, propulsion, energy storage and control one co-designed system**.

The vehicle would have several controlled modes:
- jump/launch when ground support is available;
- flapping for propulsion and high control authority;
- morphing for maneuvering and disturbance rejection;
- glide when the energy/trajectory state permits;
- a defined landing/perching or recovery mode.

Symthaea should learn a model of when each mode is advantageous from source-backed measurements and simulations, while an independent checker enforces energy, structural, operating-envelope and controller constraints. This is the promising intersection between propulsion design, embodied robotics and evidence-driven optimization.

The same general co-design principle applies to jets, ducted fans and rockets, but their physical model and mission constraints must remain separate. A rocket's specific impulse, a jet's installed thrust and a flapping robot's energy per meter are not interchangeable scores.

## 6. New-research benchmarks and evidence rules

Across all architecture families:
1. Pin mission assumptions, complete mass boundary, configuration and evidence digests.
2. Separate experiment, analytical model, simulation, synthetic fixture and derived value.
3. Hold out complete configurations or source campaigns, not merely random neighboring rows.
4. Report error and uncertainty by operating condition and architecture.
5. Preserve infeasible and indeterminate outcomes rather than forcing all designs to rank.
6. A design advantage counts only when it exceeds combined uncertainty on the same mission.
7. Keep all test declarations distinct from executed tests and all model results distinct from physical qualification.

For regime discovery, a new regime is a falsifiable hypothesis until independent measurements reproduce its boundary or distinct response. For an ornithopter, bird-like appearance is not the benchmark; measured mission energy, control, robustness, launch/landing and durability are.

## References

- NASA Glenn, [Beginner's Guide to Propulsion](https://www.grc.nasa.gov/WWW/K-12/BGP/).
- NASA, [Electrified Aircraft Propulsion configurations](https://www.nasa.gov/eap-aircraft-concepts/aircraft-configurations/).
- Nekoo et al., [A review on flapping-wing robots: Recent progress and challenges](https://doi.org/10.1177/02783649251343638), *International Journal of Robotics Research*, 2025.
- Chang et al., [Bird-inspired reflexive morphing enables rudderless flight](https://pubmed.ncbi.nlm.nih.gov/39565868/).
- Shin et al., [Fast ground-to-air transition with avian-inspired multifunctional legs](https://www.nature.com/articles/s41586-024-08228-9), *Nature*, 2024.
- Li et al., [Avian-inspired embodied perception in biohybrid flapping-wing robotics](https://www.nature.com/articles/s41467-024-53517-6), *Nature Communications*, 2024.
