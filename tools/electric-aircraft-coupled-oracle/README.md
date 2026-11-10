# Independent electric-aircraft coupled reference oracle

**Package:** `electric-aircraft-coupled-oracle` v0.1.0  
**Implementation:** dependency-free Rust, isolated Cargo workspace  
**Purpose:** independent scalar reference calculations for the next Symthaea mission electro-thermal solver.

## Why this exists

The existing mission-energy crate and thermal overlay are useful engineering increments, but the true coupled solver must be tested against an independently implemented reference. Reusing the production thermal network, mission evaluator or battery helpers as the oracle would create a circular verification argument.

This package therefore implements a deliberately small one-node reference problem using:
- a closed-form lower-current solution of the battery terminal-power quadratic;
- a temperature-dependent pack resistance law;
- an implicit backward-Euler thermal balance;
- a safeguarded scalar bisection root solve rather than the production RC network solver;
- explicit SOC, current, power, temperature and energy-balance checks;
- piecewise-constant mission segments with retained substep history;
- separate propulsion and thermal-management electrical loads, including the thermal-management share of total battery-terminal mission energy.

The crate has no third-party dependencies and is not a member of the main Symthaea workspace. It can be tested independently from the workspace root:

```sh
cargo fmt --manifest-path tools/electric-aircraft-coupled-oracle/Cargo.toml -- --check
cargo test --locked --manifest-path tools/electric-aircraft-coupled-oracle/Cargo.toml
cargo clippy --locked --manifest-path tools/electric-aircraft-coupled-oracle/Cargo.toml --all-targets -- -D warnings
```

A dedicated exact-path CI workflow is provided in `.github/workflows/electric-aircraft-coupled-oracle.yml`.

## Reference equations and model boundary

Open-circuit voltage is evaluated at the start-of-substep SOC:

```
OCV = OCV_ref + k_soc * (SOC_start - SOC_ref)
```

Resistance varies linearly with the implicit end-of-substep temperature:

```
R(T) = R_ref * [1 + alpha_T * (T - T_ref)]
```

The battery current is the lower-current physical root:

```
P_terminal = I * (OCV - I*R)
I = 2*P_terminal / (OCV + sqrt(OCV^2 - 4*R*P_terminal))
```

A negative discriminant is returned as `ElectricalInfeasible`; current is never clipped and required power is never silently reduced.

The scalar thermal state is solved from:

```
C * (T_next - T_now) / dt + G * (T_next - T_ambient) = I(T_next)^2 * R(T_next)
```

Bisection solves the implicit end temperature. Chemical energy then updates SOC. A substep is returned only after reserve, power, current, thermal-domain and energy-balance checks pass.

At mission level, each segment declares propulsion terminal power and thermal-management terminal power separately. Both are charged to the pack-terminal requirement and energy total. The report exposes propulsion energy, thermal-management energy and the TMS fraction of total terminal energy. TMS electricity is not silently reclassified as pack I²R heat: its component heat, pump/fan/compressor losses, mass, volume, pressure loss and cooling boundary still need explicit subsystem models and measurements.

## What this oracle does not model

This deliberately small reference plant:
- freezes OCV at the beginning-of-substep SOC (it does not model within-step OCV dynamics);
- uses one lumped battery thermal node;
- has a linear temperature-resistance law;
- does not model electrochemical polarization, hysteresis, C-rate dependence, SOH-dependent resistance, cell-to-cell imbalance or degradation;
- does not model motor/inverter efficiency, propeller maps, coolant flow, heat-exchanger physics, cooling mass/drag/parasitic power, aerodynamics or aircraft dynamics;
- does not model thermal runaway, cell propagation, battery safety or certification.

The oracle is useful precisely because it has a small, inspectable boundary. It is not a substitute for a validated SPMe/ECM, thermal-fluid or aircraft-level solver.

## Frozen synthetic tests

The current reference crate contains 12 deterministic Rust tests:

| Test | Purpose |
|---|---|
| Constant-resistance closed form | Check the independent bisection result against a separately derived algebraic temperature/current solution |
| Temperature-resistance feedback | Confirm higher resistance-temperature coupling changes current, heat and final temperature in the expected direction |
| No real current root | Reject terminal demand above the battery's quadratic power limit |
| Thermal ceiling | Reject a substep whose implicit thermal equilibrium lies above the declared domain |
| Reserve infringement | Keep energy-reserve failure distinct from current/thermal failure |
| Iteration limit | Return bracket width and residual instead of a partial result |
| Power/current limits | Preserve electrical constraints as separate typed outcomes |
| Initial state OOD | Reject an uncharacterized starting temperature |
| Mission replay | Retain segment identities and all substep results |
| TMS energy attribution | Keep a 20% synthetic thermal-management electrical load visible in the mission energy report |
| Time-step refinement | Compare 10 s and 1 s mission integrations |
| Failed substep rollback | Return the segment/substep failure and no partial mission PASS |

Synthetic tests prove only the reference implementation's bounded arithmetic and disposition contract. They are not physical data.

## Research state of the art — what we must not overclaim

Mission-level electro-thermal modelling is already active research, not a new claim for Luminous. An August 2026 eVTOL study reports an integrated electrical/thermal mission simulation and finds thermal-management energy exceeding 20% of total mission energy in its study cases. This directly motivates separately metering propulsion and thermal-management electrical loads in our mission contract. A 2026 framework for the whole eVTOL thermal system models interactions among battery, power electronics, cabin and avionics thermal zones, and reports up to 95% lower thermal-management energy in its best-case simulated configuration; this is a model result, not a fleet or flight-test achievement. A 2026 topology-optimized phase-change battery cooling study reports keeping the modeled battery below 41 °C at 30 °C ambient while its thermal-management mass remains below 33.3% of pack mass, with CFD predictions within 1 K of the topology-optimization result in the study's selected cases. NASA's 2026 battery-module thermal-fluid work measures component temperatures, coolant flow and pressure drop under 0.5C–2.5C discharge. SAE AIR5744, reaffirmed January 2026, encourages platform-level aircraft thermal-management engineering rather than separately optimized thermal subsystems.

These sources change our research strategy:
1. Do not claim novelty for simply coupling mission energy and temperature.
2. Charge thermal-management energy, mass, volume, pressure loss, drag and heat-rejection boundary to the aircraft mission.
3. Add ageing, SOC, electrical topology, cooling architecture and mission/season distributions once evidence exists.
4. Differentiate through traceable evidence, uncertainty calibration, strict OOD handling, independent solver checks and decision-aware experiment selection.
5. Compare our software against independent solvers and physical rigs, not merely another synthetic implementation.

Sources:
- Nakayama et al. (August 2026), *Mission-level integrated electrical–thermal simulation for energy and thermal management of eVTOL aircraft*: https://www.sciencedirect.com/science/article/pii/S1359431126025615
- Kaya (2026), *Mission profile based electrochemical-thermal battery modeling for vertical take-off and landing aerial vehicles*: https://www.sciencedirect.com/science/article/pii/S2352152X26030069
- Wilhite & Stalcup (NASA TFAWS 2026), *Thermal-Fluid Analysis of a Liquid-Cooled Battery Module for Electrified Aircraft*: https://ntrs.nasa.gov/citations/20260008248
- Lu et al. (March 2026), *Design of a compact phase change material-integrated battery thermal management system for electric aircraft via topology optimization*: https://doi.org/10.1016/j.applthermaleng.2026.129829
- SAE AIR5744, *Aircraft Thermal Management System Engineering*, reaffirmed 2026-01-21: https://saemobilus.sae.org/standards/air5744-aircraft-thermal-management-system-engineering

## Next solver acceptance gates

These are proposed internal goals, not achieved outcomes or standards.

- Exact analytical agreement for constant-resistance and one-node equilibrium cases within a preregistered tolerance.
- 100% correct typed disposition across known-answer electrical, thermal, reserve, OOD and numerical-failure controls.
- No false-feasible outcome in the frozen synthetic negative-control corpus.
- Refining the time step changes mission energy and peak temperature by less than the declared convergence tolerance.
- Cross-check the upcoming production fixed-point solver against this independently coded bisection result over a parameter grid.
- After physical testing exists, evaluate voltage/current and component-temperature residuals, prediction-interval coverage and false-safe/false-derate rates on held-out component configurations.

See [AIRROAD-ELEC-THERMAL-001 #11411](https://github.com/Luminous-Dynamics/symthaea/issues/11411) for the next fully coupled solver contract and [PR #11409](https://github.com/Luminous-Dynamics/symthaea/pull/11409) for the mission thermal feasibility overlay that this oracle is intended to help challenge.
