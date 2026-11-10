# Coupled thermal network model

**Module:** thermal_network.rs  
**Evidence class:** reduced-order numerical model; synthetic unit tests only.  
**Not:** CFD, a battery abuse model, cell-hot-spot prediction, cooling-system qualification or a thermal safety case.

## Model

The module advances a network of lumped thermal nodes using backward Euler:

\[
C_i \frac{dT_i}{dt}
=
Q_i
+
G_{i,a}(T_a-T_i)
+
\sum_j G_{ij}(T_j-T_i)
\]

where:
- C_i is the node heat capacity (J/K);
- Q_i is heat generated at the node (W);
- G_i,a is conductance to the fixed ambient boundary (W/K);
- G_ij is symmetric conductance between connected nodes (W/K).

For each step, the implementation solves the implicit linear system. Partial-pivoting Gaussian elimination is used. A time step returns a new state and an energy-balance report; it does not mutate the supplied state. Node ceilings are fail-closed.

## Intended subsystem decomposition

An initial network may contain:
- battery pack / enclosure;
- inverter power module;
- motor stator / housing;
- coolant reservoir or representative coolant mass;
- heat exchanger / aircraft skin or ambient boundary.

The network links represent effective conductive paths. A node's ambient conductance represents the combined effective heat-rejection path to the declared fixed ambient boundary. If a coolant node is included, it should have its own capacity and appropriately justified rejection conductance.

Heat generation is supplied externally per node and per timestep. It should be derived from measured or modelled component losses at the same operating point. Do not double count motor/inverter losses or assign all aircraft waste heat to the battery.

## Numerical properties

- Backward Euler is unconditionally stable for this passive linear RC network.
- The returned energy balance is: heat input minus net ambient rejection minus stored energy change. Internal link transfers cancel from the network sum.
- A small residual is a software/numerical consistency check, not evidence that the conductance network matches physical hardware.
- The model is deterministic for a fixed network, state, heat vector, time step and ambient temperature.
- The implementation is intended for small subsystem networks. Its dense solve scales cubically with node count; it is not a large CFD mesh solver.

## What it does not model

- Coolant flow, flow distribution, pump curves, pressure drop, heat-exchanger effectiveness or changing coolant inlet temperature.
- Convective coefficients as functions of airspeed, altitude, density, fan speed or Reynolds number.
- Radiation, phase change, boiling, condensation or two-phase coolant.
- Anisotropic or temperature-dependent material properties.
- Spatial gradients within each node, cell-to-cell imbalance, cell hot spots or thermal propagation.
- Electrical-thermal feedback unless the caller iteratively updates heat generation and electrical operating points.
- Battery chemistry, venting, thermal runaway, fire containment or abuse response.
- Motor winding insulation degradation or inverter semiconductor junction detail.

A single node can never establish internal hot-spot safety. Splitting a component into more nodes only improves resolution if the node capacities, conductances and heat sources are supported by physical evidence.

## Validation campaign

1. Calibrate heat-source measurements and thermal sensors, including uncertainty, response time and placement.
2. Characterize each component separately with controlled heat-input steps and steady-state holds.
3. Identify thermal capacities and conductances from independent heating/cooling tests; preserve raw time series and fitting code.
4. Couple two-node and multi-node rigs and compare predicted temperatures and energy balance with held-out experiments.
5. Test ambient extremes, hot soak, repeated high-power bursts, rapid turnaround and degraded heat rejection.
6. Compare predictions at each instrumented measurement location, reporting bias, RMSE, maximum absolute error, peak timing error and interval coverage.
7. Freeze network parameters and validate on a preregistered holdout set before using the model to compare aircraft architectures.

Proposed Luminous research target after physical instrumentation exists: component-temperature RMSE below 2 K on a preregistered independent holdout set, with empirical 95% prediction-interval coverage reported rather than assumed. This is an internal research goal, not an industry standard or achieved result. The permissible error must also be judged against the component's actual thermal margin; 2 K may be insufficient near a safety limit.

## Relationship to NASA state of the art

NASA's HEATherR work explicitly treats power-conversion losses and heat rejection as aircraft-level mass, drag and power problems. Its outer-mold-line cooling concept seeks to reduce the mass and drag of conventional pumped cooling loops. NASA's NEAT supports integrated megawatt-class powertrain testing under simulated flight altitudes up to 60,000 ft. NASA's 2026 battery-module thermal-fluid study measures an integrated liquid-cooled module and flow/pressure-drop behaviour over discharge rates up to 2.5C.

The Luminous thermal network is not competitive with those facilities or specialist thermal-fluid solvers in physical fidelity. Its value is an inspectable early-stage network that connects subsystem losses and thermal masses to mission-level trades, before coupling higher-fidelity heat-transfer solvers and physical tests.

Sources:
- https://www.nasa.gov/eap-technology/heather/
- https://www.nasa.gov/eap-labs-and-testbeds/neat/
- https://ntrs.nasa.gov/citations/20260008248
