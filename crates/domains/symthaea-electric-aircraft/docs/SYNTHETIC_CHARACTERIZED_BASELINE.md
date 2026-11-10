# Synthetic characterized mission baseline v0.2

**Evidence class:** synthetic software regression fixture only.  
**Model:** `electric-fixed-wing-energy-v0.2.0`.  
**Not a physical aircraft prediction, measured component map, or product specification.**

## Synthetic mission

Uses the v0.1 baseline aircraft and mission:
- 640 kg computed all-up mass;
- three segments: climb, cruise, descent;
- 40 kWh nominal battery with 95% state of health;
- initial SOC 95%, protected reserve 20%;
- 20% mission uncertainty margin;
- battery model remains the v0.1 constant-OCV / constant-resistance / lumped-thermal model.

## Synthetic propulsion maps

The maps below exist only to exercise software interpolation and no-extrapolation.

| Component | Independent variable | Range | Nominal efficiency | Absolute uncertainty |
|---|---|---:|---:|---:|
| Motor | Shaft-load fraction | 0.1–1.0 | 0.92–0.96 | 0.01 |
| Inverter | Input-load fraction | 0.1–1.0 | 0.97–0.985 | 0.005 |
| Propeller | Advance ratio | 0.4–1.0 | 0.80–0.86 | 0.02 |

Map geometry parameters:
- motor rated shaft power: 80 kW;
- inverter rated input power: 90 kW;
- propeller diameter: 1.6 m;
- propeller speed: 2,400 RPM.

The test evaluator uses each nominal efficiency minus its supplied absolute uncertainty bound. The curve metadata labels these synthetic datasets as Simulation, not physical measurements.

## Independently recalculated outputs

| Output | Result |
|---|---:|
| Terminal mission energy | 14.923 kWh |
| Equivalent chemical energy | 15.052 kWh |
| Energy requirement with 20% screening margin | 18.062 kWh |
| Energy available above protected reserve | 28.500 kWh |
| Calculated energy margin | 10.438 kWh |
| Climb propeller efficiency used | 0.7946875 |

The Rust unit test pins the first three outputs and climb propeller efficiency against this separately calculated oracle, within numerical tolerances.

## Interpretation

This synthetic characterized mission requires more energy than the scalar baseline because the conservative lower bounds on the component maps are less optimistic at the evaluated operating points. It demonstrates software behaviour only.

A real map must be produced from controlled motor dynamometer, inverter, and propeller tests, with raw data, instrumentation calibration, measurement uncertainty, configuration identity, test environment and verified immutable evidence. A 1D curve must not be applied outside its validity conditions. The v0.2 code validates digest formatting but does not independently fetch or hash the artifact bytes.
