# Aircraft energy screening kernel

This crate is the first executable tranche for [AIRCRAFT-ENERGY-001](https://github.com/Luminous-Dynamics/symthaea/issues/7646). Its provenance boundary composes the existing Mycelix [AeroCommons engineering/manufacturing commons](https://github.com/Luminous-Dynamics/mycelix/issues/3638) and [physical-evidence reproducibility challenge](https://github.com/Luminous-Dynamics/mycelix/issues/3643); it does not duplicate their records or mint engineering approval.

## Purpose

Provide deterministic, small, dependency-free arithmetic checks before expensive multidisciplinary aircraft analyses. The kernel separates:

- mission energy demand (Wh);
- continuous bus-power demand (W);
- peak bus-power demand and cumulative peak duration;
- usable energy after declared reserve;
- complete installed energy-system mass (kg);
- battery pack-level specific energy/power with explicit use, temperature, end-of-life, efficiency, and power derating.

The Rust workspace discovers this crate through the existing `crates/domains/*` member glob. Run its tests with:

```sh
cargo test -p symthaea-aircraft-energy
```

## Model contract

`BatteryPackSpec::to_envelope` computes this screening approximation:

```text
usable_bus_energy_Wh =
    installed_pack_mass_kg
  × pack_specific_energy_Wh_per_kg
  × usable_fraction
  × temperature_capacity_fraction
  × end_of_life_soh_fraction
  × discharge_efficiency_fraction

continuous_bus_power_W =
    installed_pack_mass_kg
  × continuous_specific_power_W_per_kg
  × power_derating_fraction

peak_bus_power_W =
    installed_pack_mass_kg
  × peak_specific_power_W_per_kg
  × power_derating_fraction
```

The energy factors are treated multiplicatively because that is the declared input contract; users must not count the same loss/derate twice. The energy and power values are estimates at a declared aircraft bus. Use pack-level, not cell-level, performance inputs. Pack mass determines battery energy and power; a separately declared complete installed-system mass must include the pack plus the other hardware inside the comparison boundary. The model rejects an installed-system mass lower than the pack mass.

`evaluate_mission` sums each segment's `required_bus_power_W × duration_s / 3600`, withholds the declared reserve from bus energy, checks each continuous segment against continuous power, checks peak segments against peak power, and sums peak-segment durations against the mission peak-duration budget.

A missing or invalid input produces `Indeterminate`, never a pass. A hard arithmetic limit failure produces `InfeasibleWithinScreen`. A pass is only a pass of this limited screen.

## What is deliberately not modeled

- a source-dispatch policy or hybrid battery/generator fuel-energy trajectory;
- proof that separate sources can operate in parallel or share a bus;
- recharge/regeneration during the mission;
- battery electrochemistry, state estimation, cell imbalance, thermal transients, aging laws, propagation of uncertainty, or fault containment;
- voltage stability, inverter/motor/generator maps, cable mass, cooling and heat-rejection drag;
- aerodynamics, structures, stability/control, climb performance, environmental envelope, or airworthiness;
- reactor core/thermal output converted into aircraft-level net electric power;
- lifecycle economics, environmental impact, manufacturing yield, certification or operating approval.

For hybrid-electric, fission, or fusion concepts, the upstream analysis must provide defensible bus-level energy/power and complete installed-mass evidence. Do not add nominal energy/power values unless simultaneous delivery and conversion loss are justified. Reactor thermal power is not bus electric power; reactor/core mass is not installed-plant mass.

## Initial adversarial tests

The 15 unit tests include hand-computed battery derating, mission-energy arithmetic, energy shortfall despite sufficient power, peak-power shortfall despite sufficient energy, continuous-power shortfall, cumulative peak-duration limit, missing energy and duty-class evidence, non-finite inputs, duplicate segment identity, invalid battery derating, invalid reserve, energy arithmetic overflow, zero-duration rejection, and an installed-system mass lower than the pack mass.

These are software-contract tests, not empirical battery characterization or aircraft validation. Before external performance claims, add an independently frozen mission corpus, independent expected values/oracle, source/model provenance, held-out reference measurements, uncertainty bounds, and comparisons against conventional tools.

## Claim ceiling

A `FeasibleWithinScreen` result is not an aircraft-feasibility, flight-performance, airworthiness, battery-safety, reactor-safety, commercial-readiness, or operational-authority claim. Physical testing and applicable external approvals remain separate gates.
