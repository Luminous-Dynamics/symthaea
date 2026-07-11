// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real-electrical-physics simulator backend, built on `symthaea-grid-physics`
//! (battery SoC/efficiency, radial feeder power flow, droop control, voltage
//! trip envelope, passive islanding detection) rather than the default
//! `SimpleInfrastructureSimulator`'s coupled heuristics.
//!
//! Behind the `grid_physics` feature — see
//! PLANETARY_ENERGY_COORDINATION_PLAN_2026-07-06.md Phase 1 ("wire it as a
//! second simulator backend... keep the heuristic sim as fast default").
//!
//! **Honesty note on channel mapping**: `InfrastructureState`'s 28 channels
//! were designed around the heuristic backend. Where a channel has a direct
//! physical referent (storage_ratio, inbound/outbound_power_kw,
//! coolant_temp_c, voltage_stability, brownout_risk, unserved_demand_ratio,
//! thermal_runaway_risk, islanding_capability/risk) this backend computes it
//! from real physics. Channels with no natural single-node electrical
//! analog (queue_depth, maintenance_backlog, switchgear_wear, deadlock_risk)
//! remain simple, clearly-labeled heuristics tied to routing effort — same
//! honest scope-narrowing as e.g. `symthaea-auv`'s "real drag/buoyancy, not
//! full Fossen" note.
//!
//! **Topology**: a single 2-node radial feeder (substation tie -> local
//! microgrid bus) carrying the battery, cooling/heating loads, and a
//! synthetic (deterministic, non-random) community demand profile. Real
//! power only — no reactive-power/Q modeling, so `symthaea-grid-physics`'s
//! `VoltageDroop`/reactive-power machinery is intentionally unused here;
//! islanded-mode voltage droop is instead approximated by a simple linear
//! inverter-output-impedance term keyed to real power, documented inline.

use symthaea_grid_physics::battery::Battery;
use symthaea_grid_physics::droop::FrequencyDroop;
use symthaea_grid_physics::feeder::{Feeder, Line, Node};
use symthaea_grid_physics::islanding::{
    PassiveIslandingDetector, RocofDetector, ThresholdDetector,
    steady_state_frequency_after_islanding,
};
use symthaea_grid_physics::trip_envelope::VoltageTripEnvelope;

use crate::simulator::InfrastructurePhysicsSimulator;
#[cfg(test)]
use crate::types::InfrastructureOperatingMode;
use crate::types::{
    BROWNOUT_RISK, COMMUNITY_DEMAND, COOLANT_TEMP_C, CRITICAL_LOAD_FRACTION, DEADLOCK_RISK,
    GRID_STRESS, ISLANDING_CAPABILITY, ISLANDING_RISK, InfrastructureCommand, InfrastructureState,
    QUEUE_DEPTH, RECOVERY_MARGIN, RELAY_HEALTH, SERVICE_INTEGRITY, SHED_LOAD_RATIO, STORAGE_RATIO,
    SWITCHGEAR_WEAR, THERMAL_LOAD, THERMAL_RUNAWAY_RISK, UNSERVED_DEMAND_RATIO, VOLTAGE_STABILITY,
};

// ── Illustrative, tunable plant parameters (documented, not derived) ──────

const BATTERY_CAPACITY_KWH: f64 = 500.0;
const BATTERY_POWER_RATING_KW: f64 = 250.0;
const BATTERY_ROUND_TRIP_EFFICIENCY: f64 = 0.90;

const LINE_RESISTANCE_OHM: f64 = 0.3;
const LINE_REACTANCE_OHM: f64 = 0.2;
const SUBSTATION_BASE_VOLTAGE_V: f64 = 7200.0;
/// Reference tie-line capacity used only to normalize `grid_stress`, not a
/// hard physical limit enforced elsewhere in this model.
const SUBSTATION_TIE_CAPACITY_KW: f64 = 1000.0;

const COOLING_RATED_KW: f64 = 20.0;
const HEATING_RATED_KW: f64 = 15.0;

/// Synthetic community demand profile: a smooth, deterministic (no RNG)
/// oscillation standing in for a diurnal load curve, compressed to a period
/// short enough for fast, reproducible tests.
const BASE_COMMUNITY_LOAD_KW: f64 = 150.0;
const LOAD_SWING_KW: f64 = 100.0;
const LOAD_PERIOD_S: f64 = 300.0;
const ROUTING_LOAD_SCALE_KW: f64 = 20.0;

const AMBIENT_TEMP_C: f64 = 22.0;
const MAX_SAFE_TEMP_C: f64 = 90.0;
/// Degrees C per kW of heat input, per second (lumped thermal-capacitance
/// stand-in — no explicit mass/specific-heat model).
const K_HEAT_TO_TEMP: f64 = 0.05;
/// Degrees C per second removed at full (1.0) cooling effort.
const K_COOL: f64 = 0.3;
/// Fractional decay per second toward ambient (passive heat loss).
const K_AMBIENT: f64 = 0.01;

/// Approximates inverter output-impedance voltage sag under real power flow
/// while islanded (a real effect — any real inverter's terminal voltage
/// sags somewhat under load — modeled here as a simple linear term rather
/// than a full per-unit-impedance derivation, since this model carries no
/// explicit reactive-power state).
const BATTERY_INTERNAL_VOLTAGE_DROOP_V_PER_KW: f64 = 0.5;

const NOMINAL_FREQUENCY_HZ: f64 = 60.0;
const FREQ_DROOP_HZ_PER_KW: f64 = 0.002;
const MAX_VOLTAGE_DEVIATION_PU_FOR_ZERO_STABILITY: f64 = 0.15;
const FREQ_DEVIATION_HZ_FOR_MAX_ISLANDING_RISK: f64 = 1.0;
const ROCOF_THRESHOLD_HZ_PER_S: f64 = 0.5;
const FREQ_THRESHOLD_MIN_HZ: f64 = 59.3;
const FREQ_THRESHOLD_MAX_HZ: f64 = 60.5;
const VOLTAGE_THRESHOLD_MIN_PU: f64 = 0.88;
const VOLTAGE_THRESHOLD_MAX_PU: f64 = 1.10;

/// Routing effort below this (all four directions combined) is treated as
/// "external ties open" -- islanded. Matches the SafeIsland fallback's own
/// convention in `embodiment.rs` (open routing = island the node).
const ISLANDING_ROUTING_THRESHOLD: f64 = 0.05;

const BASE_RELAY_DECAY_PER_S: f64 = 0.0005;
const RELAY_DECAY_PER_ABNORMAL_VOLTAGE_S: f64 = 0.002;
const RELAY_DECAY_PER_THERMAL_RISK_S: f64 = 0.001;

pub struct GridPhysicsInfrastructureSimulator {
    state: InfrastructureState,
    battery: Battery,
    trip_envelope: VoltageTripEnvelope,
    islanding_detector: PassiveIslandingDetector,
    freq_droop: FrequencyDroop,
    abnormal_voltage_elapsed_s: f64,
    prev_frequency_hz: f64,
    elapsed_s: f64,
}

impl GridPhysicsInfrastructureSimulator {
    pub fn new() -> Self {
        Self {
            state: InfrastructureState::home(),
            battery: Battery::new(
                BATTERY_CAPACITY_KWH,
                BATTERY_POWER_RATING_KW,
                BATTERY_ROUND_TRIP_EFFICIENCY,
            )
            .with_soc(0.75),
            trip_envelope: VoltageTripEnvelope::illustrative_default(),
            islanding_detector: PassiveIslandingDetector {
                rocof: RocofDetector {
                    threshold_hz_per_s: ROCOF_THRESHOLD_HZ_PER_S,
                },
                threshold: ThresholdDetector {
                    min_frequency_hz: FREQ_THRESHOLD_MIN_HZ,
                    max_frequency_hz: FREQ_THRESHOLD_MAX_HZ,
                    min_voltage_pu: VOLTAGE_THRESHOLD_MIN_PU,
                    max_voltage_pu: VOLTAGE_THRESHOLD_MAX_PU,
                },
            },
            freq_droop: FrequencyDroop {
                nominal_frequency_hz: NOMINAL_FREQUENCY_HZ,
                nominal_power_kw: 0.0,
                droop_hz_per_kw: FREQ_DROOP_HZ_PER_KW,
            },
            abnormal_voltage_elapsed_s: 0.0,
            prev_frequency_hz: NOMINAL_FREQUENCY_HZ,
            elapsed_s: 0.0,
        }
    }

    /// Direct access to the underlying battery, for tests/benchmarks that
    /// need to inspect or force a specific state of charge.
    pub fn battery(&self) -> &Battery {
        &self.battery
    }

    pub fn battery_mut(&mut self) -> &mut Battery {
        &mut self.battery
    }
}

impl Default for GridPhysicsInfrastructureSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl InfrastructurePhysicsSimulator for GridPhysicsInfrastructureSimulator {
    fn step(&mut self, cmd: &InfrastructureCommand, dt: f64) {
        self.elapsed_s += dt;
        let dt_hours = dt / 3600.0;

        let charge_frac = cmd.charge_bus().clamp(0.0, 1.0) as f64;
        let discharge_frac = cmd.discharge_bus().clamp(0.0, 1.0) as f64;
        let cooling_frac = cmd.cooling_loop().clamp(0.0, 1.0) as f64;
        let heating_frac = cmd.heating_loop().clamp(0.0, 1.0) as f64;
        let north = cmd.torques[4].abs() as f64;
        let south = cmd.torques[5].abs() as f64;
        let east = cmd.torques[6].abs() as f64;
        let west = cmd.torques[7].abs() as f64;
        let routing_total = north + south + east + west;
        let is_islanded = routing_total < ISLANDING_ROUTING_THRESHOLD;

        // ── Battery ───────────────────────────────────────────────────
        let charge_power_kw = charge_frac * self.battery.power_rating_kw;
        let discharge_power_kw = discharge_frac * self.battery.power_rating_kw;
        let charge_accepted_dc_kwh = self
            .battery
            .charge(charge_power_kw, dt_hours)
            .unwrap_or(0.0);
        let discharge_delivered_ac_kwh = self
            .battery
            .discharge(discharge_power_kw, dt_hours)
            .unwrap_or(0.0);
        let one_way_eff = self.battery.round_trip_efficiency.sqrt();
        let actual_ac_kw_for_charge = if dt_hours > 0.0 && one_way_eff > 0.0 {
            (charge_accepted_dc_kwh / one_way_eff) / dt_hours
        } else {
            0.0
        };
        let discharge_delivered_ac_kw = if dt_hours > 0.0 {
            discharge_delivered_ac_kwh / dt_hours
        } else {
            0.0
        };
        let net_battery_injection_kw = discharge_delivered_ac_kw - actual_ac_kw_for_charge;

        // ── Loads ─────────────────────────────────────────────────────
        let community_load_kw = BASE_COMMUNITY_LOAD_KW
            + LOAD_SWING_KW
                * (0.5 + 0.5 * (2.0 * std::f64::consts::PI * self.elapsed_s / LOAD_PERIOD_S).sin())
            + ROUTING_LOAD_SCALE_KW * routing_total;
        let cooling_load_kw = cooling_frac * COOLING_RATED_KW;
        let heating_load_kw = heating_frac * HEATING_RATED_KW;
        let node_load_kw =
            community_load_kw + cooling_load_kw + heating_load_kw - net_battery_injection_kw;

        // ── Feeder solve ──────────────────────────────────────────────
        // While islanded, the "reference" the local bus is measured against
        // is the battery inverter's own droop-sagged terminal voltage
        // (approximating output-impedance sag under load), not the stiff
        // substation voltage -- there is no substation connection anymore.
        let root_voltage_v = if is_islanded {
            SUBSTATION_BASE_VOLTAGE_V
                - BATTERY_INTERNAL_VOLTAGE_DROOP_V_PER_KW * net_battery_injection_kw
        } else {
            SUBSTATION_BASE_VOLTAGE_V
        };
        let feeder = Feeder::new(
            root_voltage_v,
            vec![
                Node::root(),
                Node::load(
                    0,
                    Line {
                        resistance_ohm: LINE_RESISTANCE_OHM,
                        reactance_ohm: LINE_REACTANCE_OHM,
                    },
                    node_load_kw,
                    0.0,
                ),
            ],
        )
        .expect("2-node feeder topology is always valid by construction");
        let solution = feeder.solve();
        let voltage_pu = solution.voltage_pu(1);
        let branch_loss_kw = solution.estimate_branch_loss_kw(&feeder, 1);

        // ── Thermal ───────────────────────────────────────────────────
        let heat_gain_kw = branch_loss_kw + heating_load_kw;
        let coolant_temp_c = self.state.channels[COOLANT_TEMP_C];
        let coolant_temp_c = (coolant_temp_c
            + dt * (K_HEAT_TO_TEMP * heat_gain_kw - K_COOL * cooling_frac
                + K_AMBIENT * (AMBIENT_TEMP_C - coolant_temp_c)))
            .clamp(AMBIENT_TEMP_C - 10.0, 200.0);
        let thermal_runaway_risk = ((coolant_temp_c - AMBIENT_TEMP_C)
            / (MAX_SAFE_TEMP_C - AMBIENT_TEMP_C))
            .clamp(0.0, 1.0);
        let thermal_load = thermal_runaway_risk; // same normalized quantity, kept as a distinct channel for API parity with the heuristic backend

        // ── Voltage / trip envelope ───────────────────────────────────
        if self.trip_envelope.is_continuous_operation(voltage_pu) {
            self.abnormal_voltage_elapsed_s = 0.0;
        } else {
            self.abnormal_voltage_elapsed_s += dt;
        }
        let brownout_risk = match self.trip_envelope.max_ride_through_seconds(voltage_pu) {
            Some(max_rt) if max_rt > 0.0 => {
                (self.abnormal_voltage_elapsed_s / max_rt).clamp(0.0, 1.0)
            }
            Some(_) => 1.0,
            None => 0.0,
        };
        let voltage_stability = (1.0
            - (voltage_pu - 1.0).abs() / MAX_VOLTAGE_DEVIATION_PU_FOR_ZERO_STABILITY)
            .clamp(0.0, 1.0);

        // ── Islanding / frequency ─────────────────────────────────────
        let current_frequency_hz = if is_islanded {
            steady_state_frequency_after_islanding(&self.freq_droop, community_load_kw)
        } else {
            NOMINAL_FREQUENCY_HZ
        };
        let _detected = self.islanding_detector.detect(
            self.prev_frequency_hz,
            current_frequency_hz,
            dt.max(f64::EPSILON),
            voltage_pu,
        );
        self.prev_frequency_hz = current_frequency_hz;
        let islanding_risk = if is_islanded {
            let freq_dev = (current_frequency_hz - NOMINAL_FREQUENCY_HZ).abs();
            (0.5 + freq_dev / FREQ_DEVIATION_HZ_FOR_MAX_ISLANDING_RISK).clamp(0.0, 1.0)
        } else {
            (1.0 - self.battery.soc()) * 0.2
        };
        let islanding_capability = self.battery.soc();

        // ── Unserved demand (only meaningful while islanded: grid-tied
        // assumes an infinite-bus substation that always meets local load) ──
        let unserved_demand_ratio = if is_islanded {
            let community_demand_ac_kwh = community_load_kw * dt_hours;
            let shortfall = (community_demand_ac_kwh - discharge_delivered_ac_kwh).max(0.0);
            if community_demand_ac_kwh > 0.0 {
                (shortfall / community_demand_ac_kwh).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };
        let shed_load_ratio = unserved_demand_ratio;

        // ── grid_stress: voltage deviation + tie loading, both real ────
        let grid_stress = if is_islanded {
            islanding_risk
        } else {
            ((voltage_pu - 1.0).abs() / 0.1
                + solution.branch_power_kw[1].abs() / SUBSTATION_TIE_CAPACITY_KW)
                .clamp(0.0, 1.0)
        };

        // ── Remaining channels without a clean single-node electrical
        // analog: simple, clearly-labeled heuristics tied to routing
        // effort and accumulated real-risk channels, same spirit (not the
        // same formulas) as the default heuristic backend ──
        let prev_relay_health = self.state.channels[RELAY_HEALTH];
        let relay_health = (prev_relay_health
            - dt * (BASE_RELAY_DECAY_PER_S
                + RELAY_DECAY_PER_ABNORMAL_VOLTAGE_S
                    * if self.trip_envelope.is_continuous_operation(voltage_pu) {
                        0.0
                    } else {
                        1.0
                    }
                + RELAY_DECAY_PER_THERMAL_RISK_S * thermal_runaway_risk))
            .clamp(0.0, 1.0);
        let queue_depth = (self.state.channels[QUEUE_DEPTH] * 0.9
            + routing_total.min(4.0) / 4.0 * 0.1)
            .clamp(0.0, 1.0);
        let switchgear_wear =
            (self.state.channels[SWITCHGEAR_WEAR] + routing_total * dt * 0.006).clamp(0.0, 1.0);
        // Divide by 2 (the max possible sum when both axes are fully
        // imbalanced, e.g. north=1,south=0,east=1,west=0) before clamping,
        // so full imbalance on both axes can actually reach 1.0 -- clamping
        // the raw sum first would cap this at 0.5 regardless of severity.
        let deadlock_risk = (((north - south).abs() + (east - west).abs()) / 2.0).clamp(0.0, 1.0);
        let maintenance_backlog =
            (self.state.channels[14] + (1.0 - relay_health) * dt * 0.02).clamp(0.0, 1.0);

        let service_integrity = (1.0
            - unserved_demand_ratio * 0.4
            - brownout_risk * 0.25
            - islanding_risk * 0.15
            - deadlock_risk * 0.1)
            .clamp(0.0, 1.0);
        let recovery_margin =
            (self.battery.soc() * 0.4 + voltage_stability * 0.3 + (1.0 - thermal_load) * 0.3)
                .clamp(0.0, 1.0);
        let community_demand_normalized =
            (community_load_kw / (BASE_COMMUNITY_LOAD_KW + LOAD_SWING_KW)).clamp(0.0, 1.0);

        // ── Write back ──────────────────────────────────────────────
        let s = &mut self.state.channels;
        s[STORAGE_RATIO] = self.battery.soc();
        s[THERMAL_LOAD] = thermal_load;
        s[2] = if is_islanded {
            0.0
        } else {
            solution.branch_power_kw[1].max(0.0)
        };
        s[3] = if is_islanded {
            0.0
        } else {
            (-solution.branch_power_kw[1]).max(0.0)
        };
        s[QUEUE_DEPTH] = queue_depth;
        s[RELAY_HEALTH] = relay_health;
        s[VOLTAGE_STABILITY] = voltage_stability;
        s[COOLANT_TEMP_C] = coolant_temp_c;
        s[8] = heating_frac;
        s[9] = (s[9] + dt * 0.0005).clamp(0.0, 1.0);
        s[10] = north.clamp(0.0, 1.0);
        s[11] = south.clamp(0.0, 1.0);
        s[12] = east.clamp(0.0, 1.0);
        s[13] = west.clamp(0.0, 1.0);
        s[14] = maintenance_backlog;
        s[ISLANDING_CAPABILITY] = islanding_capability;
        s[SWITCHGEAR_WEAR] = switchgear_wear;
        s[GRID_STRESS] = grid_stress;
        s[COMMUNITY_DEMAND] = community_demand_normalized;
        s[BROWNOUT_RISK] = brownout_risk;
        s[SHED_LOAD_RATIO] = shed_load_ratio;
        s[CRITICAL_LOAD_FRACTION] = 0.35;
        s[UNSERVED_DEMAND_RATIO] = unserved_demand_ratio;
        s[DEADLOCK_RISK] = deadlock_risk;
        s[ISLANDING_RISK] = islanding_risk;
        s[SERVICE_INTEGRITY] = service_integrity;
        s[RECOVERY_MARGIN] = recovery_margin;
        s[THERMAL_RUNAWAY_RISK] = thermal_runaway_risk;
    }

    fn state(&self) -> &InfrastructureState {
        &self.state
    }

    fn reset(&mut self) {
        self.state = InfrastructureState::home();
        self.battery = Battery::new(
            BATTERY_CAPACITY_KWH,
            BATTERY_POWER_RATING_KW,
            BATTERY_ROUND_TRIP_EFFICIENCY,
        )
        .with_soc(0.75);
        self.abnormal_voltage_elapsed_s = 0.0;
        self.prev_frequency_hz = NOMINAL_FREQUENCY_HZ;
        self.elapsed_s = 0.0;
    }

    fn backend_name(&self) -> &'static str {
        "grid_physics"
    }
}

/// Failure-mode benchmarks, one per target named in
/// `docs/robotics/INFRASTRUCTURE_HAZARDS_AND_SENSORIUM.md`: overload -> load
/// shedding, cooling failure -> thermal runaway, routing contention ->
/// deadlock, islanding -> degraded-but-preserved service. Deterministic
/// (fixed dt, no RNG) and replayable — same scenario re-run produces
/// identical channel trajectories, verified below.
#[cfg(test)]
mod failure_mode_tests {
    use super::*;

    #[test]
    fn test_step_runs_and_state_stays_finite() {
        let mut sim = GridPhysicsInfrastructureSimulator::new();
        let cmd = InfrastructureCommand::zero();
        for _ in 0..100 {
            sim.step(&cmd, 1.0);
        }
        assert!(sim.state().is_finite());
    }

    /// Overload -> load shedding: islanded (routing near zero) with
    /// discharge commanded far below what the synthetic community load
    /// needs. The battery simply can't deliver enough power at the
    /// commanded (low) discharge fraction to cover demand, so real,
    /// commanded-power-limited unserved demand accumulates.
    #[test]
    fn test_islanded_underprovisioned_discharge_raises_unserved_demand() {
        let mut sim = GridPhysicsInfrastructureSimulator::new();
        let mut cmd = InfrastructureCommand::zero();
        cmd.torques[1] = 0.05; // discharge far below the ~250kW peak community load
        for _ in 0..50 {
            sim.step(&cmd, 1.0);
        }
        let state = sim.state();
        assert!(
            state.unserved_demand_ratio() > 0.3,
            "expected significant unserved demand, got {}",
            state.unserved_demand_ratio()
        );
        assert!(state.shed_load_ratio() > 0.3);
        // Some escalated posture should result -- which named mode wins
        // depends on which risk channel crosses its threshold first, all
        // are legitimate responses to sustained under-service while islanded.
        assert!(matches!(
            state.inferred_mode(),
            InfrastructureOperatingMode::LoadShedding
                | InfrastructureOperatingMode::Islanding
                | InfrastructureOperatingMode::Emergency
        ));
    }

    /// Cooling failure -> thermal runaway: sustained heating with cooling
    /// disabled, grid-tied so islanding doesn't interfere with the read.
    #[test]
    fn test_cooling_failure_raises_thermal_runaway_risk() {
        let mut sim = GridPhysicsInfrastructureSimulator::new();
        let mut cmd = InfrastructureCommand::zero();
        cmd.torques[3] = 1.0; // heating_loop = full
        cmd.torques[2] = 0.0; // cooling_loop = off (the failure)
        cmd.torques[4] = 1.0; // stay grid-tied (routing above islanding threshold)
        for _ in 0..600 {
            sim.step(&cmd, 1.0);
        }
        let state = sim.state();
        assert!(
            state.thermal_runaway_risk() > 0.5,
            "expected elevated thermal runaway risk, got {}",
            state.thermal_runaway_risk()
        );
        assert!(state.channels[COOLANT_TEMP_C] > 50.0);
    }

    /// Routing contention -> deadlock: strongly imbalanced routing on both
    /// axes (all traffic pushed north and east, none south/west).
    #[test]
    fn test_routing_imbalance_raises_deadlock_risk() {
        let mut sim = GridPhysicsInfrastructureSimulator::new();
        let mut cmd = InfrastructureCommand::zero();
        cmd.torques[4] = 1.0; // north
        cmd.torques[5] = 0.0; // south
        cmd.torques[6] = 1.0; // east
        cmd.torques[7] = 0.0; // west
        sim.step(&cmd, 1.0);
        let state = sim.state();
        assert!(
            state.deadlock_risk() >= 0.6,
            "expected deadlock_risk to cross the DeadlockRecovery threshold, got {}",
            state.deadlock_risk()
        );
        assert_eq!(
            state.inferred_mode(),
            InfrastructureOperatingMode::DeadlockRecovery
        );
    }

    /// Islanding -> degraded but preserved service: routing near zero
    /// (islanded), discharge commanded high enough to actually cover the
    /// community load. Islanding risk should rise (it's a real posture
    /// change), but service should NOT collapse -- this is what "escalate
    /// operational posture" rather than "abort" means for infrastructure.
    #[test]
    fn test_islanding_preserves_service_when_discharge_covers_load() {
        let mut sim = GridPhysicsInfrastructureSimulator::new();
        let mut cmd = InfrastructureCommand::zero();
        cmd.torques[1] = 1.0; // full discharge -- battery is rated to cover peak community load
        for _ in 0..30 {
            sim.step(&cmd, 1.0);
        }
        let state = sim.state();
        assert!(
            state.islanding_risk() > 0.5,
            "islanding should show elevated (but not necessarily catastrophic) risk, got {}",
            state.islanding_risk()
        );
        assert!(
            state.service_integrity() > 0.5,
            "service should be degraded, not collapsed, got {}",
            state.service_integrity()
        );
        assert!(
            state.unserved_demand_ratio() < 0.3,
            "adequately provisioned discharge should keep unserved demand low, got {}",
            state.unserved_demand_ratio()
        );
    }

    /// Determinism: identical command sequence from a fresh simulator
    /// produces identical channel trajectories. No RNG anywhere in this
    /// backend, so this must hold exactly.
    #[test]
    fn test_deterministic_across_fresh_sims() {
        let mut cmd = InfrastructureCommand::zero();
        cmd.torques[0] = 0.3;
        cmd.torques[1] = 0.2;
        cmd.torques[2] = 0.5;
        cmd.torques[4] = 0.8;
        cmd.torques[6] = 0.4;

        let run = || {
            let mut sim = GridPhysicsInfrastructureSimulator::new();
            for _ in 0..40 {
                sim.step(&cmd, 0.5);
            }
            sim.state().channels
        };
        let first = run();
        let second = run();
        assert_eq!(first, second);
    }

    /// A rule-based scripted baseline the advisor must beat, per the plan's
    /// Phase 2 requirement ("a scripted rule-based dispatch baseline the
    /// advisor must beat"). This is the baseline itself, not the advisor --
    /// it exists so future advisor work has something concrete to compare
    /// against. Policy: discharge proportionally to (1 - storage_ratio)
    /// need is irrelevant here; simplest sane baseline is "always discharge
    /// enough to fully cover peak load, never charge unless above 90% SoC".
    fn scripted_baseline_command(
        sim: &GridPhysicsInfrastructureSimulator,
    ) -> InfrastructureCommand {
        let mut cmd = InfrastructureCommand::zero();
        if sim.battery().soc() > 0.2 {
            cmd.torques[1] = 1.0; // discharge at max to cover load
        }
        if sim.battery().soc() < 0.9 {
            cmd.torques[0] = 0.3; // modest trickle charge otherwise
        }
        cmd
    }

    #[test]
    fn test_scripted_baseline_keeps_service_integrity_reasonable_when_islanded() {
        let mut sim = GridPhysicsInfrastructureSimulator::new();
        for _ in 0..60 {
            let cmd = scripted_baseline_command(&sim);
            sim.step(&cmd, 1.0);
        }
        assert!(
            sim.state().service_integrity() > 0.4,
            "scripted baseline should keep service reasonably intact, got {}",
            sim.state().service_integrity()
        );
    }
}
