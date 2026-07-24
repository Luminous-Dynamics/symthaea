// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-pass subterranean scout simulator.
//!
//! All dynamics are expressed as rates or continuous-time relaxation terms.
//! Changing `physics_hz` therefore changes numerical resolution rather than
//! silently changing the physical world.

use crate::geology::{GeologicalLookahead, GeologySample, GeotechnicalProfile};
use crate::path_memory::{ReturnPathAssessment, ReturnPathMemory};
use crate::types::{
    ABORT_RECOMMENDATION, AQUIFER_RISK, BATTERY_RATIO, COMM_SIGNAL, CUTTER_TEMP_C, DEPTH_M,
    ESCAPE_CONFIDENCE, FORWARD_VELOCITY_MPS, GAS_RISK, HULL_STRESS, HUMIDITY,
    LOCALIZATION_CONFIDENCE, MAPPING_CONFIDENCE, MISSION_PROGRESS, MOTOR_TEMP_C,
    OBSTACLE_PROXIMITY, PITCH_RAD, RELAY_DISTANCE_NORM, RELAY_LINK_QUALITY, RETURN_PATH_CONFIDENCE,
    ROLL_RAD, ROOF_STABILITY, SEAL_INTEGRITY, SLIP_RATIO, SLURRY_LOAD, SOIL_DENSITY,
    SPOIL_BUFFER_FILL, SubterraneanCommand, SubterraneanState, THERMAL_MARGIN, TOOL_WEAR,
    VEIN_SIGNAL, VIBRATION_LEVEL, WATER_INGRESS_RATIO,
};

const AMBIENT_TEMP_C: f64 = 20.0;

fn approach(current: f64, target: f64, tau_seconds: f64, dt: f64) -> f64 {
    let tau = tau_seconds.max(1e-6);
    let alpha = 1.0 - (-dt / tau).exp();
    current + (target - current) * alpha
}

fn bounded_integrate(current: f64, rate_per_second: f64, dt: f64, min: f64, max: f64) -> f64 {
    (current + rate_per_second * dt).clamp(min, max)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecoveryResources {
    pub sealant_ratio: f64,
    pub relay_units: u8,
    pub roof_support_units: u8,
    pub dewatering_health: f64,
}

impl RecoveryResources {
    pub const fn full() -> Self {
        Self {
            sealant_ratio: 1.0,
            relay_units: 3,
            roof_support_units: 3,
            dewatering_health: 1.0,
        }
    }
}

pub trait SubterraneanPhysicsSimulator {
    fn step(&mut self, cmd: &SubterraneanCommand, dt: f64);
    fn state(&self) -> &SubterraneanState;
    fn reset(&mut self);
}

/// A deterministic boring-scout model with depth, spoil, heat, battery,
/// communication, and enclosed-environment hazards.
#[derive(Clone)]
pub struct SimpleSubterraneanSimulator {
    state: SubterraneanState,
    geology: GeotechnicalProfile,
    path_memory: ReturnPathMemory,
    recovery_resources: RecoveryResources,
    relay_effect: f64,
    roof_support_effect: f64,
    relay_command_active: bool,
    roof_support_command_active: bool,
}

impl SimpleSubterraneanSimulator {
    pub fn new() -> Self {
        Self::with_geology(GeotechnicalProfile::default())
    }

    pub fn with_geology(geology: GeotechnicalProfile) -> Self {
        Self {
            state: SubterraneanState::home(),
            geology,
            path_memory: ReturnPathMemory::default(),
            recovery_resources: RecoveryResources::full(),
            relay_effect: 0.0,
            roof_support_effect: 0.0,
            relay_command_active: false,
            roof_support_command_active: false,
        }
    }

    pub fn state_mut(&mut self) -> &mut SubterraneanState {
        &mut self.state
    }

    pub fn recovery_resources(&self) -> RecoveryResources {
        self.recovery_resources
    }

    pub fn geology(&self) -> &GeotechnicalProfile {
        &self.geology
    }

    pub fn geology_sample(&self) -> GeologySample {
        self.geology.sample(self.state.depth_m())
    }

    pub fn geology_lookahead(&self, horizon_m: f64) -> GeologicalLookahead {
        self.geology.lookahead(self.state.depth_m(), horizon_m)
    }

    pub fn return_path_memory(&self) -> &ReturnPathMemory {
        &self.path_memory
    }

    pub fn return_path_assessment(&self) -> ReturnPathAssessment {
        self.path_memory.assess(&self.state)
    }
}

impl SubterraneanPhysicsSimulator for SimpleSubterraneanSimulator {
    fn step(&mut self, cmd: &SubterraneanCommand, dt: f64) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let dt = dt.min(1.0);
        self.state.sanitize_fail_closed();
        let state_before = self.state.clone();
        let mut sanitized_command = *cmd;
        sanitized_command.sanitize();
        let cmd = &sanitized_command;
        let geology = self.geology.sample(self.state.depth_m());
        // Surface idle must not inherit the full unsupported-roof, aquifer,
        // and gas burden of the first underground stratum. Geotechnical
        // hazards become physically coupled as the machine enters confinement.
        let geological_exposure = (self.state.depth_m() / 2.0).clamp(0.0, 1.0);

        let traction = ((cmd.left_track() + cmd.right_track()) as f64 * 0.5).clamp(-1.0, 1.0);
        let reverse_bias = (-traction).max(0.0);
        let boring_cmd = cmd.cutter_head().max(0.0) as f64;
        let auger = cmd.auger_feed().max(0.0) as f64;
        let cooling = cmd.thermal_pump().max(0.0) as f64;
        let ballast = cmd.ballast_trim().clamp(-1.0, 1.0) as f64;
        let dewatering = cmd.recovery.dewatering_pump.clamp(0.0, 1.0) as f64
            * self.recovery_resources.dewatering_health;
        let requested_sealant = cmd.recovery.sealant_injector.clamp(0.0, 1.0) as f64;
        let sealant = requested_sealant.min(self.recovery_resources.sealant_ratio / dt.max(1e-6));
        let relay_active = cmd.recovery.relay_deployer >= 0.5;
        if relay_active && !self.relay_command_active && self.recovery_resources.relay_units > 0 {
            self.recovery_resources.relay_units -= 1;
            self.relay_effect = 1.0;
        }
        self.relay_command_active = relay_active;
        let support_active = cmd.recovery.roof_support >= 0.5;
        if support_active
            && !self.roof_support_command_active
            && self.recovery_resources.roof_support_units > 0
        {
            self.recovery_resources.roof_support_units -= 1;
            self.roof_support_effect = 1.0;
        }
        self.roof_support_command_active = support_active;
        self.relay_effect = approach(self.relay_effect, 0.0, 60.0, dt).clamp(0.0, 1.0);
        self.roof_support_effect =
            approach(self.roof_support_effect, 0.0, 45.0, dt).clamp(0.0, 1.0);
        self.recovery_resources.sealant_ratio = bounded_integrate(
            self.recovery_resources.sealant_ratio,
            -sealant * 0.08,
            dt,
            0.0,
            1.0,
        );

        let hazard_pressure = (self.state.channels[WATER_INGRESS_RATIO] * 0.28
            + self.state.channels[AQUIFER_RISK] * 0.18
            + self.state.channels[GAS_RISK] * 0.16
            + (1.0 - self.state.channels[ROOF_STABILITY]) * 0.18
            + self.state.channels[SPOIL_BUFFER_FILL] * 0.08
            + self.state.channels[SLIP_RATIO] * 0.07
            + self.state.channels[HULL_STRESS] * 0.05)
            .clamp(0.0, 1.0);
        let penetration_factor = (1.0 - geology.hardness * 0.68).clamp(0.18, 1.0);
        let effective_boring =
            boring_cmd * (1.0 - hazard_pressure).clamp(0.0, 1.0) * penetration_factor;
        let effective_auger = (auger + reverse_bias * 0.2).clamp(0.0, 1.0);

        self.state.channels[DEPTH_M] = bounded_integrate(
            self.state.channels[DEPTH_M],
            effective_boring * 0.45 - reverse_bias * 0.18,
            dt,
            0.0,
            200.0,
        );
        let velocity_target = traction * (1.0 - self.state.channels[SLIP_RATIO]).max(0.05);
        self.state.channels[FORWARD_VELOCITY_MPS] = approach(
            self.state.channels[FORWARD_VELOCITY_MPS],
            velocity_target,
            0.12,
            dt,
        )
        .clamp(-2.0, 2.0);
        self.state.channels[PITCH_RAD] =
            bounded_integrate(self.state.channels[PITCH_RAD], ballast * 0.2, dt, -0.6, 0.6);
        self.state.channels[ROLL_RAD] = bounded_integrate(
            self.state.channels[ROLL_RAD],
            traction * 0.1 - ballast * 0.05,
            dt,
            -0.5,
            0.5,
        );

        let cutter_cooling = (self.state.channels[CUTTER_TEMP_C] - AMBIENT_TEMP_C).max(0.0) * 0.08;
        self.state.channels[CUTTER_TEMP_C] = bounded_integrate(
            self.state.channels[CUTTER_TEMP_C],
            boring_cmd * (8.0 + geology.hardness * 20.0) - cooling * 10.0 - cutter_cooling,
            dt,
            AMBIENT_TEMP_C,
            180.0,
        );
        let motor_cooling = (self.state.channels[MOTOR_TEMP_C] - AMBIENT_TEMP_C).max(0.0) * 0.06;
        self.state.channels[MOTOR_TEMP_C] = bounded_integrate(
            self.state.channels[MOTOR_TEMP_C],
            (boring_cmd * (0.55 + geology.hardness * 0.65) + traction.abs()) * 8.0
                - cooling * 5.0
                - motor_cooling,
            dt,
            AMBIENT_TEMP_C,
            160.0,
        );
        self.state.channels[SPOIL_BUFFER_FILL] = bounded_integrate(
            self.state.channels[SPOIL_BUFFER_FILL],
            effective_boring * 0.35 - effective_auger * 0.45,
            dt,
            0.0,
            1.0,
        );
        self.state.channels[BATTERY_RATIO] = bounded_integrate(
            self.state.channels[BATTERY_RATIO],
            -(effective_boring * 0.012
                + effective_auger * 0.004
                + traction.abs() * 0.006
                + cooling * 0.004
                + dewatering * 0.009
                + sealant * 0.003
                + if relay_active { 0.001 } else { 0.0 }
                + if support_active { 0.0015 } else { 0.0 }),
            dt,
            0.0,
            1.0,
        );

        let comm_target = (1.0
            - self.state.channels[DEPTH_M] / 120.0
            - self.state.channels[WATER_INGRESS_RATIO] * 0.25)
            .clamp(0.0, 1.0);
        self.state.channels[COMM_SIGNAL] =
            approach(self.state.channels[COMM_SIGNAL], comm_target, 0.15, dt).clamp(0.0, 1.0);
        let slip_target = (0.15
            + self.state.channels[SPOIL_BUFFER_FILL] * 0.35
            + self.state.channels[SLURRY_LOAD] * 0.22
            + self.state.channels[OBSTACLE_PROXIMITY] * 0.25
            - traction.abs() * 0.05)
            .clamp(0.0, 1.0);
        self.state.channels[SLIP_RATIO] =
            approach(self.state.channels[SLIP_RATIO], slip_target, 0.25, dt).clamp(0.0, 1.0);
        self.state.channels[SOIL_DENSITY] = approach(
            self.state.channels[SOIL_DENSITY],
            (0.18 + geology.hardness * 1.72).clamp(0.0, 2.0),
            0.8,
            dt,
        );
        self.state.channels[VIBRATION_LEVEL] = approach(
            self.state.channels[VIBRATION_LEVEL],
            (boring_cmd * (0.2 + geology.hardness * 0.8)).clamp(0.0, 1.0),
            0.08,
            dt,
        )
        .clamp(0.0, 1.0);
        let humidity_target = (0.3
            + self.state.channels[DEPTH_M] / 300.0
            + self.state.channels[WATER_INGRESS_RATIO] * 0.25)
            .clamp(0.0, 1.0);
        self.state.channels[HUMIDITY] =
            approach(self.state.channels[HUMIDITY], humidity_target, 3.0, dt).clamp(0.0, 1.0);

        let obstacle_target = (geology.hardness * 0.35
            + (1.0 - geology.survey_confidence) * 0.25
            + self.state.channels[SPOIL_BUFFER_FILL] * 0.3)
            .clamp(0.0, 1.0);
        self.state.channels[OBSTACLE_PROXIMITY] = approach(
            self.state.channels[OBSTACLE_PROXIMITY],
            obstacle_target,
            1.5,
            dt,
        )
        .clamp(0.0, 1.0);
        let mapping_target = (0.45 + (1.0 - self.state.channels[OBSTACLE_PROXIMITY]) * 0.45
            - self.state.channels[SLIP_RATIO] * 0.2)
            .clamp(0.0, 1.0);
        self.state.channels[MAPPING_CONFIDENCE] = approach(
            self.state.channels[MAPPING_CONFIDENCE],
            mapping_target,
            1.2,
            dt,
        )
        .clamp(0.0, 1.0);
        self.state.channels[THERMAL_MARGIN] =
            (1.0 - self.state.channels[CUTTER_TEMP_C] / 180.0).clamp(0.0, 1.0);
        let vein_target = (geology.ore_grade * 0.82 + geology.survey_confidence * 0.08
            - self.state.channels[SLIP_RATIO] * 0.15)
            .clamp(0.0, 1.0);
        self.state.channels[VEIN_SIGNAL] =
            approach(self.state.channels[VEIN_SIGNAL], vein_target, 1.5, dt);
        self.state.channels[TOOL_WEAR] = bounded_integrate(
            self.state.channels[TOOL_WEAR],
            boring_cmd * (0.0005 + geology.abrasiveness * 0.006),
            dt,
            0.0,
            1.0,
        );

        let hull_target = (self.state.channels[SLIP_RATIO] * 0.4
            + self.state.channels[SPOIL_BUFFER_FILL] * 0.25
            + self.state.channels[SLURRY_LOAD] * 0.2
            + geology.hardness * boring_cmd * 0.18
            + (1.0 - geology.roof_cohesion) * 0.12 * geological_exposure)
            .clamp(0.0, 1.0);
        self.state.channels[HULL_STRESS] =
            approach(self.state.channels[HULL_STRESS], hull_target, 0.5, dt).clamp(0.0, 1.0);
        let return_target = (self.state.channels[COMM_SIGNAL] * 0.55
            + self.state.channels[MAPPING_CONFIDENCE] * 0.45
            - self.state.channels[SLIP_RATIO] * 0.2)
            .clamp(0.0, 1.0);
        self.state.channels[RETURN_PATH_CONFIDENCE] = approach(
            self.state.channels[RETURN_PATH_CONFIDENCE],
            return_target,
            1.5,
            dt,
        )
        .clamp(0.0, 1.0);

        let aquifer_target = (0.01
            + geological_exposure
                * (self.state.channels[HUMIDITY] * 0.18
                    + geology.permeability * 0.58
                    + (1.0 - geology.survey_confidence) * 0.08
                    + effective_boring * geology.permeability * 0.16))
            .clamp(0.0, 1.0);
        self.state.channels[AQUIFER_RISK] =
            approach(self.state.channels[AQUIFER_RISK], aquifer_target, 1.0, dt).clamp(0.0, 1.0);
        let water_rate = self.state.channels[AQUIFER_RISK] * effective_boring * 0.35
            + self.state.channels[HUMIDITY] * 0.003 * geological_exposure
            + (self.state.channels[AQUIFER_RISK] - 0.75).max(0.0) * 0.05
            - reverse_bias * 0.08
            - effective_auger * 0.03
            - dewatering * 0.38
            - self.state.channels[WATER_INGRESS_RATIO] * 0.12;
        self.state.channels[WATER_INGRESS_RATIO] = bounded_integrate(
            self.state.channels[WATER_INGRESS_RATIO],
            water_rate,
            dt,
            0.0,
            1.0,
        );
        let gas_target = (0.005
            + geological_exposure
                * (geology.gas_potential * 0.7
                    + (1.0 - geology.survey_confidence) * 0.08
                    + effective_boring * geology.gas_potential * 0.24)
            - reverse_bias * 0.08)
            .clamp(0.0, 1.0);
        self.state.channels[GAS_RISK] =
            approach(self.state.channels[GAS_RISK], gas_target, 2.0, dt).clamp(0.0, 1.0);
        let confined_roof_target = (0.18 + geology.roof_cohesion * 0.76
            - self.state.channels[OBSTACLE_PROXIMITY] * 0.18
            - self.state.channels[HULL_STRESS] * 0.2
            - self.state.channels[SLIP_RATIO] * 0.16
            - boring_cmd * (1.0 - geology.roof_cohesion) * 0.18
            + self.roof_support_effect * 0.35)
            .clamp(0.0, 1.0);
        let roof_target = 0.98 + (confined_roof_target - 0.98) * geological_exposure;
        self.state.channels[ROOF_STABILITY] =
            approach(self.state.channels[ROOF_STABILITY], roof_target, 0.35, dt).clamp(0.0, 1.0);
        let localization_target = (self.state.channels[MAPPING_CONFIDENCE] * 0.55
            + self.state.channels[COMM_SIGNAL] * 0.35
            + 0.1
            + self.relay_effect * 0.25
            - self.state.channels[SLIP_RATIO] * 0.25
            - reverse_bias * 0.03)
            .clamp(0.0, 1.0);
        self.state.channels[LOCALIZATION_CONFIDENCE] = approach(
            self.state.channels[LOCALIZATION_CONFIDENCE],
            localization_target,
            1.0,
            dt,
        )
        .clamp(0.0, 1.0);

        self.state.channels[RELAY_DISTANCE_NORM] =
            (self.state.channels[DEPTH_M] / 200.0).clamp(0.0, 1.0);
        let relay_target = (self.state.channels[COMM_SIGNAL]
            * (1.0 - self.state.channels[RELAY_DISTANCE_NORM] * 0.3)
            * (1.0 - self.state.channels[WATER_INGRESS_RATIO] * 0.35)
            + self.relay_effect * 0.65)
            .clamp(0.0, 1.0);
        self.state.channels[RELAY_LINK_QUALITY] = approach(
            self.state.channels[RELAY_LINK_QUALITY],
            relay_target,
            0.2,
            dt,
        )
        .clamp(0.0, 1.0);
        let seal_rate = -0.00005
            - self.state.channels[WATER_INGRESS_RATIO] * 0.05
            - self.state.channels[HULL_STRESS] * 0.02
            + sealant * 0.28;
        self.state.channels[SEAL_INTEGRITY] =
            bounded_integrate(self.state.channels[SEAL_INTEGRITY], seal_rate, dt, 0.0, 1.0);
        let slurry_target = (self.state.channels[WATER_INGRESS_RATIO] * 0.55
            + self.state.channels[SPOIL_BUFFER_FILL] * 0.45)
            .clamp(0.0, 1.0);
        let slurry_relaxed = approach(self.state.channels[SLURRY_LOAD], slurry_target, 0.6, dt);
        self.state.channels[SLURRY_LOAD] = bounded_integrate(
            slurry_relaxed,
            -effective_auger * 0.08 - reverse_bias * 0.05 - dewatering * 0.18,
            dt,
            0.0,
            1.0,
        );
        let escape_target = (1.0
            - self.state.channels[SLIP_RATIO] * 0.18
            - self.state.channels[SPOIL_BUFFER_FILL] * 0.16
            - self.state.channels[SLURRY_LOAD] * 0.18
            - self.state.channels[WATER_INGRESS_RATIO] * 0.2
            + self.state.channels[RETURN_PATH_CONFIDENCE] * 0.08
            + reverse_bias * 0.06)
            .clamp(0.0, 1.0);
        self.state.channels[ESCAPE_CONFIDENCE] = approach(
            self.state.channels[ESCAPE_CONFIDENCE],
            escape_target,
            0.45,
            dt,
        )
        .clamp(0.0, 1.0);
        let abort_target = (self.state.channels[WATER_INGRESS_RATIO] * 0.2
            + self.state.channels[AQUIFER_RISK] * 0.15
            + self.state.channels[GAS_RISK] * 0.16
            + (1.0 - self.state.channels[ROOF_STABILITY]) * 0.16
            + (1.0 - self.state.channels[ESCAPE_CONFIDENCE]) * 0.14
            + (1.0 - self.state.channels[SEAL_INTEGRITY]) * 0.08
            + self.state.channels[HULL_STRESS] * 0.06
            + (self.state.channels[CUTTER_TEMP_C] / 180.0) * 0.05)
            .clamp(0.0, 1.0);
        self.state.channels[ABORT_RECOMMENDATION] = approach(
            self.state.channels[ABORT_RECOMMENDATION],
            abort_target,
            0.2,
            dt,
        )
        .clamp(0.0, 1.0);
        self.state.channels[MISSION_PROGRESS] = bounded_integrate(
            self.state.channels[MISSION_PROGRESS],
            effective_boring * (1.0 - self.state.channels[ABORT_RECOMMENDATION]) * 0.01,
            dt,
            0.0,
            1.0,
        );

        self.path_memory.observe(&state_before, &self.state, cmd);
        let return_path = self.path_memory.assess(&self.state);
        self.state.channels[RETURN_PATH_CONFIDENCE] = return_path.path_confidence;
        // Route knowledge can reduce escape confidence, but must never mask
        // an immediate mobility loss such as a spoil/slurry jam.
        self.state.channels[ESCAPE_CONFIDENCE] *= 0.72 + return_path.path_confidence * 0.28;
        self.state.channels[ESCAPE_CONFIDENCE] =
            self.state.channels[ESCAPE_CONFIDENCE].clamp(0.0, 1.0);
    }

    fn state(&self) -> &SubterraneanState {
        &self.state
    }

    fn reset(&mut self) {
        self.state = SubterraneanState::home();
        self.path_memory.reset();
        self.recovery_resources = RecoveryResources::full();
        self.relay_effect = 0.0;
        self.roof_support_effect = 0.0;
        self.relay_command_active = false;
        self.roof_support_command_active = false;
    }
}

impl Default for SimpleSubterraneanSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SubterraneanOperatingMode;

    #[test]
    fn test_stable() {
        let mut sim = SimpleSubterraneanSimulator::new();
        for _ in 0..1000 {
            sim.step(&SubterraneanCommand::zero(), 0.005);
        }
        assert!(sim.state().is_finite());
    }

    #[test]
    fn malformed_command_is_sanitized_before_physics() {
        let mut sim = SimpleSubterraneanSimulator::new();
        let mut command = SubterraneanCommand::zero();
        command.torques[0] = f32::NAN;
        command.recovery.dewatering_pump = f32::INFINITY;
        sim.step(&command, 0.01);
        assert!(sim.state().is_finite());
    }

    #[test]
    fn test_torque_moves() {
        let mut sim = SimpleSubterraneanSimulator::new();
        let mut cmd = SubterraneanCommand::zero();
        cmd.set_cutter_head(1.0);
        let before = sim.state().channels[DEPTH_M];
        sim.step(&cmd, 0.01);
        assert_ne!(sim.state().channels[DEPTH_M], before);
    }

    #[test]
    fn test_boring_without_auger_fills_spoil_buffer_and_heats_cutter() {
        let mut sim = SimpleSubterraneanSimulator::new();
        let mut cmd = SubterraneanCommand::zero();
        cmd.set_cutter_head(1.0);
        for _ in 0..200 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().spoil_buffer_fill() > 0.5);
        assert!(sim.state().cutter_temp_c() > 20.0);
    }

    #[test]
    fn test_aquifer_breach_triggers_flood_response_mode() {
        let mut sim = SimpleSubterraneanSimulator::new();
        sim.state.channels[DEPTH_M] = 150.0;
        sim.state.channels[HUMIDITY] = 0.95;
        sim.state.channels[AQUIFER_RISK] = 0.9;
        sim.state.channels[WATER_INGRESS_RATIO] = 0.2;
        let mut cmd = SubterraneanCommand::zero();
        cmd.set_cutter_head(1.0);
        for _ in 0..80 {
            sim.step(&cmd, 0.02);
        }
        assert!(sim.state().water_ingress_ratio() > 0.25);
        assert!(matches!(
            sim.state().inferred_mode(),
            SubterraneanOperatingMode::FloodResponse
                | SubterraneanOperatingMode::Retreat
                | SubterraneanOperatingMode::Surface
        ));
    }

    #[test]
    fn test_gas_risk_suppresses_effective_digging() {
        let mut safe = SimpleSubterraneanSimulator::new();
        let mut hazardous = SimpleSubterraneanSimulator::new();
        hazardous.state.channels[GAS_RISK] = 0.95;
        hazardous.state.channels[ROOF_STABILITY] = 0.4;
        hazardous.state.channels[WATER_INGRESS_RATIO] = 0.35;

        let mut cmd = SubterraneanCommand::zero();
        cmd.set_cutter_head(1.0);
        safe.step(&cmd, 0.2);
        hazardous.step(&cmd, 0.2);

        assert!(hazardous.state().depth_m() < safe.state().depth_m());
        assert!(hazardous.state().abort_recommendation() > safe.state().abort_recommendation());
    }

    #[test]
    fn test_roof_instability_raises_abort_recommendation() {
        let mut sim = SimpleSubterraneanSimulator::new();
        sim.state.channels[DEPTH_M] = 20.0;
        sim.state.channels[OBSTACLE_PROXIMITY] = 0.95;
        sim.state.channels[SLIP_RATIO] = 0.8;
        sim.state.channels[HULL_STRESS] = 0.8;
        let abort_before = sim.state().abort_recommendation();
        let roof_before = sim.state().roof_stability();
        let mut cmd = SubterraneanCommand::zero();
        cmd.set_cutter_head(0.7);
        sim.step(&cmd, 0.05);
        assert!(sim.state().roof_stability() < roof_before);
        assert!(sim.state().abort_recommendation() > abort_before);
    }

    #[test]
    fn test_comms_blackout_switches_to_blackout_autonomy() {
        let mut sim = SimpleSubterraneanSimulator::new();
        // Seed a known, healthy outbound route. Teleporting directly to depth
        // without path history correctly means return is unverified and can
        // trigger a more severe surface/withdrawal response before blackout.
        for depth in 0..100 {
            let mut before = SubterraneanState::home();
            let mut after = SubterraneanState::home();
            before.channels[DEPTH_M] = depth as f64;
            after.channels[DEPTH_M] = depth as f64 + 1.0;
            sim.path_memory
                .observe(&before, &after, &SubterraneanCommand::zero());
        }
        sim.state.channels[DEPTH_M] = 100.0;
        sim.state.channels[WATER_INGRESS_RATIO] = 0.0;
        for _ in 0..50 {
            sim.step(&SubterraneanCommand::zero(), 0.02);
        }
        assert!(sim.state().relay_link_quality() < 0.2);
        assert_eq!(
            sim.state().inferred_mode(),
            SubterraneanOperatingMode::BlackoutAutonomy
        );
    }

    #[test]
    fn test_spoil_jam_reduces_escape_confidence() {
        let mut sim = SimpleSubterraneanSimulator::new();
        sim.state.channels[SPOIL_BUFFER_FILL] = 0.95;
        sim.state.channels[SLURRY_LOAD] = 0.8;
        sim.state.channels[SLIP_RATIO] = 0.7;
        for _ in 0..50 {
            sim.step(&SubterraneanCommand::zero(), 0.02);
        }
        assert!(sim.state().escape_confidence() < 0.75);
    }

    #[test]
    fn idle_dynamics_do_not_destroy_seals_or_invent_extreme_gas() {
        let mut sim = SimpleSubterraneanSimulator::new();
        for _ in 0..1000 {
            sim.step(&SubterraneanCommand::zero(), 0.005);
        }
        assert!(sim.state().seal_integrity() > 0.99);
        assert!(sim.state().gas_risk() < 0.1);
        assert_eq!(sim.state().inferred_mode(), SubterraneanOperatingMode::Dig);
    }

    #[test]
    fn flood_recovery_actuators_reduce_ingress_and_restore_seal() {
        let mut passive = SimpleSubterraneanSimulator::new();
        let mut active = SimpleSubterraneanSimulator::new();
        for sim in [&mut passive, &mut active] {
            sim.state_mut().channels[WATER_INGRESS_RATIO] = 0.8;
            sim.state_mut().channels[SEAL_INTEGRITY] = 0.35;
            sim.state_mut().channels[AQUIFER_RISK] = 0.8;
        }
        let mut recovery = SubterraneanCommand::zero();
        recovery.recovery.dewatering_pump = 1.0;
        recovery.recovery.sealant_injector = 1.0;
        for _ in 0..100 {
            passive.step(&SubterraneanCommand::zero(), 0.02);
            active.step(&recovery, 0.02);
        }
        assert!(active.state().water_ingress_ratio() < passive.state().water_ingress_ratio());
        assert!(active.state().seal_integrity() > passive.state().seal_integrity());
        assert!(active.recovery_resources().sealant_ratio < 1.0);
    }

    #[test]
    fn held_relay_command_consumes_only_one_finite_unit() {
        let mut sim = SimpleSubterraneanSimulator::new();
        sim.state_mut().channels[COMM_SIGNAL] = 0.0;
        sim.state_mut().channels[RELAY_LINK_QUALITY] = 0.0;
        let mut command = SubterraneanCommand::zero();
        command.recovery.relay_deployer = 1.0;
        for _ in 0..100 {
            sim.step(&command, 0.02);
        }
        assert_eq!(sim.recovery_resources().relay_units, 2);
        assert!(sim.state().relay_link_quality() > 0.2);
    }

    #[test]
    fn roof_support_is_finite_and_improves_stability() {
        let mut passive = SimpleSubterraneanSimulator::new();
        let mut active = SimpleSubterraneanSimulator::new();
        for sim in [&mut passive, &mut active] {
            sim.state_mut().channels[DEPTH_M] = 20.0;
            sim.state_mut().channels[ROOF_STABILITY] = 0.2;
            sim.state_mut().channels[HULL_STRESS] = 0.8;
            sim.state_mut().channels[OBSTACLE_PROXIMITY] = 0.8;
        }
        let mut support = SubterraneanCommand::zero();
        support.recovery.roof_support = 1.0;
        for _ in 0..50 {
            passive.step(&SubterraneanCommand::zero(), 0.02);
            active.step(&support, 0.02);
        }
        assert_eq!(active.recovery_resources().roof_support_units, 2);
        assert!(active.state().roof_stability() > passive.state().roof_stability());
    }

    fn rollout(hz: f64, duration_seconds: f64) -> SubterraneanState {
        let mut sim = SimpleSubterraneanSimulator::new();
        let mut cmd = SubterraneanCommand::zero();
        cmd.set_cutter_head(0.55);
        cmd.set_auger_feed(0.35);
        cmd.set_left_track(0.4);
        cmd.set_right_track(0.4);
        cmd.set_thermal_pump(0.2);
        let dt = 1.0 / hz;
        for _ in 0..(duration_seconds * hz) as usize {
            sim.step(&cmd, dt);
        }
        sim.state().clone()
    }

    #[test]
    fn equivalent_rollouts_are_rate_invariant() {
        let slow = rollout(50.0, 5.0);
        let fast = rollout(400.0, 5.0);
        let checks = [
            (DEPTH_M, 0.04),
            (CUTTER_TEMP_C, 0.75),
            (SPOIL_BUFFER_FILL, 0.04),
            (BATTERY_RATIO, 0.01),
            (GAS_RISK, 0.03),
            (SEAL_INTEGRITY, 0.01),
        ];
        for (channel, tolerance) in checks {
            let delta = (slow.channels[channel] - fast.channels[channel]).abs();
            assert!(
                delta <= tolerance,
                "channel {channel} changed with physics rate: {} vs {} (delta {delta}, tolerance {tolerance})",
                slow.channels[channel],
                fast.channels[channel]
            );
        }
    }

    #[test]
    fn hard_abrasive_geology_slows_penetration_and_increases_heat_and_wear() {
        use crate::geology::{GeotechnicalProfile, MaterialClass};

        let mut granite = SimpleSubterraneanSimulator::with_geology(
            GeotechnicalProfile::homogeneous(MaterialClass::Granite),
        );
        let mut clay = SimpleSubterraneanSimulator::with_geology(GeotechnicalProfile::homogeneous(
            MaterialClass::Clay,
        ));
        let mut command = SubterraneanCommand::zero();
        command.set_cutter_head(0.8);
        command.set_auger_feed(0.5);
        command.set_left_track(0.4);
        command.set_right_track(0.4);
        for _ in 0..1_000 {
            granite.step(&command, 0.005);
            clay.step(&command, 0.005);
        }
        assert!(granite.state().depth_m() < clay.state().depth_m());
        assert!(granite.state().cutter_temp_c() > clay.state().cutter_temp_c());
        assert!(granite.state().channels[TOOL_WEAR] > clay.state().channels[TOOL_WEAR]);
    }

    #[test]
    fn permeable_fault_gouge_elevates_aquifer_risk_relative_to_granite() {
        use crate::geology::{GeotechnicalProfile, MaterialClass};

        let mut fault = SimpleSubterraneanSimulator::with_geology(
            GeotechnicalProfile::homogeneous(MaterialClass::FaultGouge),
        );
        let mut granite = SimpleSubterraneanSimulator::with_geology(
            GeotechnicalProfile::homogeneous(MaterialClass::Granite),
        );
        fault.state_mut().channels[DEPTH_M] = 20.0;
        granite.state_mut().channels[DEPTH_M] = 20.0;
        for _ in 0..600 {
            fault.step(&SubterraneanCommand::zero(), 0.01);
            granite.step(&SubterraneanCommand::zero(), 0.01);
        }
        assert!(fault.state().aquifer_risk() > granite.state().aquifer_risk());
        assert!(fault.state().roof_stability() < granite.state().roof_stability());
    }

    #[test]
    fn simulator_records_outbound_route_and_exposes_return_assessment() {
        let mut simulator = SimpleSubterraneanSimulator::new();
        let mut command = SubterraneanCommand::zero();
        command.set_cutter_head(0.7);
        command.set_auger_feed(0.5);
        for _ in 0..4_000 {
            simulator.step(&command, 0.005);
        }
        let assessment = simulator.return_path_assessment();
        assert!(assessment.distance_home_m > 0.0);
        assert!(!simulator.return_path_memory().segments().is_empty());
        assert_eq!(
            simulator.state().channels[RETURN_PATH_CONFIDENCE],
            assessment.path_confidence
        );
    }
}
