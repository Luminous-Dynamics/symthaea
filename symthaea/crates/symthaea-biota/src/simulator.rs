// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-pass sanctuary / right-of-way simulator.
use crate::types::{
    BiotaCommand, BiotaState, ANIMAL_PRESENCE, BLACKOUT_RESILIENCE, CLASSIFICATION_CONFIDENCE,
    COMM_LINK_QUALITY, DISTRESS_SIGNAL, FLOCK_PANIC_RISK, HANDOFF_CONFIDENCE,
    INTERVENTION_OVERREACH, LOCAL_NOISE_STRESS, MISSION_PROGRESS, PATH_CONFLICT_RISK,
    RESPONSE_LATENCY, ROBOT_THREAT, ROUTE_CLEAR_CONFIDENCE, SANCTUARY_SIGNAL, THERMAL_STRESS,
    VEHICLE_THREAT, WELFARE_INTEGRITY,
};

pub trait BiotaPhysicsSimulator {
    fn step(&mut self, cmd: &BiotaCommand, dt: f64);
    fn state(&self) -> &BiotaState;
    fn reset(&mut self);
}

pub struct SimpleBiotaSimulator {
    state: BiotaState,
}

impl SimpleBiotaSimulator {
    pub fn new() -> Self {
        Self {
            state: BiotaState::home(),
        }
    }
}

impl BiotaPhysicsSimulator for SimpleBiotaSimulator {
    fn step(&mut self, cmd: &BiotaCommand, dt: f64) {
        let drive_request = (cmd.torques[0].abs() + cmd.torques[1].abs()) as f64 * 0.5;
        let acoustic = cmd.acoustic_chime().max(0.0) as f64;
        let thermal = cmd.thermal_beacon().max(0.0) as f64;
        let sanctuary = cmd.sanctuary_projector().max(0.0) as f64;
        let quiet_gate = if self.state.channels[DISTRESS_SIGNAL] >= 0.65
            || self.state.channels[LOCAL_NOISE_STRESS] >= 0.7
        {
            0.3
        } else {
            1.0
        };
        let effective_drive = drive_request * quiet_gate;
        let effective_acoustic = acoustic * quiet_gate;

        self.state.channels[ANIMAL_PRESENCE] =
            (self.state.channels[ANIMAL_PRESENCE] * 0.985 + 0.015).clamp(0.0, 1.0);
        self.state.channels[SANCTUARY_SIGNAL] = (self.state.channels[SANCTUARY_SIGNAL]
            + sanctuary * dt * 0.22
            - self.state.channels[PATH_CONFLICT_RISK] * dt * 0.05)
            .clamp(0.0, 1.0);
        self.state.channels[PATH_CONFLICT_RISK] = (self.state.channels[PATH_CONFLICT_RISK]
            + self.state.channels[VEHICLE_THREAT] * dt * 0.12
            + self.state.channels[ROBOT_THREAT] * dt * 0.08
            - sanctuary * dt * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[DISTRESS_SIGNAL] = (self.state.channels[DISTRESS_SIGNAL]
            + self.state.channels[PATH_CONFLICT_RISK] * dt * 0.08
            + self.state.channels[LOCAL_NOISE_STRESS] * dt * 0.07
            + self.state.channels[THERMAL_STRESS] * dt * 0.06
            - sanctuary * dt * 0.06
            - thermal * dt * 0.05)
            .clamp(0.0, 1.0);
        self.state.channels[THERMAL_STRESS] = (self.state.channels[THERMAL_STRESS]
            + self.state.channels[ANIMAL_PRESENCE] * dt * 0.02
            - thermal * dt * 0.1)
            .clamp(0.0, 1.0);
        self.state.channels[CLASSIFICATION_CONFIDENCE] =
            (self.state.channels[CLASSIFICATION_CONFIDENCE] * 0.995
                + sanctuary * 0.002
                + (1.0 - effective_drive) * 0.001)
                .clamp(0.0, 1.0);
        self.state.channels[HANDOFF_CONFIDENCE] = (self.state.channels[HANDOFF_CONFIDENCE]
            + sanctuary * dt * 0.08
            + effective_acoustic * dt * 0.03
            - self.state.channels[COMM_LINK_QUALITY] * dt * 0.01)
            .clamp(0.0, 1.0);
        self.state.channels[ROUTE_CLEAR_CONFIDENCE] = (self.state.channels[ROUTE_CLEAR_CONFIDENCE]
            + sanctuary * dt * 0.07
            - self.state.channels[PATH_CONFLICT_RISK] * dt * 0.06)
            .clamp(0.0, 1.0);
        self.state.channels[COMM_LINK_QUALITY] = (self.state.channels[COMM_LINK_QUALITY] * 0.998
            + self.state.channels[BLACKOUT_RESILIENCE] * 0.001)
            .clamp(0.0, 1.0);
        self.state.channels[11] =
            (self.state.channels[11] + effective_drive * dt * 0.05).clamp(0.0, 1.0);
        self.state.channels[12] =
            (self.state.channels[12] * 0.996 + effective_drive * 0.003).clamp(0.0, 1.0);
        self.state.channels[13] =
            (self.state.channels[13] * 0.997 + drive_request * 0.002).clamp(0.0, 1.0);
        self.state.channels[FLOCK_PANIC_RISK] = (self.state.channels[FLOCK_PANIC_RISK]
            + effective_acoustic * dt * 0.08
            + drive_request * dt * 0.03
            - sanctuary * dt * 0.04)
            .clamp(0.0, 1.0);
        self.state.channels[15] =
            (self.state.channels[15] * 0.995 + sanctuary * 0.003).clamp(0.0, 1.0);
        self.state.channels[16] =
            (1.0 - self.state.channels[SANCTUARY_SIGNAL] * 0.5).clamp(0.0, 1.0);
        self.state.channels[INTERVENTION_OVERREACH] = (self.state.channels[INTERVENTION_OVERREACH]
            + drive_request * dt * 0.05
            + effective_acoustic * dt * 0.06
            - quiet_gate * dt * 0.02)
            .clamp(0.0, 1.0);
        self.state.channels[RESPONSE_LATENCY] = (self.state.channels[RESPONSE_LATENCY]
            + self.state.channels[COMM_LINK_QUALITY].mul_add(-dt * 0.03, dt * 0.01))
        .clamp(0.0, 1.0);
        self.state.channels[19] = (self.state.channels[19] * 0.99 + 0.01).clamp(0.0, 1.0);
        self.state.channels[LOCAL_NOISE_STRESS] = (self.state.channels[LOCAL_NOISE_STRESS]
            + effective_acoustic * dt * 0.12
            + drive_request * dt * 0.03
            - sanctuary * dt * 0.04)
            .clamp(0.0, 1.0);
        self.state.channels[BLACKOUT_RESILIENCE] =
            (self.state.channels[BLACKOUT_RESILIENCE] * 0.999 + sanctuary * 0.001).clamp(0.0, 1.0);
        self.state.channels[MISSION_PROGRESS] =
            (self.state.channels[MISSION_PROGRESS] + sanctuary * dt * 0.04).clamp(0.0, 1.0);
        self.state.channels[WELFARE_INTEGRITY] = (1.0
            - self.state.channels[DISTRESS_SIGNAL] * 0.35
            - self.state.channels[PATH_CONFLICT_RISK] * 0.25
            - self.state.channels[INTERVENTION_OVERREACH] * 0.2
            - self.state.channels[FLOCK_PANIC_RISK] * 0.15)
            .clamp(0.0, 1.0);
    }

    fn state(&self) -> &BiotaState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = BiotaState::home();
    }
}

impl Default for SimpleBiotaSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BiotaOperatingMode;

    #[test]
    fn test_stable() {
        let mut sim = SimpleBiotaSimulator::new();
        for _ in 0..1000 {
            sim.step(&BiotaCommand::zero(), 0.005);
        }
        assert!(sim.state().is_finite());
    }

    #[test]
    fn test_sanctuary_projection_improves_signal() {
        let mut sim = SimpleBiotaSimulator::new();
        let mut cmd = BiotaCommand::zero();
        cmd.torques[5] = 1.0;
        let before = sim.state().sanctuary_signal();
        sim.step(&cmd, 0.05);
        assert!(sim.state().sanctuary_signal() > before);
    }

    #[test]
    fn test_path_conflict_enters_crossing_guard() {
        let mut sim = SimpleBiotaSimulator::new();
        sim.state.channels[PATH_CONFLICT_RISK] = 0.8;
        assert_eq!(
            sim.state().inferred_mode(),
            BiotaOperatingMode::CrossingGuard
        );
    }

    #[test]
    fn test_distress_enters_response_mode() {
        let mut sim = SimpleBiotaSimulator::new();
        sim.state.channels[DISTRESS_SIGNAL] = 0.85;
        assert_eq!(
            sim.state().inferred_mode(),
            BiotaOperatingMode::DistressResponse
        );
    }

    #[test]
    fn test_comm_loss_enters_blackout_autonomy() {
        let mut sim = SimpleBiotaSimulator::new();
        sim.state.channels[COMM_LINK_QUALITY] = 0.1;
        sim.state.channels[BLACKOUT_RESILIENCE] = 0.8;
        assert_eq!(
            sim.state().inferred_mode(),
            BiotaOperatingMode::BlackoutAutonomy
        );
    }
}
