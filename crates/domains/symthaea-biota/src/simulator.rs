// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-pass sanctuary / right-of-way simulator.
use crate::environment::EncounterEnvironment;
use crate::types::{
    ANIMAL_PRESENCE, BLACKOUT_RESILIENCE, BiotaCommand, BiotaState, CLASSIFICATION_CONFIDENCE,
    COMM_LINK_QUALITY, DISTRESS_SIGNAL, FLOCK_PANIC_RISK, HANDOFF_CONFIDENCE,
    INTERVENTION_OVERREACH, LOCAL_NOISE_STRESS, MISSION_PROGRESS, PATH_CONFLICT_RISK,
    RESPONSE_LATENCY, ROBOT_THREAT, ROUTE_CLEAR_CONFIDENCE, SANCTUARY_SIGNAL, THERMAL_STRESS,
    VEHICLE_THREAT, WELFARE_INTEGRITY,
};
use symthaea_core::genesis::GenesisSeed;

pub trait BiotaPhysicsSimulator {
    fn step(&mut self, cmd: &BiotaCommand, dt: f64);
    fn state(&self) -> &BiotaState;
    fn reset(&mut self);
}

/// How fast each sensed channel tracks its ground-truth source (per
/// second). Higher = faster/less-lagged sensing.
const ANIMAL_PRESENCE_SENSOR_GAIN: f64 = 1.5;
const DISTRESS_SENSOR_GAIN: f64 = 1.0;
const THREAT_SENSOR_GAIN: f64 = 1.2;

pub struct SimpleBiotaSimulator {
    state: BiotaState,
    environment: EncounterEnvironment,
    genesis: GenesisSeed,
}

impl SimpleBiotaSimulator {
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            state: BiotaState::home(),
            environment: EncounterEnvironment::new(genesis),
            genesis: genesis.clone(),
        }
    }

    /// Mutable state access, for tests that need to set up a degraded
    /// starting condition (e.g. a low sanctuary signal) before stepping.
    pub fn state_mut(&mut self) -> &mut BiotaState {
        &mut self.state
    }

    /// Mutable access to the ground-truth encounter world, for tests that
    /// need to script an actual animal-distress or intrusion scenario
    /// independent of the robot's own actuator history.
    pub fn environment_mut(&mut self) -> &mut EncounterEnvironment {
        &mut self.environment
    }

    pub fn environment(&self) -> &EncounterEnvironment {
        &self.environment
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

        // Ground-truth world evolves independently of the robot's own
        // actuation; `soothing` is the only channel through which this
        // robot's actions can influence it.
        let soothing = ((thermal + sanctuary) * 0.5).clamp(0.0, 1.0);
        self.environment.step(dt, soothing);

        // Sensing: each channel chases its ground-truth source through a
        // first-order lag, not a self-referential or actuator-only formula.
        let animal_present_truth = if self.environment.animal_present {
            1.0
        } else {
            0.0
        };
        self.state.channels[ANIMAL_PRESENCE] = (self.state.channels[ANIMAL_PRESENCE]
            + (animal_present_truth - self.state.channels[ANIMAL_PRESENCE])
                * ANIMAL_PRESENCE_SENSOR_GAIN
                * dt)
            .clamp(0.0, 1.0);
        self.state.channels[12] = (self.state.channels[12]
            + (self.environment.vehicle_intrusion - self.state.channels[12])
                * THREAT_SENSOR_GAIN
                * dt)
            .clamp(0.0, 1.0);
        self.state.channels[13] = (self.state.channels[13]
            + (self.environment.robot_intrusion - self.state.channels[13])
                * THREAT_SENSOR_GAIN
                * dt)
            .clamp(0.0, 1.0);

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
            + (self.environment.animal_distress - self.state.channels[DISTRESS_SIGNAL])
                * DISTRESS_SENSOR_GAIN
                * dt
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
        // channels[12] (VEHICLE_THREAT) / channels[13] (ROBOT_THREAT) are
        // now set above from the ground-truth environment, not re-derived
        // from the robot's own drive here (that was the self-referential
        // bug from SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md).
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
        self.environment = EncounterEnvironment::new(&self.genesis);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BiotaOperatingMode;

    fn test_sim() -> SimpleBiotaSimulator {
        SimpleBiotaSimulator::new(&GenesisSeed::from_phrase("test"))
    }

    #[test]
    fn test_stable() {
        let mut sim = test_sim();
        for _ in 0..1000 {
            sim.step(&BiotaCommand::zero(), 0.005);
        }
        assert!(sim.state().is_finite());
    }

    #[test]
    fn test_sanctuary_projection_improves_signal() {
        let mut sim = test_sim();
        let mut cmd = BiotaCommand::zero();
        cmd.torques[5] = 1.0;
        let before = sim.state().sanctuary_signal();
        sim.step(&cmd, 0.05);
        assert!(sim.state().sanctuary_signal() > before);
    }

    #[test]
    fn test_path_conflict_enters_crossing_guard() {
        let mut sim = test_sim();
        sim.state.channels[PATH_CONFLICT_RISK] = 0.8;
        assert_eq!(
            sim.state().inferred_mode(),
            BiotaOperatingMode::CrossingGuard
        );
    }

    #[test]
    fn test_distress_enters_response_mode() {
        let mut sim = test_sim();
        sim.state.channels[DISTRESS_SIGNAL] = 0.85;
        assert_eq!(
            sim.state().inferred_mode(),
            BiotaOperatingMode::DistressResponse
        );
    }

    #[test]
    fn test_comm_loss_enters_blackout_autonomy() {
        let mut sim = test_sim();
        sim.state.channels[COMM_LINK_QUALITY] = 0.1;
        sim.state.channels[BLACKOUT_RESILIENCE] = 0.8;
        assert_eq!(
            sim.state().inferred_mode(),
            BiotaOperatingMode::BlackoutAutonomy
        );
    }

    #[test]
    fn test_animal_presence_is_falsifiable_without_encounter() {
        // Regression for the worst finding in the review: with no animal
        // ever present in the ground-truth environment, distress_signal and
        // animal_presence must NOT rise to alarm levels no matter how long
        // the robot runs -- unlike the old pure autoregression that
        // converged to 1.0 unconditionally. Ground truth is re-pinned false
        // every step so this doesn't depend on the environment's own
        // stochastic arrival roll never firing.
        let mut sim = test_sim();
        for _ in 0..2000 {
            sim.environment_mut().animal_present = false;
            sim.environment_mut().animal_distress = 0.0;
            sim.step(&BiotaCommand::zero(), 0.01);
        }
        assert!(
            sim.state().channels[ANIMAL_PRESENCE] < 0.2,
            "animal_presence must stay low with no ground-truth encounter, got {}",
            sim.state().channels[ANIMAL_PRESENCE]
        );
        assert!(
            sim.state().distress_signal() < 0.3,
            "distress_signal must stay low with no ground-truth animal, got {}",
            sim.state().distress_signal()
        );
    }

    #[test]
    fn test_distress_signal_tracks_ground_truth_encounter() {
        // The flip side: a scripted real encounter must be sensed. Ground
        // truth is re-pinned present/distressed every step so this doesn't
        // depend on the environment's own stochastic departure roll never
        // firing mid-test.
        let mut sim = test_sim();
        for _ in 0..300 {
            sim.environment_mut().animal_present = true;
            sim.environment_mut().animal_distress = 0.9;
            sim.step(&BiotaCommand::zero(), 0.05);
        }
        assert!(
            sim.state().channels[ANIMAL_PRESENCE] > 0.6,
            "animal_presence must rise once the ground truth has an animal present, got {}",
            sim.state().channels[ANIMAL_PRESENCE]
        );
        assert!(
            sim.state().distress_signal() > 0.5,
            "distress_signal must track a real high-distress encounter, got {}",
            sim.state().distress_signal()
        );
    }

    #[test]
    fn test_vehicle_and_robot_threat_are_not_purely_actuator_derived() {
        // Regression: heavy drive commands used to directly inflate
        // vehicle_threat/robot_threat. Now they must stay near their
        // ground-truth (pinned at zero here -- no scripted intrusion) even
        // under sustained heavy driving.
        let mut sim = test_sim();
        let mut cmd = BiotaCommand::zero();
        cmd.torques[0] = 1.0;
        cmd.torques[1] = 1.0;
        for _ in 0..500 {
            sim.environment_mut().vehicle_intrusion = 0.0;
            sim.environment_mut().robot_intrusion = 0.0;
            sim.step(&cmd, 0.01);
        }
        assert!(
            sim.state().channels[12] < 0.3,
            "vehicle_threat must not be driven up by the robot's own drive command, got {}",
            sim.state().channels[12]
        );
        assert!(
            sim.state().channels[13] < 0.3,
            "robot_threat must not be driven up by the robot's own drive command, got {}",
            sim.state().channels[13]
        );
    }
}
