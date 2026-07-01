// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Perturbation crucibles for adversarial vehicle training.
//!
//! These perturbations brutally attack the vehicle's sensor and physics model
//! to force robust adaptation through Active Inference. The philosophy:
//! if she can survive black ice at 100 km/h in simulation, she can handle
//! a wet road in any city.

use serde::{Deserialize, Serialize};

use crate::simulator::VehiclePhysicsSimulator;
use crate::types::VehicleState;

/// Perturbation types for the training crucible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VehiclePerturbation {
    /// Black ice: friction drops to near zero.
    /// The tires lose grip. Active Inference must detect the ambiguity spike
    /// and autonomously reduce speed + increase mesh reliance.
    BlackIce {
        /// Friction scale (0.0 = pure ice, 0.3 = packed snow, 1.0 = dry).
        friction_scale: f64,
    },

    /// Sensor blindness: camera/LiDAR encoded channels become unreliable.
    /// Simulates heavy rain, fog, or sensor failure. The "Retina" coprocessor
    /// output degrades. Ambiguity term in EFE should drive speed reduction.
    SensorBlindness {
        /// How many channels are zeroed out (0..14). Higher = more blind.
        affected_channels: usize,
    },

    /// Crosswind: sustained lateral force.
    /// Tests the controller's ability to maintain lane position under
    /// persistent lateral disturbance.
    Crosswind {
        /// Lateral force in Newtons (body frame). Positive = push left.
        force_n: f64,
    },

    /// Tire blowout: asymmetric grip loss + yaw moment.
    /// Front-left blowout induces a pull to the left. The controller must
    /// compensate with counter-steer without overcorrecting.
    TireBlowout {
        /// Which tire: "front_left", "front_right", "rear_left", "rear_right".
        tire: String,
        /// Grip reduction factor (0.0 = no grip, 0.5 = half grip).
        grip_reduction: f64,
    },

    /// Mesh data spoof: a malicious peer broadcasts false brake signals.
    /// Tests Byzantine fault tolerance in the swarm. The vehicle must
    /// detect and ignore spoofed mesh data without panic-braking.
    MeshSpoof {
        /// Number of spoofed peers injecting false brake signals.
        num_spoofed_peers: usize,
    },

    /// Speed bump / pothole: vertical impulse.
    /// A sudden vertical force simulates road irregularities.
    /// Tests suspension response and controller stability.
    RoadImpulse {
        /// Vertical deceleration impulse (m/s²).
        decel_impulse: f64,
    },
}

/// Schedule for applying perturbations during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationSchedule {
    /// List of (step, perturbation) pairs.
    pub events: Vec<(usize, VehiclePerturbation)>,
    /// Step at which all perturbations are cleared (friction restored, etc.).
    pub clear_step: Option<usize>,
}

impl PerturbationSchedule {
    /// Empty schedule (no perturbations).
    pub fn none() -> Self {
        Self {
            events: Vec::new(),
            clear_step: None,
        }
    }

    /// Standard "Black Ice" crucible: friction drops at step 400.
    pub fn black_ice() -> Self {
        Self {
            events: vec![(
                400,
                VehiclePerturbation::BlackIce {
                    friction_scale: 0.05,
                },
            )],
            clear_step: Some(1200),
        }
    }

    /// Standard "Crosswind" crucible: 3000N lateral wind at step 300.
    pub fn crosswind() -> Self {
        Self {
            events: vec![(300, VehiclePerturbation::Crosswind { force_n: 3000.0 })],
            clear_step: Some(800),
        }
    }

    /// Front-left tire blowout at step 300: front grip drops to 30%.
    pub fn tire_blowout_front() -> Self {
        Self {
            events: vec![(
                300,
                VehiclePerturbation::TireBlowout {
                    tire: "front_left".to_string(),
                    grip_reduction: 0.3,
                },
            )],
            clear_step: Some(1000),
        }
    }

    /// Sensor blindness at step 300: 4 proprioceptive channels degraded.
    pub fn sensor_blindness() -> Self {
        Self {
            events: vec![(
                300,
                VehiclePerturbation::SensorBlindness {
                    affected_channels: 4,
                },
            )],
            clear_step: Some(800),
        }
    }

    /// Mesh spoof at step 200: 2 spoofed peers injecting false brake signals.
    pub fn mesh_spoof() -> Self {
        Self {
            events: vec![(
                200,
                VehiclePerturbation::MeshSpoof {
                    num_spoofed_peers: 2,
                },
            )],
            clear_step: Some(800),
        }
    }

    /// Road impulse (speed bump / pothole) at step 150: 5 m/s² deceleration.
    pub fn road_impulse() -> Self {
        Self {
            events: vec![(150, VehiclePerturbation::RoadImpulse { decel_impulse: 5.0 })],
            clear_step: None,
        }
    }

    /// Combined crucible: ice + blindness + crosswind at staggered intervals.
    pub fn gauntlet() -> Self {
        Self {
            events: vec![
                (200, VehiclePerturbation::Crosswind { force_n: 2000.0 }),
                (
                    500,
                    VehiclePerturbation::BlackIce {
                        friction_scale: 0.1,
                    },
                ),
                (
                    800,
                    VehiclePerturbation::SensorBlindness {
                        affected_channels: 4,
                    },
                ),
            ],
            clear_step: Some(1500),
        }
    }

    /// Apply perturbations scheduled for the given step.
    pub fn apply(&self, step: usize, sim: &mut dyn VehiclePhysicsSimulator) {
        // Clear all perturbations if we've passed the clear step
        if let Some(clear) = self.clear_step {
            if step == clear {
                sim.set_friction_scale(1.0);
                return;
            }
        }

        for (event_step, perturbation) in &self.events {
            if step == *event_step {
                match perturbation {
                    VehiclePerturbation::BlackIce { friction_scale } => {
                        sim.set_friction_scale(*friction_scale);
                    }
                    VehiclePerturbation::Crosswind { force_n } => {
                        sim.apply_external_force([0.0, *force_n]);
                    }
                    VehiclePerturbation::RoadImpulse { decel_impulse } => {
                        // F = m*a; using default ChassisParams mass (1500 kg)
                        sim.apply_external_force([-decel_impulse * 1500.0, 0.0]);
                    }
                    VehiclePerturbation::TireBlowout {
                        tire,
                        grip_reduction,
                    } => {
                        // Map tire position to front/rear axle (bicycle model collapses L/R)
                        let front = if tire.starts_with("front") {
                            *grip_reduction
                        } else {
                            1.0
                        };
                        let rear = if tire.starts_with("rear") {
                            *grip_reduction
                        } else {
                            1.0
                        };
                        sim.set_axle_friction(front, rear);
                    }
                    // SensorBlindness and MeshSpoof are applied at the state level
                    // via apply_to_state(), not at the physics simulator level.
                    VehiclePerturbation::SensorBlindness { .. }
                    | VehiclePerturbation::MeshSpoof { .. } => {}
                }
            }
        }

        // Crosswind is sustained: re-apply every step while active
        for (event_step, perturbation) in &self.events {
            if let VehiclePerturbation::Crosswind { force_n } = perturbation {
                let clear = self.clear_step.unwrap_or(usize::MAX);
                if step > *event_step && step < clear {
                    sim.apply_external_force([0.0, *force_n]);
                }
            }
        }
    }

    /// Apply state-level perturbations for the given step.
    ///
    /// Unlike `apply()` which modifies the physics simulator, this modifies
    /// the vehicle state directly — handling perturbations that operate at
    /// the sensor/encoder level rather than the physics level.
    /// SensorBlindness and MeshSpoof are sustained: active from trigger step until clear.
    pub fn apply_to_state(&self, step: usize, state: &mut VehicleState) {
        let clear = self.clear_step.unwrap_or(usize::MAX);
        if step >= clear {
            return;
        }

        for (event_step, perturbation) in &self.events {
            if step >= *event_step {
                match perturbation {
                    VehiclePerturbation::SensorBlindness { affected_channels } => {
                        state.apply_sensor_blindness(*affected_channels);
                    }
                    VehiclePerturbation::MeshSpoof { num_spoofed_peers } => {
                        state.apply_mesh_spoof(*num_spoofed_peers);
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Evaluate whether a perturbation scenario resulted in a critical failure.
///
/// Returns `true` if the vehicle has lost control (should terminate episode).
pub fn is_critical_failure(state: &VehicleState) -> bool {
    // Spin: yaw rate exceeds 1.5 rad/s (~85 deg/s)
    if state.yaw_rate.abs() > 1.5 {
        return true;
    }

    // Sideslip: lateral velocity exceeds 5 m/s (full slide)
    if state.lateral_velocity.abs() > 5.0 {
        return true;
    }

    // Backward motion
    if state.speed < -0.5 {
        return true;
    }

    // NaN/Inf
    if !state.is_finite() {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::BicycleModelSimulator;

    #[test]
    fn test_empty_schedule() {
        let schedule = PerturbationSchedule::none();
        assert!(schedule.events.is_empty());
    }

    #[test]
    fn test_black_ice_schedule() {
        let schedule = PerturbationSchedule::black_ice();
        assert_eq!(schedule.events.len(), 1);
    }

    #[test]
    fn test_gauntlet_schedule() {
        let schedule = PerturbationSchedule::gauntlet();
        assert_eq!(schedule.events.len(), 3);
    }

    #[test]
    fn test_perturbation_apply_friction() {
        let mut sim = BicycleModelSimulator::at_speed(13.4);
        let schedule = PerturbationSchedule::black_ice();

        // Before perturbation
        assert!((sim.friction_scale() - 1.0).abs() < 1e-10);

        // Apply at step 400
        schedule.apply(400, &mut sim);
        assert!(
            (sim.friction_scale() - 0.05).abs() < 1e-10,
            "Friction should drop: {}",
            sim.friction_scale()
        );

        // Clear at step 1200
        schedule.apply(1200, &mut sim);
        assert!(
            (sim.friction_scale() - 1.0).abs() < 1e-10,
            "Friction should restore: {}",
            sim.friction_scale()
        );
    }

    #[test]
    fn test_critical_failure_spin() {
        let mut state = VehicleState::cruising(13.4);
        state.yaw_rate = 2.0;
        assert!(is_critical_failure(&state));
    }

    #[test]
    fn test_critical_failure_slide() {
        let mut state = VehicleState::cruising(13.4);
        state.lateral_velocity = 6.0;
        assert!(is_critical_failure(&state));
    }

    #[test]
    fn test_no_failure_normal_driving() {
        let state = VehicleState::cruising(13.4);
        assert!(!is_critical_failure(&state));
    }

    #[test]
    fn test_tire_blowout_schedule() {
        let schedule = PerturbationSchedule::tire_blowout_front();
        assert_eq!(schedule.events.len(), 1);
    }

    #[test]
    fn test_tire_blowout_changes_handling() {
        let mut sim = BicycleModelSimulator::at_speed(13.4);
        let schedule = PerturbationSchedule::tire_blowout_front();

        let cmd = crate::types::VehicleCommand {
            steering: 0.2,
            throttle: 0.3,
            brake: 0.0,
        };

        // Drive normally before blowout
        for step in 0..300 {
            schedule.apply(step, &mut sim);
            sim.step(&cmd, 0.005);
        }
        let heading_before = sim.state().heading;

        // Apply blowout and drive more
        schedule.apply(300, &mut sim);
        for step in 301..500 {
            schedule.apply(step, &mut sim);
            sim.step(&cmd, 0.005);
        }

        assert!(
            sim.state().is_finite(),
            "Vehicle should remain stable through blowout"
        );
        assert!(
            (sim.state().heading - heading_before).abs() > 0.01,
            "Blowout should affect heading trajectory"
        );
    }

    #[test]
    fn test_crosswind_sustained() {
        let mut sim = BicycleModelSimulator::at_speed(13.4);
        let schedule = PerturbationSchedule::crosswind();
        let cmd = crate::types::VehicleCommand {
            steering: 0.0,
            throttle: 0.3,
            brake: 0.0,
        };

        // Apply crosswind and step for a while
        for step in 300..350 {
            schedule.apply(step, &mut sim);
            sim.step(&cmd, 0.005);
        }

        // Should have some lateral effect
        assert!(
            sim.state().lateral_velocity.abs() > 0.01
                || sim.state().yaw_rate.abs() > 0.001
                || sim.state().heading.abs() > 0.001,
            "Sustained crosswind should affect dynamics"
        );
    }

    #[test]
    fn test_sensor_blindness_schedule() {
        let schedule = PerturbationSchedule::sensor_blindness();
        assert_eq!(schedule.events.len(), 1);
        assert_eq!(schedule.clear_step, Some(800));
    }

    #[test]
    fn test_sensor_blindness_apply_to_state() {
        let schedule = PerturbationSchedule::sensor_blindness();
        let mut state = VehicleState::cruising(13.4);
        state.lateral_velocity = 2.0;
        state.yaw_rate = 0.5;
        state.heading = 0.3;

        // Before trigger: no effect
        schedule.apply_to_state(200, &mut state);
        assert!(
            (state.speed - 13.4).abs() < 1e-10,
            "Before trigger: speed unchanged"
        );

        // At trigger (step 300): channels zeroed
        state = VehicleState::cruising(13.4);
        state.lateral_velocity = 2.0;
        schedule.apply_to_state(300, &mut state);
        assert!(
            (state.speed - 0.0).abs() < 1e-10,
            "At trigger: speed zeroed"
        );
        assert!(
            (state.lateral_velocity - 0.0).abs() < 1e-10,
            "At trigger: lat_vel zeroed"
        );

        // Sustained (step 500): still active
        state = VehicleState::cruising(13.4);
        schedule.apply_to_state(500, &mut state);
        assert!(
            (state.speed - 0.0).abs() < 1e-10,
            "Sustained: speed still zeroed"
        );

        // After clear (step 800): no effect
        state = VehicleState::cruising(13.4);
        schedule.apply_to_state(800, &mut state);
        assert!(
            (state.speed - 13.4).abs() < 1e-10,
            "After clear: speed restored"
        );
    }

    #[test]
    fn test_mesh_spoof_schedule() {
        let schedule = PerturbationSchedule::mesh_spoof();
        assert_eq!(schedule.events.len(), 1);
        assert_eq!(schedule.clear_step, Some(800));
    }

    #[test]
    fn test_mesh_spoof_apply_to_state() {
        let schedule = PerturbationSchedule::mesh_spoof();
        let mut state = VehicleState::cruising(13.4);
        state.mesh_brake_density = 0.0;
        state.mesh_confidence = 0.3;

        // Before trigger: no effect
        schedule.apply_to_state(100, &mut state);
        assert!(
            (state.mesh_brake_density - 0.0).abs() < 1e-10,
            "Before trigger: no spoof"
        );

        // At trigger (step 200): spoofed
        schedule.apply_to_state(200, &mut state);
        assert!(
            state.mesh_brake_density > 0.4,
            "At trigger: brake density inflated: {}",
            state.mesh_brake_density
        );

        // After clear: no effect
        state.mesh_brake_density = 0.0;
        state.mesh_confidence = 0.3;
        schedule.apply_to_state(800, &mut state);
        assert!(
            (state.mesh_brake_density - 0.0).abs() < 1e-10,
            "After clear: no spoof"
        );
    }

    #[test]
    fn test_road_impulse_schedule() {
        let schedule = PerturbationSchedule::road_impulse();
        assert_eq!(schedule.events.len(), 1);
        assert!(schedule.clear_step.is_none());
    }

    #[test]
    fn test_road_impulse_applies_force() {
        let mut sim = BicycleModelSimulator::at_speed(13.4);
        let schedule = PerturbationSchedule::road_impulse();
        let cmd = crate::types::VehicleCommand {
            steering: 0.0,
            throttle: 0.3,
            brake: 0.0,
        };

        let speed_before = sim.state().speed;
        schedule.apply(150, &mut sim);
        sim.step(&cmd, 0.005);

        assert!(
            sim.state().speed < speed_before,
            "Road impulse should decelerate: before={speed_before}, after={}",
            sim.state().speed
        );
    }
}
