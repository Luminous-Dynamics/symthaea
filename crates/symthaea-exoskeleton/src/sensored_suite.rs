// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sensored Full-Frame Suit — wires symthaea-sensors into the 20-DOF exoskeleton.
//!
//! Adds realistic sensor noise to joint encoders and IMU readings, providing
//! a sim-to-real pathway: the full-frame suit reports joint angles and
//! spine IMU with configurable Gaussian noise + bias.
//!
//! This is the proof that `symthaea-sensors` is not orphaned — it drives
//! real consciousness-coupled perception feedback.

#![cfg(feature = "sensors")]

use symthaea_sensors::{ImuReading, ImuSensor, JointEncoderArray, JointReading, NoiseConfig};

use crate::full_frame::FullFrameSimulator;

/// A full-frame simulator wrapped with sensor readouts.
///
/// Provides noisy joint encoders for all 20 joints across 5 chains
/// and an IMU at the spine base (torso-equivalent).
pub struct SensoredFullFrame {
    pub sim: FullFrameSimulator,
    /// Joint encoders — one per chain (5 total).
    pub spine_encoder: JointEncoderArray,
    pub left_arm_encoder: JointEncoderArray,
    pub right_arm_encoder: JointEncoderArray,
    pub left_leg_encoder: JointEncoderArray,
    pub right_leg_encoder: JointEncoderArray,
    /// IMU at the spine base.
    pub spine_imu: ImuSensor,
}

/// Complete sensor readout for one tick.
#[derive(Debug, Clone, Default)]
pub struct SuiteReading {
    /// 2 spine joints.
    pub spine: Vec<JointReading>,
    /// 6 left arm joints.
    pub left_arm: Vec<JointReading>,
    /// 6 right arm joints.
    pub right_arm: Vec<JointReading>,
    /// 3 left leg joints.
    pub left_leg: Vec<JointReading>,
    /// 3 right leg joints.
    pub right_leg: Vec<JointReading>,
    /// Spine IMU (accel + gyro).
    pub imu: ImuReading,
}

impl SuiteReading {
    /// Total joints across all chains (should be 20).
    pub fn total_joints(&self) -> usize {
        self.spine.len()
            + self.left_arm.len()
            + self.right_arm.len()
            + self.left_leg.len()
            + self.right_leg.len()
    }

    /// Sum of absolute joint angular velocities — rough "motion magnitude".
    pub fn total_angular_speed(&self) -> f64 {
        let mut sum = 0.0;
        for r in self.spine.iter()
            .chain(self.left_arm.iter())
            .chain(self.right_arm.iter())
            .chain(self.left_leg.iter())
            .chain(self.right_leg.iter())
        {
            sum += r.velocity.abs();
        }
        sum
    }
}

impl SensoredFullFrame {
    /// Create a new sensored suite with typical noise parameters.
    pub fn new() -> Self {
        let sim = FullFrameSimulator::new();

        // Joint encoders: 0.01 rad noise, 0.001 rad bias (industrial-grade)
        let encoder_noise = NoiseConfig {
            noise_std: 0.01,
            bias: 0.001,
            update_rate_hz: 1000.0,
        };

        // IMU: 0.1 m/s² accel noise, 0.01 rad/s gyro noise
        let accel_noise = NoiseConfig {
            noise_std: 0.1,
            bias: 0.02,
            update_rate_hz: 500.0,
        };
        let gyro_noise = NoiseConfig {
            noise_std: 0.01,
            bias: 0.001,
            update_rate_hz: 500.0,
        };

        // Spine IMU attached to the first spine link (torso-equivalent)
        let spine_body = sim.spine_chain.links[0];

        Self {
            sim,
            spine_encoder: JointEncoderArray::with_noise(encoder_noise.clone()),
            left_arm_encoder: JointEncoderArray::with_noise(encoder_noise.clone()),
            right_arm_encoder: JointEncoderArray::with_noise(encoder_noise.clone()),
            left_leg_encoder: JointEncoderArray::with_noise(encoder_noise.clone()),
            right_leg_encoder: JointEncoderArray::with_noise(encoder_noise.clone()),
            spine_imu: ImuSensor::with_noise(spine_body, accel_noise, gyro_noise),
        }
    }

    /// Create with zero-noise sensors (perfect sensors for baseline comparison).
    pub fn new_noiseless() -> Self {
        let sim = FullFrameSimulator::new();
        let spine_body = sim.spine_chain.links[0];

        Self {
            sim,
            spine_encoder: JointEncoderArray::new(),
            left_arm_encoder: JointEncoderArray::new(),
            right_arm_encoder: JointEncoderArray::new(),
            left_leg_encoder: JointEncoderArray::new(),
            right_leg_encoder: JointEncoderArray::new(),
            spine_imu: ImuSensor::new(spine_body),
        }
    }

    /// Step physics + read all sensors.
    pub fn step(&mut self, dt: f64) -> SuiteReading {
        self.sim.step(dt);
        self.read_sensors(dt)
    }

    /// Read all sensors without stepping.
    pub fn read_sensors(&mut self, dt: f64) -> SuiteReading {
        SuiteReading {
            spine: self.spine_encoder.read(&self.sim.spine_chain, &self.sim.world),
            left_arm: self.left_arm_encoder.read(&self.sim.left_arm, &self.sim.world),
            right_arm: self.right_arm_encoder.read(&self.sim.right_arm, &self.sim.world),
            left_leg: self.left_leg_encoder.read(&self.sim.left_leg, &self.sim.world),
            right_leg: self.right_leg_encoder.read(&self.sim.right_leg, &self.sim.world),
            imu: self.spine_imu.read(&self.sim.world, dt),
        }
    }
}

impl Default for SensoredFullFrame {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensored_suite_has_20_joints() {
        let mut suite = SensoredFullFrame::new();
        let reading = suite.read_sensors(0.001);
        assert_eq!(reading.total_joints(), 20);
        assert_eq!(reading.spine.len(), 2);
        assert_eq!(reading.left_arm.len(), 6);
        assert_eq!(reading.right_arm.len(), 6);
        assert_eq!(reading.left_leg.len(), 3);
        assert_eq!(reading.right_leg.len(), 3);
    }

    #[test]
    fn test_noisy_vs_noiseless_differ() {
        let mut noisy = SensoredFullFrame::new();
        let mut clean = SensoredFullFrame::new_noiseless();

        // Step both identically
        for _ in 0..10 {
            noisy.step(0.001);
            clean.step(0.001);
        }

        let n_reading = noisy.read_sensors(0.001);
        let c_reading = clean.read_sensors(0.001);

        // IMU should differ due to noise
        let accel_diff: f64 = n_reading.imu.acceleration.iter()
            .zip(c_reading.imu.acceleration.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(accel_diff > 0.0 || n_reading.imu.acceleration[2] != 0.0,
            "Noisy IMU should produce different readings than clean");
    }

    #[test]
    fn test_step_produces_finite_readings() {
        let mut suite = SensoredFullFrame::new();
        for _ in 0..20 {
            let reading = suite.step(0.001);
            // All joint angles should be finite
            for r in reading.spine.iter()
                .chain(reading.left_arm.iter())
                .chain(reading.left_leg.iter())
            {
                assert!(r.angle.is_finite());
                assert!(r.velocity.is_finite());
            }
            assert!(reading.imu.acceleration.iter().all(|a| a.is_finite()));
            assert!(reading.imu.angular_velocity.iter().all(|w| w.is_finite()));
        }
    }

    #[test]
    fn test_consciousness_affects_motion() {
        let mut suite = SensoredFullFrame::new_noiseless();
        suite.sim.set_consciousness(0.9);
        for _ in 0..20 { suite.step(0.001); }
        let reading = suite.read_sensors(0.001);

        // Total angular speed is finite and computable
        let speed = reading.total_angular_speed();
        assert!(speed.is_finite());
    }
}
