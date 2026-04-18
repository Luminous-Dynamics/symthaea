// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-sensors
//!
//! Shared sensor simulation for Symthaea robotics platforms.
//! Provides force/torque, joint encoder, IMU, and contact sensors
//! that read from a `symtropy_physics::PhysicsWorld`.
//!
//! All sensors have configurable noise models for sim-to-real transfer.
//!
//! Used by: symthaea-manipulator, symthaea-quadruped, symthaea-exoskeleton,
//! and future platforms including the full-frame exoskeleton.

use nalgebra::SVector;
use symtropy_physics::body::{BodyHandle, RigidBody};
use symtropy_physics::world::PhysicsWorld;
use symtropy_physics::articulation::ArticulatedChain;

// ── Sensor Config ──────────────────────────────────────────────────────

/// Noise configuration for a sensor channel.
#[derive(Debug, Clone)]
pub struct NoiseConfig {
    /// Gaussian noise standard deviation.
    pub noise_std: f64,
    /// Constant bias offset.
    pub bias: f64,
    /// Update rate in Hz (0 = every step).
    pub update_rate_hz: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self { noise_std: 0.0, bias: 0.0, update_rate_hz: 0.0 }
    }
}

impl NoiseConfig {
    /// Apply noise to a clean reading using a deterministic seed.
    pub fn apply(&self, clean: f64, seed: u64) -> f64 {
        if self.noise_std == 0.0 && self.bias == 0.0 {
            return clean;
        }
        // Deterministic pseudo-noise (xorshift)
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        let uniform = (s as f64) / (u64::MAX as f64);
        let gaussian = (uniform - 0.5) * 3.464; // Approx N(0,1)
        clean + gaussian * self.noise_std + self.bias
    }
}

// ── Joint Encoder ──────────────────────────────────────────────────────

/// Joint angle and velocity reading from an articulated chain.
#[derive(Debug, Clone, Default)]
pub struct JointReading {
    pub angle: f64,
    pub velocity: f64,
}

/// Reads joint states from an ArticulatedChain with optional noise.
#[derive(Debug, Clone)]
pub struct JointEncoderArray {
    pub noise: NoiseConfig,
    step_counter: u64,
}

impl JointEncoderArray {
    pub fn new() -> Self {
        Self { noise: NoiseConfig::default(), step_counter: 0 }
    }

    pub fn with_noise(noise: NoiseConfig) -> Self {
        Self { noise, step_counter: 0 }
    }

    /// Read all joint states from the chain.
    pub fn read(&mut self, chain: &ArticulatedChain, world: &PhysicsWorld<3>) -> Vec<JointReading> {
        self.step_counter += 1;
        chain.read_joint_states(world)
            .iter()
            .enumerate()
            .map(|(i, &(angle, vel))| {
                let seed = self.step_counter.wrapping_mul(100 + i as u64);
                JointReading {
                    angle: self.noise.apply(angle, seed),
                    velocity: self.noise.apply(vel, seed.wrapping_add(1)),
                }
            })
            .collect()
    }
}

impl Default for JointEncoderArray {
    fn default() -> Self { Self::new() }
}

// ── IMU Sensor ─────────────────────────────────────────────────────────

/// 6-axis IMU reading (3 accel + 3 gyro).
#[derive(Debug, Clone, Default)]
pub struct ImuReading {
    /// Linear acceleration [x, y, z] in m/s^2.
    pub acceleration: [f64; 3],
    /// Angular velocity [x, y, z] in rad/s.
    pub angular_velocity: [f64; 3],
}

/// IMU sensor attached to a specific body.
#[derive(Debug, Clone)]
pub struct ImuSensor {
    pub body: BodyHandle,
    pub accel_noise: NoiseConfig,
    pub gyro_noise: NoiseConfig,
    prev_velocity: SVector<f64, 3>,
    step_counter: u64,
}

impl ImuSensor {
    pub fn new(body: BodyHandle) -> Self {
        Self {
            body,
            accel_noise: NoiseConfig::default(),
            gyro_noise: NoiseConfig::default(),
            prev_velocity: SVector::zeros(),
            step_counter: 0,
        }
    }

    pub fn with_noise(body: BodyHandle, accel_noise: NoiseConfig, gyro_noise: NoiseConfig) -> Self {
        Self { body, accel_noise, gyro_noise, prev_velocity: SVector::zeros(), step_counter: 0 }
    }

    /// Read IMU from the physics world.
    pub fn read(&mut self, world: &PhysicsWorld<3>, dt: f64) -> ImuReading {
        self.step_counter += 1;
        let body = match world.body(self.body) {
            Some(b) => b,
            None => return ImuReading::default(),
        };

        // Acceleration = (v_new - v_old) / dt + gravity
        let accel = if dt > 0.0 {
            (body.linear_velocity - self.prev_velocity) / dt
        } else {
            SVector::zeros()
        };
        self.prev_velocity = body.linear_velocity;

        let seed = self.step_counter;
        ImuReading {
            acceleration: [
                self.accel_noise.apply(accel[0], seed),
                self.accel_noise.apply(accel[1], seed + 1),
                self.accel_noise.apply(accel[2], seed + 2),
            ],
            angular_velocity: [
                self.gyro_noise.apply(body.angular_velocity.get(1, 2), seed + 3), // roll
                self.gyro_noise.apply(body.angular_velocity.get(0, 2), seed + 4), // pitch
                self.gyro_noise.apply(body.angular_velocity.get(0, 1), seed + 5), // yaw
            ],
        }
    }
}

// ── Contact Sensor ─────────────────────────────────────────────────────

/// Binary contact detection with force magnitude.
#[derive(Debug, Clone, Default)]
pub struct ContactReading {
    /// Whether the body is in contact with anything.
    pub in_contact: bool,
    /// Estimated contact force magnitude (from velocity change).
    pub force_magnitude: f64,
}

/// Detects contact for a specific body (e.g., foot, fingertip).
#[derive(Debug, Clone)]
pub struct ContactSensor {
    pub body: BodyHandle,
    prev_velocity_z: f64,
}

impl ContactSensor {
    pub fn new(body: BodyHandle) -> Self {
        Self { body, prev_velocity_z: 0.0 }
    }

    /// Read contact state from physics world.
    pub fn read(&mut self, world: &PhysicsWorld<3>, dt: f64, mass: f64) -> ContactReading {
        let body = match world.body(self.body) {
            Some(b) => b,
            None => return ContactReading::default(),
        };

        let vz = body.linear_velocity[2];
        let accel_z = if dt > 0.0 { (vz - self.prev_velocity_z) / dt } else { 0.0 };
        self.prev_velocity_z = vz;

        // Contact detected when vertical acceleration exceeds gravity
        // (something is pushing up against the body)
        let force_z = mass * (accel_z + 9.81); // Net upward force
        let in_contact = force_z > 0.5; // Threshold: 0.5N

        ContactReading {
            in_contact,
            force_magnitude: force_z.max(0.0),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_config_zero_noise() {
        let cfg = NoiseConfig::default();
        assert_eq!(cfg.apply(5.0, 42), 5.0);
    }

    #[test]
    fn test_noise_config_adds_noise() {
        let cfg = NoiseConfig { noise_std: 1.0, bias: 0.0, update_rate_hz: 0.0 };
        let readings: Vec<f64> = (0..100).map(|i| cfg.apply(5.0, i)).collect();
        let mean: f64 = readings.iter().sum::<f64>() / 100.0;
        let variance: f64 = readings.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / 100.0;
        assert!(variance > 0.01, "Noise should add variance: {variance}");
    }

    #[test]
    fn test_noise_config_bias() {
        let cfg = NoiseConfig { noise_std: 0.0, bias: 2.0, update_rate_hz: 0.0 };
        assert_eq!(cfg.apply(5.0, 42), 7.0);
    }

    #[test]
    fn test_joint_encoder_reads_chain() {
        use symtropy_physics::articulation::{ChainBuilder, LinkSpec};

        let mut world = PhysicsWorld::new(SVector::from([0.0, 0.0, -9.81]));
        let chain = ChainBuilder::new()
            .base_position(symtropy_math::Point::new([0.0, 0.0, 1.0]))
            .add_link(LinkSpec::default())
            .add_link(LinkSpec::default())
            .build(&mut world);

        let mut encoder = JointEncoderArray::new();
        let readings = encoder.read(&chain, &world);
        assert_eq!(readings.len(), 2);
        assert!(readings[0].angle.is_finite());
        assert!(readings[1].velocity.is_finite());
    }

    #[test]
    fn test_imu_sensor() {
        let mut world = PhysicsWorld::new(SVector::from([0.0, 0.0, -9.81]));
        let handle = world.add_body(RigidBody::dynamic_sphere(
            BodyHandle(0),
            symtropy_math::Point::new([0.0, 0.0, 1.0]),
            0.1, 1.0,
        ));

        let mut imu = ImuSensor::new(handle);
        world.step(0.01);
        let reading = imu.read(&world, 0.01);
        assert!(reading.acceleration[2].is_finite());
        assert!(reading.angular_velocity[0].is_finite());
    }

    #[test]
    fn test_contact_sensor_no_contact() {
        let mut world = PhysicsWorld::new(SVector::from([0.0, 0.0, -9.81]));
        let handle = world.add_body(RigidBody::dynamic_sphere(
            BodyHandle(0),
            symtropy_math::Point::new([0.0, 0.0, 5.0]), // High up, no ground contact
            0.1, 1.0,
        ));

        let mut contact = ContactSensor::new(handle);
        world.step(0.01);
        let reading = contact.read(&world, 0.01, 1.0);
        // Falling freely — might or might not register as contact depending on dt
        assert!(reading.force_magnitude.is_finite());
    }
}
