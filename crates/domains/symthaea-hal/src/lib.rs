// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-hal — Hardware Abstraction Layer
//!
//! Bridges Symthaea's cognitive loop to physical servos and sensors on a
//! Raspberry Pi (or any Linux SBC with I2C). Targets 2× PCA9685 PWM boards
//! driving 21 hobby servos, with I2C sensors (IMU) for proprioception.
//!
//! ## Architecture
//!
//! ```text
//! CognitiveLoop
//!   → HumanoidCommand (21 torques, ±1)
//!   → SafetyInterlock::filter_command()   ← watchdog, e-stop, bounds
//!   → ServoOutput::apply()                ← calibration, slew-rate limit
//!   → Pca9685 boards (I2C → PWM → servos)
//!
//! IMU sensor (I2C)
//!   → EmbeddedSensor<I2C, Mpu6050Decoder>
//!   → HalSensorAdapter::read_raw()
//!   → SensorInput bridge (in main crate)
//!   → CognitiveLoop perception
//! ```
//!
//! ## Board Layout
//!
//! ```text
//! Board 0 (0x40): joints  0–15  (abdomen + legs + right_shoulder1)
//! Board 1 (0x41): joints 16–20  (remaining arms, channels 0–4)
//! ```
//!
//! ## Feature Flags
//!
//! - `linux`: Enables `linux-embedded-hal` for real I2C devices (`/dev/i2c-*`).
//!   Without this feature, the crate compiles and tests with [`MockI2cBus`].
//! - `calibrate`: Enables the `hal-calibrate` CLI binary (adds `clap` dep).
//!
//! ## Usage
//!
//! ```rust,no_run
//! use symthaea_hal::{
//!     CalibrationProfile, SafetyInterlock, ServoOutput,
//!     mock::MockI2cBus,
//! };
//! use symthaea_humanoid::types::HumanoidCommand;
//!
//! // Create servo output with mock buses (testing)
//! let cal = CalibrationProfile::default_21();
//! let mut servo = ServoOutput::new(MockI2cBus::new(), MockI2cBus::new(), cal);
//! servo.init(50.0).unwrap();
//! servo.enable().unwrap();
//!
//! // Safety interlock filters commands
//! let mut safety = SafetyInterlock::new();
//! let cmd = HumanoidCommand::zero();
//! let safe_cmd = safety.filter_command(&cmd).unwrap();
//! servo.apply(&safe_cmd).unwrap();
//! ```

#![deny(unsafe_code)]

pub mod calibration;
pub mod error;
pub mod gpio_estop;
pub mod imu;
pub mod ina219;
pub mod interlock;
pub mod mock;
pub mod motor_safety;
pub mod pca9685;
pub mod recording;
pub mod runtime;
pub mod sensor;
pub mod servo;

// ── Public re-exports ────────────────────────────────────────────────

pub use calibration::{CalibrationProfile, JointCalibration};
pub use error::{HalError, HalResult};
pub use gpio_estop::{EstopPoller, GpioEstop};
pub use imu::{ComplementaryFilter, Mpu6050Decoder};
pub use ina219::Ina219Decoder;
pub use interlock::{SafetyConfig, SafetyInterlock};
pub use motor_safety::MotorSafetyLevel;
pub use pca9685::Pca9685;
pub use recording::{RecordingAdapter, ReplayAdapter, SensorRecording};
pub use runtime::{
    AngleMonitor, CurrentMonitor, HalRuntime, HalRuntimeBuilder, HealthStatus, RuntimeTelemetry,
};
pub use sensor::{EmbeddedSensor, HalSensorAdapter, SensorDecoder};
pub use servo::ServoOutput;

/// Re-export `embedded_hal::i2c::I2c` so downstream crates don't need a direct dependency.
pub use embedded_hal::i2c::I2c as I2cBus;

/// Re-export shared-bus wrappers for single-bus multi-device setups.
pub use embedded_hal_bus::i2c::{MutexDevice, RefCellDevice};
