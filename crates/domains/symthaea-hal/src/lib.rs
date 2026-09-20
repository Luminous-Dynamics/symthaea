// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-hal — Hardware Abstraction Layer
//!
//! Bridges Symthaea's physical sensor/actuator adapters to hardware on a
//! Raspberry Pi (or any Linux SBC with I2C). The current DMC21 backend targets
//! 2× PCA9685 PWM boards driving 21 hobby servos, with I2C sensors for
//! proprioception.
//!
//! ## Actuation semantics
//!
//! Physical command meaning must be explicit before calibration or PWM output.
//! The preferred cross-layer position representation is
//! [`PositionTargetRadiansCommand`]. [`NormalizedPositionCommand`] is narrower:
//! its `[-1,+1]` values are relative to the exact HAL calibration profile and
//! must not be confused with values normalized against another range (for
//! example humanoid morphology limits).
//!
//! The legacy `HumanoidCommand -> SafetyInterlock -> ServoOutput` runtime path
//! still exists while HAL-ACT-001 is being migrated. `HumanoidCommand` has
//! canonical normalized-torque semantics, so that legacy path is **not** the
//! strong physical actuation boundary and must not be used to justify a
//! production torque→position theorem.
//!
//! Target architecture:
//!
//! ```text
//! explicit position intent / admitted mode-tagged adaptation
//!   → PositionTargetRadiansCommand
//!   → exact current calibration + local safety/interlock
//!   → ServoOutput
//!   → PCA9685 boards (I2C → PWM → servos)
//!
//! physical sensors / encoders
//!   → typed observations with device/calibration/currentness provenance
//!   → physical-state verifier / estimator
//!   → cognition + actuation adaptation
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
//! ## Semantic command example
//!
//! ```rust
//! use symthaea_hal::PositionTargetRadiansCommand;
//!
//! let target = PositionTargetRadiansCommand::try_from_array([0.0; 21]).unwrap();
//! assert_eq!(target.values().len(), 21);
//! ```
//!
//! Constructing a typed command does not itself establish execution authority,
//! calibration qualification, physical-state currentness, trajectory safety,
//! or effect success.

#![deny(unsafe_code)]

pub mod actuation;
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

pub use actuation::{NormalizedPositionCommand, PositionTargetRadiansCommand};
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
