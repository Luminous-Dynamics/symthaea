// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Standalone subset of `symtropy-robotics-bridge-core`: just the two
//! items symthaea itself imports, `PlatformType` and `JointSafetyAuthority`.
//! The full crate's `RoboticAgent`/`HapticOracle`/`MotorPlanner` and Bevy
//! integration live in the private symtropy monorepo.

pub mod platform;
pub mod safety;

pub use platform::PlatformType;
pub use safety::JointSafetyAuthority;
