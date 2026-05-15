// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Core types and traits for robotic agents in Symtropy.

pub mod agent;
pub mod motor;
pub mod platform;

pub use agent::{spawn_robot_body, RoboticAgent};
pub use motor::{MotorPlanner, UniformGainPlanner};
pub use platform::PlatformType;
