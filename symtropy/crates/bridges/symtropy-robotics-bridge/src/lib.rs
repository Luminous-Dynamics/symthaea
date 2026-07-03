// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! robotics-bridge: links high-level CognitiveBrain commands to low-level
//! PD motor drives and robotic kinematic chains.

pub mod agent;
pub mod platform;

pub use agent::{RoboticAgent, RoboticAgentTag, RoboticBrainPlugin, RoboticJoint, spawn_robot};
/// The `-core` crate's `RoboticAgent` trait, re-exported under a distinct
/// name so it doesn't collide with this crate's concrete `agent::RoboticAgent`
/// struct. Bring this into scope (`use symtropy_robotics_bridge::RoboticAgentTrait;`)
/// to call `.tick()` / `.phi()` / `.bottleneck()` / `.can_act()` /
/// `.tick_motor_commands()` on an `agent::RoboticAgent`.
pub use symtropy_robotics_bridge_core::RoboticAgent as RoboticAgentTrait;
