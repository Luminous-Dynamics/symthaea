// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! robotics-bridge: links high-level CognitiveBrain commands to low-level
//! PD motor drives and robotic kinematic chains.

pub mod agent;

pub use agent::{RoboticAgent, RoboticBrainPlugin, RoboticJoint, spawn_robot};
