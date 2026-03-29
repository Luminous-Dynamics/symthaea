// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge Symthaea robotics platforms into the Symtropy game engine.
//!
//! Each robotic agent wraps an `ActiveInferenceAgent` (from symthaea-fep)
//! and a `MasterConsciousnessEquation`, connecting them to a physics body
//! in the game world with consciousness-gated motor authority.
//!
//! # Supported Platforms
//! The platform-specific physics simulators (flight, vehicle, humanoid, etc.)
//! live in symthaea's platform crates. This bridge provides the game-level
//! wrapper that makes them spawn-able entities with FEP behavior.

pub mod agent;
pub mod platform;

pub use agent::RoboticAgent;
pub use platform::PlatformType;
