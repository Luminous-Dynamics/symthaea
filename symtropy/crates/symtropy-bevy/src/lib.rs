// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symtropy-bevy
//!
//! Drop-in Bevy plugin for Phi-coupled N-dimensional physics.
//!
//! ```ignore
//! use symtropy_bevy::SymtropyPhysicsPlugin;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(SymtropyPhysicsPlugin::<2>::default()) // 2D physics
//!         .run();
//! }
//! ```
//!
//! The plugin provides:
//! - `PhysicsWorld<D>` + `ConsciousnessField<D>` as a combined Bevy resource
//! - Physics step system in `FixedUpdate` with Phi-coupling via `PhysicsCallback`
//! - Transform sync system that writes body positions to Bevy `Transform`
//! - Debug gizmo rendering (collider outlines, contact points, safety tiers)

pub mod plugin;
#[cfg(feature = "debug-gizmos")]
pub mod debug;

pub use plugin::{SymtropyPhysicsPlugin, SymtropyPhysics, PhysicsBody};
