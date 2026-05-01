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
//!
//! # Crate relationship
//!
//! This crate layers on top of [`symtropy-bevy-core`], which provides
//! permissive-licensed physics-only Bevy integration. The types
//! [`PhysicsBody`], [`BevyPhysics`], [`BevyPhysicsPlugin`] are re-exported
//! from bevy-core so users can drop down to no-coupling physics without
//! switching crates.

pub mod biometrics;
#[cfg(feature = "debug-gizmos")]
pub mod debug;
pub mod macro_bridge;
pub mod plugin;

pub use biometrics::{biometric_to_phi_system, PlayerBiometrics};
pub use macro_bridge::{apply_macro_modifiers_system, MacroWorldState};
pub use plugin::{SymtropyPhysics, SymtropyPhysicsPlugin};

/// Re-exported from [`symtropy-bevy-core`]. Identical type used by both
/// crates, so an entity with `PhysicsBody` works with either the permissive
/// or the Phi-coupled plugin.
pub use symtropy_bevy_core::PhysicsBody;

/// Re-exported from [`symtropy-bevy-core`] for users who want a
/// physics-only path inside a project that also uses the Phi-coupled
/// plugin (e.g., some scenes take the full consciousness integration,
/// others don't).
pub use symtropy_bevy_core::{
    BevyPhysics, BevyPhysicsCorePlugin, BevyPhysicsPlugin, NoCouplingResource,
};
