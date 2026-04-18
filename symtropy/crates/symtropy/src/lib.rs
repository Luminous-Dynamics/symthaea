// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! # symtropy
//!
//! The Symtropy distribution: an opinionated Bevy bundle for consciousness-
//! coupled simulation and game development. One dependency, four crates,
//! sensible defaults.
//!
//! ## Quick start
//!
//! ```toml
//! [dependencies]
//! symtropy = "0.1"
//! ```
//!
//! ```ignore
//! use symtropy::prelude::*;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(SymtropyScenePlugin::default())
//!         .add_plugins(SymtropyPhysicsPlugin::<3>::with_gravity([0.0, -9.81, 0.0]))
//!         .add_plugins(SymtropyDevConsolePlugin)
//!         .run();
//! }
//! ```
//!
//! ## What's bundled
//!
//! - [`bevy`] — game-engine framework (re-exported with default features).
//! - [`symtropy_bevy`] — N-D physics + Phi-coupling field; the `SymtropyPhysicsPlugin<D>`.
//! - [`symtropy_bevy_scene`] — scene scaffolding (camera, light, clear color
//!   defaults); the `SymtropyScenePlugin` + `fixed_camera()` helper.
//! - [`symtropy_devconsole`] — F1-toggleable dev console with Scene controls
//!   and (default) the Φ Inspector panel.
//!
//! ## Features
//!
//! | Feature | What it does | Default? |
//! |---|---|---|
//! | `devconsole-phi` | Surfaces the Φ Inspector panel in the devconsole. Pulls in `symtropy-bevy` (AGPL transitively). | yes |
//! | `low-level` | Re-exports `symtropy-physics` + `symtropy-math` for low-level access. | no |
//!
//! ## Naming
//!
//! `symtropy` is the meta-crate. The launcher application is
//! `symtropy-launcher`. Component crates are `symtropy-{bevy,bevy-scene,
//! devconsole,physics,math,...}`.

// Re-export bevy first so `use symtropy::prelude::*` brings in the standard
// Bevy prelude, then layer Symtropy on top.
pub use bevy;
pub use symtropy_bevy;
pub use symtropy_bevy_scene;
pub use symtropy_devconsole;

#[cfg(feature = "low-level")]
pub use symtropy_math as math;
#[cfg(feature = "low-level")]
pub use symtropy_physics as physics;

/// One-stop import for new projects:
///
/// ```ignore
/// use symtropy::prelude::*;
/// ```
///
/// Brings in Bevy's prelude plus the Symtropy-specific items most user code
/// reaches for: the three plugins, the camera helper, and the consciousness
/// inputs / physics handle types.
pub mod prelude {
    pub use bevy::prelude::*;
    pub use symtropy_bevy::{PhysicsBody, SymtropyPhysics, SymtropyPhysicsPlugin};
    pub use symtropy_bevy_scene::{fixed_camera, fixed_light, SymtropyScenePlugin};
    pub use symtropy_devconsole::{DevConsoleVisible, SymtropyDevConsolePlugin};
}
