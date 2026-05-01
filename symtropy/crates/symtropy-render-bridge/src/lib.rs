// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ND physics → Bevy rendering bridge.
//!
//! Syncs `PhysicsWorld<D>` state to Bevy `Transform` components and provides
//! projection from ND to screen coordinates.
//!
//! # Modes
//! - **2D** (D=2): Direct 1:1 mapping to Bevy 2D sprites
//! - **3D** (D=3): Maps to Bevy 3D transforms (when `bevy_pbr` is available)
//! - **4D** (D=4): Cross-section slicing — slice 4D world with 3D hyperplane,
//!   render the intersection

pub mod projection;
pub mod sync;

pub use projection::{Projector, Projector2D, Projector3D, Projector4D};
pub use sync::{sync_physics_2d, sync_physics_3d, sync_physics_4d, PhysicsBody};
