// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Shared data types, geodetic math, geometry generation, and constants
//! for Terra Atlas renderers (Leptos WebGL, Bevy wgpu).
//!
//! All math uses raw `[f32; N]` arrays to avoid dependency collisions
//! between rendering backends (web-sys, glam/nalgebra).

pub mod constants;
pub mod data;
pub mod economics;
pub mod energy_trading;
pub mod geo;
pub mod geometry;
pub mod math;
pub mod picking;
pub mod timeline;
pub mod types;

pub use types::*;
