// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interactive educational SVG explorations for Mycelix Sensorium.
//!
//! NOT Bevy games — reactive Leptos SVG components.
//! Sliders → math → SVG shapes. Zero GPU. ~2,300 LOC.
//!
//! - **math**: unit circle, parabola, stats explorer, tangent line, analytical geometry
//! - **physics**: projectile motion, circuits (Ohm's law)
//! - **graph_renderer**: shared SVG coordinate system
//! - **progress**: mastery reporting interface

pub mod graph_renderer;
pub mod math;
pub mod physics;
pub mod progress;
pub mod social_proof;

pub use graph_renderer::{GraphArea, GraphConfig};
pub use progress::{ProgressStatus, ProgressStore, use_set_progress, provide_progress_store};
