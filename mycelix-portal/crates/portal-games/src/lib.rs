// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interactive educational games for Mycelix Portal.
//!
//! Ported from mycelix-edunet/apps/leptos/src/games/.
//! All games use SVG rendering via Leptos components.

pub mod graph_renderer;
pub mod math;
pub mod physics;

pub use graph_renderer::{GraphArea, GraphConfig};
