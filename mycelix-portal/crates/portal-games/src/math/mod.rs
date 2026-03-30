// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Math games — interactive SVG explorations.
//!
//! Ported from mycelix-edunet/apps/leptos/src/games/math/.
//! Each game implements a standalone Leptos component.
//!
//! Games:
//! - unit_circle: Trigonometry (sin, cos, tan, quadrants)
//! - parabola: Function exploration (y = a(x-p)² + q)
//! - stats_explorer: Data analysis and distributions
//! - tangent_line: Calculus (derivative as slope)
//! - analytical: Coordinate geometry, distance, midpoint

// Games will be ported individually from EduNet's 10K LOC game modules.
// Each is a self-contained Leptos component with SVG rendering.
// The graph_renderer::GraphArea component provides the shared coordinate system.
//
// To port a game:
// 1. Copy the .rs file from mycelix-edunet/apps/leptos/src/games/math/
// 2. Update imports (use crate::graph_renderer::*)
// 3. Adapt to portal design system (CSS class names)
