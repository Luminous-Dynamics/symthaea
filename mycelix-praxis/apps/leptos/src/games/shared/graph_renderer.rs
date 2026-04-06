// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SVG coordinate system renderer — axes, grid, labels.
//!
//! Used by all Tier 1 and Tier 2 games. Provides a responsive SVG
//! viewBox with configurable range, grid lines, and axis labels.

use leptos::prelude::*;

/// Configuration for the graph renderer.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphConfig {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub show_grid: bool,
    pub grid_step: f64,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            x_min: -10.0,
            x_max: 10.0,
            y_min: -10.0,
            y_max: 10.0,
            show_grid: true,
            grid_step: 1.0,
        }
    }
}

/// Render an SVG coordinate system with axes, grid, and labels.
/// Children are rendered inside the SVG (curves, points, etc.).
#[component]
pub fn GraphArea(
    config: GraphConfig,
    children: Children,
) -> impl IntoView {
    let width = config.x_max - config.x_min;
    let height = config.y_max - config.y_min;
    let view_box = format!("{} {} {} {}", config.x_min, config.y_min, width, height);

    // Grid lines
    let grid_lines = if config.show_grid {
        let mut lines = Vec::new();
        // Vertical grid lines
        let mut x = (config.x_min / config.grid_step).ceil() * config.grid_step;
        while x <= config.x_max {
            if (x.abs()) > 0.01 { // skip axis
                lines.push(view! {
                    <line
                        x1=x
                        y1=config.y_min
                        x2=x
                        y2=config.y_max
                        class="graph-grid"
                    />
                });
            }
            x += config.grid_step;
        }
        // Horizontal grid lines
        let mut y = (config.y_min / config.grid_step).ceil() * config.grid_step;
        while y <= config.y_max {
            if (y.abs()) > 0.01 {
                lines.push(view! {
                    <line
                        x1=config.x_min
                        y1=y
                        x2=config.x_max
                        y2=y
                        class="graph-grid"
                    />
                });
            }
            y += config.grid_step;
        }
        lines
    } else {
        Vec::new()
    };

    // Axis tick labels (every 2 units to avoid crowding)
    let tick_step = if width > 15.0 { 5.0 } else { 2.0 };
    let mut x_labels = Vec::new();
    let mut x = (config.x_min / tick_step).ceil() * tick_step;
    while x <= config.x_max {
        if x.abs() > 0.01 {
            let label = if x == x.floor() { format!("{}", x as i32) } else { format!("{:.1}", x) };
            x_labels.push(view! {
                <text x=x y=0.8 class="graph-tick-label">{label}</text>
            });
        }
        x += tick_step;
    }

    let mut y_labels = Vec::new();
    let mut y = (config.y_min / tick_step).ceil() * tick_step;
    while y <= config.y_max {
        if y.abs() > 0.01 {
            let label = if y == y.floor() { format!("{}", -(y as i32)) } else { format!("{:.1}", -y) };
            y_labels.push(view! {
                <text x=0.3 y=y class="graph-tick-label">{label}</text>
            });
        }
        y += tick_step;
    }

    view! {
        <svg
            viewBox=view_box
            class="game-graph"
            xmlns="http://www.w3.org/2000/svg"
        >
            // Grid
            {grid_lines}

            // Axes
            <line x1=config.x_min y1="0" x2=config.x_max y2="0" class="graph-axis" />
            <line x1="0" y1=config.y_min x2="0" y2=config.y_max class="graph-axis" />

            // Tick labels
            {x_labels}
            {y_labels}

            // Children (curves, points, etc.)
            {children()}
        </svg>
    }
}
