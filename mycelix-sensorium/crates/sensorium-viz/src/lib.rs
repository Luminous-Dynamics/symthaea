// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reactive SVG visualization components for Mycelix Sensorium.
//!
//! Pure Rust replacements for D3.js/Recharts. Each component accepts
//! Leptos signals as data sources and renders reactive SVG elements.
//!
//! # Components
//!
//! - [`BarChart`] — Horizontal or vertical bar chart with labels
//! - [`LineChart`] — Multi-series line chart with optional area fill
//! - [`PieChart`] — Donut/pie chart with segments
//! - [`VoteTally`] — Governance-specific: for/against/abstain horizontal bar

pub mod bar_chart;
pub mod force_graph;
pub mod katex_bridge;
pub mod line_chart;
pub mod pie_chart;
pub mod vote_tally;

pub use bar_chart::BarChart;
pub use force_graph::ForceGraph;
pub use katex_bridge::Math;
pub use line_chart::LineChart;
pub use pie_chart::PieChart;
pub use vote_tally::VoteTallyBar;
