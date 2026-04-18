// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Graph node + edge primitives — the relational-connection layer.
//!
//! Building blocks for kinship graphs, reciprocity networks, credential
//! endorsement webs, and orbital-traffic plots. Same component, different
//! data. Ties into Indlela's [`GrowthStage`](crate::indlela::GrowthStage)
//! so a node's visual size reflects its maturity/depth in whatever graph
//! it lives in.
//!
//! # Composition
//!
//! Nodes + edges are rendered inside a consumer-owned `<svg>` element.
//! This keeps layout flexible — force-directed, fixed-grid, radial,
//! orbital — without this crate dictating the algorithm.
//!
//! ```rust,no_run
//! use mycelix_leptos_core::{GraphNode, GraphEdge, GrowthStage};
//! use leptos::prelude::*;
//! # let _ = || {
//! view! {
//!     <svg width="400" height="240" viewBox="0 0 400 240">
//!         <GraphEdge from=(80.0, 120.0) to=(200.0, 60.0) />
//!         <GraphEdge from=(80.0, 120.0) to=(200.0, 180.0) />
//!         <GraphEdge from=(200.0, 60.0) to=(320.0, 120.0) />
//!         <GraphEdge from=(200.0, 180.0) to=(320.0, 120.0) />
//!         <GraphNode x=80.0 y=120.0 label=Some("Kagiso") stage=GrowthStage::Tree />
//!         <GraphNode x=200.0 y=60.0 label=Some("Water council") stage=GrowthStage::Sapling />
//!         <GraphNode x=200.0 y=180.0 label=Some("Family circle") stage=GrowthStage::Tree />
//!         <GraphNode x=320.0 y=120.0 label=Some("Regional") stage=GrowthStage::Sprout />
//!     </svg>
//! }
//! # };
//! ```
//!
//! # Semantic CSS variables
//!
//! - `--md-node-fill` (fallback: signal cyan) — node interior
//! - `--md-node-stroke` (fallback: foreground) — node border + edge line
//! - `--md-node-label` (fallback: muted foreground) — label text
//! - `--md-mono` (fallback: ui-monospace) — label typography
//!
//! Consumers can further specialize via the `emphasis` prop (see below)
//! to flag a "consensus"-colored node or an "alarm"-colored one.

use crate::indlela::GrowthStage;
use leptos::prelude::*;

/// Visual emphasis of a node — maps to a semantic color token.
///
/// `Default` uses `--md-node-fill` (signal cyan). `Consensus`/`Care`/`Alarm`
/// reach for the cluster theme's per-semantic tokens. Unknown tokens fall
/// back to the default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NodeEmphasis {
    /// Signal / live-data color (default).
    #[default]
    Default,
    /// Consensus / verified / governance-approved.
    Consensus,
    /// Care / growth / kinship.
    Care,
    /// Alarm / urgent / flagged.
    Alarm,
}

impl NodeEmphasis {
    fn fill_var(&self) -> &'static str {
        match self {
            Self::Default => "var(--md-node-fill, var(--md-signal, #00d4ff))",
            Self::Consensus => "var(--md-consensus, #f0b72f)",
            Self::Care => "var(--md-care, #7ee787)",
            Self::Alarm => "var(--md-alarm, #ff7b72)",
        }
    }
}

/// A positioned node in a graph.
///
/// Rendered as `<circle>` + optional `<text>` label. Size comes from
/// [`GrowthStage::radius`]; opacity from [`GrowthStage::opacity`] unless
/// `opacity` is explicitly provided.
///
/// # Props
///
/// * `x` / `y` — position in the parent `<svg>` viewport (not normalized).
/// * `stage` — Indlela growth stage controlling size + default opacity.
/// * `label` — optional text rendered below the node.
/// * `emphasis` — semantic color variant (default signal, consensus, care, alarm).
/// * `opacity` — override the stage's default opacity.
/// * `label_above` — render label above the node instead of below.
#[component]
pub fn GraphNode(
    x: f64,
    y: f64,
    #[prop(optional)] stage: Option<GrowthStage>,
    #[prop(optional)] label: Option<&'static str>,
    #[prop(optional)] emphasis: Option<NodeEmphasis>,
    #[prop(optional)] opacity: Option<f64>,
    #[prop(optional)] label_above: Option<bool>,
) -> impl IntoView {
    let stage = stage.unwrap_or(GrowthStage::Sapling);
    let radius = stage.radius();
    let node_opacity = opacity.unwrap_or_else(|| stage.opacity());
    let emphasis = emphasis.unwrap_or_default();
    let fill = emphasis.fill_var();
    let stroke = "var(--md-node-stroke, var(--md-fg, #c9d1d9))";

    let label_y = if label_above.unwrap_or(false) {
        y - radius - 6.0
    } else {
        y + radius + 12.0
    };
    let baseline = if label_above.unwrap_or(false) {
        "alphabetic"
    } else {
        "hanging"
    };

    view! {
        <g class=format!("md-graph-node {}", stage.css_class())>
            <circle
                cx=x.to_string()
                cy=y.to_string()
                r=radius.to_string()
                fill=fill
                fill-opacity=node_opacity.to_string()
                stroke=stroke
                stroke-width="1"
                stroke-opacity="0.8"
            />
            {label.map(|l| view! {
                <text
                    x=x.to_string()
                    y=label_y.to_string()
                    text-anchor="middle"
                    dominant-baseline=baseline
                    fill="var(--md-node-label, var(--md-fg-muted, #6e7681))"
                    style="font-family: var(--md-mono, ui-monospace, 'JetBrains Mono', monospace); \
                           font-size: 10px; letter-spacing: 0.04em;"
                >
                    {l}
                </text>
            })}
        </g>
    }
}

/// An edge connecting two points.
///
/// Rendered as a `<line>`. Independent of `GraphNode` so consumers control
/// layering — typically render edges *before* nodes so nodes draw on top.
///
/// # Props
///
/// * `from` / `to` — `(x, y)` endpoints in the parent `<svg>` viewport.
/// * `strength` — 0.0..=1.0 strength of the connection (controls opacity).
///   Defaults to 0.6.
/// * `dashed` — render as dashed line. Default false.
#[component]
pub fn GraphEdge(
    from: (f64, f64),
    to: (f64, f64),
    #[prop(optional)] strength: Option<f64>,
    #[prop(optional)] dashed: Option<bool>,
) -> impl IntoView {
    let strength = strength.unwrap_or(0.6).clamp(0.0, 1.0);
    let dash = if dashed.unwrap_or(false) { "2 3" } else { "0" };

    view! {
        <line
            class="md-graph-edge"
            x1=from.0.to_string()
            y1=from.1.to_string()
            x2=to.0.to_string()
            y2=to.1.to_string()
            stroke="var(--md-node-stroke, var(--md-fg, #c9d1d9))"
            stroke-opacity=strength.to_string()
            stroke-width="1"
            stroke-dasharray=dash
        />
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emphasis_default_uses_signal_token() {
        let e = NodeEmphasis::Default;
        assert!(e.fill_var().contains("--md-signal"));
    }

    #[test]
    fn emphasis_consensus_uses_consensus_token() {
        let e = NodeEmphasis::Consensus;
        assert!(e.fill_var().contains("--md-consensus"));
    }

    #[test]
    fn emphasis_care_uses_care_token() {
        assert!(NodeEmphasis::Care.fill_var().contains("--md-care"));
    }

    #[test]
    fn emphasis_alarm_uses_alarm_token() {
        assert!(NodeEmphasis::Alarm.fill_var().contains("--md-alarm"));
    }

    #[test]
    fn edge_strength_clamps_to_0_1() {
        // Replicate the clamp logic.
        let s: f64 = 1.5;
        assert_eq!(s.clamp(0.0, 1.0), 1.0);
        let s: f64 = -0.2;
        assert_eq!(s.clamp(0.0, 1.0), 0.0);
        let s: f64 = 0.3;
        assert_eq!(s.clamp(0.0, 1.0), 0.3);
    }

    #[test]
    fn growth_stage_drives_node_radius() {
        // Confirm Indlela's stages produce increasing radii.
        assert!(GrowthStage::Seed.radius() < GrowthStage::Sprout.radius());
        assert!(GrowthStage::Sprout.radius() < GrowthStage::Sapling.radius());
        assert!(GrowthStage::Sapling.radius() < GrowthStage::Tree.radius());
    }
}
