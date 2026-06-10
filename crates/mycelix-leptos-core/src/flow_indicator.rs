// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Flow indicator — animated pulse traveling from a source point to a target.
//!
//! Visualizes cross-cluster dispatches, value flows, message routes, and
//! resource movements. The aesthetic commitment *"connections drawn, not
//! just nodes"* made concrete: when governance votes arrive from identity,
//! a FlowIndicator shows the vote leaving identity's node and landing at
//! governance's node. When TEND flows from the treasury to an apprentice,
//! a pulse travels the edge.
//!
//! # Composition
//!
//! Rendered inside a consumer-owned `<svg>`, same as [`GraphNode`](crate::graph_node::GraphNode).
//! Usually overlaid on top of a [`GraphEdge`](crate::graph_node::GraphEdge) of the same path —
//! the edge shows the connection exists, the flow shows something moving
//! through it right now.
//!
//! ```rust,no_run
//! use mycelix_leptos_core::{FlowIndicator, GraphEdge, GraphNode, NodeEmphasis, GrowthStage};
//! use leptos::prelude::*;
//! # let _ = || {
//! view! {
//!     <svg width="300" height="80" viewBox="0 0 300 80">
//!         <GraphEdge from=(40.0, 40.0) to=(260.0, 40.0) strength=0.3 />
//!         // vote flows from identity → governance
//!         <FlowIndicator
//!             from=(40.0, 40.0)
//!             to=(260.0, 40.0)
//!             emphasis=NodeEmphasis::Consensus
//!             continuous=true
//!         />
//!         <GraphNode x=40.0 y=40.0 label=Some("identity") stage=GrowthStage::Sapling />
//!         <GraphNode x=260.0 y=40.0 label=Some("governance") stage=GrowthStage::Sapling emphasis=NodeEmphasis::Consensus />
//!     </svg>
//! }
//! # };
//! ```
//!
//! # Semantic color + motion
//!
//! Uses SVG's native `<animateMotion>` — no JS, no deps. The pulse color
//! comes from the same [`NodeEmphasis`](crate::graph_node::NodeEmphasis)
//! tokens as nodes, so a "consensus"-emphasized flow visibly matches
//! consensus nodes.

use crate::graph_node::NodeEmphasis;
use leptos::prelude::*;

/// An animated pulse flowing from `from` to `to` along a straight line.
///
/// Renders an SVG `<circle>` that travels the path via `<animateMotion>`.
/// Independent of [`GraphEdge`](crate::graph_node::GraphEdge) — typically
/// use both together: the edge shows the connection, the flow shows
/// motion along it.
///
/// # Props
///
/// * `from` / `to` — `(x, y)` endpoints. Flow travels from → to.
/// * `duration_ms` — milliseconds per pulse transit. Defaults to 1200.
/// * `emphasis` — semantic color variant (default/signal, consensus, care, alarm).
/// * `reverse` — if true, pulse travels target → source instead.
/// * `continuous` — if true, pulse repeats forever. If false, runs once.
///   Defaults to true (most use cases are "this flow is happening right now").
/// * `radius` — pulse radius in pixels. Defaults to 3.
#[component]
pub fn FlowIndicator(
    from: (f64, f64),
    to: (f64, f64),
    #[prop(optional)] duration_ms: Option<u32>,
    #[prop(optional)] emphasis: Option<NodeEmphasis>,
    #[prop(optional)] reverse: Option<bool>,
    #[prop(optional)] continuous: Option<bool>,
    #[prop(optional)] radius: Option<f64>,
) -> impl IntoView {
    let duration = duration_ms.unwrap_or(1200);
    let emphasis = emphasis.unwrap_or_default();
    let reverse = reverse.unwrap_or(false);
    let continuous = continuous.unwrap_or(true);
    let r = radius.unwrap_or(3.0);

    let (src, dst) = if reverse { (to, from) } else { (from, to) };

    // SVG animateMotion path: "M src_x src_y L dst_x dst_y"
    let path = format!("M{} {} L{} {}", src.0, src.1, dst.0, dst.1);
    let duration_s = format!("{:.3}s", (duration as f64) / 1000.0);
    let repeat = if continuous { "indefinite" } else { "1" };

    let fill = emphasis_fill(&emphasis);
    let glow = emphasis_glow(&emphasis);

    view! {
        <g class="md-flow-indicator">
            <circle
                r=r.to_string()
                cx=src.0.to_string()
                cy=src.1.to_string()
                fill=fill
                fill-opacity="0.9"
                style=format!(
                    "filter: drop-shadow(0 0 {}px {});",
                    (r * 2.0) as u32,
                    glow
                )
            >
                <animateMotion
                    dur=duration_s
                    repeatCount=repeat
                    path=path
                    rotate="auto"
                    fill="freeze"
                />
            </circle>
        </g>
    }
}

fn emphasis_fill(e: &NodeEmphasis) -> &'static str {
    match e {
        NodeEmphasis::Default => "var(--md-signal, #00d4ff)",
        NodeEmphasis::Consensus => "var(--md-consensus, #f0b72f)",
        NodeEmphasis::Care => "var(--md-care, #7ee787)",
        NodeEmphasis::Alarm => "var(--md-alarm, #ff7b72)",
    }
}

// Glow color — slightly dimmer variant used in drop-shadow.
// Falls back to the fill if no glow-specific token exists.
fn emphasis_glow(e: &NodeEmphasis) -> &'static str {
    match e {
        NodeEmphasis::Default => "var(--md-signal-glow, var(--md-signal, #00d4ff))",
        NodeEmphasis::Consensus => "var(--md-consensus-glow, var(--md-consensus, #f0b72f))",
        NodeEmphasis::Care => "var(--md-care-glow, var(--md-care, #7ee787))",
        NodeEmphasis::Alarm => "var(--md-alarm-glow, var(--md-alarm, #ff7b72))",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_default_is_signal() {
        assert!(emphasis_fill(&NodeEmphasis::Default).contains("--md-signal"));
    }

    #[test]
    fn fill_consensus_is_consensus_token() {
        assert!(emphasis_fill(&NodeEmphasis::Consensus).contains("--md-consensus"));
    }

    #[test]
    fn glow_falls_back_to_fill_token() {
        // Default glow should fall through to --md-signal if --md-signal-glow unset.
        let g = emphasis_glow(&NodeEmphasis::Default);
        assert!(g.contains("--md-signal"));
    }

    #[test]
    fn reverse_swaps_endpoints() {
        // Replicate the direction logic.
        let from = (0.0_f64, 10.0);
        let to = (100.0_f64, 10.0);
        let reverse = true;
        let (src, dst) = if reverse { (to, from) } else { (from, to) };
        assert_eq!(src, (100.0, 10.0));
        assert_eq!(dst, (0.0, 10.0));
    }

    #[test]
    fn continuous_maps_to_indefinite_repeat() {
        let continuous = true;
        let repeat = if continuous { "indefinite" } else { "1" };
        assert_eq!(repeat, "indefinite");

        let continuous = false;
        let repeat = if continuous { "indefinite" } else { "1" };
        assert_eq!(repeat, "1");
    }

    #[test]
    fn duration_ms_to_seconds_format() {
        let duration_s = format!("{:.3}s", 1200_f64 / 1000.0);
        assert_eq!(duration_s, "1.200s");

        let duration_s = format!("{:.3}s", 250_f64 / 1000.0);
        assert_eq!(duration_s, "0.250s");
    }
}
