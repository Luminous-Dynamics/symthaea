// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Aesthetic showcase — reference implementation of the Mycelix design language.
//!
//! Single component that exercises every shared primitive on one page with
//! mock data, demonstrating the intended visual grammar for a Type 1 civ
//! substrate: info density, temporal motion, visible connections, semantic
//! color tokens, monospace-infrastructure typography.
//!
//! # Purpose
//!
//! Three roles:
//! 1. **Documentation-as-code.** Copy-paste reference for cluster frontends
//!    adopting the design system. Shows canonical usage of every primitive.
//! 2. **Visual regression anchor.** If the aesthetic drifts, this page
//!    breaks first.
//! 3. **Onboarding surface.** A new frontend author points their dev page
//!    at `<Showcase />` to see what "cohesive" looks like.
//!
//! # Composition
//!
//! One reactive `<Showcase />` component. Uses `RwSignal` internally for
//! the mock telemetry streams so the TelemetryLines actually move. Relies
//! on `gloo_timers` for the tick loop (already a dep of this crate).
//!
//! # Usage
//!
//! ```rust,no_run
//! use mycelix_leptos_core::Showcase;
//! use leptos::prelude::*;
//! # let _ = || {
//! view! { <Showcase /> }
//! # };
//! ```
//!
//! Drop it onto any dev page — no other wiring required. The page styles
//! itself via inline CSS variables so it renders standalone.

use crate::flow_indicator::FlowIndicator;
use crate::graph_node::{GraphEdge, GraphNode, NodeEmphasis};
use crate::indlela::GrowthStage;
use crate::stat_card::StatCard;
use crate::telemetry_line::TelemetryLine;
use gloo_timers::future::TimeoutFuture;
use leptos::prelude::*;
use leptos::task::spawn_local;

/// Aesthetic showcase page — all primitives, one cohesive scene.
///
/// Emulates the morning moment of the "Kagiso" demo storyboard: a caretaker
/// surveying her regional watershed with live telemetry, kinship network,
/// and governance flows visible on one dashboard.
#[component]
pub fn Showcase() -> impl IntoView {
    // Reactive mock-data streams. They tick once per second via gloo_timers
    // so the TelemetryLines read as live.
    let flow_rate = RwSignal::new(vec![
        1.2_f64, 1.3, 1.2, 1.5, 1.4, 1.6, 1.5, 1.7, 1.8, 1.6, 1.9, 2.0,
    ]);
    let proposal_rate = RwSignal::new(vec![3.0_f64, 4.0, 2.0, 5.0, 6.0, 4.0, 3.0, 7.0, 5.0, 8.0]);
    let tend_flow = RwSignal::new(vec![
        12.0_f64, 18.0, 15.0, 22.0, 19.0, 25.0, 28.0, 24.0, 30.0, 27.0,
    ]);

    spawn_local(async move {
        // Tick each stream forward with a small random walk so the
        // sparklines visibly move. This is a showcase, not a simulation —
        // deterministic enough to read, alive enough to feel live.
        let mut phase = 0_u32;
        loop {
            TimeoutFuture::new(1200).await;
            phase = phase.wrapping_add(1);

            let jitter = |seed: u32, base: f64, amp: f64| -> f64 {
                let s = (seed.wrapping_mul(2654435761)) as i32 as f64 / i32::MAX as f64;
                (base + s * amp).max(0.0)
            };

            flow_rate.update(|v| {
                v.push(jitter(phase.wrapping_mul(13), 1.6, 0.6));
                if v.len() > 40 {
                    v.remove(0);
                }
            });
            proposal_rate.update(|v| {
                v.push(jitter(phase.wrapping_mul(17), 5.0, 4.0));
                if v.len() > 40 {
                    v.remove(0);
                }
            });
            tend_flow.update(|v| {
                v.push(jitter(phase.wrapping_mul(23), 22.0, 10.0));
                if v.len() > 40 {
                    v.remove(0);
                }
            });
        }
    });

    view! {
        <div style=SHOWCASE_STYLE>
            <header style=HEADER_STYLE>
                <div style="display: flex; align-items: baseline; gap: 16px;">
                    <span style="font-family: var(--md-mono, ui-monospace, 'JetBrains Mono', monospace); \
                                 font-size: 10px; color: var(--md-signal, #00d4ff); \
                                 letter-spacing: 0.2em;">
                        "MYCELIX / SUBSTRATE"
                    </span>
                    <span style="font-family: var(--md-mono, ui-monospace, monospace); \
                                 font-size: 10px; color: var(--md-fg-muted, #6e7681); \
                                 letter-spacing: 0.15em;">
                        "01 · 7° 47' 32 E · " {move || format!("PHASE {:02}", 14)}
                    </span>
                </div>
                <h1 style=TITLE_STYLE>
                    "Regional watershed · morning survey"
                </h1>
            </header>

            // KPI row — static numbers
            <section style=GRID_ROW_STYLE>
                <StatCard label="Active nodes" value="47".to_string() subtitle="watershed sensors".to_string() />
                <StatCard label="Uptime" value="99.84%".to_string() subtitle="30-day rolling".to_string() />
                <StatCard label="Flow anomalies" value="2".to_string() subtitle="node-12, node-31".to_string() />
                <StatCard label="Pending proposals" value="3".to_string() subtitle="water-allocation council".to_string() />
            </section>

            // Telemetry row — live streams
            <section style=PANEL_STYLE>
                <h2 style=SECTION_TITLE_STYLE>"live telemetry"</h2>
                <div style="display: flex; flex-direction: column; gap: 4px;">
                    <TelemetryLine
                        label="Flow rate"
                        values=flow_rate.into()
                        unit="L/s"
                        min=0.0
                        max=3.0
                    />
                    <TelemetryLine
                        label="Proposals / hr"
                        values=proposal_rate.into()
                        decimals=0
                        min=0.0
                    />
                    <TelemetryLine
                        label="TEND flow"
                        values=tend_flow.into()
                        unit="T/min"
                        decimals=0
                        min=0.0
                    />
                </div>
            </section>

            // Kinship graph — Indlela growth tied to relational primitives
            <section style=PANEL_STYLE>
                <h2 style=SECTION_TITLE_STYLE>"kinship · trust web"</h2>
                <svg width="100%" height="180" viewBox="0 0 400 180" preserveAspectRatio="xMidYMid meet">
                    // Edges first, nodes on top (recommended layering).
                    <GraphEdge from=(60.0, 90.0) to=(200.0, 40.0) strength=0.7 />
                    <GraphEdge from=(60.0, 90.0) to=(200.0, 140.0) strength=0.85 />
                    <GraphEdge from=(200.0, 40.0) to=(340.0, 90.0) strength=0.55 />
                    <GraphEdge from=(200.0, 140.0) to=(340.0, 90.0) strength=0.5 />
                    <GraphEdge from=(200.0, 40.0) to=(200.0, 140.0) strength=0.4 dashed=true />

                    // Pulse: kinship acknowledgement flowing along one edge.
                    <FlowIndicator
                        from=(60.0, 90.0)
                        to=(200.0, 140.0)
                        emphasis=NodeEmphasis::Care
                        duration_ms=1800
                    />
                    // Pulse: governance confirmation flowing the other way.
                    <FlowIndicator
                        from=(340.0, 90.0)
                        to=(200.0, 40.0)
                        emphasis=NodeEmphasis::Consensus
                        duration_ms=1500
                    />

                    // Nodes — sized by Indlela growth stage.
                    <GraphNode x=60.0 y=90.0 label="Kagiso" stage=GrowthStage::Tree emphasis=NodeEmphasis::Care />
                    <GraphNode x=200.0 y=40.0 label="water council" stage=GrowthStage::Sapling emphasis=NodeEmphasis::Consensus />
                    <GraphNode x=200.0 y=140.0 label="family circle" stage=GrowthStage::Tree emphasis=NodeEmphasis::Care />
                    <GraphNode x=340.0 y=90.0 label="regional" stage=GrowthStage::Sprout />
                </svg>
                <p style="font-size: 10px; color: var(--md-fg-muted, #6e7681); \
                          font-family: var(--md-mono, ui-monospace, monospace); \
                          margin: 8px 0 0 0; letter-spacing: 0.05em;">
                    "node size · growth stage  |  edge opacity · trust strength  \
                     |  pulse · live dispatch"
                </p>
            </section>

            // Semantic-color chip row — documents the token system.
            <section style=PANEL_STYLE>
                <h2 style=SECTION_TITLE_STYLE>"semantic color tokens"</h2>
                <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                    <ColorChip name="signal" var="--md-signal" fallback="#00d4ff" note="live data / streams" />
                    <ColorChip name="consensus" var="--md-consensus" fallback="#f0b72f" note="governance / verified" />
                    <ColorChip name="care" var="--md-care" fallback="#7ee787" note="hearth / kinship / growth" />
                    <ColorChip name="alarm" var="--md-alarm" fallback="#ff7b72" note="emergencies / flagged" />
                </div>
            </section>

            <footer style=FOOTER_STYLE>
                <span>"mycelix-leptos-core / showcase  ·  "</span>
                <span style="color: var(--md-signal, #00d4ff);">
                    "type 1 civilizational substrate"
                </span>
            </footer>
        </div>
    }
}

/// Small chip showing a color token name, its hex fallback, and a note.
/// Intended only for the showcase — not a general-purpose primitive.
#[component]
fn ColorChip(
    name: &'static str,
    var: &'static str,
    fallback: &'static str,
    note: &'static str,
) -> impl IntoView {
    view! {
        <div style="display: flex; align-items: center; gap: 10px; \
                    padding: 6px 10px; border: 1px solid var(--md-fg-muted, #6e7681); \
                    border-radius: 2px; font-family: var(--md-mono, ui-monospace, monospace); \
                    font-size: 10px;">
            <span
                style=format!(
                    "display: inline-block; width: 20px; height: 20px; \
                     background: var({}, {}); border-radius: 2px;",
                    var, fallback
                )
            ></span>
            <div style="display: flex; flex-direction: column; gap: 2px;">
                <span style="color: var(--md-fg, #c9d1d9); letter-spacing: 0.1em;">
                    {name}
                </span>
                <span style="color: var(--md-fg-muted, #6e7681);">
                    {note}
                </span>
            </div>
        </div>
    }
}

// --- Inline styles --- these live here so the showcase renders cohesively
// without the consumer needing to import an external stylesheet. Real cluster
// frontends should migrate these to their own theme CSS.

const SHOWCASE_STYLE: &str = "display: flex; flex-direction: column; gap: 20px; \
    padding: 24px 32px; min-height: 100vh; \
    background: var(--md-base, #0a0e14); \
    color: var(--md-fg, #c9d1d9); \
    font-family: system-ui, -apple-system, sans-serif; font-size: 14px;";

const HEADER_STYLE: &str = "display: flex; flex-direction: column; gap: 6px; \
    padding-bottom: 12px; \
    border-bottom: 1px solid var(--md-fg-muted, #6e7681);";

const TITLE_STYLE: &str = "margin: 0; \
    font-family: var(--md-mono, ui-monospace, 'JetBrains Mono', monospace); \
    font-size: 22px; font-weight: 500; letter-spacing: -0.01em; \
    color: var(--md-fg, #c9d1d9);";

const GRID_ROW_STYLE: &str = "display: grid; \
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px;";

const PANEL_STYLE: &str = "display: flex; flex-direction: column; gap: 10px; \
    padding: 16px 18px; \
    border: 1px solid var(--md-fg-muted, #6e7681); \
    border-radius: 2px; \
    background: rgba(255, 255, 255, 0.01);";

const SECTION_TITLE_STYLE: &str = "margin: 0 0 4px 0; \
    font-family: var(--md-mono, ui-monospace, monospace); \
    font-size: 10px; font-weight: 500; text-transform: uppercase; \
    letter-spacing: 0.2em; color: var(--md-fg-muted, #6e7681);";

const FOOTER_STYLE: &str = "margin-top: auto; padding-top: 12px; \
    font-family: var(--md-mono, ui-monospace, monospace); \
    font-size: 10px; letter-spacing: 0.1em; \
    color: var(--md-fg-muted, #6e7681); \
    border-top: 1px solid var(--md-fg-muted, #6e7681);";

#[cfg(test)]
mod tests {
    // Showcase is a Leptos component — structural tests live in consumer
    // apps via wasm-bindgen-test. These tests cover the pure logic only.

    #[test]
    fn jitter_formula_is_bounded_to_nonneg() {
        // Replicate the jitter closure's behavior.
        let jitter = |seed: u32, base: f64, amp: f64| -> f64 {
            let s = (seed.wrapping_mul(2654435761)) as i32 as f64 / i32::MAX as f64;
            (base + s * amp).max(0.0)
        };
        // Should never go below zero for positive base + bounded amp.
        for seed in 0..200 {
            let v = jitter(seed, 1.6, 0.6);
            assert!(v >= 0.0, "jitter produced negative: {v} at seed {seed}");
        }
    }

    #[test]
    fn jitter_scales_near_base_when_amp_small() {
        let jitter = |seed: u32, base: f64, amp: f64| -> f64 {
            let s = (seed.wrapping_mul(2654435761)) as i32 as f64 / i32::MAX as f64;
            (base + s * amp).max(0.0)
        };
        // With amp=0.1 and base=5.0, values should stay in roughly [4.9, 5.1].
        for seed in 0..50 {
            let v = jitter(seed, 5.0, 0.1);
            assert!(v >= 4.9 && v <= 5.1, "v={v} out of [4.9, 5.1]");
        }
    }

    #[test]
    fn window_cap_keeps_last_n() {
        // Showcase caps each stream at 40 points. Replicate that logic.
        let mut vs: Vec<f64> = (0..50).map(|i| i as f64).collect();
        while vs.len() > 40 {
            vs.remove(0);
        }
        assert_eq!(vs.len(), 40);
        // Oldest retained should be index 10 (we pushed 0..50, dropped 10).
        assert_eq!(vs[0], 10.0);
        assert_eq!(vs[39], 49.0);
    }
}
