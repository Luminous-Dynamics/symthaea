// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::components::glass_panel::GlassPanel;
use crate::state::AppState;

/// Maximum data points in the trend chart.
const CHART_WIDTH: f32 = 600.0;
const CHART_HEIGHT: f32 = 200.0;
const CHART_PADDING: f32 = 30.0;

/// Tab: QOL Trends — personal consciousness + harmony over time.
#[component]
pub fn TrendsPage() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");

    view! {
        <div class="trends-page" style="display: flex; flex-direction: column; gap: 1rem; padding: 1rem;">

            // ── Personal Trend Chart ──
            <GlassPanel title="Personal QOL Trends">
                <div style="position: relative;">
                    <svg
                        viewBox=format!("0 0 {} {}", CHART_WIDTH, CHART_HEIGHT + CHART_PADDING)
                        style="width: 100%; height: auto; max-height: 240px;"
                    >
                        // Grid lines
                        <line x1={CHART_PADDING} y1="0" x2={CHART_PADDING} y2={CHART_HEIGHT}
                            stroke="var(--fg-muted, #555)" stroke-width="0.5" />
                        <line x1={CHART_PADDING} y1={CHART_HEIGHT} x2={CHART_WIDTH} y2={CHART_HEIGHT}
                            stroke="var(--fg-muted, #555)" stroke-width="0.5" />

                        // Y-axis labels
                        <text x="2" y="10" fill="var(--fg-muted, #888)" font-size="9">"1.0"</text>
                        <text x="2" y={CHART_HEIGHT / 2.0} fill="var(--fg-muted, #888)" font-size="9">"0.5"</text>
                        <text x="2" y={CHART_HEIGHT} fill="var(--fg-muted, #888)" font-size="9">"0.0"</text>

                        // Consciousness line (blue)
                        {move || {
                            let data = state.trend_consciousness.get();
                            if data.len() < 2 {
                                return String::new();
                            }
                            build_polyline_points(&data)
                        }}

                        // Harmony line (green)
                        {move || {
                            let data = state.trend_harmony.get();
                            if data.len() < 2 {
                                return String::new();
                            }
                            build_polyline_points_harmony(&data)
                        }}

                        // Legend
                        <rect x={CHART_WIDTH - 140.0} y="8" width="10" height="10" fill="var(--consciousness-blue, #4488cc)" />
                        <text x={CHART_WIDTH - 125.0} y="17" fill="var(--fg, #ccc)" font-size="10">"Consciousness"</text>
                        <rect x={CHART_WIDTH - 140.0} y="24" width="10" height="10" fill="var(--leaf-green, #7ec8a0)" />
                        <text x={CHART_WIDTH - 125.0} y="33" fill="var(--fg, #ccc)" font-size="10">"Harmony"</text>
                    </svg>
                </div>
            </GlassPanel>

            // ── Trend Summary ──
            <GlassPanel title="Trend Summary">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; text-align: center;">
                    <div>
                        <div style="font-size: 0.75rem; color: var(--fg-muted, #888);">"Consciousness"</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">
                            {move || format!("{:.1}%", state.consciousness_level.get() * 100.0)}
                        </div>
                        <div style="font-size: 0.7rem;">
                            {move || {
                                let data = state.trend_consciousness.get();
                                if data.len() >= 2 {
                                    let delta = data.last().unwrap_or(&0.0) - data.first().unwrap_or(&0.0);
                                    if delta > 0.01 {
                                        format!("Rising +{:.1}%", delta * 100.0)
                                    } else if delta < -0.01 {
                                        format!("Falling {:.1}%", delta * 100.0)
                                    } else {
                                        "Stable".to_string()
                                    }
                                } else {
                                    "Gathering data...".to_string()
                                }
                            }}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: var(--fg-muted, #888);">"Harmony"</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">
                            {move || format!("{:.1}%", state.harmony_alignment.get() * 100.0)}
                        </div>
                        <div style="font-size: 0.7rem;">
                            {move || state.dominant_harmony.get()}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: var(--fg-muted, #888);">"Stability"</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">
                            {move || {
                                let data = state.trend_consciousness.get();
                                if data.len() < 5 {
                                    return "...".to_string();
                                }
                                let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                                let var: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
                                let stability = 1.0 - var.sqrt().min(1.0);
                                format!("{:.0}%", stability * 100.0)
                            }}
                        </div>
                        <div style="font-size: 0.7rem; color: var(--fg-muted, #888);">"Low variance = stable"</div>
                    </div>
                </div>
            </GlassPanel>

            // ── Wellbeing Profile ──
            <GlassPanel title="Wellbeing Profile">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="font-size: 1.1rem; font-weight: bold;">
                        {move || state.wellbeing_profile.get()}
                    </div>
                    <div style="font-size: 0.8rem; color: var(--fg-muted, #888);">
                        {move || {
                            match state.wellbeing_profile.get().as_str() {
                                "Default" => "Baseline consciousness dynamics",
                                "ADHD Support" => "Elevated novelty response, faster dynamics",
                                "Anxiety Buffer" => "Calming bias during high stress",
                                "Grief Mode" => "Slower, gentler dynamics with rest emphasis",
                                "Sensory Gentle" => "Dampened surprise, reduced alertness",
                                "Focus Flow" => "Sustained attention with moderate engagement",
                                _ => "Custom profile",
                            }
                        }}
                    </div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.7rem; color: var(--fg-muted, #666);">
                    "Profiles modulate the AI companion's behavior, not your neurochemistry. \
                     Change via Soma app or API: set_wellbeing_profile(\"grief_mode\")"
                </div>
            </GlassPanel>

            // ── Data Points ──
            <GlassPanel title="Trend Data">
                <div style="font-size: 0.8rem; color: var(--fg-muted, #888);">
                    {move || {
                        let n = state.trend_consciousness.get().len();
                        format!("{} data points collected ({:.0} seconds of observation)",
                            n, n as f32 * 0.05) // ~20Hz, broadcast every cycle
                    }}
                </div>
            </GlassPanel>
        </div>
    }
}

/// Build SVG polyline points string for consciousness data (blue line).
fn build_polyline_points(data: &[f32]) -> String {
    if data.is_empty() {
        return String::new();
    }
    let n = data.len();
    let usable_width = CHART_WIDTH - CHART_PADDING;
    let points: String = data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = CHART_PADDING + (i as f32 / n.max(1) as f32) * usable_width;
            let y = CHART_HEIGHT * (1.0 - v.clamp(0.0, 1.0));
            format!("{:.1},{:.1}", x, y)
        })
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        "<polyline points=\"{}\" fill=\"none\" stroke=\"var(--consciousness-blue, #4488cc)\" stroke-width=\"1.5\" opacity=\"0.8\" />",
        points
    )
}

/// Build SVG polyline points string for harmony data (green line).
fn build_polyline_points_harmony(data: &[f32]) -> String {
    if data.is_empty() {
        return String::new();
    }
    let n = data.len();
    let usable_width = CHART_WIDTH - CHART_PADDING;
    let points: String = data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = CHART_PADDING + (i as f32 / n.max(1) as f32) * usable_width;
            let y = CHART_HEIGHT * (1.0 - v.clamp(0.0, 1.0));
            format!("{:.1},{:.1}", x, y)
        })
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        "<polyline points=\"{}\" fill=\"none\" stroke=\"var(--leaf-green, #7ec8a0)\" stroke-width=\"1.5\" opacity=\"0.8\" />",
        points
    )
}
