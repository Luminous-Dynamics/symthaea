// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governance vote tally visualization — for/against/abstain horizontal bar.

use leptos::prelude::*;

/// Horizontal stacked bar showing vote distribution.
///
/// Renders a single horizontal bar split into for (green), against (red),
/// and abstain (grey) segments with counts and percentage labels.
#[component]
pub fn VoteTallyBar(
    votes_for: u32,
    votes_against: u32,
    votes_abstain: u32,
    #[prop(default = 300.0)] width: f64,
    #[prop(default = 40.0)] height: f64,
) -> impl IntoView {
    let total = (votes_for + votes_against + votes_abstain).max(1) as f64;
    let pct_for = votes_for as f64 / total;
    let pct_against = votes_against as f64 / total;
    let pct_abstain = votes_abstain as f64 / total;

    let padding = 4.0;
    let bar_w = width - 2.0 * padding;
    let bar_h = 16.0;
    let bar_y = 4.0;

    let w_for = pct_for * bar_w;
    let w_against = pct_against * bar_w;
    let w_abstain = pct_abstain * bar_w;

    let x_for = padding;
    let x_against = x_for + w_for;
    let x_abstain = x_against + w_against;

    view! {
        <svg
            viewBox=format!("0 0 {width} {height}")
            style="width: 100%; height: auto;"
            xmlns="http://www.w3.org/2000/svg"
        >
            // Background
            <rect x=padding y=bar_y width=bar_w height=bar_h rx="4"
                fill="rgba(255,255,255,0.05)" />

            // For segment
            {(w_for > 0.5).then(|| view! {
                <rect x=x_for y=bar_y width=w_for height=bar_h
                    fill="#22c55e" rx="4" opacity="0.85" />
            })}

            // Against segment
            {(w_against > 0.5).then(|| view! {
                <rect x=x_against y=bar_y width=w_against height=bar_h
                    fill="#ef4444" opacity="0.85" />
            })}

            // Abstain segment
            {(w_abstain > 0.5).then(|| view! {
                <rect x=x_abstain y=bar_y width=w_abstain height=bar_h
                    fill="#6b7280" opacity="0.6" />
            })}

            // Labels below
            <text x=padding y={bar_y + bar_h + 12.0}
                fill="#22c55e" font-size="9" font-family="monospace">
                {format!("{} for ({:.0}%)", votes_for, pct_for * 100.0)}
            </text>
            <text x={width / 2.0} y={bar_y + bar_h + 12.0}
                text-anchor="middle"
                fill="#ef4444" font-size="9" font-family="monospace">
                {format!("{} against ({:.0}%)", votes_against, pct_against * 100.0)}
            </text>
            <text x={width - padding} y={bar_y + bar_h + 12.0}
                text-anchor="end"
                fill="#6b7280" font-size="9" font-family="monospace">
                {format!("{} abstain", votes_abstain)}
            </text>
        </svg>
    }
}
