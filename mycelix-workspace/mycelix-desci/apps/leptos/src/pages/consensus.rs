// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consensus tracker — paradigm shift detection.

use leptos::prelude::*;
use crate::types::*;

fn demo_consensus() -> Vec<(String, Vec<ConsensusSnapshot>)> {
    vec![
        ("Room-Temperature Superconductivity (LK-99)".into(), vec![
            ConsensusSnapshot { timestamp: "2023-07-01".into(), support_ratio: 0.65, verification_count: 2, state: "Emerging".into(), trend: "Strengthening".into() },
            ConsensusSnapshot { timestamp: "2023-07-15".into(), support_ratio: 0.45, verification_count: 8, state: "Contested".into(), trend: "Weakening".into() },
            ConsensusSnapshot { timestamp: "2023-08-01".into(), support_ratio: 0.10, verification_count: 15, state: "Contested".into(), trend: "Reversing".into() },
            ConsensusSnapshot { timestamp: "2023-08-15".into(), support_ratio: 0.02, verification_count: 20, state: "NearUniversal".into(), trend: "Stable".into() },
        ]),
        ("Hubble Tension (H₀ discrepancy)".into(), vec![
            ConsensusSnapshot { timestamp: "2024-01-01".into(), support_ratio: 0.70, verification_count: 50, state: "Strong".into(), trend: "Stable".into() },
            ConsensusSnapshot { timestamp: "2025-01-01".into(), support_ratio: 0.75, verification_count: 80, state: "Strong".into(), trend: "Strengthening".into() },
            ConsensusSnapshot { timestamp: "2026-01-01".into(), support_ratio: 0.80, verification_count: 120, state: "Strong".into(), trend: "Strengthening".into() },
        ]),
        ("Superheavy Island of Stability".into(), vec![
            ConsensusSnapshot { timestamp: "2026-04-04".into(), support_ratio: 0.60, verification_count: 3, state: "Emerging".into(), trend: "Strengthening".into() },
        ]),
    ]
}

#[component]
pub fn ConsensusPage() -> impl IntoView {
    let consensus_data = demo_consensus();

    view! {
        <div class="page-container">
            <h1 class="page-title">"Consensus Tracker"</h1>
            <p style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 1.5rem;">
                "Tracks how scientific consensus evolves over time. Detects paradigm shifts, reversals, and emerging agreements."
            </p>

            {consensus_data.iter().map(|(name, snapshots)| {
                let latest = snapshots.last().unwrap();
                let trend_color = match latest.trend.as_str() {
                    "Strengthening" => "var(--accent-emerald)",
                    "Weakening" | "Reversing" => "var(--tier-e0)",
                    _ => "var(--text-secondary)",
                };

                view! {
                    <div class="glass-panel" style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                            <h3 style="font-size: 1rem;">{name.clone()}</h3>
                            <div style="display: flex; gap: 0.5rem; align-items: center;">
                                <span style=format!("font-size: 0.75rem; color: {};", trend_color)>{latest.trend.clone()}</span>
                                <span style="font-size: 1.25rem; font-weight: 700; color: var(--accent-indigo);">{format!("{:.0}%", latest.support_ratio * 100.0)}</span>
                            </div>
                        </div>

                        // Timeline
                        <div style="display: flex; gap: 2px; align-items: end; height: 40px; margin-bottom: 0.5rem;">
                            {snapshots.iter().map(|s| {
                                let h = format!("{}px", (s.support_ratio * 40.0) as u32);
                                let color = if s.support_ratio > 0.7 { "var(--accent-emerald)" } else if s.support_ratio > 0.3 { "var(--tier-e2)" } else { "var(--tier-e0)" };
                                view! {
                                    <div style=format!("flex: 1; height: {}; background: {}; border-radius: 2px 2px 0 0;", h, color) title=format!("{}: {:.0}%", s.timestamp, s.support_ratio * 100.0)></div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>

                        <div style="font-size: 0.75rem; color: var(--text-secondary);">
                            {format!("{} verification(s) | State: {} | {} snapshots", latest.verification_count, latest.state, snapshots.len())}
                        </div>
                    </div>
                }
            }).collect::<Vec<_>>()}
        </div>
    }
}
