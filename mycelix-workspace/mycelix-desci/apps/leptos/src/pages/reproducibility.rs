// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducibility dashboard.

use leptos::prelude::*;
use crate::types::*;

fn demo_stats() -> Vec<ReproducibilityStats> {
    vec![
        ReproducibilityStats { claim_id: "lk99-superconductor".into(), score: 0.05, confidence: 0.95, total_attempts: 12, successful: 0, failed: 11 },
        ReproducibilityStats { claim_id: "island-stability-z120".into(), score: 0.85, confidence: 0.4, total_attempts: 2, successful: 1, failed: 0 },
        ReproducibilityStats { claim_id: "arts-parts-waveguide".into(), score: 0.15, confidence: 0.8, total_attempts: 3, successful: 0, failed: 2 },
    ]
}

#[component]
pub fn ReproducibilityPage() -> impl IntoView {
    let stats = demo_stats();

    view! {
        <div class="page-container">
            <h1 class="page-title">"Reproducibility Dashboard"</h1>

            <div class="stat-grid" style="margin-bottom: 1.5rem;">
                <div class="glass-panel stat-card">
                    <div class="stat-value">{stats.iter().map(|s| s.total_attempts).sum::<usize>()}</div>
                    <div class="stat-label">"Total Replication Attempts"</div>
                </div>
                <div class="glass-panel stat-card">
                    <div class="stat-value">{format!("{:.0}%", stats.iter().map(|s| s.score).sum::<f64>() / stats.len() as f64 * 100.0)}</div>
                    <div class="stat-label">"Average Reproducibility"</div>
                </div>
                <div class="glass-panel stat-card">
                    <div class="stat-value">{stats.iter().filter(|s| s.score < 0.3).count()}</div>
                    <div class="stat-label">"Crisis Claims (< 30%)"</div>
                </div>
            </div>

            <div class="glass-panel">
                <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">"Claims by Reproducibility Score"</h2>
                {stats.iter().map(|s| {
                    let bar_color = if s.score > 0.7 { "var(--accent-emerald)" } else if s.score > 0.3 { "var(--tier-e2)" } else { "var(--tier-e0)" };
                    let width = format!("{}%", (s.score * 100.0) as u32);
                    view! {
                        <div style="margin-bottom: 1rem; padding: 0.75rem; background: var(--bg-secondary); border-radius: 0.5rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.375rem;">
                                <span style="font-size: 0.875rem; font-weight: 600;">{s.claim_id.clone()}</span>
                                <span style=format!("font-size: 0.875rem; font-weight: 700; color: {};", bar_color)>{format!("{:.0}%", s.score * 100.0)}</span>
                            </div>
                            <div style="height: 8px; background: var(--bg-primary); border-radius: 4px; margin-bottom: 0.25rem;">
                                <div style=format!("width: {}; height: 100%; background: {}; border-radius: 4px;", width, bar_color)></div>
                            </div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary);">
                                {format!("{} attempts: {} successful, {} failed | confidence: {:.0}%", s.total_attempts, s.successful, s.failed, s.confidence * 100.0)}
                            </div>
                        </div>
                    }
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}
