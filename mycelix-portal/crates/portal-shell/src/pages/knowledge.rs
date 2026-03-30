// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Knowledge domain pages — epistemic commons dashboard.

use leptos::prelude::*;
use portal_viz::{BarChart, bar_chart::Bar};

#[component]
pub fn KnowledgeOverview() -> impl IntoView {
    view! {
        <div class="knowledge-content">
            <div class="governance-nav">
                <button class="domain-nav-btn active">"Claims"</button>
                <button class="domain-nav-btn">"Graph"</button>
                <button class="domain-nav-btn">"Fact Check"</button>
                <button class="domain-nav-btn">"Markets"</button>
            </div>

            <div class="commons-stats-grid">
                <div class="thought-card">
                    <div class="thought-type" style="color: #0891B2">"CLAIMS"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"342"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"In the knowledge graph"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #22D3EE">"VERIFIED"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"78%"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Fact-check pass rate"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #f59e0b">"MARKETS"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"12"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Active prediction markets"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #22c55e">"COHERENCE"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"0.71"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Graph coherence score"</p>
                </div>
            </div>

            <h3 class="section-title">"Verdicts by Domain"</h3>
            <BarChart
                data=vec![
                    Bar { label: "Science".into(), value: 89.0, color: "#0891B2".into() },
                    Bar { label: "Policy".into(), value: 62.0, color: "#22D3EE".into() },
                    Bar { label: "Health".into(), value: 78.0, color: "#0D7377".into() },
                    Bar { label: "Economics".into(), value: 54.0, color: "#D97706".into() },
                    Bar { label: "Tech".into(), value: 71.0, color: "#2563EB".into() },
                ]
                width=350.0
                height=180.0
            />
        </div>
    }
}
