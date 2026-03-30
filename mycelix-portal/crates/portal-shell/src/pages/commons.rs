// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Commons domain pages — shared resources dashboard.

use leptos::prelude::*;
use portal_viz::{BarChart, PieChart, bar_chart::Bar, pie_chart::Segment};

#[component]
pub fn CommonsOverview() -> impl IntoView {
    view! {
        <div class="commons-content">
            <div class="governance-nav">
                <button class="domain-nav-btn active">"Overview"</button>
                <button class="domain-nav-btn">"Property"</button>
                <button class="domain-nav-btn">"Housing"</button>
                <button class="domain-nav-btn">"Care"</button>
                <button class="domain-nav-btn">"Water"</button>
                <button class="domain-nav-btn">"Food"</button>
                <button class="domain-nav-btn">"Transport"</button>
            </div>

            <div class="commons-stats-grid">
                <div class="thought-card">
                    <div class="thought-type" style="color: #059669">"HOUSING"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"142"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Units across 12 buildings"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #06b6d4">"WATER"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"8"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Active water sources"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #f59e0b">"CARE"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"23"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Active care plans"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #ec4899">"MUTUAL AID"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"7"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Open requests"</p>
                </div>
            </div>

            <h3 class="section-title">"Resource Distribution"</h3>
            <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
                <PieChart
                    segments=vec![
                        Segment { label: "Housing".into(), value: 42.0, color: "#059669".into() },
                        Segment { label: "Water".into(), value: 18.0, color: "#06b6d4".into() },
                        Segment { label: "Food".into(), value: 25.0, color: "#f59e0b".into() },
                        Segment { label: "Transport".into(), value: 15.0, color: "#8b5cf6".into() },
                    ]
                />
                <BarChart
                    data=vec![
                        Bar { label: "Jan".into(), value: 12.0, color: "#059669".into() },
                        Bar { label: "Feb".into(), value: 18.0, color: "#059669".into() },
                        Bar { label: "Mar".into(), value: 15.0, color: "#059669".into() },
                        Bar { label: "Apr".into(), value: 22.0, color: "#34D399".into() },
                    ]
                    width=280.0
                    height=160.0
                />
            </div>
        </div>
    }
}
