// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! EduNet domain pages — learning dashboard.

use leptos::prelude::*;
use portal_viz::{BarChart, bar_chart::Bar};

#[component]
pub fn EdunetOverview() -> impl IntoView {
    view! {
        <div class="edunet-content">
            <div class="governance-nav">
                <button class="domain-nav-btn active">"Dashboard"</button>
                <button class="domain-nav-btn">"Courses"</button>
                <button class="domain-nav-btn">"Review"</button>
                <button class="domain-nav-btn">"Skill Map"</button>
                <button class="domain-nav-btn">"Credentials"</button>
            </div>

            <div class="commons-stats-grid">
                <div class="thought-card">
                    <div class="thought-type" style="color: #2563EB">"LEVEL"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"7"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"4,280 XP total"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #60A5FA">"STREAK"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"12d"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"1.5x bonus active"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #f59e0b">"DUE"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"18"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Cards to review (3 overdue)"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #22c55e">"MASTERY"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"68%"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"CAPS curriculum average"</p>
                </div>
            </div>

            <h3 class="section-title">"Top Skills"</h3>
            <BarChart
                data=vec![
                    Bar { label: "Multiplication".into(), value: 85.0, color: "#2563EB".into() },
                    Bar { label: "Fractions".into(), value: 72.0, color: "#60A5FA".into() },
                    Bar { label: "Algebra".into(), value: 68.0, color: "#60A5FA".into() },
                    Bar { label: "Geometry".into(), value: 55.0, color: "#93c5fd".into() },
                    Bar { label: "Statistics".into(), value: 42.0, color: "#93c5fd".into() },
                ]
                width=350.0
                height=180.0
            />
        </div>
    }
}
