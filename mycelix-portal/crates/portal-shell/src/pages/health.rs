// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Health domain pages — homeostasis dashboard.

use leptos::prelude::*;
use portal_viz::{PieChart, LineChart, pie_chart::Segment, line_chart::Series};

#[component]
pub fn HealthOverview() -> impl IntoView {
    view! {
        <div class="health-content">
            <div class="governance-nav">
                <button class="domain-nav-btn active">"Homeostasis"</button>
                <button class="domain-nav-btn">"Records"</button>
                <button class="domain-nav-btn">"Consent"</button>
                <button class="domain-nav-btn">"Privacy"</button>
                <button class="domain-nav-btn">"Metabolism"</button>
            </div>

            <div class="commons-stats-grid">
                <div class="thought-card">
                    <div class="thought-type" style="color: #0D7377">"RECORDS"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"47"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Encrypted in vault"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #06D6C8">"CONSENTS"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"3"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Active (1 Part 2 protected)"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #f59e0b">"PRIVACY \u{03b5}"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"2.4"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"of 10.0 budget remaining"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #22c55e">"DIVIDENDS"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"$128"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Earned from FL research"</p>
                </div>
            </div>

            <h3 class="section-title">"Record Types"</h3>
            <PieChart
                segments=vec![
                    Segment { label: "Lab".into(), value: 18.0, color: "#0D7377".into() },
                    Segment { label: "Vitals".into(), value: 12.0, color: "#06D6C8".into() },
                    Segment { label: "Meds".into(), value: 8.0, color: "#22c55e".into() },
                    Segment { label: "Imaging".into(), value: 5.0, color: "#60a5fa".into() },
                    Segment { label: "Other".into(), value: 4.0, color: "#6b7280".into() },
                ]
            />

            <h3 class="section-title" style="margin-top: 1rem">"Privacy Budget Consumption"</h3>
            <LineChart
                series=vec![
                    Series { label: "\u{03b5} spent".into(), color: "#f59e0b".into(), data: vec![0.0, 0.8, 1.5, 2.1, 3.2, 4.8, 6.1, 7.6] },
                ]
                width=350.0
                height=140.0
            />
        </div>
    }
}
