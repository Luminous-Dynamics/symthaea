// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finance domain pages — economic metabolism dashboard.

use leptos::prelude::*;
use portal_viz::{BarChart, LineChart, bar_chart::Bar, line_chart::Series};

#[component]
pub fn FinanceOverview() -> impl IntoView {
    view! {
        <div class="finance-content">
            <div class="governance-nav">
                <button class="domain-nav-btn active">"Overview"</button>
                <button class="domain-nav-btn">"Payments"</button>
                <button class="domain-nav-btn">"Treasury"</button>
                <button class="domain-nav-btn">"Staking"</button>
                <button class="domain-nav-btn">"Recognition"</button>
            </div>

            <div class="commons-stats-grid">
                <div class="thought-card">
                    <div class="thought-type" style="color: #D97706">"SAP BALANCE"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"4,280"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"After demurrage"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #FBBF24">"MYCEL SCORE"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"0.72"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Participation + Recognition"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #22c55e">"STAKED"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"1,200"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"SAP collateral (weight 1.72x)"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #8b5cf6">"TREASURY"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"84,500"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Commons pool (inalienable)"</p>
                </div>
            </div>

            <h3 class="section-title">"TEND Flow (last 8 weeks)"</h3>
            <LineChart
                series=vec![
                    Series { label: "Inflow".into(), color: "#22c55e".into(), data: vec![120.0, 185.0, 210.0, 175.0, 230.0, 195.0, 245.0, 220.0] },
                    Series { label: "Outflow".into(), color: "#f59e0b".into(), data: vec![80.0, 95.0, 110.0, 140.0, 105.0, 130.0, 115.0, 145.0] },
                ]
                width=400.0
                height=180.0
            />

            <h3 class="section-title" style="margin-top: 1rem">"Recognition This Cycle"</h3>
            <BarChart
                data=vec![
                    Bar { label: "Care".into(), value: 35.0, color: "#ec4899".into() },
                    Bar { label: "Teaching".into(), value: 28.0, color: "#2563EB".into() },
                    Bar { label: "Governance".into(), value: 18.0, color: "#7C3AED".into() },
                    Bar { label: "Infra".into(), value: 12.0, color: "#059669".into() },
                    Bar { label: "Art".into(), value: 8.0, color: "#FBBF24".into() },
                ]
                width=350.0
                height=160.0
            />
        </div>
    }
}
