// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_finance_context;

#[component]
pub fn TendPage() -> impl IntoView {
    let ctx = use_finance_context();

    view! {
        <div class="page-tend">
            <h1>"TEND"</h1>
            <p class="subtitle">"Mutual credit \u{2014} 1 hour of care = 1 TEND"</p>

            <div class="tend-balance-display">
                <span class=move || format!("tend-balance {}", ctx.tend_balance.get().balance_class())>
                    {move || format!("{:+}", ctx.tend_balance.get().balance)}
                </span>
                <span class="tend-label">{move || ctx.tend_balance.get().equilibrium_label()}</span>
            </div>

            <div class="summary-bar">
                <div class="summary-item">
                    <span class="summary-label">"Given"</span>
                    <span class="summary-value">{move || format!("{:.1}h", ctx.tend_balance.get().total_provided)}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">"Received"</span>
                    <span class="summary-value">{move || format!("{:.1}h", ctx.tend_balance.get().total_received)}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">"Exchanges"</span>
                    <span class="summary-value">{move || ctx.tend_balance.get().exchange_count.to_string()}</span>
                </div>
            </div>

            <h2>"Recent Exchanges"</h2>
            <div class="exchange-list">
                {move || ctx.tend_exchanges.get().iter().map(|ex| {
                    let desc = ex.service_description.clone();
                    let hours = ex.hours;
                    let category = ex.service_category.label().to_string();
                    let status = ex.status.label();
                    view! {
                        <div class="exchange-card">
                            <div class="exchange-header">
                                <span class="exchange-hours">{format!("{hours:.1}h")}</span>
                                <span class="badge">{status}</span>
                            </div>
                            <p class="exchange-desc">{desc}</p>
                            <span class="exchange-category">{category}</span>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
