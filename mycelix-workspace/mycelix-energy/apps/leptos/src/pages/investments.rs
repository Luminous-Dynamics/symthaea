// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_energy_context;

#[component]
pub fn InvestmentsPage() -> impl IntoView {
    let ctx = use_energy_context();

    let total_invested = move || {
        ctx.investments.get().iter()
            .filter(|i| i.status == energy_leptos_types::InvestmentStatus::Confirmed)
            .map(|i| i.amount).sum::<f64>()
    };

    view! {
        <div class="page-investments">
            <h1>"Investments"</h1>
            <p class="subtitle">"Your energy investment portfolio."</p>

            <div class="summary-bar">
                <div class="summary-item">
                    <span class="summary-label">"Total Invested"</span>
                    <span class="summary-value">{move || format!("R {:.0}", total_invested())}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">"Positions"</span>
                    <span class="summary-value">{move || ctx.investments.get().len().to_string()}</span>
                </div>
            </div>

            <div class="investment-list">
                {move || ctx.investments.get().iter().map(|inv| {
                    let project = inv.project_id.clone();
                    let amount = inv.amount;
                    let currency = inv.currency.clone();
                    let inv_type = inv.investment_type.label();
                    let status = inv.status.label();
                    let share_pct = inv.share_percentage;
                    view! {
                        <div class="investment-card">
                            <div class="inv-header">
                                <span class="inv-project">{project}</span>
                                <span class="badge">{status}</span>
                            </div>
                            <div class="inv-body">
                                <span class="inv-amount">{format!("{currency} {amount:.0}")}</span>
                                <span class="inv-meta">{format!("{inv_type} | {share_pct:.1}% share")}</span>
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
