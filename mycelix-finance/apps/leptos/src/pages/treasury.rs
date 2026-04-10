// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::context::use_finance_context;
use leptos::prelude::*;

#[component]
pub fn TreasuryPage() -> impl IntoView {
    let ctx = use_finance_context();

    view! {
        <div class="page-treasury">
            <h1>"Treasury"</h1>
            <p class="subtitle">"The commons reserve \u{2014} 25% inalienable."</p>

            {move || ctx.treasury.get().map(|t| {
                let health = t.reserve_health() * 100.0;
                view! {
                    <div class="treasury-display">
                        <div class="dashboard-grid">
                            <div class="dash-card">
                                <span class="dash-label">"Total Balance"</span>
                                <span class="dash-value">{format!("{:.2} SAP", t.display_balance())}</span>
                            </div>
                            <div class="dash-card">
                                <span class="dash-label">"Inalienable Reserve"</span>
                                <span class="dash-value">{format!("{:.2} SAP", t.display_reserve())}</span>
                                <span class="dash-sub">"25% of total \u{2014} cannot be spent"</span>
                            </div>
                            <div class="dash-card">
                                <span class="dash-label">"Available"</span>
                                <span class="dash-value accent">{format!("{:.2} SAP", t.display_available())}</span>
                            </div>
                            <div class="dash-card">
                                <span class="dash-label">"Reserve Health"</span>
                                <span class="dash-value">{format!("{health:.0}%")}</span>
                            </div>
                        </div>
                    </div>
                }
            })}
        </div>
    }
}
