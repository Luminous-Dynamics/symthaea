// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_finance_context;

#[component]
pub fn HomePage() -> impl IntoView {
    let ctx = use_finance_context();

    view! {
        <div class="page-home">
            <h1>"Finance"</h1>
            <p class="subtitle">"Three currencies, one commons."</p>

            <div class="currency-grid">
                <div class="currency-card tend-card">
                    <span class="currency-icon">"🤝"</span>
                    <span class="currency-name">"TEND"</span>
                    <span class="currency-value">{move || format!("{} hours", ctx.tend_balance.get().balance)}</span>
                    <span class="currency-desc">{move || ctx.tend_balance.get().equilibrium_label()}</span>
                </div>
                <div class="currency-card sap-card">
                    <span class="currency-icon">"💧"</span>
                    <span class="currency-name">"SAP"</span>
                    <span class="currency-value">{move || format!("{:.2}", ctx.sap_balance.get().display_balance())}</span>
                    <span class="currency-desc">"transferable with 2%/yr demurrage"</span>
                </div>
                <div class="currency-card mycel-card">
                    <span class="currency-icon">"🍄"</span>
                    <span class="currency-name">"MYCEL"</span>
                    <span class="currency-value">{move || format!("{:.2}", ctx.mycel_score.get().score)}</span>
                    <span class="currency-desc">{move || ctx.mycel_score.get().tier.label()}</span>
                </div>
            </div>
        </div>
    }
}
