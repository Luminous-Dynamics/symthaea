// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::context::use_finance_context;
use leptos::prelude::*;
use leptos_router::components::A;

#[component]
pub fn HomePage() -> impl IntoView {
    let ctx = use_finance_context();

    view! {
        <div class="page-home">
            <section class="hero">
                <h1>"Finance"</h1>
                <p class="hero-subtitle">"Three currencies. One commons. No extraction."</p>
                <div class="hero-cta">
                    <A href="/tend" attr:class="btn btn-primary">"View Balance"</A>
                    <A href="/tend" attr:class="btn btn-ghost">"Give Care"</A>
                </div>
            </section>

            <section class="how-it-works">
                <h2>"How it works"</h2>
                <div class="steps">
                    <div class="step">
                        <span class="step-icon">"🤝"</span>
                        <h3>"TEND"</h3>
                        <p>"Mutual credit \u{2014} 1 hour = 1 TEND"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"💧"</span>
                        <h3>"SAP"</h3>
                        <p>"Transferable value with demurrage"</p>
                    </div>
                    <div class="step">
                        <span class="step-icon">"🍄"</span>
                        <h3>"MYCEL"</h3>
                        <p>"Soulbound reputation"</p>
                    </div>
                </div>
            </section>

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
