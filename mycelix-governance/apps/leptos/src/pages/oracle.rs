// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Price oracle page: decentralized price discovery via crowd-sourced reports.
//! Prices denominated in TEND units — "what does 1 TEND buy?"
//! AI-interactable: data-oracle-vitality, data-tier, data-form.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;

#[component]
pub fn OraclePage() -> impl IntoView {
    let fin = use_finance();

    let (report_item, set_report_item) = signal(String::new());
    let (report_price, set_report_price) = signal(String::new());

    view! {
        <div class="oracle-page" data-page="oracle" role="main">
            <h1 class="page-title">"Price Oracle"</h1>
            <p class="page-subtitle">"what does one hour of care buy? the community decides"</p>

            // Oracle state
            <section class="oracle-state-section" aria-label="oracle state" data-section="oracle-state">
                {move || {
                    let oracle = fin.oracle_state.get();
                    let css = oracle.tier.css_class().to_string();
                    let tier_label = oracle.tier.label().to_string();
                    let tier_label2 = tier_label.clone();
                    let limit = oracle.tier.credit_limit();
                    view! {
                        <div
                            class=format!("oracle-state-card {css}")
                            data-oracle-vitality=oracle.vitality.to_string()
                            data-oracle-tier=tier_label.clone()
                            data-credit-limit=limit.to_string()
                        >
                            <div class="oracle-vitality">
                                <span class="vitality-label">"community vitality"</span>
                                <div class="vitality-bar">
                                    <div
                                        class="vitality-fill"
                                        style=format!("width: {}%", oracle.vitality)
                                    ></div>
                                </div>
                                <span class="vitality-value" data-metric="vitality">
                                    {format!("{}/100", oracle.vitality)}
                                </span>
                            </div>
                            <div class="oracle-tier-info">
                                <span class="tier-label">"tier: "</span>
                                <span class="tier-value">{tier_label2}</span>
                                <span class="tier-limit">
                                    {format!(" — TEND credit limit ±{limit}h")}
                                </span>
                            </div>
                            <p class="oracle-explanation">
                                "when community vitality is high, trust expands and credit limits grow. "
                                "when stressed, the organism contracts to protect itself."
                            </p>
                        </div>
                    }
                }}
            </section>

            // Report a price
            <section class="price-report-section" aria-label="report a price" data-section="price-report">
                <h2 class="section-title">"Report a price"</h2>
                <p class="section-desc">
                    "help the community understand what 1 TEND hour is worth in real goods"
                </p>
                <form
                    class="price-report-form"
                    data-form="price-report"
                    on:submit=move |ev| {
                        ev.prevent_default();
                        // In Phase 5, wire to price_oracle.report_price
                        set_report_item.set(String::new());
                        set_report_price.set(String::new());
                    }
                >
                    <div class="form-row">
                        <div class="form-field">
                            <label for="report-item">"Item"</label>
                            <input
                                id="report-item"
                                type="text"
                                class="form-input"
                                placeholder="e.g. 1kg bread, 1L milk"
                                data-field="item"
                                prop:value=move || report_item.get()
                                on:input=move |ev| set_report_item.set(event_target_value(&ev))
                            />
                        </div>
                        <div class="form-field">
                            <label for="report-price">"TEND price"</label>
                            <input
                                id="report-price"
                                type="number"
                                class="form-input"
                                step="0.01"
                                placeholder="0.25"
                                data-field="price-tend"
                                prop:value=move || report_price.get()
                                on:input=move |ev| set_report_price.set(event_target_value(&ev))
                            />
                        </div>
                    </div>
                    <button
                        type="submit"
                        class="submit-btn"
                        data-action="report-price"
                        disabled=move || {
                            report_item.get().trim().is_empty()
                                || report_price.get().parse::<f64>().unwrap_or(0.0) <= 0.0
                        }
                    >
                        "broadcast price signal"
                    </button>
                </form>
            </section>
        </div>
    }
}
