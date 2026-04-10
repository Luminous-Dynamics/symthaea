// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::actions;
use crate::context::use_finance_context;
use leptos::prelude::*;

#[component]
pub fn TendPage() -> impl IntoView {
    let ctx = use_finance_context();

    // Exchange form state
    let (receiver, set_receiver) = signal(String::new());
    let (hours, set_hours) = signal(String::new());
    let (description, set_description) = signal(String::new());
    let (show_form, set_show_form) = signal(false);

    let can_submit = Memo::new(move |_| {
        !receiver.get().trim().is_empty()
            && hours.get().parse::<f32>().unwrap_or(0.0) > 0.0
            && !description.get().trim().is_empty()
    });

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

            // Exchange form
            <div class="action-section">
                <button class="btn btn-primary" on:click=move |_| set_show_form.update(|s| *s = !*s)>
                    {move || if show_form.get() { "Cancel" } else { "Record care given" }}
                </button>

                <div style=move || if show_form.get() { "display: block" } else { "display: none" }>
                    <form class="create-form" on:submit=move |ev: leptos::ev::SubmitEvent| {
                        ev.prevent_default();
                        if can_submit.get_untracked() {
                            actions::record_tend_exchange(
                                receiver.get_untracked(),
                                hours.get_untracked().parse::<f32>().unwrap_or(1.0),
                                description.get_untracked(),
                            );
                            set_receiver.set(String::new());
                            set_hours.set(String::new());
                            set_description.set(String::new());
                            set_show_form.set(false);
                        }
                    }>
                        <div class="form-field">
                            <label>"Who received your care?"</label>
                            <input class="form-input" type="text" placeholder="Name or DID"
                                prop:value=move || receiver.get()
                                on:input=move |ev| set_receiver.set(event_target_value(&ev))
                            />
                        </div>
                        <div class="form-field">
                            <label>"Hours of care"</label>
                            <input class="form-input" type="number" step="0.5" min="0.5" max="168" placeholder="1.0"
                                prop:value=move || hours.get()
                                on:input=move |ev| set_hours.set(event_target_value(&ev))
                            />
                        </div>
                        <div class="form-field">
                            <label>"What care did you give?"</label>
                            <textarea class="form-textarea" rows="3" placeholder="Describe the care you provided..."
                                prop:value=move || description.get()
                                on:input=move |ev| set_description.set(event_target_value(&ev))
                            ></textarea>
                        </div>
                        <button type="submit" class="btn btn-primary" disabled=move || !can_submit.get()>
                            "Record this exchange"
                        </button>
                    </form>
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
