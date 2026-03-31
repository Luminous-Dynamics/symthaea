// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! TEND exchange page: mutual credit as reciprocal care.
//!
//! Balance display uses equilibrium language, not accounting.
//! Oracle tier indicator modulates the section's breathing rate.
//! AI-interactable: data-balance, data-oracle-tier, data-exchange-id.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;
use crate::contexts::civic_actions;
use finance_leptos_types::*;

#[component]
pub fn TendPage() -> impl IntoView {
    let fin = use_finance();

    let (receiver_did, set_receiver_did) = signal(String::new());
    let (hours, set_hours) = signal(String::new());
    let (description, set_description) = signal(String::new());

    let can_submit = Memo::new(move |_| {
        !receiver_did.get().trim().is_empty()
            && hours.get().parse::<f32>().unwrap_or(0.0) > 0.0
            && !description.get().trim().is_empty()
    });

    view! {
        <div class="tend-page" data-page="tend" role="main">
            <h1 class="page-title">"Mutual Care"</h1>

            // Balance + Oracle (above fold)
            <section class="tend-balance-section" aria-label="TEND balance" data-section="tend-balance">
                {move || {
                    let bal = fin.tend_balance.get();
                    let oracle = fin.oracle_state.get();
                    view! {
                        <div
                            class=format!("tend-balance-card {}", bal.balance_class())
                            data-balance=bal.balance.to_string()
                            data-oracle-tier=format!("{:?}", oracle.tier)
                            data-oracle-vitality=oracle.vitality.to_string()
                            data-credit-limit=oracle.tier.credit_limit().to_string()
                        >
                            <div class="balance-main">
                                <span class="balance-hours" data-metric="balance">
                                    {format!("{:+} hours", bal.balance)}
                                </span>
                                <span class="balance-state">
                                    {bal.equilibrium_label()}
                                </span>
                            </div>
                            <div class="balance-context">
                                <span class="balance-given" data-metric="provided">
                                    {format!("{:.1}h given", bal.total_provided)}
                                </span>
                                <span class="balance-received" data-metric="received">
                                    {format!("{:.1}h received", bal.total_received)}
                                </span>
                                <span class="balance-exchanges" data-metric="exchanges">
                                    {format!("{} exchanges", bal.exchange_count)}
                                </span>
                            </div>
                            <div class=format!("oracle-indicator {}", oracle.tier.css_class()) data-component="oracle">
                                <span class="oracle-label">"community vitality"</span>
                                <span class="oracle-value">
                                    {format!("{} — ±{} hour limit", oracle.tier.label(), oracle.tier.credit_limit())}
                                </span>
                            </div>
                        </div>
                    }
                }}
            </section>

            // Exchange form (action-first)
            <section class="tend-exchange-section" aria-label="record an exchange" data-section="exchange-form">
                <h2 class="section-title">"Record care given"</h2>
                <form
                    class="exchange-form"
                    data-form="tend-exchange"
                    on:submit=move |ev| {
                        ev.prevent_default();
                        if can_submit.get_untracked() {
                            civic_actions::record_tend_exchange(
                                receiver_did.get_untracked(),
                                hours.get_untracked().parse::<f32>().unwrap_or(1.0),
                                description.get_untracked(),
                            );
                            set_receiver_did.set(String::new());
                            set_hours.set(String::new());
                            set_description.set(String::new());
                        }
                    }
                >
                    <div class="form-field">
                        <label for="exchange-receiver">"Who received your care?"</label>
                        <input
                            id="exchange-receiver"
                            type="text"
                            class="form-input"
                            placeholder="did:mycelix:..."
                            data-field="receiver-did"
                            prop:value=move || receiver_did.get()
                            on:input=move |ev| set_receiver_did.set(event_target_value(&ev))
                        />
                    </div>
                    <div class="form-field">
                        <label for="exchange-hours">"Hours of care"</label>
                        <input
                            id="exchange-hours"
                            type="number"
                            class="form-input"
                            step="0.5"
                            min="0.5"
                            placeholder="1.0"
                            data-field="hours"
                            prop:value=move || hours.get()
                            on:input=move |ev| set_hours.set(event_target_value(&ev))
                        />
                    </div>
                    <div class="form-field">
                        <label for="exchange-description">"What care did you give?"</label>
                        <textarea
                            id="exchange-description"
                            class="form-textarea"
                            rows="3"
                            placeholder="describe the care you provided"
                            data-field="description"
                            prop:value=move || description.get()
                            on:input=move |ev| set_description.set(event_target_value(&ev))
                        ></textarea>
                    </div>
                    <button
                        type="submit"
                        class="submit-btn"
                        data-action="record-exchange"
                        disabled=move || !can_submit.get()
                    >
                        "record this exchange"
                    </button>
                </form>
            </section>

            // Recent exchanges
            <section class="tend-history-section" aria-label="exchange history" data-section="exchange-history">
                <h2 class="section-title">"Recent exchanges"</h2>
                <div class="exchange-list" role="list">
                    {move || {
                        let exchanges = fin.tend_exchanges.get();
                        if exchanges.is_empty() {
                            view! {
                                <p class="empty-state">"no exchanges yet — care begins with giving"</p>
                            }.into_any()
                        } else {
                            exchanges.into_iter().map(|ex| {
                                let is_provider = ex.provider_did.contains("mock-citizen");
                                let direction = if is_provider { "gave" } else { "received" };
                                let other = if is_provider { &ex.receiver_did } else { &ex.provider_did };
                                let other_short = other.split(':').last().unwrap_or(other).to_string();
                                view! {
                                    <div
                                        class=format!("exchange-card exchange-{direction}")
                                        data-exchange-id=ex.id.clone()
                                        data-direction=direction
                                        data-hours=format!("{:.1}", ex.hours)
                                        data-status=format!("{:?}", ex.status)
                                        data-category=ex.service_category.label().to_string()
                                        role="listitem"
                                    >
                                        <div class="exchange-header">
                                            <span class="exchange-direction">
                                                {format!("{direction} {:.1}h", ex.hours)}
                                            </span>
                                            <span class="exchange-with">
                                                {format!("{} {other_short}",
                                                    if is_provider { "to" } else { "from" })}
                                            </span>
                                            <span class="exchange-status">{ex.status.label()}</span>
                                        </div>
                                        <p class="exchange-description">{ex.service_description}</p>
                                    </div>
                                }
                            }).collect_view().into_any()
                        }
                    }}
                </div>
            </section>

            // Marketplace (listings + requests)
            <section class="marketplace-section" aria-label="service marketplace" data-section="marketplace">
                <h2 class="section-title">"Marketplace"</h2>
                <div class="marketplace-grid">
                    <div class="marketplace-column" data-subsection="offerings">
                        <h3 class="subsection-title">"Offerings"</h3>
                        {move || {
                            fin.marketplace_listings.get().into_iter().map(|l| {
                                let cat = l.category.label().to_string();
                                let cat2 = cat.clone();
                                let hours_str = format!("~{:.1}h", l.estimated_hours);
                                view! {
                                    <div
                                        class="listing-card"
                                        data-listing-id=l.id.clone()
                                        data-category=cat
                                        data-hours=format!("{:.1}", l.estimated_hours)
                                        role="listitem"
                                    >
                                        <h4 class="listing-title">{l.title}</h4>
                                        <p class="listing-desc">{l.description}</p>
                                        <div class="listing-meta">
                                            <span>{cat2}</span>
                                            <span>{hours_str}</span>
                                        </div>
                                    </div>
                                }
                            }).collect_view()
                        }}
                    </div>
                    <div class="marketplace-column" data-subsection="needs">
                        <h3 class="subsection-title">"Needs"</h3>
                        {move || {
                            fin.marketplace_requests.get().into_iter().map(|r| {
                                let cat = r.category.label().to_string();
                                let cat2 = cat.clone();
                                let hours_str = format!("~{:.1}h", r.estimated_hours);
                                view! {
                                    <div
                                        class="request-card"
                                        data-request-id=r.id.clone()
                                        data-category=cat
                                        data-hours=format!("{:.1}", r.estimated_hours)
                                        role="listitem"
                                    >
                                        <h4 class="request-title">{r.title}</h4>
                                        <p class="request-desc">{r.description}</p>
                                        <div class="request-meta">
                                            <span>{cat2}</span>
                                            <span>{hours_str}</span>
                                        </div>
                                    </div>
                                }
                            }).collect_view()
                        }}
                    </div>
                </div>
            </section>
        </div>
    }
}
