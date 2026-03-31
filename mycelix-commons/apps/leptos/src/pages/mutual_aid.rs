// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::contexts::commons_context::use_commons;
use crate::contexts::commons_actions;
use commons_leptos_types::*;

#[component]
pub fn MutualAidPage() -> impl IntoView {
    let commons = use_commons();
    let (title, set_title) = signal(String::new());
    let (desc, set_desc) = signal(String::new());

    view! {
        <div class="mutual-aid-page" data-page="mutual-aid" role="main">
            <h1 class="page-title">"Mutual Aid"</h1>
            <p class="page-subtitle">"share needs and offer help — care flows both ways"</p>

            // Post a need
            <section class="post-need-section" data-section="post-need">
                <h2 class="section-title">"Share a need"</h2>
                <form class="need-form" data-form="post-need" on:submit=move |ev| {
                    ev.prevent_default();
                    if !title.get_untracked().trim().is_empty() {
                        commons_actions::post_need(title.get_untracked(), desc.get_untracked(), NeedCategory::Other("General".into()), Urgency::Medium);
                        set_title.set(String::new()); set_desc.set(String::new());
                    }
                }>
                    <div class="form-field">
                        <label for="need-title">"What do you need?"</label>
                        <input id="need-title" type="text" class="form-input" placeholder="a brief description" data-field="title"
                            prop:value=move || title.get() on:input=move |ev| set_title.set(event_target_value(&ev)) />
                    </div>
                    <div class="form-field">
                        <label for="need-desc">"Details"</label>
                        <textarea id="need-desc" class="form-textarea" rows="3" placeholder="when, where, how much" data-field="description"
                            prop:value=move || desc.get() on:input=move |ev| set_desc.set(event_target_value(&ev))></textarea>
                    </div>
                    <button type="submit" class="submit-btn" data-action="post-need" disabled=move || title.get().trim().is_empty()>"share this need"</button>
                </form>
            </section>

            // Open needs
            <section class="needs-section" data-section="needs" aria-label="open needs">
                <h2 class="section-title">"Open needs"</h2>
                <div class="needs-list" role="list">
                    {move || {
                        let needs: Vec<_> = commons.needs.get().into_iter().filter(|n| n.status == NeedStatus::Open).collect();
                        if needs.is_empty() {
                            view! { <p class="empty-state">"all needs are met — the commons is whole"</p> }.into_any()
                        } else {
                            needs.into_iter().map(|n| {
                                let cat_label = n.category.label().to_string();
                                let cat_label2 = cat_label.clone();
                                let urg_class = n.urgency.css_class().to_string();
                                let urg_label = n.urgency.label().to_string();
                                let urg_label2 = urg_label.clone();
                                let requester = n.requester_did.split(':').last().unwrap_or(&n.requester_did).to_string();
                                view! {
                                    <div class=format!("need-card {urg_class}") data-need-id=n.id.clone() data-urgency=urg_label data-category=cat_label role="listitem">
                                        <div class="need-header">
                                            <span class="need-urgency">{urg_label2}</span>
                                            <span class="need-category">{cat_label2}</span>
                                        </div>
                                        <h3 class="need-title">{n.title}</h3>
                                        <p class="need-desc">{n.description}</p>
                                        <span class="need-requester">{format!("from {requester}")}</span>
                                    </div>
                                }
                            }).collect_view().into_any()
                        }
                    }}
                </div>
            </section>

            // Offers
            <section class="offers-section" data-section="offers" aria-label="standing offers">
                <h2 class="section-title">"Standing offers"</h2>
                <div class="offers-list" role="list">
                    {move || {
                        commons.offers.get().into_iter().map(|o| {
                            let cat = o.category.label().to_string();
                            let cat2 = cat.clone();
                            let offerer = o.offerer_did.split(':').last().unwrap_or(&o.offerer_did).to_string();
                            view! {
                                <div class="offer-card" data-offer-id=o.id.clone() data-category=cat role="listitem">
                                    <h3 class="offer-title">{o.title}</h3>
                                    <p class="offer-desc">{o.description}</p>
                                    <div class="offer-meta">
                                        <span>{cat2}</span>
                                        <span>{format!("from {offerer}")}</span>
                                    </div>
                                </div>
                            }
                        }).collect_view()
                    }}
                </div>
            </section>
        </div>
    }
}
