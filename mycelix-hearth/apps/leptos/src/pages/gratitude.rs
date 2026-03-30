// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::hearth_context::{use_hearth, member_name};
use crate::hearth_actions;
use crate::components::TierGate;
use personal_leptos_types::TrustTier;

#[component]
pub fn GratitudePage() -> impl IntoView {
    let hearth = use_hearth();

    let (show_form, set_show_form) = signal(false);
    let (recipient, set_recipient) = signal(String::new());
    let (message, set_message) = signal(String::new());

    let submit_gratitude = move |_| {
        let to = recipient.get();
        let msg = message.get();
        if to.is_empty() || msg.is_empty() { return; }

        hearth_actions::express_gratitude(to, msg);

        set_message.set(String::new());
        set_recipient.set(String::new());
        set_show_form.set(false);
    };

    view! {
        <div class="page gratitude-page">
            <h1 class="page-title">"gratitude wall"</h1>
            <p class="page-subtitle">"expressions of appreciation that strengthen bonds"</p>

            <TierGate min_tier=TrustTier::Basic action_label="express gratitude">
                <button
                    class="action-btn"
                    on:click=move |_| set_show_form.set(!show_form.get())
                >
                    {move || if show_form.get() { "Cancel" } else { "+ Express Gratitude" }}
                </button>
            </TierGate>

            // Gratitude form
            {move || show_form.get().then(|| {
                let members = hearth.members.get();
                let my_agent = hearth.my_agent.get();
                let others: Vec<_> = members.iter()
                    .filter(|m| m.agent != my_agent)
                    .cloned().collect();

                view! {
                    <div class="form-card">
                        <div class="form-row">
                            <label>"To"</label>
                            <select on:change=move |ev| {
                                use wasm_bindgen::JsCast;
                                if let Some(sel) = ev.target().and_then(|t| t.dyn_ref::<web_sys::HtmlSelectElement>().cloned()) {
                                    set_recipient.set(sel.value());
                                }
                            }>
                                <option value="">"Select someone..."</option>
                                {others.iter().map(|m| {
                                    let agent = m.agent.clone();
                                    let name = m.display_name.clone();
                                    view! { <option value=agent>{name}</option> }
                                }).collect_view()}
                            </select>
                        </div>
                        <div class="form-row">
                            <label>"Message"</label>
                            <input
                                type="text"
                                placeholder="What are you grateful for?"
                                prop:value=move || message.get()
                                on:input=move |ev| {
                                    use wasm_bindgen::JsCast;
                                    if let Some(input) = ev.target().and_then(|t| t.dyn_ref::<web_sys::HtmlInputElement>().cloned()) {
                                        set_message.set(input.value());
                                    }
                                }
                            />
                        </div>
                        <button class="action-btn" on:click=submit_gratitude>"Send"</button>
                    </div>
                }
            })}

            // Gratitude stream
            {move || {
                let members = hearth.members.get();
                let gratitude = hearth.gratitude.get();

                if gratitude.is_empty() {
                    view! {
                        <div class="empty-state">"the wall is waiting for its first mark of warmth. who do you appreciate?"</div>
                    }.into_any()
                } else {
                    view! {
                        <div class="gratitude-stream">
                            {gratitude.iter().map(|g| {
                                let from = member_name(&members, &g.from_agent);
                                let to = member_name(&members, &g.to_agent);
                                let msg = g.message.clone();
                                let gtype = format!("{:?}", g.gratitude_type);
                                view! {
                                    <div class="gratitude-card">
                                        <div class="gratitude-header">
                                            <span class="gratitude-from">{from}</span>
                                            <span class="gratitude-arrow">" → "</span>
                                            <span class="gratitude-to">{to}</span>
                                            <span class="gratitude-type">{gtype}</span>
                                        </div>
                                        <p class="gratitude-message">"“" {msg} "”"</p>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
