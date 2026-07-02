// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Create proposal page: tier-gated proposal planting.

use leptos::prelude::*;
use personal_leptos_types::TrustTier;
use governance_leptos_types::ProposalType;
use crate::components::TierGate;
use crate::contexts::civic_actions;

#[component]
pub fn CreateProposalPage() -> impl IntoView {
    let (title, set_title) = signal(String::new());
    let (description, set_description) = signal(String::new());
    let (proposal_type, set_proposal_type) = signal(ProposalType::Standard);

    let can_submit = Memo::new(move |_| {
        !title.get().trim().is_empty() && !description.get().trim().is_empty()
    });

    view! {
        <div class="create-proposal-page">
            <h1 class="page-title">"Plant a new proposal"</h1>

            <TierGate min_tier=TrustTier::Basic action="creating proposals">
                <form class="proposal-form" on:submit=move |ev| {
                    ev.prevent_default();
                    if can_submit.get_untracked() {
                        civic_actions::create_proposal(
                            title.get_untracked(),
                            description.get_untracked(),
                            proposal_type.get_untracked(),
                        );
                        set_title.set(String::new());
                        set_description.set(String::new());
                    }
                }>
                    <div class="form-field">
                        <label for="proposal-title">"What is growing?"</label>
                        <input
                            id="proposal-title"
                            type="text"
                            class="form-input"
                            placeholder="a clear title for your proposal"
                            prop:value=move || title.get()
                            on:input=move |ev| {
                                set_title.set(event_target_value(&ev));
                            }
                        />
                    </div>

                    <div class="form-field">
                        <label for="proposal-type">"What kind?"</label>
                        <select
                            id="proposal-type"
                            class="form-select"
                            on:change=move |ev| {
                                let val = event_target_value(&ev);
                                set_proposal_type.set(match val.as_str() {
                                    "Emergency" => ProposalType::Emergency,
                                    "Constitutional" => ProposalType::Constitutional,
                                    "Parameter" => ProposalType::Parameter,
                                    "Funding" => ProposalType::Funding,
                                    _ => ProposalType::Standard,
                                });
                            }
                        >
                            <option value="Standard" selected>"Standard"</option>
                            <option value="Funding">"Funding"</option>
                            <option value="Parameter">"Parameter Change"</option>
                            <option value="Emergency">"Emergency"</option>
                            <option value="Constitutional">"Constitutional"</option>
                        </select>
                    </div>

                    <div class="form-field">
                        <label for="proposal-description">"Describe what you envision"</label>
                        <textarea
                            id="proposal-description"
                            class="form-textarea"
                            rows="8"
                            placeholder="why does this matter? what changes?"
                            prop:value=move || description.get()
                            on:input=move |ev| {
                                set_description.set(event_target_value(&ev));
                            }
                        ></textarea>
                    </div>

                    <button
                        type="submit"
                        class="submit-btn"
                        disabled=move || !can_submit.get()
                    >
                        "plant this proposal"
                    </button>
                </form>
            </TierGate>
        </div>
    }
}
