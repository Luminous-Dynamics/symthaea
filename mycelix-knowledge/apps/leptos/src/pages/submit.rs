// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use knowledge_leptos_types::ClaimType;
use crate::actions;

#[component]
pub fn SubmitPage() -> impl IntoView {
    // Form state
    let (content, set_content) = signal(String::new());
    let (claim_type, set_claim_type) = signal("Fact".to_string());
    let (tags, set_tags) = signal(String::new());
    let (sources, set_sources) = signal(String::new());
    let (show_form, set_show_form) = signal(false);

    let can_submit = Memo::new(move |_| {
        !content.get().trim().is_empty()
    });

    let on_submit = move |ev: leptos::ev::SubmitEvent| {
        ev.prevent_default();
        if can_submit.get_untracked() {
            let ct = match claim_type.get_untracked().as_str() {
                "Opinion" => ClaimType::Opinion,
                "Prediction" => ClaimType::Prediction,
                "Hypothesis" => ClaimType::Hypothesis,
                "Definition" => ClaimType::Definition,
                "Historical" => ClaimType::Historical,
                "Normative" => ClaimType::Normative,
                "Narrative" => ClaimType::Narrative,
                _ => ClaimType::Fact,
            };
            let tags_vec: Vec<String> = tags
                .get_untracked()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let sources_vec: Vec<String> = sources
                .get_untracked()
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            actions::submit_claim(content.get_untracked(), ct, tags_vec, sources_vec);
            set_content.set(String::new());
            set_tags.set(String::new());
            set_sources.set(String::new());
            set_show_form.set(false);
        }
    };

    view! {
        <div class="page-submit">
            <h1>"Submit a Claim"</h1>
            <p class="subtitle">"Contribute knowledge to the commons."</p>

            <button
                class="btn btn-primary"
                on:click=move |_| set_show_form.update(|s| *s = !*s)
            >
                {move || if show_form.get() { "Cancel" } else { "New Claim" }}
            </button>

            <div style=move || if show_form.get() { "display: block" } else { "display: none" }>
                <form class="create-form" on:submit=on_submit>
                    <div class="form-field">
                        <label>"Content"</label>
                        <textarea class="form-input" rows="4"
                            placeholder="State the claim clearly..."
                            prop:value=move || content.get()
                            on:input=move |ev| set_content.set(event_target_value(&ev))
                        />
                    </div>
                    <div class="form-field">
                        <label>"Claim Type"</label>
                        <select class="form-select"
                            on:change=move |ev| set_claim_type.set(event_target_value(&ev))
                        >
                            {ClaimType::all().iter().map(|ct| {
                                let val = ct.label();
                                view! { <option value=val>{val}</option> }
                            }).collect_view()}
                        </select>
                    </div>
                    <div class="form-field">
                        <label>"Tags (comma-separated)"</label>
                        <input class="form-input" type="text"
                            placeholder="e.g. physics, energy, climate"
                            prop:value=move || tags.get()
                            on:input=move |ev| set_tags.set(event_target_value(&ev))
                        />
                    </div>
                    <div class="form-field">
                        <label>"Sources (one per line)"</label>
                        <textarea class="form-input" rows="3"
                            placeholder="https://example.com/paper.pdf&#10;DOI:10.1234/example"
                            prop:value=move || sources.get()
                            on:input=move |ev| set_sources.set(event_target_value(&ev))
                        />
                    </div>
                    <button type="submit" class="btn btn-primary" disabled=move || !can_submit.get()>
                        "Submit Claim"
                    </button>
                </form>
            </div>
        </div>
    }
}
