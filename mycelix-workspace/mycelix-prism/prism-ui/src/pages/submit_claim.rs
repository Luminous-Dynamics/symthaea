// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim submission page — let users contribute their own knowledge.
//!
//! The voice of the people: users don't just consume knowledge,
//! they contribute it with evidence levels and sources.

use crate::holochain::DhtClaim;
use leptos::prelude::*;
use prism_common::{EmpiricalLevel, MaterialityLevel, NormativeLevel};

#[component]
pub fn SubmitClaimPage() -> impl IntoView {
    let (content, set_content) = signal(String::new());
    let (source, set_source) = signal(String::new());
    let (tags, set_tags) = signal(String::new());
    let (e_level, set_e_level) = signal("E3".to_string());
    let (submitted, set_submitted) = signal(false);
    let (submitting, set_submitting) = signal(false);

    let on_submit = move |ev: leptos::ev::SubmitEvent| {
        ev.prevent_default();
        let claim_text = content.get();
        if claim_text.trim().is_empty() {
            return;
        }

        set_submitting.set(true);

        let e = match e_level.get().as_str() {
            "E4" => EmpiricalLevel::E4,
            "E3" => EmpiricalLevel::E3,
            "E2" => EmpiricalLevel::E2,
            "E1" => EmpiricalLevel::E1,
            _ => EmpiricalLevel::E0,
        };

        let claim = DhtClaim {
            content: claim_text.clone(),
            empirical: e,
            normative: NormativeLevel::N2,
            materiality: MaterialityLevel::M2,
            sources: source
                .get()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            tags: tags
                .get()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            claim_type: "Fact".to_string(),
            confidence: 0.8,
        };

        wasm_bindgen_futures::spawn_local(async move {
            let _ = crate::holochain::publish_claim(&claim).await;
            set_submitted.set(true);
            set_submitting.set(false);
        });
    };

    view! {
        <div class="reader-content">
            <h1>"Submit a Claim"</h1>
            <p style="color: var(--content-text-secondary); margin-bottom: 24px;">
                "Contribute knowledge to the Prism network. Every claim is epistemically "
                "classified and stored with its evidence level and sources."
            </p>

            <Show when=move || !submitted.get()>
                <form class="claim-submit-form" on:submit=on_submit>
                    <div class="claim-form-group">
                        <label class="claim-form-label">"Claim (a factual statement)"</label>
                        <textarea
                            class="claim-form-textarea"
                            placeholder="e.g., The speed of light in vacuum is approximately 299,792,458 meters per second"
                            prop:value=content
                            on:input:target=move |ev| set_content.set(ev.target().value())
                        ></textarea>
                    </div>

                    <div class="claim-form-group">
                        <label class="claim-form-label">"Evidence Level"</label>
                        <select
                            class="claim-form-select"
                            prop:value=e_level
                            on:change:target=move |ev| set_e_level.set(ev.target().value())
                        >
                            <option value="E4">"E4 — Established (peer-reviewed, authoritative)"</option>
                            <option value="E3" selected>"E3 — Replicated (verified by multiple sources)"</option>
                            <option value="E2">"E2 — Tested (some evidence, needs more)"</option>
                            <option value="E1">"E1 — Preliminary (initial observation)"</option>
                            <option value="E0">"E0 — Unverified (personal claim)"</option>
                        </select>
                    </div>

                    <div class="claim-form-group">
                        <label class="claim-form-label">"Source(s) — comma separated"</label>
                        <input
                            class="claim-form-input"
                            type="text"
                            placeholder="e.g., NIST, Wikipedia, doi:10.1234/5678"
                            prop:value=source
                            on:input:target=move |ev| set_source.set(ev.target().value())
                        />
                    </div>

                    <div class="claim-form-group">
                        <label class="claim-form-label">"Tags — comma separated"</label>
                        <input
                            class="claim-form-input"
                            type="text"
                            placeholder="e.g., physics, light, constants"
                            prop:value=tags
                            on:input:target=move |ev| set_tags.set(ev.target().value())
                        />
                    </div>

                    <button
                        class="claim-submit-btn"
                        type="submit"
                        disabled=move || submitting.get() || content.get().trim().is_empty()
                    >
                        {move || if submitting.get() { "Submitting..." } else { "Submit Claim" }}
                    </button>
                </form>
            </Show>

            <Show when=move || submitted.get()>
                <div class="claim-success">
                    "Claim saved locally and queued for DHT sync. "
                    "When the Holochain conductor is connected, it will be published to the Mycelix knowledge network."
                </div>
                <button
                    class="claim-submit-btn"
                    style="margin-top: 16px;"
                    on:click=move |_| {
                        set_submitted.set(false);
                        set_content.set(String::new());
                        set_source.set(String::new());
                        set_tags.set(String::new());
                    }
                >
                    "Submit Another"
                </button>
            </Show>
        </div>
    }
}
