// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Submit new epistemic claim.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

/// Claim submission form.
#[component]
pub fn SubmitPage() -> impl IntoView {
    let (description, set_description) = signal(String::new());
    let (category, set_category) = signal("Nuclear Physics".to_string());
    let (submitted, set_submitted) = signal(false);

    let on_submit = move |_| {
        if !description.get().is_empty() {
            set_submitted.set(true);
        }
    };

    view! {
        <div class="page-container">
            <h1 class="page-title">"Submit Epistemic Claim"</h1>

            {move || if submitted.get() {
                view! {
                    <div class="glass-panel" style="text-align: center; padding: 2rem;">
                        <h2 style="color: var(--accent-emerald); margin-bottom: 1rem;">"Claim Submitted"</h2>
                        <p style="color: var(--text-secondary);">"Your claim has been registered as E0 (Unverified). It will be classified by the discovery bridge and made available for peer verification."</p>
                        <button class="btn btn-primary" style="margin-top: 1rem;" on:click=move |_| set_submitted.set(false)>"Submit Another"</button>
                    </div>
                }.into_any()
            } else {
                view! {
                    <div class="glass-panel">
                        <div style="margin-bottom: 1rem;">
                            <label style="font-size: 0.875rem; color: var(--text-secondary); display: block; margin-bottom: 0.375rem;">"Description"</label>
                            <textarea
                                style="width: 100%; min-height: 120px; padding: 0.75rem; background: var(--bg-secondary); border: 1px solid var(--border-glass); border-radius: 0.375rem; color: var(--text-primary); font-size: 0.875rem; resize: vertical;"
                                placeholder="Describe your scientific claim..."
                                on:input=move |ev| {
                                    use wasm_bindgen::JsCast;
                                    let target: web_sys::HtmlTextAreaElement = ev.target().unwrap().unchecked_into();
                                    set_description.set(target.value());
                                }
                            ></textarea>
                        </div>

                        <div style="margin-bottom: 1rem;">
                            <label style="font-size: 0.875rem; color: var(--text-secondary); display: block; margin-bottom: 0.375rem;">"Category"</label>
                            <select
                                style="width: 100%; padding: 0.5rem; background: var(--bg-secondary); border: 1px solid var(--border-glass); border-radius: 0.375rem; color: var(--text-primary);"
                                on:change=move |ev| {
                                    use wasm_bindgen::JsCast;
                                    let target: web_sys::HtmlSelectElement = ev.target().unwrap().unchecked_into();
                                    set_category.set(target.value());
                                }
                            >
                                <option>"Nuclear Physics"</option>
                                <option>"Modified Gravity"</option>
                                <option>"Metamaterials"</option>
                                <option>"Quantum Mechanics"</option>
                                <option>"General Relativity"</option>
                                <option>"Information Theory"</option>
                                <option>"Biophysics"</option>
                                <option>"Thermodynamics"</option>
                                <option>"Other"</option>
                            </select>
                        </div>

                        <div style="display: flex; gap: 1rem; align-items: center;">
                            <button class="btn btn-primary" on:click=on_submit>"Submit Claim"</button>
                            <span style="font-size: 0.75rem; color: var(--text-secondary);">
                                "Claims start at E0 (Unverified) and advance with peer verification."
                            </span>
                        </div>
                    </div>
                }.into_any()
            }}
        </div>
    }
}
