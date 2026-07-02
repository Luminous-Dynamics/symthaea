// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Submit new epistemic claim.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

/// Claim submission form with auto-LEM classification preview.
#[component]
pub fn SubmitPage() -> impl IntoView {
    let (description, set_description) = signal(String::new());
    let (category, set_category) = signal("Nuclear Physics".to_string());
    let (keywords, set_keywords) = signal(String::new());
    let (dataset_hash, set_dataset_hash) = signal(String::new());
    let (license, set_license) = signal("CC-BY-4.0".to_string());
    let (creator, set_creator) = signal(String::new());
    let (submitted, set_submitted) = signal(false);

    // Auto-LEM classification preview using WASM catalog search
    let lem_preview = move || {
        let desc = description.get();
        if desc.len() < 10 { return ("—".to_string(), "—".to_string(), "—".to_string(), None); }

        // Run WASM structural search
        let catalog = symthaea_physics_catalog::catalog::PhysicsCatalog::new();
        let results = symthaea_physics_catalog::search::search_by_text(&catalog, &desc, 1);
        let (nearest, similarity) = results
            .first()
            .map(|r| (Some(r.name.clone()), r.score as f64))
            .unwrap_or((None, 0.0));

        // E-axis from keywords (simulation confidence proxy)
        let desc_lower = desc.to_lowercase();
        let e = if desc_lower.contains("replicated") || desc_lower.contains("reproduced") { "E3" }
                else if desc_lower.contains("verified") || desc_lower.contains("confirmed") { "E2" }
                else if desc_lower.contains("observed") || desc_lower.contains("measured") { "E1" }
                else { "E0" };

        // N-axis from catalog similarity
        let n = if similarity > 0.9 { "N3" }
                else if similarity > 0.6 { "N2" }
                else if similarity > 0.3 { "N1" }
                else { "N0" };

        // M-axis from domain keywords
        let m = if desc_lower.contains("relativity") || desc_lower.contains("quantum") { "M3" }
                else if desc_lower.contains("nuclear") || desc_lower.contains("electromagnetic") { "M2" }
                else if desc_lower.contains("fluid") || desc_lower.contains("thermal") { "M1" }
                else { "M0" };

        (e.to_string(), n.to_string(), m.to_string(), nearest)
    };

    let on_submit = move |_| {
        if description.get().is_empty() {
            return;
        }
        let desc = description.get();
        let cat = category.get();
        leptos::task::spawn_local(async move {
            let req = crate::types::CreateClaimRequest {
                tier: crate::types::EpistemicTier::E0,
                content: crate::types::ClaimContent {
                    dataset_hash: String::new(),
                    description: desc,
                    category: cat,
                    keywords: vec![],
                    storage_ref: None,
                    reproducibility_score: None,
                    license: None,
                },
                creator: "anonymous".to_string(),
            };
            let _ = crate::api::create_claim(&req).await;
        });
        set_submitted.set(true);
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

                        // Additional fields
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                            <div>
                                <label style="font-size: 0.875rem; color: var(--text-secondary); display: block; margin-bottom: 0.375rem;">"Keywords (comma-separated)"</label>
                                <input type="text" placeholder="e.g., fusion, plasma, ignition"
                                    style="width: 100%; padding: 0.5rem; background: var(--bg-secondary); border: 1px solid var(--border-glass); border-radius: 0.375rem; color: var(--text-primary); font-size: 0.875rem;"
                                    on:input=move |ev| { use wasm_bindgen::JsCast; let t: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into(); set_keywords.set(t.value()); }
                                />
                            </div>
                            <div>
                                <label style="font-size: 0.875rem; color: var(--text-secondary); display: block; margin-bottom: 0.375rem;">"Creator / Author"</label>
                                <input type="text" placeholder="Your name or institution"
                                    style="width: 100%; padding: 0.5rem; background: var(--bg-secondary); border: 1px solid var(--border-glass); border-radius: 0.375rem; color: var(--text-primary); font-size: 0.875rem;"
                                    on:input=move |ev| { use wasm_bindgen::JsCast; let t: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into(); set_creator.set(t.value()); }
                                />
                            </div>
                        </div>

                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                            <div>
                                <label style="font-size: 0.875rem; color: var(--text-secondary); display: block; margin-bottom: 0.375rem;">"Dataset Hash (optional)"</label>
                                <input type="text" placeholder="blake3:... or IPFS CID"
                                    style="width: 100%; padding: 0.5rem; background: var(--bg-secondary); border: 1px solid var(--border-glass); border-radius: 0.375rem; color: var(--text-primary); font-size: 0.875rem; font-family: monospace;"
                                    on:input=move |ev| { use wasm_bindgen::JsCast; let t: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into(); set_dataset_hash.set(t.value()); }
                                />
                            </div>
                            <div>
                                <label style="font-size: 0.875rem; color: var(--text-secondary); display: block; margin-bottom: 0.375rem;">"License"</label>
                                <select
                                    style="width: 100%; padding: 0.5rem; background: var(--bg-secondary); border: 1px solid var(--border-glass); border-radius: 0.375rem; color: var(--text-primary);"
                                    on:change=move |ev| { use wasm_bindgen::JsCast; let t: web_sys::HtmlSelectElement = ev.target().unwrap().unchecked_into(); set_license.set(t.value()); }
                                >
                                    <option>"CC-BY-4.0"</option>
                                    <option>"CC-BY-SA-4.0"</option>
                                    <option>"CC-BY-NC-4.0"</option>
                                    <option>"MIT"</option>
                                    <option>"AGPL-3.0"</option>
                                    <option>"All Rights Reserved"</option>
                                </select>
                            </div>
                        </div>

                        // Auto-LEM classification preview
                        <div style="margin-bottom: 1rem; padding: 0.75rem; background: var(--bg-secondary); border-radius: 0.5rem; border: 1px dashed var(--border-glass);">
                            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.375rem;">"Auto-Classification Preview:"</div>
                            {move || {
                                let (e, n, m, nearest) = lem_preview();
                                view! {
                                    <div style="display: flex; gap: 1rem; font-size: 0.875rem; flex-wrap: wrap;">
                                        <span>"E: " <strong style="color: var(--accent-indigo);">{e}</strong></span>
                                        <span>"N: " <strong style="color: var(--accent-indigo);">{n}</strong></span>
                                        <span>"M: " <strong style="color: var(--accent-indigo);">{m}</strong></span>
                                        {nearest.map(|n| view! {
                                            <span style="font-size: 0.75rem; color: var(--accent-emerald);">"Nearest: " {n}</span>
                                        })}
                                    </div>
                                }
                            }}
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
