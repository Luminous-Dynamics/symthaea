// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verification workflow page.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn VerifyPage() -> impl IntoView {
    let (evidence, set_evidence) = signal(String::new());
    let (methodology, set_methodology) = signal("Exact".to_string());
    let (submitted, set_submitted) = signal(false);

    view! {
        <div class="page-container">
            <h1 class="page-title">"Verify Claim"</h1>

            {move || if submitted.get() {
                view! {
                    <div class="glass-panel" style="text-align: center; padding: 2rem;">
                        <h2 style="color: var(--accent-emerald);">"Verification Submitted"</h2>
                        <p style="color: var(--text-secondary); margin-top: 0.5rem;">"Your verification has been recorded. The claim's epistemic tier will be re-evaluated."</p>
                        <a href="/browse" class="btn btn-primary" style="display: inline-block; margin-top: 1rem; text-decoration: none;">"Back to Claims"</a>
                    </div>
                }.into_any()
            } else {
                view! {
                    <div class="glass-panel">
                        <div style="margin-bottom: 1rem;">
                            <label style="font-size: 0.875rem; color: var(--text-secondary); display: block; margin-bottom: 0.375rem;">"Evidence / Replication Description"</label>
                            <textarea
                                style="width: 100%; min-height: 120px; padding: 0.75rem; background: var(--bg-secondary); border: 1px solid var(--border-glass); border-radius: 0.375rem; color: var(--text-primary); font-size: 0.875rem; resize: vertical;"
                                placeholder="Describe your verification evidence, replication attempt, or review findings..."
                                on:input=move |ev| {
                                    let target: web_sys::HtmlTextAreaElement = ev.target().unwrap().unchecked_into();
                                    set_evidence.set(target.value());
                                }
                            ></textarea>
                        </div>

                        <div style="margin-bottom: 1rem;">
                            <label style="font-size: 0.875rem; color: var(--text-secondary); display: block; margin-bottom: 0.375rem;">"Methodology Match"</label>
                            <select
                                style="width: 100%; padding: 0.5rem; background: var(--bg-secondary); border: 1px solid var(--border-glass); border-radius: 0.375rem; color: var(--text-primary);"
                                on:change=move |ev| {
                                    let target: web_sys::HtmlSelectElement = ev.target().unwrap().unchecked_into();
                                    set_methodology.set(target.value());
                                }
                            >
                                <option value="Exact">"Exact — Same methodology, same conditions"</option>
                                <option value="Close">"Close — Minor variations in methodology"</option>
                                <option value="Conceptual">"Conceptual — Different approach, same question"</option>
                                <option value="Extension">"Extension — Builds on original with new data"</option>
                            </select>
                        </div>

                        <div style="margin-bottom: 1rem;">
                            <label style="font-size: 0.875rem; color: var(--text-secondary); display: block; margin-bottom: 0.375rem;">"Outcome"</label>
                            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                                <button class="btn btn-emerald" on:click=move |_| set_submitted.set(true)>"Full Replication"</button>
                                <button class="btn" style="background: var(--tier-e2); color: white;" on:click=move |_| set_submitted.set(true)>"Partial Replication"</button>
                                <button class="btn" style="background: var(--tier-e0); color: white;" on:click=move |_| set_submitted.set(true)>"Failure to Replicate"</button>
                                <button class="btn" style="background: var(--text-secondary); color: white;" on:click=move |_| set_submitted.set(true)>"Inconclusive"</button>
                            </div>
                        </div>
                    </div>
                }.into_any()
            }}
        </div>
    }
}
