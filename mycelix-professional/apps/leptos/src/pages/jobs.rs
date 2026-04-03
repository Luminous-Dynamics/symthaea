// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Jobs page — search, browse, and post job opportunities.
//! Job matching runs client-side using the matching engine.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn JobsPage() -> impl IntoView {
    let (search_query, set_search_query) = signal(String::new());

    view! {
        <div class="page jobs-page">
            <h1>"Job Opportunities"</h1>

            <div class="search-bar">
                <input
                    type="text"
                    placeholder="Search by skill (e.g., rust, holochain, python)..."
                    class="search-input"
                    aria-label="Search jobs by skill"
                    prop:value=move || search_query.get()
                    on:input=move |ev| {
                        let input: web_sys::HtmlInputElement = ev.target().unwrap().dyn_into().unwrap();
                        set_search_query.set(input.value());
                    }
                />
                <button class="btn-primary">"Search"</button>
            </div>

            <div class="jobs-grid">
                <div class="job-card">
                    <h3>"Rust Developer"</h3>
                    <p class="job-org">"Luminous Dynamics"</p>
                    <div class="job-skills">
                        <span class="skill-tag">"rust"</span>
                        <span class="skill-tag">"holochain"</span>
                        <span class="skill-tag">"wasm"</span>
                    </div>
                    <div class="job-meta">
                        <span>"Remote"</span>
                        <span>"$80k-$120k"</span>
                    </div>
                </div>

                <div class="placeholder-card">
                    <p>"Job listings will appear here when the Holochain conductor is connected."</p>
                    <p class="text-secondary">"Post a job or search by skill to get started."</p>
                </div>
            </div>

            <div class="post-job-section">
                <h3>"Post a Job"</h3>
                <p class="text-secondary">"Create a job posting visible to the entire professional network."</p>
                <button class="btn-primary" disabled title="Connect to Holochain conductor to post jobs">"Create Job Posting"</button>
            </div>
        </div>
    }
}
