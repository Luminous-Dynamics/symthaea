// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_knowledge_context;

#[component]
pub fn BrowsePage() -> impl IntoView {
    let ctx = use_knowledge_context();
    let (query, set_query) = signal(String::new());

    let filtered = move || {
        let q = query.get().to_lowercase();
        ctx.claims.get().iter().filter(|c| {
            q.is_empty() || c.content.to_lowercase().contains(&q) || c.tags.iter().any(|t| t.to_lowercase().contains(&q))
        }).cloned().collect::<Vec<_>>()
    };

    view! {
        <div class="page-browse">
            <h1>"Browse Claims"</h1>
            <div class="search-bar">
                <input class="form-input search-input" type="search" placeholder="Search claims or tags..."
                    prop:value=move || query.get()
                    on:input=move |ev| set_query.set(event_target_value(&ev))
                />
            </div>

            <div class="claim-list">
                {move || {
                    let claims = filtered();
                    if claims.is_empty() {
                        view! { <p class="no-results">"No claims match your search."</p> }.into_any()
                    } else {
                        claims.iter().map(|c| {
                            let id = c.id.clone();
                            let content = c.content.clone();
                            let ct = c.claim_type.label();
                            let e = c.classification.empirical.label();
                            let n = c.classification.normative.label();
                            let m = c.classification.materiality.label();
                            let conf = c.classification.confidence;
                            let tags = c.tags.clone();
                            let author = c.author.clone();
                            view! {
                                <div class="claim-card">
                                    <div class="claim-header">
                                        <span class="claim-id">{id}</span>
                                        <span class="claim-type-badge">{ct}</span>
                                    </div>
                                    <p class="claim-content">{content}</p>
                                    <div class="epistemic-bar">
                                        <span class="e-tag">{e}</span>
                                        <span class="n-tag">{n}</span>
                                        <span class="m-tag">{m}</span>
                                        <span class="conf-tag">{format!("{:.0}%", conf * 100.0)}</span>
                                    </div>
                                    <div class="claim-tags">
                                        {tags.iter().map(|t| {
                                            let tag = t.clone();
                                            view! { <span class="tag">{tag}</span> }
                                        }).collect_view()}
                                    </div>
                                    <span class="claim-author">{author}</span>
                                </div>
                            }
                        }).collect_view().into_any()
                    }
                }}
            </div>
        </div>
    }
}
