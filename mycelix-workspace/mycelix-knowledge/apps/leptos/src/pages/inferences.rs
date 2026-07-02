// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_knowledge_context;

#[component]
pub fn InferencesPage() -> impl IntoView {
    let ctx = use_knowledge_context();

    view! {
        <div class="page-inferences">
            <h1>"Inferences"</h1>
            <p class="subtitle">"Automated patterns and conclusions from the knowledge graph."</p>

            <div class="inference-list">
                {move || ctx.inferences.get().iter().map(|inf| {
                    let itype = inf.inference_type.label();
                    let conclusion = inf.conclusion.clone();
                    let reasoning = inf.reasoning.clone();
                    let confidence = inf.confidence;
                    let verified = inf.verified;
                    let sources = inf.source_claims.len();
                    view! {
                        <div class="inference-card">
                            <div class="inf-header">
                                <span class="inf-type">{itype}</span>
                                {verified.then(|| view! { <span class="badge badge-success">"Verified"</span> })}
                                {(!verified).then(|| view! { <span class="badge">"Unverified"</span> })}
                            </div>
                            <p class="inf-conclusion">{conclusion}</p>
                            <p class="inf-reasoning">{reasoning}</p>
                            <div class="inf-meta">
                                <span>{format!("{:.0}% confidence", confidence * 100.0)}</span>
                                <span>{format!("{sources} source claims")}</span>
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
