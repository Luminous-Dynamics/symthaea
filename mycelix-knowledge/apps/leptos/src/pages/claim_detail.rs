// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::hooks::use_params_map;
use crate::context::use_knowledge_context;

#[component]
pub fn ClaimDetailPage() -> impl IntoView {
    let ctx = use_knowledge_context();
    let params = use_params_map();

    let claim = move || {
        let id = params.get().get("id").unwrap_or_default();
        ctx.claims.get().iter().find(|c| c.id == id).cloned()
    };

    view! {
        <div class="page-claim-detail">
            {move || match claim() {
                Some(c) => {
                    let e = c.classification.empirical.label();
                    let n = c.classification.normative.label();
                    let m = c.classification.materiality.label();
                    let conf = c.classification.confidence;
                    view! {
                        <div class="claim-detail">
                            <div class="claim-detail-header">
                                <span class="claim-id">{c.id.clone()}</span>
                                <span class="claim-type-badge">{c.claim_type.label()}</span>
                            </div>
                            <h1 class="claim-content-full">{c.content.clone()}</h1>

                            <div class="epistemic-detail">
                                <h2>"Epistemic Classification"</h2>
                                <div class="epistemic-grid">
                                    <div class="epistemic-dim">
                                        <span class="dim-label">"Empirical"</span>
                                        <span class="dim-value e-tag">{e}</span>
                                    </div>
                                    <div class="epistemic-dim">
                                        <span class="dim-label">"Normative"</span>
                                        <span class="dim-value n-tag">{n}</span>
                                    </div>
                                    <div class="epistemic-dim">
                                        <span class="dim-label">"Materiality"</span>
                                        <span class="dim-value m-tag">{m}</span>
                                    </div>
                                    <div class="epistemic-dim">
                                        <span class="dim-label">"Confidence"</span>
                                        <span class="dim-value">{format!("{:.0}%", conf * 100.0)}</span>
                                    </div>
                                </div>
                            </div>

                            <div class="claim-meta-section">
                                <h2>"Sources"</h2>
                                {if c.sources.is_empty() {
                                    view! { <p class="no-data">"No sources provided."</p> }.into_any()
                                } else {
                                    c.sources.iter().map(|s| {
                                        let src = s.clone();
                                        view! { <div class="source-item">{src}</div> }
                                    }).collect_view().into_any()
                                }}
                            </div>

                            <div class="claim-meta-section">
                                <h2>"Tags"</h2>
                                <div class="claim-tags">
                                    {c.tags.iter().map(|t| {
                                        let tag = t.clone();
                                        view! { <span class="tag">{tag}</span> }
                                    }).collect_view()}
                                </div>
                            </div>

                            <div class="claim-meta-section">
                                <span class="claim-author">{format!("Author: {}", c.author)}</span>
                                <span class="claim-version">{format!("Version {}", c.version)}</span>
                            </div>
                        </div>
                    }.into_any()
                }
                None => view! {
                    <div class="empty-state">
                        <div class="empty-state-icon">"?"</div>
                        <h3 class="empty-state-title">"Claim not found"</h3>
                    </div>
                }.into_any()
            }}
        </div>
    }
}
