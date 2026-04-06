// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_knowledge_context;

#[component]
pub fn FactCheckPage() -> impl IntoView {
    let ctx = use_knowledge_context();

    view! {
        <div class="page-fact-check">
            <h1>"Fact Check"</h1>
            <p class="subtitle">"Verify statements against the knowledge graph."</p>

            <div class="fact-check-list">
                {move || ctx.fact_checks.get().iter().map(|fc| {
                    let statement = fc.statement.clone();
                    let verdict = fc.verdict.label();
                    let verdict_css = fc.verdict.css_class();
                    let confidence = fc.verdict_confidence;
                    let credibility = fc.credibility_score;
                    let evidence = fc.evidence_count;
                    view! {
                        <div class="fact-check-card">
                            <div class="fc-header">
                                <span class=format!("verdict-badge badge-{verdict_css}")>{verdict}</span>
                                <span class="fc-confidence">{format!("{:.0}% confidence", confidence * 100.0)}</span>
                            </div>
                            <p class="fc-statement">{statement}</p>
                            <div class="fc-meta">
                                <span>{format!("Credibility: {:.0}%", credibility * 100.0)}</span>
                                <span>{format!("{evidence} evidence items")}</span>
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
