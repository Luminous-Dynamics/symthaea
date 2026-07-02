// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_knowledge_context;
use crate::actions;

#[component]
pub fn FactCheckPage() -> impl IntoView {
    let ctx = use_knowledge_context();

    let (statement, set_statement) = signal(String::new());
    let can_check = Memo::new(move |_| !statement.get().trim().is_empty());

    let on_check = move |ev: leptos::ev::SubmitEvent| {
        ev.prevent_default();
        if can_check.get_untracked() {
            actions::run_fact_check(statement.get_untracked());
            set_statement.set(String::new());
        }
    };

    view! {
        <div class="page-fact-check">
            <h1>"Fact Check"</h1>
            <p class="subtitle">"Verify statements against the knowledge graph."</p>

            <div class="check-statement-section">
                <h2>"Check a statement"</h2>
                <form class="create-form inline-form" on:submit=on_check>
                    <div class="form-field">
                        <input class="form-input" type="text"
                            placeholder="Enter a statement to fact-check..."
                            prop:value=move || statement.get()
                            on:input=move |ev| set_statement.set(event_target_value(&ev))
                        />
                    </div>
                    <button type="submit" class="btn btn-primary" disabled=move || !can_check.get()>
                        "Check"
                    </button>
                </form>
            </div>

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
