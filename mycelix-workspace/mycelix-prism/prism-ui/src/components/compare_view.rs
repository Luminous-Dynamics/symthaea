// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Search engine comparison view — Prism vs DuckDuckGo vs Wikipedia vs Ollama vs Brave vs Perplexity.
//!
//! Shows the same query across 6 engines side-by-side.
//! Only Prism has epistemic classification. The contrast sells itself.

use leptos::prelude::*;
use prism_common::SearchResult;
use prism_search::SearchEngine;

use crate::components::engine_card::EngineCard;
use crate::components::result_card::ResultCard;
use crate::external_search::{self, ExternalHit, ExternalResult, SearchStatus};

#[component]
pub fn CompareView(query: String) -> impl IntoView {
    let search_engine = expect_context::<StoredValue<Option<SearchEngine>>>();

    // Prism results (instant, local)
    let prism_results: Vec<SearchResult> =
        search_engine.with_value(|e| e.as_ref().map(|s| s.search(&query, 5)).unwrap_or_default());

    // External search signals
    let (wiki, set_wiki) = signal(ExternalResult::loading("Wikipedia"));
    let (ddg, set_ddg) = signal(ExternalResult::loading("DuckDuckGo"));
    let (ollama, set_ollama) = signal(ExternalResult::loading("Ollama"));
    let (brave, set_brave) = signal(ExternalResult::loading("Brave"));
    let (perplexity, set_perplexity) = signal(ExternalResult::loading("Perplexity"));

    // Fire all external searches in parallel
    let q1 = query.clone();
    let q2 = query.clone();
    let q3 = query.clone();
    let q4 = query.clone();
    let q5 = query.clone();

    wasm_bindgen_futures::spawn_local(async move {
        set_wiki.set(external_search::search_wikipedia(&q1).await);
    });
    wasm_bindgen_futures::spawn_local(async move {
        set_ddg.set(external_search::search_duckduckgo(&q2).await);
    });
    wasm_bindgen_futures::spawn_local(async move {
        set_ollama.set(external_search::query_ollama(&q3).await);
    });
    wasm_bindgen_futures::spawn_local(async move {
        set_brave.set(external_search::search_brave(&q4).await);
    });
    wasm_bindgen_futures::spawn_local(async move {
        set_perplexity.set(external_search::query_perplexity(&q5).await);
    });

    let prism_count = prism_results.len();

    view! {
        <div class="compare-container">
            <div class="compare-header">
                "Comparing results for "<strong>{query.clone()}</strong>
                " across 6 engines"
            </div>

            <div class="compare-grid">
                // ── Prism Column (with E-levels) ──
                <div class="engine-card prism">
                    <div class="engine-header">
                        <span class="engine-label prism-label">"Prism"</span>
                        <span class="engine-badge">"Epistemic"</span>
                    </div>
                    <div class="engine-results">
                        {prism_results.into_iter().enumerate().map(|(i, r)| {
                            view! { <ResultCard result=r rank={i+1} /> }
                        }).collect_view()}
                    </div>
                    <div class="engine-footer">
                        {prism_count}" classified results with E-levels"
                    </div>
                </div>

                // ── Wikipedia Column ──
                {move || {
                    let w = wiki.get();
                    view! { <EngineCard result=w is_prism=false /> }
                }}

                // ── DuckDuckGo Column ──
                {move || {
                    let d = ddg.get();
                    view! { <EngineCard result=d is_prism=false /> }
                }}

                // ── Ollama Column ──
                {move || {
                    let o = ollama.get();
                    view! { <EngineCard result=o is_prism=false /> }
                }}

                // ── Brave Column ──
                {move || {
                    let b = brave.get();
                    view! { <EngineCard result=b is_prism=false /> }
                }}

                // ── Perplexity Column ──
                {move || {
                    let p = perplexity.get();
                    view! { <EngineCard result=p is_prism=false /> }
                }}
            </div>

            <div class="compare-footer">
                <p>"Only Prism classifies every result with evidence levels (E0-E4), "
                   "source attribution, and trust scores. Other engines return unclassified data."</p>
            </div>
        </div>
    }
}
