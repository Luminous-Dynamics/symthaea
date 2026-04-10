// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Routes to the correct page view based on BrowserState.

use leptos::prelude::*;
use prism_common::SearchResult;
use prism_search::SearchEngine;
use crate::components::compare_view::CompareView;
use crate::components::result_card::ResultCard;
use crate::components::sentient_overlay;
use crate::pages::settings::SettingsPage;
use crate::pages::submit_claim::SubmitClaimPage;
use crate::state::{BrowserState, PageView};

#[component]
pub fn ContentRouter() -> impl IntoView {
    let state = expect_context::<BrowserState>();
    let search_engine = expect_context::<StoredValue<SearchEngine>>();

    move || {
        if state.loading.get() {
            return view! { <div class="loading">"Loading..."</div> }.into_any();
        }

        match state.view.get() {
            PageView::Welcome => view! { <WelcomePage /> }.into_any(),
            PageView::Search { query, results } => {
                view! { <SearchResultsPage query=query.clone() results=results.clone() /> }.into_any()
            }
            PageView::Page { html } => {
                // Apply the Sentient Overlay — annotate page content with epistemic claims
                let annotated = search_engine.with_value(|e| {
                    sentient_overlay::annotate_html(&html, e)
                });
                view! {
                    <div class="reader-content" inner_html=annotated></div>
                }.into_any()
            }
            PageView::Compare { query } => {
                view! { <CompareView query=query.clone() /> }.into_any()
            }
            PageView::Settings => {
                view! { <SettingsPage /> }.into_any()
            }
            PageView::SubmitClaim => {
                view! { <SubmitClaimPage /> }.into_any()
            }
            PageView::Error { message } => {
                view! {
                    <div class="reader-content">
                        <h1>"Navigation Error"</h1>
                        <p>{message.clone()}</p>
                    </div>
                }.into_any()
            }
            PageView::Loading => {
                view! { <div class="loading">"Loading..."</div> }.into_any()
            }
        }
    }
}

#[component]
fn WelcomePage() -> impl IntoView {
    view! {
        <div class="reader-content welcome">
            <h1>"Mycelix Prism"</h1>
            <p class="tagline">"Consciousness-aware epistemic search. Pure Rust."</p>

            <div class="try-section">
                <h2>"Try Searching"</h2>
                <p>"Type a query in the search bar above:"</p>
                <ul>
                    <li><strong>"ocean acidification"</strong>" — climate science from NOAA"</li>
                    <li><strong>"consciousness"</strong>" — IIT, GWT, hard problem"</li>
                    <li><strong>"rust programming"</strong>" — memory safety, ownership"</li>
                    <li><strong>"quantum physics"</strong>" — entanglement, uncertainty"</li>
                    <li><strong>"DNA genetics"</strong>" — double helix, CRISPR"</li>
                </ul>

                <h2>"How It Works"</h2>
                <p>"Every query is encoded into a "<strong>"16,384-bit hyperdimensional vector"</strong>
                   " using word-level random indexing with IDF weighting."</p>

                <h2>"Security"</h2>
                <p>"Every page passes through a two-stage immune system before rendering."</p>

                <hr />
                <p style="color: #888; font-size: 14px; text-align: center; margin-top: 24px;">
                    "The browser is the distribution mechanism. The knowledge engine is the value."
                </p>
                <p style="text-align: center; margin-top: 8px;">
                    <a href="javascript:void(0)" style="color: var(--prism-accent); font-size: 14px;"
                       on:click=move |_| {
                           use leptos::prelude::Set;
                           let state = expect_context::<BrowserState>();
                           state.set_view.set(PageView::SubmitClaim);
                           state.set_current_url.set("prism://submit".to_string());
                           state.set_page_title.set("Submit Claim".to_string());
                       }
                    >"Submit your own claim →"</a>
                </p>
            </div>
        </div>
    }
}

#[component]
fn SearchResultsPage(query: String, results: Vec<SearchResult>) -> impl IntoView {
    let count = results.len();
    let search_engine = expect_context::<StoredValue<SearchEngine>>();
    let total_claims = search_engine.with_value(|e| e.claim_count());

    let max_sim = results.iter().map(|r| r.query_similarity).fold(0.0_f32, f32::max);
    let weak_results = max_sim < 0.08;

    view! {
        <div class="search-results">
            <div class="search-header">
                {count}" results for "<strong>{query.clone()}</strong>
                <span style="float:right; font-size:12px; color:var(--content-text-secondary);">
                    {total_claims}" claims indexed"
                </span>
            </div>

            {if weak_results {
                view! {
                    <div class="no-strong-matches">
                        <p>"No strong matches found for "<strong>{query.clone()}</strong>"."</p>
                        <p>"The knowledge base doesn't contain claims closely related to this query. "
                           "Try more specific terms like:"</p>
                        <ul>
                            <li><strong>"ocean acidification"</strong></li>
                            <li><strong>"consciousness"</strong></li>
                            <li><strong>"rust programming"</strong></li>
                            <li><strong>"quantum physics"</strong></li>
                        </ul>
                        <p style="margin-top: 12px; font-size: 13px; color: var(--content-text-secondary);">
                            "Showing best approximate matches below:"
                        </p>
                    </div>
                }.into_any()
            } else {
                view! { <div></div> }.into_any()
            }}

            {results.into_iter().enumerate().map(|(i, r)| {
                view! { <ResultCard result=r rank={i+1} /> }
            }).collect_view()}
        </div>
    }
}
