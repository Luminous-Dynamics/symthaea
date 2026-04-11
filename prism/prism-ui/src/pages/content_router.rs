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
    let search_engine = expect_context::<StoredValue<Option<SearchEngine>>>();

    move || {
        let current_view = state.view.get();

        // Show full-page loading only for URL navigation, not search augmentation.
        // Search results show an inline loading banner via SearchResultsPage.
        if state.loading.get() && !matches!(current_view, PageView::Search { .. }) {
            return view! { <div class="loading">"Loading\u{2026}"</div> }.into_any();
        }

        match current_view {
            PageView::Welcome => view! { <WelcomePage /> }.into_any(),
            PageView::Search { query, results } => {
                view! { <SearchResultsPage query=query.clone() results=results.clone() /> }.into_any()
            }
            PageView::Page { html } => {
                // Apply the Sentient Overlay if enabled in settings
                let overlay_enabled = crate::persistence::load::<bool>("overlay").unwrap_or(true);
                let annotated = if overlay_enabled {
                    search_engine.with_value(|e| {
                        match e {
                            Some(eng) => sentient_overlay::annotate_html(&html, eng),
                            None => html.clone(),
                        }
                    })
                } else {
                    html.clone()
                };
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
            PageView::Bookmarks => {
                view! { <BookmarksPage /> }.into_any()
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
            <img src="/static/prism-hero.jpg" alt="Mycelix Prism" class="welcome-hero" width="800" height="436" />
            <h1>"Mycelix Prism"</h1>
            <p class="tagline">"Consciousness-aware epistemic search. Pure Rust."</p>

            <div class="try-section">
                <h2>"Try Searching"</h2>
                <p>"Tap a query to search:"</p>
                <ul>
                    {[
                        ("ocean acidification", "climate science from NOAA"),
                        ("consciousness", "IIT, GWT, hard problem"),
                        ("rust programming", "memory safety, ownership"),
                        ("quantum physics", "entanglement, uncertainty"),
                        ("DNA genetics", "double helix, CRISPR"),
                    ].into_iter().map(|(query, desc)| {
                        let q = query.to_string();
                        view! {
                            <li>
                                <button class="example-query"
                                   type="button"
                                   on:click=move |_| {
                                       crate::engine::trigger_search(&q);
                                   }
                                >
                                    {query}
                                </button>
                                " — "{desc}
                            </li>
                        }
                    }).collect::<Vec<_>>()}
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
                    <button class="link-button"
                       on:click=move |_| {
                           use leptos::prelude::Set;
                           let state = expect_context::<BrowserState>();
                           let view = PageView::SubmitClaim;
                           state.set_view.set(view.clone());
                           state.set_current_url.set("prism://submit".to_string());
                           state.set_page_title.set("Submit Claim".to_string());
                           state.push_history("prism://submit", "Submit Claim", &view);
                       }
                    >"Submit your own claim \u{2192}"</button>
                </p>
            </div>
        </div>
    }
}

#[component]
fn SearchResultsPage(query: String, results: Vec<SearchResult>) -> impl IntoView {
    let state = expect_context::<BrowserState>();
    let count = results.len();
    let search_engine = expect_context::<StoredValue<Option<SearchEngine>>>();
    let total_claims = search_engine.with_value(|e| e.as_ref().map(|s| s.claim_count()).unwrap_or(0));

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

            // Loading indicator for web-augmented search
            <Show when=move || state.loading.get()>
                <div class="search-loading-banner">
                    <span class="spinner"></span>
                    " Searching external sources\u{2026}"
                </div>
            </Show>

            // Spore consciousness-generated epistemic summary
            <Show when=move || !state.spore_summary.get().is_empty()>
                <div class="spore-summary">
                    <div class="spore-summary-header">
                        <span class="spore-psi">{move || format!("\u{03A8} {:.0}%", state.consciousness.get() * 100.0)}</span>
                        " Epistemic Summary"
                    </div>
                    <p class="spore-summary-text">{move || state.spore_summary.get()}</p>
                    <p class="spore-disclaimer">{move || {
                        let conf = state.epistemic_confidence.get();
                        if conf < 0.2 {
                            "This is a simulated consciousness metric, not a claim of knowledge."
                        } else {
                            ""
                        }
                    }}</p>
                </div>
            </Show>

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

#[component]
fn BookmarksPage() -> impl IntoView {
    let state = expect_context::<BrowserState>();

    view! {
        <div class="reader-content">
            <h1>"Bookmarks"</h1>

            {move || {
                let bm = state.bookmarks.get();
                if bm.is_empty() {
                    view! {
                        <p class="bookmarks-empty">"No bookmarks yet. Click the \u{2606} star in the search bar to bookmark a page."</p>
                    }.into_any()
                } else {
                    view! {
                        <div class="bookmarks-list">
                            {bm.into_iter().map(|b| {
                                let url = b.url.clone();
                                let url_display = b.url.clone();
                                let url_delete = b.url.clone();
                                let title = if b.title.is_empty() { b.url.clone() } else { b.title.clone() };
                                let s_nav = expect_context::<BrowserState>();
                                let s_del = expect_context::<BrowserState>();
                                view! {
                                    <div class="bookmark-item">
                                        <div class="bookmark-info"
                                            on:click=move |_| {
                                                crate::engine::navigate_history(&url);
                                            }
                                        >
                                            <span class="bookmark-title">{title}</span>
                                            <span class="bookmark-url">{url_display}</span>
                                        </div>
                                        <button class="bookmark-delete" title="Remove bookmark"
                                            on:click=move |_| {
                                                s_del.remove_bookmark(&url_delete);
                                            }
                                        >"\u{00D7}"</button>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
