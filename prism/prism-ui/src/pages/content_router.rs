// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Routes to the correct page view based on BrowserState.

use crate::components::compare_view::CompareView;
use crate::components::result_card::ResultCard;
use crate::components::sentient_overlay;
use crate::pages::settings::SettingsPage;
use crate::pages::submit_claim::SubmitClaimPage;
use crate::state::{BrowserState, PageView};
use leptos::prelude::*;
use prism_common::SearchResult;
use prism_search::SearchEngine;

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
                view! { <SearchResultsPage query=query.clone() results=results.clone() /> }
                    .into_any()
            }
            PageView::Page { html } => {
                // Apply the Sentient Overlay if enabled in settings
                let overlay_enabled = crate::persistence::load::<bool>("overlay").unwrap_or(true);
                let annotated = if overlay_enabled {
                    search_engine.with_value(|e| match e {
                        Some(eng) => sentient_overlay::annotate_html(&html, eng),
                        None => crate::engine::sanitize_html(&html, None),
                    })
                } else {
                    crate::engine::sanitize_html(&html, None)
                };
                view! {
                    <div class="reader-content" inner_html=annotated></div>
                }
                .into_any()
            }
            PageView::Compare { query } => view! { <CompareView query=query.clone() /> }.into_any(),
            PageView::FullPageIframe { url } => {
                let url_display = url.clone();
                view! {
                    <div class="fullpage-container">
                        <iframe
                            class="fullpage-iframe"
                            src=url.clone()
                            sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
                        ></iframe>
                        <EpistemicDrawer url=url_display />
                    </div>
                }
                .into_any()
            }
            PageView::Settings => view! { <SettingsPage /> }.into_any(),
            PageView::SubmitClaim => view! { <SubmitClaimPage /> }.into_any(),
            PageView::Bookmarks => view! { <BookmarksPage /> }.into_any(),
            PageView::Error { message } => view! {
                <div class="reader-content">
                    <h1>"Navigation Error"</h1>
                    <p>{message.clone()}</p>
                </div>
            }
            .into_any(),
            PageView::Loading => view! { <div class="loading">"Loading..."</div> }.into_any(),
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
    let total_claims =
        search_engine.with_value(|e| e.as_ref().map(|s| s.claim_count()).unwrap_or(0));

    let max_sim = results
        .iter()
        .map(|r| r.query_similarity)
        .fold(0.0_f32, f32::max);
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
                        " Consciousness Reflection"
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
                let q = query.clone();
                view! {
                    <div class="no-knowledge-banner">
                        <div class="no-knowledge-icon">"\u{1F50D}"</div>
                        <div class="no-knowledge-text">
                            <p class="no-knowledge-title">
                                "Prism has no verified knowledge about "<strong>{q.clone()}</strong>"."
                            </p>
                            <p class="no-knowledge-detail">
                                "The epistemic knowledge base ("{total_claims}" claims) doesn't contain "
                                "strong matches for this query. Results below are approximate matches "
                                "and external sources \u{2014} treat them as unverified."
                            </p>
                        </div>
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

/// Floating epistemic panel that appears alongside the Full Page iframe.
/// Shows consciousness state, epistemic confidence, and knowledge base matches
/// for the current URL — without replacing the page content.
#[component]
fn EpistemicDrawer(url: String) -> impl IntoView {
    let state = expect_context::<BrowserState>();
    let (drawer_open, set_drawer_open) = signal(false);

    // Search knowledge base for claims related to the URL's domain/topic
    let search_engine = expect_context::<StoredValue<Option<SearchEngine>>>();
    let url_query = url.clone();
    let related_claims = search_engine.with_value(|opt| {
        if let Some(engine) = opt {
            // Extract host + path keywords for search
            let query = url::Url::parse(&url_query)
                .ok()
                .map(|u| {
                    let host = u
                        .host_str()
                        .unwrap_or("")
                        .replace("www.", "")
                        .replace(".com", "")
                        .replace(".org", "");
                    let path = u
                        .path()
                        .replace('/', " ")
                        .replace('-', " ")
                        .replace('_', " ");
                    format!("{} {}", host, path)
                })
                .unwrap_or_default();
            if query.len() > 3 {
                engine.search(&query, 5)
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    });

    let has_claims = !related_claims.is_empty();
    let claim_count = related_claims.len();

    view! {
        // Toggle button - always visible
        <button
            class=move || if drawer_open.get() { "drawer-toggle open" } else { "drawer-toggle" }
            on:click=move |_| set_drawer_open.set(!drawer_open.get())
            title="Epistemic Analysis"
            attr:aria-label="Toggle epistemic panel"
        >
            {move || if drawer_open.get() { "\u{00BB}" } else {
                if has_claims { "\u{03A8}" } else { "\u{03A8}" }
            }}
        </button>

        // Drawer panel
        <Show when=move || drawer_open.get()>
            <div class="epistemic-drawer">
                <div class="drawer-header">
                    <span class="drawer-title">"Epistemic Analysis"</span>
                    <button class="drawer-close" on:click=move |_| set_drawer_open.set(false)>"\u{00D7}"</button>
                </div>

                // Consciousness state
                <div class="drawer-section">
                    <div class="drawer-label">"Consciousness"</div>
                    <div class="drawer-value">
                        <span class="spore-psi">{move || format!("\u{03A8} {:.0}%", state.consciousness.get() * 100.0)}</span>
                        <span class="drawer-conf">{move || format!("Confidence: {:.0}%", state.epistemic_confidence.get() * 100.0)}</span>
                    </div>
                </div>

                // Related knowledge
                <div class="drawer-section">
                    <div class="drawer-label">{claim_count}" related claims"</div>
                    {if has_claims {
                        view! {
                            <div class="drawer-claims">
                                {related_claims.into_iter().map(|r| {
                                    let e_class = match r.empirical_level {
                                        prism_common::EmpiricalLevel::E4 => "e-badge e4",
                                        prism_common::EmpiricalLevel::E3 => "e-badge e3",
                                        prism_common::EmpiricalLevel::E2 => "e-badge e2",
                                        prism_common::EmpiricalLevel::E1 => "e-badge e1",
                                        prism_common::EmpiricalLevel::E0 => "e-badge e0",
                                    };
                                    let e_label = match r.empirical_level {
                                        prism_common::EmpiricalLevel::E4 => "E4",
                                        prism_common::EmpiricalLevel::E3 => "E3",
                                        prism_common::EmpiricalLevel::E2 => "E2",
                                        prism_common::EmpiricalLevel::E1 => "E1",
                                        prism_common::EmpiricalLevel::E0 => "E0",
                                    };
                                    let sim = (r.query_similarity * 100.0) as u32;
                                    let snippet: String = r.content.chars().take(120).collect();
                                    view! {
                                        <div class="drawer-claim">
                                            <span class=e_class>{e_label}</span>
                                            <span class="drawer-sim">{sim}"%"</span>
                                            <p class="drawer-claim-text">{snippet}</p>
                                        </div>
                                    }
                                }).collect_view()}
                            </div>
                        }.into_any()
                    } else {
                        view! {
                            <p class="drawer-empty">"No claims in the knowledge base match this page."</p>
                        }.into_any()
                    }}
                </div>

                <p class="drawer-disclaimer">
                    "Epistemic analysis based on curated knowledge base. "
                    "Full Page mode allows scripts and cookies."
                </p>
            </div>
        </Show>
    }
}
