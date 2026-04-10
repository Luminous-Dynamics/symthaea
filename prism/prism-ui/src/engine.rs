// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge between the Leptos UI and Prism engine crates.

use leptos::prelude::*;
use prism_common::{ContentZone, SafetyLevel};
use prism_privacy::ConsentStore;
use prism_reflex::ReflexArc;
use prism_search::SearchEngine;

use crate::state::{BrowserState, PageView, SearchMode};
use prism_common::{EmpiricalLevel, NormativeLevel, MaterialityLevel, SearchResult};

/// Trigger a search from outside the search bar (e.g., clicking an example query).
pub fn trigger_search(query: &str) {
    log::info!("trigger_search called with: {}", query);
    let state = expect_context::<BrowserState>();
    let engine_cell = expect_context::<StoredValue<Option<SearchEngine>>>();
    engine_cell.with_value(|opt| {
        log::info!("engine_cell has value: {}", opt.is_some());
        if let Some(se) = opt {
            search_query(query, &state, se);
        }
    });
}

/// Sanitize HTML for safe innerHTML rendering (reader mode).
///
/// More permissive than ammonia::clean() — allows structural and formatting
/// tags needed for readable content, while stripping scripts and event handlers.
fn sanitize_html(html: &str) -> String {
    let mut builder = ammonia::Builder::new();
    builder
        .add_tags(&[
            "h1", "h2", "h3", "h4", "h5", "h6",
            "p", "br", "hr", "div", "span", "section", "article", "main",
            "ul", "ol", "li", "dl", "dt", "dd",
            "strong", "em", "b", "i", "u", "s", "mark", "small", "sub", "sup",
            "a", "blockquote", "pre", "code", "kbd", "var", "samp",
            "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption",
            "img", "figure", "figcaption", "picture", "source",
            "details", "summary", "time", "abbr", "cite",
            "nav", "header", "footer", "aside",
        ])
        .add_tag_attributes("a", &["href", "title"])
        .add_tag_attributes("img", &["src", "alt", "width", "height"])
        .add_tag_attributes("td", &["colspan", "rowspan"])
        .add_tag_attributes("th", &["colspan", "rowspan", "scope"])
        .add_tag_attributes("time", &["datetime"])
        .add_tag_attributes("abbr", &["title"])
        .link_rel(Some("noopener noreferrer"))
        .url_relative(ammonia::UrlRelative::PassThrough);
    builder.clean(html).to_string()
}

/// CORS proxy URL. Routes external fetches through the local proxy
/// to bypass browser same-origin restrictions.
/// Falls back to direct fetch if proxy is not running.
fn proxy_url(url: &str) -> String {
    // Try proxy at :8131 (same host as the app)
    // The WASM app detects its own origin and constructs the proxy URL
    if let Some(window) = web_sys::window() {
        if let Ok(origin) = window.location().origin() {
            // Replace the port with 8131
            if let Some(base) = origin.rsplit_once(':') {
                return format!("{}:8131/proxy?url={}", base.0, url);
            }
        }
    }
    // Fallback: try localhost proxy
    format!("http://127.0.0.1:8131/proxy?url={}", url)
}

pub fn is_url(input: &str) -> bool {
    let input = input.trim();
    // Explicit schemes
    if input.starts_with("http://")
        || input.starts_with("https://")
        || input.starts_with("prism://")
    {
        return true;
    }
    // Bare domain detection: contains a dot, no spaces, looks like a domain
    // e.g. "google.com", "en.wikipedia.org/wiki/Rust"
    if !input.contains(' ') && input.contains('.') {
        let parts: Vec<&str> = input.split('.').collect();
        if parts.len() >= 2 {
            let tld = parts.last().unwrap_or(&"").split('/').next().unwrap_or("");
            // Common TLDs or at least 2 chars after the dot
            if tld.len() >= 2 && tld.chars().all(|c| c.is_ascii_alphabetic()) {
                return true;
            }
        }
    }
    false
}

/// Normalize a bare domain into a full URL.
fn normalize_url(input: &str) -> String {
    let input = input.trim();
    if input.starts_with("http://") || input.starts_with("https://") || input.starts_with("prism://") {
        input.to_string()
    } else {
        format!("https://{}", input)
    }
}

/// Process user input: navigate to URL or perform epistemic search.
pub fn process_input(
    input: &str,
    state: &BrowserState,
    search_engine: &SearchEngine,
    reflex: &ReflexArc,
) {
    let input = input.trim();
    if input.is_empty() {
        return;
    }

    if input == "prism://settings" {
        let view = PageView::Settings;
        state.set_current_url.set("prism://settings".to_string());
        state.set_page_title.set("Settings".to_string());
        state.set_view.set(view.clone());
        state.set_zone.set(ContentZone::Local);
        state.set_safety.set(SafetyLevel::Green);
        state.set_threat_count.set(0);
        state.push_history("prism://settings", "Settings", &view);
        return;
    }

    if input == "prism://submit" {
        let view = PageView::SubmitClaim;
        state.set_current_url.set("prism://submit".to_string());
        state.set_page_title.set("Submit Claim".to_string());
        state.set_view.set(view.clone());
        state.set_zone.set(ContentZone::Local);
        state.set_safety.set(SafetyLevel::Green);
        state.set_threat_count.set(0);
        state.push_history("prism://submit", "Submit Claim", &view);
        return;
    }

    if input == "prism://welcome" || input.starts_with("prism://") && !input.contains("search") {
        let view = PageView::Welcome;
        state.set_current_url.set(input.to_string());
        state.set_page_title.set("Prism".to_string());
        state.set_view.set(view.clone());
        state.set_zone.set(ContentZone::Local);
        state.set_safety.set(SafetyLevel::Green);
        state.set_threat_count.set(0);
        state.push_history(input, "Prism", &view);
        return;
    }

    if is_url(input) {
        let url = normalize_url(input);
        navigate_url(&url, state, reflex);
    } else {
        search_query(input, state, search_engine);
    }
}

fn search_query(query: &str, state: &BrowserState, engine: &SearchEngine) {
    let mode = state.search_mode.get_untracked();
    let local_results = engine.search(query, 10);

    let url = format!("prism://search?q={}", query);
    let title = format!("Search: {}", query);
    let view = PageView::Search {
        query: query.to_string(),
        results: local_results.clone(),
    };

    // Show local results immediately
    state.set_current_url.set(url.clone());
    state.set_page_title.set(title.clone());
    state.set_view.set(view.clone());
    state.set_zone.set(ContentZone::Local);
    state.set_safety.set(SafetyLevel::Green);
    state.set_threat_count.set(0);
    state.push_history(&url, &title, &view);

    // Signal loading for web-augmented modes
    if mode != SearchMode::Basic {
        state.set_loading.set(true);
    }

    // Advanced/Paradigm: augment with web sources in background
    if mode == SearchMode::Basic {
        return;
    }

    let query_owned = query.to_string();
    let set_view = state.set_view;
    let set_loading = state.set_loading;
    let local = local_results;

    wasm_bindgen_futures::spawn_local(async move {
        let mut merged = local;
        let mut failed_sources: Vec<String> = Vec::new();

        // Wikipedia (direct CORS, most reliable)
        let wiki = crate::external_search::search_wikipedia(&query_owned).await;
        if matches!(wiki.status, crate::external_search::SearchStatus::Error(_)) {
            failed_sources.push("Wikipedia".into());
        }
        for hit in wiki.hits {
            merged.push(external_hit_to_result(&hit, "Wikipedia", EmpiricalLevel::E3));
        }

        // DuckDuckGo (via proxy)
        let ddg = crate::external_search::search_duckduckgo(&query_owned).await;
        if matches!(ddg.status, crate::external_search::SearchStatus::Error(_)) {
            failed_sources.push("DuckDuckGo".into());
        }
        for hit in ddg.hits {
            merged.push(external_hit_to_result(&hit, "DuckDuckGo", EmpiricalLevel::E2));
        }

        // Paradigm mode: also try Knowledge DHT + Ollama
        if mode == SearchMode::Paradigm {
            let dht = prism_knowledge_bridge::unified_search(vec![], &query_owned).await;
            merged.extend(dht);

            let ollama = crate::external_search::query_ollama(&query_owned).await;
            if matches!(ollama.status, crate::external_search::SearchStatus::Error(_)) {
                failed_sources.push("Ollama".into());
            }
            for hit in ollama.hits {
                merged.push(external_hit_to_result(&hit, "Ollama", EmpiricalLevel::E1));
            }
        }

        if !failed_sources.is_empty() {
            log::warn!("External sources failed: {}", failed_sources.join(", "));
        }

        // Deduplicate by content prefix
        let mut seen = std::collections::HashSet::new();
        merged.retain(|r| {
            let key: String = r.content.chars().take(60).collect::<String>().to_lowercase();
            seen.insert(key)
        });

        // Re-rank by composite score
        merged.sort_by(|a, b| b.rank_score().partial_cmp(&a.rank_score()).unwrap_or(std::cmp::Ordering::Equal));

        // Limit to top 20
        merged.truncate(20);

        set_view.set(PageView::Search {
            query: query_owned,
            results: merged,
        });
        set_loading.set(false);
    });
}

/// Convert an external search hit into a Prism SearchResult with epistemic annotation.
fn external_hit_to_result(
    hit: &crate::external_search::ExternalHit,
    source_name: &str,
    base_level: EmpiricalLevel,
) -> SearchResult {
    let content = if hit.snippet.is_empty() {
        hit.title.clone()
    } else {
        hit.snippet.clone()
    };
    let source = hit.url.clone().unwrap_or_else(|| source_name.to_string());

    SearchResult {
        content,
        sources: vec![source],
        empirical_level: base_level,
        normative_level: NormativeLevel::N1,
        materiality_level: MaterialityLevel::M2,
        query_similarity: 0.4, // Web results get moderate base relevance
        author_reputation: match source_name {
            "Wikipedia" => 0.85,
            "DuckDuckGo" => 0.60,
            "Ollama" => 0.50,
            _ => 0.70,
        },
        age_days: 7,
        tags: vec![source_name.to_lowercase()],
    }
}

fn navigate_url(url_str: &str, state: &BrowserState, reflex: &ReflexArc) {
    let url = match url::Url::parse(url_str) {
        Ok(u) => u,
        Err(e) => {
            state.set_view.set(PageView::Error {
                message: format!("Invalid URL: {}", e),
            });
            return;
        }
    };

    let pre = reflex.pre_fetch(&url, false, false);
    let url_string = url_str.to_string();
    let state_clone = state.clone();
    let state_history = state.clone();

    state.set_loading.set(true);
    state.set_current_url.set(url_str.to_string());

    wasm_bindgen_futures::spawn_local(async move {
        // Route through CORS proxy if available, fall back to direct fetch
        let fetch_url = proxy_url(&url_string);
        match gloo_net::http::Request::get(&fetch_url).send().await {
            Ok(resp) => {
                let status = resp.status();
                // Check for error status codes (proxy errors, server errors, etc.)
                if status >= 400 {
                    let body = resp.text().await.unwrap_or_else(|_| "Unknown error".into());
                    state_clone.set_view.set(PageView::Error {
                        message: format!(
                            "Server returned HTTP {} for this URL.\n\n{}\n\n\
                             If the CORS proxy is running but the site is unreachable, \
                             the site may be blocking proxy requests.",
                            status, body
                        ),
                    });
                    state_clone.set_loading.set(false);
                    return;
                }
                match resp.text().await {
                    Ok(html) => {
                        if html.trim().is_empty() || html.len() < 20 {
                            state_clone.set_view.set(PageView::Error {
                                message: "Received an empty response. The page may require JavaScript to render.".into(),
                            });
                            state_clone.set_loading.set(false);
                            return;
                        }
                        let clean = sanitize_html(&html);
                        let dom = prism_dom::parse_html(&html);
                        let reflex = ReflexArc::new();
                        let post = reflex.post_parse(&dom, &pre);
                        let consent = ConsentStore::new();
                        let zone = consent.resolve_zone(
                            post.zone,
                            url::Url::parse(&url_string)
                                .ok()
                                .and_then(|u| u.host_str().map(|s| s.to_string()))
                                .as_deref()
                                .unwrap_or(""),
                        );
                        let title = dom.title().unwrap_or_else(|| "Untitled".to_string());

                        let view = PageView::Page { html: clean };
                        state_clone.set_page_title.set(title.clone());
                        state_clone.set_zone.set(zone);
                        state_clone.set_safety.set(post.safety_level);
                        state_clone.set_threat_count.set(post.threats.len());
                        state_clone.set_view.set(view.clone());
                        state_clone.set_loading.set(false);
                        state_history.push_history(&url_string, &title, &view);
                    }
                    Err(e) => {
                        state_clone.set_view.set(PageView::Error {
                            message: format!("Failed to read response: {}", e),
                        });
                        state_clone.set_loading.set(false);
                    }
                }
            }
            Err(e) => {
                let err_str = format!("{}", e);
                let message = if err_str.contains("Failed to fetch") || err_str.contains("NetworkError") {
                    format!(
                        "Could not load this page. The CORS proxy may not be running.\n\n\
                         Start the proxy: cargo run -p prism-proxy\n\n\
                         Or use the Tauri desktop app for unrestricted browsing.\n\n\
                         Technical: {}", e
                    )
                } else {
                    format!("Fetch failed: {}", e)
                };
                state_clone.set_view.set(PageView::Error { message });
                state_clone.set_loading.set(false);
            }
        }
    });
}
