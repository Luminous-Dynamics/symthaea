// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge between the Leptos UI and Prism engine crates.

use leptos::prelude::Set;
use prism_common::{ContentZone, SafetyLevel};
use prism_privacy::ConsentStore;
use prism_reflex::ReflexArc;
use prism_search::SearchEngine;

use crate::state::{BrowserState, PageView};

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
        state.set_current_url.set("prism://settings".to_string());
        state.set_page_title.set("Settings".to_string());
        state.set_view.set(PageView::Settings);
        state.set_zone.set(ContentZone::Local);
        state.set_safety.set(SafetyLevel::Green);
        state.set_threat_count.set(0);
        return;
    }

    if input == "prism://submit" {
        state.set_current_url.set("prism://submit".to_string());
        state.set_page_title.set("Submit Claim".to_string());
        state.set_view.set(PageView::SubmitClaim);
        state.set_zone.set(ContentZone::Local);
        state.set_safety.set(SafetyLevel::Green);
        state.set_threat_count.set(0);
        return;
    }

    if input == "prism://welcome" || input.starts_with("prism://") && !input.contains("search") {
        state.set_current_url.set(input.to_string());
        state.set_page_title.set("Symthaea Prism".to_string());
        state.set_view.set(PageView::Welcome);
        state.set_zone.set(ContentZone::Local);
        state.set_safety.set(SafetyLevel::Green);
        state.set_threat_count.set(0);
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
    let results = engine.search(query, 10);
    state.set_current_url.set(format!("prism://search?q={}", query));
    state.set_page_title.set(format!("Search: {}", query));
    state.set_view.set(PageView::Search {
        query: query.to_string(),
        results,
    });
    state.set_zone.set(ContentZone::Local);
    state.set_safety.set(SafetyLevel::Green);
    state.set_threat_count.set(0);
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

                        state_clone.set_page_title.set(title);
                        state_clone.set_zone.set(zone);
                        state_clone.set_safety.set(post.safety_level);
                        state_clone.set_threat_count.set(post.threats.len());
                        state_clone.set_view.set(PageView::Page { html: clean });
                        state_clone.set_loading.set(false);
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
