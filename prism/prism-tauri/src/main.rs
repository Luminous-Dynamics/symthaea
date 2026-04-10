// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prism — Tauri v2 desktop app.
//!
//! The Local Hub: direct URL fetching (no CORS), bundled search engine,
//! native filesystem access for persistent storage.
//!
//! Tauri commands are called from the Leptos WASM frontend via IPC.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use prism_common::{ContentZone, SafetyLevel};
use prism_dom::parse_html;
use prism_privacy::ConsentStore;
use prism_reflex::ReflexArc;
use prism_search::SearchEngine;
use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Mutex;

/// Shared application state managed by Tauri.
struct PrismState {
    search: SearchEngine,
    reflex: ReflexArc,
    client: reqwest::Client,
}

/// Result of fetching and analyzing a URL.
#[derive(Serialize)]
struct FetchResult {
    html: String,
    title: String,
    zone: String,
    safety: String,
    threat_count: usize,
}

/// Search result for the frontend.
#[derive(Serialize)]
struct SearchResultJson {
    content: String,
    sources: Vec<String>,
    empirical_level: String,
    query_similarity: f32,
    author_reputation: f32,
    tags: Vec<String>,
    rank_score: f32,
}

/// Fetch a URL directly (no CORS restrictions in native Tauri).
#[tauri::command]
async fn fetch_url(
    url: String,
    state: State<'_, Mutex<PrismState>>,
) -> Result<FetchResult, String> {
    // Clone the client before dropping the lock (reqwest::Client is cheaply cloneable)
    let client = {
        let s = state.lock().map_err(|e| format!("Lock error: {}", e))?;
        s.client.clone()
    };

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Fetch failed: {}", e))?;

    let html = response
        .text()
        .await
        .map_err(|e| format!("Read failed: {}", e))?;

    // Sanitize
    let clean = ammonia::clean(&html);

    // Security analysis (re-acquire lock for reflex arc)
    let parsed_url = url::Url::parse(&url).map_err(|e| format!("Bad URL: {}", e))?;
    let dom = parse_html(&html);
    let title = dom.title().unwrap_or_else(|| "Untitled".to_string());

    let s = state.lock().map_err(|e| format!("Lock error: {}", e))?;
    let pre = s.reflex.pre_fetch(&parsed_url, false, false);
    let post = s.reflex.post_parse(&dom, &pre);
    let consent = ConsentStore::new();
    let zone = consent.resolve_zone(post.zone, parsed_url.host_str().unwrap_or(""));

    Ok(FetchResult {
        html: clean,
        title,
        zone: format!("{:?}", zone),
        safety: format!("{:?}", post.safety_level),
        threat_count: post.threats.len(),
    })
}

/// Perform epistemic search.
#[tauri::command]
fn search_claims(
    query: String,
    top_k: usize,
    state: State<'_, Mutex<PrismState>>,
) -> Result<Vec<SearchResultJson>, String> {
    let state = state.lock().map_err(|e| format!("Lock error: {}", e))?;
    let results = state.search.search(&query, top_k);

    Ok(results.into_iter().map(|r| {
        let score = r.rank_score();
        SearchResultJson {
            content: r.content,
            sources: r.sources,
            empirical_level: format!("{:?}", r.empirical_level),
            query_similarity: r.query_similarity,
            author_reputation: r.author_reputation,
            tags: r.tags,
            rank_score: score,
        }
    }).collect())
}

/// Get claim count.
#[tauri::command]
fn claim_count(state: State<'_, Mutex<PrismState>>) -> Result<usize, String> {
    let state = state.lock().map_err(|e| format!("Lock error: {}", e))?;
    Ok(state.search.claim_count())
}

fn main() {
    env_logger::init();

    let prism_state = PrismState {
        search: SearchEngine::with_core_claims(),
        reflex: ReflexArc::new(),
        client: reqwest::Client::builder()
            .user_agent("Prism/0.2 (Tauri; +https://mycelix.net)")
            .redirect(reqwest::redirect::Policy::limited(10))
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("HTTP client"),
    };

    log::info!("Prism desktop starting ({} claims indexed)", prism_state.search.claim_count());

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(Mutex::new(prism_state))
        .invoke_handler(tauri::generate_handler![
            fetch_url,
            search_claims,
            claim_count,
        ])
        .run(tauri::generate_context!())
        .expect("error running Prism");
}
