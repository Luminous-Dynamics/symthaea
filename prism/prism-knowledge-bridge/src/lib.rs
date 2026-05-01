// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bidirectional bridge between Prism HDC search and Knowledge DHT.
//!
//! Search chain: Local HDC → Knowledge DHT → (future: Wikidata)
//! Publish chain: Prism claim → Knowledge DHT (via holochain.rs)
//!
//! This crate provides the DHT query functions that the Prism search
//! engine can use as a fallback when local results are insufficient.

use mycelix_claim_types::*;
use prism_common::{EmpiricalLevel, MaterialityLevel, NormativeLevel, SearchResult};
use serde::{Deserialize, Serialize};

const CONDUCTOR_URL: &str = "ws://localhost:8888";
const APP_ID: &str = "mycelix-knowledge";
const MIN_LOCAL_RESULTS: usize = 3;

/// Query the Knowledge DHT for claims matching a tag or keyword.
///
/// Returns SearchResults with DHT-sourced claims, scored at 0.0 similarity
/// (caller should re-rank with their own scoring).
pub async fn query_dht_claims(tag: &str) -> Vec<SearchResult> {
    let transport = mycelix_leptos_client::BrowserWsTransport::new();
    let client = mycelix_leptos_client::HolochainClient::new(transport, APP_ID, "knowledge");

    if client.connect(CONDUCTOR_URL, None).await.is_err() {
        log::info!("[KnowledgeBridge] Conductor not available for DHT search");
        return vec![];
    }

    // Query claims by tag
    let records: Vec<serde_json::Value> = match client
        .call_zome("claims", "get_claims_by_tag", &tag.to_string())
        .await
    {
        Ok(r) => r,
        Err(e) => {
            log::info!("[KnowledgeBridge] DHT query failed: {e}");
            return vec![];
        }
    };

    // Convert Records to SearchResults
    records
        .iter()
        .filter_map(|record| {
            let entry = record.get("entry")?.get("Present")?;
            let content = entry.get("content")?.as_str()?;
            let sources: Vec<String> = entry
                .get("sources")
                .and_then(|s| serde_json::from_value(s.clone()).ok())
                .unwrap_or_default();
            let tags: Vec<String> = entry
                .get("tags")
                .and_then(|t| serde_json::from_value(t.clone()).ok())
                .unwrap_or_default();

            // Parse epistemic classification
            let classification = entry.get("classification")?;
            let e_val = classification.get("empirical")?.as_f64().unwrap_or(0.0) as f32;
            let n_val = classification.get("normative")?.as_f64().unwrap_or(0.5) as f32;
            let m_val = classification
                .get("mythic")
                .or_else(|| classification.get("materiality"))?
                .as_f64()
                .unwrap_or(0.5) as f32;

            Some(SearchResult {
                content: content.to_string(),
                sources,
                empirical_level: EmpiricalLevel::from_f32(e_val),
                normative_level: if n_val >= 0.75 {
                    NormativeLevel::N3
                } else if n_val >= 0.5 {
                    NormativeLevel::N2
                } else if n_val >= 0.25 {
                    NormativeLevel::N1
                } else {
                    NormativeLevel::N0
                },
                materiality_level: if m_val >= 0.75 {
                    MaterialityLevel::M3
                } else if m_val >= 0.5 {
                    MaterialityLevel::M2
                } else if m_val >= 0.25 {
                    MaterialityLevel::M1
                } else {
                    MaterialityLevel::M0
                },
                query_similarity: 0.5,   // DHT results get base relevance
                author_reputation: 0.85, // DHT-verified = decent trust
                age_days: 30,            // Assume moderate age for DHT claims
                tags,
            })
        })
        .collect()
}

/// Unified search: local HDC results + DHT fallback.
///
/// If local results are fewer than MIN_LOCAL_RESULTS, queries the DHT
/// for additional claims and merges them (deduped by content prefix).
pub async fn unified_search(local_results: Vec<SearchResult>, query: &str) -> Vec<SearchResult> {
    if local_results.len() >= MIN_LOCAL_RESULTS {
        return local_results;
    }

    log::info!(
        "[KnowledgeBridge] Local results ({}) below threshold ({}), querying DHT...",
        local_results.len(),
        MIN_LOCAL_RESULTS
    );

    // Use the first tag from the query as DHT search key
    let dht_results = query_dht_claims(query).await;

    if dht_results.is_empty() {
        return local_results;
    }

    log::info!(
        "[KnowledgeBridge] DHT returned {} additional claims",
        dht_results.len()
    );

    // Merge and deduplicate by content prefix (first 50 chars)
    let mut merged = local_results;
    let existing_prefixes: std::collections::HashSet<String> = merged
        .iter()
        .map(|r| r.content.chars().take(50).collect())
        .collect();

    for result in dht_results {
        let prefix: String = result.content.chars().take(50).collect();
        if !existing_prefixes.contains(&prefix) {
            merged.push(result);
        }
    }

    // Re-rank by composite score
    merged.sort_by(|a, b| {
        b.rank_score()
            .partial_cmp(&a.rank_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merged
}
