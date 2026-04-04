// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! REST API client for the DeSci backend.

use crate::types::*;
use gloo_net::http::Request;

const API_BASE: &str = "/api/v1";

/// Fetch aggregate statistics.
pub async fn fetch_stats() -> Result<StatsResponse, String> {
    Request::get(&format!("{API_BASE}/query/stats"))
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?
        .json::<StatsResponse>()
        .await
        .map_err(|e| format!("Parse error: {e}"))
}

/// Fetch available categories.
pub async fn fetch_categories() -> Result<Vec<String>, String> {
    #[derive(serde::Deserialize)]
    struct CategoriesResponse {
        categories: Vec<String>,
    }

    let resp = Request::get(&format!("{API_BASE}/query/categories"))
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?
        .json::<CategoriesResponse>()
        .await
        .map_err(|e| format!("Parse error: {e}"))?;

    Ok(resp.categories)
}

/// Query claims with filters.
pub async fn query_claims(query: &QueryRequest) -> Result<QueryResponse, String> {
    Request::post(&format!("{API_BASE}/query"))
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(query).map_err(|e| format!("Serialize error: {e}"))?)
        .map_err(|e| format!("Request build error: {e}"))?
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?
        .json::<QueryResponse>()
        .await
        .map_err(|e| format!("Parse error: {e}"))
}

/// Get a single claim by ID.
pub async fn get_claim(id: &str) -> Result<ClaimResponse, String> {
    Request::get(&format!("{API_BASE}/claims/{id}"))
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?
        .json::<ClaimResponse>()
        .await
        .map_err(|e| format!("Parse error: {e}"))
}

/// Create a new claim.
pub async fn create_claim(req: &CreateClaimRequest) -> Result<ClaimResponse, String> {
    Request::post(&format!("{API_BASE}/claims"))
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(req).map_err(|e| format!("Serialize error: {e}"))?)
        .map_err(|e| format!("Request build error: {e}"))?
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?
        .json::<ClaimResponse>()
        .await
        .map_err(|e| format!("Parse error: {e}"))
}
