// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Query API handlers

use axum::{Json, extract::State};
use mycelix_desci_core::query::{QueryFilter, SortBy};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

use crate::{error::Result, models::*, state::AppState};

/// Execute a query against claims
#[utoipa::path(
    post,
    path = "/api/v1/query",
    request_body = QueryRequest,
    responses(
        (status = 200, description = "Query executed successfully", body = QueryResponse),
        (status = 400, description = "Invalid query", body = ApiError),
    ),
    tag = "query"
)]
pub async fn execute_query(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>> {
    info!("Executing query");

    let query_engine = state.query_engine.read().await;

    // Build query filter from request
    let page = req.page.unwrap_or(1).max(1);
    let page_size = req.page_size.unwrap_or(20).clamp(1, 100);
    let offset = (page - 1) * page_size;

    let sort_by = req.sort.as_ref().map(|s| match s.field.as_str() {
        "created_at" => SortBy::CreatedAt,
        "tier" => SortBy::EpistemicTier,
        "updated_at" => SortBy::UpdatedAt,
        "verification_count" => SortBy::VerificationCount,
        "category" => SortBy::Category,
        _ => SortBy::CreatedAt,
    });

    let filter = QueryFilter {
        category: req.category.clone(),
        keywords: req.keywords.clone().unwrap_or_default(),
        min_tier: req.tier,
        max_tier: None,
        creator: req.creator.clone(),
        date_from: None,
        date_to: None,
        min_verifications: None,
        license: None,
        sort_by,
        sort_order: req.sort.as_ref().map(|s| s.order),
        offset: Some(offset),
        limit: Some(page_size),
    };

    // Execute query
    let result = query_engine.query(&filter).await?;

    let response = QueryResponse {
        results: result
            .claims
            .iter()
            .map(|c| ClaimResponse::from(c))
            .collect(),
        total_count: result.total_count,
        page,
        page_size,
        total_pages: (result.total_count + page_size - 1) / page_size,
    };

    info!(
        total_results = result.total_count,
        page = page,
        page_size = page_size,
        execution_time_ms = result.execution_time_ms,
        "Query executed"
    );

    Ok(Json(response))
}

/// Get all available categories
#[utoipa::path(
    get,
    path = "/api/v1/query/categories",
    responses(
        (status = 200, description = "Categories retrieved", body = CategoriesResponse),
    ),
    tag = "query"
)]
pub async fn get_categories(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<CategoriesResponse>> {
    info!("Retrieving categories");

    // TODO: Implement category extraction from index
    // For now, return empty list
    let categories = Vec::new();

    Ok(Json(CategoriesResponse { categories }))
}

/// Get query statistics
#[utoipa::path(
    get,
    path = "/api/v1/query/stats",
    responses(
        (status = 200, description = "Statistics retrieved", body = QueryStatsResponse),
    ),
    tag = "query"
)]
pub async fn get_stats(State(_state): State<Arc<AppState>>) -> Result<Json<QueryStatsResponse>> {
    info!("Retrieving query statistics");

    // TODO: Implement stats calculation from index
    // For now, return placeholder data
    let response = QueryStatsResponse {
        total_claims: 0,
        claims_by_tier: HashMap::new(),
        total_categories: 0,
        total_keywords: 0,
    };

    Ok(Json(response))
}
