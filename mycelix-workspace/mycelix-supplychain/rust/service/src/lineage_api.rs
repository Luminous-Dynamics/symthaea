// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lineage and batch query API endpoints
//!
//! Provides endpoints for querying supply chain lineage and batch history

use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use claim_model::DkgClaim;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

use crate::api::ApiError;
use crate::AppState;

/// Response for batch claims query
#[derive(Debug, Serialize)]
pub struct BatchClaimsResponse {
    /// Batch ID queried
    pub batch_id: String,

    /// All claims for this batch
    pub claims: Vec<DkgClaim>,

    /// Total number of claims
    pub total_claims: usize,

    /// Timestamp of query
    pub timestamp: String,
}

/// Response for lineage query
#[derive(Debug, Serialize)]
pub struct LineageResponse {
    /// Batch ID queried
    pub batch_id: String,

    /// Claims for this batch (in chronological order)
    pub claims: Vec<DkgClaim>,

    /// Upstream batches (source materials)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<Vec<UpstreamBatch>>,

    /// Downstream batches (derived products)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downstream: Option<Vec<DownstreamBatch>>,

    /// Total number of claims in lineage
    pub total_claims: usize,

    /// Maximum lineage depth
    pub depth: usize,
}

/// Upstream batch information
#[derive(Debug, Serialize)]
pub struct UpstreamBatch {
    /// Batch ID
    pub batch_id: String,

    /// Number of claims for this batch
    pub claim_count: usize,

    /// First event type
    pub first_event_type: String,
}

/// Downstream batch information
#[derive(Debug, Serialize)]
pub struct DownstreamBatch {
    /// Batch ID
    pub batch_id: String,

    /// Number of claims for this batch
    pub claim_count: usize,

    /// First event type
    pub first_event_type: String,
}

/// Query parameters for claims search
#[derive(Debug, Deserialize)]
pub struct ClaimFilters {
    /// Filter by product ID
    #[serde(rename = "product_id")]
    pub product_id: Option<String>,

    /// Filter by batch ID
    #[serde(rename = "batch_id")]
    pub batch_id: Option<String>,

    /// Filter by facility ID
    #[serde(rename = "facility_id")]
    pub facility_id: Option<String>,

    /// Filter by event type
    #[serde(rename = "event_type")]
    pub event_type: Option<String>,

    /// Filter by date range (ISO 8601)
    #[serde(rename = "from")]
    pub from: Option<String>,

    /// Filter by date range (ISO 8601)
    #[serde(rename = "to")]
    pub to: Option<String>,

    /// Limit results
    #[serde(default = "default_limit")]
    pub limit: usize,

    /// Offset for pagination
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    50
}

/// Search response
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    /// Matching claims
    pub claims: Vec<DkgClaim>,

    /// Total matching claims (before pagination)
    pub total: usize,

    /// Limit applied
    pub limit: usize,

    /// Offset applied
    pub offset: usize,

    /// Whether there are more results
    pub has_more: bool,
}

/// GET /v1/batches/:batch_id/claims
///
/// Get all claims for a specific batch ID
pub async fn get_batch_claims(
    State(state): State<Arc<AppState>>,
    Path(batch_id): Path<String>,
) -> Result<Json<BatchClaimsResponse>, ApiError> {
    info!(batch_id = %batch_id, "Querying claims for batch");

    let claims = if let Some(ref db) = state.db {
        // Use database if available
        db.get_claims_by_batch(&batch_id)
            .await
            .map_err(|e| ApiError::Internal(format!("Database error: {}", e)))?
    } else {
        // Fallback to in-memory storage
        let claims_map = state.claims.read().await;
        claims_map
            .values()
            .filter(|claim| claim.subject.batch_id == batch_id)
            .cloned()
            .collect()
    };

    let total = claims.len();

    debug!(
        batch_id = %batch_id,
        total_claims = total,
        "Found claims for batch"
    );

    // Record metrics
    crate::metrics::QUERY_RESULTS_COUNT
        .with_label_values(&["batch_claims"])
        .observe(total as f64);

    Ok(Json(BatchClaimsResponse {
        batch_id,
        claims,
        total_claims: total,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// GET /v1/lineage/:batch_id
///
/// Get full lineage for a batch (upstream and downstream)
pub async fn get_lineage(
    State(state): State<Arc<AppState>>,
    Path(batch_id): Path<String>,
) -> Result<Json<LineageResponse>, ApiError> {
    info!(batch_id = %batch_id, "Querying lineage for batch");

    // Get claims for this batch
    let claims = if let Some(ref db) = state.db {
        db.get_claims_by_batch(&batch_id)
            .await
            .map_err(|e| ApiError::Internal(format!("Database error: {}", e)))?
    } else {
        let claims_map = state.claims.read().await;
        claims_map
            .values()
            .filter(|claim| claim.subject.batch_id == batch_id)
            .cloned()
            .collect()
    };

    // Return empty lineage if no claims found (instead of 404)
    if claims.is_empty() {
        debug!(
            batch_id = %batch_id,
            "No claims found for batch, returning empty lineage"
        );

        return Ok(Json(LineageResponse {
            batch_id,
            claims: vec![],
            upstream: None,
            downstream: None,
            total_claims: 0,
            depth: 0,
        }));
    }

    // Find upstream batches (sources)
    let upstream = find_upstream_batches(&claims, &state).await?;

    // Find downstream batches (derivatives)
    let downstream = find_downstream_batches(&batch_id, &state).await?;

    let total_claims = claims.len()
        + upstream.as_ref().map(|u| u.iter().map(|b| b.claim_count).sum::<usize>()).unwrap_or(0)
        + downstream.as_ref().map(|d| d.iter().map(|b| b.claim_count).sum::<usize>()).unwrap_or(0);

    let depth = calculate_depth(&upstream, &downstream);

    debug!(
        batch_id = %batch_id,
        total_claims = total_claims,
        depth = depth,
        "Lineage query complete"
    );

    // Record metrics
    crate::metrics::LINEAGE_DEPTH
        .with_label_values(&["full"])
        .observe(depth as f64);

    Ok(Json(LineageResponse {
        batch_id,
        claims,
        upstream,
        downstream,
        total_claims,
        depth,
    }))
}

/// GET /v1/claims
///
/// Search and filter claims
pub async fn search_claims(
    State(state): State<Arc<AppState>>,
    Query(filters): Query<ClaimFilters>,
) -> Result<Json<SearchResponse>, ApiError> {
    info!(?filters, "Searching claims");

    let all_claims: Vec<DkgClaim> = if let Some(ref db) = state.db {
        // If we have database, we could implement efficient filtering
        // For now, get all and filter in memory
        db.get_all_claims()
            .await
            .map_err(|e| ApiError::Internal(format!("Database error: {}", e)))?
    } else {
        let claims_map = state.claims.read().await;
        claims_map.values().cloned().collect()
    };

    // Apply filters
    let mut filtered: Vec<DkgClaim> = all_claims
        .into_iter()
        .filter(|claim| {
            // Filter by product_id
            if let Some(ref product_id) = filters.product_id {
                if &claim.subject.product_id != product_id {
                    return false;
                }
            }

            // Filter by batch_id
            if let Some(ref batch_id) = filters.batch_id {
                if &claim.subject.batch_id != batch_id {
                    return false;
                }
            }

            // Filter by facility_id
            if let Some(ref facility_id) = filters.facility_id {
                if let Some(ref claim_facility_id) = claim.assertion.facility_id {
                    if claim_facility_id != facility_id {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            // Filter by event_type
            if let Some(ref event_type) = filters.event_type {
                if format!("{:?}", claim.assertion.event_type).to_uppercase() != event_type.to_uppercase() {
                    return false;
                }
            }

            // Date filters would go here if we add timestamp to claims

            true
        })
        .collect();

    // Sort by timestamp (newest first)
    filtered.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    let total = filtered.len();
    let has_more = total > (filters.offset + filters.limit);

    // Apply pagination
    let paginated: Vec<DkgClaim> = filtered
        .into_iter()
        .skip(filters.offset)
        .take(filters.limit)
        .collect();

    debug!(
        total = total,
        returned = paginated.len(),
        "Search complete"
    );

    // Record metrics
    crate::metrics::QUERY_RESULTS_COUNT
        .with_label_values(&["search"])
        .observe(paginated.len() as f64);

    Ok(Json(SearchResponse {
        claims: paginated,
        total,
        limit: filters.limit,
        offset: filters.offset,
        has_more,
    }))
}

/// Find upstream batches (source materials)
async fn find_upstream_batches(
    claims: &[DkgClaim],
    state: &Arc<AppState>,
) -> Result<Option<Vec<UpstreamBatch>>, ApiError> {
    let mut upstream_batch_ids = std::collections::HashSet::new();

    // Collect all previous batch IDs from claims
    for claim in claims {
        if let Some(ref prev_claims) = claim.lineage.previous_claims {
            for prev_claim_id in prev_claims {
                // Find the claim and get its batch_id
                if let Some(ref db) = state.db {
                    if let Ok(Some(prev_claim)) = db.get_claim(prev_claim_id).await {
                        upstream_batch_ids.insert(prev_claim.subject.batch_id.clone());
                    }
                } else {
                    let claims_map = state.claims.read().await;
                    if let Some(prev_claim) = claims_map.get(prev_claim_id) {
                        upstream_batch_ids.insert(prev_claim.subject.batch_id.clone());
                    }
                }
            }
        }
    }

    if upstream_batch_ids.is_empty() {
        return Ok(None);
    }

    let mut upstream = Vec::new();
    for batch_id in upstream_batch_ids {
        let batch_claims = if let Some(ref db) = state.db {
            db.get_claims_by_batch(&batch_id).await.unwrap_or_default()
        } else {
            let claims_map = state.claims.read().await;
            claims_map
                .values()
                .filter(|c| c.subject.batch_id == batch_id)
                .cloned()
                .collect()
        };

        if !batch_claims.is_empty() {
            upstream.push(UpstreamBatch {
                batch_id,
                claim_count: batch_claims.len(),
                first_event_type: format!("{:?}", batch_claims[0].assertion.event_type),
            });
        }
    }

    Ok(Some(upstream))
}

/// Find downstream batches (derived products)
async fn find_downstream_batches(
    batch_id: &str,
    state: &Arc<AppState>,
) -> Result<Option<Vec<DownstreamBatch>>, ApiError> {
    // Get all claims and find those that reference this batch
    let all_claims: Vec<DkgClaim> = if let Some(ref db) = state.db {
        db.get_all_claims().await.unwrap_or_default()
    } else {
        let claims_map = state.claims.read().await;
        claims_map.values().cloned().collect()
    };

    let mut downstream_batch_ids = std::collections::HashSet::new();

    // Find claims that have this batch in their lineage
    for claim in &all_claims {
        if claim.subject.batch_id == batch_id {
            continue; // Skip claims from the same batch
        }

        // Check if this claim's previous claims include any from our batch
        if let Some(ref prev_claims) = claim.lineage.previous_claims {
            for prev_claim_id in prev_claims {
                // Check if this previous claim is from our batch
                let is_from_batch = if let Some(ref db) = state.db {
                    if let Ok(Some(prev_claim)) = db.get_claim(prev_claim_id).await {
                        prev_claim.subject.batch_id == batch_id
                    } else {
                        false
                    }
                } else {
                    let claims_map = state.claims.read().await;
                    claims_map
                        .get(prev_claim_id)
                        .map(|c| c.subject.batch_id == batch_id)
                        .unwrap_or(false)
                };

                if is_from_batch {
                    downstream_batch_ids.insert(claim.subject.batch_id.clone());
                }
            }
        }
    }

    if downstream_batch_ids.is_empty() {
        return Ok(None);
    }

    let mut downstream = Vec::new();
    for batch_id in downstream_batch_ids {
        let batch_claims: Vec<DkgClaim> = all_claims
            .iter()
            .filter(|c| c.subject.batch_id == batch_id)
            .cloned()
            .collect();

        if !batch_claims.is_empty() {
            downstream.push(DownstreamBatch {
                batch_id,
                claim_count: batch_claims.len(),
                first_event_type: format!("{:?}", batch_claims[0].assertion.event_type),
            });
        }
    }

    Ok(Some(downstream))
}

/// Calculate lineage depth
fn calculate_depth(upstream: &Option<Vec<UpstreamBatch>>, downstream: &Option<Vec<DownstreamBatch>>) -> usize {
    let upstream_depth = upstream.as_ref().map(|u| if u.is_empty() { 0 } else { 1 }).unwrap_or(0);
    let downstream_depth = downstream.as_ref().map(|d| if d.is_empty() { 0 } else { 1 }).unwrap_or(0);
    upstream_depth + downstream_depth + 1 // +1 for current batch
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_depth() {
        assert_eq!(calculate_depth(&None, &None), 1);
        assert_eq!(calculate_depth(&Some(vec![]), &None), 1);
        assert_eq!(
            calculate_depth(&Some(vec![UpstreamBatch {
                batch_id: "test".to_string(),
                claim_count: 1,
                first_event_type: "PRODUCED".to_string(),
            }]), &None),
            2
        );
    }

    #[test]
    fn test_default_limit() {
        assert_eq!(default_limit(), 50);
    }
}
