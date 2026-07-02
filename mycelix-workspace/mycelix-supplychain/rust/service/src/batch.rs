// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use claim_model::SupplyEventVC;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use crate::AppState;

/// Maximum number of events allowed in a single batch request
const MAX_BATCH_SIZE: usize = 100;

/// Request body for batch event ingestion
#[derive(Debug, Deserialize)]
pub struct BatchIngestRequest {
    /// List of supply chain events to ingest
    pub events: Vec<SupplyEventVC>,

    /// Processing mode: "best-effort" (default) or "atomic"
    /// - best-effort: Process all events, return partial success
    /// - atomic: All events succeed or all fail (transaction)
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_mode() -> String {
    "best-effort".to_string()
}

/// Response for batch event ingestion
#[derive(Debug, Serialize)]
pub struct BatchIngestResponse {
    /// Total number of events in the batch
    pub total: usize,

    /// Number of events successfully ingested
    pub succeeded: usize,

    /// Number of events that failed
    pub failed: usize,

    /// Processing time in milliseconds
    pub duration_ms: u64,

    /// Detailed results for each event
    pub results: Vec<EventResult>,
}

/// Result for a single event in the batch
#[derive(Debug, Serialize)]
pub struct EventResult {
    /// Index of the event in the request array
    pub index: usize,

    /// Processing status: "success" or "failed"
    pub status: String,

    /// Claim ID (only for successful events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claim_id: Option<String>,

    /// Lineage hash (only for successful events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lineage_hash: Option<String>,

    /// Error message (only for failed events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// Processing time for this event in milliseconds
    pub duration_ms: u64,
}

impl EventResult {
    fn success(index: usize, claim_id: String, lineage_hash: String, duration_ms: u64) -> Self {
        Self {
            index,
            status: "success".to_string(),
            claim_id: Some(claim_id),
            lineage_hash: Some(lineage_hash),
            error: None,
            duration_ms,
        }
    }

    fn failure(index: usize, error: String, duration_ms: u64) -> Self {
        Self {
            index,
            status: "failed".to_string(),
            claim_id: None,
            lineage_hash: None,
            error: Some(error),
            duration_ms,
        }
    }
}

/// Error type for batch operations
#[derive(Debug)]
pub enum BatchError {
    TooManyEvents(usize),
    EmptyBatch,
    InvalidMode(String),
    ProcessingError(String),
}

impl IntoResponse for BatchError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            BatchError::TooManyEvents(count) => (
                StatusCode::BAD_REQUEST,
                format!(
                    "Batch size {} exceeds maximum of {}",
                    count, MAX_BATCH_SIZE
                ),
            ),
            BatchError::EmptyBatch => (StatusCode::BAD_REQUEST, "Batch cannot be empty".to_string()),
            BatchError::InvalidMode(mode) => (
                StatusCode::BAD_REQUEST,
                format!("Invalid mode '{}' (must be 'best-effort' or 'atomic')", mode),
            ),
            BatchError::ProcessingError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

/// Process a single event (extracted for reusability)
async fn process_single_event(
    event: SupplyEventVC,
    state: Arc<AppState>,
) -> Result<(String, String), String> {
    // Validate the event
    if let Err(e) = event.validate() {
        return Err(format!("Validation failed: {}", e));
    }

    // Process through pipeline (signing, lineage resolution)
    let result = match crate::pipeline::process_event(&state, event).await {
        Ok(r) => r,
        Err(e) => return Err(format!("Processing failed: {}", e)),
    };

    let claim_id = result.claim.id.clone();
    let lineage_hash = result.claim.lineage.hash.clone();

    // Store the claim (use database if available, otherwise in-memory)
    if let Some(ref db) = state.db {
        if let Err(e) = db.store_claim(&result.claim).await {
            return Err(format!("Database storage failed: {}", e));
        }
    } else {
        let mut claims = state.claims.write().await;
        claims.insert(result.claim.id.clone(), result.claim.clone());
    }

    Ok((claim_id, lineage_hash))
}

/// Handler for POST /v1/events/batch
pub async fn ingest_batch(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BatchIngestRequest>,
) -> Result<(StatusCode, Json<BatchIngestResponse>), BatchError> {
    let start_time = Instant::now();
    let total_events = request.events.len();

    info!(
        total_events = total_events,
        mode = %request.mode,
        "Starting batch event ingestion"
    );

    // Validation
    if total_events == 0 {
        warn!("Rejected empty batch");
        return Err(BatchError::EmptyBatch);
    }

    if total_events > MAX_BATCH_SIZE {
        warn!(
            total_events = total_events,
            max_batch_size = MAX_BATCH_SIZE,
            "Rejected batch: too many events"
        );
        return Err(BatchError::TooManyEvents(total_events));
    }

    if request.mode != "best-effort" && request.mode != "atomic" {
        warn!(mode = %request.mode, "Invalid processing mode");
        return Err(BatchError::InvalidMode(request.mode));
    }

    // Record metrics
    crate::metrics::EVENTS_INGESTED
        .with_label_values(&["batch"])
        .inc_by(total_events as f64);

    // Process events based on mode
    let results = if request.mode == "atomic" {
        process_atomic(request.events, state.clone()).await?
    } else {
        process_best_effort(request.events, state.clone()).await
    };

    // Calculate statistics
    let succeeded = results.iter().filter(|r| r.status == "success").count();
    let failed = results.iter().filter(|r| r.status == "failed").count();
    let duration_ms = start_time.elapsed().as_millis() as u64;

    info!(
        total = total_events,
        succeeded = succeeded,
        failed = failed,
        duration_ms = duration_ms,
        "Batch ingestion completed"
    );

    // Record batch metrics
    crate::metrics::API_REQUEST_DURATION
        .with_label_values(&["POST", "/v1/events/batch", "201"])
        .observe(duration_ms as f64 / 1000.0);

    Ok((
        StatusCode::CREATED,
        Json(BatchIngestResponse {
            total: total_events,
            succeeded,
            failed,
            duration_ms,
            results,
        }),
    ))
}

/// Process events in best-effort mode (partial success allowed)
async fn process_best_effort(
    events: Vec<SupplyEventVC>,
    state: Arc<AppState>,
) -> Vec<EventResult> {
    debug!(count = events.len(), "Processing batch in best-effort mode");

    let mut tasks = JoinSet::new();

    // Spawn parallel tasks for each event
    for (index, event) in events.into_iter().enumerate() {
        let state_clone = state.clone();

        tasks.spawn(async move {
            let event_start = Instant::now();

            let result = match process_single_event(event, state_clone).await {
                Ok((claim_id, lineage_hash)) => {
                    let duration = event_start.elapsed().as_millis() as u64;
                    EventResult::success(index, claim_id, lineage_hash, duration)
                }
                Err(error) => {
                    let duration = event_start.elapsed().as_millis() as u64;
                    warn!(index = index, error = %error, "Event processing failed");
                    EventResult::failure(index, error, duration)
                }
            };

            result
        });
    }

    // Collect results
    let mut results = Vec::new();
    while let Some(res) = tasks.join_next().await {
        match res {
            Ok(event_result) => results.push(event_result),
            Err(e) => {
                warn!(error = %e, "Task join error");
                // This shouldn't happen, but handle it gracefully
                results.push(EventResult::failure(
                    results.len(),
                    format!("Task error: {}", e),
                    0,
                ));
            }
        }
    }

    // Sort by index to maintain order
    results.sort_by_key(|r| r.index);

    results
}

/// Process events in atomic mode (all or nothing)
async fn process_atomic(
    events: Vec<SupplyEventVC>,
    state: Arc<AppState>,
) -> Result<Vec<EventResult>, BatchError> {
    debug!(count = events.len(), "Processing batch in atomic mode");

    // Note: For true atomic behavior, we'd need transaction support in the database
    // For now, we process sequentially and stop on first error
    let mut results = Vec::new();

    for (index, event) in events.into_iter().enumerate() {
        let event_start = Instant::now();

        match process_single_event(event, state.clone()).await {
            Ok((claim_id, lineage_hash)) => {
                let duration = event_start.elapsed().as_millis() as u64;
                results.push(EventResult::success(
                    index,
                    claim_id,
                    lineage_hash,
                    duration,
                ));
            }
            Err(e) => {
                warn!(index = index, error = %e, "Event processing failed in atomic mode");
                return Err(BatchError::ProcessingError(format!(
                    "Event at index {} failed: {}. Note: Previous {} events were stored.",
                    index, e, index
                )));
            }
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_batch_size() {
        assert_eq!(MAX_BATCH_SIZE, 100);
    }

    #[test]
    fn test_default_mode() {
        assert_eq!(default_mode(), "best-effort");
    }

    #[test]
    fn test_event_result_success() {
        let result = EventResult::success(
            0,
            "claim-123".to_string(),
            "hash-abc".to_string(),
            42,
        );

        assert_eq!(result.index, 0);
        assert_eq!(result.status, "success");
        assert_eq!(result.claim_id, Some("claim-123".to_string()));
        assert_eq!(result.lineage_hash, Some("hash-abc".to_string()));
        assert_eq!(result.error, None);
        assert_eq!(result.duration_ms, 42);
    }

    #[test]
    fn test_event_result_failure() {
        let result = EventResult::failure(5, "Invalid event".to_string(), 10);

        assert_eq!(result.index, 5);
        assert_eq!(result.status, "failed");
        assert_eq!(result.claim_id, None);
        assert_eq!(result.lineage_hash, None);
        assert_eq!(result.error, Some("Invalid event".to_string()));
        assert_eq!(result.duration_ms, 10);
    }
}
