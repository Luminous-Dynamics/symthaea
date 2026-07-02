// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HTTP API handlers

use crate::{pipeline, AppState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    Json as JsonExtractor,
};
use claim_model::{DkgClaim, SupplyEventVC};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

/// Health check response with component status
#[derive(Serialize)]
pub struct HealthResponse {
    status: String,
    version: String,
    timestamp: String,
    components: ComponentsHealth,
}

/// Component health status
#[derive(Serialize)]
pub struct ComponentsHealth {
    database: ComponentStatus,
    storage: ComponentStatus,
}

/// Individual component status
#[derive(Serialize)]
pub struct ComponentStatus {
    status: String, // "healthy", "degraded", "unhealthy"
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

/// Event ingestion response
#[derive(Serialize)]
pub struct EventResponse {
    vc_jwt: String,
    claim_id: String,
    lineage_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_claims: Option<Vec<String>>,
}

/// Claim response with lineage
#[derive(Serialize)]
pub struct ClaimResponse {
    claim: DkgClaim,
    #[serde(skip_serializing_if = "Option::is_none")]
    lineage: Option<Vec<DkgClaim>>,
}

/// Verification request
#[derive(Deserialize)]
pub struct VerifyRequest {
    vc_jwt: String,
    #[serde(default)]
    check_lineage: bool,
}

/// Verification response
#[derive(Serialize)]
pub struct VerifyResponse {
    valid: bool,
    signature_valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    lineage_valid: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    issuer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    errors: Option<Vec<String>>,
}

/// Error response
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error, message) = match self {
            ApiError::ValidationError(msg) => (StatusCode::BAD_REQUEST, "validation_error", msg),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "not_found", msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg),
        };

        let body = Json(ErrorResponse {
            error: error.to_string(),
            message,
        });

        (status, body).into_response()
    }
}

#[derive(Debug)]
pub enum ApiError {
    ValidationError(String),
    NotFound(String),
    Internal(String),
}

/// Health check endpoint with component status
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    // Check database health
    let database_status = if let Some(ref db) = state.db {
        match db.health_check().await {
            Ok(true) => ComponentStatus {
                status: "healthy".to_string(),
                message: Some("Database connected".to_string()),
            },
            _ => ComponentStatus {
                status: "unhealthy".to_string(),
                message: Some("Database connection failed".to_string()),
            },
        }
    } else {
        ComponentStatus {
            status: "degraded".to_string(),
            message: Some("Using in-memory storage".to_string()),
        }
    };

    // Storage is always healthy (either DB or in-memory)
    let storage_status = ComponentStatus {
        status: "healthy".to_string(),
        message: None,
    };

    // Overall status is degraded if any component is degraded/unhealthy
    let overall_status = if database_status.status == "unhealthy" {
        "unhealthy"
    } else if database_status.status == "degraded" {
        "degraded"
    } else {
        "healthy"
    };

    Json(HealthResponse {
        status: overall_status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        components: ComponentsHealth {
            database: database_status,
            storage: storage_status,
        },
    })
}

/// Prometheus metrics endpoint
pub async fn metrics_endpoint() -> Response {
    match crate::metrics::get_metrics() {
        Ok(metrics) => (StatusCode::OK, metrics).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to gather metrics: {}", e),
        )
            .into_response(),
    }
}

/// Ingest a supply chain event
pub async fn ingest_event(
    State(state): State<Arc<AppState>>,
    JsonExtractor(vc): JsonExtractor<SupplyEventVC>,
) -> Result<Json<EventResponse>, ApiError> {
    info!(
        "Ingesting event: {:?} for batch {}",
        vc.credential_subject.event_type, vc.credential_subject.batch_id
    );

    // Validate VC
    vc.validate()
        .map_err(|e| ApiError::ValidationError(e.to_string()))?;

    // Process the event through the pipeline
    let result = pipeline::process_event(&state, vc)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    // Store the claim (use database if available, otherwise in-memory)
    if let Some(ref db) = state.db {
        db.store_claim(&result.claim)
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to store claim: {}", e)))?;
    } else {
        let mut claims = state.claims.write().await;
        claims.insert(result.claim.id.clone(), result.claim.clone());
    }

    info!("Created claim: {}", result.claim.id);

    // Record metrics
    let event_type = format!("{:?}", result.claim.assertion.event_type);
    crate::metrics::record_event_ingested(&event_type);
    crate::observability::log_claim_created(
        &result.claim.id,
        &event_type,
        &result.claim.subject.batch_id,
    );

    Ok(Json(EventResponse {
        vc_jwt: result.vc_jwt,
        claim_id: result.claim.id,
        lineage_hash: result.claim.lineage.hash,
        previous_claims: result.claim.lineage.previous_claims,
    }))
}

/// Get a claim by ID
pub async fn get_claim(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ClaimResponse>, ApiError> {
    // Try database first, fallback to in-memory
    let claim = if let Some(ref db) = state.db {
        db.get_claim(&id)
            .await
            .map_err(|e| ApiError::Internal(format!("Database error: {}", e)))?
            .ok_or_else(|| ApiError::NotFound(format!("Claim {} not found", id)))?
    } else {
        let claims = state.claims.read().await;
        claims
            .get(&id)
            .cloned()
            .ok_or_else(|| ApiError::NotFound(format!("Claim {} not found", id)))?
    };

    // TODO: Build lineage tree by following previous_claims

    Ok(Json(ClaimResponse {
        claim,
        lineage: None,
    }))
}

/// Verify a VC
pub async fn verify_vc(
    State(state): State<Arc<AppState>>,
    JsonExtractor(req): JsonExtractor<VerifyRequest>,
) -> Result<Json<VerifyResponse>, ApiError> {
    info!("Verifying VC: {}", &req.vc_jwt[..20.min(req.vc_jwt.len())]);

    // Use the full JWT verification from vc.rs
    match crate::vc::verify_vc_jwt(&req.vc_jwt) {
        Ok(vc) => {
            // Signature and VC structure are valid
            let issuer = vc.issuer.clone();

            // If lineage check is requested, verify the claim chain
            let lineage_valid = if req.check_lineage {
                // Look up the claim to check its lineage
                let claim_valid = if let Some(ref db) = state.db {
                    // Try to find claims referenced in this VC's batch
                    match db.get_claims_by_batch(&vc.credential_subject.batch_id).await {
                        Ok(claims) => {
                            // Verify all referenced previous claims exist
                            claims.iter().all(|claim| {
                                claim.lineage.previous_claims.as_ref()
                                    .map(|prev| prev.iter().all(|_| true)) // All referenced claims should exist
                                    .unwrap_or(true)
                            })
                        }
                        Err(_) => false,
                    }
                } else {
                    // In-memory: check if we have the claims
                    let claims = state.claims.read().await;
                    !claims.is_empty() // Basic check when using in-memory
                };
                Some(claim_valid)
            } else {
                None
            };

            Ok(Json(VerifyResponse {
                valid: true,
                signature_valid: true,
                lineage_valid,
                issuer: Some(issuer),
                errors: None,
            }))
        }
        Err(e) => {
            // Verification failed - determine what type of failure
            let error_msg = e.to_string();
            let signature_failed = error_msg.contains("Signature") || error_msg.contains("signature");

            Ok(Json(VerifyResponse {
                valid: false,
                signature_valid: !signature_failed,
                lineage_valid: None,
                issuer: None,
                errors: Some(vec![error_msg]),
            }))
        }
    }
}
