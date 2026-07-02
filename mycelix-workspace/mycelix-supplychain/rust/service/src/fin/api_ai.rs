// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AI Invoice Processing API endpoints

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use axum_extra::extract::Multipart;
use serde::Deserialize;
use serde_json::json;
use sqlx::PgPool;
use uuid::Uuid;

use super::ai_processing::{AiInvoiceService, ExtractionConfig};

/// AI Processing API state
#[derive(Clone)]
pub struct AiProcessingState {
    pub service: AiInvoiceService,
}

impl AiProcessingState {
    pub fn new(pool: PgPool) -> Self {
        Self {
            service: AiInvoiceService::new(pool),
        }
    }
}

/// Create AI processing router
pub fn ai_processing_router(state: AiProcessingState) -> Router {
    Router::new()
        // Queue management
        .route("/v1/fin/ai/queue", get(list_queue))
        .route("/v1/fin/ai/queue", post(queue_document))
        .route("/v1/fin/ai/queue/:id", get(get_queue_item))
        .route("/v1/fin/ai/queue/:id/process", post(process_document))
        // Extracted data
        .route("/v1/fin/ai/extracted/:queue_id", get(get_extracted_data))
        .route("/v1/fin/ai/extracted/:queue_id/approve", post(approve_extraction))
        .route("/v1/fin/ai/extracted/:queue_id/correct", post(record_correction))
        // Vendor patterns
        .route("/v1/fin/ai/vendor-patterns", post(add_vendor_pattern))
        // File upload
        .route("/v1/fin/ai/upload", post(upload_document))
        .with_state(state)
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AiTenantQuery {
    pub tenant_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct ListQueueQuery {
    pub tenant_id: Uuid,
    pub status: Option<String>,
    pub limit: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct QueueDocumentRequest {
    pub tenant_id: Uuid,
    pub source_type: String,
    pub file_name: Option<String>,
    pub file_path: Option<String>,
    pub file_size: Option<i32>,
    pub mime_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ProcessRequest {
    pub extract_line_items: Option<bool>,
    pub extract_tax_details: Option<bool>,
    pub match_vendors: Option<bool>,
    pub suggest_gl_accounts: Option<bool>,
    pub min_confidence: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct ApproveExtractionRequest {
    pub user_id: Uuid,
    pub create_as: String,  // INVOICE or BILL
}

#[derive(Debug, Deserialize)]
pub struct RecordCorrectionRequest {
    pub sample_type: String,
    pub input_text: String,
    pub expected_output: String,
    pub user_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct AddVendorPatternRequest {
    pub tenant_id: Uuid,
    pub vendor_id: Uuid,
    pub pattern_type: String,
    pub pattern_value: String,
}

// ============================================================================
// Handlers
// ============================================================================

async fn list_queue(
    State(state): State<AiProcessingState>,
    Query(query): Query<ListQueueQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50);

    match state.service.get_pending_items(query.tenant_id, limit).await {
        Ok(items) => (StatusCode::OK, Json(json!({
            "items": items,
            "count": items.len()
        }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn queue_document(
    State(state): State<AiProcessingState>,
    Json(req): Json<QueueDocumentRequest>,
) -> impl IntoResponse {
    match state.service.queue_document(
        req.tenant_id,
        &req.source_type,
        req.file_name.as_deref(),
        req.file_path.as_deref(),
        req.file_size,
        req.mime_type.as_deref(),
    ).await {
        Ok(item) => (StatusCode::CREATED, Json(json!(item))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn get_queue_item(
    State(state): State<AiProcessingState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.service.get_queue_item(id).await {
        Ok(item) => (StatusCode::OK, Json(json!(item))).into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn process_document(
    State(state): State<AiProcessingState>,
    Path(queue_id): Path<Uuid>,
    Json(req): Json<ProcessRequest>,
) -> impl IntoResponse {
    let config = ExtractionConfig {
        extract_line_items: req.extract_line_items.unwrap_or(true),
        extract_tax_details: req.extract_tax_details.unwrap_or(true),
        match_vendors: req.match_vendors.unwrap_or(true),
        suggest_gl_accounts: req.suggest_gl_accounts.unwrap_or(true),
        min_confidence: req.min_confidence.unwrap_or(0.7),
    };

    match state.service.process_document(queue_id, config).await {
        Ok(extracted) => (StatusCode::OK, Json(json!({
            "extracted": extracted,
            "queue_id": queue_id
        }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn get_extracted_data(
    State(state): State<AiProcessingState>,
    Path(queue_id): Path<Uuid>,
) -> impl IntoResponse {
    match state.service.get_extracted_data(queue_id).await {
        Ok(Some(extracted)) => (
            StatusCode::OK,
            Json(json!({
                "queue_id": queue_id,
                "extracted": extracted
            })),
        ).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": "No extracted data found for this queue item",
                "queue_id": queue_id
            })),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn approve_extraction(
    State(state): State<AiProcessingState>,
    Path(queue_id): Path<Uuid>,
    Json(req): Json<ApproveExtractionRequest>,
) -> impl IntoResponse {
    match state.service.approve_extraction(queue_id, req.user_id, &req.create_as).await {
        Ok(result) => (
            StatusCode::OK,
            Json(json!({
                "success": true,
                "queue_id": result.queue_id,
                "created_type": result.created_type,
                "created_id": result.created_id,
                "message": format!("{} created successfully from extracted data", result.created_type)
            })),
        ).into_response(),
        Err(e) => {
            let status = match &e {
                super::ai_processing::AiProcessingError::NotFound(_) => StatusCode::NOT_FOUND,
                super::ai_processing::AiProcessingError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (
                status,
                Json(json!({ "error": e.to_string() })),
            ).into_response()
        }
    }
}

async fn record_correction(
    State(state): State<AiProcessingState>,
    Path(queue_id): Path<Uuid>,
    Query(query): Query<AiTenantQuery>,
    Json(req): Json<RecordCorrectionRequest>,
) -> impl IntoResponse {
    match state.service.record_correction(
        query.tenant_id,
        queue_id,
        &req.sample_type,
        &req.input_text,
        &req.expected_output,
        req.user_id,
    ).await {
        Ok(()) => (StatusCode::OK, Json(json!({
            "success": true,
            "message": "Correction recorded for AI training"
        }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn add_vendor_pattern(
    State(state): State<AiProcessingState>,
    Json(req): Json<AddVendorPatternRequest>,
) -> impl IntoResponse {
    match state.service.add_vendor_pattern(
        req.tenant_id,
        req.vendor_id,
        &req.pattern_type,
        &req.pattern_value,
    ).await {
        Ok(()) => (StatusCode::CREATED, Json(json!({
            "success": true,
            "pattern_type": req.pattern_type,
            "pattern_value": req.pattern_value
        }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn upload_document(
    State(state): State<AiProcessingState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut tenant_id: Option<Uuid> = None;
    let mut file_name: Option<String> = None;
    let mut file_data: Option<Vec<u8>> = None;
    let mut mime_type: Option<String> = None;

    // Parse multipart form
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "tenant_id" => {
                if let Ok(text) = field.text().await {
                    tenant_id = Uuid::parse_str(&text).ok();
                }
            }
            "file" => {
                file_name = field.file_name().map(|s| s.to_string());
                mime_type = field.content_type().map(|s| s.to_string());
                if let Ok(data) = field.bytes().await {
                    file_data = Some(data.to_vec());
                }
            }
            _ => {}
        }
    }

    // Validate required fields
    let tenant_id = match tenant_id {
        Some(id) => id,
        None => return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "tenant_id is required" })),
        ).into_response(),
    };

    let file_size = file_data.as_ref().map(|d| d.len() as i32);

    // Queue the document
    match state.service.queue_document(
        tenant_id,
        "UPLOAD",
        file_name.as_deref(),
        None, // Would save file and provide path in production
        file_size,
        mime_type.as_deref(),
    ).await {
        Ok(item) => {
            // In production, would also save file to storage here
            (StatusCode::CREATED, Json(json!({
                "queue_item": item,
                "file_name": file_name,
                "file_size": file_size,
                "mime_type": mime_type
            }))).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}
