// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared Error Types for Mycelix ERP
//!
//! Common error types used across all ERP modules.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

/// Service error types
#[derive(Debug, Error)]
pub enum ServiceError {
    /// Database operation failed
    #[error("Database error: {0}")]
    Database(String),

    /// Resource not found
    #[error("Not found: {0}")]
    NotFound(String),

    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),

    /// Authorization error
    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    /// Forbidden
    #[error("Forbidden: {0}")]
    Forbidden(String),

    /// Conflict (e.g., duplicate key)
    #[error("Conflict: {0}")]
    Conflict(String),

    /// Internal server error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Bad request
    #[error("Bad request: {0}")]
    BadRequest(String),

    /// External service error
    #[error("External service error: {0}")]
    ExternalService(String),
}

/// Error response body
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            ServiceError::Database(msg) => {
                tracing::error!("Database error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, "database_error", msg.clone())
            }
            ServiceError::NotFound(msg) => {
                (StatusCode::NOT_FOUND, "not_found", msg.clone())
            }
            ServiceError::Validation(msg) => {
                (StatusCode::BAD_REQUEST, "validation_error", msg.clone())
            }
            ServiceError::Unauthorized(msg) => {
                (StatusCode::UNAUTHORIZED, "unauthorized", msg.clone())
            }
            ServiceError::Forbidden(msg) => {
                (StatusCode::FORBIDDEN, "forbidden", msg.clone())
            }
            ServiceError::Conflict(msg) => {
                (StatusCode::CONFLICT, "conflict", msg.clone())
            }
            ServiceError::Internal(msg) => {
                tracing::error!("Internal error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", "An internal error occurred".to_string())
            }
            ServiceError::BadRequest(msg) => {
                (StatusCode::BAD_REQUEST, "bad_request", msg.clone())
            }
            ServiceError::ExternalService(msg) => {
                tracing::warn!("External service error: {}", msg);
                (StatusCode::BAD_GATEWAY, "external_service_error", msg.clone())
            }
        };

        let body = ErrorResponse {
            error: error_type.to_string(),
            message,
            details: None,
        };

        (status, Json(body)).into_response()
    }
}

impl From<sqlx::Error> for ServiceError {
    fn from(err: sqlx::Error) -> Self {
        match err {
            sqlx::Error::RowNotFound => ServiceError::NotFound("Resource not found".into()),
            sqlx::Error::Database(db_err) => {
                // Check for unique constraint violations
                if db_err.code().map_or(false, |c| c == "23505") {
                    ServiceError::Conflict("Resource already exists".into())
                } else {
                    ServiceError::Database(db_err.to_string())
                }
            }
            _ => ServiceError::Database(err.to_string()),
        }
    }
}

impl From<std::io::Error> for ServiceError {
    fn from(err: std::io::Error) -> Self {
        ServiceError::Internal(err.to_string())
    }
}
