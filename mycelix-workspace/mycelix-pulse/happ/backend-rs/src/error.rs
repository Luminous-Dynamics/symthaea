// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error handling for Mycelix-Mail backend

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use thiserror::Error;

use crate::types::ApiError;

/// Application-level errors
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("Authentication error: {0}")]
    AuthenticationError(String),

    #[error("Authorization denied: {0}")]
    AuthorizationDenied(String),

    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Holochain error: {0}")]
    HolochainError(String),

    #[error("DID not registered: {0}")]
    DidNotFound(String),

    #[error("DID already registered: {0}")]
    DidAlreadyExists(String),

    #[error("Trust score unavailable for: {0}")]
    TrustUnavailable(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Encryption error: {0}")]
    EncryptionError(String),

    #[error("Internal server error: {0}")]
    InternalError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, code, message) = match &self {
            AppError::AuthenticationFailed(msg) => {
                (StatusCode::UNAUTHORIZED, "AUTH_FAILED", msg.clone())
            }
            AppError::AuthenticationError(msg) => {
                (StatusCode::UNAUTHORIZED, "AUTH_ERROR", msg.clone())
            }
            AppError::AuthorizationDenied(msg) => {
                (StatusCode::FORBIDDEN, "FORBIDDEN", msg.clone())
            }
            AppError::NotFound(msg) => {
                (StatusCode::NOT_FOUND, "NOT_FOUND", msg.clone())
            }
            AppError::ValidationError(msg) => {
                (StatusCode::BAD_REQUEST, "VALIDATION_ERROR", msg.clone())
            }
            AppError::HolochainError(msg) => {
                (StatusCode::BAD_GATEWAY, "HOLOCHAIN_ERROR", msg.clone())
            }
            AppError::DidNotFound(did) => {
                (StatusCode::NOT_FOUND, "DID_NOT_FOUND", format!("DID not registered: {}", did))
            }
            AppError::DidAlreadyExists(did) => {
                (StatusCode::CONFLICT, "DID_EXISTS", format!("DID already registered: {}", did))
            }
            AppError::TrustUnavailable(did) => {
                (StatusCode::SERVICE_UNAVAILABLE, "TRUST_UNAVAILABLE", format!("Trust score unavailable for: {}", did))
            }
            AppError::RateLimitExceeded => {
                (StatusCode::TOO_MANY_REQUESTS, "RATE_LIMIT", "Rate limit exceeded".to_string())
            }
            AppError::EncryptionError(msg) => {
                tracing::error!("Encryption error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, "ENCRYPTION_ERROR", "Encryption operation failed".to_string())
            }
            AppError::InternalError(msg) => {
                // Log internal errors but don't expose details
                tracing::error!("Internal error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", "An internal error occurred".to_string())
            }
            AppError::ConfigError(msg) => {
                tracing::error!("Configuration error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, "CONFIG_ERROR", "Server configuration error".to_string())
            }
        };

        let body = Json(ApiError::new(code, message));
        (status, body).into_response()
    }
}

/// Result type alias for handlers
pub type AppResult<T> = Result<T, AppError>;

/// Convert anyhow errors to AppError
impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::InternalError(err.to_string())
    }
}

/// Convert JWT errors
impl From<jsonwebtoken::errors::Error> for AppError {
    fn from(err: jsonwebtoken::errors::Error) -> Self {
        AppError::AuthenticationFailed(format!("Token error: {}", err))
    }
}

/// Convert AI service errors
impl From<crate::services::ai::AIError> for AppError {
    fn from(err: crate::services::ai::AIError) -> Self {
        AppError::InternalError(err.to_string())
    }
}

/// Convert trust graph errors
impl From<crate::services::trust_graph::TrustGraphError> for AppError {
    fn from(err: crate::services::trust_graph::TrustGraphError) -> Self {
        match err {
            crate::services::trust_graph::TrustGraphError::DidNotFound(did) => {
                AppError::NotFound(format!("Trust node not found: {}", did))
            }
            crate::services::trust_graph::TrustGraphError::NoPathExists => {
                AppError::NotFound("No trust path exists between DIDs".to_string())
            }
            crate::services::trust_graph::TrustGraphError::OperationFailed(msg) => {
                AppError::InternalError(msg)
            }
        }
    }
}

/// Convert claims service errors
impl From<crate::services::claims::ClaimsError> for AppError {
    fn from(err: crate::services::claims::ClaimsError) -> Self {
        AppError::InternalError(err.to_string())
    }
}
