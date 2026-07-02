// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! External Integrations Module
//!
//! This module provides integrations for external financial services.
//! Each integration follows a common pattern:
//! - Configuration via environment variables
//! - Async API clients with retry logic
//! - Webhook handling for real-time events
//! - Error mapping to our internal types

pub mod client;
pub mod plaid;
pub mod stripe;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Common integration error type
#[derive(Error, Debug)]
pub enum IntegrationError {
    #[error("Configuration missing: {0}")]
    ConfigMissing(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Rate limited, retry after {0} seconds")]
    RateLimited(u64),

    #[error("Webhook verification failed")]
    WebhookVerificationFailed,

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Common webhook event wrapper
#[derive(Debug, Serialize, Deserialize)]
pub struct WebhookEvent<T> {
    pub event_id: String,
    pub event_type: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub data: T,
}

/// Integration status for health checks
#[derive(Debug, Serialize, Deserialize)]
pub struct IntegrationStatus {
    pub name: String,
    pub configured: bool,
    pub connected: bool,
    pub last_sync: Option<chrono::DateTime<chrono::Utc>>,
    pub error: Option<String>,
}

/// Check all integration statuses
pub async fn check_all_integrations() -> Vec<IntegrationStatus> {
    vec![
        plaid::check_status().await,
        stripe::check_status().await,
    ]
}
