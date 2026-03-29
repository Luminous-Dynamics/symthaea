// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Health check endpoints for Kubernetes and monitoring
//!
//! Provides liveness, readiness, and detailed health checks

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crate::AppState;

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Individual component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
}

/// Detailed health response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub version: String,
    pub uptime_seconds: u64,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub components: Option<Vec<ComponentHealth>>,
}

/// Liveness probe - basic "is the process running"
///
/// This endpoint should always return 200 OK unless the service is completely dead.
/// Kubernetes uses this to determine if the pod should be restarted.
///
/// GET /health/live
pub async fn liveness() -> Response {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "healthy",
            "message": "Service is alive"
        })),
    )
        .into_response()
}

/// Readiness probe - "can the service handle traffic"
///
/// This endpoint checks if the service is ready to accept requests.
/// Kubernetes uses this to determine if traffic should be routed to the pod.
///
/// Checks:
/// - Database connectivity (if configured)
/// - Essential dependencies
///
/// GET /health/ready
pub async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    let mut components = Vec::new();
    let mut overall_status = HealthStatus::Healthy;

    // Check database connectivity if available
    if let Some(ref db) = state.db {
        let start = std::time::Instant::now();
        match db.health_check().await {
            Ok(_) => {
                components.push(ComponentHealth {
                    name: "database".to_string(),
                    status: HealthStatus::Healthy,
                    message: Some("Database connection healthy".to_string()),
                    latency_ms: Some(start.elapsed().as_millis() as u64),
                });
            }
            Err(e) => {
                tracing::error!("Database health check failed: {}", e);
                components.push(ComponentHealth {
                    name: "database".to_string(),
                    status: HealthStatus::Unhealthy,
                    message: Some(format!("Database check failed: {}", e)),
                    latency_ms: Some(start.elapsed().as_millis() as u64),
                });
                overall_status = HealthStatus::Unhealthy;
            }
        }
    } else {
        // In-memory mode - always ready
        components.push(ComponentHealth {
            name: "storage".to_string(),
            status: HealthStatus::Healthy,
            message: Some("In-memory storage (no database configured)".to_string()),
            latency_ms: None,
        });
    }

    // Check cryptographic keypair
    let did = state.keypair.did();
    if did.starts_with("did:key:") {
        components.push(ComponentHealth {
            name: "crypto".to_string(),
            status: HealthStatus::Healthy,
            message: Some("Keypair initialized".to_string()),
            latency_ms: None,
        });
    } else {
        components.push(ComponentHealth {
            name: "crypto".to_string(),
            status: HealthStatus::Unhealthy,
            message: Some("Invalid keypair DID format".to_string()),
            latency_ms: None,
        });
        overall_status = HealthStatus::Unhealthy;
    }

    let status_code = match overall_status {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Degraded => StatusCode::OK, // Still accepting traffic
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };

    (
        status_code,
        Json(serde_json::json!({
            "status": overall_status,
            "components": components,
        })),
    )
        .into_response()
}

/// Detailed health endpoint with comprehensive system information
///
/// Provides detailed health information including:
/// - Overall status
/// - Service version
/// - Uptime
/// - Component health status
///
/// GET /health
pub async fn health(State(state): State<Arc<AppState>>) -> Response {
    let start_time = START_TIME.get().copied().unwrap_or_else(SystemTime::now);
    let uptime = SystemTime::now()
        .duration_since(start_time)
        .unwrap_or(Duration::from_secs(0));

    let mut components = Vec::new();
    let mut overall_status = HealthStatus::Healthy;

    // Check database
    if let Some(ref db) = state.db {
        let db_start = std::time::Instant::now();
        match db.health_check().await {
            Ok(_) => {
                components.push(ComponentHealth {
                    name: "database".to_string(),
                    status: HealthStatus::Healthy,
                    message: Some("SQLite connection healthy".to_string()),
                    latency_ms: Some(db_start.elapsed().as_millis() as u64),
                });
            }
            Err(e) => {
                tracing::error!("Database health check failed: {}", e);
                components.push(ComponentHealth {
                    name: "database".to_string(),
                    status: HealthStatus::Unhealthy,
                    message: Some(format!("Database error: {}", e)),
                    latency_ms: Some(db_start.elapsed().as_millis() as u64),
                });
                overall_status = HealthStatus::Unhealthy;
            }
        }
    } else {
        components.push(ComponentHealth {
            name: "storage".to_string(),
            status: HealthStatus::Healthy,
            message: Some("In-memory storage (no database)".to_string()),
            latency_ms: None,
        });
    }

    // Check keypair
    let did = state.keypair.did();
    components.push(ComponentHealth {
        name: "crypto".to_string(),
        status: if did.starts_with("did:key:") {
            HealthStatus::Healthy
        } else {
            overall_status = HealthStatus::Unhealthy;
            HealthStatus::Unhealthy
        },
        message: Some(format!("Service DID: {}", did)),
        latency_ms: None,
    });

    // Check in-memory claims storage
    let claims_count = state.claims.read().await.len();
    components.push(ComponentHealth {
        name: "claims_cache".to_string(),
        status: HealthStatus::Healthy,
        message: Some(format!("{} claims in memory cache", claims_count)),
        latency_ms: None,
    });

    let response = HealthResponse {
        status: overall_status,
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: uptime.as_secs(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        components: Some(components),
    };

    let status_code = match response.status {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Degraded => StatusCode::OK,
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };

    (status_code, Json(response)).into_response()
}

/// Service start time for uptime calculation
static START_TIME: once_cell::sync::OnceCell<SystemTime> = once_cell::sync::OnceCell::new();

/// Initialize health tracking
pub fn init() {
    START_TIME.get_or_init(SystemTime::now);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_serialization() {
        let status = HealthStatus::Healthy;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"healthy\"");

        let status = HealthStatus::Degraded;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"degraded\"");

        let status = HealthStatus::Unhealthy;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"unhealthy\"");
    }

    #[test]
    fn test_component_health_creation() {
        let component = ComponentHealth {
            name: "database".to_string(),
            status: HealthStatus::Healthy,
            message: Some("OK".to_string()),
            latency_ms: Some(5),
        };

        assert_eq!(component.name, "database");
        assert_eq!(component.status, HealthStatus::Healthy);
        assert_eq!(component.message, Some("OK".to_string()));
        assert_eq!(component.latency_ms, Some(5));
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: HealthStatus::Healthy,
            version: "0.1.0".to_string(),
            uptime_seconds: 3600,
            timestamp: "2025-11-16T12:00:00Z".to_string(),
            components: None,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "healthy");
        assert_eq!(json["version"], "0.1.0");
        assert_eq!(json["uptime_seconds"], 3600);
    }

    #[test]
    fn test_init_sets_start_time() {
        init();
        assert!(START_TIME.get().is_some());
    }
}
