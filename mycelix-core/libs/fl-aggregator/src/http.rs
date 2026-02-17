//! HTTP API for federated learning aggregator.
//!
//! Provides REST endpoints for gradient submission and aggregation control.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::trace::TraceLayer;

use crate::aggregator::{AggregatorStatus, AsyncAggregator};
use crate::metrics::AggregatorMetrics;
use crate::Round;

/// Application state shared across handlers.
pub struct AppState {
    pub aggregator: AsyncAggregator,
}

/// Create the HTTP router.
///
/// Note: No CORS layer is configured since FL clients are typically
/// backend services, not browsers. For browser-based dashboards,
/// consider adding CORS or using a separate gateway.
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Health and status
        .route("/health", get(health))
        .route("/status", get(get_status))
        .route("/metrics", get(get_metrics))
        .route("/metrics/prometheus", get(get_prometheus_metrics))
        // Node management
        .route("/nodes/:node_id", post(register_node))
        .route("/nodes/:node_id", axum::routing::delete(unregister_node))
        // Gradient operations
        .route("/gradients", post(submit_gradient))
        .route("/aggregate", post(trigger_aggregation))
        .route("/result", get(get_result))
        // Round management
        .route("/round", get(get_current_round))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
}

/// Health check response.
#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
}

/// Health check endpoint.
async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy",
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// Get aggregator status.
async fn get_status(State(state): State<Arc<AppState>>) -> Json<AggregatorStatus> {
    Json(state.aggregator.status().await)
}

/// Get aggregator metrics.
async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<AggregatorMetrics> {
    Json(state.aggregator.metrics().await)
}

/// Get Prometheus-format metrics.
async fn get_prometheus_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let metrics = state.aggregator.metrics().await;
    (
        StatusCode::OK,
        [("content-type", "text/plain; charset=utf-8")],
        metrics.to_prometheus(),
    )
}

/// Register a node.
async fn register_node(
    State(state): State<Arc<AppState>>,
    Path(node_id): Path<String>,
) -> impl IntoResponse {
    match state.aggregator.register_node(&node_id).await {
        Ok(_) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "status": "registered",
                "node_id": node_id
            })),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": e.to_string()
            })),
        ),
    }
}

/// Unregister a node.
async fn unregister_node(
    State(state): State<Arc<AppState>>,
    Path(node_id): Path<String>,
) -> impl IntoResponse {
    match state.aggregator.unregister_node(&node_id).await {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "unregistered",
                "node_id": node_id
            })),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": e.to_string()
            })),
        ),
    }
}

/// Maximum allowed gradient size (10 million elements)
const MAX_GRADIENT_SIZE: usize = 10_000_000;

/// Maximum allowed node ID length
const MAX_NODE_ID_LENGTH: usize = 256;

/// Gradient submission request.
#[derive(Deserialize)]
struct GradientRequest {
    node_id: String,
    gradient: Vec<f32>,
    #[serde(default)]
    signature: Option<String>,
}

impl GradientRequest {
    /// Validate the gradient request for security and data integrity.
    fn validate(&self) -> Result<(), String> {
        // Validate node_id
        if self.node_id.is_empty() {
            return Err("node_id cannot be empty".into());
        }
        if self.node_id.len() > MAX_NODE_ID_LENGTH {
            return Err(format!(
                "node_id too long: {} bytes (max {})",
                self.node_id.len(),
                MAX_NODE_ID_LENGTH
            ));
        }

        // Validate gradient
        if self.gradient.is_empty() {
            return Err("gradient cannot be empty".into());
        }
        if self.gradient.len() > MAX_GRADIENT_SIZE {
            return Err(format!(
                "gradient too large: {} elements (max {})",
                self.gradient.len(),
                MAX_GRADIENT_SIZE
            ));
        }

        // Check for non-finite values (NaN, Infinity)
        for (i, &val) in self.gradient.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!(
                    "gradient contains non-finite value at index {}: {}",
                    i, val
                ));
            }
        }

        Ok(())
    }
}

/// Gradient submission response.
#[derive(Serialize)]
struct GradientResponse {
    status: String,
    round: Round,
    submitted: usize,
    expected: usize,
    is_complete: bool,
}

/// Submit a gradient.
async fn submit_gradient(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GradientRequest>,
) -> impl IntoResponse {
    // Validate request before processing (security: prevent NaN/Inf, DoS)
    if let Err(validation_error) = req.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(GradientResponse {
                status: format!("validation error: {}", validation_error),
                round: 0,
                submitted: 0,
                expected: 0,
                is_complete: false,
            }),
        );
    }

    let gradient = ndarray::Array1::from(req.gradient);

    match state.aggregator.submit(&req.node_id, gradient).await {
        Ok(_) => {
            let status = state.aggregator.status().await;
            (
                StatusCode::ACCEPTED,
                Json(GradientResponse {
                    status: "accepted".to_string(),
                    round: status.round,
                    submitted: status.submitted_nodes,
                    expected: status.expected_nodes,
                    is_complete: status.is_complete,
                }),
            )
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(GradientResponse {
                status: format!("error: {}", e),
                round: 0,
                submitted: 0,
                expected: 0,
                is_complete: false,
            }),
        ),
    }
}

/// Aggregation response.
#[derive(Serialize)]
struct AggregationResponse {
    status: String,
    round: Round,
    gradient: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Trigger aggregation (force finalize).
async fn trigger_aggregation(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.aggregator.force_finalize().await {
        Ok(gradient) => {
            let status = state.aggregator.status().await;
            (
                StatusCode::OK,
                Json(AggregationResponse {
                    status: "aggregated".to_string(),
                    round: status.round.saturating_sub(1),
                    gradient: Some(gradient.to_vec()),
                    error: None,
                }),
            )
        }
        Err(e) => {
            let status = state.aggregator.status().await;
            (
                StatusCode::BAD_REQUEST,
                Json(AggregationResponse {
                    status: "error".to_string(),
                    round: status.round,
                    gradient: None,
                    error: Some(e.to_string()),
                }),
            )
        }
    }
}

/// Get aggregated result if round is complete.
async fn get_result(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.aggregator.get_aggregated_gradient().await {
        Ok(Some(gradient)) => {
            let status = state.aggregator.status().await;
            (
                StatusCode::OK,
                Json(AggregationResponse {
                    status: "complete".to_string(),
                    round: status.round.saturating_sub(1),
                    gradient: Some(gradient.to_vec()),
                    error: None,
                }),
            )
        }
        Ok(None) => {
            let status = state.aggregator.status().await;
            (
                StatusCode::ACCEPTED,
                Json(AggregationResponse {
                    status: "pending".to_string(),
                    round: status.round,
                    gradient: None,
                    error: None,
                }),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(AggregationResponse {
                status: "error".to_string(),
                round: 0,
                gradient: None,
                error: Some(e.to_string()),
            }),
        ),
    }
}

/// Round response.
#[derive(Serialize)]
struct RoundResponse {
    round: Round,
    submitted: usize,
    expected: usize,
    is_complete: bool,
}

/// Get current round info.
async fn get_current_round(State(state): State<Arc<AppState>>) -> Json<RoundResponse> {
    let status = state.aggregator.status().await;
    Json(RoundResponse {
        round: status.round,
        submitted: status.submitted_nodes,
        expected: status.expected_nodes,
        is_complete: status.is_complete,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;
    use crate::byzantine::Defense;

    fn create_test_app() -> Router {
        let config = crate::aggregator::AggregatorConfig::default()
            .with_expected_nodes(2)
            .with_defense(Defense::FedAvg);

        let state = Arc::new(AppState {
            aggregator: AsyncAggregator::new(config),
        });

        create_router(state)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_test_app();

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_status_endpoint() {
        let app = create_test_app();

        let response = app
            .oneshot(Request::builder().uri("/status").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
