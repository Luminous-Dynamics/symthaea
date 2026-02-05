//! API integration tests
//!
//! Tests the API server endpoints using axum's router directly.
//! Requires the `api_module` feature.
//!
//! ```bash
//! cargo test --test api_integration --features api_module
//! ```

#![cfg(feature = "api_module")]

use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use tower::ServiceExt; // for `oneshot`

fn create_test_app() -> Router {
    symthaea::api::create_router()
}

#[tokio::test]
async fn test_health_check() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("parse JSON");

    assert_eq!(json["status"], "healthy");
    assert_eq!(json["service"], "symthaea-benchmark-api");
}

#[tokio::test]
async fn test_v1_health_check() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/health")
                .body(Body::empty())
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_metrics_prometheus() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);

    let content_type = response
        .headers()
        .get("content-type")
        .expect("content-type header")
        .to_str()
        .expect("header string");
    assert!(
        content_type.contains("text/plain"),
        "Expected text/plain content type, got: {content_type}"
    );

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let text = String::from_utf8(body.to_vec()).expect("UTF-8 body");

    // Prometheus format should contain HELP and TYPE lines
    assert!(
        text.contains("phi_calculations_total") || text.contains("api_requests_total"),
        "Expected Prometheus metric names in response"
    );
}

#[tokio::test]
async fn test_metrics_json() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/metrics")
                .body(Body::empty())
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("parse JSON");

    // JSON metrics should have timestamp and metrics fields
    assert!(json.get("timestamp").is_some(), "Expected timestamp field");
    assert!(json.get("metrics").is_some(), "Expected metrics field");
}

#[tokio::test]
async fn test_leaderboard() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/leaderboard")
                .body(Body::empty())
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("parse JSON");

    assert!(json.get("entries").is_some(), "Expected entries field");
    assert!(json.get("total_submissions").is_some(), "Expected total_submissions field");
}

#[tokio::test]
async fn test_topology_rankings() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/leaderboard/topologies")
                .body(Body::empty())
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("parse JSON");

    assert!(json.get("rankings").is_some(), "Expected rankings field");
}

#[tokio::test]
async fn test_datasets_list() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/datasets")
                .body(Body::empty())
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("parse JSON");

    assert!(json.get("datasets").is_some(), "Expected datasets field");
}

#[tokio::test]
async fn test_submit_with_valid_payload() {
    let app: Router = create_test_app();

    // POST with valid payload should succeed (202 Accepted)
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model_name":"test-model","topology_type":"ring","n_nodes":8}"#,
                ))
                .expect("request builder"),
        )
        .await
        .expect("response");

    // Should return 202 Accepted with submission_id
    assert_eq!(response.status(), StatusCode::ACCEPTED);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("parse JSON");

    assert!(json.get("submission_id").is_some(), "Expected submission_id field");
    assert!(
        json["status"] == "processing" || json["status"] == "completed",
        "Expected status to be processing or completed, got: {}",
        json["status"]
    );
}
