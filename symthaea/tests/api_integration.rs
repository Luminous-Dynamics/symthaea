// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    Router,
    body::Body,
    http::{Request, StatusCode, header},
};
use tower::util::ServiceExt; // for `oneshot`

fn create_test_app() -> Router {
    symthaea::api::create_router()
}

fn create_authed_app() -> Router {
    symthaea::api::create_router_with_config(symthaea::api::ApiConfig {
        bearer_token: Some("test-token".to_string()),
        ..symthaea::api::ApiConfig::default()
    })
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
    assert!(
        json.get("total_submissions").is_some(),
        "Expected total_submissions field"
    );
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

    assert!(
        json.get("submission_id").is_some(),
        "Expected submission_id field"
    );
    assert!(
        json["status"] == "processing" || json["status"] == "completed",
        "Expected status to be processing or completed, got: {}",
        json["status"]
    );
}

#[tokio::test]
async fn test_private_submission_requires_auth_config() {
    let app: Router = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model_name":"private-model","topology_type":"ring","n_nodes":8,"public":false}"#,
                ))
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_private_submission_hidden_without_auth() {
    let app: Router = create_authed_app();

    let submit_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/submit")
                .header("content-type", "application/json")
                .header(header::AUTHORIZATION, "Bearer test-token")
                .body(Body::from(
                    r#"{"model_name":"private-model","topology_type":"ring","n_nodes":8,"public":false}"#,
                ))
                .expect("request builder"),
        )
        .await
        .expect("submit response");

    assert_eq!(submit_response.status(), StatusCode::ACCEPTED);
    let body = axum::body::to_bytes(submit_response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("parse JSON");
    let submission_id = json["submission_id"]
        .as_str()
        .expect("submission_id")
        .to_string();

    for _ in 0..20 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/results/{submission_id}"))
                    .body(Body::empty())
                    .expect("request builder"),
            )
            .await
            .expect("response");

        if response.status() == StatusCode::UNAUTHORIZED {
            break;
        }

        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    let unauthorized = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/results/{submission_id}"))
                .body(Body::empty())
                .expect("request builder"),
        )
        .await
        .expect("response");
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

    let authorized = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/results/{submission_id}"))
                .header(header::AUTHORIZATION, "Bearer test-token")
                .body(Body::empty())
                .expect("request builder"),
        )
        .await
        .expect("response");
    assert!(
        authorized.status() == StatusCode::OK || authorized.status() == StatusCode::ACCEPTED,
        "expected OK or ACCEPTED, got {}",
        authorized.status()
    );

    let leaderboard = app
        .oneshot(
            Request::builder()
                .uri("/v1/leaderboard")
                .body(Body::empty())
                .expect("request builder"),
        )
        .await
        .expect("response");
    assert_eq!(leaderboard.status(), StatusCode::OK);

    let body = axum::body::to_bytes(leaderboard.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("parse JSON");
    let entries = json["entries"].as_array().expect("entries array");
    assert!(
        !entries
            .iter()
            .any(|entry| entry["model_name"] == "private-model"),
        "private submission should not appear on leaderboard"
    );
}

#[tokio::test]
async fn test_submit_requires_auth_when_bearer_is_configured() {
    let app: Router = create_authed_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model_name":"unauthorized-model","topology_type":"ring","n_nodes":8}"#,
                ))
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_audit_events_requires_auth_when_bearer_is_configured() {
    let app: Router = create_authed_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audit/events")
                .body(Body::empty())
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_audit_events_return_recent_entries() {
    let app: Router = create_authed_app();

    let _ = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/submit")
                .header("content-type", "application/json")
                .header(header::AUTHORIZATION, "Bearer test-token")
                .body(Body::from(
                    r#"{"model_name":"audit-model","topology_type":"ring","n_nodes":8}"#,
                ))
                .expect("request builder"),
        )
        .await
        .expect("submit response");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audit/events?limit=10")
                .header(header::AUTHORIZATION, "Bearer test-token")
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
    let events = json["events"].as_array().expect("events");
    assert!(!events.is_empty(), "expected at least one audit event");
}

#[tokio::test]
async fn test_large_submission_is_processed_not_permanently_queued() {
    let app: Router = create_test_app();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model_name":"large-model","topology_type":"ring","n_nodes":17}"#,
                ))
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("parse JSON");
    let submission_id = json["submission_id"]
        .as_str()
        .expect("submission_id")
        .to_string();

    let mut saw_ok = false;
    for _ in 0..40 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/results/{submission_id}"))
                    .body(Body::empty())
                    .expect("request builder"),
            )
            .await
            .expect("response");

        if response.status() == StatusCode::OK {
            saw_ok = true;
            break;
        }

        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    assert!(saw_ok, "large submission should eventually complete");
}

#[tokio::test]
async fn test_dimensional_sweep_returns_not_implemented() {
    let app: Router = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/dimensional-sweep")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"topology_type":"ring","min_dimension":128,"max_dimension":256,"samples_per_dimension":4}"#))
                .expect("request builder"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
}