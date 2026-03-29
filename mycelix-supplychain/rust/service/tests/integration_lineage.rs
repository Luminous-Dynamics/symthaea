// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Integration tests for lineage query API endpoints
 *
 * Tests:
 * - GET /v1/batches/:batch_id/claims
 * - GET /v1/lineage/:batch_id
 * - GET /v1/claims (search with filters)
 *
 * Note: These tests verify endpoint availability and response structure.
 * Functional tests with actual data would require a different test pattern.
 */

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

mod common;
use common::*;

// ============================================================================
// GET /v1/batches/:batch_id/claims Tests
// ============================================================================

#[tokio::test]
async fn test_get_batch_claims_empty() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/batches/NONEXISTENT/claims")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["batch_id"], "NONEXISTENT");
    assert_eq!(result["total_claims"], 0);
    assert_eq!(result["claims"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_get_batch_claims_response_structure() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/batches/TEST-BATCH/claims")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    // Verify response structure
    assert!(result.get("batch_id").is_some());
    assert!(result.get("claims").is_some());
    assert!(result.get("total_claims").is_some());
    assert!(result["claims"].is_array());
}

// ============================================================================
// GET /v1/lineage/:batch_id Tests
// ============================================================================

#[tokio::test]
async fn test_get_lineage_nonexistent_batch() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/lineage/NONEXISTENT")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return 200 with empty lineage
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["batch_id"], "NONEXISTENT");
    assert_eq!(result["total_claims"], 0);
}

#[tokio::test]
async fn test_get_lineage_response_structure() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/lineage/TEST-BATCH")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    // Verify response structure
    assert!(result.get("batch_id").is_some());
    assert!(result.get("claims").is_some());
    assert!(result.get("total_claims").is_some());
    assert!(result.get("depth").is_some());
    assert!(result["claims"].is_array());
}

// ============================================================================
// GET /v1/claims (Search) Tests
// ============================================================================

#[tokio::test]
async fn test_search_claims_empty_results() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/claims?product_id=NONEXISTENT")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["total"], 0);
    assert_eq!(result["claims"].as_array().unwrap().len(), 0);
    assert_eq!(result["has_more"], false);
}

#[tokio::test]
async fn test_search_claims_default_pagination() {
    let app = create_test_app().await;

    // Search without limit/offset (should use defaults)
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/claims")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    // Should have default limit of 50
    assert_eq!(result["limit"], 50);
    assert_eq!(result["offset"], 0);
    assert!(result["claims"].is_array());
    assert!(result.get("total").is_some());
    assert!(result.get("has_more").is_some());
}

#[tokio::test]
async fn test_search_claims_custom_pagination() {
    let app = create_test_app().await;

    // Search with custom limit and offset
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/claims?limit=10&offset=5")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["limit"], 10);
    assert_eq!(result["offset"], 5);
}

#[tokio::test]
async fn test_search_claims_filter_by_product() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/claims?product_id=SKU-COFFEE")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    // Verify response structure
    assert!(result["claims"].is_array());
    assert!(result["total"].is_number());
    assert!(result["has_more"].is_boolean());
}

#[tokio::test]
async fn test_search_claims_filter_by_batch() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/claims?batch_id=BATCH-001")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    assert!(result["claims"].is_array());
    assert_eq!(result["limit"], 50);
    assert_eq!(result["offset"], 0);
}

#[tokio::test]
async fn test_search_claims_filter_by_event_type() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/claims?event_type=PRODUCED")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    assert!(result["claims"].is_array());
}

#[tokio::test]
async fn test_search_claims_combined_filters() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/claims?product_id=SKU-001&event_type=PRODUCED&limit=25")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    assert!(result["claims"].is_array());
    assert_eq!(result["limit"], 25);
}

#[tokio::test]
async fn test_search_claims_date_range_filter() {
    let app = create_test_app().await;

    // Test date range filtering
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/claims?from=2025-11-01T00:00:00Z&to=2025-11-30T23:59:59Z")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    assert!(result["claims"].is_array());
}

#[tokio::test]
async fn test_search_claims_facility_filter() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/claims?facility_id=FAC-001")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    assert!(result["claims"].is_array());
}

#[tokio::test]
async fn test_search_claims_max_limit() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/claims?limit=1000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    // Should accept limit up to 1000
    assert_eq!(result["limit"], 1000);
}

#[tokio::test]
async fn test_search_claims_response_structure() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/claims")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    // Verify all required fields are present
    assert!(result.get("claims").is_some());
    assert!(result.get("total").is_some());
    assert!(result.get("limit").is_some());
    assert!(result.get("offset").is_some());
    assert!(result.get("has_more").is_some());

    // Verify types
    assert!(result["claims"].is_array());
    assert!(result["total"].is_number());
    assert!(result["limit"].is_number());
    assert!(result["offset"].is_number());
    assert!(result["has_more"].is_boolean());
}
