// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for batch event ingestion API
//!
//! Tests the POST /v1/events/batch endpoint with various scenarios

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use claim_model::{CredentialSubject, EventType, Facility, SupplyEventVC};
use http_body_util::BodyExt; // for `collect`
use serde_json::json;
use tower::ServiceExt; // for `oneshot`

mod common;
use common::*;

/// Helper to create a valid test event
fn create_test_event(batch_id: &str, product_id: &str, event_type: EventType) -> SupplyEventVC {
    SupplyEventVC {
        context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
        vc_type: vec!["VerifiableCredential".to_string()],
        issuer: "did:mycelix:org:test".to_string(),
        issuance_date: chrono::Utc::now(),
        expiration_date: None,
        credential_subject: CredentialSubject {
            event_type,
            product_id: product_id.to_string(),
            batch_id: batch_id.to_string(),
            quantity: 100.0,
            unit: "kg".to_string(),
            facility: Facility {
                id: "TEST-FACILITY".to_string(),
                name: "Test Facility".to_string(),
                location: None,
            },
            timestamp: chrono::Utc::now(),
            shipment: None,
            certification: None,
            metadata: None,
            prev_batch_ids: None,
        },
        proof: None,
    }
}

#[tokio::test]
async fn test_batch_best_effort_mode_all_success() {
    let app = create_test_app().await;

    // Create 10 valid events
    let events: Vec<SupplyEventVC> = (0..10)
        .map(|i| {
            create_test_event(
                &format!("BATCH-{:03}", i),
                "PRODUCT-TEST",
                EventType::Produced,
            )
        })
        .collect();

    let request_body = json!({
        "events": events,
        "mode": "best-effort"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/events/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["total"], 10);
    assert_eq!(result["succeeded"], 10);
    assert_eq!(result["failed"], 0);
    assert!(result["duration_ms"].as_u64().unwrap() > 0);

    // Verify all results are success
    let results = result["results"].as_array().unwrap();
    assert_eq!(results.len(), 10);

    for (i, result) in results.iter().enumerate() {
        assert_eq!(result["index"], i);
        assert_eq!(result["status"], "success");
        assert!(result["claim_id"].is_string());
        assert!(result["lineage_hash"].is_string());
        assert!(result["error"].is_null());
    }
}

#[tokio::test]
async fn test_batch_best_effort_mode_partial_success() {
    let app = create_test_app().await;

    // Create 5 valid events and 3 invalid events (missing required fields)
    let mut events = vec![];

    // Valid events
    for i in 0..5 {
        events.push(create_test_event(
            &format!("BATCH-{:03}", i),
            "PRODUCT-TEST",
            EventType::Produced,
        ));
    }

    // Invalid events (will fail validation)
    for i in 5..8 {
        let mut invalid_event = create_test_event(
            &format!("BATCH-{:03}", i),
            "PRODUCT-TEST",
            EventType::Produced,
        );
        // Make it invalid by setting negative quantity (will fail validation)
        invalid_event.credential_subject.quantity = -100.0;
        events.push(invalid_event);
    }

    let request_body = json!({
        "events": events,
        "mode": "best-effort"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/events/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["total"], 8);
    assert_eq!(result["succeeded"], 5);
    assert_eq!(result["failed"], 3);

    // Verify first 5 are success, last 3 are failures
    let results = result["results"].as_array().unwrap();
    for i in 0..5 {
        assert_eq!(results[i]["status"], "success");
        assert!(results[i]["claim_id"].is_string());
    }

    for i in 5..8 {
        assert_eq!(results[i]["status"], "failed");
        assert!(results[i]["error"].is_string());
        assert!(results[i]["claim_id"].is_null());
    }
}

#[tokio::test]
async fn test_batch_atomic_mode_all_success() {
    let app = create_test_app().await;

    // Create 5 valid events
    let events: Vec<SupplyEventVC> = (0..5)
        .map(|i| {
            create_test_event(
                &format!("BATCH-ATOMIC-{:03}", i),
                "PRODUCT-TEST",
                EventType::Produced,
            )
        })
        .collect();

    let request_body = json!({
        "events": events,
        "mode": "atomic"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/events/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["total"], 5);
    assert_eq!(result["succeeded"], 5);
    assert_eq!(result["failed"], 0);
}

#[tokio::test]
async fn test_batch_atomic_mode_failure() {
    let app = create_test_app().await;

    // Create 3 valid events + 1 invalid event
    let mut events = vec![];

    for i in 0..3 {
        events.push(create_test_event(
            &format!("BATCH-{:03}", i),
            "PRODUCT-TEST",
            EventType::Produced,
        ));
    }

    // Add invalid event
    let mut invalid_event = create_test_event("BATCH-003", "PRODUCT-TEST", EventType::Produced);
    invalid_event.credential_subject.quantity = -100.0; // Negative quantity will fail validation
    events.push(invalid_event);

    let request_body = json!({
        "events": events,
        "mode": "atomic"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/events/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Atomic mode should return error when any event fails
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Should have error message mentioning the failed event index
    let error_msg = result["error"].as_str().unwrap();
    assert!(error_msg.contains("index 3") || error_msg.contains("failed"));
}

#[tokio::test]
async fn test_batch_max_size_exceeded() {
    let app = create_test_app().await;

    // Create 101 events (exceeds MAX_BATCH_SIZE of 100)
    let events: Vec<SupplyEventVC> = (0..101)
        .map(|i| {
            create_test_event(
                &format!("BATCH-{:03}", i),
                "PRODUCT-TEST",
                EventType::Produced,
            )
        })
        .collect();

    let request_body = json!({
        "events": events,
        "mode": "best-effort"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/events/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

    let error_msg = result["error"].as_str().unwrap();
    assert!(error_msg.contains("101"));
    assert!(error_msg.contains("100") || error_msg.contains("maximum"));
}

#[tokio::test]
async fn test_batch_empty_array() {
    let app = create_test_app().await;

    let request_body = json!({
        "events": [],
        "mode": "best-effort"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/events/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

    let error_msg = result["error"].as_str().unwrap();
    assert!(error_msg.contains("empty") || error_msg.contains("cannot be empty"));
}

#[tokio::test]
async fn test_batch_invalid_mode() {
    let app = create_test_app().await;

    let events = vec![create_test_event("BATCH-001", "PRODUCT-TEST", EventType::Produced)];

    let request_body = json!({
        "events": events,
        "mode": "invalid-mode"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/events/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

    let error_msg = result["error"].as_str().unwrap();
    assert!(error_msg.contains("invalid-mode") || error_msg.contains("Invalid mode"));
}

#[tokio::test]
async fn test_batch_performance() {
    let app = create_test_app().await;

    // Create 50 events
    let events: Vec<SupplyEventVC> = (0..50)
        .map(|i| {
            create_test_event(
                &format!("BATCH-PERF-{:03}", i),
                "PRODUCT-TEST",
                EventType::Produced,
            )
        })
        .collect();

    let request_body = json!({
        "events": events,
        "mode": "best-effort"
    });

    let start = std::time::Instant::now();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/events/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let elapsed = start.elapsed();

    assert_eq!(response.status(), StatusCode::CREATED);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["total"], 50);
    assert_eq!(result["succeeded"], 50);

    let duration_ms = result["duration_ms"].as_u64().unwrap();
    println!("Batch of 50 events processed in {}ms", duration_ms);
    println!("Actual wall-clock time: {}ms", elapsed.as_millis());
    println!(
        "Throughput: {:.2} events/second",
        50000.0 / duration_ms as f64
    );

    // Performance target: <500ms for 50 events
    assert!(
        duration_ms < 500,
        "Batch processing took {}ms, expected <500ms",
        duration_ms
    );
}

#[tokio::test]
async fn test_batch_lineage_resolution() {
    let app = create_test_app().await;

    // Create batch of events with lineage relationships
    let mut events = vec![];

    // Create 3 PRODUCED events
    for i in 0..3 {
        events.push(create_test_event(
            &format!("BATCH-LINEAGE-{:03}", i),
            "COFFEE",
            EventType::Produced,
        ));
    }

    let batch_request = json!({
        "events": events,
        "mode": "best-effort"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/events/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&batch_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["succeeded"], 3);
    assert_eq!(result["failed"], 0);

    // Verify all events have lineage_hash (indicating lineage calculation worked)
    let results = result["results"].as_array().unwrap();
    for result in results {
        assert_eq!(result["status"], "success");
        assert!(result["lineage_hash"].is_string());
        let lineage_hash = result["lineage_hash"].as_str().unwrap();
        assert!(!lineage_hash.is_empty(), "Lineage hash should not be empty");
    }
}

#[tokio::test]
async fn test_batch_default_mode() {
    let app = create_test_app().await;

    let events = vec![create_test_event("BATCH-001", "PRODUCT-TEST", EventType::Produced)];

    // Don't specify mode - should default to "best-effort"
    let request_body = json!({
        "events": events
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/events/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["total"], 1);
    assert_eq!(result["succeeded"], 1);
}
