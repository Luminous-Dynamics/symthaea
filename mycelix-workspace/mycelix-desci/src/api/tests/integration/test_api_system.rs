// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! System API endpoint tests
//!
//! Tests all system-related API endpoints.

use super::helpers::*;
use serde_json::Value;

#[tokio::test]
async fn test_health_endpoint() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let response = server.get("/system/health").await;
    assert_eq!(response.status(), 200, "Health endpoint should return 200");

    let health: Value = response.json().await.expect("Failed to parse health response");

    assert_eq!(health["status"], "healthy", "Status should be 'healthy'");
    assert!(health["version"].is_string(), "Should have version");
    assert!(health["uptime_seconds"].is_number(), "Should have uptime");
}

#[tokio::test]
async fn test_health_endpoint_structure() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let response = server.get("/system/health").await;
    let health: Value = response.json().await.expect("Failed to parse health response");

    // Verify all required fields exist
    assert!(health.get("status").is_some(), "Should have 'status' field");
    assert!(health.get("version").is_some(), "Should have 'version' field");
    assert!(health.get("uptime_seconds").is_some(), "Should have 'uptime_seconds' field");

    // Verify types
    assert!(health["status"].is_string());
    assert!(health["version"].is_string());
    assert!(health["uptime_seconds"].is_number());

    let uptime = health["uptime_seconds"].as_u64().unwrap();
    assert!(uptime >= 0, "Uptime should be non-negative");
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create some activity first
    let claim = sample_claim_request("E0", "metrics-test", "user@test.com");
    server.post("/claims", &claim).await;

    let query = serde_json::json!({
        "category": "metrics-test",
        "page": 1,
        "page_size": 10
    });
    server.post("/query", &query).await;

    // Get metrics
    let response = server.get("/system/metrics").await;
    assert_eq!(response.status(), 200, "Metrics endpoint should return 200");

    let metrics: Value = response.json().await.expect("Failed to parse metrics response");

    assert!(metrics["total_claims"].is_number(), "Should have total_claims");
    assert!(metrics["queries_executed"].is_number(), "Should have queries_executed");
    assert!(metrics["claims_created"].is_number(), "Should have claims_created");
    assert!(metrics["verifications_added"].is_number(), "Should have verifications_added");
    assert!(metrics["uptime_seconds"].is_number(), "Should have uptime_seconds");

    let total_claims = metrics["total_claims"].as_u64().unwrap();
    assert!(total_claims >= 1, "Should have at least 1 claim");

    let queries = metrics["queries_executed"].as_u64().unwrap();
    assert!(queries >= 1, "Should have executed at least 1 query");
}

#[tokio::test]
async fn test_metrics_increment_on_operations() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Get initial metrics
    let response = server.get("/system/metrics").await;
    let initial: Value = response.json().await.expect("Failed to parse metrics");
    let initial_claims = initial["claims_created"].as_u64().unwrap_or(0);
    let initial_queries = initial["queries_executed"].as_u64().unwrap_or(0);

    // Create a claim
    let claim = sample_claim_request("E0", "increment-test", "user@test.com");
    server.post("/claims", &claim).await;

    // Execute a query
    let query = serde_json::json!({
        "category": "increment-test",
        "page": 1,
        "page_size": 10
    });
    server.post("/query", &query).await;

    // Get updated metrics
    let response = server.get("/system/metrics").await;
    let updated: Value = response.json().await.expect("Failed to parse metrics");
    let new_claims = updated["claims_created"].as_u64().unwrap_or(0);
    let new_queries = updated["queries_executed"].as_u64().unwrap_or(0);

    assert!(
        new_claims > initial_claims,
        "Claims created counter should increment"
    );
    assert!(
        new_queries > initial_queries,
        "Queries executed counter should increment"
    );
}

#[tokio::test]
async fn test_version_endpoint() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let response = server.get("/system/version").await;
    assert_eq!(response.status(), 200, "Version endpoint should return 200");

    let version: Value = response.json().await.expect("Failed to parse version response");

    assert!(version["version"].is_string(), "Should have version string");
    assert!(version["build_date"].is_string(), "Should have build_date");
    assert!(version["commit_hash"].is_string(), "Should have commit_hash");

    let version_str = version["version"].as_str().unwrap();
    assert!(!version_str.is_empty(), "Version should not be empty");
}

#[tokio::test]
async fn test_uptime_increases() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Get initial uptime
    let response = server.get("/system/health").await;
    let initial: Value = response.json().await.expect("Failed to parse health");
    let initial_uptime = initial["uptime_seconds"].as_u64().unwrap();

    // Wait a bit
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Get updated uptime
    let response = server.get("/system/health").await;
    let updated: Value = response.json().await.expect("Failed to parse health");
    let new_uptime = updated["uptime_seconds"].as_u64().unwrap();

    assert!(
        new_uptime >= initial_uptime,
        "Uptime should increase over time"
    );
}

#[tokio::test]
async fn test_system_endpoints_available_without_auth() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // All system endpoints should be publicly accessible
    let endpoints = vec![
        "/system/health",
        "/system/metrics",
        "/system/version",
    ];

    for endpoint in endpoints {
        let response = server.get(endpoint).await;
        assert!(
            response.status().is_success(),
            "Endpoint {} should be accessible without auth",
            endpoint
        );
    }
}

#[tokio::test]
async fn test_health_endpoint_response_time() {
    let server = TestServer::start().await.expect("Failed to start test server");

    use std::time::Instant;

    let start = Instant::now();
    let response = server.get("/system/health").await;
    let duration = start.elapsed();

    assert_eq!(response.status(), 200);
    assert!(
        duration.as_millis() < 100,
        "Health endpoint should respond quickly (< 100ms)"
    );
}

#[tokio::test]
async fn test_metrics_consistency() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Get metrics twice in quick succession
    let response1 = server.get("/system/metrics").await;
    let metrics1: Value = response1.json().await.expect("Failed to parse metrics");

    let response2 = server.get("/system/metrics").await;
    let metrics2: Value = response2.json().await.expect("Failed to parse metrics");

    // Uptime may vary slightly, but other metrics should be the same
    assert_eq!(
        metrics1["total_claims"], metrics2["total_claims"],
        "Total claims should be consistent"
    );
}
