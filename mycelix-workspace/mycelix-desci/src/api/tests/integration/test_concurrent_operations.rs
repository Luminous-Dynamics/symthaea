// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Concurrent operations integration tests
//!
//! Tests the API's ability to handle multiple simultaneous requests
//! and ensure data consistency under concurrent load.

use super::helpers::*;
use serde_json::Value;
use std::sync::Arc;
use tokio::task::JoinSet;

#[tokio::test]
async fn test_concurrent_claim_creation() {
    let server = Arc::new(TestServer::start().await.expect("Failed to start test server"));
    let num_claims = 10;

    // Create multiple claims concurrently
    let mut join_set = JoinSet::new();

    for i in 0..num_claims {
        let server = Arc::clone(&server);
        join_set.spawn(async move {
            let claim_request = sample_claim_request(
                "E0",
                "concurrent-test",
                &format!("user{}@test.com", i),
            );
            server.post("/claims", &claim_request).await
        });
    }

    // Wait for all requests to complete
    let mut successful = 0;
    while let Some(result) = join_set.join_next().await {
        let response = result.expect("Task panicked");
        if response.status() == 201 {
            successful += 1;
        }
    }

    assert_eq!(successful, num_claims, "All concurrent claims should be created");

    // Verify all claims exist via query
    let query_request = serde_json::json!({
        "category": "concurrent-test",
        "page": 1,
        "page_size": 20
    });

    let response = server.post("/query", &query_request).await;
    assert_eq!(response.status(), 200);

    let query_result: Value = response.json().await.expect("Failed to parse query result");
    assert_eq!(query_result["total_count"], num_claims);
}

#[tokio::test]
async fn test_concurrent_verifications_on_same_claim() {
    let server = Arc::new(TestServer::start().await.expect("Failed to start test server"));

    // Create a single claim
    let claim_request = sample_claim_request("E0", "concurrency", "creator@test.com");
    let response = server.post("/claims", &claim_request).await;
    assert_eq!(response.status(), 201);

    let claim: Value = response.json().await.expect("Failed to parse claim");
    let claim_id = claim["id"].as_str().expect("No claim ID").to_string();

    // Add multiple verifications concurrently
    let num_verifications = 5;
    let mut join_set = JoinSet::new();

    for i in 0..num_verifications {
        let server = Arc::clone(&server);
        let claim_id = claim_id.clone();
        join_set.spawn(async move {
            let verification = sample_verification_request(&format!("verifier{}@test.com", i));
            server.put(&format!("/claims/{}/verify", claim_id), &verification).await
        });
    }

    // Wait for all verifications to complete
    let mut successful = 0;
    while let Some(result) = join_set.join_next().await {
        let response = result.expect("Task panicked");
        if response.status().is_success() {
            successful += 1;
        }
    }

    assert_eq!(successful, num_verifications, "All verifications should succeed");

    // Verify final state - should be E4 with 5 verifications
    let response = server.get(&format!("/claims/{}", claim_id)).await;
    assert_eq!(response.status(), 200);

    let final_claim: Value = response.json().await.expect("Failed to parse final claim");
    assert_eq!(final_claim["tier"], "E4");
    assert_eq!(final_claim["verifications"].as_array().unwrap().len(), 5);
}

#[tokio::test]
async fn test_concurrent_queries() {
    let server = Arc::new(TestServer::start().await.expect("Failed to start test server"));

    // Create several claims first
    for i in 0..5 {
        let claim_request = sample_claim_request("E0", "query-test", &format!("user{}@test.com", i));
        server.post("/claims", &claim_request).await;
    }

    // Execute multiple queries concurrently
    let num_queries = 10;
    let mut join_set = JoinSet::new();

    for _ in 0..num_queries {
        let server = Arc::clone(&server);
        join_set.spawn(async move {
            let query_request = serde_json::json!({
                "category": "query-test",
                "page": 1,
                "page_size": 10
            });
            server.post("/query", &query_request).await
        });
    }

    // Wait for all queries to complete
    let mut successful = 0;
    while let Some(result) = join_set.join_next().await {
        let response = result.expect("Task panicked");
        if response.status() == 200 {
            successful += 1;
            let query_result: Value = response.json().await.expect("Failed to parse query result");
            assert_eq!(query_result["total_count"], 5, "All queries should return same count");
        }
    }

    assert_eq!(successful, num_queries, "All concurrent queries should succeed");
}

#[tokio::test]
async fn test_concurrent_trust_updates() {
    let server = Arc::new(TestServer::start().await.expect("Failed to start test server"));

    let participant = "test_participant@example.com";
    let num_updates = 10;

    // Perform multiple trust updates concurrently
    let mut join_set = JoinSet::new();

    for i in 0..num_updates {
        let server = Arc::clone(&server);
        let participant = participant.to_string();
        join_set.spawn(async move {
            let update_request = serde_json::json!({
                "positive": i % 2 == 0,
                "weight": 1.0
            });
            server.put(&format!("/trust/{}", participant), &update_request).await
        });
    }

    // Wait for all updates to complete
    let mut successful = 0;
    while let Some(result) = join_set.join_next().await {
        let response = result.expect("Task panicked");
        if response.status().is_success() {
            successful += 1;
        }
    }

    assert_eq!(successful, num_updates, "All trust updates should succeed");

    // Verify trust score exists
    let response = server.get(&format!("/trust/{}", participant)).await;
    assert_eq!(response.status(), 200);

    let trust_response: Value = response.json().await.expect("Failed to parse trust response");
    assert!(trust_response["score"].is_number(), "Should have a numeric trust score");
}

#[tokio::test]
async fn test_concurrent_mixed_operations() {
    let server = Arc::new(TestServer::start().await.expect("Failed to start test server"));

    let mut join_set = JoinSet::new();

    // Mix of different operations
    for i in 0..20 {
        let server = Arc::clone(&server);
        match i % 4 {
            0 => {
                // Create claim
                join_set.spawn(async move {
                    let claim_request = sample_claim_request("E0", "mixed", &format!("user{}@test.com", i));
                    server.post("/claims", &claim_request).await
                });
            }
            1 => {
                // Query
                join_set.spawn(async move {
                    let query_request = serde_json::json!({
                        "category": "mixed",
                        "page": 1,
                        "page_size": 10
                    });
                    server.post("/query", &query_request).await
                });
            }
            2 => {
                // Get stats
                join_set.spawn(async move {
                    server.get("/query/stats").await
                });
            }
            3 => {
                // Health check
                join_set.spawn(async move {
                    server.get("/system/health").await
                });
            }
            _ => unreachable!(),
        }
    }

    // Wait for all operations to complete
    let mut successful = 0;
    while let Some(result) = join_set.join_next().await {
        let response = result.expect("Task panicked");
        if response.status().is_success() {
            successful += 1;
        }
    }

    assert!(successful >= 15, "Most concurrent mixed operations should succeed");
}
