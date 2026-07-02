// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Query API endpoint tests
//!
//! Tests all query-related API endpoints.

use super::helpers::*;
use serde_json::Value;

#[tokio::test]
async fn test_query_claims_by_category() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create claims in different categories
    for i in 0..3 {
        let claim = sample_claim_request("E0", "longevity", &format!("user{}@test.com", i));
        server.post("/claims", &claim).await;
    }

    for i in 0..2 {
        let claim = sample_claim_request("E0", "physics", &format!("user{}@test.com", i + 3));
        server.post("/claims", &claim).await;
    }

    // Query for longevity claims
    let query_request = serde_json::json!({
        "category": "longevity",
        "page": 1,
        "page_size": 10
    });

    let response = server.post("/query", &query_request).await;
    assert_eq!(response.status(), 200);

    let result: Value = response.json().await.expect("Failed to parse query result");
    assert_eq!(result["total_count"], 3, "Should find 3 longevity claims");
    assert_eq!(result["page"], 1);
    assert!(result["claims"].as_array().unwrap().len() == 3);
}

#[tokio::test]
async fn test_query_claims_with_pagination() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create 15 claims
    for i in 0..15 {
        let claim = sample_claim_request("E0", "pagination-test", &format!("user{}@test.com", i));
        server.post("/claims", &claim).await;
    }

    // Query first page (5 items)
    let query_request = serde_json::json!({
        "category": "pagination-test",
        "page": 1,
        "page_size": 5
    });

    let response = server.post("/query", &query_request).await;
    assert_eq!(response.status(), 200);

    let result: Value = response.json().await.expect("Failed to parse query result");
    assert_eq!(result["total_count"], 15);
    assert_eq!(result["page"], 1);
    assert_eq!(result["total_pages"], 3);
    assert_eq!(result["claims"].as_array().unwrap().len(), 5);

    // Query second page
    let query_request = serde_json::json!({
        "category": "pagination-test",
        "page": 2,
        "page_size": 5
    });

    let response = server.post("/query", &query_request).await;
    let result: Value = response.json().await.expect("Failed to parse query result");
    assert_eq!(result["page"], 2);
    assert_eq!(result["claims"].as_array().unwrap().len(), 5);

    // Query third page (last 5 items)
    let query_request = serde_json::json!({
        "category": "pagination-test",
        "page": 3,
        "page_size": 5
    });

    let response = server.post("/query", &query_request).await;
    let result: Value = response.json().await.expect("Failed to parse query result");
    assert_eq!(result["page"], 3);
    assert_eq!(result["claims"].as_array().unwrap().len(), 5);
}

#[tokio::test]
async fn test_query_claims_by_tier() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create E0 claim
    let claim = sample_claim_request("E0", "tier-test", "user0@test.com");
    let response = server.post("/claims", &claim).await;
    let claim0: Value = response.json().await.expect("Failed to parse claim");
    let claim0_id = claim0["id"].as_str().unwrap();

    // Create another E0 claim and upgrade it to E1
    let claim = sample_claim_request("E0", "tier-test", "user1@test.com");
    let response = server.post("/claims", &claim).await;
    let claim1: Value = response.json().await.expect("Failed to parse claim");
    let claim1_id = claim1["id"].as_str().unwrap();

    // Verify to upgrade to E1
    let verification = sample_verification_request("verifier@test.com");
    server.put(&format!("/claims/{}/verify", claim1_id), &verification).await;

    // Query for E0 claims
    let query_request = serde_json::json!({
        "category": "tier-test",
        "tier": "E0",
        "page": 1,
        "page_size": 10
    });

    let response = server.post("/query", &query_request).await;
    assert_eq!(response.status(), 200);

    let result: Value = response.json().await.expect("Failed to parse query result");
    assert_eq!(result["total_count"], 1, "Should find only 1 E0 claim");

    // Query for E1 claims
    let query_request = serde_json::json!({
        "category": "tier-test",
        "tier": "E1",
        "page": 1,
        "page_size": 10
    });

    let response = server.post("/query", &query_request).await;
    let result: Value = response.json().await.expect("Failed to parse query result");
    assert_eq!(result["total_count"], 1, "Should find only 1 E1 claim");
}

#[tokio::test]
async fn test_query_claims_by_creator() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let creator = "specific.researcher@uni.edu";

    // Create claims by specific creator
    for i in 0..3 {
        let claim = sample_claim_request("E0", &format!("category{}", i), creator);
        server.post("/claims", &claim).await;
    }

    // Create claims by other creators
    for i in 0..2 {
        let claim = sample_claim_request("E0", "other-category", &format!("other{}@test.com", i));
        server.post("/claims", &claim).await;
    }

    // Query by creator
    let query_request = serde_json::json!({
        "creator": creator,
        "page": 1,
        "page_size": 10
    });

    let response = server.post("/query", &query_request).await;
    assert_eq!(response.status(), 200);

    let result: Value = response.json().await.expect("Failed to parse query result");
    assert_eq!(result["total_count"], 3, "Should find 3 claims by specific creator");

    let claims = result["claims"].as_array().unwrap();
    for claim in claims {
        assert_eq!(claim["creator"], creator);
    }
}

#[tokio::test]
async fn test_get_categories_endpoint() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create claims in different categories
    let categories = vec!["biology", "physics", "chemistry"];
    for (i, category) in categories.iter().enumerate() {
        let claim = sample_claim_request("E0", category, &format!("user{}@test.com", i));
        server.post("/claims", &claim).await;
    }

    // Get categories
    let response = server.get("/query/categories").await;
    assert_eq!(response.status(), 200);

    let result: Value = response.json().await.expect("Failed to parse categories response");
    let returned_categories = result["categories"].as_array().expect("No categories array");

    assert!(returned_categories.len() >= 3, "Should have at least 3 categories");

    for category in &categories {
        assert!(
            returned_categories.iter().any(|c| c.as_str() == Some(category)),
            "Should include category: {}",
            category
        );
    }
}

#[tokio::test]
async fn test_get_stats_endpoint() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create some claims
    for i in 0..5 {
        let claim = sample_claim_request("E0", "stats-test", &format!("user{}@test.com", i));
        server.post("/claims", &claim).await;
    }

    // Get stats
    let response = server.get("/query/stats").await;
    assert_eq!(response.status(), 200);

    let stats: Value = response.json().await.expect("Failed to parse stats response");

    assert!(stats["total_claims"].is_number(), "Should have total_claims");
    assert!(stats["claims_by_tier"].is_object(), "Should have claims_by_tier");
    assert!(stats["claims_by_category"].is_object(), "Should have claims_by_category");

    let total = stats["total_claims"].as_u64().unwrap();
    assert!(total >= 5, "Should have at least 5 claims");
}

#[tokio::test]
async fn test_query_with_empty_results() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Query for non-existent category
    let query_request = serde_json::json!({
        "category": "nonexistent-category-xyz",
        "page": 1,
        "page_size": 10
    });

    let response = server.post("/query", &query_request).await;
    assert_eq!(response.status(), 200);

    let result: Value = response.json().await.expect("Failed to parse query result");
    assert_eq!(result["total_count"], 0);
    assert_eq!(result["claims"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_query_with_sorting() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create claims with slight delays to ensure different timestamps
    for i in 0..3 {
        let claim = sample_claim_request("E0", "sort-test", &format!("user{}@test.com", i));
        server.post("/claims", &claim).await;
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    // Query with descending sort (newest first)
    let query_request = serde_json::json!({
        "category": "sort-test",
        "page": 1,
        "page_size": 10,
        "sort": {
            "field": "created_at",
            "order": "desc"
        }
    });

    let response = server.post("/query", &query_request).await;
    assert_eq!(response.status(), 200);

    let result: Value = response.json().await.expect("Failed to parse query result");
    let claims = result["claims"].as_array().unwrap();
    assert_eq!(claims.len(), 3);

    // Verify descending order
    if claims.len() >= 2 {
        let first_time = claims[0]["created_at"].as_str().unwrap();
        let second_time = claims[1]["created_at"].as_str().unwrap();
        assert!(first_time >= second_time, "Should be in descending order");
    }
}
