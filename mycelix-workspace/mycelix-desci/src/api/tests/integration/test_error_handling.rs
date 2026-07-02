// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error handling integration tests
//!
//! Tests that the API correctly handles invalid requests and error conditions.

use super::helpers::*;
use serde_json::Value;

#[tokio::test]
async fn test_get_nonexistent_claim() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let fake_id = "00000000-0000-0000-0000-000000000000";
    let response = server.get(&format!("/claims/{}", fake_id)).await;

    assert_eq!(response.status(), 404, "Should return 404 for nonexistent claim");
}

#[tokio::test]
async fn test_create_claim_with_invalid_tier() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let invalid_request = serde_json::json!({
        "tier": "E99",  // Invalid tier
        "content": {
            "dataset_hash": "blake3:test",
            "description": "Test",
            "category": "test",
            "keywords": ["test"]
        },
        "creator": "test@example.com"
    });

    let response = server.post("/claims", &invalid_request).await;

    assert!(
        response.status().is_client_error(),
        "Should return 4xx error for invalid tier"
    );
}

#[tokio::test]
async fn test_create_claim_with_missing_fields() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let invalid_request = serde_json::json!({
        "tier": "E0",
        "content": {
            "dataset_hash": "blake3:test"
            // Missing required fields
        }
    });

    let response = server.post("/claims", &invalid_request).await;

    assert!(
        response.status().is_client_error(),
        "Should return 4xx error for missing fields"
    );
}

#[tokio::test]
async fn test_verify_nonexistent_claim() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let fake_id = "00000000-0000-0000-0000-000000000000";
    let verification = sample_verification_request("verifier@test.com");

    let response = server.put(&format!("/claims/{}/verify", fake_id), &verification).await;

    assert_eq!(response.status(), 404, "Should return 404 for nonexistent claim");
}

#[tokio::test]
async fn test_add_provenance_to_nonexistent_claim() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let fake_id = "00000000-0000-0000-0000-000000000000";
    let provenance = sample_provenance_request(None);

    let response = server.put(&format!("/claims/{}/provenance", fake_id), &provenance).await;

    assert_eq!(response.status(), 404, "Should return 404 for nonexistent claim");
}

#[tokio::test]
async fn test_query_with_invalid_pagination() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Invalid page size (0)
    let invalid_query = serde_json::json!({
        "category": "test",
        "page": 1,
        "page_size": 0
    });

    let response = server.post("/query", &invalid_query).await;

    assert!(
        response.status().is_client_error() || response.status().is_success(),
        "Should handle invalid page size gracefully"
    );

    // Invalid page number (0)
    let invalid_query = serde_json::json!({
        "category": "test",
        "page": 0,
        "page_size": 10
    });

    let response = server.post("/query", &invalid_query).await;

    assert!(
        response.status().is_client_error() || response.status().is_success(),
        "Should handle invalid page number gracefully"
    );
}

#[tokio::test]
async fn test_malformed_json_request() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let response = server
        .client
        .post(server.api_url("/claims"))
        .header("Content-Type", "application/json")
        .body("{this is not valid json}")
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(response.status(), 400, "Should return 400 for malformed JSON");
}

#[tokio::test]
async fn test_empty_request_body() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let response = server
        .client
        .post(server.api_url("/claims"))
        .header("Content-Type", "application/json")
        .body("")
        .send()
        .await
        .expect("Failed to send request");

    assert!(
        response.status().is_client_error(),
        "Should return 4xx for empty request body"
    );
}

#[tokio::test]
async fn test_trust_update_with_invalid_participant() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let update_request = serde_json::json!({
        "positive": true,
        "weight": 1.0
    });

    let response = server.put("/trust/", &update_request).await;

    assert!(
        response.status().is_client_error(),
        "Should handle empty participant ID"
    );
}

#[tokio::test]
async fn test_verification_with_missing_signature() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create a claim first
    let claim_request = sample_claim_request("E0", "test", "creator@test.com");
    let response = server.post("/claims", &claim_request).await;
    assert_eq!(response.status(), 201);

    let claim: Value = response.json().await.expect("Failed to parse claim");
    let claim_id = claim["id"].as_str().expect("No claim ID");

    // Try to verify without signature
    let invalid_verification = serde_json::json!({
        "verifier": "verifier@test.com"
        // Missing signature
    });

    let response = server.put(&format!("/claims/{}/verify", claim_id), &invalid_verification).await;

    assert!(
        response.status().is_client_error(),
        "Should return 4xx for verification without signature"
    );
}

#[tokio::test]
async fn test_query_with_extremely_large_page_size() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let query = serde_json::json!({
        "category": "test",
        "page": 1,
        "page_size": 999999
    });

    let response = server.post("/query", &query).await;

    // Should either reject or cap the page size
    assert!(response.status().is_success() || response.status().is_client_error());

    if response.status().is_success() {
        let result: Value = response.json().await.expect("Failed to parse result");
        let page_size = result["page_size"].as_u64().unwrap_or(0);
        assert!(page_size <= 1000, "Should cap page size to reasonable limit");
    }
}

#[tokio::test]
async fn test_duplicate_verification_from_same_verifier() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create a claim
    let claim_request = sample_claim_request("E0", "test", "creator@test.com");
    let response = server.post("/claims", &claim_request).await;
    assert_eq!(response.status(), 201);

    let claim: Value = response.json().await.expect("Failed to parse claim");
    let claim_id = claim["id"].as_str().expect("No claim ID");

    // Add verification
    let verification = sample_verification_request("verifier@test.com");
    let response = server.put(&format!("/claims/{}/verify", claim_id), &verification).await;
    assert_eq!(response.status(), 200);

    // Try to add same verifier again
    let duplicate_verification = sample_verification_request("verifier@test.com");
    let response = server.put(&format!("/claims/{}/verify", claim_id), &duplicate_verification).await;

    // Should either accept (updating existing) or reject duplicate
    assert!(response.status().is_success() || response.status().is_client_error());
}
