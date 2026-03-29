// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claims API endpoint tests
//!
//! Tests all claim-related API endpoints.

use super::helpers::*;
use serde_json::Value;

#[tokio::test]
async fn test_create_claim_endpoint() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let claim_request = sample_claim_request("E0", "biology", "researcher@uni.edu");
    let response = server.post("/claims", &claim_request).await;

    assert_eq!(response.status(), 201, "Should return 201 Created");

    let claim: Value = response.json().await.expect("Failed to parse response");
    assert!(claim["id"].is_string(), "Should have an ID");
    assert_eq!(claim["tier"], "E0");
    assert_eq!(claim["content"]["category"], "biology");
    assert_eq!(claim["creator"], "researcher@uni.edu");
    assert!(claim["created_at"].is_string(), "Should have created_at timestamp");
    assert!(claim["updated_at"].is_string(), "Should have updated_at timestamp");
}

#[tokio::test]
async fn test_get_claim_endpoint() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create a claim first
    let claim_request = sample_claim_request("E0", "chemistry", "dr.chem@lab.org");
    let create_response = server.post("/claims", &claim_request).await;
    assert_eq!(create_response.status(), 201);

    let created_claim: Value = create_response.json().await.expect("Failed to parse claim");
    let claim_id = created_claim["id"].as_str().expect("No claim ID");

    // Retrieve it
    let response = server.get(&format!("/claims/{}", claim_id)).await;
    assert_eq!(response.status(), 200, "Should return 200 OK");

    let retrieved_claim: Value = response.json().await.expect("Failed to parse response");
    assert_eq!(retrieved_claim["id"], claim_id);
    assert_eq!(retrieved_claim["tier"], "E0");
    assert_eq!(retrieved_claim["content"]["category"], "chemistry");
    assert_eq!(retrieved_claim["creator"], "dr.chem@lab.org");
}

#[tokio::test]
async fn test_verify_claim_endpoint() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create a claim
    let claim_request = sample_claim_request("E0", "physics", "dr.physics@institute.edu");
    let create_response = server.post("/claims", &claim_request).await;
    assert_eq!(create_response.status(), 201);

    let claim: Value = create_response.json().await.expect("Failed to parse claim");
    let claim_id = claim["id"].as_str().expect("No claim ID");

    // Add verification
    let verification = sample_verification_request("peer@reviewer.org");
    let response = server.put(&format!("/claims/{}/verify", claim_id), &verification).await;

    assert_eq!(response.status(), 200, "Should return 200 OK");

    let verified_claim: Value = response.json().await.expect("Failed to parse response");
    assert_eq!(verified_claim["tier"], "E1", "Should upgrade to E1");

    let verifications = verified_claim["verifications"].as_array().expect("No verifications array");
    assert_eq!(verifications.len(), 1);
    assert_eq!(verifications[0]["verifier"], "peer@reviewer.org");
    assert!(verifications[0]["verified_at"].is_string());
}

#[tokio::test]
async fn test_add_provenance_endpoint() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create a claim
    let claim_request = sample_claim_request("E0", "astronomy", "dr.astro@observatory.edu");
    let create_response = server.post("/claims", &claim_request).await;
    assert_eq!(create_response.status(), 201);

    let claim: Value = create_response.json().await.expect("Failed to parse claim");
    let claim_id = claim["id"].as_str().expect("No claim ID");

    // Add provenance
    let provenance = sample_provenance_request(None);
    let response = server.put(&format!("/claims/{}/provenance", claim_id), &provenance).await;

    assert_eq!(response.status(), 200, "Should return 200 OK");

    let updated_claim: Value = response.json().await.expect("Failed to parse response");
    let provenance_entries = updated_claim["provenance"].as_array().expect("No provenance array");
    assert_eq!(provenance_entries.len(), 1);
    assert_eq!(provenance_entries[0]["transformation"], "test_transformation");
    assert!(provenance_entries[0]["timestamp"].is_string());
}

#[tokio::test]
async fn test_create_multiple_claims_different_categories() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let categories = vec!["biology", "physics", "chemistry", "astronomy", "longevity"];

    for (i, category) in categories.iter().enumerate() {
        let claim_request = sample_claim_request("E0", category, &format!("researcher{}@test.com", i));
        let response = server.post("/claims", &claim_request).await;

        assert_eq!(response.status(), 201);

        let claim: Value = response.json().await.expect("Failed to parse claim");
        assert_eq!(claim["content"]["category"], *category);
    }
}

#[tokio::test]
async fn test_verify_claim_multiple_times() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create a claim
    let claim_request = sample_claim_request("E0", "test", "creator@test.com");
    let create_response = server.post("/claims", &claim_request).await;
    let claim: Value = create_response.json().await.expect("Failed to parse claim");
    let claim_id = claim["id"].as_str().expect("No claim ID");

    // Add 3 verifications
    for i in 0..3 {
        let verification = sample_verification_request(&format!("verifier{}@test.com", i));
        let response = server.put(&format!("/claims/{}/verify", claim_id), &verification).await;
        assert_eq!(response.status(), 200);
    }

    // Check final state
    let response = server.get(&format!("/claims/{}", claim_id)).await;
    let final_claim: Value = response.json().await.expect("Failed to parse claim");

    assert_eq!(final_claim["tier"], "E2");
    assert_eq!(final_claim["verifications"].as_array().unwrap().len(), 3);
}
