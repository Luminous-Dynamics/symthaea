// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim lifecycle integration tests
//!
//! Tests the complete lifecycle of a scientific claim from creation
//! through peer review to publication-ready status.

use super::helpers::*;
use serde_json::Value;

#[tokio::test]
async fn test_complete_claim_lifecycle_e0_to_e4() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Step 1: Create an E0 claim
    let claim_request = sample_claim_request("E0", "longevity", "dr.alice@uni.edu");
    let response = server.post("/claims", &claim_request).await;
    assert_eq!(response.status(), 201, "Failed to create claim");

    let claim: Value = response.json().await.expect("Failed to parse claim");
    let claim_id = claim["id"].as_str().expect("No claim ID");
    assert_eq!(claim["tier"], "E0");
    assert_eq!(claim["verifications"].as_array().unwrap().len(), 0);

    // Step 2: Add first verification (E0 → E1)
    let verification = sample_verification_request("peer1@uni.edu");
    let response = server.put(&format!("/claims/{}/verify", claim_id), &verification).await;
    assert_eq!(response.status(), 200);

    let updated_claim: Value = response.json().await.expect("Failed to parse updated claim");
    assert_eq!(updated_claim["tier"], "E1", "Should upgrade to E1 after 1 verification");
    assert_eq!(updated_claim["verifications"].as_array().unwrap().len(), 1);

    // Step 3: Add second verification (E1 → E1, need 3 for E2)
    let verification = sample_verification_request("peer2@uni.edu");
    let response = server.put(&format!("/claims/{}/verify", claim_id), &verification).await;
    assert_eq!(response.status(), 200);

    let updated_claim: Value = response.json().await.expect("Failed to parse updated claim");
    assert_eq!(updated_claim["tier"], "E1", "Should stay E1 with 2 verifications");
    assert_eq!(updated_claim["verifications"].as_array().unwrap().len(), 2);

    // Step 4: Add third verification (E1 → E2)
    let verification = sample_verification_request("peer3@uni.edu");
    let response = server.put(&format!("/claims/{}/verify", claim_id), &verification).await;
    assert_eq!(response.status(), 200);

    let updated_claim: Value = response.json().await.expect("Failed to parse updated claim");
    assert_eq!(updated_claim["tier"], "E2", "Should upgrade to E2 after 3 verifications");
    assert_eq!(updated_claim["verifications"].as_array().unwrap().len(), 3);

    // Step 5: Add fourth verification (E2 → E3)
    let verification = sample_verification_request("peer4@uni.edu");
    let response = server.put(&format!("/claims/{}/verify", claim_id), &verification).await;
    assert_eq!(response.status(), 200);

    let updated_claim: Value = response.json().await.expect("Failed to parse updated claim");
    assert_eq!(updated_claim["tier"], "E3", "Should upgrade to E3 after 4 verifications");
    assert_eq!(updated_claim["verifications"].as_array().unwrap().len(), 4);

    // Step 6: Add fifth verification (E3 → E4, publication-ready!)
    let verification = sample_verification_request("peer5@uni.edu");
    let response = server.put(&format!("/claims/{}/verify", claim_id), &verification).await;
    assert_eq!(response.status(), 200);

    let updated_claim: Value = response.json().await.expect("Failed to parse updated claim");
    assert_eq!(updated_claim["tier"], "E4", "Should upgrade to E4 after 5 verifications");
    assert_eq!(updated_claim["verifications"].as_array().unwrap().len(), 5);

    // Step 7: Verify final state
    let response = server.get(&format!("/claims/{}", claim_id)).await;
    assert_eq!(response.status(), 200);

    let final_claim: Value = response.json().await.expect("Failed to parse final claim");
    assert_eq!(final_claim["tier"], "E4");
    assert_eq!(final_claim["verifications"].as_array().unwrap().len(), 5);
    assert_eq!(final_claim["creator"], "dr.alice@uni.edu");
}

#[tokio::test]
async fn test_claim_with_provenance() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create source claim
    let source_request = sample_claim_request("E0", "physics", "dr.bob@lab.org");
    let response = server.post("/claims", &source_request).await;
    assert_eq!(response.status(), 201);

    let source_claim: Value = response.json().await.expect("Failed to parse source claim");
    let source_id = source_claim["id"].as_str().expect("No source ID");

    // Create derived claim
    let derived_request = sample_claim_request("E0", "physics", "dr.carol@lab.org");
    let response = server.post("/claims", &derived_request).await;
    assert_eq!(response.status(), 201);

    let derived_claim: Value = response.json().await.expect("Failed to parse derived claim");
    let derived_id = derived_claim["id"].as_str().expect("No derived ID");

    // Add provenance linking them
    let provenance = sample_provenance_request(Some(source_id));
    let response = server.put(&format!("/claims/{}/provenance", derived_id), &provenance).await;
    assert_eq!(response.status(), 200);

    // Verify provenance was added
    let response = server.get(&format!("/claims/{}", derived_id)).await;
    assert_eq!(response.status(), 200);

    let updated_claim: Value = response.json().await.expect("Failed to parse claim");
    let provenance_entries = updated_claim["provenance"].as_array().expect("No provenance");
    assert_eq!(provenance_entries.len(), 1);
    assert_eq!(provenance_entries[0]["source_id"], source_id);
}

#[tokio::test]
async fn test_claim_metadata_preservation() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create claim with specific metadata
    let claim_request = serde_json::json!({
        "tier": "E0",
        "content": {
            "dataset_hash": "blake3:specific_test_hash",
            "description": "Specific test description",
            "category": "biology",
            "keywords": vec!["DNA", "CRISPR", "gene-editing"]
        },
        "creator": "dr.test@example.com"
    });

    let response = server.post("/claims", &claim_request).await;
    assert_eq!(response.status(), 201);

    let claim: Value = response.json().await.expect("Failed to parse claim");
    let claim_id = claim["id"].as_str().expect("No claim ID");

    // Retrieve and verify all metadata is preserved
    let response = server.get(&format!("/claims/{}", claim_id)).await;
    assert_eq!(response.status(), 200);

    let retrieved: Value = response.json().await.expect("Failed to parse retrieved claim");
    assert_eq!(retrieved["content"]["dataset_hash"], "blake3:specific_test_hash");
    assert_eq!(retrieved["content"]["description"], "Specific test description");
    assert_eq!(retrieved["content"]["category"], "biology");
    assert_eq!(retrieved["creator"], "dr.test@example.com");

    let keywords = retrieved["content"]["keywords"].as_array().expect("No keywords");
    assert_eq!(keywords.len(), 3);
    assert!(keywords.iter().any(|k| k == "DNA"));
    assert!(keywords.iter().any(|k| k == "CRISPR"));
    assert!(keywords.iter().any(|k| k == "gene-editing"));
}
