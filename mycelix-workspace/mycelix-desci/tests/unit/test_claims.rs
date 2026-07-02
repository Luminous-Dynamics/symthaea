// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Comprehensive tests for the claims module

use mycelix_desci_core::{
    claims::{ClaimContent, DesciClaim, EpistemicTier, Provenance, Verification},
    hash,
};
use chrono::Utc;

mod fixtures;
use fixtures::*;

#[test]
fn test_create_basic_claim() {
    let content = ClaimContent {
        dataset_hash: "blake3:test123".to_string(),
        description: "Test claim".to_string(),
        category: "test".to_string(),
        keywords: vec!["test".to_string()],
        storage_ref: None,
        reproducibility_score: None,
        license: None,
    };

    let claim = DesciClaim::new(EpistemicTier::E0, content.clone(), "creator_123".to_string());

    assert_eq!(claim.epistemic_tier, EpistemicTier::E0);
    assert_eq!(claim.content.description, "Test claim");
    assert_eq!(claim.creator, "creator_123");
    assert_eq!(claim.verifications.len(), 0);
    assert_eq!(claim.provenance.len(), 0);
}

#[test]
fn test_epistemic_tier_ordering() {
    assert!(EpistemicTier::E4 > EpistemicTier::E3);
    assert!(EpistemicTier::E3 > EpistemicTier::E2);
    assert!(EpistemicTier::E2 > EpistemicTier::E1);
    assert!(EpistemicTier::E1 > EpistemicTier::E0);
}

#[test]
fn test_epistemic_tier_descriptions() {
    assert_eq!(EpistemicTier::E0.description(), "Unverified claim");
    assert_eq!(EpistemicTier::E4.description(), "Peer-reviewed and independently reproduced");
}

#[test]
fn test_epistemic_tier_min_verifications() {
    assert_eq!(EpistemicTier::E0.min_verifications(), 0);
    assert_eq!(EpistemicTier::E1.min_verifications(), 1);
    assert_eq!(EpistemicTier::E2.min_verifications(), 2);
    assert_eq!(EpistemicTier::E3.min_verifications(), 3);
    assert_eq!(EpistemicTier::E4.min_verifications(), 5);
}

#[test]
fn test_add_provenance() {
    let mut claim = create_sample_claim();
    let initial_count = claim.provenance.len();

    let prov = Provenance::new("Test Source".to_string(), "test".to_string());
    claim.add_provenance(prov);

    assert_eq!(claim.provenance.len(), initial_count + 1);
    assert_eq!(claim.provenance.last().unwrap().source, "Test Source");
}

#[test]
fn test_add_verification() {
    let mut claim = create_sample_claim();

    let verification = Verification {
        verifier: "verifier_1".to_string(),
        timestamp: Utc::now(),
        signature: vec![1, 2, 3, 4],
        notes: Some("Verified successfully".to_string()),
    };

    claim.add_verification(verification);

    assert_eq!(claim.verifications.len(), 1);
    assert_eq!(claim.verifications[0].verifier, "verifier_1");
}

#[test]
fn test_automatic_tier_upgrade() {
    let mut claim = create_sample_claim();
    assert_eq!(claim.epistemic_tier, EpistemicTier::E0);

    // Add 1 verification -> should upgrade to E1
    let ver1 = Verification {
        verifier: "v1".to_string(),
        timestamp: Utc::now(),
        signature: vec![1],
        notes: None,
    };
    claim.add_verification(ver1);
    assert_eq!(claim.epistemic_tier, EpistemicTier::E1);

    // Add 2nd verification -> should upgrade to E2
    let ver2 = Verification {
        verifier: "v2".to_string(),
        timestamp: Utc::now(),
        signature: vec![2],
        notes: None,
    };
    claim.add_verification(ver2);
    assert_eq!(claim.epistemic_tier, EpistemicTier::E2);

    // Add more to reach E3
    for i in 3..=5 {
        let ver = Verification {
            verifier: format!("v{}", i),
            timestamp: Utc::now(),
            signature: vec![i],
            notes: None,
        };
        claim.add_verification(ver);
    }
    assert_eq!(claim.epistemic_tier, EpistemicTier::E3);
}

#[test]
fn test_tier_never_downgrades() {
    let mut claim = create_claim_with_tier(EpistemicTier::E3);

    // Even with only 1 verification, tier should stay at E3
    claim.verifications.clear();
    let ver = Verification {
        verifier: "v1".to_string(),
        timestamp: Utc::now(),
        signature: vec![1],
        notes: None,
    };
    claim.add_verification(ver);

    assert_eq!(claim.epistemic_tier, EpistemicTier::E3);  // Should not downgrade
}

#[test]
fn test_is_valid_for_tier() {
    // E0 claim needs 0 verifications
    let claim_e0 = create_claim_with_tier(EpistemicTier::E0);
    assert!(claim_e0.is_valid_for_tier());

    // E3 claim needs 3 verifications
    let claim_e3_invalid = create_claim_with_tier(EpistemicTier::E3);
    assert!(!claim_e3_invalid.is_valid_for_tier());  // Has 0 verifications

    let claim_e3_valid = create_verified_claim(3);
    assert!(claim_e3_valid.is_valid_for_tier());  // Has 3 verifications
}

#[test]
fn test_json_serialization() {
    let claim = create_sample_claim();

    // Serialize
    let json = claim.to_json().expect("Failed to serialize");
    assert!(!json.is_empty());
    assert!(json.contains("epistemic_tier"));
    assert!(json.contains("dataset_hash"));

    // Deserialize
    let deserialized = DesciClaim::from_json(&json).expect("Failed to deserialize");

    assert_eq!(deserialized.id, claim.id);
    assert_eq!(deserialized.epistemic_tier, claim.epistemic_tier);
    assert_eq!(deserialized.content.description, claim.content.description);
}

#[test]
fn test_json_roundtrip() {
    let original = create_verified_claim(3);
    let json = original.to_json().unwrap();
    let restored = DesciClaim::from_json(&json).unwrap();

    assert_eq!(original.id, restored.id);
    assert_eq!(original.verifications.len(), restored.verifications.len());
    assert_eq!(original.provenance.len(), restored.provenance.len());
}

#[test]
fn test_provenance_with_url() {
    let prov = Provenance::new("Test".to_string(), "test".to_string())
        .with_url("https://example.com".to_string());

    assert_eq!(prov.url, Some("https://example.com".to_string()));
}

#[test]
fn test_provenance_with_metadata() {
    let prov = Provenance::new("Test".to_string(), "test".to_string())
        .with_metadata("key1", serde_json::json!("value1"))
        .with_metadata("key2", serde_json::json!(42));

    assert!(prov.metadata.get("key1").is_some());
    assert_eq!(prov.metadata.get("key1").unwrap(), "value1");
}

#[test]
fn test_claim_timestamps() {
    let claim = create_sample_claim();

    assert!(claim.created_at <= claim.updated_at);

    // After adding verification, updated_at should change
    std::thread::sleep(std::time::Duration::from_millis(10));

    let mut claim_modified = claim.clone();
    let ver = Verification {
        verifier: "v1".to_string(),
        timestamp: Utc::now(),
        signature: vec![1],
        notes: None,
    };
    claim_modified.add_verification(ver);

    assert!(claim_modified.updated_at > claim.updated_at);
}

#[test]
fn test_claim_content_keywords() {
    let content = ClaimContent {
        dataset_hash: "test".to_string(),
        description: "Test".to_string(),
        category: "test".to_string(),
        keywords: vec!["keyword1".to_string(), "keyword2".to_string(), "keyword3".to_string()],
        storage_ref: None,
        reproducibility_score: None,
        license: None,
    };

    let claim = DesciClaim::new(EpistemicTier::E0, content, "creator".to_string());

    assert_eq!(claim.content.keywords.len(), 3);
    assert!(claim.content.keywords.contains(&"keyword2".to_string()));
}

#[test]
fn test_reproducibility_score_bounds() {
    let mut content = ClaimContent {
        dataset_hash: "test".to_string(),
        description: "Test".to_string(),
        category: "test".to_string(),
        keywords: vec![],
        storage_ref: None,
        reproducibility_score: Some(0.95),
        license: None,
    };

    assert!(content.reproducibility_score.unwrap() <= 1.0);
    assert!(content.reproducibility_score.unwrap() >= 0.0);

    // Edge cases
    content.reproducibility_score = Some(1.0);
    assert_eq!(content.reproducibility_score, Some(1.0));

    content.reproducibility_score = Some(0.0);
    assert_eq!(content.reproducibility_score, Some(0.0));
}

#[test]
fn test_multiple_verifications_same_verifier() {
    let mut claim = create_sample_claim();

    // Add two verifications from same verifier
    for i in 0..2 {
        let ver = Verification {
            verifier: "same_verifier".to_string(),
            timestamp: Utc::now(),
            signature: vec![i],
            notes: None,
        };
        claim.add_verification(ver);
    }

    assert_eq!(claim.verifications.len(), 2);
    // Both should be recorded even though same verifier
}

#[test]
fn test_claim_with_empty_keywords() {
    let content = ClaimContent {
        dataset_hash: "test".to_string(),
        description: "Test".to_string(),
        category: "test".to_string(),
        keywords: vec![],
        storage_ref: None,
        reproducibility_score: None,
        license: None,
    };

    let claim = DesciClaim::new(EpistemicTier::E0, content, "creator".to_string());

    assert_eq!(claim.content.keywords.len(), 0);
}

#[test]
fn test_claim_with_ipfs_storage() {
    let content = ClaimContent {
        dataset_hash: hash::hash_bytes(b"test_data").to_string(),
        description: "Test with IPFS".to_string(),
        category: "test".to_string(),
        keywords: vec![],
        storage_ref: Some("ipfs://QmTest123456".to_string()),
        reproducibility_score: None,
        license: None,
    };

    let claim = DesciClaim::new(EpistemicTier::E0, content, "creator".to_string());

    assert!(claim.content.storage_ref.is_some());
    assert!(claim.content.storage_ref.unwrap().starts_with("ipfs://"));
}

#[test]
fn test_license_types() {
    let licenses = vec!["MIT", "CC-BY-4.0", "CC0-1.0", "Apache-2.0"];

    for license in licenses {
        let content = ClaimContent {
            dataset_hash: "test".to_string(),
            description: "Test".to_string(),
            category: "test".to_string(),
            keywords: vec![],
            storage_ref: None,
            reproducibility_score: None,
            license: Some(license.to_string()),
        };

        let claim = DesciClaim::new(EpistemicTier::E0, content, "creator".to_string());

        assert_eq!(claim.content.license.as_ref().unwrap(), license);
    }
}
