// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Comprehensive tests for the storage module

use mycelix_desci_core::{
    claims::{DesciClaim, EpistemicTier},
    storage::{MemoryStorage, StorageBackend},
    Error,
};

mod fixtures;
use fixtures::*;

#[tokio::test]
async fn test_memory_storage_store() {
    let storage = MemoryStorage::new();
    let claim = create_sample_claim();

    let cid = storage.store(&claim).await.expect("Failed to store");

    assert!(!cid.is_empty());
    assert_eq!(cid, claim.id.to_string());
}

#[tokio::test]
async fn test_memory_storage_retrieve() {
    let storage = MemoryStorage::new();
    let claim = create_sample_claim();
    let claim_id = claim.id;

    let cid = storage.store(&claim).await.unwrap();
    let retrieved = storage.retrieve(&cid).await.unwrap();

    assert_eq!(retrieved.id, claim_id);
    assert_eq!(retrieved.content.description, claim.content.description);
}

#[tokio::test]
async fn test_memory_storage_retrieve_nonexistent() {
    let storage = MemoryStorage::new();

    let result = storage.retrieve("nonexistent_id").await;

    assert!(result.is_err());
    match result {
        Err(Error::NotFound(_)) => (),  // Expected
        _ => panic!("Expected NotFound error"),
    }
}

#[tokio::test]
async fn test_memory_storage_exists() {
    let storage = MemoryStorage::new();
    let claim = create_sample_claim();

    let cid = storage.store(&claim).await.unwrap();

    assert!(storage.exists(&cid).await.unwrap());
    assert!(!storage.exists("nonexistent").await.unwrap());
}

#[tokio::test]
async fn test_memory_storage_delete() {
    let storage = MemoryStorage::new();
    let claim = create_sample_claim();

    let cid = storage.store(&claim).await.unwrap();
    assert!(storage.exists(&cid).await.unwrap());

    storage.delete(&cid).await.unwrap();
    assert!(!storage.exists(&cid).await.unwrap());
}

#[tokio::test]
async fn test_memory_storage_multiple_claims() {
    let storage = MemoryStorage::new();
    let claims = create_test_claim_set();

    let mut stored_ids = Vec::new();

    for claim in &claims {
        let cid = storage.store(claim).await.unwrap();
        stored_ids.push(cid);
    }

    assert_eq!(stored_ids.len(), claims.len());

    // Verify all can be retrieved
    for cid in stored_ids {
        assert!(storage.exists(&cid).await.unwrap());
        let retrieved = storage.retrieve(&cid).await.unwrap();
        assert!(!retrieved.content.description.is_empty());
    }
}

#[tokio::test]
async fn test_memory_storage_update() {
    let storage = MemoryStorage::new();
    let mut claim = create_sample_claim();

    let cid = storage.store(&claim).await.unwrap();

    // Modify claim
    let verification = mycelix_desci_core::claims::Verification {
        verifier: "new_verifier".to_string(),
        timestamp: chrono::Utc::now(),
        signature: vec![1, 2, 3],
        notes: Some("Updated".to_string()),
    };
    claim.add_verification(verification);

    // Store updated version
    storage.store(&claim).await.unwrap();

    // Retrieve and verify update
    let retrieved = storage.retrieve(&cid).await.unwrap();
    assert_eq!(retrieved.verifications.len(), 1);
    assert_eq!(retrieved.verifications[0].verifier, "new_verifier");
}

#[tokio::test]
async fn test_memory_storage_concurrent_access() {
    use std::sync::Arc;

    let storage = Arc::new(MemoryStorage::new());
    let claim = create_sample_claim();
    let cid = storage.store(&claim).await.unwrap();

    let mut handles = vec![];

    // Spawn multiple concurrent readers
    for _ in 0..10 {
        let storage_clone = Arc::clone(&storage);
        let cid_clone = cid.clone();

        let handle = tokio::spawn(async move {
            storage_clone.retrieve(&cid_clone).await.unwrap()
        });

        handles.push(handle);
    }

    // All should succeed
    for handle in handles {
        let retrieved = handle.await.unwrap();
        assert_eq!(retrieved.id, claim.id);
    }
}

#[tokio::test]
async fn test_memory_storage_large_claim() {
    let storage = MemoryStorage::new();

    // Create claim with large description and many keywords
    let large_description = "Large description ".repeat(1000);
    let large_keywords: Vec<String> = (0..100).map(|i| format!("keyword_{}", i)).collect();

    let content = mycelix_desci_core::claims::ClaimContent {
        dataset_hash: "test".to_string(),
        description: large_description.clone(),
        category: "test".to_string(),
        keywords: large_keywords.clone(),
        storage_ref: None,
        reproducibility_score: None,
        license: None,
    };

    let claim = DesciClaim::new(EpistemicTier::E0, content, "creator".to_string());

    let cid = storage.store(&claim).await.unwrap();
    let retrieved = storage.retrieve(&cid).await.unwrap();

    assert_eq!(retrieved.content.description.len(), large_description.len());
    assert_eq!(retrieved.content.keywords.len(), 100);
}

#[tokio::test]
async fn test_memory_storage_delete_nonexistent() {
    let storage = MemoryStorage::new();

    // Deleting nonexistent should not error (idempotent)
    let result = storage.delete("nonexistent").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_memory_storage_preserves_metadata() {
    let storage = MemoryStorage::new();
    let claim = create_claim_with_provenance();

    let cid = storage.store(&claim).await.unwrap();
    let retrieved = storage.retrieve(&cid).await.unwrap();

    // Verify provenance preserved
    assert_eq!(retrieved.provenance.len(), claim.provenance.len());
    assert_eq!(retrieved.provenance[0].source, claim.provenance[0].source);

    // Verify timestamps preserved
    assert_eq!(retrieved.created_at, claim.created_at);
    assert_eq!(retrieved.updated_at, claim.updated_at);
}

#[tokio::test]
async fn test_memory_storage_different_tiers() {
    let storage = MemoryStorage::new();

    let tiers = vec![
        EpistemicTier::E0,
        EpistemicTier::E1,
        EpistemicTier::E2,
        EpistemicTier::E3,
        EpistemicTier::E4,
    ];

    for tier in tiers {
        let claim = create_claim_with_tier(tier);
        let cid = storage.store(&claim).await.unwrap();
        let retrieved = storage.retrieve(&cid).await.unwrap();

        assert_eq!(retrieved.epistemic_tier, tier);
    }
}

#[tokio::test]
async fn test_memory_storage_special_characters() {
    let storage = MemoryStorage::new();

    let content = mycelix_desci_core::claims::ClaimContent {
        dataset_hash: "test".to_string(),
        description: "Test with special chars: 日本語, émojis 🔬, symbols <>&\"'".to_string(),
        category: "test".to_string(),
        keywords: vec!["tag-with-dash".to_string(), "tag_with_underscore".to_string()],
        storage_ref: None,
        reproducibility_score: None,
        license: None,
    };

    let claim = DesciClaim::new(EpistemicTier::E0, content, "creator".to_string());

    let cid = storage.store(&claim).await.unwrap();
    let retrieved = storage.retrieve(&cid).await.unwrap();

    assert_eq!(retrieved.content.description, claim.content.description);
}

#[tokio::test]
async fn test_memory_storage_id_uniqueness() {
    let storage = MemoryStorage::new();

    let claim1 = create_sample_claim();
    let claim2 = create_sample_claim();

    let cid1 = storage.store(&claim1).await.unwrap();
    let cid2 = storage.store(&claim2).await.unwrap();

    // IDs should be different (UUIDs)
    assert_ne!(cid1, cid2);
}
