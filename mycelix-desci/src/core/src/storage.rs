// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Storage abstraction for DeSci claims
//!
//! Supports multiple backends: IPFS, Filecoin, DHT

use crate::{DesciClaim, Error, Result};
use async_trait::async_trait;
use chrono::Utc;

/// Storage backend trait
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store a claim and return a content identifier
    async fn store(&self, claim: &DesciClaim) -> Result<String>;

    /// Retrieve a claim by its content identifier
    async fn retrieve(&self, cid: &str) -> Result<DesciClaim>;

    /// Check if a claim exists
    async fn exists(&self, cid: &str) -> Result<bool>;

    /// Delete a claim (if supported)
    async fn delete(&self, cid: &str) -> Result<()>;

    /// Retrieve the full authenticated proof package for a claim.
    async fn get_full_proof(&self, id: uuid::Uuid) -> Result<mycelix_zkp_core::AuthenticatedProof>;

    /// Update a claim with its permanent Arweave Quantum Anchor CID.
    async fn update_quantum_anchor_pointer(&self, id: uuid::Uuid, cid: &str) -> Result<()>;
}

/// In-memory storage for testing
#[derive(Clone)]
pub struct MemoryStorage {
    claims: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, DesciClaim>>>,
}

impl MemoryStorage {
    /// Create a new in-memory storage
    pub fn new() -> Self {
        Self {
            claims: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StorageBackend for MemoryStorage {
    async fn store(&self, claim: &DesciClaim) -> Result<String> {
        let cid = claim.id.to_string();
        let mut claims = self.claims.write().await;
        claims.insert(cid.clone(), claim.clone());
        Ok(cid)
    }

    async fn retrieve(&self, cid: &str) -> Result<DesciClaim> {
        let claims = self.claims.read().await;
        claims
            .get(cid)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("Claim not found: {}", cid)))
    }

    async fn exists(&self, cid: &str) -> Result<bool> {
        let claims = self.claims.read().await;
        Ok(claims.contains_key(cid))
    }

    async fn delete(&self, cid: &str) -> Result<()> {
        let mut claims = self.claims.write().await;
        claims.remove(cid);
        Ok(())
    }

    async fn get_full_proof(
        &self,
        _id: uuid::Uuid,
    ) -> Result<mycelix_zkp_core::AuthenticatedProof> {
        // Mock implementation for simulation
        Ok(mycelix_zkp_core::AuthenticatedProof {
            proof: vec![0; 64],
            signature: vec![0; 3300],
            metadata: mycelix_zkp_core::ProofMetadata {
                domain_tag: mycelix_zkp_core::DomainTag::new("Mycelix", "DeSci", 1),
                protocol_version: 1,
                client_id: [0xAA; 32],
                timestamp: Utc::now().timestamp() as u64,
                nonce: [0xBB; 32],
                backend: mycelix_zkp_core::types::BackendId::Winterfell,
            },
            public_inputs_hash: [0; 32],
            joules_consumed: 0.0,
        })
    }

    async fn update_quantum_anchor_pointer(&self, _id: uuid::Uuid, _cid: &str) -> Result<()> {
        // Mock implementation for simulation
        Ok(())
    }
}

/// IPFS storage backend (placeholder)
pub struct IpfsStorage {
    // TODO: Add IPFS client
}

impl IpfsStorage {
    /// Create a new IPFS storage backend
    pub fn new(_api_url: &str) -> Result<Self> {
        // TODO: Initialize IPFS client
        Ok(Self {})
    }
}

#[async_trait]
impl StorageBackend for IpfsStorage {
    async fn store(&self, _claim: &DesciClaim) -> Result<String> {
        // TODO: Implement IPFS upload
        Err(Error::Storage(
            "IPFS storage not yet implemented".to_string(),
        ))
    }

    async fn retrieve(&self, _cid: &str) -> Result<DesciClaim> {
        Err(Error::Storage(
            "IPFS storage not yet implemented".to_string(),
        ))
    }

    async fn exists(&self, _cid: &str) -> Result<bool> {
        Err(Error::Storage(
            "IPFS storage not yet implemented".to_string(),
        ))
    }

    async fn delete(&self, _cid: &str) -> Result<()> {
        Err(Error::Storage(
            "IPFS storage not yet implemented".to_string(),
        ))
    }

    async fn get_full_proof(
        &self,
        _id: uuid::Uuid,
    ) -> Result<mycelix_zkp_core::AuthenticatedProof> {
        Err(Error::Storage(
            "IPFS storage not yet implemented".to_string(),
        ))
    }

    async fn update_quantum_anchor_pointer(&self, _id: uuid::Uuid, _cid: &str) -> Result<()> {
        Err(Error::Storage(
            "IPFS storage not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::{ClaimContent, EpistemicTier};

    #[tokio::test]
    async fn test_memory_storage() {
        let storage = MemoryStorage::new();

        let content = ClaimContent {
            dataset_hash: "test_hash".to_string(),
            description: "Test claim".to_string(),
            category: "test".to_string(),
            keywords: vec![],
            storage_ref: None,
            reproducibility_score: None,
            license: None,
        };

        let claim = DesciClaim::new(EpistemicTier::E0, content, "test_creator".to_string());

        // Store
        let cid = storage.store(&claim).await.unwrap();

        // Retrieve
        let retrieved = storage.retrieve(&cid).await.unwrap();
        assert_eq!(retrieved.id, claim.id);

        // Check existence
        assert!(storage.exists(&cid).await.unwrap());

        // Delete
        storage.delete(&cid).await.unwrap();
        assert!(!storage.exists(&cid).await.unwrap());
    }
}
