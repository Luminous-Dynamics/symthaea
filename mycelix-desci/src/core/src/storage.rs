//! Storage abstraction for DeSci claims
//!
//! Supports multiple backends: IPFS, Filecoin, DHT

use crate::{DesciClaim, Error, Result};
use async_trait::async_trait;

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
        Err(Error::Storage("IPFS storage not yet implemented".to_string()))
    }

    async fn retrieve(&self, _cid: &str) -> Result<DesciClaim> {
        Err(Error::Storage("IPFS storage not yet implemented".to_string()))
    }

    async fn exists(&self, _cid: &str) -> Result<bool> {
        Err(Error::Storage("IPFS storage not yet implemented".to_string()))
    }

    async fn delete(&self, _cid: &str) -> Result<()> {
        Err(Error::Storage("IPFS storage not yet implemented".to_string()))
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

        let claim = DesciClaim::new(
            EpistemicTier::E0,
            content,
            "test_creator".to_string(),
        );

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
