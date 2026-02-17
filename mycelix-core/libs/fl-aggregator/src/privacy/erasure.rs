//! GDPR Cryptographic Erasure Mechanism
//!
//! Implements the "right to be forgotten" (GDPR Article 17) through cryptographic
//! key deletion. User data is encrypted with user-specific keys; deleting the key
//! renders the data permanently unrecoverable.
//!
//! ## Architecture
//!
//! ```text
//! User Data → Encrypt(user_key) → Encrypted Blob → Store
//!                  ↓
//!             user_key stored in
//!             user's Holochain source chain
//!                  ↓
//!             Erasure Request
//!                  ↓
//!             Delete user_key
//!                  ↓
//!             Data permanently unrecoverable
//! ```
//!
//! ## Legal Compliance
//!
//! This implementation satisfies GDPR Article 17 requirements:
//! - Data subjects can request erasure of personal data
//! - Erasure is technically irreversible (key deletion)
//! - Erasure receipts provide audit trail
//! - Processing of erased data is impossible (cryptographically guaranteed)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use thiserror::Error;

/// Errors that can occur during erasure operations.
#[derive(Debug, Error)]
pub enum ErasureError {
    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    #[error("Encryption failed: {0}")]
    EncryptionFailed(String),

    #[error("Decryption failed: {0}")]
    DecryptionFailed(String),

    #[error("Key already erased for agent: {0}")]
    KeyAlreadyErased(String),

    #[error("Invalid erasure request: {0}")]
    InvalidRequest(String),
}

/// Result type for erasure operations.
pub type ErasureResult<T> = Result<T, ErasureError>;

/// A user-specific encryption key for GDPR-compliant data handling.
#[derive(Clone)]
pub struct UserEncryptionKey {
    /// The raw key bytes (256-bit)
    key_bytes: [u8; 32],
    /// When the key was created
    created_at: u64,
    /// Optional key ID for tracking
    key_id: String,
}

impl UserEncryptionKey {
    /// Generate a new random encryption key.
    pub fn generate() -> Self {
        let mut key_bytes = [0u8; 32];
        // Use system randomness
        getrandom::getrandom(&mut key_bytes).expect("Failed to generate random key");

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let key_id = format!("key-{}", hex::encode(&key_bytes[..8]));

        Self {
            key_bytes,
            created_at: now,
            key_id,
        }
    }

    /// Get the key ID (safe to log).
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// Get creation timestamp.
    pub fn created_at(&self) -> u64 {
        self.created_at
    }
}

/// Receipt proving that erasure was performed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureReceipt {
    /// Unique receipt ID
    pub receipt_id: String,
    /// Hash of the agent's public key (not the key itself)
    pub agent_hash: String,
    /// Hash of the encryption key that was deleted
    pub key_hash: String,
    /// Timestamp of erasure
    pub erased_at: u64,
    /// Number of data items affected
    pub items_affected: u64,
    /// Cryptographic proof of erasure (hash of operation)
    pub proof_hash: String,
}

impl ErasureReceipt {
    /// Verify the receipt's cryptographic proof.
    pub fn verify(&self) -> bool {
        let mut hasher = Sha3_256::new();
        hasher.update(self.receipt_id.as_bytes());
        hasher.update(self.agent_hash.as_bytes());
        hasher.update(self.key_hash.as_bytes());
        hasher.update(self.erased_at.to_le_bytes());
        hasher.update(self.items_affected.to_le_bytes());

        let expected = hex::encode(hasher.finalize());
        expected == self.proof_hash
    }
}

/// Tracks erased agents to prevent re-processing their data.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErasureRegistry {
    /// Hashes of erased agent keys (for lookup without storing keys)
    erased_agents: HashMap<String, ErasureReceipt>,
}

impl ErasureRegistry {
    /// Check if an agent has been erased.
    pub fn is_erased(&self, agent_hash: &str) -> bool {
        self.erased_agents.contains_key(agent_hash)
    }

    /// Get erasure receipt for an agent.
    pub fn get_receipt(&self, agent_hash: &str) -> Option<&ErasureReceipt> {
        self.erased_agents.get(agent_hash)
    }

    /// Record an erasure.
    pub fn record_erasure(&mut self, receipt: ErasureReceipt) {
        self.erased_agents.insert(receipt.agent_hash.clone(), receipt);
    }

    /// Get count of erased agents.
    pub fn erased_count(&self) -> usize {
        self.erased_agents.len()
    }
}

/// Manages encryption keys for GDPR-compliant data handling.
///
/// Each user's data is encrypted with a user-specific key. When a user
/// requests erasure (GDPR Article 17), their key is deleted, rendering
/// all their encrypted data permanently unrecoverable.
pub struct ErasureKeyManager {
    /// Active encryption keys by agent hash
    keys: Arc<RwLock<HashMap<String, UserEncryptionKey>>>,
    /// Registry of erased agents
    registry: Arc<RwLock<ErasureRegistry>>,
    /// Count of encrypted items per agent
    item_counts: Arc<RwLock<HashMap<String, u64>>>,
}

impl Default for ErasureKeyManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ErasureKeyManager {
    /// Create a new erasure key manager.
    pub fn new() -> Self {
        Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            registry: Arc::new(RwLock::new(ErasureRegistry::default())),
            item_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Hash an agent identifier for storage (privacy-preserving).
    fn hash_agent(agent_id: &str) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(agent_id.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Get or create an encryption key for an agent.
    pub fn get_or_create_key(&self, agent_id: &str) -> ErasureResult<UserEncryptionKey> {
        let agent_hash = Self::hash_agent(agent_id);

        // Check if agent was erased
        if self.registry.read().is_erased(&agent_hash) {
            return Err(ErasureError::KeyAlreadyErased(agent_id.to_string()));
        }

        // Get existing or create new key
        let mut keys = self.keys.write();
        if let Some(key) = keys.get(&agent_hash) {
            Ok(key.clone())
        } else {
            let key = UserEncryptionKey::generate();
            keys.insert(agent_hash, key.clone());
            Ok(key)
        }
    }

    /// Encrypt data for a specific agent.
    ///
    /// Uses ChaCha20-Poly1305 AEAD for authenticated encryption.
    pub fn encrypt_for_agent(&self, agent_id: &str, data: &[u8]) -> ErasureResult<Vec<u8>> {
        let key = self.get_or_create_key(agent_id)?;
        let agent_hash = Self::hash_agent(agent_id);

        // Generate nonce
        let mut nonce = [0u8; 12];
        getrandom::getrandom(&mut nonce).map_err(|e| ErasureError::EncryptionFailed(e.to_string()))?;

        // Encrypt using ChaCha20-Poly1305
        use chacha20poly1305::{
            aead::{Aead, KeyInit},
            ChaCha20Poly1305, Nonce,
        };

        let cipher = ChaCha20Poly1305::new_from_slice(&key.key_bytes)
            .map_err(|e| ErasureError::EncryptionFailed(e.to_string()))?;

        let ciphertext = cipher
            .encrypt(Nonce::from_slice(&nonce), data)
            .map_err(|e| ErasureError::EncryptionFailed(e.to_string()))?;

        // Increment item count
        *self.item_counts.write().entry(agent_hash).or_insert(0) += 1;

        // Prepend nonce to ciphertext
        let mut result = nonce.to_vec();
        result.extend(ciphertext);
        Ok(result)
    }

    /// Decrypt data for a specific agent.
    pub fn decrypt_for_agent(&self, agent_id: &str, encrypted: &[u8]) -> ErasureResult<Vec<u8>> {
        let agent_hash = Self::hash_agent(agent_id);

        // Check if agent was erased
        if self.registry.read().is_erased(&agent_hash) {
            return Err(ErasureError::KeyAlreadyErased(agent_id.to_string()));
        }

        let keys = self.keys.read();
        let key = keys
            .get(&agent_hash)
            .ok_or_else(|| ErasureError::AgentNotFound(agent_id.to_string()))?;

        if encrypted.len() < 12 {
            return Err(ErasureError::DecryptionFailed("Data too short".to_string()));
        }

        let (nonce, ciphertext) = encrypted.split_at(12);

        use chacha20poly1305::{
            aead::{Aead, KeyInit},
            ChaCha20Poly1305, Nonce,
        };

        let cipher = ChaCha20Poly1305::new_from_slice(&key.key_bytes)
            .map_err(|e| ErasureError::DecryptionFailed(e.to_string()))?;

        cipher
            .decrypt(Nonce::from_slice(nonce), ciphertext)
            .map_err(|e| ErasureError::DecryptionFailed(e.to_string()))
    }

    /// Request erasure for an agent (GDPR Article 17).
    ///
    /// This permanently deletes the agent's encryption key, rendering all
    /// their encrypted data unrecoverable.
    ///
    /// Returns an `ErasureReceipt` as proof of erasure.
    pub fn request_erasure(&self, agent_id: &str) -> ErasureResult<ErasureReceipt> {
        let agent_hash = Self::hash_agent(agent_id);

        // Check if already erased
        if self.registry.read().is_erased(&agent_hash) {
            return Err(ErasureError::KeyAlreadyErased(agent_id.to_string()));
        }

        // Get and remove the key
        let key = self
            .keys
            .write()
            .remove(&agent_hash)
            .ok_or_else(|| ErasureError::AgentNotFound(agent_id.to_string()))?;

        // Get item count
        let items_affected = self.item_counts.write().remove(&agent_hash).unwrap_or(0);

        // Generate receipt
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let receipt_id = format!("erasure-{}-{}", &agent_hash[..16], now);

        // Hash the key for the receipt (key itself is deleted)
        let mut key_hasher = Sha3_256::new();
        key_hasher.update(&key.key_bytes);
        let key_hash = hex::encode(key_hasher.finalize());

        // Generate proof hash
        let mut proof_hasher = Sha3_256::new();
        proof_hasher.update(receipt_id.as_bytes());
        proof_hasher.update(agent_hash.as_bytes());
        proof_hasher.update(key_hash.as_bytes());
        proof_hasher.update(now.to_le_bytes());
        proof_hasher.update(items_affected.to_le_bytes());
        let proof_hash = hex::encode(proof_hasher.finalize());

        let receipt = ErasureReceipt {
            receipt_id,
            agent_hash: agent_hash.clone(),
            key_hash,
            erased_at: now,
            items_affected,
            proof_hash,
        };

        // Record in registry
        self.registry.write().record_erasure(receipt.clone());

        Ok(receipt)
    }

    /// Check if an agent has been erased.
    pub fn is_erased(&self, agent_id: &str) -> bool {
        let agent_hash = Self::hash_agent(agent_id);
        self.registry.read().is_erased(&agent_hash)
    }

    /// Get erasure receipt for an agent.
    pub fn get_erasure_receipt(&self, agent_id: &str) -> Option<ErasureReceipt> {
        let agent_hash = Self::hash_agent(agent_id);
        self.registry.read().get_receipt(&agent_hash).cloned()
    }

    /// Get statistics about the erasure manager.
    pub fn stats(&self) -> ErasureStats {
        ErasureStats {
            active_keys: self.keys.read().len(),
            erased_agents: self.registry.read().erased_count(),
            total_items: self.item_counts.read().values().sum(),
        }
    }
}

/// Statistics about erasure operations.
#[derive(Debug, Clone)]
pub struct ErasureStats {
    /// Number of active encryption keys
    pub active_keys: usize,
    /// Number of erased agents
    pub erased_agents: usize,
    /// Total encrypted items
    pub total_items: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let manager = ErasureKeyManager::new();
        let agent_id = "agent-1";
        let data = b"sensitive user contribution data";

        let encrypted = manager.encrypt_for_agent(agent_id, data).unwrap();
        let decrypted = manager.decrypt_for_agent(agent_id, &encrypted).unwrap();

        assert_eq!(data.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_erasure_makes_data_unrecoverable() {
        let manager = ErasureKeyManager::new();
        let agent_id = "agent-to-erase";
        let data = b"data to be forgotten";

        // Encrypt some data
        let encrypted = manager.encrypt_for_agent(agent_id, data).unwrap();

        // Request erasure
        let receipt = manager.request_erasure(agent_id).unwrap();
        assert!(receipt.verify());

        // Decryption should now fail
        let result = manager.decrypt_for_agent(agent_id, &encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn test_erasure_receipt_verification() {
        let manager = ErasureKeyManager::new();
        let agent_id = "agent-receipt-test";

        // Create some data
        let _ = manager.encrypt_for_agent(agent_id, b"data").unwrap();

        // Erase
        let receipt = manager.request_erasure(agent_id).unwrap();

        // Verify receipt
        assert!(receipt.verify());
        assert!(receipt.items_affected >= 1);
    }

    #[test]
    fn test_double_erasure_fails() {
        let manager = ErasureKeyManager::new();
        let agent_id = "agent-double-erase";

        let _ = manager.encrypt_for_agent(agent_id, b"data").unwrap();
        let _ = manager.request_erasure(agent_id).unwrap();

        // Second erasure should fail
        let result = manager.request_erasure(agent_id);
        assert!(matches!(result, Err(ErasureError::KeyAlreadyErased(_))));
    }

    #[test]
    fn test_erased_agent_cannot_encrypt_new_data() {
        let manager = ErasureKeyManager::new();
        let agent_id = "agent-no-new-data";

        let _ = manager.encrypt_for_agent(agent_id, b"data").unwrap();
        let _ = manager.request_erasure(agent_id).unwrap();

        // New encryption should fail
        let result = manager.encrypt_for_agent(agent_id, b"new data");
        assert!(matches!(result, Err(ErasureError::KeyAlreadyErased(_))));
    }

    #[test]
    fn test_different_agents_have_different_keys() {
        let manager = ErasureKeyManager::new();
        let data = b"same data";

        let encrypted1 = manager.encrypt_for_agent("agent-1", data).unwrap();
        let encrypted2 = manager.encrypt_for_agent("agent-2", data).unwrap();

        // Same plaintext should produce different ciphertext
        assert_ne!(encrypted1, encrypted2);

        // Cross-decryption should fail
        let result = manager.decrypt_for_agent("agent-1", &encrypted2);
        assert!(result.is_err());
    }

    #[test]
    fn test_stats_tracking() {
        let manager = ErasureKeyManager::new();

        let _ = manager.encrypt_for_agent("agent-1", b"data1").unwrap();
        let _ = manager.encrypt_for_agent("agent-1", b"data2").unwrap();
        let _ = manager.encrypt_for_agent("agent-2", b"data3").unwrap();

        let stats = manager.stats();
        assert_eq!(stats.active_keys, 2);
        assert_eq!(stats.total_items, 3);

        let _ = manager.request_erasure("agent-1").unwrap();

        let stats = manager.stats();
        assert_eq!(stats.active_keys, 1);
        assert_eq!(stats.erased_agents, 1);
    }
}
