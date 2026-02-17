//! Hardware Security Module (HSM) Integration
//!
//! Provides secure key management and signing operations:
//! - Key generation in HSM
//! - Proof signing
//! - Key attestation
//! - Secure random generation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

// ============================================================================
// HSM Configuration
// ============================================================================

/// HSM provider type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HsmProvider {
    /// Software-based (for testing/development)
    Software,
    /// AWS CloudHSM
    AwsCloudHsm,
    /// Azure Key Vault HSM
    AzureKeyVault,
    /// Google Cloud HSM
    GoogleCloudHsm,
    /// YubiHSM
    YubiHsm,
    /// Thales Luna HSM
    ThalesLuna,
    /// Generic PKCS#11
    Pkcs11,
}

/// HSM configuration
#[derive(Clone, Debug)]
pub struct HsmConfig {
    /// Provider type
    pub provider: HsmProvider,
    /// Connection endpoint
    pub endpoint: Option<String>,
    /// Authentication credentials
    pub credentials: HsmCredentials,
    /// Key slot/partition
    pub slot: Option<u32>,
    /// Connection timeout
    pub timeout: Duration,
    /// Retry attempts
    pub retry_attempts: u32,
    /// Key derivation path (for HD keys)
    pub derivation_path: Option<String>,
}

impl Default for HsmConfig {
    fn default() -> Self {
        Self {
            provider: HsmProvider::Software,
            endpoint: None,
            credentials: HsmCredentials::None,
            slot: None,
            timeout: Duration::from_secs(30),
            retry_attempts: 3,
            derivation_path: None,
        }
    }
}

/// HSM credentials
#[derive(Clone, Debug)]
pub enum HsmCredentials {
    /// No authentication
    None,
    /// PIN-based authentication
    Pin(String),
    /// Certificate-based authentication
    Certificate { cert_path: String, key_path: String },
    /// Token-based authentication
    Token(String),
    /// AWS-style credentials
    Aws { access_key: String, secret_key: String, region: String },
}

// ============================================================================
// Key Types
// ============================================================================

/// Key purpose
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyPurpose {
    /// Proof signing
    ProofSigning,
    /// Proof verification
    ProofVerification,
    /// Commitment generation
    Commitment,
    /// Random generation
    RandomGeneration,
    /// Key derivation master
    KeyDerivation,
}

/// Key algorithm
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyAlgorithm {
    /// Ed25519 signing
    Ed25519,
    /// ECDSA with secp256k1
    EcdsaSecp256k1,
    /// ECDSA with P-256
    EcdsaP256,
    /// AES-256 (for symmetric operations)
    Aes256,
    /// Dilithium (post-quantum)
    Dilithium,
}

/// Key metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyMetadata {
    /// Key identifier
    pub id: String,
    /// Key label/name
    pub label: String,
    /// Key purpose
    pub purpose: KeyPurpose,
    /// Key algorithm
    pub algorithm: KeyAlgorithm,
    /// Creation timestamp
    pub created_at: u64,
    /// Expiration timestamp (if any)
    pub expires_at: Option<u64>,
    /// Whether key is extractable
    pub extractable: bool,
    /// Associated data
    pub metadata: HashMap<String, String>,
}

/// Key handle (opaque reference to HSM key)
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct KeyHandle {
    /// Internal key ID
    pub id: String,
    /// Provider-specific handle
    handle: Vec<u8>,
    /// Key metadata
    pub metadata: KeyMetadata,
}

// ============================================================================
// HSM Operations
// ============================================================================

/// Result of an HSM operation
pub type HsmResult<T> = Result<T, HsmError>;

/// HSM error types
#[derive(Clone, Debug, thiserror::Error)]
pub enum HsmError {
    #[error("HSM connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    #[error("Key not found: {0}")]
    KeyNotFound(String),
    #[error("Operation not supported: {0}")]
    NotSupported(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("HSM operation failed: {0}")]
    OperationFailed(String),
    #[error("Timeout")]
    Timeout,
    #[error("Key generation failed: {0}")]
    KeyGenerationFailed(String),
}

/// Signature result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Signature {
    /// Raw signature bytes
    pub bytes: Vec<u8>,
    /// Algorithm used
    pub algorithm: KeyAlgorithm,
    /// Key ID that signed
    pub key_id: String,
    /// Timestamp of signing
    pub timestamp: u64,
}

/// Key attestation (proof that key was generated in HSM)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyAttestation {
    /// Attestation certificate chain
    pub certificates: Vec<Vec<u8>>,
    /// Attestation statement
    pub statement: Vec<u8>,
    /// Provider-specific attestation data
    pub provider_data: HashMap<String, String>,
}

// ============================================================================
// HSM Trait
// ============================================================================

/// Trait for HSM implementations
#[async_trait]
pub trait Hsm: Send + Sync {
    /// Connect to the HSM
    async fn connect(&mut self) -> HsmResult<()>;

    /// Disconnect from the HSM
    async fn disconnect(&mut self) -> HsmResult<()>;

    /// Check if connected
    fn is_connected(&self) -> bool;

    /// Generate a new key
    async fn generate_key(
        &self,
        label: &str,
        algorithm: KeyAlgorithm,
        purpose: KeyPurpose,
    ) -> HsmResult<KeyHandle>;

    /// Get a key by ID
    async fn get_key(&self, key_id: &str) -> HsmResult<KeyHandle>;

    /// List all keys
    async fn list_keys(&self) -> HsmResult<Vec<KeyMetadata>>;

    /// Delete a key
    async fn delete_key(&self, key_id: &str) -> HsmResult<()>;

    /// Sign data with a key
    async fn sign(&self, key_id: &str, data: &[u8]) -> HsmResult<Signature>;

    /// Verify a signature
    async fn verify(&self, key_id: &str, data: &[u8], signature: &Signature) -> HsmResult<bool>;

    /// Get random bytes from HSM
    async fn get_random(&self, length: usize) -> HsmResult<Vec<u8>>;

    /// Get key attestation
    async fn get_attestation(&self, key_id: &str) -> HsmResult<KeyAttestation>;

    /// Export public key (if allowed)
    async fn export_public_key(&self, key_id: &str) -> HsmResult<Vec<u8>>;
}

// ============================================================================
// Software HSM Implementation
// ============================================================================

/// Software-based HSM (for testing/development)
#[allow(dead_code)]
pub struct SoftwareHsm {
    config: HsmConfig,
    connected: bool,
    keys: Arc<RwLock<HashMap<String, SoftwareKey>>>,
    stats: Arc<RwLock<HsmStats>>,
}

struct SoftwareKey {
    handle: KeyHandle,
    private_key: Vec<u8>,
    public_key: Vec<u8>,
}

/// HSM usage statistics
#[derive(Clone, Debug, Default)]
pub struct HsmStats {
    pub keys_generated: u64,
    pub signatures_created: u64,
    pub signatures_verified: u64,
    pub random_bytes_generated: u64,
}

impl SoftwareHsm {
    /// Create a new software HSM
    pub fn new(config: HsmConfig) -> Self {
        Self {
            config,
            connected: false,
            keys: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HsmStats::default())),
        }
    }

    /// Get statistics
    pub async fn stats(&self) -> HsmStats {
        self.stats.read().await.clone()
    }

    fn generate_key_id(&self) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();
        format!("sw-key-{}", timestamp)
    }
}

#[async_trait]
impl Hsm for SoftwareHsm {
    async fn connect(&mut self) -> HsmResult<()> {
        // Software HSM is always "connected"
        self.connected = true;
        Ok(())
    }

    async fn disconnect(&mut self) -> HsmResult<()> {
        self.connected = false;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    async fn generate_key(
        &self,
        label: &str,
        algorithm: KeyAlgorithm,
        purpose: KeyPurpose,
    ) -> HsmResult<KeyHandle> {
        if !self.connected {
            return Err(HsmError::ConnectionFailed("Not connected".to_string()));
        }

        let key_id = self.generate_key_id();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Generate key material based on algorithm
        let (private_key, public_key) = match algorithm {
            KeyAlgorithm::Ed25519 => {
                // Generate random 32-byte seed
                let seed: [u8; 32] = rand::random();
                let private = seed.to_vec();
                // In real impl, would derive public key from private
                let public = blake3::hash(&seed).as_bytes().to_vec();
                (private, public)
            }
            KeyAlgorithm::EcdsaP256 | KeyAlgorithm::EcdsaSecp256k1 => {
                let private: [u8; 32] = rand::random();
                let public = blake3::hash(&private).as_bytes().to_vec();
                (private.to_vec(), public)
            }
            KeyAlgorithm::Aes256 => {
                let key: [u8; 32] = rand::random();
                (key.to_vec(), Vec::new())
            }
            KeyAlgorithm::Dilithium => {
                // Placeholder for post-quantum key
                let mut private = vec![0u8; 64];
                for i in 0..64 {
                    private[i] = rand::random();
                }
                let public = blake3::hash(&private).as_bytes().to_vec();
                (private, public)
            }
        };

        let metadata = KeyMetadata {
            id: key_id.clone(),
            label: label.to_string(),
            purpose,
            algorithm,
            created_at: timestamp,
            expires_at: None,
            extractable: false,
            metadata: HashMap::new(),
        };

        let handle = KeyHandle {
            id: key_id.clone(),
            handle: key_id.as_bytes().to_vec(),
            metadata: metadata.clone(),
        };

        let software_key = SoftwareKey {
            handle: handle.clone(),
            private_key,
            public_key,
        };

        self.keys.write().await.insert(key_id, software_key);

        let mut stats = self.stats.write().await;
        stats.keys_generated += 1;

        Ok(handle)
    }

    async fn get_key(&self, key_id: &str) -> HsmResult<KeyHandle> {
        let keys = self.keys.read().await;
        keys.get(key_id)
            .map(|k| k.handle.clone())
            .ok_or_else(|| HsmError::KeyNotFound(key_id.to_string()))
    }

    async fn list_keys(&self) -> HsmResult<Vec<KeyMetadata>> {
        let keys = self.keys.read().await;
        Ok(keys.values().map(|k| k.handle.metadata.clone()).collect())
    }

    async fn delete_key(&self, key_id: &str) -> HsmResult<()> {
        let mut keys = self.keys.write().await;
        if keys.remove(key_id).is_some() {
            Ok(())
        } else {
            Err(HsmError::KeyNotFound(key_id.to_string()))
        }
    }

    async fn sign(&self, key_id: &str, data: &[u8]) -> HsmResult<Signature> {
        let (signature_bytes, algorithm) = {
            let keys = self.keys.read().await;
            let key = keys
                .get(key_id)
                .ok_or_else(|| HsmError::KeyNotFound(key_id.to_string()))?;

            // Create signature (simplified - real impl would use proper crypto)
            let mut hasher = blake3::Hasher::new();
            hasher.update(&key.private_key);
            hasher.update(data);
            let sig_bytes = hasher.finalize().as_bytes().to_vec();
            let alg = key.handle.metadata.algorithm;
            (sig_bytes, alg)
        };

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut stats = self.stats.write().await;
        stats.signatures_created += 1;

        Ok(Signature {
            bytes: signature_bytes,
            algorithm,
            key_id: key_id.to_string(),
            timestamp,
        })
    }

    async fn verify(&self, key_id: &str, data: &[u8], signature: &Signature) -> HsmResult<bool> {
        let expected = {
            let keys = self.keys.read().await;
            let key = keys
                .get(key_id)
                .ok_or_else(|| HsmError::KeyNotFound(key_id.to_string()))?;

            // Verify signature (simplified)
            let mut hasher = blake3::Hasher::new();
            hasher.update(&key.private_key);
            hasher.update(data);
            hasher.finalize().as_bytes().to_vec()
        };

        let mut stats = self.stats.write().await;
        stats.signatures_verified += 1;

        Ok(signature.bytes == expected)
    }

    async fn get_random(&self, length: usize) -> HsmResult<Vec<u8>> {
        let mut random_bytes = vec![0u8; length];
        for byte in &mut random_bytes {
            *byte = rand::random();
        }

        let mut stats = self.stats.write().await;
        stats.random_bytes_generated += length as u64;

        Ok(random_bytes)
    }

    async fn get_attestation(&self, key_id: &str) -> HsmResult<KeyAttestation> {
        // Software HSM doesn't support real attestation
        let keys = self.keys.read().await;
        if !keys.contains_key(key_id) {
            return Err(HsmError::KeyNotFound(key_id.to_string()));
        }

        Ok(KeyAttestation {
            certificates: Vec::new(),
            statement: b"software-hsm-attestation".to_vec(),
            provider_data: {
                let mut data = HashMap::new();
                data.insert("provider".to_string(), "software".to_string());
                data.insert("note".to_string(), "Not cryptographically verifiable".to_string());
                data
            },
        })
    }

    async fn export_public_key(&self, key_id: &str) -> HsmResult<Vec<u8>> {
        let keys = self.keys.read().await;
        let key = keys
            .get(key_id)
            .ok_or_else(|| HsmError::KeyNotFound(key_id.to_string()))?;

        if key.public_key.is_empty() {
            Err(HsmError::NotSupported(
                "Key does not have exportable public key".to_string(),
            ))
        } else {
            Ok(key.public_key.clone())
        }
    }
}

// ============================================================================
// Proof Signer
// ============================================================================

/// Proof signer using HSM
pub struct ProofSigner {
    hsm: Arc<dyn Hsm>,
    signing_key: Option<String>,
}

impl ProofSigner {
    /// Create a new proof signer
    pub fn new(hsm: Arc<dyn Hsm>) -> Self {
        Self {
            hsm,
            signing_key: None,
        }
    }

    /// Initialize with a new signing key
    pub async fn initialize(&mut self, label: &str) -> HsmResult<KeyHandle> {
        let handle = self
            .hsm
            .generate_key(label, KeyAlgorithm::Ed25519, KeyPurpose::ProofSigning)
            .await?;
        self.signing_key = Some(handle.id.clone());
        Ok(handle)
    }

    /// Use an existing signing key
    pub fn use_key(&mut self, key_id: &str) {
        self.signing_key = Some(key_id.to_string());
    }

    /// Sign proof bytes
    pub async fn sign_proof(&self, proof_bytes: &[u8]) -> HsmResult<SignedProof> {
        let key_id = self
            .signing_key
            .as_ref()
            .ok_or_else(|| HsmError::InvalidParameters("No signing key configured".to_string()))?;

        // Create proof hash
        let proof_hash = blake3::hash(proof_bytes);

        // Sign the hash
        let signature = self.hsm.sign(key_id, proof_hash.as_bytes()).await?;

        Ok(SignedProof {
            proof_bytes: proof_bytes.to_vec(),
            proof_hash: proof_hash.as_bytes().to_vec(),
            signature,
        })
    }

    /// Verify a signed proof
    pub async fn verify_signed_proof(&self, signed_proof: &SignedProof) -> HsmResult<bool> {
        // Verify hash
        let expected_hash = blake3::hash(&signed_proof.proof_bytes);
        if expected_hash.as_bytes() != signed_proof.proof_hash.as_slice() {
            return Ok(false);
        }

        // Verify signature
        self.hsm
            .verify(
                &signed_proof.signature.key_id,
                &signed_proof.proof_hash,
                &signed_proof.signature,
            )
            .await
    }
}

/// A signed proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignedProof {
    /// Original proof bytes
    pub proof_bytes: Vec<u8>,
    /// Hash of proof
    pub proof_hash: Vec<u8>,
    /// HSM signature
    pub signature: Signature,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_software_hsm_connect() {
        let mut hsm = SoftwareHsm::new(HsmConfig::default());
        assert!(!hsm.is_connected());

        hsm.connect().await.unwrap();
        assert!(hsm.is_connected());

        hsm.disconnect().await.unwrap();
        assert!(!hsm.is_connected());
    }

    #[tokio::test]
    async fn test_generate_key() {
        let mut hsm = SoftwareHsm::new(HsmConfig::default());
        hsm.connect().await.unwrap();

        let handle = hsm
            .generate_key("test-key", KeyAlgorithm::Ed25519, KeyPurpose::ProofSigning)
            .await
            .unwrap();

        assert!(!handle.id.is_empty());
        assert_eq!(handle.metadata.label, "test-key");
        assert_eq!(handle.metadata.algorithm, KeyAlgorithm::Ed25519);
    }

    #[tokio::test]
    async fn test_list_keys() {
        let mut hsm = SoftwareHsm::new(HsmConfig::default());
        hsm.connect().await.unwrap();

        hsm.generate_key("key1", KeyAlgorithm::Ed25519, KeyPurpose::ProofSigning)
            .await
            .unwrap();
        hsm.generate_key("key2", KeyAlgorithm::EcdsaP256, KeyPurpose::Commitment)
            .await
            .unwrap();

        let keys = hsm.list_keys().await.unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[tokio::test]
    async fn test_sign_and_verify() {
        let mut hsm = SoftwareHsm::new(HsmConfig::default());
        hsm.connect().await.unwrap();

        let handle = hsm
            .generate_key("signing-key", KeyAlgorithm::Ed25519, KeyPurpose::ProofSigning)
            .await
            .unwrap();

        let data = b"test data to sign";
        let signature = hsm.sign(&handle.id, data).await.unwrap();

        assert!(!signature.bytes.is_empty());
        assert_eq!(signature.key_id, handle.id);

        let valid = hsm.verify(&handle.id, data, &signature).await.unwrap();
        assert!(valid);

        // Verify fails with wrong data
        let invalid = hsm.verify(&handle.id, b"wrong data", &signature).await.unwrap();
        assert!(!invalid);
    }

    #[tokio::test]
    async fn test_get_random() {
        let mut hsm = SoftwareHsm::new(HsmConfig::default());
        hsm.connect().await.unwrap();

        let random1 = hsm.get_random(32).await.unwrap();
        let random2 = hsm.get_random(32).await.unwrap();

        assert_eq!(random1.len(), 32);
        assert_eq!(random2.len(), 32);
        assert_ne!(random1, random2);
    }

    #[tokio::test]
    async fn test_delete_key() {
        let mut hsm = SoftwareHsm::new(HsmConfig::default());
        hsm.connect().await.unwrap();

        let handle = hsm
            .generate_key("to-delete", KeyAlgorithm::Ed25519, KeyPurpose::ProofSigning)
            .await
            .unwrap();

        // Key exists
        assert!(hsm.get_key(&handle.id).await.is_ok());

        // Delete
        hsm.delete_key(&handle.id).await.unwrap();

        // Key no longer exists
        assert!(hsm.get_key(&handle.id).await.is_err());
    }

    #[tokio::test]
    async fn test_export_public_key() {
        let mut hsm = SoftwareHsm::new(HsmConfig::default());
        hsm.connect().await.unwrap();

        let handle = hsm
            .generate_key("export-test", KeyAlgorithm::Ed25519, KeyPurpose::ProofSigning)
            .await
            .unwrap();

        let public_key = hsm.export_public_key(&handle.id).await.unwrap();
        assert!(!public_key.is_empty());
    }

    #[tokio::test]
    async fn test_attestation() {
        let mut hsm = SoftwareHsm::new(HsmConfig::default());
        hsm.connect().await.unwrap();

        let handle = hsm
            .generate_key("attest-key", KeyAlgorithm::Ed25519, KeyPurpose::ProofSigning)
            .await
            .unwrap();

        let attestation = hsm.get_attestation(&handle.id).await.unwrap();
        assert!(!attestation.statement.is_empty());
    }

    #[tokio::test]
    async fn test_proof_signer() {
        let mut hsm = SoftwareHsm::new(HsmConfig::default());
        hsm.connect().await.unwrap();

        let hsm_arc: Arc<dyn Hsm> = Arc::new(hsm);
        let mut signer = ProofSigner::new(Arc::clone(&hsm_arc));

        signer.initialize("proof-signing-key").await.unwrap();

        let proof_bytes = b"fake proof data for testing";
        let signed = signer.sign_proof(proof_bytes).await.unwrap();

        assert_eq!(signed.proof_bytes, proof_bytes);
        assert!(!signed.signature.bytes.is_empty());

        let valid = signer.verify_signed_proof(&signed).await.unwrap();
        assert!(valid);
    }

    #[tokio::test]
    async fn test_hsm_stats() {
        let mut hsm = SoftwareHsm::new(HsmConfig::default());
        hsm.connect().await.unwrap();

        let stats_before = hsm.stats().await;
        assert_eq!(stats_before.keys_generated, 0);

        let handle = hsm
            .generate_key("stats-key", KeyAlgorithm::Ed25519, KeyPurpose::ProofSigning)
            .await
            .unwrap();

        let _ = hsm.sign(&handle.id, b"test").await.unwrap();
        let _ = hsm.get_random(16).await.unwrap();

        let stats_after = hsm.stats().await;
        assert_eq!(stats_after.keys_generated, 1);
        assert_eq!(stats_after.signatures_created, 1);
        assert!(stats_after.random_bytes_generated >= 16);
    }

    #[test]
    fn test_hsm_config_default() {
        let config = HsmConfig::default();
        assert_eq!(config.provider, HsmProvider::Software);
        assert_eq!(config.retry_attempts, 3);
    }

    #[test]
    fn test_key_metadata() {
        let metadata = KeyMetadata {
            id: "test-id".to_string(),
            label: "test-label".to_string(),
            purpose: KeyPurpose::ProofSigning,
            algorithm: KeyAlgorithm::Ed25519,
            created_at: 0,
            expires_at: None,
            extractable: false,
            metadata: HashMap::new(),
        };

        assert_eq!(metadata.label, "test-label");
        assert_eq!(metadata.purpose, KeyPurpose::ProofSigning);
    }
}
