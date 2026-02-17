//! Hardware Security Module (HSM) Integration
//!
//! Provides secure key management and cryptographic operations via HSM.
//!
//! ## Supported HSMs
//!
//! - AWS CloudHSM
//! - Azure Key Vault
//! - HashiCorp Vault
//! - PKCS#11 compatible devices
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::hsm::{HsmProvider, VaultHsm};
//!
//! // Initialize Vault HSM
//! let hsm = VaultHsm::new("https://vault.example.com", "my-token").await?;
//!
//! // Sign proof commitment
//! let signature = hsm.sign("proof-signing-key", &commitment).await?;
//!
//! // Verify signature
//! let valid = hsm.verify("proof-signing-key", &commitment, &signature).await?;
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::proofs::{ProofError, ProofResult};

/// Key type for HSM operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyType {
    /// ECDSA P-256
    EcdsaP256,
    /// ECDSA P-384
    EcdsaP384,
    /// Ed25519
    Ed25519,
    /// RSA 2048
    Rsa2048,
    /// RSA 4096
    Rsa4096,
    /// AES-256 (symmetric)
    Aes256,
}

impl Default for KeyType {
    fn default() -> Self {
        Self::Ed25519
    }
}

/// Key metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetadata {
    /// Key identifier
    pub id: String,
    /// Key type
    pub key_type: KeyType,
    /// Creation timestamp
    pub created_at: u64,
    /// Last used timestamp
    pub last_used_at: Option<u64>,
    /// Key version (for rotation)
    pub version: u32,
    /// Whether key is enabled
    pub enabled: bool,
    /// Custom labels
    pub labels: HashMap<String, String>,
}

/// HSM operation result
#[derive(Debug, Clone)]
pub struct HsmResult<T> {
    /// Operation result
    pub data: T,
    /// Operation latency in milliseconds
    pub latency_ms: u64,
    /// HSM device used
    pub device_id: Option<String>,
}

/// Signature from HSM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HsmSignature {
    /// Raw signature bytes
    pub bytes: Vec<u8>,
    /// Key ID used
    pub key_id: String,
    /// Key version
    pub key_version: u32,
    /// Algorithm used
    pub algorithm: String,
}

/// HSM provider trait
#[async_trait]
pub trait HsmProvider: Send + Sync {
    /// Get provider name
    fn name(&self) -> &str;

    /// Check if HSM is available
    async fn is_available(&self) -> bool;

    /// List available keys
    async fn list_keys(&self) -> ProofResult<Vec<KeyMetadata>>;

    /// Get key metadata
    async fn get_key(&self, key_id: &str) -> ProofResult<KeyMetadata>;

    /// Generate a new key
    async fn generate_key(
        &self,
        key_id: &str,
        key_type: KeyType,
        labels: HashMap<String, String>,
    ) -> ProofResult<KeyMetadata>;

    /// Sign data with a key
    async fn sign(&self, key_id: &str, data: &[u8]) -> ProofResult<HsmSignature>;

    /// Verify a signature
    async fn verify(
        &self,
        key_id: &str,
        data: &[u8],
        signature: &HsmSignature,
    ) -> ProofResult<bool>;

    /// Encrypt data
    async fn encrypt(&self, key_id: &str, plaintext: &[u8]) -> ProofResult<Vec<u8>>;

    /// Decrypt data
    async fn decrypt(&self, key_id: &str, ciphertext: &[u8]) -> ProofResult<Vec<u8>>;

    /// Rotate a key
    async fn rotate_key(&self, key_id: &str) -> ProofResult<KeyMetadata>;

    /// Delete a key (if allowed)
    async fn delete_key(&self, key_id: &str) -> ProofResult<()>;
}

/// HSM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HsmConfig {
    /// Provider type
    pub provider: HsmProviderType,
    /// Connection endpoint
    pub endpoint: String,
    /// Authentication method
    pub auth: HsmAuth,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries
    pub max_retries: u32,
    /// Enable audit logging
    pub audit_logging: bool,
}

impl Default for HsmConfig {
    fn default() -> Self {
        Self {
            provider: HsmProviderType::Vault,
            endpoint: "http://localhost:8200".to_string(),
            auth: HsmAuth::Token("".to_string()),
            timeout_secs: 30,
            max_retries: 3,
            audit_logging: true,
        }
    }
}

/// HSM provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HsmProviderType {
    /// HashiCorp Vault
    Vault,
    /// AWS CloudHSM
    AwsCloudHsm,
    /// Azure Key Vault
    AzureKeyVault,
    /// PKCS#11
    Pkcs11,
    /// Local software HSM (for testing)
    SoftHsm,
}

/// HSM authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HsmAuth {
    /// Token-based auth
    Token(String),
    /// AppRole (Vault)
    AppRole { role_id: String, secret_id: String },
    /// AWS IAM
    AwsIam { role_arn: String },
    /// Azure Managed Identity
    AzureManagedIdentity,
    /// Certificate-based
    Certificate { cert_path: String, key_path: String },
    /// PKCS#11 PIN
    Pkcs11Pin(String),
}

/// HashiCorp Vault HSM implementation
pub struct VaultHsm {
    config: HsmConfig,
    client: reqwest::Client,
    transit_path: String,
}

impl VaultHsm {
    /// Create a new Vault HSM client
    pub fn new(endpoint: &str, token: &str) -> ProofResult<Self> {
        let config = HsmConfig {
            provider: HsmProviderType::Vault,
            endpoint: endpoint.to_string(),
            auth: HsmAuth::Token(token.to_string()),
            ..Default::default()
        };

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| ProofError::HsmError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            config,
            client,
            transit_path: "transit".to_string(),
        })
    }

    /// Set the transit engine path
    pub fn with_transit_path(mut self, path: &str) -> Self {
        self.transit_path = path.to_string();
        self
    }

    fn get_token(&self) -> ProofResult<&str> {
        match &self.config.auth {
            HsmAuth::Token(token) => Ok(token),
            _ => Err(ProofError::HsmError("Vault requires token auth".to_string())),
        }
    }
}

#[async_trait]
impl HsmProvider for VaultHsm {
    fn name(&self) -> &str {
        "HashiCorp Vault"
    }

    async fn is_available(&self) -> bool {
        let url = format!("{}/v1/sys/health", self.config.endpoint);
        self.client.get(&url).send().await.is_ok()
    }

    async fn list_keys(&self) -> ProofResult<Vec<KeyMetadata>> {
        let url = format!(
            "{}/v1/{}/keys",
            self.config.endpoint, self.transit_path
        );

        let response = self
            .client
            .request(reqwest::Method::from_bytes(b"LIST").unwrap(), &url)
            .header("X-Vault-Token", self.get_token()?)
            .send()
            .await
            .map_err(|e| ProofError::HsmError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ProofError::HsmError(format!(
                "List keys failed: {}",
                response.status()
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ProofError::HsmError(format!("Failed to parse response: {}", e)))?;

        let keys = body["data"]["keys"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|id| KeyMetadata {
                        id: id.to_string(),
                        key_type: KeyType::Ed25519, // Would need to query each key
                        created_at: 0,
                        last_used_at: None,
                        version: 1,
                        enabled: true,
                        labels: HashMap::new(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(keys)
    }

    async fn get_key(&self, key_id: &str) -> ProofResult<KeyMetadata> {
        let url = format!(
            "{}/v1/{}/keys/{}",
            self.config.endpoint, self.transit_path, key_id
        );

        let response = self
            .client
            .get(&url)
            .header("X-Vault-Token", self.get_token()?)
            .send()
            .await
            .map_err(|e| ProofError::HsmError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ProofError::HsmError(format!(
                "Get key failed: {}",
                response.status()
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ProofError::HsmError(format!("Failed to parse response: {}", e)))?;

        let data = &body["data"];

        Ok(KeyMetadata {
            id: key_id.to_string(),
            key_type: match data["type"].as_str() {
                Some("ed25519") => KeyType::Ed25519,
                Some("ecdsa-p256") => KeyType::EcdsaP256,
                Some("ecdsa-p384") => KeyType::EcdsaP384,
                Some("rsa-2048") => KeyType::Rsa2048,
                Some("rsa-4096") => KeyType::Rsa4096,
                Some("aes256-gcm96") => KeyType::Aes256,
                _ => KeyType::Ed25519,
            },
            created_at: 0,
            last_used_at: None,
            version: data["latest_version"].as_u64().unwrap_or(1) as u32,
            enabled: !data["deletion_allowed"].as_bool().unwrap_or(false),
            labels: HashMap::new(),
        })
    }

    async fn generate_key(
        &self,
        key_id: &str,
        key_type: KeyType,
        _labels: HashMap<String, String>,
    ) -> ProofResult<KeyMetadata> {
        let url = format!(
            "{}/v1/{}/keys/{}",
            self.config.endpoint, self.transit_path, key_id
        );

        let vault_type = match key_type {
            KeyType::Ed25519 => "ed25519",
            KeyType::EcdsaP256 => "ecdsa-p256",
            KeyType::EcdsaP384 => "ecdsa-p384",
            KeyType::Rsa2048 => "rsa-2048",
            KeyType::Rsa4096 => "rsa-4096",
            KeyType::Aes256 => "aes256-gcm96",
        };

        let body = serde_json::json!({
            "type": vault_type,
            "exportable": false,
        });

        let response = self
            .client
            .post(&url)
            .header("X-Vault-Token", self.get_token()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProofError::HsmError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ProofError::HsmError(format!(
                "Generate key failed: {}",
                response.status()
            )));
        }

        Ok(KeyMetadata {
            id: key_id.to_string(),
            key_type,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            last_used_at: None,
            version: 1,
            enabled: true,
            labels: HashMap::new(),
        })
    }

    async fn sign(&self, key_id: &str, data: &[u8]) -> ProofResult<HsmSignature> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};

        let url = format!(
            "{}/v1/{}/sign/{}",
            self.config.endpoint, self.transit_path, key_id
        );

        let input = STANDARD.encode(data);

        let body = serde_json::json!({
            "input": input,
        });

        let response = self
            .client
            .post(&url)
            .header("X-Vault-Token", self.get_token()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProofError::HsmError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ProofError::HsmError(format!(
                "Sign failed: {}",
                response.status()
            )));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ProofError::HsmError(format!("Failed to parse response: {}", e)))?;

        let signature_str = result["data"]["signature"]
            .as_str()
            .ok_or_else(|| ProofError::HsmError("Missing signature in response".to_string()))?;

        // Vault format: vault:v1:base64_signature
        let parts: Vec<&str> = signature_str.split(':').collect();
        if parts.len() != 3 {
            return Err(ProofError::HsmError("Invalid signature format".to_string()));
        }

        let sig_bytes = STANDARD
            .decode(parts[2])
            .map_err(|e| ProofError::HsmError(format!("Failed to decode signature: {}", e)))?;

        let version: u32 = parts[1]
            .strip_prefix('v')
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        Ok(HsmSignature {
            bytes: sig_bytes,
            key_id: key_id.to_string(),
            key_version: version,
            algorithm: "ed25519".to_string(),
        })
    }

    async fn verify(
        &self,
        key_id: &str,
        data: &[u8],
        signature: &HsmSignature,
    ) -> ProofResult<bool> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};

        let url = format!(
            "{}/v1/{}/verify/{}",
            self.config.endpoint, self.transit_path, key_id
        );

        let input = STANDARD.encode(data);
        let sig_b64 = STANDARD.encode(&signature.bytes);
        let vault_sig = format!("vault:v{}:{}", signature.key_version, sig_b64);

        let body = serde_json::json!({
            "input": input,
            "signature": vault_sig,
        });

        let response = self
            .client
            .post(&url)
            .header("X-Vault-Token", self.get_token()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProofError::HsmError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ProofError::HsmError(format!(
                "Verify failed: {}",
                response.status()
            )));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ProofError::HsmError(format!("Failed to parse response: {}", e)))?;

        Ok(result["data"]["valid"].as_bool().unwrap_or(false))
    }

    async fn encrypt(&self, key_id: &str, plaintext: &[u8]) -> ProofResult<Vec<u8>> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};

        let url = format!(
            "{}/v1/{}/encrypt/{}",
            self.config.endpoint, self.transit_path, key_id
        );

        let input = STANDARD.encode(plaintext);

        let body = serde_json::json!({
            "plaintext": input,
        });

        let response = self
            .client
            .post(&url)
            .header("X-Vault-Token", self.get_token()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProofError::HsmError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ProofError::HsmError(format!(
                "Encrypt failed: {}",
                response.status()
            )));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ProofError::HsmError(format!("Failed to parse response: {}", e)))?;

        let ciphertext = result["data"]["ciphertext"]
            .as_str()
            .ok_or_else(|| ProofError::HsmError("Missing ciphertext".to_string()))?;

        Ok(ciphertext.as_bytes().to_vec())
    }

    async fn decrypt(&self, key_id: &str, ciphertext: &[u8]) -> ProofResult<Vec<u8>> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};

        let url = format!(
            "{}/v1/{}/decrypt/{}",
            self.config.endpoint, self.transit_path, key_id
        );

        let ciphertext_str = std::str::from_utf8(ciphertext)
            .map_err(|e| ProofError::HsmError(format!("Invalid ciphertext: {}", e)))?;

        let body = serde_json::json!({
            "ciphertext": ciphertext_str,
        });

        let response = self
            .client
            .post(&url)
            .header("X-Vault-Token", self.get_token()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProofError::HsmError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ProofError::HsmError(format!(
                "Decrypt failed: {}",
                response.status()
            )));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ProofError::HsmError(format!("Failed to parse response: {}", e)))?;

        let plaintext_b64 = result["data"]["plaintext"]
            .as_str()
            .ok_or_else(|| ProofError::HsmError("Missing plaintext".to_string()))?;

        STANDARD
            .decode(plaintext_b64)
            .map_err(|e| ProofError::HsmError(format!("Failed to decode plaintext: {}", e)))
    }

    async fn rotate_key(&self, key_id: &str) -> ProofResult<KeyMetadata> {
        let url = format!(
            "{}/v1/{}/keys/{}/rotate",
            self.config.endpoint, self.transit_path, key_id
        );

        let response = self
            .client
            .post(&url)
            .header("X-Vault-Token", self.get_token()?)
            .send()
            .await
            .map_err(|e| ProofError::HsmError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ProofError::HsmError(format!(
                "Rotate key failed: {}",
                response.status()
            )));
        }

        // Fetch updated key info
        self.get_key(key_id).await
    }

    async fn delete_key(&self, key_id: &str) -> ProofResult<()> {
        // First, enable deletion
        let config_url = format!(
            "{}/v1/{}/keys/{}/config",
            self.config.endpoint, self.transit_path, key_id
        );

        let config_body = serde_json::json!({
            "deletion_allowed": true,
        });

        self.client
            .post(&config_url)
            .header("X-Vault-Token", self.get_token()?)
            .json(&config_body)
            .send()
            .await
            .map_err(|e| ProofError::HsmError(format!("Request failed: {}", e)))?;

        // Then delete
        let url = format!(
            "{}/v1/{}/keys/{}",
            self.config.endpoint, self.transit_path, key_id
        );

        let response = self
            .client
            .delete(&url)
            .header("X-Vault-Token", self.get_token()?)
            .send()
            .await
            .map_err(|e| ProofError::HsmError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ProofError::HsmError(format!(
                "Delete key failed: {}",
                response.status()
            )));
        }

        Ok(())
    }
}

/// Software HSM for testing (NOT for production)
pub struct SoftHsm {
    keys: std::sync::RwLock<HashMap<String, SoftKey>>,
}

struct SoftKey {
    metadata: KeyMetadata,
    private_key: Vec<u8>,
}

impl SoftHsm {
    /// Create a new software HSM for testing
    pub fn new() -> Self {
        Self {
            keys: std::sync::RwLock::new(HashMap::new()),
        }
    }
}

impl Default for SoftHsm {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl HsmProvider for SoftHsm {
    fn name(&self) -> &str {
        "SoftHSM (Testing Only)"
    }

    async fn is_available(&self) -> bool {
        true
    }

    async fn list_keys(&self) -> ProofResult<Vec<KeyMetadata>> {
        let keys = self.keys.read().unwrap();
        Ok(keys.values().map(|k| k.metadata.clone()).collect())
    }

    async fn get_key(&self, key_id: &str) -> ProofResult<KeyMetadata> {
        let keys = self.keys.read().unwrap();
        keys.get(key_id)
            .map(|k| k.metadata.clone())
            .ok_or_else(|| ProofError::HsmError(format!("Key not found: {}", key_id)))
    }

    async fn generate_key(
        &self,
        key_id: &str,
        key_type: KeyType,
        labels: HashMap<String, String>,
    ) -> ProofResult<KeyMetadata> {
        use rand::RngCore;

        let mut private_key = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut private_key);

        let metadata = KeyMetadata {
            id: key_id.to_string(),
            key_type,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            last_used_at: None,
            version: 1,
            enabled: true,
            labels,
        };

        let soft_key = SoftKey {
            metadata: metadata.clone(),
            private_key,
        };

        self.keys.write().unwrap().insert(key_id.to_string(), soft_key);

        Ok(metadata)
    }

    async fn sign(&self, key_id: &str, data: &[u8]) -> ProofResult<HsmSignature> {
        use sha2::{Sha256, Digest};

        let keys = self.keys.read().unwrap();
        let key = keys
            .get(key_id)
            .ok_or_else(|| ProofError::HsmError(format!("Key not found: {}", key_id)))?;

        // Simple HMAC-like signature for testing
        let mut hasher = Sha256::new();
        hasher.update(&key.private_key);
        hasher.update(data);
        let sig: [u8; 32] = hasher.finalize().into();

        Ok(HsmSignature {
            bytes: sig.to_vec(),
            key_id: key_id.to_string(),
            key_version: key.metadata.version,
            algorithm: "hmac-sha256".to_string(),
        })
    }

    async fn verify(
        &self,
        key_id: &str,
        data: &[u8],
        signature: &HsmSignature,
    ) -> ProofResult<bool> {
        let expected = self.sign(key_id, data).await?;
        Ok(expected.bytes == signature.bytes)
    }

    async fn encrypt(&self, key_id: &str, plaintext: &[u8]) -> ProofResult<Vec<u8>> {
        let keys = self.keys.read().unwrap();
        let key = keys
            .get(key_id)
            .ok_or_else(|| ProofError::HsmError(format!("Key not found: {}", key_id)))?;

        // Simple XOR "encryption" for testing only
        let ciphertext: Vec<u8> = plaintext
            .iter()
            .zip(key.private_key.iter().cycle())
            .map(|(p, k)| p ^ k)
            .collect();

        Ok(ciphertext)
    }

    async fn decrypt(&self, key_id: &str, ciphertext: &[u8]) -> ProofResult<Vec<u8>> {
        // XOR is its own inverse
        self.encrypt(key_id, ciphertext).await
    }

    async fn rotate_key(&self, key_id: &str) -> ProofResult<KeyMetadata> {
        use rand::RngCore;

        let mut keys = self.keys.write().unwrap();
        let key = keys
            .get_mut(key_id)
            .ok_or_else(|| ProofError::HsmError(format!("Key not found: {}", key_id)))?;

        rand::thread_rng().fill_bytes(&mut key.private_key);
        key.metadata.version += 1;

        Ok(key.metadata.clone())
    }

    async fn delete_key(&self, key_id: &str) -> ProofResult<()> {
        let mut keys = self.keys.write().unwrap();
        keys.remove(key_id)
            .map(|_| ())
            .ok_or_else(|| ProofError::HsmError(format!("Key not found: {}", key_id)))
    }
}

/// Create an HSM provider from configuration
pub fn create_provider(config: HsmConfig) -> ProofResult<Arc<dyn HsmProvider>> {
    match config.provider {
        HsmProviderType::Vault => {
            let token = match &config.auth {
                HsmAuth::Token(t) => t.clone(),
                _ => return Err(ProofError::HsmError("Vault requires token auth".to_string())),
            };
            Ok(Arc::new(VaultHsm::new(&config.endpoint, &token)?))
        }
        HsmProviderType::SoftHsm => Ok(Arc::new(SoftHsm::new())),
        _ => Err(ProofError::HsmError(format!(
            "Provider {:?} not yet implemented",
            config.provider
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_soft_hsm_key_lifecycle() {
        let hsm = SoftHsm::new();

        // Generate key
        let metadata = hsm
            .generate_key("test-key", KeyType::Ed25519, HashMap::new())
            .await
            .unwrap();

        assert_eq!(metadata.id, "test-key");
        assert_eq!(metadata.version, 1);

        // List keys
        let keys = hsm.list_keys().await.unwrap();
        assert_eq!(keys.len(), 1);

        // Get key
        let fetched = hsm.get_key("test-key").await.unwrap();
        assert_eq!(fetched.id, "test-key");

        // Delete key
        hsm.delete_key("test-key").await.unwrap();

        // Should be gone
        let keys = hsm.list_keys().await.unwrap();
        assert_eq!(keys.len(), 0);
    }

    #[tokio::test]
    async fn test_soft_hsm_sign_verify() {
        let hsm = SoftHsm::new();

        hsm.generate_key("sign-key", KeyType::Ed25519, HashMap::new())
            .await
            .unwrap();

        let data = b"hello world";
        let signature = hsm.sign("sign-key", data).await.unwrap();

        assert!(hsm.verify("sign-key", data, &signature).await.unwrap());

        // Different data should fail
        assert!(!hsm.verify("sign-key", b"different", &signature).await.unwrap());
    }

    #[tokio::test]
    async fn test_soft_hsm_encrypt_decrypt() {
        let hsm = SoftHsm::new();

        hsm.generate_key("enc-key", KeyType::Aes256, HashMap::new())
            .await
            .unwrap();

        let plaintext = b"secret message";
        let ciphertext = hsm.encrypt("enc-key", plaintext).await.unwrap();

        assert_ne!(&ciphertext, plaintext);

        let decrypted = hsm.decrypt("enc-key", &ciphertext).await.unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[tokio::test]
    async fn test_soft_hsm_rotate_key() {
        let hsm = SoftHsm::new();

        hsm.generate_key("rotate-key", KeyType::Ed25519, HashMap::new())
            .await
            .unwrap();

        let data = b"test data";
        let sig1 = hsm.sign("rotate-key", data).await.unwrap();

        // Rotate
        let metadata = hsm.rotate_key("rotate-key").await.unwrap();
        assert_eq!(metadata.version, 2);

        // New signature should be different
        let sig2 = hsm.sign("rotate-key", data).await.unwrap();
        assert_ne!(sig1.bytes, sig2.bytes);
    }

    #[test]
    fn test_key_type_default() {
        assert_eq!(KeyType::default(), KeyType::Ed25519);
    }

    #[test]
    fn test_hsm_config_default() {
        let config = HsmConfig::default();
        assert_eq!(config.provider, HsmProviderType::Vault);
        assert_eq!(config.timeout_secs, 30);
    }
}
