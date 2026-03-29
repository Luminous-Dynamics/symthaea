// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-End Encryption Module
//!
//! PGP/OpenPGP support, key management, and encrypted message handling

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Key Management
// ============================================================================

pub struct KeyManager {
    pool: PgPool,
    keyring: Keyring,
}

impl KeyManager {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            keyring: Keyring::new(),
        }
    }

    /// Generate a new key pair for a user
    pub async fn generate_key_pair(
        &self,
        user_id: Uuid,
        params: KeyGenerationParams,
    ) -> Result<KeyPair, EncryptionError> {
        // Validate parameters
        if params.key_size < 2048 {
            return Err(EncryptionError::InvalidKeySize(
                "Key size must be at least 2048 bits".into(),
            ));
        }

        // Generate key pair (would use actual crypto library)
        let key_pair = self.keyring.generate_pgp_key(
            &params.name,
            &params.email,
            params.key_size,
            params.algorithm,
        )?;

        // Store in database
        sqlx::query(
            r#"
            INSERT INTO user_keys (id, user_id, public_key, private_key_encrypted,
                                   fingerprint, algorithm, key_size, created_at, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(user_id)
        .bind(&key_pair.public_key)
        .bind(&key_pair.private_key_encrypted)
        .bind(&key_pair.fingerprint)
        .bind(params.algorithm.to_string())
        .bind(params.key_size as i32)
        .bind(Utc::now())
        .bind(params.expires_at)
        .execute(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        Ok(key_pair)
    }

    /// Import an existing public key
    pub async fn import_public_key(
        &self,
        user_id: Uuid,
        armored_key: &str,
    ) -> Result<PublicKeyInfo, EncryptionError> {
        let key_info = self.keyring.parse_public_key(armored_key)?;

        sqlx::query(
            r#"
            INSERT INTO contact_keys (id, user_id, email, public_key, fingerprint,
                                      verified, trust_level, imported_at)
            VALUES ($1, $2, $3, $4, $5, false, 'unknown', $6)
            ON CONFLICT (user_id, fingerprint) DO UPDATE SET
                public_key = EXCLUDED.public_key,
                imported_at = EXCLUDED.imported_at
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(user_id)
        .bind(&key_info.email)
        .bind(armored_key)
        .bind(&key_info.fingerprint)
        .bind(Utc::now())
        .execute(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        Ok(key_info)
    }

    /// Get user's public key
    pub async fn get_public_key(&self, user_id: Uuid) -> Result<Option<String>, EncryptionError> {
        let result: Option<(String,)> = sqlx::query_as(
            "SELECT public_key FROM user_keys WHERE user_id = $1 AND revoked = false ORDER BY created_at DESC LIMIT 1",
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        Ok(result.map(|(key,)| key))
    }

    /// Get public key for a contact by email
    pub async fn get_contact_key(
        &self,
        user_id: Uuid,
        email: &str,
    ) -> Result<Option<ContactKey>, EncryptionError> {
        let result: Option<ContactKey> = sqlx::query_as(
            r#"
            SELECT email, public_key, fingerprint, verified, trust_level, imported_at
            FROM contact_keys
            WHERE user_id = $1 AND email = $2
            ORDER BY imported_at DESC LIMIT 1
            "#,
        )
        .bind(user_id)
        .bind(email)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        Ok(result)
    }

    /// Set trust level for a contact's key
    pub async fn set_key_trust(
        &self,
        user_id: Uuid,
        fingerprint: &str,
        trust_level: TrustLevel,
        verified: bool,
    ) -> Result<(), EncryptionError> {
        sqlx::query(
            "UPDATE contact_keys SET trust_level = $1, verified = $2 WHERE user_id = $3 AND fingerprint = $4",
        )
        .bind(trust_level.to_string())
        .bind(verified)
        .bind(user_id)
        .bind(fingerprint)
        .execute(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        Ok(())
    }

    /// Revoke a user's key
    pub async fn revoke_key(
        &self,
        user_id: Uuid,
        fingerprint: &str,
        reason: &str,
    ) -> Result<(), EncryptionError> {
        sqlx::query(
            "UPDATE user_keys SET revoked = true, revocation_reason = $1, revoked_at = $2 WHERE user_id = $3 AND fingerprint = $4",
        )
        .bind(reason)
        .bind(Utc::now())
        .bind(user_id)
        .bind(fingerprint)
        .execute(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        Ok(())
    }

    /// Publish key to key server
    pub async fn publish_to_keyserver(
        &self,
        user_id: Uuid,
        keyserver: &str,
    ) -> Result<(), EncryptionError> {
        let public_key = self
            .get_public_key(user_id)
            .await?
            .ok_or(EncryptionError::KeyNotFound)?;

        // Would make HTTP request to keyserver
        // For now, just mark as published
        sqlx::query(
            "UPDATE user_keys SET published_to = array_append(published_to, $1) WHERE user_id = $2",
        )
        .bind(keyserver)
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// Encryption Service
// ============================================================================

pub struct EncryptionService {
    pool: PgPool,
    key_manager: KeyManager,
    keyring: Keyring,
}

impl EncryptionService {
    pub fn new(pool: PgPool) -> Self {
        let key_manager = KeyManager::new(pool.clone());
        Self {
            pool,
            key_manager,
            keyring: Keyring::new(),
        }
    }

    /// Encrypt a message for one or more recipients
    pub async fn encrypt_message(
        &self,
        user_id: Uuid,
        content: &str,
        recipients: &[String],
        options: EncryptionOptions,
    ) -> Result<EncryptedMessage, EncryptionError> {
        let mut recipient_keys = Vec::new();

        // Gather recipient public keys
        for email in recipients {
            if let Some(contact_key) = self.key_manager.get_contact_key(user_id, email).await? {
                if contact_key.trust_level == TrustLevel::Untrusted {
                    return Err(EncryptionError::UntrustedKey(email.clone()));
                }
                recipient_keys.push(contact_key.public_key);
            } else {
                return Err(EncryptionError::KeyNotFound);
            }
        }

        // Also include sender's key so they can read sent messages
        if let Some(sender_key) = self.key_manager.get_public_key(user_id).await? {
            recipient_keys.push(sender_key);
        }

        // Encrypt the content
        let encrypted_content = self.keyring.encrypt_pgp(content, &recipient_keys)?;

        // Sign if requested
        let signature = if options.sign {
            Some(self.sign_content(user_id, content).await?)
        } else {
            None
        };

        Ok(EncryptedMessage {
            encrypted_content,
            signature,
            algorithm: "PGP".to_string(),
            recipients: recipients.to_vec(),
            encrypted_at: Utc::now(),
        })
    }

    /// Decrypt a message
    pub async fn decrypt_message(
        &self,
        user_id: Uuid,
        encrypted_content: &str,
        passphrase: &str,
    ) -> Result<DecryptedMessage, EncryptionError> {
        // Get user's private key
        let private_key: Option<(String,)> = sqlx::query_as(
            "SELECT private_key_encrypted FROM user_keys WHERE user_id = $1 AND revoked = false ORDER BY created_at DESC LIMIT 1",
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        let private_key = private_key.ok_or(EncryptionError::KeyNotFound)?.0;

        // Decrypt
        let decrypted = self
            .keyring
            .decrypt_pgp(encrypted_content, &private_key, passphrase)?;

        Ok(DecryptedMessage {
            content: decrypted,
            decrypted_at: Utc::now(),
            signature_verified: None,
        })
    }

    /// Sign content with user's private key
    pub async fn sign_content(
        &self,
        user_id: Uuid,
        content: &str,
    ) -> Result<String, EncryptionError> {
        // Get private key (would need passphrase in real implementation)
        let private_key: Option<(String,)> = sqlx::query_as(
            "SELECT private_key_encrypted FROM user_keys WHERE user_id = $1 AND revoked = false ORDER BY created_at DESC LIMIT 1",
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        let private_key = private_key.ok_or(EncryptionError::KeyNotFound)?.0;

        self.keyring.sign_pgp(content, &private_key)
    }

    /// Verify a signature
    pub async fn verify_signature(
        &self,
        user_id: Uuid,
        content: &str,
        signature: &str,
        signer_email: &str,
    ) -> Result<SignatureVerification, EncryptionError> {
        let contact_key = self
            .key_manager
            .get_contact_key(user_id, signer_email)
            .await?
            .ok_or(EncryptionError::KeyNotFound)?;

        let valid = self
            .keyring
            .verify_pgp(content, signature, &contact_key.public_key)?;

        Ok(SignatureVerification {
            valid,
            signer_email: signer_email.to_string(),
            signer_fingerprint: contact_key.fingerprint,
            trust_level: contact_key.trust_level,
            verified_at: Utc::now(),
        })
    }

    /// Encrypt an attachment
    pub async fn encrypt_attachment(
        &self,
        user_id: Uuid,
        data: &[u8],
        recipients: &[String],
    ) -> Result<Vec<u8>, EncryptionError> {
        let mut recipient_keys = Vec::new();

        for email in recipients {
            if let Some(contact_key) = self.key_manager.get_contact_key(user_id, email).await? {
                recipient_keys.push(contact_key.public_key);
            }
        }

        self.keyring.encrypt_binary(&recipient_keys, data)
    }

    /// Check if a contact has a public key available
    pub async fn can_encrypt_to(&self, user_id: Uuid, email: &str) -> Result<bool, EncryptionError> {
        Ok(self
            .key_manager
            .get_contact_key(user_id, email)
            .await?
            .is_some())
    }
}

// ============================================================================
// Self-Destructing Messages
// ============================================================================

pub struct SelfDestructService {
    pool: PgPool,
}

impl SelfDestructService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a self-destructing message
    pub async fn create_self_destruct_message(
        &self,
        user_id: Uuid,
        email_id: Uuid,
        config: SelfDestructConfig,
    ) -> Result<(), EncryptionError> {
        let expires_at = match config.destruct_type {
            DestructType::AfterRead { delay_seconds } => None, // Calculated on read
            DestructType::AtTime { time } => Some(time),
            DestructType::AfterDuration { seconds } => Some(Utc::now() + chrono::Duration::seconds(seconds as i64)),
        };

        sqlx::query(
            r#"
            INSERT INTO self_destruct_messages
                (id, email_id, user_id, destruct_type, max_reads, current_reads,
                 delay_after_read_seconds, expires_at, created_at)
            VALUES ($1, $2, $3, $4, $5, 0, $6, $7, $8)
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(email_id)
        .bind(user_id)
        .bind(config.destruct_type.to_string())
        .bind(config.max_reads)
        .bind(config.delay_after_read_seconds)
        .bind(expires_at)
        .bind(Utc::now())
        .execute(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        Ok(())
    }

    /// Record a read and check if message should be destroyed
    pub async fn record_read(&self, email_id: Uuid) -> Result<DestructStatus, EncryptionError> {
        let record: Option<SelfDestructRecord> = sqlx::query_as(
            r#"
            UPDATE self_destruct_messages
            SET current_reads = current_reads + 1,
                first_read_at = COALESCE(first_read_at, NOW())
            WHERE email_id = $1
            RETURNING destruct_type, max_reads, current_reads, delay_after_read_seconds,
                      expires_at, first_read_at
            "#,
        )
        .bind(email_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        let record = match record {
            Some(r) => r,
            None => return Ok(DestructStatus::NotSelfDestruct),
        };

        // Check if should destruct
        if let Some(max_reads) = record.max_reads {
            if record.current_reads >= max_reads {
                self.destroy_message(email_id).await?;
                return Ok(DestructStatus::Destroyed);
            }
        }

        if let Some(expires_at) = record.expires_at {
            if Utc::now() >= expires_at {
                self.destroy_message(email_id).await?;
                return Ok(DestructStatus::Destroyed);
            }
        }

        // Calculate destruction time for "after read" type
        if record.destruct_type == "after_read" {
            if let (Some(first_read), Some(delay)) = (record.first_read_at, record.delay_after_read_seconds) {
                let destruct_at = first_read + chrono::Duration::seconds(delay as i64);
                if Utc::now() >= destruct_at {
                    self.destroy_message(email_id).await?;
                    return Ok(DestructStatus::Destroyed);
                }
                return Ok(DestructStatus::WillDestruct { at: destruct_at });
            }
        }

        Ok(DestructStatus::Active {
            reads_remaining: record.max_reads.map(|m| m - record.current_reads),
            expires_at: record.expires_at,
        })
    }

    async fn destroy_message(&self, email_id: Uuid) -> Result<(), EncryptionError> {
        // Delete the email content (keep metadata for audit)
        sqlx::query(
            "UPDATE emails SET body_text = '[Message destroyed]', body_html = NULL, destroyed_at = NOW() WHERE id = $1",
        )
        .bind(email_id)
        .execute(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        // Delete attachments
        sqlx::query("DELETE FROM attachments WHERE email_id = $1")
            .bind(email_id)
            .execute(&self.pool)
            .await
            .map_err(|e| EncryptionError::Database(e.to_string()))?;

        Ok(())
    }

    /// Cleanup expired self-destruct messages (run periodically)
    pub async fn cleanup_expired(&self) -> Result<u64, EncryptionError> {
        let result = sqlx::query(
            r#"
            SELECT email_id FROM self_destruct_messages
            WHERE expires_at IS NOT NULL AND expires_at < NOW()
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| EncryptionError::Database(e.to_string()))?;

        let count = result.len() as u64;

        for row in result {
            let email_id: Uuid = row.get("email_id");
            self.destroy_message(email_id).await?;
        }

        Ok(count)
    }
}

// ============================================================================
// Keyring (Crypto Operations - would use actual PGP library)
// ============================================================================

struct Keyring;

impl Keyring {
    fn new() -> Self {
        Self
    }

    fn generate_pgp_key(
        &self,
        name: &str,
        email: &str,
        key_size: u32,
        algorithm: KeyAlgorithm,
    ) -> Result<KeyPair, EncryptionError> {
        // In real implementation, would use sequoia-pgp or similar
        Ok(KeyPair {
            public_key: format!("-----BEGIN PGP PUBLIC KEY BLOCK-----\n...\n-----END PGP PUBLIC KEY BLOCK-----"),
            private_key_encrypted: format!("-----BEGIN PGP PRIVATE KEY BLOCK-----\n...\n-----END PGP PRIVATE KEY BLOCK-----"),
            fingerprint: format!("{:040X}", uuid::Uuid::new_v4().as_u128()),
        })
    }

    fn parse_public_key(&self, armored_key: &str) -> Result<PublicKeyInfo, EncryptionError> {
        // Parse PGP key to extract info
        Ok(PublicKeyInfo {
            fingerprint: format!("{:040X}", uuid::Uuid::new_v4().as_u128()),
            email: "user@example.com".to_string(),
            name: Some("User".to_string()),
            created_at: Utc::now(),
            expires_at: None,
            algorithm: KeyAlgorithm::RSA,
            key_size: 4096,
        })
    }

    fn encrypt_pgp(&self, content: &str, recipient_keys: &[String]) -> Result<String, EncryptionError> {
        // Would use actual PGP encryption
        Ok(format!(
            "-----BEGIN PGP MESSAGE-----\n{}\n-----END PGP MESSAGE-----",
            base64::encode(content)
        ))
    }

    fn decrypt_pgp(
        &self,
        encrypted: &str,
        private_key: &str,
        passphrase: &str,
    ) -> Result<String, EncryptionError> {
        // Would use actual PGP decryption
        // For now, just extract base64 content (mock)
        let content = encrypted
            .lines()
            .filter(|l| !l.starts_with("-----"))
            .collect::<String>();
        String::from_utf8(base64::decode(&content).unwrap_or_default())
            .map_err(|_| EncryptionError::DecryptionFailed)
    }

    fn sign_pgp(&self, content: &str, private_key: &str) -> Result<String, EncryptionError> {
        Ok(format!(
            "-----BEGIN PGP SIGNATURE-----\n{}\n-----END PGP SIGNATURE-----",
            base64::encode(content.as_bytes())
        ))
    }

    fn verify_pgp(
        &self,
        content: &str,
        signature: &str,
        public_key: &str,
    ) -> Result<bool, EncryptionError> {
        // Would verify actual signature
        Ok(true)
    }

    fn encrypt_binary(&self, recipient_keys: &[String], data: &[u8]) -> Result<Vec<u8>, EncryptionError> {
        // Would encrypt binary data
        Ok(data.to_vec())
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyGenerationParams {
    pub name: String,
    pub email: String,
    pub key_size: u32,
    pub algorithm: KeyAlgorithm,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum KeyAlgorithm {
    RSA,
    Ed25519,
    Curve25519,
}

impl ToString for KeyAlgorithm {
    fn to_string(&self) -> String {
        match self {
            KeyAlgorithm::RSA => "RSA".to_string(),
            KeyAlgorithm::Ed25519 => "Ed25519".to_string(),
            KeyAlgorithm::Curve25519 => "Curve25519".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeyPair {
    pub public_key: String,
    pub private_key_encrypted: String,
    pub fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicKeyInfo {
    pub fingerprint: String,
    pub email: String,
    pub name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub algorithm: KeyAlgorithm,
    pub key_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ContactKey {
    pub email: String,
    pub public_key: String,
    pub fingerprint: String,
    pub verified: bool,
    pub trust_level: TrustLevel,
    pub imported_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "text")]
pub enum TrustLevel {
    Untrusted,
    Unknown,
    Marginal,
    Full,
    Ultimate,
}

impl ToString for TrustLevel {
    fn to_string(&self) -> String {
        match self {
            TrustLevel::Untrusted => "untrusted".to_string(),
            TrustLevel::Unknown => "unknown".to_string(),
            TrustLevel::Marginal => "marginal".to_string(),
            TrustLevel::Full => "full".to_string(),
            TrustLevel::Ultimate => "ultimate".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionOptions {
    pub sign: bool,
    pub compress: bool,
    pub armor: bool,
}

impl Default for EncryptionOptions {
    fn default() -> Self {
        Self {
            sign: true,
            compress: true,
            armor: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedMessage {
    pub encrypted_content: String,
    pub signature: Option<String>,
    pub algorithm: String,
    pub recipients: Vec<String>,
    pub encrypted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecryptedMessage {
    pub content: String,
    pub decrypted_at: DateTime<Utc>,
    pub signature_verified: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureVerification {
    pub valid: bool,
    pub signer_email: String,
    pub signer_fingerprint: String,
    pub trust_level: TrustLevel,
    pub verified_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfDestructConfig {
    pub destruct_type: DestructType,
    pub max_reads: Option<i32>,
    pub delay_after_read_seconds: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DestructType {
    AfterRead { delay_seconds: i32 },
    AtTime { time: DateTime<Utc> },
    AfterDuration { seconds: i32 },
}

impl ToString for DestructType {
    fn to_string(&self) -> String {
        match self {
            DestructType::AfterRead { .. } => "after_read".to_string(),
            DestructType::AtTime { .. } => "at_time".to_string(),
            DestructType::AfterDuration { .. } => "after_duration".to_string(),
        }
    }
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct SelfDestructRecord {
    destruct_type: String,
    max_reads: Option<i32>,
    current_reads: i32,
    delay_after_read_seconds: Option<i32>,
    expires_at: Option<DateTime<Utc>>,
    first_read_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DestructStatus {
    NotSelfDestruct,
    Active {
        reads_remaining: Option<i32>,
        expires_at: Option<DateTime<Utc>>,
    },
    WillDestruct {
        at: DateTime<Utc>,
    },
    Destroyed,
}

#[derive(Debug, thiserror::Error)]
pub enum EncryptionError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Key not found")]
    KeyNotFound,
    #[error("Invalid key size: {0}")]
    InvalidKeySize(String),
    #[error("Untrusted key for: {0}")]
    UntrustedKey(String),
    #[error("Decryption failed")]
    DecryptionFailed,
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("Key parsing error: {0}")]
    KeyParsingError(String),
}

// Trait for row.get
use sqlx::Row;
