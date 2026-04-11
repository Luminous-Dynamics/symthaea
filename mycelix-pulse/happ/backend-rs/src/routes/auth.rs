// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Authentication routes
//!
//! Handles user registration and login via Holochain agent keys.
//! DIDs are registered on the DHT, not in any external database.
//!
//! SECURITY: Password hashes are stored in a secure local file encrypted with
//! ChaCha20-Poly1305. The encryption key is derived from an environment variable.

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{AppError, AppResult};
use crate::middleware::{AuthenticatedUser, Claims, create_token};
use crate::routes::AppState;
use crate::types::{ApiError, LoginRequest, LoginResponse, RegisterRequest};

/// Thread-safe credential store for password hashes with encrypted file persistence
#[derive(Clone)]
pub struct CredentialStore {
    /// Map of DID -> password hash (PHC format)
    hashes: Arc<RwLock<HashMap<String, String>>>,
    /// Path to the encrypted credentials file
    storage_path: PathBuf,
    /// Encryption key derived from environment (32 bytes)
    encryption_key: [u8; 32],
}

impl CredentialStore {
    /// Create a new credential store with secure file persistence
    pub fn new() -> Self {
        let storage_path = Self::get_storage_path();
        let encryption_key = Self::derive_encryption_key();

        let store = Self {
            hashes: Arc::new(RwLock::new(HashMap::new())),
            storage_path,
            encryption_key,
        };

        // Load existing credentials from disk
        if let Err(e) = store.load_from_disk_sync() {
            tracing::warn!("Could not load credentials from disk: {}. Starting fresh.", e);
        }

        store
    }

    /// Get the storage path for credentials file
    fn get_storage_path() -> PathBuf {
        let data_dir = std::env::var("MYCELIX_DATA_DIR")
            .unwrap_or_else(|_| {
                // Default to .mycelix-mail in current directory or home
                std::env::var("HOME")
                    .map(|home| format!("{}/.mycelix-mail", home))
                    .unwrap_or_else(|_| ".mycelix-mail".to_string())
            });

        PathBuf::from(data_dir).join("credentials.enc")
    }

    /// Derive encryption key from environment variable
    fn derive_encryption_key() -> [u8; 32] {
        use sha2::{Digest, Sha256};

        let secret = std::env::var("CREDENTIAL_ENCRYPTION_SECRET").unwrap_or_else(|_| {
            panic!(
                "CREDENTIAL_ENCRYPTION_SECRET environment variable must be set.\n\
                 Generate one with: openssl rand -base64 48"
            );
        });
        if secret.len() < 32 {
            panic!("CREDENTIAL_ENCRYPTION_SECRET must be at least 32 characters");
        }

        let mut hasher = Sha256::new();
        hasher.update(b"mycelix-credential-encryption:");
        hasher.update(secret.as_bytes());
        let hash = hasher.finalize();

        let mut key = [0u8; 32];
        key.copy_from_slice(&hash);
        key
    }

    /// Load credentials from encrypted file (synchronous for initialization)
    fn load_from_disk_sync(&self) -> Result<(), Box<dyn std::error::Error>> {
        use crate::services::crypto::decrypt_symmetric;

        if !self.storage_path.exists() {
            tracing::debug!("Credentials file does not exist yet: {:?}", self.storage_path);
            return Ok(());
        }

        let encrypted_data = std::fs::read(&self.storage_path)?;

        if encrypted_data.len() < 12 {
            return Err("Invalid credentials file: too short".into());
        }

        // Extract nonce (first 12 bytes) and ciphertext
        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&encrypted_data[..12]);
        let ciphertext = &encrypted_data[12..];

        let plaintext = decrypt_symmetric(ciphertext, &nonce, &self.encryption_key)
            .map_err(|e| format!("Decryption failed: {}", e))?;

        let hashes: HashMap<String, String> = serde_json::from_slice(&plaintext)?;

        // Update in-memory store using blocking lock for sync context
        let mut store = futures::executor::block_on(self.hashes.write());
        *store = hashes;

        tracing::info!(
            "Loaded {} credential(s) from secure storage",
            store.len()
        );

        Ok(())
    }

    /// Save credentials to encrypted file
    async fn save_to_disk(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::services::crypto::encrypt_symmetric;

        // Ensure parent directory exists
        if let Some(parent) = self.storage_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Serialize hashes to JSON
        let hashes = self.hashes.read().await;
        let json = serde_json::to_vec(&*hashes)?;

        // Encrypt with ChaCha20-Poly1305
        let (ciphertext, nonce) = encrypt_symmetric(&json, &self.encryption_key)
            .map_err(|e| format!("Encryption failed: {}", e))?;

        // Write nonce + ciphertext to file
        let mut encrypted_data = nonce.to_vec();
        encrypted_data.extend(ciphertext);

        tokio::fs::write(&self.storage_path, &encrypted_data).await?;

        tracing::debug!(
            "Saved {} credential(s) to secure storage",
            hashes.len()
        );

        Ok(())
    }

    /// Store a password hash for a DID
    pub async fn store_hash(&self, did: &str, password_hash: &str) {
        {
            let mut hashes = self.hashes.write().await;
            hashes.insert(did.to_string(), password_hash.to_string());
        }

        // Persist to disk
        if let Err(e) = self.save_to_disk().await {
            tracing::error!("Failed to persist credentials to disk: {}", e);
        }

        tracing::debug!("Stored password hash for DID: {}", did);
    }

    /// Retrieve password hash for a DID
    pub async fn get_hash(&self, did: &str) -> Option<String> {
        let hashes = self.hashes.read().await;
        hashes.get(did).cloned()
    }

    /// Check if a DID has a stored password
    pub async fn has_password(&self, did: &str) -> bool {
        let hashes = self.hashes.read().await;
        hashes.contains_key(did)
    }

    /// Remove password hash for a DID
    pub async fn remove_hash(&self, did: &str) {
        {
            let mut hashes = self.hashes.write().await;
            hashes.remove(did);
        }

        // Persist to disk
        if let Err(e) = self.save_to_disk().await {
            tracing::error!("Failed to persist credentials to disk: {}", e);
        }
    }
}

impl Default for CredentialStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Global credential store - lazy initialized with secure file persistence
static CREDENTIAL_STORE: once_cell::sync::Lazy<CredentialStore> =
    once_cell::sync::Lazy::new(CredentialStore::new);

/// Create auth routes
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/register", post(register))
        .route("/login", post(login))
        .route("/me", get(get_current_user))
        .route("/refresh", post(refresh_token))
}

/// Register a new user
///
/// This:
/// 1. Creates a Holochain agent (if not already connected)
/// 2. Registers the DID on the DHT
/// 3. Stores password hash locally (for this backend instance)
/// 4. Returns a JWT token
#[utoipa::path(
    post,
    path = "/auth/register",
    tag = "auth",
    request_body = RegisterRequest,
    responses(
        (status = 200, description = "Registration successful", body = LoginResponse),
        (status = 400, description = "Invalid DID format", body = ApiError),
        (status = 500, description = "Registration failed", body = ApiError)
    )
)]
pub async fn register(
    State(state): State<AppState>,
    Json(input): Json<RegisterRequest>,
) -> AppResult<Json<LoginResponse>> {
    tracing::info!("Registering new user with DID: {}", input.did);

    // Validate DID format
    if !input.did.starts_with("did:") {
        return Err(AppError::ValidationError(
            "DID must start with 'did:'".to_string(),
        ));
    }

    // Ensure connected to Holochain
    state.holochain.connect().await?;

    // Register DID on DHT
    let _hash = state.holochain.register_did(&input.did).await?;

    // Get agent pubkey
    let agent_pubkey = state.holochain.get_agent_pubkey().await?;
    let agent_pubkey_b64 = BASE64.encode(agent_pubkey.get_raw_39());

    // Hash password using Argon2id
    let password_hash = hash_password(&input.password)?;

    // Store password hash in credential store
    // SECURITY NOTE: In production, use persistent encrypted storage
    CREDENTIAL_STORE.store_hash(&input.did, &password_hash).await;
    tracing::info!("Password hash stored for DID: {}", input.did);

    // Create JWT token
    let claims = Claims::new(
        input.did.clone(),
        agent_pubkey_b64.clone(),
        state.config.jwt_expiration_hours,
    );
    let token = create_token(&claims, &state.config.jwt_secret)?;

    tracing::info!("User registered successfully: {}", input.did);

    Ok(Json(LoginResponse {
        token,
        did: input.did,
        agent_pub_key: agent_pubkey_b64,
        expires_at: chrono::DateTime::from_timestamp(claims.exp, 0)
            .unwrap_or_default(),
    }))
}

/// Login an existing user
#[utoipa::path(
    post,
    path = "/auth/login",
    tag = "auth",
    request_body = LoginRequest,
    responses(
        (status = 200, description = "Login successful", body = LoginResponse),
        (status = 401, description = "Invalid credentials", body = ApiError),
        (status = 404, description = "DID not found", body = ApiError)
    )
)]
pub async fn login(
    State(state): State<AppState>,
    Json(input): Json<LoginRequest>,
) -> AppResult<Json<LoginResponse>> {
    tracing::info!("User login attempt: {}", input.did);

    // Ensure connected to Holochain
    state.holochain.connect().await?;

    // Resolve DID to verify it exists
    let agent_pubkey = state.holochain.resolve_did(&input.did).await?;
    let agent_pubkey_b64 = BASE64.encode(agent_pubkey.get_raw_39());

    // Verify password against stored hash
    verify_password(&input.did, &input.password).await?;
    tracing::debug!("Password verified for DID: {}", input.did);

    // Create JWT token
    let claims = Claims::new(
        input.did.clone(),
        agent_pubkey_b64.clone(),
        state.config.jwt_expiration_hours,
    );
    let token = create_token(&claims, &state.config.jwt_secret)?;

    tracing::info!("User logged in successfully: {}", input.did);

    Ok(Json(LoginResponse {
        token,
        did: input.did,
        agent_pub_key: agent_pubkey_b64,
        expires_at: chrono::DateTime::from_timestamp(claims.exp, 0)
            .unwrap_or_default(),
    }))
}

/// Get current user info
#[utoipa::path(
    get,
    path = "/auth/me",
    tag = "auth",
    responses(
        (status = 200, description = "Current user info", body = CurrentUserResponse),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_current_user(
    State(state): State<AppState>,
    user: AuthenticatedUser,
) -> AppResult<Json<CurrentUserResponse>> {
    // Get trust score for self
    let trust = state.trust_cache.get_trust(&user.did).await.ok();

    Ok(Json(CurrentUserResponse {
        did: user.did,
        agent_pub_key: user.agent_pub_key,
        trust_score: trust.as_ref().map(|t| t.score),
        is_verified: true, // They have a valid JWT
    }))
}

/// Refresh JWT token
#[utoipa::path(
    post,
    path = "/auth/refresh",
    tag = "auth",
    responses(
        (status = 200, description = "Token refreshed", body = LoginResponse),
        (status = 401, description = "Invalid token", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn refresh_token(
    State(state): State<AppState>,
    user: AuthenticatedUser,
) -> AppResult<Json<LoginResponse>> {
    // Create new JWT token with fresh expiration
    let claims = Claims::new(
        user.did.clone(),
        user.agent_pub_key.clone(),
        state.config.jwt_expiration_hours,
    );
    let token = create_token(&claims, &state.config.jwt_secret)?;

    Ok(Json(LoginResponse {
        token,
        did: user.did,
        agent_pub_key: user.agent_pub_key,
        expires_at: chrono::DateTime::from_timestamp(claims.exp, 0)
            .unwrap_or_default(),
    }))
}

// Helper types

#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct CurrentUserResponse {
    /// User's DID
    pub did: String,
    /// Agent public key (base64)
    pub agent_pub_key: String,
    /// User's trust score
    pub trust_score: Option<f64>,
    /// Whether JWT is valid
    pub is_verified: bool,
}

/// Hash a password using Argon2id
///
/// Returns the password hash in PHC string format, which includes:
/// - Algorithm identifier (argon2id)
/// - Version
/// - Memory cost, time cost, parallelism
/// - Salt (base64)
/// - Hash (base64)
fn hash_password(password: &str) -> AppResult<String> {
    // Validate password requirements
    if password.len() < 8 {
        return Err(AppError::ValidationError(
            "Password must be at least 8 characters".to_string(),
        ));
    }

    // Generate random salt
    let salt = SaltString::generate(&mut OsRng);

    // Use Argon2id with default secure parameters
    // Default: m=19456 (19 MiB), t=2, p=1
    let argon2 = Argon2::default();

    let hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| AppError::InternalError(format!("Password hashing failed: {}", e)))?;

    Ok(hash.to_string())
}

/// Verify a password against a stored hash
///
/// This function:
/// 1. Retrieves the stored hash from the credential store
/// 2. Parses the PHC format hash string
/// 3. Verifies the password using Argon2id
///
/// Returns an error if:
/// - No password hash is stored for the DID
/// - The password doesn't match the stored hash
/// - The hash format is invalid
async fn verify_password(did: &str, password: &str) -> AppResult<()> {
    // Retrieve stored hash
    let stored_hash = CREDENTIAL_STORE
        .get_hash(did)
        .await
        .ok_or_else(|| {
            tracing::warn!("No password hash found for DID: {}", did);
            AppError::AuthenticationError("Invalid credentials".to_string())
        })?;

    // Parse the PHC format hash
    let parsed_hash = PasswordHash::new(&stored_hash).map_err(|e| {
        tracing::error!("Invalid stored password hash format for DID {}: {}", did, e);
        AppError::InternalError("Password verification failed".to_string())
    })?;

    // Verify password using Argon2id
    Argon2::default()
        .verify_password(password.as_bytes(), &parsed_hash)
        .map_err(|_| {
            tracing::warn!("Password verification failed for DID: {}", did);
            AppError::AuthenticationError("Invalid credentials".to_string())
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_password_valid() {
        let password = "secure_password_123";
        let result = hash_password(password);
        assert!(result.is_ok());

        let hash = result.unwrap();
        // PHC format starts with $argon2id$
        assert!(hash.starts_with("$argon2"));
    }

    #[test]
    fn test_hash_password_too_short() {
        let password = "short";
        let result = hash_password(password);
        assert!(result.is_err());
    }

    #[test]
    fn test_hash_password_different_hashes() {
        let password = "same_password_123";
        let hash1 = hash_password(password).unwrap();
        let hash2 = hash_password(password).unwrap();

        // Same password should produce different hashes (due to random salt)
        assert_ne!(hash1, hash2);
    }

    #[tokio::test]
    async fn test_verify_password_success() {
        let did = "did:test:verify_success";
        let password = "correct_password_123";

        // Hash and store
        let hash = hash_password(password).unwrap();
        CREDENTIAL_STORE.store_hash(did, &hash).await;

        // Verify
        let result = verify_password(did, password).await;
        assert!(result.is_ok());

        // Cleanup
        CREDENTIAL_STORE.remove_hash(did).await;
    }

    #[tokio::test]
    async fn test_verify_password_wrong_password() {
        let did = "did:test:verify_wrong";
        let password = "correct_password_123";
        let wrong_password = "wrong_password_456";

        // Hash and store
        let hash = hash_password(password).unwrap();
        CREDENTIAL_STORE.store_hash(did, &hash).await;

        // Verify with wrong password
        let result = verify_password(did, wrong_password).await;
        assert!(result.is_err());

        // Cleanup
        CREDENTIAL_STORE.remove_hash(did).await;
    }

    #[tokio::test]
    async fn test_verify_password_no_hash_stored() {
        let did = "did:test:no_hash";
        let password = "any_password_123";

        // Verify without storing hash
        let result = verify_password(did, password).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_credential_store_operations() {
        let did = "did:test:store_ops";
        let hash = "test_hash";

        // Store
        CREDENTIAL_STORE.store_hash(did, hash).await;
        assert!(CREDENTIAL_STORE.has_password(did).await);

        // Get
        let retrieved = CREDENTIAL_STORE.get_hash(did).await;
        assert_eq!(retrieved, Some(hash.to_string()));

        // Remove
        CREDENTIAL_STORE.remove_hash(did).await;
        assert!(!CREDENTIAL_STORE.has_password(did).await);
    }
}
