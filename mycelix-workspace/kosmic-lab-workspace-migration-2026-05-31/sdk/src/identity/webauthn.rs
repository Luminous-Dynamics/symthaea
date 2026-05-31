// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! WebAuthn/FIDO2 Integration for Mycelix Identity System
//!
//! This module provides WebAuthn (Web Authentication) support for hardware key
//! authentication, enabling passwordless and phishing-resistant authentication
//! using security keys (YubiKey, SoloKey, etc.) and platform authenticators.
//!
//! # WebAuthn Flow Overview
//!
//! ## Registration (Credential Creation)
//! ```text
//! 1. Server creates RegistrationChallenge with:
//!    - Random challenge bytes
//!    - Relying party (RP) info
//!    - User info
//!    - Attestation preference
//!
//! 2. Client/authenticator creates credential:
//!    - Generates keypair bound to RP
//!    - Signs challenge with private key
//!    - Returns attestation object
//!
//! 3. Server verifies:
//!    - Challenge matches
//!    - Origin matches RP
//!    - Attestation signature valid
//!    - Stores public key + credential ID
//! ```
//!
//! ## Authentication (Assertion)
//! ```text
//! 1. Server creates AuthenticationChallenge with:
//!    - Random challenge bytes
//!    - Allowed credential IDs
//!    - User verification requirement
//!
//! 2. Client/authenticator creates assertion:
//!    - Signs challenge + authenticator data
//!    - Increments sign counter
//!    - Returns signature
//!
//! 3. Server verifies:
//!    - Signature over clientDataHash + authData
//!    - Sign counter > stored counter (replay protection)
//!    - Updates stored counter
//! ```
//!
//! # Security Properties
//!
//! - **Phishing Resistance**: Origin bound - credentials only work for registered domain
//! - **Replay Protection**: Sign counter prevents credential cloning detection
//! - **User Verification**: Optional PIN/biometric for 2FA
//! - **Attestation**: Hardware attestation proves authenticator type
//!
//! # Integration with Byzantine Identity
//!
//! WebAuthn credentials can be bound to Mycelix identities (DIDs) to provide:
//! - Hardware-backed authentication for high-stakes operations
//! - Additional verification factor for governance actions
//! - Recovery mechanism via multiple registered authenticators

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Core Types
// ============================================================================

/// WebAuthn credential for hardware key binding
///
/// Represents a registered credential that can be used for authentication.
/// The credential_id and public_key are provided by the authenticator during
/// registration and must be stored securely.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub struct WebAuthnCredential {
    /// Unique identifier for this credential (assigned by authenticator)
    /// This is used to identify the credential during authentication
    pub credential_id: Vec<u8>,

    /// COSE-encoded public key from the authenticator
    /// Used to verify signatures during authentication
    pub public_key: Vec<u8>,

    /// Signature counter for replay detection
    /// Must be incremented by authenticator on each use
    pub sign_count: u32,

    /// User handle (opaque identifier for the user)
    /// Can be the Mycelix DID or agent public key
    pub user_handle: Vec<u8>,

    /// Attestation format used during registration
    pub attestation_format: AttestationFormat,

    /// Unix timestamp of credential creation
    pub created_at: u64,

    /// Optional friendly name for the credential
    pub friendly_name: Option<String>,

    /// AAGUID of the authenticator (if available)
    /// Can be used to identify authenticator make/model
    pub aaguid: Option<Vec<u8>>,

    /// Transports supported by the authenticator
    pub transports: Vec<AuthenticatorTransport>,

    /// Whether this credential requires user verification
    pub user_verification_required: bool,
}

impl WebAuthnCredential {
    /// Create a new credential from registration response
    pub fn new(
        credential_id: Vec<u8>,
        public_key: Vec<u8>,
        user_handle: Vec<u8>,
        attestation_format: AttestationFormat,
    ) -> Self {
        Self {
            credential_id,
            public_key,
            sign_count: 0,
            user_handle,
            attestation_format,
            created_at: current_timestamp(),
            friendly_name: None,
            aaguid: None,
            transports: vec![],
            user_verification_required: false,
        }
    }

    /// Set a friendly name for the credential
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.friendly_name = Some(name.into());
        self
    }

    /// Set the AAGUID
    pub fn with_aaguid(mut self, aaguid: Vec<u8>) -> Self {
        self.aaguid = Some(aaguid);
        self
    }

    /// Set supported transports
    pub fn with_transports(mut self, transports: Vec<AuthenticatorTransport>) -> Self {
        self.transports = transports;
        self
    }

    /// Update sign counter after successful authentication
    pub fn update_sign_count(&mut self, new_count: u32) -> Result<(), WebAuthnError> {
        if new_count <= self.sign_count && new_count != 0 {
            return Err(WebAuthnError::SignCounterNotIncremented {
                stored: self.sign_count,
                received: new_count,
            });
        }
        self.sign_count = new_count;
        Ok(())
    }

    /// Check if sign counter is valid (replay protection)
    pub fn validate_sign_count(&self, new_count: u32) -> bool {
        // Counter of 0 from authenticator means counter not supported
        if new_count == 0 {
            return true;
        }
        new_count > self.sign_count
    }

    /// Get credential age in seconds
    pub fn age_seconds(&self) -> u64 {
        current_timestamp().saturating_sub(self.created_at)
    }
}

/// Attestation format indicating how the authenticator's attestation statement
/// should be verified
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub enum AttestationFormat {
    /// No attestation provided
    #[default]
    None,
    /// Packed attestation (common for hardware keys)
    Packed,
    /// TPM attestation (Trusted Platform Module)
    Tpm,
    /// Android Key Attestation
    AndroidKey,
    /// Android SafetyNet attestation
    AndroidSafetyNet,
    /// FIDO U2F attestation (legacy)
    FidoU2f,
    /// Apple attestation
    Apple,
}

/// Transport mechanisms supported by authenticators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub enum AuthenticatorTransport {
    /// USB transport
    Usb,
    /// NFC transport
    Nfc,
    /// Bluetooth Low Energy
    Ble,
    /// Internal/platform authenticator
    Internal,
    /// Hybrid (CTAP 2.2 cross-device authentication)
    Hybrid,
}

// ============================================================================
// Challenge Types
// ============================================================================

/// Registration challenge sent to client to initiate credential creation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub struct RegistrationChallenge {
    /// Random challenge bytes (at least 16 bytes, recommended 32)
    pub challenge: Vec<u8>,

    /// Relying party identifier (typically the domain)
    pub rp_id: String,

    /// Human-readable relying party name
    pub rp_name: String,

    /// User identifier (opaque, could be DID bytes)
    pub user_id: Vec<u8>,

    /// Human-readable username for display
    pub user_name: String,

    /// Display name for the user
    pub user_display_name: String,

    /// Timeout in milliseconds for the operation
    pub timeout_ms: u32,

    /// Attestation preference
    pub attestation_preference: AttestationConveyance,

    /// Authenticator selection criteria
    pub authenticator_selection: AuthenticatorSelectionCriteria,

    /// Credentials to exclude (prevent re-registration)
    pub exclude_credentials: Vec<Vec<u8>>,

    /// Unix timestamp when challenge was created
    pub created_at: u64,

    /// Unix timestamp when challenge expires
    pub expires_at: u64,
}

impl RegistrationChallenge {
    /// Check if challenge has expired
    pub fn is_expired(&self) -> bool {
        current_timestamp() > self.expires_at
    }

    /// Get remaining time until expiration in seconds
    pub fn remaining_seconds(&self) -> u64 {
        self.expires_at.saturating_sub(current_timestamp())
    }
}

/// Attestation conveyance preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub enum AttestationConveyance {
    /// No attestation required
    #[default]
    None,
    /// Attestation may be anonymized by the client
    Indirect,
    /// Request direct attestation from authenticator
    Direct,
    /// Request enterprise attestation (requires AAGUID allowlist)
    Enterprise,
}

/// Criteria for selecting authenticators during registration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub struct AuthenticatorSelectionCriteria {
    /// Attachment modality (platform vs cross-platform)
    pub authenticator_attachment: Option<AuthenticatorAttachment>,

    /// Require resident/discoverable credential
    pub resident_key: ResidentKeyRequirement,

    /// User verification requirement
    pub user_verification: UserVerification,
}

impl Default for AuthenticatorSelectionCriteria {
    fn default() -> Self {
        Self {
            authenticator_attachment: None,
            resident_key: ResidentKeyRequirement::Discouraged,
            user_verification: UserVerification::Preferred,
        }
    }
}

/// Authenticator attachment modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub enum AuthenticatorAttachment {
    /// Platform authenticator (Touch ID, Windows Hello, etc.)
    Platform,
    /// Roaming/cross-platform authenticator (YubiKey, etc.)
    CrossPlatform,
}

/// Resident key (discoverable credential) requirement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub enum ResidentKeyRequirement {
    /// Resident key not required but preferred
    #[default]
    Discouraged,
    /// Resident key preferred but not required
    Preferred,
    /// Resident key required (for usernameless login)
    Required,
}

/// User verification requirement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub enum UserVerification {
    /// UV required (PIN or biometric must be verified)
    Required,
    /// UV preferred but not required
    #[default]
    Preferred,
    /// UV discouraged (skip if possible)
    Discouraged,
}

/// Authentication challenge for assertion
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub struct AuthenticationChallenge {
    /// Random challenge bytes
    pub challenge: Vec<u8>,

    /// Relying party identifier
    pub rp_id: String,

    /// List of allowed credential IDs
    pub allowed_credentials: Vec<AllowedCredential>,

    /// Timeout in milliseconds
    pub timeout_ms: u32,

    /// User verification requirement
    pub user_verification: UserVerification,

    /// Unix timestamp when challenge was created
    pub created_at: u64,

    /// Unix timestamp when challenge expires
    pub expires_at: u64,
}

impl AuthenticationChallenge {
    /// Check if challenge has expired
    pub fn is_expired(&self) -> bool {
        current_timestamp() > self.expires_at
    }
}

/// Allowed credential descriptor for authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub struct AllowedCredential {
    /// Credential identifier
    pub id: Vec<u8>,
    /// Allowed transports (hints for client)
    pub transports: Vec<AuthenticatorTransport>,
}

// ============================================================================
// Response Types
// ============================================================================

/// Registration response from authenticator (attestation)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub struct RegistrationResponse {
    /// Credential ID assigned by authenticator
    pub credential_id: Vec<u8>,

    /// Client data JSON (base64url encoded by client)
    pub client_data_json: Vec<u8>,

    /// Attestation object (CBOR encoded)
    pub attestation_object: Vec<u8>,

    /// Transports reported by authenticator
    pub transports: Vec<AuthenticatorTransport>,
}

impl RegistrationResponse {
    /// Compute SHA-256 hash of client data
    pub fn client_data_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.client_data_json);
        hasher.finalize().into()
    }
}

/// Authentication response from authenticator (assertion)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub struct AuthenticationResponse {
    /// Credential ID used for authentication
    pub credential_id: Vec<u8>,

    /// Client data JSON
    pub client_data_json: Vec<u8>,

    /// Authenticator data (includes RP ID hash, flags, counter)
    pub authenticator_data: Vec<u8>,

    /// Signature over authenticator_data || client_data_hash
    pub signature: Vec<u8>,

    /// User handle (optional, returned if credential is resident)
    pub user_handle: Option<Vec<u8>>,
}

impl AuthenticationResponse {
    /// Compute SHA-256 hash of client data
    pub fn client_data_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.client_data_json);
        hasher.finalize().into()
    }

    /// Get the data that was signed (auth_data || client_data_hash)
    pub fn signed_data(&self) -> Vec<u8> {
        let mut data = self.authenticator_data.clone();
        data.extend_from_slice(&self.client_data_hash());
        data
    }

    /// Parse sign counter from authenticator data (bytes 33-36)
    pub fn parse_sign_count(&self) -> Option<u32> {
        if self.authenticator_data.len() < 37 {
            return None;
        }
        let bytes: [u8; 4] = self.authenticator_data[33..37].try_into().ok()?;
        Some(u32::from_be_bytes(bytes))
    }

    /// Parse flags byte from authenticator data
    pub fn parse_flags(&self) -> Option<AuthenticatorFlags> {
        if self.authenticator_data.is_empty() {
            return None;
        }
        // Flags are at byte 32 (after 32-byte RP ID hash)
        if self.authenticator_data.len() < 33 {
            return None;
        }
        Some(AuthenticatorFlags::from_byte(self.authenticator_data[32]))
    }
}

/// Parsed authenticator flags
#[derive(Debug, Clone, Copy)]
pub struct AuthenticatorFlags {
    /// User was present (UP flag)
    pub user_present: bool,
    /// User was verified (UV flag)
    pub user_verified: bool,
    /// Backup eligibility (BE flag, CTAP 2.1)
    pub backup_eligible: bool,
    /// Backup state (BS flag, CTAP 2.1)
    pub backed_up: bool,
    /// Attested credential data included (AT flag)
    pub attested_credential_data: bool,
    /// Extension data included (ED flag)
    pub extension_data: bool,
}

impl AuthenticatorFlags {
    /// Parse flags from a single byte
    pub fn from_byte(byte: u8) -> Self {
        Self {
            user_present: byte & 0x01 != 0,
            user_verified: byte & 0x04 != 0,
            backup_eligible: byte & 0x08 != 0,
            backed_up: byte & 0x10 != 0,
            attested_credential_data: byte & 0x40 != 0,
            extension_data: byte & 0x80 != 0,
        }
    }
}

/// Result of successful authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub struct AuthenticationResult {
    /// The credential that was authenticated
    pub credential_id: Vec<u8>,

    /// User handle associated with credential
    pub user_handle: Vec<u8>,

    /// New sign count (should be stored)
    pub new_sign_count: u32,

    /// Whether user verification was performed
    pub user_verified: bool,

    /// Timestamp of authentication
    pub authenticated_at: u64,
}

// ============================================================================
// Error Types
// ============================================================================

/// WebAuthn operation errors
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/identity/"))]
pub enum WebAuthnError {
    /// Challenge has expired
    ChallengeExpired,

    /// Challenge was not found
    ChallengeNotFound,

    /// Origin mismatch (potential phishing)
    OriginMismatch {
        /// Expected origin.
        expected: String,
        /// Received origin.
        received: String,
    },

    /// RP ID mismatch
    RpIdMismatch {
        /// Expected relying party ID.
        expected: String,
        /// Received relying party ID.
        received: String,
    },

    /// Sign counter did not increment (potential cloned authenticator)
    SignCounterNotIncremented {
        /// Previously stored counter value.
        stored: u32,
        /// Received counter value.
        received: u32,
    },

    /// Credential not found
    CredentialNotFound {
        /// Credential identifier that was not found.
        credential_id: Vec<u8>,
    },

    /// User verification required but not performed
    UserVerificationRequired,

    /// User presence required but not confirmed
    UserPresenceRequired,

    /// Invalid attestation format
    InvalidAttestationFormat(String),

    /// Attestation verification failed
    AttestationVerificationFailed(String),

    /// Signature verification failed
    SignatureVerificationFailed(String),

    /// Invalid client data
    InvalidClientData(String),

    /// Invalid authenticator data
    InvalidAuthenticatorData(String),

    /// COSE key parsing error
    CoseKeyError(String),

    /// Credential already registered
    CredentialAlreadyRegistered,

    /// Internal error
    InternalError(String),
}

impl std::fmt::Display for WebAuthnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChallengeExpired => write!(f, "Challenge has expired"),
            Self::ChallengeNotFound => write!(f, "Challenge not found"),
            Self::OriginMismatch { expected, received } => {
                write!(
                    f,
                    "Origin mismatch: expected {}, received {}",
                    expected, received
                )
            }
            Self::RpIdMismatch { expected, received } => {
                write!(
                    f,
                    "RP ID mismatch: expected {}, received {}",
                    expected, received
                )
            }
            Self::SignCounterNotIncremented { stored, received } => {
                write!(
                    f,
                    "Sign counter not incremented: stored {}, received {}",
                    stored, received
                )
            }
            Self::CredentialNotFound { credential_id } => {
                write!(f, "Credential not found: {:?}", credential_id)
            }
            Self::UserVerificationRequired => {
                write!(f, "User verification required but not performed")
            }
            Self::UserPresenceRequired => write!(f, "User presence required but not confirmed"),
            Self::InvalidAttestationFormat(s) => write!(f, "Invalid attestation format: {}", s),
            Self::AttestationVerificationFailed(s) => {
                write!(f, "Attestation verification failed: {}", s)
            }
            Self::SignatureVerificationFailed(s) => {
                write!(f, "Signature verification failed: {}", s)
            }
            Self::InvalidClientData(s) => write!(f, "Invalid client data: {}", s),
            Self::InvalidAuthenticatorData(s) => write!(f, "Invalid authenticator data: {}", s),
            Self::CoseKeyError(s) => write!(f, "COSE key error: {}", s),
            Self::CredentialAlreadyRegistered => write!(f, "Credential already registered"),
            Self::InternalError(s) => write!(f, "Internal error: {}", s),
        }
    }
}

impl std::error::Error for WebAuthnError {}

// ============================================================================
// WebAuthn Service
// ============================================================================

/// Configuration for WebAuthn service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebAuthnConfig {
    /// Relying party identifier (typically domain name)
    pub rp_id: String,

    /// Relying party origin (full URL, e.g., `https://example.com`)
    pub rp_origin: String,

    /// Relying party display name
    pub rp_name: String,

    /// Challenge time-to-live in seconds (default: 300 = 5 minutes)
    pub challenge_ttl_seconds: u64,

    /// Default timeout for operations in milliseconds
    pub default_timeout_ms: u32,

    /// Require user verification for authentication
    pub require_user_verification: bool,

    /// Allowed origins (for multi-origin support)
    pub allowed_origins: Vec<String>,

    /// Allowed RP IDs (for subdomain support)
    pub allowed_rp_ids: Vec<String>,
}

impl WebAuthnConfig {
    /// Create config for a single domain
    pub fn single_domain(domain: &str, name: &str) -> Self {
        Self {
            rp_id: domain.to_string(),
            rp_origin: format!("https://{}", domain),
            rp_name: name.to_string(),
            challenge_ttl_seconds: 300,
            default_timeout_ms: 60000,
            require_user_verification: false,
            allowed_origins: vec![format!("https://{}", domain)],
            allowed_rp_ids: vec![domain.to_string()],
        }
    }
}

/// WebAuthn service for managing registration and authentication
pub struct WebAuthnService {
    /// Configuration
    config: WebAuthnConfig,

    /// Pending registration challenges (keyed by challenge bytes)
    pending_registrations: HashMap<Vec<u8>, RegistrationChallenge>,

    /// Pending authentication challenges (keyed by challenge bytes)
    pending_authentications: HashMap<Vec<u8>, AuthenticationChallenge>,
}

impl WebAuthnService {
    /// Create a new WebAuthn service
    pub fn new(rp_id: String, rp_origin: String) -> Self {
        Self::with_config(WebAuthnConfig {
            rp_id: rp_id.clone(),
            rp_origin: rp_origin.clone(),
            rp_name: rp_id.clone(),
            challenge_ttl_seconds: 300,
            default_timeout_ms: 60000,
            require_user_verification: false,
            allowed_origins: vec![rp_origin],
            allowed_rp_ids: vec![rp_id],
        })
    }

    /// Create a new WebAuthn service with full configuration
    pub fn with_config(config: WebAuthnConfig) -> Self {
        Self {
            config,
            pending_registrations: HashMap::new(),
            pending_authentications: HashMap::new(),
        }
    }

    /// Generate cryptographically secure random challenge bytes
    fn generate_challenge(&self) -> Vec<u8> {
        // Generate 32 bytes of randomness
        // In production, this should use a CSPRNG
        let mut challenge = vec![0u8; 32];

        // Use timestamp + hash for basic randomness
        // In production, use getrandom or os-level CSPRNG
        let now = current_timestamp();
        let seed_data = format!(
            "{}{}{:?}",
            now,
            self.config.rp_id,
            std::time::Instant::now()
        );
        let mut hasher = Sha256::new();
        hasher.update(seed_data.as_bytes());
        challenge.copy_from_slice(&hasher.finalize());

        challenge
    }

    // =========================================================================
    // Registration Flow
    // =========================================================================

    /// Create a registration challenge for a new credential
    ///
    /// # Arguments
    /// * `user_id` - Opaque user identifier (e.g., DID bytes)
    /// * `user_name` - Human-readable username
    ///
    /// # Returns
    /// Registration challenge to send to the client
    pub fn create_registration_challenge(
        &mut self,
        user_id: &[u8],
        user_name: &str,
    ) -> Result<RegistrationChallenge, WebAuthnError> {
        self.create_registration_challenge_with_options(
            user_id,
            user_name,
            user_name,
            AttestationConveyance::None,
            AuthenticatorSelectionCriteria::default(),
            vec![],
        )
    }

    /// Create a registration challenge with full options
    pub fn create_registration_challenge_with_options(
        &mut self,
        user_id: &[u8],
        user_name: &str,
        display_name: &str,
        attestation: AttestationConveyance,
        auth_selection: AuthenticatorSelectionCriteria,
        exclude_credentials: Vec<Vec<u8>>,
    ) -> Result<RegistrationChallenge, WebAuthnError> {
        let challenge = self.generate_challenge();
        let now = current_timestamp();

        let reg_challenge = RegistrationChallenge {
            challenge: challenge.clone(),
            rp_id: self.config.rp_id.clone(),
            rp_name: self.config.rp_name.clone(),
            user_id: user_id.to_vec(),
            user_name: user_name.to_string(),
            user_display_name: display_name.to_string(),
            timeout_ms: self.config.default_timeout_ms,
            attestation_preference: attestation,
            authenticator_selection: auth_selection,
            exclude_credentials,
            created_at: now,
            expires_at: now + self.config.challenge_ttl_seconds,
        };

        // Store the challenge
        self.pending_registrations
            .insert(challenge, reg_challenge.clone());

        // Clean up expired challenges
        self.cleanup_expired_challenges();

        Ok(reg_challenge)
    }

    /// Verify a registration response and create a credential
    ///
    /// # Arguments
    /// * `challenge` - The registration challenge that was sent to client
    /// * `response` - The response from the authenticator
    ///
    /// # Returns
    /// A new WebAuthnCredential on success
    pub fn verify_registration(
        &mut self,
        challenge: &RegistrationChallenge,
        response: &RegistrationResponse,
    ) -> Result<WebAuthnCredential, WebAuthnError> {
        // 1. Verify challenge is valid and not expired
        let stored = self
            .pending_registrations
            .get(&challenge.challenge)
            .ok_or(WebAuthnError::ChallengeNotFound)?;

        if stored.is_expired() {
            self.pending_registrations.remove(&challenge.challenge);
            return Err(WebAuthnError::ChallengeExpired);
        }

        // 2. Parse and verify client data
        let client_data = self.parse_client_data(&response.client_data_json)?;

        // Verify type is "webauthn.create"
        if client_data.get("type").and_then(|v| v.as_str()) != Some("webauthn.create") {
            return Err(WebAuthnError::InvalidClientData(
                "Invalid type, expected webauthn.create".to_string(),
            ));
        }

        // Verify challenge matches (base64url encoded)
        let challenge_b64 = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            &challenge.challenge,
        );
        if client_data.get("challenge").and_then(|v| v.as_str()) != Some(&challenge_b64) {
            return Err(WebAuthnError::InvalidClientData(
                "Challenge mismatch".to_string(),
            ));
        }

        // Verify origin
        if let Some(origin) = client_data.get("origin").and_then(|v| v.as_str()) {
            if !self.config.allowed_origins.contains(&origin.to_string()) {
                return Err(WebAuthnError::OriginMismatch {
                    expected: self.config.rp_origin.clone(),
                    received: origin.to_string(),
                });
            }
        }

        // 3. Parse attestation object (simplified - full impl would use CBOR)
        let (auth_data, attestation_fmt, public_key) =
            self.parse_attestation_object(&response.attestation_object)?;

        // 4. Verify RP ID hash
        let expected_rp_id_hash = {
            let mut hasher = Sha256::new();
            hasher.update(self.config.rp_id.as_bytes());
            hasher.finalize()
        };

        if auth_data.len() < 32 || auth_data[..32] != expected_rp_id_hash[..] {
            return Err(WebAuthnError::RpIdMismatch {
                expected: self.config.rp_id.clone(),
                received: "Hash mismatch".to_string(),
            });
        }

        // 5. Verify flags
        if auth_data.len() < 33 {
            return Err(WebAuthnError::InvalidAuthenticatorData(
                "Auth data too short".to_string(),
            ));
        }
        let flags = AuthenticatorFlags::from_byte(auth_data[32]);

        if !flags.user_present {
            return Err(WebAuthnError::UserPresenceRequired);
        }

        if challenge.authenticator_selection.user_verification == UserVerification::Required
            && !flags.user_verified
        {
            return Err(WebAuthnError::UserVerificationRequired);
        }

        // 6. Create credential
        let credential = WebAuthnCredential::new(
            response.credential_id.clone(),
            public_key,
            challenge.user_id.clone(),
            attestation_fmt,
        )
        .with_transports(response.transports.clone());

        // 7. Remove used challenge
        self.pending_registrations.remove(&challenge.challenge);

        Ok(credential)
    }

    // =========================================================================
    // Authentication Flow
    // =========================================================================

    /// Create an authentication challenge
    ///
    /// # Arguments
    /// * `credential_ids` - List of allowed credential IDs for the user
    ///
    /// # Returns
    /// Authentication challenge to send to the client
    pub fn create_authentication_challenge(
        &mut self,
        credential_ids: &[Vec<u8>],
    ) -> Result<AuthenticationChallenge, WebAuthnError> {
        self.create_authentication_challenge_with_options(
            credential_ids,
            if self.config.require_user_verification {
                UserVerification::Required
            } else {
                UserVerification::Preferred
            },
        )
    }

    /// Create an authentication challenge with options
    pub fn create_authentication_challenge_with_options(
        &mut self,
        credential_ids: &[Vec<u8>],
        user_verification: UserVerification,
    ) -> Result<AuthenticationChallenge, WebAuthnError> {
        let challenge = self.generate_challenge();
        let now = current_timestamp();

        let allowed_credentials = credential_ids
            .iter()
            .map(|id| AllowedCredential {
                id: id.clone(),
                transports: vec![], // Let client determine transports
            })
            .collect();

        let auth_challenge = AuthenticationChallenge {
            challenge: challenge.clone(),
            rp_id: self.config.rp_id.clone(),
            allowed_credentials,
            timeout_ms: self.config.default_timeout_ms,
            user_verification,
            created_at: now,
            expires_at: now + self.config.challenge_ttl_seconds,
        };

        // Store the challenge
        self.pending_authentications
            .insert(challenge, auth_challenge.clone());

        // Clean up expired challenges
        self.cleanup_expired_challenges();

        Ok(auth_challenge)
    }

    /// Verify an authentication response
    ///
    /// # Arguments
    /// * `challenge` - The authentication challenge that was sent
    /// * `response` - The response from the authenticator
    /// * `credential` - The stored credential to verify against
    ///
    /// # Returns
    /// Authentication result with updated sign count
    pub fn verify_authentication(
        &mut self,
        challenge: &AuthenticationChallenge,
        response: &AuthenticationResponse,
        credential: &WebAuthnCredential,
    ) -> Result<AuthenticationResult, WebAuthnError> {
        // 1. Verify challenge is valid
        let stored = self
            .pending_authentications
            .get(&challenge.challenge)
            .ok_or(WebAuthnError::ChallengeNotFound)?;

        if stored.is_expired() {
            self.pending_authentications.remove(&challenge.challenge);
            return Err(WebAuthnError::ChallengeExpired);
        }

        // 2. Verify credential ID matches
        if response.credential_id != credential.credential_id {
            return Err(WebAuthnError::CredentialNotFound {
                credential_id: response.credential_id.clone(),
            });
        }

        // 3. Parse and verify client data
        let client_data = self.parse_client_data(&response.client_data_json)?;

        // Verify type is "webauthn.get"
        if client_data.get("type").and_then(|v| v.as_str()) != Some("webauthn.get") {
            return Err(WebAuthnError::InvalidClientData(
                "Invalid type, expected webauthn.get".to_string(),
            ));
        }

        // Verify challenge matches
        let challenge_b64 = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            &challenge.challenge,
        );
        if client_data.get("challenge").and_then(|v| v.as_str()) != Some(&challenge_b64) {
            return Err(WebAuthnError::InvalidClientData(
                "Challenge mismatch".to_string(),
            ));
        }

        // Verify origin
        if let Some(origin) = client_data.get("origin").and_then(|v| v.as_str()) {
            if !self.config.allowed_origins.contains(&origin.to_string()) {
                return Err(WebAuthnError::OriginMismatch {
                    expected: self.config.rp_origin.clone(),
                    received: origin.to_string(),
                });
            }
        }

        // 4. Verify RP ID hash in authenticator data
        if response.authenticator_data.len() < 37 {
            return Err(WebAuthnError::InvalidAuthenticatorData(
                "Auth data too short".to_string(),
            ));
        }

        let expected_rp_id_hash = {
            let mut hasher = Sha256::new();
            hasher.update(self.config.rp_id.as_bytes());
            hasher.finalize()
        };

        if response.authenticator_data[..32] != expected_rp_id_hash[..] {
            return Err(WebAuthnError::RpIdMismatch {
                expected: self.config.rp_id.clone(),
                received: "Hash mismatch".to_string(),
            });
        }

        // 5. Verify flags
        let flags = response.parse_flags().ok_or_else(|| {
            WebAuthnError::InvalidAuthenticatorData("Cannot parse flags".to_string())
        })?;

        if !flags.user_present {
            return Err(WebAuthnError::UserPresenceRequired);
        }

        if challenge.user_verification == UserVerification::Required && !flags.user_verified {
            return Err(WebAuthnError::UserVerificationRequired);
        }

        // 6. Verify signature
        let signed_data = response.signed_data();
        self.verify_signature(&credential.public_key, &signed_data, &response.signature)?;

        // 7. Verify sign counter
        let new_sign_count = response.parse_sign_count().unwrap_or(0);
        if !credential.validate_sign_count(new_sign_count) {
            return Err(WebAuthnError::SignCounterNotIncremented {
                stored: credential.sign_count,
                received: new_sign_count,
            });
        }

        // 8. Remove used challenge
        self.pending_authentications.remove(&challenge.challenge);

        Ok(AuthenticationResult {
            credential_id: credential.credential_id.clone(),
            user_handle: credential.user_handle.clone(),
            new_sign_count,
            user_verified: flags.user_verified,
            authenticated_at: current_timestamp(),
        })
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /// Parse client data JSON
    fn parse_client_data(
        &self,
        client_data_json: &[u8],
    ) -> Result<serde_json::Value, WebAuthnError> {
        serde_json::from_slice(client_data_json)
            .map_err(|e| WebAuthnError::InvalidClientData(format!("JSON parse error: {}", e)))
    }

    /// Parse attestation object (simplified - full impl would parse CBOR)
    fn parse_attestation_object(
        &self,
        _attestation_object: &[u8],
    ) -> Result<(Vec<u8>, AttestationFormat, Vec<u8>), WebAuthnError> {
        // In production, this would:
        // 1. Parse CBOR to extract authData, fmt, attStmt
        // 2. Verify attestation statement based on fmt
        // 3. Extract credential public key from authData
        //
        // For now, return a placeholder
        // This would need a CBOR library (e.g., ciborium) for full implementation

        Err(WebAuthnError::InternalError(
            "Attestation parsing requires CBOR library - stub implementation".to_string(),
        ))
    }

    /// Verify a signature using COSE public key
    ///
    /// # Safety
    /// When the `stub-webauthn` feature is enabled, this accepts any signature
    /// for testing purposes only. Without that feature, signature verification
    /// is enforced and will reject unverified signatures.
    fn verify_signature(
        &self,
        _public_key: &[u8],
        _signed_data: &[u8],
        _signature: &[u8],
    ) -> Result<(), WebAuthnError> {
        // Production: reject until real COSE verification is implemented.
        // This requires a CBOR/COSE library (e.g., coset + ring/p256) to:
        // 1. Parse COSE key to determine algorithm (ES256, RS256, EdDSA)
        // 2. Extract key parameters
        // 3. Verify signature using appropriate algorithm

        #[cfg(feature = "stub-webauthn")]
        {
            return Ok(());
        }

        #[cfg(not(feature = "stub-webauthn"))]
        {
            Err(WebAuthnError::InternalError(
                "WebAuthn signature verification not yet implemented. \
                 Enable feature 'stub-webauthn' for testing only."
                    .to_string(),
            ))
        }
    }

    /// Clean up expired challenges
    fn cleanup_expired_challenges(&mut self) {
        let now = current_timestamp();

        self.pending_registrations.retain(|_, c| c.expires_at > now);
        self.pending_authentications
            .retain(|_, c| c.expires_at > now);
    }

    /// Get number of pending registrations
    pub fn pending_registration_count(&self) -> usize {
        self.pending_registrations.len()
    }

    /// Get number of pending authentications
    pub fn pending_authentication_count(&self) -> usize {
        self.pending_authentications.len()
    }

    /// Get the service configuration
    pub fn config(&self) -> &WebAuthnConfig {
        &self.config
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get current Unix timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_service() -> WebAuthnService {
        WebAuthnService::new(
            "mycelix.example.com".to_string(),
            "https://mycelix.example.com".to_string(),
        )
    }

    #[test]
    fn test_credential_creation() {
        let cred = WebAuthnCredential::new(
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10],
            AttestationFormat::None,
        )
        .with_name("My YubiKey")
        .with_transports(vec![AuthenticatorTransport::Usb]);

        assert_eq!(cred.credential_id, vec![1, 2, 3, 4]);
        assert_eq!(cred.friendly_name, Some("My YubiKey".to_string()));
        assert_eq!(cred.sign_count, 0);
    }

    #[test]
    fn test_sign_counter_validation() {
        let mut cred = WebAuthnCredential::new(
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10],
            AttestationFormat::None,
        );

        // Valid increment
        assert!(cred.validate_sign_count(1));
        assert!(cred.update_sign_count(1).is_ok());
        assert_eq!(cred.sign_count, 1);

        // Valid increment
        assert!(cred.validate_sign_count(5));
        assert!(cred.update_sign_count(5).is_ok());
        assert_eq!(cred.sign_count, 5);

        // Invalid - counter not incremented
        assert!(!cred.validate_sign_count(3));
        assert!(cred.update_sign_count(3).is_err());

        // Counter of 0 is always valid (authenticator doesn't support counters)
        assert!(cred.validate_sign_count(0));
    }

    #[test]
    fn test_registration_challenge_creation() {
        let mut service = create_test_service();

        let user_id = b"did:mycelix:test123".to_vec();
        let challenge = service
            .create_registration_challenge(&user_id, "test_user")
            .unwrap();

        assert_eq!(challenge.rp_id, "mycelix.example.com");
        assert_eq!(challenge.user_id, user_id);
        assert_eq!(challenge.user_name, "test_user");
        assert_eq!(challenge.challenge.len(), 32);
        assert!(!challenge.is_expired());
    }

    #[test]
    fn test_authentication_challenge_creation() {
        let mut service = create_test_service();

        let cred_ids = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];

        let challenge = service.create_authentication_challenge(&cred_ids).unwrap();

        assert_eq!(challenge.rp_id, "mycelix.example.com");
        assert_eq!(challenge.allowed_credentials.len(), 2);
        assert_eq!(challenge.challenge.len(), 32);
        assert!(!challenge.is_expired());
    }

    #[test]
    fn test_authenticator_flags_parsing() {
        // UP + UV flags set
        let flags = AuthenticatorFlags::from_byte(0x05);
        assert!(flags.user_present);
        assert!(flags.user_verified);
        assert!(!flags.backup_eligible);
        assert!(!flags.attested_credential_data);

        // All flags set
        let flags_all = AuthenticatorFlags::from_byte(0xFF);
        assert!(flags_all.user_present);
        assert!(flags_all.user_verified);
        assert!(flags_all.backup_eligible);
        assert!(flags_all.backed_up);
        assert!(flags_all.attested_credential_data);
        assert!(flags_all.extension_data);
    }

    #[test]
    fn test_authentication_response_parsing() {
        // Create mock auth data: 32 bytes RP ID hash + 1 byte flags + 4 bytes counter
        let mut auth_data = vec![0u8; 32]; // RP ID hash
        auth_data.push(0x05); // Flags: UP + UV
        auth_data.extend_from_slice(&42u32.to_be_bytes()); // Counter = 42

        let response = AuthenticationResponse {
            credential_id: vec![1, 2, 3],
            client_data_json: br#"{"type":"webauthn.get"}"#.to_vec(),
            authenticator_data: auth_data,
            signature: vec![],
            user_handle: None,
        };

        assert_eq!(response.parse_sign_count(), Some(42));

        let flags = response.parse_flags().unwrap();
        assert!(flags.user_present);
        assert!(flags.user_verified);
    }

    #[test]
    fn test_client_data_hash() {
        let response = RegistrationResponse {
            credential_id: vec![],
            client_data_json: b"test data".to_vec(),
            attestation_object: vec![],
            transports: vec![],
        };

        let hash = response.client_data_hash();
        assert_eq!(hash.len(), 32);

        // Same data should produce same hash
        let hash2 = response.client_data_hash();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_webauthn_config() {
        let config = WebAuthnConfig::single_domain("example.com", "Example App");

        assert_eq!(config.rp_id, "example.com");
        assert_eq!(config.rp_origin, "https://example.com");
        assert_eq!(config.rp_name, "Example App");
        assert!(
            config
                .allowed_origins
                .contains(&"https://example.com".to_string())
        );
    }

    #[test]
    fn test_error_display() {
        let err = WebAuthnError::ChallengeExpired;
        assert_eq!(err.to_string(), "Challenge has expired");

        let err = WebAuthnError::OriginMismatch {
            expected: "https://a.com".to_string(),
            received: "https://b.com".to_string(),
        };
        assert!(err.to_string().contains("Origin mismatch"));
    }

    #[test]
    fn test_challenge_expiration() {
        let challenge = RegistrationChallenge {
            challenge: vec![1, 2, 3],
            rp_id: "test".to_string(),
            rp_name: "Test".to_string(),
            user_id: vec![],
            user_name: "user".to_string(),
            user_display_name: "User".to_string(),
            timeout_ms: 60000,
            attestation_preference: AttestationConveyance::None,
            authenticator_selection: AuthenticatorSelectionCriteria::default(),
            exclude_credentials: vec![],
            created_at: current_timestamp(),
            expires_at: current_timestamp() + 300,
        };

        assert!(!challenge.is_expired());
        assert!(challenge.remaining_seconds() > 0);
    }

    #[test]
    fn test_transport_variants() {
        let transports = vec![
            AuthenticatorTransport::Usb,
            AuthenticatorTransport::Nfc,
            AuthenticatorTransport::Ble,
            AuthenticatorTransport::Internal,
            AuthenticatorTransport::Hybrid,
        ];

        assert_eq!(transports.len(), 5);
    }

    #[test]
    fn test_attestation_format_variants() {
        let formats = vec![
            AttestationFormat::None,
            AttestationFormat::Packed,
            AttestationFormat::Tpm,
            AttestationFormat::AndroidKey,
            AttestationFormat::AndroidSafetyNet,
            AttestationFormat::FidoU2f,
            AttestationFormat::Apple,
        ];

        assert_eq!(formats.len(), 7);
        assert_eq!(AttestationFormat::default(), AttestationFormat::None);
    }
}
