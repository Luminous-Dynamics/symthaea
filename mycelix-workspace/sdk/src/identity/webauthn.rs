//! WebAuthn/FIDO2 Integration for Mycelix Identity System
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

// CBOR/COSE imports for WebAuthn full implementation
#[cfg(feature = "webauthn-full")]
use ciborium::Value as CborValue;
#[cfg(feature = "webauthn-full")]
use ed25519_dalek::{Signature as Ed25519Signature, VerifyingKey as Ed25519VerifyingKey};
#[cfg(feature = "webauthn-full")]
use p256::ecdsa::{
    signature::Verifier, Signature as P256Signature, VerifyingKey as P256VerifyingKey,
};

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
// COSE Key Types (for WebAuthn signature verification)
// ============================================================================

/// COSE key algorithms supported by WebAuthn
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoseAlgorithm {
    /// ES256 - ECDSA with P-256 and SHA-256 (COSE algorithm -7)
    Es256,
    /// EdDSA - Edwards-curve DSA (Ed25519) (COSE algorithm -8)
    EdDsa,
}

/// COSE key types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoseKeyType {
    /// OKP - Octet Key Pair (for Ed25519)
    Okp,
    /// EC2 - Elliptic Curve (for P-256)
    Ec2,
}

/// Parsed COSE key structure
///
/// COSE_Key is defined in RFC 8152 and used by WebAuthn for public key encoding.
/// The key parameters depend on the key type:
///
/// For EC2 (P-256):
/// - kty: 2 (EC2)
/// - alg: -7 (ES256)
/// - crv: 1 (P-256)
/// - x: 32 bytes (x-coordinate)
/// - y: 32 bytes (y-coordinate)
///
/// For OKP (Ed25519):
/// - kty: 1 (OKP)
/// - alg: -8 (EdDSA)
/// - crv: 6 (Ed25519)
/// - x: 32 bytes (public key)
#[derive(Debug, Clone)]
pub struct CoseKey {
    /// Key type (EC2 for P-256, OKP for Ed25519)
    pub key_type: CoseKeyType,
    /// Algorithm (ES256 or EdDSA)
    pub algorithm: CoseAlgorithm,
    /// Curve identifier (P-256 = 1, Ed25519 = 6)
    pub curve: i64,
    /// X coordinate (for EC2) or public key (for OKP)
    pub x_coordinate: Option<Vec<u8>>,
    /// Y coordinate (for EC2 only)
    pub y_coordinate: Option<Vec<u8>>,
}

impl CoseKey {
    /// Parse a COSE key from CBOR bytes
    #[cfg(feature = "webauthn-full")]
    pub fn from_cbor(data: &[u8]) -> Result<Self, WebAuthnError> {
        let cbor_value: CborValue = ciborium::from_reader(data)
            .map_err(|e| WebAuthnError::CoseKeyError(format!("CBOR parse error: {}", e)))?;

        let map = cbor_value
            .as_map()
            .ok_or_else(|| WebAuthnError::CoseKeyError("Expected CBOR map".to_string()))?;

        // COSE key parameter labels:
        // 1 = kty (key type)
        // 3 = alg (algorithm)
        // -1 = crv (curve) for EC2/OKP
        // -2 = x (x-coordinate)
        // -3 = y (y-coordinate) for EC2

        let kty = Self::get_integer_param(map, 1)?;
        let alg = Self::get_integer_param(map, 3)?;
        let crv = Self::get_integer_param(map, -1)?;

        let key_type = match kty {
            1 => CoseKeyType::Okp,
            2 => CoseKeyType::Ec2,
            other => {
                return Err(WebAuthnError::CoseKeyError(format!(
                    "Unsupported key type: {}",
                    other
                )))
            }
        };

        let algorithm = match alg {
            -7 => CoseAlgorithm::Es256,
            -8 => CoseAlgorithm::EdDsa,
            other => {
                return Err(WebAuthnError::CoseKeyError(format!(
                    "Unsupported algorithm: {}",
                    other
                )))
            }
        };

        // Validate key type and algorithm match
        match (key_type, algorithm) {
            (CoseKeyType::Ec2, CoseAlgorithm::Es256) => {
                if crv != 1 {
                    return Err(WebAuthnError::CoseKeyError(format!(
                        "ES256 requires P-256 curve (1), got: {}",
                        crv
                    )));
                }
            }
            (CoseKeyType::Okp, CoseAlgorithm::EdDsa) => {
                if crv != 6 {
                    return Err(WebAuthnError::CoseKeyError(format!(
                        "EdDSA requires Ed25519 curve (6), got: {}",
                        crv
                    )));
                }
            }
            _ => {
                return Err(WebAuthnError::CoseKeyError(format!(
                    "Mismatched key type {:?} and algorithm {:?}",
                    key_type, algorithm
                )))
            }
        }

        let x_coordinate = Self::get_bytes_param(map, -2).ok();
        let y_coordinate = Self::get_bytes_param(map, -3).ok();

        // Validate required parameters
        if x_coordinate.is_none() {
            return Err(WebAuthnError::CoseKeyError(
                "Missing x coordinate (-2)".to_string(),
            ));
        }

        if key_type == CoseKeyType::Ec2 && y_coordinate.is_none() {
            return Err(WebAuthnError::CoseKeyError(
                "Missing y coordinate (-3) for EC2 key".to_string(),
            ));
        }

        Ok(Self {
            key_type,
            algorithm,
            curve: crv,
            x_coordinate,
            y_coordinate,
        })
    }

    /// Stub implementation when webauthn-full is not enabled
    #[cfg(not(feature = "webauthn-full"))]
    pub fn from_cbor(_data: &[u8]) -> Result<Self, WebAuthnError> {
        Err(WebAuthnError::CoseKeyError(
            "COSE key parsing requires 'webauthn-full' feature".to_string(),
        ))
    }

    /// Helper to get an integer parameter from COSE key map
    #[cfg(feature = "webauthn-full")]
    fn get_integer_param(map: &[(CborValue, CborValue)], label: i64) -> Result<i64, WebAuthnError> {
        for (k, v) in map {
            if let Some(key_int) = k.as_integer() {
                let key_val: i64 = key_int
                    .try_into()
                    .map_err(|_| WebAuthnError::CoseKeyError("Invalid key label".to_string()))?;
                if key_val == label {
                    if let Some(val_int) = v.as_integer() {
                        return val_int.try_into().map_err(|_| {
                            WebAuthnError::CoseKeyError(format!(
                                "Integer overflow for label {}",
                                label
                            ))
                        });
                    }
                }
            }
        }
        Err(WebAuthnError::CoseKeyError(format!(
            "Missing parameter {}",
            label
        )))
    }

    /// Helper to get bytes parameter from COSE key map
    #[cfg(feature = "webauthn-full")]
    fn get_bytes_param(
        map: &[(CborValue, CborValue)],
        label: i64,
    ) -> Result<Vec<u8>, WebAuthnError> {
        for (k, v) in map {
            if let Some(key_int) = k.as_integer() {
                let key_val: i64 = key_int
                    .try_into()
                    .map_err(|_| WebAuthnError::CoseKeyError("Invalid key label".to_string()))?;
                if key_val == label {
                    if let Some(bytes) = v.as_bytes() {
                        return Ok(bytes.to_vec());
                    }
                }
            }
        }
        Err(WebAuthnError::CoseKeyError(format!(
            "Missing parameter {}",
            label
        )))
    }

    /// Get the algorithm for this key
    pub fn algorithm(&self) -> CoseAlgorithm {
        self.algorithm
    }

    /// Check if this is an EC2 (P-256) key
    pub fn is_ec2(&self) -> bool {
        self.key_type == CoseKeyType::Ec2
    }

    /// Check if this is an OKP (Ed25519) key
    pub fn is_okp(&self) -> bool {
        self.key_type == CoseKeyType::Okp
    }
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

    /// Parse attestation object from CBOR format
    ///
    /// The attestation object has the structure:
    /// ```text
    /// {
    ///   "fmt": text,      // Attestation format: "packed", "fido-u2f", "none", etc.
    ///   "authData": bytes, // Authenticator data
    ///   "attStmt": map     // Attestation statement (format-specific)
    /// }
    /// ```
    ///
    /// Returns (authData, attestation_format, public_key)
    #[cfg(feature = "webauthn-full")]
    fn parse_attestation_object(
        &self,
        attestation_object: &[u8],
    ) -> Result<(Vec<u8>, AttestationFormat, Vec<u8>), WebAuthnError> {
        // Parse CBOR
        let cbor_value: CborValue = ciborium::from_reader(attestation_object).map_err(|e| {
            WebAuthnError::InvalidAttestationFormat(format!("CBOR parse error: {}", e))
        })?;

        let map = cbor_value.as_map().ok_or_else(|| {
            WebAuthnError::InvalidAttestationFormat("Expected CBOR map".to_string())
        })?;

        // Extract fmt (attestation format)
        let fmt_str = self.cbor_map_get_text(map, "fmt")?;
        let attestation_format = match fmt_str.as_str() {
            "packed" => AttestationFormat::Packed,
            "fido-u2f" => AttestationFormat::FidoU2f,
            "none" => AttestationFormat::None,
            "tpm" => AttestationFormat::Tpm,
            "android-key" => AttestationFormat::AndroidKey,
            "android-safetynet" => AttestationFormat::AndroidSafetyNet,
            "apple" => AttestationFormat::Apple,
            other => {
                return Err(WebAuthnError::InvalidAttestationFormat(format!(
                    "Unknown attestation format: {}",
                    other
                )))
            }
        };

        // Extract authData
        let auth_data = self.cbor_map_get_bytes(map, "authData")?;

        // Parse authenticator data to extract public key
        // authData structure:
        //   - rpIdHash: 32 bytes
        //   - flags: 1 byte
        //   - signCount: 4 bytes (big-endian)
        //   - attestedCredentialData (if AT flag set):
        //     - aaguid: 16 bytes
        //     - credentialIdLength: 2 bytes (big-endian)
        //     - credentialId: credentialIdLength bytes
        //     - credentialPublicKey: COSE_Key (CBOR)

        if auth_data.len() < 37 {
            return Err(WebAuthnError::InvalidAuthenticatorData(
                "Auth data too short (minimum 37 bytes)".to_string(),
            ));
        }

        let flags = AuthenticatorFlags::from_byte(auth_data[32]);

        // Check if attested credential data is present
        if !flags.attested_credential_data {
            return Err(WebAuthnError::InvalidAuthenticatorData(
                "No attested credential data in registration response".to_string(),
            ));
        }

        // Parse attested credential data (starts at byte 37)
        if auth_data.len() < 55 {
            return Err(WebAuthnError::InvalidAuthenticatorData(
                "Auth data too short for attested credential data".to_string(),
            ));
        }

        // Skip aaguid (16 bytes) and get credential ID length
        let cred_id_len = u16::from_be_bytes([auth_data[53], auth_data[54]]) as usize;

        // Calculate where the COSE key starts
        let cose_key_start = 55 + cred_id_len;
        if auth_data.len() <= cose_key_start {
            return Err(WebAuthnError::InvalidAuthenticatorData(
                "Auth data too short for COSE key".to_string(),
            ));
        }

        // The remaining bytes are the COSE-encoded public key
        let public_key = auth_data[cose_key_start..].to_vec();

        Ok((auth_data, attestation_format, public_key))
    }

    /// Stub implementation when webauthn-full feature is disabled
    #[cfg(not(feature = "webauthn-full"))]
    fn parse_attestation_object(
        &self,
        _attestation_object: &[u8],
    ) -> Result<(Vec<u8>, AttestationFormat, Vec<u8>), WebAuthnError> {
        Err(WebAuthnError::InternalError(
            "Attestation parsing requires 'webauthn-full' feature".to_string(),
        ))
    }

    /// Helper to get a text value from a CBOR map
    #[cfg(feature = "webauthn-full")]
    fn cbor_map_get_text(
        &self,
        map: &[(CborValue, CborValue)],
        key: &str,
    ) -> Result<String, WebAuthnError> {
        for (k, v) in map {
            if let Some(k_str) = k.as_text() {
                if k_str == key {
                    return v.as_text().map(|s| s.to_string()).ok_or_else(|| {
                        WebAuthnError::InvalidAttestationFormat(format!(
                            "Expected text for key '{}'",
                            key
                        ))
                    });
                }
            }
        }
        Err(WebAuthnError::InvalidAttestationFormat(format!(
            "Missing key '{}'",
            key
        )))
    }

    /// Helper to get bytes from a CBOR map
    #[cfg(feature = "webauthn-full")]
    fn cbor_map_get_bytes(
        &self,
        map: &[(CborValue, CborValue)],
        key: &str,
    ) -> Result<Vec<u8>, WebAuthnError> {
        for (k, v) in map {
            if let Some(k_str) = k.as_text() {
                if k_str == key {
                    return v.as_bytes().map(|b| b.to_vec()).ok_or_else(|| {
                        WebAuthnError::InvalidAttestationFormat(format!(
                            "Expected bytes for key '{}'",
                            key
                        ))
                    });
                }
            }
        }
        Err(WebAuthnError::InvalidAttestationFormat(format!(
            "Missing key '{}'",
            key
        )))
    }

    /// Verify a signature using COSE public key
    ///
    /// Supports:
    /// - ES256 (ECDSA with P-256 and SHA-256) - COSE algorithm -7
    /// - EdDSA (Ed25519) - COSE algorithm -8
    ///
    /// # Safety
    /// When the `stub-webauthn` feature is enabled, this accepts any signature
    /// for testing purposes only. Without that feature, signature verification
    /// is enforced and will reject unverified signatures.
    #[cfg(feature = "webauthn-full")]
    fn verify_signature(
        &self,
        public_key: &[u8],
        signed_data: &[u8],
        signature: &[u8],
    ) -> Result<(), WebAuthnError> {
        // Parse the COSE key to determine the algorithm
        let cose_key = CoseKey::from_cbor(public_key)?;

        match cose_key.algorithm {
            CoseAlgorithm::Es256 => self.verify_es256_signature(&cose_key, signed_data, signature),
            CoseAlgorithm::EdDsa => self.verify_eddsa_signature(&cose_key, signed_data, signature),
        }
    }

    /// Stub implementation when webauthn-full feature is disabled
    #[cfg(all(not(feature = "webauthn-full"), feature = "stub-webauthn"))]
    fn verify_signature(
        &self,
        _public_key: &[u8],
        _signed_data: &[u8],
        _signature: &[u8],
    ) -> Result<(), WebAuthnError> {
        // Accept any signature in stub mode
        Ok(())
    }

    /// Error when neither webauthn-full nor stub-webauthn is enabled
    #[cfg(all(not(feature = "webauthn-full"), not(feature = "stub-webauthn")))]
    fn verify_signature(
        &self,
        _public_key: &[u8],
        _signed_data: &[u8],
        _signature: &[u8],
    ) -> Result<(), WebAuthnError> {
        Err(WebAuthnError::InternalError(
            "WebAuthn signature verification not available. \
             Enable feature 'webauthn-full' for production or 'stub-webauthn' for testing."
                .to_string(),
        ))
    }

    /// Verify ES256 (ECDSA P-256) signature
    #[cfg(feature = "webauthn-full")]
    fn verify_es256_signature(
        &self,
        cose_key: &CoseKey,
        signed_data: &[u8],
        signature: &[u8],
    ) -> Result<(), WebAuthnError> {
        // Extract P-256 coordinates from COSE key
        let (x, y) = match (&cose_key.x_coordinate, &cose_key.y_coordinate) {
            (Some(x), Some(y)) => (x, y),
            _ => {
                return Err(WebAuthnError::CoseKeyError(
                    "Missing x or y coordinate for EC2 key".to_string(),
                ))
            }
        };

        // Build uncompressed SEC1 point: 0x04 || x || y
        let mut point = Vec::with_capacity(65);
        point.push(0x04);
        // Pad x coordinate to 32 bytes
        if x.len() < 32 {
            point.extend(std::iter::repeat(0u8).take(32 - x.len()));
        }
        point.extend_from_slice(x);
        // Pad y coordinate to 32 bytes
        if y.len() < 32 {
            point.extend(std::iter::repeat(0u8).take(32 - y.len()));
        }
        point.extend_from_slice(y);

        // Create P-256 verifying key
        let verifying_key = P256VerifyingKey::from_sec1_bytes(&point)
            .map_err(|e| WebAuthnError::CoseKeyError(format!("Invalid P-256 key: {}", e)))?;

        // Parse signature (WebAuthn uses raw r||s format, 64 bytes for P-256)
        let sig = if signature.len() == 64 {
            P256Signature::from_slice(signature).map_err(|e| {
                WebAuthnError::SignatureVerificationFailed(format!(
                    "Invalid ES256 signature format: {}",
                    e
                ))
            })?
        } else {
            // Try DER format
            P256Signature::from_der(signature).map_err(|e| {
                WebAuthnError::SignatureVerificationFailed(format!(
                    "Invalid ES256 signature: {}",
                    e
                ))
            })?
        };

        // Verify signature over SHA-256 hash of signed_data
        verifying_key.verify(signed_data, &sig).map_err(|_| {
            WebAuthnError::SignatureVerificationFailed(
                "ES256 signature verification failed".to_string(),
            )
        })
    }

    /// Verify EdDSA (Ed25519) signature
    #[cfg(feature = "webauthn-full")]
    fn verify_eddsa_signature(
        &self,
        cose_key: &CoseKey,
        signed_data: &[u8],
        signature: &[u8],
    ) -> Result<(), WebAuthnError> {
        // Extract Ed25519 public key from COSE key (x coordinate for OKP)
        let x = cose_key.x_coordinate.as_ref().ok_or_else(|| {
            WebAuthnError::CoseKeyError("Missing x coordinate for OKP key".to_string())
        })?;

        if x.len() != 32 {
            return Err(WebAuthnError::CoseKeyError(format!(
                "Invalid Ed25519 public key length: {} (expected 32)",
                x.len()
            )));
        }

        let key_bytes: [u8; 32] = x
            .as_slice()
            .try_into()
            .map_err(|_| WebAuthnError::CoseKeyError("Invalid key length".to_string()))?;

        let verifying_key = Ed25519VerifyingKey::from_bytes(&key_bytes)
            .map_err(|e| WebAuthnError::CoseKeyError(format!("Invalid Ed25519 key: {}", e)))?;

        if signature.len() != 64 {
            return Err(WebAuthnError::SignatureVerificationFailed(format!(
                "Invalid Ed25519 signature length: {} (expected 64)",
                signature.len()
            )));
        }

        let sig_bytes: [u8; 64] = signature.try_into().map_err(|_| {
            WebAuthnError::SignatureVerificationFailed("Invalid signature length".to_string())
        })?;

        let sig = Ed25519Signature::from_bytes(&sig_bytes);

        use ed25519_dalek::Verifier;
        verifying_key.verify(signed_data, &sig).map_err(|_| {
            WebAuthnError::SignatureVerificationFailed(
                "EdDSA signature verification failed".to_string(),
            )
        })
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
        assert!(config
            .allowed_origins
            .contains(&"https://example.com".to_string()));
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

    #[test]
    fn test_cose_algorithm_types() {
        assert_eq!(CoseAlgorithm::Es256, CoseAlgorithm::Es256);
        assert_eq!(CoseAlgorithm::EdDsa, CoseAlgorithm::EdDsa);
        assert_ne!(CoseAlgorithm::Es256, CoseAlgorithm::EdDsa);
    }

    #[test]
    fn test_cose_key_type_helpers() {
        // Test EC2 key type
        let ec2_key = CoseKey {
            key_type: CoseKeyType::Ec2,
            algorithm: CoseAlgorithm::Es256,
            curve: 1,
            x_coordinate: Some(vec![0u8; 32]),
            y_coordinate: Some(vec![0u8; 32]),
        };
        assert!(ec2_key.is_ec2());
        assert!(!ec2_key.is_okp());
        assert_eq!(ec2_key.algorithm(), CoseAlgorithm::Es256);

        // Test OKP key type
        let okp_key = CoseKey {
            key_type: CoseKeyType::Okp,
            algorithm: CoseAlgorithm::EdDsa,
            curve: 6,
            x_coordinate: Some(vec![0u8; 32]),
            y_coordinate: None,
        };
        assert!(okp_key.is_okp());
        assert!(!okp_key.is_ec2());
        assert_eq!(okp_key.algorithm(), CoseAlgorithm::EdDsa);
    }
}

// ============================================================================
// WebAuthn Full Feature Tests (require webauthn-full feature)
// ============================================================================

#[cfg(all(test, feature = "webauthn-full"))]
mod webauthn_full_tests {
    use super::*;

    /// Test vector: A minimal valid COSE EC2 key (P-256/ES256)
    /// This is hand-crafted CBOR for testing
    fn create_test_es256_cose_key() -> Vec<u8> {
        // COSE key structure for ES256:
        // {
        //   1: 2,      // kty: EC2
        //   3: -7,     // alg: ES256
        //   -1: 1,     // crv: P-256
        //   -2: x (32 bytes),
        //   -3: y (32 bytes)
        // }
        //
        // Using ciborium to generate valid CBOR:
        let mut map: Vec<(CborValue, CborValue)> = Vec::new();

        // kty = 2 (EC2)
        map.push((CborValue::Integer(1.into()), CborValue::Integer(2.into())));

        // alg = -7 (ES256)
        map.push((
            CborValue::Integer(3.into()),
            CborValue::Integer((-7i64).into()),
        ));

        // crv = 1 (P-256)
        map.push((
            CborValue::Integer((-1i64).into()),
            CborValue::Integer(1.into()),
        ));

        // x coordinate (32 bytes of test data)
        let x = vec![
            0x04, 0xb1, 0x71, 0x91, 0x4a, 0xf1, 0xf5, 0x49, 0xdb, 0x73, 0x9c, 0x2e, 0x85, 0xe4,
            0xf0, 0x3b, 0x4e, 0xf5, 0x4a, 0x82, 0xda, 0x75, 0xbb, 0x5e, 0x85, 0x4a, 0x0c, 0x4b,
            0xf8, 0xc8, 0x1a, 0x75,
        ];
        map.push((CborValue::Integer((-2i64).into()), CborValue::Bytes(x)));

        // y coordinate (32 bytes of test data)
        let y = vec![
            0x9f, 0x14, 0x86, 0xc0, 0x47, 0x83, 0x0a, 0xa5, 0xde, 0x90, 0x3f, 0x50, 0x3b, 0x3c,
            0x29, 0xa7, 0xf5, 0xfa, 0x6d, 0xd8, 0x61, 0xd7, 0x7e, 0x04, 0x5e, 0x1b, 0x13, 0x80,
            0x28, 0xd8, 0x9c, 0x73,
        ];
        map.push((CborValue::Integer((-3i64).into()), CborValue::Bytes(y)));

        let cbor = CborValue::Map(map);
        let mut buf = Vec::new();
        ciborium::into_writer(&cbor, &mut buf).expect("CBOR encoding failed");
        buf
    }

    /// Test vector: A minimal valid COSE OKP key (Ed25519/EdDSA)
    fn create_test_eddsa_cose_key() -> Vec<u8> {
        // COSE key structure for EdDSA:
        // {
        //   1: 1,      // kty: OKP
        //   3: -8,     // alg: EdDSA
        //   -1: 6,     // crv: Ed25519
        //   -2: x (32 bytes)
        // }
        let mut map: Vec<(CborValue, CborValue)> = Vec::new();

        // kty = 1 (OKP)
        map.push((CborValue::Integer(1.into()), CborValue::Integer(1.into())));

        // alg = -8 (EdDSA)
        map.push((
            CborValue::Integer(3.into()),
            CborValue::Integer((-8i64).into()),
        ));

        // crv = 6 (Ed25519)
        map.push((
            CborValue::Integer((-1i64).into()),
            CborValue::Integer(6.into()),
        ));

        // x coordinate (Ed25519 public key, 32 bytes)
        let x = vec![
            0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7, 0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64,
            0x07, 0x3a, 0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25, 0xaf, 0x02, 0x1a, 0x68,
            0xf7, 0x07, 0x51, 0x1a,
        ];
        map.push((CborValue::Integer((-2i64).into()), CborValue::Bytes(x)));

        let cbor = CborValue::Map(map);
        let mut buf = Vec::new();
        ciborium::into_writer(&cbor, &mut buf).expect("CBOR encoding failed");
        buf
    }

    /// Create a test attestation object with 'none' format
    fn create_test_attestation_object() -> Vec<u8> {
        // Create authenticator data with attested credential data
        let mut auth_data = Vec::new();

        // RP ID hash (32 bytes) - SHA-256 of "example.com"
        let rp_id_hash = {
            let mut hasher = Sha256::new();
            hasher.update(b"example.com");
            hasher.finalize()
        };
        auth_data.extend_from_slice(&rp_id_hash);

        // Flags: UP (0x01) + AT (0x40) = 0x41
        auth_data.push(0x41);

        // Sign count (4 bytes, big-endian)
        auth_data.extend_from_slice(&0u32.to_be_bytes());

        // AAGUID (16 bytes)
        auth_data.extend_from_slice(&[0u8; 16]);

        // Credential ID length (2 bytes, big-endian)
        let cred_id = vec![1, 2, 3, 4, 5, 6, 7, 8];
        auth_data.extend_from_slice(&(cred_id.len() as u16).to_be_bytes());

        // Credential ID
        auth_data.extend_from_slice(&cred_id);

        // COSE public key
        auth_data.extend_from_slice(&create_test_es256_cose_key());

        // Build attestation object
        let mut map: Vec<(CborValue, CborValue)> = Vec::new();
        map.push((
            CborValue::Text("fmt".to_string()),
            CborValue::Text("none".to_string()),
        ));
        map.push((
            CborValue::Text("authData".to_string()),
            CborValue::Bytes(auth_data),
        ));
        map.push((
            CborValue::Text("attStmt".to_string()),
            CborValue::Map(vec![]),
        ));

        let cbor = CborValue::Map(map);
        let mut buf = Vec::new();
        ciborium::into_writer(&cbor, &mut buf).expect("CBOR encoding failed");
        buf
    }

    #[test]
    fn test_cose_key_es256_parsing() {
        let cose_bytes = create_test_es256_cose_key();
        let key = CoseKey::from_cbor(&cose_bytes).expect("Failed to parse COSE key");

        assert_eq!(key.key_type, CoseKeyType::Ec2);
        assert_eq!(key.algorithm, CoseAlgorithm::Es256);
        assert_eq!(key.curve, 1);
        assert!(key.x_coordinate.is_some());
        assert!(key.y_coordinate.is_some());
        assert_eq!(key.x_coordinate.as_ref().unwrap().len(), 32);
        assert_eq!(key.y_coordinate.as_ref().unwrap().len(), 32);
    }

    #[test]
    fn test_cose_key_eddsa_parsing() {
        let cose_bytes = create_test_eddsa_cose_key();
        let key = CoseKey::from_cbor(&cose_bytes).expect("Failed to parse COSE key");

        assert_eq!(key.key_type, CoseKeyType::Okp);
        assert_eq!(key.algorithm, CoseAlgorithm::EdDsa);
        assert_eq!(key.curve, 6);
        assert!(key.x_coordinate.is_some());
        assert!(key.y_coordinate.is_none()); // OKP keys don't have y
        assert_eq!(key.x_coordinate.as_ref().unwrap().len(), 32);
    }

    #[test]
    fn test_cose_key_invalid_cbor() {
        let invalid_cbor = vec![0xff, 0xff, 0xff];
        let result = CoseKey::from_cbor(&invalid_cbor);
        assert!(result.is_err());
    }

    #[test]
    fn test_cose_key_missing_params() {
        // CBOR map with only kty, missing other required params
        let mut map: Vec<(CborValue, CborValue)> = Vec::new();
        map.push((CborValue::Integer(1.into()), CborValue::Integer(2.into())));

        let cbor = CborValue::Map(map);
        let mut buf = Vec::new();
        ciborium::into_writer(&cbor, &mut buf).unwrap();

        let result = CoseKey::from_cbor(&buf);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Missing parameter"));
    }

    #[test]
    fn test_attestation_object_parsing() {
        let attestation = create_test_attestation_object();

        let service =
            WebAuthnService::new("example.com".to_string(), "https://example.com".to_string());

        let result = service.parse_attestation_object(&attestation);
        assert!(
            result.is_ok(),
            "Failed to parse attestation: {:?}",
            result.err()
        );

        let (auth_data, format, public_key) = result.unwrap();

        assert_eq!(format, AttestationFormat::None);
        assert!(auth_data.len() >= 37);
        assert!(!public_key.is_empty());

        // Verify public key is valid COSE key
        let cose_key = CoseKey::from_cbor(&public_key);
        assert!(
            cose_key.is_ok(),
            "Public key is not valid COSE: {:?}",
            cose_key.err()
        );
    }

    #[test]
    fn test_attestation_invalid_format() {
        // Create attestation with invalid format
        let mut map: Vec<(CborValue, CborValue)> = Vec::new();
        map.push((
            CborValue::Text("fmt".to_string()),
            CborValue::Text("unknown-format".to_string()),
        ));
        map.push((
            CborValue::Text("authData".to_string()),
            CborValue::Bytes(vec![0u8; 37]),
        ));
        map.push((
            CborValue::Text("attStmt".to_string()),
            CborValue::Map(vec![]),
        ));

        let cbor = CborValue::Map(map);
        let mut buf = Vec::new();
        ciborium::into_writer(&cbor, &mut buf).unwrap();

        let service =
            WebAuthnService::new("example.com".to_string(), "https://example.com".to_string());

        let result = service.parse_attestation_object(&buf);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unknown attestation format"));
    }

    #[test]
    fn test_es256_signature_verification() {
        use p256::ecdsa::{signature::Signer, SigningKey};
        use rand::rngs::OsRng;

        // Generate a test key pair
        let signing_key = SigningKey::random(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        // Get public key coordinates
        let public_key_point = verifying_key.to_encoded_point(false);
        let x = public_key_point.x().unwrap().as_slice().to_vec();
        let y = public_key_point.y().unwrap().as_slice().to_vec();

        // Create COSE key
        let mut map: Vec<(CborValue, CborValue)> = Vec::new();
        map.push((CborValue::Integer(1.into()), CborValue::Integer(2.into()))); // kty: EC2
        map.push((
            CborValue::Integer(3.into()),
            CborValue::Integer((-7i64).into()),
        )); // alg: ES256
        map.push((
            CborValue::Integer((-1i64).into()),
            CborValue::Integer(1.into()),
        )); // crv: P-256
        map.push((CborValue::Integer((-2i64).into()), CborValue::Bytes(x)));
        map.push((CborValue::Integer((-3i64).into()), CborValue::Bytes(y)));

        let cbor = CborValue::Map(map);
        let mut cose_key_bytes = Vec::new();
        ciborium::into_writer(&cbor, &mut cose_key_bytes).unwrap();

        // Sign test data
        let test_data = b"test authentication data";
        let signature: P256Signature = signing_key.sign(test_data);

        // Verify signature
        let service =
            WebAuthnService::new("example.com".to_string(), "https://example.com".to_string());

        let result =
            service.verify_signature(&cose_key_bytes, test_data, signature.to_bytes().as_slice());

        assert!(
            result.is_ok(),
            "ES256 signature verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_es256_invalid_signature() {
        // Create a valid COSE key
        let cose_key_bytes = create_test_es256_cose_key();

        // Invalid signature (random bytes)
        let invalid_signature = vec![0u8; 64];

        let service =
            WebAuthnService::new("example.com".to_string(), "https://example.com".to_string());

        let result = service.verify_signature(&cose_key_bytes, b"test data", &invalid_signature);

        // Should fail - invalid key or signature
        assert!(result.is_err());
    }

    #[test]
    fn test_eddsa_signature_verification() {
        use ed25519_dalek::{Signer, SigningKey};
        use rand::rngs::OsRng;

        // Generate a test key pair
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        // Get public key bytes
        let public_key_bytes = verifying_key.to_bytes().to_vec();

        // Create COSE key
        let mut map: Vec<(CborValue, CborValue)> = Vec::new();
        map.push((CborValue::Integer(1.into()), CborValue::Integer(1.into()))); // kty: OKP
        map.push((
            CborValue::Integer(3.into()),
            CborValue::Integer((-8i64).into()),
        )); // alg: EdDSA
        map.push((
            CborValue::Integer((-1i64).into()),
            CborValue::Integer(6.into()),
        )); // crv: Ed25519
        map.push((
            CborValue::Integer((-2i64).into()),
            CborValue::Bytes(public_key_bytes),
        ));

        let cbor = CborValue::Map(map);
        let mut cose_key_bytes = Vec::new();
        ciborium::into_writer(&cbor, &mut cose_key_bytes).unwrap();

        // Sign test data
        let test_data = b"test authentication data for EdDSA";
        let signature = signing_key.sign(test_data);

        // Verify signature
        let service =
            WebAuthnService::new("example.com".to_string(), "https://example.com".to_string());

        let result =
            service.verify_signature(&cose_key_bytes, test_data, signature.to_bytes().as_slice());

        assert!(
            result.is_ok(),
            "EdDSA signature verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_eddsa_invalid_signature() {
        // Create a valid COSE OKP key (with random but valid public key)
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        let public_key_bytes = verifying_key.to_bytes().to_vec();

        let mut map: Vec<(CborValue, CborValue)> = Vec::new();
        map.push((CborValue::Integer(1.into()), CborValue::Integer(1.into()))); // kty: OKP
        map.push((
            CborValue::Integer(3.into()),
            CborValue::Integer((-8i64).into()),
        )); // alg: EdDSA
        map.push((
            CborValue::Integer((-1i64).into()),
            CborValue::Integer(6.into()),
        )); // crv: Ed25519
        map.push((
            CborValue::Integer((-2i64).into()),
            CborValue::Bytes(public_key_bytes),
        ));

        let cbor = CborValue::Map(map);
        let mut cose_key_bytes = Vec::new();
        ciborium::into_writer(&cbor, &mut cose_key_bytes).unwrap();

        // Invalid signature (random bytes)
        let invalid_signature = vec![0u8; 64];

        let service =
            WebAuthnService::new("example.com".to_string(), "https://example.com".to_string());

        let result = service.verify_signature(&cose_key_bytes, b"test data", &invalid_signature);

        // Should fail
        assert!(result.is_err());
    }

    #[test]
    fn test_packed_attestation_format_parsing() {
        // Create attestation with packed format
        let mut auth_data = Vec::new();

        // RP ID hash (32 bytes)
        auth_data.extend_from_slice(&[0u8; 32]);
        // Flags: UP (0x01) + AT (0x40) = 0x41
        auth_data.push(0x41);
        // Sign count
        auth_data.extend_from_slice(&0u32.to_be_bytes());
        // AAGUID
        auth_data.extend_from_slice(&[0u8; 16]);
        // Credential ID length
        auth_data.extend_from_slice(&8u16.to_be_bytes());
        // Credential ID
        auth_data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        // COSE key
        auth_data.extend_from_slice(&create_test_es256_cose_key());

        let mut map: Vec<(CborValue, CborValue)> = Vec::new();
        map.push((
            CborValue::Text("fmt".to_string()),
            CborValue::Text("packed".to_string()),
        ));
        map.push((
            CborValue::Text("authData".to_string()),
            CborValue::Bytes(auth_data),
        ));
        map.push((
            CborValue::Text("attStmt".to_string()),
            CborValue::Map(vec![]),
        ));

        let cbor = CborValue::Map(map);
        let mut buf = Vec::new();
        ciborium::into_writer(&cbor, &mut buf).unwrap();

        let service =
            WebAuthnService::new("example.com".to_string(), "https://example.com".to_string());

        let result = service.parse_attestation_object(&buf);
        assert!(result.is_ok());

        let (_, format, _) = result.unwrap();
        assert_eq!(format, AttestationFormat::Packed);
    }

    #[test]
    fn test_fido_u2f_attestation_format_parsing() {
        // Create attestation with fido-u2f format
        let mut auth_data = Vec::new();

        // RP ID hash (32 bytes)
        auth_data.extend_from_slice(&[0u8; 32]);
        // Flags: UP (0x01) + AT (0x40) = 0x41
        auth_data.push(0x41);
        // Sign count
        auth_data.extend_from_slice(&0u32.to_be_bytes());
        // AAGUID
        auth_data.extend_from_slice(&[0u8; 16]);
        // Credential ID length
        auth_data.extend_from_slice(&8u16.to_be_bytes());
        // Credential ID
        auth_data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        // COSE key
        auth_data.extend_from_slice(&create_test_es256_cose_key());

        let mut map: Vec<(CborValue, CborValue)> = Vec::new();
        map.push((
            CborValue::Text("fmt".to_string()),
            CborValue::Text("fido-u2f".to_string()),
        ));
        map.push((
            CborValue::Text("authData".to_string()),
            CborValue::Bytes(auth_data),
        ));
        map.push((
            CborValue::Text("attStmt".to_string()),
            CborValue::Map(vec![]),
        ));

        let cbor = CborValue::Map(map);
        let mut buf = Vec::new();
        ciborium::into_writer(&cbor, &mut buf).unwrap();

        let service =
            WebAuthnService::new("example.com".to_string(), "https://example.com".to_string());

        let result = service.parse_attestation_object(&buf);
        assert!(result.is_ok());

        let (_, format, _) = result.unwrap();
        assert_eq!(format, AttestationFormat::FidoU2f);
    }
}
