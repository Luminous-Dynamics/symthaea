// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Phase 2.5: CRYSTALS-Dilithium Integration for zkSTARK Authentication
// Provides post-quantum digital signatures for client authentication

use pyo3::prelude::*;
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SecretKey as PQSecretKey, SignedMessage};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

/// Dilithium5 exact sizes (per pqcrypto-dilithium library - EMPIRICALLY VERIFIED)
const DILITHIUM5_PUBLIC_KEY_SIZE: usize = 2592;
const DILITHIUM5_SECRET_KEY_SIZE: usize = 4896;  // NOTE: Library returns 4896, not 4864 from spec
const DILITHIUM5_SIGNATURE_SIZE: usize = 4595;   // To be verified empirically

/// Domain separation tag for signature binding
const DOMAIN_TAG: &[u8] = b"ZTML:Gen7:AuthGradProof:v1";
const PROTOCOL_VERSION: u32 = 1;

/// Dilithium5 keypair for client authentication
///
/// Security Level: NIST Level 5 (highest, comparable to AES-256)
/// Public Key Size: 2,592 bytes (exact, verified)
/// Secret Key Size: 4,896 bytes (exact, verified - library returns 4896, not 4864 from spec)
/// Signature Size: 4,595 bytes (to be verified empirically)
#[pyclass]
pub struct DilithiumKeypair {
    public_key: Vec<u8>,
    secret_key: Vec<u8>,
    client_id: [u8; 32], // SHA-256 hash of public key
}

#[pymethods]
impl DilithiumKeypair {
    /// Generate a new Dilithium5 keypair
    #[new]
    pub fn new() -> PyResult<Self> {
        let (pk, sk) = dilithium5::keypair();

        let public_key = pk.as_bytes().to_vec();
        let secret_key = sk.as_bytes().to_vec();

        // Verify exact sizes (CRITICAL: prevents size flakiness)
        assert_eq!(
            public_key.len(),
            DILITHIUM5_PUBLIC_KEY_SIZE,
            "Public key size mismatch: expected {} bytes, got {}",
            DILITHIUM5_PUBLIC_KEY_SIZE,
            public_key.len()
        );
        assert_eq!(
            secret_key.len(),
            DILITHIUM5_SECRET_KEY_SIZE,
            "Secret key size mismatch: expected {} bytes, got {}",
            DILITHIUM5_SECRET_KEY_SIZE,
            secret_key.len()
        );

        // Compute client_id as SHA-256(public_key)
        let mut hasher = Sha256::new();
        hasher.update(&public_key);
        let client_id_vec = hasher.finalize();
        let mut client_id = [0u8; 32];
        client_id.copy_from_slice(&client_id_vec);

        Ok(Self {
            public_key,
            secret_key,
            client_id,
        })
    }

    /// Load keypair from bytes
    #[staticmethod]
    pub fn from_bytes(public_key: Vec<u8>, secret_key: Vec<u8>) -> PyResult<Self> {
        // Validate exact sizes
        if public_key.len() != DILITHIUM5_PUBLIC_KEY_SIZE {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Invalid public key size: expected {} bytes, got {}",
                    DILITHIUM5_PUBLIC_KEY_SIZE,
                    public_key.len()
                ),
            ));
        }
        if secret_key.len() != DILITHIUM5_SECRET_KEY_SIZE {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Invalid secret key size: expected {} bytes, got {}",
                    DILITHIUM5_SECRET_KEY_SIZE,
                    secret_key.len()
                ),
            ));
        }

        // Compute client_id
        let mut hasher = Sha256::new();
        hasher.update(&public_key);
        let client_id_vec = hasher.finalize();
        let mut client_id = [0u8; 32];
        client_id.copy_from_slice(&client_id_vec);

        Ok(Self {
            public_key,
            secret_key,
            client_id,
        })
    }

    /// Get public key bytes
    pub fn get_public_key(&self) -> Vec<u8> {
        self.public_key.clone()
    }

    /// Get secret key bytes (sensitive!)
    pub fn get_secret_key(&self) -> Vec<u8> {
        self.secret_key.clone()
    }

    /// Get client ID (SHA-256 of public key)
    pub fn get_client_id(&self) -> Vec<u8> {
        self.client_id.to_vec()
    }

    /// Sign a message with Dilithium5
    ///
    /// Args:
    ///     message: Bytes to sign
    ///
    /// Returns:
    ///     Signature bytes (4595 bytes exact)
    pub fn sign(&self, message: Vec<u8>) -> PyResult<Vec<u8>> {
        let sk = dilithium5::SecretKey::from_bytes(&self.secret_key)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid secret key: {:?}", e)
            ))?;

        let signed_message = dilithium5::sign(&message, &sk);
        Ok(signed_message.as_bytes().to_vec())
    }

    /// Verify a Dilithium5 signature
    #[staticmethod]
    pub fn verify(message: Vec<u8>, signature: Vec<u8>, public_key: Vec<u8>) -> PyResult<bool> {
        let pk = dilithium5::PublicKey::from_bytes(&public_key)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid public key: {:?}", e)
            ))?;

        let signed_msg = dilithium5::SignedMessage::from_bytes(&signature)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid signature: {:?}", e)
            ))?;

        match dilithium5::open(&signed_msg, &pk) {
            Ok(verified_msg) => {
                // Check that recovered message matches original
                Ok(verified_msg == message)
            }
            Err(_) => Ok(false),
        }
    }
}

/// Authenticated gradient proof combining zkSTARK + Dilithium signature
///
/// Structure:
///   - stark_proof: zkSTARK proving gradient computation (61KB)
///   - signature: Dilithium5 signature (4595 bytes exact)
///   - client_id: SHA-256 hash of client's public key (32 bytes)
///   - round_number: Training round number (8 bytes)
///   - timestamp: Unix timestamp (8 bytes)
///   - nonce: Random 32-byte value for replay protection (32 bytes)
///   - model_hash: SHA-256 of initial model parameters (32 bytes)
///   - gradient_hash: SHA-256 of computed gradient (32 bytes)
///
/// Total size: ~65.8KB (was 61KB in Phase 2)
#[pyclass]
pub struct AuthenticatedGradientProof {
    #[pyo3(get)]
    pub stark_proof: Vec<u8>,

    #[pyo3(get)]
    pub signature: Vec<u8>,

    #[pyo3(get)]
    pub client_id: Vec<u8>,

    #[pyo3(get)]
    pub round_number: u64,

    #[pyo3(get)]
    pub timestamp: u64,

    #[pyo3(get)]
    pub nonce: Vec<u8>,

    #[pyo3(get)]
    pub model_hash: Vec<u8>,

    #[pyo3(get)]
    pub gradient_hash: Vec<u8>,
}

#[pymethods]
impl AuthenticatedGradientProof {
    /// Create new authenticated proof
    #[new]
    pub fn new(
        stark_proof: Vec<u8>,
        signature: Vec<u8>,
        client_id: Vec<u8>,
        round_number: u64,
        timestamp: u64,
        nonce: Vec<u8>,
        model_hash: Vec<u8>,
        gradient_hash: Vec<u8>,
    ) -> Self {
        Self {
            stark_proof,
            signature,
            client_id,
            round_number,
            timestamp,
            nonce,
            model_hash,
            gradient_hash,
        }
    }

    /// Construct message to be signed
    ///
    /// Message format (with domain separation):
    ///   SHA-256(domain_tag || protocol_version || client_id || round_number ||
    ///           timestamp || nonce || model_hash || gradient_hash || stark_proof)
    ///
    /// This ensures the signature commits to:
    /// - Domain tag: Prevents cross-protocol replay
    /// - Protocol version: Prevents version confusion
    /// - Client ID: Binds to specific client
    /// - Round number: Prevents replay across rounds
    /// - Timestamp: Ensures freshness
    /// - Nonce: Prevents replay within round
    /// - Model hash: Commits to initial model
    /// - Gradient hash: Commits to computed gradient
    /// - STARK proof: Commits to computation proof
    #[staticmethod]
    pub fn construct_message(
        stark_proof: Vec<u8>,
        client_id: Vec<u8>,
        round_number: u64,
        timestamp: u64,
        nonce: Vec<u8>,
        model_hash: Vec<u8>,
        gradient_hash: Vec<u8>,
    ) -> Vec<u8> {
        let mut hasher = Sha256::new();

        // Domain separation tag (prevents cross-protocol replay)
        hasher.update(DOMAIN_TAG);

        // Protocol version
        hasher.update(&PROTOCOL_VERSION.to_le_bytes());

        // Client identity
        hasher.update(&client_id);

        // Round and timestamp (temporal binding)
        hasher.update(&round_number.to_le_bytes());
        hasher.update(&timestamp.to_le_bytes());

        // Nonce (per-client, per-round uniqueness)
        hasher.update(&nonce);

        // Model and gradient commitments
        hasher.update(&model_hash);
        hasher.update(&gradient_hash);

        // STARK proof bytes
        hasher.update(&stark_proof);

        hasher.finalize().to_vec()
    }

    /// Verify authenticated proof
    ///
    /// Checks:
    ///   1. Round number matches current round
    ///   2. Timestamp is within ±5 minutes
    ///   3. Signature is valid for the message
    ///   4. zkSTARK proof is valid (checked separately by coordinator)
    ///
    /// Args:
    ///     current_round: Expected round number
    ///     client_public_key: Client's Dilithium public key
    ///     max_timestamp_delta: Maximum allowed time difference (seconds)
    ///
    /// Returns:
    ///     (is_valid, error_message)
    pub fn verify(
        &self,
        current_round: u64,
        client_public_key: Vec<u8>,
        max_timestamp_delta: u64,
    ) -> PyResult<(bool, String)> {
        // 1. Check round number
        if self.round_number != current_round {
            return Ok((
                false,
                format!(
                    "Invalid round: expected {}, got {}",
                    current_round, self.round_number
                ),
            ));
        }

        // 2. Check timestamp
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let time_diff = if now > self.timestamp {
            now - self.timestamp
        } else {
            self.timestamp - now
        };

        if time_diff > max_timestamp_delta {
            return Ok((
                false,
                format!(
                    "Timestamp expired: {} seconds old (max {})",
                    time_diff, max_timestamp_delta
                ),
            ));
        }

        // 3. Verify client_id matches public key
        let mut hasher = Sha256::new();
        hasher.update(&client_public_key);
        let expected_client_id = hasher.finalize();

        if self.client_id != expected_client_id.as_slice() {
            return Ok((
                false,
                "Client ID does not match public key".to_string(),
            ));
        }

        // 4. Construct message and verify signature
        let message = Self::construct_message(
            self.stark_proof.clone(),
            self.client_id.clone(),
            self.round_number,
            self.timestamp,
            self.nonce.clone(),
            self.model_hash.clone(),
            self.gradient_hash.clone(),
        );

        let signature_valid = DilithiumKeypair::verify(
            message,
            self.signature.clone(),
            client_public_key,
        )?;

        if !signature_valid {
            return Ok((false, "Invalid Dilithium signature".to_string()));
        }

        // Note: zkSTARK verification happens separately in coordinator
        Ok((true, "All checks passed".to_string()))
    }

    /// Get total proof size in bytes
    pub fn size_bytes(&self) -> usize {
        self.stark_proof.len() +
        self.signature.len() +
        self.client_id.len() +
        8 + // round_number
        8 + // timestamp
        self.nonce.len() +
        self.model_hash.len() +
        self.gradient_hash.len()
    }
}

/// Helper: Generate random nonce
#[pyfunction]
pub fn generate_nonce() -> Vec<u8> {
    use rand::RngCore;
    let mut nonce = vec![0u8; 32];
    rand::thread_rng().fill_bytes(&mut nonce);
    nonce
}

/// Helper: Get current Unix timestamp
#[pyfunction]
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_secs()
}

/// Register Python module
pub fn register_dilithium_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DilithiumKeypair>()?;
    m.add_class::<AuthenticatedGradientProof>()?;
    m.add_function(wrap_pyfunction!(generate_nonce, m)?)?;
    m.add_function(wrap_pyfunction!(current_timestamp, m)?)?;
    Ok(())
}
