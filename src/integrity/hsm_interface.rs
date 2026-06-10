// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hardware Security Module (HSM/TPM) Interface (Design Only)
//!
//! Defines traits for storing integrity-critical values in hardware security modules.
//! Values stored in an HSM cannot be modified without physical access to the hardened chip,
//! protecting against software-level tampering of moral algebra weights, safety thresholds,
//! and harmony basis vectors.
//!
//! This is a future capability — actual HSM binding requires platform-specific
//! TPM2 or PKCS#11 libraries. The traits are defined now for interface stability.

use std::fmt;

/// Handle to a value stored in the HSM.
#[derive(Debug, Clone)]
pub struct SecureHandle {
    /// Unique identifier for the stored value.
    pub id: [u8; 32],
    /// Human-readable name for the stored value.
    pub name: String,
}

/// Error from HSM operations.
#[derive(Debug)]
pub enum HsmError {
    /// HSM is not available on this platform.
    NotAvailable,
    /// Value not found by handle.
    NotFound(String),
    /// Integrity verification failed during retrieval.
    IntegrityViolation(String),
    /// General HSM communication error.
    Communication(String),
}

impl fmt::Display for HsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAvailable => write!(f, "HSM not available"),
            Self::NotFound(name) => write!(f, "HSM value not found: {name}"),
            Self::IntegrityViolation(msg) => write!(f, "HSM integrity violation: {msg}"),
            Self::Communication(msg) => write!(f, "HSM communication error: {msg}"),
        }
    }
}

impl std::error::Error for HsmError {}

/// Trait for secure storage of integrity-critical values.
///
/// Implementations may use TPM2, PKCS#11, Intel SGX, or other hardware
/// security mechanisms. The `SoftwareSecureStore` provides an in-memory
/// fallback for platforms without hardware HSM support.
pub trait SecureStore: Send + Sync {
    /// Store a named value in the HSM. Returns a handle for retrieval.
    fn store(&self, name: &str, data: &[u8]) -> Result<SecureHandle, HsmError>;

    /// Retrieve a value by handle. The HSM verifies integrity.
    fn retrieve(&self, handle: &SecureHandle) -> Result<Vec<u8>, HsmError>;

    /// Sign data with the HSM's attestation key.
    fn sign_attestation(&self, data: &[u8]) -> Result<Vec<u8>, HsmError>;

    /// Verify a signature produced by this or a peer HSM.
    fn verify_attestation(&self, data: &[u8], signature: &[u8]) -> Result<bool, HsmError>;

    /// Check if the HSM is available and operational.
    fn is_available(&self) -> bool;
}

/// Software-only fallback store (no actual hardware security).
///
/// Stores values in memory with BLAKE3 integrity verification.
/// Protects against accidental corruption but NOT deliberate tampering
/// (an attacker with memory access can modify both the value and its hash).
pub struct SoftwareSecureStore {
    entries: std::sync::Mutex<Vec<(SecureHandle, Vec<u8>, [u8; 32])>>,
}

impl SoftwareSecureStore {
    /// Create a new software-only store.
    pub fn new() -> Self {
        Self {
            entries: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl Default for SoftwareSecureStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SecureStore for SoftwareSecureStore {
    fn store(&self, name: &str, data: &[u8]) -> Result<SecureHandle, HsmError> {
        let hash = *blake3::hash(data).as_bytes();
        let handle = SecureHandle {
            id: *blake3::hash(name.as_bytes()).as_bytes(),
            name: name.to_string(),
        };
        let mut entries = self
            .entries
            .lock()
            .map_err(|e| HsmError::Communication(format!("Lock poisoned: {e}")))?;
        entries.push((handle.clone(), data.to_vec(), hash));
        Ok(handle)
    }

    fn retrieve(&self, handle: &SecureHandle) -> Result<Vec<u8>, HsmError> {
        let entries = self
            .entries
            .lock()
            .map_err(|e| HsmError::Communication(format!("Lock poisoned: {e}")))?;
        for (h, data, stored_hash) in entries.iter() {
            if h.id == handle.id {
                // Verify integrity
                let current_hash = *blake3::hash(data).as_bytes();
                if current_hash != *stored_hash {
                    return Err(HsmError::IntegrityViolation(format!(
                        "Value '{}' has been modified in memory",
                        handle.name
                    )));
                }
                return Ok(data.clone());
            }
        }
        Err(HsmError::NotFound(handle.name.clone()))
    }

    fn sign_attestation(&self, data: &[u8]) -> Result<Vec<u8>, HsmError> {
        // Software fallback: BLAKE3 keyed hash (not cryptographically secure as a signature,
        // but sufficient for detecting accidental corruption)
        let key = [0x53u8; 32]; // Fixed key — NOT secure, just a stub
        let hash = blake3::keyed_hash(&key, data);
        Ok(hash.as_bytes().to_vec())
    }

    fn verify_attestation(&self, data: &[u8], signature: &[u8]) -> Result<bool, HsmError> {
        let key = [0x53u8; 32];
        let expected = blake3::keyed_hash(&key, data);
        Ok(signature == expected.as_bytes())
    }

    fn is_available(&self) -> bool {
        true // Software store is always available
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_software_store_roundtrip() {
        let store = SoftwareSecureStore::new();
        let data = b"moral_algebra_weights_v1";
        let handle = store.store("moral_weights", data).unwrap();
        let retrieved = store.retrieve(&handle).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_software_store_not_found() {
        let store = SoftwareSecureStore::new();
        let handle = SecureHandle {
            id: [0; 32],
            name: "nonexistent".into(),
        };
        assert!(matches!(
            store.retrieve(&handle),
            Err(HsmError::NotFound(_))
        ));
    }

    #[test]
    fn test_software_attestation_roundtrip() {
        let store = SoftwareSecureStore::new();
        let data = b"test attestation data";
        let sig = store.sign_attestation(data).unwrap();
        assert!(store.verify_attestation(data, &sig).unwrap());
    }

    #[test]
    fn test_software_attestation_fails_on_tamper() {
        let store = SoftwareSecureStore::new();
        let data = b"original data";
        let sig = store.sign_attestation(data).unwrap();
        let tampered = b"tampered data";
        assert!(!store.verify_attestation(tampered, &sig).unwrap());
    }

    #[test]
    fn test_software_store_is_available() {
        let store = SoftwareSecureStore::new();
        assert!(store.is_available());
    }
}
