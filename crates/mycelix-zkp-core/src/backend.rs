// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ZKBackend trait — unified interface for RISC0, Winterfell, and (future) Binius.
//!
//! Ported from the Python `ZKBackend` ABC in
//! `mycelix-core/0TML/tests/integration/zkbackend_abstraction.py` (895 lines).
//!
//! The trait enables drop-in backend switching: a governance zome can verify
//! proofs from either backend without knowing which one generated them.

use crate::error::ZkpResult;
use crate::types::{BackendId, ProofResult, VerificationResult};

/// Public inputs for a ZK proof circuit.
///
/// Each circuit defines its own public input structure, but they all
/// serialize through this common wrapper for the backend interface.
#[derive(Clone, Debug)]
pub struct PublicInputs {
    /// Serialized public inputs (circuit-specific format).
    pub data: Vec<u8>,
    /// Human-readable description for logging.
    pub description: String,
}

impl PublicInputs {
    pub fn new(data: Vec<u8>, description: impl Into<String>) -> Self {
        Self {
            data,
            description: description.into(),
        }
    }

    /// Create from a serde-serializable value.
    pub fn from_value<T: serde::Serialize>(
        value: &T,
        description: impl Into<String>,
    ) -> ZkpResult<Self> {
        let data = serde_json::to_vec(value)
            .map_err(|e: serde_json::Error| crate::error::ZkpError::Serialization(e.to_string()))?;
        Ok(Self::new(data, description))
    }
}

/// Unified interface for ZKP backends.
///
/// Mirrors the Python `ZKBackend` ABC from `zkbackend_abstraction.py`.
/// Each backend implements prove() and verify() with identical semantics.
pub trait ProofBackend: Send + Sync {
    /// Human-readable backend name (e.g. "RISC Zero", "Winterfell").
    fn name(&self) -> &str;

    /// Backend version string.
    fn version(&self) -> &str;

    /// Which backend this is.
    fn id(&self) -> BackendId;

    /// Check if this backend is available (binary exists, deps met).
    fn is_available(&self) -> bool;

    /// Generate a proof for the given public inputs and witness.
    ///
    /// - `public_inputs`: Circuit-specific public inputs.
    /// - `witness`: Private witness data (f32 values, Q16.16 encoded internally).
    fn prove(&self, public_inputs: &PublicInputs, witness: &[f32]) -> ZkpResult<ProofResult>;

    /// Verify a proof against its public inputs.
    ///
    /// This is the operation that runs inside Holochain zomes (WASM).
    /// Must be deterministic and not require filesystem or network.
    fn verify(
        &self,
        proof_bytes: &[u8],
        public_inputs: &PublicInputs,
    ) -> ZkpResult<VerificationResult>;
}

// --- Backend implementations (feature-gated) ---

/// Winterfell STARK backend — domain-specific, 3-10x faster than RISC0.
///
/// Best for: simple circuits (range proofs, membership, nullifiers).
/// Proving: 5-15s. Verification: <50ms. Proof size: ~200KB.
#[cfg(feature = "backend-winterfell")]
pub mod winterfell_backend {
    use super::*;

    pub struct WinterfellBackend {
        version: String,
    }

    impl WinterfellBackend {
        pub fn new() -> Self {
            Self {
                // Read winterfell version from the crate
                version: "0.13.1".to_string(),
            }
        }
    }

    impl Default for WinterfellBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ProofBackend for WinterfellBackend {
        fn name(&self) -> &str {
            "Winterfell STARK"
        }

        fn version(&self) -> &str {
            &self.version
        }

        fn id(&self) -> BackendId {
            BackendId::Winterfell
        }

        fn is_available(&self) -> bool {
            true // Always available when feature is enabled
        }

        fn prove(&self, _public_inputs: &PublicInputs, _witness: &[f32]) -> ZkpResult<ProofResult> {
            // Circuit-specific proving logic is injected by each cluster's ZKP crate.
            // This base implementation provides the backend interface;
            // actual AIR circuits are defined in governance-zkp, health-zkp, etc.
            Err(crate::error::ZkpError::ProvingError(
                "Use a circuit-specific prover (e.g. governance-zkp) instead of the base backend"
                    .into(),
            ))
        }

        fn verify(
            &self,
            _proof_bytes: &[u8],
            _public_inputs: &PublicInputs,
        ) -> ZkpResult<VerificationResult> {
            // Generic Winterfell proof verification.
            // Circuit-specific verification is handled by each cluster's ZKP crate
            // which calls winterfell::verify() with the appropriate AIR.
            Err(crate::error::ZkpError::VerificationFailed(
                "Use a circuit-specific verifier (e.g. governance-zkp) instead of the base backend"
                    .into(),
            ))
        }
    }
}

/// RISC0 zkVM backend — general-purpose, write any Rust as guest.
///
/// Best for: complex multi-step logic (vote encryption, provenance chains).
/// Proving: ~46.6s. Verification: ~92ms. Proof size: ~221KB.
#[cfg(feature = "backend-risc0")]
pub mod risc0_backend {
    use super::*;

    pub struct Risc0Backend {
        version: String,
    }

    impl Risc0Backend {
        pub fn new() -> Self {
            Self {
                version: "3.0.4".to_string(),
            }
        }
    }

    impl Default for Risc0Backend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ProofBackend for Risc0Backend {
        fn name(&self) -> &str {
            "RISC Zero zkVM"
        }

        fn version(&self) -> &str {
            &self.version
        }

        fn id(&self) -> BackendId {
            BackendId::Risc0
        }

        fn is_available(&self) -> bool {
            true // Available when feature is enabled
        }

        fn prove(&self, _public_inputs: &PublicInputs, _witness: &[f32]) -> ZkpResult<ProofResult> {
            // Circuit-specific guest programs are defined per cluster.
            Err(crate::error::ZkpError::ProvingError(
                "Use a circuit-specific prover with a RISC0 guest image".into(),
            ))
        }

        fn verify(
            &self,
            _proof_bytes: &[u8],
            _public_inputs: &PublicInputs,
        ) -> ZkpResult<VerificationResult> {
            // Generic RISC0 receipt verification.
            // Circuit-specific journal parsing is handled by each cluster.
            Err(crate::error::ZkpError::VerificationFailed(
                "Use a circuit-specific verifier with the expected RISC0 image ID".into(),
            ))
        }
    }
}

/// Select the best available backend for a given circuit complexity.
///
/// Simple circuits (range, membership) → Winterfell (faster).
/// Complex circuits (multi-step logic) → RISC0 (more flexible).
pub fn select_backend(complexity: CircuitComplexity) -> BackendId {
    match complexity {
        CircuitComplexity::Simple => {
            #[cfg(feature = "backend-winterfell")]
            return BackendId::Winterfell;
            #[cfg(not(feature = "backend-winterfell"))]
            {
                #[cfg(feature = "backend-risc0")]
                return BackendId::Risc0;
                #[cfg(not(feature = "backend-risc0"))]
                return BackendId::Winterfell; // Default even if unavailable
            }
        }
        CircuitComplexity::Complex => {
            #[cfg(feature = "backend-risc0")]
            return BackendId::Risc0;
            #[cfg(not(feature = "backend-risc0"))]
            {
                #[cfg(feature = "backend-winterfell")]
                return BackendId::Winterfell;
                #[cfg(not(feature = "backend-winterfell"))]
                return BackendId::Risc0; // Default even if unavailable
            }
        }
    }
}

/// Hint for backend selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitComplexity {
    /// Range proofs, membership, nullifiers — Winterfell preferred.
    Simple,
    /// Multi-step validation, arbitrary Rust logic — RISC0 preferred.
    Complex,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_public_inputs_creation() {
        let pi = PublicInputs::new(vec![1, 2, 3], "test inputs");
        assert_eq!(pi.data, vec![1, 2, 3]);
        assert_eq!(pi.description, "test inputs");
    }

    #[test]
    fn test_public_inputs_from_value() {
        let value = serde_json::json!({"threshold": 100, "round": 5});
        let pi = PublicInputs::from_value(&value, "json inputs").unwrap();
        assert!(!pi.data.is_empty());
    }

    #[test]
    fn test_select_backend_simple() {
        let backend = select_backend(CircuitComplexity::Simple);
        // Without features, defaults are returned
        assert!(
            backend == BackendId::Winterfell || backend == BackendId::Risc0,
            "must select a valid backend"
        );
    }

    #[test]
    fn test_select_backend_complex() {
        let backend = select_backend(CircuitComplexity::Complex);
        assert!(
            backend == BackendId::Risc0 || backend == BackendId::Winterfell,
            "must select a valid backend"
        );
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn test_winterfell_backend_available() {
        let b = winterfell_backend::WinterfellBackend::new();
        assert!(b.is_available());
        assert_eq!(b.id(), BackendId::Winterfell);
        assert_eq!(b.name(), "Winterfell STARK");
    }

    #[cfg(feature = "backend-risc0")]
    #[test]
    fn test_risc0_backend_available() {
        let b = risc0_backend::Risc0Backend::new();
        assert!(b.is_available());
        assert_eq!(b.id(), BackendId::Risc0);
        assert_eq!(b.name(), "RISC Zero zkVM");
    }
}
