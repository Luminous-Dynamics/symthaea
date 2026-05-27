// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # mycelix-zkp-core
//!
//! Shared dual-backend ZKP verification library for the Mycelix ecosystem.
//!
//! Implements the DASTARK pattern: RISC0 zkVM (general-purpose) + Winterfell STARK
//! (domain-specific, 3-10x faster) + CRYSTALS-Dilithium5 (post-quantum authentication).
//!
//! ## Architecture
//!
//! ```text
//! Client (native)                    Zome (WASM)
//! ┌──────────────┐                  ┌──────────────────┐
//! │ ZKBackend    │  proof bytes     │ verify_proof()   │
//! │  .prove()    │ ──────────────>  │  (300-500KB)     │
//! │ Dilithium5   │  + signature     │ verify_dilithium │
//! │  .sign()     │                  │  (<1ms)          │
//! └──────────────┘                  └──────────────────┘
//! ```
//!
//! ## Feature Flags
//!
//! - `backend-winterfell`: Winterfell STARK verifier (~200-400KB WASM)
//! - `backend-risc0`: RISC0 zkVM verifier (~300-500KB WASM)
//! - `backend-dual`: Both backends
//! - `dilithium`: CRYSTALS-Dilithium5 PQ signatures (~2.6KB WASM)
//! - `full`: All backends + Dilithium (native testing only)

pub mod backend;
pub mod circuits;
pub mod consciousness;
#[cfg(feature = "dilithium")]
pub mod dilithium;
pub mod domain;
pub mod error;
pub mod fixed_point;
pub mod mission_audit;
pub mod pogq;
pub mod types;

// Re-exports
pub use backend::ProofBackend;
pub use consciousness::{CivicTier, ConsciousnessProofRequest, ConsciousnessProofResult};
#[cfg(feature = "dilithium")]
pub use dilithium::DilithiumKeypair;
pub use domain::DomainTag;
pub use error::{ZkpError, ZkpResult};
pub use fixed_point::{FixedPoint, Q16_16_SCALE};
pub use pogq::{DualBackendComparison, PoGQPublicInputs, PoGQResult, PoGQWitness, simulate_pogq};
pub use types::{AuthenticatedProof, ProofBytes, ProofMetadata, ProofResult, VerificationResult};

// Re-export shared ecosystem types
pub use proofs_commitment::{CommitmentHash, CommitmentScheme, Sha3Commitment};
pub use proofs_config::{ProofConfig, SecurityLevel, UseCase};
