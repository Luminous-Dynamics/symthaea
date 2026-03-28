// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # mycelix-crypto
//!
//! Crypto-agile types for the Mycelix Identity system.
//!
//! This crate provides algorithm-tagged wrappers for public keys, signatures,
//! and encrypted envelopes that enable the identity hApp to work with multiple
//! cryptographic algorithms (Ed25519 today, ML-DSA/hybrid/ML-KEM in the future)
//! without hardcoding any single algorithm.
//!
//! ## Feature flags
//!
//! - **`wasm`** — Types and validation only; suitable for `wasm32-unknown-unknown`
//!   targets (Holochain WASM zomes). No actual cryptographic operations.
//!
//! - **`native`** — Adds trait implementations backed by real cryptographic
//!   libraries (ed25519-dalek, pqcrypto-dilithium, pqcrypto-kyber, etc.).
//!   Used by the CLI and tests.

pub mod algorithm;
pub mod envelope;
pub mod error;

#[cfg(feature = "native")]
pub mod traits;

#[cfg(feature = "native")]
pub mod pqc;

pub use algorithm::AlgorithmId;
pub use envelope::{EncryptedEnvelope, TaggedPublicKey, TaggedSignature};
pub use error::CryptoError;

#[cfg(feature = "native")]
pub use traits::{Encryptor, KeyEncapsulator, Signer, Verifier};
