// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # symthaea-hdc-crypto
//!
//! **INSECURE EXPERIMENTAL RESEARCH CODE. DO NOT USE FOR SECURITY.**
//!
//! This crate preserves HDC masking and algebra demonstrations for compatibility
//! and adversarial study. It does not provide a secure MAC, threshold secret
//! sharing, commitment scheme, FHE, privacy-preserving aggregation, or a secure
//! sensor-derived KDF.
//!
//! ## Core Idea
//!
//! Binary hypervectors support natural homomorphic operations:
//!
//! ```text
//! bind(enc(A), enc(B)) = enc(bind(A, B))       [XOR distributes over XOR]
//! bundle(enc(A), enc(B)) ~ enc(bundle(A, B))   [majority vote preserves under shared mask]
//! ```
//!
//! These identities demonstrate XOR/majority algebra. They do not establish a
//! privacy-preserving protocol.
//!
//! ## Security Model
//!
//! A fresh uniform XOR mask can implement a one-time pad for one message. The
//! higher-level constructions in this crate violate or fail to enforce the
//! assumptions required for their former security claims.
//!
//! ## What This Is NOT
//!
//! - Not a general-purpose FHE scheme (supports only HDC algebra: bind, bundle, similarity)
//! - Not a replacement for AES/ChaCha20 (use those for bulk data encryption)
//! - Not a digital signature scheme (use Ed25519 for signatures)
//!
//! ## References
//!
//! - Shannon, C. (1949). Communication theory of secrecy systems.
//! - Kanerva, P. (2009). Hyperdimensional computing: An introduction.
//! - Imani et al. (2019). A framework for collaborative learning in secure HDC.

pub mod binary_hv;
pub mod crypto;
pub mod fhe;

pub use binary_hv::BinaryHV;

/// Root re-exports of the quarantined types below require this explicit
/// feature so a consumer of this crate cannot stumble onto
/// `symthaea_hdc_crypto::HdcMac` or `::EncryptedHV` looking like ordinary,
/// safe security APIs. Use `crypto::HdcMac` / `fhe::EncryptedHV` etc.
/// directly (still available unconditionally) if you specifically need to
/// study or exercise these constructions.
#[cfg(feature = "insecure-experimental-crypto")]
pub use crypto::{HdcCommitment, HdcContextKey, HdcMac, HdcShare, HdcThresholdSharing};
#[cfg(feature = "insecure-experimental-crypto")]
pub use fhe::{CollectiveWisdomPool, EncryptedHV, generate_collective_mask};
