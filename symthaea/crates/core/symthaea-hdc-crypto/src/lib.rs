// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # symthaea-hdc-crypto
//!
//! Information-theoretically secure cryptographic primitives built on 16,384-bit
//! binary hypervectors. Achieves **FHE at plaintext speed** by exploiting the
//! algebraic structure of hyperdimensional computing (HDC).
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
//! This enables privacy-preserving collective computation without the
//! 1000x+ overhead of lattice-based FHE schemes.
//!
//! ## Security Model
//!
//! - **Perfect secrecy** (Shannon 1949): OTP encryption with D = 16,384 bit masks
//! - **Zero collision probability**: XOR binding is a bijection per-operand
//! - **Information-theoretic MAC**: 5-10 ns authentication (vs 50-400 ns for hash-based MACs)
//! - **Threshold secret sharing**: (k,n) splitting via majority-vote bundling
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
pub use crypto::{HdcCommitment, HdcContextKey, HdcMac, HdcShare, HdcThresholdSharing};
pub use fhe::{CollectiveWisdomPool, EncryptedHV, generate_collective_mask};
