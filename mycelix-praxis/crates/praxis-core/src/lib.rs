// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # EduNet Core
//!
//! Core types, cryptographic helpers, and provenance utilities for Mycelix EduNet.
//!
//! This crate provides:
//! - Common data structures used across zomes
//! - Cryptographic primitives (hashing, signatures)
//! - Provenance tracking for models and credentials
//! - Validation utilities
//! - Proof of Learning (PoL) for verifying genuine learning
//! - Structured error handling with descriptive messages

pub mod crypto;
pub mod errors;
pub mod provenance;
pub mod proof_of_learning;
pub mod types;
pub mod validation;
pub mod export_formats;
mod benchmarks;

pub use crypto::*;
pub use errors::*;
pub use provenance::*;
pub use proof_of_learning::*;
pub use types::*;

/// Current protocol version
pub const PROTOCOL_VERSION: &str = "0.1.0";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version() {
        assert_eq!(PROTOCOL_VERSION, "0.1.0");
    }
}
