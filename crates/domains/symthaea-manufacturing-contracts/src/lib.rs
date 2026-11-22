// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public composition contracts for Symthaea manufacturing semantics.
//!
//! This crate intentionally depends on the small `symthaea-manufacturing-process`
//! identity kernel and lifts the already-frozen state, capability, recipe, and
//! process-plan semantics into reusable public APIs. It does not add machine
//! execution authority.

pub mod capability;
pub mod plan;
pub mod recipe;
pub mod state;

pub use capability::*;
pub use plan::*;
pub use recipe::*;
pub use state::*;

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ManufacturingContractError {
    #[error("{field} must be canonical and non-empty")]
    NonCanonical { field: &'static str },
    #[error("duplicate reference in {field}: {value}")]
    DuplicateReference { field: &'static str, value: String },
    #[error("invalid manufacturing contract: {0}")]
    Invalid(&'static str),
    #[error("{field} must be a lowercase 64-character hexadecimal digest")]
    InvalidDigest { field: &'static str },
}

pub(crate) fn canonical_token(
    field: &'static str,
    value: &str,
) -> Result<(), ManufacturingContractError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(ManufacturingContractError::NonCanonical { field });
    }
    Ok(())
}

pub(crate) fn validate_digest(
    field: &'static str,
    value: &str,
) -> Result<(), ManufacturingContractError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ManufacturingContractError::InvalidDigest { field });
    }
    Ok(())
}

pub(crate) fn hash_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}
