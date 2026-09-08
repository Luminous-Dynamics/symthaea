// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Point-of-use composition of Forge compilation receipt binding and freshly
//! revalidated qualified-attestation currentness.
//!
//! This module closes one substitution class:
//!
//! ```text
//! receipt bound to qualified attestation A
//!     + fresh currentness witness for qualified attestation B
//!         != current receipt trust
//! ```
//!
//! The positive witness still grants no effect authority and cannot deserialize
//! or execute a compiled artifact.

use super::attestation_currentness::RevalidatedQualifiedAttestationCurrentnessV1;
use super::forge_compilation::ForgeCompiledArtifactIdentityV1;
use super::forge_compilation_attestation::ForgeCompilationAttestationBindingV1;
use std::error::Error;
use std::fmt;

/// Private in-process witness that one exact Forge compilation-attestation
/// binding is current at one exact point of use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentForgeCompilationAttestationV1 {
    binding: ForgeCompilationAttestationBindingV1,
    currentness: RevalidatedQualifiedAttestationCurrentnessV1,
    point_of_use_unix_s: u64,
    natural_valid_until_unix_s: u64,
}

impl CurrentForgeCompilationAttestationV1 {
    pub fn binding(&self) -> &ForgeCompilationAttestationBindingV1 {
        &self.binding
    }

    pub fn currentness(&self) -> &RevalidatedQualifiedAttestationCurrentnessV1 {
        &self.currentness
    }

    pub fn point_of_use_unix_s(&self) -> u64 {
        self.point_of_use_unix_s
    }

    pub fn natural_valid_until_unix_s(&self) -> u64 {
        self.natural_valid_until_unix_s
    }

    pub fn compiled_artifact(&self) -> &ForgeCompiledArtifactIdentityV1 {
        self.binding.compiled_artifact()
    }

    pub fn compilation_receipt_commitment(&self) -> [u8; 32] {
        self.binding.compilation_receipt_commitment()
    }

    pub fn execution_profile_commitment(&self) -> [u8; 32] {
        self.binding.execution_profile_commitment()
    }

    /// Time-only check. Any later consequential use should perform a fresh
    /// currentness revalidation rather than treating this witness as durable.
    pub fn remains_within_time_bounds(&self, at_unix_s: u64) -> bool {
        self.point_of_use_unix_s <= at_unix_s
            && at_unix_s < self.natural_valid_until_unix_s
    }
}

/// Compose one receipt-bound qualified worker attestation with one currentness
/// witness for the exact same qualified attestation at the exact point of use.
pub fn bind_current_forge_compilation_attestation_v1(
    binding: &ForgeCompilationAttestationBindingV1,
    currentness: &RevalidatedQualifiedAttestationCurrentnessV1,
    point_of_use_unix_s: u64,
) -> Result<CurrentForgeCompilationAttestationV1, CurrentForgeCompilationAttestationErrorV1> {
    if point_of_use_unix_s != currentness.revalidated_at_unix_s() {
        return Err(
            CurrentForgeCompilationAttestationErrorV1::PointOfUseTimeMismatch {
                currentness_at_unix_s: currentness.revalidated_at_unix_s(),
                requested_at_unix_s: point_of_use_unix_s,
            },
        );
    }

    if binding.qualified_attestation() != currentness.qualified() {
        return Err(
            CurrentForgeCompilationAttestationErrorV1::QualifiedAttestationMismatch,
        );
    }

    let natural_valid_until_unix_s = binding
        .natural_valid_until_unix_s()
        .min(currentness.natural_valid_until_unix_s());

    Ok(CurrentForgeCompilationAttestationV1 {
        binding: binding.clone(),
        currentness: currentness.clone(),
        point_of_use_unix_s,
        natural_valid_until_unix_s,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurrentForgeCompilationAttestationErrorV1 {
    PointOfUseTimeMismatch {
        currentness_at_unix_s: u64,
        requested_at_unix_s: u64,
    },
    QualifiedAttestationMismatch,
}

impl fmt::Display for CurrentForgeCompilationAttestationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PointOfUseTimeMismatch {
                currentness_at_unix_s,
                requested_at_unix_s,
            } => write!(
                f,
                "Forge point-of-use time {requested_at_unix_s} differs from currentness revalidation time {currentness_at_unix_s}"
            ),
            Self::QualifiedAttestationMismatch => write!(
                f,
                "Forge compilation receipt binding and point-of-use currentness refer to different qualified attestations"
            ),
        }
    }
}

impl Error for CurrentForgeCompilationAttestationErrorV1 {}

#[cfg(test)]
mod tests;
