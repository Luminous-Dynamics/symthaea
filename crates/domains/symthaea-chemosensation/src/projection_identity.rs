// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Content identities for the lossy ContinuousHV -> BinaryHV root projection.
//!
//! Chemical evidence, continuous representation geometry, and root projection
//! policy are separate contracts. The resulting BinaryHV comparison space is
//! therefore identified by both the source [`ChemicalEncodingSpaceId`] and the
//! projection policy used to quantize it.

use std::fmt;

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::{
    ChemicalEncodingSpaceId, ChemicalProjectionQuality, ChemicalRootProjection,
    ChemicalRootProjectionConfig, ChemicalRootProjectionError,
};

/// Identity of the algorithm/parameters used to project a continuous chemical
/// representation into BinaryHV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChemicalRootProjectionPolicyId([u8; 32]);

impl ChemicalRootProjectionPolicyId {
    pub fn sign_threshold(
        threshold: f32,
    ) -> Result<Self, ChemicalRootProjectionError> {
        if !threshold.is_finite() {
            return Err(ChemicalRootProjectionError::NonFiniteThreshold);
        }
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-chemosensation-root-projection-policy-v1");
        hasher.update(b"continuous-to-binary-sign-threshold");
        hasher.update(&threshold.to_bits().to_le_bytes());
        Ok(Self(*hasher.finalize().as_bytes()))
    }

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ChemicalRootProjectionPolicyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(f, &self.0)
    }
}

/// Identity of the BinaryHV coordinate/quantization space actually entering the
/// root multimodal integrator.
///
/// Equal binary-space IDs mean both the source ContinuousHV geometry and the
/// projection policy match. Equal dimensionality alone is not sufficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChemicalRootBinarySpaceId([u8; 32]);

impl ChemicalRootBinarySpaceId {
    pub fn from_parts(
        source_space: ChemicalEncodingSpaceId,
        policy: ChemicalRootProjectionPolicyId,
    ) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-chemosensation-root-binary-space-v1");
        hasher.update(source_space.as_bytes());
        hasher.update(policy.as_bytes());
        Self(*hasher.finalize().as_bytes())
    }

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ChemicalRootBinarySpaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(f, &self.0)
    }
}

impl ChemicalRootProjectionConfig {
    pub fn policy_id(
        &self,
    ) -> Result<ChemicalRootProjectionPolicyId, ChemicalRootProjectionError> {
        ChemicalRootProjectionPolicyId::sign_threshold(self.threshold)
    }
}

impl ChemicalProjectionQuality {
    pub fn policy_id(
        &self,
    ) -> Result<ChemicalRootProjectionPolicyId, ChemicalRootProjectionError> {
        ChemicalRootProjectionPolicyId::sign_threshold(self.threshold)
    }
}

impl ChemicalRootProjection {
    pub fn projection_policy_id(
        &self,
    ) -> Result<ChemicalRootProjectionPolicyId, ChemicalRootProjectionError> {
        self.quality.policy_id()
    }

    pub fn binary_space_id(
        &self,
    ) -> Result<ChemicalRootBinarySpaceId, ChemicalRootProjectionError> {
        Ok(ChemicalRootBinarySpaceId::from_parts(
            self.encoding_space_id,
            self.projection_policy_id()?,
        ))
    }
}

fn write_hex(f: &mut fmt::Formatter<'_>, bytes: &[u8; 32]) -> fmt::Result {
    for byte in bytes {
        write!(f, "{byte:02x}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChemicalBridgeTarget, ChemicalEvidenceBundleId};
    use symthaea_core::hdc::binary_hv::BinaryHV;

    #[test]
    fn projection_policy_identity_is_deterministic() {
        let a = ChemicalRootProjectionConfig { threshold: 0.0 };
        let b = ChemicalRootProjectionConfig { threshold: 0.0 };
        assert_eq!(a.policy_id().unwrap(), b.policy_id().unwrap());
    }

    #[test]
    fn invalid_threshold_cannot_obtain_projection_policy_identity() {
        assert!(matches!(
            ChemicalRootProjectionPolicyId::sign_threshold(f32::NAN),
            Err(ChemicalRootProjectionError::NonFiniteThreshold)
        ));
        assert!(matches!(
            ChemicalRootProjectionConfig {
                threshold: f32::INFINITY,
            }
            .policy_id(),
            Err(ChemicalRootProjectionError::NonFiniteThreshold)
        ));
    }

    #[test]
    fn changing_threshold_changes_projection_policy_identity() {
        let zero = ChemicalRootProjectionConfig { threshold: 0.0 }
            .policy_id()
            .unwrap();
        let shifted = ChemicalRootProjectionConfig { threshold: 0.01 }
            .policy_id()
            .unwrap();
        assert_ne!(zero, shifted);
    }

    #[test]
    fn binary_space_identity_depends_on_source_space_and_policy() {
        let source_a = ChemicalEncodingSpaceId::from_bytes([1; 32]);
        let source_b = ChemicalEncodingSpaceId::from_bytes([2; 32]);
        let policy_a = ChemicalRootProjectionPolicyId::sign_threshold(0.0).unwrap();
        let policy_b = ChemicalRootProjectionPolicyId::sign_threshold(0.01).unwrap();

        let baseline = ChemicalRootBinarySpaceId::from_parts(source_a, policy_a);
        assert_ne!(
            baseline,
            ChemicalRootBinarySpaceId::from_parts(source_b, policy_a)
        );
        assert_ne!(
            baseline,
            ChemicalRootBinarySpaceId::from_parts(source_a, policy_b)
        );
    }

    #[test]
    fn projection_exposes_policy_and_binary_space_identity() {
        let source_space = ChemicalEncodingSpaceId::from_bytes([3; 32]);
        let projection = ChemicalRootProjection {
            target: ChemicalBridgeTarget::Olfactory,
            evidence_bundle_id: ChemicalEvidenceBundleId::from_bytes([4; 32]),
            encoding_space_id: source_space,
            binary_vector: BinaryHV::zero(),
            confidence: 0.8,
            agreement: 0.9,
            earliest_timestamp_us: 10,
            latest_timestamp_us: 20,
            component_count: 2,
            quality: ChemicalProjectionQuality {
                threshold: 0.0,
                source_to_bipolar_similarity: 0.8,
                positive_fraction: 0.5,
            },
        };

        let policy = ChemicalRootProjectionPolicyId::sign_threshold(0.0).unwrap();
        assert_eq!(projection.projection_policy_id().unwrap(), policy);
        assert_eq!(
            projection.binary_space_id().unwrap(),
            ChemicalRootBinarySpaceId::from_parts(source_space, policy)
        );
    }
}
