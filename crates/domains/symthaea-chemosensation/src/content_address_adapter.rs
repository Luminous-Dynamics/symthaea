// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Adapters from chemosensation's strong domain IDs into the shared
//! [`ContentAddress32`] envelope.
//!
//! The domain IDs remain authoritative inside this crate. These adapters exist
//! only for crossing generic subsystem boundaries such as multimodal cognition.
//! No digest is recomputed: each generic address carries the exact 32-byte
//! BLAKE3-derived domain identity under an explicit semantic namespace.
//!
//! Observation and evidence-bundle addresses use v2 namespaces because their
//! underlying strong IDs now include declared chemical clock-domain provenance.
//! A v1 digest must never be silently reinterpreted as a v2 evidence contract.

use std::fmt;

use symthaea_evidence_plane::{ContentAddress32, ContentAddressError};

use crate::{
    ChemicalEncodingSpaceId, ChemicalEvidenceBundleId, ChemicalObservationId,
    ChemicalRootBinarySpaceId, ChemicalRootProjection, ChemicalRootProjectionError,
    ChemicalRootProjectionPolicyId,
};

pub const CHEMICAL_OBSERVATION_NAMESPACE: &str = "symthaea-chemosensation-observation-v2";
pub const CHEMICAL_EVIDENCE_BUNDLE_NAMESPACE: &str =
    "symthaea-chemosensation-evidence-bundle-v2";
pub const CHEMICAL_ENCODING_SPACE_NAMESPACE: &str =
    "symthaea-chemosensation-encoding-space-v1";
pub const CHEMICAL_ROOT_PROJECTION_POLICY_NAMESPACE: &str =
    "symthaea-chemosensation-root-projection-policy-v1";
pub const CHEMICAL_ROOT_BINARY_SPACE_NAMESPACE: &str =
    "symthaea-chemosensation-root-binary-space-v1";

const BLAKE3_256: &str = "blake3-256";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemicalContentAddressError {
    Projection(ChemicalRootProjectionError),
    Address(ContentAddressError),
}

impl fmt::Display for ChemicalContentAddressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Projection(error) => write!(f, "chemical projection identity failed: {error:?}"),
            Self::Address(error) => write!(f, "generic content-address construction failed: {error}"),
        }
    }
}

impl std::error::Error for ChemicalContentAddressError {}

impl From<ChemicalRootProjectionError> for ChemicalContentAddressError {
    fn from(value: ChemicalRootProjectionError) -> Self {
        Self::Projection(value)
    }
}

impl From<ContentAddressError> for ChemicalContentAddressError {
    fn from(value: ContentAddressError) -> Self {
        Self::Address(value)
    }
}

impl ChemicalObservationId {
    pub fn content_address(&self) -> Result<ContentAddress32, ContentAddressError> {
        address(CHEMICAL_OBSERVATION_NAMESPACE, *self.as_bytes())
    }
}

impl ChemicalEvidenceBundleId {
    pub fn content_address(&self) -> Result<ContentAddress32, ContentAddressError> {
        address(CHEMICAL_EVIDENCE_BUNDLE_NAMESPACE, *self.as_bytes())
    }
}

impl ChemicalEncodingSpaceId {
    pub fn content_address(&self) -> Result<ContentAddress32, ContentAddressError> {
        address(CHEMICAL_ENCODING_SPACE_NAMESPACE, *self.as_bytes())
    }
}

impl ChemicalRootProjectionPolicyId {
    pub fn content_address(&self) -> Result<ContentAddress32, ContentAddressError> {
        address(
            CHEMICAL_ROOT_PROJECTION_POLICY_NAMESPACE,
            *self.as_bytes(),
        )
    }
}

impl ChemicalRootBinarySpaceId {
    pub fn content_address(&self) -> Result<ContentAddress32, ContentAddressError> {
        address(CHEMICAL_ROOT_BINARY_SPACE_NAMESPACE, *self.as_bytes())
    }
}

/// Generic content identities required to explain a projected chemical modal
/// representation without importing chemosensation-specific ID types downstream.
///
/// This is lineage, not trust: it states what evidence and representation
/// contracts produced the projected bits, not whether those contracts should be
/// believed or how much confidence cognition should assign them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChemicalRootContentLineage {
    pub evidence_bundle: ContentAddress32,
    pub input_space: ContentAddress32,
    pub projection_policy: ContentAddress32,
    pub output_space: ContentAddress32,
}

impl ChemicalRootContentLineage {
    pub fn from_projection(
        projection: &ChemicalRootProjection,
    ) -> Result<Self, ChemicalContentAddressError> {
        let projection_policy_id = projection.projection_policy_id()?;
        let binary_space_id = projection.binary_space_id()?;

        Ok(Self {
            evidence_bundle: projection.evidence_bundle_id.content_address()?,
            input_space: projection.encoding_space_id.content_address()?,
            projection_policy: projection_policy_id.content_address()?,
            output_space: binary_space_id.content_address()?,
        })
    }
}

fn address(namespace: &str, digest: [u8; 32]) -> Result<ContentAddress32, ContentAddressError> {
    ContentAddress32::new(BLAKE3_256, namespace, digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChemicalBridgeTarget, ChemicalProjectionQuality, ChemicalRootProjection};
    use symthaea_core::hdc::binary_hv::BinaryHV;

    #[test]
    fn strong_ids_map_to_distinct_generic_namespaces_without_rehashing() {
        let bytes = [7; 32];
        let evidence = ChemicalEvidenceBundleId::from_bytes(bytes)
            .content_address()
            .unwrap();
        let space = ChemicalEncodingSpaceId::from_bytes(bytes)
            .content_address()
            .unwrap();

        assert_eq!(evidence.digest(), &bytes);
        assert_eq!(space.digest(), &bytes);
        assert_ne!(evidence, space);
        assert_eq!(evidence.algorithm(), BLAKE3_256);
        assert_eq!(evidence.namespace(), CHEMICAL_EVIDENCE_BUNDLE_NAMESPACE);
        assert_eq!(space.namespace(), CHEMICAL_ENCODING_SPACE_NAMESPACE);
        assert!(CHEMICAL_OBSERVATION_NAMESPACE.ends_with("-v2"));
        assert!(CHEMICAL_EVIDENCE_BUNDLE_NAMESPACE.ends_with("-v2"));
    }

    #[test]
    fn projected_lineage_preserves_all_four_identity_roles() {
        let projection = ChemicalRootProjection {
            target: ChemicalBridgeTarget::Olfactory,
            evidence_bundle_id: ChemicalEvidenceBundleId::from_bytes([1; 32]),
            encoding_space_id: ChemicalEncodingSpaceId::from_bytes([2; 32]),
            clock_domain: None,
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

        let lineage = ChemicalRootContentLineage::from_projection(&projection).unwrap();

        assert_eq!(lineage.evidence_bundle.digest(), &[1; 32]);
        assert_eq!(lineage.input_space.digest(), &[2; 32]);
        assert_eq!(
            lineage.projection_policy.namespace(),
            CHEMICAL_ROOT_PROJECTION_POLICY_NAMESPACE
        );
        assert_eq!(
            lineage.output_space.namespace(),
            CHEMICAL_ROOT_BINARY_SPACE_NAMESPACE
        );
        assert_ne!(lineage.input_space, lineage.output_space);
    }

    #[test]
    fn invalid_projection_identity_is_not_hidden_by_generic_adapter() {
        let projection = ChemicalRootProjection {
            target: ChemicalBridgeTarget::Olfactory,
            evidence_bundle_id: ChemicalEvidenceBundleId::from_bytes([1; 32]),
            encoding_space_id: ChemicalEncodingSpaceId::from_bytes([2; 32]),
            clock_domain: None,
            binary_vector: BinaryHV::zero(),
            confidence: 0.8,
            agreement: 0.9,
            earliest_timestamp_us: 10,
            latest_timestamp_us: 20,
            component_count: 2,
            quality: ChemicalProjectionQuality {
                threshold: f32::NAN,
                source_to_bipolar_similarity: 0.8,
                positive_fraction: 0.5,
            },
        };

        assert!(matches!(
            ChemicalRootContentLineage::from_projection(&projection),
            Err(ChemicalContentAddressError::Projection(
                ChemicalRootProjectionError::NonFiniteThreshold
            ))
        ));
    }

    #[test]
    fn namespaces_are_valid_content_address_contracts() {
        for namespace in [
            CHEMICAL_OBSERVATION_NAMESPACE,
            CHEMICAL_EVIDENCE_BUNDLE_NAMESPACE,
            CHEMICAL_ENCODING_SPACE_NAMESPACE,
            CHEMICAL_ROOT_PROJECTION_POLICY_NAMESPACE,
            CHEMICAL_ROOT_BINARY_SPACE_NAMESPACE,
        ] {
            assert!(ContentAddress32::new(BLAKE3_256, namespace, [0; 32]).is_ok());
        }
    }
}
