// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed entity-identity semantics for VIS-004A.
//!
//! The central invariant is that tracking continuity is not equivalent to object identity:
//!
//! ```text
//! observation-local hypothesis != track != entity hypothesis != external identity assertion
//! ```
//!
//! This module provides namespaced references and evidence-bearing associations without granting
//! any reference canonical world-identity authority.

use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

use crate::epistemic::{VisualEvidence, VisualObservationRef, VisualOrigin};

const MAX_EXTERNAL_AUTHORITY_LEN: usize = 128;
const MAX_EXTERNAL_ID_LEN: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct ObservationEntityRef {
    observation: VisualObservationRef,
    local_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct ObservationEntityRefWire {
    observation: VisualObservationRef,
    local_id: u32,
}

impl ObservationEntityRef {
    pub fn new(observation: VisualObservationRef, local_id: u32) -> Result<Self, EntityIdentityError> {
        if local_id == 0 {
            return Err(EntityIdentityError::MissingLocalEntityId);
        }
        Ok(Self {
            observation,
            local_id,
        })
    }

    pub const fn observation(self) -> VisualObservationRef {
        self.observation
    }

    pub const fn local_id(self) -> u32 {
        self.local_id
    }
}

impl<'de> Deserialize<'de> for ObservationEntityRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ObservationEntityRefWire::deserialize(deserializer)?;
        Self::new(wire.observation, wire.local_id).map_err(serde::de::Error::custom)
    }
}

/// One tracker-local identity handle.
///
/// `track_id` is deliberately scoped by `tracker_namespace`; numeric track IDs alone are not
/// globally meaningful and may collide across processes, devices, restored experiments, or
/// independently-created `VisionManifold` instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct VisualTrackRef {
    tracker_namespace: u64,
    track_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct VisualTrackRefWire {
    tracker_namespace: u64,
    track_id: u64,
}

impl VisualTrackRef {
    pub fn new(tracker_namespace: u64, track_id: u64) -> Result<Self, EntityIdentityError> {
        if tracker_namespace == 0 {
            return Err(EntityIdentityError::MissingTrackerNamespace);
        }
        Ok(Self {
            tracker_namespace,
            track_id,
        })
    }

    pub const fn tracker_namespace(self) -> u64 {
        self.tracker_namespace
    }

    pub const fn track_id(self) -> u64 {
        self.track_id
    }
}

impl<'de> Deserialize<'de> for VisualTrackRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = VisualTrackRefWire::deserialize(deserializer)?;
        Self::new(wire.tracker_namespace, wire.track_id).map_err(serde::de::Error::custom)
    }
}

/// A belief-layer entity handle that may outlive any single tracker handle.
///
/// This is still a hypothesis namespace, not a canonical physical-world identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct VisualEntityHypothesisRef {
    belief_namespace: u64,
    entity_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct VisualEntityHypothesisRefWire {
    belief_namespace: u64,
    entity_id: u64,
}

impl VisualEntityHypothesisRef {
    pub fn new(belief_namespace: u64, entity_id: u64) -> Result<Self, EntityIdentityError> {
        if belief_namespace == 0 {
            return Err(EntityIdentityError::MissingBeliefNamespace);
        }
        if entity_id == 0 {
            return Err(EntityIdentityError::MissingEntityHypothesisId);
        }
        Ok(Self {
            belief_namespace,
            entity_id,
        })
    }

    pub const fn belief_namespace(self) -> u64 {
        self.belief_namespace
    }

    pub const fn entity_id(self) -> u64 {
        self.entity_id
    }
}

impl<'de> Deserialize<'de> for VisualEntityHypothesisRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = VisualEntityHypothesisRefWire::deserialize(deserializer)?;
        Self::new(wire.belief_namespace, wire.entity_id).map_err(serde::de::Error::custom)
    }
}

/// Identifier asserted by an external authority/source.
///
/// Its existence means only that a source supplied this identifier. It does not establish that
/// the identifier is correct, globally unique, authenticated, or equivalent to Symthaea's entity
/// hypothesis.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct ExternalEntityAssertionRef {
    authority: String,
    external_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ExternalEntityAssertionRefWire {
    authority: String,
    external_id: String,
}

impl ExternalEntityAssertionRef {
    pub fn new(
        authority: impl Into<String>,
        external_id: impl Into<String>,
    ) -> Result<Self, EntityIdentityError> {
        let authority = authority.into();
        let external_id = external_id.into();
        validate_identifier_text(&authority, MAX_EXTERNAL_AUTHORITY_LEN, true)?;
        validate_identifier_text(&external_id, MAX_EXTERNAL_ID_LEN, false)?;
        Ok(Self {
            authority,
            external_id,
        })
    }

    pub fn authority(&self) -> &str {
        &self.authority
    }

    pub fn external_id(&self) -> &str {
        &self.external_id
    }
}

impl<'de> Deserialize<'de> for ExternalEntityAssertionRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ExternalEntityAssertionRefWire::deserialize(deserializer)?;
        Self::new(wire.authority, wire.external_id).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityIdentityLevel {
    ObservationLocal,
    Track,
    EntityHypothesis,
    ExternalAssertion,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum EntityReference {
    ObservationLocal(ObservationEntityRef),
    Track(VisualTrackRef),
    EntityHypothesis(VisualEntityHypothesisRef),
    ExternalAssertion(ExternalEntityAssertionRef),
}

impl EntityReference {
    pub const fn level(&self) -> EntityIdentityLevel {
        match self {
            Self::ObservationLocal(_) => EntityIdentityLevel::ObservationLocal,
            Self::Track(_) => EntityIdentityLevel::Track,
            Self::EntityHypothesis(_) => EntityIdentityLevel::EntityHypothesis,
            Self::ExternalAssertion(_) => EntityIdentityLevel::ExternalAssertion,
        }
    }

    /// No reference variant is a canonical world identity in VIS-004A.
    pub const fn is_canonical_world_identity(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityAssociationKind {
    SupportsSameEntity,
    SupportsDistinctEntity,
}

/// Evidence-bearing relation between two identity references.
///
/// Identity relations are interpretations and therefore cannot be direct `Observed` evidence.
/// VIS-004A accepts only `Inferred` or `Remembered` evidence. Predictions/simulations may propose
/// future identity relations elsewhere, but they cannot enter this historical association record.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EntityAssociation {
    left: EntityReference,
    right: EntityReference,
    kind: EntityAssociationKind,
    evidence: VisualEvidence,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct EntityAssociationWire {
    left: EntityReference,
    right: EntityReference,
    kind: EntityAssociationKind,
    evidence: VisualEvidence,
}

impl EntityAssociation {
    pub fn new(
        left: EntityReference,
        right: EntityReference,
        kind: EntityAssociationKind,
        evidence: VisualEvidence,
    ) -> Result<Self, EntityIdentityError> {
        if left == right {
            return Err(EntityIdentityError::SelfAssociation);
        }
        match evidence.origin() {
            VisualOrigin::Inferred | VisualOrigin::Remembered => {}
            VisualOrigin::Observed => return Err(EntityIdentityError::ObservedIdentityAssociation),
            VisualOrigin::Predicted | VisualOrigin::Simulated | VisualOrigin::Counterfactual => {
                return Err(EntityIdentityError::GenerativeIdentityAssociation)
            }
        }
        Ok(Self {
            left,
            right,
            kind,
            evidence,
        })
    }

    pub fn left(&self) -> &EntityReference {
        &self.left
    }

    pub fn right(&self) -> &EntityReference {
        &self.right
    }

    pub const fn kind(&self) -> EntityAssociationKind {
        self.kind
    }

    pub fn evidence(&self) -> &VisualEvidence {
        &self.evidence
    }
}

impl<'de> Deserialize<'de> for EntityAssociation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = EntityAssociationWire::deserialize(deserializer)?;
        Self::new(wire.left, wire.right, wire.kind, wire.evidence)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityIdentityError {
    MissingLocalEntityId,
    MissingTrackerNamespace,
    MissingBeliefNamespace,
    MissingEntityHypothesisId,
    EmptyExternalAuthority,
    EmptyExternalId,
    ExternalAuthorityTooLong,
    ExternalIdTooLong,
    IdentifierContainsControlCharacter,
    SelfAssociation,
    ObservedIdentityAssociation,
    GenerativeIdentityAssociation,
}

impl fmt::Display for EntityIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MissingLocalEntityId => "observation-local entity ID must be non-zero",
            Self::MissingTrackerNamespace => "tracker namespace must be non-zero",
            Self::MissingBeliefNamespace => "entity-belief namespace must be non-zero",
            Self::MissingEntityHypothesisId => "entity hypothesis ID must be non-zero",
            Self::EmptyExternalAuthority => "external identity authority must be non-empty",
            Self::EmptyExternalId => "external identity value must be non-empty",
            Self::ExternalAuthorityTooLong => "external identity authority exceeds length limit",
            Self::ExternalIdTooLong => "external identity value exceeds length limit",
            Self::IdentifierContainsControlCharacter => {
                "external identity fields cannot contain control characters"
            }
            Self::SelfAssociation => "entity association requires two distinct references",
            Self::ObservedIdentityAssociation => {
                "entity identity association is inferred, not direct observation"
            }
            Self::GenerativeIdentityAssociation => {
                "predicted/simulated/counterfactual identity cannot enter historical association state"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for EntityIdentityError {}

fn validate_identifier_text(
    value: &str,
    max_len: usize,
    authority: bool,
) -> Result<(), EntityIdentityError> {
    if value.trim().is_empty() {
        return Err(if authority {
            EntityIdentityError::EmptyExternalAuthority
        } else {
            EntityIdentityError::EmptyExternalId
        });
    }
    if value.len() > max_len {
        return Err(if authority {
            EntityIdentityError::ExternalAuthorityTooLong
        } else {
            EntityIdentityError::ExternalIdTooLong
        });
    }
    if value.chars().any(char::is_control) {
        return Err(EntityIdentityError::IdentifierContainsControlCharacter);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{VisualCaptureClock, VisualStreamRef};

    fn observation(frame_id: u64) -> VisualObservationRef {
        VisualObservationRef::new(
            VisualStreamRef::new(7, 11).unwrap(),
            frame_id,
            1_000 + frame_id,
            VisualCaptureClock::StreamMonotonic,
        )
    }

    fn inferred(frame_id: u64, confidence: f32) -> VisualEvidence {
        VisualEvidence::inferred(vec![observation(frame_id)], confidence).unwrap()
    }

    #[test]
    fn track_ids_are_namespaced() {
        let a = VisualTrackRef::new(10, 0).unwrap();
        let b = VisualTrackRef::new(11, 0).unwrap();
        assert_ne!(a, b);
        assert!(VisualTrackRef::new(0, 0).is_err());
    }

    #[test]
    fn observation_local_reference_is_bound_to_exact_observation() {
        let a = ObservationEntityRef::new(observation(1), 1).unwrap();
        let b = ObservationEntityRef::new(observation(2), 1).unwrap();
        assert_ne!(a, b);
        assert!(ObservationEntityRef::new(observation(1), 0).is_err());
    }

    #[test]
    fn entity_hypothesis_is_not_canonical_world_identity() {
        let entity = EntityReference::EntityHypothesis(
            VisualEntityHypothesisRef::new(99, 1).unwrap(),
        );
        assert_eq!(entity.level(), EntityIdentityLevel::EntityHypothesis);
        assert!(!entity.is_canonical_world_identity());
    }

    #[test]
    fn external_assertion_is_validated_but_not_promoted() {
        let external = EntityReference::ExternalAssertion(
            ExternalEntityAssertionRef::new("inventory-system", "sku:abc-123").unwrap(),
        );
        assert_eq!(external.level(), EntityIdentityLevel::ExternalAssertion);
        assert!(!external.is_canonical_world_identity());
        assert!(ExternalEntityAssertionRef::new("", "abc").is_err());
        assert!(ExternalEntityAssertionRef::new("source", "\n").is_err());
    }

    #[test]
    fn identity_association_requires_inference_or_memory() {
        let left = EntityReference::Track(VisualTrackRef::new(5, 1).unwrap());
        let right = EntityReference::EntityHypothesis(
            VisualEntityHypothesisRef::new(8, 1).unwrap(),
        );

        let assoc = EntityAssociation::new(
            left.clone(),
            right.clone(),
            EntityAssociationKind::SupportsSameEntity,
            inferred(2, 0.7),
        )
        .unwrap();
        assert_eq!(assoc.kind(), EntityAssociationKind::SupportsSameEntity);

        let observed = VisualEvidence::observed(observation(2), 1.0).unwrap();
        assert_eq!(
            EntityAssociation::new(
                left.clone(),
                right.clone(),
                EntityAssociationKind::SupportsSameEntity,
                observed,
            )
            .unwrap_err(),
            EntityIdentityError::ObservedIdentityAssociation
        );

        let predicted = VisualEvidence::predicted(vec![observation(2)], 0.8).unwrap();
        assert_eq!(
            EntityAssociation::new(
                left,
                right,
                EntityAssociationKind::SupportsSameEntity,
                predicted,
            )
            .unwrap_err(),
            EntityIdentityError::GenerativeIdentityAssociation
        );
    }

    #[test]
    fn self_association_is_rejected() {
        let reference = EntityReference::Track(VisualTrackRef::new(4, 9).unwrap());
        assert_eq!(
            EntityAssociation::new(
                reference.clone(),
                reference,
                EntityAssociationKind::SupportsSameEntity,
                inferred(3, 0.6),
            )
            .unwrap_err(),
            EntityIdentityError::SelfAssociation
        );
    }

    #[test]
    fn deserialization_revalidates_track_namespace() {
        let malformed = r#"{"tracker_namespace":0,"track_id":7}"#;
        assert!(serde_json::from_str::<VisualTrackRef>(malformed).is_err());
    }

    #[test]
    fn deserialization_revalidates_identity_association_origin() {
        let left = EntityReference::Track(VisualTrackRef::new(4, 1).unwrap());
        let right = EntityReference::Track(VisualTrackRef::new(4, 2).unwrap());
        let wire = serde_json::json!({
            "left": left,
            "right": right,
            "kind": "supports_same_entity",
            "evidence": VisualEvidence::observed(observation(5), 1.0).unwrap(),
        });
        assert!(serde_json::from_value::<EntityAssociation>(wire).is_err());
    }
}
