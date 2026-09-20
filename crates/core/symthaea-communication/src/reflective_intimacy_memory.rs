// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Perspective-aware reflective memory metadata for intimate dialogue.
//!
//! This module deliberately stores no raw intimate payload. A memory item binds
//! an opaque content reference/commitment to perspective, reality namespace,
//! provenance, retention, and an exact privacy-graph artifact. Retrieval remains
//! conditional on that privacy artifact still being active and unchanged.

use crate::intimate_memory_privacy::{
    IntimateArtifactKindV1, IntimateDependencyStateV1, IntimateMemoryPrivacyGraphV1,
    IntimateRetentionClassV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REFLECTIVE_INTIMACY_MEMORY_SCHEMA_V1: &str =
    "symthaea.communication.reflective-intimacy-memory.v1";
const CONTENT_DOMAIN_V1: &[u8] = b"symthaea:reflective-intimacy-content:v1\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReflectiveMemoryPerspectiveV1 {
    Participant,
    Symthaea,
    Shared,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReflectiveRealityNamespaceV1 {
    RealWorld,
    Fantasy { world_id: String },
}

impl ReflectiveRealityNamespaceV1 {
    fn validate(&self) -> Result<(), ReflectiveMemoryErrorV1> {
        if let Self::Fantasy { world_id } = self {
            validate_id(world_id, ReflectiveMemoryErrorV1::InvalidFantasyWorldId)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReflectiveMemoryNamespaceV1 {
    ParticipantFact,
    SymthaeaPersona,
    SharedRelationship,
    FantasyWorld,
    ExplicitPreference,
    InferredPreference,
    Boundary,
    Reflection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReflectiveMemorySensitivityV1 {
    Ordinary,
    Personal,
    Intimate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReflectiveMemoryProvenanceV1 {
    ExplicitParticipantStatement,
    ValidatedSelfReport,
    SymthaeaAuthored,
    BehavioralInference,
    SharedDerived,
    SystemReflection,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReflectiveIntimacyMemoryItemV1 {
    pub memory_id: String,
    /// Exact artifact in KAMA-PRIV-001B governing whether this memory remains
    /// retrievable. The payload itself is not stored here.
    pub privacy_artifact_id: String,
    pub content_ref: String,
    pub content_commitment: String,
    pub namespace: ReflectiveMemoryNamespaceV1,
    pub reality: ReflectiveRealityNamespaceV1,
    pub perspective: ReflectiveMemoryPerspectiveV1,
    pub provenance: ReflectiveMemoryProvenanceV1,
    pub sensitivity: ReflectiveMemorySensitivityV1,
    pub confidence: f32,
    pub retention: IntimateRetentionClassV1,
    /// Exact source lineage expected on the governing privacy artifact.
    pub source_artifact_ids: BTreeSet<String>,
    pub created_at_ns: u64,
    pub updated_at_ns: u64,
    /// Optional prior reflective-memory identity superseded by this item.
    pub supersedes: Option<String>,
}

impl ReflectiveIntimacyMemoryItemV1 {
    fn validate_shape(&self) -> Result<(), ReflectiveMemoryErrorV1> {
        validate_id(&self.memory_id, ReflectiveMemoryErrorV1::InvalidMemoryId)?;
        validate_id(
            &self.privacy_artifact_id,
            ReflectiveMemoryErrorV1::InvalidPrivacyArtifactId,
        )?;
        validate_id(&self.content_ref, ReflectiveMemoryErrorV1::InvalidContentRef)?;
        validate_content_commitment(&self.content_commitment)?;
        self.reality.validate()?;
        if let Some(prior) = &self.supersedes {
            validate_id(prior, ReflectiveMemoryErrorV1::InvalidSupersedesId)?;
            if prior == &self.memory_id {
                return Err(ReflectiveMemoryErrorV1::SelfSupersession);
            }
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(ReflectiveMemoryErrorV1::InvalidConfidence);
        }
        if self.updated_at_ns < self.created_at_ns {
            return Err(ReflectiveMemoryErrorV1::InvalidTimestamps);
        }
        if self.source_artifact_ids.is_empty() {
            return Err(ReflectiveMemoryErrorV1::MissingSourceLineage);
        }
        if self
            .source_artifact_ids
            .iter()
            .any(|source| canonical_id(source).is_none())
        {
            return Err(ReflectiveMemoryErrorV1::InvalidSourceArtifactId);
        }
        if self.retention == IntimateRetentionClassV1::ExternalExport {
            return Err(ReflectiveMemoryErrorV1::ExternalExportIsNotMemoryRetention);
        }
        validate_semantics(self)?;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct ReflectiveIntimacyMemoryIndexV1 {
    items: BTreeMap<String, ReflectiveIntimacyMemoryItemV1>,
    known_ids: BTreeSet<String>,
    superseded_by: BTreeMap<String, String>,
}

impl ReflectiveIntimacyMemoryIndexV1 {
    pub fn admit(
        &mut self,
        item: ReflectiveIntimacyMemoryItemV1,
        privacy: &IntimateMemoryPrivacyGraphV1,
    ) -> Result<(), ReflectiveMemoryErrorV1> {
        item.validate_shape()?;
        if self.known_ids.contains(&item.memory_id) {
            return Err(ReflectiveMemoryErrorV1::DuplicateMemoryId);
        }

        let privacy_artifact = privacy
            .active_artifact(&item.privacy_artifact_id)
            .ok_or(ReflectiveMemoryErrorV1::PrivacyArtifactUnavailable)?;
        if privacy_artifact.dependency_state != IntimateDependencyStateV1::Complete
            || !privacy.is_retrievable(&item.privacy_artifact_id)
        {
            return Err(ReflectiveMemoryErrorV1::PrivacyArtifactNotRetrievable);
        }
        if privacy_artifact.retention != item.retention {
            return Err(ReflectiveMemoryErrorV1::RetentionMismatch);
        }
        if privacy_artifact.source_ids != item.source_artifact_ids {
            return Err(ReflectiveMemoryErrorV1::SourceLineageMismatch);
        }
        if privacy_artifact.kind != expected_privacy_kind(item.namespace) {
            return Err(ReflectiveMemoryErrorV1::PrivacyArtifactKindMismatch);
        }

        if let Some(prior_id) = &item.supersedes {
            let prior = self
                .items
                .get(prior_id)
                .ok_or(ReflectiveMemoryErrorV1::UnknownSupersededMemory)?;
            if self.superseded_by.contains_key(prior_id) {
                return Err(ReflectiveMemoryErrorV1::PriorMemoryAlreadySuperseded);
            }
            if prior.namespace != item.namespace || prior.perspective != item.perspective {
                return Err(ReflectiveMemoryErrorV1::SemanticLaneSupersessionMismatch);
            }
            if prior.reality != item.reality {
                return Err(ReflectiveMemoryErrorV1::CrossRealitySupersession);
            }
            if item.updated_at_ns < prior.updated_at_ns {
                return Err(ReflectiveMemoryErrorV1::SupersessionMovesBackwardInTime);
            }
        }

        let memory_id = item.memory_id.clone();
        if let Some(prior_id) = item.supersedes.clone() {
            self.superseded_by.insert(prior_id, memory_id.clone());
        }
        self.known_ids.insert(memory_id.clone());
        self.items.insert(memory_id, item);
        Ok(())
    }

    /// Returns only a current item whose exact privacy lineage remains active.
    /// If KAMA-PRIV removes or mutates the governing artifact lineage, the item
    /// fails closed until a fresh reflective-memory item is admitted.
    pub fn current<'a>(
        &'a self,
        memory_id: &str,
        privacy: &IntimateMemoryPrivacyGraphV1,
    ) -> Option<&'a ReflectiveIntimacyMemoryItemV1> {
        let item = self.items.get(memory_id)?;
        if self.superseded_by.contains_key(memory_id) {
            return None;
        }
        let privacy_artifact = privacy.active_artifact(&item.privacy_artifact_id)?;
        if !privacy.is_retrievable(&item.privacy_artifact_id)
            || privacy_artifact.dependency_state != IntimateDependencyStateV1::Complete
            || privacy_artifact.retention != item.retention
            || privacy_artifact.source_ids != item.source_artifact_ids
            || privacy_artifact.kind != expected_privacy_kind(item.namespace)
        {
            return None;
        }
        Some(item)
    }

    pub fn superseded_by(&self, memory_id: &str) -> Option<&str> {
        self.superseded_by.get(memory_id).map(String::as_str)
    }
}

pub fn reflective_content_commitment_v1(bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CONTENT_DOMAIN_V1);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn expected_privacy_kind(namespace: ReflectiveMemoryNamespaceV1) -> IntimateArtifactKindV1 {
    match namespace {
        ReflectiveMemoryNamespaceV1::FantasyWorld => IntimateArtifactKindV1::FantasyHistory,
        ReflectiveMemoryNamespaceV1::ExplicitPreference
        | ReflectiveMemoryNamespaceV1::Boundary => IntimateArtifactKindV1::ExplicitPreference,
        ReflectiveMemoryNamespaceV1::InferredPreference => IntimateArtifactKindV1::InferredPreference,
        ReflectiveMemoryNamespaceV1::ParticipantFact
        | ReflectiveMemoryNamespaceV1::SymthaeaPersona
        | ReflectiveMemoryNamespaceV1::SharedRelationship
        | ReflectiveMemoryNamespaceV1::Reflection => IntimateArtifactKindV1::ReflectiveMemory,
    }
}

fn validate_semantics(item: &ReflectiveIntimacyMemoryItemV1) -> Result<(), ReflectiveMemoryErrorV1> {
    match item.namespace {
        ReflectiveMemoryNamespaceV1::ParticipantFact => {
            if item.perspective != ReflectiveMemoryPerspectiveV1::Participant
                || !matches!(
                    item.provenance,
                    ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement
                        | ReflectiveMemoryProvenanceV1::ValidatedSelfReport
                )
            {
                return Err(ReflectiveMemoryErrorV1::PerspectiveProvenanceMismatch);
            }
        }
        ReflectiveMemoryNamespaceV1::SymthaeaPersona => {
            if item.perspective != ReflectiveMemoryPerspectiveV1::Symthaea
                || item.provenance != ReflectiveMemoryProvenanceV1::SymthaeaAuthored
            {
                return Err(ReflectiveMemoryErrorV1::PerspectiveProvenanceMismatch);
            }
        }
        ReflectiveMemoryNamespaceV1::SharedRelationship => {
            if item.perspective != ReflectiveMemoryPerspectiveV1::Shared
                || !matches!(
                    item.provenance,
                    ReflectiveMemoryProvenanceV1::SharedDerived
                        | ReflectiveMemoryProvenanceV1::SystemReflection
                )
            {
                return Err(ReflectiveMemoryErrorV1::PerspectiveProvenanceMismatch);
            }
        }
        ReflectiveMemoryNamespaceV1::FantasyWorld => {
            if !matches!(item.reality, ReflectiveRealityNamespaceV1::Fantasy { .. }) {
                return Err(ReflectiveMemoryErrorV1::FantasyMemoryRequiresFantasyReality);
            }
        }
        ReflectiveMemoryNamespaceV1::ExplicitPreference => {
            if item.perspective != ReflectiveMemoryPerspectiveV1::Participant
                || !matches!(
                    item.provenance,
                    ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement
                        | ReflectiveMemoryProvenanceV1::ValidatedSelfReport
                )
            {
                return Err(ReflectiveMemoryErrorV1::PerspectiveProvenanceMismatch);
            }
        }
        ReflectiveMemoryNamespaceV1::InferredPreference => {
            if item.perspective != ReflectiveMemoryPerspectiveV1::Participant
                || item.provenance != ReflectiveMemoryProvenanceV1::BehavioralInference
            {
                return Err(ReflectiveMemoryErrorV1::PerspectiveProvenanceMismatch);
            }
        }
        ReflectiveMemoryNamespaceV1::Boundary => {
            if item.perspective != ReflectiveMemoryPerspectiveV1::Participant
                || item.provenance
                    != ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement
            {
                return Err(ReflectiveMemoryErrorV1::BoundaryMustBeExplicitParticipantStatement);
            }
        }
        ReflectiveMemoryNamespaceV1::Reflection => {
            if item.perspective != ReflectiveMemoryPerspectiveV1::Shared
                || item.provenance != ReflectiveMemoryProvenanceV1::SystemReflection
            {
                return Err(ReflectiveMemoryErrorV1::PerspectiveProvenanceMismatch);
            }
        }
    }
    Ok(())
}

fn validate_content_commitment(value: &str) -> Result<(), ReflectiveMemoryErrorV1> {
    let Some(hex) = value.strip_prefix("blake3:") else {
        return Err(ReflectiveMemoryErrorV1::InvalidContentCommitment);
    };
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ReflectiveMemoryErrorV1::InvalidContentCommitment);
    }
    Ok(())
}

fn validate_id(value: &str, error: ReflectiveMemoryErrorV1) -> Result<(), ReflectiveMemoryErrorV1> {
    canonical_id(value).map(|_| ()).ok_or(error)
}

fn canonical_id(value: &str) -> Option<String> {
    let value = value.trim().to_owned();
    (!value.is_empty() && value.len() <= 256).then_some(value)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectiveMemoryErrorV1 {
    InvalidMemoryId,
    InvalidPrivacyArtifactId,
    InvalidContentRef,
    InvalidContentCommitment,
    InvalidFantasyWorldId,
    InvalidSupersedesId,
    InvalidSourceArtifactId,
    InvalidConfidence,
    InvalidTimestamps,
    MissingSourceLineage,
    SelfSupersession,
    ExternalExportIsNotMemoryRetention,
    PerspectiveProvenanceMismatch,
    BoundaryMustBeExplicitParticipantStatement,
    FantasyMemoryRequiresFantasyReality,
    DuplicateMemoryId,
    PrivacyArtifactUnavailable,
    PrivacyArtifactNotRetrievable,
    RetentionMismatch,
    SourceLineageMismatch,
    PrivacyArtifactKindMismatch,
    UnknownSupersededMemory,
    PriorMemoryAlreadySuperseded,
    SemanticLaneSupersessionMismatch,
    CrossRealitySupersession,
    SupersessionMovesBackwardInTime,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intimate_memory_privacy::{
        IntimateDerivationPolicyV1, IntimateMemoryArtifactV1,
    };

    fn graph_with_memory_artifact(
        memory_artifact_id: &str,
        kind: IntimateArtifactKindV1,
        retention: IntimateRetentionClassV1,
    ) -> IntimateMemoryPrivacyGraphV1 {
        let mut graph = IntimateMemoryPrivacyGraphV1::default();
        graph
            .insert(
                IntimateMemoryArtifactV1::root(
                    "source-1",
                    IntimateArtifactKindV1::PsychologyEvidence,
                    retention,
                )
                .unwrap(),
            )
            .unwrap();
        graph
            .insert(IntimateMemoryArtifactV1 {
                artifact_id: memory_artifact_id.into(),
                kind,
                retention,
                dependency_state: IntimateDependencyStateV1::Complete,
                derivation_policy: IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
                reconstructive: false,
                source_ids: BTreeSet::from(["source-1".into()]),
                durability_authorization_ref: None,
                export_receipt_ref: None,
            })
            .unwrap();
        graph
    }

    fn explicit_preference(id: &str, privacy_artifact_id: &str) -> ReflectiveIntimacyMemoryItemV1 {
        ReflectiveIntimacyMemoryItemV1 {
            memory_id: id.into(),
            privacy_artifact_id: privacy_artifact_id.into(),
            content_ref: format!("vault:{id}"),
            content_commitment: reflective_content_commitment_v1(id.as_bytes()),
            namespace: ReflectiveMemoryNamespaceV1::ExplicitPreference,
            reality: ReflectiveRealityNamespaceV1::RealWorld,
            perspective: ReflectiveMemoryPerspectiveV1::Participant,
            provenance: ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement,
            sensitivity: ReflectiveMemorySensitivityV1::Intimate,
            confidence: 1.0,
            retention: IntimateRetentionClassV1::DurableOptIn,
            source_artifact_ids: BTreeSet::from(["source-1".into()]),
            created_at_ns: 10,
            updated_at_ns: 10,
            supersedes: None,
        }
    }

    #[test]
    fn privacy_retraction_immediately_removes_memory_from_retrieval() {
        let mut graph = graph_with_memory_artifact(
            "privacy-memory-1",
            IntimateArtifactKindV1::ExplicitPreference,
            IntimateRetentionClassV1::DurableOptIn,
        );
        let mut index = ReflectiveIntimacyMemoryIndexV1::default();
        index
            .admit(explicit_preference("memory-1", "privacy-memory-1"), &graph)
            .unwrap();
        assert!(index.current("memory-1", &graph).is_some());
        graph.retract_source("source-1").unwrap();
        assert!(index.current("memory-1", &graph).is_none());
    }

    #[test]
    fn fantasy_cannot_supersede_real_world_memory() {
        let mut graph = graph_with_memory_artifact(
            "privacy-memory-1",
            IntimateArtifactKindV1::ExplicitPreference,
            IntimateRetentionClassV1::DurableOptIn,
        );
        graph
            .insert(IntimateMemoryArtifactV1 {
                artifact_id: "privacy-memory-2".into(),
                kind: IntimateArtifactKindV1::ExplicitPreference,
                retention: IntimateRetentionClassV1::DurableOptIn,
                dependency_state: IntimateDependencyStateV1::Complete,
                derivation_policy: IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
                reconstructive: false,
                source_ids: BTreeSet::from(["source-1".into()]),
                durability_authorization_ref: None,
                export_receipt_ref: None,
            })
            .unwrap();
        let mut index = ReflectiveIntimacyMemoryIndexV1::default();
        index
            .admit(explicit_preference("memory-1", "privacy-memory-1"), &graph)
            .unwrap();
        let mut replacement = explicit_preference("memory-2", "privacy-memory-2");
        replacement.reality = ReflectiveRealityNamespaceV1::Fantasy {
            world_id: "world-1".into(),
        };
        replacement.updated_at_ns = 20;
        replacement.supersedes = Some("memory-1".into());
        assert_eq!(
            index.admit(replacement, &graph),
            Err(ReflectiveMemoryErrorV1::CrossRealitySupersession)
        );
    }

    #[test]
    fn boundary_cannot_be_inferred() {
        let mut item = explicit_preference("boundary-1", "privacy-boundary-1");
        item.namespace = ReflectiveMemoryNamespaceV1::Boundary;
        item.provenance = ReflectiveMemoryProvenanceV1::BehavioralInference;
        assert_eq!(
            item.validate_shape(),
            Err(ReflectiveMemoryErrorV1::BoundaryMustBeExplicitParticipantStatement)
        );
    }

    #[test]
    fn supersession_preserves_old_identity_but_only_new_item_is_current() {
        let mut graph = graph_with_memory_artifact(
            "privacy-memory-1",
            IntimateArtifactKindV1::ExplicitPreference,
            IntimateRetentionClassV1::DurableOptIn,
        );
        graph
            .insert(IntimateMemoryArtifactV1 {
                artifact_id: "privacy-memory-2".into(),
                kind: IntimateArtifactKindV1::ExplicitPreference,
                retention: IntimateRetentionClassV1::DurableOptIn,
                dependency_state: IntimateDependencyStateV1::Complete,
                derivation_policy: IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
                reconstructive: false,
                source_ids: BTreeSet::from(["source-1".into()]),
                durability_authorization_ref: None,
                export_receipt_ref: None,
            })
            .unwrap();
        let mut index = ReflectiveIntimacyMemoryIndexV1::default();
        index
            .admit(explicit_preference("memory-1", "privacy-memory-1"), &graph)
            .unwrap();
        let mut replacement = explicit_preference("memory-2", "privacy-memory-2");
        replacement.updated_at_ns = 20;
        replacement.supersedes = Some("memory-1".into());
        index.admit(replacement, &graph).unwrap();
        assert!(index.current("memory-1", &graph).is_none());
        assert!(index.current("memory-2", &graph).is_some());
        assert_eq!(index.superseded_by("memory-1"), Some("memory-2"));
    }

    #[test]
    fn stale_lineage_after_privacy_mutation_fails_closed() {
        let mut graph = IntimateMemoryPrivacyGraphV1::default();
        for id in ["source-1", "source-2"] {
            graph
                .insert(
                    IntimateMemoryArtifactV1::root(
                        id,
                        IntimateArtifactKindV1::PsychologyEvidence,
                        IntimateRetentionClassV1::DurableOptIn,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        graph
            .insert(IntimateMemoryArtifactV1 {
                artifact_id: "privacy-memory".into(),
                kind: IntimateArtifactKindV1::ExplicitPreference,
                retention: IntimateRetentionClassV1::DurableOptIn,
                dependency_state: IntimateDependencyStateV1::Complete,
                derivation_policy: IntimateDerivationPolicyV1::MayRetainWithIndependentBasis,
                reconstructive: false,
                source_ids: BTreeSet::from(["source-1".into(), "source-2".into()]),
                durability_authorization_ref: None,
                export_receipt_ref: None,
            })
            .unwrap();
        let mut item = explicit_preference("memory", "privacy-memory");
        item.source_artifact_ids = BTreeSet::from(["source-1".into(), "source-2".into()]);
        let mut index = ReflectiveIntimacyMemoryIndexV1::default();
        index.admit(item, &graph).unwrap();
        assert!(index.current("memory", &graph).is_some());
        graph.retract_source("source-1").unwrap();
        assert!(graph.is_retrievable("privacy-memory"));
        assert!(index.current("memory", &graph).is_none());
    }
}
