// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dependency-aware privacy semantics for intimate derived memory.
//!
//! This graph stores metadata only. It does not contain raw intimate content.
//! Retraction is local to the governed graph and does not claim erasure of
//! external exports, user copies, or systems outside the governed domain.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const INTIMATE_MEMORY_PRIVACY_SCHEMA_V1: &str =
    "symthaea.communication.intimate-memory-privacy.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntimateArtifactKindV1 {
    RawTranscript,
    SessionSummary,
    ExplicitPreference,
    InferredPreference,
    PsychologyEvidence,
    ReflectiveMemory,
    FantasyHistory,
    EmbeddingOrIndex,
    ExternalExport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntimateRetentionClassV1 {
    EphemeralSession,
    DurableOptIn,
    ExternalExport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntimateDependencyStateV1 {
    Complete,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntimateDerivationPolicyV1 {
    RootSource,
    MustDeleteOnSourceRetraction,
    MustRecomputeWithoutRetractedSource,
    MayRetainWithIndependentBasis,
    ExternalExportRequiresNotice,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntimateMemoryArtifactV1 {
    pub artifact_id: String,
    pub kind: IntimateArtifactKindV1,
    pub retention: IntimateRetentionClassV1,
    pub dependency_state: IntimateDependencyStateV1,
    pub derivation_policy: IntimateDerivationPolicyV1,
    pub reconstructive: bool,
    pub source_ids: BTreeSet<String>,
    /// Required when an ephemeral source is intentionally promoted into a
    /// durable derived artifact. The graph stores only the opaque evidence ref.
    pub durability_authorization_ref: Option<String>,
    /// Required for external exports so retraction receipts can disclose that
    /// deletion beyond the governed domain cannot be proven here.
    pub export_receipt_ref: Option<String>,
}

impl IntimateMemoryArtifactV1 {
    pub fn root(
        artifact_id: impl Into<String>,
        kind: IntimateArtifactKindV1,
        retention: IntimateRetentionClassV1,
    ) -> Result<Self, IntimateMemoryPrivacyErrorV1> {
        let artifact = Self {
            artifact_id: artifact_id.into(),
            kind,
            retention,
            dependency_state: IntimateDependencyStateV1::Complete,
            derivation_policy: IntimateDerivationPolicyV1::RootSource,
            reconstructive: false,
            source_ids: BTreeSet::new(),
            durability_authorization_ref: None,
            export_receipt_ref: None,
        };
        artifact.validate_shape()?;
        Ok(artifact)
    }

    fn validate_shape(&self) -> Result<(), IntimateMemoryPrivacyErrorV1> {
        validate_id(
            &self.artifact_id,
            IntimateMemoryPrivacyErrorV1::InvalidArtifactId,
        )?;
        if let Some(reference) = &self.durability_authorization_ref {
            validate_id(
                reference,
                IntimateMemoryPrivacyErrorV1::InvalidAuthorizationRef,
            )?;
        }
        if let Some(reference) = &self.export_receipt_ref {
            validate_id(reference, IntimateMemoryPrivacyErrorV1::InvalidExportRef)?;
        }
        if self.source_ids.contains(&self.artifact_id) {
            return Err(IntimateMemoryPrivacyErrorV1::SelfDependency);
        }

        match self.derivation_policy {
            IntimateDerivationPolicyV1::RootSource => {
                if !self.source_ids.is_empty() {
                    return Err(IntimateMemoryPrivacyErrorV1::RootHasDependencies);
                }
            }
            IntimateDerivationPolicyV1::ExternalExportRequiresNotice => {
                if self.retention != IntimateRetentionClassV1::ExternalExport
                    || self.export_receipt_ref.is_none()
                {
                    return Err(IntimateMemoryPrivacyErrorV1::InvalidExternalExport);
                }
                if self.source_ids.is_empty()
                    && self.dependency_state == IntimateDependencyStateV1::Complete
                {
                    return Err(IntimateMemoryPrivacyErrorV1::DerivedArtifactMissingSource);
                }
            }
            _ => {
                if self.source_ids.is_empty()
                    && self.dependency_state == IntimateDependencyStateV1::Complete
                {
                    return Err(IntimateMemoryPrivacyErrorV1::DerivedArtifactMissingSource);
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntimateRetractionActionV1 {
    DeleteLocal,
    RecomputeRequired,
    RetainIndependentBasis,
    ExternalExportNotice,
    BlockUnknownDependency,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntimateRetractionEventV1 {
    pub artifact_id: String,
    pub action: IntimateRetractionActionV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntimateRetractionStatusV1 {
    Applied,
    AlreadyRetracted,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntimateRetractionReceiptV1 {
    pub source_id: String,
    pub status: IntimateRetractionStatusV1,
    pub events: Vec<IntimateRetractionEventV1>,
    pub external_deletion_unproven: bool,
}

#[derive(Debug, Default)]
pub struct IntimateMemoryPrivacyGraphV1 {
    active: BTreeMap<String, IntimateMemoryArtifactV1>,
    known_ids: BTreeSet<String>,
}

impl IntimateMemoryPrivacyGraphV1 {
    pub fn insert(
        &mut self,
        artifact: IntimateMemoryArtifactV1,
    ) -> Result<(), IntimateMemoryPrivacyErrorV1> {
        artifact.validate_shape()?;
        if self.known_ids.contains(&artifact.artifact_id) {
            return Err(IntimateMemoryPrivacyErrorV1::DuplicateArtifactId);
        }

        if artifact.dependency_state == IntimateDependencyStateV1::Complete {
            for source_id in &artifact.source_ids {
                if !self.active.contains_key(source_id) {
                    return Err(IntimateMemoryPrivacyErrorV1::UnknownSourceArtifact);
                }
            }
        }

        if artifact.retention == IntimateRetentionClassV1::DurableOptIn {
            let promotes_ephemeral = artifact.source_ids.iter().any(|source_id| {
                self.active
                    .get(source_id)
                    .map(|source| source.retention == IntimateRetentionClassV1::EphemeralSession)
                    .unwrap_or(false)
            });
            if promotes_ephemeral && artifact.durability_authorization_ref.is_none() {
                return Err(
                    IntimateMemoryPrivacyErrorV1::EphemeralToDurableRequiresAuthorization,
                );
            }
        }

        self.known_ids.insert(artifact.artifact_id.clone());
        self.active.insert(artifact.artifact_id.clone(), artifact);
        Ok(())
    }

    pub fn is_retrievable(&self, artifact_id: &str) -> bool {
        self.active
            .get(artifact_id)
            .map(|artifact| artifact.dependency_state == IntimateDependencyStateV1::Complete)
            .unwrap_or(false)
    }

    pub fn active_artifact(&self, artifact_id: &str) -> Option<&IntimateMemoryArtifactV1> {
        self.active.get(artifact_id)
    }

    pub fn retract_source(
        &mut self,
        source_id: &str,
    ) -> Result<IntimateRetractionReceiptV1, IntimateMemoryPrivacyErrorV1> {
        let source_id = canonical_id(source_id)
            .ok_or(IntimateMemoryPrivacyErrorV1::InvalidArtifactId)?;
        if !self.known_ids.contains(&source_id) {
            return Err(IntimateMemoryPrivacyErrorV1::UnknownArtifactId);
        }
        if !self.active.contains_key(&source_id) {
            return Ok(IntimateRetractionReceiptV1 {
                source_id,
                status: IntimateRetractionStatusV1::AlreadyRetracted,
                events: Vec::new(),
                external_deletion_unproven: false,
            });
        }

        let mut events = Vec::new();
        let mut invalidated = VecDeque::new();
        self.active.remove(&source_id);
        invalidated.push_back(source_id.clone());
        events.push(IntimateRetractionEventV1 {
            artifact_id: source_id.clone(),
            action: IntimateRetractionActionV1::DeleteLocal,
        });

        // Dependency metadata marked Unknown cannot establish that it is
        // independent of the retracted source. Remove it from retrieval first.
        let unknown_ids: Vec<String> = self
            .active
            .iter()
            .filter(|(_, artifact)| {
                artifact.dependency_state == IntimateDependencyStateV1::Unknown
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in unknown_ids {
            self.active.remove(&id);
            invalidated.push_back(id.clone());
            events.push(IntimateRetractionEventV1 {
                artifact_id: id,
                action: IntimateRetractionActionV1::BlockUnknownDependency,
            });
        }

        let mut external_deletion_unproven = false;
        while let Some(invalidated_id) = invalidated.pop_front() {
            let dependent_ids: Vec<String> = self
                .active
                .iter()
                .filter(|(_, artifact)| artifact.source_ids.contains(&invalidated_id))
                .map(|(id, _)| id.clone())
                .collect();

            for dependent_id in dependent_ids {
                let Some(snapshot) = self.active.get(&dependent_id).cloned() else {
                    continue;
                };
                let action = if snapshot.reconstructive {
                    IntimateRetractionActionV1::DeleteLocal
                } else {
                    match snapshot.derivation_policy {
                        IntimateDerivationPolicyV1::RootSource
                        | IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction => {
                            IntimateRetractionActionV1::DeleteLocal
                        }
                        IntimateDerivationPolicyV1::MustRecomputeWithoutRetractedSource => {
                            IntimateRetractionActionV1::RecomputeRequired
                        }
                        IntimateDerivationPolicyV1::MayRetainWithIndependentBasis => {
                            let has_other_active_source = snapshot.source_ids.iter().any(|source_id| {
                                source_id != &invalidated_id && self.active.contains_key(source_id)
                            });
                            if has_other_active_source {
                                IntimateRetractionActionV1::RetainIndependentBasis
                            } else {
                                IntimateRetractionActionV1::DeleteLocal
                            }
                        }
                        IntimateDerivationPolicyV1::ExternalExportRequiresNotice => {
                            IntimateRetractionActionV1::ExternalExportNotice
                        }
                    }
                };

                match action {
                    IntimateRetractionActionV1::RetainIndependentBasis => {
                        if let Some(artifact) = self.active.get_mut(&dependent_id) {
                            artifact.source_ids.remove(&invalidated_id);
                        }
                    }
                    IntimateRetractionActionV1::ExternalExportNotice => {
                        external_deletion_unproven = true;
                        self.active.remove(&dependent_id);
                        invalidated.push_back(dependent_id.clone());
                    }
                    IntimateRetractionActionV1::DeleteLocal
                    | IntimateRetractionActionV1::RecomputeRequired
                    | IntimateRetractionActionV1::BlockUnknownDependency => {
                        self.active.remove(&dependent_id);
                        invalidated.push_back(dependent_id.clone());
                    }
                }
                events.push(IntimateRetractionEventV1 {
                    artifact_id: dependent_id,
                    action,
                });
            }
        }

        events.sort_by(|a, b| a.artifact_id.cmp(&b.artifact_id).then(a.action.cmp(&b.action)));
        Ok(IntimateRetractionReceiptV1 {
            source_id,
            status: IntimateRetractionStatusV1::Applied,
            events,
            external_deletion_unproven,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntimateMemoryPrivacyErrorV1 {
    InvalidArtifactId,
    InvalidAuthorizationRef,
    InvalidExportRef,
    DuplicateArtifactId,
    UnknownArtifactId,
    UnknownSourceArtifact,
    SelfDependency,
    RootHasDependencies,
    DerivedArtifactMissingSource,
    InvalidExternalExport,
    EphemeralToDurableRequiresAuthorization,
}

fn validate_id(value: &str, error: IntimateMemoryPrivacyErrorV1) -> Result<(), IntimateMemoryPrivacyErrorV1> {
    canonical_id(value).map(|_| ()).ok_or(error)
}

fn canonical_id(value: &str) -> Option<String> {
    let value = value.trim().to_owned();
    (!value.is_empty() && value.len() <= 256).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn derived(
        id: &str,
        source_ids: &[&str],
        policy: IntimateDerivationPolicyV1,
    ) -> IntimateMemoryArtifactV1 {
        IntimateMemoryArtifactV1 {
            artifact_id: id.into(),
            kind: IntimateArtifactKindV1::SessionSummary,
            retention: IntimateRetentionClassV1::EphemeralSession,
            dependency_state: IntimateDependencyStateV1::Complete,
            derivation_policy: policy,
            reconstructive: false,
            source_ids: source_ids.iter().map(|id| (*id).to_owned()).collect(),
            durability_authorization_ref: None,
            export_receipt_ref: None,
        }
    }

    #[test]
    fn fan_out_deletion_propagates_to_reconstructive_descendants() {
        let mut graph = IntimateMemoryPrivacyGraphV1::default();
        graph.insert(IntimateMemoryArtifactV1::root(
            "transcript",
            IntimateArtifactKindV1::RawTranscript,
            IntimateRetentionClassV1::EphemeralSession,
        ).unwrap()).unwrap();
        graph.insert(derived(
            "summary",
            &["transcript"],
            IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
        )).unwrap();
        let mut embedding = derived(
            "embedding",
            &["summary"],
            IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
        );
        embedding.kind = IntimateArtifactKindV1::EmbeddingOrIndex;
        embedding.reconstructive = true;
        graph.insert(embedding).unwrap();

        let receipt = graph.retract_source("transcript").unwrap();
        assert_eq!(receipt.status, IntimateRetractionStatusV1::Applied);
        assert!(!graph.is_retrievable("summary"));
        assert!(!graph.is_retrievable("embedding"));
        assert!(receipt.events.iter().any(|event| {
            event.artifact_id == "embedding"
                && event.action == IntimateRetractionActionV1::DeleteLocal
        }));
    }

    #[test]
    fn fan_in_can_retain_explicit_independent_basis() {
        let mut graph = IntimateMemoryPrivacyGraphV1::default();
        for id in ["source-a", "source-b"] {
            graph.insert(IntimateMemoryArtifactV1::root(
                id,
                IntimateArtifactKindV1::ExplicitPreference,
                IntimateRetentionClassV1::DurableOptIn,
            ).unwrap()).unwrap();
        }
        let mut preference = derived(
            "preference",
            &["source-a", "source-b"],
            IntimateDerivationPolicyV1::MayRetainWithIndependentBasis,
        );
        preference.kind = IntimateArtifactKindV1::ExplicitPreference;
        preference.retention = IntimateRetentionClassV1::DurableOptIn;
        graph.insert(preference).unwrap();

        let receipt = graph.retract_source("source-a").unwrap();
        assert!(graph.is_retrievable("preference"));
        assert_eq!(
            graph.active_artifact("preference").unwrap().source_ids,
            BTreeSet::from(["source-b".to_owned()])
        );
        assert!(receipt.events.iter().any(|event| {
            event.artifact_id == "preference"
                && event.action == IntimateRetractionActionV1::RetainIndependentBasis
        }));
    }

    #[test]
    fn ephemeral_source_cannot_silently_become_durable() {
        let mut graph = IntimateMemoryPrivacyGraphV1::default();
        graph.insert(IntimateMemoryArtifactV1::root(
            "ephemeral",
            IntimateArtifactKindV1::RawTranscript,
            IntimateRetentionClassV1::EphemeralSession,
        ).unwrap()).unwrap();
        let mut summary = derived(
            "durable-summary",
            &["ephemeral"],
            IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
        );
        summary.retention = IntimateRetentionClassV1::DurableOptIn;
        assert_eq!(
            graph.insert(summary.clone()),
            Err(IntimateMemoryPrivacyErrorV1::EphemeralToDurableRequiresAuthorization)
        );
        summary.durability_authorization_ref = Some("user-opt-in:receipt".into());
        assert!(graph.insert(summary).is_ok());
    }

    #[test]
    fn external_export_surfaces_unproven_deletion() {
        let mut graph = IntimateMemoryPrivacyGraphV1::default();
        graph.insert(IntimateMemoryArtifactV1::root(
            "preference",
            IntimateArtifactKindV1::ExplicitPreference,
            IntimateRetentionClassV1::DurableOptIn,
        ).unwrap()).unwrap();
        graph.insert(IntimateMemoryArtifactV1 {
            artifact_id: "export".into(),
            kind: IntimateArtifactKindV1::ExternalExport,
            retention: IntimateRetentionClassV1::ExternalExport,
            dependency_state: IntimateDependencyStateV1::Complete,
            derivation_policy: IntimateDerivationPolicyV1::ExternalExportRequiresNotice,
            reconstructive: false,
            source_ids: BTreeSet::from(["preference".into()]),
            durability_authorization_ref: None,
            export_receipt_ref: Some("export:receipt".into()),
        }).unwrap();

        let receipt = graph.retract_source("preference").unwrap();
        assert!(receipt.external_deletion_unproven);
        assert!(receipt.events.iter().any(|event| {
            event.artifact_id == "export"
                && event.action == IntimateRetractionActionV1::ExternalExportNotice
        }));
    }

    #[test]
    fn unknown_dependency_fails_closed_on_retraction() {
        let mut graph = IntimateMemoryPrivacyGraphV1::default();
        graph.insert(IntimateMemoryArtifactV1::root(
            "source",
            IntimateArtifactKindV1::RawTranscript,
            IntimateRetentionClassV1::EphemeralSession,
        ).unwrap()).unwrap();
        graph.insert(IntimateMemoryArtifactV1 {
            artifact_id: "unknown-summary".into(),
            kind: IntimateArtifactKindV1::ReflectiveMemory,
            retention: IntimateRetentionClassV1::EphemeralSession,
            dependency_state: IntimateDependencyStateV1::Unknown,
            derivation_policy: IntimateDerivationPolicyV1::MustRecomputeWithoutRetractedSource,
            reconstructive: false,
            source_ids: BTreeSet::new(),
            durability_authorization_ref: None,
            export_receipt_ref: None,
        }).unwrap();
        assert!(!graph.is_retrievable("unknown-summary"));
        let receipt = graph.retract_source("source").unwrap();
        assert!(receipt.events.iter().any(|event| {
            event.artifact_id == "unknown-summary"
                && event.action == IntimateRetractionActionV1::BlockUnknownDependency
        }));
    }

    #[test]
    fn repeated_retraction_is_idempotent() {
        let mut graph = IntimateMemoryPrivacyGraphV1::default();
        graph.insert(IntimateMemoryArtifactV1::root(
            "source",
            IntimateArtifactKindV1::PsychologyEvidence,
            IntimateRetentionClassV1::DurableOptIn,
        ).unwrap()).unwrap();
        let first = graph.retract_source("source").unwrap();
        let second = graph.retract_source("source").unwrap();
        assert_eq!(first.status, IntimateRetractionStatusV1::Applied);
        assert_eq!(second.status, IntimateRetractionStatusV1::AlreadyRetracted);
        assert!(second.events.is_empty());
    }

    #[test]
    fn retired_artifact_identity_cannot_be_reused() {
        let mut graph = IntimateMemoryPrivacyGraphV1::default();
        let source = IntimateMemoryArtifactV1::root(
            "source",
            IntimateArtifactKindV1::PsychologyEvidence,
            IntimateRetentionClassV1::DurableOptIn,
        ).unwrap();
        graph.insert(source.clone()).unwrap();
        graph.retract_source("source").unwrap();
        assert_eq!(
            graph.insert(source),
            Err(IntimateMemoryPrivacyErrorV1::DuplicateArtifactId)
        );
    }
}
