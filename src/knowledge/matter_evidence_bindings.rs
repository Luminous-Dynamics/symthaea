// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Matter-domain bindings to canonical external evidence references.
//!
//! These adapters sit above `symthaea-nuclear` and `symthaea-evidence-plane`,
//! both of which are already root dependencies. They therefore strengthen
//! provenance checks without adding a new dependency edge to the nuclear crate.
//!
//! The adapters validate local consistency only. They never turn a caller-
//! supplied digest/date/role into independently verified chronology, experiment,
//! preregistration, or replication authority.

use std::collections::BTreeSet;
use std::fmt;

use symthaea_evidence_plane::external_receipt::{
    ClaimedUtcDate, DeclaredChronologyInterpretation, DeclaredTemporalRelation,
    EvidenceReferenceInterpretation, EvidenceRole, ExternalEvidenceBundle, ExternalEvidenceError,
};
use symthaea_nuclear::{
    HistoricalMassSnapshot, HistoricalModelKnowledge, PreregisteredBlindCriteria, SnapshotAuthority,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatterEvidenceBindingError {
    ExternalEvidence(ExternalEvidenceError),
    InvalidHistoricalSnapshot(String),
    SyntheticSnapshotHasNoExternalArtifact,
    SnapshotArtifactHashMismatch,
    SnapshotChronologyDateMismatch {
        snapshot_yyyymmdd: u32,
        reference_yyyymmdd: u32,
    },
    InvalidModelKnowledge,
    MissingReferenceDate(String),
    ReferenceDateAfterKnowledgeCutoff {
        evidence_id: String,
        reference_yyyymmdd: u32,
        cutoff_yyyymmdd: u32,
    },
}

impl From<ExternalEvidenceError> for MatterEvidenceBindingError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for MatterEvidenceBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExternalEvidence(error) => write!(f, "{error}"),
            Self::InvalidHistoricalSnapshot(error) => {
                write!(f, "historical snapshot failed local validation: {error}")
            }
            Self::SyntheticSnapshotHasNoExternalArtifact => write!(
                f,
                "synthetic historical fixtures cannot be bound as external artifacts"
            ),
            Self::SnapshotArtifactHashMismatch => write!(
                f,
                "snapshot authority digest does not match referenced artifact digest"
            ),
            Self::SnapshotChronologyDateMismatch {
                snapshot_yyyymmdd,
                reference_yyyymmdd,
            } => write!(
                f,
                "snapshot release date {snapshot_yyyymmdd} differs from chronology reference date {reference_yyyymmdd}"
            ),
            Self::InvalidModelKnowledge => write!(
                f,
                "historical model knowledge has an invalid cutoff or malformed provenance identifiers"
            ),
            Self::MissingReferenceDate(id) => {
                write!(f, "external evidence `{id}` has no claimed UTC date")
            }
            Self::ReferenceDateAfterKnowledgeCutoff {
                evidence_id,
                reference_yyyymmdd,
                cutoff_yyyymmdd,
            } => write!(
                f,
                "external evidence `{evidence_id}` date {reference_yyyymmdd} postdates model knowledge cutoff {cutoff_yyyymmdd}"
            ),
        }
    }
}

impl std::error::Error for MatterEvidenceBindingError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalSnapshotReferenceBinding {
    pub source_id: String,
    pub artifact_evidence_id: String,
    pub chronology_evidence_id: String,
    pub release_yyyymmdd: u32,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub chronology_interpretation: DeclaredChronologyInterpretation,
}

/// Bind one historical snapshot to an artifact reference and its declared
/// chronology reference. Role/hash/date agreement is checked; external truth is
/// not.
pub fn bind_historical_snapshot_references(
    snapshot: &HistoricalMassSnapshot,
    artifact_evidence_id: &str,
    bundle: &ExternalEvidenceBundle,
) -> Result<HistoricalSnapshotReferenceBinding, MatterEvidenceBindingError> {
    snapshot
        .validate()
        .map_err(|error| MatterEvidenceBindingError::InvalidHistoricalSnapshot(error.to_string()))?;

    let metadata = snapshot.metadata();
    let expected_digest = match &metadata.authority {
        SnapshotAuthority::ExternalArtifactSha256 { sha256_hex } => sha256_hex.as_str(),
        SnapshotAuthority::SyntheticFixture => {
            return Err(MatterEvidenceBindingError::SyntheticSnapshotHasNoExternalArtifact)
        }
    };

    let artifact = bundle.require_role(artifact_evidence_id, EvidenceRole::ArtifactContent)?;
    if !artifact
        .content_sha256
        .as_str()
        .eq_ignore_ascii_case(expected_digest)
    {
        return Err(MatterEvidenceBindingError::SnapshotArtifactHashMismatch);
    }

    let chronology = bundle.require_role(
        &metadata.chronology_evidence_id,
        EvidenceRole::ChronologyAttestation,
    )?;
    let chronology_date = chronology
        .claimed_utc_date
        .ok_or_else(|| MatterEvidenceBindingError::MissingReferenceDate(metadata.chronology_evidence_id.clone()))?;
    if chronology_date.yyyymmdd() != metadata.release_yyyymmdd {
        return Err(MatterEvidenceBindingError::SnapshotChronologyDateMismatch {
            snapshot_yyyymmdd: metadata.release_yyyymmdd,
            reference_yyyymmdd: chronology_date.yyyymmdd(),
        });
    }

    Ok(HistoricalSnapshotReferenceBinding {
        source_id: metadata.source_id.clone(),
        artifact_evidence_id: artifact.id.as_str().to_string(),
        chronology_evidence_id: chronology.id.as_str().to_string(),
        release_yyyymmdd: metadata.release_yyyymmdd,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalSnapshotPairReferenceBinding {
    pub earlier: HistoricalSnapshotReferenceBinding,
    pub later: HistoricalSnapshotReferenceBinding,
    pub declared_order: DeclaredTemporalRelation,
}

/// Bind an earlier/later snapshot pair to two distinct content artifacts and an
/// increasing declared chronology.
pub fn bind_historical_snapshot_pair_references(
    earlier: &HistoricalMassSnapshot,
    earlier_artifact_evidence_id: &str,
    later: &HistoricalMassSnapshot,
    later_artifact_evidence_id: &str,
    bundle: &ExternalEvidenceBundle,
) -> Result<HistoricalSnapshotPairReferenceBinding, MatterEvidenceBindingError> {
    let earlier_binding =
        bind_historical_snapshot_references(earlier, earlier_artifact_evidence_id, bundle)?;
    let later_binding = bind_historical_snapshot_references(later, later_artifact_evidence_id, bundle)?;

    bundle.require_distinct_content(earlier_artifact_evidence_id, later_artifact_evidence_id)?;
    let declared_order = bundle.declared_temporal_order(
        &earlier_binding.chronology_evidence_id,
        &later_binding.chronology_evidence_id,
    )?;

    Ok(HistoricalSnapshotPairReferenceBinding {
        earlier: earlier_binding,
        later: later_binding,
        declared_order,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalModelKnowledgeReferenceBinding {
    pub model_id: String,
    pub cutoff_yyyymmdd: u32,
    pub implementation_evidence_id: String,
    pub dependency_evidence_ids: Vec<String>,
    pub interpretation: EvidenceReferenceInterpretation,
}

/// Bind model implementation/dependency references and ensure every claimed
/// reference date is no later than the model's own declared knowledge cutoff.
pub fn bind_historical_model_knowledge_references(
    knowledge: &HistoricalModelKnowledge,
    bundle: &ExternalEvidenceBundle,
) -> Result<HistoricalModelKnowledgeReferenceBinding, MatterEvidenceBindingError> {
    if knowledge.model_id.trim().is_empty()
        || knowledge.frozen_implementation_evidence_id.trim().is_empty()
        || knowledge.dependency_evidence_ids.is_empty()
        || knowledge
            .dependency_evidence_ids
            .iter()
            .any(|id| id.trim().is_empty())
        || knowledge.latest_dependency_yyyymmdd < 19000101
        || ClaimedUtcDate::new(knowledge.latest_dependency_yyyymmdd).is_err()
    {
        return Err(MatterEvidenceBindingError::InvalidModelKnowledge);
    }

    let unique: BTreeSet<_> = knowledge.dependency_evidence_ids.iter().collect();
    if unique.len() != knowledge.dependency_evidence_ids.len() {
        return Err(MatterEvidenceBindingError::InvalidModelKnowledge);
    }

    let implementation = bundle.require_role(
        &knowledge.frozen_implementation_evidence_id,
        EvidenceRole::ImplementationSnapshot,
    )?;
    require_reference_no_later_than(
        implementation.id.as_str(),
        implementation.claimed_utc_date,
        knowledge.latest_dependency_yyyymmdd,
    )?;

    for dependency_id in &knowledge.dependency_evidence_ids {
        let dependency = bundle.require_role(dependency_id, EvidenceRole::DependencySnapshot)?;
        require_reference_no_later_than(
            dependency.id.as_str(),
            dependency.claimed_utc_date,
            knowledge.latest_dependency_yyyymmdd,
        )?;
    }

    Ok(HistoricalModelKnowledgeReferenceBinding {
        model_id: knowledge.model_id.clone(),
        cutoff_yyyymmdd: knowledge.latest_dependency_yyyymmdd,
        implementation_evidence_id: implementation.id.as_str().to_string(),
        dependency_evidence_ids: knowledge.dependency_evidence_ids.clone(),
        interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
    })
}

fn require_reference_no_later_than(
    evidence_id: &str,
    date: Option<ClaimedUtcDate>,
    cutoff_yyyymmdd: u32,
) -> Result<(), MatterEvidenceBindingError> {
    let date = date
        .ok_or_else(|| MatterEvidenceBindingError::MissingReferenceDate(evidence_id.to_string()))?;
    if date.yyyymmdd() > cutoff_yyyymmdd {
        return Err(MatterEvidenceBindingError::ReferenceDateAfterKnowledgeCutoff {
            evidence_id: evidence_id.to_string(),
            reference_yyyymmdd: date.yyyymmdd(),
            cutoff_yyyymmdd,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreregistrationReferenceBinding {
    pub registration_evidence_id: String,
    pub result_evidence_id: String,
    pub declared_order: DeclaredTemporalRelation,
}

/// Bind a blind-criteria registration reference to a later solver-result
/// reference. The output remains declared chronology only.
pub fn bind_preregistered_blind_criteria_reference(
    criteria: &PreregisteredBlindCriteria,
    result_evidence_id: &str,
    bundle: &ExternalEvidenceBundle,
) -> Result<PreregistrationReferenceBinding, MatterEvidenceBindingError> {
    bundle.require_role(
        criteria.registration_evidence_id(),
        EvidenceRole::Preregistration,
    )?;
    bundle.require_role(result_evidence_id, EvidenceRole::SolverExecution)?;
    let declared_order = bundle.require_preregistration_before(
        criteria.registration_evidence_id(),
        result_evidence_id,
    )?;

    Ok(PreregistrationReferenceBinding {
        registration_evidence_id: criteria.registration_evidence_id().to_string(),
        result_evidence_id: result_evidence_id.to_string(),
        declared_order,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_evidence_plane::external_receipt::ExternalEvidenceReference;
    use symthaea_nuclear::{BlindHoldout, BlindMetricBounds, MeasuredNucleus};

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn reference(
        id: &str,
        role: EvidenceRole,
        digest_char: char,
        date: u32,
    ) -> ExternalEvidenceReference {
        ExternalEvidenceReference::try_new(
            id,
            role,
            digest(digest_char),
            format!("artifact://{id}"),
            format!("subject:{id}"),
            Some(date),
            Some("fixture issuer".to_string()),
        )
        .unwrap()
    }

    fn snapshot(
        source: &str,
        release: u32,
        chronology_id: &str,
        digest_char: char,
        z: u16,
    ) -> HistoricalMassSnapshot {
        HistoricalMassSnapshot::new(
            source,
            release,
            chronology_id,
            SnapshotAuthority::ExternalArtifactSha256 {
                sha256_hex: digest(digest_char),
            },
            &[MeasuredNucleus {
                z,
                n: z + 1,
                binding_energy_mev: 100.0 + z as f64,
                is_measured: true,
            }],
        )
        .unwrap()
    }

    #[test]
    fn historical_pair_stays_declared_chronology_only() {
        let early = snapshot("old", 20121201, "old-time", 'a', 20);
        let late = snapshot("new", 20201201, "new-time", 'b', 21);
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("old-artifact", EvidenceRole::ArtifactContent, 'a', 20121201),
            reference("old-time", EvidenceRole::ChronologyAttestation, 'c', 20121201),
            reference("new-artifact", EvidenceRole::ArtifactContent, 'b', 20201201),
            reference("new-time", EvidenceRole::ChronologyAttestation, 'd', 20201201),
        ])
        .unwrap();

        let binding = bind_historical_snapshot_pair_references(
            &early,
            "old-artifact",
            &late,
            "new-artifact",
            &bundle,
        )
        .unwrap();
        assert_eq!(
            binding.declared_order.interpretation(),
            DeclaredChronologyInterpretation::DeclaredChronologyOnly
        );
        assert_eq!(
            binding.earlier.reference_interpretation,
            EvidenceReferenceInterpretation::ReferenceOnly
        );
    }

    #[test]
    fn snapshot_hash_mismatch_fails_closed() {
        let snap = snapshot("old", 20121201, "old-time", 'a', 20);
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("artifact", EvidenceRole::ArtifactContent, 'b', 20121201),
            reference("old-time", EvidenceRole::ChronologyAttestation, 'c', 20121201),
        ])
        .unwrap();
        assert!(matches!(
            bind_historical_snapshot_references(&snap, "artifact", &bundle),
            Err(MatterEvidenceBindingError::SnapshotArtifactHashMismatch)
        ));
    }

    #[test]
    fn dependency_reference_cannot_postdate_model_cutoff() {
        let knowledge = HistoricalModelKnowledge::new(
            "rf-vintage",
            20121201,
            "implementation",
            vec!["dz-prior".to_string()],
        )
        .unwrap();
        let bundle = ExternalEvidenceBundle::new(vec![
            reference(
                "implementation",
                EvidenceRole::ImplementationSnapshot,
                'a',
                20120101,
            ),
            reference("dz-prior", EvidenceRole::DependencySnapshot, 'b', 20130101),
        ])
        .unwrap();
        assert!(matches!(
            bind_historical_model_knowledge_references(&knowledge, &bundle),
            Err(MatterEvidenceBindingError::ReferenceDateAfterKnowledgeCutoff { .. })
        ));
    }

    #[test]
    fn malformed_deserialized_model_cutoff_is_rechecked() {
        let knowledge = HistoricalModelKnowledge {
            model_id: "rf-vintage".to_string(),
            latest_dependency_yyyymmdd: 20261301,
            frozen_implementation_evidence_id: "implementation".to_string(),
            dependency_evidence_ids: vec!["dz-prior".to_string()],
        };
        let bundle = ExternalEvidenceBundle::new(vec![
            reference(
                "implementation",
                EvidenceRole::ImplementationSnapshot,
                'a',
                20120101,
            ),
            reference("dz-prior", EvidenceRole::DependencySnapshot, 'b', 20120101),
        ])
        .unwrap();
        assert!(matches!(
            bind_historical_model_knowledge_references(&knowledge, &bundle),
            Err(MatterEvidenceBindingError::InvalidModelKnowledge)
        ));
    }

    #[test]
    fn preregistration_binding_stays_declared_only() {
        let criteria = PreregisteredBlindCriteria::new(
            "registration",
            BlindHoldout::ProtonFrontier { train_z_max: 82 },
            "candidate",
            BlindMetricBounds {
                max_rms_mev: 1.0,
                max_mae_mev: 1.0,
                max_abs_bias_mev: 1.0,
                max_max_abs_error_mev: 2.0,
            },
            None,
        )
        .unwrap();
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("registration", EvidenceRole::Preregistration, 'a', 20260101),
            reference("result", EvidenceRole::SolverExecution, 'b', 20260201),
        ])
        .unwrap();
        let binding =
            bind_preregistered_blind_criteria_reference(&criteria, "result", &bundle).unwrap();
        assert_eq!(
            binding.declared_order.interpretation(),
            DeclaredChronologyInterpretation::DeclaredChronologyOnly
        );
    }
}
