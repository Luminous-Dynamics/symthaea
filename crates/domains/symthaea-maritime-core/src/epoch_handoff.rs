// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semantic qualification for inventory carried across regenerative lineage epochs.
//!
//! Symtropy owns runtime accounting/conservation. This module independently checks
//! whether the *declared mappings* between predecessor and successor dependencies
//! are admissible under exact Symthaea closure/Genome evidence. It contains no
//! manufacturing recipes, process settings, physical actuation, or operating authority.

use crate::{
    validate_regenerative_lineage_successor, DependencyGovernance, RegenerativeClosureModel,
    RegenerativeGenomeError, RegenerativeGenomeV1, RegenerativeLineageViabilityError,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REGENERATIVE_EPOCH_HANDOFF_SCHEMA_V1: u8 = 1;

const MAX_ITEMS: usize = 4096;
const MAX_ID_LEN: usize = 256;
const MAX_BINDING_LEN: usize = 1024;

/// Qualification evidence for one predecessor->successor inventory mapping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeEpochTransferQualificationV1 {
    pub source_dependency_id: String,
    pub successor_dependency_id: String,
    /// Evidence that inventory represented by the predecessor dependency may be
    /// admitted under the successor dependency identity.
    pub transfer_qualification_binding: String,
    /// Required only for `SafeguardedExternal` continuity.
    pub safeguarded_continuity_binding: Option<String>,
}

/// Qualification evidence for inventory entering the successor from outside the predecessor epoch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeEpochExternalAdmissionQualificationV1 {
    pub successor_dependency_id: String,
    /// Exact evidence identity of the externally admitted inventory/service entitlement.
    pub inventory_evidence_binding: String,
    /// Evidence that this admission is acceptable for the successor dependency.
    pub admission_qualification_binding: String,
    /// Required only when admitting a safeguarded-external dependency.
    pub safeguarded_admission_binding: Option<String>,
}

/// Exact cross-system evidence contract for one industrial-lineage epoch handoff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeEpochHandoffEvidenceV1 {
    pub schema_version: u8,
    pub handoff_id: String,
    pub source_epoch_id: String,
    pub source_epoch_evidence_binding: String,
    pub successor_epoch_id: String,
    pub successor_epoch_evidence_binding: String,
    pub source_genome_id: String,
    pub source_genome_evidence_binding: String,
    pub successor_genome_id: String,
    pub successor_genome_evidence_binding: String,
    /// Opaque binding to the exact dynamic accounting receipt produced downstream.
    pub dynamic_handoff_receipt_binding: String,
    /// Sorted and unique by `(source_dependency_id, successor_dependency_id)`.
    pub transfer_qualifications: Vec<RegenerativeEpochTransferQualificationV1>,
    /// Sorted and unique by `successor_dependency_id`.
    pub external_admission_qualifications:
        Vec<RegenerativeEpochExternalAdmissionQualificationV1>,
    pub evidence_binding: String,
}

/// Diagnostic result. This qualifies evidence relationships, not physical execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeEpochHandoffQualificationReportV1 {
    pub handoff_id: String,
    pub source_genome_id: String,
    pub successor_genome_id: String,
    pub dynamic_handoff_receipt_binding: String,
    pub qualified_transfer_count: u16,
    pub cross_id_transfer_count: u16,
    pub safeguarded_transfer_count: u16,
    pub external_admission_count: u16,
    pub safeguarded_external_admission_count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeEpochHandoffError {
    UnsupportedSchemaVersion { schema_version: u8 },
    InvalidIdentifier,
    InvalidEvidenceBinding,
    TooManyEvidenceItems,
    NoInventoryAdmissions,
    EpochIdentityNotDistinct,
    GenomeIdentityMismatch,
    SourceGenomeInvalid(RegenerativeGenomeError),
    SuccessorGenomeInvalid(RegenerativeGenomeError),
    InvalidLineageSuccessor(RegenerativeLineageViabilityError),
    NonCanonicalTransferOrder,
    NonCanonicalExternalAdmissionOrder,
    DuplicateSuccessorTransferTarget { dependency_id: String },
    UnknownSourceDependency { dependency_id: String },
    UnknownSuccessorDependency { dependency_id: String },
    DependencyKindMismatch {
        source_dependency_id: String,
        successor_dependency_id: String,
    },
    DependencyGovernanceMismatch {
        source_dependency_id: String,
        successor_dependency_id: String,
    },
    MissingSafeguardedContinuity { dependency_id: String },
    UnexpectedSafeguardedContinuity { dependency_id: String },
    MissingSafeguardedAdmission { dependency_id: String },
    UnexpectedSafeguardedAdmission { dependency_id: String },
    CountOverflow,
}

/// Validate semantic admissibility of a dynamic epoch handoff against exact Genome/closure evidence.
pub fn qualify_regenerative_epoch_handoff(
    evidence: &RegenerativeEpochHandoffEvidenceV1,
    source_genome: &RegenerativeGenomeV1,
    source_model: &RegenerativeClosureModel,
    successor_genome: &RegenerativeGenomeV1,
    successor_model: &RegenerativeClosureModel,
) -> Result<RegenerativeEpochHandoffQualificationReportV1, RegenerativeEpochHandoffError> {
    validate_shape(evidence)?;
    source_genome
        .validate_against_model(source_model)
        .map_err(RegenerativeEpochHandoffError::SourceGenomeInvalid)?;
    successor_genome
        .validate_against_model(successor_model)
        .map_err(RegenerativeEpochHandoffError::SuccessorGenomeInvalid)?;
    validate_regenerative_lineage_successor(source_genome, successor_genome)
        .map_err(RegenerativeEpochHandoffError::InvalidLineageSuccessor)?;

    if evidence.source_genome_id != source_genome.genome_id
        || evidence.source_genome_evidence_binding != source_genome.evidence_binding
        || evidence.successor_genome_id != successor_genome.genome_id
        || evidence.successor_genome_evidence_binding != successor_genome.evidence_binding
    {
        return Err(RegenerativeEpochHandoffError::GenomeIdentityMismatch);
    }

    let source_dependencies: BTreeMap<&str, _> = source_model
        .dependencies
        .iter()
        .map(|dependency| (dependency.dependency_id.as_str(), dependency))
        .collect();
    let successor_dependencies: BTreeMap<&str, _> = successor_model
        .dependencies
        .iter()
        .map(|dependency| (dependency.dependency_id.as_str(), dependency))
        .collect();

    let mut successor_targets = BTreeSet::new();
    let mut cross_id = 0usize;
    let mut safeguarded_transfers = 0usize;
    for transfer in &evidence.transfer_qualifications {
        let source = source_dependencies
            .get(transfer.source_dependency_id.as_str())
            .ok_or_else(|| RegenerativeEpochHandoffError::UnknownSourceDependency {
                dependency_id: transfer.source_dependency_id.clone(),
            })?;
        let successor = successor_dependencies
            .get(transfer.successor_dependency_id.as_str())
            .ok_or_else(|| RegenerativeEpochHandoffError::UnknownSuccessorDependency {
                dependency_id: transfer.successor_dependency_id.clone(),
            })?;
        if !successor_targets.insert(transfer.successor_dependency_id.clone()) {
            return Err(RegenerativeEpochHandoffError::DuplicateSuccessorTransferTarget {
                dependency_id: transfer.successor_dependency_id.clone(),
            });
        }
        if source.kind != successor.kind {
            return Err(RegenerativeEpochHandoffError::DependencyKindMismatch {
                source_dependency_id: transfer.source_dependency_id.clone(),
                successor_dependency_id: transfer.successor_dependency_id.clone(),
            });
        }
        if source.governance != successor.governance {
            return Err(RegenerativeEpochHandoffError::DependencyGovernanceMismatch {
                source_dependency_id: transfer.source_dependency_id.clone(),
                successor_dependency_id: transfer.successor_dependency_id.clone(),
            });
        }
        if transfer.source_dependency_id != transfer.successor_dependency_id {
            cross_id += 1;
        }
        match source.governance {
            DependencyGovernance::SafeguardedExternal => {
                safeguarded_transfers += 1;
                if transfer.safeguarded_continuity_binding.is_none() {
                    return Err(RegenerativeEpochHandoffError::MissingSafeguardedContinuity {
                        dependency_id: transfer.source_dependency_id.clone(),
                    });
                }
            }
            DependencyGovernance::Ordinary => {
                if transfer.safeguarded_continuity_binding.is_some() {
                    return Err(RegenerativeEpochHandoffError::UnexpectedSafeguardedContinuity {
                        dependency_id: transfer.source_dependency_id.clone(),
                    });
                }
            }
        }
    }

    let mut safeguarded_external_admissions = 0usize;
    for admission in &evidence.external_admission_qualifications {
        let successor = successor_dependencies
            .get(admission.successor_dependency_id.as_str())
            .ok_or_else(|| RegenerativeEpochHandoffError::UnknownSuccessorDependency {
                dependency_id: admission.successor_dependency_id.clone(),
            })?;
        match successor.governance {
            DependencyGovernance::SafeguardedExternal => {
                safeguarded_external_admissions += 1;
                if admission.safeguarded_admission_binding.is_none() {
                    return Err(RegenerativeEpochHandoffError::MissingSafeguardedAdmission {
                        dependency_id: admission.successor_dependency_id.clone(),
                    });
                }
            }
            DependencyGovernance::Ordinary => {
                if admission.safeguarded_admission_binding.is_some() {
                    return Err(RegenerativeEpochHandoffError::UnexpectedSafeguardedAdmission {
                        dependency_id: admission.successor_dependency_id.clone(),
                    });
                }
            }
        }
    }

    Ok(RegenerativeEpochHandoffQualificationReportV1 {
        handoff_id: evidence.handoff_id.clone(),
        source_genome_id: source_genome.genome_id.clone(),
        successor_genome_id: successor_genome.genome_id.clone(),
        dynamic_handoff_receipt_binding: evidence.dynamic_handoff_receipt_binding.clone(),
        qualified_transfer_count: count(evidence.transfer_qualifications.len())?,
        cross_id_transfer_count: count(cross_id)?,
        safeguarded_transfer_count: count(safeguarded_transfers)?,
        external_admission_count: count(evidence.external_admission_qualifications.len())?,
        safeguarded_external_admission_count: count(safeguarded_external_admissions)?,
    })
}

fn validate_shape(
    evidence: &RegenerativeEpochHandoffEvidenceV1,
) -> Result<(), RegenerativeEpochHandoffError> {
    if evidence.schema_version != REGENERATIVE_EPOCH_HANDOFF_SCHEMA_V1 {
        return Err(RegenerativeEpochHandoffError::UnsupportedSchemaVersion {
            schema_version: evidence.schema_version,
        });
    }
    for id in [
        &evidence.handoff_id,
        &evidence.source_epoch_id,
        &evidence.successor_epoch_id,
        &evidence.source_genome_id,
        &evidence.successor_genome_id,
    ] {
        validate_id(id)?;
    }
    for binding in [
        &evidence.source_epoch_evidence_binding,
        &evidence.successor_epoch_evidence_binding,
        &evidence.source_genome_evidence_binding,
        &evidence.successor_genome_evidence_binding,
        &evidence.dynamic_handoff_receipt_binding,
        &evidence.evidence_binding,
    ] {
        validate_binding(binding)?;
    }
    if evidence.source_epoch_id == evidence.successor_epoch_id
        || evidence.source_epoch_evidence_binding == evidence.successor_epoch_evidence_binding
    {
        return Err(RegenerativeEpochHandoffError::EpochIdentityNotDistinct);
    }
    if evidence.transfer_qualifications.len() > MAX_ITEMS
        || evidence.external_admission_qualifications.len() > MAX_ITEMS
    {
        return Err(RegenerativeEpochHandoffError::TooManyEvidenceItems);
    }
    if evidence.transfer_qualifications.is_empty()
        && evidence.external_admission_qualifications.is_empty()
    {
        return Err(RegenerativeEpochHandoffError::NoInventoryAdmissions);
    }

    for transfer in &evidence.transfer_qualifications {
        validate_id(&transfer.source_dependency_id)?;
        validate_id(&transfer.successor_dependency_id)?;
        validate_binding(&transfer.transfer_qualification_binding)?;
        if let Some(binding) = &transfer.safeguarded_continuity_binding {
            validate_binding(binding)?;
        }
    }
    if evidence.transfer_qualifications.windows(2).any(|pair| {
        (
            pair[0].source_dependency_id.as_str(),
            pair[0].successor_dependency_id.as_str(),
        ) >= (
            pair[1].source_dependency_id.as_str(),
            pair[1].successor_dependency_id.as_str(),
        )
    }) {
        return Err(RegenerativeEpochHandoffError::NonCanonicalTransferOrder);
    }

    for admission in &evidence.external_admission_qualifications {
        validate_id(&admission.successor_dependency_id)?;
        validate_binding(&admission.inventory_evidence_binding)?;
        validate_binding(&admission.admission_qualification_binding)?;
        if let Some(binding) = &admission.safeguarded_admission_binding {
            validate_binding(binding)?;
        }
    }
    if evidence
        .external_admission_qualifications
        .windows(2)
        .any(|pair| pair[0].successor_dependency_id >= pair[1].successor_dependency_id)
    {
        return Err(RegenerativeEpochHandoffError::NonCanonicalExternalAdmissionOrder);
    }
    Ok(())
}

fn count(value: usize) -> Result<u16, RegenerativeEpochHandoffError> {
    u16::try_from(value).map_err(|_| RegenerativeEpochHandoffError::CountOverflow)
}

fn validate_id(value: &str) -> Result<(), RegenerativeEpochHandoffError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeEpochHandoffError::InvalidIdentifier)
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativeEpochHandoffError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || !value.contains(':')
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeEpochHandoffError::InvalidEvidenceBinding)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        RegenerativeCapability, RegenerativeDependency, RegenerativeDependencyKind,
        RegenerativeGenomeRequirementV1, REGENERATIVE_GENOME_SCHEMA_V1,
    };

    fn dependency(
        id: &str,
        kind: RegenerativeDependencyKind,
        governance: DependencyGovernance,
    ) -> RegenerativeDependency {
        RegenerativeDependency {
            dependency_id: id.into(),
            kind,
            governance,
            demand_units_per_period: 1,
            local_production_units_per_period: 0,
            recycling_units_per_period: 0,
            stockpile_units: 10,
            unit_mass_grams: None,
            evidence_binding: format!("dependency:{id}"),
        }
    }

    fn model(version: &str) -> RegenerativeClosureModel {
        RegenerativeClosureModel {
            model_id: format!("manta-{version}"),
            period_duration_ms: 1,
            dependencies: vec![
                dependency(
                    &format!("metrology-{version}"),
                    RegenerativeDependencyKind::Metrology,
                    DependencyGovernance::Ordinary,
                ),
                dependency(
                    &format!("reactor-service-{version}"),
                    RegenerativeDependencyKind::ExternalService,
                    DependencyGovernance::SafeguardedExternal,
                ),
            ],
            capabilities: vec![RegenerativeCapability {
                capability_id: format!("qualification-{version}"),
                essential: true,
                dependency_ids: BTreeSet::from([
                    format!("metrology-{version}"),
                    format!("reactor-service-{version}"),
                ]),
                evidence_binding: format!("capability:qualification-{version}"),
            }],
            evidence_binding: format!("model:manta-{version}"),
        }
    }

    fn requirement(version: &str, suffix: &str) -> RegenerativeGenomeRequirementV1 {
        RegenerativeGenomeRequirementV1 {
            requirement_id: format!("req-{suffix}"),
            capability_id: format!("qualification-{version}"),
            baseline_dependency_id: format!("{suffix}-{version}"),
            design_binding: format!("design:{suffix}-{version}"),
            metrology_profile_binding: format!("metrology-profile:{suffix}-{version}"),
            requalification_profile_binding: format!("requal:{suffix}-{version}"),
            disassembly_profile_binding: format!("disassembly:{suffix}-{version}"),
            recovery_profile_binding: format!("recovery:{suffix}-{version}"),
            qualified_substitution_bindings: Vec::new(),
        }
    }

    fn genome(version: &str, parent: Option<&str>) -> RegenerativeGenomeV1 {
        RegenerativeGenomeV1 {
            schema_version: REGENERATIVE_GENOME_SCHEMA_V1,
            genome_id: format!("genome-{version}"),
            lineage_parent_binding: parent.map(str::to_owned),
            closure_model_id: format!("manta-{version}"),
            closure_model_evidence_binding: format!("model:manta-{version}"),
            requirements: vec![
                requirement(version, "metrology"),
                requirement(version, "reactor-service"),
            ],
            evidence_binding: format!("genome:manta-{version}"),
        }
    }

    fn evidence() -> RegenerativeEpochHandoffEvidenceV1 {
        RegenerativeEpochHandoffEvidenceV1 {
            schema_version: REGENERATIVE_EPOCH_HANDOFF_SCHEMA_V1,
            handoff_id: "manta-v1-v2".into(),
            source_epoch_id: "epoch-v1".into(),
            source_epoch_evidence_binding: "epoch:manta-v1".into(),
            successor_epoch_id: "epoch-v2".into(),
            successor_epoch_evidence_binding: "epoch:manta-v2".into(),
            source_genome_id: "genome-v1".into(),
            source_genome_evidence_binding: "genome:manta-v1".into(),
            successor_genome_id: "genome-v2".into(),
            successor_genome_evidence_binding: "genome:manta-v2".into(),
            dynamic_handoff_receipt_binding: "symtropy-handoff:receipt-v1-v2".into(),
            transfer_qualifications: vec![
                RegenerativeEpochTransferQualificationV1 {
                    source_dependency_id: "metrology-v1".into(),
                    successor_dependency_id: "metrology-v2".into(),
                    transfer_qualification_binding: "qualification:metrology-v1-v2".into(),
                    safeguarded_continuity_binding: None,
                },
                RegenerativeEpochTransferQualificationV1 {
                    source_dependency_id: "reactor-service-v1".into(),
                    successor_dependency_id: "reactor-service-v2".into(),
                    transfer_qualification_binding: "qualification:reactor-service-v1-v2".into(),
                    safeguarded_continuity_binding: Some(
                        "safeguarded:reactor-service-continuity-v1-v2".into(),
                    ),
                },
            ],
            external_admission_qualifications: vec![
                RegenerativeEpochExternalAdmissionQualificationV1 {
                    successor_dependency_id: "metrology-v2".into(),
                    inventory_evidence_binding: "external:metrology-reference-stock".into(),
                    admission_qualification_binding: "qualification:external-metrology-v2".into(),
                    safeguarded_admission_binding: None,
                },
            ],
            evidence_binding: "epoch-handoff:manta-v1-v2".into(),
        }
    }

    #[test]
    fn exact_lineage_handoff_qualifies_cross_id_and_safeguarded_continuity() {
        let source_model = model("v1");
        let successor_model = model("v2");
        let source_genome = genome("v1", None);
        let successor_genome = genome("v2", Some("genome:manta-v1"));
        let report = qualify_regenerative_epoch_handoff(
            &evidence(),
            &source_genome,
            &source_model,
            &successor_genome,
            &successor_model,
        )
        .unwrap();
        assert_eq!(report.qualified_transfer_count, 2);
        assert_eq!(report.cross_id_transfer_count, 2);
        assert_eq!(report.safeguarded_transfer_count, 1);
        assert_eq!(report.external_admission_count, 1);
        assert_eq!(report.safeguarded_external_admission_count, 0);
    }

    #[test]
    fn safeguarded_transfer_requires_separate_continuity_evidence() {
        let source_model = model("v1");
        let successor_model = model("v2");
        let source_genome = genome("v1", None);
        let successor_genome = genome("v2", Some("genome:manta-v1"));
        let mut claim = evidence();
        claim.transfer_qualifications[1].safeguarded_continuity_binding = None;
        assert_eq!(
            qualify_regenerative_epoch_handoff(
                &claim,
                &source_genome,
                &source_model,
                &successor_genome,
                &successor_model,
            ),
            Err(RegenerativeEpochHandoffError::MissingSafeguardedContinuity {
                dependency_id: "reactor-service-v1".into(),
            })
        );
    }

    #[test]
    fn transfer_cannot_relabel_inventory_across_dependency_kinds() {
        let source_model = model("v1");
        let mut successor_model = model("v2");
        successor_model.dependencies[0].kind = RegenerativeDependencyKind::Component;
        let source_genome = genome("v1", None);
        let successor_genome = genome("v2", Some("genome:manta-v1"));
        assert!(matches!(
            qualify_regenerative_epoch_handoff(
                &evidence(),
                &source_genome,
                &source_model,
                &successor_genome,
                &successor_model,
            ),
            Err(RegenerativeEpochHandoffError::DependencyKindMismatch { .. })
        ));
    }

    #[test]
    fn ordinary_transfer_cannot_carry_safeguarded_continuity_semantics() {
        let source_model = model("v1");
        let successor_model = model("v2");
        let source_genome = genome("v1", None);
        let successor_genome = genome("v2", Some("genome:manta-v1"));
        let mut claim = evidence();
        claim.transfer_qualifications[0].safeguarded_continuity_binding =
            Some("safeguarded:not-applicable".into());
        assert_eq!(
            qualify_regenerative_epoch_handoff(
                &claim,
                &source_genome,
                &source_model,
                &successor_genome,
                &successor_model,
            ),
            Err(RegenerativeEpochHandoffError::UnexpectedSafeguardedContinuity {
                dependency_id: "metrology-v1".into(),
            })
        );
    }

    #[test]
    fn exact_genome_identity_is_bound_to_the_handoff() {
        let source_model = model("v1");
        let successor_model = model("v2");
        let source_genome = genome("v1", None);
        let successor_genome = genome("v2", Some("genome:manta-v1"));
        let mut claim = evidence();
        claim.successor_genome_evidence_binding = "genome:other".into();
        assert_eq!(
            qualify_regenerative_epoch_handoff(
                &claim,
                &source_genome,
                &source_model,
                &successor_genome,
                &successor_model,
            ),
            Err(RegenerativeEpochHandoffError::GenomeIdentityMismatch)
        );
    }
}
