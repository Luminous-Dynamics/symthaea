//! Fail-closed intake audit for reference datasets and held-out model validation.
//!
//! This module audits the declared provenance/rights/data contract and a frozen
//! operating-group split. It does not download data, verify legal conclusions,
//! authenticate publishers, prove blinding, run a solver, or validate hydrodynamics.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArtifactAvailability {
    CatalogOnly,
    PublicAvailabilityReported,
    RawBytesInspected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RightsStatus {
    PermittedForDeclaredUse,
    Restricted,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MeasurementUncertaintyStatus {
    ReportedAndReviewed,
    NotReported,
    NotReviewed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservableContract {
    /// Stable quantity key, e.g. "total-resistance" or "propeller-thrust".
    pub observable_id: String,
    /// Declared canonical unit, e.g. "N" or "N*m"; conversions are not performed here.
    pub canonical_unit: String,
    /// Meaning and boundary of the measured/predicted quantity.
    pub definition: String,
    /// True only after the source's units, sign convention and quantity definition were reviewed.
    pub semantics_reviewed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceDataset {
    pub dataset_id: String,
    pub source_url: String,
    pub citation: String,
    pub source_revision: String,
    pub availability: ArtifactAvailability,
    /// SHA-256 of the exact raw dataset artifact, not of the paper or its landing page.
    pub artifact_sha256: Option<String>,
    pub rights_status: RightsStatus,
    /// Link or durable identifier for the documented rights decision.
    pub rights_reference: String,
    pub observables: Vec<ObservableContract>,
    /// Split unit: whole operating-condition groups, not randomly selected individual rows.
    pub operating_group_ids: Vec<String>,
    pub uncertainty_status: MeasurementUncertaintyStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldOutValidationPlan {
    pub dataset_id: String,
    /// Frozen split-manifest digest. Hash validity is checked; its contents are not parsed here.
    pub split_manifest_sha256: String,
    /// Hash of the exact model, solver and configuration intended for this comparison.
    pub model_configuration_sha256: String,
    /// Explicit declaration that the configuration was frozen before hold-out results were accessed.
    /// This is an attestation, not cryptographic proof of blinding.
    pub model_frozen_before_holdout_access: bool,
    pub calibration_group_ids: Vec<String>,
    pub holdout_group_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntakeDisposition {
    ReadyForUncertaintyAwareComparison,
    ReadyForDescriptiveComparisonOnly,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntakeIssue {
    EmptyField { field: &'static str },
    InvalidSourceUrl,
    InvalidDigest { field: &'static str },
    MissingArtifactDigest,
    RawBytesNotInspected,
    RightsNotPermitted { status: RightsStatus },
    NoObservables,
    DuplicateObservable { observable_id: String },
    InvalidObservable { observable_id: String, field: &'static str },
    ObservableSemanticsNotReviewed { observable_id: String },
    NoOperatingGroups,
    InvalidGroupId { field: &'static str },
    DuplicateGroupId { field: &'static str, group_id: String },
    DuplicateOperatingGroup { group_id: String },
    EmptyCalibrationSplit,
    EmptyHoldoutSplit,
    UnknownCalibrationGroup { group_id: String },
    UnknownHoldoutGroup { group_id: String },
    CalibrationHoldoutOverlap { group_id: String },
    UnassignedOperatingGroup { group_id: String },
    DatasetPlanMismatch,
    ModelNotFrozenBeforeHoldoutAccess,
    MeasurementUncertaintyNotReviewed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceIntakeAudit {
    pub dataset_id: String,
    pub disposition: IntakeDisposition,
    /// Sorted and deduplicated to make the result independent of input-vector order.
    pub issues: Vec<IntakeIssue>,
}

fn is_nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

fn normalized_ids(
    values: &[String],
    field: &'static str,
    issues: &mut Vec<IntakeIssue>,
) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    for value in values {
        let normalized = value.trim().to_owned();
        if normalized.is_empty() {
            issues.push(IntakeIssue::InvalidGroupId { field });
        } else if !ids.insert(normalized.clone()) {
            issues.push(IntakeIssue::DuplicateGroupId {
                field,
                group_id: normalized,
            });
        }
    }
    ids
}

/// Audit one candidate dataset and one frozen group-level calibration/hold-out split.
///
/// A passing disposition only permits a declared comparison to begin. It does not
/// prove source authenticity, licensing, model blinding, numerical correctness or
/// real-world performance. Any missing/invalid identity, rights, artifact, semantic
/// review or split condition blocks both comparison dispositions.
pub fn audit_reference_intake(
    dataset: &ReferenceDataset,
    plan: &HeldOutValidationPlan,
) -> ReferenceIntakeAudit {
    let mut issues = Vec::new();

    for (value, field) in [
        (dataset.dataset_id.as_str(), "dataset_id"),
        (dataset.source_url.as_str(), "source_url"),
        (dataset.citation.as_str(), "citation"),
        (dataset.source_revision.as_str(), "source_revision"),
        (dataset.rights_reference.as_str(), "rights_reference"),
        (plan.dataset_id.as_str(), "plan.dataset_id"),
    ] {
        if !is_nonempty(value) {
            issues.push(IntakeIssue::EmptyField { field });
        }
    }

    if !dataset.source_url.trim().starts_with("https://") {
        issues.push(IntakeIssue::InvalidSourceUrl);
    }
    if dataset.dataset_id != plan.dataset_id {
        issues.push(IntakeIssue::DatasetPlanMismatch);
    }
    if dataset.availability != ArtifactAvailability::RawBytesInspected {
        issues.push(IntakeIssue::RawBytesNotInspected);
    }
    match dataset.artifact_sha256.as_deref() {
        None => issues.push(IntakeIssue::MissingArtifactDigest),
        Some(digest) if !valid_sha256_hex(digest) => {
            issues.push(IntakeIssue::InvalidDigest { field: "artifact_sha256" });
        }
        Some(_) => {}
    }

    if dataset.rights_status != RightsStatus::PermittedForDeclaredUse {
        issues.push(IntakeIssue::RightsNotPermitted {
            status: dataset.rights_status,
        });
    }
    if !valid_sha256_hex(&plan.split_manifest_sha256) {
        issues.push(IntakeIssue::InvalidDigest {
            field: "split_manifest_sha256",
        });
    }
    if !valid_sha256_hex(&plan.model_configuration_sha256) {
        issues.push(IntakeIssue::InvalidDigest {
            field: "model_configuration_sha256",
        });
    }
    if !plan.model_frozen_before_holdout_access {
        issues.push(IntakeIssue::ModelNotFrozenBeforeHoldoutAccess);
    }

    if dataset.observables.is_empty() {
        issues.push(IntakeIssue::NoObservables);
    }
    let mut observable_ids = BTreeSet::new();
    for observable in &dataset.observables {
        let id = observable.observable_id.trim().to_owned();
        if id.is_empty() {
            issues.push(IntakeIssue::InvalidObservable {
                observable_id: id,
                field: "observable_id",
            });
            continue;
        }
        if !observable_ids.insert(id.clone()) {
            issues.push(IntakeIssue::DuplicateObservable { observable_id: id.clone() });
        }
        for (value, field) in [
            (observable.canonical_unit.as_str(), "canonical_unit"),
            (observable.definition.as_str(), "definition"),
        ] {
            if !is_nonempty(value) {
                issues.push(IntakeIssue::InvalidObservable {
                    observable_id: id.clone(),
                    field,
                });
            }
        }
        if !observable.semantics_reviewed {
            issues.push(IntakeIssue::ObservableSemanticsNotReviewed {
                observable_id: id,
            });
        }
    }

    if dataset.operating_group_ids.is_empty() {
        issues.push(IntakeIssue::NoOperatingGroups);
    }
    let declared = normalized_ids(
        &dataset.operating_group_ids,
        "operating_group_ids",
        &mut issues,
    );
    // Use a separate set so duplicate declared operating groups have a domain-specific
    // issue in addition to the generic duplicate-group issue.
    let mut declared_seen = BTreeSet::new();
    for value in &dataset.operating_group_ids {
        let normalized = value.trim().to_owned();
        if !normalized.is_empty() && !declared_seen.insert(normalized.clone()) {
            issues.push(IntakeIssue::DuplicateOperatingGroup {
                group_id: normalized,
            });
        }
    }

    let calibration = normalized_ids(
        &plan.calibration_group_ids,
        "calibration_group_ids",
        &mut issues,
    );
    let holdout = normalized_ids(
        &plan.holdout_group_ids,
        "holdout_group_ids",
        &mut issues,
    );

    if calibration.is_empty() {
        issues.push(IntakeIssue::EmptyCalibrationSplit);
    }
    if holdout.is_empty() {
        issues.push(IntakeIssue::EmptyHoldoutSplit);
    }
    for group_id in &calibration {
        if !declared.contains(group_id) {
            issues.push(IntakeIssue::UnknownCalibrationGroup {
                group_id: group_id.clone(),
            });
        }
    }
    for group_id in &holdout {
        if !declared.contains(group_id) {
            issues.push(IntakeIssue::UnknownHoldoutGroup {
                group_id: group_id.clone(),
            });
        }
    }
    for group_id in calibration.intersection(&holdout) {
        issues.push(IntakeIssue::CalibrationHoldoutOverlap {
            group_id: group_id.clone(),
        });
    }
    let assigned: BTreeSet<String> = calibration.union(&holdout).cloned().collect();
    for group_id in declared.difference(&assigned) {
        issues.push(IntakeIssue::UnassignedOperatingGroup {
            group_id: group_id.clone(),
        });
    }

    if dataset.uncertainty_status == MeasurementUncertaintyStatus::NotReviewed {
        issues.push(IntakeIssue::MeasurementUncertaintyNotReviewed);
    }

    issues.sort();
    issues.dedup();
    let disposition = if !issues.is_empty() {
        IntakeDisposition::Blocked
    } else if dataset.uncertainty_status == MeasurementUncertaintyStatus::ReportedAndReviewed {
        IntakeDisposition::ReadyForUncertaintyAwareComparison
    } else {
        // An explicitly reviewed dataset that does not report measurement uncertainty
        // can support descriptive errors only; it cannot support coverage calibration.
        IntakeDisposition::ReadyForDescriptiveComparisonOnly
    };

    ReferenceIntakeAudit {
        dataset_id: dataset.dataset_id.clone(),
        disposition,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (ReferenceDataset, HeldOutValidationPlan) {
        let dataset = ReferenceDataset {
            dataset_id: "synthetic-dtc-fixture".to_owned(),
            source_url: "https://example.invalid/synthetic-fixture".to_owned(),
            citation: "Synthetic unit-test fixture; not published or measured data".to_owned(),
            source_revision: "fixture-r1".to_owned(),
            availability: ArtifactAvailability::RawBytesInspected,
            artifact_sha256: Some("a".repeat(64)),
            rights_status: RightsStatus::PermittedForDeclaredUse,
            rights_reference: "test-only-rights-fixture".to_owned(),
            observables: vec![
                ObservableContract {
                    observable_id: "resistance".to_owned(),
                    canonical_unit: "N".to_owned(),
                    definition: "Synthetic fixture total resistance".to_owned(),
                    semantics_reviewed: true,
                },
                ObservableContract {
                    observable_id: "propeller-thrust".to_owned(),
                    canonical_unit: "N".to_owned(),
                    definition: "Synthetic fixture propeller thrust".to_owned(),
                    semantics_reviewed: true,
                },
            ],
            operating_group_ids: vec![
                "depth-ratio-4.0".to_owned(),
                "depth-ratio-2.0".to_owned(),
                "depth-ratio-1.2".to_owned(),
            ],
            uncertainty_status: MeasurementUncertaintyStatus::ReportedAndReviewed,
        };
        let plan = HeldOutValidationPlan {
            dataset_id: dataset.dataset_id.clone(),
            split_manifest_sha256: "b".repeat(64),
            model_configuration_sha256: "c".repeat(64),
            model_frozen_before_holdout_access: true,
            calibration_group_ids: vec![
                "depth-ratio-4.0".to_owned(),
                "depth-ratio-2.0".to_owned(),
            ],
            holdout_group_ids: vec!["depth-ratio-1.2".to_owned()],
        };
        (dataset, plan)
    }

    #[test]
    fn valid_frozen_split_allows_uncertainty_aware_comparison() {
        let (dataset, plan) = fixture();
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(
            audit.disposition,
            IntakeDisposition::ReadyForUncertaintyAwareComparison
        );
        assert!(audit.issues.is_empty());
    }

    #[test]
    fn catalog_reference_is_not_raw_measured_evidence() {
        let (mut dataset, plan) = fixture();
        dataset.availability = ArtifactAvailability::CatalogOnly;
        dataset.artifact_sha256 = None;
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::RawBytesNotInspected));
        assert!(audit.issues.contains(&IntakeIssue::MissingArtifactDigest));
    }

    #[test]
    fn public_availability_claim_does_not_replace_inspected_bytes() {
        let (mut dataset, plan) = fixture();
        dataset.availability = ArtifactAvailability::PublicAvailabilityReported;
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
    }

    #[test]
    fn unknown_or_restricted_rights_block_intake() {
        for status in [RightsStatus::Unknown, RightsStatus::Restricted] {
            let (mut dataset, plan) = fixture();
            dataset.rights_status = status;
            let audit = audit_reference_intake(&dataset, &plan);
            assert_eq!(audit.disposition, IntakeDisposition::Blocked);
            assert!(audit.issues.contains(&IntakeIssue::RightsNotPermitted { status }));
        }
    }

    #[test]
    fn malformed_artifact_or_split_digest_blocks_intake() {
        let (mut dataset, mut plan) = fixture();
        dataset.artifact_sha256 = Some("not-a-digest".to_owned());
        plan.split_manifest_sha256 = "also-not-a-digest".to_owned();
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.iter().any(|issue| matches!(
            issue,
            IntakeIssue::InvalidDigest { field: "artifact_sha256" }
        )));
        assert!(audit.issues.iter().any(|issue| matches!(
            issue,
            IntakeIssue::InvalidDigest { field: "split_manifest_sha256" }
        )));
    }

    #[test]
    fn calibration_and_holdout_groups_must_be_disjoint() {
        let (dataset, mut plan) = fixture();
        plan.holdout_group_ids = vec!["depth-ratio-2.0".to_owned()];
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::CalibrationHoldoutOverlap {
            group_id: "depth-ratio-2.0".to_owned(),
        }));
    }

    #[test]
    fn groups_must_be_declared_and_all_declared_groups_assigned() {
        let (dataset, mut plan) = fixture();
        plan.holdout_group_ids = vec!["unregistered-depth".to_owned()];
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::UnknownHoldoutGroup {
            group_id: "unregistered-depth".to_owned(),
        }));
        assert!(audit.issues.iter().any(|issue| matches!(
            issue,
            IntakeIssue::UnassignedOperatingGroup { .. }
        )));
    }

    #[test]
    fn duplicate_normalized_group_and_observable_ids_block_intake() {
        let (mut dataset, mut plan) = fixture();
        dataset.operating_group_ids.push(" depth-ratio-4.0 ".to_owned());
        dataset.observables.push(dataset.observables[0].clone());
        plan.calibration_group_ids[0] = " depth-ratio-4.0 ".to_owned();
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.iter().any(|issue| matches!(
            issue,
            IntakeIssue::DuplicateOperatingGroup { .. }
        )));
        assert!(audit.issues.iter().any(|issue| matches!(
            issue,
            IntakeIssue::DuplicateObservable { .. }
        )));
    }

    #[test]
    fn missing_uncertainty_allows_descriptive_comparison_only() {
        let (mut dataset, plan) = fixture();
        dataset.uncertainty_status = MeasurementUncertaintyStatus::NotReported;
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(
            audit.disposition,
            IntakeDisposition::ReadyForDescriptiveComparisonOnly
        );
        assert!(audit.issues.is_empty());
    }

    #[test]
    fn unreviewed_uncertainty_blocks_all_comparison_claims() {
        let (mut dataset, plan) = fixture();
        dataset.uncertainty_status = MeasurementUncertaintyStatus::NotReviewed;
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::MeasurementUncertaintyNotReviewed));
    }

    #[test]
    fn model_must_be_frozen_before_holdout_access() {
        let (dataset, mut plan) = fixture();
        plan.model_frozen_before_holdout_access = false;
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::ModelNotFrozenBeforeHoldoutAccess));
    }

    #[test]
    fn split_must_refer_to_the_same_dataset() {
        let (dataset, mut plan) = fixture();
        plan.dataset_id = "different-dataset".to_owned();
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::DatasetPlanMismatch));
    }

    #[test]
    fn reviewed_observable_semantics_are_required() {
        let (mut dataset, plan) = fixture();
        dataset.observables[0].semantics_reviewed = false;
        dataset.observables[1].canonical_unit.clear();
        let audit = audit_reference_intake(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.iter().any(|issue| matches!(
            issue,
            IntakeIssue::ObservableSemanticsNotReviewed { observable_id }
                if observable_id == "resistance"
        )));
    }

    #[test]
    fn audit_is_invariant_to_input_vector_permutation() {
        let (mut dataset, mut plan) = fixture();
        let original = audit_reference_intake(&dataset, &plan);
        dataset.observables.reverse();
        dataset.operating_group_ids.reverse();
        plan.calibration_group_ids.reverse();
        plan.holdout_group_ids.reverse();
        let permuted = audit_reference_intake(&dataset, &plan);
        assert_eq!(original, permuted);
    }
}
