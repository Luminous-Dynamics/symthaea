//! Fail-closed intake audit for reference datasets and held-out model validation.
//!
//! This module audits the declared provenance/rights/data contract and a frozen
//! operating-group split. It does not download data, verify legal conclusions,
//! authenticate publishers, prove blinding, run a solver, or validate hydrodynamics.

use std::collections::BTreeSet;
use sha2::{Digest, Sha256};

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
    MissingRawArtifactBytes,
    ArtifactDigestMismatch,
    MissingModelConfigurationBytes,
    ModelConfigurationDigestMismatch,
    SplitManifestDigestMismatch,
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


/// SHA-256 hex of the exact bytes supplied by the caller. The caller is responsible
/// for reading the intended artifact/configuration bytes; this function does not fetch them.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    use std::fmt::Write as _;
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[derive(Default)]
struct CanonicalEncoder(Vec<u8>);

impl CanonicalEncoder {
    fn new(domain: &str) -> Self {
        let mut encoder = Self::default();
        encoder.text("domain", domain);
        encoder
    }

    fn field(&mut self, name: &str, value: &[u8]) {
        // Explicit length-prefixing prevents separator/delimiter ambiguities.
        self.0.extend_from_slice(&(name.len() as u64).to_be_bytes());
        self.0.extend_from_slice(name.as_bytes());
        self.0.extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.0.extend_from_slice(value);
    }

    fn text(&mut self, name: &str, value: &str) {
        self.field(name, value.as_bytes());
    }

    fn boolean(&mut self, name: &str, value: bool) {
        self.field(name, &[u8::from(value)]);
    }

    fn count(&mut self, name: &str, value: usize) {
        self.field(name, &(value as u64).to_be_bytes());
    }
}

fn sorted_normalized_ids(values: &[String]) -> Vec<String> {
    let mut ids: Vec<String> = values.iter().map(|value| value.trim().to_owned()).collect();
    ids.sort();
    ids
}

/// Canonical, versioned SHA-256 of a dataset manifest bound to the exact artifact bytes.
/// Observable contracts and operating-group inventories are treated as sets and sorted;
/// any change to their content changes the digest.
pub fn canonical_dataset_manifest_sha256(
    dataset: &ReferenceDataset,
    raw_artifact_bytes: &[u8],
) -> String {
    let mut encoder = CanonicalEncoder::new("marine-reference-dataset-manifest-v1");
    encoder.text("dataset_id", &dataset.dataset_id);
    encoder.text("source_url", &dataset.source_url);
    encoder.text("citation", &dataset.citation);
    encoder.text("source_revision", &dataset.source_revision);
    encoder.text(
        "availability",
        match dataset.availability {
            ArtifactAvailability::CatalogOnly => "catalog-only",
            ArtifactAvailability::PublicAvailabilityReported => "public-availability-reported",
            ArtifactAvailability::RawBytesInspected => "raw-bytes-inspected",
        },
    );
    encoder.text("raw_artifact_sha256", &sha256_hex(raw_artifact_bytes));
    encoder.text(
        "rights_status",
        match dataset.rights_status {
            RightsStatus::PermittedForDeclaredUse => "permitted-for-declared-use",
            RightsStatus::Restricted => "restricted",
            RightsStatus::Unknown => "unknown",
        },
    );
    encoder.text("rights_reference", &dataset.rights_reference);
    encoder.text(
        "uncertainty_status",
        match dataset.uncertainty_status {
            MeasurementUncertaintyStatus::ReportedAndReviewed => "reported-and-reviewed",
            MeasurementUncertaintyStatus::NotReported => "not-reported",
            MeasurementUncertaintyStatus::NotReviewed => "not-reviewed",
        },
    );

    let mut observables = dataset.observables.clone();
    observables.sort_by(|a, b| {
        a.observable_id
            .cmp(&b.observable_id)
            .then_with(|| a.canonical_unit.cmp(&b.canonical_unit))
            .then_with(|| a.definition.cmp(&b.definition))
            .then_with(|| a.semantics_reviewed.cmp(&b.semantics_reviewed))
    });
    encoder.count("observable_count", observables.len());
    for observable in &observables {
        let mut item = CanonicalEncoder::new("marine-observable-contract-v1");
        item.text("observable_id", &observable.observable_id);
        item.text("canonical_unit", &observable.canonical_unit);
        item.text("definition", &observable.definition);
        item.boolean("semantics_reviewed", observable.semantics_reviewed);
        encoder.field("observable", &item.0);
    }

    let groups = sorted_normalized_ids(&dataset.operating_group_ids);
    encoder.count("operating_group_count", groups.len());
    for group in groups {
        encoder.text("operating_group", &group);
    }
    sha256_hex(&encoder.0)
}

/// Canonical SHA-256 of the exact validation split, bound to the dataset manifest,
/// raw artifact bytes and exact model/configuration bytes. The digest field supplied
/// in the plan is intentionally excluded to avoid self-reference.
pub fn canonical_split_manifest_sha256(
    dataset: &ReferenceDataset,
    raw_artifact_bytes: &[u8],
    plan: &HeldOutValidationPlan,
    model_configuration_bytes: &[u8],
) -> String {
    let mut encoder = CanonicalEncoder::new("marine-heldout-split-manifest-v1");
    encoder.text(
        "dataset_manifest_sha256",
        &canonical_dataset_manifest_sha256(dataset, raw_artifact_bytes),
    );
    encoder.text("dataset_id", &plan.dataset_id);
    encoder.text(
        "model_configuration_sha256",
        &sha256_hex(model_configuration_bytes),
    );
    encoder.boolean("model_frozen_before_holdout_access", plan.model_frozen_before_holdout_access);

    let calibration = sorted_normalized_ids(&plan.calibration_group_ids);
    encoder.count("calibration_group_count", calibration.len());
    for group in calibration {
        encoder.text("calibration_group", &group);
    }
    let holdout = sorted_normalized_ids(&plan.holdout_group_ids);
    encoder.count("holdout_group_count", holdout.len());
    for group in holdout {
        encoder.text("holdout_group", &group);
    }
    sha256_hex(&encoder.0)
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
    raw_artifact_bytes: Option<&[u8]>,
    model_configuration_bytes: Option<&[u8]>,
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
    match raw_artifact_bytes {
        None => issues.push(IntakeIssue::MissingRawArtifactBytes),
        Some(bytes) => {
            let actual_digest = sha256_hex(bytes);
            if !dataset
                .artifact_sha256
                .as_deref()
                .is_some_and(|declared| declared.eq_ignore_ascii_case(&actual_digest))
            {
                issues.push(IntakeIssue::ArtifactDigestMismatch);
            }
        }
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
    match model_configuration_bytes {
        None => issues.push(IntakeIssue::MissingModelConfigurationBytes),
        Some(bytes) => {
            let actual_digest = sha256_hex(bytes);
            if !plan
                .model_configuration_sha256
                .eq_ignore_ascii_case(&actual_digest)
            {
                issues.push(IntakeIssue::ModelConfigurationDigestMismatch);
            }
        }
    }
    if let (Some(raw_bytes), Some(model_bytes)) =
        (raw_artifact_bytes, model_configuration_bytes)
    {
        let expected = canonical_split_manifest_sha256(dataset, raw_bytes, plan, model_bytes);
        if !plan
            .split_manifest_sha256
            .eq_ignore_ascii_case(&expected)
        {
            issues.push(IntakeIssue::SplitManifestDigestMismatch);
        }
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

    const RAW_BYTES: &[u8] = b"synthetic-only reference artifact bytes";
    const MODEL_BYTES: &[u8] = b"synthetic-only frozen model configuration";

    fn audit(dataset: &ReferenceDataset, plan: &HeldOutValidationPlan) -> ReferenceIntakeAudit {
        audit_reference_intake(dataset, plan, Some(RAW_BYTES), Some(MODEL_BYTES))
    }


    fn fixture() -> (ReferenceDataset, HeldOutValidationPlan) {
        let dataset = ReferenceDataset {
            dataset_id: "synthetic-dtc-fixture".to_owned(),
            source_url: "https://example.invalid/synthetic-fixture".to_owned(),
            citation: "Synthetic unit-test fixture; not published or measured data".to_owned(),
            source_revision: "fixture-r1".to_owned(),
            availability: ArtifactAvailability::RawBytesInspected,
            artifact_sha256: Some(sha256_hex(RAW_BYTES)),
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
        let mut plan = HeldOutValidationPlan {
            dataset_id: dataset.dataset_id.clone(),
            split_manifest_sha256: String::new(),
            model_configuration_sha256: sha256_hex(MODEL_BYTES),
            model_frozen_before_holdout_access: true,
            calibration_group_ids: vec![
                "depth-ratio-4.0".to_owned(),
                "depth-ratio-2.0".to_owned(),
            ],
            holdout_group_ids: vec!["depth-ratio-1.2".to_owned()],
        };
        plan.split_manifest_sha256 =
            canonical_split_manifest_sha256(&dataset, RAW_BYTES, &plan, MODEL_BYTES);
        (dataset, plan)
    }

    #[test]
    fn valid_frozen_split_allows_uncertainty_aware_comparison() {
        let (dataset, plan) = fixture();
        let audit = audit(&dataset, &plan);
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
        let audit = audit(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::RawBytesNotInspected));
        assert!(audit.issues.contains(&IntakeIssue::MissingArtifactDigest));
    }

    #[test]
    fn public_availability_claim_does_not_replace_inspected_bytes() {
        let (mut dataset, plan) = fixture();
        dataset.availability = ArtifactAvailability::PublicAvailabilityReported;
        let audit = audit(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
    }

    #[test]
    fn unknown_or_restricted_rights_block_intake() {
        for status in [RightsStatus::Unknown, RightsStatus::Restricted] {
            let (mut dataset, plan) = fixture();
            dataset.rights_status = status;
            let audit = audit(&dataset, &plan);
            assert_eq!(audit.disposition, IntakeDisposition::Blocked);
            assert!(audit.issues.contains(&IntakeIssue::RightsNotPermitted { status }));
        }
    }

    #[test]
    fn malformed_artifact_or_split_digest_blocks_intake() {
        let (mut dataset, mut plan) = fixture();
        dataset.artifact_sha256 = Some("not-a-digest".to_owned());
        plan.split_manifest_sha256 = "also-not-a-digest".to_owned();
        let audit = audit(&dataset, &plan);
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
        let audit = audit(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::CalibrationHoldoutOverlap {
            group_id: "depth-ratio-2.0".to_owned(),
        }));
    }

    #[test]
    fn groups_must_be_declared_and_all_declared_groups_assigned() {
        let (dataset, mut plan) = fixture();
        plan.holdout_group_ids = vec!["unregistered-depth".to_owned()];
        let audit = audit(&dataset, &plan);
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
        let audit = audit(&dataset, &plan);
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
        let audit = audit(&dataset, &plan);
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
        let audit = audit(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::MeasurementUncertaintyNotReviewed));
    }

    #[test]
    fn model_must_be_frozen_before_holdout_access() {
        let (dataset, mut plan) = fixture();
        plan.model_frozen_before_holdout_access = false;
        let audit = audit(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::ModelNotFrozenBeforeHoldoutAccess));
    }

    #[test]
    fn split_must_refer_to_the_same_dataset() {
        let (dataset, mut plan) = fixture();
        plan.dataset_id = "different-dataset".to_owned();
        let audit = audit(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::DatasetPlanMismatch));
    }

    #[test]
    fn reviewed_observable_semantics_are_required() {
        let (mut dataset, plan) = fixture();
        dataset.observables[0].semantics_reviewed = false;
        dataset.observables[1].canonical_unit.clear();
        let audit = audit(&dataset, &plan);
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
        let original = audit(&dataset, &plan);
        dataset.observables.reverse();
        dataset.operating_group_ids.reverse();
        plan.calibration_group_ids.reverse();
        plan.holdout_group_ids.reverse();
        let permuted = audit(&dataset, &plan);
        assert_eq!(original, permuted);
    }

    #[test]
    fn sha256_matches_standard_known_answer() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn mutated_artifact_bytes_are_blocked_even_when_declared_digest_is_stale() {
        let (dataset, plan) = fixture();
        let audit = audit_reference_intake(&dataset, &plan, Some(b"mutated raw artifact"), Some(MODEL_BYTES));
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::ArtifactDigestMismatch));
        assert!(audit.issues.contains(&IntakeIssue::SplitManifestDigestMismatch));
    }

    #[test]
    fn mutated_model_configuration_bytes_are_blocked() {
        let (dataset, plan) = fixture();
        let audit = audit_reference_intake(&dataset, &plan, Some(RAW_BYTES), Some(b"changed config"));
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::ModelConfigurationDigestMismatch));
        assert!(audit.issues.contains(&IntakeIssue::SplitManifestDigestMismatch));
    }

    #[test]
    fn unchanged_split_digest_cannot_survive_a_holdout_group_change() {
        let (dataset, mut plan) = fixture();
        plan.holdout_group_ids = vec!["depth-ratio-2.0".to_owned()];
        let audit = audit(&dataset, &plan);
        assert_eq!(audit.disposition, IntakeDisposition::Blocked);
        assert!(audit.issues.contains(&IntakeIssue::SplitManifestDigestMismatch));
    }

    #[test]
    fn stale_split_digest_blocks_changed_observable_units_or_source_revision() {
        let (mut dataset, plan) = fixture();
        dataset.observables[0].canonical_unit = "kN".to_owned();
        let changed_units = audit(&dataset, &plan);
        assert_eq!(changed_units.disposition, IntakeDisposition::Blocked);
        assert!(changed_units.issues.contains(&IntakeIssue::SplitManifestDigestMismatch));

        let (mut dataset, plan) = fixture();
        dataset.source_revision = "source-revision-r2".to_owned();
        let changed_source = audit(&dataset, &plan);
        assert_eq!(changed_source.disposition, IntakeDisposition::Blocked);
        assert!(changed_source.issues.contains(&IntakeIssue::SplitManifestDigestMismatch));
    }

    #[test]
    fn absent_raw_bytes_or_model_bytes_block_intake() {
        let (dataset, plan) = fixture();
        let no_raw = audit_reference_intake(&dataset, &plan, None, Some(MODEL_BYTES));
        assert_eq!(no_raw.disposition, IntakeDisposition::Blocked);
        assert!(no_raw.issues.contains(&IntakeIssue::MissingRawArtifactBytes));

        let no_model = audit_reference_intake(&dataset, &plan, Some(RAW_BYTES), None);
        assert_eq!(no_model.disposition, IntakeDisposition::Blocked);
        assert!(no_model.issues.contains(&IntakeIssue::MissingModelConfigurationBytes));
    }

}
