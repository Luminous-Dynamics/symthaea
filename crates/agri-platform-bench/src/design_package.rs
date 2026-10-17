// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structural qualification for open agricultural design packages.
//!
//! This module checks package completeness and provenance declarations before
//! independent build review. It does not inspect CAD geometry, parse license
//! text, calculate mechanical/electrical safety, certify manufacturing, or
//! establish agronomic performance. A complete result is NOT a safety approval
//! and does NOT authorize operation of equipment.

use std::collections::BTreeSet;

pub const CONTRACT_VERSION: &str = "OPEN-FARM-DESIGN-PACKAGE-V1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArtifactKind {
    EditableSource,
    NeutralExport,
    DimensionedDrawings,
    BillOfMaterials,
    LicenseNotice,
    OperatingLimits,
    AssemblyInstructions,
    InspectionPlan,
    TestProtocol,
    MaintenanceRepairGuide,
}

impl ArtifactKind {
    pub const REQUIRED: [Self; 10] = [
        Self::EditableSource,
        Self::NeutralExport,
        Self::DimensionedDrawings,
        Self::BillOfMaterials,
        Self::LicenseNotice,
        Self::OperatingLimits,
        Self::AssemblyInstructions,
        Self::InspectionPlan,
        Self::TestProtocol,
        Self::MaintenanceRepairGuide,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RightsReview {
    OpenRightsReviewed,
    Unknown,
    Incompatible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRecord {
    pub kind: ArtifactKind,
    /// Must bind the artifact to exactly the package revision under review.
    pub revision: String,
    /// Source URI or stable evidence locator; synthetic fixtures may use a fixture URI.
    pub source_ref: String,
    /// SHA-256 of referenced bytes (64 hexadecimal chars). This module validates
    /// digest shape only; it does not fetch the artifact or recalculate the hash.
    pub sha256: String,
    /// SPDX identifier or human-reviewed license identifier.
    pub license_id: String,
    /// A separate human review, not a legal opinion from this code.
    pub rights_review: RightsReview,
    /// True only for the preferred editable source format, not an export mesh.
    pub preferred_editable_source: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartSourceStatus {
    VerifiedSource {
        source_ref: String,
        lead_time_days: u32,
    },
    ReviewedAlternative {
        source_ref: String,
        compatibility_review_ref: String,
    },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BomLine {
    pub part_id: String,
    pub quantity: u64,
    pub unit: String,
    pub material_or_part_spec: String,
    pub manufacturing_process: String,
    pub source_status: PartSourceStatus,
    /// Exact minor-currency-unit quote for one BOM unit, if sourced.
    pub unit_cost_minor: Option<u64>,
    pub currency_code: Option<String>,
    /// Quote date in ISO 8601 calendar date form YYYY-MM-DD (shape checked here).
    pub quote_date: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CriticalInterfaceReview {
    Reviewed {
        reviewer_ref: String,
        evidence_sha256: String,
    },
    NotApplicable {
        rationale: String,
    },
    Pending,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesignPackage {
    pub design_id: String,
    pub revision: String,
    pub intended_service: String,
    /// Exact fabrication context this package is being screened for.
    pub target_fabrication_context: String,
    pub operating_domain: Vec<String>,
    pub exclusions: Vec<String>,
    pub substitution_policy: String,
    pub critical_interface_review: CriticalInterfaceReview,
    pub bom: Vec<BomLine>,
    pub artifacts: Vec<ArtifactRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationIssue {
    MissingIdentity(&'static str),
    MissingOperatingDomain,
    MissingExclusions,
    MissingSubstitutionPolicy,
    MissingRequiredArtifact(ArtifactKind),
    MissingPreferredEditableSource,
    EmptyBom,
    InvalidBomLine(String),
    UnknownPartAvailability(String),
    MissingArtifactSource(ArtifactKind),
    MissingLicenseIdentifier(ArtifactKind),
    UnknownArtifactRights(ArtifactKind),
    IncompatibleArtifactRights(ArtifactKind),
    InvalidArtifactDigest(ArtifactKind),
    ArtifactRevisionMismatch {
        kind: ArtifactKind,
        artifact_revision: String,
        package_revision: String,
    },
    CriticalInterfaceReviewPending,
    InvalidCriticalInterfaceReview,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostEvidenceStatus {
    Complete {
        currency_code: String,
        total_minor: u64,
    },
    Incomplete {
        lines: Vec<String>,
    },
    MixedCurrencies {
        currencies: Vec<String>,
    },
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageReadiness {
    CompleteForIndependentBuildReview,
    Incomplete,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationReport {
    pub contract_version: &'static str,
    pub design_id: String,
    pub revision: String,
    pub readiness: PackageReadiness,
    pub issues: Vec<QualificationIssue>,
    pub cost_evidence: CostEvidenceStatus,
    /// Always false: this contract never issues an equipment safety approval.
    pub safety_certified: bool,
    /// Always false: this contract never grants permission to operate equipment.
    pub operation_authorized: bool,
}

pub fn qualify_package(package: &DesignPackage) -> QualificationReport {
    let mut issues = Vec::new();
    let mut blocked = false;

    for (value, field) in [
        (package.design_id.as_str(), "design_id"),
        (package.revision.as_str(), "revision"),
        (package.intended_service.as_str(), "intended_service"),
        (
            package.target_fabrication_context.as_str(),
            "target_fabrication_context",
        ),
    ] {
        if value.trim().is_empty() {
            issues.push(QualificationIssue::MissingIdentity(field));
        }
    }
    if package.operating_domain.iter().all(|item| item.trim().is_empty()) {
        issues.push(QualificationIssue::MissingOperatingDomain);
    }
    if package.exclusions.iter().all(|item| item.trim().is_empty()) {
        issues.push(QualificationIssue::MissingExclusions);
    }
    if package.substitution_policy.trim().is_empty() {
        issues.push(QualificationIssue::MissingSubstitutionPolicy);
    }

    match &package.critical_interface_review {
        CriticalInterfaceReview::Pending => {
            issues.push(QualificationIssue::CriticalInterfaceReviewPending);
            blocked = true;
        }
        CriticalInterfaceReview::Reviewed {
            reviewer_ref,
            evidence_sha256,
        } => {
            if reviewer_ref.trim().is_empty() || !is_sha256(evidence_sha256) {
                issues.push(QualificationIssue::InvalidCriticalInterfaceReview);
                blocked = true;
            }
        }
        CriticalInterfaceReview::NotApplicable { rationale } => {
            if rationale.trim().is_empty() {
                issues.push(QualificationIssue::InvalidCriticalInterfaceReview);
                blocked = true;
            }
        }
    }

    let present: BTreeSet<ArtifactKind> = package.artifacts.iter().map(|a| a.kind).collect();
    for required in ArtifactKind::REQUIRED {
        if !present.contains(&required) {
            issues.push(QualificationIssue::MissingRequiredArtifact(required));
        }
    }
    if !package.artifacts.iter().any(|a| {
        a.kind == ArtifactKind::EditableSource && a.preferred_editable_source
    }) {
        issues.push(QualificationIssue::MissingPreferredEditableSource);
    }

    for artifact in &package.artifacts {
        if artifact.source_ref.trim().is_empty() {
            issues.push(QualificationIssue::MissingArtifactSource(artifact.kind));
            blocked = true;
        }
        if artifact.license_id.trim().is_empty() {
            issues.push(QualificationIssue::MissingLicenseIdentifier(artifact.kind));
            blocked = true;
        }
        if !is_sha256(&artifact.sha256) {
            issues.push(QualificationIssue::InvalidArtifactDigest(artifact.kind));
            blocked = true;
        }
        if artifact.revision != package.revision {
            issues.push(QualificationIssue::ArtifactRevisionMismatch {
                kind: artifact.kind,
                artifact_revision: artifact.revision.clone(),
                package_revision: package.revision.clone(),
            });
            blocked = true;
        }
        match artifact.rights_review {
            RightsReview::OpenRightsReviewed => {}
            RightsReview::Unknown => {
                issues.push(QualificationIssue::UnknownArtifactRights(artifact.kind));
                blocked = true;
            }
            RightsReview::Incompatible => {
                issues.push(QualificationIssue::IncompatibleArtifactRights(artifact.kind));
                blocked = true;
            }
        }
    }

    if package.bom.is_empty() {
        issues.push(QualificationIssue::EmptyBom);
    }

    let mut incomplete_cost_lines = Vec::new();
    let mut currencies = BTreeSet::new();
    let mut quote_total = 0_u64;
    let mut cost_overflow = false;

    for line in &package.bom {
        if line.part_id.trim().is_empty()
            || line.quantity == 0
            || line.unit.trim().is_empty()
            || line.material_or_part_spec.trim().is_empty()
            || line.manufacturing_process.trim().is_empty()
        {
            issues.push(QualificationIssue::InvalidBomLine(line.part_id.clone()));
        }

        match &line.source_status {
            PartSourceStatus::VerifiedSource { source_ref, .. } => {
                if source_ref.trim().is_empty() {
                    issues.push(QualificationIssue::UnknownPartAvailability(
                        line.part_id.clone(),
                    ));
                    blocked = true;
                }
            }
            PartSourceStatus::ReviewedAlternative {
                source_ref,
                compatibility_review_ref,
            } => {
                if source_ref.trim().is_empty() || compatibility_review_ref.trim().is_empty() {
                    issues.push(QualificationIssue::UnknownPartAvailability(
                        line.part_id.clone(),
                    ));
                    blocked = true;
                }
            }
            PartSourceStatus::Unknown => {
                issues.push(QualificationIssue::UnknownPartAvailability(
                    line.part_id.clone(),
                ));
                blocked = true;
            }
        }

        match (
            line.unit_cost_minor,
            line.currency_code.as_deref(),
            line.quote_date.as_deref(),
        ) {
            (Some(unit_cost), Some(currency), Some(date))
                if is_currency_code(currency) && is_iso_date_shape(date) =>
            {
                currencies.insert(currency.to_owned());
                match unit_cost
                    .checked_mul(line.quantity)
                    .and_then(|cost| quote_total.checked_add(cost))
                {
                    Some(total) => quote_total = total,
                    None => cost_overflow = true,
                }
            }
            _ => incomplete_cost_lines.push(line.part_id.clone()),
        }
    }

    let cost_evidence = if cost_overflow {
        CostEvidenceStatus::ArithmeticOverflow
    } else if !incomplete_cost_lines.is_empty() {
        CostEvidenceStatus::Incomplete {
            lines: unique_sorted(incomplete_cost_lines),
        }
    } else if currencies.len() > 1 {
        CostEvidenceStatus::MixedCurrencies {
            currencies: currencies.into_iter().collect(),
        }
    } else if let Some(currency_code) = currencies.into_iter().next() {
        CostEvidenceStatus::Complete {
            currency_code,
            total_minor: quote_total,
        }
    } else {
        CostEvidenceStatus::Incomplete {
            lines: package.bom.iter().map(|line| line.part_id.clone()).collect(),
        }
    };

    // Missing prices do not by themselves make the source package unbuildable;
    // they do prevent a complete-cost claim. Arithmetic overflow invalidates
    // the total and blocks the report.
    if matches!(cost_evidence, CostEvidenceStatus::ArithmeticOverflow) {
        blocked = true;
    }

    let incomplete = issues.iter().any(|issue| {
        matches!(
            issue,
            QualificationIssue::MissingIdentity(_)
                | QualificationIssue::MissingOperatingDomain
                | QualificationIssue::MissingExclusions
                | QualificationIssue::MissingSubstitutionPolicy
                | QualificationIssue::MissingRequiredArtifact(_)
                | QualificationIssue::MissingPreferredEditableSource
                | QualificationIssue::EmptyBom
                | QualificationIssue::InvalidBomLine(_)
        )
    });

    let readiness = if blocked {
        PackageReadiness::Blocked
    } else if incomplete {
        PackageReadiness::Incomplete
    } else {
        PackageReadiness::CompleteForIndependentBuildReview
    };

    QualificationReport {
        contract_version: CONTRACT_VERSION,
        design_id: package.design_id.clone(),
        revision: package.revision.clone(),
        readiness,
        issues,
        cost_evidence,
        safety_certified: false,
        operation_authorized: false,
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn is_currency_code(value: &str) -> bool {
    value.len() == 3 && value.bytes().all(|b| b.is_ascii_uppercase())
}

fn is_iso_date_shape(value: &str) -> bool {
    value.len() == 10
        && value.as_bytes()[4] == b'-'
        && value.as_bytes()[7] == b'-'
        && value
            .bytes()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
}

fn unique_sorted(values: Vec<String>) -> Vec<String> {
    values.into_iter().collect::<BTreeSet<_>>().into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(kind: ArtifactKind, revision: &str) -> ArtifactRecord {
        ArtifactRecord {
            kind,
            revision: revision.to_owned(),
            source_ref: format!("fixture://farm-design/rev-{revision}/{kind:?}"),
            sha256: "a".repeat(64),
            license_id: "CERN-OHL-S-2.0".into(),
            rights_review: RightsReview::OpenRightsReviewed,
            preferred_editable_source: kind == ArtifactKind::EditableSource,
        }
    }

    fn complete_package() -> DesignPackage {
        DesignPackage {
            design_id: "manual-implement-demo".into(),
            revision: "A".into(),
            intended_service: "synthetic manual weeding task".into(),
            target_fabrication_context: "synthetic fabrication hub".into(),
            operating_domain: vec!["hand-operated; low-speed; declared crop/row envelope".into()],
            exclusions: vec!["not for powered operation or use outside declared dimensions".into()],
            substitution_policy: "review fit, material, strength and interface effects before substitution".into(),
            critical_interface_review: CriticalInterfaceReview::NotApplicable {
                rationale: "synthetic fixture: no powered or safety-critical interfaces represented".into(),
            },
            bom: vec![
                BomLine {
                    part_id: "tube-001".into(),
                    quantity: 2,
                    unit: "each".into(),
                    material_or_part_spec: "declared steel tube profile".into(),
                    manufacturing_process: "cut and deburr".into(),
                    source_status: PartSourceStatus::VerifiedSource {
                        source_ref: "fixture://supplier/tube-001".into(),
                        lead_time_days: 5,
                    },
                    unit_cost_minor: Some(2_500),
                    currency_code: Some("ZAR".into()),
                    quote_date: Some("2026-10-01".into()),
                },
                BomLine {
                    part_id: "fastener-001".into(),
                    quantity: 4,
                    unit: "each".into(),
                    material_or_part_spec: "declared fastener size and grade".into(),
                    manufacturing_process: "purchase".into(),
                    source_status: PartSourceStatus::VerifiedSource {
                        source_ref: "fixture://supplier/fastener-001".into(),
                        lead_time_days: 3,
                    },
                    unit_cost_minor: Some(250),
                    currency_code: Some("ZAR".into()),
                    quote_date: Some("2026-10-01".into()),
                },
            ],
            artifacts: ArtifactKind::REQUIRED
                .into_iter()
                .map(|kind| artifact(kind, "A"))
                .collect(),
        }
    }

    #[test]
    fn complete_fixture_is_only_ready_for_independent_build_review() {
        let report = qualify_package(&complete_package());
        assert_eq!(
            report.readiness,
            PackageReadiness::CompleteForIndependentBuildReview
        );
        assert_eq!(
            report.cost_evidence,
            CostEvidenceStatus::Complete {
                currency_code: "ZAR".into(),
                total_minor: 6_000
            }
        );
        assert!(!report.safety_certified);
        assert!(!report.operation_authorized);
    }

    #[test]
    fn missing_editable_source_cannot_be_replaced_by_neutral_export() {
        let mut package = complete_package();
        package.artifacts.retain(|artifact| artifact.kind != ArtifactKind::EditableSource);
        let report = qualify_package(&package);
        assert_eq!(report.readiness, PackageReadiness::Incomplete);
        assert!(report.issues.contains(
            &QualificationIssue::MissingRequiredArtifact(ArtifactKind::EditableSource)
        ));
        assert!(report.issues.contains(&QualificationIssue::MissingPreferredEditableSource));
    }

    #[test]
    fn unknown_or_incompatible_license_rights_fail_closed() {
        let mut unknown = complete_package();
        unknown.artifacts[0].rights_review = RightsReview::Unknown;
        let report = qualify_package(&unknown);
        assert_eq!(report.readiness, PackageReadiness::Blocked);
        assert!(report.issues.contains(
            &QualificationIssue::UnknownArtifactRights(ArtifactKind::EditableSource)
        ));

        let mut incompatible = complete_package();
        incompatible.artifacts[0].rights_review = RightsReview::Incompatible;
        assert_eq!(qualify_package(&incompatible).readiness, PackageReadiness::Blocked);
    }

    #[test]
    fn a_revision_mismatch_blocks_the_package() {
        let mut package = complete_package();
        package.artifacts[0].revision = "older".into();
        let report = qualify_package(&package);
        assert_eq!(report.readiness, PackageReadiness::Blocked);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            QualificationIssue::ArtifactRevisionMismatch { .. }
        )));
    }

    #[test]
    fn unknown_bom_part_availability_blocks_local_build_review() {
        let mut package = complete_package();
        package.bom[0].source_status = PartSourceStatus::Unknown;
        let report = qualify_package(&package);
        assert_eq!(report.readiness, PackageReadiness::Blocked);
        assert!(report.issues.contains(
            &QualificationIssue::UnknownPartAvailability("tube-001".into())
        ));
    }

    #[test]
    fn missing_quote_blocks_complete_cost_claim_but_not_package_review() {
        let mut package = complete_package();
        package.bom[0].unit_cost_minor = None;
        package.bom[0].currency_code = None;
        package.bom[0].quote_date = None;
        let report = qualify_package(&package);
        assert_eq!(
            report.readiness,
            PackageReadiness::CompleteForIndependentBuildReview
        );
        assert!(matches!(report.cost_evidence, CostEvidenceStatus::Incomplete { .. }));
    }

    #[test]
    fn mixed_currency_quotes_never_form_a_single_cost_total() {
        let mut package = complete_package();
        package.bom[1].currency_code = Some("USD".into());
        let report = qualify_package(&package);
        assert!(matches!(report.cost_evidence, CostEvidenceStatus::MixedCurrencies { .. }));
    }

    #[test]
    fn arithmetic_overflow_does_not_wrap_cost_total() {
        let mut package = complete_package();
        package.bom[0].quantity = u64::MAX;
        package.bom[0].unit_cost_minor = Some(2);
        let report = qualify_package(&package);
        assert_eq!(report.readiness, PackageReadiness::Blocked);
        assert_eq!(report.cost_evidence, CostEvidenceStatus::ArithmeticOverflow);
    }

    #[test]
    fn malformed_digest_and_pending_interface_review_block() {
        let mut package = complete_package();
        package.artifacts[0].sha256 = "not-a-digest".into();
        package.critical_interface_review = CriticalInterfaceReview::Pending;
        let report = qualify_package(&package);
        assert_eq!(report.readiness, PackageReadiness::Blocked);
        assert!(report.issues.contains(
            &QualificationIssue::InvalidArtifactDigest(ArtifactKind::EditableSource)
        ));
        assert!(report.issues.contains(&QualificationIssue::CriticalInterfaceReviewPending));
    }

    #[test]
    fn empty_bom_and_undeclared_exclusions_are_not_ready() {
        let mut package = complete_package();
        package.bom.clear();
        package.exclusions.clear();
        let report = qualify_package(&package);
        assert_eq!(report.readiness, PackageReadiness::Incomplete);
        assert!(report.issues.contains(&QualificationIssue::EmptyBom));
        assert!(report.issues.contains(&QualificationIssue::MissingExclusions));
    }
}
