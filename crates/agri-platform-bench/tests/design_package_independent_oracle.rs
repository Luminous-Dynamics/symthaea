// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hand-authored expected outcomes for the open design package contract.
//!
//! This integration test builds its own reference fixture (including an
//! explicit artifact inventory) instead of reusing the production module's
//! fixture builder or required-artifact constant. It checks decision outcomes;
//! it is not a verifier of real CAD bytes, legal license text, or physical safety.

use agri_platform_bench::design_package::{
    qualify_package, ArtifactKind, ArtifactRecord, BomLine, CostEvidenceStatus,
    CriticalInterfaceReview, DesignPackage, PackageReadiness, PartSourceStatus,
    QualificationIssue, RightsReview,
};

fn artifact(kind: ArtifactKind, revision: &str) -> ArtifactRecord {
    ArtifactRecord {
        kind,
        revision: revision.to_owned(),
        source_ref: format!("fixture://golden/rev-{revision}/{kind:?}"),
        sha256: "a".repeat(64),
        license_id: "CERN-OHL-S-2.0".into(),
        rights_review: RightsReview::OpenRightsReviewed {
            reviewer_ref: "fixture://review/rights-001".into(),
            evidence_sha256: "b".repeat(64),
        },
        preferred_editable_source: kind == ArtifactKind::EditableSource,
    }
}

fn part(
    id: &str,
    quantity: u64,
    unit_cost_minor: Option<u64>,
    currency_code: Option<&str>,
) -> BomLine {
    BomLine {
        part_id: id.into(),
        quantity,
        unit: "each".into(),
        material_or_part_spec: format!("declared synthetic spec for {id}"),
        manufacturing_process: "purchase and inspect".into(),
        source_status: PartSourceStatus::VerifiedSource {
            source_ref: format!("fixture://supplier/{id}"),
            lead_time_days: 5,
        },
        unit_cost_minor,
        currency_code: currency_code.map(str::to_owned),
        quote_date: unit_cost_minor.map(|_| "2026-10-01".to_owned()),
    }
}

fn reference_package() -> DesignPackage {
    // Explicit inventory on purpose: do not reuse ArtifactKind::REQUIRED.
    DesignPackage {
        design_id: "golden-manual-implement".into(),
        revision: "A".into(),
        intended_service: "synthetic manual field task".into(),
        target_fabrication_context: "synthetic local fabrication hub".into(),
        operating_domain: vec![
            "hand-operated; low-speed; declared dimensions only".into(),
        ],
        exclusions: vec![
            "no powered operation; no use outside the stated dimensions".into(),
        ],
        substitution_policy: "approve interface, material and fit impacts before substitution".into(),
        critical_interface_review: CriticalInterfaceReview::NotApplicable {
            rationale: "fixture contains no powered or safety-critical interfaces".into(),
        },
        bom: vec![
            part("tube-001", 2, Some(2_500), Some("ZAR")),
            part("fastener-001", 4, Some(250), Some("ZAR")),
        ],
        artifacts: vec![
            artifact(ArtifactKind::EditableSource, "A"),
            artifact(ArtifactKind::NeutralExport, "A"),
            artifact(ArtifactKind::DimensionedDrawings, "A"),
            artifact(ArtifactKind::BillOfMaterials, "A"),
            artifact(ArtifactKind::LicenseNotice, "A"),
            artifact(ArtifactKind::OperatingLimits, "A"),
            artifact(ArtifactKind::AssemblyInstructions, "A"),
            artifact(ArtifactKind::InspectionPlan, "A"),
            artifact(ArtifactKind::TestProtocol, "A"),
            artifact(ArtifactKind::MaintenanceRepairGuide, "A"),
        ],
    }
}

#[test]
fn golden_complete_fixture_reports_review_readiness_not_safety_approval() {
    let report = qualify_package(&reference_package());

    assert_eq!(
        report.readiness,
        PackageReadiness::CompleteForIndependentBuildReview
    );
    assert_eq!(
        report.cost_evidence,
        CostEvidenceStatus::Complete {
            currency_code: "ZAR".into(),
            total_minor: 6_000,
        }
    );
    assert!(!report.safety_certified);
    assert!(!report.operation_authorized);
}

#[test]
fn no_editable_source_is_incomplete_even_when_neutral_export_exists() {
    let mut package = reference_package();
    package.artifacts.retain(|item| item.kind != ArtifactKind::EditableSource);

    let report = qualify_package(&package);
    assert_eq!(report.readiness, PackageReadiness::Incomplete);
    assert!(report.issues.contains(
        &QualificationIssue::MissingRequiredArtifact(ArtifactKind::EditableSource)
    ));
    assert!(report.issues.contains(&QualificationIssue::MissingPreferredEditableSource));
}

#[test]
fn unknown_license_rights_and_missing_review_receipt_are_blockers() {
    let mut package = reference_package();
    package.artifacts[0].rights_review = RightsReview::Unknown;
    assert_eq!(qualify_package(&package).readiness, PackageReadiness::Blocked);

    let mut package = reference_package();
    package.artifacts[0].rights_review = RightsReview::OpenRightsReviewed {
        reviewer_ref: String::new(),
        evidence_sha256: "not-sha256".into(),
    };
    let report = qualify_package(&package);
    assert_eq!(report.readiness, PackageReadiness::Blocked);
    assert!(report.issues.contains(
        &QualificationIssue::InvalidRightsReview(ArtifactKind::EditableSource)
    ));
}

#[test]
fn artifacts_from_another_revision_are_not_implicitly_compatible() {
    let mut package = reference_package();
    package.artifacts[0].revision = "A-older".into();
    assert_eq!(qualify_package(&package).readiness, PackageReadiness::Blocked);
}

#[test]
fn unknown_part_source_blocks_site_specific_build_review() {
    let mut package = reference_package();
    package.bom[0].source_status = PartSourceStatus::Unknown;
    let report = qualify_package(&package);
    assert_eq!(report.readiness, PackageReadiness::Blocked);
    assert!(report.issues.contains(
        &QualificationIssue::UnknownPartAvailability("tube-001".into())
    ));
}

#[test]
fn unavailable_price_evidence_is_not_zero_cost() {
    let mut package = reference_package();
    package.bom[0].unit_cost_minor = None;
    package.bom[0].currency_code = None;
    package.bom[0].quote_date = None;

    let report = qualify_package(&package);
    assert_eq!(
        report.readiness,
        PackageReadiness::CompleteForIndependentBuildReview
    );
    assert_eq!(
        report.cost_evidence,
        CostEvidenceStatus::Incomplete {
            lines: vec!["tube-001".into()]
        }
    );
}

#[test]
fn mixed_currencies_do_not_produce_a_single_total() {
    let mut package = reference_package();
    package.bom[1].currency_code = Some("USD".into());

    assert_eq!(
        qualify_package(&package).cost_evidence,
        CostEvidenceStatus::MixedCurrencies {
            currencies: vec!["USD".into(), "ZAR".into()]
        }
    );
}

#[test]
fn checked_cost_overflow_blocks_the_result_instead_of_wrapping() {
    let mut package = reference_package();
    package.bom[0].quantity = u64::MAX;
    package.bom[0].unit_cost_minor = Some(2);
    package.bom[0].currency_code = Some("ZAR".into());
    package.bom[0].quote_date = Some("2026-10-01".into());

    let report = qualify_package(&package);
    assert_eq!(report.readiness, PackageReadiness::Blocked);
    assert_eq!(report.cost_evidence, CostEvidenceStatus::ArithmeticOverflow);
}

#[test]
fn missing_bom_and_exclusions_are_incomplete_not_successful_defaults() {
    let mut package = reference_package();
    package.bom.clear();
    package.exclusions.clear();

    let report = qualify_package(&package);
    assert_eq!(report.readiness, PackageReadiness::Incomplete);
    assert!(report.issues.contains(&QualificationIssue::EmptyBom));
    assert!(report.issues.contains(&QualificationIssue::MissingExclusions));
}

#[test]
fn pending_critical_interface_review_blocks_the_package() {
    let mut package = reference_package();
    package.critical_interface_review = CriticalInterfaceReview::Pending;
    assert_eq!(qualify_package(&package).readiness, PackageReadiness::Blocked);
}
