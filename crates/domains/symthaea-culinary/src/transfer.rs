// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Leakage-resistant evaluation contracts for culinary transfer benchmarks.
//!
//! A system that performs well on familiar recipes may fail when cuisine,
//! region, equipment, season, or ingredient families change. This module
//! validates the split manifest before any candidate policy is scored. It
//! does not claim that synthetic fixtures are observed kitchen evidence, and
//! it does not measure taste or food safety.
//!
//! A valid manifest has development, validation, and test cases; explicit
//! held-out axes; unique case and domain identifiers; evidence provenance;
//! and no outcome-record reuse across cases. Every value on a held-out axis
//! must be absent from development and validation data.
//!
//! This is an evaluation-integrity contract, not a recipe optimizer.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CULINARY_TRANSFER_SCHEMA_VERSION: u32 = 1;

/// Role of a case in the frozen evaluation protocol.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferSplit {
    Development,
    Validation,
    Test,
}

impl TransferSplit {
    fn index(self) -> usize {
        match self {
            Self::Development => 0,
            Self::Validation => 1,
            Self::Test => 2,
        }
    }
}

/// Evidence class for the outcome associated with a case.
///
/// Synthetic and simulated cases are useful for testing evaluator mechanics,
/// but are not substitutes for a real kitchen observation or sensory panel.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CulinaryEvidenceClass {
    ObservedKitchenTrial,
    SimulatedOutcome,
    SyntheticFixture,
}

/// Axis that must be absent from development and validation data when it
/// appears in the held-out test set.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferAxis {
    Region,
    Cuisine,
    Setting,
    EquipmentProfile,
    MarketPeriod,
    RecipeFamily,
    IngredientFamily,
}

/// Context profile for a family of culinary evaluation cases.
///
/// Use stable, normalized IDs rather than free-form display labels for split
/// axes. The caller owns normalization and must version that policy with the
/// benchmark manifest.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CulinaryDomainV1 {
    pub domain_id: String,
    pub region: String,
    pub cuisine: String,
    pub setting: Option<String>,
    pub equipment_profile: Option<String>,
    pub market_period: Option<String>,
}

/// One frozen example in a transfer benchmark.
///
/// Outcome record IDs identify the actual response/measurement records used
/// for scoring (for example, one prepared-batch outcome or one blinded panel
/// result), not general reference tables that may legitimately support many
/// cases. A record may not be scored more than once under different cases.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CulinaryTransferCaseV1 {
    pub case_id: String,
    pub split: TransferSplit,
    pub domain_id: String,
    pub recipe_family_id: String,
    pub ingredient_family_ids: Vec<String>,
    pub evidence_class: CulinaryEvidenceClass,
    pub outcome_record_ids: Vec<String>,
    /// Protocol, simulator, or generator identity appropriate to evidence_class.
    pub evidence_method_id: Option<String>,
}

/// Frozen split contract for a culinary transfer benchmark.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CulinaryTransferManifestV1 {
    pub schema_version: u32,
    pub benchmark_id: String,
    pub held_out_axes: Vec<TransferAxis>,
    /// When true, every scored test case must be an observed kitchen trial.
    /// Set false only for explicitly non-empirical evaluator/fixture runs.
    pub require_observed_test_cases: bool,
    pub domains: Vec<CulinaryDomainV1>,
    pub cases: Vec<CulinaryTransferCaseV1>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferViolationCode {
    UnsupportedSchemaVersion,
    MissingRequiredValue,
    DuplicateIdentifier,
    DuplicateHeldOutAxis,
    NoCases,
    MissingSplit,
    UnknownDomain,
    MissingAxisValue,
    HeldOutAxisLeakage,
    MissingOutcomeRecord,
    OutcomeRecordReused,
    MissingEvidenceMethod,
    NonObservedTestCase,
}

/// Machine-readable explanation of a manifest-integrity failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TransferViolation {
    pub code: TransferViolationCode,
    pub case_id: Option<String>,
    pub detail: String,
}

/// Deterministic validation summary. A valid report has no violations.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TransferValidationReport {
    pub benchmark_id: String,
    pub case_count: usize,
    pub development_count: usize,
    pub validation_count: usize,
    pub test_count: usize,
    pub observed_test_count: usize,
    pub synthetic_test_count: usize,
    pub violations: Vec<TransferViolation>,
}

impl TransferValidationReport {
    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }
}

fn is_nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn add_violation(
    violations: &mut Vec<TransferViolation>,
    code: TransferViolationCode,
    case_id: Option<&str>,
    detail: impl Into<String>,
) {
    violations.push(TransferViolation {
        code,
        case_id: case_id.map(str::to_owned),
        detail: detail.into(),
    });
}

fn domain_axis_value<'a>(
    domain: &'a CulinaryDomainV1,
    axis: TransferAxis,
) -> Option<&'a str> {
    match axis {
        TransferAxis::Region => Some(domain.region.as_str()),
        TransferAxis::Cuisine => Some(domain.cuisine.as_str()),
        TransferAxis::Setting => domain.setting.as_deref(),
        TransferAxis::EquipmentProfile => domain.equipment_profile.as_deref(),
        TransferAxis::MarketPeriod => domain.market_period.as_deref(),
        TransferAxis::RecipeFamily | TransferAxis::IngredientFamily => None,
    }
}

fn case_axis_values<'a>(
    case: &'a CulinaryTransferCaseV1,
    domain: &'a CulinaryDomainV1,
    axis: TransferAxis,
) -> Vec<&'a str> {
    match axis {
        TransferAxis::RecipeFamily => vec![case.recipe_family_id.as_str()],
        TransferAxis::IngredientFamily => case
            .ingredient_family_ids
            .iter()
            .map(String::as_str)
            .collect(),
        _ => domain_axis_value(domain, axis).into_iter().collect(),
    }
}

/// Validate that a transfer benchmark's train/validation/test design is
/// internally consistent and resistant to common data-leakage mistakes.
///
/// This validator deliberately does not compute recipe quality, prediction
/// error, sensory preference, nutrition, price, or safety. Those require their
/// own evidence contracts and scorers.
pub fn validate_transfer_manifest(
    manifest: &CulinaryTransferManifestV1,
) -> TransferValidationReport {
    let mut violations = Vec::new();

    if manifest.schema_version != CULINARY_TRANSFER_SCHEMA_VERSION {
        add_violation(
            &mut violations,
            TransferViolationCode::UnsupportedSchemaVersion,
            None,
            format!(
                "schema_version {} is unsupported; expected {}",
                manifest.schema_version, CULINARY_TRANSFER_SCHEMA_VERSION
            ),
        );
    }
    if !is_nonempty(&manifest.benchmark_id) {
        add_violation(
            &mut violations,
            TransferViolationCode::MissingRequiredValue,
            None,
            "benchmark_id must be non-empty",
        );
    }
    if manifest.cases.is_empty() {
        add_violation(
            &mut violations,
            TransferViolationCode::NoCases,
            None,
            "manifest has no cases",
        );
    }
    if manifest.held_out_axes.is_empty() {
        add_violation(
            &mut violations,
            TransferViolationCode::MissingRequiredValue,
            None,
            "at least one held_out_axis is required",
        );
    }

    let mut seen_axes = BTreeSet::new();
    for axis in &manifest.held_out_axes {
        if !seen_axes.insert(*axis) {
            add_violation(
                &mut violations,
                TransferViolationCode::DuplicateHeldOutAxis,
                None,
                format!("held_out_axis {axis:?} is declared more than once"),
            );
        }
    }

    let mut domains: BTreeMap<&str, &CulinaryDomainV1> = BTreeMap::new();
    for domain in &manifest.domains {
        if !is_nonempty(&domain.domain_id) {
            add_violation(
                &mut violations,
                TransferViolationCode::MissingRequiredValue,
                None,
                "domain_id must be non-empty",
            );
        } else if domains.insert(domain.domain_id.as_str(), domain).is_some() {
            add_violation(
                &mut violations,
                TransferViolationCode::DuplicateIdentifier,
                None,
                format!("duplicate domain_id {}", domain.domain_id),
            );
        }
        if !is_nonempty(&domain.region) || !is_nonempty(&domain.cuisine) {
            add_violation(
                &mut violations,
                TransferViolationCode::MissingRequiredValue,
                None,
                format!(
                    "domain {} requires non-empty normalized region and cuisine IDs",
                    domain.domain_id
                ),
            );
        }
    }

    let mut case_ids = BTreeSet::new();
    let mut outcome_record_owner: BTreeMap<&str, (&str, TransferSplit)> = BTreeMap::new();
    let mut split_counts = [0_usize; 3];
    let mut observed_test_count = 0_usize;
    let mut synthetic_test_count = 0_usize;

    for case in &manifest.cases {
        split_counts[case.split.index()] += 1;

        if !is_nonempty(&case.case_id) {
            add_violation(
                &mut violations,
                TransferViolationCode::MissingRequiredValue,
                None,
                "case_id must be non-empty",
            );
        } else if !case_ids.insert(case.case_id.as_str()) {
            add_violation(
                &mut violations,
                TransferViolationCode::DuplicateIdentifier,
                Some(&case.case_id),
                format!("duplicate case_id {}", case.case_id),
            );
        }

        if !is_nonempty(&case.recipe_family_id) {
            add_violation(
                &mut violations,
                TransferViolationCode::MissingRequiredValue,
                Some(&case.case_id),
                "recipe_family_id must be non-empty",
            );
        }

        let domain = match domains.get(case.domain_id.as_str()) {
            Some(domain) => Some(*domain),
            None => {
                add_violation(
                    &mut violations,
                    TransferViolationCode::UnknownDomain,
                    Some(&case.case_id),
                    format!("domain_id {} is not declared", case.domain_id),
                );
                None
            }
        };

        let mut local_records = BTreeSet::new();
        for record_id in &case.outcome_record_ids {
            if !is_nonempty(record_id) {
                add_violation(
                    &mut violations,
                    TransferViolationCode::MissingRequiredValue,
                    Some(&case.case_id),
                    "outcome_record_ids may not contain empty IDs",
                );
                continue;
            }
            if !local_records.insert(record_id.as_str()) {
                add_violation(
                    &mut violations,
                    TransferViolationCode::OutcomeRecordReused,
                    Some(&case.case_id),
                    format!("outcome record {record_id} is repeated within the case"),
                );
                continue;
            }
            if let Some((owner_case, owner_split)) =
                outcome_record_owner.insert(record_id.as_str(), (&case.case_id, case.split))
            {
                add_violation(
                    &mut violations,
                    TransferViolationCode::OutcomeRecordReused,
                    Some(&case.case_id),
                    format!(
                        "outcome record {record_id} is already assigned to case {owner_case} ({owner_split:?})"
                    ),
                );
            }
        }

        match case.evidence_class {
            CulinaryEvidenceClass::ObservedKitchenTrial => {
                if case.outcome_record_ids.is_empty() {
                    add_violation(
                        &mut violations,
                        TransferViolationCode::MissingOutcomeRecord,
                        Some(&case.case_id),
                        "observed kitchen trials require at least one outcome record",
                    );
                }
                if case.split == TransferSplit::Test {
                    observed_test_count += 1;
                }
            }
            CulinaryEvidenceClass::SimulatedOutcome
            | CulinaryEvidenceClass::SyntheticFixture => {
                if case
                    .evidence_method_id
                    .as_deref()
                    .is_none_or(|method| !is_nonempty(method))
                {
                    add_violation(
                        &mut violations,
                        TransferViolationCode::MissingEvidenceMethod,
                        Some(&case.case_id),
                        "simulated and synthetic cases require an explicit model or generator identity",
                    );
                }
                if case.split == TransferSplit::Test {
                    synthetic_test_count += 1;
                }
                if manifest.require_observed_test_cases && case.split == TransferSplit::Test {
                    add_violation(
                        &mut violations,
                        TransferViolationCode::NonObservedTestCase,
                        Some(&case.case_id),
                        "test case is not an observed kitchen trial; it cannot support an empirical transfer claim",
                    );
                }
            }
        }

        if let Some(domain) = domain {
            for axis in &manifest.held_out_axes {
                let values = case_axis_values(case, domain, *axis);
                let mut found_value = false;
                for value in values {
                    if !is_nonempty(value) {
                        continue;
                    }
                    found_value = true;
                }
                if !found_value {
                    add_violation(
                        &mut violations,
                        TransferViolationCode::MissingAxisValue,
                        Some(&case.case_id),
                        format!("case has no usable value for held-out axis {axis:?}"),
                    );
                }
            }
        }

        for family_id in &case.ingredient_family_ids {
            if !is_nonempty(family_id) {
                add_violation(
                    &mut violations,
                    TransferViolationCode::MissingRequiredValue,
                    Some(&case.case_id),
                    "ingredient_family_ids may not contain empty IDs",
                );
            }
        }
    }

    for (index, split) in [
        TransferSplit::Development,
        TransferSplit::Validation,
        TransferSplit::Test,
    ]
    .into_iter()
    .enumerate()
    {
        if split_counts[index] == 0 {
            add_violation(
                &mut violations,
                TransferViolationCode::MissingSplit,
                None,
                format!("split {split:?} has no cases"),
            );
        }
    }

    // Test cases must be outside every selected axis used by development or
    // validation. This is the central transfer/generalization guarantee.
    let reference_cases: Vec<&CulinaryTransferCaseV1> = manifest
        .cases
        .iter()
        .filter(|case| case.split != TransferSplit::Test)
        .collect();
    for axis in &manifest.held_out_axes {
        let mut reference_values = BTreeSet::new();
        for case in &reference_cases {
            if let Some(domain) = domains.get(case.domain_id.as_str()) {
                for value in case_axis_values(case, domain, *axis) {
                    if is_nonempty(value) {
                        reference_values.insert(value.to_owned());
                    }
                }
            }
        }
        for case in manifest
            .cases
            .iter()
            .filter(|case| case.split == TransferSplit::Test)
        {
            if let Some(domain) = domains.get(case.domain_id.as_str()) {
                for value in case_axis_values(case, domain, *axis) {
                    if is_nonempty(value) && reference_values.contains(value) {
                        add_violation(
                            &mut violations,
                            TransferViolationCode::HeldOutAxisLeakage,
                            Some(&case.case_id),
                            format!(
                                "test value {value} on held-out axis {axis:?} also appears in development or validation"
                            ),
                        );
                    }
                }
            }
        }
    }

    // Stable violation order makes serialized reports comparable across runs.
    violations.sort_by(|left, right| {
        (
            left.code,
            left.case_id.as_deref().unwrap_or(""),
            left.detail.as_str(),
        )
            .cmp(&(
                right.code,
                right.case_id.as_deref().unwrap_or(""),
                right.detail.as_str(),
            ))
    });

    TransferValidationReport {
        benchmark_id: manifest.benchmark_id.clone(),
        case_count: manifest.cases.len(),
        development_count: split_counts[TransferSplit::Development.index()],
        validation_count: split_counts[TransferSplit::Validation.index()],
        test_count: split_counts[TransferSplit::Test.index()],
        observed_test_count,
        synthetic_test_count,
        violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain(
        domain_id: &str,
        region: &str,
        cuisine: &str,
        equipment: &str,
    ) -> CulinaryDomainV1 {
        CulinaryDomainV1 {
            domain_id: domain_id.to_owned(),
            region: region.to_owned(),
            cuisine: cuisine.to_owned(),
            setting: Some("household".to_owned()),
            equipment_profile: Some(equipment.to_owned()),
            market_period: Some("2026-q3".to_owned()),
        }
    }

    fn case(
        case_id: &str,
        split: TransferSplit,
        domain_id: &str,
        recipe_family_id: &str,
        ingredient_family_id: &str,
        outcome_record_id: &str,
    ) -> CulinaryTransferCaseV1 {
        CulinaryTransferCaseV1 {
            case_id: case_id.to_owned(),
            split,
            domain_id: domain_id.to_owned(),
            recipe_family_id: recipe_family_id.to_owned(),
            ingredient_family_ids: vec![ingredient_family_id.to_owned()],
            evidence_class: CulinaryEvidenceClass::ObservedKitchenTrial,
            outcome_record_ids: vec![outcome_record_id.to_owned()],
            evidence_method_id: None,
        }
    }

    fn valid_manifest() -> CulinaryTransferManifestV1 {
        CulinaryTransferManifestV1 {
            schema_version: CULINARY_TRANSFER_SCHEMA_VERSION,
            benchmark_id: "culinary-transfer-v1".to_owned(),
            held_out_axes: vec![TransferAxis::Cuisine, TransferAxis::Region],
            require_observed_test_cases: true,
            domains: vec![
                domain("dev", "region-a", "cuisine-a", "oven-a"),
                domain("val", "region-b", "cuisine-b", "oven-a"),
                domain("test", "region-c", "cuisine-c", "stovetop-b"),
            ],
            cases: vec![
                case(
                    "case-dev",
                    TransferSplit::Development,
                    "dev",
                    "family-dev",
                    "grain-a",
                    "outcome-dev",
                ),
                case(
                    "case-val",
                    TransferSplit::Validation,
                    "val",
                    "family-val",
                    "grain-b",
                    "outcome-val",
                ),
                case(
                    "case-test",
                    TransferSplit::Test,
                    "test",
                    "family-test",
                    "grain-c",
                    "outcome-test",
                ),
            ],
        }
    }

    #[test]
    fn disjoint_observed_transfer_manifest_is_accepted() {
        let report = validate_transfer_manifest(&valid_manifest());
        assert!(report.is_valid(), "{:?}", report.violations);
        assert_eq!(report.development_count, 1);
        assert_eq!(report.validation_count, 1);
        assert_eq!(report.test_count, 1);
        assert_eq!(report.observed_test_count, 1);
    }

    #[test]
    fn held_out_cuisine_leakage_is_rejected() {
        let mut manifest = valid_manifest();
        manifest.domains[2].cuisine = "cuisine-a".to_owned();

        let report = validate_transfer_manifest(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code == TransferViolationCode::HeldOutAxisLeakage
                && violation.case_id.as_deref() == Some("case-test")
        }));
    }

    #[test]
    fn outcome_record_reuse_across_splits_is_rejected() {
        let mut manifest = valid_manifest();
        manifest.cases[2].outcome_record_ids = vec!["outcome-dev".to_owned()];

        let report = validate_transfer_manifest(&manifest);
        assert!(report
            .violations
            .iter()
            .any(|violation| violation.code == TransferViolationCode::OutcomeRecordReused));
    }

    #[test]
    fn synthetic_test_data_cannot_masquerade_as_observed_evidence() {
        let mut manifest = valid_manifest();
        manifest.cases[2].evidence_class = CulinaryEvidenceClass::SyntheticFixture;
        manifest.cases[2].evidence_method_id = Some("fixture-generator-v1".to_owned());

        let report = validate_transfer_manifest(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code == TransferViolationCode::NonObservedTestCase
                && violation.case_id.as_deref() == Some("case-test")
        }));
        assert_eq!(report.synthetic_test_count, 1);
    }

    #[test]
    fn duplicate_ids_and_missing_splits_fail_closed() {
        let mut manifest = valid_manifest();
        manifest.cases[1].case_id = "case-dev".to_owned();
        manifest.cases.pop();

        let report = validate_transfer_manifest(&manifest);
        assert!(report
            .violations
            .iter()
            .any(|violation| violation.code == TransferViolationCode::DuplicateIdentifier));
        assert!(report
            .violations
            .iter()
            .any(|violation| violation.code == TransferViolationCode::MissingSplit));
    }

    #[test]
    fn synthetic_fixtures_are_allowed_only_when_empirical_claims_are_not_required() {
        let mut manifest = valid_manifest();
        manifest.require_observed_test_cases = false;
        manifest.cases[2].evidence_class = CulinaryEvidenceClass::SyntheticFixture;
        manifest.cases[2].evidence_method_id = Some("fixture-generator-v1".to_owned());

        let report = validate_transfer_manifest(&manifest);
        assert!(report.is_valid(), "{:?}", report.violations);
        assert_eq!(report.synthetic_test_count, 1);
        assert_eq!(report.observed_test_count, 0);
    }
}
