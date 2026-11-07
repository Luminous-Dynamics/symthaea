//! Multi-resource mission-endurance screening for marine vessel design.
//!
//! The crate evaluates caller-supplied usable-stock and net-depletion intervals.
//! It does not model water treatment, nutrition, waste processing, batteries,
//! energy, or physical life support. The result is a per-dimension budget
//! disposition, never a scalar "autonomy days" claim.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

impl Interval {
    pub const fn new(lower: f64, upper: f64) -> Self {
        Self { lower, upper }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum EvidenceClass {
    Synthetic,
    Assumed,
    ModelEstimate,
    ExternalSource,
    Measured,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DimensionDisposition {
    SufficientWithinDeclaredBounds,
    InsufficientWithinDeclaredBounds,
    Indeterminate,
    Invalid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverallDisposition {
    AllDimensionsCoverTargetWithinDeclaredBounds,
    AtLeastOneDimensionDoesNotCoverTarget,
    Indeterminate,
    Invalid,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnduranceInterval {
    /// None means no finite lower bound is derived because the maximum
    /// declared depletion rate is zero.
    pub minimum_days: Option<f64>,
    /// None means no finite upper bound is derived because the minimum
    /// declared depletion rate is zero.
    pub maximum_days: Option<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Finding {
    MissingProfileId,
    MissingProfileManifestDigest,
    InvalidProfileManifestDigest,
    MissingConfigurationDigest,
    InvalidConfigurationDigest,
    MissingOperatingCaseId,
    MissingConditionsDigest,
    InvalidConditionsDigest,
    InvalidTargetDays,
    EmptyRequiredDimensionRoster,
    EmptyRequiredDimensionId,
    DuplicateRequiredDimensionId { dimension_id: String },
    MissingRequiredDimension { dimension_id: String },
    MissingDimensionId,
    DuplicateBudgetDimensionId { dimension_id: String },
    UnexpectedDimensionId { dimension_id: String },
    MissingQuantityUnit { dimension_id: String },
    MissingRateUnit { dimension_id: String },
    RateUnitMismatch { dimension_id: String },
    MissingUsableStock { dimension_id: String },
    MissingNetDepletionRate { dimension_id: String },
    MissingEvidenceReference { dimension_id: String },
    MissingEvidenceClass { dimension_id: String },
    MissingConfigurationBinding { dimension_id: String },
    MissingOperatingCaseBinding { dimension_id: String },
    MissingConditionsBinding { dimension_id: String },
    StaleConfigurationBinding { dimension_id: String },
    StaleOperatingCaseBinding { dimension_id: String },
    StaleConditionsBinding { dimension_id: String },
    InvalidNonFinite { dimension_id: String, field: String },
    InvalidNegative { dimension_id: String, field: String },
    InvertedInterval { dimension_id: String, field: String },
    DerivedOverflow { dimension_id: String, field: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutonomyProfile {
    pub profile_id: String,
    /// Caller-supplied SHA-256 identity of the frozen profile manifest.
    pub profile_manifest_digest: String,
    pub target_days: f64,
    /// All dimensions required for this service profile. Order is not meaningful.
    pub required_dimension_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceBudget {
    pub dimension_id: String,
    /// Canonical quantity unit, e.g. "L", "kg", or "g".
    pub quantity_unit: Option<String>,
    /// Must be expressed in quantity_unit per day, e.g. "L/day".
    pub rate_unit: Option<String>,
    /// Remaining usable stock or remaining storage headroom.
    pub usable_stock: Option<Interval>,
    /// Net depletion of stock / consumption of remaining storage headroom.
    /// All process physics and retained residuals are upstream caller-owned inputs.
    pub net_depletion_per_day: Option<Interval>,
    /// Exact subjects to which this row applies.
    pub configuration_digest: Option<String>,
    pub operating_case_id: Option<String>,
    pub conditions_digest: Option<String>,
    pub evidence_ref: Option<String>,
    pub evidence_class: Option<EvidenceClass>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutonomyCase {
    pub profile: AutonomyProfile,
    pub configuration_digest: String,
    pub operating_case_id: String,
    pub conditions_digest: String,
    pub resources: Vec<ResourceBudget>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DimensionAssessment {
    pub dimension_id: String,
    pub evidence_class: Option<EvidenceClass>,
    pub endurance: Option<EnduranceInterval>,
    pub disposition: DimensionDisposition,
    pub findings: Vec<Finding>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutonomyReport {
    pub profile_id: String,
    pub profile_manifest_digest: String,
    pub configuration_digest: String,
    pub operating_case_id: String,
    pub conditions_digest: String,
    pub target_days: f64,
    pub disposition: OverallDisposition,
    /// Individual resource results are retained; there is deliberately no
    /// aggregate "autonomy_days" field.
    pub dimensions: Vec<DimensionAssessment>,
    pub findings: Vec<Finding>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sort_findings(findings: &mut Vec<Finding>) {
    findings.sort();
    findings.dedup();
}

fn validate_interval(
    dimension_id: &str,
    field: &str,
    interval: Interval,
    findings: &mut Vec<Finding>,
    invalid: &mut bool,
) -> bool {
    if !interval.lower.is_finite() {
        findings.push(Finding::InvalidNonFinite {
            dimension_id: dimension_id.to_owned(),
            field: format!("{field}.lower"),
        });
        *invalid = true;
        return false;
    }
    if !interval.upper.is_finite() {
        findings.push(Finding::InvalidNonFinite {
            dimension_id: dimension_id.to_owned(),
            field: format!("{field}.upper"),
        });
        *invalid = true;
        return false;
    }
    if interval.lower < 0.0 || interval.upper < 0.0 {
        findings.push(Finding::InvalidNegative {
            dimension_id: dimension_id.to_owned(),
            field: field.to_owned(),
        });
        *invalid = true;
        return false;
    }
    if interval.lower > interval.upper {
        findings.push(Finding::InvertedInterval {
            dimension_id: dimension_id.to_owned(),
            field: field.to_owned(),
        });
        *invalid = true;
        return false;
    }
    true
}

fn assess_resource(
    resource: &ResourceBudget,
    case: &AutonomyCase,
    duplicate_id: bool,
    expected_ids: &[String],
) -> DimensionAssessment {
    let dimension_id = if nonempty(&resource.dimension_id) {
        resource.dimension_id.clone()
    } else {
        "<missing>".to_owned()
    };
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut indeterminate = false;

    if dimension_id == "<missing>" {
        findings.push(Finding::MissingDimensionId);
        indeterminate = true;
    }
    if duplicate_id {
        findings.push(Finding::DuplicateBudgetDimensionId {
            dimension_id: dimension_id.clone(),
        });
        invalid = true;
    }
    if dimension_id != "<missing>" && !expected_ids.contains(&dimension_id) {
        findings.push(Finding::UnexpectedDimensionId {
            dimension_id: dimension_id.clone(),
        });
        invalid = true;
    }

    let quantity_unit = match resource.quantity_unit.as_deref() {
        Some(unit) if nonempty(unit) => Some(unit),
        _ => {
            findings.push(Finding::MissingQuantityUnit {
                dimension_id: dimension_id.clone(),
            });
            indeterminate = true;
            None
        }
    };
    match resource.rate_unit.as_deref() {
        Some(unit) if nonempty(unit) => {
            if let Some(quantity_unit) = quantity_unit {
                let expected_rate_unit = format!("{quantity_unit}/day");
                if unit != expected_rate_unit.as_str() {
                    findings.push(Finding::RateUnitMismatch {
                        dimension_id: dimension_id.clone(),
                    });
                    invalid = true;
                }
            }
        }
        _ => {
            findings.push(Finding::MissingRateUnit {
                dimension_id: dimension_id.clone(),
            });
            indeterminate = true;
        }
    }

    match resource.configuration_digest.as_deref() {
        Some(value) if nonempty(value) => {
            if value != case.configuration_digest {
                findings.push(Finding::StaleConfigurationBinding {
                    dimension_id: dimension_id.clone(),
                });
                invalid = true;
            }
        }
        _ => {
            findings.push(Finding::MissingConfigurationBinding {
                dimension_id: dimension_id.clone(),
            });
            indeterminate = true;
        }
    }
    match resource.operating_case_id.as_deref() {
        Some(value) if nonempty(value) => {
            if value != case.operating_case_id {
                findings.push(Finding::StaleOperatingCaseBinding {
                    dimension_id: dimension_id.clone(),
                });
                invalid = true;
            }
        }
        _ => {
            findings.push(Finding::MissingOperatingCaseBinding {
                dimension_id: dimension_id.clone(),
            });
            indeterminate = true;
        }
    }
    match resource.conditions_digest.as_deref() {
        Some(value) if nonempty(value) => {
            if value != case.conditions_digest {
                findings.push(Finding::StaleConditionsBinding {
                    dimension_id: dimension_id.clone(),
                });
                invalid = true;
            }
        }
        _ => {
            findings.push(Finding::MissingConditionsBinding {
                dimension_id: dimension_id.clone(),
            });
            indeterminate = true;
        }
    }
    if resource.evidence_ref.as_deref().is_none_or(|value| !nonempty(value)) {
        findings.push(Finding::MissingEvidenceReference {
            dimension_id: dimension_id.clone(),
        });
        indeterminate = true;
    }
    if resource.evidence_class.is_none() {
        findings.push(Finding::MissingEvidenceClass {
            dimension_id: dimension_id.clone(),
        });
        indeterminate = true;
    }

    let stock = match resource.usable_stock {
        Some(interval) => {
            validate_interval(&dimension_id, "usable_stock", interval, &mut findings, &mut invalid);
            Some(interval)
        }
        None => {
            findings.push(Finding::MissingUsableStock {
                dimension_id: dimension_id.clone(),
            });
            indeterminate = true;
            None
        }
    };
    let rate = match resource.net_depletion_per_day {
        Some(interval) => {
            validate_interval(
                &dimension_id,
                "net_depletion_per_day",
                interval,
                &mut findings,
                &mut invalid,
            );
            Some(interval)
        }
        None => {
            findings.push(Finding::MissingNetDepletionRate {
                dimension_id: dimension_id.clone(),
            });
            indeterminate = true;
            None
        }
    };

    let endurance = if !invalid {
        match (stock, rate) {
            (Some(stock), Some(rate)) => {
                let minimum_days = if rate.upper == 0.0 {
                    None
                } else {
                    Some(stock.lower / rate.upper)
                };
                let maximum_days = if rate.lower == 0.0 {
                    None
                } else {
                    Some(stock.upper / rate.lower)
                };
                let overflow = minimum_days.is_some_and(|value| !value.is_finite())
                    || maximum_days.is_some_and(|value| !value.is_finite());
                if overflow {
                    findings.push(Finding::DerivedOverflow {
                        dimension_id: dimension_id.clone(),
                        field: "endurance_days".to_owned(),
                    });
                    invalid = true;
                    None
                } else {
                    Some(EnduranceInterval {
                        minimum_days,
                        maximum_days,
                    })
                }
            }
            _ => None,
        }
    } else {
        None
    };

    let numeric_disposition = endurance.map(|bounds| {
        if bounds.minimum_days.is_none_or(|days| days >= case.profile.target_days) {
            DimensionDisposition::SufficientWithinDeclaredBounds
        } else if bounds.maximum_days.is_some_and(|days| days < case.profile.target_days) {
            DimensionDisposition::InsufficientWithinDeclaredBounds
        } else {
            DimensionDisposition::Indeterminate
        }
    });

    let disposition = if invalid {
        DimensionDisposition::Invalid
    } else if indeterminate {
        DimensionDisposition::Indeterminate
    } else {
        numeric_disposition.unwrap_or(DimensionDisposition::Indeterminate)
    };

    sort_findings(&mut findings);
    DimensionAssessment {
        dimension_id,
        evidence_class: resource.evidence_class,
        endurance,
        disposition,
        findings,
    }
}

/// Validate a resource vector and evaluate each dimension independently.
///
/// Identifiers and SHA-256 strings are checked for format and exact equality.
/// Digests are identity bindings only: this crate does not authenticate sources,
/// recalculate manifests, or prove evidence quality.
pub fn assess_case(case: &AutonomyCase) -> AutonomyReport {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut indeterminate = false;

    if !nonempty(&case.profile.profile_id) {
        findings.push(Finding::MissingProfileId);
        indeterminate = true;
    }
    if !is_sha256(&case.profile.profile_manifest_digest) {
        if nonempty(&case.profile.profile_manifest_digest) {
            findings.push(Finding::InvalidProfileManifestDigest);
            invalid = true;
        } else {
            findings.push(Finding::MissingProfileManifestDigest);
            indeterminate = true;
        }
    }
    if !is_sha256(&case.configuration_digest) {
        if nonempty(&case.configuration_digest) {
            findings.push(Finding::InvalidConfigurationDigest);
            invalid = true;
        } else {
            findings.push(Finding::MissingConfigurationDigest);
            indeterminate = true;
        }
    }
    if !nonempty(&case.operating_case_id) {
        findings.push(Finding::MissingOperatingCaseId);
        indeterminate = true;
    }
    if !is_sha256(&case.conditions_digest) {
        if nonempty(&case.conditions_digest) {
            findings.push(Finding::InvalidConditionsDigest);
            invalid = true;
        } else {
            findings.push(Finding::MissingConditionsDigest);
            indeterminate = true;
        }
    }
    if !case.profile.target_days.is_finite() || case.profile.target_days <= 0.0 {
        findings.push(Finding::InvalidTargetDays);
        invalid = true;
    }
    if case.profile.required_dimension_ids.is_empty() {
        findings.push(Finding::EmptyRequiredDimensionRoster);
        invalid = true;
    }

    let mut required_counts = BTreeMap::<String, usize>::new();
    for id in &case.profile.required_dimension_ids {
        if !nonempty(id) {
            findings.push(Finding::EmptyRequiredDimensionId);
            invalid = true;
        } else {
            *required_counts.entry(id.clone()).or_default() += 1;
        }
    }
    for (dimension_id, count) in &required_counts {
        if *count > 1 {
            findings.push(Finding::DuplicateRequiredDimensionId {
                dimension_id: dimension_id.clone(),
            });
            invalid = true;
        }
    }
    let expected_ids: Vec<String> = required_counts.keys().cloned().collect();

    let mut budget_counts = BTreeMap::<String, usize>::new();
    for resource in &case.resources {
        if nonempty(&resource.dimension_id) {
            *budget_counts.entry(resource.dimension_id.clone()).or_default() += 1;
        }
    }
    for dimension_id in &expected_ids {
        if !budget_counts.contains_key(dimension_id) {
            findings.push(Finding::MissingRequiredDimension {
                dimension_id: dimension_id.clone(),
            });
            indeterminate = true;
        }
    }

    let mut resources = case.resources.iter().collect::<Vec<_>>();
    resources.sort_by_key(|resource| format!("{resource:?}"));

    let mut dimensions = resources
        .into_iter()
        .map(|resource| {
            let duplicate = nonempty(&resource.dimension_id)
                && budget_counts.get(&resource.dimension_id).copied().unwrap_or(0) > 1;
            assess_resource(resource, case, duplicate, &expected_ids)
        })
        .collect::<Vec<_>>();

    for dimension_id in &expected_ids {
        if !budget_counts.contains_key(dimension_id) {
            dimensions.push(DimensionAssessment {
                dimension_id: dimension_id.clone(),
                evidence_class: None,
                endurance: None,
                disposition: DimensionDisposition::Indeterminate,
                findings: vec![Finding::MissingRequiredDimension {
                    dimension_id: dimension_id.clone(),
                }],
            });
        }
    }

    dimensions.sort_by(|left, right| {
        left.dimension_id
            .cmp(&right.dimension_id)
            .then_with(|| format!("{left:?}").cmp(&format!("{right:?}")))
    });
    for dimension in &dimensions {
        findings.extend(dimension.findings.clone());
        match dimension.disposition {
            DimensionDisposition::Invalid => invalid = true,
            DimensionDisposition::Indeterminate => indeterminate = true,
            DimensionDisposition::SufficientWithinDeclaredBounds
            | DimensionDisposition::InsufficientWithinDeclaredBounds => {}
        }
    }
    sort_findings(&mut findings);

    let disposition = if invalid {
        OverallDisposition::Invalid
    } else if dimensions.iter().any(|dimension| {
        dimension.disposition == DimensionDisposition::InsufficientWithinDeclaredBounds
    }) {
        OverallDisposition::AtLeastOneDimensionDoesNotCoverTarget
    } else if indeterminate {
        OverallDisposition::Indeterminate
    } else {
        OverallDisposition::AllDimensionsCoverTargetWithinDeclaredBounds
    };

    AutonomyReport {
        profile_id: case.profile.profile_id.clone(),
        profile_manifest_digest: case.profile.profile_manifest_digest.clone(),
        configuration_digest: case.configuration_digest.clone(),
        operating_case_id: case.operating_case_id.clone(),
        conditions_digest: case.conditions_digest.clone(),
        target_days: case.profile.target_days,
        disposition,
        dimensions,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE_DIGEST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CONFIG_DIGEST: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const CONDITIONS_DIGEST: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn resource(id: &str, stock: Interval, rate: Interval) -> ResourceBudget {
        ResourceBudget {
            dimension_id: id.to_owned(),
            quantity_unit: Some("L".to_owned()),
            rate_unit: Some("L/day".to_owned()),
            usable_stock: Some(stock),
            net_depletion_per_day: Some(rate),
            configuration_digest: Some(CONFIG_DIGEST.to_owned()),
            operating_case_id: Some("offshore-14d-v1".to_owned()),
            conditions_digest: Some(CONDITIONS_DIGEST.to_owned()),
            evidence_ref: Some(format!("fixture:{id}")),
            evidence_class: Some(EvidenceClass::Synthetic),
        }
    }

    fn case(resources: Vec<ResourceBudget>) -> AutonomyCase {
        AutonomyCase {
            profile: AutonomyProfile {
                profile_id: "liveaboard-14d-v1".to_owned(),
                profile_manifest_digest: PROFILE_DIGEST.to_owned(),
                target_days: 14.0,
                required_dimension_ids: vec![
                    "potable_water".to_owned(),
                    "food_protein".to_owned(),
                    "blackwater_headroom".to_owned(),
                ],
            },
            configuration_digest: CONFIG_DIGEST.to_owned(),
            operating_case_id: "offshore-14d-v1".to_owned(),
            conditions_digest: CONDITIONS_DIGEST.to_owned(),
            resources,
        }
    }

    fn adequate_case() -> AutonomyCase {
        case(vec![
            resource("potable_water", Interval::new(300.0, 300.0), Interval::new(10.0, 10.0)),
            resource("food_protein", Interval::new(300.0, 300.0), Interval::new(10.0, 10.0)),
            resource(
                "blackwater_headroom",
                Interval::new(150.0, 150.0),
                Interval::new(10.0, 10.0),
            ),
        ])
    }

    #[test]
    fn adequate_case_covers_every_dimension_without_aggregate_days() {
        let report = assess_case(&adequate_case());
        assert_eq!(
            report.disposition,
            OverallDisposition::AllDimensionsCoverTargetWithinDeclaredBounds
        );
        assert_eq!(report.dimensions.len(), 3);
        assert!(report.dimensions.iter().all(|dimension| {
            dimension.disposition == DimensionDisposition::SufficientWithinDeclaredBounds
        }));
        let water = report.dimensions.iter().find(|dimension| dimension.dimension_id == "potable_water").unwrap();
        assert_eq!(water.endurance.unwrap().minimum_days, Some(30.0));
    }

    #[test]
    fn shortest_resource_bottleneck_is_not_averaged_away() {
        let report = assess_case(&case(vec![
            resource("potable_water", Interval::new(30.0, 30.0), Interval::new(10.0, 10.0)),
            resource("food_protein", Interval::new(300.0, 300.0), Interval::new(10.0, 10.0)),
            resource(
                "blackwater_headroom",
                Interval::new(50.0, 50.0),
                Interval::new(10.0, 10.0),
            ),
        ]));
        assert_eq!(
            report.disposition,
            OverallDisposition::AtLeastOneDimensionDoesNotCoverTarget
        );
        let by_id = report
            .dimensions
            .iter()
            .map(|dimension| (dimension.dimension_id.as_str(), dimension.endurance.unwrap()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(by_id["potable_water"].minimum_days, Some(3.0));
        assert_eq!(by_id["food_protein"].minimum_days, Some(30.0));
        assert_eq!(by_id["blackwater_headroom"].minimum_days, Some(5.0));
    }

    #[test]
    fn one_food_nutrient_can_block_an_otherwise_adequate_food_bundle() {
        let mut test_case = adequate_case();
        test_case.profile.required_dimension_ids.push("food_micronutrients".to_owned());
        test_case.resources.push(resource(
            "food_micronutrients",
            Interval::new(20.0, 20.0),
            Interval::new(2.0, 2.0),
        ));
        let report = assess_case(&test_case);
        assert_eq!(
            report.disposition,
            OverallDisposition::AtLeastOneDimensionDoesNotCoverTarget
        );
        let micro = report
            .dimensions
            .iter()
            .find(|dimension| dimension.dimension_id == "food_micronutrients")
            .unwrap();
        assert_eq!(
            micro.disposition,
            DimensionDisposition::InsufficientWithinDeclaredBounds
        );
    }

    #[test]
    fn omitted_required_dimension_never_passes() {
        let mut test_case = adequate_case();
        test_case.resources.pop();
        let report = assess_case(&test_case);
        assert_eq!(report.disposition, OverallDisposition::Indeterminate);
        assert!(report.findings.iter().any(|finding| matches!(
            finding,
            Finding::MissingRequiredDimension { dimension_id }
                if dimension_id == "blackwater_headroom"
        )));
    }

    #[test]
    fn duplicate_budget_dimension_is_invalid() {
        let mut test_case = adequate_case();
        test_case.resources.push(test_case.resources[0].clone());
        let report = assess_case(&test_case);
        assert_eq!(report.disposition, OverallDisposition::Invalid);
        assert!(report.findings.iter().any(|finding| matches!(
            finding,
            Finding::DuplicateBudgetDimensionId { dimension_id }
                if dimension_id == "potable_water"
        )));
    }

    #[test]
    fn unexpected_dimension_is_invalid() {
        let mut test_case = adequate_case();
        test_case.resources.push(resource(
            "unregistered_bonus_resource",
            Interval::new(100.0, 100.0),
            Interval::new(1.0, 1.0),
        ));
        assert_eq!(assess_case(&test_case).disposition, OverallDisposition::Invalid);
    }

    #[test]
    fn unknown_stock_is_indeterminate_not_zero() {
        let mut test_case = adequate_case();
        test_case.resources[0].usable_stock = None;
        let report = assess_case(&test_case);
        assert_eq!(report.disposition, OverallDisposition::Indeterminate);
        assert!(report.findings.contains(&Finding::MissingUsableStock {
            dimension_id: "potable_water".to_owned()
        }));
    }

    #[test]
    fn zero_rate_lower_and_upper_bounds_are_distinct() {
        let mut test_case = adequate_case();
        test_case.resources[0].net_depletion_per_day = Some(Interval::new(0.0, 0.0));
        test_case.resources[1].net_depletion_per_day = Some(Interval::new(0.0, 2.0));
        let report = assess_case(&test_case);
        let water = report.dimensions.iter().find(|d| d.dimension_id == "potable_water").unwrap();
        let food = report.dimensions.iter().find(|d| d.dimension_id == "food_protein").unwrap();
        assert_eq!(water.endurance.unwrap().minimum_days, None);
        assert_eq!(water.endurance.unwrap().maximum_days, None);
        assert_eq!(food.endurance.unwrap().minimum_days, Some(150.0));
        assert_eq!(food.endurance.unwrap().maximum_days, None);
        assert_eq!(food.disposition, DimensionDisposition::SufficientWithinDeclaredBounds);
    }

    #[test]
    fn crossing_target_interval_is_indeterminate() {
        let mut test_case = adequate_case();
        test_case.resources[0].usable_stock = Some(Interval::new(150.0, 200.0));
        test_case.resources[0].net_depletion_per_day = Some(Interval::new(5.0, 10.0));
        let report = assess_case(&test_case);
        let water = report.dimensions.iter().find(|d| d.dimension_id == "potable_water").unwrap();
        assert_eq!(water.endurance.unwrap().minimum_days, Some(15.0));
        assert_eq!(water.endurance.unwrap().maximum_days, Some(40.0));
        assert_eq!(water.disposition, DimensionDisposition::Indeterminate);
    }

    #[test]
    fn rate_unit_must_match_quantity_unit_per_day() {
        let mut test_case = adequate_case();
        test_case.resources[0].rate_unit = Some("m3/day".to_owned());
        assert_eq!(assess_case(&test_case).disposition, OverallDisposition::Invalid);
    }

    #[test]
    fn stale_condition_binding_is_invalid() {
        let mut test_case = adequate_case();
        test_case.resources[0].conditions_digest = Some("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_owned());
        assert_eq!(assess_case(&test_case).disposition, OverallDisposition::Invalid);
    }

    #[test]
    fn inverted_and_negative_intervals_are_invalid() {
        let mut inverted = adequate_case();
        inverted.resources[0].usable_stock = Some(Interval::new(20.0, 10.0));
        assert_eq!(assess_case(&inverted).disposition, OverallDisposition::Invalid);

        let mut negative = adequate_case();
        negative.resources[0].net_depletion_per_day = Some(Interval::new(-1.0, 1.0));
        assert_eq!(assess_case(&negative).disposition, OverallDisposition::Invalid);
    }

    #[test]
    fn non_finite_values_are_invalid() {
        let mut test_case = adequate_case();
        test_case.resources[0].usable_stock = Some(Interval::new(f64::NAN, 10.0));
        assert_eq!(assess_case(&test_case).disposition, OverallDisposition::Invalid);
    }

    #[test]
    fn invalid_digest_and_non_positive_target_do_not_pass() {
        let mut test_case = adequate_case();
        test_case.profile.profile_manifest_digest = "not-a-digest".to_owned();
        test_case.profile.target_days = 0.0;
        let report = assess_case(&test_case);
        assert_eq!(report.disposition, OverallDisposition::Invalid);
        assert!(report.findings.contains(&Finding::InvalidProfileManifestDigest));
        assert!(report.findings.contains(&Finding::InvalidTargetDays));
    }

    #[test]
    fn assumed_or_synthetic_inputs_keep_their_evidence_class() {
        let mut test_case = adequate_case();
        test_case.resources[0].evidence_class = Some(EvidenceClass::Assumed);
        let report = assess_case(&test_case);
        let water = report.dimensions.iter().find(|d| d.dimension_id == "potable_water").unwrap();
        assert_eq!(water.evidence_class, Some(EvidenceClass::Assumed));
        assert_eq!(report.disposition, OverallDisposition::AllDimensionsCoverTargetWithinDeclaredBounds);
    }

    #[test]
    fn naive_average_can_false_pass_when_two_dimensions_fail() {
        let mut test_case = case(vec![
            resource("potable_water", Interval::new(30.0, 30.0), Interval::new(10.0, 10.0)),
            resource("food_protein", Interval::new(300.0, 300.0), Interval::new(10.0, 10.0)),
            resource(
                "blackwater_headroom",
                Interval::new(50.0, 50.0),
                Interval::new(10.0, 10.0),
            ),
        ]);
        test_case.profile.target_days = 7.0;

        // Independent naive baseline: averaging 3, 30 and 5 days gives
        // 12.67 days and would incorrectly approve a 7-day target.
        let naive_average_days = (3.0_f64 + 30.0 + 5.0) / 3.0;
        assert!(naive_average_days >= test_case.profile.target_days);

        let report = assess_case(&test_case);
        assert_eq!(
            report.disposition,
            OverallDisposition::AtLeastOneDimensionDoesNotCoverTarget
        );
    }

    #[test]
    fn stale_configuration_binding_does_not_inherit_budget() {
        let mut test_case = adequate_case();
        test_case.resources[0].configuration_digest = Some(
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_owned(),
        );
        assert_eq!(assess_case(&test_case).disposition, OverallDisposition::Invalid);
    }

    #[test]
    fn missing_evidence_class_preserves_indeterminate_state() {
        let mut test_case = adequate_case();
        test_case.resources[0].evidence_class = None;
        let report = assess_case(&test_case);
        assert_eq!(report.disposition, OverallDisposition::Indeterminate);
        assert!(report.findings.contains(&Finding::MissingEvidenceClass {
            dimension_id: "potable_water".to_owned()
        }));
    }

    #[test]
    fn input_permutation_produces_identical_report() {
        let forward = adequate_case();
        let mut reverse = adequate_case();
        reverse.resources.reverse();
        reverse.profile.required_dimension_ids.reverse();
        assert_eq!(assess_case(&forward), assess_case(&reverse));
    }
}
