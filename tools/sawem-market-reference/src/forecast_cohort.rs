//! Forecast-calibration cohort selection audit.
//!
//! Calibration scores are only as representative as the assets admitted to
//! the evaluation cohort. This module makes the target population, every
//! inclusion/exclusion, the rule version and selection time explicit. It
//! prevents post-outcome filtering from being presented as population-level
//! generalization. It does not decide which assets are legally/technically
//! eligible or determine forecast accuracy.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohortObjective {
    FullPopulation,
    MerchantOnly,
    SensitivitySubset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohortDisposition {
    Included,
    Excluded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohortExclusionReason {
    None,
    PreCommissioning,
    IncompleteTelemetry,
    ContractedRevenue,
    PerformanceOutlier,
    OtherDeclaredEligibility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohortSelectionBasis {
    PreRegisteredEligibilityRule,
    OutcomeDependentPostHoc,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CohortPopulationAsset<'a> {
    pub asset_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CohortSelectionRecord<'a> {
    pub asset_id: &'a str,
    pub disposition: CohortDisposition,
    pub exclusion_reason: CohortExclusionReason,
    pub selection_basis: CohortSelectionBasis,
    pub selection_rule_ref: &'a str,
    pub selection_rule_sha256: &'a str,
    pub selected_at_utc: i64,
    pub evidence_ref: &'a str,
    pub evidence_sha256: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForecastCohortAuditInputs<'a> {
    pub cohort_id: &'a str,
    pub objective: CohortObjective,
    pub plan_ref: &'a str,
    pub plan_sha256: &'a str,
    pub plan_registered_at_utc: i64,
    pub evaluation_start_utc: i64,
    pub evaluation_end_utc: i64,
    pub population: &'a [CohortPopulationAsset<'a>],
    pub selections: &'a [CohortSelectionRecord<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastCohortAuditError {
    InvalidEvidenceReference,
    InvalidDigest,
    InvalidEvaluationWindow,
    EmptyPopulation,
    IncompleteCensus,
    DuplicateAssetId,
    DuplicateSelection,
    NonRepresentativeExclusion,
    UnknownAsset,
    ExclusionReasonOnIncludedAsset,
    MissingExclusionReason,
    SelectionAfterEvaluationStart,
    PlanNotRegisteredBeforeEvaluation,
    PostHocOutcomeSelection,
    UnregisteredSelectionRule,
    InvalidMerchantOnlyExclusion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastCohortClaimClass {
    PopulationGeneralization,
    MerchantSubset,
    SensitivityOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForecastCohortAudit {
    pub cohort_id: String,
    pub objective: CohortObjective,
    pub claim_class: ForecastCohortClaimClass,
    pub population_asset_count: u64,
    pub included_asset_count: u64,
    pub excluded_asset_count: u64,
    pub pre_commissioning_excluded_count: u64,
    pub incomplete_telemetry_excluded_count: u64,
    pub contracted_revenue_excluded_count: u64,
    pub performance_outlier_excluded_count: u64,
    pub other_declared_exclusion_count: u64,
    pub population_coverage_bps: u16,
    pub post_hoc_selection_count: u64,
    pub full_population_generalization_claim_allowed: bool,
    pub selection_bias_warning: bool,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Audit one frozen forecast-calibration cohort. An asset population census
/// must be complete and every asset must have exactly one disposition.
/// Outcome-driven exclusions cannot be used to claim population generalization.
/// They may be reported only as sensitivity subsets, with an explicit warning.
pub fn audit_forecast_calibration_cohort(
    inputs: ForecastCohortAuditInputs<'_>,
) -> Result<ForecastCohortAudit, ForecastCohortAuditError> {
    if !nonempty(inputs.cohort_id)
        || !nonempty(inputs.plan_ref)
        || !valid_sha256(inputs.plan_sha256)
    {
        return Err(ForecastCohortAuditError::InvalidEvidenceReference);
    }
    if inputs.evaluation_start_utc >= inputs.evaluation_end_utc {
        return Err(ForecastCohortAuditError::InvalidEvaluationWindow);
    }
    if inputs.population.is_empty() {
        return Err(ForecastCohortAuditError::EmptyPopulation);
    }

    if inputs.objective != CohortObjective::SensitivitySubset
        && inputs.plan_registered_at_utc > inputs.evaluation_start_utc
    {
        return Err(ForecastCohortAuditError::PlanNotRegisteredBeforeEvaluation);
    }

    for (index, asset) in inputs.population.iter().copied().enumerate() {
        if !nonempty(asset.asset_id) || !nonempty(asset.source_ref) {
            return Err(ForecastCohortAuditError::InvalidEvidenceReference);
        }
        if !valid_sha256(asset.source_sha256) {
            return Err(ForecastCohortAuditError::InvalidDigest);
        }
        for previous in &inputs.population[..index] {
            if previous.asset_id == asset.asset_id {
                return Err(ForecastCohortAuditError::DuplicateAssetId);
            }
        }
    }

    for (index, selection) in inputs.selections.iter().copied().enumerate() {
        if !nonempty(selection.asset_id)
            || !nonempty(selection.selection_rule_ref)
            || !nonempty(selection.evidence_ref)
        {
            return Err(ForecastCohortAuditError::InvalidEvidenceReference);
        }
        if !valid_sha256(selection.selection_rule_sha256)
            || !valid_sha256(selection.evidence_sha256)
        {
            return Err(ForecastCohortAuditError::InvalidDigest);
        }
        if !inputs
            .population
            .iter()
            .any(|asset| asset.asset_id == selection.asset_id)
        {
            return Err(ForecastCohortAuditError::UnknownAsset);
        }
        for previous in &inputs.selections[..index] {
            if previous.asset_id == selection.asset_id {
                return Err(ForecastCohortAuditError::DuplicateSelection);
            }
        }
        if selection.disposition == CohortDisposition::Included
            && selection.exclusion_reason != CohortExclusionReason::None
        {
            return Err(ForecastCohortAuditError::ExclusionReasonOnIncludedAsset);
        }
        if selection.disposition == CohortDisposition::Excluded
            && selection.exclusion_reason == CohortExclusionReason::None
        {
            return Err(ForecastCohortAuditError::MissingExclusionReason);
        }
        if inputs.objective != CohortObjective::SensitivitySubset
            && selection.selected_at_utc > inputs.evaluation_start_utc
        {
            return Err(ForecastCohortAuditError::SelectionAfterEvaluationStart);
        }
        if selection.selection_basis == CohortSelectionBasis::OutcomeDependentPostHoc
            && inputs.objective != CohortObjective::SensitivitySubset
        {
            return Err(ForecastCohortAuditError::PostHocOutcomeSelection);
        }
        if selection.selection_basis == CohortSelectionBasis::Unknown {
            return Err(ForecastCohortAuditError::UnregisteredSelectionRule);
        }
        if inputs.objective == CohortObjective::FullPopulation
            && selection.disposition == CohortDisposition::Excluded
            && (selection.exclusion_reason == CohortExclusionReason::ContractedRevenue
                || selection.exclusion_reason == CohortExclusionReason::PerformanceOutlier)
        {
            return Err(ForecastCohortAuditError::NonRepresentativeExclusion);
        }
        if inputs.objective == CohortObjective::MerchantOnly
            && selection.disposition == CohortDisposition::Excluded
            && selection.exclusion_reason != CohortExclusionReason::ContractedRevenue
            && selection.exclusion_reason != CohortExclusionReason::PreCommissioning
            && selection.exclusion_reason != CohortExclusionReason::IncompleteTelemetry
            && selection.exclusion_reason != CohortExclusionReason::OtherDeclaredEligibility
        {
            return Err(ForecastCohortAuditError::InvalidMerchantOnlyExclusion);
        }
    }

    if inputs.selections.len() != inputs.population.len() {
        return Err(ForecastCohortAuditError::IncompleteCensus);
    }

    let mut included = 0_u64;
    let mut excluded = 0_u64;
    let mut pre_commissioning = 0_u64;
    let mut incomplete_telemetry = 0_u64;
    let mut contracted_revenue = 0_u64;
    let mut performance_outlier = 0_u64;
    let mut other = 0_u64;
    let mut post_hoc = 0_u64;

    for selection in inputs.selections {
        if selection.selection_basis == CohortSelectionBasis::OutcomeDependentPostHoc {
            post_hoc = post_hoc.saturating_add(1);
        }
        match selection.disposition {
            CohortDisposition::Included => included = included.saturating_add(1),
            CohortDisposition::Excluded => {
                excluded = excluded.saturating_add(1);
                match selection.exclusion_reason {
                    CohortExclusionReason::None => unreachable!(),
                    CohortExclusionReason::PreCommissioning => {
                        pre_commissioning = pre_commissioning.saturating_add(1)
                    }
                    CohortExclusionReason::IncompleteTelemetry => {
                        incomplete_telemetry = incomplete_telemetry.saturating_add(1)
                    }
                    CohortExclusionReason::ContractedRevenue => {
                        contracted_revenue = contracted_revenue.saturating_add(1)
                    }
                    CohortExclusionReason::PerformanceOutlier => {
                        performance_outlier = performance_outlier.saturating_add(1)
                    }
                    CohortExclusionReason::OtherDeclaredEligibility => {
                        other = other.saturating_add(1)
                    }
                }
            }
        }
    }

    let population_count = u64::try_from(inputs.population.len())
        .map_err(|_| ForecastCohortAuditError::IncompleteCensus)?;
    let coverage = included
        .checked_mul(10_000)
        .ok_or(ForecastCohortAuditError::IncompleteCensus)?
        / population_count;
    let coverage_bps =
        u16::try_from(coverage).map_err(|_| ForecastCohortAuditError::IncompleteCensus)?;

    let claim_class = match inputs.objective {
        CohortObjective::FullPopulation => ForecastCohortClaimClass::PopulationGeneralization,
        CohortObjective::MerchantOnly => ForecastCohortClaimClass::MerchantSubset,
        CohortObjective::SensitivitySubset => ForecastCohortClaimClass::SensitivityOnly,
    };
    let selection_bias_warning = inputs.objective == CohortObjective::SensitivitySubset
        || performance_outlier > 0
        || incomplete_telemetry > 0
        || post_hoc > 0
        || (inputs.objective == CohortObjective::FullPopulation && contracted_revenue > 0);
    let full_population_generalization_claim_allowed = inputs.objective
        == CohortObjective::FullPopulation
        && coverage_bps == 10_000
        && !selection_bias_warning
        && included == population_count;

    Ok(ForecastCohortAudit {
        cohort_id: inputs.cohort_id.to_owned(),
        objective: inputs.objective,
        claim_class,
        population_asset_count: population_count,
        included_asset_count: included,
        excluded_asset_count: excluded,
        pre_commissioning_excluded_count: pre_commissioning,
        incomplete_telemetry_excluded_count: incomplete_telemetry,
        contracted_revenue_excluded_count: contracted_revenue,
        performance_outlier_excluded_count: performance_outlier,
        other_declared_exclusion_count: other,
        population_coverage_bps: coverage_bps,
        post_hoc_selection_count: post_hoc,
        full_population_generalization_claim_allowed,
        selection_bias_warning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const PLAN_SHA: &str =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn population() -> [CohortPopulationAsset<'static>; 3] {
        [
            CohortPopulationAsset {
                asset_id: "battery-a",
                source_ref: "asset-register/a",
                source_sha256: SOURCE_SHA,
            },
            CohortPopulationAsset {
                asset_id: "battery-b",
                source_ref: "asset-register/b",
                source_sha256: SOURCE_SHA,
            },
            CohortPopulationAsset {
                asset_id: "battery-c",
                source_ref: "asset-register/c",
                source_sha256: SOURCE_SHA,
            },
        ]
    }

    fn selection(
        asset_id: &'static str,
        disposition: CohortDisposition,
        reason: CohortExclusionReason,
        basis: CohortSelectionBasis,
    ) -> CohortSelectionRecord<'static> {
        CohortSelectionRecord {
            asset_id,
            disposition,
            exclusion_reason: reason,
            selection_basis: basis,
            selection_rule_ref: "calibration-cohort-rule/v1",
            selection_rule_sha256: PLAN_SHA,
            selected_at_utc: 20,
            evidence_ref: asset_id,
            evidence_sha256: SOURCE_SHA,
        }
    }

    fn inputs<'a>(
        population: &'a [CohortPopulationAsset<'a>],
        selections: &'a [CohortSelectionRecord<'a>],
        objective: CohortObjective,
    ) -> ForecastCohortAuditInputs<'a> {
        ForecastCohortAuditInputs {
            cohort_id: "sawem-bess-calibration-cohort/v1",
            objective,
            plan_ref: "calibration-cohort-rule/v1",
            plan_sha256: PLAN_SHA,
            plan_registered_at_utc: 10,
            evaluation_start_utc: 30,
            evaluation_end_utc: 100,
            population,
            selections,
        }
    }

    #[test]
    fn full_population_census_with_predeclared_eligibility_allows_generalization() {
        let population = population();
        let selections = [
            selection(
                "battery-a",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-b",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-c",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
        ];
        let result = audit_forecast_calibration_cohort(inputs(
            &population,
            &selections,
            CohortObjective::FullPopulation,
        ))
        .unwrap();
        assert_eq!(result.claim_class, ForecastCohortClaimClass::PopulationGeneralization);
        assert_eq!(result.population_coverage_bps, 10_000);
        assert!(result.full_population_generalization_claim_allowed);
        assert!(!result.selection_bias_warning);
    }

    #[test]
    fn incomplete_telemetry_exclusion_discloses_that_full_population_coverage_is_not_met() {
        let population = population();
        let selections = [
            selection(
                "battery-a",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-b",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-c",
                CohortDisposition::Excluded,
                CohortExclusionReason::IncompleteTelemetry,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
        ];
        let result = audit_forecast_calibration_cohort(inputs(
            &population,
            &selections,
            CohortObjective::FullPopulation,
        ))
        .unwrap();
        assert_eq!(result.population_coverage_bps, 6_666);
        assert!(!result.full_population_generalization_claim_allowed);
        assert!(result.selection_bias_warning);
    }

    #[test]
    fn performance_based_post_hoc_exclusion_cannot_support_population_generalization() {
        let population = population();
        let selections = [
            selection(
                "battery-a",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-b",
                CohortDisposition::Excluded,
                CohortExclusionReason::PerformanceOutlier,
                CohortSelectionBasis::OutcomeDependentPostHoc,
            ),
            selection(
                "battery-c",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
        ];
        assert_eq!(
            audit_forecast_calibration_cohort(inputs(
                &population,
                &selections,
                CohortObjective::FullPopulation,
            )),
            Err(ForecastCohortAuditError::PostHocOutcomeSelection)
        );
    }

    #[test]
    fn performance_filtered_sensitivity_subset_is_labeled_non_generalizable() {
        let population = population();
        let selections = [
            selection(
                "battery-a",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-b",
                CohortDisposition::Excluded,
                CohortExclusionReason::PerformanceOutlier,
                CohortSelectionBasis::OutcomeDependentPostHoc,
            ),
            selection(
                "battery-c",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
        ];
        let result = audit_forecast_calibration_cohort(inputs(
            &population,
            &selections,
            CohortObjective::SensitivitySubset,
        ))
        .unwrap();
        assert_eq!(result.claim_class, ForecastCohortClaimClass::SensitivityOnly);
        assert!(result.selection_bias_warning);
        assert!(!result.full_population_generalization_claim_allowed);
        assert_eq!(result.performance_outlier_excluded_count, 1);
    }

    #[test]
    fn merchant_subset_can_exclude_contractual_assets_but_remains_scoped() {
        let population = population();
        let selections = [
            selection(
                "battery-a",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-b",
                CohortDisposition::Excluded,
                CohortExclusionReason::ContractedRevenue,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-c",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
        ];
        let result = audit_forecast_calibration_cohort(inputs(
            &population,
            &selections,
            CohortObjective::MerchantOnly,
        ))
        .unwrap();
        assert_eq!(result.claim_class, ForecastCohortClaimClass::MerchantSubset);
        assert!(!result.full_population_generalization_claim_allowed);
        assert!(!result.selection_bias_warning);
        assert_eq!(result.contracted_revenue_excluded_count, 1);
    }

    #[test]
    fn missing_asset_census_and_duplicate_membership_fail_closed() {
        let population = population();
        let incomplete = [
            selection(
                "battery-a",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-b",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
        ];
        assert_eq!(
            audit_forecast_calibration_cohort(inputs(
                &population,
                &incomplete,
                CohortObjective::FullPopulation,
            )),
            Err(ForecastCohortAuditError::IncompleteCensus)
        );

        let duplicates = [
            selection(
                "battery-a",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-a",
                CohortDisposition::Excluded,
                CohortExclusionReason::PerformanceOutlier,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-c",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
        ];
        assert_eq!(
            audit_forecast_calibration_cohort(inputs(
                &population,
                &duplicates,
                CohortObjective::FullPopulation,
            )),
            Err(ForecastCohortAuditError::DuplicateSelection)
        );
    }

    #[test]
    fn plan_and_selection_must_be_frozen_before_evaluation() {
        let population = population();
        let selections = [
            selection(
                "battery-a",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-b",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-c",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
        ];
        assert_eq!(
            audit_forecast_calibration_cohort(ForecastCohortAuditInputs {
                plan_registered_at_utc: 31,
                ..inputs(&population, &selections, CohortObjective::FullPopulation)
            }),
            Err(ForecastCohortAuditError::PlanNotRegisteredBeforeEvaluation)
        );
        let late_selection = [
            CohortSelectionRecord {
                selected_at_utc: 31,
                ..selections[0]
            },
            selections[1],
            selections[2],
        ];
        assert_eq!(
            audit_forecast_calibration_cohort(inputs(
                &population,
                &late_selection,
                CohortObjective::FullPopulation,
            )),
            Err(ForecastCohortAuditError::SelectionAfterEvaluationStart)
        );
    }

    #[test]
    fn invalid_reasons_ids_and_digests_fail_closed() {
        let population = population();
        let missing_reason = [
            selection(
                "battery-a",
                CohortDisposition::Excluded,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-b",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-c",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
        ];
        assert_eq!(
            audit_forecast_calibration_cohort(inputs(
                &population,
                &missing_reason,
                CohortObjective::FullPopulation,
            )),
            Err(ForecastCohortAuditError::MissingExclusionReason)
        );

        let wrong_id = [
            selection(
                "battery-a",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-b",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-unknown",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
        ];
        assert_eq!(
            audit_forecast_calibration_cohort(inputs(
                &population,
                &wrong_id,
                CohortObjective::FullPopulation,
            )),
            Err(ForecastCohortAuditError::UnknownAsset)
        );

        let bad_digest = [
            CohortSelectionRecord {
                selection_rule_sha256: "bad",
                ..selection(
                    "battery-a",
                    CohortDisposition::Included,
                    CohortExclusionReason::None,
                    CohortSelectionBasis::PreRegisteredEligibilityRule,
                )
            },
            selection(
                "battery-b",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
            selection(
                "battery-c",
                CohortDisposition::Included,
                CohortExclusionReason::None,
                CohortSelectionBasis::PreRegisteredEligibilityRule,
            ),
        ];
        assert_eq!(
            audit_forecast_calibration_cohort(inputs(
                &population,
                &bad_digest,
                CohortObjective::FullPopulation,
            )),
            Err(ForecastCohortAuditError::InvalidDigest)
        );
    }
}
