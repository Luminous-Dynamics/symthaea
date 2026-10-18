//! Independent oracle checks for ICCB's finite named capital-policy catalog.
//!
//! The oracle does not call `CapitalPolicyScenario::evaluate` to calculate any
//! expected value. It independently re-derives policy payoffs and checks the
//! reported catalog, feasibility, tie-break and improvement-vs-one-shot fields.
//! This verifies the current four-template contract, not arbitrary decision trees
//! or real-options value.

use iccb_evaluator::policy::{
    CapitalPolicy, CapitalPolicyScenario, PolicyAssessment, PolicyComparison, PolicyInfeasibility,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyOracleError {
    InvalidScenario,
    ScenarioIdMismatch,
    MissingPolicyAssessment,
    DuplicatePolicyAssessment,
    FeasibilityMismatch,
    CommitmentMismatch,
    ValueMismatch,
    BestPolicyMismatch,
    ImprovementMismatch,
}

const BPS: i128 = 10_000;
const BPS_SQUARED: i128 = 100_000_000;

fn floor_ratio(numerator: i128, positive_denominator: i128) -> i128 {
    let quotient = numerator / positive_denominator;
    let remainder = numerator % positive_denominator;
    if remainder < 0 {
        quotient - 1
    } else {
        quotient
    }
}

fn period_one(amount_minor: i64, discount_factor_bps: u32) -> i128 {
    floor_ratio(
        i128::from(amount_minor) * i128::from(discount_factor_bps),
        BPS,
    )
}

fn period_two(amount_minor: i64, discount_factor_bps: u32) -> i128 {
    let factor = i128::from(discount_factor_bps);
    floor_ratio(i128::from(amount_minor) * factor * factor, BPS_SQUARED)
}

fn policy_commitment(
    scenario: &CapitalPolicyScenario,
    policy: CapitalPolicy,
) -> Option<i64> {
    match policy {
        CapitalPolicy::Abandon => Some(0),
        CapitalPolicy::CommitFullNow | CapitalPolicy::WaitThenInvest => {
            Some(scenario.full_commitment_minor)
        }
        CapitalPolicy::StageThenAdapt => scenario
            .staged_commitment_minor
            .checked_add(scenario.expansion_commitment_minor),
    }
}

fn case_value(
    scenario: &CapitalPolicyScenario,
    policy: CapitalPolicy,
    case_index: usize,
) -> i128 {
    let case = &scenario.demand_cases[case_index];
    let factor = scenario.discount_factor_bps;
    match policy {
        CapitalPolicy::Abandon => 0,
        CapitalPolicy::CommitFullNow => {
            -i128::from(scenario.full_commitment_minor)
                + period_one(case.full_capacity_cashflow_period1_minor, factor)
                + period_two(case.full_capacity_cashflow_period2_minor, factor)
        }
        CapitalPolicy::WaitThenInvest => {
            let continuation = -period_one(scenario.full_commitment_minor, factor)
                + period_two(case.full_capacity_cashflow_period2_minor, factor);
            -i128::from(scenario.wait_cost_minor) + continuation.max(0)
        }
        CapitalPolicy::StageThenAdapt => {
            let continuation = -period_one(scenario.expansion_commitment_minor, factor)
                + period_two(case.expanded_cashflow_period2_minor, factor);
            -i128::from(scenario.staged_commitment_minor)
                + period_one(case.staged_cashflow_period1_minor, factor)
                + continuation.max(0)
        }
    }
}

fn independently_expected_npv(
    scenario: &CapitalPolicyScenario,
    policy: CapitalPolicy,
) -> Option<i64> {
    let mut weighted = 0_i128;
    for index in 0..scenario.demand_cases.len() {
        weighted += i128::from(scenario.demand_cases[index].probability_bps)
            * case_value(scenario, policy, index);
    }
    i64::try_from(floor_ratio(weighted, BPS)).ok()
}

fn input_is_well_formed(scenario: &CapitalPolicyScenario) -> bool {
    if scenario.id.trim().is_empty()
        || scenario.demand_cases.is_empty()
        || scenario.discount_factor_bps > 10_000
        || scenario.capital_budget_minor < 0
        || scenario.full_commitment_minor < 0
        || scenario.staged_commitment_minor < 0
        || scenario.expansion_commitment_minor < 0
        || scenario.wait_cost_minor < 0
    {
        return false;
    }
    if scenario
        .staged_commitment_minor
        .checked_add(scenario.expansion_commitment_minor)
        .is_none()
    {
        return false;
    }
    let mut probability_total = 0_u64;
    for case in &scenario.demand_cases {
        if case.id.trim().is_empty() {
            return false;
        }
        probability_total += u64::from(case.probability_bps);
    }
    probability_total == 10_000
}

fn expected_assessment(
    scenario: &CapitalPolicyScenario,
    policy: CapitalPolicy,
) -> Result<PolicyAssessment, PolicyOracleError> {
    let commitment = policy_commitment(scenario, policy)
        .ok_or(PolicyOracleError::CommitmentMismatch)?;
    if commitment > scenario.capital_budget_minor {
        return Ok(PolicyAssessment {
            policy,
            feasible: false,
            infeasibility: Some(PolicyInfeasibility::CommitmentExceedsBudget),
            maximum_authorized_commitment_minor: commitment,
            expected_npv_minor: None,
        });
    }

    let value = independently_expected_npv(scenario, policy)
        .ok_or(PolicyOracleError::ValueMismatch)?;
    Ok(PolicyAssessment {
        policy,
        feasible: true,
        infeasibility: None,
        maximum_authorized_commitment_minor: commitment,
        expected_npv_minor: Some(value),
    })
}

const POLICY_ORDER: [CapitalPolicy; 4] = [
    CapitalPolicy::Abandon,
    CapitalPolicy::CommitFullNow,
    CapitalPolicy::WaitThenInvest,
    CapitalPolicy::StageThenAdapt,
];

/// Checks a result against independently calculated values for every named policy.
///
/// The input must satisfy the evaluator's finite-state contract. Invalid inputs
/// are not scored as if they were valid benchmark episodes.
pub fn verify_policy_comparison(
    scenario: &CapitalPolicyScenario,
    reported: &PolicyComparison,
) -> Result<(), PolicyOracleError> {
    if !input_is_well_formed(scenario) {
        return Err(PolicyOracleError::InvalidScenario);
    }
    if reported.scenario_id != scenario.id {
        return Err(PolicyOracleError::ScenarioIdMismatch);
    }

    let mut actual = BTreeMap::new();
    for assessment in &reported.assessments {
        if actual.insert(assessment.policy, assessment).is_some() {
            return Err(PolicyOracleError::DuplicatePolicyAssessment);
        }
    }
    if actual.len() != POLICY_ORDER.len()
        || POLICY_ORDER.iter().any(|policy| !actual.contains_key(policy))
    {
        return Err(PolicyOracleError::MissingPolicyAssessment);
    }

    for policy in POLICY_ORDER {
        let expected = expected_assessment(scenario, policy)?;
        let reported_assessment = actual
            .get(&policy)
            .ok_or(PolicyOracleError::MissingPolicyAssessment)?;
        if reported_assessment.feasible != expected.feasible
            || reported_assessment.infeasibility != expected.infeasibility
        {
            return Err(PolicyOracleError::FeasibilityMismatch);
        }
        if reported_assessment.maximum_authorized_commitment_minor
            != expected.maximum_authorized_commitment_minor
        {
            return Err(PolicyOracleError::CommitmentMismatch);
        }
        if reported_assessment.expected_npv_minor != expected.expected_npv_minor {
            return Err(PolicyOracleError::ValueMismatch);
        }
    }

    // The reference enumerates an explicit stable policy order. Strict improvement
    // retains the earliest policy on ties; Abandon therefore wins exact zero ties.
    let mut best_policy = CapitalPolicy::Abandon;
    let mut best_value = 0_i64;
    for policy in POLICY_ORDER {
        let expected = expected_assessment(scenario, policy)?;
        if let Some(value) = expected.expected_npv_minor {
            if value > best_value {
                best_policy = policy;
                best_value = value;
            }
        }
    }
    if reported.best_policy_within_catalog != best_policy {
        return Err(PolicyOracleError::BestPolicyMismatch);
    }

    let one_shot = expected_assessment(scenario, CapitalPolicy::CommitFullNow)?;
    let expected_improvement = one_shot
        .expected_npv_minor
        .map(|value| best_value.checked_sub(value).ok_or(PolicyOracleError::ImprovementMismatch))
        .transpose()?;
    if reported.improvement_vs_one_shot_minor != expected_improvement {
        return Err(PolicyOracleError::ImprovementMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use iccb_evaluator::policy::DemandCase;

    fn scenario() -> CapitalPolicyScenario {
        CapitalPolicyScenario {
            id: "ICCB-POLICY-ORACLE-001".into(),
            capital_budget_minor: 120,
            full_commitment_minor: 100,
            staged_commitment_minor: 15,
            expansion_commitment_minor: 90,
            wait_cost_minor: 70,
            discount_factor_bps: 9_000,
            demand_cases: vec![
                DemandCase {
                    id: "high-demand".into(),
                    probability_bps: 5_000,
                    full_capacity_cashflow_period1_minor: 40,
                    full_capacity_cashflow_period2_minor: 200,
                    staged_cashflow_period1_minor: 30,
                    expanded_cashflow_period2_minor: 180,
                },
                DemandCase {
                    id: "low-demand".into(),
                    probability_bps: 5_000,
                    full_capacity_cashflow_period1_minor: 10,
                    full_capacity_cashflow_period2_minor: 20,
                    staged_cashflow_period1_minor: 10,
                    expanded_cashflow_period2_minor: 20,
                },
            ],
        }
    }

    fn generated_scenario(index: u32) -> CapitalPolicyScenario {
        let low_probability = 1_000 + (index * 137) % 4_000;
        let high_probability = 1_000 + (index * 251) % 3_000;
        let middle_probability = 10_000 - low_probability - high_probability;
        let shift = i64::from(index % 17);
        CapitalPolicyScenario {
            id: format!("ICCB-POLICY-GRID-{index:03}"),
            capital_budget_minor: 80 + i64::from(index % 71),
            full_commitment_minor: 100,
            staged_commitment_minor: 10 + i64::from(index % 16),
            expansion_commitment_minor: 80 + i64::from(index % 31),
            wait_cost_minor: i64::from(index % 41),
            discount_factor_bps: 8_000 + (index * 173) % 2_001,
            demand_cases: vec![
                DemandCase {
                    id: "low".into(),
                    probability_bps: low_probability,
                    full_capacity_cashflow_period1_minor: 5 + shift,
                    full_capacity_cashflow_period2_minor: 10 + shift,
                    staged_cashflow_period1_minor: 4 + shift / 2,
                    expanded_cashflow_period2_minor: 7 + shift,
                },
                DemandCase {
                    id: "middle".into(),
                    probability_bps: middle_probability,
                    full_capacity_cashflow_period1_minor: 25 + shift,
                    full_capacity_cashflow_period2_minor: 70 + 2 * shift,
                    staged_cashflow_period1_minor: 18 + shift,
                    expanded_cashflow_period2_minor: 60 + shift,
                },
                DemandCase {
                    id: "high".into(),
                    probability_bps: high_probability,
                    full_capacity_cashflow_period1_minor: 45 + 2 * shift,
                    full_capacity_cashflow_period2_minor: 130 + 3 * shift,
                    staged_cashflow_period1_minor: 35 + shift,
                    expanded_cashflow_period2_minor: 110 + 2 * shift,
                },
            ],
        }
    }

    #[test]
    fn reference_case_matches_independent_policy_calculation() {
        let input = scenario();
        let comparison = input.evaluate().unwrap();
        verify_policy_comparison(&input, &comparison).unwrap();
    }

    #[test]
    fn oracle_rejects_tampered_policy_value() {
        let input = scenario();
        let mut comparison = input.evaluate().unwrap();
        comparison
            .assessments
            .iter_mut()
            .find(|assessment| assessment.policy == CapitalPolicy::StageThenAdapt)
            .unwrap()
            .expected_npv_minor = Some(123_456);
        assert_eq!(
            verify_policy_comparison(&input, &comparison),
            Err(PolicyOracleError::ValueMismatch)
        );
    }

    #[test]
    fn oracle_rejects_wrong_best_policy() {
        let input = scenario();
        let mut comparison = input.evaluate().unwrap();
        comparison.best_policy_within_catalog = CapitalPolicy::Abandon;
        assert_eq!(
            verify_policy_comparison(&input, &comparison),
            Err(PolicyOracleError::BestPolicyMismatch)
        );
    }

    #[test]
    fn oracle_rejects_missing_duplicate_and_invalid_cases() {
        let input = scenario();
        let mut missing = input.evaluate().unwrap();
        missing.assessments.pop();
        assert_eq!(
            verify_policy_comparison(&input, &missing),
            Err(PolicyOracleError::MissingPolicyAssessment)
        );

        let mut duplicate = input.evaluate().unwrap();
        duplicate.assessments[1].policy = CapitalPolicy::Abandon;
        assert_eq!(
            verify_policy_comparison(&input, &duplicate),
            Err(PolicyOracleError::DuplicatePolicyAssessment)
        );

        let mut invalid = scenario();
        invalid.demand_cases[0].probability_bps = 4_999;
        assert_eq!(
            verify_policy_comparison(&invalid, &input.evaluate().unwrap()),
            Err(PolicyOracleError::InvalidScenario)
        );
    }

    #[test]
    fn deterministic_hundred_scenario_sweep_matches_oracle() {
        for index in 0..100 {
            let input = generated_scenario(index);
            let comparison = input.evaluate().unwrap();
            verify_policy_comparison(&input, &comparison)
                .unwrap_or_else(|error| panic!("grid scenario {index} diverged: {error:?}"));
        }
    }
}
