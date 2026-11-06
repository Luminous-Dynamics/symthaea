//! Matched-information decision regret for scenario-based BESS dispatch.
//!
//! The implementable policy is audited with the non-anticipativity checker.
//! A separate perfect-information (clairvoyant) result is evaluated only as
//! a mathematical upper bound under the same market-rule, asset, topology and
//! settlement basis. The difference is EVPI / expected regret, not forecast
//! model error and not a policy-quality guarantee.
//!
//! Scenario probabilities must be available to the policy by every decision
//! time. Scenario outcomes and both result records must be available by the
//! later evaluation timestamp. This module evaluates submitted outcomes; it is
//! not a stochastic optimizer or a source-authentication mechanism.

use std::collections::BTreeSet;

use crate::non_anticipativity::{
    NonAnticipativityAssessment, NonAnticipativityError, ScenarioDecision,
    audit_non_anticipative_policy,
};

const PROBABILITY_BPS: u128 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionScenarioOutcome<'a> {
    pub scenario_id: &'a str,
    pub scenario_source_ref: &'a str,
    pub scenario_source_sha256: &'a str,
    pub scenario_available_at_utc: i64,
    pub probability_source_ref: &'a str,
    pub probability_source_sha256: &'a str,
    pub probability_available_at_utc: i64,
    pub probability_bps: u16,
    pub policy_result_source_ref: &'a str,
    pub policy_result_sha256: &'a str,
    pub policy_result_available_at_utc: i64,
    pub policy_net_cash_minor_zar: i64,
    pub perfect_information_result_source_ref: &'a str,
    pub perfect_information_result_sha256: &'a str,
    pub perfect_information_result_available_at_utc: i64,
    pub perfect_information_net_cash_minor_zar: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionRegretInputs<'a> {
    pub portfolio_id: &'a str,
    pub evaluation_as_of_utc: i64,
    /// Upper-tail probability for worst-case regret. 500 = worst 5%.
    pub regret_tail_probability_bps: u16,
    pub market_rule_source_ref: &'a str,
    pub market_rule_source_sha256: &'a str,
    pub asset_constraint_source_ref: &'a str,
    pub asset_constraint_source_sha256: &'a str,
    pub network_topology_source_ref: &'a str,
    pub network_topology_source_sha256: &'a str,
    pub settlement_basis_source_ref: &'a str,
    pub settlement_basis_source_sha256: &'a str,
    pub declared_scenario_ids: &'a [&'a str],
    pub decisions: &'a [ScenarioDecision<'a>],
    pub outcomes: &'a [DecisionScenarioOutcome<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionRegretError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidTailProbability,
    EmptyScenarioSet,
    EmptyOutcomeSet,
    ScenarioCensusMismatch,
    DuplicateScenarioId,
    DuplicateOutcomeSourceReference,
    DuplicatePolicyResultSourceReference,
    DuplicateOracleResultSourceReference,
    ProbabilitiesDoNotSumToOne,
    InvalidScenarioProbability,
    FutureKnownEvidence,
    ProbabilityNotAvailableAtDecision,
    NonAnticipativePolicyInvalid,
    NoSharedInformationFrontier,
    PerfectInformationOracleBelowPolicy,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionRegretAssessment<'a> {
    pub portfolio_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub scenario_count: u64,
    pub total_probability_bps: u16,
    pub policy_expected_net_cash_numerator_minor_zar_bps: i128,
    pub perfect_information_expected_net_cash_numerator_minor_zar_bps: i128,
    /// EVPI = expected perfect-information value less implementable policy value.
    pub expected_regret_numerator_minor_zar_bps: i128,
    pub expected_value_of_perfect_information_numerator_minor_zar_bps: i128,
    pub expected_net_cash_denominator_bps: u128,
    /// Worst-tail regret distribution; exact numerator / tail probability.
    pub upper_tail_regret_var_minor_zar: i128,
    pub upper_tail_regret_cvar_numerator_minor_zar_bps: i128,
    pub upper_tail_regret_cvar_denominator_bps: u16,
    pub worst_case_regret_minor_zar: i128,
    /// Share of expected perfect-information value lost to imperfect foresight,
    /// basis points. None when expected oracle cash is not positive.
    pub evpi_as_share_of_oracle_bps: Option<u64>,
    pub non_anticipativity: NonAnticipativityAssessment<'a>,
    /// The oracle is deliberately non-implementable and must never be reported
    /// as the achievable value of the submitted forecast-informed policy.
    pub perfect_information_is_non_implementable_bound: bool,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn add_i128(current: i128, value: i128) -> Result<i128, DecisionRegretError> {
    current
        .checked_add(value)
        .ok_or(DecisionRegretError::ArithmeticOverflow)
}

fn validate_ref_and_hash(reference: &str, digest: &str) -> Result<(), DecisionRegretError> {
    if !nonempty(reference) {
        return Err(DecisionRegretError::InvalidEvidenceReference);
    }
    if !valid_sha256(digest) {
        return Err(DecisionRegretError::InvalidSourceDigest);
    }
    Ok(())
}

/// Compare one implementable scenario policy with a matched perfect-information
/// oracle. The oracle is allowed to see each scenario's realized future, but
/// the supplied actions are independently checked for non-anticipativity.
pub fn evaluate_decision_regret<'a>(
    inputs: DecisionRegretInputs<'a>,
) -> Result<DecisionRegretAssessment<'a>, DecisionRegretError> {
    if !nonempty(inputs.portfolio_id) {
        return Err(DecisionRegretError::InvalidEvidenceReference);
    }
    if inputs.declared_scenario_ids.is_empty() {
        return Err(DecisionRegretError::EmptyScenarioSet);
    }
    if inputs.outcomes.is_empty() {
        return Err(DecisionRegretError::EmptyOutcomeSet);
    }
    if inputs.regret_tail_probability_bps == 0
        || u128::from(inputs.regret_tail_probability_bps) > PROBABILITY_BPS
    {
        return Err(DecisionRegretError::InvalidTailProbability);
    }

    let basis_refs = [
        inputs.market_rule_source_ref,
        inputs.asset_constraint_source_ref,
        inputs.network_topology_source_ref,
        inputs.settlement_basis_source_ref,
    ];
    let basis_hashes = [
        inputs.market_rule_source_sha256,
        inputs.asset_constraint_source_sha256,
        inputs.network_topology_source_sha256,
        inputs.settlement_basis_source_sha256,
    ];
    let mut unique_basis_references = BTreeSet::new();
    for (reference, digest) in basis_refs.into_iter().zip(basis_hashes) {
        validate_ref_and_hash(reference, digest)?;
        if !unique_basis_references.insert(reference) {
            return Err(DecisionRegretError::InvalidEvidenceReference);
        }
    }
    let mut declared = BTreeSet::new();
    for scenario_id in inputs.declared_scenario_ids.iter().copied() {
        if !nonempty(scenario_id) {
            return Err(DecisionRegretError::InvalidEvidenceReference);
        }
        if !declared.insert(scenario_id) {
            return Err(DecisionRegretError::DuplicateScenarioId);
        }
    }
    if inputs.outcomes.len() != declared.len() {
        return Err(DecisionRegretError::ScenarioCensusMismatch);
    }

    let non_anticipativity =
        audit_non_anticipative_policy(inputs.declared_scenario_ids, inputs.decisions).map_err(
            |_: NonAnticipativityError| DecisionRegretError::NonAnticipativePolicyInvalid,
        )?;
    if non_anticipativity.shared_information_set_count == 0 && declared.len() > 1 {
        return Err(DecisionRegretError::NoSharedInformationFrontier);
    }

    let mut outcome_ids = BTreeSet::new();
    let mut scenario_refs = BTreeSet::new();
    let mut policy_refs = BTreeSet::new();
    let mut oracle_refs = BTreeSet::new();
    let mut total_probability = 0_u32;
    let mut policy_expected_numerator = 0_i128;
    let mut oracle_expected_numerator = 0_i128;
    let mut regret_expected_numerator = 0_i128;
    let mut regret_points = Vec::with_capacity(inputs.outcomes.len());
    let mut worst_case_regret = 0_i128;

    for outcome in inputs.outcomes.iter().copied() {
        if !declared.contains(outcome.scenario_id) {
            return Err(DecisionRegretError::ScenarioCensusMismatch);
        }
        if !outcome_ids.insert(outcome.scenario_id) {
            return Err(DecisionRegretError::DuplicateScenarioId);
        }
        validate_ref_and_hash(outcome.scenario_source_ref, outcome.scenario_source_sha256)?;
        validate_ref_and_hash(
            outcome.probability_source_ref,
            outcome.probability_source_sha256,
        )?;
        validate_ref_and_hash(
            outcome.policy_result_source_ref,
            outcome.policy_result_sha256,
        )?;
        validate_ref_and_hash(
            outcome.perfect_information_result_source_ref,
            outcome.perfect_information_result_sha256,
        )?;

        if outcome.scenario_available_at_utc > inputs.evaluation_as_of_utc
            || outcome.probability_available_at_utc > inputs.evaluation_as_of_utc
            || outcome.policy_result_available_at_utc > inputs.evaluation_as_of_utc
            || outcome.perfect_information_result_available_at_utc > inputs.evaluation_as_of_utc
        {
            return Err(DecisionRegretError::FutureKnownEvidence);
        }
        if outcome.probability_bps == 0 {
            return Err(DecisionRegretError::InvalidScenarioProbability);
        }
        if !scenario_refs.insert(outcome.scenario_source_ref) {
            return Err(DecisionRegretError::DuplicateOutcomeSourceReference);
        }
        if !policy_refs.insert(outcome.policy_result_source_ref) {
            return Err(DecisionRegretError::DuplicatePolicyResultSourceReference);
        }
        if !oracle_refs.insert(outcome.perfect_information_result_source_ref) {
            return Err(DecisionRegretError::DuplicateOracleResultSourceReference);
        }

        let scenario_decisions = inputs
            .decisions
            .iter()
            .filter(|decision| decision.scenario_id == outcome.scenario_id);
        if scenario_decisions
            .into_iter()
            .any(|decision| outcome.probability_available_at_utc > decision.decision_time_utc)
        {
            return Err(DecisionRegretError::ProbabilityNotAvailableAtDecision);
        }

        if outcome.perfect_information_net_cash_minor_zar < outcome.policy_net_cash_minor_zar {
            return Err(DecisionRegretError::PerfectInformationOracleBelowPolicy);
        }

        total_probability = total_probability
            .checked_add(u32::from(outcome.probability_bps))
            .ok_or(DecisionRegretError::ArithmeticOverflow)?;
        let probability = i128::from(outcome.probability_bps);
        let policy_weighted = i128::from(outcome.policy_net_cash_minor_zar)
            .checked_mul(probability)
            .ok_or(DecisionRegretError::ArithmeticOverflow)?;
        let oracle_weighted = i128::from(outcome.perfect_information_net_cash_minor_zar)
            .checked_mul(probability)
            .ok_or(DecisionRegretError::ArithmeticOverflow)?;
        let regret = i128::from(outcome.perfect_information_net_cash_minor_zar)
            .checked_sub(i128::from(outcome.policy_net_cash_minor_zar))
            .ok_or(DecisionRegretError::ArithmeticOverflow)?;
        let regret_weighted = regret
            .checked_mul(probability)
            .ok_or(DecisionRegretError::ArithmeticOverflow)?;

        policy_expected_numerator = add_i128(policy_expected_numerator, policy_weighted)?;
        oracle_expected_numerator = add_i128(oracle_expected_numerator, oracle_weighted)?;
        regret_expected_numerator = add_i128(regret_expected_numerator, regret_weighted)?;
        worst_case_regret = worst_case_regret.max(regret);
        regret_points.push((regret, outcome.probability_bps));
    }

    if outcome_ids != declared {
        return Err(DecisionRegretError::ScenarioCensusMismatch);
    }
    if u128::from(total_probability) != PROBABILITY_BPS {
        return Err(DecisionRegretError::ProbabilitiesDoNotSumToOne);
    }
    if oracle_expected_numerator.checked_sub(policy_expected_numerator)
        != Some(regret_expected_numerator)
    {
        return Err(DecisionRegretError::ArithmeticOverflow);
    }

    regret_points.sort_by(|left, right| right.0.cmp(&left.0));
    let tail_target = u32::from(inputs.regret_tail_probability_bps);
    let mut tail_remaining = tail_target;
    let mut tail_numerator = 0_i128;
    let mut tail_var = 0_i128;
    for (regret, probability_bps) in regret_points {
        let take = u32::from(probability_bps).min(tail_remaining);
        if take > 0 {
            tail_var = regret;
            let weighted = regret
                .checked_mul(i128::from(take))
                .ok_or(DecisionRegretError::ArithmeticOverflow)?;
            tail_numerator = add_i128(tail_numerator, weighted)?;
            tail_remaining = tail_remaining
                .checked_sub(take)
                .ok_or(DecisionRegretError::ArithmeticOverflow)?;
        }
        if tail_remaining == 0 {
            break;
        }
    }
    if tail_remaining != 0 {
        return Err(DecisionRegretError::ArithmeticOverflow);
    }

    let evpi_share_bps = if oracle_expected_numerator > 0 {
        let probability_basis =
            i128::try_from(PROBABILITY_BPS).map_err(|_| DecisionRegretError::ArithmeticOverflow)?;
        let numerator = regret_expected_numerator
            .checked_mul(probability_basis)
            .ok_or(DecisionRegretError::ArithmeticOverflow)?;
        Some(
            u64::try_from(numerator / oracle_expected_numerator)
                .map_err(|_| DecisionRegretError::ArithmeticOverflow)?,
        )
    } else {
        None
    };

    Ok(DecisionRegretAssessment {
        portfolio_id: inputs.portfolio_id,
        evaluation_as_of_utc: inputs.evaluation_as_of_utc,
        scenario_count: u64::try_from(inputs.outcomes.len())
            .map_err(|_| DecisionRegretError::ArithmeticOverflow)?,
        total_probability_bps: u16::try_from(total_probability)
            .map_err(|_| DecisionRegretError::ArithmeticOverflow)?,
        policy_expected_net_cash_numerator_minor_zar_bps: policy_expected_numerator,
        perfect_information_expected_net_cash_numerator_minor_zar_bps: oracle_expected_numerator,
        expected_regret_numerator_minor_zar_bps: regret_expected_numerator,
        expected_value_of_perfect_information_numerator_minor_zar_bps: regret_expected_numerator,
        expected_net_cash_denominator_bps: PROBABILITY_BPS,
        upper_tail_regret_var_minor_zar: tail_var,
        upper_tail_regret_cvar_numerator_minor_zar_bps: tail_numerator,
        upper_tail_regret_cvar_denominator_bps: inputs.regret_tail_probability_bps,
        worst_case_regret_minor_zar: worst_case_regret,
        evpi_as_share_of_oracle_bps: evpi_share_bps,
        non_anticipativity,
        perfect_information_is_non_implementable_bound: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::non_anticipativity::DispatchAction;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const INFO_SHA: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn decision(
        scenario_id: &'static str,
        node: &'static str,
        available_at: i64,
        decision_at: i64,
        net_power: i64,
    ) -> ScenarioDecision<'static> {
        ScenarioDecision {
            scenario_id,
            information_node_id: node,
            information_set_sha256: INFO_SHA,
            information_available_at_utc: available_at,
            decision_time_utc: decision_at,
            action: DispatchAction {
                net_power_w: net_power,
                reserve_up_w: 0,
                reserve_down_w: 0,
                target_soc_bps: 5_000,
            },
        }
    }

    fn outcome(
        scenario_id: &'static str,
        probability_bps: u16,
        policy: i64,
        oracle: i64,
    ) -> DecisionScenarioOutcome<'static> {
        DecisionScenarioOutcome {
            scenario_id,
            scenario_source_ref: scenario_id,
            scenario_source_sha256: SOURCE_SHA,
            scenario_available_at_utc: 150,
            probability_source_ref: "probability/manifest/v1",
            probability_source_sha256: SOURCE_SHA,
            probability_available_at_utc: 80,
            probability_bps,
            policy_result_source_ref: if scenario_id == "low" || scenario_id == "a" {
                "policy-result/low"
            } else if scenario_id == "high" || scenario_id == "b" {
                "policy-result/high"
            } else {
                "policy-result/third"
            },
            policy_result_sha256: SOURCE_SHA,
            policy_result_available_at_utc: 200,
            policy_net_cash_minor_zar: policy,
            perfect_information_result_source_ref: if scenario_id == "low" || scenario_id == "a" {
                "oracle-result/low"
            } else if scenario_id == "high" || scenario_id == "b" {
                "oracle-result/high"
            } else {
                "oracle-result/third"
            },
            perfect_information_result_sha256: SOURCE_SHA,
            perfect_information_result_available_at_utc: 210,
            perfect_information_net_cash_minor_zar: oracle,
        }
    }

    fn inputs<'a>(
        declared: &'a [&'a str],
        decisions: &'a [ScenarioDecision<'a>],
        outcomes: &'a [DecisionScenarioOutcome<'a>],
        tail_bps: u16,
    ) -> DecisionRegretInputs<'a> {
        DecisionRegretInputs {
            portfolio_id: "battery-001/rolling-horizon",
            evaluation_as_of_utc: 220,
            regret_tail_probability_bps: tail_bps,
            market_rule_source_ref: "market-code/rev2.3",
            market_rule_source_sha256: SOURCE_SHA,
            asset_constraint_source_ref: "asset/limits/v1",
            asset_constraint_source_sha256: SOURCE_SHA,
            network_topology_source_ref: "network/poi/v1",
            network_topology_source_sha256: SOURCE_SHA,
            settlement_basis_source_ref: "settlement/basis/v1",
            settlement_basis_source_sha256: SOURCE_SHA,
            declared_scenario_ids: declared,
            decisions,
            outcomes,
        }
    }

    #[test]
    fn computes_evpi_expected_regret_and_upper_tail_regret_exactly() {
        let declared = ["low", "high"];
        let decisions = [
            decision("low", "root", 70, 100, 10_000),
            decision("high", "root", 70, 100, 10_000),
        ];
        let outcomes = [
            outcome("low", 5_000, 80, 100),
            outcome("high", 5_000, 50, 70),
        ];
        let result =
            evaluate_decision_regret(inputs(&declared, &decisions, &outcomes, 5_000)).unwrap();
        assert_eq!(
            result.policy_expected_net_cash_numerator_minor_zar_bps,
            650_000
        );
        assert_eq!(
            result.perfect_information_expected_net_cash_numerator_minor_zar_bps,
            850_000
        );
        assert_eq!(result.expected_regret_numerator_minor_zar_bps, 200_000);
        assert_eq!(
            result.expected_value_of_perfect_information_numerator_minor_zar_bps,
            200_000
        );
        assert_eq!(result.upper_tail_regret_var_minor_zar, 20);
        assert_eq!(
            result.upper_tail_regret_cvar_numerator_minor_zar_bps,
            100_000
        );
        assert_eq!(result.upper_tail_regret_cvar_denominator_bps, 5_000);
        assert_eq!(result.evpi_as_share_of_oracle_bps, Some(2_352));
        assert!(result.perfect_information_is_non_implementable_bound);
        assert_eq!(result.non_anticipativity.shared_information_set_count, 1);
    }

    #[test]
    fn upper_tail_cvar_partially_weights_probability_at_the_boundary() {
        let declared = ["a", "b", "c"];
        let decisions = [
            decision("a", "root", 70, 100, 1),
            decision("b", "root", 70, 100, 1),
            decision("c", "root", 70, 100, 1),
        ];
        let outcomes = [
            outcome("a", 1_000, 0, 100),
            outcome("b", 4_000, 0, 50),
            outcome("c", 5_000, 0, 0),
        ];
        let result =
            evaluate_decision_regret(inputs(&declared, &decisions, &outcomes, 3_000)).unwrap();
        assert_eq!(result.upper_tail_regret_var_minor_zar, 50);
        assert_eq!(
            result.upper_tail_regret_cvar_numerator_minor_zar_bps,
            200_000
        );
        assert_eq!(result.upper_tail_regret_cvar_denominator_bps, 3_000);
    }

    #[test]
    fn shared_information_actions_must_match_before_the_scenario_tree_branches() {
        let declared = ["low", "high"];
        let decisions = [
            decision("low", "root", 70, 100, 10_000),
            decision("high", "root", 70, 100, 11_000),
        ];
        let outcomes = [
            outcome("low", 5_000, 80, 100),
            outcome("high", 5_000, 50, 70),
        ];
        assert_eq!(
            evaluate_decision_regret(inputs(&declared, &decisions, &outcomes, 5_000)),
            Err(DecisionRegretError::NonAnticipativePolicyInvalid)
        );
    }

    #[test]
    fn decisions_may_diverge_only_after_the_information_frontier_branches() {
        let declared = ["low", "high"];
        let decisions = [
            decision("low", "root", 70, 100, 10_000),
            decision("high", "root", 70, 100, 10_000),
            decision("low", "low-branch", 110, 120, -5_000),
            decision("high", "high-branch", 110, 120, 20_000),
        ];
        let outcomes = [
            outcome("low", 5_000, 80, 100),
            outcome("high", 5_000, 50, 70),
        ];
        let result =
            evaluate_decision_regret(inputs(&declared, &decisions, &outcomes, 5_000)).unwrap();
        assert_eq!(result.non_anticipativity.shared_information_set_count, 1);
        assert!(result.non_anticipativity.all_shared_decisions_identical);
    }

    #[test]
    fn oracle_below_policy_future_probability_and_unmatched_scenarios_fail_closed() {
        let declared = ["low", "high"];
        let decisions = [
            decision("low", "root", 70, 100, 10_000),
            decision("high", "root", 70, 100, 10_000),
        ];
        let oracle_below = [
            outcome("low", 5_000, 80, 70),
            outcome("high", 5_000, 50, 70),
        ];
        assert_eq!(
            evaluate_decision_regret(inputs(&declared, &decisions, &oracle_below, 5_000)),
            Err(DecisionRegretError::PerfectInformationOracleBelowPolicy)
        );

        let mut future_probability = outcome("low", 5_000, 80, 100);
        future_probability.probability_available_at_utc = 101;
        let future = [future_probability, outcome("high", 5_000, 50, 70)];
        assert_eq!(
            evaluate_decision_regret(inputs(&declared, &decisions, &future, 5_000)),
            Err(DecisionRegretError::ProbabilityNotAvailableAtDecision)
        );

        let missing = [outcome("low", 10_000, 80, 100)];
        assert_eq!(
            evaluate_decision_regret(inputs(&declared, &decisions, &missing, 5_000)),
            Err(DecisionRegretError::ScenarioCensusMismatch)
        );
    }

    #[test]
    fn scenario_probabilities_and_matched_source_basis_are_mandatory() {
        let declared = ["low", "high"];
        let decisions = [
            decision("low", "root", 70, 100, 10_000),
            decision("high", "root", 70, 100, 10_000),
        ];
        let invalid_probability = [
            outcome("low", 4_000, 80, 100),
            outcome("high", 5_000, 50, 70),
        ];
        assert_eq!(
            evaluate_decision_regret(inputs(&declared, &decisions, &invalid_probability, 5_000)),
            Err(DecisionRegretError::ProbabilitiesDoNotSumToOne)
        );

        let invalid_basis_outcomes = [
            outcome("low", 5_000, 80, 100),
            outcome("high", 5_000, 50, 70),
        ];
        let invalid_basis = DecisionRegretInputs {
            network_topology_source_ref: "asset/limits/v1",
            ..inputs(&declared, &decisions, &invalid_basis_outcomes, 5_000)
        };
        assert_eq!(
            evaluate_decision_regret(invalid_basis),
            Err(DecisionRegretError::InvalidEvidenceReference)
        );
    }
}
