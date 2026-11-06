//! Matched, paired comparison of two implementable BESS decision policies.
//!
//! This is the direct policy-to-policy comparator required for defensible
//! "better than baseline/vendor" claims. Both policies must be non-anticipative,
//! must have the same scenario census and probability manifest, and must use
//! identical decision-time information frontiers. Market rule, asset, topology,
//! settlement and scenario sources are content-addressed and shared.
//!
//! The evaluator scores source-bound outcomes supplied by adapters; it does not
//! execute PLEXOS, Chronos, HOMER, SAM, PyPSA, GridPath or a Luminous optimizer.
//! Positive candidate-minus-comparator cash delta means the candidate is better.

use std::collections::BTreeSet;

use crate::non_anticipativity::{
    NonAnticipativityAssessment, NonAnticipativityError, ScenarioDecision,
    audit_non_anticipative_policy,
};

const PROBABILITY_BPS: u128 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyOutcomePair<'a> {
    pub scenario_id: &'a str,
    pub scenario_source_ref: &'a str,
    pub scenario_source_sha256: &'a str,
    pub probability_source_ref: &'a str,
    pub probability_source_sha256: &'a str,
    pub probability_available_at_utc: i64,
    pub probability_bps: u16,
    pub candidate_result_record_id: &'a str,
    pub candidate_result_source_ref: &'a str,
    pub candidate_result_sha256: &'a str,
    pub candidate_result_available_at_utc: i64,
    pub candidate_net_cash_minor_zar: i64,
    pub comparator_result_record_id: &'a str,
    pub comparator_result_source_ref: &'a str,
    pub comparator_result_sha256: &'a str,
    pub comparator_result_available_at_utc: i64,
    pub comparator_net_cash_minor_zar: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairedPolicyComparisonInputs<'a> {
    pub comparison_id: &'a str,
    pub evaluation_as_of_utc: i64,
    /// Lower-tail probability for candidate-minus-comparator cash delta.
    /// 500 = worst 5% of paired outcomes.
    pub lower_tail_probability_bps: u16,
    pub candidate_policy_id: &'a str,
    pub candidate_policy_sha256: &'a str,
    pub comparator_policy_id: &'a str,
    pub comparator_policy_sha256: &'a str,
    pub market_rule_source_ref: &'a str,
    pub market_rule_source_sha256: &'a str,
    pub asset_constraints_source_ref: &'a str,
    pub asset_constraints_source_sha256: &'a str,
    pub network_topology_source_ref: &'a str,
    pub network_topology_source_sha256: &'a str,
    pub settlement_basis_source_ref: &'a str,
    pub settlement_basis_source_sha256: &'a str,
    pub scenario_manifest_source_ref: &'a str,
    pub scenario_manifest_source_sha256: &'a str,
    pub declared_scenario_ids: &'a [&'a str],
    pub candidate_decisions: &'a [ScenarioDecision<'a>],
    pub comparator_decisions: &'a [ScenarioDecision<'a>],
    pub paired_outcomes: &'a [PolicyOutcomePair<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairedPolicyComparisonError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidTailProbability,
    EmptyScenarioSet,
    EmptyDecisionSet,
    EmptyOutcomeSet,
    PolicyIdentityNotDistinct,
    ScenarioCensusMismatch,
    DuplicateScenarioId,
    DuplicateDecisionKey,
    DecisionCoverageMismatch,
    InformationFrontierMismatch,
    ProbabilityNotAvailableAtDecision,
    FutureKnownEvidence,
    InvalidScenarioProbability,
    ProbabilitiesDoNotSumToOne,
    DuplicateOutcomeRecordId,
    NonAnticipativeCandidateInvalid,
    NonAnticipativeComparatorInvalid,
    NoSharedInformationFrontier,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairedPolicyComparisonAssessment<'a> {
    pub comparison_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub candidate_policy_id: &'a str,
    pub candidate_policy_sha256: &'a str,
    pub comparator_policy_id: &'a str,
    pub comparator_policy_sha256: &'a str,
    pub scenario_count: u64,
    pub total_probability_bps: u16,
    pub candidate_expected_net_cash_numerator_minor_zar_bps: i128,
    pub comparator_expected_net_cash_numerator_minor_zar_bps: i128,
    /// Candidate minus comparator. Positive favours the candidate.
    pub expected_cash_delta_numerator_minor_zar_bps: i128,
    pub expected_cash_delta_denominator_bps: u128,
    pub candidate_win_probability_bps: u16,
    pub comparator_win_probability_bps: u16,
    pub tie_probability_bps: u16,
    pub candidate_win_scenario_count: u64,
    pub comparator_win_scenario_count: u64,
    pub tie_scenario_count: u64,
    pub worst_case_cash_delta_minor_zar: i128,
    pub best_case_cash_delta_minor_zar: i128,
    pub lower_tail_delta_var_minor_zar: i128,
    pub lower_tail_delta_cvar_numerator_minor_zar_bps: i128,
    pub lower_tail_delta_cvar_denominator_bps: u16,
    /// True only if both policies passed non-anticipativity and were replayed
    /// against identical information frontiers and common input basis.
    pub comparison_is_matched_and_non_anticipative: bool,
    pub candidate_non_anticipativity: NonAnticipativityAssessment<'a>,
    pub comparator_non_anticipativity: NonAnticipativityAssessment<'a>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_reference(reference: &str, digest: &str) -> Result<(), PairedPolicyComparisonError> {
    if !nonempty(reference) {
        return Err(PairedPolicyComparisonError::InvalidEvidenceReference);
    }
    if !valid_sha256(digest) {
        return Err(PairedPolicyComparisonError::InvalidSourceDigest);
    }
    Ok(())
}

fn add_i128(
    current: i128,
    value: i128,
) -> Result<i128, PairedPolicyComparisonError> {
    current
        .checked_add(value)
        .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)
}

fn validate_decision_coverage_and_frontier(
    candidate: &[ScenarioDecision<'_>],
    comparator: &[ScenarioDecision<'_>],
) -> Result<(), PairedPolicyComparisonError> {
    if candidate.is_empty() || comparator.is_empty() {
        return Err(PairedPolicyComparisonError::EmptyDecisionSet);
    }
    if candidate.len() != comparator.len() {
        return Err(PairedPolicyComparisonError::DecisionCoverageMismatch);
    }

    for (index, decision) in candidate.iter().copied().enumerate() {
        if candidate[..index].iter().any(|previous| {
            previous.scenario_id == decision.scenario_id
                && previous.decision_time_utc == decision.decision_time_utc
        }) {
            return Err(PairedPolicyComparisonError::DuplicateDecisionKey);
        }
    }
    for (index, decision) in comparator.iter().copied().enumerate() {
        if comparator[..index].iter().any(|previous| {
            previous.scenario_id == decision.scenario_id
                && previous.decision_time_utc == decision.decision_time_utc
        }) {
            return Err(PairedPolicyComparisonError::DuplicateDecisionKey);
        }
    }

    for candidate_decision in candidate.iter().copied() {
        let mut matches = comparator.iter().copied().filter(|other| {
            other.scenario_id == candidate_decision.scenario_id
                && other.decision_time_utc == candidate_decision.decision_time_utc
        });
        let Some(comparator_decision) = matches.next() else {
            return Err(PairedPolicyComparisonError::DecisionCoverageMismatch);
        };
        if matches.next().is_some() {
            return Err(PairedPolicyComparisonError::DuplicateDecisionKey);
        }
        if candidate_decision.information_node_id != comparator_decision.information_node_id
            || candidate_decision.information_set_sha256
                != comparator_decision.information_set_sha256
            || candidate_decision.information_available_at_utc
                != comparator_decision.information_available_at_utc
        {
            return Err(PairedPolicyComparisonError::InformationFrontierMismatch);
        }
    }
    Ok(())
}

/// Score two implementable, source-bound policies on the same scenario tree.
/// The common source basis prevents a different rule version, battery rating,
/// topology or settlement interpretation from being smuggled into the delta.
pub fn evaluate_paired_policy_comparison<'a>(
    inputs: PairedPolicyComparisonInputs<'a>,
) -> Result<PairedPolicyComparisonAssessment<'a>, PairedPolicyComparisonError> {
    if !nonempty(inputs.comparison_id)
        || !nonempty(inputs.candidate_policy_id)
        || !nonempty(inputs.comparator_policy_id)
    {
        return Err(PairedPolicyComparisonError::InvalidEvidenceReference);
    }
    if inputs.candidate_policy_id == inputs.comparator_policy_id
        && inputs.candidate_policy_sha256 == inputs.comparator_policy_sha256
    {
        return Err(PairedPolicyComparisonError::PolicyIdentityNotDistinct);
    }
    validate_reference(
        inputs.candidate_policy_id,
        inputs.candidate_policy_sha256,
    )?;
    validate_reference(
        inputs.comparator_policy_id,
        inputs.comparator_policy_sha256,
    )?;

    let common_references = [
        inputs.market_rule_source_ref,
        inputs.asset_constraints_source_ref,
        inputs.network_topology_source_ref,
        inputs.settlement_basis_source_ref,
        inputs.scenario_manifest_source_ref,
    ];
    let common_digests = [
        inputs.market_rule_source_sha256,
        inputs.asset_constraints_source_sha256,
        inputs.network_topology_source_sha256,
        inputs.settlement_basis_source_sha256,
        inputs.scenario_manifest_source_sha256,
    ];
    for (reference, digest) in common_references.into_iter().zip(common_digests) {
        validate_reference(reference, digest)?;
    }
    for (index, reference) in common_references.iter().enumerate() {
        if common_references[..index].contains(reference) {
            return Err(PairedPolicyComparisonError::InvalidEvidenceReference);
        }
    }

    if inputs.declared_scenario_ids.is_empty() {
        return Err(PairedPolicyComparisonError::EmptyScenarioSet);
    }
    if inputs.paired_outcomes.is_empty() {
        return Err(PairedPolicyComparisonError::EmptyOutcomeSet);
    }
    if inputs.lower_tail_probability_bps == 0
        || u128::from(inputs.lower_tail_probability_bps) > PROBABILITY_BPS
    {
        return Err(PairedPolicyComparisonError::InvalidTailProbability);
    }

    let mut declared = BTreeSet::new();
    for scenario_id in inputs.declared_scenario_ids.iter().copied() {
        if !nonempty(scenario_id) {
            return Err(PairedPolicyComparisonError::InvalidEvidenceReference);
        }
        if !declared.insert(scenario_id) {
            return Err(PairedPolicyComparisonError::DuplicateScenarioId);
        }
    }
    if inputs.paired_outcomes.len() != declared.len() {
        return Err(PairedPolicyComparisonError::ScenarioCensusMismatch);
    }

    validate_decision_coverage_and_frontier(
        inputs.candidate_decisions,
        inputs.comparator_decisions,
    )?;

    let candidate_non_anticipativity = audit_non_anticipative_policy(
        inputs.declared_scenario_ids,
        inputs.candidate_decisions,
    )
    .map_err(|_: NonAnticipativityError| {
        PairedPolicyComparisonError::NonAnticipativeCandidateInvalid
    })?;
    let comparator_non_anticipativity = audit_non_anticipative_policy(
        inputs.declared_scenario_ids,
        inputs.comparator_decisions,
    )
    .map_err(|_: NonAnticipativityError| {
        PairedPolicyComparisonError::NonAnticipativeComparatorInvalid
    })?;
    if declared.len() > 1
        && (candidate_non_anticipativity.shared_information_set_count == 0
            || comparator_non_anticipativity.shared_information_set_count == 0)
    {
        return Err(PairedPolicyComparisonError::NoSharedInformationFrontier);
    }

    let mut outcome_ids = BTreeSet::new();
    let mut scenario_source_refs = BTreeSet::new();
    let mut candidate_record_ids = BTreeSet::new();
    let mut comparator_record_ids = BTreeSet::new();
    let mut total_probability = 0_u32;
    let mut candidate_expected = 0_i128;
    let mut comparator_expected = 0_i128;
    let mut delta_expected = 0_i128;
    let mut candidate_wins_probability = 0_u32;
    let mut comparator_wins_probability = 0_u32;
    let mut tie_probability = 0_u32;
    let mut candidate_win_count = 0_u64;
    let mut comparator_win_count = 0_u64;
    let mut tie_count = 0_u64;
    let mut worst_delta: Option<i128> = None;
    let mut best_delta: Option<i128> = None;
    let mut deltas = Vec::with_capacity(inputs.paired_outcomes.len());

    for outcome in inputs.paired_outcomes.iter().copied() {
        if !declared.contains(outcome.scenario_id) || !outcome_ids.insert(outcome.scenario_id) {
            return Err(PairedPolicyComparisonError::ScenarioCensusMismatch);
        }
        validate_reference(outcome.scenario_source_ref, outcome.scenario_source_sha256)?;
        validate_reference(outcome.probability_source_ref, outcome.probability_source_sha256)?;
        validate_reference(
            outcome.candidate_result_source_ref,
            outcome.candidate_result_sha256,
        )?;
        validate_reference(
            outcome.comparator_result_source_ref,
            outcome.comparator_result_sha256,
        )?;

        if outcome.probability_bps == 0 {
            return Err(PairedPolicyComparisonError::InvalidScenarioProbability);
        }
        if outcome.probability_available_at_utc > inputs.evaluation_as_of_utc
            || outcome.candidate_result_available_at_utc > inputs.evaluation_as_of_utc
            || outcome.comparator_result_available_at_utc > inputs.evaluation_as_of_utc
        {
            return Err(PairedPolicyComparisonError::FutureKnownEvidence);
        }
        if !scenario_source_refs.insert(outcome.scenario_source_ref)
            || !candidate_record_ids.insert(outcome.candidate_result_record_id)
            || !comparator_record_ids.insert(outcome.comparator_result_record_id)
        {
            return Err(PairedPolicyComparisonError::DuplicateOutcomeRecordId);
        }

        let matching_decisions = inputs
            .candidate_decisions
            .iter()
            .copied()
            .filter(|decision| decision.scenario_id == outcome.scenario_id);
        for decision in matching_decisions {
            if outcome.probability_available_at_utc > decision.decision_time_utc {
                return Err(PairedPolicyComparisonError::ProbabilityNotAvailableAtDecision);
            }
        }

        total_probability = total_probability
            .checked_add(u32::from(outcome.probability_bps))
            .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;
        let probability = i128::from(outcome.probability_bps);
        let candidate_weighted = i128::from(outcome.candidate_net_cash_minor_zar)
            .checked_mul(probability)
            .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;
        let comparator_weighted = i128::from(outcome.comparator_net_cash_minor_zar)
            .checked_mul(probability)
            .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;
        let delta = i128::from(outcome.candidate_net_cash_minor_zar)
            .checked_sub(i128::from(outcome.comparator_net_cash_minor_zar))
            .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;
        let delta_weighted = delta
            .checked_mul(probability)
            .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;

        candidate_expected = add_i128(candidate_expected, candidate_weighted)?;
        comparator_expected = add_i128(comparator_expected, comparator_weighted)?;
        delta_expected = add_i128(delta_expected, delta_weighted)?;

        if delta > 0 {
            candidate_wins_probability = candidate_wins_probability
                .checked_add(u32::from(outcome.probability_bps))
                .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;
            candidate_win_count = candidate_win_count
                .checked_add(1)
                .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;
        } else if delta < 0 {
            comparator_wins_probability = comparator_wins_probability
                .checked_add(u32::from(outcome.probability_bps))
                .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;
            comparator_win_count = comparator_win_count
                .checked_add(1)
                .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;
        } else {
            tie_probability = tie_probability
                .checked_add(u32::from(outcome.probability_bps))
                .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;
            tie_count = tie_count
                .checked_add(1)
                .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;
        }

        worst_delta = Some(worst_delta.map_or(delta, |current| current.min(delta)));
        best_delta = Some(best_delta.map_or(delta, |current| current.max(delta)));
        deltas.push((delta, outcome.probability_bps));
    }

    if outcome_ids != declared {
        return Err(PairedPolicyComparisonError::ScenarioCensusMismatch);
    }
    if u128::from(total_probability) != PROBABILITY_BPS {
        return Err(PairedPolicyComparisonError::ProbabilitiesDoNotSumToOne);
    }
    if candidate_expected.checked_sub(comparator_expected) != Some(delta_expected) {
        return Err(PairedPolicyComparisonError::ArithmeticOverflow);
    }

    deltas.sort_by(|left, right| left.0.cmp(&right.0));
    let tail_target = u32::from(inputs.lower_tail_probability_bps);
    let mut remaining = tail_target;
    let mut tail_numerator = 0_i128;
    let mut tail_var = 0_i128;
    for (delta, probability_bps) in deltas {
        let take = u32::from(probability_bps).min(remaining);
        if take > 0 {
            tail_var = delta;
            tail_numerator = add_i128(
                tail_numerator,
                delta
                    .checked_mul(i128::from(take))
                    .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?,
            )?;
            remaining = remaining
                .checked_sub(take)
                .ok_or(PairedPolicyComparisonError::ArithmeticOverflow)?;
        }
        if remaining == 0 {
            break;
        }
    }
    if remaining != 0 {
        return Err(PairedPolicyComparisonError::ArithmeticOverflow);
    }

    Ok(PairedPolicyComparisonAssessment {
        comparison_id: inputs.comparison_id,
        evaluation_as_of_utc: inputs.evaluation_as_of_utc,
        candidate_policy_id: inputs.candidate_policy_id,
        candidate_policy_sha256: inputs.candidate_policy_sha256,
        comparator_policy_id: inputs.comparator_policy_id,
        comparator_policy_sha256: inputs.comparator_policy_sha256,
        scenario_count: u64::try_from(inputs.paired_outcomes.len())
            .map_err(|_| PairedPolicyComparisonError::ArithmeticOverflow)?,
        total_probability_bps: u16::try_from(total_probability)
            .map_err(|_| PairedPolicyComparisonError::ArithmeticOverflow)?,
        candidate_expected_net_cash_numerator_minor_zar_bps: candidate_expected,
        comparator_expected_net_cash_numerator_minor_zar_bps: comparator_expected,
        expected_cash_delta_numerator_minor_zar_bps: delta_expected,
        expected_cash_delta_denominator_bps: PROBABILITY_BPS,
        candidate_win_probability_bps: u16::try_from(candidate_wins_probability)
            .map_err(|_| PairedPolicyComparisonError::ArithmeticOverflow)?,
        comparator_win_probability_bps: u16::try_from(comparator_wins_probability)
            .map_err(|_| PairedPolicyComparisonError::ArithmeticOverflow)?,
        tie_probability_bps: u16::try_from(tie_probability)
            .map_err(|_| PairedPolicyComparisonError::ArithmeticOverflow)?,
        candidate_win_scenario_count: candidate_win_count,
        comparator_win_scenario_count: comparator_win_count,
        tie_scenario_count: tie_count,
        worst_case_cash_delta_minor_zar: worst_delta
            .ok_or(PairedPolicyComparisonError::EmptyOutcomeSet)?,
        best_case_cash_delta_minor_zar: best_delta
            .ok_or(PairedPolicyComparisonError::EmptyOutcomeSet)?,
        lower_tail_delta_var_minor_zar: tail_var,
        lower_tail_delta_cvar_numerator_minor_zar_bps: tail_numerator,
        lower_tail_delta_cvar_denominator_bps: inputs.lower_tail_probability_bps,
        comparison_is_matched_and_non_anticipative: true,
        candidate_non_anticipativity,
        comparator_non_anticipativity,
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
        action_power: i64,
    ) -> ScenarioDecision<'static> {
        ScenarioDecision {
            scenario_id,
            information_node_id: node,
            information_set_sha256: INFO_SHA,
            information_available_at_utc: available_at,
            decision_time_utc: decision_at,
            action: DispatchAction {
                net_power_w: action_power,
                reserve_up_w: 0,
                reserve_down_w: 0,
                target_soc_bps: 5_000,
            },
        }
    }

    fn outcome(
        scenario_id: &'static str,
        probability_bps: u16,
        candidate_cash: i64,
        comparator_cash: i64,
    ) -> PolicyOutcomePair<'static> {
        PolicyOutcomePair {
            scenario_id,
            scenario_source_ref: scenario_id,
            scenario_source_sha256: SOURCE_SHA,
            probability_source_ref: "probability/manifest/v1",
            probability_source_sha256: SOURCE_SHA,
            probability_available_at_utc: 80,
            probability_bps,
            candidate_result_record_id: if scenario_id == "low" {
                "candidate/low"
            } else if scenario_id == "high" {
                "candidate/high"
            } else {
                "candidate/third"
            },
            candidate_result_source_ref: if scenario_id == "low" {
                "candidate/result/low"
            } else if scenario_id == "high" {
                "candidate/result/high"
            } else {
                "candidate/result/third"
            },
            candidate_result_sha256: SOURCE_SHA,
            candidate_result_available_at_utc: 200,
            candidate_net_cash_minor_zar: candidate_cash,
            comparator_result_record_id: if scenario_id == "low" {
                "comparator/low"
            } else if scenario_id == "high" {
                "comparator/high"
            } else {
                "comparator/third"
            },
            comparator_result_source_ref: if scenario_id == "low" {
                "comparator/result/low"
            } else if scenario_id == "high" {
                "comparator/result/high"
            } else {
                "comparator/result/third"
            },
            comparator_result_sha256: SOURCE_SHA,
            comparator_result_available_at_utc: 200,
            comparator_net_cash_minor_zar: comparator_cash,
        }
    }

    fn inputs<'a>(
        declared: &'a [&'a str],
        candidate: &'a [ScenarioDecision<'a>],
        comparator: &'a [ScenarioDecision<'a>],
        outcomes: &'a [PolicyOutcomePair<'a>],
        tail_bps: u16,
    ) -> PairedPolicyComparisonInputs<'a> {
        PairedPolicyComparisonInputs {
            comparison_id: "paired/battery-001",
            evaluation_as_of_utc: 220,
            lower_tail_probability_bps: tail_bps,
            candidate_policy_id: "luminous-policy",
            candidate_policy_sha256: SOURCE_SHA,
            comparator_policy_id: "baseline-policy",
            comparator_policy_sha256: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            market_rule_source_ref: "market-code/rev2.3",
            market_rule_source_sha256: SOURCE_SHA,
            asset_constraints_source_ref: "asset/limits/v1",
            asset_constraints_source_sha256: SOURCE_SHA,
            network_topology_source_ref: "network/topology/v1",
            network_topology_source_sha256: SOURCE_SHA,
            settlement_basis_source_ref: "settlement/basis/v1",
            settlement_basis_source_sha256: SOURCE_SHA,
            scenario_manifest_source_ref: "scenario/manifest/v1",
            scenario_manifest_source_sha256: SOURCE_SHA,
            declared_scenario_ids: declared,
            candidate_decisions: candidate,
            comparator_decisions: comparator,
            paired_outcomes: outcomes,
        }
    }

    #[test]
    fn paired_comparison_reports_expected_delta_win_probability_and_lower_tail() {
        let declared = ["low", "high"];
        let candidate = [
            decision("low", "root", 70, 100, 10_000),
            decision("high", "root", 70, 100, 10_000),
        ];
        let comparator = [
            decision("low", "root", 70, 100, 8_000),
            decision("high", "root", 70, 100, 8_000),
        ];
        let outcomes = [
            outcome("low", 5_000, 100, 80),
            outcome("high", 5_000, 50, 70),
        ];
        let result =
            evaluate_paired_policy_comparison(inputs(&declared, &candidate, &comparator, &outcomes, 5_000))
                .unwrap();
        assert_eq!(result.candidate_expected_net_cash_numerator_minor_zar_bps, 750_000);
        assert_eq!(result.comparator_expected_net_cash_numerator_minor_zar_bps, 750_000);
        assert_eq!(result.expected_cash_delta_numerator_minor_zar_bps, 0);
        assert_eq!(result.candidate_win_probability_bps, 5_000);
        assert_eq!(result.comparator_win_probability_bps, 5_000);
        assert_eq!(result.tie_probability_bps, 0);
        assert_eq!(result.worst_case_cash_delta_minor_zar, -20);
        assert_eq!(result.best_case_cash_delta_minor_zar, 20);
        assert_eq!(result.lower_tail_delta_var_minor_zar, -20);
        assert_eq!(result.lower_tail_delta_cvar_numerator_minor_zar_bps, -100_000);
        assert!(result.comparison_is_matched_and_non_anticipative);
    }

    #[test]
    fn exact_information_frontier_must_match_between_policies() {
        let declared = ["low", "high"];
        let candidate = [
            decision("low", "root", 70, 100, 10_000),
            decision("high", "root", 70, 100, 10_000),
        ];
        let comparator = [
            decision("low", "root", 70, 100, 8_000),
            ScenarioDecision {
                information_set_sha256: "different-information-set-digest",
                ..decision("high", "root", 70, 100, 8_000)
            },
        ];
        let outcomes = [
            outcome("low", 5_000, 100, 80),
            outcome("high", 5_000, 50, 70),
        ];
        assert_eq!(
            evaluate_paired_policy_comparison(inputs(
                &declared,
                &candidate,
                &comparator,
                &outcomes,
                5_000
            )),
            Err(PairedPolicyComparisonError::InformationFrontierMismatch)
        );
    }

    #[test]
    fn both_policies_must_be_non_anticipative() {
        let declared = ["low", "high"];
        let candidate = [
            decision("low", "root", 70, 100, 10_000),
            decision("high", "root", 70, 100, 11_000),
        ];
        let comparator = [
            decision("low", "root", 70, 100, 8_000),
            decision("high", "root", 70, 100, 8_000),
        ];
        let outcomes = [
            outcome("low", 5_000, 100, 80),
            outcome("high", 5_000, 50, 70),
        ];
        assert_eq!(
            evaluate_paired_policy_comparison(inputs(
                &declared,
                &candidate,
                &comparator,
                &outcomes,
                5_000
            )),
            Err(PairedPolicyComparisonError::NonAnticipativeCandidateInvalid)
        );
    }

    #[test]
    fn source_basis_probability_census_and_time_coverage_fail_closed() {
        let declared = ["low", "high"];
        let candidate = [
            decision("low", "root", 70, 100, 10_000),
            decision("high", "root", 70, 100, 10_000),
        ];
        let comparator = [
            decision("low", "root", 70, 100, 8_000),
            decision("high", "root", 70, 100, 8_000),
        ];
        let outcomes = [
            outcome("low", 5_000, 100, 80),
            outcome("high", 5_000, 50, 70),
        ];
        let bad_probability = [
            outcomes[0],
            PolicyOutcomePair {
                probability_bps: 4_000,
                ..outcomes[1]
            },
        ];
        assert_eq!(
            evaluate_paired_policy_comparison(inputs(
                &declared,
                &candidate,
                &comparator,
                &bad_probability,
                5_000
            )),
            Err(PairedPolicyComparisonError::ProbabilitiesDoNotSumToOne)
        );

        let bad_time = [
            outcomes[0],
            PolicyOutcomePair {
                candidate_result_available_at_utc: 221,
                ..outcomes[1]
            },
        ];
        assert_eq!(
            evaluate_paired_policy_comparison(inputs(
                &declared,
                &candidate,
                &comparator,
                &bad_time,
                5_000
            )),
            Err(PairedPolicyComparisonError::FutureKnownEvidence)
        );

        let invalid_basis = PairedPolicyComparisonInputs {
            network_topology_source_ref: "asset/limits/v1",
            ..inputs(&declared, &candidate, &comparator, &outcomes, 5_000)
        };
        assert_eq!(
            evaluate_paired_policy_comparison(invalid_basis),
            Err(PairedPolicyComparisonError::InvalidEvidenceReference)
        );
    }
}
