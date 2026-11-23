// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two-stage metrology acquisition: low-cost screening followed by
//! candidate- and outcome-specific high-fidelity confirmation.
//!
//! This is a deterministic synthetic decision planner. It does not execute
//! measurements, qualify calibration evidence or grant process authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const TWO_STAGE_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.two-stage-metrology-input.v1";
pub const TWO_STAGE_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.two-stage-metrology-report.v1";
pub const MAX_TWO_STAGE_CANDIDATES_EXACT: usize = 18;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryAssayCharacteristicsV1 {
    pub sensitivity: f64,
    pub specificity: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryDecisionLossV1 {
    pub false_negative_cost: f64,
    pub false_positive_cost: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoStageMetrologyCandidateV1 {
    pub candidate_id: String,
    pub prior_defect_probability: f64,
    pub screening: BinaryAssayCharacteristicsV1,
    pub screening_cost: u64,
    pub confirmation: BinaryAssayCharacteristicsV1,
    pub confirmation_cost: u64,
    pub decision_loss: BinaryDecisionLossV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoStageMetrologyPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    /// Digest of the candidate packet after canonical candidate-ID ordering.
    pub candidate_packet_sha256: String,
    /// Screening budget is consumed before any high-fidelity outcome is known.
    pub screening_cost_budget: u64,
    /// Hard capacity: the sum of confirmation costs if every selected
    /// candidate reaches any branch that invokes confirmation.
    pub confirmation_worst_case_cost_budget: u64,
    pub max_candidates_exact: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreenOutcomeV1 {
    Positive,
    Negative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfirmationActionV1 {
    Confirm,
    StopAfterScreen,
    OutcomeImpossible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceDispositionV1 {
    HoldForConfirmation,
    ScreenOnlyDecision,
    NoEvidenceDecision,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoStageOutcomeBranchV1 {
    pub screen_outcome: ScreenOutcomeV1,
    pub screen_outcome_probability: f64,
    pub posterior_defect_probability: Option<f64>,
    pub decision_loss_without_confirmation: Option<f64>,
    pub expected_decision_loss_after_confirmation: Option<f64>,
    pub gross_confirmation_evsi: f64,
    pub net_confirmation_value: f64,
    pub action: ConfirmationActionV1,
    pub expected_confirmation_cost: f64,
    pub worst_case_confirmation_cost: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoStageCandidatePlanV1 {
    pub candidate_id: String,
    pub prior_defect_probability: f64,
    pub low_screen_cost: u64,
    pub high_confirmation_cost: u64,
    pub screen_positive_probability: f64,
    pub screen_negative_probability: f64,
    pub confirm_on_positive: bool,
    pub confirm_on_negative: bool,
    pub expected_incremental_net_value: f64,
    pub expected_confirmation_cost: f64,
    pub worst_case_confirmation_cost: u64,
    pub selected_for_confirmation_portfolio: bool,
    pub positive_branch: TwoStageOutcomeBranchV1,
    pub negative_branch: TwoStageOutcomeBranchV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoStageClaimLimitsV1 {
    pub plan_does_not_execute_measurements: bool,
    pub planned_confirmation_is_not_a_completed_measurement: bool,
    pub qualification_gate_must_pass_before_label_promotion: bool,
    pub no_fab_process_authority_is_granted: bool,
    pub model_assay_parameters_are_declared_synthetic_inputs: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoStageMetrologyReportV1 {
    pub schema: String,
    pub policy_id: String,
    pub policy_sha256: String,
    pub candidate_packet_sha256: String,
    pub screening_cost_budget: u64,
    pub screening_cost_used: u64,
    pub no_measurement_total_cost: u64,
    pub screen_only_total_cost: u64,
    pub high_fidelity_only_total_cost: u64,
    pub confirmation_worst_case_cost_budget: u64,
    pub confirmation_worst_case_cost_used: u64,
    pub total_worst_case_measurement_cost: u64,
    pub confirmation_expected_cost: f64,
    pub total_expected_measurement_cost: f64,
    pub no_measurement_expected_decision_loss: f64,
    pub screen_only_expected_decision_loss: f64,
    pub high_fidelity_only_expected_decision_loss: f64,
    pub two_stage_expected_decision_loss: f64,
    pub expected_incremental_net_value: f64,
    pub subsets_examined: usize,
    pub selected_confirmation_candidates: Vec<String>,
    pub candidates: Vec<TwoStageCandidatePlanV1>,
    pub report_sha256: String,
    pub claim_limits: TwoStageClaimLimitsV1,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct TwoStageMetrologyErrorV1(pub String);

#[derive(Clone, Debug)]
struct BinaryEvsiV1 {
    probability_positive: f64,
    probability_negative: f64,
    posterior_positive: Option<f64>,
    posterior_negative: Option<f64>,
    expected_loss_without_measurement: f64,
    expected_loss_after_measurement: f64,
    gross_evsi: f64,
}

#[derive(Clone, Debug)]
struct ConfirmationPortfolioScoreV1 {
    indices: Vec<usize>,
    cost: u64,
    expected_net_value: f64,
    ids: Vec<String>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, TwoStageMetrologyErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        TwoStageMetrologyErrorV1(format!("canonical serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn finite_non_negative(value: f64, field: &str) -> Result<(), TwoStageMetrologyErrorV1> {
    if !value.is_finite() || value < 0.0 {
        return Err(TwoStageMetrologyErrorV1(format!(
            "{field}: expected a finite non-negative value"
        )));
    }
    Ok(())
}

fn validate_assay(
    assay: BinaryAssayCharacteristicsV1,
    field: &str,
) -> Result<(), TwoStageMetrologyErrorV1> {
    if !assay.sensitivity.is_finite()
        || !(0.0..=1.0).contains(&assay.sensitivity)
        || !assay.specificity.is_finite()
        || !(0.0..=1.0).contains(&assay.specificity)
    {
        return Err(TwoStageMetrologyErrorV1(format!(
            "{field}: sensitivity and specificity must be finite and in [0, 1]"
        )));
    }
    Ok(())
}

fn optimal_binary_decision_loss(
    prior_defect_probability: f64,
    loss: BinaryDecisionLossV1,
) -> f64 {
    let loss_when_disposed_as_pass = prior_defect_probability * loss.false_negative_cost;
    let loss_when_disposed_as_defect = (1.0 - prior_defect_probability) * loss.false_positive_cost;
    loss_when_disposed_as_pass.min(loss_when_disposed_as_defect)
}

/// Expected loss before/after one binary assay under the declared loss model.
/// A numerical-zero tolerance prevents an uninformative assay from obtaining
/// a positive EVSI score through floating-point cancellation.
fn binary_evsi(
    prior: f64,
    assay: BinaryAssayCharacteristicsV1,
    loss: BinaryDecisionLossV1,
) -> BinaryEvsiV1 {
    let probability_positive =
        prior * assay.sensitivity + (1.0 - prior) * (1.0 - assay.specificity);
    let probability_negative = (1.0 - probability_positive).clamp(0.0, 1.0);

    let posterior_positive = (probability_positive > 0.0).then(|| {
        (prior * assay.sensitivity / probability_positive).clamp(0.0, 1.0)
    });
    let posterior_negative = (probability_negative > 0.0).then(|| {
        (prior * (1.0 - assay.sensitivity) / probability_negative).clamp(0.0, 1.0)
    });

    let expected_loss_without_measurement = optimal_binary_decision_loss(prior, loss);
    let positive_loss = posterior_positive
        .map(|posterior| optimal_binary_decision_loss(posterior, loss))
        .unwrap_or(0.0);
    let negative_loss = posterior_negative
        .map(|posterior| optimal_binary_decision_loss(posterior, loss))
        .unwrap_or(0.0);
    let expected_loss_after_measurement =
        probability_positive * positive_loss + probability_negative * negative_loss;

    let raw_gross = expected_loss_without_measurement - expected_loss_after_measurement;
    let tolerance = 1.0e-12
        * expected_loss_without_measurement
            .abs()
            .max(expected_loss_after_measurement.abs())
            .max(1.0);
    let gross_evsi = if raw_gross > tolerance { raw_gross } else { 0.0 };

    BinaryEvsiV1 {
        probability_positive,
        probability_negative,
        posterior_positive,
        posterior_negative,
        expected_loss_without_measurement,
        expected_loss_after_measurement,
        gross_evsi,
    }
}

fn validate_candidate(
    candidate: &TwoStageMetrologyCandidateV1,
    index: usize,
) -> Result<(), TwoStageMetrologyErrorV1> {
    if candidate.candidate_id.trim().is_empty() {
        return Err(TwoStageMetrologyErrorV1(format!(
            "candidates[{index}].candidate_id: expected non-empty string"
        )));
    }
    if !candidate.prior_defect_probability.is_finite()
        || !(0.0..=1.0).contains(&candidate.prior_defect_probability)
    {
        return Err(TwoStageMetrologyErrorV1(format!(
            "candidates[{index}].prior_defect_probability: expected finite value in [0, 1]"
        )));
    }
    validate_assay(candidate.screening, &format!("candidates[{index}].screening"))?;
    validate_assay(
        candidate.confirmation,
        &format!("candidates[{index}].confirmation"),
    )?;
    finite_non_negative(
        candidate.decision_loss.false_negative_cost,
        &format!("candidates[{index}].false_negative_cost"),
    )?;
    finite_non_negative(
        candidate.decision_loss.false_positive_cost,
        &format!("candidates[{index}].false_positive_cost"),
    )?;
    if candidate.decision_loss.false_negative_cost.max(candidate.decision_loss.false_positive_cost)
        <= 0.0
    {
        return Err(TwoStageMetrologyErrorV1(format!(
            "candidates[{index}].decision_loss: at least one error cost must be positive"
        )));
    }
    if candidate.screening_cost == 0 || candidate.confirmation_cost == 0 {
        return Err(TwoStageMetrologyErrorV1(format!(
            "candidates[{index}]: screening and confirmation costs must be positive"
        )));
    }
    Ok(())
}

fn validate_policy_and_candidates(
    policy: &TwoStageMetrologyPolicyV1,
    candidates: &[TwoStageMetrologyCandidateV1],
) -> Result<Vec<TwoStageMetrologyCandidateV1>, TwoStageMetrologyErrorV1> {
    if policy.schema != TWO_STAGE_INPUT_SCHEMA_V1 {
        return Err(TwoStageMetrologyErrorV1("policy.schema: unsupported".into()));
    }
    if policy.policy_id.trim().is_empty() {
        return Err(TwoStageMetrologyErrorV1("policy.policy_id: empty".into()));
    }
    if policy.max_candidates_exact == 0
        || policy.max_candidates_exact > MAX_TWO_STAGE_CANDIDATES_EXACT
    {
        return Err(TwoStageMetrologyErrorV1(format!(
            "policy.max_candidates_exact must be in 1..={MAX_TWO_STAGE_CANDIDATES_EXACT}"
        )));
    }
    if candidates.is_empty() || candidates.len() > policy.max_candidates_exact {
        return Err(TwoStageMetrologyErrorV1(
            "candidate count must be non-empty and within the exact subset limit".into(),
        ));
    }

    let mut canonical = candidates.to_vec();
    canonical.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    let mut seen = BTreeSet::new();
    let mut screening_total = 0_u64;

    for (index, candidate) in canonical.iter().enumerate() {
        validate_candidate(candidate, index)?;
        if !seen.insert(candidate.candidate_id.as_str()) {
            return Err(TwoStageMetrologyErrorV1(format!(
                "candidates[{index}].candidate_id: duplicate identity"
            )));
        }
        screening_total = screening_total
            .checked_add(candidate.screening_cost)
            .ok_or_else(|| TwoStageMetrologyErrorV1("screening cost overflow".into()))?;
    }
    if screening_total > policy.screening_cost_budget {
        return Err(TwoStageMetrologyErrorV1(
            "screening stage exceeds its fixed screening-cost budget".into(),
        ));
    }

    let packet_digest = json_sha256(&canonical)?;
    if packet_digest != policy.candidate_packet_sha256 {
        return Err(TwoStageMetrologyErrorV1(
            "candidate packet digest does not match canonical candidates".into(),
        ));
    }
    Ok(canonical)
}

fn candidate_branch(
    outcome: ScreenOutcomeV1,
    screen_probability: f64,
    posterior: Option<f64>,
    candidate: &TwoStageMetrologyCandidateV1,
    portfolio_selected: bool,
) -> TwoStageOutcomeBranchV1 {
    let Some(posterior_probability) = posterior else {
        return TwoStageOutcomeBranchV1 {
            screen_outcome: outcome,
            screen_outcome_probability: screen_probability,
            posterior_defect_probability: None,
            decision_loss_without_confirmation: None,
            expected_decision_loss_after_confirmation: None,
            gross_confirmation_evsi: 0.0,
            net_confirmation_value: 0.0,
            action: ConfirmationActionV1::OutcomeImpossible,
            expected_confirmation_cost: 0.0,
            worst_case_confirmation_cost: 0,
        };
    };

    let high_evsi = binary_evsi(
        posterior_probability,
        candidate.confirmation,
        candidate.decision_loss,
    );
    let net_value = high_evsi.gross_evsi - candidate.confirmation_cost as f64;
    let should_confirm = portfolio_selected && net_value > 1.0e-12;
    let action = if should_confirm {
        ConfirmationActionV1::Confirm
    } else {
        ConfirmationActionV1::StopAfterScreen
    };
    let expected_confirmation_cost = if should_confirm {
        screen_probability * candidate.confirmation_cost as f64
    } else {
        0.0
    };

    TwoStageOutcomeBranchV1 {
        screen_outcome: outcome,
        screen_outcome_probability: screen_probability,
        posterior_defect_probability: Some(posterior_probability),
        decision_loss_without_confirmation: Some(high_evsi.expected_loss_without_measurement),
        expected_decision_loss_after_confirmation: Some(high_evsi.expected_loss_after_measurement),
        gross_confirmation_evsi: high_evsi.gross_evsi,
        net_confirmation_value: if should_confirm { net_value } else { 0.0 },
        action,
        expected_confirmation_cost,
        worst_case_confirmation_cost: if should_confirm {
            candidate.confirmation_cost
        } else {
            0
        },
    }
}

fn is_better_portfolio(
    candidate: &ConfirmationPortfolioScoreV1,
    incumbent: &ConfirmationPortfolioScoreV1,
) -> bool {
    const EPSILON: f64 = 1.0e-12;
    if candidate.expected_net_value > incumbent.expected_net_value + EPSILON {
        return true;
    }
    if incumbent.expected_net_value > candidate.expected_net_value + EPSILON {
        return false;
    }
    if candidate.cost != incumbent.cost {
        return candidate.cost < incumbent.cost;
    }
    candidate.ids < incumbent.ids
}

/// Plans a low-fidelity screening stage and an outcome-conditional
/// high-fidelity confirmation portfolio under a hard worst-case budget.
///
/// Screening is applied to all candidates; its fixed cost must fit the
/// screening budget. A selected candidate may invoke confirmation on positive,
/// negative or both screen outcomes, but only where posterior-conditional
/// gross EVSI exceeds the confirmation cost. Portfolio selection is exact for
/// up to 18 candidates and uses deterministic candidate-ID tie-breaking.
pub fn plan_two_stage_metrology_v1(
    policy: &TwoStageMetrologyPolicyV1,
    candidates: &[TwoStageMetrologyCandidateV1],
) -> Result<TwoStageMetrologyReportV1, TwoStageMetrologyErrorV1> {
    let canonical_candidates = validate_policy_and_candidates(policy, candidates)?;
    let candidate_packet_sha256 = json_sha256(&canonical_candidates)?;
    let policy_sha256 = json_sha256(policy)?;
    let screening_cost_used = canonical_candidates
        .iter()
        .map(|candidate| candidate.screening_cost)
        .sum::<u64>();
    let high_fidelity_only_total_cost = canonical_candidates
        .iter()
        .try_fold(0_u64, |total, candidate| total.checked_add(candidate.confirmation_cost))
        .ok_or_else(|| TwoStageMetrologyErrorV1("high-fidelity baseline cost overflow".into()))?;

    let mut preliminary = Vec::with_capacity(canonical_candidates.len());
    for candidate in &canonical_candidates {
        let screen_evsi = binary_evsi(
            candidate.prior_defect_probability,
            candidate.screening,
            candidate.decision_loss,
        );
        let positive = candidate_branch(
            ScreenOutcomeV1::Positive,
            screen_evsi.probability_positive,
            screen_evsi.posterior_positive,
            candidate,
            true,
        );
        let negative = candidate_branch(
            ScreenOutcomeV1::Negative,
            screen_evsi.probability_negative,
            screen_evsi.posterior_negative,
            candidate,
            true,
        );
        let expected_incremental_net_value =
            positive.screen_outcome_probability * positive.net_confirmation_value
                + negative.screen_outcome_probability * negative.net_confirmation_value;
        let confirmation_worst_case_cost = positive
            .worst_case_confirmation_cost
            .max(negative.worst_case_confirmation_cost);
        let expected_confirmation_cost =
            positive.expected_confirmation_cost + negative.expected_confirmation_cost;

        preliminary.push((
            screen_evsi,
            positive,
            negative,
            expected_incremental_net_value,
            expected_confirmation_cost,
            confirmation_worst_case_cost,
        ));
    }

    let subset_count = 1_usize << canonical_candidates.len();
    let mut best = ConfirmationPortfolioScoreV1 {
        indices: Vec::new(),
        cost: 0,
        expected_net_value: 0.0,
        ids: Vec::new(),
    };

    for mask in 1_usize..subset_count {
        let mut indices = Vec::new();
        let mut ids = Vec::new();
        let mut cost = 0_u64;
        let mut expected_net_value = 0.0;
        let mut valid = true;

        for index in 0..canonical_candidates.len() {
            if mask & (1_usize << index) == 0 {
                continue;
            }
            let candidate_cost = preliminary[index].5;
            if candidate_cost == 0 {
                valid = false;
                break;
            }
            match cost.checked_add(candidate_cost) {
                Some(next_cost)
                    if next_cost <= policy.confirmation_worst_case_cost_budget =>
                {
                    cost = next_cost;
                }
                _ => {
                    valid = false;
                    break;
                }
            }
            indices.push(index);
            ids.push(canonical_candidates[index].candidate_id.clone());
            expected_net_value += preliminary[index].3;
        }
        if valid {
            let score = ConfirmationPortfolioScoreV1 {
                indices,
                cost,
                expected_net_value,
                ids,
            };
            if is_better_portfolio(&score, &best) {
                best = score;
            }
        }
    }

    let selected_indices: BTreeSet<usize> = best.indices.iter().copied().collect();
    let mut candidate_plans = Vec::with_capacity(canonical_candidates.len());
    let mut no_measurement_expected_decision_loss = 0.0;
    let mut screen_only_expected_decision_loss = 0.0;
    let mut high_fidelity_only_expected_decision_loss = 0.0;
    let mut two_stage_expected_decision_loss = 0.0;
    let mut confirmation_expected_cost = 0.0;

    for (index, candidate) in canonical_candidates.iter().enumerate() {
        let (screen_evsi, _, _, _, _, _) = &preliminary[index];
        let selected = selected_indices.contains(&index);
        let positive = candidate_branch(
            ScreenOutcomeV1::Positive,
            screen_evsi.probability_positive,
            screen_evsi.posterior_positive,
            candidate,
            selected,
        );
        let negative = candidate_branch(
            ScreenOutcomeV1::Negative,
            screen_evsi.probability_negative,
            screen_evsi.posterior_negative,
            candidate,
            selected,
        );

        let no_measurement_loss =
            optimal_binary_decision_loss(candidate.prior_defect_probability, candidate.decision_loss);
        no_measurement_expected_decision_loss += no_measurement_loss;

        screen_only_expected_decision_loss +=
            screen_evsi.expected_loss_after_measurement;

        let high_only = binary_evsi(
            candidate.prior_defect_probability,
            candidate.confirmation,
            candidate.decision_loss,
        );
        high_fidelity_only_expected_decision_loss += high_only.expected_loss_after_measurement;

        let positive_loss = if positive.action == ConfirmationActionV1::Confirm {
            positive.expected_decision_loss_after_confirmation.unwrap_or(0.0)
        } else {
            positive.decision_loss_without_confirmation.unwrap_or(0.0)
        };
        let negative_loss = if negative.action == ConfirmationActionV1::Confirm {
            negative.expected_decision_loss_after_confirmation.unwrap_or(0.0)
        } else {
            negative.decision_loss_without_confirmation.unwrap_or(0.0)
        };
        two_stage_expected_decision_loss +=
            positive.screen_outcome_probability * positive_loss
                + negative.screen_outcome_probability * negative_loss;
        confirmation_expected_cost +=
            positive.expected_confirmation_cost + negative.expected_confirmation_cost;

        let confirm_on_positive = positive.action == ConfirmationActionV1::Confirm;
        let confirm_on_negative = negative.action == ConfirmationActionV1::Confirm;
        let worst_case_confirmation_cost = if confirm_on_positive || confirm_on_negative {
            candidate.confirmation_cost
        } else {
            0
        };
        let expected_incremental_net_value =
            positive.screen_outcome_probability * positive.net_confirmation_value
                + negative.screen_outcome_probability * negative.net_confirmation_value;

        candidate_plans.push(TwoStageCandidatePlanV1 {
            candidate_id: candidate.candidate_id.clone(),
            prior_defect_probability: candidate.prior_defect_probability,
            low_screen_cost: candidate.screening_cost,
            high_confirmation_cost: candidate.confirmation_cost,
            screen_positive_probability: positive.screen_outcome_probability,
            screen_negative_probability: negative.screen_outcome_probability,
            confirm_on_positive,
            confirm_on_negative,
            expected_incremental_net_value,
            expected_confirmation_cost: positive.expected_confirmation_cost
                + negative.expected_confirmation_cost,
            worst_case_confirmation_cost,
            selected_for_confirmation_portfolio: selected,
            positive_branch: positive,
            negative_branch: negative,
        });
    }

    let expected_incremental_net_value = screen_only_expected_decision_loss
        - two_stage_expected_decision_loss
        - confirmation_expected_cost;
    let total_expected_measurement_cost =
        screening_cost_used as f64 + confirmation_expected_cost;
    let total_worst_case_measurement_cost = screening_cost_used
        .checked_add(best.cost)
        .ok_or_else(|| TwoStageMetrologyErrorV1("two-stage worst-case cost overflow".into()))?;
    let selected_confirmation_candidates = best.ids.clone();

    let mut report = TwoStageMetrologyReportV1 {
        schema: TWO_STAGE_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_sha256,
        candidate_packet_sha256,
        screening_cost_budget: policy.screening_cost_budget,
        screening_cost_used,
        no_measurement_total_cost: 0,
        screen_only_total_cost: screening_cost_used,
        high_fidelity_only_total_cost,
        confirmation_worst_case_cost_budget: policy.confirmation_worst_case_cost_budget,
        confirmation_worst_case_cost_used: best.cost,
        total_worst_case_measurement_cost,
        confirmation_expected_cost,
        total_expected_measurement_cost,
        no_measurement_expected_decision_loss,
        screen_only_expected_decision_loss,
        high_fidelity_only_expected_decision_loss,
        two_stage_expected_decision_loss,
        expected_incremental_net_value,
        subsets_examined: subset_count,
        selected_confirmation_candidates,
        candidates: candidate_plans,
        report_sha256: String::new(),
        claim_limits: TwoStageClaimLimitsV1 {
            plan_does_not_execute_measurements: true,
            planned_confirmation_is_not_a_completed_measurement: true,
            qualification_gate_must_pass_before_label_promotion: true,
            no_fab_process_authority_is_granted: true,
            model_assay_parameters_are_declared_synthetic_inputs: true,
        },
    };
    report.report_sha256 = json_sha256(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        candidate_id: &str,
        prior: f64,
        screening: BinaryAssayCharacteristicsV1,
        screening_cost: u64,
        confirmation: BinaryAssayCharacteristicsV1,
        confirmation_cost: u64,
    ) -> TwoStageMetrologyCandidateV1 {
        TwoStageMetrologyCandidateV1 {
            candidate_id: candidate_id.into(),
            prior_defect_probability: prior,
            screening,
            screening_cost,
            confirmation,
            confirmation_cost,
            decision_loss: BinaryDecisionLossV1 {
                false_negative_cost: 100.0,
                false_positive_cost: 10.0,
            },
        }
    }

    fn policy(candidates: &[TwoStageMetrologyCandidateV1], confirmation_budget: u64) -> TwoStageMetrologyPolicyV1 {
        let mut canonical = candidates.to_vec();
        canonical.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
        TwoStageMetrologyPolicyV1 {
            schema: TWO_STAGE_INPUT_SCHEMA_V1.into(),
            policy_id: "synthetic-two-stage-v1".into(),
            candidate_packet_sha256: json_sha256(&canonical).unwrap(),
            screening_cost_budget: candidates
                .iter()
                .map(|candidate| candidate.screening_cost)
                .sum(),
            confirmation_worst_case_cost_budget: confirmation_budget,
            max_candidates_exact: MAX_TWO_STAGE_CANDIDATES_EXACT,
        }
    }

    fn assay(good: bool) -> BinaryAssayCharacteristicsV1 {
        if good {
            BinaryAssayCharacteristicsV1 {
                sensitivity: 0.95,
                specificity: 0.99,
            }
        } else {
            BinaryAssayCharacteristicsV1 {
                sensitivity: 0.70,
                specificity: 0.30,
            }
        }
    }

    #[test]
    fn exact_plan_replays_and_never_exceeds_worst_case_confirmation_budget() {
        let candidates = vec![
            candidate("wafer-a", 0.20, assay(true), 1, assay(true), 5),
            candidate("wafer-b", 0.05, assay(true), 1, assay(true), 5),
            candidate("wafer-c", 0.65, assay(true), 1, assay(true), 5),
        ];
        let policy = policy(&candidates, 10);
        let first = plan_two_stage_metrology_v1(&policy, &candidates).unwrap();
        let second = plan_two_stage_metrology_v1(&policy, &candidates).unwrap();

        assert_eq!(first.report_sha256, second.report_sha256);
        assert_eq!(first.screening_cost_used, 3);
        assert_eq!(first.no_measurement_total_cost, 0);
        assert_eq!(first.screen_only_total_cost, 3);
        assert_eq!(first.high_fidelity_only_total_cost, 15);
        assert!(first.confirmation_worst_case_cost_used <= 10);
        assert_eq!(
            first.total_worst_case_measurement_cost,
            first.screening_cost_used + first.confirmation_worst_case_cost_used
        );
        assert_eq!(first.subsets_examined, 8);
        assert_eq!(first.candidates.len(), 3);
        assert!(first.expected_incremental_net_value >= 0.0);
        assert!(first.claim_limits.plan_does_not_execute_measurements);
    }

    #[test]
    fn no_confirmation_when_high_fidelity_assay_is_uninformative() {
        let candidates = vec![candidate(
            "wafer-uninformative",
            0.4,
            assay(true),
            1,
            assay(false),
            1,
        )];
        let policy = policy(&candidates, 10);
        let report = plan_two_stage_metrology_v1(&policy, &candidates).unwrap();
        assert!(report.selected_confirmation_candidates.is_empty());
        assert_eq!(report.confirmation_worst_case_cost_used, 0);
        assert_eq!(report.candidates[0].confirm_on_positive, false);
        assert_eq!(report.candidates[0].confirm_on_negative, false);
    }

    #[test]
    fn no_confirmation_when_confirmation_cost_exceeds_gross_evsi() {
        let candidates = vec![candidate(
            "wafer-expensive",
            0.3,
            assay(true),
            1,
            assay(true),
            1_000_000,
        )];
        let policy = policy(&candidates, 1_000_000);
        let report = plan_two_stage_metrology_v1(&policy, &candidates).unwrap();
        assert!(report.selected_confirmation_candidates.is_empty());
        assert_eq!(report.confirmation_expected_cost, 0.0);
    }

    #[test]
    fn exact_worst_case_budget_is_not_replaced_by_expected_cost() {
        let candidates = vec![
            candidate("wafer-a", 0.25, assay(true), 1, assay(true), 10),
            candidate("wafer-b", 0.30, assay(true), 1, assay(true), 10),
        ];
        let policy = policy(&candidates, 10);
        let report = plan_two_stage_metrology_v1(&policy, &candidates).unwrap();
        assert!(report.confirmation_worst_case_cost_used <= 10);
        assert!(report.selected_confirmation_candidates.len() <= 1);
    }

    #[test]
    fn uninformative_screen_does_not_claim_information_gain() {
        let candidates = vec![candidate(
            "wafer-uninformative-screen",
            0.4,
            assay(false),
            1,
            assay(true),
            5,
        )];
        let policy = policy(&candidates, 5);
        let report = plan_two_stage_metrology_v1(&policy, &candidates).unwrap();
        assert!((report.screen_only_expected_decision_loss
            - report.no_measurement_expected_decision_loss)
            .abs()
            < 1.0e-10);
    }

    #[test]
    fn reordered_candidates_produce_identical_report() {
        let first = vec![
            candidate("wafer-a", 0.2, assay(true), 1, assay(true), 5),
            candidate("wafer-b", 0.6, assay(true), 1, assay(true), 5),
        ];
        let second = vec![first[1].clone(), first[0].clone()];
        let first_policy = policy(&first, 5);
        let second_policy = policy(&second, 5);
        assert_eq!(
            plan_two_stage_metrology_v1(&first_policy, &first).unwrap(),
            plan_two_stage_metrology_v1(&second_policy, &second).unwrap()
        );
    }

    #[test]
    fn screening_budget_and_candidate_packet_digest_are_enforced() {
        let candidates = vec![candidate("wafer-a", 0.2, assay(true), 2, assay(true), 5)];
        let mut policy = policy(&candidates, 5);
        policy.screening_cost_budget = 1;
        assert!(plan_two_stage_metrology_v1(&policy, &candidates).is_err());

        let mut policy = policy(&candidates, 5);
        policy.candidate_packet_sha256 = "0".repeat(64);
        assert!(plan_two_stage_metrology_v1(&policy, &candidates).is_err());
    }

    #[test]
    fn invalid_probabilities_costs_and_duplicate_ids_are_rejected() {
        let mut candidates = vec![candidate("wafer-a", 0.2, assay(true), 1, assay(true), 5)];
        candidates[0].screening.sensitivity = 1.2;
        assert!(plan_two_stage_metrology_v1(&policy(&candidates, 5), &candidates).is_err());

        let mut candidates = vec![
            candidate("wafer-a", 0.2, assay(true), 1, assay(true), 5),
            candidate("wafer-a", 0.3, assay(true), 1, assay(true), 5),
        ];
        assert!(plan_two_stage_metrology_v1(&policy(&candidates, 10), &candidates).is_err());
        candidates[0].screening_cost = 0;
        assert!(plan_two_stage_metrology_v1(&policy(&candidates, 10), &candidates).is_err());
    }

    #[test]
    fn zero_probability_screen_outcome_is_explicitly_impossible() {
        let candidates = vec![candidate(
            "wafer-perfect-prior",
            1.0,
            BinaryAssayCharacteristicsV1 {
                sensitivity: 1.0,
                specificity: 1.0,
            },
            1,
            assay(true),
            5,
        )];
        let policy = policy(&candidates, 5);
        let report = plan_two_stage_metrology_v1(&policy, &candidates).unwrap();
        assert_eq!(
            report.candidates[0].negative_branch.action,
            ConfirmationActionV1::OutcomeImpossible
        );
        assert_eq!(
            report.candidates[0].negative_branch.posterior_defect_probability,
            None
        );
    }
}
