// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Report-only agency-budget selection for discrete shared-autonomy research.
//!
//! This deterministic high-level candidate selector is for simulation and offline
//! replay only. It does not send commands, enforce mechanical safety, authenticate
//! intent probabilities, or measure subjective agency. Its Agency Loss Index (ALI)
//! is a model-side proxy only.
//!
//! Motivated by agency-constrained shared autonomy for prosthetic grasp selection
//! (Zhang et al., 2026): https://journals.sagepub.com/doi/10.1177/10711813261485183

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Current schema version for agency-budget requests and decisions.
pub const AGENCY_BUDGET_SCHEMA_VERSION: u16 = 1;
/// Probability-mass validation tolerance for input distributions.
pub const PROBABILITY_SUM_TOLERANCE_V1: f32 = 1.0e-4;
/// Numeric allowance used only for agency-envelope comparison.
pub const AGENCY_ENVELOPE_EPSILON_V1: f32 = 1.0e-6;

/// Versioned request for high-level agency-envelope evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgencyBudgetRequestV1 {
    /// Must equal AGENCY_BUDGET_SCHEMA_VERSION.
    pub schema_version: u16,
    /// Stable, non-empty, unique candidate IDs in distribution order.
    pub candidate_ids: Vec<String>,
    /// User-intent probability for each candidate, in the same order.
    pub user_intent_probabilities: Vec<f32>,
    /// Autonomy-preference probability for each candidate, in the same order.
    pub autonomy_preference_probabilities: Vec<f32>,
    /// Maximum tolerated model-side Agency Loss Index, in [0, 1].
    ///
    /// This is a research control, not a universal human-factors threshold.
    pub agency_budget: f32,
}

/// The distribution that failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgencyDistributionV1 {
    /// User-intent distribution.
    UserIntent,
    /// Autonomy-preference distribution.
    AutonomyPreference,
}

/// Why a probability-like numeric value failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericValidationErrorV1 {
    /// The value was NaN or positive/negative infinity.
    NonFinite,
    /// The finite value was outside [0, 1].
    OutsideUnitInterval,
}

/// Stable input-validation errors. An error contains no selected candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgencyBudgetErrorV1 {
    /// Unsupported request schema.
    UnsupportedSchemaVersion { received: u16 },
    /// No candidate actions were supplied.
    NoCandidates,
    /// Candidate IDs and probability arrays have different lengths.
    CandidateCountMismatch {
        candidate_count: usize,
        user_probability_count: usize,
        autonomy_probability_count: usize,
    },
    /// A candidate ID is empty or has surrounding whitespace.
    InvalidCandidateId { index: usize },
    /// Candidate IDs must be unique.
    DuplicateCandidateId { candidate_id: String },
    /// The agency budget is not finite and within [0, 1].
    InvalidAgencyBudget { reason: NumericValidationErrorV1 },
    /// A probability is not finite and within [0, 1].
    InvalidProbability {
        distribution: AgencyDistributionV1,
        index: usize,
        reason: NumericValidationErrorV1,
    },
    /// The distribution's probability mass is not sufficiently close to one.
    ProbabilityMassMismatch {
        distribution: AgencyDistributionV1,
        observed_sum: f32,
    },
}

/// How the selected candidate relates to the unconstrained autonomy proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgencySelectionStatusV1 {
    /// The highest autonomy-preference candidate was inside the agency envelope.
    AutonomyProposalAdmitted,
    /// The unconstrained proposal exceeded the budget; an in-envelope candidate
    /// with the greatest autonomy preference was selected.
    AutonomyProposalConstrained,
    /// Defensive fallback used only if the computed envelope is unexpectedly empty.
    UserOnlyFallback,
}

/// Candidate-level values included in an offline replay receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgencyCandidateAssessmentV1 {
    /// Stable candidate ID.
    pub candidate_id: String,
    /// User-intent probability.
    pub user_intent_probability: f32,
    /// Autonomy-preference probability.
    pub autonomy_preference_probability: f32,
    /// max(user-intent probability) minus this candidate's user-intent probability,
    /// clamped at zero. This is a model-derived proxy, not subjective agency.
    pub agency_loss_index: f32,
    /// Whether this candidate is inside the request's agency envelope.
    pub within_agency_envelope: bool,
}

/// Deterministic report from the discrete agency-budget selector.
///
/// A selected candidate is for simulation/replay only, not an actuator command
/// or physical-safety authorization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgencyBudgetDecisionV1 {
    /// Output schema version.
    pub schema_version: u16,
    /// Research budget supplied in the request.
    pub agency_budget: f32,
    /// User-only argmax candidate, with stable first-candidate tie breaking.
    pub user_only_candidate_id: String,
    /// Unconstrained autonomy argmax candidate, with stable tie breaking.
    pub unconstrained_autonomy_candidate_id: String,
    /// Candidate selected for simulation/replay.
    pub selected_candidate_id: String,
    /// Selected candidate's index in the request arrays.
    pub selected_index: usize,
    /// Model-side ALI of the selected candidate.
    pub selected_agency_loss_index: f32,
    /// Whether autonomy was admitted, constrained, or fell back to user-only.
    pub status: AgencySelectionStatusV1,
    /// Per-candidate assessments in exact input order.
    pub candidate_assessments: Vec<AgencyCandidateAssessmentV1>,
}

/// Evaluate a discrete action proposal under a model-side agency-loss budget.
///
/// The user-only candidate is the first maximum of the user-intent distribution.
/// The unconstrained proposal is the first maximum of the autonomy-preference
/// distribution. If that proposal is outside the envelope, this set-based method
/// selects the candidate with greatest autonomy preference among candidates whose
/// ALI is within budget. Exact ties preserve the earliest input candidate.
///
/// For valid distributions and a budget in [0, 1], the user-only argmax always has
/// ALI zero; an empty envelope should therefore be unreachable. A defensive
/// user-only fallback remains. Invalid inputs return an error and select nothing.
///
/// This function is for deterministic software experiments only. Real systems must
/// independently validate calibration, preserve an accessible stop path, and use
/// their platform-specific safety controller.
pub fn evaluate_agency_budget(
    request: &AgencyBudgetRequestV1,
) -> Result<AgencyBudgetDecisionV1, AgencyBudgetErrorV1> {
    if request.schema_version != AGENCY_BUDGET_SCHEMA_VERSION {
        return Err(AgencyBudgetErrorV1::UnsupportedSchemaVersion {
            received: request.schema_version,
        });
    }
    if request.candidate_ids.is_empty() {
        return Err(AgencyBudgetErrorV1::NoCandidates);
    }

    let count = request.candidate_ids.len();
    if request.user_intent_probabilities.len() != count
        || request.autonomy_preference_probabilities.len() != count
    {
        return Err(AgencyBudgetErrorV1::CandidateCountMismatch {
            candidate_count: count,
            user_probability_count: request.user_intent_probabilities.len(),
            autonomy_probability_count: request.autonomy_preference_probabilities.len(),
        });
    }

    let mut seen = HashSet::with_capacity(count);
    for (index, candidate_id) in request.candidate_ids.iter().enumerate() {
        if candidate_id.trim().is_empty() || candidate_id.trim() != candidate_id {
            return Err(AgencyBudgetErrorV1::InvalidCandidateId { index });
        }
        if !seen.insert(candidate_id.as_str()) {
            return Err(AgencyBudgetErrorV1::DuplicateCandidateId {
                candidate_id: candidate_id.clone(),
            });
        }
    }

    validate_unit_interval(request.agency_budget)
        .map_err(|reason| AgencyBudgetErrorV1::InvalidAgencyBudget { reason })?;
    validate_distribution(
        &request.user_intent_probabilities,
        AgencyDistributionV1::UserIntent,
    )?;
    validate_distribution(
        &request.autonomy_preference_probabilities,
        AgencyDistributionV1::AutonomyPreference,
    )?;

    let user_only_index = first_argmax(&request.user_intent_probabilities)
        .expect("validated request has at least one candidate");
    let autonomy_index = first_argmax(&request.autonomy_preference_probabilities)
        .expect("validated request has at least one candidate");
    let top_user_probability = request.user_intent_probabilities[user_only_index];

    let candidate_assessments: Vec<AgencyCandidateAssessmentV1> = request
        .candidate_ids
        .iter()
        .enumerate()
        .map(|(index, candidate_id)| {
            let agency_loss_index =
                (top_user_probability - request.user_intent_probabilities[index]).max(0.0);
            AgencyCandidateAssessmentV1 {
                candidate_id: candidate_id.clone(),
                user_intent_probability: request.user_intent_probabilities[index],
                autonomy_preference_probability: request.autonomy_preference_probabilities[index],
                agency_loss_index,
                within_agency_envelope: agency_loss_index
                    <= request.agency_budget + AGENCY_ENVELOPE_EPSILON_V1,
            }
        })
        .collect();

    let eligible: Vec<bool> = candidate_assessments
        .iter()
        .map(|assessment| assessment.within_agency_envelope)
        .collect();

    // Replaces a current winner only on strict improvement, preserving first-on-tie.
    let (selected_index, status) = match first_argmax_eligible(
        &request.autonomy_preference_probabilities,
        &eligible,
    ) {
        Some(index) if index == autonomy_index => {
            (index, AgencySelectionStatusV1::AutonomyProposalAdmitted)
        }
        Some(index) => (index, AgencySelectionStatusV1::AutonomyProposalConstrained),
        None => (user_only_index, AgencySelectionStatusV1::UserOnlyFallback),
    };

    Ok(AgencyBudgetDecisionV1 {
        schema_version: AGENCY_BUDGET_SCHEMA_VERSION,
        agency_budget: request.agency_budget,
        user_only_candidate_id: request.candidate_ids[user_only_index].clone(),
        unconstrained_autonomy_candidate_id: request.candidate_ids[autonomy_index].clone(),
        selected_candidate_id: request.candidate_ids[selected_index].clone(),
        selected_index,
        selected_agency_loss_index: candidate_assessments[selected_index].agency_loss_index,
        status,
        candidate_assessments,
    })
}

fn validate_distribution(
    probabilities: &[f32],
    distribution: AgencyDistributionV1,
) -> Result<(), AgencyBudgetErrorV1> {
    let mut sum = 0.0_f32;
    for (index, probability) in probabilities.iter().copied().enumerate() {
        validate_unit_interval(probability).map_err(|reason| {
            AgencyBudgetErrorV1::InvalidProbability {
                distribution,
                index,
                reason,
            }
        })?;
        sum += probability;
    }
    if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE_V1 {
        return Err(AgencyBudgetErrorV1::ProbabilityMassMismatch {
            distribution,
            observed_sum: sum,
        });
    }
    Ok(())
}

fn validate_unit_interval(value: f32) -> Result<(), NumericValidationErrorV1> {
    if !value.is_finite() {
        Err(NumericValidationErrorV1::NonFinite)
    } else if !(0.0..=1.0).contains(&value) {
        Err(NumericValidationErrorV1::OutsideUnitInterval)
    } else {
        Ok(())
    }
}

fn first_argmax(values: &[f32]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (index, value) in values.iter().enumerate() {
        if best.is_none_or(|best_index| *value > values[best_index]) {
            best = Some(index);
        }
    }
    best
}

fn first_argmax_eligible(values: &[f32], eligible: &[bool]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (index, value) in values.iter().enumerate() {
        if eligible.get(index).copied().unwrap_or(false)
            && best.is_none_or(|best_index| *value > values[best_index])
        {
            best = Some(index);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(user: Vec<f32>, autonomy: Vec<f32>, agency_budget: f32) -> AgencyBudgetRequestV1 {
        let candidate_ids = (0..user.len())
            .map(|index| format!("candidate-{index}"))
            .collect();
        AgencyBudgetRequestV1 {
            schema_version: AGENCY_BUDGET_SCHEMA_VERSION,
            candidate_ids,
            user_intent_probabilities: user,
            autonomy_preference_probabilities: autonomy,
            agency_budget,
        }
    }

    fn compositions_of_ten() -> Vec<Vec<f32>> {
        let mut distributions = Vec::new();
        for first in 0..=10 {
            for second in 0..=(10 - first) {
                let third = 10 - first - second;
                distributions.push(vec![
                    first as f32 / 10.0,
                    second as f32 / 10.0,
                    third as f32 / 10.0,
                ]);
            }
        }
        distributions
    }

    #[test]
    fn zero_budget_keeps_selection_at_top_user_probability() {
        let result = evaluate_agency_budget(&request(
            vec![0.7, 0.2, 0.1],
            vec![0.0, 1.0, 0.0],
            0.0,
        ))
        .expect("valid distributions");
        assert_eq!(result.user_only_candidate_id, "candidate-0");
        assert_eq!(result.selected_candidate_id, "candidate-0");
        assert_eq!(result.status, AgencySelectionStatusV1::AutonomyProposalConstrained);
        assert_eq!(result.selected_agency_loss_index, 0.0);
    }

    #[test]
    fn unconstrained_proposal_is_admitted_when_inside_budget() {
        let result = evaluate_agency_budget(&request(
            vec![0.5, 0.3, 0.2],
            vec![0.1, 0.8, 0.1],
            0.25,
        ))
        .expect("valid distributions");
        assert_eq!(result.unconstrained_autonomy_candidate_id, "candidate-1");
        assert_eq!(result.selected_candidate_id, "candidate-1");
        assert_eq!(result.status, AgencySelectionStatusV1::AutonomyProposalAdmitted);
        assert!(result.selected_agency_loss_index <= 0.25 + AGENCY_ENVELOPE_EPSILON_V1);
    }

    #[test]
    fn autonomy_proposal_outside_budget_is_constrained_to_best_feasible_candidate() {
        let result = evaluate_agency_budget(&request(
            vec![0.6, 0.3, 0.1],
            vec![0.05, 0.1, 0.85],
            0.25,
        ))
        .expect("valid distributions");
        assert_eq!(result.unconstrained_autonomy_candidate_id, "candidate-2");
        assert_eq!(result.selected_candidate_id, "candidate-1");
        assert_eq!(result.status, AgencySelectionStatusV1::AutonomyProposalConstrained);
        assert!(result.selected_agency_loss_index <= 0.25 + AGENCY_ENVELOPE_EPSILON_V1);
    }

    #[test]
    fn ties_preserve_first_candidate_order() {
        let result = evaluate_agency_budget(&request(
            vec![0.4, 0.4, 0.2],
            vec![0.1, 0.45, 0.45],
            0.0,
        ))
        .expect("valid distributions");
        // Two candidates tie for highest user probability, so both have ALI zero.
        assert_eq!(result.selected_candidate_id, "candidate-1");
        assert_eq!(result.selected_agency_loss_index, 0.0);
    }

    #[test]
    fn probability_mass_mismatch_is_rejected() {
        let error = evaluate_agency_budget(&request(
            vec![0.7, 0.2, 0.2],
            vec![0.2, 0.3, 0.5],
            0.2,
        ))
        .expect_err("malformed user distribution must be rejected");
        assert!(matches!(
            error,
            AgencyBudgetErrorV1::ProbabilityMassMismatch {
                distribution: AgencyDistributionV1::UserIntent,
                ..
            }
        ));
    }

    #[test]
    fn non_finite_probability_is_rejected_without_selecting_a_candidate() {
        let error = evaluate_agency_budget(&request(
            vec![f32::NAN, 0.3, 0.7],
            vec![0.2, 0.3, 0.5],
            0.2,
        ))
        .expect_err("NaN in intent distribution must fail closed");
        assert!(matches!(
            error,
            AgencyBudgetErrorV1::InvalidProbability {
                distribution: AgencyDistributionV1::UserIntent,
                index: 0,
                reason: NumericValidationErrorV1::NonFinite
            }
        ));
    }

    #[test]
    fn length_mismatch_duplicate_ids_and_invalid_budget_are_rejected() {
        let mut mismatch = request(vec![0.5, 0.5], vec![0.5, 0.5], 0.2);
        mismatch.candidate_ids.pop();
        assert!(matches!(
            evaluate_agency_budget(&mismatch),
            Err(AgencyBudgetErrorV1::CandidateCountMismatch { .. })
        ));

        let mut duplicate = request(vec![0.5, 0.5], vec![0.5, 0.5], 0.2);
        duplicate.candidate_ids[1] = duplicate.candidate_ids[0].clone();
        assert!(matches!(
            evaluate_agency_budget(&duplicate),
            Err(AgencyBudgetErrorV1::DuplicateCandidateId { .. })
        ));

        assert!(matches!(
            evaluate_agency_budget(&request(
                vec![0.5, 0.5],
                vec![0.5, 0.5],
                f32::INFINITY
            )),
            Err(AgencyBudgetErrorV1::InvalidAgencyBudget {
                reason: NumericValidationErrorV1::NonFinite
            })
        ));
    }

    #[test]
    fn unsupported_schema_is_rejected() {
        let mut input = request(vec![0.5, 0.5], vec![0.5, 0.5], 0.2);
        input.schema_version += 1;
        assert!(matches!(
            evaluate_agency_budget(&input),
            Err(AgencyBudgetErrorV1::UnsupportedSchemaVersion { .. })
        ));
    }

    #[test]
    fn exhaustive_three_candidate_sweep_never_exceeds_budget() {
        let distributions = compositions_of_ten();
        let budgets = [0.0_f32, 0.1, 0.2, 0.5, 1.0];
        let mut evaluated = 0_usize;
        for user in &distributions {
            for autonomy in &distributions {
                for budget in budgets {
                    let result = evaluate_agency_budget(&request(
                        user.clone(),
                        autonomy.clone(),
                        budget,
                    ))
                    .expect("composition distributions are valid");

                    assert!(
                        result.selected_agency_loss_index <= budget + AGENCY_ENVELOPE_EPSILON_V1,
                        "selected ALI {} exceeded budget {} for user={user:?}, autonomy={autonomy:?}",
                        result.selected_agency_loss_index,
                        budget
                    );
                    assert!(
                        result.candidate_assessments[result.selected_index].within_agency_envelope
                    );
                    let max_user = user.iter().copied().fold(0.0_f32, f32::max);
                    let expected_ali = (max_user - user[result.selected_index]).max(0.0);
                    assert!((result.selected_agency_loss_index - expected_ali).abs() <= 1.0e-6);

                    let best_feasible_preference = result
                        .candidate_assessments
                        .iter()
                        .filter(|candidate| candidate.within_agency_envelope)
                        .map(|candidate| candidate.autonomy_preference_probability)
                        .fold(-1.0_f32, f32::max);
                    assert!(
                        (result.candidate_assessments[result.selected_index]
                            .autonomy_preference_probability
                            - best_feasible_preference)
                            .abs()
                            <= 1.0e-6,
                        "selection was not autonomy-optimal inside envelope"
                    );
                    if budget == 1.0 {
                        assert_eq!(
                            result.selected_candidate_id,
                            result.unconstrained_autonomy_candidate_id
                        );
                    }
                    evaluated += 1;
                }
            }
        }
        // 66 user distributions × 66 autonomy distributions × 5 budgets.
        assert_eq!(evaluated, 21_780);
    }

    #[test]
    fn serialized_decisions_are_deterministic() {
        let input = request(vec![0.6, 0.3, 0.1], vec![0.05, 0.1, 0.85], 0.25);
        let expected = evaluate_agency_budget(&input).expect("valid distributions");
        let expected_bytes = serde_json::to_vec(&expected).expect("decision serializes");
        for _ in 0..1_000 {
            let replay = evaluate_agency_budget(&input).expect("valid distributions");
            assert_eq!(replay, expected);
            assert_eq!(serde_json::to_vec(&replay).expect("decision serializes"), expected_bytes);
        }
    }
}
