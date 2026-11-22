// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic budgeted multi-fidelity acquisition baseline.
//!
//! This selects which candidates should receive expensive evaluation. It is an
//! experiment scheduler, not a design approver, simulator, or source of truth.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

pub const INPUT_SCHEMA_V1: &str = "luminous.semi-bench.mf-acquisition.v1";
pub const OUTPUT_SCHEMA_V1: &str = "luminous.semi-bench.mf-acquisition-report.v1";
pub const MAX_CANDIDATES_EXACT: usize = 18;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcquisitionCandidateV1 {
    pub candidate_id: String,
    /// Cost in the declared budget unit.
    pub estimated_cost: u64,
    /// Expected feasible hypervolume gain from an independent prediction packet.
    pub expected_feasible_hv_gain: f64,
    /// Calibrated epistemic standard deviation in the same hypervolume units.
    pub epistemic_stddev: f64,
    /// Relevance to the current Pareto frontier, constrained to [0, 1].
    pub frontier_relevance: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcquisitionSubjectV1 {
    pub schema: String,
    pub policy_id: String,
    pub prediction_packet_sha256: String,
    pub model_sha256: String,
    pub objective_spec_sha256: String,
    pub budget_unit: String,
    pub total_cost_budget: u64,
    /// Exploration bonus per unit of epistemic uncertainty.
    pub exploration_weight: f64,
    pub candidates: Vec<AcquisitionCandidateV1>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct AcquisitionError(pub String);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectedCandidateV1 {
    pub candidate_id: String,
    pub estimated_cost: u64,
    pub expected_feasible_hv_gain: f64,
    pub exploration_bonus: f64,
    pub acquisition_utility: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcquisitionClaimLimitsV1 {
    pub selection_is_not_design_approval: bool,
    pub selection_does_not_execute_high_fidelity_simulation: bool,
    pub selected_candidates_require_independent_final_evaluation: bool,
    pub does_not_establish_physical_process_or_fabrication_authority: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcquisitionReportV1 {
    pub schema: String,
    pub policy_id: String,
    pub prediction_packet_sha256: String,
    pub model_sha256: String,
    pub objective_spec_sha256: String,
    pub budget_unit: String,
    pub total_cost_budget: u64,
    pub used_cost_budget: u64,
    pub subsets_examined: usize,
    pub total_acquisition_utility: f64,
    pub selected: Vec<SelectedCandidateV1>,
    pub claim_limits: AcquisitionClaimLimitsV1,
}

fn require_text(value: &str, field: &str) -> Result<(), AcquisitionError> {
    if value.trim().is_empty() {
        return Err(AcquisitionError(format!(
            "{field}: expected non-empty string"
        )));
    }
    Ok(())
}

fn require_sha256(value: &str, field: &str) -> Result<(), AcquisitionError> {
    require_text(value, field)?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(AcquisitionError(format!(
            "{field}: expected lowercase SHA-256 hex"
        )));
    }
    Ok(())
}

fn finite(value: f64, field: &str) -> Result<(), AcquisitionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AcquisitionError(format!("{field}: expected finite number")))
    }
}

fn rounded(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}

fn validate_subject(subject: &AcquisitionSubjectV1) -> Result<(), AcquisitionError> {
    if subject.schema != INPUT_SCHEMA_V1 {
        return Err(AcquisitionError(
            "schema: unsupported or missing acquisition subject".to_string(),
        ));
    }
    require_text(&subject.policy_id, "policy_id")?;
    require_sha256(
        &subject.prediction_packet_sha256,
        "prediction_packet_sha256",
    )?;
    require_sha256(&subject.model_sha256, "model_sha256")?;
    require_sha256(&subject.objective_spec_sha256, "objective_spec_sha256")?;
    require_text(&subject.budget_unit, "budget_unit")?;
    finite(subject.exploration_weight, "exploration_weight")?;
    if subject.exploration_weight < 0.0 {
        return Err(AcquisitionError(
            "exploration_weight: must be non-negative".to_string(),
        ));
    }
    if subject.candidates.is_empty() {
        return Err(AcquisitionError(
            "candidates: expected non-empty array".to_string(),
        ));
    }
    if subject.candidates.len() > MAX_CANDIDATES_EXACT {
        return Err(AcquisitionError(format!(
            "candidates: exact subset oracle is limited to {MAX_CANDIDATES_EXACT}"
        )));
    }

    let mut seen = BTreeSet::new();
    for (index, candidate) in subject.candidates.iter().enumerate() {
        require_text(
            &candidate.candidate_id,
            &format!("candidates[{index}].candidate_id"),
        )?;
        if !seen.insert(candidate.candidate_id.as_str()) {
            return Err(AcquisitionError(format!(
                "candidates[{index}].candidate_id: duplicate candidate identity"
            )));
        }
        if candidate.estimated_cost == 0 {
            return Err(AcquisitionError(format!(
                "candidates[{index}].estimated_cost: must be greater than zero"
            )));
        }
        finite(
            candidate.expected_feasible_hv_gain,
            &format!("candidates[{index}].expected_feasible_hv_gain"),
        )?;
        finite(
            candidate.epistemic_stddev,
            &format!("candidates[{index}].epistemic_stddev"),
        )?;
        if candidate.epistemic_stddev < 0.0 {
            return Err(AcquisitionError(format!(
                "candidates[{index}].epistemic_stddev: must be non-negative"
            )));
        }
        finite(
            candidate.frontier_relevance,
            &format!("candidates[{index}].frontier_relevance"),
        )?;
        if !(0.0..=1.0).contains(&candidate.frontier_relevance) {
            return Err(AcquisitionError(format!(
                "candidates[{index}].frontier_relevance: must be in [0, 1]"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct SubsetScore {
    selected_indices: Vec<usize>,
    cost: u64,
    utility: f64,
    ids: Vec<String>,
}

fn is_better(candidate: &SubsetScore, incumbent: &SubsetScore) -> bool {
    const EPSILON: f64 = 1e-12;
    if candidate.utility > incumbent.utility + EPSILON {
        return true;
    }
    if incumbent.utility > candidate.utility + EPSILON {
        return false;
    }
    if candidate.cost != incumbent.cost {
        return candidate.cost < incumbent.cost;
    }
    candidate.ids < incumbent.ids
}

/// Chooses the exact maximum-utility candidate subset within the declared
/// cost budget. For candidate i:
///
/// utility_i = expected_feasible_hv_gain_i
///           + exploration_weight * epistemic_stddev_i * frontier_relevance_i
///
/// Expected feasible gain must already reflect the upstream feasibility/risk
/// model. Selection only schedules independent high-fidelity evaluation.
pub fn select_high_fidelity_candidates_v1(
    subject: &AcquisitionSubjectV1,
) -> Result<AcquisitionReportV1, AcquisitionError> {
    validate_subject(subject)?;

    let mut candidates: Vec<&AcquisitionCandidateV1> = subject.candidates.iter().collect();
    candidates.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let mut scores = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        let exploration_bonus =
            subject.exploration_weight * candidate.epistemic_stddev * candidate.frontier_relevance;
        let utility = candidate.expected_feasible_hv_gain + exploration_bonus;
        scores.push((exploration_bonus, utility));
    }

    let mut best = SubsetScore {
        selected_indices: Vec::new(),
        cost: 0,
        utility: 0.0,
        ids: Vec::new(),
    };
    let subsets_examined = 1usize << candidates.len();

    for mask in 1usize..subsets_examined {
        let mut selected_indices = Vec::new();
        let mut ids = Vec::new();
        let mut cost = 0u64;
        let mut utility = 0.0f64;
        let mut valid = true;

        for (index, candidate) in candidates.iter().enumerate() {
            if mask & (1usize << index) == 0 {
                continue;
            }
            match cost.checked_add(candidate.estimated_cost) {
                Some(next_cost) if next_cost <= subject.total_cost_budget => cost = next_cost,
                _ => {
                    valid = false;
                    break;
                }
            }
            selected_indices.push(index);
            ids.push(candidate.candidate_id.clone());
            utility += scores[index].1;
        }
        if valid {
            let score = SubsetScore {
                selected_indices,
                cost,
                utility,
                ids,
            };
            if is_better(&score, &best) {
                best = score;
            }
        }
    }

    let selected: Vec<SelectedCandidateV1> = best
        .selected_indices
        .iter()
        .map(|index| SelectedCandidateV1 {
            candidate_id: candidates[*index].candidate_id.clone(),
            estimated_cost: candidates[*index].estimated_cost,
            expected_feasible_hv_gain: rounded(candidates[*index].expected_feasible_hv_gain),
            exploration_bonus: rounded(scores[*index].0),
            acquisition_utility: rounded(scores[*index].1),
        })
        .collect();

    Ok(AcquisitionReportV1 {
        schema: OUTPUT_SCHEMA_V1.to_string(),
        policy_id: subject.policy_id.clone(),
        prediction_packet_sha256: subject.prediction_packet_sha256.clone(),
        model_sha256: subject.model_sha256.clone(),
        objective_spec_sha256: subject.objective_spec_sha256.clone(),
        budget_unit: subject.budget_unit.clone(),
        total_cost_budget: subject.total_cost_budget,
        used_cost_budget: best.cost,
        subsets_examined,
        total_acquisition_utility: rounded(best.utility),
        selected,
        claim_limits: AcquisitionClaimLimitsV1 {
            selection_is_not_design_approval: true,
            selection_does_not_execute_high_fidelity_simulation: true,
            selected_candidates_require_independent_final_evaluation: true,
            does_not_establish_physical_process_or_fabrication_authority: true,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const INPUT: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/mf-acquisition-known-answer-v1.json"
    );
    const EXPECTED: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/mf-acquisition-known-answer-v1.expected.json"
    );

    fn known_answer() -> AcquisitionSubjectV1 {
        serde_json::from_str(INPUT).expect("known-answer acquisition subject must deserialize")
    }

    #[test]
    fn exact_subset_matches_frozen_known_answer() {
        let report = select_high_fidelity_candidates_v1(&known_answer()).unwrap();
        let actual = serde_json::to_value(report).unwrap();
        let expected: Value = serde_json::from_str(EXPECTED).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn candidate_input_order_does_not_change_selection() {
        let mut subject = known_answer();
        let expected = select_high_fidelity_candidates_v1(&subject).unwrap();
        subject.candidates.reverse();
        assert_eq!(select_high_fidelity_candidates_v1(&subject).unwrap(), expected);
    }

    #[test]
    fn exact_budget_is_never_exceeded() {
        let mut subject = known_answer();
        subject.total_cost_budget = 12;
        let report = select_high_fidelity_candidates_v1(&subject).unwrap();
        assert!(report.used_cost_budget <= 12);
        assert!(!report.selected.iter().any(|item| item.candidate_id == "B"));
    }

    #[test]
    fn empty_selection_is_valid_when_budget_is_insufficient() {
        let mut subject = known_answer();
        subject.total_cost_budget = 1;
        let report = select_high_fidelity_candidates_v1(&subject).unwrap();
        assert!(report.selected.is_empty());
        assert_eq!(report.used_cost_budget, 0);
        assert_eq!(report.total_acquisition_utility, 0.0);
    }

    #[test]
    fn duplicate_candidate_identity_is_rejected() {
        let mut subject = known_answer();
        subject.candidates[1].candidate_id = subject.candidates[0].candidate_id.clone();
        assert!(
            select_high_fidelity_candidates_v1(&subject)
                .unwrap_err()
                .to_string()
                .contains("duplicate candidate identity")
        );
    }

    #[test]
    fn more_than_eighteen_candidates_is_rejected() {
        let mut subject = known_answer();
        subject.candidates = (0..19)
            .map(|index| AcquisitionCandidateV1 {
                candidate_id: format!("C-{index:02}"),
                estimated_cost: 1,
                expected_feasible_hv_gain: 0.01,
                epistemic_stddev: 0.01,
                frontier_relevance: 0.5,
            })
            .collect();
        assert!(
            select_high_fidelity_candidates_v1(&subject)
                .unwrap_err()
                .to_string()
                .contains("limited to 18")
        );
    }

    #[test]
    fn malformed_relevance_and_negative_uncertainty_are_rejected() {
        let mut subject = known_answer();
        subject.candidates[0].frontier_relevance = 1.1;
        assert!(select_high_fidelity_candidates_v1(&subject).is_err());

        let mut subject = known_answer();
        subject.candidates[0].epistemic_stddev = -0.01;
        assert!(select_high_fidelity_candidates_v1(&subject).is_err());
    }

    #[test]
    fn input_digests_and_exploration_weight_are_validated() {
        let mut subject = known_answer();
        subject.model_sha256 = "invalid".to_string();
        assert!(select_high_fidelity_candidates_v1(&subject).is_err());

        let mut subject = known_answer();
        subject.exploration_weight = -1.0;
        assert!(select_high_fidelity_candidates_v1(&subject).is_err());
    }
}
