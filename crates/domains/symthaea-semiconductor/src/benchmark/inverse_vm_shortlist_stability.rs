// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario stability audit for evidence-qualified inverse-VM shortlists.
//!
//! Scenario frequencies are stress-test frequencies, not probabilities unless
//! the scenario generator is separately calibrated. Every scenario is replayed
//! through the same hard admission and receipt verification path as the base
//! shortlist. This module does not change candidates or authorize actions.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::inverse_vm_shortlist::{
    CandidateScreenStateV1, InverseVmShortlistErrorV1, InverseVmShortlistInputV1,
    InverseVmShortlistReportV1, evaluate_inverse_vm_shortlist_v1,
    seal_inverse_vm_candidate_v1, seal_inverse_vm_shortlist_policy_v1,
};

pub const INVERSE_VM_STABILITY_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.inverse-vm-stability-input.v1";
pub const INVERSE_VM_STABILITY_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.inverse-vm-stability-report.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmRobustnessScenarioV1 {
    pub scenario_id: String,
    /// Examples: reference_drift, sensor_dropout, model_disagreement,
    /// support_sparsification. This is a declared stress label, not a probability.
    pub scenario_family: String,
    pub perturbation_profile_sha256: String,
    pub input: InverseVmShortlistInputV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmStabilityInputV1 {
    pub schema: String,
    pub stability_input_id: String,
    pub baseline_input: InverseVmShortlistInputV1,
    pub scenarios: Vec<InverseVmRobustnessScenarioV1>,
    /// Maximum allowed range between best and worst selected rank.
    pub maximum_rank_spread: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmScenarioReplayV1 {
    pub scenario_id: String,
    pub scenario_family: String,
    pub perturbation_profile_sha256: String,
    pub shortlist_report_sha256: String,
    pub selected_candidate_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmCandidateScenarioStateV1 {
    pub scenario_id: String,
    pub state: CandidateScreenStateV1,
    pub selected_rank: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmCandidateStabilityV1 {
    pub candidate_id: String,
    pub baseline_state: CandidateScreenStateV1,
    pub scenario_states: Vec<InverseVmCandidateScenarioStateV1>,
    pub feasible_scenario_count: usize,
    pub selected_scenario_count: usize,
    /// Descriptive fraction over the supplied stress scenarios; not a probability.
    pub feasibility_fraction: f64,
    /// Descriptive fraction over the supplied stress scenarios; not a probability.
    pub selection_fraction: f64,
    pub feasible_in_all_scenarios: bool,
    pub selected_in_all_scenarios: bool,
    pub best_selected_rank: Option<usize>,
    pub worst_selected_rank: Option<usize>,
    pub median_selected_rank: Option<f64>,
    pub selected_rank_spread: Option<usize>,
    pub rank_spread_within_limit: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmStabilityReportV1 {
    pub schema: String,
    pub stability_input_id: String,
    pub process_snapshot_sha256: String,
    pub decision_at_unix_seconds: i64,
    pub baseline_report_sha256: String,
    pub scenario_count: usize,
    pub scenario_replays: Vec<InverseVmScenarioReplayV1>,
    pub candidates: Vec<InverseVmCandidateStabilityV1>,
    pub scenario_frequency_is_probability: bool,
    pub inverse_candidate_generation_executed: bool,
    pub recipe_write_executed: bool,
    pub physical_experiment_authorized: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct InverseVmStabilityErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), InverseVmStabilityErrorV1> {
    if value.trim().is_empty() {
        return Err(InverseVmStabilityErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn digest_json<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, InverseVmStabilityErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        InverseVmStabilityErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn recipe_settings_digest(
    candidate: &super::inverse_vm_shortlist::InverseVmCandidateV1,
) -> Result<String, InverseVmStabilityErrorV1> {
    let mut settings = candidate.recipe_settings.clone();
    settings.sort_by(|left, right| left.knob_id.cmp(&right.knob_id));
    digest_json(&settings)
}

fn report_digest(
    report: &InverseVmStabilityReportV1,
) -> Result<String, InverseVmStabilityErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    digest_json(&canonical)
}

fn candidate_state(
    report: &InverseVmShortlistReportV1,
    candidate_id: &str,
) -> Result<(CandidateScreenStateV1, Option<usize>), InverseVmStabilityErrorV1> {
    let evaluation = report
        .evaluations
        .iter()
        .find(|entry| entry.candidate_id == candidate_id)
        .ok_or_else(|| {
            InverseVmStabilityErrorV1(format!(
                "scenario report omitted candidate {candidate_id}"
            ))
        })?;
    Ok((evaluation.state, evaluation.selected_rank))
}

fn median_rank(ranks: &mut [usize]) -> Option<f64> {
    if ranks.is_empty() {
        return None;
    }
    ranks.sort_unstable();
    let middle = ranks.len() / 2;
    if ranks.len().is_multiple_of(2) {
        Some((ranks[middle - 1] as f64 + ranks[middle] as f64) / 2.0)
    } else {
        Some(ranks[middle] as f64)
    }
}

fn validate_same_decision_context(
    baseline: &InverseVmShortlistInputV1,
    scenario: &InverseVmShortlistInputV1,
) -> Result<(), InverseVmStabilityErrorV1> {
    if baseline.process_snapshot_sha256 != scenario.process_snapshot_sha256
        || baseline.decision_at_unix_seconds != scenario.decision_at_unix_seconds
        || baseline.clock_domain_id != scenario.clock_domain_id
        || baseline.clock_mapping_sha256 != scenario.clock_mapping_sha256
    {
        return Err(InverseVmStabilityErrorV1(
            "scenario changed the process snapshot or decision clock context".into(),
        ));
    }
    let baseline_policy = seal_inverse_vm_shortlist_policy_v1(&baseline.policy)
        .map_err(|error: InverseVmShortlistErrorV1| InverseVmStabilityErrorV1(error.to_string()))?;
    let scenario_policy = seal_inverse_vm_shortlist_policy_v1(&scenario.policy)
        .map_err(|error: InverseVmShortlistErrorV1| InverseVmStabilityErrorV1(error.to_string()))?;
    if baseline_policy != scenario_policy {
        return Err(InverseVmStabilityErrorV1(
            "scenario changed the target, hard constraints, support policy or ranking policy".into(),
        ));
    }

    let baseline_candidates: BTreeMap<&str, &super::inverse_vm_shortlist::InverseVmCandidateV1> =
        baseline
            .candidates
            .iter()
            .map(|candidate| (candidate.candidate_id.as_str(), candidate))
            .collect();
    let scenario_candidates: BTreeMap<&str, &super::inverse_vm_shortlist::InverseVmCandidateV1> =
        scenario
            .candidates
            .iter()
            .map(|candidate| (candidate.candidate_id.as_str(), candidate))
            .collect();
    if baseline_candidates.len() != baseline.candidates.len()
        || scenario_candidates.len() != scenario.candidates.len()
        || baseline_candidates.keys().collect::<BTreeSet<_>>()
            != scenario_candidates.keys().collect::<BTreeSet<_>>()
    {
        return Err(InverseVmStabilityErrorV1(
            "scenario candidate inventory differs from the frozen baseline inventory".into(),
        ));
    }

    for (candidate_id, baseline_candidate) in &baseline_candidates {
        let scenario_candidate = scenario_candidates[candidate_id];
        if recipe_settings_digest(baseline_candidate)? != recipe_settings_digest(scenario_candidate)? {
            return Err(InverseVmStabilityErrorV1(format!(
                "scenario {candidate_id} changed the candidate recipe vector"
            )));
        }
        if baseline_candidate.process_snapshot_sha256
            != scenario_candidate.process_snapshot_sha256
        {
            return Err(InverseVmStabilityErrorV1(format!(
                "scenario {candidate_id} changed its process snapshot"
            )));
        }
    }
    Ok(())
}

/// Replays each declared stress scenario through the same hard shortlist
/// evaluator. The output reports descriptive scenario frequencies and rank
/// envelopes only; it does not interpret stress-scenario frequency as a
/// probability or confidence level.
pub fn evaluate_inverse_vm_stability_v1(
    input: &InverseVmStabilityInputV1,
) -> Result<InverseVmStabilityReportV1, InverseVmStabilityErrorV1> {
    if input.schema != INVERSE_VM_STABILITY_INPUT_SCHEMA_V1 {
        return Err(InverseVmStabilityErrorV1(
            "stability input schema is unsupported".into(),
        ));
    }
    require_text(&input.stability_input_id, "stability_input_id")?;
    if input.scenarios.is_empty() {
        return Err(InverseVmStabilityErrorV1(
            "at least one stress scenario is required".into(),
        ));
    }
    let baseline_report = evaluate_inverse_vm_shortlist_v1(&input.baseline_input)
        .map_err(|error| InverseVmStabilityErrorV1(error.to_string()))?;
    let baseline_candidates: BTreeMap<String, &super::inverse_vm_shortlist::InverseVmCandidateV1> =
        input
            .baseline_input
            .candidates
            .iter()
            .map(|candidate| (candidate.candidate_id.clone(), candidate))
            .collect();

    let mut scenario_ids = BTreeSet::new();
    let mut scenarios = input.scenarios.clone();
    scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut replayed = Vec::with_capacity(scenarios.len());
    let mut scenario_reports = Vec::with_capacity(scenarios.len());

    for scenario in &scenarios {
        require_text(&scenario.scenario_id, "scenario.scenario_id")?;
        require_text(&scenario.scenario_family, "scenario.scenario_family")?;
        if !scenario_ids.insert(scenario.scenario_id.as_str()) {
            return Err(InverseVmStabilityErrorV1(
                "duplicate robustness scenario ID".into(),
            ));
        }
        if !valid_sha256(&scenario.perturbation_profile_sha256) {
            return Err(InverseVmStabilityErrorV1(format!(
                "scenario {} has malformed perturbation profile digest",
                scenario.scenario_id
            )));
        }
        validate_same_decision_context(&input.baseline_input, &scenario.input)?;
        let report = evaluate_inverse_vm_shortlist_v1(&scenario.input)
            .map_err(|error| InverseVmStabilityErrorV1(error.to_string()))?;
        let selected_candidate_ids = report.top_k_candidate_ids.clone();
        replayed.push(InverseVmScenarioReplayV1 {
            scenario_id: scenario.scenario_id.clone(),
            scenario_family: scenario.scenario_family.clone(),
            perturbation_profile_sha256: scenario.perturbation_profile_sha256.clone(),
            shortlist_report_sha256: report.report_sha256.clone(),
            selected_candidate_ids,
        });
        scenario_reports.push((scenario.scenario_id.clone(), report));
    }

    let mut candidate_reports = Vec::with_capacity(baseline_candidates.len());
    for (candidate_id, _) in &baseline_candidates {
        let (baseline_state, _) = candidate_state(&baseline_report, candidate_id)?;
        let mut scenario_states = Vec::with_capacity(scenario_reports.len());
        let mut feasible_scenario_count = 0_usize;
        let mut selected_ranks = Vec::new();
        let mut selected_scenario_count = 0_usize;

        for (scenario_id, report) in &scenario_reports {
            let (state, selected_rank) = candidate_state(report, candidate_id)?;
            if state == CandidateScreenStateV1::Feasible {
                feasible_scenario_count += 1;
            }
            if let Some(rank) = selected_rank {
                selected_scenario_count += 1;
                selected_ranks.push(rank);
            }
            scenario_states.push(InverseVmCandidateScenarioStateV1 {
                scenario_id: scenario_id.clone(),
                state,
                selected_rank,
            });
        }

        let scenario_count = scenario_reports.len();
        let all_selected = selected_scenario_count == scenario_count;
        let all_feasible = feasible_scenario_count == scenario_count;
        let best = selected_ranks.iter().copied().min();
        let worst = selected_ranks.iter().copied().max();
        let spread = best.zip(worst).map(|(minimum, maximum)| maximum - minimum);
        let median = median_rank(&mut selected_ranks);
        let rank_spread_within_limit =
            spread.is_some_and(|value| value <= input.maximum_rank_spread);

        candidate_reports.push(InverseVmCandidateStabilityV1 {
            candidate_id: candidate_id.clone(),
            baseline_state,
            scenario_states,
            feasible_scenario_count,
            selected_scenario_count,
            feasibility_fraction: feasible_scenario_count as f64 / scenario_count as f64,
            selection_fraction: selected_scenario_count as f64 / scenario_count as f64,
            feasible_in_all_scenarios: all_feasible,
            selected_in_all_scenarios: all_selected,
            best_selected_rank: best,
            worst_selected_rank: worst,
            median_selected_rank: median,
            selected_rank_spread: spread,
            rank_spread_within_limit,
        });
    }

    candidate_reports.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    let mut report = InverseVmStabilityReportV1 {
        schema: INVERSE_VM_STABILITY_REPORT_SCHEMA_V1.into(),
        stability_input_id: input.stability_input_id.clone(),
        process_snapshot_sha256: input.baseline_input.process_snapshot_sha256.clone(),
        decision_at_unix_seconds: input.baseline_input.decision_at_unix_seconds,
        baseline_report_sha256: baseline_report.report_sha256,
        scenario_count: scenario_reports.len(),
        scenario_replays: replayed,
        candidates: candidate_reports,
        scenario_frequency_is_probability: false,
        inverse_candidate_generation_executed: false,
        recipe_write_executed: false,
        physical_experiment_authorized: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

/// Replays the baseline and every stress scenario, then compares the complete
/// content-addressed stability report. Scenario ordering does not matter.
pub fn verify_inverse_vm_stability_v1(
    input: &InverseVmStabilityInputV1,
    supplied: &InverseVmStabilityReportV1,
) -> Result<bool, InverseVmStabilityErrorV1> {
    if !valid_sha256(&supplied.report_sha256) || report_digest(supplied)? != supplied.report_sha256 {
        return Ok(false);
    }
    let expected = evaluate_inverse_vm_stability_v1(input)?;
    Ok(expected == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::inverse_vm_shortlist::{
        ForwardEvaluationReceiptV1, ForwardEvidenceStateV1, HistoricalSupportEvidenceV1,
        HistoricalSupportReceiptV1, FORWARD_EVALUATION_RECEIPT_SCHEMA_V1,
        HISTORICAL_SUPPORT_RECEIPT_SCHEMA_V1, InverseVmCandidateV1, MetrologyTargetV1,
        NumericRangeV1, PredictedMetricIntervalV1, RecipeKnobConstraintV1, RecipeSettingV1,
        SideConstraintV1, INVERSE_VM_SHORTLIST_INPUT_SCHEMA_V1,
        INVERSE_VM_SHORTLIST_POLICY_SCHEMA_V1, seal_forward_evaluation_receipt_v1,
        seal_historical_support_receipt_v1, seal_inverse_vm_candidate_v1,
    };

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy() -> super::super::inverse_vm_shortlist::InverseVmShortlistPolicyV1 {
        super::super::inverse_vm_shortlist::InverseVmShortlistPolicyV1 {
            schema: INVERSE_VM_SHORTLIST_POLICY_SCHEMA_V1.into(),
            policy_id: "inverse-stability.v1".into(),
            revision: 1,
            recipe_constraint_source_sha256: digest('a'),
            recipe_constraints_complete: true,
            recipe_knob_constraints: vec![
                RecipeKnobConstraintV1 {
                    knob_id: "rf_power".into(),
                    unit: "W".into(),
                    allowed: NumericRangeV1 { lower: 100.0, upper: 500.0 },
                },
                RecipeKnobConstraintV1 {
                    knob_id: "pressure".into(),
                    unit: "mTorr".into(),
                    allowed: NumericRangeV1 { lower: 10.0, upper: 100.0 },
                },
            ],
            side_constraints_complete: true,
            side_constraints: vec![SideConstraintV1 {
                constraint_id: "uniformity".into(),
                measurand_id: "uniformity".into(),
                unit: "%".into(),
                allowed: NumericRangeV1 { lower: 0.0, upper: 3.0 },
            }],
            target: MetrologyTargetV1 {
                measurand_id: "epi_thickness".into(),
                unit: "nm".into(),
                target_value: 100.0,
                acceptable: NumericRangeV1 { lower: 98.0, upper: 102.0 },
            },
            required_forward_model_sha256: digest('b'),
            support_manifest_sha256: digest('c'),
            support_policy_sha256: digest('d'),
            minimum_support_neighbours: 2,
            maximum_normalized_support_distance: 0.75,
            maximum_interval_width_fraction: 0.75,
            target_error_weight: 1.0,
            interval_width_weight: 0.5,
            support_distance_weight: 0.25,
            diversity_weight: 0.05,
            minimum_recipe_diversity_distance: 0.02,
            top_k: 2,
        }
    }

    fn candidate(id: &str, rf: f64, pressure: f64, lower: f64, upper: f64) -> InverseVmCandidateV1 {
        let mut value = InverseVmCandidateV1 {
            candidate_id: id.into(),
            process_snapshot_sha256: digest('e'),
            recipe_settings: vec![
                RecipeSettingV1 { knob_id: "rf_power".into(), unit: "W".into(), value: rf },
                RecipeSettingV1 { knob_id: "pressure".into(), unit: "mTorr".into(), value: pressure },
            ],
            historical_support: Some(HistoricalSupportEvidenceV1 {
                support_manifest_sha256: digest('c'),
                neighbour_ids: vec![format!("n-{id}-1"), format!("n-{id}-2")],
                normalized_distance: 0.2,
                support_policy_sha256: digest('d'),
                receipt: None,
            }),
            forward_model_sha256: digest('b'),
            forward_evidence_state: ForwardEvidenceStateV1::Qualified,
            forward_evaluation_receipt: None,
            predicted_metrics: vec![
                PredictedMetricIntervalV1 {
                    measurand_id: "epi_thickness".into(),
                    unit: "nm".into(),
                    lower,
                    upper,
                },
                PredictedMetricIntervalV1 {
                    measurand_id: "uniformity".into(),
                    unit: "%",
                    lower: 1.0,
                    upper: 2.0,
                },
            ],
            candidate_evidence_sha256: String::new(),
        };

        let support = value.historical_support.as_ref().unwrap();
        let mut support_receipt = HistoricalSupportReceiptV1 {
            schema: HISTORICAL_SUPPORT_RECEIPT_SCHEMA_V1.into(),
            evaluator_id: "stability-support.v1".into(),
            process_snapshot_sha256: value.process_snapshot_sha256.clone(),
            support_manifest_sha256: support.support_manifest_sha256.clone(),
            support_policy_sha256: support.support_policy_sha256.clone(),
            recipe_settings_sha256: {
                let mut settings = value.recipe_settings.clone();
                settings.sort_by(|a, b| a.knob_id.cmp(&b.knob_id));
                digest_bytes(&settings)
            },
            neighbour_ids_sha256: {
                let mut neighbours = support.neighbour_ids.clone();
                neighbours.sort();
                digest_bytes(&neighbours)
            },
            normalized_distance: support.normalized_distance,
            clock_domain_id: "fab-clock".into(),
            clock_mapping_sha256: digest('9'),
            evaluated_at_unix_seconds: 80,
            available_at_unix_seconds: 90,
            receipt_sha256: String::new(),
        };
        support_receipt.receipt_sha256 = seal_historical_support_receipt_v1(&support_receipt).unwrap();
        value.historical_support.as_mut().unwrap().receipt = Some(support_receipt);

        let mut forward_receipt = ForwardEvaluationReceiptV1 {
            schema: FORWARD_EVALUATION_RECEIPT_SCHEMA_V1.into(),
            evaluator_id: "stability-forward.v1".into(),
            clock_domain_id: "fab-clock".into(),
            clock_mapping_sha256: digest('9'),
            process_snapshot_sha256: value.process_snapshot_sha256.clone(),
            forward_model_sha256: value.forward_model_sha256.clone(),
            model_qualification_report_sha256: Some(digest('f')),
            recipe_settings_sha256: {
                let mut settings = value.recipe_settings.clone();
                settings.sort_by(|a, b| a.knob_id.cmp(&b.knob_id));
                digest_bytes(&settings)
            },
            predicted_metrics_sha256: {
                let mut metrics = value.predicted_metrics.clone();
                metrics.sort_by(|a, b| a.measurand_id.cmp(&b.measurand_id));
                digest_bytes(&metrics)
            },
            evidence_state: value.forward_evidence_state,
            evaluated_at_unix_seconds: 80,
            available_at_unix_seconds: 90,
            receipt_sha256: String::new(),
        };
        forward_receipt.receipt_sha256 = seal_forward_evaluation_receipt_v1(&forward_receipt).unwrap();
        value.forward_evaluation_receipt = Some(forward_receipt);
        value.candidate_evidence_sha256 = seal_inverse_vm_candidate_v1(&value).unwrap();
        value
    }

    fn digest_bytes<T: Serialize + ?Sized>(value: &T) -> String {
        format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
    }

    fn input(candidates: Vec<InverseVmCandidateV1>) -> InverseVmShortlistInputV1 {
        InverseVmShortlistInputV1 {
            schema: INVERSE_VM_SHORTLIST_INPUT_SCHEMA_V1.into(),
            input_id: "stability-base-input.v1".into(),
            process_snapshot_sha256: digest('e'),
            decision_at_unix_seconds: 100,
            clock_domain_id: "fab-clock".into(),
            clock_mapping_sha256: digest('9'),
            policy: policy(),
            candidates,
        }
    }

    fn scenario(id: &str, family: &str, input: InverseVmShortlistInputV1) -> InverseVmRobustnessScenarioV1 {
        InverseVmRobustnessScenarioV1 {
            scenario_id: id.into(),
            scenario_family: family.into(),
            perturbation_profile_sha256: digest('8'),
            input,
        }
    }

    #[test]
    fn identical_scenarios_report_stable_top_k() {
        let baseline = input(vec![
            candidate("a", 220.0, 45.0, 99.0, 101.0),
            candidate("b", 400.0, 80.0, 99.4, 100.6),
        ]);
        let source1 = baseline.clone();
        let source2 = baseline.clone();
        let stability = InverseVmStabilityInputV1 {
            schema: INVERSE_VM_STABILITY_INPUT_SCHEMA_V1.into(),
            stability_input_id: "stable-input.v1".into(),
            baseline_input: baseline,
            scenarios: vec![
                scenario("dropout-low", "sensor_dropout", source1),
                scenario("reference-nominal", "reference_drift", source2),
            ],
            maximum_rank_spread: 0,
        };
        let report = evaluate_inverse_vm_stability_v1(&stability).unwrap();
        assert_eq!(report.scenario_count, 2);
        assert!(report.candidates.iter().all(|item| item.feasible_in_all_scenarios));
        assert!(report.candidates.iter().all(|item| item.selected_in_all_scenarios));
        assert!(report.candidates.iter().all(|item| item.rank_spread_within_limit));
        assert!(!report.scenario_frequency_is_probability);
        assert!(verify_inverse_vm_stability_v1(&stability, &report).unwrap());
    }

    #[test]
    fn interval_widening_can_reduce_scenario_feasibility_and_selection() {
        let baseline = input(vec![
            candidate("a", 220.0, 45.0, 99.0, 101.0),
            candidate("b", 400.0, 80.0, 99.4, 100.6),
        ]);
        let mut stressed = baseline.clone();
        stressed.candidates[0].predicted_metrics[0].lower = 97.0;
        stressed.candidates[0].candidate_evidence_sha256 =
            seal_inverse_vm_candidate_v1(&stressed.candidates[0]).unwrap();
        let mut metrics = stressed.candidates[0].predicted_metrics.clone();
        metrics.sort_by(|a, b| a.measurand_id.cmp(&b.measurand_id));
        let metrics_digest = digest_bytes(&metrics);
        let forward = stressed.candidates[0]
            .forward_evaluation_receipt
            .as_mut()
            .unwrap();
        forward.predicted_metrics_sha256 = metrics_digest;
        forward.receipt_sha256 = seal_forward_evaluation_receipt_v1(forward).unwrap();
        stressed.candidates[0].candidate_evidence_sha256 =
            seal_inverse_vm_candidate_v1(&stressed.candidates[0]).unwrap();

        let stability = InverseVmStabilityInputV1 {
            schema: INVERSE_VM_STABILITY_INPUT_SCHEMA_V1.into(),
            stability_input_id: "shift-stability.v1".into(),
            baseline_input: baseline,
            scenarios: vec![scenario("interval-widening", "model_uncertainty", stressed)],
            maximum_rank_spread: 1,
        };
        let report = evaluate_inverse_vm_stability_v1(&stability).unwrap();
        let a = report.candidates.iter().find(|x| x.candidate_id == "a").unwrap();
        assert_eq!(a.feasible_scenario_count, 0);
        assert!(!a.feasible_in_all_scenarios);
        assert_eq!(a.selection_fraction, 0.0);
    }

    #[test]
    fn scenario_recipe_or_policy_changes_are_rejected() {
        let baseline = input(vec![candidate("a", 220.0, 45.0, 99.0, 101.0)]);
        let mut changed = baseline.clone();
        changed.candidates[0].recipe_settings[0].value += 1.0;
        let recipe_result = evaluate_inverse_vm_stability_v1(&InverseVmStabilityInputV1 {
            schema: INVERSE_VM_STABILITY_INPUT_SCHEMA_V1.into(),
            stability_input_id: "bad-recipe.v1".into(),
            baseline_input: baseline.clone(),
            scenarios: vec![scenario("recipe-changed", "invalid", changed)],
            maximum_rank_spread: 1,
        });
        assert!(recipe_result.is_err());

        let mut changed_policy = baseline.clone();
        changed_policy.policy.target.target_value = 101.0;
        let policy_result = evaluate_inverse_vm_stability_v1(&InverseVmStabilityInputV1 {
            schema: INVERSE_VM_STABILITY_INPUT_SCHEMA_V1.into(),
            stability_input_id: "bad-policy.v1".into(),
            baseline_input: baseline,
            scenarios: vec![scenario("policy-changed", "invalid", changed_policy)],
            maximum_rank_spread: 1,
        });
        assert!(policy_result.is_err());
    }

    #[test]
    fn candidate_inventory_must_be_identical_in_every_scenario() {
        let baseline = input(vec![candidate("a", 220.0, 45.0, 99.0, 101.0)]);
        let changed = input(vec![candidate("b", 220.0, 45.0, 99.0, 101.0)]);
        let result = evaluate_inverse_vm_stability_v1(&InverseVmStabilityInputV1 {
            schema: INVERSE_VM_STABILITY_INPUT_SCHEMA_V1.into(),
            stability_input_id: "wrong-inventory.v1".into(),
            baseline_input: baseline,
            scenarios: vec![scenario("inventory-shift", "sensor_dropout", changed)],
            maximum_rank_spread: 1,
        });
        assert!(result.is_err());
    }

    #[test]
    fn late_forward_receipt_in_stress_scenario_reduces_feasibility_not_probability() {
        let baseline = input(vec![candidate("a", 220.0, 45.0, 99.0, 101.0)]);
        let mut late = baseline.clone();
        let receipt = late.candidates[0].forward_evaluation_receipt.as_mut().unwrap();
        receipt.available_at_unix_seconds = 101;
        receipt.receipt_sha256 = seal_forward_evaluation_receipt_v1(receipt).unwrap();
        late.candidates[0].candidate_evidence_sha256 =
            seal_inverse_vm_candidate_v1(&late.candidates[0]).unwrap();

        let stability = InverseVmStabilityInputV1 {
            schema: INVERSE_VM_STABILITY_INPUT_SCHEMA_V1.into(),
            stability_input_id: "late-receipt-stability.v1".into(),
            baseline_input: baseline,
            scenarios: vec![scenario("late-receipt", "label_delay", late)],
            maximum_rank_spread: 0,
        };
        let report = evaluate_inverse_vm_stability_v1(&stability).unwrap();
        let entry = &report.candidates[0];
        assert_eq!(entry.feasible_scenario_count, 0);
        assert_eq!(entry.selected_scenario_count, 0);
        assert_eq!(entry.feasibility_fraction, 0.0);
        assert_eq!(entry.selection_fraction, 0.0);
        assert!(!report.scenario_frequency_is_probability);
    }

    #[test]
    fn rank_reversal_is_exposed_as_a_scenario_rank_envelope() {
        let mut base_policy = policy();
        base_policy.diversity_weight = 0.0;
        base_policy.minimum_recipe_diversity_distance = 0.02;
        let baseline = InverseVmShortlistInputV1 {
            schema: INVERSE_VM_SHORTLIST_INPUT_SCHEMA_V1.into(),
            input_id: "rank-envelope-base.v1".into(),
            process_snapshot_sha256: digest('e'),
            decision_at_unix_seconds: 100,
            clock_domain_id: "fab-clock".into(),
            clock_mapping_sha256: digest('9'),
            policy: base_policy,
            candidates: vec![
                candidate("a", 220.0, 45.0, 99.95, 100.05),
                candidate("b", 470.0, 90.0, 99.0, 101.8),
            ],
        };
        let mut stressed = baseline.clone();
        stressed.candidates[0].predicted_metrics[0].lower = 100.5;
        stressed.candidates[0].predicted_metrics[0].upper = 102.0;
        let mut metrics = stressed.candidates[0].predicted_metrics.clone();
        metrics.sort_by(|left, right| left.measurand_id.cmp(&right.measurand_id));
        let metrics_sha256 = digest_bytes(&metrics);
        let receipt = stressed.candidates[0]
            .forward_evaluation_receipt
            .as_mut()
            .unwrap();
        receipt.predicted_metrics_sha256 = metrics_sha256;
        receipt.receipt_sha256 = seal_forward_evaluation_receipt_v1(receipt).unwrap();
        stressed.candidates[0].candidate_evidence_sha256 =
            seal_inverse_vm_candidate_v1(&stressed.candidates[0]).unwrap();

        let stability = InverseVmStabilityInputV1 {
            schema: INVERSE_VM_STABILITY_INPUT_SCHEMA_V1.into(),
            stability_input_id: "rank-envelope.v1".into(),
            baseline_input: baseline,
            scenarios: vec![scenario("reference-shift", "reference_drift", stressed)],
            maximum_rank_spread: 0,
        };
        let report = evaluate_inverse_vm_stability_v1(&stability).unwrap();
        let a = report.candidates.iter().find(|entry| entry.candidate_id == "a").unwrap();
        assert!(a.selected_in_all_scenarios);
        assert_eq!(a.best_selected_rank, Some(1));
        assert_eq!(a.worst_selected_rank, Some(2));
        assert_eq!(a.selected_rank_spread, Some(1));
        assert!(!a.rank_spread_within_limit);
    }

    #[test]
    fn scenario_order_does_not_change_stability_digest() {
        let baseline = input(vec![
            candidate("a", 220.0, 45.0, 99.0, 101.0),
            candidate("b", 400.0, 80.0, 99.4, 100.6),
        ]);
        let scenarios = vec![
            scenario("z", "reference_drift", baseline.clone()),
            scenario("a", "sensor_dropout", baseline.clone()),
        ];
        let first = InverseVmStabilityInputV1 {
            schema: INVERSE_VM_STABILITY_INPUT_SCHEMA_V1.into(),
            stability_input_id: "order-test.v1".into(),
            baseline_input: baseline.clone(),
            scenarios: scenarios.clone(),
            maximum_rank_spread: 1,
        };
        let mut second = first.clone();
        second.scenarios.reverse();
        let left = evaluate_inverse_vm_stability_v1(&first).unwrap();
        let right = evaluate_inverse_vm_stability_v1(&second).unwrap();
        assert_eq!(left.report_sha256, right.report_sha256);
    }
}
