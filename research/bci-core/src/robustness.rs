// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Preregistered signal-perturbation robustness summaries for BCI decoders.
//!
//! Scores are pre-aggregated over the same frozen evaluation units by an upstream
//! scorer. This module verifies a complete model × scenario grid and prevents a
//! strong average from concealing a weak perturbation scenario. It does not
//! synthesize EEG, establish that a perturbation is physiologically realistic,
//! or calculate confidence intervals.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// All score values use the closed interval [0, 1_000_000] (parts per million).
pub const ROBUSTNESS_SCORE_SCALE_PPM: u32 = 1_000_000;

/// Perturbation families to declare before looking at held-out results.
/// Severity is a preregistered ordinal level within a family, not a universal
/// physical unit and not directly comparable across different families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerturbationFamily {
    Clean,
    ChannelDropout,
    AmplitudeScaling,
    AdditiveArtifact,
    PacketLoss,
    TemporalJitter,
    SessionShift,
    ReferenceShift,
    MontageShift,
    Other,
}

/// One frozen scenario in a robustness battery. Exactly one scenario must use
/// Clean with severity zero; every perturbed scenario must have severity > 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustnessScenario {
    pub scenario_id: String,
    pub family: PerturbationFamily,
    pub severity_permille: u16,
}

/// One pre-aggregated score per model and scenario. Upstream evaluation must use
/// the same frozen held-out evaluation units for every model/scenario pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustnessScore {
    pub model_id: String,
    pub scenario_id: String,
    pub score_ppm: u32,
}

/// Thresholds are aspirational study gates, not claims about a validated device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RobustnessConfig {
    pub minimum_mean_perturbed_score_ppm: u32,
    pub maximum_worst_case_degradation_ppm: u32,
}

/// Point summary for one model over the complete preregistered stress battery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustnessModelReport {
    pub model_id: String,
    pub clean_score_ppm: u32,
    pub mean_perturbed_score_ppm: u32,
    pub worst_case_score_ppm: u32,
    pub worst_case_scenario_id: String,
    pub mean_degradation_from_clean_ppm: u32,
    pub worst_case_degradation_from_clean_ppm: u32,
}

/// Direct per-scenario comparison; positive delta means the candidate scored higher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustnessScenarioDelta {
    pub scenario_id: String,
    pub family: PerturbationFamily,
    pub severity_permille: u16,
    pub baseline_score_ppm: u32,
    pub candidate_score_ppm: u32,
    pub candidate_minus_baseline_ppm: i64,
}

/// Comparison plus threshold decisions for the candidate. These are descriptive
/// point-estimate gates; uncertainty must be calculated on paired unit-level data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustnessEvaluationReport {
    pub baseline_model_id: String,
    pub candidate_model_id: String,
    pub model_reports: Vec<RobustnessModelReport>,
    pub scenario_deltas: Vec<RobustnessScenarioDelta>,
    pub candidate_minus_baseline_mean_perturbed_ppm: i64,
    pub candidate_minus_baseline_worst_case_ppm: i64,
    pub candidate_meets_mean_score_floor: bool,
    pub candidate_meets_worst_case_degradation_limit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RobustnessError {
    EmptyModels,
    TooFewModels { actual: usize },
    EmptyScenarios,
    NoPerturbedScenarios,
    EmptyScores,
    EmptyIdentity { field: &'static str, index: usize },
    DuplicateModelId(String),
    DuplicateScenarioId(String),
    InvalidScenarioSeverity { scenario_id: String },
    ExpectedExactlyOneCleanScenario { actual: usize },
    BaselineModelNotInInventory(String),
    CandidateModelNotInInventory(String),
    SameBaselineAndCandidate(String),
    UnexpectedModel(String),
    UnexpectedScenario(String),
    DuplicateModelScenario { model_id: String, scenario_id: String },
    ScoreOutOfRange { model_id: String, scenario_id: String, score_ppm: u32 },
    IncompleteScoreGrid { expected: usize, actual: usize },
    InvalidThreshold { field: &'static str, value_ppm: u32 },
    ComputationBudgetExceeded,
}

impl fmt::Display for RobustnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RobustnessError {}

/// Evaluate a frozen perturbation battery for a baseline and candidate.
///
/// expected_model_ids, scenarios, and their Cartesian product must be fixed
/// before inspecting final results. Scores must come from the same upstream
/// aggregation protocol and the same held-out units. The function refuses missing
/// or unexpected model/scenario rows rather than silently averaging what remains.
///
/// The mean and worst-case values are equal-weighted across declared perturbed
/// scenarios. The worst-case degradation is the largest positive loss relative to
/// that model's clean score; a scenario that scores above clean has zero degradation.
pub fn evaluate_robustness(
    scores: &[RobustnessScore],
    expected_model_ids: &[&str],
    scenarios: &[RobustnessScenario],
    baseline_model_id: &str,
    candidate_model_id: &str,
    config: RobustnessConfig,
) -> Result<RobustnessEvaluationReport, RobustnessError> {
    if scores.is_empty() {
        return Err(RobustnessError::EmptyScores);
    }
    if expected_model_ids.is_empty() {
        return Err(RobustnessError::EmptyModels);
    }
    if expected_model_ids.len() < 2 {
        return Err(RobustnessError::TooFewModels { actual: expected_model_ids.len() });
    }
    if scenarios.is_empty() {
        return Err(RobustnessError::EmptyScenarios);
    }
    if config.minimum_mean_perturbed_score_ppm > ROBUSTNESS_SCORE_SCALE_PPM {
        return Err(RobustnessError::InvalidThreshold {
            field: "minimum_mean_perturbed_score_ppm",
            value_ppm: config.minimum_mean_perturbed_score_ppm,
        });
    }
    if config.maximum_worst_case_degradation_ppm > ROBUSTNESS_SCORE_SCALE_PPM {
        return Err(RobustnessError::InvalidThreshold {
            field: "maximum_worst_case_degradation_ppm",
            value_ppm: config.maximum_worst_case_degradation_ppm,
        });
    }
    if baseline_model_id == candidate_model_id {
        return Err(RobustnessError::SameBaselineAndCandidate(baseline_model_id.to_owned()));
    }

    let mut models = BTreeSet::new();
    for (index, model_id) in expected_model_ids.iter().enumerate() {
        if model_id.trim().is_empty() {
            return Err(RobustnessError::EmptyIdentity { field: "model_id", index });
        }
        if !models.insert((*model_id).to_owned()) {
            return Err(RobustnessError::DuplicateModelId((*model_id).to_owned()));
        }
    }
    if !models.contains(baseline_model_id) {
        return Err(RobustnessError::BaselineModelNotInInventory(baseline_model_id.to_owned()));
    }
    if !models.contains(candidate_model_id) {
        return Err(RobustnessError::CandidateModelNotInInventory(candidate_model_id.to_owned()));
    }

    let mut scenario_by_id = BTreeMap::new();
    let mut clean_scenario_ids = Vec::new();
    for (index, scenario) in scenarios.iter().enumerate() {
        if scenario.scenario_id.trim().is_empty() {
            return Err(RobustnessError::EmptyIdentity { field: "scenario_id", index });
        }
        let severity_valid = match scenario.family {
            PerturbationFamily::Clean => {
                if scenario.severity_permille == 0 {
                    clean_scenario_ids.push(scenario.scenario_id.clone());
                    true
                } else {
                    false
                }
            }
            _ => scenario.severity_permille > 0 && scenario.severity_permille <= 1000,
        };
        if !severity_valid {
            return Err(RobustnessError::InvalidScenarioSeverity {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        if scenario_by_id.insert(scenario.scenario_id.clone(), scenario.clone()).is_some() {
            return Err(RobustnessError::DuplicateScenarioId(scenario.scenario_id.clone()));
        }
    }
    if clean_scenario_ids.len() != 1 {
        return Err(RobustnessError::ExpectedExactlyOneCleanScenario {
            actual: clean_scenario_ids.len(),
        });
    }
    let clean_scenario_id = clean_scenario_ids[0].clone();
    let mut stress_scenarios: Vec<RobustnessScenario> = scenario_by_id
        .values()
        .filter(|scenario| scenario.family != PerturbationFamily::Clean)
        .cloned()
        .collect();
    if stress_scenarios.is_empty() {
        return Err(RobustnessError::NoPerturbedScenarios);
    }
    stress_scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

    let mut score_by_pair: BTreeMap<(String, String), u32> = BTreeMap::new();
    for row in scores {
        if !models.contains(&row.model_id) {
            return Err(RobustnessError::UnexpectedModel(row.model_id.clone()));
        }
        if !scenario_by_id.contains_key(&row.scenario_id) {
            return Err(RobustnessError::UnexpectedScenario(row.scenario_id.clone()));
        }
        if row.score_ppm > ROBUSTNESS_SCORE_SCALE_PPM {
            return Err(RobustnessError::ScoreOutOfRange {
                model_id: row.model_id.clone(),
                scenario_id: row.scenario_id.clone(),
                score_ppm: row.score_ppm,
            });
        }
        let key = (row.model_id.clone(), row.scenario_id.clone());
        if score_by_pair.insert(key, row.score_ppm).is_some() {
            return Err(RobustnessError::DuplicateModelScenario {
                model_id: row.model_id.clone(),
                scenario_id: row.scenario_id.clone(),
            });
        }
    }
    let expected_rows = models
        .len()
        .checked_mul(scenario_by_id.len())
        .ok_or(RobustnessError::ComputationBudgetExceeded)?;
    if score_by_pair.len() != expected_rows {
        return Err(RobustnessError::IncompleteScoreGrid {
            expected: expected_rows,
            actual: score_by_pair.len(),
        });
    }

    let mut model_reports = Vec::with_capacity(models.len());
    for model_id in &models {
        let clean_score = score_by_pair[&(model_id.clone(), clean_scenario_id.clone())];
        let mut perturbed_sum = 0_u128;
        let mut degradation_sum = 0_u128;
        let mut worst_case_score = u32::MAX;
        let mut worst_case_scenario_id = String::new();
        let mut worst_case_degradation = 0_u32;

        for scenario in &stress_scenarios {
            let score = score_by_pair[&(model_id.clone(), scenario.scenario_id.clone())];
            perturbed_sum += u128::from(score);
            let degradation = clean_score.saturating_sub(score);
            degradation_sum += u128::from(degradation);
            worst_case_degradation = worst_case_degradation.max(degradation);
            // Scenarios are sorted by ID; retaining the first equal minimum makes
            // the reported worst-case scenario deterministic under ties.
            if score < worst_case_score {
                worst_case_score = score;
                worst_case_scenario_id = scenario.scenario_id.clone();
            }
        }
        let scenario_count = u128::try_from(stress_scenarios.len())
            .map_err(|_| RobustnessError::ComputationBudgetExceeded)?;
        model_reports.push(RobustnessModelReport {
            model_id: model_id.clone(),
            clean_score_ppm: clean_score,
            mean_perturbed_score_ppm: u32::try_from(perturbed_sum / scenario_count)
                .map_err(|_| RobustnessError::ComputationBudgetExceeded)?,
            worst_case_score_ppm: worst_case_score,
            worst_case_scenario_id,
            mean_degradation_from_clean_ppm: u32::try_from(degradation_sum / scenario_count)
                .map_err(|_| RobustnessError::ComputationBudgetExceeded)?,
            worst_case_degradation_from_clean_ppm: worst_case_degradation,
        });
    }
    model_reports.sort_by(|left, right| left.model_id.cmp(&right.model_id));

    let baseline = model_reports.iter().find(|model| model.model_id == baseline_model_id)
        .expect("baseline model was verified in the inventory");
    let candidate = model_reports.iter().find(|model| model.model_id == candidate_model_id)
        .expect("candidate model was verified in the inventory");
    let mut scenario_deltas = Vec::with_capacity(scenario_by_id.len());
    let mut all_scenarios: Vec<RobustnessScenario> = scenario_by_id.values().cloned().collect();
    all_scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    for scenario in all_scenarios {
        let baseline_score = score_by_pair[&(baseline_model_id.to_owned(), scenario.scenario_id.clone())];
        let candidate_score = score_by_pair[&(candidate_model_id.to_owned(), scenario.scenario_id.clone())];
        scenario_deltas.push(RobustnessScenarioDelta {
            scenario_id: scenario.scenario_id,
            family: scenario.family,
            severity_permille: scenario.severity_permille,
            baseline_score_ppm: baseline_score,
            candidate_score_ppm: candidate_score,
            candidate_minus_baseline_ppm: i64::from(candidate_score) - i64::from(baseline_score),
        });
    }

    Ok(RobustnessEvaluationReport {
        baseline_model_id: baseline_model_id.to_owned(),
        candidate_model_id: candidate_model_id.to_owned(),
        candidate_minus_baseline_mean_perturbed_ppm:
            i64::from(candidate.mean_perturbed_score_ppm) - i64::from(baseline.mean_perturbed_score_ppm),
        candidate_minus_baseline_worst_case_ppm:
            i64::from(candidate.worst_case_score_ppm) - i64::from(baseline.worst_case_score_ppm),
        candidate_meets_mean_score_floor:
            candidate.mean_perturbed_score_ppm >= config.minimum_mean_perturbed_score_ppm,
        candidate_meets_worst_case_degradation_limit:
            candidate.worst_case_degradation_from_clean_ppm <= config.maximum_worst_case_degradation_ppm,
        model_reports,
        scenario_deltas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean() -> RobustnessScenario {
        RobustnessScenario {
            scenario_id: "clean".into(),
            family: PerturbationFamily::Clean,
            severity_permille: 0,
        }
    }

    fn scenario(id: &str, family: PerturbationFamily, severity_permille: u16) -> RobustnessScenario {
        RobustnessScenario { scenario_id: id.into(), family, severity_permille }
    }

    fn score(model_id: &str, scenario_id: &str, score_ppm: u32) -> RobustnessScore {
        RobustnessScore { model_id: model_id.into(), scenario_id: scenario_id.into(), score_ppm }
    }

    fn config() -> RobustnessConfig {
        RobustnessConfig {
            minimum_mean_perturbed_score_ppm: 700_000,
            maximum_worst_case_degradation_ppm: 200_000,
        }
    }

    fn scenarios() -> Vec<RobustnessScenario> {
        vec![
            clean(),
            scenario("channel-dropout-mild", PerturbationFamily::ChannelDropout, 100),
            scenario("artifact-severe", PerturbationFamily::AdditiveArtifact, 800),
        ]
    }

    fn scores() -> Vec<RobustnessScore> {
        vec![
            score("baseline", "clean", 900_000),
            score("baseline", "channel-dropout-mild", 800_000),
            score("baseline", "artifact-severe", 600_000),
            score("candidate", "clean", 900_000),
            score("candidate", "channel-dropout-mild", 850_000),
            score("candidate", "artifact-severe", 700_000),
        ]
    }

    fn evaluate(
        rows: &[RobustnessScore],
        scenario_inventory: &[RobustnessScenario],
        threshold: RobustnessConfig,
    ) -> Result<RobustnessEvaluationReport, RobustnessError> {
        evaluate_robustness(rows, &["candidate", "baseline"], scenario_inventory, "baseline", "candidate", threshold)
    }

    #[test]
    fn complete_grid_reports_equal_weight_mean_worst_case_and_paired_scenario_deltas() {
        let report = evaluate(&scores(), &scenarios(), config()).unwrap();
        let baseline = report.model_reports.iter().find(|item| item.model_id == "baseline").unwrap();
        let candidate = report.model_reports.iter().find(|item| item.model_id == "candidate").unwrap();
        assert_eq!(baseline.mean_perturbed_score_ppm, 700_000);
        assert_eq!(baseline.worst_case_score_ppm, 600_000);
        assert_eq!(baseline.worst_case_scenario_id, "artifact-severe");
        assert_eq!(baseline.worst_case_degradation_from_clean_ppm, 300_000);
        assert_eq!(candidate.mean_perturbed_score_ppm, 775_000);
        assert_eq!(candidate.worst_case_score_ppm, 700_000);
        assert_eq!(candidate.mean_degradation_from_clean_ppm, 125_000);
        assert_eq!(candidate.worst_case_degradation_from_clean_ppm, 200_000);
        assert_eq!(report.candidate_minus_baseline_mean_perturbed_ppm, 75_000);
        assert_eq!(report.candidate_minus_baseline_worst_case_ppm, 100_000);
        assert!(report.candidate_meets_mean_score_floor);
        assert!(report.candidate_meets_worst_case_degradation_limit);
        assert_eq!(report.scenario_deltas.len(), 3);
    }

    #[test]
    fn strong_average_cannot_hide_a_worst_case_gate_failure() {
        let mut rows = scores();
        rows.iter_mut().find(|row| row.model_id == "candidate" && row.scenario_id == "clean").unwrap().score_ppm = 1_000_000;
        rows.iter_mut().find(|row| row.model_id == "candidate" && row.scenario_id == "channel-dropout-mild").unwrap().score_ppm = 950_000;
        rows.iter_mut().find(|row| row.model_id == "candidate" && row.scenario_id == "artifact-severe").unwrap().score_ppm = 750_000;
        let threshold = RobustnessConfig {
            minimum_mean_perturbed_score_ppm: 800_000,
            maximum_worst_case_degradation_ppm: 200_000,
        };
        let report = evaluate(&rows, &scenarios(), threshold).unwrap();
        let candidate = report.model_reports.iter().find(|item| item.model_id == "candidate").unwrap();
        assert_eq!(candidate.mean_perturbed_score_ppm, 850_000);
        assert_eq!(candidate.worst_case_degradation_from_clean_ppm, 250_000);
        assert!(report.candidate_meets_mean_score_floor);
        assert!(!report.candidate_meets_worst_case_degradation_limit);
    }

    #[test]
    fn rejects_incomplete_duplicate_unexpected_and_out_of_range_score_grids() {
        let mut rows = scores();
        rows.pop();
        assert_eq!(
            evaluate(&rows, &scenarios(), config()),
            Err(RobustnessError::IncompleteScoreGrid { expected: 6, actual: 5 })
        );

        let mut rows = scores();
        rows.push(rows[0].clone());
        assert!(matches!(
            evaluate(&rows, &scenarios(), config()),
            Err(RobustnessError::DuplicateModelScenario { .. })
        ));

        let mut rows = scores();
        rows[0].model_id = "unexpected-model".into();
        assert_eq!(
            evaluate(&rows, &scenarios(), config()),
            Err(RobustnessError::UnexpectedModel("unexpected-model".into()))
        );

        let mut rows = scores();
        rows[0].score_ppm = ROBUSTNESS_SCORE_SCALE_PPM + 1;
        assert!(matches!(
            evaluate(&rows, &scenarios(), config()),
            Err(RobustnessError::ScoreOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_invalid_scenario_battery_and_comparison_controls() {
        let invalid = vec![
            clean(),
            scenario("bad-zero-stress", PerturbationFamily::ChannelDropout, 0),
        ];
        assert_eq!(
            evaluate(&scores(), &invalid, config()),
            Err(RobustnessError::InvalidScenarioSeverity { scenario_id: "bad-zero-stress".into() })
        );
        let clean_only = vec![clean()];
        assert_eq!(
            evaluate(&scores(), &clean_only, config()),
            Err(RobustnessError::NoPerturbedScenarios)
        );
        let same_model = evaluate_robustness(
            &scores(),
            &["baseline", "candidate"],
            &scenarios(),
            "baseline",
            "baseline",
            config(),
        );
        assert_eq!(
            same_model,
            Err(RobustnessError::SameBaselineAndCandidate("baseline".into()))
        );
        let bad_threshold = RobustnessConfig {
            minimum_mean_perturbed_score_ppm: ROBUSTNESS_SCORE_SCALE_PPM + 1,
            ..config()
        };
        assert!(matches!(
            evaluate(&scores(), &scenarios(), bad_threshold),
            Err(RobustnessError::InvalidThreshold { .. })
        ));
    }

    #[test]
    fn output_is_deterministic_under_input_order_changes() {
        let first = evaluate(&scores(), &scenarios(), config()).unwrap();
        let mut reversed_scores = scores();
        reversed_scores.reverse();
        let mut reversed_scenarios = scenarios();
        reversed_scenarios.reverse();
        let second = evaluate(&reversed_scores, &reversed_scenarios, config()).unwrap();
        assert_eq!(first, second);
    }
}
