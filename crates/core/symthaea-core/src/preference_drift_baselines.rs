// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic fixed-weight and seeded-random baselines for the synthetic mobility benchmark.
//!
//! Baselines select from scenario-provided eligible candidates and are scored only after their
//! decisions are frozen. This is an offline synthetic comparison, not a preference model.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::preference_drift_benchmark::{
    validate_preference_drift_fixture, PreferenceDriftFixtureFindingV1,
};

/// Current schema version for selection-baseline receipts.
pub const PREFERENCE_SELECTION_BASELINES_SCHEMA_VERSION_V1: u64 = 1;

/// Frozen task-success weight.
pub const FIXED_WEIGHT_TASK_SUCCESS_V1: f64 = 0.40;
/// Frozen stability-margin weight.
pub const FIXED_WEIGHT_STABILITY_V1: f64 = 0.20;
/// Frozen wearer-borne mass weight.
pub const FIXED_WEIGHT_MASS_V1: f64 = 0.15;
/// Frozen energy-use weight.
pub const FIXED_WEIGHT_ENERGY_V1: f64 = 0.10;
/// Frozen movement-freedom weight.
pub const FIXED_WEIGHT_MOVEMENT_FREEDOM_V1: f64 = 0.15;
/// Fixed normalization scale for the synthetic mass values, in kilograms.
pub const FIXED_WEIGHT_MASS_SCALE_KG_V1: f64 = 7.0;
/// Fixed normalization scale for the synthetic energy values, in watt-hours per task.
pub const FIXED_WEIGHT_ENERGY_SCALE_WH_V1: f64 = 6.0;

/// One candidate-selection output from an engineering or random baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceSelectionBaselineAssessmentV1 {
    /// Scenario identity.
    pub scenario_id: String,
    /// Stable baseline name.
    pub baseline_id: String,
    /// Selected candidate, or none if no candidate is eligible.
    pub selected_candidate_id: Option<String>,
    /// Engineering utility for fixed-weight selection; absent for random selection.
    pub fixed_weight_utility: Option<f64>,
    /// Reproducible seed; present for seeded-random selection.
    pub seed: Option<u64>,
}

/// Scores computed only after selection decisions are frozen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceSelectionBaselineScoreV1 {
    /// Stable baseline name.
    pub baseline_id: String,
    /// Seed used for the stochastic comparator, if applicable.
    pub seed: Option<u64>,
    /// Number of fixture scenarios.
    pub scenario_count: usize,
    /// Number of scenarios for which the evaluation oracle supplies a ranking.
    pub oracle_ranking_scenario_count: usize,
    /// Number of candidates selected in scenarios with a ranking oracle.
    pub selections_with_ranking_count: usize,
    /// Selections belonging to the oracle's acceptable set for ranked scenarios.
    pub acceptable_selection_count: usize,
    /// Selections outside the oracle's acceptable set for ranked scenarios.
    pub unacceptable_selection_count: usize,
    /// Selections emitted when the oracle says the evidence is insufficient or rejects all.
    pub selection_without_ranking_count: usize,
    /// Selections emitted in explicit reject-all scenarios.
    pub selection_on_reject_all_count: usize,
    /// Selections made while at least one preference observation is task/context mismatched.
    pub stale_context_selection_count: usize,
    /// Per-scenario output, preserving fixture order.
    pub assessments: Vec<PreferenceSelectionBaselineAssessmentV1>,
}

/// Paired receipt for frozen fixed-weight and seeded-random selection baselines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceSelectionBaselinesReportV1 {
    /// Report schema.
    pub schema_version: u64,
    /// Version of the synthetic fixture.
    pub fixture_version: String,
    /// Number of scenarios.
    pub scenario_count: usize,
    /// Number of candidates in the fixture catalog.
    pub candidate_count: usize,
    /// Root seed for the random baseline.
    pub seed: u64,
    /// Fixed-weight comparator results.
    pub fixed_weight_score: PreferenceSelectionBaselineScoreV1,
    /// Seeded-random comparator results.
    pub seeded_random_score: PreferenceSelectionBaselineScoreV1,
    /// Explicitly bounded claim.
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Copy)]
struct EngineeringProfileV1 {
    mass_kg: f64,
    energy_wh: f64,
    movement_restriction: f64,
    stability_margin: f64,
}

fn finding(path: impl Into<String>, reason: impl Into<String>) -> PreferenceDriftFixtureFindingV1 {
    PreferenceDriftFixtureFindingV1 {
        path: path.into(),
        reason: reason.into(),
    }
}

fn finite_number(value: &Value, path: &str) -> Result<f64, PreferenceDriftFixtureFindingV1> {
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| finding(path, "expected a finite numeric value"))
}

fn read_profiles(
    root: &Value,
) -> Result<HashMap<String, EngineeringProfileV1>, Vec<PreferenceDriftFixtureFindingV1>> {
    let candidates = root
        .get("candidate_catalog")
        .and_then(Value::as_array)
        .ok_or_else(|| vec![finding("$.candidate_catalog", "missing candidate catalog")])?;
    let mut profiles = HashMap::new();

    for (index, candidate) in candidates.iter().enumerate() {
        let path = format!("$.candidate_catalog[{index}]");
        let candidate_id = candidate
            .get("candidate_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| vec![finding(format!("{path}.candidate_id"), "missing candidate ID")])?;
        let profile = candidate
            .get("synthetic_engineering_profile")
            .ok_or_else(|| vec![finding(
                format!("{path}.synthetic_engineering_profile"),
                "missing synthetic engineering profile",
            )])?;

        let mass = finite_number(
            profile.get("wearer_borne_mass_kg").unwrap_or(&Value::Null),
            &format!("{path}.synthetic_engineering_profile.wearer_borne_mass_kg"),
        ).map_err(|error| vec![error])?;
        let energy = finite_number(
            profile.get("energy_wh_per_task").unwrap_or(&Value::Null),
            &format!("{path}.synthetic_engineering_profile.energy_wh_per_task"),
        ).map_err(|error| vec![error])?;
        let restriction = finite_number(
            profile.get("movement_restriction_index_0_to_1").unwrap_or(&Value::Null),
            &format!("{path}.synthetic_engineering_profile.movement_restriction_index_0_to_1"),
        ).map_err(|error| vec![error])?;
        let stability = finite_number(
            profile.get("stability_margin_0_to_1").unwrap_or(&Value::Null),
            &format!("{path}.synthetic_engineering_profile.stability_margin_0_to_1"),
        ).map_err(|error| vec![error])?;

        if !(0.0..=FIXED_WEIGHT_MASS_SCALE_KG_V1).contains(&mass) {
            return Err(vec![finding(
                format!("{path}.synthetic_engineering_profile.wearer_borne_mass_kg"),
                "mass is outside the fixed-weight fixture normalization range",
            )]);
        }
        if !(0.0..=FIXED_WEIGHT_ENERGY_SCALE_WH_V1).contains(&energy) {
            return Err(vec![finding(
                format!("{path}.synthetic_engineering_profile.energy_wh_per_task"),
                "energy is outside the fixed-weight fixture normalization range",
            )]);
        }
        if !(0.0..=1.0).contains(&restriction) || !(0.0..=1.0).contains(&stability) {
            return Err(vec![finding(
                format!("{path}.synthetic_engineering_profile"),
                "restriction and stability metrics must be normalized to [0, 1]",
            )]);
        }

        let profile = EngineeringProfileV1 {
            mass_kg: mass,
            energy_wh: energy,
            movement_restriction: restriction,
            stability_margin: stability,
        };
        if profiles.insert(candidate_id.clone(), profile).is_some() {
            return Err(vec![finding(
                format!("{path}.candidate_id"),
                format!("duplicate candidate ID: {candidate_id}"),
            )]);
        }
    }
    Ok(profiles)
}

fn task_success(
    candidate: &Value,
    task_id: &str,
    candidate_id: &str,
) -> Result<f64, PreferenceDriftFixtureFindingV1> {
    let value = candidate
        .pointer("/synthetic_engineering_profile/task_success_rate_0_to_1")
        .and_then(|rates| rates.get(task_id))
        .ok_or_else(|| finding(
            format!("$.candidate_catalog[{candidate_id}].task_success_rate_0_to_1.{task_id}"),
            "task-specific success metric is missing",
        ))?;
    match value.as_f64().filter(|number| number.is_finite()) {
        Some(rate) if (0.0..=1.0).contains(&rate) => Ok(rate),
        _ => Err(finding(
            format!("$.candidate_catalog[{candidate_id}].task_success_rate_0_to_1.{task_id}"),
            "task-success rate must be finite and normalized to [0, 1]",
        )),
    }
}

fn fixed_weight_utility(
    candidate: &Value,
    profile: EngineeringProfileV1,
    task_id: &str,
    candidate_id: &str,
) -> Result<f64, PreferenceDriftFixtureFindingV1> {
    let success = task_success(candidate, task_id, candidate_id)?;
    Ok(FIXED_WEIGHT_TASK_SUCCESS_V1 * success
        + FIXED_WEIGHT_STABILITY_V1 * profile.stability_margin
        + FIXED_WEIGHT_MASS_V1 * (1.0 - profile.mass_kg / FIXED_WEIGHT_MASS_SCALE_KG_V1)
        + FIXED_WEIGHT_ENERGY_V1 * (1.0 - profile.energy_wh / FIXED_WEIGHT_ENERGY_SCALE_WH_V1)
        + FIXED_WEIGHT_MOVEMENT_FREEDOM_V1 * (1.0 - profile.movement_restriction))
}

fn choose_fixed_weight(
    scenario: &Value,
    candidate_values: &HashMap<String, &Value>,
    profiles: &HashMap<String, EngineeringProfileV1>,
) -> Result<PreferenceSelectionBaselineAssessmentV1, Vec<PreferenceDriftFixtureFindingV1>> {
    let scenario_id = scenario.get("scenario_id").and_then(Value::as_str).unwrap_or_default();
    let task_id = scenario.get("task_id").and_then(Value::as_str).unwrap_or_default();
    let eligible = scenario
        .get("eligible_candidate_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| vec![finding(
            format!("$.scenarios[{scenario_id}].eligible_candidate_ids"),
            "missing eligible candidate array",
        )])?;

    let mut selected: Option<(String, f64)> = None;
    for item in eligible {
        let candidate_id = item.as_str().ok_or_else(|| vec![finding(
            format!("$.scenarios[{scenario_id}].eligible_candidate_ids"),
            "candidate ID must be a string",
        )])?;
        let candidate = candidate_values.get(candidate_id).ok_or_else(|| vec![finding(
            format!("$.scenarios[{scenario_id}].eligible_candidate_ids"),
            format!("unknown candidate ID: {candidate_id}"),
        )])?;
        let profile = profiles.get(candidate_id).ok_or_else(|| vec![finding(
            format!("$.candidate_catalog[{candidate_id}]"),
            "missing engineering profile",
        )])?;
        let utility = fixed_weight_utility(candidate, *profile, task_id, candidate_id)
            .map_err(|error| vec![error])?;
        // Preserve the first eligible candidate on exact ties; tie-breaking is deterministic.
        if selected.as_ref().is_none_or(|(_, best)| utility > *best) {
            selected = Some((candidate_id.to_owned(), utility));
        }
    }

    let (selected_candidate_id, fixed_weight_utility) = match selected {
        Some((id, score)) => (Some(id), Some(score)),
        None => (None, None),
    };
    Ok(PreferenceSelectionBaselineAssessmentV1 {
        scenario_id: scenario_id.to_owned(),
        baseline_id: "fixed_weight_engineering".to_owned(),
        selected_candidate_id,
        fixed_weight_utility,
        seed: None,
    })
}

/// Stable FNV-1a 64-bit selector; it is deterministic, not cryptographic.
fn seeded_index(seed: u64, scenario_id: &str, upper_bound: usize) -> Option<usize> {
    if upper_bound == 0 {
        return None;
    }
    let key = format!("{seed}:{scenario_id}");
    let mut hash = 0xcbf29ce484222325u64;
    for byte in key.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Some((hash % upper_bound as u64) as usize)
}

fn choose_seeded_random(
    scenario: &Value,
    seed: u64,
) -> Result<PreferenceSelectionBaselineAssessmentV1, Vec<PreferenceDriftFixtureFindingV1>> {
    let scenario_id = scenario.get("scenario_id").and_then(Value::as_str).unwrap_or_default();
    let eligible = scenario
        .get("eligible_candidate_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| vec![finding(
            format!("$.scenarios[{scenario_id}].eligible_candidate_ids"),
            "missing eligible candidate array",
        )])?;
    let ids = eligible
        .iter()
        .map(|item| item.as_str().map(str::to_owned))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| vec![finding(
            format!("$.scenarios[{scenario_id}].eligible_candidate_ids"),
            "candidate IDs must be strings",
        )])?;
    let selected_candidate_id =
        seeded_index(seed, scenario_id, ids.len()).and_then(|index| ids.get(index).cloned());

    Ok(PreferenceSelectionBaselineAssessmentV1 {
        scenario_id: scenario_id.to_owned(),
        baseline_id: "seeded_random".to_owned(),
        selected_candidate_id,
        fixed_weight_utility: None,
        seed: Some(seed),
    })
}

fn score_selections(
    scenarios: &[Value],
    assessments: &[PreferenceSelectionBaselineAssessmentV1],
    baseline_id: &str,
    seed: Option<u64>,
) -> PreferenceSelectionBaselineScoreV1 {
    let mut score = PreferenceSelectionBaselineScoreV1 {
        baseline_id: baseline_id.to_owned(),
        seed,
        scenario_count: scenarios.len(),
        oracle_ranking_scenario_count: 0,
        selections_with_ranking_count: 0,
        acceptable_selection_count: 0,
        unacceptable_selection_count: 0,
        selection_without_ranking_count: 0,
        selection_on_reject_all_count: 0,
        stale_context_selection_count: 0,
        assessments: assessments.to_vec(),
    };

    for (scenario, assessment) in scenarios.iter().zip(assessments) {
        let oracle = scenario.get("evaluation_oracle").unwrap_or(&Value::Null);
        let status = oracle.get("status").and_then(Value::as_str);
        let selected = assessment.selected_candidate_id.as_ref();
        if status == Some("ranking") {
            score.oracle_ranking_scenario_count += 1;
            if let Some(candidate_id) = selected {
                score.selections_with_ranking_count += 1;
                let acceptable = oracle
                    .get("acceptable_candidate_ids")
                    .and_then(Value::as_array)
                    .is_some_and(|items| items.iter().any(|item| item.as_str() == Some(candidate_id.as_str())));
                if acceptable {
                    score.acceptable_selection_count += 1;
                } else {
                    score.unacceptable_selection_count += 1;
                }
            }
        } else if selected.is_some() {
            score.selection_without_ranking_count += 1;
        }

        if status == Some("reject_all") && selected.is_some() {
            score.selection_on_reject_all_count += 1;
        }
        if selected.is_some() {
            let task_id = scenario.get("task_id").and_then(Value::as_str);
            let context_id = scenario.get("current_context_id").and_then(Value::as_str);
            let has_mismatch = scenario
                .get("preference_observations")
                .and_then(Value::as_array)
                .is_some_and(|observations| {
                    observations.iter().any(|observation| {
                        observation.get("task_id").and_then(Value::as_str) != task_id
                            || observation.get("context_id").and_then(Value::as_str) != context_id
                    })
                });
            if has_mismatch {
                score.stale_context_selection_count += 1;
            }
        }
    }
    score
}

/// Evaluate fixed-weight and seeded-random selection over the same validated synthetic episodes.
///
/// The weights and normalization scales are fixed constants, not fitted on the evaluation oracle.
/// Both policies select only from scenario-provided eligible candidates. The seed controls only
/// random baseline selection. Oracle labels are consulted after decisions have been produced.
pub fn evaluate_preference_selection_baselines(
    input: &str,
    seed: u64,
) -> Result<PreferenceSelectionBaselinesReportV1, Vec<PreferenceDriftFixtureFindingV1>> {
    validate_preference_drift_fixture(input)?;
    let root: Value = serde_json::from_str(input).map_err(|error| {
        vec![finding("$", format!("invalid JSON: {error}"))]
    })?;
    let fixture_version = root
        .get("fixture_version")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| vec![finding("$.fixture_version", "missing fixture version")])?;
    let scenarios = root
        .get("scenarios")
        .and_then(Value::as_array)
        .ok_or_else(|| vec![finding("$.scenarios", "missing scenario list")])?;
    let candidate_values = root
        .get("candidate_catalog")
        .and_then(Value::as_array)
        .ok_or_else(|| vec![finding("$.candidate_catalog", "missing candidate catalog")])?
        .iter()
        .filter_map(|candidate| {
            candidate
                .get("candidate_id")
                .and_then(Value::as_str)
                .map(|id| (id.to_owned(), candidate))
        })
        .collect::<HashMap<_, _>>();
    let profiles = read_profiles(&root)?;

    let mut fixed_assessments = Vec::with_capacity(scenarios.len());
    let mut random_assessments = Vec::with_capacity(scenarios.len());
    for scenario in scenarios {
        fixed_assessments.push(choose_fixed_weight(scenario, &candidate_values, &profiles)?);
        random_assessments.push(choose_seeded_random(scenario, seed)?);
    }

    // Score only after the two selection baselines have frozen all outputs.
    let fixed_weight_score =
        score_selections(scenarios, &fixed_assessments, "fixed_weight_engineering", None);
    let seeded_random_score =
        score_selections(scenarios, &random_assessments, "seeded_random", Some(seed));

    Ok(PreferenceSelectionBaselinesReportV1 {
        schema_version: PREFERENCE_SELECTION_BASELINES_SCHEMA_VERSION_V1,
        fixture_version,
        scenario_count: scenarios.len(),
        candidate_count: candidate_values.len(),
        seed,
        fixed_weight_score,
        seeded_random_score,
        claim_ceiling: "Synthetic fixed-weight/random behavior only; no real preference, safety, clinical, or deployment claim.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!(concat!(
        "../../../../docs/research/open-mobility/fixtures/",
        "preference-drift-scenarios-v1.json"
    ));

    #[test]
    fn fixture_baselines_validate_and_emit_expected_counts() {
        let report = evaluate_preference_selection_baselines(FIXTURE, 20_261_018)
            .expect("fixture baselines should evaluate");
        assert_eq!(report.scenario_count, 12);
        assert_eq!(report.candidate_count, 3);
        assert_eq!(report.fixed_weight_score.oracle_ranking_scenario_count, 9);
        assert_eq!(report.fixed_weight_score.acceptable_selection_count, 8);
        assert_eq!(report.fixed_weight_score.unacceptable_selection_count, 1);
        assert_eq!(report.fixed_weight_score.selection_without_ranking_count, 3);
        assert_eq!(report.fixed_weight_score.selection_on_reject_all_count, 1);
        assert_eq!(report.fixed_weight_score.stale_context_selection_count, 4);
        assert_eq!(report.seeded_random_score.acceptable_selection_count, 6);
        assert_eq!(report.seeded_random_score.unacceptable_selection_count, 3);
        assert_eq!(report.seeded_random_score.selection_without_ranking_count, 3);
        assert_eq!(report.seeded_random_score.selection_on_reject_all_count, 1);
        assert_eq!(report.seeded_random_score.stale_context_selection_count, 4);
    }

    #[test]
    fn seeded_random_baseline_is_repeatable_for_a_fixed_seed() {
        let first = evaluate_preference_selection_baselines(FIXTURE, 12345).unwrap();
        let second = evaluate_preference_selection_baselines(FIXTURE, 12345).unwrap();
        assert_eq!(first.seeded_random_score, second.seeded_random_score);
    }

    #[test]
    fn oracle_changes_do_not_change_any_baseline_selection() {
        let mut root: Value = serde_json::from_str(FIXTURE).unwrap();
        let original = evaluate_preference_selection_baselines(FIXTURE, 20261018).unwrap();
        root["scenarios"][0]["evaluation_oracle"]["preferred_order"] =
            serde_json::json!(["assist", "stable", "flex"]);
        root["scenarios"][0]["evaluation_oracle"]["acceptable_candidate_ids"] =
            serde_json::json!(["assist"]);
        let changed_json = serde_json::to_string(&root).unwrap();
        let changed = evaluate_preference_selection_baselines(&changed_json, 20261018).unwrap();

        for (before, after) in original
            .fixed_weight_score
            .assessments
            .iter()
            .zip(changed.fixed_weight_score.assessments.iter())
        {
            assert_eq!(before.selected_candidate_id, after.selected_candidate_id);
        }
        for (before, after) in original
            .seeded_random_score
            .assessments
            .iter()
            .zip(changed.seeded_random_score.assessments.iter())
        {
            assert_eq!(before.selected_candidate_id, after.selected_candidate_id);
        }
    }

    #[test]
    fn fixed_weights_sum_to_one_and_are_not_fitted_at_runtime() {
        let sum = FIXED_WEIGHT_TASK_SUCCESS_V1
            + FIXED_WEIGHT_STABILITY_V1
            + FIXED_WEIGHT_MASS_V1
            + FIXED_WEIGHT_ENERGY_V1
            + FIXED_WEIGHT_MOVEMENT_FREEDOM_V1;
        assert!((sum - 1.0).abs() < f64::EPSILON);
        assert_eq!(FIXED_WEIGHT_TASK_SUCCESS_V1, 0.40);
        assert_eq!(FIXED_WEIGHT_STABILITY_V1, 0.20);
        assert_eq!(FIXED_WEIGHT_MASS_V1, 0.15);
        assert_eq!(FIXED_WEIGHT_ENERGY_V1, 0.10);
        assert_eq!(FIXED_WEIGHT_MOVEMENT_FREEDOM_V1, 0.15);
    }

    #[test]
    fn non_finite_or_out_of_range_profile_is_rejected() {
        let mut root: Value = serde_json::from_str(FIXTURE).unwrap();
        root["candidate_catalog"][0]["synthetic_engineering_profile"]
            ["movement_restriction_index_0_to_1"] = serde_json::json!(1.5);
        let modified = serde_json::to_string(&root).unwrap();
        assert!(evaluate_preference_selection_baselines(&modified, 1).is_err());
    }

    #[test]
    fn unsupported_task_success_key_fails_closed() {
        let mut root: Value = serde_json::from_str(FIXTURE).unwrap();
        root["scenarios"][0]["task_id"] = serde_json::json!("unsupported_task");
        let modified = serde_json::to_string(&root).unwrap();
        assert!(evaluate_preference_selection_baselines(&modified, 1).is_err());
    }

    #[test]
    fn selection_never_escapes_declared_eligible_candidates() {
        let root: Value = serde_json::from_str(FIXTURE).unwrap();
        let report = evaluate_preference_selection_baselines(FIXTURE, 20261018).unwrap();
        for assessment in report
            .fixed_weight_score
            .assessments
            .iter()
            .chain(report.seeded_random_score.assessments.iter())
        {
            let scenario = root["scenarios"]
                .as_array()
                .unwrap()
                .iter()
                .find(|scenario| scenario["scenario_id"] == assessment.scenario_id)
                .unwrap();
            assert!(assessment.selected_candidate_id.as_ref().is_none_or(|candidate| {
                scenario["eligible_candidate_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|eligible| eligible.as_str() == Some(candidate))
            }));
        }
    }
}
