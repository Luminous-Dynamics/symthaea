// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic contract validation for the Open Mobility preference-drift fixture.
//!
//! This module validates the hand-authored synthetic corpus only. It does not fit a preference
//! model, rank candidates, estimate calibration, operate hardware, or establish human evidence.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

/// Supported version of the preference-drift fixture contract.
pub const PREFERENCE_DRIFT_FIXTURE_SCHEMA_VERSION_V1: u64 = 1;

/// Counts from a validated fixture. Counts are bookkeeping, not model-performance metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreferenceDriftFixtureSummaryV1 {
    /// Fixture schema version.
    pub schema_version: u64,
    /// Number of catalogued synthetic candidates.
    pub candidate_count: usize,
    /// Total number of scenario records.
    pub scenario_count: usize,
    /// Number of development scenarios.
    pub development_scenario_count: usize,
    /// Number of held-out scenarios.
    pub held_out_scenario_count: usize,
    /// Number of explicit synthetic feedback/rejection records.
    pub observation_count: usize,
}

/// One fail-closed fixture-contract violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreferenceDriftFixtureFindingV1 {
    /// JSON path identifying the malformed or inconsistent field.
    pub path: String,
    /// Stable human-readable explanation.
    pub reason: String,
}

fn finding(
    findings: &mut Vec<PreferenceDriftFixtureFindingV1>,
    path: impl Into<String>,
    reason: impl Into<String>,
) {
    findings.push(PreferenceDriftFixtureFindingV1 {
        path: path.into(),
        reason: reason.into(),
    });
}

fn nonempty_string(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

fn string_array(value: &Value, field: &str) -> Option<Vec<String>> {
    value.get(field).and_then(Value::as_array).and_then(|items| {
        let strings = items.iter().map(Value::as_str).collect::<Option<Vec<_>>>()?;
        Some(strings.into_iter().map(str::to_owned).collect())
    })
}

fn record_unique_strings(
    values: &[String],
    path: &str,
    findings: &mut Vec<PreferenceDriftFixtureFindingV1>,
) {
    let mut seen = HashSet::with_capacity(values.len());
    for value in values {
        if !seen.insert(value.as_str()) {
            finding(
                findings,
                path,
                format!("duplicate identifier `{value}`"),
            );
        }
    }
}

fn validate_known_ids(
    values: &[String],
    candidate_ids: &HashSet<String>,
    path: &str,
    findings: &mut Vec<PreferenceDriftFixtureFindingV1>,
) {
    for value in values {
        if !candidate_ids.contains(value) {
            finding(
                findings,
                path,
                format!("unknown candidate identifier `{value}`"),
            );
        }
    }
}

/// Validate the versioned synthetic fixture before a replay/evaluator consumes it.
///
/// Validation rejects an unsupported schema, duplicate IDs, unknown candidate references,
/// malformed preference feedback, oracle leakage markers, inconsistent query budgets, and
/// candidates that violate a declared hard wearer-borne-mass constraint while marked eligible.
/// It cannot authenticate the truth of any observation and must not be treated as such.
pub fn validate_preference_drift_fixture(
    input: &str,
) -> Result<PreferenceDriftFixtureSummaryV1, Vec<PreferenceDriftFixtureFindingV1>> {
    let root: Value = match serde_json::from_str(input) {
        Ok(value) => value,
        Err(error) => {
            return Err(vec![PreferenceDriftFixtureFindingV1 {
                path: "$".to_owned(),
                reason: format!("invalid JSON: {error}"),
            }]);
        }
    };

    let mut findings = Vec::new();
    let schema_version = root.get("schema_version").and_then(Value::as_u64);
    if schema_version != Some(PREFERENCE_DRIFT_FIXTURE_SCHEMA_VERSION_V1) {
        finding(
            &mut findings,
            "$.schema_version",
            format!(
                "unsupported or missing schema version; expected {}",
                PREFERENCE_DRIFT_FIXTURE_SCHEMA_VERSION_V1
            ),
        );
    }

    if nonempty_string(&root, "benchmark_id").is_none() {
        finding(&mut findings, "$.benchmark_id", "missing non-empty benchmark ID");
    }
    if nonempty_string(&root, "fixture_version").is_none() {
        finding(
            &mut findings,
            "$.fixture_version",
            "missing non-empty fixture version",
        );
    }
    if nonempty_string(&root, "claim_ceiling").is_none() {
        finding(&mut findings, "$.claim_ceiling", "missing claim ceiling");
    }
    if root
        .get("evidence_class")
        .and_then(Value::as_str)
        != Some("synthetic_contract_fixture")
    {
        finding(
            &mut findings,
            "$.evidence_class",
            "fixture must remain explicitly synthetic",
        );
    }
    if root
        .pointer("/generation/random_generator_used")
        .and_then(Value::as_bool)
        != Some(false)
        || root
            .pointer("/generation/empirical_data_used")
            .and_then(Value::as_bool)
            != Some(false)
    {
        finding(
            &mut findings,
            "$.generation",
            "hand-authored fixture must not claim random generation or empirical data",
        );
    }
    if root
        .pointer("/metric_separation/engineering_metrics_are_preference_labels")
        .and_then(Value::as_bool)
        != Some(false)
    {
        finding(
            &mut findings,
            "$.metric_separation.engineering_metrics_are_preference_labels",
            "engineering metrics must remain distinct from preference evidence",
        );
    }

    let candidates = match root.get("candidate_catalog").and_then(Value::as_array) {
        Some(items) if !items.is_empty() => items.clone(),
        _ => {
            finding(
                &mut findings,
                "$.candidate_catalog",
                "missing or empty candidate catalog",
            );
            Vec::new()
        }
    };
    let mut candidate_ids = HashSet::<String>::with_capacity(candidates.len());
    let mut candidate_mass_kg = HashMap::<String, f64>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let path = format!("$.candidate_catalog[{index}]");
        let candidate_id = match nonempty_string(candidate, "candidate_id") {
            Some(value) => value,
            None => {
                finding(
                    &mut findings,
                    format!("{path}.candidate_id"),
                    "missing non-empty candidate ID",
                );
                continue;
            }
        };
        if !candidate_ids.insert(candidate_id.clone()) {
            finding(
                &mut findings,
                format!("{path}.candidate_id"),
                format!("duplicate candidate ID `{candidate_id}`"),
            );
        }
        if candidate.get("values_are_empirical").and_then(Value::as_bool) != Some(false) {
            finding(
                &mut findings,
                format!("{path}.values_are_empirical"),
                "candidate values must be marked non-empirical",
            );
        }
        match candidate
            .pointer("/synthetic_engineering_profile/wearer_borne_mass_kg")
            .and_then(Value::as_f64)
        {
            Some(mass) if mass.is_finite() && mass >= 0.0 => {
                candidate_mass_kg.insert(candidate_id, mass);
            }
            _ => finding(
                &mut findings,
                format!("{path}.synthetic_engineering_profile.wearer_borne_mass_kg"),
                "missing, non-finite, or negative synthetic mass value",
            ),
        }
    }

    let scenarios = match root.get("scenarios").and_then(Value::as_array) {
        Some(items) if !items.is_empty() => items.clone(),
        _ => {
            finding(
                &mut findings,
                "$.scenarios",
                "missing or empty scenario array",
            );
            Vec::new()
        }
    };
    let mut scenario_ids = HashSet::<String>::with_capacity(scenarios.len());
    let mut observation_ids = HashSet::<String>::new();
    let mut family_splits = HashMap::<String, String>::new();
    let mut development_count = 0usize;
    let mut held_out_count = 0usize;
    let mut observation_count = 0usize;

    for (index, scenario) in scenarios.iter().enumerate() {
        let path = format!("$.scenarios[{index}]");
        match nonempty_string(scenario, "scenario_id") {
            Some(id) => {
                if !scenario_ids.insert(id.clone()) {
                    finding(
                        &mut findings,
                        format!("{path}.scenario_id"),
                        format!("duplicate scenario ID `{id}`"),
                    );
                }
            }
            None => finding(
                &mut findings,
                format!("{path}.scenario_id"),
                "missing non-empty scenario ID",
            ),
        }
        let family = nonempty_string(scenario, "scenario_family");
        if family.is_none() {
            finding(
                &mut findings,
                format!("{path}.scenario_family"),
                "missing scenario-family identity",
            );
        }
        if nonempty_string(scenario, "task_id").is_none() {
            finding(
                &mut findings,
                format!("{path}.task_id"),
                "missing task identity",
            );
        }
        if scenario
            .get("current_context")
            .and_then(Value::as_object)
            .is_none_or(|context| context.is_empty())
        {
            finding(
                &mut findings,
                format!("{path}.current_context"),
                "missing non-empty current context",
            );
        }
        if nonempty_string(scenario, "current_context_id").is_none() {
            finding(
                &mut findings,
                format!("{path}.current_context_id"),
                "missing non-empty current context ID",
            );
        }

        let split = scenario.get("split").and_then(Value::as_str);
        match split {
            Some("development") => development_count += 1,
            Some("held_out") => held_out_count += 1,
            _ => finding(
                &mut findings,
                format!("{path}.split"),
                "split must be development or held_out",
            ),
        }
        if let (Some(family), Some(split @ ("development" | "held_out"))) = (family, split) {
            if let Some(previous_split) = family_splits.get(&family) {
                if previous_split != split {
                    finding(
                        &mut findings,
                        format!("{path}.scenario_family"),
                        format!(
                            "scenario family `{}` leaks across `{}` and `{}` splits",
                            family, previous_split, split
                        ),
                    );
                }
            } else {
                family_splits.insert(family, split.to_owned());
            }
        }

        let eligible = match string_array(scenario, "eligible_candidate_ids") {
            Some(ids) => ids,
            None => {
                finding(
                    &mut findings,
                    format!("{path}.eligible_candidate_ids"),
                    "missing candidate eligibility array",
                );
                Vec::new()
            }
        };
        record_unique_strings(
            &eligible,
            &format!("{path}.eligible_candidate_ids"),
            &mut findings,
        );
        validate_known_ids(
            &eligible,
            &candidate_ids,
            &format!("{path}.eligible_candidate_ids"),
            &mut findings,
        );

        if let Some(ineligible) = string_array(scenario, "explicitly_ineligible_candidate_ids") {
            record_unique_strings(
                &ineligible,
                &format!("{path}.explicitly_ineligible_candidate_ids"),
                &mut findings,
            );
            validate_known_ids(
                &ineligible,
                &candidate_ids,
                &format!("{path}.explicitly_ineligible_candidate_ids"),
                &mut findings,
            );
            for id in &ineligible {
                if eligible.contains(id) {
                    finding(
                        &mut findings,
                        format!("{path}.eligible_candidate_ids"),
                        format!("hard-ineligible candidate `{id}` is also eligible"),
                    );
                }
            }
        }

        let budget = scenario.get("query_budget_total").and_then(Value::as_u64);
        let used = scenario.get("queries_already_used").and_then(Value::as_u64);
        match (budget, used) {
            (Some(total), Some(consumed)) if consumed <= total => {}
            _ => finding(
                &mut findings,
                format!("{path}.query_budget_total"),
                "query budget must be unsigned and queries_already_used must not exceed it",
            ),
        }

        let observations = match scenario
            .get("preference_observations")
            .and_then(Value::as_array)
        {
            Some(items) => items.clone(),
            None => {
                finding(
                    &mut findings,
                    format!("{path}.preference_observations"),
                    "missing preference-observation array",
                );
                Vec::new()
            }
        };
        for (observation_index, observation) in observations.iter().enumerate() {
            observation_count += 1;
            let observation_path =
                format!("{path}.preference_observations[{observation_index}]");
            match nonempty_string(observation, "observation_id") {
                Some(id) => {
                    if !observation_ids.insert(id.clone()) {
                        finding(
                            &mut findings,
                            format!("{observation_path}.observation_id"),
                            format!("duplicate observation ID `{id}`"),
                        );
                    }
                }
                None => finding(
                    &mut findings,
                    format!("{observation_path}.observation_id"),
                    "missing non-empty observation ID",
                ),
            }
            if observation
                .get("source")
                .and_then(Value::as_str)
                .is_none_or(|source| !source.starts_with("explicit_synthetic_"))
            {
                finding(
                    &mut findings,
                    format!("{observation_path}.source"),
                    "preference must be an explicit synthetic feedback record",
                );
            }
            if observation.get("step").and_then(Value::as_u64).is_none()
                || nonempty_string(observation, "task_id").is_none()
                || nonempty_string(observation, "context_id").is_none()
            {
                finding(
                    &mut findings,
                    &observation_path,
                    "observation requires non-negative step, task ID, and context ID",
                );
            }

            let preferred = nonempty_string(observation, "preferred_candidate_id");
            let rejected = nonempty_string(observation, "rejected_candidate_id");
            let rejected_all = string_array(observation, "rejected_candidate_ids");
            match (preferred, rejected, rejected_all) {
                (Some(winner), Some(loser), None) => {
                    if winner == loser {
                        finding(
                            &mut findings,
                            &observation_path,
                            "pairwise preference must compare two distinct candidates",
                        );
                    }
                    validate_known_ids(
                        &[winner],
                        &candidate_ids,
                        &observation_path,
                        &mut findings,
                    );
                    validate_known_ids(
                        &[loser],
                        &candidate_ids,
                        &observation_path,
                        &mut findings,
                    );
                }
                (None, None, Some(rejected_ids)) if !rejected_ids.is_empty() => {
                    record_unique_strings(
                        &rejected_ids,
                        &observation_path,
                        &mut findings,
                    );
                    validate_known_ids(
                        &rejected_ids,
                        &candidate_ids,
                        &observation_path,
                        &mut findings,
                    );
                }
                _ => finding(
                    &mut findings,
                    &observation_path,
                    "expected a preference pair or explicit non-empty reject-all list",
                ),
            }
        }

        match scenario.get("evaluation_oracle") {
            Some(oracle) => {
                if oracle.get("visibility").and_then(Value::as_str)
                    != Some("evaluation_only_never_policy_input")
                {
                    finding(
                        &mut findings,
                        format!("{path}.evaluation_oracle.visibility"),
                        "evaluation oracle must be explicitly barred from policy input",
                    );
                }
                let status = oracle.get("status").and_then(Value::as_str);
                if !matches!(
                    status,
                    Some("ranking" | "insufficient_evidence" | "reject_all")
                ) {
                    finding(
                        &mut findings,
                        format!("{path}.evaluation_oracle.status"),
                        "unknown oracle status",
                    );
                }
                let preferred_order = string_array(oracle, "preferred_order");
                let acceptable = string_array(oracle, "acceptable_candidate_ids");
                if preferred_order.is_none() || acceptable.is_none() {
                    finding(
                        &mut findings,
                        format!("{path}.evaluation_oracle"),
                        "oracle must provide preferred_order and acceptable_candidate_ids arrays",
                    );
                } else {
                    let order = preferred_order.unwrap_or_default();
                    let accepted = acceptable.unwrap_or_default();
                    record_unique_strings(
                        &order,
                        &format!("{path}.evaluation_oracle.preferred_order"),
                        &mut findings,
                    );
                    record_unique_strings(
                        &accepted,
                        &format!("{path}.evaluation_oracle.acceptable_candidate_ids"),
                        &mut findings,
                    );
                    validate_known_ids(
                        &order,
                        &candidate_ids,
                        &format!("{path}.evaluation_oracle.preferred_order"),
                        &mut findings,
                    );
                    validate_known_ids(
                        &accepted,
                        &candidate_ids,
                        &format!("{path}.evaluation_oracle.acceptable_candidate_ids"),
                        &mut findings,
                    );
                    for candidate_id in &accepted {
                        if !eligible.contains(candidate_id) {
                            finding(
                                &mut findings,
                                format!("{path}.evaluation_oracle.acceptable_candidate_ids"),
                                format!("oracle accepts ineligible candidate `{candidate_id}`"),
                            );
                        }
                    }
                    if status == Some("ranking") && order.is_empty() {
                        finding(
                            &mut findings,
                            format!("{path}.evaluation_oracle.preferred_order"),
                            "ranking oracle must contain an order",
                        );
                    }
                    if status != Some("ranking") && !accepted.is_empty() {
                        finding(
                            &mut findings,
                            format!("{path}.evaluation_oracle.acceptable_candidate_ids"),
                            "non-ranking oracle must not mark candidates acceptable",
                        );
                    }
                    if status != Some("ranking") && !order.is_empty() {
                        finding(
                            &mut findings,
                            format!("{path}.evaluation_oracle.preferred_order"),
                            "non-ranking oracle must not manufacture a total order",
                        );
                    }
                }
            }
            None => finding(
                &mut findings,
                format!("{path}.evaluation_oracle"),
                "missing evaluation-only oracle",
            ),
        }

        if let Some(constraints) = scenario
            .get("eligibility_constraints")
            .and_then(Value::as_array)
        {
            for (constraint_index, constraint) in constraints.iter().enumerate() {
                if constraint.get("kind").and_then(Value::as_str)
                    == Some("hard_constraint")
                    && constraint.get("metric").and_then(Value::as_str)
                        == Some("wearer_borne_mass_kg")
                {
                    let maximum = constraint.get("max_value").and_then(Value::as_f64);
                    if let Some(maximum) = maximum.filter(|value| {
                        value.is_finite() && *value >= 0.0
                    })
                    {
                        for candidate_id in &eligible {
                            if candidate_mass_kg
                                .get(candidate_id)
                                .is_some_and(|mass| *mass > maximum)
                            {
                                finding(
                                    &mut findings,
                                    format!(
                                        "{path}.eligibility_constraints[{constraint_index}]"
                                    ),
                                    format!(
                                        "eligible candidate `{candidate_id}` exceeds mass ceiling"
                                    ),
                                );
                            }
                        }
                    } else {
                        finding(
                            &mut findings,
                            format!(
                                "{path}.eligibility_constraints[{constraint_index}].max_value"
                            ),
                            "hard mass ceiling must be finite and non-negative",
                        );
                    }
                }
            }
        }
    }

    if !findings.is_empty() {
        return Err(findings);
    }

    Ok(PreferenceDriftFixtureSummaryV1 {
        schema_version: schema_version.unwrap_or_default(),
        candidate_count: candidates.len(),
        scenario_count: scenarios.len(),
        development_scenario_count: development_count,
        held_out_scenario_count: held_out_count,
        observation_count,
    })
}

/// Action emitted by the context-matched pairwise-feedback replay baseline.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceReplayActionV1 {
    /// One eligible candidate directly beats every other eligible candidate in applicable feedback.
    Recommend,
    /// Current-context preference evidence is incomplete or contradictory; budget remains.
    Query,
    /// A decision cannot be supported, and no query budget remains.
    Abstain,
    /// The explicit current-context feedback rejects every eligible candidate.
    RejectAll,
    /// No candidate is technically eligible under the supplied scenario.
    NoFeasibleCandidate,
}

/// Why the replay baseline emitted its action.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceReplayReasonV1 {
    /// A unique candidate beats every other eligible candidate under exact-context explicit feedback.
    UniqueExplicitPairwiseWinner,
    /// Explicit current-context feedback rejects all eligible candidates.
    ExplicitRejectAll,
    /// The scenario has no eligible candidate.
    NoEligibleCandidates,
    /// No observations match both the current task and current context.
    NoContextMatchedPreferenceEvidence,
    /// Observations exist, but the evidence does not compare a unique winner to every eligible option.
    InsufficientPairwiseCoverage,
    /// Current-context feedback contains both directions for the same pair.
    ContradictoryFeedback,
}

/// Report-only output of a simple context-matched preference baseline.
///
/// This intentionally does not read `evaluation_oracle`, engineering scores, latent states, or
/// sensor data. A recommendation requires explicit current-task/current-context pairwise feedback
/// against every other eligible candidate; a model proxy or stale observation cannot fill a gap.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceReplayAssessmentV1 {
    /// Scenario identity.
    pub scenario_id: String,
    /// Baseline action.
    pub action: PreferenceReplayActionV1,
    /// Evidence-driven reason for the action.
    pub reason: PreferenceReplayReasonV1,
    /// Candidate selected only when `action` is `recommend`.
    pub recommended_candidate_id: Option<String>,
    /// Eligible candidates after the scenario's upstream hard-gate filter.
    pub eligible_candidate_ids: Vec<String>,
    /// Number of observations matching both exact current task and context.
    pub context_matched_observation_count: usize,
    /// Number of observations not applicable to the current task/context.
    pub stale_or_mismatched_observation_count: usize,
    /// Additional explicit feedback queries still allowed in this scenario.
    pub remaining_query_budget: u64,
}

/// Replay a conservative baseline using explicit, context-matched ordinal observations only.
///
/// This is not a learned preference model. It recommends only when a single eligible candidate
/// has explicit pairwise support against every other eligible candidate. Contradictory feedback,
/// missing comparisons, context/task drift, and absent observations yield a query if permitted,
/// otherwise abstention. A current-context reject-all record is preserved. The function does not
/// inspect the evaluation-only oracle or infer preferences from engineering metrics.
pub fn replay_preference_episode(
    scenario: &Value,
) -> Result<PreferenceReplayAssessmentV1, Vec<PreferenceDriftFixtureFindingV1>> {
    let mut findings = Vec::new();
    let scenario_id = nonempty_string(scenario, "scenario_id");
    let task_id = nonempty_string(scenario, "task_id");
    let current_context_id = nonempty_string(scenario, "current_context_id");
    if scenario_id.is_none() {
        finding(&mut findings, "$.scenario_id", "missing scenario ID");
    }
    if task_id.is_none() {
        finding(&mut findings, "$.task_id", "missing task ID");
    }
    if current_context_id.is_none() {
        finding(
            &mut findings,
            "$.current_context_id",
            "missing current context ID",
        );
    }

    let eligible = match string_array(scenario, "eligible_candidate_ids") {
        Some(ids) => ids,
        None => {
            finding(
                &mut findings,
                "$.eligible_candidate_ids",
                "missing or malformed eligible-candidate array",
            );
            Vec::new()
        }
    };
    record_unique_strings(&eligible, "$.eligible_candidate_ids", &mut findings);

    let total_budget = scenario.get("query_budget_total").and_then(Value::as_u64);
    let consumed_budget = scenario.get("queries_already_used").and_then(Value::as_u64);
    let remaining_budget = match (total_budget, consumed_budget) {
        (Some(total), Some(consumed)) if consumed <= total => total - consumed,
        _ => {
            finding(
                &mut findings,
                "$.query_budget_total",
                "invalid query budget or consumed queries exceed total",
            );
            0
        }
    };

    let observations = match scenario
        .get("preference_observations")
        .and_then(Value::as_array)
    {
        Some(items) => items,
        None => {
            return Err(vec![PreferenceDriftFixtureFindingV1 {
                path: "$.preference_observations".to_owned(),
                reason: "missing or malformed preference-observation array".to_owned(),
            }]);
        }
    };
    if !findings.is_empty() {
        return Err(findings);
    }

    let task_id = task_id.unwrap_or_default();
    let current_context_id = current_context_id.unwrap_or_default();
    let mut matched = Vec::new();
    let mut mismatched_count = 0usize;

    for observation in observations {
        let observation_task = nonempty_string(observation, "task_id");
        let observation_context = nonempty_string(observation, "context_id");
        if observation_task.as_deref() == Some(task_id.as_str())
            && observation_context.as_deref() == Some(current_context_id.as_str())
        {
            matched.push(observation);
        } else {
            mismatched_count += 1;
        }
    }

    let scenario_id = scenario_id.unwrap_or_default();
    let build = |action, reason, recommended_candidate_id| PreferenceReplayAssessmentV1 {
        scenario_id: scenario_id.clone(),
        action,
        reason,
        recommended_candidate_id,
        eligible_candidate_ids: eligible.clone(),
        context_matched_observation_count: matched.len(),
        stale_or_mismatched_observation_count: mismatched_count,
        remaining_query_budget: remaining_budget,
    };

    if eligible.is_empty() {
        return Ok(build(
            PreferenceReplayActionV1::NoFeasibleCandidate,
            PreferenceReplayReasonV1::NoEligibleCandidates,
            None,
        ));
    }

    let eligible_set: HashSet<String> = eligible.iter().cloned().collect();
    for observation in &matched {
        if let Some(rejected) = string_array(observation, "rejected_candidate_ids") {
            let rejected_set: HashSet<String> = rejected.iter().cloned().collect();
            if eligible_set.iter().all(|candidate| rejected_set.contains(candidate)) {
                return Ok(build(
                    PreferenceReplayActionV1::RejectAll,
                    PreferenceReplayReasonV1::ExplicitRejectAll,
                    None,
                ));
            }
        }
    }

    let mut directed_pairs = HashSet::<(String, String)>::new();
    let mut malformed_current_observation = false;
    for observation in &matched {
        let preferred = nonempty_string(observation, "preferred_candidate_id");
        let rejected = nonempty_string(observation, "rejected_candidate_id");
        match (preferred, rejected) {
            (Some(winner), Some(loser)) if winner != loser => {
                if eligible_set.contains(&winner) && eligible_set.contains(&loser)
                {
                    directed_pairs.insert((winner, loser));
                }
            }
            (None, None) if string_array(observation, "rejected_candidate_ids").is_some() => {}
            _ => malformed_current_observation = true,
        }
    }

    if malformed_current_observation {
        return Err(vec![PreferenceDriftFixtureFindingV1 {
            path: "$.preference_observations".to_owned(),
            reason: "malformed current-context preference observation".to_owned(),
        }]);
    }

    let has_conflict = directed_pairs
        .iter()
        .any(|(winner, loser)| directed_pairs.contains(&(loser.clone(), winner.clone())));
    if has_conflict {
        let action = if remaining_budget > 0 {
            PreferenceReplayActionV1::Query
        } else {
            PreferenceReplayActionV1::Abstain
        };
        return Ok(build(action, PreferenceReplayReasonV1::ContradictoryFeedback, None));
    }

    let winner_candidates: Vec<String> = eligible
        .iter()
        .filter(|candidate| {
            let candidate_id = candidate.as_str();
            eligible
                .iter()
                .filter(|other| other.as_str() != candidate_id)
                .all(|other| {
                    directed_pairs.contains(&(
                        candidate_id.to_owned(),
                        other.as_str().to_owned(),
                    ))
                })
        })
        .cloned()
        .collect();

    if winner_candidates.len() == 1 && eligible.len() > 1 {
        let candidate = winner_candidates.into_iter().next();
        return Ok(build(
            PreferenceReplayActionV1::Recommend,
            PreferenceReplayReasonV1::UniqueExplicitPairwiseWinner,
            candidate,
        ));
    }

    let reason = if matched.is_empty() {
        PreferenceReplayReasonV1::NoContextMatchedPreferenceEvidence
    } else {
        PreferenceReplayReasonV1::InsufficientPairwiseCoverage
    };
    let action = if remaining_budget > 0 {
        PreferenceReplayActionV1::Query
    } else {
        PreferenceReplayActionV1::Abstain
    };
    Ok(build(action, reason, None))
}

/// Deterministic report from replaying every episode in one validated fixture.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceReplayBenchmarkReportV1 {
    /// Fixture schema version.
    pub schema_version: u64,
    /// Version string identifying the synthetic fixture corpus.
    pub fixture_version: String,
    /// Number of candidates in the fixture catalog.
    pub candidate_count: usize,
    /// Number of replayed scenarios.
    pub scenario_count: usize,
    /// Number of development scenarios.
    pub development_scenario_count: usize,
    /// Number of held-out scenarios.
    pub held_out_scenario_count: usize,
    /// Number of synthetic preference/rejection observations in the corpus.
    pub observation_count: usize,
    /// Replay action counts; these are synthetic fixture dispositions, not real-world rates.
    pub recommendation_count: usize,
    /// Number of episodes where the policy requests current-context feedback.
    pub query_count: usize,
    /// Number of episodes where it abstains.
    pub abstention_count: usize,
    /// Number of episodes where explicit feedback rejects every eligible candidate.
    pub reject_all_count: usize,
    /// Number of episodes with no technically eligible candidate.
    pub no_feasible_candidate_count: usize,
    /// Per-scenario deterministic reports, preserving fixture order.
    pub assessments: Vec<PreferenceReplayAssessmentV1>,
    /// Explicit limitation of any inference from this synthetic replay.
    pub claim_ceiling: String,
}

/// Validate and replay the synthetic fixture into a deterministic, report-only benchmark receipt.
///
/// The policy input is allow-listed to scenario ID, task/context IDs, eligible candidate IDs,
/// query budget, and explicit preference observations. Evaluation oracles, scenario split labels,
/// synthetic engineering profiles, and other evaluation metadata are not copied into a policy
/// episode. This entry point reports baseline behavior; it does not compare learned policies or
/// estimate real-world preference accuracy.
pub fn replay_preference_fixture(
    input: &str,
) -> Result<PreferenceReplayBenchmarkReportV1, Vec<PreferenceDriftFixtureFindingV1>> {
    let summary = validate_preference_drift_fixture(input)?;
    let root: Value = serde_json::from_str(input).map_err(|error| {
        vec![PreferenceDriftFixtureFindingV1 {
            path: "$".to_owned(),
            reason: format!("invalid JSON after validation: {error}"),
        }]
    })?;
    let fixture_version = nonempty_string(&root, "fixture_version").ok_or_else(|| {
        vec![PreferenceDriftFixtureFindingV1 {
            path: "$.fixture_version".to_owned(),
            reason: "missing fixture version".to_owned(),
        }]
    })?;
    let scenarios = root
        .get("scenarios")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            vec![PreferenceDriftFixtureFindingV1 {
                path: "$.scenarios".to_owned(),
                reason: "missing scenario array after validation".to_owned(),
            }]
        })?;

    let mut assessments = Vec::with_capacity(scenarios.len());
    for scenario in scenarios {
        let policy_input = serde_json::json!({
            "scenario_id": scenario.get("scenario_id"),
            "task_id": scenario.get("task_id"),
            "current_context_id": scenario.get("current_context_id"),
            "eligible_candidate_ids": scenario.get("eligible_candidate_ids"),
            "query_budget_total": scenario.get("query_budget_total"),
            "queries_already_used": scenario.get("queries_already_used"),
            "preference_observations": scenario.get("preference_observations"),
        });
        assessments.push(replay_preference_episode(&policy_input)?);
    }

    let recommendation_count = assessments
        .iter()
        .filter(|item| item.action == PreferenceReplayActionV1::Recommend)
        .count();
    let query_count = assessments
        .iter()
        .filter(|item| item.action == PreferenceReplayActionV1::Query)
        .count();
    let abstention_count = assessments
        .iter()
        .filter(|item| item.action == PreferenceReplayActionV1::Abstain)
        .count();
    let reject_all_count = assessments
        .iter()
        .filter(|item| item.action == PreferenceReplayActionV1::RejectAll)
        .count();
    let no_feasible_candidate_count = assessments
        .iter()
        .filter(|item| item.action == PreferenceReplayActionV1::NoFeasibleCandidate)
        .count();

    Ok(PreferenceReplayBenchmarkReportV1 {
        schema_version: summary.schema_version,
        fixture_version,
        candidate_count: summary.candidate_count,
        scenario_count: summary.scenario_count,
        development_scenario_count: summary.development_scenario_count,
        held_out_scenario_count: summary.held_out_scenario_count,
        observation_count: summary.observation_count,
        recommendation_count,
        query_count,
        abstention_count,
        reject_all_count,
        no_feasible_candidate_count,
        assessments,
        claim_ceiling: "Synthetic replay behavior only; no real wearer preference, safety, clinical, or deployment claim."
            .to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!(concat!(
        "../../../../docs/research/open-mobility/fixtures/",
        "preference-drift-scenarios-v1.json"
    ));

    fn parse_fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("checked-in preference fixture must be valid JSON")
    }

    fn encode(value: &Value) -> String {
        serde_json::to_string(value).expect("fixture mutation remains serializable")
    }

    fn has_path_error(findings: &[PreferenceDriftFixtureFindingV1], path: &str) -> bool {
        findings.iter().any(|item| item.path == path)
    }

    #[test]
    fn batch_replay_emits_complete_deterministic_receipt() {
        let report = replay_preference_fixture(FIXTURE)
            .expect("checked-in fixture should replay");
        assert_eq!(report.schema_version, 1);
        assert_eq!(report.fixture_version, "0.1.0");
        assert_eq!(report.candidate_count, 3);
        assert_eq!(report.scenario_count, 12);
        assert_eq!(report.development_scenario_count, 2);
        assert_eq!(report.held_out_scenario_count, 10);
        assert_eq!(report.observation_count, 19);
        assert_eq!(report.assessments.len(), 12);
        assert_eq!(report.recommendation_count, 4);
        assert_eq!(report.query_count, 6);
        assert_eq!(report.abstention_count, 1);
        assert_eq!(report.reject_all_count, 1);
        assert_eq!(report.no_feasible_candidate_count, 0);
    }

    #[test]
    fn batch_policy_receipt_is_invariant_to_oracle_ranking_changes() {
        let mut fixture = parse_fixture();
        let expected = replay_preference_fixture(&encode(&fixture)).unwrap();
        fixture["scenarios"][0]["evaluation_oracle"]["preferred_order"] =
            serde_json::json!(["assist", "stable", "flex"]);
        let changed = replay_preference_fixture(&encode(&fixture)).unwrap();
        assert_eq!(changed, expected);
    }

    #[test]
    fn batch_replay_report_is_byte_stable_for_one_hundred_repeats() {
        let expected = serde_json::to_vec(&replay_preference_fixture(FIXTURE).unwrap()).unwrap();
        for _ in 0..100 {
            let actual =
                serde_json::to_vec(&replay_preference_fixture(FIXTURE).unwrap()).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn empty_eligible_set_returns_no_feasible_candidate() {
        let fixture = parse_fixture();
        let mut scenario = scenario_by_id(&fixture, "DEV-WALK-001").clone();
        scenario["eligible_candidate_ids"] = serde_json::json!([]);
        let report = replay_preference_episode(&scenario).unwrap();
        assert_eq!(report.action, PreferenceReplayActionV1::NoFeasibleCandidate);
        assert_eq!(report.recommended_candidate_id, None);
    }

    #[test]
    fn versioned_fixture_validates_and_reports_expected_counts() {
        let summary = validate_preference_drift_fixture(FIXTURE)
            .expect("the versioned fixture must satisfy its contract");
        assert_eq!(summary.schema_version, 1);
        assert_eq!(summary.candidate_count, 3);
        assert_eq!(summary.scenario_count, 12);
        assert_eq!(summary.development_scenario_count, 2);
        assert_eq!(summary.held_out_scenario_count, 10);
        assert_eq!(summary.observation_count, 19);
    }

    #[test]
    fn repeated_validation_is_deterministic() {
        let expected = validate_preference_drift_fixture(FIXTURE).unwrap();
        for _ in 0..100 {
            assert_eq!(validate_preference_drift_fixture(FIXTURE).unwrap(), expected);
        }
    }

    #[test]
    fn unsupported_schema_fails_closed() {
        let mut value = parse_fixture();
        value["schema_version"] = serde_json::json!(99);
        let findings = validate_preference_drift_fixture(&encode(&value)).unwrap_err();
        assert!(has_path_error(&findings, "$.schema_version"));
    }

    #[test]
    fn duplicate_scenario_identity_is_rejected() {
        let mut value = parse_fixture();
        let duplicate_id = value["scenarios"][0]["scenario_id"].clone();
        value["scenarios"][1]["scenario_id"] = duplicate_id;
        let findings = validate_preference_drift_fixture(&encode(&value)).unwrap_err();
        assert!(findings.iter().any(|item| item.reason.contains("duplicate scenario ID")));
    }

    #[test]
    fn evaluation_oracle_cannot_be_marked_as_policy_input() {
        let mut value = parse_fixture();
        value["scenarios"][0]["evaluation_oracle"]["visibility"] =
            serde_json::json!("policy_input");
        let findings = validate_preference_drift_fixture(&encode(&value)).unwrap_err();
        assert!(has_path_error(
            &findings,
            "$.scenarios[0].evaluation_oracle.visibility"
        ));
    }

    #[test]
    fn unknown_candidate_references_are_rejected() {
        let mut value = parse_fixture();
        value["scenarios"][0]["eligible_candidate_ids"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!("not-in-catalog"));
        let findings = validate_preference_drift_fixture(&encode(&value)).unwrap_err();
        assert!(has_path_error(
            &findings,
            "$.scenarios[0].eligible_candidate_ids"
        ));
    }

    #[test]
    fn query_budget_inversion_is_rejected() {
        let mut value = parse_fixture();
        value["scenarios"][9]["queries_already_used"] = serde_json::json!(2);
        let findings = validate_preference_drift_fixture(&encode(&value)).unwrap_err();
        assert!(has_path_error(
            &findings,
            "$.scenarios[9].query_budget_total"
        ));
    }

    #[test]
    fn preference_evidence_requires_explicit_feedback_provenance() {
        let mut value = parse_fixture();
        value["scenarios"][0]["preference_observations"][0]["source"] =
            serde_json::json!("sensor_inferred_preference");
        let findings = validate_preference_drift_fixture(&encode(&value)).unwrap_err();
        assert!(has_path_error(
            &findings,
            "$.scenarios[0].preference_observations[0].source"
        ));
    }

    #[test]
    fn hard_ineligible_candidate_cannot_remain_eligible() {
        let mut value = parse_fixture();
        value["scenarios"][5]["eligible_candidate_ids"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!("assist"));
        let findings = validate_preference_drift_fixture(&encode(&value)).unwrap_err();
        assert!(findings.iter().any(|item| {
            item.reason.contains("violates the declared mass ceiling")
                || item.reason.contains("hard-ineligible candidate")
        }));
    }

    #[test]
    fn contradictory_feedback_does_not_require_an_oracle_total_order() {
        let value = parse_fixture();
        let summary = validate_preference_drift_fixture(&encode(&value)).unwrap();
        assert_eq!(summary.scenario_count, 12);
        let conflict = value["scenarios"]
            .as_array()
            .unwrap()
            .iter()
            .find(|scenario| {
                scenario["scenario_id"] == serde_json::json!("HOLD-CONFLICT-001")
            })
            .unwrap();
        assert_eq!(
            conflict["evaluation_oracle"]["status"],
            serde_json::json!("insufficient_evidence")
        );
        assert!(conflict["evaluation_oracle"]["preferred_order"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn mixed_type_candidate_arrays_fail_closed() {
        let mut value = parse_fixture();
        value["scenarios"][0]["eligible_candidate_ids"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!(false));
        let findings = validate_preference_drift_fixture(&encode(&value)).unwrap_err();
        assert!(has_path_error(
            &findings,
            "$.scenarios[0].eligible_candidate_ids"
        ));
    }

    fn scenario_by_id<'a>(fixture: &'a Value, scenario_id: &str) -> &'a Value {
        fixture["scenarios"]
            .as_array()
            .expect("fixture scenario array")
            .iter()
            .find(|scenario| scenario["scenario_id"] == serde_json::json!(scenario_id))
            .expect("scenario ID exists")
    }

    #[test]
    fn explicit_context_matched_pairwise_winner_is_recommended() {
        let fixture = parse_fixture();
        let report = replay_preference_episode(scenario_by_id(&fixture, "DEV-WALK-001"))
            .expect("valid synthetic episode");
        assert_eq!(report.action, PreferenceReplayActionV1::Recommend);
        assert_eq!(report.recommended_candidate_id.as_deref(), Some("flex"));
        assert_eq!(report.context_matched_observation_count, 2);
        assert_eq!(report.stale_or_mismatched_observation_count, 0);
    }

    #[test]
    fn stale_preference_is_not_transferred_across_fatigue_context() {
        let fixture = parse_fixture();
        let report = replay_preference_episode(scenario_by_id(&fixture, "HOLD-SHIFT-001"))
            .expect("valid synthetic episode");
        assert_eq!(report.action, PreferenceReplayActionV1::Query);
        assert_eq!(
            report.reason,
            PreferenceReplayReasonV1::NoContextMatchedPreferenceEvidence
        );
        assert_eq!(report.context_matched_observation_count, 0);
        assert_eq!(report.stale_or_mismatched_observation_count, 2);
    }

    #[test]
    fn context_blind_baseline_fails_the_fatigue_shift_control() {
        let fixture = parse_fixture();
        let scenario = scenario_by_id(&fixture, "HOLD-SHIFT-001");
        let task = nonempty_string(scenario, "task_id").unwrap();
        let eligible = string_array(scenario, "eligible_candidate_ids").unwrap();
        let observations = scenario["preference_observations"]
            .as_array()
            .expect("observation array");
        let mut task_only_pairs = HashSet::<(String, String)>::new();
        for observation in observations {
            if nonempty_string(observation, "task_id").as_deref() == Some(task.as_str()) {
                if let (Some(preferred), Some(rejected)) = (
                    nonempty_string(observation, "preferred_candidate_id"),
                    nonempty_string(observation, "rejected_candidate_id"),
                ) {
                    task_only_pairs.insert((preferred, rejected));
                }
            }
        }
        let context_blind_winner = eligible.iter().find(|candidate| {
            let candidate_id = candidate.as_str();
            eligible.iter().filter(|other| other.as_str() != candidate_id).all(|other| {
                task_only_pairs.contains(&(
                    candidate_id.to_owned(),
                    other.as_str().to_owned(),
                ))
            })
        });
        assert_eq!(context_blind_winner.map(String::as_str), Some("flex"));

        let context_aware = replay_preference_episode(scenario).unwrap();
        assert_eq!(context_aware.action, PreferenceReplayActionV1::Query);
        assert_eq!(context_aware.recommended_candidate_id, None);
    }

    #[test]
    fn contradictory_pairwise_feedback_requests_clarification() {
        let fixture = parse_fixture();
        let report = replay_preference_episode(scenario_by_id(&fixture, "HOLD-CONFLICT-001"))
            .expect("valid synthetic episode");
        assert_eq!(report.action, PreferenceReplayActionV1::Query);
        assert_eq!(report.reason, PreferenceReplayReasonV1::ContradictoryFeedback);
    }

    #[test]
    fn hard_ineligible_preferred_candidate_is_never_recommended() {
        let fixture = parse_fixture();
        let report = replay_preference_episode(scenario_by_id(&fixture, "HOLD-GATE-001"))
            .expect("valid synthetic episode");
        assert_eq!(report.action, PreferenceReplayActionV1::Recommend);
        assert_eq!(report.recommended_candidate_id.as_deref(), Some("stable"));
        assert!(!report.eligible_candidate_ids.iter().any(|id| id == "assist"));
    }

    #[test]
    fn explicit_rejection_of_every_eligible_option_is_preserved() {
        let fixture = parse_fixture();
        let report = replay_preference_episode(scenario_by_id(&fixture, "HOLD-REJECT-001"))
            .expect("valid synthetic episode");
        assert_eq!(report.action, PreferenceReplayActionV1::RejectAll);
        assert_eq!(report.recommended_candidate_id, None);
    }

    #[test]
    fn exhausted_query_budget_causes_abstention_on_stale_evidence() {
        let fixture = parse_fixture();
        let report = replay_preference_episode(scenario_by_id(&fixture, "HOLD-BUDGET-001"))
            .expect("valid synthetic episode");
        assert_eq!(report.action, PreferenceReplayActionV1::Abstain);
        assert_eq!(report.remaining_query_budget, 0);
        assert_eq!(report.stale_or_mismatched_observation_count, 1);
    }

    #[test]
    fn absence_of_preference_evidence_does_not_create_a_ranking() {
        let fixture = parse_fixture();
        let report = replay_preference_episode(scenario_by_id(&fixture, "HOLD-UNKNOWN-001"))
            .expect("valid synthetic episode");
        assert_eq!(report.action, PreferenceReplayActionV1::Query);
        assert_eq!(report.recommended_candidate_id, None);
    }

    #[test]
    fn incomplete_pairwise_coverage_queries_instead_of_inventing_a_winner() {
        let fixture = parse_fixture();
        let report = replay_preference_episode(scenario_by_id(&fixture, "HOLD-TRANSFER-001"))
            .expect("valid synthetic episode");
        assert_eq!(report.action, PreferenceReplayActionV1::Query);
        assert_eq!(
            report.reason,
            PreferenceReplayReasonV1::NoContextMatchedPreferenceEvidence
        );
    }

    #[test]
    fn evaluation_oracle_changes_cannot_change_policy_replay() {
        let fixture = parse_fixture();
        let original = scenario_by_id(&fixture, "DEV-WALK-001");
        let expected = replay_preference_episode(original).expect("valid synthetic episode");
        let mut changed = original.clone();
        changed["evaluation_oracle"]["preferred_order"] =
            serde_json::json!(["assist", "stable", "flex"]);
        changed["evaluation_oracle"]["acceptable_candidate_ids"] = serde_json::json!(["assist"]);
        let actual = replay_preference_episode(&changed).expect("mutated evaluation oracle ignored");
        assert_eq!(actual, expected);
    }

    #[test]
    fn replay_output_is_byte_stable_for_one_hundred_repeats() {
        let fixture = parse_fixture();
        let scenario = scenario_by_id(&fixture, "HOLD-NOISE-001");
        let expected = serde_json::to_vec(&replay_preference_episode(scenario).unwrap()).unwrap();
        for _ in 0..100 {
            let actual =
                serde_json::to_vec(&replay_preference_episode(scenario).unwrap()).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn malformed_replay_budget_fails_closed() {
        let fixture = parse_fixture();
        let mut scenario = scenario_by_id(&fixture, "DEV-WALK-001").clone();
        scenario["queries_already_used"] = serde_json::json!(99);
        assert!(replay_preference_episode(&scenario).is_err());
    }

    #[test]
    fn malformed_current_context_id_fails_closed() {
        let mut fixture = parse_fixture();
        fixture["scenarios"][0]
            .as_object_mut()
            .expect("scenario object")
            .remove("current_context_id");
        let findings = validate_preference_drift_fixture(&encode(&fixture)).unwrap_err();
        assert!(has_path_error(
            &findings,
            "$.scenarios[0].current_context_id"
        ));
    }

    #[test]
    fn malformed_pairwise_feedback_is_rejected() {
        let mut value = parse_fixture();
        value["scenarios"][0]["preference_observations"][0]["rejected_candidate_id"] =
            serde_json::json!("flex");
        let findings = validate_preference_drift_fixture(&encode(&value)).unwrap_err();
        assert!(findings
            .iter()
            .any(|item| item.reason.contains("distinct candidates")));
    }
}
