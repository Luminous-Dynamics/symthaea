// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Offline, synthetic query-budget frontier for preference-drift episodes.
//!
//! Query selection receives an allow-listed policy view. The evaluation oracle is consulted only
//! after a candidate pair has been frozen, to simulate a synthetic response, and after all episodes
//! finish, to score outcomes. This is not a human study, preference model, hardware controller, or
//! evidence about real wearer interaction cost.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::preference_drift_benchmark::{
    replay_preference_episode, validate_preference_drift_fixture,
    PreferenceDriftFixtureFindingV1, PreferenceReplayActionV1, PreferenceReplayReasonV1,
};

/// Schema version for the interaction-budget frontier report.
pub const PREFERENCE_QUERY_FRONTIER_SCHEMA_VERSION_V1: u64 = 1;
const MAX_QUERY_BUDGET_FOR_FRONTIER_V1: u64 = 8;

/// Query selector used in the controlled synthetic interaction simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceQuerySelectionStrategyV1 {
    /// Ask the first unobserved eligible pair in canonical candidate-ID order.
    LexicographicUnaskedPair,
    /// Select an unobserved eligible pair using a stable seed/scenario/query-index hash.
    SeededRandomUnaskedPair,
    /// Prefer a pair whose candidates have the largest number of unresolved direct comparisons.
    MaximizeUncomparedCandidateCoverage,
}

/// One simulated query and its synthetic response disposition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceQueryReceiptV1 {
    /// One-based query index within the episode.
    pub query_index: usize,
    /// First candidate ID in canonical lexical order.
    pub candidate_a_id: String,
    /// Second candidate ID in canonical lexical order.
    pub candidate_b_id: String,
    /// Synthetic time step when the pair was selected.
    pub asked_at_step: u64,
    /// Synthetic time step when the response was simulated.
    pub answered_at_step: u64,
    /// One of simulated_pairwise_feedback, simulated_reject_all, or no_synthetic_answer_available.
    pub response_kind: String,
}

/// Interactive replay disposition for one scenario and query budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceQueryEpisodeAssessmentV1 {
    /// Scenario identity.
    pub scenario_id: String,
    /// Query-selection strategy used.
    pub strategy: PreferenceQuerySelectionStrategyV1,
    /// Maximum number of pairwise questions this run could ask.
    pub query_budget_limit: u64,
    /// Final disposition after simulated query responses and replay.
    pub final_action: PreferenceReplayActionV1,
    /// Final evidence-driven reason from replay or selector exhaustion.
    pub final_reason: PreferenceReplayReasonV1,
    /// Candidate selected, if the final action recommends one.
    pub recommended_candidate_id: Option<String>,
    /// Number of pairwise questions asked, including unanswered questions.
    pub query_count: usize,
    /// Questions for which the synthetic oracle could not provide a ranking answer.
    pub unanswered_query_count: usize,
    /// True when replay requested a query but no unasked eligible pair remained.
    pub query_selector_exhausted: bool,
    /// Ordered query/response provenance.
    pub queries: Vec<PreferenceQueryReceiptV1>,
}

/// Per-family disposition counts for one query budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceQueryBudgetFamilyScoreV1 {
    /// Number of episodes in this scenario family.
    pub scenario_count: usize,
    /// Episodes for which the evaluation oracle supplies an acceptable-option ranking.
    pub ranking_scenario_count: usize,
    /// Final recommendations on ranked scenarios.
    pub recommendation_count: usize,
    /// Recommendations in the oracle's acceptable set.
    pub acceptable_recommendation_count: usize,
    /// Recommendations outside the oracle's acceptable set.
    pub unacceptable_recommendation_count: usize,
    /// Pairwise questions asked across family episodes.
    pub query_count_total: usize,
    /// Questions with no ranking answer available from the synthetic response oracle.
    pub unanswered_query_count_total: usize,
    /// Episodes that finish by abstaining.
    pub abstention_count: usize,
}

/// Summary of all episodes at one maximum-query budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceQueryBudgetFrontierPointV1 {
    /// Maximum number of additional questions per episode, inclusive of this frontier point.
    pub query_budget_limit: u64,
    /// Number of episodes scored.
    pub scenario_count: usize,
    /// Number of episodes with an oracle ranking.
    pub ranking_scenario_count: usize,
    /// Final recommendations on ranked scenarios.
    pub recommendation_count: usize,
    /// Recommendations within each oracle's acceptable set.
    pub acceptable_recommendation_count: usize,
    /// Recommendations outside each oracle's acceptable set.
    pub unacceptable_recommendation_count: usize,
    /// Recommendations emitted when the oracle supplies no ranking.
    pub recommendation_without_ranking_count: usize,
    /// Total questions asked across episodes.
    pub query_count_total: usize,
    /// Total questions for which no ranking answer was available.
    pub unanswered_query_count_total: usize,
    /// Episodes that finish by abstaining.
    pub abstention_count: usize,
    /// Episodes that finish by explicitly rejecting all currently eligible options.
    pub reject_all_count: usize,
    /// Correct explicit reject-all dispositions.
    pub correct_reject_all_count: usize,
    /// Reject-all dispositions when the oracle does not label reject-all.
    pub incorrect_reject_all_count: usize,
    /// Episodes with no technically eligible candidate.
    pub no_feasible_candidate_count: usize,
    /// Abstentions after using the full positive query budget.
    pub query_budget_exhaustion_abstention_count: usize,
    /// Family-scoped counts, scored after episode decisions are fixed.
    pub family_scores: BTreeMap<String, PreferenceQueryBudgetFamilyScoreV1>,
    /// Per-scenario dispositions with query receipts.
    pub assessments: Vec<PreferenceQueryEpisodeAssessmentV1>,
}

/// Comparison of query cost versus final disposition for two query selectors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceQueryBudgetFrontierReportV1 {
    /// Report schema version.
    pub schema_version: u64,
    /// Synthetic fixture version.
    pub fixture_version: String,
    /// Root seed for the seeded-random query selector.
    pub seed: u64,
    /// Highest query budget included; all limits from zero through this value are reported.
    pub max_query_budget: u64,
    /// Scenario count used at every budget.
    pub scenario_count: usize,
    /// Lexicographic query-selector frontier.
    pub lexicographic_unasked_pair: Vec<PreferenceQueryBudgetFrontierPointV1>,
    /// Seeded-random query-selector frontier.
    pub seeded_random_unasked_pair: Vec<PreferenceQueryBudgetFrontierPointV1>,
    /// Coverage-deficit query-selector frontier.
    pub coverage_deficit_unasked_pair: Vec<PreferenceQueryBudgetFrontierPointV1>,
    /// Explicit separation between policy input, simulated responses, and scoring.
    pub oracle_separation_contract: String,
    /// Limit on all claims made by this synthetic interaction simulation.
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SimulatedPreferenceResponseV1 {
    Pairwise { preferred: String, rejected: String },
    RejectAll,
    Unanswered,
}

fn finding(
    path: impl Into<String>,
    reason: impl Into<String>,
) -> Vec<PreferenceDriftFixtureFindingV1> {
    vec![PreferenceDriftFixtureFindingV1 {
        path: path.into(),
        reason: reason.into(),
    }]
}

fn canonical_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_owned(), b.to_owned())
    } else {
        (b.to_owned(), a.to_owned())
    }
}

fn current_policy_view(
    scenario: &Value,
    current_step: u64,
    query_budget: u64,
    queries_used: u64,
    observations: &[Value],
) -> Value {
    json!({
        "scenario_id": scenario.get("scenario_id"),
        "task_id": scenario.get("task_id"),
        "current_context_id": scenario.get("current_context_id"),
        "current_step": current_step,
        "preference_max_age_steps": scenario.get("preference_max_age_steps"),
        "eligible_candidate_ids": scenario.get("eligible_candidate_ids"),
        "query_budget_total": query_budget,
        "queries_already_used": queries_used,
        "preference_observations": observations
    })
}

fn known_current_pairs(policy_view: &Value) -> HashSet<(String, String)> {
    let task_id = policy_view.get("task_id").and_then(Value::as_str);
    let context_id = policy_view.get("current_context_id").and_then(Value::as_str);
    let current_step = policy_view.get("current_step").and_then(Value::as_u64);
    let max_age = policy_view
        .get("preference_max_age_steps")
        .and_then(Value::as_u64);
    let mut known = HashSet::new();

    let observations = policy_view
        .get("preference_observations")
        .and_then(Value::as_array);
    if let (Some(task_id), Some(context_id), Some(current_step), Some(max_age), Some(observations)) =
        (task_id, context_id, current_step, max_age, observations)
    {
        for observation in observations {
            let matches_context =
                observation.get("task_id").and_then(Value::as_str) == Some(task_id)
                    && observation.get("context_id").and_then(Value::as_str) == Some(context_id);
            let step = observation.get("step").and_then(Value::as_u64);
            let is_fresh = step.is_some_and(|step| {
                step <= current_step && current_step - step <= max_age
            });
            if !matches_context || !is_fresh {
                continue;
            }

            let preferred = observation
                .get("preferred_candidate_id")
                .and_then(Value::as_str);
            let rejected = observation
                .get("rejected_candidate_id")
                .and_then(Value::as_str);
            if let (Some(preferred), Some(rejected)) = (preferred, rejected) {
                if preferred != rejected {
                    known.insert(canonical_pair(preferred, rejected));
                }
            }
        }
    }
    known
}

/// Choose a question using only the current policy view, asked-pair history, strategy, and seed.
/// This function has no scenario/oracle parameter by design.
fn select_next_pair(
    policy_view: &Value,
    already_asked: &HashSet<(String, String)>,
    strategy: PreferenceQuerySelectionStrategyV1,
    seed: u64,
    query_index: usize,
) -> Option<(String, String)> {
    let mut eligible = policy_view
        .get("eligible_candidate_ids")?
        .as_array()?
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    eligible.sort_unstable();
    eligible.dedup();

    let eligible_set: HashSet<String> = eligible.iter().cloned().collect();
    let known: HashSet<(String, String)> = known_current_pairs(policy_view)
        .into_iter()
        .filter(|(a, b)| eligible_set.contains(a) && eligible_set.contains(b))
        .collect();
    let mut candidates = Vec::new();
    for i in 0..eligible.len() {
        for j in (i + 1)..eligible.len() {
            let pair = canonical_pair(&eligible[i], &eligible[j]);
            if !known.contains(&pair) && !already_asked.contains(&pair) {
                candidates.push(pair);
            }
        }
    }
    if candidates.is_empty() {
        return None;
    }

    match strategy {
        PreferenceQuerySelectionStrategyV1::LexicographicUnaskedPair => {
            candidates.into_iter().next()
        }
        PreferenceQuerySelectionStrategyV1::SeededRandomUnaskedPair => {
            let scenario_id = policy_view.get("scenario_id").and_then(Value::as_str)?;
            let key = format!("{seed}:{scenario_id}:{query_index}");
            let mut hash = 0xcbf29ce484222325u64;
            for byte in key.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
            candidates.get((hash % candidates.len() as u64) as usize).cloned()
        }
        PreferenceQuerySelectionStrategyV1::MaximizeUncomparedCandidateCoverage => {
            // Count only directly observed, fresh comparisons. No transitive preference inference.
            let mut compared_per_candidate = BTreeMap::<String, usize>::new();
            for pair in &known {
                *compared_per_candidate.entry(pair.0.clone()).or_default() += 1;
                *compared_per_candidate.entry(pair.1.clone()).or_default() += 1;
            }
            let mut best_pair: Option<(usize, (String, String))> = None;
            for pair in candidates {
                let left_coverage = compared_per_candidate.get(&pair.0).copied().unwrap_or_default();
                let right_coverage = compared_per_candidate.get(&pair.1).copied().unwrap_or_default();
                let unresolved_score = eligible.len().saturating_sub(1 + left_coverage)
                    + eligible.len().saturating_sub(1 + right_coverage);
                if best_pair.as_ref().is_none_or(|(best_score, _)| unresolved_score > *best_score) {
                    best_pair = Some((unresolved_score, pair));
                }
            }
            best_pair.map(|(_, pair)| pair)
        }
    }
}

/// Simulate a response only after the candidate pair has already been selected.
fn simulate_response(
    scenario: &Value,
    pair: &(String, String),
) -> SimulatedPreferenceResponseV1 {
    let oracle = scenario.get("evaluation_oracle").unwrap_or(&Value::Null);
    match oracle.get("status").and_then(Value::as_str) {
        Some("reject_all") => SimulatedPreferenceResponseV1::RejectAll,
        Some("ranking") => {
            let order = oracle.get("preferred_order").and_then(Value::as_array);
            let positions = order
                .map(|order| {
                    let position = |candidate: &str| {
                        order.iter().position(|item| item.as_str() == Some(candidate))
                    };
                    (position(&pair.0), position(&pair.1))
                })
                .unwrap_or((None, None));
            match positions {
                (Some(a), Some(b)) if a < b => SimulatedPreferenceResponseV1::Pairwise {
                    preferred: pair.0.clone(),
                    rejected: pair.1.clone(),
                },
                (Some(a), Some(b)) if b < a => SimulatedPreferenceResponseV1::Pairwise {
                    preferred: pair.1.clone(),
                    rejected: pair.0.clone(),
                },
                _ => SimulatedPreferenceResponseV1::Unanswered,
            }
        }
        _ => SimulatedPreferenceResponseV1::Unanswered,
    }
}

fn append_simulated_response(
    scenario: &Value,
    observations: &mut Vec<Value>,
    eligible: &[String],
    current_step: u64,
    query_index: usize,
    response: &SimulatedPreferenceResponseV1,
) -> String {
    let scenario_id = scenario
        .get("scenario_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown-scenario");
    let task_id = scenario.get("task_id").and_then(Value::as_str).unwrap_or("");
    let context_id = scenario
        .get("current_context_id")
        .and_then(Value::as_str)
        .unwrap_or("");

    match response {
        SimulatedPreferenceResponseV1::Pairwise { preferred, rejected } => {
            observations.push(json!({
                "observation_id": format!("{scenario_id}-INTERACTION-Q{query_index:02}"),
                "step": current_step,
                "task_id": task_id,
                "context_id": context_id,
                "source": "explicit_synthetic_simulated_query_feedback",
                "preferred_candidate_id": preferred,
                "rejected_candidate_id": rejected
            }));
            "simulated_pairwise_feedback".to_owned()
        }
        SimulatedPreferenceResponseV1::RejectAll => {
            observations.push(json!({
                "observation_id": format!("{scenario_id}-INTERACTION-Q{query_index:02}"),
                "step": current_step,
                "task_id": task_id,
                "context_id": context_id,
                "source": "explicit_synthetic_simulated_query_reject_all",
                "rejected_candidate_ids": eligible
            }));
            "simulated_reject_all".to_owned()
        }
        SimulatedPreferenceResponseV1::Unanswered => {
            "no_synthetic_answer_available".to_owned()
        }
    }
}

fn run_interactive_episode(
    scenario: &Value,
    query_budget: u64,
    strategy: PreferenceQuerySelectionStrategyV1,
    seed: u64,
) -> Result<PreferenceQueryEpisodeAssessmentV1, Vec<PreferenceDriftFixtureFindingV1>> {
    let scenario_id = scenario
        .get("scenario_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| finding("$.scenario_id", "missing scenario ID"))?;
    let mut current_step = scenario
        .get("current_step")
        .and_then(Value::as_u64)
        .ok_or_else(|| finding("$.current_step", "missing or invalid current_step"))?;
    let eligible = scenario
        .get("eligible_candidate_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| finding("$.eligible_candidate_ids", "missing eligible-candidate array"))?
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| finding("$.eligible_candidate_ids", "candidate IDs must be strings"))?
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut observations = scenario
        .get("preference_observations")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| finding("$.preference_observations", "missing observation array"))?;

    let prior_queries_used = scenario
        .get("queries_already_used")
        .and_then(Value::as_u64)
        .ok_or_else(|| finding("$.queries_already_used", "missing or invalid prior query count"))?;
    let mut asked_pairs = HashSet::<(String, String)>::new();
    let mut queries = Vec::<PreferenceQueryReceiptV1>::new();
    let mut unanswered_query_count = 0usize;
    let mut query_selector_exhausted = false;

    loop {
        let total_budget = prior_queries_used.checked_add(query_budget).ok_or_else(|| {
            finding("$.query_budget", "prior plus additional query budget would overflow")
        })?;
        let cumulative_queries_used = prior_queries_used
            .checked_add(queries.len() as u64)
            .ok_or_else(|| finding("$.queries_already_used", "query count would overflow"))?;
        let policy_view = current_policy_view(
            scenario,
            current_step,
            total_budget,
            cumulative_queries_used,
            &observations,
        );
        let assessment = replay_preference_episode(&policy_view)?;
        if assessment.action != PreferenceReplayActionV1::Query {
            return Ok(PreferenceQueryEpisodeAssessmentV1 {
                scenario_id,
                strategy,
                query_budget_limit: query_budget,
                final_action: assessment.action,
                final_reason: assessment.reason,
                recommended_candidate_id: assessment.recommended_candidate_id,
                query_count: queries.len(),
                unanswered_query_count,
                query_selector_exhausted,
                queries,
            });
        }
        if queries.len() >= query_budget as usize {
            return Ok(PreferenceQueryEpisodeAssessmentV1 {
                scenario_id,
                strategy,
                query_budget_limit: query_budget,
                final_action: PreferenceReplayActionV1::Abstain,
                final_reason: assessment.reason,
                recommended_candidate_id: None,
                query_count: queries.len(),
                unanswered_query_count,
                query_selector_exhausted,
                queries,
            });
        }

        let query_index = queries.len();
        let pair = select_next_pair(
            &policy_view,
            &asked_pairs,
            strategy,
            seed,
            query_index,
        );
        let Some(pair) = pair else {
            query_selector_exhausted = true;
            return Ok(PreferenceQueryEpisodeAssessmentV1 {
                scenario_id,
                strategy,
                query_budget_limit: query_budget,
                final_action: PreferenceReplayActionV1::Abstain,
                final_reason: assessment.reason,
                recommended_candidate_id: None,
                query_count: queries.len(),
                unanswered_query_count,
                query_selector_exhausted,
                queries,
            });
        };

        asked_pairs.insert(pair.clone());
        let asked_at_step = current_step;
        current_step = current_step.checked_add(1).ok_or_else(|| {
            finding("$.current_step", "synthetic interaction step would overflow")
        })?;

        // Oracle access begins only after select_next_pair has returned the pair.
        let response = simulate_response(scenario, &pair);
        let response_kind = append_simulated_response(
            scenario,
            &mut observations,
            &eligible,
            current_step,
            query_index + 1,
            &response,
        );
        if response == SimulatedPreferenceResponseV1::Unanswered {
            unanswered_query_count += 1;
        }

        queries.push(PreferenceQueryReceiptV1 {
            query_index: query_index + 1,
            candidate_a_id: pair.0,
            candidate_b_id: pair.1,
            asked_at_step,
            answered_at_step: current_step,
            response_kind,
        });
    }
}

fn score_frontier_point(
    scenarios: &[Value],
    assessments: &[PreferenceQueryEpisodeAssessmentV1],
    query_budget_limit: u64,
) -> PreferenceQueryBudgetFrontierPointV1 {
    let mut point = PreferenceQueryBudgetFrontierPointV1 {
        query_budget_limit,
        scenario_count: scenarios.len(),
        ranking_scenario_count: 0,
        recommendation_count: 0,
        acceptable_recommendation_count: 0,
        unacceptable_recommendation_count: 0,
        recommendation_without_ranking_count: 0,
        query_count_total: 0,
        unanswered_query_count_total: 0,
        abstention_count: 0,
        reject_all_count: 0,
        correct_reject_all_count: 0,
        incorrect_reject_all_count: 0,
        no_feasible_candidate_count: 0,
        query_budget_exhaustion_abstention_count: 0,
        family_scores: BTreeMap::new(),
        assessments: assessments.to_vec(),
    };

    for (scenario, assessment) in scenarios.iter().zip(assessments) {
        let oracle = scenario.get("evaluation_oracle").unwrap_or(&Value::Null);
        let oracle_status = oracle.get("status").and_then(Value::as_str);
        let family = scenario
            .get("scenario_family")
            .and_then(Value::as_str)
            .unwrap_or("unknown-family")
            .to_owned();
        let family_score = point
            .family_scores
            .entry(family)
            .or_insert_with(|| PreferenceQueryBudgetFamilyScoreV1 {
                scenario_count: 0,
                ranking_scenario_count: 0,
                recommendation_count: 0,
                acceptable_recommendation_count: 0,
                unacceptable_recommendation_count: 0,
                query_count_total: 0,
                unanswered_query_count_total: 0,
                abstention_count: 0,
            });

        point.query_count_total += assessment.query_count;
        point.unanswered_query_count_total += assessment.unanswered_query_count;
        family_score.query_count_total += assessment.query_count;
        family_score.unanswered_query_count_total += assessment.unanswered_query_count;
        family_score.scenario_count += 1;

        if oracle_status == Some("ranking") {
            point.ranking_scenario_count += 1;
            family_score.ranking_scenario_count += 1;
            if assessment.final_action == PreferenceReplayActionV1::Recommend {
                point.recommendation_count += 1;
                family_score.recommendation_count += 1;
                let acceptable = oracle
                    .get("acceptable_candidate_ids")
                    .and_then(Value::as_array)
                    .is_some_and(|ids| {
                        assessment.recommended_candidate_id.as_ref().is_some_and(|selected| {
                            ids.iter().any(|candidate| {
                                candidate.as_str() == Some(selected.as_str())
                            })
                        })
                    });
                if acceptable {
                    point.acceptable_recommendation_count += 1;
                    family_score.acceptable_recommendation_count += 1;
                } else {
                    point.unacceptable_recommendation_count += 1;
                    family_score.unacceptable_recommendation_count += 1;
                }
            }
        } else if assessment.final_action == PreferenceReplayActionV1::Recommend {
            point.recommendation_without_ranking_count += 1;
        }

        match &assessment.final_action {
            PreferenceReplayActionV1::Abstain => {
                point.abstention_count += 1;
                family_score.abstention_count += 1;
                if query_budget_limit > 0
                    && assessment.query_count as u64 >= query_budget_limit
                {
                    point.query_budget_exhaustion_abstention_count += 1;
                }
            }
            PreferenceReplayActionV1::RejectAll => {
                point.reject_all_count += 1;
                if oracle_status == Some("reject_all") {
                    point.correct_reject_all_count += 1;
                } else {
                    point.incorrect_reject_all_count += 1;
                }
            }
            PreferenceReplayActionV1::NoFeasibleCandidate => {
                point.no_feasible_candidate_count += 1;
            }
            PreferenceReplayActionV1::Recommend | PreferenceReplayActionV1::Query => {}
        }
    }
    point
}

/// Evaluate three query policies over every *additional-query* budget from zero to the configured
/// maximum. Prior questions already used in the source episode remain counted. The evaluation
/// oracle is omitted from policy views; it acts only as a simulated respondent after each question
/// has been selected and as a post-decision scorer.
pub fn evaluate_preference_query_budget_frontier(
    input: &str,
    max_query_budget: u64,
    seed: u64,
) -> Result<PreferenceQueryBudgetFrontierReportV1, Vec<PreferenceDriftFixtureFindingV1>> {
    let summary = validate_preference_drift_fixture(input)?;
    if max_query_budget > MAX_QUERY_BUDGET_FOR_FRONTIER_V1 {
        return Err(finding(
            "$.max_query_budget",
            format!(
                "max_query_budget must not exceed {}",
                MAX_QUERY_BUDGET_FOR_FRONTIER_V1
            ),
        ));
    }
    let root: Value = serde_json::from_str(input)
        .map_err(|error| finding("$", format!("invalid fixture JSON: {error}")))?;
    let fixture_version = root
        .get("fixture_version")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| finding("$.fixture_version", "missing fixture_version"))?;
    let scenarios = root
        .get("scenarios")
        .and_then(Value::as_array)
        .ok_or_else(|| finding("$.scenarios", "missing scenarios"))?;

    let mut lexicographic_points = Vec::with_capacity(max_query_budget as usize + 1);
    let mut seeded_random_points = Vec::with_capacity(max_query_budget as usize + 1);
    let mut coverage_deficit_points = Vec::with_capacity(max_query_budget as usize + 1);
    for budget in 0..=max_query_budget {
        let mut lexicographic_assessments = Vec::with_capacity(scenarios.len());
        let mut random_assessments = Vec::with_capacity(scenarios.len());
        let mut coverage_assessments = Vec::with_capacity(scenarios.len());
        for scenario in scenarios {
            lexicographic_assessments.push(run_interactive_episode(
                scenario,
                budget,
                PreferenceQuerySelectionStrategyV1::LexicographicUnaskedPair,
                seed,
            )?);
            random_assessments.push(run_interactive_episode(
                scenario,
                budget,
                PreferenceQuerySelectionStrategyV1::SeededRandomUnaskedPair,
                seed,
            )?);
            coverage_assessments.push(run_interactive_episode(
                scenario,
                budget,
                PreferenceQuerySelectionStrategyV1::MaximizeUncomparedCandidateCoverage,
                seed,
            )?);
        }
        // All episode decisions are frozen before this scoring stage reads oracle labels.
        lexicographic_points.push(score_frontier_point(
            scenarios,
            &lexicographic_assessments,
            budget,
        ));
        seeded_random_points.push(score_frontier_point(
            scenarios,
            &random_assessments,
            budget,
        ));
        coverage_deficit_points.push(score_frontier_point(
            scenarios,
            &coverage_assessments,
            budget,
        ));
    }

    Ok(PreferenceQueryBudgetFrontierReportV1 {
        schema_version: PREFERENCE_QUERY_FRONTIER_SCHEMA_VERSION_V1,
        fixture_version,
        seed,
        max_query_budget,
        scenario_count: summary.scenario_count,
        lexicographic_unasked_pair: lexicographic_points,
        seeded_random_unasked_pair: seeded_random_points,
        coverage_deficit_unasked_pair: coverage_deficit_points,
        oracle_separation_contract: concat!(
            "Candidate-pair selection reads only task/context-scoped observations, the eligible ",
            "candidate set, prior asked-pair receipts, the selector strategy, and the seed. ",
            "The evaluation oracle is used after pair selection to simulate a response and after ",
            "the complete episode to score the final disposition."
        )
        .to_owned(),
        claim_ceiling: concat!(
            "Synthetic interactive-protocol behavior only. Query counts and response rates are ",
            "not human interaction-time, cognitive burden, preference accuracy, clinical, or ",
            "deployment estimates. No physical hardware is connected."
        )
        .to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!(
        "../../../../docs/research/open-mobility/fixtures/preference-drift-scenarios-v2.json"
    );

    fn scenario_by_id<'a>(fixture: &'a Value, id: &str) -> &'a Value {
        fixture["scenarios"]
            .as_array()
            .unwrap()
            .iter()
            .find(|scenario| scenario["scenario_id"] == id)
            .expect("scenario ID exists")
    }

    #[test]
    fn query_budget_frontier_is_deterministic_and_query_cost_is_monotone() {
        let first = evaluate_preference_query_budget_frontier(FIXTURE, 3, 20261018).unwrap();
        let second = evaluate_preference_query_budget_frontier(FIXTURE, 3, 20261018).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.scenario_count, 12);
        assert_eq!(first.lexicographic_unasked_pair.len(), 4);
        assert_eq!(first.seeded_random_unasked_pair.len(), 4);
        assert_eq!(first.coverage_deficit_unasked_pair.len(), 4);

        for points in [
            &first.lexicographic_unasked_pair,
            &first.seeded_random_unasked_pair,
            &first.coverage_deficit_unasked_pair,
        ] {
            for pair in points.windows(2) {
                assert!(
                    pair[1].query_count_total >= pair[0].query_count_total,
                    "increasing the allowed budget must not lower total questions asked"
                );
            }
        }
    }

    #[test]
    fn query_selector_is_oracle_isolated_before_pair_choice() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let scenario = scenario_by_id(&fixture, "HOLD-UNKNOWN-001");
        let observations = scenario["preference_observations"].as_array().unwrap().clone();
        let policy = current_policy_view(scenario, 0, 2, 0, &observations);
        assert!(policy.get("evaluation_oracle").is_none());
        let asked = HashSet::new();
        let before = select_next_pair(
            &policy,
            &asked,
            PreferenceQuerySelectionStrategyV1::LexicographicUnaskedPair,
            55,
            0,
        );

        let mut changed_oracle = scenario.clone();
        changed_oracle["evaluation_oracle"]["preferred_order"] =
            json!(["stable", "assist", "flex"]);
        let observations = changed_oracle["preference_observations"]
            .as_array()
            .unwrap()
            .clone();
        let changed_policy = current_policy_view(&changed_oracle, 0, 2, 0, &observations);
        let after = select_next_pair(
            &changed_policy,
            &asked,
            PreferenceQuerySelectionStrategyV1::LexicographicUnaskedPair,
            55,
            0,
        );
        let before_coverage = select_next_pair(
            &policy,
            &asked,
            PreferenceQuerySelectionStrategyV1::MaximizeUncomparedCandidateCoverage,
            55,
            0,
        );
        let after_coverage = select_next_pair(
            &changed_policy,
            &asked,
            PreferenceQuerySelectionStrategyV1::MaximizeUncomparedCandidateCoverage,
            55,
            0,
        );
        assert_eq!(before, after);
        assert_eq!(before_coverage, after_coverage);
        assert_eq!(policy, changed_policy);
    }

    #[test]
    fn coverage_selector_targets_less_compared_candidates_without_transitivity() {
        let policy = json!({
            "scenario_id": "coverage-control",
            "task_id": "walk",
            "current_context_id": "level-rested",
            "current_step": 0,
            "preference_max_age_steps": 2,
            "eligible_candidate_ids": ["A", "B", "C", "D"],
            "query_budget_total": 2,
            "queries_already_used": 0,
            "preference_observations": [
                {
                    "observation_id": "AB",
                    "step": 0,
                    "task_id": "walk",
                    "context_id": "level-rested",
                    "source": "explicit_synthetic_ordinal_feedback",
                    "preferred_candidate_id": "A",
                    "rejected_candidate_id": "B"
                },
                {
                    "observation_id": "AC",
                    "step": 0,
                    "task_id": "walk",
                    "context_id": "level-rested",
                    "source": "explicit_synthetic_ordinal_feedback",
                    "preferred_candidate_id": "A",
                    "rejected_candidate_id": "C"
                }
            ]
        });
        let asked = HashSet::new();
        let lexicographic = select_next_pair(
            &policy,
            &asked,
            PreferenceQuerySelectionStrategyV1::LexicographicUnaskedPair,
            1,
            0,
        );
        let coverage = select_next_pair(
            &policy,
            &asked,
            PreferenceQuerySelectionStrategyV1::MaximizeUncomparedCandidateCoverage,
            1,
            0,
        );
        assert_eq!(lexicographic, Some(("A".to_owned(), "D".to_owned())));
        assert_eq!(coverage, Some(("B".to_owned(), "D".to_owned())));
    }

    #[test]
    fn candidate_and_observation_order_do_not_change_query_frontier_receipts() {
        let original = evaluate_preference_query_budget_frontier(FIXTURE, 2, 20261018).unwrap();
        let mut fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        for scenario in fixture["scenarios"].as_array_mut().unwrap() {
            scenario["eligible_candidate_ids"]
                .as_array_mut()
                .unwrap()
                .reverse();
            scenario["preference_observations"]
                .as_array_mut()
                .unwrap()
                .reverse();
        }
        let permuted =
            evaluate_preference_query_budget_frontier(&fixture.to_string(), 2, 20261018).unwrap();
        assert_eq!(original, permuted);
    }

    #[test]
    fn reject_all_response_is_explicitly_synthetic_and_stops_interaction() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let mut scenario = scenario_by_id(&fixture, "HOLD-REJECT-001").clone();
        scenario["preference_observations"] = json!([]);
        let report = run_interactive_episode(
            &scenario,
            2,
            PreferenceQuerySelectionStrategyV1::LexicographicUnaskedPair,
            7,
        )
        .unwrap();
        assert_eq!(report.final_action, PreferenceReplayActionV1::RejectAll);
        assert_eq!(report.query_count, 1);
        assert_eq!(report.queries[0].response_kind, "simulated_reject_all");
        assert_eq!(report.unanswered_query_count, 0);
    }

    #[test]
    fn insufficient_oracle_does_not_fabricate_pairwise_feedback() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let scenario = scenario_by_id(&fixture, "HOLD-UNKNOWN-001");
        let report = run_interactive_episode(
            scenario,
            3,
            PreferenceQuerySelectionStrategyV1::SeededRandomUnaskedPair,
            20261018,
        )
        .unwrap();
        assert_eq!(report.final_action, PreferenceReplayActionV1::Abstain);
        assert_eq!(report.query_count, 3);
        assert_eq!(report.unanswered_query_count, 3);
        assert!(report.queries.iter().all(|query| {
            query.response_kind == "no_synthetic_answer_available"
        }));
    }

    #[test]
    fn zero_budget_never_emits_a_query() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let scenario = scenario_by_id(&fixture, "HOLD-UNKNOWN-001");
        let report = run_interactive_episode(
            scenario,
            0,
            PreferenceQuerySelectionStrategyV1::LexicographicUnaskedPair,
            1,
        )
        .unwrap();
        assert_eq!(report.final_action, PreferenceReplayActionV1::Abstain);
        assert_eq!(report.query_count, 0);
        assert!(report.queries.is_empty());
    }

    #[test]
    fn generated_factorized_suite_runs_through_query_budget_frontier() {
        use crate::preference_drift_generator::{
            generate_preference_drift_suite, PreferenceDriftGeneratorConfigV1,
        };

        let config = PreferenceDriftGeneratorConfigV1 {
            seed: 42,
            wearer_session_count: 10,
            development_wearer_session_count: 2,
            held_out_repetitions_per_family: 1,
        };
        let suite = generate_preference_drift_suite(FIXTURE, &config)
            .expect("factorized suite should validate");
        let report = evaluate_preference_query_budget_frontier(
            &suite.fixture_json,
            2,
            config.seed,
        )
        .expect("generated suite should evaluate in the query frontier");

        assert_eq!(report.scenario_count, 84);
        assert_eq!(report.lexicographic_unasked_pair.len(), 3);
        assert_eq!(report.seeded_random_unasked_pair.len(), 3);
        assert_eq!(report.coverage_deficit_unasked_pair.len(), 3);
        assert_eq!(
            report.lexicographic_unasked_pair[2].family_scores.len(),
            11
        );
        for frontier in [
            &report.lexicographic_unasked_pair,
            &report.seeded_random_unasked_pair,
            &report.coverage_deficit_unasked_pair,
        ] {
            assert_eq!(
                frontier[2].family_scores.values()
                    .map(|score| score.scenario_count)
                    .sum::<usize>(),
                84
            );
        }
    }

    #[test]
    fn query_budget_limit_is_validated() {
        assert!(evaluate_preference_query_budget_frontier(FIXTURE, 9, 1).is_err());
    }
}
