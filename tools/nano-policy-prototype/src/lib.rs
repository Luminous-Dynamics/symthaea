#![forbid(unsafe_code)]

//! Deterministic, pre-query candidate selection for research benchmarking.
//!
//! This is a policy prototype, not an optical/materials solver. It must never
//! receive hidden oracle labels or post-query observations.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const SCHEMA_VERSION: &str = "nano-policy-input-v1";
const SCORE_EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    Random,
    GoalOnly,
    UncertaintyOnly,
    DiversityOnly,
    /// Alternate between low and high pre-query predicted error to improve response-space coverage.
    BalancedExtremes,
    Hybrid,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureSpace {
    /// A pinned identifier for descriptor/embedding semantics and revision.
    pub id: String,
    /// Human-readable description of what each dimension means.
    pub semantics: String,
    /// True only if features were transformed with a frozen, recorded mapping.
    pub normalized_to_minus_one_one: bool,
    /// SHA-256 of the frozen feature-preprocessing specification/code.
    pub transform_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Weights {
    pub goal_fit: f64,
    pub uncertainty: f64,
    pub cost_efficiency: f64,
    pub diversity: f64,
}

impl Weights {
    fn validate(&self) -> Result<(), String> {
        let values = [
            self.goal_fit,
            self.uncertainty,
            self.cost_efficiency,
            self.diversity,
        ];
        if values.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err("weights must be finite and non-negative".to_string());
        }
        let total: f64 = values.iter().sum();
        if (total - 1.0).abs() > 1.0e-9 {
            return Err("hybrid weights must sum to 1 within 1e-9".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub id: String,
    /// Frozen normalized features in [-1, 1]; same feature space for every row.
    pub features: Vec<f64>,
    /// Lower is better; must be a pre-query model estimate, never a true oracle label.
    pub predicted_goal_error: Option<f64>,
    /// Non-negative pre-query uncertainty estimate. Not automatically calibrated.
    pub uncertainty: Option<f64>,
    /// Positive pre-query cost estimate in the declared cost unit.
    pub estimated_cost: Option<f64>,
    pub eligible: bool,
    pub already_queried: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EstimateProvenance {
    pub predicted_goal_error_model_id: Option<String>,
    pub predicted_goal_error_model_sha256: Option<String>,
    pub uncertainty_method_id: Option<String>,
    pub uncertainty_method_sha256: Option<String>,
    pub estimated_cost_model_id: Option<String>,
    pub estimated_cost_model_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: String,
    pub request_id: String,
    /// Caller-supplied immutable code revision for the policy implementation.
    pub policy_revision: String,
    /// Digest of the frozen source candidate-pool manifest; independently recompute it in the harness.
    pub candidate_pool_sha256: String,
    pub policy: Policy,
    pub seed: u64,
    pub batch_size: usize,
    pub feature_space: FeatureSpace,
    pub weights: Weights,
    /// Immutable identity of the models/methods used for pre-query estimates.
    pub estimate_provenance: EstimateProvenance,
    /// Features of previously evaluated/reference candidates in the same frozen space.
    pub anchors: Vec<Vec<f64>>,
    pub candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ComponentScores {
    pub goal_fit: f64,
    pub uncertainty: f64,
    pub cost_efficiency: f64,
    pub diversity: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SelectedCandidate {
    pub rank: usize,
    pub candidate_id: String,
    pub score: f64,
    pub component_scores: ComponentScores,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Response {
    pub schema_version: String,
    pub request_id: String,
    pub policy_revision: String,
    pub candidate_pool_sha256: String,
    pub policy: Policy,
    pub seed: u64,
    pub selected: Vec<SelectedCandidate>,
    pub candidate_pool_size: usize,
    pub eligible_pool_size: usize,
    pub information_boundary: String,
}

fn validate_request(request: &Request) -> Result<Vec<usize>, String> {
    if request.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported schema_version {:?}; expected {:?}",
            request.schema_version, SCHEMA_VERSION
        ));
    }
    if request.request_id.trim().is_empty() {
        return Err("request_id must not be empty".to_string());
    }
    if request.policy_revision.trim().is_empty() {
        return Err("policy_revision must not be empty".to_string());
    }
    validate_sha256(&request.candidate_pool_sha256, "candidate_pool_sha256")?;
    if request.batch_size == 0 {
        return Err("batch_size must be at least 1".to_string());
    }
    if request.feature_space.id.trim().is_empty()
        || request.feature_space.semantics.trim().is_empty()
    {
        return Err("feature-space id and semantics must be declared".to_string());
    }
    if !request.feature_space.normalized_to_minus_one_one {
        return Err("features must be normalized to [-1, 1] by a frozen preprocessing contract".to_string());
    }
    request.weights.validate()?;
    validate_sha256(&request.feature_space.transform_sha256, "feature_space.transform_sha256")?;

    let Some(first) = request.candidates.first() else {
        return Err("candidate pool is empty".to_string());
    };
    let feature_dim = first.features.len();
    if feature_dim == 0 {
        return Err("feature vectors must not be empty".to_string());
    }

    for (index, anchor) in request.anchors.iter().enumerate() {
        if anchor.len() != feature_dim {
            return Err(format!("anchor {index} has feature dimension {}, expected {feature_dim}", anchor.len()));
        }
        validate_features(anchor, &format!("anchor {index}"))?;
    }

    let mut ids = HashSet::with_capacity(request.candidates.len());
    let mut eligible_indices = Vec::new();

    for (index, candidate) in request.candidates.iter().enumerate() {
        if candidate.id.trim().is_empty() {
            return Err(format!("candidate {index} has an empty id"));
        }
        if !ids.insert(candidate.id.as_str()) {
            return Err(format!("duplicate candidate id {:?}", candidate.id));
        }
        if candidate.features.len() != feature_dim {
            return Err(format!(
                "candidate {:?} has feature dimension {}, expected {feature_dim}",
                candidate.id, candidate.features.len()
            ));
        }
        validate_features(&candidate.features, &format!("candidate {:?}", candidate.id))?;

        if let Some(value) = candidate.predicted_goal_error {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("candidate {:?} predicted_goal_error must be finite and >= 0", candidate.id));
            }
        }
        if let Some(value) = candidate.uncertainty {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("candidate {:?} uncertainty must be finite and >= 0", candidate.id));
            }
        }
        if let Some(value) = candidate.estimated_cost {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("candidate {:?} estimated_cost must be finite and > 0", candidate.id));
            }
        }

        if candidate.eligible && !candidate.already_queried {
            eligible_indices.push(index);
        }
    }

    if eligible_indices.len() < request.batch_size {
        return Err(format!(
            "requested batch_size {} exceeds eligible candidate count {}",
            request.batch_size,
            eligible_indices.len()
        ));
    }

    let requires_goal = request.policy == Policy::GoalOnly
        || request.policy == Policy::BalancedExtremes
        || (request.policy == Policy::Hybrid && request.weights.goal_fit > 0.0);
    let requires_uncertainty = request.policy == Policy::UncertaintyOnly
        || (request.policy == Policy::Hybrid && request.weights.uncertainty > 0.0);
    let requires_cost = request.policy == Policy::Hybrid && request.weights.cost_efficiency > 0.0;

    validate_estimator_pair(
        &request.estimate_provenance.predicted_goal_error_model_id,
        &request.estimate_provenance.predicted_goal_error_model_sha256,
        "predicted_goal_error_model",
        requires_goal,
    )?;
    validate_estimator_pair(
        &request.estimate_provenance.uncertainty_method_id,
        &request.estimate_provenance.uncertainty_method_sha256,
        "uncertainty_method",
        requires_uncertainty,
    )?;
    validate_estimator_pair(
        &request.estimate_provenance.estimated_cost_model_id,
        &request.estimate_provenance.estimated_cost_model_sha256,
        "estimated_cost_model",
        requires_cost,
    )?;

    for index in &eligible_indices {
        let candidate = &request.candidates[*index];
        if requires_goal && candidate.predicted_goal_error.is_none() {
            return Err(format!("candidate {:?} lacks required pre-query predicted_goal_error", candidate.id));
        }
        if requires_uncertainty && candidate.uncertainty.is_none() {
            return Err(format!("candidate {:?} lacks required pre-query uncertainty", candidate.id));
        }
        if requires_cost && candidate.estimated_cost.is_none() {
            return Err(format!("candidate {:?} lacks required pre-query estimated_cost", candidate.id));
        }
    }

    Ok(eligible_indices)
}

fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    if bytes.len() != 64
        || !bytes.iter().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err(format!("{label} must be a lowercase 64-character SHA-256 hex digest"));
    }
    Ok(())
}

fn validate_estimator_pair(
    id: &Option<String>,
    digest: &Option<String>,
    label: &str,
    required: bool,
) -> Result<(), String> {
    match (id, digest) {
        (Some(id), Some(digest)) if !id.trim().is_empty() => {
            validate_sha256(digest, &format!("{label}.sha256"))
        }
        (None, None) if !required => Ok(()),
        (None, None) => Err(format!("{label} identity is required for this policy")),
        _ => Err(format!("{label} requires both a non-empty id and a SHA-256 digest")),
    }
}

fn validate_features(features: &[f64], label: &str) -> Result<(), String> {
    for (dimension, value) in features.iter().enumerate() {
        if !value.is_finite() || *value < -1.0 || *value > 1.0 {
            return Err(format!(
                "{label} feature[{dimension}] must be finite and within [-1, 1]"
            ));
        }
    }
    Ok(())
}

fn normalize(values: &[f64], higher_is_better: bool) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range <= SCORE_EPSILON {
        return vec![0.5; values.len()];
    }
    values
        .iter()
        .map(|value| {
            let low_to_high = (value - min) / range;
            if higher_is_better {
                low_to_high
            } else {
                1.0 - low_to_high
            }
        })
        .collect()
}

fn normalize_component(
    request: &Request,
    eligible_indices: &[usize],
    get_value: impl Fn(&Candidate) -> Option<f64>,
    higher_is_better: bool,
) -> Vec<f64> {
    let raw: Vec<Option<f64>> = eligible_indices
        .iter()
        .map(|index| get_value(&request.candidates[*index]))
        .collect();

    let present_values: Vec<f64> = raw.iter().flatten().copied().collect();
    if present_values.is_empty() {
        return vec![0.5; eligible_indices.len()];
    }

    let normalized_present = normalize(&present_values, higher_is_better);
    let mut iterator = normalized_present.into_iter();
    raw.iter()
        .map(|value| {
            if value.is_some() {
                iterator.next().unwrap_or(0.5)
            } else {
                0.5
            }
        })
        .collect()
}

fn normalized_distance(left: &[f64], right: &[f64]) -> f64 {
    let sum_squares: f64 = left
        .iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum();
    (sum_squares.sqrt() / (2.0 * (left.len() as f64).sqrt())).clamp(0.0, 1.0)
}

fn diversity_score(
    candidate: &[f64],
    anchors: &[Vec<f64>],
    selected_features: &[Vec<f64>],
) -> f64 {
    let mut min_distance = 1.0_f64;
    let mut seen_reference = false;

    for reference in anchors.iter().chain(selected_features.iter()) {
        seen_reference = true;
        min_distance = min_distance.min(normalized_distance(candidate, reference));
    }

    if seen_reference {
        min_distance
    } else {
        1.0
    }
}

// Stable SplitMix64 finalizer. Used only to randomize baseline order; not cryptographic.
fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn random_priority(seed: u64, candidate_id: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64 ^ seed;
    for byte in candidate_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    mix64(hash)
}

fn policy_reason(policy: Policy) -> String {
    match policy {
        Policy::Random => "deterministic seeded-random baseline; not a quality claim".to_string(),
        Policy::GoalOnly => "lowest pre-query predicted goal error; not an observed oracle result".to_string(),
        Policy::UncertaintyOnly => "highest pre-query uncertainty estimate; not a calibrated probability unless separately validated".to_string(),
        Policy::DiversityOnly => "greedy maximin distance in the declared normalized feature space".to_string(),
        Policy::BalancedExtremes => "alternates between low- and high-predicted-error regions for model coverage; not a best-candidate-only objective".to_string(),
        Policy::Hybrid => "weighted pre-query goal fit, uncertainty, cost efficiency and maximin diversity".to_string(),
    }
}

pub fn select(request: &Request) -> Result<Response, String> {
    let eligible_indices = validate_request(request)?;
    let count = request.candidates.len();

    let goal_values = normalize_component(
        request,
        &eligible_indices,
        |candidate| candidate.predicted_goal_error,
        false,
    );
    let uncertainty_values = normalize_component(
        request,
        &eligible_indices,
        |candidate| candidate.uncertainty,
        true,
    );
    let cost_values = normalize_component(
        request,
        &eligible_indices,
        |candidate| candidate.estimated_cost,
        false,
    );

    let mut goal_score = vec![0.5; count];
    let mut uncertainty_score = vec![0.5; count];
    let mut cost_score = vec![0.5; count];

    for (position, index) in eligible_indices.iter().enumerate() {
        goal_score[*index] = goal_values[position];
        uncertainty_score[*index] = uncertainty_values[position];
        cost_score[*index] = cost_values[position];
    }

    let eligible_pool_size = eligible_indices.len();
    let mut remaining = eligible_indices;
    let mut selected_indices: Vec<usize> = Vec::with_capacity(request.batch_size);
    let mut selected = Vec::with_capacity(request.batch_size);
    let reason = policy_reason(request.policy);

    for rank in 1..=request.batch_size {
        let selected_features: Vec<Vec<f64>> = selected_indices
            .iter()
            .map(|index| request.candidates[*index].features.clone())
            .collect();

        let mut scored: Vec<(usize, f64, f64)> = remaining
            .iter()
            .map(|index| {
                let candidate = &request.candidates[*index];
                let diversity = diversity_score(
                    &candidate.features,
                    &request.anchors,
                    &selected_features,
                );
                let score = match request.policy {
                    Policy::Random => random_priority(request.seed, &candidate.id) as f64
                        / u64::MAX as f64,
                    Policy::GoalOnly => goal_score[*index],
                    Policy::UncertaintyOnly => uncertainty_score[*index],
                    Policy::DiversityOnly => diversity,
                    Policy::BalancedExtremes => {
                        if rank % 2 == 1 {
                            goal_score[*index]
                        } else {
                            1.0 - goal_score[*index]
                        }
                    }
                    Policy::Hybrid => {
                        request.weights.goal_fit * goal_score[*index]
                            + request.weights.uncertainty * uncertainty_score[*index]
                            + request.weights.cost_efficiency * cost_score[*index]
                            + request.weights.diversity * diversity
                    }
                };
                (*index, score, diversity)
            })
            .collect();

        scored.sort_by(|left, right| {
            let score_order = right.1.total_cmp(&left.1);
            let coverage_tie_break = if request.policy == Policy::BalancedExtremes {
                right.2.total_cmp(&left.2)
            } else {
                std::cmp::Ordering::Equal
            };
            score_order
                .then(coverage_tie_break)
                .then_with(|| request.candidates[left.0].id.cmp(&request.candidates[right.0].id))
        });

        let (winner_index, score, diversity) = scored
            .first()
            .copied()
            .ok_or_else(|| "no remaining eligible candidate".to_string())?;
        let winner = &request.candidates[winner_index];

        let component_scores = ComponentScores {
            goal_fit: goal_score[winner_index],
            uncertainty: uncertainty_score[winner_index],
            cost_efficiency: cost_score[winner_index],
            diversity,
        };
        selected.push(SelectedCandidate {
            rank,
            candidate_id: winner.id.clone(),
            score,
            component_scores,
            reason: reason.clone(),
        });

        selected_indices.push(winner_index);
        remaining.retain(|index| *index != winner_index);
    }

    Ok(Response {
        schema_version: "nano-policy-output-v1".to_string(),
        request_id: request.request_id.clone(),
        policy_revision: request.policy_revision.clone(),
        candidate_pool_sha256: request.candidate_pool_sha256.clone(),
        policy: request.policy,
        seed: request.seed,
        selected,
        candidate_pool_size: request.candidates.len(),
        eligible_pool_size,
        information_boundary: "Selection used only candidate ids, declared normalized pre-query features, explicitly supplied pre-query estimates, eligibility flags, anchors, policy configuration and seed. No oracle labels or post-query observations are accepted by the schema.".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn candidate(
        id: &str,
        features: Vec<f64>,
        error: Option<f64>,
        uncertainty: Option<f64>,
        cost: Option<f64>,
    ) -> Candidate {
        Candidate {
            id: id.to_string(),
            features,
            predicted_goal_error: error,
            uncertainty,
            estimated_cost: cost,
            eligible: true,
            already_queried: false,
        }
    }

    fn request(policy: Policy) -> Request {
        Request {
            schema_version: SCHEMA_VERSION.to_string(),
            request_id: "test-1".to_string(),
            policy_revision: "test-policy-revision-v1".to_string(),
            candidate_pool_sha256: "a".repeat(64),
            policy,
            seed: 42,
            batch_size: 2,
            feature_space: FeatureSpace {
                id: "test-embedding-v1".to_string(),
                semantics: "two normalized toy features".to_string(),
                normalized_to_minus_one_one: true,
                transform_sha256: "f88f161ae64d0edcf2a65d9cb4397593435304ca7388bc3e688bce952d4deda6".to_string(),
            },
            weights: Weights {
                goal_fit: 0.30,
                uncertainty: 0.20,
                cost_efficiency: 0.10,
                diversity: 0.40,
            },
            estimate_provenance: EstimateProvenance {
                predicted_goal_error_model_id: Some("toy-prequery-goal-model-v1".to_string()),
                predicted_goal_error_model_sha256: Some("106e028be1f1ba742cff35f33db40ce3fa27db974d2d1ef4e2661e7fa418699a".to_string()),
                uncertainty_method_id: Some("toy-uncertainty-estimator-v1".to_string()),
                uncertainty_method_sha256: Some("3ade88dde4e76fd4bd910d9a088b6e9584589bf2c46512bc6ef0841f2667795a".to_string()),
                estimated_cost_model_id: Some("toy-cost-estimator-v1".to_string()),
                estimated_cost_model_sha256: Some("6206e804b6f1d23644dd89d832852f7403c150295ef0a4d6266359fb268991a8".to_string()),
            },
            anchors: vec![],
            candidates: vec![
                candidate("a", vec![0.0, 0.0], Some(0.1), Some(0.2), Some(1.0)),
                candidate("b", vec![0.02, 0.02], Some(0.2), Some(0.4), Some(1.1)),
                candidate("c", vec![0.9, 0.9], Some(0.3), Some(0.8), Some(1.5)),
            ],
        }
    }

    #[test]
    fn diversity_policy_spreads_a_batch_and_is_order_invariant() {
        let mut input = request(Policy::DiversityOnly);
        input.batch_size = 2;
        let output = select(&input).expect("valid request");
        assert_eq!(output.selected.len(), 2);
        assert_eq!(output.selected[0].candidate_id, "a");
        assert_eq!(output.selected[1].candidate_id, "c");
        assert!(output.selected.iter().all(|item| item.score.is_finite()));

        let expected_ids: Vec<_> = output.selected.iter().map(|item| item.candidate_id.clone()).collect();
        input.candidates.reverse();
        let reordered = select(&input).expect("valid reordered request");
        let reordered_ids: Vec<_> = reordered.selected.iter().map(|item| item.candidate_id.clone()).collect();
        assert_eq!(expected_ids, reordered_ids);
    }

    #[test]
    fn random_policy_is_deterministic_and_order_invariant() {
        let input = request(Policy::Random);
        let first = select(&input).expect("valid request");
        let mut reordered = input.clone();
        reordered.candidates.reverse();
        let second = select(&reordered).expect("valid reordered request");
        let first_ids: Vec<_> = first.selected.iter().map(|item| item.candidate_id.clone()).collect();
        let second_ids: Vec<_> = second.selected.iter().map(|item| item.candidate_id.clone()).collect();
        assert_eq!(first_ids, second_ids);
    }

    #[test]
    fn already_queried_and_ineligible_candidates_are_never_selected() {
        let mut input = request(Policy::DiversityOnly);
        input.candidates[0].already_queried = true;
        input.candidates[2].eligible = false;
        input.batch_size = 1;
        let output = select(&input).expect("one candidate remains eligible");
        assert_eq!(output.selected[0].candidate_id, "b");
    }

    #[test]
    fn missing_required_pre_query_fields_fail_closed() {
        let mut input = request(Policy::GoalOnly);
        input.candidates[1].predicted_goal_error = None;
        assert!(select(&input).is_err());
    }

    #[test]
    fn duplicate_candidate_ids_fail_closed() {
        let mut input = request(Policy::DiversityOnly);
        input.candidates[1].id = "a".to_string();
        assert!(select(&input).unwrap_err().contains("duplicate candidate id"));
    }

    #[test]
    fn over_budget_batch_fails_closed() {
        let mut input = request(Policy::DiversityOnly);
        input.batch_size = 4;
        assert!(select(&input).unwrap_err().contains("exceeds eligible candidate count"));
    }

    #[test]
    fn out_of_range_or_non_finite_features_fail_closed() {
        let mut input = request(Policy::DiversityOnly);
        input.candidates[0].features[0] = 1.01;
        assert!(select(&input).is_err());

        input.candidates[0].features[0] = f64::NAN;
        assert!(select(&input).is_err());
    }

    #[test]
    fn undeclared_information_fields_are_rejected_by_json_deserialization() {
        let raw = json!({
            "schema_version": "nano-policy-input-v1",
            "request_id": "test",
            "policy_revision": "test-policy-revision-v1",
            "candidate_pool_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "policy": "random",
            "seed": 1,
            "batch_size": 1,
            "feature_space": {
                "id": "x",
                "semantics": "x",
                "normalized_to_minus_one_one": true,
                "transform_sha256": "f88f161ae64d0edcf2a65d9cb4397593435304ca7388bc3e688bce952d4deda6"
            },
            "weights": {
                "goal_fit": 0.3,
                "uncertainty": 0.2,
                "cost_efficiency": 0.1,
                "diversity": 0.4
            },
            "estimate_provenance": {
                "predicted_goal_error_model_id": null,
                "predicted_goal_error_model_sha256": null,
                "uncertainty_method_id": null,
                "uncertainty_method_sha256": null,
                "estimated_cost_model_id": null,
                "estimated_cost_model_sha256": null
            },
            "anchors": [],
            "candidates": [{
                "id": "x",
                "features": [0.0],
                "predicted_goal_error": null,
                "uncertainty": null,
                "estimated_cost": null,
                "eligible": true,
                "already_queried": false,
                "hidden_oracle_label": 0.0
            }]
        });
        assert!(serde_json::from_value::<Request>(raw).is_err());
    }

    #[test]
    fn hybrid_requires_immutable_estimator_provenance() {
        let mut input = request(Policy::Hybrid);
        input.estimate_provenance.predicted_goal_error_model_sha256 = None;
        assert!(select(&input).unwrap_err().contains("requires both a non-empty id and a SHA-256 digest"));
    }

    #[test]
    fn invalid_feature_or_pool_digests_fail_closed() {
        let mut input = request(Policy::DiversityOnly);
        input.candidate_pool_sha256 = "not-a-digest".to_string();
        assert!(select(&input).unwrap_err().contains("candidate_pool_sha256"));

        let mut input = request(Policy::DiversityOnly);
        input.feature_space.transform_sha256 = "G".repeat(64);
        assert!(select(&input).unwrap_err().contains("feature_space.transform_sha256"));
    }

    #[test]
    fn balanced_extremes_samples_both_low_and_high_predicted_error() {
        let input = request(Policy::BalancedExtremes);
        let output = select(&input).expect("valid request");
        assert_eq!(output.selected[0].candidate_id, "a");
        assert_eq!(output.selected[1].candidate_id, "c");
        assert!(output.selected.iter().all(|item| item.score.is_finite()));
    }

    #[test]
    fn hybrid_scores_are_finite_and_preserve_seed_and_request_identity() {
        let input = request(Policy::Hybrid);
        let output = select(&input).expect("valid request");
        assert_eq!(output.request_id, "test-1");
        assert_eq!(output.policy_revision, "test-policy-revision-v1");
        assert_eq!(output.candidate_pool_sha256, "a".repeat(64));
        assert_eq!(output.seed, 42);
        assert_eq!(output.policy, Policy::Hybrid);
        assert_eq!(output.selected[0].candidate_id, "a");
        assert_eq!(output.selected[1].candidate_id, "c");
        assert!(output.selected.iter().all(|item| item.score.is_finite()));
    }
}
