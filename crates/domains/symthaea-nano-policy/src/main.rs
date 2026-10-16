//! Research-only selector: deterministic diversity and random-control policies.
//!
//! Deliberately accepts candidate IDs and unlabeled numeric descriptors only.
//! It does not implement nanophysics or predict a material property.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};

const SCHEMA_VERSION: &str = "nano-selection-input-v1";
const POLICY_VERSION: &str = "nano-selector-v1";
const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CANDIDATES: usize = 100_000;
const MAX_FEATURES: usize = 4_096;
const MAX_FEATURE_CELLS: usize = 10_000_000;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema_version: String,
    policy: Policy,
    budget: usize,
    seed: u64,
    #[serde(default)]
    already_selected: Vec<String>,
    candidates: Vec<Candidate>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Policy {
    Maximin,
    Random,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    id: String,
    features: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct Response {
    schema_version: &'static str,
    policy_version: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_blake3: Option<String>,
    policy: Policy,
    candidate_count: usize,
    feature_count: usize,
    budget_requested: usize,
    already_selected: Vec<String>,
    normalization: &'static str,
    selected: Vec<SelectedCandidate>,
}

#[derive(Debug, Serialize)]
struct SelectedCandidate {
    id: String,
    rank: usize,
    score_kind: &'static str,
    /// Squared Euclidean distance in min-max-normalized descriptor space.
    /// Absent for the random-control policy.
    score: Option<f64>,
}

fn validate_and_prepare(
    request: &Request,
) -> Result<(Vec<Candidate>, Vec<Vec<f64>>, usize, Vec<usize>, Vec<String>), String> {
    if request.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported schema_version {:?}; expected {:?}",
            request.schema_version, SCHEMA_VERSION
        ));
    }
    if request.candidates.is_empty() {
        return Err("candidate pool must not be empty".to_owned());
    }
    if request.candidates.len() > MAX_CANDIDATES {
        return Err(format!("candidate pool exceeds limit of {MAX_CANDIDATES}"));
    }
    if request.budget == 0 {
        return Err("budget must be at least 1".to_owned());
    }

    let mut candidates = request.candidates.clone();
    for candidate in &candidates {
        if candidate.id.trim().is_empty() {
            return Err("candidate id must not be empty".to_owned());
        }
        if candidate.features.is_empty() {
            return Err(format!("candidate {:?} has no features", candidate.id));
        }
        if candidate.features.iter().any(|value| !value.is_finite()) {
            return Err(format!("candidate {:?} has non-finite features", candidate.id));
        }
    }
    candidates.sort_by(|left, right| left.id.cmp(&right.id));

    for pair in candidates.windows(2) {
        if pair[0].id == pair[1].id {
            return Err(format!("duplicate candidate id {:?}", pair[0].id));
        }
    }

    let feature_count = candidates[0].features.len();
    if feature_count > MAX_FEATURES {
        return Err(format!("feature count exceeds limit of {MAX_FEATURES}"));
    }
    if candidates.iter().any(|candidate| candidate.features.len() != feature_count) {
        return Err("all candidates must have the same feature dimension".to_owned());
    }
    if candidates
        .len()
        .checked_mul(feature_count)
        .is_none_or(|cells| cells > MAX_FEATURE_CELLS)
    {
        return Err(format!(
            "candidate_count * feature_count exceeds limit of {MAX_FEATURE_CELLS}"
        ));
    }

    let mut prior_ids = request.already_selected.clone();
    prior_ids.sort();
    if prior_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("already_selected contains duplicate ids".to_owned());
    }

    let id_to_index: HashMap<&str, usize> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.id.as_str(), index))
        .collect();
    let mut prior_indices = Vec::with_capacity(prior_ids.len());
    for id in &prior_ids {
        let index = id_to_index
            .get(id.as_str())
            .copied()
            .ok_or_else(|| format!("already_selected id {id:?} is absent from candidate pool"))?;
        prior_indices.push(index);
    }

    let remaining = candidates.len().saturating_sub(prior_indices.len());
    if request.budget > remaining {
        return Err(format!(
            "budget {} exceeds {} unselected candidates",
            request.budget, remaining
        ));
    }

    let normalized = match request.policy {
        Policy::Maximin => normalize_features(&candidates)?,
        // The random-control arm must not pay the cost of descriptor normalization.
        Policy::Random => Vec::new(),
    };
    Ok((candidates, normalized, feature_count, prior_indices, prior_ids))
}

fn normalize_features(candidates: &[Candidate]) -> Result<Vec<Vec<f64>>, String> {
    let dimensions = candidates[0].features.len();
    let mut mins = vec![f64::INFINITY; dimensions];
    let mut maxs = vec![f64::NEG_INFINITY; dimensions];

    for candidate in candidates {
        for (dimension, value) in candidate.features.iter().enumerate() {
            mins[dimension] = mins[dimension].min(*value);
            maxs[dimension] = maxs[dimension].max(*value);
        }
    }

    let mut ranges = Vec::with_capacity(dimensions);
    for (dimension, (min, max)) in mins.iter().zip(&maxs).enumerate() {
        let range = max - min;
        if !range.is_finite() {
            return Err(format!(
                "feature dimension {dimension} has a non-finite numeric span"
            ));
        }
        ranges.push(range);
    }

    Ok(candidates
        .iter()
        .map(|candidate| {
            candidate
                .features
                .iter()
                .enumerate()
                .map(|(dimension, value)| {
                    if ranges[dimension] == 0.0 {
                        0.0
                    } else {
                        (value - mins[dimension]) / ranges[dimension]
                    }
                })
                .collect()
        })
        .collect())
}

fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let difference = a - b;
            difference * difference
        })
        .sum()
}

fn centroid(features: &[Vec<f64>]) -> Vec<f64> {
    let dimensions = features[0].len();
    let mut center = vec![0.0; dimensions];
    for row in features {
        for (index, value) in row.iter().enumerate() {
            center[index] += value;
        }
    }
    let count = features.len() as f64;
    for value in &mut center {
        *value /= count;
    }
    center
}

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Unbiased bounded integer using multiply-high plus rejection.
    fn index_below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        let bound = bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = (self.next_u64() as u128) * (bound as u128);
            let low = product as u64;
            if low >= threshold {
                return (product >> 64) as usize;
            }
        }
    }
}

fn choose(
    request: &Request,
    candidates: &[Candidate],
    features: &[Vec<f64>],
    feature_count: usize,
    prior_indices: &[usize],
    prior_ids: Vec<String>,
) -> Result<Response, String> {
    let mut selected_indices = prior_indices.to_vec();
    let mut remaining: Vec<usize> = (0..candidates.len())
        .filter(|index| !selected_indices.contains(index))
        .collect();
    let mut selected = Vec::with_capacity(request.budget);

    match request.policy {
        Policy::Random => {
            // The pool is sorted by opaque ID during validation, so a fixed seed
            // yields the same permutation regardless of input row order.
            let mut rng = SplitMix64::new(request.seed);
            for index in (1..remaining.len()).rev() {
                let other = rng.index_below(index + 1);
                remaining.swap(index, other);
            }
            for (rank, index) in remaining.into_iter().take(request.budget).enumerate() {
                selected.push(SelectedCandidate {
                    id: candidates[index].id.clone(),
                    rank: rank + 1,
                    score_kind: "random_order",
                    score: None,
                });
            }
        }
        Policy::Maximin => {
            // If no history exists, choose a deterministic first point farthest
            // from the pool centroid. This avoids consuming target labels as a seed.
            if selected_indices.is_empty() {
                let center = centroid(features);
                let mut best: Option<(usize, f64)> = None;
                for &index in &remaining {
                    let score = squared_distance(&features[index], &center);
                    if best.is_none_or(|(_, best_score)| score > best_score) {
                        best = Some((index, score));
                    }
                }
                let (index, score) =
                    best.ok_or_else(|| "no eligible candidate remains".to_owned())?;
                selected.push(SelectedCandidate {
                    id: candidates[index].id.clone(),
                    rank: 1,
                    score_kind: "distance_to_pool_centroid",
                    score: Some(score),
                });
                selected_indices.push(index);
                remaining.retain(|candidate| *candidate != index);
            }

            if selected.len() < request.budget {
                // Compute each candidate's distance to the nearest selected point
                // once, then update it only against each newly chosen point. This
                // is O(pool * features * (history + budget)), not O(pool * features * budget^2).
                let mut minimum_distances: Vec<f64> = features
                    .iter()
                    .map(|feature| {
                        selected_indices
                            .iter()
                            .map(|chosen| squared_distance(feature, &features[*chosen]))
                            .reduce(f64::min)
                            .unwrap_or(f64::INFINITY)
                    })
                    .collect();

                while selected.len() < request.budget {
                    let mut best: Option<(usize, f64)> = None;
                    for &index in &remaining {
                        let score = minimum_distances[index];
                        // Strict greater-than plus lexical ordering fixes tie behavior.
                        if best.is_none_or(|(_, best_score)| score > best_score) {
                            best = Some((index, score));
                        }
                    }

                    let (index, score) =
                        best.ok_or_else(|| "no eligible candidate remains".to_owned())?;
                    selected.push(SelectedCandidate {
                        id: candidates[index].id.clone(),
                        rank: selected.len() + 1,
                        score_kind: "min_distance_to_selected",
                        score: Some(score),
                    });
                    selected_indices.push(index);
                    remaining.retain(|candidate| *candidate != index);

                    if selected.len() < request.budget {
                        for (candidate_index, feature) in features.iter().enumerate() {
                            let distance = squared_distance(feature, &features[index]);
                            minimum_distances[candidate_index] =
                                minimum_distances[candidate_index].min(distance);
                        }
                    }
                }
            }
        }
    }

    Ok(Response {
        schema_version: SCHEMA_VERSION,
        policy_version: POLICY_VERSION,
        input_blake3: None,
        policy: request.policy,
        candidate_count: candidates.len(),
        feature_count,
        budget_requested: request.budget,
        already_selected: prior_ids,
        normalization: match request.policy {
            Policy::Maximin => "per_feature_min_max_over_candidate_pool",
            Policy::Random => "not_used_by_random_policy",
        },
        selected,
    })
}

fn select(request: &Request) -> Result<Response, String> {
    let (candidates, features, feature_count, prior_indices, prior_ids) =
        validate_and_prepare(request)?;
    choose(
        request,
        &candidates,
        &features,
        feature_count,
        &prior_indices,
        prior_ids,
    )
}

fn select_with_request_digest(request: &Request, raw_request: &[u8]) -> Result<Response, String> {
    let mut response = select(request)?;
    response.input_blake3 = Some(blake3::hash(raw_request).to_hex().to_string());
    Ok(response)
}

fn execute() -> Result<(), String> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read stdin: {error}"))?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        return Err(format!("input exceeds {MAX_INPUT_BYTES} byte limit"));
    }

    let request: Request = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid request JSON: {error}"))?;
    let response = select_with_request_digest(&request, &bytes)?;
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, &response)
        .map_err(|error| format!("failed to encode response: {error}"))?;
    output
        .write_all(b"\n")
        .map_err(|error| format!("failed to write response: {error}"))?;
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("nano-policy: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, features: &[f64]) -> Candidate {
        Candidate {
            id: id.to_owned(),
            features: features.to_vec(),
        }
    }

    fn request(policy: Policy, budget: usize, candidates: Vec<Candidate>) -> Request {
        Request {
            schema_version: SCHEMA_VERSION.to_owned(),
            policy,
            budget,
            seed: 42,
            already_selected: Vec::new(),
            candidates,
        }
    }

    fn selected_ids(response: &Response) -> Vec<String> {
        response.selected.iter().map(|item| item.id.clone()).collect()
    }

    #[test]
    fn maximin_selects_descriptor_extremes_deterministically() {
        let req = request(
            Policy::Maximin,
            2,
            vec![
                candidate("a", &[0.0]),
                candidate("b", &[0.1]),
                candidate("c", &[0.9]),
                candidate("d", &[1.0]),
            ],
        );
        let result = select(&req).expect("valid request");
        assert_eq!(selected_ids(&result), vec!["a", "d"]);
        assert_eq!(result.selected[0].score_kind, "distance_to_pool_centroid");
        assert_eq!(result.selected[1].score_kind, "min_distance_to_selected");
    }

    #[test]
    fn maximin_updates_nearest_distances_after_each_pick() {
        let req = request(
            Policy::Maximin,
            3,
            vec![
                candidate("a", &[0.0]),
                candidate("b", &[0.1]),
                candidate("c", &[0.5]),
                candidate("d", &[1.0]),
            ],
        );
        let result = select(&req).expect("valid request");
        assert_eq!(selected_ids(&result), vec!["d", "a", "c"]);
        assert_eq!(result.selected[2].score_kind, "min_distance_to_selected");
    }

    #[test]
    fn input_row_order_does_not_change_maximin_result() {
        let first = request(
            Policy::Maximin,
            3,
            vec![
                candidate("a", &[0.0, 0.0]),
                candidate("b", &[0.5, 0.2]),
                candidate("c", &[1.0, 1.0]),
            ],
        );
        let mut second = first.clone();
        second.candidates.reverse();
        assert_eq!(
            selected_ids(&select(&first).expect("first")),
            selected_ids(&select(&second).expect("second"))
        );
    }

    #[test]
    fn random_control_skips_unneeded_descriptor_normalization() {
        let req = request(
            Policy::Random,
            1,
            vec![candidate("low", &[f64::MIN]), candidate("high", &[f64::MAX])],
        );
        let result = select(&req).expect("random policy does not normalize descriptors");
        assert_eq!(result.normalization, "not_used_by_random_policy");
        assert_eq!(result.feature_count, 1);
    }

    #[test]
    fn random_baseline_is_seeded_and_input_order_independent() {
        let first = request(
            Policy::Random,
            3,
            vec![
                candidate("a", &[0.0]),
                candidate("b", &[0.25]),
                candidate("c", &[0.5]),
                candidate("d", &[0.75]),
                candidate("e", &[1.0]),
            ],
        );
        let mut second = first.clone();
        second.candidates.reverse();
        assert_eq!(
            selected_ids(&select(&first).expect("first")),
            selected_ids(&select(&second).expect("second"))
        );
    }

    #[test]
    fn prior_selection_is_excluded_and_used_for_diversity() {
        let mut req = request(
            Policy::Maximin,
            1,
            vec![candidate("a", &[0.0]), candidate("b", &[0.1]), candidate("c", &[1.0])],
        );
        req.already_selected = vec!["a".to_owned()];
        let result = select(&req).expect("valid request");
        assert_eq!(selected_ids(&result), vec!["c"]);
        assert_eq!(result.already_selected, vec!["a"]);
    }

    #[test]
    fn rejects_extra_candidate_fields_that_could_smuggle_labels() {
        let raw = r#"{
            "schema_version":"nano-selection-input-v1",
            "policy":"maximin",
            "budget":1,
            "seed":0,
            "already_selected":[],
            "candidates":[{"id":"x","features":[0.0],"target_energy_ev":-3.2}]
        }"#;
        assert!(serde_json::from_str::<Request>(raw).is_err());
    }

    #[test]
    fn rejects_duplicate_ids() {
        let req = request(Policy::Random, 1, vec![candidate("x", &[0.0]), candidate("x", &[1.0])]);
        assert!(select(&req).unwrap_err().contains("duplicate candidate id"));
    }

    #[test]
    fn rejects_mixed_feature_dimensions() {
        let req = request(
            Policy::Maximin,
            1,
            vec![candidate("a", &[0.0]), candidate("b", &[0.5, 0.6])],
        );
        assert!(select(&req).unwrap_err().contains("same feature dimension"));
    }

    #[test]
    fn rejects_invalid_prior_selection_and_over_budget() {
        let mut req = request(
            Policy::Maximin,
            1,
            vec![candidate("a", &[0.0]), candidate("b", &[1.0])],
        );
        req.already_selected = vec!["missing".to_owned()];
        assert!(select(&req).unwrap_err().contains("absent from candidate pool"));
        req.already_selected = vec!["a".to_owned(), "b".to_owned()];
        assert!(select(&req).unwrap_err().contains("budget"));
    }

    #[test]
    fn constant_feature_dimensions_are_safe() {
        let req = request(
            Policy::Maximin,
            2,
            vec![candidate("b", &[5.0, 1.0]), candidate("a", &[5.0, 1.0])],
        );
        let result = select(&req).expect("constant dimensions are valid");
        assert_eq!(selected_ids(&result), vec!["a", "b"]);
        assert!(result.selected.iter().all(|item| item.score == Some(0.0)));
    }

    #[test]
    fn rejects_duplicate_prior_ids() {
        let mut req = request(
            Policy::Maximin,
            1,
            vec![candidate("a", &[0.0]), candidate("b", &[1.0])],
        );
        req.already_selected = vec!["a".to_owned(), "a".to_owned()];
        assert!(select(&req).unwrap_err().contains("duplicate ids"));
    }

    #[test]
    fn rejects_non_finite_numeric_span() {
        let req = request(
            Policy::Maximin,
            1,
            vec![candidate("low", &[f64::MIN]), candidate("high", &[f64::MAX])],
        );
        assert!(select(&req).unwrap_err().contains("non-finite numeric span"));
    }

    #[test]
    fn cli_receipt_binds_selection_to_exact_request_bytes() {
        let raw = include_bytes!("../fixtures/smoke-request.json");
        let req: Request = serde_json::from_slice(raw).expect("valid smoke fixture");
        let response =
            select_with_request_digest(&req, raw).expect("valid selection with request digest");
        let expected = blake3::hash(raw).to_hex().to_string();
        assert_eq!(response.input_blake3.as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn pinned_smoke_fixture_produces_expected_selection() {
        let req: Request = serde_json::from_str(include_str!("../fixtures/smoke-request.json"))
            .expect("valid smoke request fixture");
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/smoke-expected.json"))
                .expect("valid smoke expectation fixture");
        let result = select(&req).expect("smoke request selects candidates");
        let actual = serde_json::to_value(selected_ids(&result)).expect("serialize selected IDs");
        assert_eq!(actual, expected["expected_selected_ids"]);
    }

    #[test]
    fn checked_in_schemas_are_json_and_fail_closed_at_object_boundaries() {
        let input_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/input.schema.json"))
                .expect("input schema parses as JSON");
        let output_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/output.schema.json"))
                .expect("output schema parses as JSON");
        assert_eq!(input_schema["additionalProperties"].as_bool(), Some(false));
        assert_eq!(
            input_schema["properties"]["candidates"]["items"]["additionalProperties"].as_bool(),
            Some(false)
        );
        assert_eq!(output_schema["additionalProperties"].as_bool(), Some(false));
        assert_eq!(
            output_schema["properties"]["policy_version"]["const"].as_str(),
            Some(POLICY_VERSION)
        );
        assert_eq!(
            output_schema["properties"]["input_blake3"]["pattern"].as_str(),
            Some("^[a-f0-9]{64}$")
        );
    }

    #[test]
    fn rejects_wrong_schema_version() {
        let mut req = request(Policy::Maximin, 1, vec![candidate("a", &[0.0])]);
        req.schema_version = "future-schema".to_owned();
        assert!(select(&req).unwrap_err().contains("unsupported schema_version"));
    }
}
