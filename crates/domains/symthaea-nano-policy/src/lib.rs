// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Research-only deterministic candidate-selection policies.
//!
//! This crate does not predict physical properties. It selects from a frozen
//! candidate pool using externally supplied normalized descriptors. Target
//! labels, oracle results, test labels and post-outcome fields are deliberately
//! not representable in the version-1 request structures.

#![deny(unsafe_code)]
#![warn(missing_docs)]

/// Structural verifier for selector request/response receipts.
/// Independent post-outcome benchmark metric computation.
pub mod metrics;
pub mod receipt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Current JSONL request/response contract version.
pub const SCHEMA_VERSION: &str = "nano-policy-request-v1";
/// Maximum accepted request-line size.
pub const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;
/// Maximum candidates in one request.
pub const MAX_CANDIDATES: usize = 10_000;
/// Maximum descriptor dimension per candidate.
pub const MAX_FEATURE_DIMENSIONS: usize = 256;
/// Maximum number of candidates to return in one query.
pub const MAX_BUDGET: usize = 1_000;
/// Maximum candidate/family identifier length in bytes.
pub const MAX_ID_BYTES: usize = 256;

/// Available deterministic policies. These are baselines, not a physical model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    /// Deterministic pseudo-random sampling without replacement.
    RandomSeeded,
    /// Greedy farthest-first selection over normalized descriptors.
    FarthestFirst,
    /// Farthest-first with a preference for underrepresented families.
    GroupBalancedFarthestFirst,
    /// Greedy descriptor-space diversity per unit of externally declared cost.
    CostAwareFarthestFirst,
}

/// Candidate inputs available to the selector.
///
/// Unknown JSON fields are rejected. Targets, predictions, test membership and
/// oracle feedback are not accepted by this version of the contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    /// Stable, unique candidate identifier.
    pub candidate_id: String,
    /// Optional positive estimated cost in one frozen cost unit (e.g. simulation seconds).
    ///
    /// Required by CostAwareFarthestFirst. Cost provenance and calibration are
    /// external responsibilities; this field is not predicted by this crate.
    #[serde(default)]
    pub estimated_cost: Option<f64>,
    /// Predeclared scientific family/group key used by the group-balanced policy.
    pub family_id: String,
    /// Frozen descriptor vector; values must be finite and lie in [0, 1].
    ///
    /// Scaling and feature selection must be fitted only on permitted training
    /// data and identified by the request's feature-contract digest.
    pub normalized_features: Vec<f64>,
}

/// One JSONL request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionRequest {
    /// Must equal SCHEMA_VERSION.
    pub schema_version: String,
    /// Selection algorithm.
    pub policy: Policy,
    /// Maximum number of candidates to return.
    pub budget: usize,
    /// Seed used by the deterministic random baseline and group-policy anchor.
    pub seed: u64,
    /// SHA-256 identity for the externally frozen feature/preprocessing contract.
    pub feature_contract_sha256: String,
    /// Optional unit for all declared candidate costs (e.g. "simulation_seconds").
    /// Required by CostAwareFarthestFirst and included in the request receipt.
    #[serde(default)]
    pub cost_unit: Option<String>,
    /// Complete frozen candidate pool.
    pub candidates: Vec<Candidate>,
}

/// One selected candidate and its policy diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionDecision {
    /// One-based selection order.
    pub rank: usize,
    /// Selected candidate identifier.
    pub candidate_id: String,
    /// Family/group used for provenance and diversity analysis.
    pub family_id: String,
    /// Declared cost carried through from the request, if supplied.
    pub estimated_cost: Option<f64>,
    /// Minimum Euclidean distance to an already-selected candidate in normalized
    /// descriptor space. None means no distance score was used.
    pub min_distance: Option<f64>,
    /// Policy explanation, not scientific evidence.
    pub reason: String,
}

/// Successful response and compact audit receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionResponse {
    /// Current response schema.
    pub schema_version: String,
    /// Always completed for a successful response.
    pub status: String,
    /// Selection policy.
    pub policy: Policy,
    /// Seed used for reproducible policy replay.
    pub seed: u64,
    /// SHA-256 of exact request-line bytes, excluding line terminator.
    pub request_sha256: String,
    /// Identity supplied for the frozen feature/preprocessing contract.
    pub feature_contract_sha256: String,
    /// Cost unit, if the request declared one.
    pub cost_unit: Option<String>,
    /// Candidate count after validating the input pool.
    pub candidate_count: usize,
    /// Requested maximum selection budget.
    pub requested_budget: usize,
    /// Number actually selected.
    pub selected_count: usize,
    /// Selection trace in exact output order.
    pub selected: Vec<SelectionDecision>,
    /// Limitations attached to every response.
    pub limitations: Vec<String>,
}

/// Rejected request; this is not a scientific experiment outcome.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ErrorResponse {
    /// Current response schema.
    pub schema_version: String,
    /// Always rejected_input for this response type.
    pub status: String,
    /// SHA-256 of exact request-line bytes.
    pub request_sha256: String,
    /// Stable machine-readable error category.
    pub error_code: String,
    /// Bounded diagnostic text.
    pub message: String,
}

/// Request validation or policy execution error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    /// Invalid JSON or forbidden unknown fields.
    InvalidJson(String),
    /// Unsupported schema version.
    UnsupportedSchemaVersion,
    /// Invalid feature-contract digest.
    InvalidFeatureContractDigest,
    /// Zero or excessively large budget.
    InvalidBudget,
    /// Empty or excessively large candidate pool.
    InvalidCandidateCount,
    /// Candidate/family ID is invalid or candidate ID is duplicated.
    InvalidCandidateId,
    /// Empty, too-large or inconsistent feature vector.
    InvalidFeatureDimensions,
    /// Feature is non-finite or outside [0, 1].
    InvalidFeatureValue,
    /// A supplied cost is non-finite or non-positive.
    InvalidEstimatedCost,
    /// The cost-aware policy was requested but at least one candidate has no cost.
    MissingEstimatedCost,
    /// The cost-aware policy was requested without a cost unit.
    MissingCostUnit,
    /// Request line exceeds the hard byte limit.
    RequestTooLarge,
}

impl SelectionError {
    /// Stable machine-readable error category.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidJson(_) => "invalid_json_or_forbidden_field",
            Self::UnsupportedSchemaVersion => "unsupported_schema_version",
            Self::InvalidFeatureContractDigest => "invalid_feature_contract_digest",
            Self::InvalidBudget => "invalid_budget",
            Self::InvalidCandidateCount => "invalid_candidate_count",
            Self::InvalidCandidateId => "invalid_candidate_id",
            Self::InvalidFeatureDimensions => "invalid_feature_dimensions",
            Self::InvalidFeatureValue => "invalid_feature_value",
            Self::InvalidEstimatedCost => "invalid_estimated_cost",
            Self::MissingEstimatedCost => "missing_estimated_cost",
            Self::MissingCostUnit => "missing_cost_unit",
            Self::RequestTooLarge => "request_too_large",
        }
    }

    fn message(&self) -> String {
        let text = match self {
            Self::InvalidJson(message) => message.clone(),
            Self::UnsupportedSchemaVersion => {
                format!("schema_version must be {SCHEMA_VERSION}")
            }
            Self::InvalidFeatureContractDigest => {
                "feature_contract_sha256 must be 64 lowercase hexadecimal characters".to_owned()
            }
            Self::InvalidBudget => format!("budget must be between 1 and {MAX_BUDGET}"),
            Self::InvalidCandidateCount => {
                format!("candidate pool must contain 1..={MAX_CANDIDATES} candidates")
            }
            Self::InvalidCandidateId => format!(
                "candidate_id and family_id must be nonempty, <= {MAX_ID_BYTES} bytes; candidate_id must be unique"
            ),
            Self::InvalidFeatureDimensions => format!(
                "normalized_features dimension must be 1..={MAX_FEATURE_DIMENSIONS}, equal for all candidates"
            ),
            Self::InvalidFeatureValue => {
                "every normalized feature must be finite and in [0, 1]".to_owned()
            }
            Self::InvalidEstimatedCost => {
                "estimated_cost must be finite and strictly positive".to_owned()
            }
            Self::MissingEstimatedCost => {
                "cost_aware_farthest_first requires estimated_cost for every candidate".to_owned()
            }
            Self::MissingCostUnit => {
                "cost_aware_farthest_first requires a nonempty cost_unit".to_owned()
            }
            Self::RequestTooLarge => format!("request line exceeds {MAX_LINE_BYTES} bytes"),
        };
        text.chars().take(300).collect()
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Parse and handle one JSONL request, returning one JSONL response.
///
/// Invalid requests yield a machine-readable error and do not panic. The input
/// digest is computed before parsing and is not a claim about data truth.
pub fn handle_line(line: &str) -> String {
    let digest = sha256_hex(line.as_bytes());
    if line.len() > MAX_LINE_BYTES {
        return error_json(&digest, &SelectionError::RequestTooLarge);
    }

    match serde_json::from_str::<SelectionRequest>(line) {
        Ok(request) => match select(request, line.as_bytes()) {
            Ok(response) => serde_json::to_string(&response)
                .unwrap_or_else(|_| fallback_error_json(&digest)),
            Err(error) => error_json(&digest, &error),
        },
        Err(error) => error_json(
            &digest,
            &SelectionError::InvalidJson(error.to_string()),
        ),
    }
}

fn error_json(digest: &str, error: &SelectionError) -> String {
    let response = ErrorResponse {
        schema_version: SCHEMA_VERSION.to_owned(),
        status: "rejected_input".to_owned(),
        request_sha256: digest.to_owned(),
        error_code: error.code().to_owned(),
        message: error.message(),
    };
    serde_json::to_string(&response)
        .unwrap_or_else(|_| fallback_error_json(digest))
}

fn fallback_error_json(digest: &str) -> String {
    format!(
        "{{\"schema_version\":\"{}\",\"status\":\"internal_serialization_error\",\"request_sha256\":\"{}\"}}",
        SCHEMA_VERSION, digest
    )
}

/// Validate a request and select up to its budget.
///
/// No target label, predicted property, oracle result or post-outcome field is
/// consulted by any policy implemented here.
fn select(
    request: SelectionRequest,
    request_bytes: &[u8],
) -> Result<SelectionResponse, SelectionError> {
    validate(&request)?;
    let request_sha256 = sha256_hex(request_bytes);

    let mut sorted: Vec<usize> = (0..request.candidates.len()).collect();
    sorted.sort_by(|a, b| {
        request.candidates[*a]
            .candidate_id
            .cmp(&request.candidates[*b].candidate_id)
    });
    let budget = request.budget.min(sorted.len());

    let decisions = match request.policy {
        Policy::RandomSeeded => random_selection(&request, &sorted, budget),
        Policy::FarthestFirst => farthest_first(&request, &sorted, budget),
        Policy::GroupBalancedFarthestFirst => {
            group_balanced_farthest_first(&request, &sorted, budget)
        }
        Policy::CostAwareFarthestFirst => {
            cost_aware_farthest_first(&request, &sorted, budget)
        }
    };

    Ok(SelectionResponse {
        schema_version: SCHEMA_VERSION.to_owned(),
        status: "completed".to_owned(),
        policy: request.policy,
        seed: request.seed,
        request_sha256: request_sha256.clone(),
        feature_contract_sha256: request.feature_contract_sha256,
        cost_unit: request.cost_unit,
        candidate_count: request.candidates.len(),
        requested_budget: request.budget,
        selected_count: decisions.len(),
        selected: decisions,
        limitations: vec![
            "Descriptor-space diversity is not a physical property, calibrated uncertainty, or proof of novelty.".to_owned(),
            "Feature normalization, provenance, rights, and holdout integrity must be established outside this selector.".to_owned(),
            "This policy prototype is not an atomistic, optical, biological, or manufacturing simulator.".to_owned(),
        ],
    })
}

fn validate(request: &SelectionRequest) -> Result<(), SelectionError> {
    if request.schema_version != SCHEMA_VERSION {
        return Err(SelectionError::UnsupportedSchemaVersion);
    }
    if !is_lowercase_sha256(&request.feature_contract_sha256) {
        return Err(SelectionError::InvalidFeatureContractDigest);
    }
    if request.budget == 0 || request.budget > MAX_BUDGET {
        return Err(SelectionError::InvalidBudget);
    }
    if request.candidates.is_empty()
        || request.candidates.len() > MAX_CANDIDATES
    {
        return Err(SelectionError::InvalidCandidateCount);
    }

    let first_dimension = request.candidates[0].normalized_features.len();
    if first_dimension == 0 || first_dimension > MAX_FEATURE_DIMENSIONS {
        return Err(SelectionError::InvalidFeatureDimensions);
    }

    if request.policy == Policy::CostAwareFarthestFirst {
        if request
            .cost_unit
            .as_deref()
            .is_none_or(|unit| unit.trim().is_empty() || unit.len() > 64)
        {
            return Err(SelectionError::MissingCostUnit);
        }
        if request.candidates.iter().any(|candidate| candidate.estimated_cost.is_none()) {
            return Err(SelectionError::MissingEstimatedCost);
        }
    }

    let mut ids = BTreeSet::new();
    for candidate in &request.candidates {
        if candidate.candidate_id.trim().is_empty()
            || candidate.family_id.trim().is_empty()
            || candidate.candidate_id.len() > MAX_ID_BYTES
            || candidate.family_id.len() > MAX_ID_BYTES
            || !ids.insert(candidate.candidate_id.as_str())
        {
            return Err(SelectionError::InvalidCandidateId);
        }
        if candidate.normalized_features.len() != first_dimension {
            return Err(SelectionError::InvalidFeatureDimensions);
        }
        if candidate
            .normalized_features
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(SelectionError::InvalidFeatureValue);
        }
        if candidate
            .estimated_cost
            .is_some_and(|cost| !cost.is_finite() || cost <= 0.0)
        {
            return Err(SelectionError::InvalidEstimatedCost);
        }
    }
    Ok(())
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn random_selection(
    request: &SelectionRequest,
    sorted: &[usize],
    budget: usize,
) -> Vec<SelectionDecision> {
    let mut order = sorted.to_vec();
    let mut rng = SplitMix64::new(request.seed);
    for end in (1..order.len()).rev() {
        let index = rng.index(end + 1);
        order.swap(end, index);
    }
    order
        .into_iter()
        .take(budget)
        .enumerate()
        .map(|(index, candidate_index)| {
            decision(
                &request.candidates[candidate_index],
                index + 1,
                None,
                "seeded_random_without_replacement",
            )
        })
        .collect()
}

fn farthest_first(
    request: &SelectionRequest,
    sorted: &[usize],
    budget: usize,
) -> Vec<SelectionDecision> {
    let first = sorted[0];
    let mut selected_flags = vec![false; request.candidates.len()];
    let mut minimum_squared_distances = vec![f64::INFINITY; request.candidates.len()];
    let mut selected_count = 0usize;
    let mut decisions = Vec::with_capacity(budget);

    selected_flags[first] = true;
    selected_count += 1;
    decisions.push(decision(
        &request.candidates[first],
        selected_count,
        None,
        "lexicographic_seed",
    ));
    update_distances(
        request,
        first,
        sorted,
        &selected_flags,
        &mut minimum_squared_distances,
    );

    while selected_count < budget {
        let next = select_farthest(
            sorted,
            &selected_flags,
            &minimum_squared_distances,
        )
        .expect("budget is bounded by validated candidate count");
        let distance = minimum_squared_distances[next].sqrt();
        selected_flags[next] = true;
        selected_count += 1;
        decisions.push(decision(
            &request.candidates[next],
            selected_count,
            Some(distance),
            "maximin_normalized_descriptor_distance",
        ));
        update_distances(
            request,
            next,
            sorted,
            &selected_flags,
            &mut minimum_squared_distances,
        );
    }
    decisions
}

fn group_balanced_farthest_first(
    request: &SelectionRequest,
    sorted: &[usize],
    budget: usize,
) -> Vec<SelectionDecision> {
    let mut order = sorted.to_vec();
    let mut rng = SplitMix64::new(request.seed);
    let initial_position = rng.index(order.len());
    let first = order.remove(initial_position);
    let mut selected_flags = vec![false; request.candidates.len()];
    let mut minimum_squared_distances = vec![f64::INFINITY; request.candidates.len()];
    let mut family_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut decisions = Vec::with_capacity(budget);

    selected_flags[first] = true;
    *family_counts
        .entry(request.candidates[first].family_id.as_str())
        .or_insert(0) += 1;
    decisions.push(decision(
        &request.candidates[first],
        1,
        None,
        "seeded_initial_candidate",
    ));
    update_distances(
        request,
        first,
        sorted,
        &selected_flags,
        &mut minimum_squared_distances,
    );

    while decisions.len() < budget {
        let least_represented_count = sorted
            .iter()
            .filter(|index| !selected_flags[**index])
            .map(|index| {
                *family_counts
                    .get(request.candidates[*index].family_id.as_str())
                    .unwrap_or(&0)
            })
            .min()
            .expect("budget is bounded by validated candidate count");

        // Sorted candidate order and strict comparisons make ties deterministic.
        let mut candidates = sorted.iter().copied().filter(|index| {
            !selected_flags[*index]
                && family_counts
                    .get(request.candidates[*index].family_id.as_str())
                    .copied()
                    .unwrap_or(0)
                    == least_represented_count
        });
        let mut best = candidates
            .next()
            .expect("a least-represented candidate exists");
        for candidate_index in candidates {
            if minimum_squared_distances[candidate_index]
                > minimum_squared_distances[best]
            {
                best = candidate_index;
            }
        }

        let distance = minimum_squared_distances[best].sqrt();
        selected_flags[best] = true;
        *family_counts
            .entry(request.candidates[best].family_id.as_str())
            .or_insert(0) += 1;
        decisions.push(decision(
            &request.candidates[best],
            decisions.len() + 1,
            Some(distance),
            "least_represented_family_then_maximin_distance",
        ));
        update_distances(
            request,
            best,
            sorted,
            &selected_flags,
            &mut minimum_squared_distances,
        );
    }
    decisions
}

/// Select a cheapest anchor, then maximize minimum descriptor distance per
/// declared evaluation cost. This is a cost-aware diversity baseline, not
/// expected information gain, calibrated utility, or a physical prediction.
fn cost_aware_farthest_first(
    request: &SelectionRequest,
    sorted: &[usize],
    budget: usize,
) -> Vec<SelectionDecision> {
    let first = sorted
        .iter()
        .copied()
        .min_by(|left, right| {
            request.candidates[*left]
                .estimated_cost
                .expect("cost-aware requests are validated")
                .total_cmp(
                    &request.candidates[*right]
                        .estimated_cost
                        .expect("cost-aware requests are validated"),
                )
                .then_with(|| {
                    request.candidates[*left]
                        .candidate_id
                        .cmp(&request.candidates[*right].candidate_id)
                })
        })
        .expect("validated candidate pool is nonempty");

    let mut selected_flags = vec![false; request.candidates.len()];
    let mut minimum_squared_distances = vec![f64::INFINITY; request.candidates.len()];
    let mut decisions = Vec::with_capacity(budget);

    selected_flags[first] = true;
    decisions.push(decision(
        &request.candidates[first],
        1,
        None,
        "lowest_declared_cost_anchor",
    ));
    update_distances(
        request,
        first,
        sorted,
        &selected_flags,
        &mut minimum_squared_distances,
    );

    while decisions.len() < budget {
        let mut best: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        for candidate_index in sorted.iter().copied() {
            if selected_flags[candidate_index] {
                continue;
            }
            let cost = request.candidates[candidate_index]
                .estimated_cost
                .expect("cost-aware requests are validated");
            let score =
                minimum_squared_distances[candidate_index].sqrt() / cost;
            if score > best_score {
                best = Some(candidate_index);
                best_score = score;
            }
        }

        let next = best
            .expect("budget is bounded by validated candidate count");
        let distance = minimum_squared_distances[next].sqrt();
        selected_flags[next] = true;
        decisions.push(decision(
            &request.candidates[next],
            decisions.len() + 1,
            Some(distance),
            "maximin_normalized_descriptor_distance_per_cost",
        ));
        update_distances(
            request,
            next,
            sorted,
            &selected_flags,
            &mut minimum_squared_distances,
        );
    }
    decisions
}

fn select_farthest(
    sorted: &[usize],
    selected_flags: &[bool],
    minimum_squared_distances: &[f64],
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for candidate_index in sorted.iter().copied() {
        if selected_flags[candidate_index] {
            continue;
        }
        match best {
            None => best = Some(candidate_index),
            Some(current)
                if minimum_squared_distances[candidate_index]
                    > minimum_squared_distances[current] =>
            {
                best = Some(candidate_index);
            }
            _ => {}
        }
    }
    best
}

fn update_distances(
    request: &SelectionRequest,
    newly_selected: usize,
    sorted: &[usize],
    selected_flags: &[bool],
    minimum_squared_distances: &mut [f64],
) {
    let reference =
        &request.candidates[newly_selected].normalized_features;
    for candidate_index in sorted.iter().copied() {
        if selected_flags[candidate_index] {
            continue;
        }
        let distance = squared_distance(
            reference,
            &request.candidates[candidate_index].normalized_features,
        );
        if distance < minimum_squared_distances[candidate_index] {
            minimum_squared_distances[candidate_index] = distance;
        }
    }
}

fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum()
}

fn decision(
    candidate: &Candidate,
    rank: usize,
    min_distance: Option<f64>,
    reason: &str,
) -> SelectionDecision {
    SelectionDecision {
        rank,
        candidate_id: candidate.candidate_id.clone(),
        family_id: candidate.family_id.clone(),
        estimated_cost: candidate.estimated_cost,
        min_distance,
        reason: reason.to_owned(),
    }
}

/// Small deterministic PRNG; avoids dependence on third-party RNG versions.
#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn index(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        let upper = upper_exclusive as u64;
        let rejection_threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u64();
            if value >= rejection_threshold {
                return (value % upper) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest() -> String {
        "a".repeat(64)
    }

    fn candidate(id: &str, family: &str, x: f64, y: f64) -> Candidate {
        Candidate {
            candidate_id: id.to_owned(),
            estimated_cost: Some(1.0),
            family_id: family.to_owned(),
            normalized_features: vec![x, y],
        }
    }

    fn request(policy: Policy, budget: usize) -> SelectionRequest {
        SelectionRequest {
            schema_version: SCHEMA_VERSION.to_owned(),
            policy,
            budget,
            seed: 42,
            feature_contract_sha256: digest(),
            cost_unit: None,
            candidates: vec![
                candidate("candidate-c", "family-b", 1.0, 1.0),
                candidate("candidate-a", "family-a", 0.0, 0.0),
                candidate("candidate-b", "family-a", 0.2, 0.2),
                candidate("candidate-d", "family-c", 0.8, 0.8),
            ],
        }
    }

    #[test]
    fn cost_aware_policy_prefers_coverage_per_cost() {
        let mut req = request(Policy::CostAwareFarthestFirst, 2);
        req.cost_unit = Some("simulation_seconds".to_owned());
        req.candidates[0].estimated_cost = Some(10.0);
        req.candidates[1].estimated_cost = Some(2.0);
        req.candidates[2].estimated_cost = Some(0.5);
        req.candidates[3].estimated_cost = Some(1.0);

        let result = select(req, b"cost-aware-test").unwrap();
        assert_eq!(result.selected[0].candidate_id, "candidate-b");
        assert_eq!(result.selected[1].candidate_id, "candidate-d");
        assert_eq!(
            result.selected[1].reason,
            "maximin_normalized_descriptor_distance_per_cost"
        );
    }

    #[test]
    fn cost_aware_policy_rejects_missing_and_invalid_costs() {
        let mut req = request(Policy::CostAwareFarthestFirst, 1);
        assert_eq!(validate(&req), Err(SelectionError::MissingCostUnit));
        req.cost_unit = Some("simulation_seconds".to_owned());
        req.candidates[0].estimated_cost = None;
        assert_eq!(
            validate(&req),
            Err(SelectionError::MissingEstimatedCost)
        );
        req.policy = Policy::FarthestFirst;
        req.candidates[0].estimated_cost = Some(0.0);
        assert_eq!(
            validate(&req),
            Err(SelectionError::InvalidEstimatedCost)
        );
    }

    #[test]
    fn cost_aware_receipt_preserves_cost_unit_and_selected_cost() {
        let mut req = request(Policy::CostAwareFarthestFirst, 1);
        req.cost_unit = Some("simulation_seconds".to_owned());
        let result = select(req, b"cost-receipt-test").unwrap();
        assert_eq!(
            result.cost_unit.as_deref(),
            Some("simulation_seconds")
        );
        assert_eq!(result.selected[0].estimated_cost, Some(1.0));
    }

    #[test]
    fn farthest_first_selects_distant_candidates() {
        let request_bytes = b"farthest-first-test";
        let result =
            select(request(Policy::FarthestFirst, 2), request_bytes).unwrap();
        assert_eq!(result.selected_count, 2);
        assert_eq!(result.selected[0].candidate_id, "candidate-a");
        assert_eq!(result.selected[1].candidate_id, "candidate-c");
        assert!(
            (result.selected[1].min_distance.unwrap() - 2.0_f64.sqrt()).abs() < 1e-12
        );
    }

    #[test]
    fn random_policy_is_repeatable_and_without_replacement() {
        let req = request(Policy::RandomSeeded, 3);
        let first = select(req.clone(), b"random-test").unwrap();
        let second = select(req, b"random-test").unwrap();
        assert_eq!(first, second);
        let ids: BTreeSet<_> = first
            .selected
            .iter()
            .map(|item| item.candidate_id.as_str())
            .collect();
        assert_eq!(ids.len(), first.selected_count);
    }

    #[test]
    fn group_balanced_policy_spreads_across_families() {
        let result = select(
            request(Policy::GroupBalancedFarthestFirst, 3),
            b"group-test",
        )
        .unwrap();
        let family_count = result
            .selected
            .iter()
            .map(|item| item.family_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(family_count.len(), 3);
    }

    #[test]
    fn request_order_does_not_change_farthest_first_result() {
        let first_req = request(Policy::FarthestFirst, 3);
        let mut second_req = first_req.clone();
        second_req.candidates.reverse();
        let first = select(first_req, b"order-test-one").unwrap();
        let second = select(second_req, b"order-test-two").unwrap();
        let first_ids: Vec<_> = first
            .selected
            .iter()
            .map(|item| item.candidate_id.clone())
            .collect();
        let second_ids: Vec<_> = second
            .selected
            .iter()
            .map(|item| item.candidate_id.clone())
            .collect();
        assert_eq!(first_ids, second_ids);
    }

    #[test]
    fn invalid_normalized_values_are_rejected() {
        let mut req = request(Policy::RandomSeeded, 1);
        req.candidates[0].normalized_features[0] = f64::NAN;
        assert_eq!(validate(&req), Err(SelectionError::InvalidFeatureValue));
        req.candidates[0].normalized_features[0] = 1.01;
        assert_eq!(validate(&req), Err(SelectionError::InvalidFeatureValue));
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let mut req = request(Policy::RandomSeeded, 1);
        req.candidates[0].candidate_id = req.candidates[1].candidate_id.clone();
        assert_eq!(validate(&req), Err(SelectionError::InvalidCandidateId));
    }

    #[test]
    fn inconsistent_descriptor_dimensions_are_rejected() {
        let mut req = request(Policy::RandomSeeded, 1);
        req.candidates[0].normalized_features.pop();
        assert_eq!(validate(&req), Err(SelectionError::InvalidFeatureDimensions));
    }

    #[test]
    fn unknown_target_fields_are_rejected_by_deserializer() {
        let json = format!(
            r#"{{"schema_version":"{}","policy":"random_seeded","budget":1,"seed":1,"feature_contract_sha256":"{}","candidates":[{{"candidate_id":"a","family_id":"x","normalized_features":[0.1],"target_energy_eV":-9.1}}]}}"#,
            SCHEMA_VERSION,
            digest()
        );
        assert!(serde_json::from_str::<SelectionRequest>(&json).is_err());
    }

    #[test]
    fn zero_budget_is_rejected() {
        let req = request(Policy::RandomSeeded, 0);
        assert_eq!(validate(&req), Err(SelectionError::InvalidBudget));
    }

    #[test]
    fn handle_line_returns_receipt_for_valid_request() {
        let req = request(Policy::FarthestFirst, 2);
        let json = serde_json::to_string(&req).unwrap();
        let response: SelectionResponse =
            serde_json::from_str(&handle_line(&json)).unwrap();
        assert_eq!(response.status, "completed");
        assert_eq!(response.request_sha256, sha256_hex(json.as_bytes()));
    }

    #[test]
    fn checked_in_example_fixture_runs_end_to_end() {
        let line = include_str!("../fixtures/example-request.jsonl").trim_end();
        let response: SelectionResponse =
            serde_json::from_str(&handle_line(line)).unwrap();
        let ids: Vec<_> = response
            .selected
            .iter()
            .map(|item| item.candidate_id.as_str())
            .collect();

        assert_eq!(ids, vec!["c-a", "c-c"]);
        assert_eq!(response.request_sha256, sha256_hex(line.as_bytes()));
    }

    #[test]
    fn checked_in_json_schemas_are_parseable_and_fail_closed() {
        let request_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/request.schema.json")).unwrap();
        let response_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/response.schema.json")).unwrap();

        assert_eq!(request_schema["additionalProperties"], false);
        assert_eq!(
            request_schema["properties"]["candidates"]["items"]["additionalProperties"],
            false
        );
        assert!(request_schema["properties"]["policy"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "cost_aware_farthest_first"));
        assert!(request_schema["allOf"].as_array().unwrap().iter().any(|gate| {
            gate["if"]["properties"]["policy"]["const"] == "cost_aware_farthest_first"
        }));
        assert_eq!(response_schema["oneOf"].as_array().unwrap().len(), 3);
        assert_eq!(
            response_schema["oneOf"][0]["properties"]["cost_unit"]["type"][0],
            "string"
        );
    }

    #[test]
    fn malformed_line_returns_json_error_instead_of_panicking() {
        let response: ErrorResponse =
            serde_json::from_str(&handle_line("{")).unwrap();
        assert_eq!(response.status, "rejected_input");
        assert_eq!(response.error_code, "invalid_json_or_forbidden_field");
    }
}
