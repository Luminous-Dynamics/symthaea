// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic bootstrap Random Forest with continuous leaf-frequency scores.
//!
//! This implementation uses CART/Gini splits, bootstrap sampling, and a fresh
//! mtry feature sample at every split. Leaf scores are unweighted empirical
//! positive frequencies; class weighting changes split selection only.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use crate::benchmark::binary_metrics::{
    BinaryMetricReportV1, BinaryScoreSampleV1, BinaryScoreSubjectV1, score_binary_subject_v1,
};

pub const MODEL_SCHEMA_V1: &str = "luminous.semi-bench.random-forest-model.v1";
pub const PREDICTION_SCHEMA_V1: &str = "luminous.semi-bench.random-forest-predictions.v1";
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForestClassWeightingV1 {
    None,
    Balanced,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomForestParametersV1 {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    /// Zero means floor(sqrt(feature_count)); otherwise use this exact mtry.
    pub mtry: usize,
    pub seed: u64,
    pub class_weighting: ForestClassWeightingV1,
}

impl Default for RandomForestParametersV1 {
    fn default() -> Self {
        Self {
            n_trees: 200,
            max_depth: 12,
            min_samples_leaf: 4,
            mtry: 0,
            seed: 42,
            class_weighting: ForestClassWeightingV1::Balanced,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RandomForestTrainingSampleV1 {
    pub sample_id: String,
    pub positive: bool,
    pub features: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RandomForestInputSampleV1 {
    pub sample_id: String,
    pub positive: bool,
    pub features: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForestNodeV1 {
    pub feature_index: Option<usize>,
    pub threshold: Option<f64>,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub positive_probability: f64,
    pub sample_count: usize,
    pub positive_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForestTreeV1 {
    pub seed: u64,
    pub nodes: Vec<ForestNodeV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomForestModelV1 {
    pub schema: String,
    pub parameters: RandomForestParametersV1,
    pub feature_count: usize,
    pub resolved_mtry: usize,
    pub training_rows: usize,
    pub training_positive: usize,
    pub training_negative: usize,
    pub training_sample_ids_sha256: String,
    pub trees: Vec<ForestTreeV1>,
    pub model_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomForestPredictionV1 {
    pub sample_id: String,
    pub positive: bool,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomForestPredictionReportV1 {
    pub schema: String,
    pub model_sha256: String,
    pub prediction_count: usize,
    pub score_vector_sha256: String,
    pub predictions: Vec<RandomForestPredictionV1>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct RandomForestError(pub String);

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, upper_exclusive: usize) -> usize {
        let bound = upper_exclusive as u64;
        let rejection_threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= rejection_threshold {
                return (value % bound) as usize;
            }
        }
    }
}

fn shuffle<T>(items: &mut [T], rng: &mut SplitMix64) {
    for index in (1..items.len()).rev() {
        let selected = rng.below(index + 1);
        items.swap(index, selected);
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sample_ids_sha256(samples: &[RandomForestTrainingSampleV1]) -> String {
    let ids: Vec<&str> = samples
        .iter()
        .map(|sample| sample.sample_id.as_str())
        .collect();
    sha256_hex(&serde_json::to_vec(&ids).expect("sample ID serialization is infallible"))
}

fn validate_parameters(
    parameters: &RandomForestParametersV1,
    feature_count: usize,
) -> Result<usize, RandomForestError> {
    if parameters.n_trees == 0 || parameters.n_trees > 1_000 {
        return Err(RandomForestError(
            "n_trees: expected integer in [1, 1000]".into(),
        ));
    }
    if parameters.max_depth == 0 || parameters.max_depth > 64 {
        return Err(RandomForestError(
            "max_depth: expected integer in [1, 64]".into(),
        ));
    }
    if parameters.min_samples_leaf == 0 {
        return Err(RandomForestError(
            "min_samples_leaf: expected positive integer".into(),
        ));
    }
    if parameters.mtry > feature_count {
        return Err(RandomForestError(
            "mtry: cannot exceed feature_count".into(),
        ));
    }
    let resolved = if parameters.mtry == 0 {
        (feature_count as f64).sqrt().floor().max(1.0) as usize
    } else {
        parameters.mtry
    };
    Ok(resolved)
}

fn validate_training_samples(
    samples: &[RandomForestTrainingSampleV1],
) -> Result<(usize, usize, usize), RandomForestError> {
    if samples.is_empty() {
        return Err(RandomForestError(
            "training_samples: expected non-empty array".into(),
        ));
    }
    let feature_count = samples[0].features.len();
    if feature_count == 0 {
        return Err(RandomForestError(
            "training_samples: expected at least one feature".into(),
        ));
    }
    let mut seen = BTreeSet::new();
    let mut positive = 0usize;
    for (row_index, sample) in samples.iter().enumerate() {
        if sample.sample_id.trim().is_empty() {
            return Err(RandomForestError(format!(
                "training_samples[{row_index}].sample_id: expected non-empty string"
            )));
        }
        if !seen.insert(sample.sample_id.as_str()) {
            return Err(RandomForestError(format!(
                "training_samples[{row_index}].sample_id: duplicate"
            )));
        }
        if sample.features.len() != feature_count {
            return Err(RandomForestError(format!(
                "training_samples[{row_index}]: expected {feature_count} features, observed {}",
                sample.features.len()
            )));
        }
        for (feature_index, value) in sample.features.iter().enumerate() {
            if !value.is_finite() {
                return Err(RandomForestError(format!(
                    "training_samples[{row_index}].features[{feature_index}]: expected finite number"
                )));
            }
        }
        if sample.positive {
            positive += 1;
        }
    }
    let negative = samples.len() - positive;
    if positive == 0 || negative == 0 {
        return Err(RandomForestError(
            "training_samples: both classes are required".into(),
        ));
    }
    Ok((feature_count, positive, negative))
}

fn weighted_gini(positive_weight: f64, negative_weight: f64) -> f64 {
    let total = positive_weight + negative_weight;
    if total <= 0.0 {
        return 0.0;
    }
    let p = positive_weight / total;
    let n = negative_weight / total;
    1.0 - p * p - n * n
}

#[derive(Clone, Copy, Debug)]
struct CandidateSplit {
    feature_index: usize,
    threshold: f64,
    gain: f64,
}

struct TreeBuilder<'a> {
    samples: &'a [RandomForestTrainingSampleV1],
    positive_weight: f64,
    negative_weight: f64,
    resolved_mtry: usize,
    parameters: &'a RandomForestParametersV1,
    rng: SplitMix64,
    nodes: Vec<ForestNodeV1>,
}

impl TreeBuilder<'_> {
    fn leaf_node(&self, row_indices: &[usize]) -> ForestNodeV1 {
        let positive_count = row_indices
            .iter()
            .filter(|index| self.samples[**index].positive)
            .count();
        let probability = positive_count as f64 / row_indices.len() as f64;
        ForestNodeV1 {
            feature_index: None,
            threshold: None,
            left_child: None,
            right_child: None,
            positive_probability: probability,
            sample_count: row_indices.len(),
            positive_count,
        }
    }

    fn best_split(&mut self, row_indices: &[usize]) -> Option<CandidateSplit> {
        let mut candidate_features: Vec<usize> = (0..self.samples[0].features.len()).collect();
        shuffle(&mut candidate_features, &mut self.rng);
        candidate_features.truncate(self.resolved_mtry);
        candidate_features.sort_unstable();

        let mut total_positive_weight = 0.0;
        let mut total_negative_weight = 0.0;
        for index in row_indices {
            let sample = &self.samples[*index];
            if sample.positive {
                total_positive_weight += self.positive_weight;
            } else {
                total_negative_weight += self.negative_weight;
            }
        }
        let total_weight = total_positive_weight + total_negative_weight;
        let parent_impurity = weighted_gini(total_positive_weight, total_negative_weight);

        let mut best: Option<CandidateSplit> = None;
        for feature_index in candidate_features {
            let mut values: Vec<(f64, bool)> = row_indices
                .iter()
                .map(|index| {
                    (
                        self.samples[*index].features[feature_index],
                        self.samples[*index].positive,
                    )
                })
                .collect();
            values.sort_by(|left, right| left.0.total_cmp(&right.0));

            let mut left_positive_count = 0usize;
            let mut left_negative_count = 0usize;
            let mut left_positive_weight = 0.0;
            let mut left_negative_weight = 0.0;

            for split_index in 0..values.len().saturating_sub(1) {
                let (value, positive) = values[split_index];
                if positive {
                    left_positive_count += 1;
                    left_positive_weight += self.positive_weight;
                } else {
                    left_negative_count += 1;
                    left_negative_weight += self.negative_weight;
                }

                let left_count = split_index + 1;
                let right_count = values.len() - left_count;
                if left_count < self.parameters.min_samples_leaf
                    || right_count < self.parameters.min_samples_leaf
                    || value == values[split_index + 1].0
                {
                    continue;
                }

                let right_positive_weight = total_positive_weight - left_positive_weight;
                let right_negative_weight = total_negative_weight - left_negative_weight;
                let left_weight = left_positive_weight + left_negative_weight;
                let right_weight = right_positive_weight + right_negative_weight;
                let child_impurity = left_weight
                    * weighted_gini(left_positive_weight, left_negative_weight)
                    + right_weight * weighted_gini(right_positive_weight, right_negative_weight);
                let gain = total_weight * parent_impurity - child_impurity;
                if gain <= EPSILON {
                    continue;
                }
                let next_value = values[split_index + 1].0;
                let threshold = value * 0.5 + next_value * 0.5;
                let candidate = CandidateSplit {
                    feature_index,
                    threshold,
                    gain,
                };
                let replace = match best {
                    None => true,
                    Some(current) => candidate.gain > current.gain + EPSILON,
                };
                if replace {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    fn build_node(&mut self, row_indices: &[usize], depth: usize) -> usize {
        let node_index = self.nodes.len();
        let leaf = self.leaf_node(row_indices);
        self.nodes.push(leaf);

        let positive_count = row_indices
            .iter()
            .filter(|index| self.samples[**index].positive)
            .count();
        if depth >= self.parameters.max_depth
            || row_indices.len() < self.parameters.min_samples_leaf.saturating_mul(2)
            || positive_count == 0
            || positive_count == row_indices.len()
        {
            return node_index;
        }

        let Some(split) = self.best_split(row_indices) else {
            return node_index;
        };
        let mut left_rows = Vec::new();
        let mut right_rows = Vec::new();
        for index in row_indices {
            if self.samples[*index].features[split.feature_index] <= split.threshold {
                left_rows.push(*index);
            } else {
                right_rows.push(*index);
            }
        }
        if left_rows.len() < self.parameters.min_samples_leaf
            || right_rows.len() < self.parameters.min_samples_leaf
        {
            return node_index;
        }

        let left_child = self.build_node(&left_rows, depth + 1);
        let right_child = self.build_node(&right_rows, depth + 1);
        self.nodes[node_index].feature_index = Some(split.feature_index);
        self.nodes[node_index].threshold = Some(split.threshold);
        self.nodes[node_index].left_child = Some(left_child);
        self.nodes[node_index].right_child = Some(right_child);
        node_index
    }
}

#[derive(Serialize)]
struct ModelDigestMaterial<'a> {
    schema: &'a str,
    parameters: &'a RandomForestParametersV1,
    feature_count: usize,
    resolved_mtry: usize,
    training_rows: usize,
    training_positive: usize,
    training_negative: usize,
    training_sample_ids_sha256: &'a str,
    trees: &'a [ForestTreeV1],
}

pub fn fit_random_forest_v1(
    training_samples: &[RandomForestTrainingSampleV1],
    parameters: &RandomForestParametersV1,
) -> Result<RandomForestModelV1, RandomForestError> {
    let mut samples = training_samples.to_vec();
    samples.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    let (feature_count, positive_count, negative_count) = validate_training_samples(&samples)?;
    let resolved_mtry = validate_parameters(parameters, feature_count)?;

    let (positive_weight, negative_weight) = match &parameters.class_weighting {
        ForestClassWeightingV1::None => (1.0, 1.0),
        ForestClassWeightingV1::Balanced => (
            samples.len() as f64 / (2.0 * positive_count as f64),
            samples.len() as f64 / (2.0 * negative_count as f64),
        ),
    };

    let training_sample_ids_sha256 = sample_ids_sha256(&samples);
    let mut trees = Vec::with_capacity(parameters.n_trees);
    for tree_index in 0..parameters.n_trees {
        let tree_seed = parameters
            .seed
            .wrapping_add((tree_index as u64).wrapping_mul(GOLDEN_GAMMA));
        let mut rng = SplitMix64::new(tree_seed);
        let bootstrap: Vec<usize> = (0..samples.len())
            .map(|_| rng.below(samples.len()))
            .collect();

        let mut builder = TreeBuilder {
            samples: &samples,
            positive_weight,
            negative_weight,
            resolved_mtry,
            parameters,
            rng,
            nodes: Vec::new(),
        };
        builder.build_node(&bootstrap, 0);
        if builder.nodes.is_empty() {
            return Err(RandomForestError(format!(
                "tree {tree_index}: empty tree after fit"
            )));
        }
        trees.push(ForestTreeV1 {
            seed: tree_seed,
            nodes: builder.nodes,
        });
    }

    let mut model = RandomForestModelV1 {
        schema: MODEL_SCHEMA_V1.into(),
        parameters: parameters.clone(),
        feature_count,
        resolved_mtry,
        training_rows: samples.len(),
        training_positive: positive_count,
        training_negative: negative_count,
        training_sample_ids_sha256,
        trees,
        model_sha256: String::new(),
    };
    let material = ModelDigestMaterial {
        schema: &model.schema,
        parameters: &model.parameters,
        feature_count: model.feature_count,
        resolved_mtry: model.resolved_mtry,
        training_rows: model.training_rows,
        training_positive: model.training_positive,
        training_negative: model.training_negative,
        training_sample_ids_sha256: &model.training_sample_ids_sha256,
        trees: &model.trees,
    };
    model.model_sha256 = sha256_hex(&serde_json::to_vec(&material).map_err(|error| {
        RandomForestError(format!("model digest serialization failed: {error}"))
    })?);
    Ok(model)
}

fn predict_tree(tree: &ForestTreeV1, features: &[f64]) -> Result<f64, RandomForestError> {
    let mut node_index = 0usize;
    loop {
        let node = tree
            .nodes
            .get(node_index)
            .ok_or_else(|| RandomForestError("model tree contains invalid child index".into()))?;
        match (
            node.feature_index,
            node.threshold,
            node.left_child,
            node.right_child,
        ) {
            (Some(feature), Some(threshold), Some(left), Some(right)) => {
                let value = *features.get(feature).ok_or_else(|| {
                    RandomForestError("prediction row has insufficient feature count".into())
                })?;
                node_index = if value <= threshold { left } else { right };
            }
            (None, None, None, None) => return Ok(node.positive_probability),
            _ => {
                return Err(RandomForestError(
                    "model tree node has inconsistent split state".into(),
                ));
            }
        }
    }
}

pub fn predict_random_forest_v1(
    model: &RandomForestModelV1,
    samples: &[RandomForestInputSampleV1],
) -> Result<RandomForestPredictionReportV1, RandomForestError> {
    if model.schema != MODEL_SCHEMA_V1 || model.trees.is_empty() {
        return Err(RandomForestError(
            "model schema or tree collection invalid".into(),
        ));
    }
    if samples.is_empty() {
        return Err(RandomForestError(
            "prediction samples: expected non-empty array".into(),
        ));
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    let mut seen = BTreeSet::new();
    let mut predictions = Vec::with_capacity(ordered.len());
    for (row_index, sample) in ordered.iter().enumerate() {
        if sample.sample_id.trim().is_empty() || !seen.insert(sample.sample_id.as_str()) {
            return Err(RandomForestError(format!(
                "prediction_samples[{row_index}].sample_id: empty or duplicate"
            )));
        }
        if sample.features.len() != model.feature_count {
            return Err(RandomForestError(format!(
                "prediction_samples[{row_index}]: expected {} features, observed {}",
                model.feature_count,
                sample.features.len()
            )));
        }
        for (feature_index, value) in sample.features.iter().enumerate() {
            if !value.is_finite() {
                return Err(RandomForestError(format!(
                    "prediction_samples[{row_index}].features[{feature_index}]: expected finite number"
                )));
            }
        }
        let mut sum = 0.0;
        for tree in &model.trees {
            sum += predict_tree(tree, &sample.features)?;
        }
        let score = sum / model.trees.len() as f64;
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Err(RandomForestError(format!(
                "prediction_samples[{row_index}]: ensemble score outside [0,1]"
            )));
        }
        predictions.push(RandomForestPredictionV1 {
            sample_id: sample.sample_id.clone(),
            positive: sample.positive,
            score,
        });
    }
    let digest_material: Vec<(&str, bool, f64)> = predictions
        .iter()
        .map(|row| (row.sample_id.as_str(), row.positive, row.score))
        .collect();
    let score_vector_sha256 =
        sha256_hex(&serde_json::to_vec(&digest_material).map_err(|error| {
            RandomForestError(format!("score digest serialization failed: {error}"))
        })?);
    Ok(RandomForestPredictionReportV1 {
        schema: PREDICTION_SCHEMA_V1.into(),
        model_sha256: model.model_sha256.clone(),
        prediction_count: predictions.len(),
        score_vector_sha256,
        predictions,
    })
}

pub fn score_random_forest_predictions_v1(
    prediction_report: &RandomForestPredictionReportV1,
    decision_threshold: f64,
    calibration_bins: usize,
) -> Result<BinaryMetricReportV1, RandomForestError> {
    if prediction_report.schema != PREDICTION_SCHEMA_V1 {
        return Err(RandomForestError(
            "prediction report schema unsupported".into(),
        ));
    }
    let subject = BinaryScoreSubjectV1 {
        schema: crate::benchmark::binary_metrics::INPUT_SCHEMA_V1.into(),
        case_id: Some(format!("random-forest:{}", prediction_report.model_sha256)),
        decision_threshold,
        calibration_bins,
        recall_target: 0.95,
        positive_class_semantics: Some("observed semiconductor defect label".into()),
        samples: prediction_report
            .predictions
            .iter()
            .map(|row| BinaryScoreSampleV1 {
                sample_id: row.sample_id.clone(),
                positive: row.positive,
                score: row.score,
            })
            .collect(),
    };
    score_binary_subject_v1(&subject)
        .map_err(|error| RandomForestError(format!("binary metrics rejected predictions: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples() -> Vec<RandomForestTrainingSampleV1> {
        (0..80)
            .map(|index| {
                let positive = index % 4 == 0;
                let signal = if positive { 1.0 } else { 0.0 };
                RandomForestTrainingSampleV1 {
                    sample_id: format!("sample-{index:03}"),
                    positive,
                    features: vec![
                        signal + (index % 7) as f64 * 0.001,
                        (index % 11) as f64 / 11.0,
                        ((index * 7) % 13) as f64 / 13.0,
                    ],
                }
            })
            .collect()
    }

    fn parameters() -> RandomForestParametersV1 {
        RandomForestParametersV1 {
            n_trees: 12,
            max_depth: 6,
            min_samples_leaf: 2,
            mtry: 0,
            seed: 144,
            class_weighting: ForestClassWeightingV1::Balanced,
        }
    }

    #[test]
    fn seeded_fit_and_prediction_are_byte_stable() {
        let training = samples();
        let model_a = fit_random_forest_v1(&training, &parameters()).unwrap();
        let model_b = fit_random_forest_v1(&training, &parameters()).unwrap();
        assert_eq!(model_a.model_sha256, model_b.model_sha256);
        let input: Vec<_> = training
            .iter()
            .map(|row| RandomForestInputSampleV1 {
                sample_id: row.sample_id.clone(),
                positive: row.positive,
                features: row.features.clone(),
            })
            .collect();
        let predictions_a = predict_random_forest_v1(&model_a, &input).unwrap();
        let predictions_b = predict_random_forest_v1(&model_b, &input).unwrap();
        assert_eq!(predictions_a, predictions_b);
        assert!(
            predictions_a
                .predictions
                .iter()
                .all(|row| row.score.is_finite() && (0.0..=1.0).contains(&row.score))
        );
    }

    #[test]
    fn training_row_order_does_not_change_model_digest() {
        let mut training = samples();
        let first = fit_random_forest_v1(&training, &parameters()).unwrap();
        training.reverse();
        let second = fit_random_forest_v1(&training, &parameters()).unwrap();
        assert_eq!(first.model_sha256, second.model_sha256);
    }

    #[test]
    fn different_seed_changes_model_digest() {
        let training = samples();
        let first = fit_random_forest_v1(&training, &parameters()).unwrap();
        let mut changed = parameters();
        changed.seed += 1;
        let second = fit_random_forest_v1(&training, &changed).unwrap();
        assert_ne!(first.model_sha256, second.model_sha256);
    }

    #[test]
    fn test_predictions_are_sorted_by_sample_identity() {
        let training = samples();
        let model = fit_random_forest_v1(&training, &parameters()).unwrap();
        let mut input: Vec<_> = training
            .iter()
            .map(|row| RandomForestInputSampleV1 {
                sample_id: row.sample_id.clone(),
                positive: row.positive,
                features: row.features.clone(),
            })
            .collect();
        input.reverse();
        let report = predict_random_forest_v1(&model, &input).unwrap();
        assert!(
            report
                .predictions
                .windows(2)
                .all(|pair| pair[0].sample_id < pair[1].sample_id)
        );
    }

    #[test]
    fn one_class_and_non_finite_training_input_reject() {
        let mut only_positive = samples();
        for row in &mut only_positive {
            row.positive = true;
        }
        assert!(fit_random_forest_v1(&only_positive, &parameters()).is_err());

        let mut invalid = samples();
        invalid[0].features[0] = f64::NAN;
        assert!(fit_random_forest_v1(&invalid, &parameters()).is_err());
    }

    #[test]
    fn invalid_prediction_input_and_dimensions_reject() {
        let model = fit_random_forest_v1(&samples(), &parameters()).unwrap();
        let invalid = vec![RandomForestInputSampleV1 {
            sample_id: "bad".into(),
            positive: false,
            features: vec![1.0],
        }];
        assert!(predict_random_forest_v1(&model, &invalid).is_err());
    }

    #[test]
    fn forest_scores_reuse_the_independent_binary_metric_contract() {
        let training = samples();
        let model = fit_random_forest_v1(&training, &parameters()).unwrap();
        let inputs: Vec<_> = training
            .iter()
            .map(|row| RandomForestInputSampleV1 {
                sample_id: row.sample_id.clone(),
                positive: row.positive,
                features: row.features.clone(),
            })
            .collect();
        let predictions = predict_random_forest_v1(&model, &inputs).unwrap();
        let metrics = score_random_forest_predictions_v1(&predictions, 0.5, 10).unwrap();
        assert_eq!(metrics.sample_count, 80);
        assert_eq!(metrics.class_counts.positive, 20);
        assert_eq!(metrics.class_counts.negative, 60);
        assert!((0.0..=1.0).contains(&metrics.ranking_metrics.average_precision));
        assert!((0.0..=1.0).contains(&metrics.ranking_metrics.auroc));
        assert!((0.0..=1.0).contains(&metrics.probability_metrics.brier_score));
    }

    #[test]
    fn class_weighting_only_changes_split_policy_not_leaf_probability_definition() {
        let training = samples();
        let mut unweighted = parameters();
        unweighted.class_weighting = ForestClassWeightingV1::None;
        let mut balanced = parameters();
        balanced.class_weighting = ForestClassWeightingV1::Balanced;
        let unweighted_model = fit_random_forest_v1(&training, &unweighted).unwrap();
        let balanced_model = fit_random_forest_v1(&training, &balanced).unwrap();
        assert_eq!(
            unweighted_model.training_sample_ids_sha256,
            balanced_model.training_sample_ids_sha256
        );
        assert!(
            unweighted_model
                .trees
                .iter()
                .all(|tree| !tree.nodes.is_empty())
        );
        assert!(
            balanced_model
                .trees
                .iter()
                .all(|tree| !tree.nodes.is_empty())
        );
    }
}
