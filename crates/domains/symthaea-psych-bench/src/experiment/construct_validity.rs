// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Adversarial construct-validity controls for generated architecture benchmarks.
//!
//! A benchmark that is structurally valid can still be scientifically weak if a
//! trivial train-only shortcut solves its evaluation split. This module attacks
//! generated tasks with deliberately simple controls before any architecture
//! result is interpreted.
//!
//! The controls consume learner-visible `features` only. Example ids and
//! `support_tags` are metadata and are never supplied to fitted shortcut models.

use crate::experiment::TaskProgram;
use crate::experiment_validity::{
    evaluate_rule, validate_generated_task, BenchmarkValidityPolicy, BenchmarkValidityReport,
    ExampleRecord, GeneratedTaskDataset,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CONSTRUCT_VALIDITY_SCHEMA_V1: &str = "symthaea.construct-validity/v1";

const CHANCE_HASH_DOMAIN: &[u8] = b"symthaea.construct-validity.chance/v1";
const SHUFFLE_HASH_DOMAIN: &[u8] = b"symthaea.construct-validity.shuffle/v1";

fn accuracy(predictions: &[bool], examples: &[ExampleRecord]) -> Result<f64, String> {
    if predictions.len() != examples.len() || examples.is_empty() {
        return Err("predictions must match a non-empty evaluation split".into());
    }
    let correct = predictions
        .iter()
        .zip(examples)
        .filter(|(prediction, example)| **prediction == example.expected_label)
        .count();
    Ok(correct as f64 / examples.len() as f64)
}

fn majority_label(labels: &[bool]) -> Result<bool, String> {
    if labels.is_empty() {
        return Err("majority predictor requires training labels".into());
    }
    let positives = labels.iter().filter(|label| **label).count();
    // Deterministic negative-class tie break. The choice is arbitrary but frozen.
    Ok(positives * 2 > labels.len())
}

fn labels(examples: &[ExampleRecord]) -> Vec<bool> {
    examples.iter().map(|example| example.expected_label).collect()
}

fn learner_feature_keys(examples: &[ExampleRecord]) -> BTreeSet<String> {
    examples
        .iter()
        .flat_map(|example| example.features.keys().cloned())
        .collect()
}

fn feature_distance(left: &ExampleRecord, right: &ExampleRecord) -> usize {
    let keys: BTreeSet<&String> = left
        .features
        .keys()
        .chain(right.features.keys())
        .collect();
    keys.into_iter()
        .filter(|key| left.features.get(*key) != right.features.get(*key))
        .count()
}

#[derive(Debug, Clone)]
struct SingleFeatureModel {
    feature: String,
    value_labels: BTreeMap<i64, bool>,
    fallback: bool,
    train_accuracy: f64,
}

fn fit_single_feature_model(
    train: &[ExampleRecord],
    train_labels: &[bool],
) -> Result<SingleFeatureModel, String> {
    if train.is_empty() || train.len() != train_labels.len() {
        return Err("single-feature model requires paired non-empty training data".into());
    }
    let fallback = majority_label(train_labels)?;
    let keys = learner_feature_keys(train);
    if keys.is_empty() {
        return Err("single-feature model requires learner-visible features".into());
    }

    let mut best: Option<SingleFeatureModel> = None;
    for feature in keys {
        let mut counts: BTreeMap<i64, (usize, usize)> = BTreeMap::new();
        for (example, label) in train.iter().zip(train_labels) {
            if let Some(value) = example.features.get(&feature) {
                let entry = counts.entry(*value).or_default();
                if *label {
                    entry.0 += 1;
                } else {
                    entry.1 += 1;
                }
            }
        }
        let value_labels: BTreeMap<i64, bool> = counts
            .into_iter()
            .map(|(value, (positive, negative))| {
                let label = if positive == negative {
                    fallback
                } else {
                    positive > negative
                };
                (value, label)
            })
            .collect();
        let predictions: Vec<bool> = train
            .iter()
            .map(|example| {
                example
                    .features
                    .get(&feature)
                    .and_then(|value| value_labels.get(value))
                    .copied()
                    .unwrap_or(fallback)
            })
            .collect();
        let correct = predictions
            .iter()
            .zip(train_labels)
            .filter(|(prediction, label)| **prediction == **label)
            .count();
        let train_accuracy = correct as f64 / train.len() as f64;

        let candidate = SingleFeatureModel {
            feature,
            value_labels,
            fallback,
            train_accuracy,
        };
        let replace = match &best {
            None => true,
            Some(current) => {
                candidate.train_accuracy > current.train_accuracy
                    || (candidate.train_accuracy == current.train_accuracy
                        && candidate.feature < current.feature)
            }
        };
        if replace {
            best = Some(candidate);
        }
    }

    best.ok_or_else(|| "failed to fit single-feature model".into())
}

fn predict_single_feature(model: &SingleFeatureModel, example: &ExampleRecord) -> bool {
    example
        .features
        .get(&model.feature)
        .and_then(|value| model.value_labels.get(value))
        .copied()
        .unwrap_or(model.fallback)
}

fn exact_lookup_predictions(
    train: &[ExampleRecord],
    train_labels: &[bool],
    eval: &[ExampleRecord],
) -> Result<Vec<bool>, String> {
    if train.is_empty() || train.len() != train_labels.len() {
        return Err("exact lookup requires paired non-empty training data".into());
    }
    let fallback = majority_label(train_labels)?;
    let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for (example, label) in train.iter().zip(train_labels) {
        let digest = example.feature_digest()?;
        let entry = counts.entry(digest).or_default();
        if *label {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }
    Ok(eval
        .iter()
        .map(|example| {
            example
                .feature_digest()
                .ok()
                .and_then(|digest| counts.get(&digest).copied())
                .map(|(positive, negative)| {
                    if positive == negative {
                        fallback
                    } else {
                        positive > negative
                    }
                })
                .unwrap_or(fallback)
        })
        .collect())
}

fn nearest_neighbor_predictions(
    train: &[ExampleRecord],
    train_labels: &[bool],
    eval: &[ExampleRecord],
) -> Result<Vec<bool>, String> {
    if train.is_empty() || train.len() != train_labels.len() {
        return Err("nearest neighbor requires paired non-empty training data".into());
    }
    let fallback = majority_label(train_labels)?;
    let mut predictions = Vec::with_capacity(eval.len());
    for target in eval {
        let mut best_distance = usize::MAX;
        let mut positive = 0usize;
        let mut negative = 0usize;
        for (candidate, label) in train.iter().zip(train_labels) {
            let distance = feature_distance(candidate, target);
            if distance < best_distance {
                best_distance = distance;
                positive = 0;
                negative = 0;
            }
            if distance == best_distance {
                if *label {
                    positive += 1;
                } else {
                    negative += 1;
                }
            }
        }
        predictions.push(if positive == negative {
            fallback
        } else {
            positive > negative
        });
    }
    Ok(predictions)
}

fn deterministic_chance_prediction(seed: u64, example_id: &str) -> bool {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CHANCE_HASH_DOMAIN);
    hasher.update(&[0]);
    hasher.update(&seed.to_le_bytes());
    hasher.update(&[0]);
    hasher.update(example_id.as_bytes());
    hasher.finalize().as_bytes()[0] & 1 == 1
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn shuffle_seed(seed: u64) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SHUFFLE_HASH_DOMAIN);
    hasher.update(&[0]);
    hasher.update(&seed.to_le_bytes());
    let bytes = hasher.finalize();
    u64::from_le_bytes(bytes.as_bytes()[..8].try_into().expect("eight bytes"))
}

fn shuffled_labels(original: &[bool], seed: u64) -> Result<Vec<bool>, String> {
    if original.len() < 2 {
        return Err("label shuffle requires at least two training examples".into());
    }
    let mut shuffled = original.to_vec();
    let mut state = shuffle_seed(seed);
    for index in (1..shuffled.len()).rev() {
        let swap = (splitmix64(&mut state) % (index as u64 + 1)) as usize;
        shuffled.swap(index, swap);
    }
    Ok(shuffled)
}

/// 95% Wilson upper bound for a fair Bernoulli classifier's realized accuracy.
/// This is used only as a benchmark-resolution check, not as a hypothesis test.
fn chance_accuracy_upper95(n: usize) -> Result<f64, String> {
    if n == 0 {
        return Err("chance interval requires a non-empty evaluation split".into());
    }
    let p = 0.5;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let n = n as f64;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    Ok((center + half).min(1.0))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortcutControlKind {
    Majority,
    SingleFeatureMarginal,
    ExactLookup,
    NearestNeighbor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShortcutControlScore {
    pub kind: ShortcutControlKind,
    pub train_accuracy: Option<f64>,
    pub eval_accuracy: f64,
    pub selected_feature: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShuffledRelationSummary {
    pub seeds: Vec<u64>,
    pub single_feature_mean_accuracy: f64,
    pub single_feature_max_accuracy: f64,
    pub nearest_neighbor_mean_accuracy: f64,
    pub nearest_neighbor_max_accuracy: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstructValidityPolicy {
    /// Structural/oracle policy inherited from SYM-ARCH-002A3.
    pub structural_policy: BenchmarkValidityPolicy,
    /// Maximum allowed evaluation accuracy for any train-only shortcut. Reaching
    /// this ceiling is sufficient to make the benchmark inconclusive.
    pub shortcut_accuracy_ceiling: f64,
    /// Maximum allowed mean evaluation accuracy across shuffled-label controls.
    pub shuffled_mean_accuracy_ceiling: f64,
    /// Minimum required executable-oracle accuracy.
    pub oracle_accuracy_floor: f64,
    /// Deterministic seed for the realized chance sanity control.
    pub chance_seed: u64,
    /// Frozen seeds for prevalence-preserving shuffled-label controls.
    pub shuffle_seeds: Vec<u64>,
}

impl ConstructValidityPolicy {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("shortcut_accuracy_ceiling", self.shortcut_accuracy_ceiling),
            (
                "shuffled_mean_accuracy_ceiling",
                self.shuffled_mean_accuracy_ceiling,
            ),
            ("oracle_accuracy_floor", self.oracle_accuracy_floor),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must be a finite probability"));
            }
        }
        if self.shortcut_accuracy_ceiling <= 0.5 {
            return Err("shortcut ceiling must be above fair-chance accuracy".into());
        }
        if self.shuffled_mean_accuracy_ceiling <= 0.5 {
            return Err("shuffled-label ceiling must be above fair-chance accuracy".into());
        }
        if self.oracle_accuracy_floor < 0.5 {
            return Err("oracle accuracy floor must be at least chance".into());
        }
        if self.shuffle_seeds.len() < 4 {
            return Err("at least four frozen shuffle seeds are required".into());
        }
        let unique: BTreeSet<u64> = self.shuffle_seeds.iter().copied().collect();
        if unique.len() != self.shuffle_seeds.len() {
            return Err("shuffle seeds must be unique".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstructValidityStatus {
    Passed,
    InconclusiveBenchmark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstructViolationKind {
    StructuralValidityFailed,
    InsufficientEvaluationResolution,
    OracleControlFailed,
    ShortcutControlTooStrong,
    ShuffledRelationControlTooStrong,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstructViolation {
    pub kind: ConstructViolationKind,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstructValidityReport {
    pub schema: String,
    pub status: ConstructValidityStatus,
    pub dataset_digest: Option<String>,
    pub structural_report: BenchmarkValidityReport,
    pub eval_examples: usize,
    pub chance_expected_accuracy: f64,
    pub chance_observed_accuracy: Option<f64>,
    pub chance_accuracy_upper95: Option<f64>,
    pub oracle_accuracy: Option<f64>,
    pub shortcut_scores: Vec<ShortcutControlScore>,
    pub shuffled_relation: Option<ShuffledRelationSummary>,
    pub violations: Vec<ConstructViolation>,
}

impl ConstructValidityReport {
    pub fn passed(&self) -> bool {
        self.status == ConstructValidityStatus::Passed
    }
}

fn push_violation(
    violations: &mut Vec<ConstructViolation>,
    kind: ConstructViolationKind,
    detail: impl Into<String>,
) {
    violations.push(ConstructViolation {
        kind,
        detail: detail.into(),
    });
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Run adversarial construct-validity controls against a generated task.
///
/// Model selection for the single-feature control uses training accuracy only.
/// Evaluation labels are used solely for final scoring. Exact lookup and nearest
/// neighbor receive only training features/labels plus evaluation features.
/// Support tags and example ids are excluded from fitted shortcut inputs.
pub fn run_construct_validity(
    program: &TaskProgram,
    dataset: &GeneratedTaskDataset,
    policy: &ConstructValidityPolicy,
) -> Result<ConstructValidityReport, String> {
    policy.validate()?;
    let structural_report = validate_generated_task(program, dataset, &policy.structural_policy);
    let mut violations = Vec::new();

    if !structural_report.is_valid() {
        push_violation(
            &mut violations,
            ConstructViolationKind::StructuralValidityFailed,
            "SYM-ARCH-002A3 structural/oracle validity must pass before shortcut controls count",
        );
        return Ok(ConstructValidityReport {
            schema: CONSTRUCT_VALIDITY_SCHEMA_V1.to_string(),
            status: ConstructValidityStatus::InconclusiveBenchmark,
            dataset_digest: structural_report.dataset_digest.clone(),
            structural_report,
            eval_examples: dataset.eval.len(),
            chance_expected_accuracy: 0.5,
            chance_observed_accuracy: None,
            chance_accuracy_upper95: None,
            oracle_accuracy: None,
            shortcut_scores: Vec::new(),
            shuffled_relation: None,
            violations,
        });
    }

    let dataset_digest = Some(dataset.digest()?);
    let eval_examples = dataset.eval.len();
    let chance_upper = chance_accuracy_upper95(eval_examples)?;
    let resolution_ceiling = policy
        .shortcut_accuracy_ceiling
        .min(policy.shuffled_mean_accuracy_ceiling);
    if chance_upper >= resolution_ceiling {
        push_violation(
            &mut violations,
            ConstructViolationKind::InsufficientEvaluationResolution,
            format!(
                "fair-chance 95% upper accuracy {chance_upper:.4} reaches/exceeds the tighter preregistered shortcut ceiling {resolution_ceiling:.4}; increase evaluation support"
            ),
        );
    }

    let chance_predictions: Vec<bool> = dataset
        .eval
        .iter()
        .map(|example| deterministic_chance_prediction(policy.chance_seed, &example.example_id))
        .collect();
    let chance_observed_accuracy = accuracy(&chance_predictions, &dataset.eval)?;

    let oracle_predictions: Vec<bool> = dataset
        .eval
        .iter()
        .map(|example| evaluate_rule(&program.rule, &example.features))
        .collect::<Result<Vec<_>, _>>()?;
    let oracle_accuracy = accuracy(&oracle_predictions, &dataset.eval)?;
    if oracle_accuracy < policy.oracle_accuracy_floor {
        push_violation(
            &mut violations,
            ConstructViolationKind::OracleControlFailed,
            format!(
                "symbolic oracle accuracy {oracle_accuracy:.4} is below floor {:.4}",
                policy.oracle_accuracy_floor
            ),
        );
    }

    let train_labels = labels(&dataset.train);
    let train_majority = majority_label(&train_labels)?;
    let majority_predictions = vec![train_majority; dataset.eval.len()];
    let majority_train_accuracy = train_labels
        .iter()
        .filter(|label| **label == train_majority)
        .count() as f64
        / train_labels.len() as f64;
    let majority_score = ShortcutControlScore {
        kind: ShortcutControlKind::Majority,
        train_accuracy: Some(majority_train_accuracy),
        eval_accuracy: accuracy(&majority_predictions, &dataset.eval)?,
        selected_feature: None,
    };

    let single_feature_model = fit_single_feature_model(&dataset.train, &train_labels)?;
    let single_feature_predictions: Vec<bool> = dataset
        .eval
        .iter()
        .map(|example| predict_single_feature(&single_feature_model, example))
        .collect();
    let single_feature_score = ShortcutControlScore {
        kind: ShortcutControlKind::SingleFeatureMarginal,
        train_accuracy: Some(single_feature_model.train_accuracy),
        eval_accuracy: accuracy(&single_feature_predictions, &dataset.eval)?,
        selected_feature: Some(single_feature_model.feature.clone()),
    };

    let exact_lookup_predictions =
        exact_lookup_predictions(&dataset.train, &train_labels, &dataset.eval)?;
    let exact_lookup_score = ShortcutControlScore {
        kind: ShortcutControlKind::ExactLookup,
        train_accuracy: Some(1.0),
        eval_accuracy: accuracy(&exact_lookup_predictions, &dataset.eval)?,
        selected_feature: None,
    };

    let nearest_predictions =
        nearest_neighbor_predictions(&dataset.train, &train_labels, &dataset.eval)?;
    let nearest_score = ShortcutControlScore {
        kind: ShortcutControlKind::NearestNeighbor,
        train_accuracy: Some(1.0),
        eval_accuracy: accuracy(&nearest_predictions, &dataset.eval)?,
        selected_feature: None,
    };

    let shortcut_scores = vec![
        majority_score,
        single_feature_score,
        exact_lookup_score,
        nearest_score,
    ];
    for score in &shortcut_scores {
        if score.eval_accuracy >= policy.shortcut_accuracy_ceiling {
            push_violation(
                &mut violations,
                ConstructViolationKind::ShortcutControlTooStrong,
                format!(
                    "{:?} evaluation accuracy {:.4} reaches/exceeds ceiling {:.4}",
                    score.kind, score.eval_accuracy, policy.shortcut_accuracy_ceiling
                ),
            );
        }
    }

    let mut shuffled_single_feature = Vec::with_capacity(policy.shuffle_seeds.len());
    let mut shuffled_nearest = Vec::with_capacity(policy.shuffle_seeds.len());
    for &seed in &policy.shuffle_seeds {
        let permuted = shuffled_labels(&train_labels, seed)?;
        let model = fit_single_feature_model(&dataset.train, &permuted)?;
        let predictions: Vec<bool> = dataset
            .eval
            .iter()
            .map(|example| predict_single_feature(&model, example))
            .collect();
        shuffled_single_feature.push(accuracy(&predictions, &dataset.eval)?);

        let predictions = nearest_neighbor_predictions(&dataset.train, &permuted, &dataset.eval)?;
        shuffled_nearest.push(accuracy(&predictions, &dataset.eval)?);
    }
    let single_feature_mean_accuracy = mean(&shuffled_single_feature);
    let nearest_neighbor_mean_accuracy = mean(&shuffled_nearest);
    let shuffled_relation = ShuffledRelationSummary {
        seeds: policy.shuffle_seeds.clone(),
        single_feature_mean_accuracy,
        single_feature_max_accuracy: shuffled_single_feature
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        nearest_neighbor_mean_accuracy,
        nearest_neighbor_max_accuracy: shuffled_nearest
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    };

    for (name, value) in [
        ("single_feature", single_feature_mean_accuracy),
        ("nearest_neighbor", nearest_neighbor_mean_accuracy),
    ] {
        if value >= policy.shuffled_mean_accuracy_ceiling {
            push_violation(
                &mut violations,
                ConstructViolationKind::ShuffledRelationControlTooStrong,
                format!(
                    "shuffled-label {name} mean accuracy {value:.4} reaches/exceeds ceiling {:.4}",
                    policy.shuffled_mean_accuracy_ceiling
                ),
            );
        }
    }

    Ok(ConstructValidityReport {
        schema: CONSTRUCT_VALIDITY_SCHEMA_V1.to_string(),
        status: if violations.is_empty() {
            ConstructValidityStatus::Passed
        } else {
            ConstructValidityStatus::InconclusiveBenchmark
        },
        dataset_digest,
        structural_report,
        eval_examples,
        chance_expected_accuracy: 0.5,
        chance_observed_accuracy: Some(chance_observed_accuracy),
        chance_accuracy_upper95: Some(chance_upper),
        oracle_accuracy: Some(oracle_accuracy),
        shortcut_scores,
        shuffled_relation: Some(shuffled_relation),
        violations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{
        ContextVisibility, RuleExpr, TaskProgram, TimingRegime, TASK_PROGRAM_SCHEMA_V1,
    };
    use crate::experiment_validity::symbolic_oracle_digest;

    fn xor_rule() -> RuleExpr {
        RuleExpr::Xor {
            left: Box::new(RuleExpr::Parity {
                factor: "a".into(),
                modulus: 2,
                remainder: 0,
            }),
            right: Box::new(RuleExpr::Parity {
                factor: "b".into(),
                modulus: 2,
                remainder: 0,
            }),
        }
    }

    fn example(id: &str, a: i64, b: i64, split: &str, rule: &RuleExpr) -> ExampleRecord {
        let features = BTreeMap::from([("a".to_string(), a), ("b".to_string(), b)]);
        let expected_label = evaluate_rule(rule, &features).unwrap();
        ExampleRecord {
            example_id: id.into(),
            features,
            support_tags: vec![split.into()],
            expected_label,
        }
    }

    fn clean_dataset(eval_count: usize) -> (TaskProgram, GeneratedTaskDataset) {
        let rule = xor_rule();
        let train_pairs = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
        ];
        let eval_pairs = [
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
        ];
        let train: Vec<_> = train_pairs
            .into_iter()
            .enumerate()
            .map(|(index, (a, b))| example(&format!("train-{index}"), a, b, "train", &rule))
            .collect();
        let eval: Vec<_> = eval_pairs
            .into_iter()
            .take(eval_count)
            .enumerate()
            .map(|(index, (a, b))| example(&format!("eval-{index}"), a, b, "eval", &rule))
            .collect();
        let positive_examples = train
            .iter()
            .chain(&eval)
            .filter(|example| example.expected_label)
            .count();
        let negative_examples = train.len() + eval.len() - positive_examples;
        let mut program = TaskProgram {
            schema: TASK_PROGRAM_SCHEMA_V1.into(),
            program_id: "construct-validity-xor".into(),
            family: "unit-test".into(),
            rule,
            context_visibility: ContextVisibility::TaskFree,
            timing_regime: TimingRegime::Uniform,
            positive_examples,
            negative_examples,
            train_support: vec!["train".into()],
            eval_support: vec!["eval".into()],
            oracle_digest: String::new(),
        };
        program.oracle_digest = symbolic_oracle_digest(&program.rule).unwrap();
        let dataset = GeneratedTaskDataset {
            program_digest: program.digest().unwrap(),
            train,
            eval,
        };
        (program, dataset)
    }

    fn policy() -> ConstructValidityPolicy {
        ConstructValidityPolicy {
            structural_policy: BenchmarkValidityPolicy::task_free_strict(),
            shortcut_accuracy_ceiling: 0.80,
            shuffled_mean_accuracy_ceiling: 0.80,
            oracle_accuracy_floor: 1.0,
            chance_seed: 17,
            shuffle_seeds: vec![1, 2, 3, 4, 5, 7, 9, 11],
        }
    }

    #[test]
    fn clean_xor_benchmark_passes_shortcut_gate() {
        let (program, dataset) = clean_dataset(8);
        let report = run_construct_validity(&program, &dataset, &policy()).unwrap();
        assert!(report.passed(), "violations: {:?}", report.violations);
        assert_eq!(report.oracle_accuracy, Some(1.0));
        assert_eq!(report.eval_examples, 8);
        assert!(report.chance_accuracy_upper95.unwrap() < 0.80);
        assert!(report
            .shortcut_scores
            .iter()
            .all(|score| score.eval_accuracy < 0.80));
    }

    #[test]
    fn single_feature_label_channel_is_rejected() {
        let (program, mut dataset) = clean_dataset(8);
        for example in dataset.train.iter_mut().chain(&mut dataset.eval) {
            example.features.insert(
                "shortcut".into(),
                if example.expected_label { 1 } else { 0 },
            );
        }
        let report = run_construct_validity(&program, &dataset, &policy()).unwrap();
        assert_eq!(report.status, ConstructValidityStatus::InconclusiveBenchmark);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == ConstructViolationKind::ShortcutControlTooStrong
        }));
        let marginal = report
            .shortcut_scores
            .iter()
            .find(|score| score.kind == ShortcutControlKind::SingleFeatureMarginal)
            .unwrap();
        assert_eq!(marginal.selected_feature.as_deref(), Some("shortcut"));
        assert_eq!(marginal.eval_accuracy, 1.0);
    }

    #[test]
    fn four_item_eval_is_too_coarse_for_eighty_percent_ceiling() {
        let (program, dataset) = clean_dataset(4);
        let report = run_construct_validity(&program, &dataset, &policy()).unwrap();
        assert_eq!(report.status, ConstructValidityStatus::InconclusiveBenchmark);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == ConstructViolationKind::InsufficientEvaluationResolution
        }));
        assert!(report.chance_accuracy_upper95.unwrap() >= 0.80);
    }

    #[test]
    fn support_metadata_never_enters_shortcut_models() {
        let (mut program, dataset) = clean_dataset(8);
        let baseline = run_construct_validity(&program, &dataset, &policy()).unwrap();

        program.train_support = vec!["train".into(), "positive".into(), "negative".into()];
        program.eval_support = vec!["eval".into(), "positive".into(), "negative".into()];
        let mut tagged = dataset.clone();
        for example in tagged.train.iter_mut().chain(&mut tagged.eval) {
            example.support_tags.push(if example.expected_label {
                "positive".into()
            } else {
                "negative".into()
            });
        }
        tagged.program_digest = program.digest().unwrap();
        let retagged = run_construct_validity(&program, &tagged, &policy()).unwrap();

        assert_eq!(baseline.shortcut_scores, retagged.shortcut_scores);
        assert_eq!(baseline.oracle_accuracy, retagged.oracle_accuracy);
    }

    #[test]
    fn shuffled_labels_preserve_prevalence() {
        let (_, dataset) = clean_dataset(8);
        let original = labels(&dataset.train);
        let positives = original.iter().filter(|label| **label).count();
        for seed in [1, 2, 3, 4, 5] {
            let shuffled = shuffled_labels(&original, seed).unwrap();
            assert_eq!(
                shuffled.iter().filter(|label| **label).count(),
                positives
            );
        }
    }

    #[test]
    fn policy_rejects_weak_or_duplicate_shuffle_contracts() {
        let mut invalid = policy();
        invalid.shuffle_seeds = vec![1, 2, 3];
        assert!(invalid.validate().is_err());

        invalid = policy();
        invalid.shuffle_seeds = vec![1, 2, 3, 3];
        assert!(invalid.validate().is_err());

        invalid = policy();
        invalid.shortcut_accuracy_ceiling = 0.5;
        assert!(invalid.validate().is_err());
    }
}
