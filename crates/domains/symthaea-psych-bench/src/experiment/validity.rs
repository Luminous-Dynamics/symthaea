// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed benchmark validity and mutation testing for architecture research.
//!
//! This module validates generated task datasets before any architecture score is
//! interpreted. It provides an executable symbolic oracle for the v1 `RuleExpr`
//! language, split/leakage checks, support-contract checks, deterministic dataset
//! identity, and adversarial benchmark mutations that the validator must detect.

use crate::experiment::{RuleExpr, TaskProgram};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const RULE_ORACLE_HASH_DOMAIN: &[u8] = b"symthaea.rule-oracle.hash/v1";
const EXAMPLE_FEATURE_HASH_DOMAIN: &[u8] = b"symthaea.example-features.hash/v1";
const DATASET_HASH_DOMAIN: &[u8] = b"symthaea.generated-task-dataset.hash/v1";

fn canonical_hash<T: Serialize>(domain: &[u8], value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&[0]);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn looks_like_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Deterministic digest of the executable symbolic oracle represented by a rule.
pub fn symbolic_oracle_digest(rule: &RuleExpr) -> Result<String, String> {
    canonical_hash(RULE_ORACLE_HASH_DOMAIN, rule)
}

/// Evaluate the v1 symbolic rule over an integer-valued feature assignment.
///
/// `Eq` and `Ne` interpret both operands as feature names. `Parity` reads one
/// named integer factor. Missing factors fail closed rather than defaulting.
pub fn evaluate_rule(rule: &RuleExpr, features: &BTreeMap<String, i64>) -> Result<bool, String> {
    let value = |name: &str| {
        features
            .get(name)
            .copied()
            .ok_or_else(|| format!("missing feature required by oracle: {name}"))
    };

    match rule {
        RuleExpr::Eq { left, right } => Ok(value(left)? == value(right)?),
        RuleExpr::Ne { left, right } => Ok(value(left)? != value(right)?),
        RuleExpr::Parity {
            factor,
            modulus,
            remainder,
        } => {
            if *modulus == 0 || *remainder >= *modulus {
                return Err("invalid parity rule reached oracle execution".into());
            }
            let modulus = i64::from(*modulus);
            let remainder = i64::from(*remainder);
            Ok(value(factor)?.rem_euclid(modulus) == remainder)
        }
        RuleExpr::Not { inner } => Ok(!evaluate_rule(inner, features)?),
        RuleExpr::And { terms } => {
            for term in terms {
                if !evaluate_rule(term, features)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        RuleExpr::Or { terms } => {
            for term in terms {
                if evaluate_rule(term, features)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        RuleExpr::Xor { left, right } => {
            Ok(evaluate_rule(left, features)? ^ evaluate_rule(right, features)?)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExampleRecord {
    pub example_id: String,
    pub features: BTreeMap<String, i64>,
    pub support_tags: Vec<String>,
    pub expected_label: bool,
}

impl ExampleRecord {
    pub fn validate_structure(&self) -> Result<(), String> {
        if self.example_id.trim().is_empty() {
            return Err("example id must be non-empty".into());
        }
        if self.features.is_empty() {
            return Err("example must contain at least one feature".into());
        }
        let mut tags = BTreeSet::new();
        for tag in &self.support_tags {
            let normalized = tag.trim();
            if normalized.is_empty() {
                return Err("support tag must be non-empty".into());
            }
            if !tags.insert(normalized.to_string()) {
                return Err("duplicate support tag on example".into());
            }
        }
        Ok(())
    }

    /// Feature-only identity used to detect train/evaluation leakage independent
    /// of labels, ids, or support annotations.
    pub fn feature_digest(&self) -> Result<String, String> {
        canonical_hash(EXAMPLE_FEATURE_HASH_DOMAIN, &self.features)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedTaskDataset {
    pub program_digest: String,
    pub train: Vec<ExampleRecord>,
    pub eval: Vec<ExampleRecord>,
}

impl GeneratedTaskDataset {
    /// Canonical set-style digest. Example ordering does not change dataset
    /// identity, but ids/features/labels/support tags do.
    pub fn digest(&self) -> Result<String, String> {
        fn canonical_examples(examples: &[ExampleRecord]) -> Vec<ExampleRecord> {
            let mut canonical = examples.to_vec();
            for example in &mut canonical {
                example.support_tags.sort();
            }
            canonical.sort_by(|left, right| left.example_id.cmp(&right.example_id));
            canonical
        }

        canonical_hash(
            DATASET_HASH_DOMAIN,
            &(
                self.program_digest.as_str(),
                canonical_examples(&self.train),
                canonical_examples(&self.eval),
            ),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkValidityPolicy {
    /// Feature keys forbidden because they reveal task/world/boundary identity.
    pub forbidden_feature_keys: BTreeSet<String>,
    /// Reject feature-identical examples across train and evaluation splits.
    pub require_feature_disjoint_splits: bool,
    /// Require each example to carry at least one support tag and require every
    /// support tag to be declared by the split's TaskProgram support.
    pub require_declared_support_tags: bool,
    /// Require positive and negative examples in both train and evaluation splits.
    pub require_both_classes_per_split: bool,
}

impl BenchmarkValidityPolicy {
    /// Conservative default for latent-context/task-free confirmatory tasks.
    pub fn task_free_strict() -> Self {
        Self {
            forbidden_feature_keys: [
                "__task_id",
                "task_id",
                "__world_id",
                "world_id",
                "boundary_marker",
                "time_to_switch",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            require_feature_disjoint_splits: true,
            require_declared_support_tags: true,
            require_both_classes_per_split: true,
        }
    }

    /// Explicit-context experiments may construct a narrower forbidden-key set,
    /// but doing so is an explicit policy decision rather than an implicit leak.
    pub fn with_forbidden_feature_keys(mut self, keys: impl IntoIterator<Item = String>) -> Self {
        self.forbidden_feature_keys = keys.into_iter().collect();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidityViolationKind {
    ProgramInvalid,
    ProgramDigestMismatch,
    OracleDigestMismatch,
    DatasetEmpty,
    ExampleInvalid,
    DuplicateExampleId,
    TrainEvalFeatureLeak,
    ForbiddenFeatureLeak,
    UndeclaredSupport,
    OracleMismatch,
    DeclaredClassCountMismatch,
    SplitClassDegeneracy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidityViolation {
    pub kind: ValidityViolationKind,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkValidityStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkValidityReport {
    pub status: BenchmarkValidityStatus,
    pub dataset_digest: Option<String>,
    pub train_examples: usize,
    pub eval_examples: usize,
    pub positives: usize,
    pub negatives: usize,
    pub violations: Vec<ValidityViolation>,
}

impl BenchmarkValidityReport {
    pub fn is_valid(&self) -> bool {
        self.status == BenchmarkValidityStatus::Valid
    }
}

fn push_violation(
    violations: &mut Vec<ValidityViolation>,
    kind: ValidityViolationKind,
    detail: impl Into<String>,
) {
    violations.push(ValidityViolation {
        kind,
        detail: detail.into(),
    });
}

/// Validate a generated benchmark before any architecture result can be emitted.
pub fn validate_generated_task(
    program: &TaskProgram,
    dataset: &GeneratedTaskDataset,
    policy: &BenchmarkValidityPolicy,
) -> BenchmarkValidityReport {
    let mut violations = Vec::new();

    if let Err(error) = program.validate() {
        push_violation(
            &mut violations,
            ValidityViolationKind::ProgramInvalid,
            error,
        );
    }

    let expected_program_digest = program.digest().map_err(|error| error.to_string());
    match expected_program_digest {
        Ok(digest) if dataset.program_digest != digest => push_violation(
            &mut violations,
            ValidityViolationKind::ProgramDigestMismatch,
            "dataset is not bound to the supplied TaskProgram",
        ),
        Err(error) => push_violation(
            &mut violations,
            ValidityViolationKind::ProgramInvalid,
            format!("program digest failed: {error}"),
        ),
        _ => {}
    }

    match symbolic_oracle_digest(&program.rule) {
        Ok(digest) if program.oracle_digest.to_ascii_lowercase() != digest => push_violation(
            &mut violations,
            ValidityViolationKind::OracleDigestMismatch,
            "TaskProgram oracle digest does not match its executable RuleExpr oracle",
        ),
        Err(error) => push_violation(
            &mut violations,
            ValidityViolationKind::ProgramInvalid,
            format!("symbolic oracle digest failed: {error}"),
        ),
        _ => {}
    }

    if dataset.train.is_empty() || dataset.eval.is_empty() {
        push_violation(
            &mut violations,
            ValidityViolationKind::DatasetEmpty,
            "training and evaluation splits must both be non-empty",
        );
    }

    let train_support: BTreeSet<String> = program
        .train_support
        .iter()
        .map(|tag| tag.trim().to_string())
        .collect();
    let eval_support: BTreeSet<String> = program
        .eval_support
        .iter()
        .map(|tag| tag.trim().to_string())
        .collect();
    let forbidden: BTreeSet<String> = policy
        .forbidden_feature_keys
        .iter()
        .map(|key| key.trim().to_ascii_lowercase())
        .collect();

    let mut ids = BTreeSet::new();
    let mut train_feature_digests = BTreeSet::new();
    let mut positives = 0usize;
    let mut negatives = 0usize;
    let mut train_positive = 0usize;
    let mut train_negative = 0usize;
    let mut eval_positive = 0usize;
    let mut eval_negative = 0usize;

    for (split_name, examples, declared_support) in [
        ("train", &dataset.train, &train_support),
        ("eval", &dataset.eval, &eval_support),
    ] {
        for example in examples {
            if let Err(error) = example.validate_structure() {
                push_violation(
                    &mut violations,
                    ValidityViolationKind::ExampleInvalid,
                    format!("{}:{}: {error}", split_name, example.example_id),
                );
            }

            if !ids.insert(example.example_id.clone()) {
                push_violation(
                    &mut violations,
                    ValidityViolationKind::DuplicateExampleId,
                    format!("duplicate example id: {}", example.example_id),
                );
            }

            for key in example.features.keys() {
                if forbidden.contains(&key.to_ascii_lowercase()) {
                    push_violation(
                        &mut violations,
                        ValidityViolationKind::ForbiddenFeatureLeak,
                        format!(
                            "{}:{} contains forbidden feature {key}",
                            split_name, example.example_id
                        ),
                    );
                }
            }

            if policy.require_declared_support_tags {
                if example.support_tags.is_empty() {
                    push_violation(
                        &mut violations,
                        ValidityViolationKind::UndeclaredSupport,
                        format!("{}:{} has no support provenance tag", split_name, example.example_id),
                    );
                }
                for tag in &example.support_tags {
                    let normalized = tag.trim().to_string();
                    if !declared_support.contains(&normalized) {
                        push_violation(
                            &mut violations,
                            ValidityViolationKind::UndeclaredSupport,
                            format!(
                                "{}:{} uses undeclared support tag {}",
                                split_name, example.example_id, normalized
                            ),
                        );
                    }
                }
            }

            match evaluate_rule(&program.rule, &example.features) {
                Ok(label) if label != example.expected_label => push_violation(
                    &mut violations,
                    ValidityViolationKind::OracleMismatch,
                    format!(
                        "{}:{} label disagrees with symbolic oracle",
                        split_name, example.example_id
                    ),
                ),
                Err(error) => push_violation(
                    &mut violations,
                    ValidityViolationKind::OracleMismatch,
                    format!("{}:{} oracle failed: {error}", split_name, example.example_id),
                ),
                _ => {}
            }

            if example.expected_label {
                positives += 1;
                if split_name == "train" {
                    train_positive += 1;
                } else {
                    eval_positive += 1;
                }
            } else {
                negatives += 1;
                if split_name == "train" {
                    train_negative += 1;
                } else {
                    eval_negative += 1;
                }
            }

            if split_name == "train" {
                if let Ok(digest) = example.feature_digest() {
                    train_feature_digests.insert(digest);
                }
            }
        }
    }

    if policy.require_feature_disjoint_splits {
        for example in &dataset.eval {
            if let Ok(digest) = example.feature_digest() {
                if train_feature_digests.contains(&digest) {
                    push_violation(
                        &mut violations,
                        ValidityViolationKind::TrainEvalFeatureLeak,
                        format!(
                            "evaluation example {} duplicates training features",
                            example.example_id
                        ),
                    );
                }
            }
        }
    }

    if positives != program.positive_examples || negatives != program.negative_examples {
        push_violation(
            &mut violations,
            ValidityViolationKind::DeclaredClassCountMismatch,
            format!(
                "observed class counts +{} / -{} do not match declared +{} / -{}",
                positives, negatives, program.positive_examples, program.negative_examples
            ),
        );
    }

    if policy.require_both_classes_per_split
        && (train_positive == 0 || train_negative == 0 || eval_positive == 0 || eval_negative == 0)
    {
        push_violation(
            &mut violations,
            ValidityViolationKind::SplitClassDegeneracy,
            "both train and evaluation splits must contain both classes",
        );
    }

    let dataset_digest = dataset.digest().ok();
    BenchmarkValidityReport {
        status: if violations.is_empty() {
            BenchmarkValidityStatus::Valid
        } else {
            BenchmarkValidityStatus::Invalid
        },
        dataset_digest,
        train_examples: dataset.train.len(),
        eval_examples: dataset.eval.len(),
        positives,
        negatives,
        violations,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMutation {
    FlipFirstEvalLabel,
    LeakFirstTrainExampleIntoEval,
    CorruptProgramDigest,
    InjectForbiddenTaskId,
    InjectUndeclaredEvalSupport,
    RemoveFirstEvalSupport,
}

/// Apply an intentionally invalid mutation used to test the validator itself.
pub fn apply_mutation(
    dataset: &GeneratedTaskDataset,
    mutation: BenchmarkMutation,
) -> Result<GeneratedTaskDataset, String> {
    let mut mutated = dataset.clone();
    match mutation {
        BenchmarkMutation::FlipFirstEvalLabel => {
            let example = mutated
                .eval
                .first_mut()
                .ok_or_else(|| "mutation requires a non-empty evaluation split".to_string())?;
            example.expected_label = !example.expected_label;
        }
        BenchmarkMutation::LeakFirstTrainExampleIntoEval => {
            let train = mutated
                .train
                .first()
                .cloned()
                .ok_or_else(|| "mutation requires a non-empty training split".to_string())?;
            let eval = mutated
                .eval
                .first_mut()
                .ok_or_else(|| "mutation requires a non-empty evaluation split".to_string())?;
            eval.features = train.features;
        }
        BenchmarkMutation::CorruptProgramDigest => {
            mutated.program_digest = "00".repeat(32);
        }
        BenchmarkMutation::InjectForbiddenTaskId => {
            let example = mutated
                .eval
                .first_mut()
                .ok_or_else(|| "mutation requires a non-empty evaluation split".to_string())?;
            example.features.insert("__task_id".into(), 1);
        }
        BenchmarkMutation::InjectUndeclaredEvalSupport => {
            let example = mutated
                .eval
                .first_mut()
                .ok_or_else(|| "mutation requires a non-empty evaluation split".to_string())?;
            example.support_tags.push("undeclared-support".into());
        }
        BenchmarkMutation::RemoveFirstEvalSupport => {
            let example = mutated
                .eval
                .first_mut()
                .ok_or_else(|| "mutation requires a non-empty evaluation split".to_string())?;
            example.support_tags.clear();
        }
    }
    Ok(mutated)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationDetection {
    pub mutation: BenchmarkMutation,
    pub detected: bool,
    pub violation_kinds: Vec<ValidityViolationKind>,
}

/// Run the standard adversarial benchmark mutations and require each to become invalid.
pub fn mutation_detection_suite(
    program: &TaskProgram,
    dataset: &GeneratedTaskDataset,
    policy: &BenchmarkValidityPolicy,
) -> Result<Vec<MutationDetection>, String> {
    let baseline = validate_generated_task(program, dataset, policy);
    if !baseline.is_valid() {
        return Err("mutation suite requires a valid baseline dataset".into());
    }

    let mutations = [
        BenchmarkMutation::FlipFirstEvalLabel,
        BenchmarkMutation::LeakFirstTrainExampleIntoEval,
        BenchmarkMutation::CorruptProgramDigest,
        BenchmarkMutation::InjectForbiddenTaskId,
        BenchmarkMutation::InjectUndeclaredEvalSupport,
        BenchmarkMutation::RemoveFirstEvalSupport,
    ];

    let mut detections = Vec::with_capacity(mutations.len());
    for mutation in mutations {
        let mutated = apply_mutation(dataset, mutation)?;
        let report = validate_generated_task(program, &mutated, policy);
        detections.push(MutationDetection {
            mutation,
            detected: !report.is_valid(),
            violation_kinds: report.violations.into_iter().map(|violation| violation.kind).collect(),
        });
    }
    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{
        ContextVisibility, RuleExpr, TASK_PROGRAM_SCHEMA_V1, TimingRegime,
    };

    fn example(id: &str, x: i64, support: &str) -> ExampleRecord {
        let mut features = BTreeMap::new();
        features.insert("x".into(), x);
        ExampleRecord {
            example_id: id.into(),
            features,
            support_tags: vec![support.into()],
            expected_label: x.rem_euclid(2) == 0,
        }
    }

    fn fixture() -> (TaskProgram, GeneratedTaskDataset, BenchmarkValidityPolicy) {
        let rule = RuleExpr::Parity {
            factor: "x".into(),
            modulus: 2,
            remainder: 0,
        };
        let oracle_digest = symbolic_oracle_digest(&rule).unwrap();
        let program = TaskProgram {
            schema: TASK_PROGRAM_SCHEMA_V1.into(),
            program_id: "parity-validity-001".into(),
            family: "relational".into(),
            rule,
            context_visibility: ContextVisibility::TaskFree,
            timing_regime: TimingRegime::Uniform,
            positive_examples: 4,
            negative_examples: 4,
            train_support: vec!["train-range".into()],
            eval_support: vec!["eval-range".into()],
            oracle_digest,
        };
        let program_digest = program.digest().unwrap();
        let dataset = GeneratedTaskDataset {
            program_digest,
            train: vec![
                example("train-0", 0, "train-range"),
                example("train-1", 1, "train-range"),
                example("train-2", 2, "train-range"),
                example("train-3", 3, "train-range"),
            ],
            eval: vec![
                example("eval-4", 4, "eval-range"),
                example("eval-5", 5, "eval-range"),
                example("eval-6", 6, "eval-range"),
                example("eval-7", 7, "eval-range"),
            ],
        };
        (program, dataset, BenchmarkValidityPolicy::task_free_strict())
    }

    #[test]
    fn symbolic_oracle_handles_nested_boolean_rules() {
        let rule = RuleExpr::And {
            terms: vec![
                RuleExpr::Parity {
                    factor: "x".into(),
                    modulus: 2,
                    remainder: 0,
                },
                RuleExpr::Ne {
                    left: "x".into(),
                    right: "y".into(),
                },
            ],
        };
        let mut features = BTreeMap::new();
        features.insert("x".into(), 4);
        features.insert("y".into(), 3);
        assert!(evaluate_rule(&rule, &features).unwrap());
        features.insert("y".into(), 4);
        assert!(!evaluate_rule(&rule, &features).unwrap());
    }

    #[test]
    fn valid_fixture_passes_and_digest_ignores_example_order() {
        let (program, dataset, policy) = fixture();
        let report = validate_generated_task(&program, &dataset, &policy);
        assert!(report.is_valid(), "violations={:?}", report.violations);

        let mut reordered = dataset.clone();
        reordered.train.reverse();
        reordered.eval.reverse();
        assert_eq!(dataset.digest().unwrap(), reordered.digest().unwrap());
    }

    #[test]
    fn oracle_digest_must_match_executable_rule() {
        let (mut program, dataset, policy) = fixture();
        program.oracle_digest = "ab".repeat(32);
        let report = validate_generated_task(&program, &dataset, &policy);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == ValidityViolationKind::OracleDigestMismatch
        }));
    }

    #[test]
    fn feature_identical_train_eval_example_is_leakage_even_with_new_id() {
        let (program, mut dataset, policy) = fixture();
        dataset.eval[0].features = dataset.train[0].features.clone();
        let report = validate_generated_task(&program, &dataset, &policy);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == ValidityViolationKind::TrainEvalFeatureLeak
        }));
    }

    #[test]
    fn mutation_suite_detects_every_standard_benchmark_corruption() {
        let (program, dataset, policy) = fixture();
        let detections = mutation_detection_suite(&program, &dataset, &policy).unwrap();
        assert_eq!(detections.len(), 6);
        assert!(detections.iter().all(|detection| detection.detected));
    }

    #[test]
    fn forbidden_task_identity_feature_fails_closed() {
        let (program, mut dataset, policy) = fixture();
        dataset.eval[0].features.insert("world_id".into(), 3);
        let report = validate_generated_task(&program, &dataset, &policy);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == ValidityViolationKind::ForbiddenFeatureLeak
        }));
    }

    #[test]
    fn missing_support_provenance_fails_closed() {
        let (program, mut dataset, policy) = fixture();
        dataset.eval[0].support_tags.clear();
        let report = validate_generated_task(&program, &dataset, &policy);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == ValidityViolationKind::UndeclaredSupport
        }));
    }

    #[test]
    fn class_count_contract_catches_missing_or_duplicated_examples() {
        let (program, mut dataset, policy) = fixture();
        dataset.eval.pop();
        let report = validate_generated_task(&program, &dataset, &policy);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == ValidityViolationKind::DeclaredClassCountMismatch
        }));
    }
}
