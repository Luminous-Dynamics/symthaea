// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Differential/reference validation for generated architecture benchmarks.
//!
//! A structurally valid benchmark can still be wrong if its generator/runtime
//! and oracle share the same mistaken assumption. This module introduces a
//! second, independently implemented RuleExpr evaluator plus an explicit finite
//! factor domain and optional reference train/evaluation partition.

use crate::experiment::{RuleExpr, TaskProgram};
use crate::experiment_validity::{
    evaluate_rule, validate_generated_task, BenchmarkValidityPolicy, BenchmarkValidityReport,
    ExampleRecord, GeneratedTaskDataset,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DIFFERENTIAL_VALIDITY_SCHEMA_V1: &str = "symthaea.differential-validity/v1";
const ASSIGNMENT_HASH_DOMAIN: &[u8] = b"symthaea.reference-assignment.hash/v1";
const TRUTH_TABLE_HASH_DOMAIN: &[u8] = b"symthaea.reference-truth-table.hash/v1";

fn canonical_hash<T: Serialize>(domain: &[u8], value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&[0]);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn assignment_digest(features: &BTreeMap<String, i64>) -> Result<String, String> {
    canonical_hash(ASSIGNMENT_HASH_DOMAIN, features)
}

/// Independent reference semantics for RuleExpr.
///
/// This intentionally does not call A3's `evaluate_rule`. Differential agreement
/// therefore exercises two separate implementations of the rule semantics.
pub fn reference_evaluate_rule(
    rule: &RuleExpr,
    features: &BTreeMap<String, i64>,
) -> Result<bool, String> {
    fn read(features: &BTreeMap<String, i64>, name: &str) -> Result<i64, String> {
        features
            .get(name)
            .copied()
            .ok_or_else(|| format!("reference evaluator missing feature: {name}"))
    }

    match rule {
        RuleExpr::Eq { left, right } => Ok(read(features, left)? == read(features, right)?),
        RuleExpr::Ne { left, right } => Ok(read(features, left)? != read(features, right)?),
        RuleExpr::Parity {
            factor,
            modulus,
            remainder,
        } => {
            if *modulus == 0 || *remainder >= *modulus {
                return Err("reference evaluator received invalid parity rule".into());
            }
            Ok(read(features, factor)?.rem_euclid(i64::from(*modulus))
                == i64::from(*remainder))
        }
        RuleExpr::Not { inner } => Ok(!reference_evaluate_rule(inner, features)?),
        RuleExpr::And { terms } => {
            if terms.len() < 2 {
                return Err("reference evaluator requires >=2 AND terms".into());
            }
            for term in terms {
                if !reference_evaluate_rule(term, features)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        RuleExpr::Or { terms } => {
            if terms.len() < 2 {
                return Err("reference evaluator requires >=2 OR terms".into());
            }
            for term in terms {
                if reference_evaluate_rule(term, features)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        RuleExpr::Xor { left, right } => {
            let left = reference_evaluate_rule(left, features)?;
            let right = reference_evaluate_rule(right, features)?;
            Ok((left || right) && !(left && right))
        }
    }
}

fn referenced_features(rule: &RuleExpr, out: &mut BTreeSet<String>) {
    match rule {
        RuleExpr::Eq { left, right } | RuleExpr::Ne { left, right } => {
            out.insert(left.clone());
            out.insert(right.clone());
        }
        RuleExpr::Parity { factor, .. } => {
            out.insert(factor.clone());
        }
        RuleExpr::Not { inner } => referenced_features(inner, out),
        RuleExpr::And { terms } | RuleExpr::Or { terms } => {
            for term in terms {
                referenced_features(term, out);
            }
        }
        RuleExpr::Xor { left, right } => {
            referenced_features(left, out);
            referenced_features(right, out);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferencePartition {
    pub train_assignments: Vec<BTreeMap<String, i64>>,
    pub eval_assignments: Vec<BTreeMap<String, i64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DifferentialValidityPolicy {
    pub structural_policy: BenchmarkValidityPolicy,
    /// Complete finite domain for every factor governed by the reference check.
    pub factor_domains: BTreeMap<String, Vec<i64>>,
    /// Fail before enumeration if the Cartesian product exceeds this cap.
    pub max_reference_assignments: usize,
    /// If true, learner-visible generated features must be exactly the declared
    /// reference factors. If false, extra nuisance features are allowed but are
    /// projected out of reference truth/coverage/partition identity.
    pub require_exact_feature_schema: bool,
    /// Minimum fraction of the finite reference universe represented by unique
    /// projected assignments across train + eval.
    pub minimum_coverage_fraction: f64,
    /// Optional independent source of truth for exact train/eval membership.
    pub reference_partition: Option<ReferencePartition>,
}

impl DifferentialValidityPolicy {
    pub fn validate(&self, rule: &RuleExpr) -> Result<usize, String> {
        if self.factor_domains.is_empty() {
            return Err("reference factor domain must be non-empty".into());
        }
        if self.max_reference_assignments == 0 {
            return Err("max_reference_assignments must be positive".into());
        }
        if !self.minimum_coverage_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_coverage_fraction)
        {
            return Err("minimum coverage must be a finite probability".into());
        }

        let mut product = 1usize;
        for (factor, values) in &self.factor_domains {
            if factor.trim().is_empty() || values.is_empty() {
                return Err("reference factors and value domains must be non-empty".into());
            }
            let unique: BTreeSet<i64> = values.iter().copied().collect();
            if unique.len() != values.len() {
                return Err(format!("duplicate value in reference domain for {factor}"));
            }
            product = product
                .checked_mul(values.len())
                .ok_or_else(|| "reference Cartesian product overflow".to_string())?;
        }
        if product > self.max_reference_assignments {
            return Err(format!(
                "reference universe {product} exceeds cap {}",
                self.max_reference_assignments
            ));
        }

        let mut required = BTreeSet::new();
        referenced_features(rule, &mut required);
        let declared: BTreeSet<String> = self.factor_domains.keys().cloned().collect();
        for feature in required {
            if !declared.contains(&feature) {
                return Err(format!(
                    "rule feature {feature} missing from reference factor domain"
                ));
            }
        }

        if let Some(partition) = &self.reference_partition {
            validate_reference_partition(partition, self)?;
        }
        Ok(product)
    }
}

/// Project an assignment onto the declared reference factors while validating
/// that every reference factor is present and in-domain.
fn project_assignment(
    assignment: &BTreeMap<String, i64>,
    policy: &DifferentialValidityPolicy,
) -> Result<BTreeMap<String, i64>, String> {
    let declared: BTreeSet<&String> = policy.factor_domains.keys().collect();
    let observed: BTreeSet<&String> = assignment.keys().collect();
    if policy.require_exact_feature_schema && observed != declared {
        return Err("assignment feature schema differs from reference domain".into());
    }

    let mut projected = BTreeMap::new();
    for (factor, domain) in &policy.factor_domains {
        let value = assignment
            .get(factor)
            .copied()
            .ok_or_else(|| format!("assignment missing declared reference factor: {factor}"))?;
        if !domain.contains(&value) {
            return Err(format!(
                "feature {factor} value {value} is outside reference domain"
            ));
        }
        projected.insert(factor.clone(), value);
    }
    Ok(projected)
}

fn projected_digest(
    assignment: &BTreeMap<String, i64>,
    policy: &DifferentialValidityPolicy,
) -> Result<String, String> {
    assignment_digest(&project_assignment(assignment, policy)?)
}

fn validate_reference_partition(
    partition: &ReferencePartition,
    policy: &DifferentialValidityPolicy,
) -> Result<(), String> {
    if partition.train_assignments.is_empty() || partition.eval_assignments.is_empty() {
        return Err("reference partition requires non-empty train and eval assignments".into());
    }
    let mut train = BTreeSet::new();
    let mut eval = BTreeSet::new();
    for assignment in &partition.train_assignments {
        let digest = projected_digest(assignment, policy)?;
        if !train.insert(digest) {
            return Err("duplicate projected assignment in reference train partition".into());
        }
    }
    for assignment in &partition.eval_assignments {
        let digest = projected_digest(assignment, policy)?;
        if !eval.insert(digest.clone()) {
            return Err("duplicate projected assignment in reference eval partition".into());
        }
        if train.contains(&digest) {
            return Err("reference train/eval partition overlap".into());
        }
    }
    Ok(())
}

fn enumerate_assignments(domains: &BTreeMap<String, Vec<i64>>) -> Vec<BTreeMap<String, i64>> {
    fn recurse(
        factors: &[(&String, &Vec<i64>)],
        index: usize,
        current: &mut BTreeMap<String, i64>,
        out: &mut Vec<BTreeMap<String, i64>>,
    ) {
        if index == factors.len() {
            out.push(current.clone());
            return;
        }
        let (factor, values) = factors[index];
        for value in values {
            current.insert(factor.clone(), *value);
            recurse(factors, index + 1, current, out);
        }
        current.remove(factor.as_str());
    }

    let factors: Vec<_> = domains.iter().collect();
    let mut out = Vec::new();
    recurse(&factors, 0, &mut BTreeMap::new(), &mut out);
    out
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceTruthEntry {
    pub assignment_digest: String,
    pub label: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialValidityStatus {
    Passed,
    InconclusiveBenchmark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialViolationKind {
    StructuralValidityFailed,
    InvalidReferenceSpecification,
    OracleImplementationDisagreement,
    FeatureSchemaMismatch,
    OutOfDomainValue,
    DuplicateFeatureAssignment,
    ReferenceLabelMismatch,
    CoverageTooLow,
    ReferencePartitionMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DifferentialViolation {
    pub kind: DifferentialViolationKind,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DifferentialValidityReport {
    pub schema: String,
    pub status: DifferentialValidityStatus,
    pub structural_report: BenchmarkValidityReport,
    pub reference_universe_size: usize,
    pub generated_unique_assignments: usize,
    pub coverage_fraction: f64,
    pub oracle_disagreements: usize,
    pub reference_label_disagreements: usize,
    pub reference_truth_table_digest: Option<String>,
    pub violations: Vec<DifferentialViolation>,
}

impl DifferentialValidityReport {
    pub fn passed(&self) -> bool {
        self.status == DifferentialValidityStatus::Passed
    }
}

fn push_violation(
    violations: &mut Vec<DifferentialViolation>,
    kind: DifferentialViolationKind,
    detail: impl Into<String>,
) {
    violations.push(DifferentialViolation {
        kind,
        detail: detail.into(),
    });
}

fn projected_digest_set(
    examples: &[ExampleRecord],
    policy: &DifferentialValidityPolicy,
) -> Result<BTreeSet<String>, String> {
    examples
        .iter()
        .map(|example| projected_digest(&example.features, policy))
        .collect()
}

fn run_with_reference<F>(
    program: &TaskProgram,
    dataset: &GeneratedTaskDataset,
    policy: &DifferentialValidityPolicy,
    mut reference: F,
) -> Result<DifferentialValidityReport, String>
where
    F: FnMut(&RuleExpr, &BTreeMap<String, i64>) -> Result<bool, String>,
{
    let structural_report = validate_generated_task(program, dataset, &policy.structural_policy);
    let mut violations = Vec::new();
    if !structural_report.is_valid() {
        push_violation(
            &mut violations,
            DifferentialViolationKind::StructuralValidityFailed,
            "A3 structural/oracle validity must pass before differential evidence counts",
        );
    }

    let reference_universe_size = match policy.validate(&program.rule) {
        Ok(size) => size,
        Err(error) => {
            push_violation(
                &mut violations,
                DifferentialViolationKind::InvalidReferenceSpecification,
                error,
            );
            0
        }
    };
    if reference_universe_size == 0 {
        return Ok(DifferentialValidityReport {
            schema: DIFFERENTIAL_VALIDITY_SCHEMA_V1.to_string(),
            status: DifferentialValidityStatus::InconclusiveBenchmark,
            structural_report,
            reference_universe_size: 0,
            generated_unique_assignments: 0,
            coverage_fraction: 0.0,
            oracle_disagreements: 0,
            reference_label_disagreements: 0,
            reference_truth_table_digest: None,
            violations,
        });
    }

    let universe = enumerate_assignments(&policy.factor_domains);
    debug_assert_eq!(universe.len(), reference_universe_size);
    let mut truth_table = Vec::with_capacity(universe.len());
    let mut oracle_disagreements = 0usize;
    for assignment in &universe {
        let digest = assignment_digest(assignment)?;
        let production = evaluate_rule(&program.rule, assignment);
        let independent = reference(&program.rule, assignment);
        match (production, independent) {
            (Ok(left), Ok(right)) => {
                if left != right {
                    oracle_disagreements += 1;
                    push_violation(
                        &mut violations,
                        DifferentialViolationKind::OracleImplementationDisagreement,
                        format!("production/reference oracle disagree on assignment {digest}"),
                    );
                }
                truth_table.push(ReferenceTruthEntry {
                    assignment_digest: digest,
                    label: right,
                });
            }
            (Err(left), Err(right)) => {
                oracle_disagreements += 1;
                push_violation(
                    &mut violations,
                    DifferentialViolationKind::OracleImplementationDisagreement,
                    format!("both evaluators errored on finite assignment: production={left}; reference={right}"),
                );
            }
            (Err(error), Ok(label)) => {
                oracle_disagreements += 1;
                truth_table.push(ReferenceTruthEntry {
                    assignment_digest: digest,
                    label,
                });
                push_violation(
                    &mut violations,
                    DifferentialViolationKind::OracleImplementationDisagreement,
                    format!("production oracle errored while reference succeeded: {error}"),
                );
            }
            (Ok(_), Err(error)) => {
                oracle_disagreements += 1;
                push_violation(
                    &mut violations,
                    DifferentialViolationKind::OracleImplementationDisagreement,
                    format!("reference oracle errored while production succeeded: {error}"),
                );
            }
        }
    }

    truth_table.sort_by(|left, right| left.assignment_digest.cmp(&right.assignment_digest));
    let reference_truth_table_digest = canonical_hash(TRUTH_TABLE_HASH_DOMAIN, &truth_table).ok();
    let truth_labels: BTreeMap<String, bool> = truth_table
        .iter()
        .map(|entry| (entry.assignment_digest.clone(), entry.label))
        .collect();

    let declared_keys: BTreeSet<&String> = policy.factor_domains.keys().collect();
    let mut generated = BTreeSet::new();
    let mut reference_label_disagreements = 0usize;
    for (split, examples) in [("train", &dataset.train), ("eval", &dataset.eval)] {
        for example in examples {
            let observed_keys: BTreeSet<&String> = example.features.keys().collect();
            if policy.require_exact_feature_schema && observed_keys != declared_keys {
                push_violation(
                    &mut violations,
                    DifferentialViolationKind::FeatureSchemaMismatch,
                    format!("{split}:{} feature schema differs from reference domain", example.example_id),
                );
            }

            let projected = match project_assignment(&example.features, policy) {
                Ok(projected) => projected,
                Err(error) => {
                    let kind = if error.contains("outside reference domain") {
                        DifferentialViolationKind::OutOfDomainValue
                    } else {
                        DifferentialViolationKind::FeatureSchemaMismatch
                    };
                    push_violation(
                        &mut violations,
                        kind,
                        format!("{split}:{}: {error}", example.example_id),
                    );
                    continue;
                }
            };
            let digest = assignment_digest(&projected)?;
            if !generated.insert(digest.clone()) {
                push_violation(
                    &mut violations,
                    DifferentialViolationKind::DuplicateFeatureAssignment,
                    format!("projected reference assignment appears more than once: {digest}"),
                );
            }

            match truth_labels.get(&digest) {
                Some(reference_label) if *reference_label != example.expected_label => {
                    reference_label_disagreements += 1;
                    push_violation(
                        &mut violations,
                        DifferentialViolationKind::ReferenceLabelMismatch,
                        format!("{split}:{} disagrees with independent reference label", example.example_id),
                    );
                }
                None => {
                    reference_label_disagreements += 1;
                    push_violation(
                        &mut violations,
                        DifferentialViolationKind::ReferenceLabelMismatch,
                        format!("{split}:{} has no independent reference label", example.example_id),
                    );
                }
                _ => {}
            }
        }
    }

    let coverage_fraction = generated.len() as f64 / reference_universe_size as f64;
    if coverage_fraction < policy.minimum_coverage_fraction {
        push_violation(
            &mut violations,
            DifferentialViolationKind::CoverageTooLow,
            format!(
                "generated projected coverage {coverage_fraction:.4} is below frozen minimum {:.4}",
                policy.minimum_coverage_fraction
            ),
        );
    }

    if let Some(reference_partition) = &policy.reference_partition {
        let expected_train: BTreeSet<String> = reference_partition
            .train_assignments
            .iter()
            .map(|assignment| projected_digest(assignment, policy))
            .collect::<Result<_, _>>()?;
        let expected_eval: BTreeSet<String> = reference_partition
            .eval_assignments
            .iter()
            .map(|assignment| projected_digest(assignment, policy))
            .collect::<Result<_, _>>()?;
        let actual_train = projected_digest_set(&dataset.train, policy)?;
        let actual_eval = projected_digest_set(&dataset.eval, policy)?;
        if actual_train != expected_train || actual_eval != expected_eval {
            push_violation(
                &mut violations,
                DifferentialViolationKind::ReferencePartitionMismatch,
                format!(
                    "generated split differs from frozen reference partition: train actual/expected={}/{}, eval actual/expected={}/{}",
                    actual_train.len(),
                    expected_train.len(),
                    actual_eval.len(),
                    expected_eval.len()
                ),
            );
        }
    }

    Ok(DifferentialValidityReport {
        schema: DIFFERENTIAL_VALIDITY_SCHEMA_V1.to_string(),
        status: if violations.is_empty() {
            DifferentialValidityStatus::Passed
        } else {
            DifferentialValidityStatus::InconclusiveBenchmark
        },
        structural_report,
        reference_universe_size,
        generated_unique_assignments: generated.len(),
        coverage_fraction,
        oracle_disagreements,
        reference_label_disagreements,
        reference_truth_table_digest,
        violations,
    })
}

/// Compare A3 semantics, an independent reference evaluator, finite-domain truth
/// table, generated examples, and an optional frozen reference partition.
pub fn run_differential_validation(
    program: &TaskProgram,
    dataset: &GeneratedTaskDataset,
    policy: &DifferentialValidityPolicy,
) -> Result<DifferentialValidityReport, String> {
    run_with_reference(program, dataset, policy, reference_evaluate_rule)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{
        ContextVisibility, RuleExpr, TaskProgram, TimingRegime, TASK_PROGRAM_SCHEMA_V1,
    };
    use crate::experiment_validity::symbolic_oracle_digest;

    fn rule() -> RuleExpr {
        RuleExpr::Xor {
            left: Box::new(RuleExpr::Parity {
                factor: "a".into(),
                modulus: 2,
                remainder: 0,
            }),
            right: Box::new(RuleExpr::Eq {
                left: "b".into(),
                right: "c".into(),
            }),
        }
    }

    fn assignment(a: i64, b: i64, c: i64) -> BTreeMap<String, i64> {
        BTreeMap::from([
            ("a".into(), a),
            ("b".into(), b),
            ("c".into(), c),
        ])
    }

    fn example(
        id: &str,
        features: BTreeMap<String, i64>,
        split: &str,
        rule: &RuleExpr,
    ) -> ExampleRecord {
        let expected_label = evaluate_rule(rule, &features).unwrap();
        ExampleRecord {
            example_id: id.into(),
            features,
            support_tags: vec![split.into()],
            expected_label,
        }
    }

    fn fixture() -> (TaskProgram, GeneratedTaskDataset, DifferentialValidityPolicy) {
        let rule = rule();
        let universe = [
            assignment(0, 0, 0),
            assignment(0, 0, 1),
            assignment(0, 1, 0),
            assignment(0, 1, 1),
            assignment(1, 0, 0),
            assignment(1, 0, 1),
            assignment(1, 1, 0),
            assignment(1, 1, 1),
        ];
        let train_assignments = universe[..4].to_vec();
        let eval_assignments = universe[4..].to_vec();
        let train: Vec<_> = train_assignments
            .iter()
            .enumerate()
            .map(|(index, features)| {
                example(&format!("train-{index}"), features.clone(), "train", &rule)
            })
            .collect();
        let eval: Vec<_> = eval_assignments
            .iter()
            .enumerate()
            .map(|(index, features)| {
                example(&format!("eval-{index}"), features.clone(), "eval", &rule)
            })
            .collect();
        let positive_examples = train
            .iter()
            .chain(&eval)
            .filter(|example| example.expected_label)
            .count();
        let negative_examples = train.len() + eval.len() - positive_examples;
        let mut program = TaskProgram {
            schema: TASK_PROGRAM_SCHEMA_V1.into(),
            program_id: "differential-fixture".into(),
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
        let policy = DifferentialValidityPolicy {
            structural_policy: BenchmarkValidityPolicy::task_free_strict(),
            factor_domains: BTreeMap::from([
                ("a".into(), vec![0, 1]),
                ("b".into(), vec![0, 1]),
                ("c".into(), vec![0, 1]),
            ]),
            max_reference_assignments: 32,
            require_exact_feature_schema: true,
            minimum_coverage_fraction: 1.0,
            reference_partition: Some(ReferencePartition {
                train_assignments,
                eval_assignments,
            }),
        };
        (program, dataset, policy)
    }

    #[test]
    fn reference_and_a3_semantics_agree_exhaustively() {
        let (program, dataset, policy) = fixture();
        let report = run_differential_validation(&program, &dataset, &policy).unwrap();
        assert!(report.passed(), "violations: {:?}", report.violations);
        assert_eq!(report.reference_universe_size, 8);
        assert_eq!(report.generated_unique_assignments, 8);
        assert_eq!(report.coverage_fraction, 1.0);
        assert_eq!(report.oracle_disagreements, 0);
        assert_eq!(report.reference_label_disagreements, 0);
        assert!(report.reference_truth_table_digest.is_some());
    }

    #[test]
    fn checker_detects_injected_reference_semantic_disagreement() {
        let (program, dataset, policy) = fixture();
        let mut flipped_once = false;
        let report = run_with_reference(&program, &dataset, &policy, |rule, features| {
            let value = reference_evaluate_rule(rule, features)?;
            if !flipped_once {
                flipped_once = true;
                Ok(!value)
            } else {
                Ok(value)
            }
        })
        .unwrap();
        assert_eq!(report.status, DifferentialValidityStatus::InconclusiveBenchmark);
        assert_eq!(report.oracle_disagreements, 1);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == DifferentialViolationKind::OracleImplementationDisagreement
        }));
    }

    #[test]
    fn wrong_reference_partition_is_detected_even_with_correct_labels() {
        let (program, dataset, mut policy) = fixture();
        let partition = policy.reference_partition.as_mut().unwrap();
        let moved = partition.train_assignments.pop().unwrap();
        partition.eval_assignments.push(moved);
        let report = run_differential_validation(&program, &dataset, &policy).unwrap();
        assert_eq!(report.status, DifferentialValidityStatus::InconclusiveBenchmark);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == DifferentialViolationKind::ReferencePartitionMismatch
        }));
    }

    #[test]
    fn extra_learner_feature_is_rejected_by_exact_schema() {
        let (program, mut dataset, policy) = fixture();
        for example in dataset.train.iter_mut().chain(&mut dataset.eval) {
            example.features.insert("nuisance".into(), 7);
        }
        let report = run_differential_validation(&program, &dataset, &policy).unwrap();
        assert_eq!(report.status, DifferentialValidityStatus::InconclusiveBenchmark);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == DifferentialViolationKind::FeatureSchemaMismatch
        }));
    }

    #[test]
    fn relaxed_schema_projects_nuisance_out_of_reference_identity() {
        let (program, mut dataset, mut policy) = fixture();
        policy.require_exact_feature_schema = false;
        policy.reference_partition = None;
        for (index, example) in dataset
            .train
            .iter_mut()
            .chain(&mut dataset.eval)
            .enumerate()
        {
            example.features.insert("nuisance".into(), index as i64);
        }
        let report = run_differential_validation(&program, &dataset, &policy).unwrap();
        assert!(report.passed(), "violations: {:?}", report.violations);
        assert_eq!(report.generated_unique_assignments, 8);
        assert_eq!(report.coverage_fraction, 1.0);
    }

    #[test]
    fn out_of_domain_value_is_rejected() {
        let (mut program, mut dataset, policy) = fixture();
        dataset.eval[0].features.insert("a".into(), 9);
        dataset.eval[0].expected_label =
            evaluate_rule(&program.rule, &dataset.eval[0].features).unwrap();
        program.positive_examples = dataset
            .train
            .iter()
            .chain(&dataset.eval)
            .filter(|example| example.expected_label)
            .count();
        program.negative_examples = dataset.train.len() + dataset.eval.len() - program.positive_examples;
        dataset.program_digest = program.digest().unwrap();

        let report = run_differential_validation(&program, &dataset, &policy).unwrap();
        assert_eq!(report.status, DifferentialValidityStatus::InconclusiveBenchmark);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == DifferentialViolationKind::OutOfDomainValue
        }));
    }

    #[test]
    fn low_coverage_is_rejected_without_claiming_generator_failure() {
        let (mut program, mut dataset, mut policy) = fixture();
        dataset.eval.pop();
        program.positive_examples = dataset
            .train
            .iter()
            .chain(&dataset.eval)
            .filter(|example| example.expected_label)
            .count();
        program.negative_examples = dataset.train.len() + dataset.eval.len() - program.positive_examples;
        dataset.program_digest = program.digest().unwrap();
        policy.reference_partition = None;
        policy.minimum_coverage_fraction = 1.0;

        let report = run_differential_validation(&program, &dataset, &policy).unwrap();
        assert_eq!(report.status, DifferentialValidityStatus::InconclusiveBenchmark);
        assert!(report.coverage_fraction < 1.0);
        assert!(report.violations.iter().any(|violation| {
            violation.kind == DifferentialViolationKind::CoverageTooLow
        }));
    }

    #[test]
    fn reference_spec_rejects_missing_rule_factor_and_excessive_universe() {
        let (program, _, mut policy) = fixture();
        policy.factor_domains.remove("c");
        assert!(policy.validate(&program.rule).is_err());

        let (_, _, mut policy) = fixture();
        policy.max_reference_assignments = 4;
        assert!(policy.validate(&program.rule).is_err());
    }
}
