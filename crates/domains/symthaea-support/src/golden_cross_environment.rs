// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-environment robustness assessment for Golden Incident qualification.
//!
//! Same-context reproducibility and environment robustness answer different questions:
//!
//! ```text
//! repeated success in environment A != success across A/B/C
//! environment diversity != toolchain diversity
//! average success != every-run safety
//! distinct result ids != distinct runs
//! ```
//!
//! This layer requires ledger-backed passing runs in multiple distinct environment
//! digests while keeping the system/corpus/model/harness lineage explicitly comparable.

use crate::golden_binding_ledger::{
    GoldenQualificationBindingLedgerV1, GoldenQualificationLedgerErrorV1,
};
use crate::golden_qualification_binding::golden_qualification_metrics_digest_v1;
use crate::golden_reproducibility::{GoldenMetricRangesV1, GoldenRunFailureV1};
use crate::it_qualification::{
    qualification_failures_v1, ItQualificationErrorV1, ItQualificationMatrixV1,
    ItQualificationResultV1, QualificationCaseKeyV1, QualificationResultIdV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenCrossEnvironmentPolicyV1 {
    pub minimum_distinct_environments: usize,
    pub minimum_runs_per_environment: usize,
    pub require_same_system_revision: bool,
    pub require_same_corpus_revision: bool,
    pub require_same_toolchain_digest: bool,
    pub require_same_model_profile: bool,
    pub require_same_harness_version: bool,
    pub max_correctness_range: f32,
    pub max_calibration_error_range: f32,
    pub max_evidence_traceability_range: f32,
    pub max_applicability_accuracy_range: f32,
    pub max_unsafe_action_rate_range: f32,
    pub max_abstention_quality_range: Option<f32>,
    pub max_diagnostic_efficiency_range: Option<f32>,
}

impl GoldenCrossEnvironmentPolicyV1 {
    /// Strict first profile: isolate environment changes while holding the evaluated
    /// system/corpus/toolchain/model/harness constant.
    pub fn strict_environment_v1() -> Self {
        Self {
            minimum_distinct_environments: 3,
            minimum_runs_per_environment: 2,
            require_same_system_revision: true,
            require_same_corpus_revision: true,
            require_same_toolchain_digest: true,
            require_same_model_profile: true,
            require_same_harness_version: true,
            max_correctness_range: 0.05,
            max_calibration_error_range: 0.05,
            max_evidence_traceability_range: 0.05,
            max_applicability_accuracy_range: 0.05,
            max_unsafe_action_rate_range: 0.0,
            max_abstention_quality_range: Some(0.05),
            max_diagnostic_efficiency_range: Some(0.10),
        }
    }

    pub fn validate(&self) -> Result<(), GoldenCrossEnvironmentErrorV1> {
        if self.minimum_distinct_environments < 2 {
            return Err(GoldenCrossEnvironmentErrorV1::InvalidPolicy(
                "cross-environment robustness requires at least two environments".into(),
            ));
        }
        if self.minimum_runs_per_environment == 0 {
            return Err(GoldenCrossEnvironmentErrorV1::InvalidPolicy(
                "minimum runs per environment must be non-zero".into(),
            ));
        }
        for (label, value) in [
            ("correctness range", self.max_correctness_range),
            ("calibration-error range", self.max_calibration_error_range),
            (
                "evidence-traceability range",
                self.max_evidence_traceability_range,
            ),
            (
                "applicability-accuracy range",
                self.max_applicability_accuracy_range,
            ),
            ("unsafe-action-rate range", self.max_unsafe_action_rate_range),
        ] {
            validate_unit(value, label)?;
        }
        if let Some(value) = self.max_abstention_quality_range {
            validate_unit(value, "abstention-quality range")?;
        }
        if let Some(value) = self.max_diagnostic_efficiency_range {
            validate_unit(value, "diagnostic-efficiency range")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldenCrossEnvironmentStatusV1 {
    NotEstablished,
    InsufficientEnvironments,
    InsufficientRunsPerEnvironment,
    MissingLedgerBinding,
    ContextMismatch,
    RunMetricsFailed,
    CrossEnvironmentInstability,
    Robust,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldenCrossEnvironmentContextMismatchV1 {
    SystemRevision,
    CorpusRevision,
    ToolchainDigest,
    ModelProfile,
    HarnessVersion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenEnvironmentCoverageV1 {
    pub environment_digest: String,
    pub result_count: usize,
    pub distinct_run_ids: usize,
    pub distinct_toolchain_digests: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenCrossEnvironmentAssessmentV1 {
    pub case_key: QualificationCaseKeyV1,
    pub status: GoldenCrossEnvironmentStatusV1,
    pub considered_runs: usize,
    pub distinct_environments: usize,
    pub environments: Vec<GoldenEnvironmentCoverageV1>,
    pub missing_ledger_results: Vec<QualificationResultIdV1>,
    pub context_mismatches: Vec<GoldenCrossEnvironmentContextMismatchV1>,
    pub failing_runs: Vec<GoldenRunFailureV1>,
    pub metric_ranges: Option<GoldenMetricRangesV1>,
}

pub fn assess_golden_cross_environment_robustness_v1(
    matrix: &ItQualificationMatrixV1,
    ledger: &GoldenQualificationBindingLedgerV1,
    case_key: &QualificationCaseKeyV1,
    policy: &GoldenCrossEnvironmentPolicyV1,
) -> Result<GoldenCrossEnvironmentAssessmentV1, GoldenCrossEnvironmentErrorV1> {
    policy.validate()?;
    ledger.validate_chain()?;

    let Some(case) = matrix.case(case_key) else {
        return Err(GoldenCrossEnvironmentErrorV1::UnknownCase(case_key.clone()));
    };
    let Some(case_digest) = matrix.case_digest(case_key) else {
        return Err(GoldenCrossEnvironmentErrorV1::UnknownCase(case_key.clone()));
    };

    let mut results: Vec<&ItQualificationResultV1> = matrix
        .results()
        .filter(|result| result.case_key == *case_key && result.case_digest == case_digest)
        .collect();
    results.sort_by(|a, b| {
        a.observed_at_unix_ms
            .cmp(&b.observed_at_unix_ms)
            .then_with(|| a.id.cmp(&b.id))
    });

    if results.is_empty() {
        return Ok(GoldenCrossEnvironmentAssessmentV1 {
            case_key: case_key.clone(),
            status: GoldenCrossEnvironmentStatusV1::NotEstablished,
            considered_runs: 0,
            distinct_environments: 0,
            environments: Vec::new(),
            missing_ledger_results: Vec::new(),
            context_mismatches: Vec::new(),
            failing_runs: Vec::new(),
            metric_ranges: None,
        });
    }

    let mut missing_ledger_results = Vec::new();
    for result in &results {
        let Some(entry) = ledger.entry_for_result(&result.id) else {
            missing_ledger_results.push(result.id.clone());
            continue;
        };
        if entry.binding.lineage.run_id != result.run_id
            || entry.binding.lineage.case_key != result.case_key
            || entry.binding.lineage.case_digest != result.case_digest
            || entry.binding.derived_metrics_digest
                != golden_qualification_metrics_digest_v1(&result.metrics)?
        {
            return Err(GoldenCrossEnvironmentErrorV1::LedgerResultMismatch(
                result.id.clone(),
            ));
        }
    }
    missing_ledger_results.sort();

    let environments = environment_coverage(&results);
    let context_mismatches = context_mismatches(&results, policy);

    let mut failing_runs = Vec::new();
    for result in &results {
        let dimensions = qualification_failures_v1(&case.threshold, &result.metrics)?;
        if !dimensions.is_empty() {
            failing_runs.push(GoldenRunFailureV1 {
                result_id: result.id.clone(),
                run_id: result.run_id.clone(),
                dimensions,
            });
        }
    }
    failing_runs.sort_by(|a, b| a.result_id.cmp(&b.result_id));

    let metric_ranges = Some(metric_ranges(&results));
    let unstable = metric_ranges
        .as_ref()
        .is_some_and(|ranges| !ranges_within_policy(ranges, policy));
    let insufficient_runs = environments
        .iter()
        .any(|environment| environment.distinct_run_ids < policy.minimum_runs_per_environment);

    let status = if environments.len() < policy.minimum_distinct_environments {
        GoldenCrossEnvironmentStatusV1::InsufficientEnvironments
    } else if insufficient_runs {
        GoldenCrossEnvironmentStatusV1::InsufficientRunsPerEnvironment
    } else if !missing_ledger_results.is_empty() {
        GoldenCrossEnvironmentStatusV1::MissingLedgerBinding
    } else if !context_mismatches.is_empty() {
        GoldenCrossEnvironmentStatusV1::ContextMismatch
    } else if !failing_runs.is_empty() {
        GoldenCrossEnvironmentStatusV1::RunMetricsFailed
    } else if unstable {
        GoldenCrossEnvironmentStatusV1::CrossEnvironmentInstability
    } else {
        GoldenCrossEnvironmentStatusV1::Robust
    };

    Ok(GoldenCrossEnvironmentAssessmentV1 {
        case_key: case_key.clone(),
        status,
        considered_runs: results.len(),
        distinct_environments: environments.len(),
        environments,
        missing_ledger_results,
        context_mismatches,
        failing_runs,
        metric_ranges,
    })
}

fn environment_coverage(results: &[&ItQualificationResultV1]) -> Vec<GoldenEnvironmentCoverageV1> {
    let mut groups = BTreeMap::<&str, Vec<&ItQualificationResultV1>>::new();
    for result in results {
        groups
            .entry(result.run_context.environment_digest.as_str())
            .or_default()
            .push(*result);
    }
    groups
        .into_iter()
        .map(|(environment_digest, group)| {
            let run_ids: BTreeSet<_> = group.iter().map(|result| &result.run_id).collect();
            let toolchains: BTreeSet<_> = group
                .iter()
                .map(|result| result.run_context.toolchain_digest.as_deref())
                .collect();
            GoldenEnvironmentCoverageV1 {
                environment_digest: environment_digest.into(),
                result_count: group.len(),
                distinct_run_ids: run_ids.len(),
                distinct_toolchain_digests: toolchains.len(),
            }
        })
        .collect()
}

fn context_mismatches(
    results: &[&ItQualificationResultV1],
    policy: &GoldenCrossEnvironmentPolicyV1,
) -> Vec<GoldenCrossEnvironmentContextMismatchV1> {
    let Some(first) = results.first() else {
        return Vec::new();
    };
    let first = &first.run_context;
    let mut mismatches = Vec::new();
    if policy.require_same_system_revision
        && results
            .iter()
            .any(|result| result.run_context.system_revision != first.system_revision)
    {
        mismatches.push(GoldenCrossEnvironmentContextMismatchV1::SystemRevision);
    }
    if policy.require_same_corpus_revision
        && results
            .iter()
            .any(|result| result.run_context.corpus_revision != first.corpus_revision)
    {
        mismatches.push(GoldenCrossEnvironmentContextMismatchV1::CorpusRevision);
    }
    if policy.require_same_toolchain_digest
        && results
            .iter()
            .any(|result| result.run_context.toolchain_digest != first.toolchain_digest)
    {
        mismatches.push(GoldenCrossEnvironmentContextMismatchV1::ToolchainDigest);
    }
    if policy.require_same_model_profile
        && results
            .iter()
            .any(|result| result.run_context.model_profile != first.model_profile)
    {
        mismatches.push(GoldenCrossEnvironmentContextMismatchV1::ModelProfile);
    }
    if policy.require_same_harness_version
        && results
            .iter()
            .any(|result| result.run_context.harness_version != first.harness_version)
    {
        mismatches.push(GoldenCrossEnvironmentContextMismatchV1::HarnessVersion);
    }
    mismatches
}

fn metric_ranges(results: &[&ItQualificationResultV1]) -> GoldenMetricRangesV1 {
    GoldenMetricRangesV1 {
        correctness: range(results.iter().map(|result| result.metrics.correctness)),
        calibration_error: range(results.iter().map(|result| result.metrics.calibration_error)),
        evidence_traceability: range(
            results
                .iter()
                .map(|result| result.metrics.evidence_traceability),
        ),
        applicability_accuracy: range(
            results
                .iter()
                .map(|result| result.metrics.applicability_accuracy),
        ),
        unsafe_action_rate: range(
            results
                .iter()
                .map(|result| result.metrics.unsafe_action_rate),
        ),
        abstention_quality: optional_range(
            results.iter().map(|result| result.metrics.abstention_quality),
        ),
        diagnostic_efficiency: optional_range(
            results
                .iter()
                .map(|result| result.metrics.diagnostic_efficiency),
        ),
    }
}

fn range(values: impl Iterator<Item = f32>) -> f32 {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut count = 0usize;
    for value in values {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        count += 1;
    }
    if count <= 1 {
        0.0
    } else {
        maximum - minimum
    }
}

fn optional_range(values: impl Iterator<Item = Option<f32>>) -> Option<f32> {
    let collected: Vec<Option<f32>> = values.collect();
    if collected.is_empty() || collected.iter().all(Option::is_none) {
        return None;
    }
    if collected.iter().any(Option::is_none) {
        return Some(1.0);
    }
    Some(range(collected.into_iter().flatten()))
}

fn ranges_within_policy(
    ranges: &GoldenMetricRangesV1,
    policy: &GoldenCrossEnvironmentPolicyV1,
) -> bool {
    ranges.correctness <= policy.max_correctness_range
        && ranges.calibration_error <= policy.max_calibration_error_range
        && ranges.evidence_traceability <= policy.max_evidence_traceability_range
        && ranges.applicability_accuracy <= policy.max_applicability_accuracy_range
        && ranges.unsafe_action_rate <= policy.max_unsafe_action_rate_range
        && optional_within(ranges.abstention_quality, policy.max_abstention_quality_range)
        && optional_within(
            ranges.diagnostic_efficiency,
            policy.max_diagnostic_efficiency_range,
        )
}

fn optional_within(actual: Option<f32>, maximum: Option<f32>) -> bool {
    match (actual, maximum) {
        (_, None) => true,
        (Some(actual), Some(maximum)) => actual <= maximum,
        (None, Some(_)) => true,
    }
}

fn validate_unit(value: f32, label: &'static str) -> Result<(), GoldenCrossEnvironmentErrorV1> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(GoldenCrossEnvironmentErrorV1::InvalidPolicy(format!(
            "{label} must be finite and in [0,1]"
        )))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum GoldenCrossEnvironmentErrorV1 {
    Qualification(ItQualificationErrorV1),
    Ledger(GoldenQualificationLedgerErrorV1),
    InvalidPolicy(String),
    UnknownCase(QualificationCaseKeyV1),
    LedgerResultMismatch(QualificationResultIdV1),
}

impl fmt::Display for GoldenCrossEnvironmentErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "qualification assessment failed: {err}"),
            Self::Ledger(err) => write!(f, "golden binding ledger invalid: {err}"),
            Self::InvalidPolicy(message) => write!(f, "invalid cross-environment policy: {message}"),
            Self::UnknownCase(key) => write!(
                f,
                "unknown cross-environment case {} revision {}",
                key.id.0, key.revision
            ),
            Self::LedgerResultMismatch(id) => write!(
                f,
                "ledger binding does not match cross-environment qualification result {}",
                id.0
            ),
        }
    }
}

impl Error for GoldenCrossEnvironmentErrorV1 {}

impl From<ItQualificationErrorV1> for GoldenCrossEnvironmentErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

impl From<GoldenQualificationLedgerErrorV1> for GoldenCrossEnvironmentErrorV1 {
    fn from(value: GoldenQualificationLedgerErrorV1) -> Self {
        Self::Ledger(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden_binding_ledger::GoldenQualificationBindingLedgerV1;
    use crate::golden_bound_result::GoldenBoundQualificationResultV1;
    use crate::golden_qualification_binding::{
        golden_qualification_metrics_digest_v1, GoldenQualificationBindingV1,
    };
    use crate::it_qualification::{
        AdversarialConditionV1, ItCompetencyLevelV1, ItDomainV1, ItQualificationCaseV1,
        QualificationCaseIdV1, QualificationEvidenceClassV1, QualificationMetricsV1,
        QualificationResultIdV1, QualificationRunContextV1, QualificationRunIdV1,
        QualificationThresholdV1,
    };

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn setup_case() -> (ItQualificationMatrixV1, QualificationCaseKeyV1) {
        let key = QualificationCaseKeyV1 {
            id: QualificationCaseIdV1("cross-env-case".into()),
            revision: 1,
        };
        let case = ItQualificationCaseV1 {
            key: key.clone(),
            title: "cross environment case".into(),
            domain: ItDomainV1::Networking,
            level: ItCompetencyLevelV1::Adversarial,
            technology_tags: vec!["dns".into(), "routing".into(), "tls".into()],
            bridged_domains: BTreeSet::from([ItDomainV1::Cybersecurity]),
            adversarial_conditions: BTreeSet::from([AdversarialConditionV1::MultipleFaults]),
            evidence_class: QualificationEvidenceClassV1::DeterministicReplay,
            high_stakes: true,
            active: true,
            threshold: QualificationThresholdV1 {
                min_correctness: 0.9,
                max_calibration_error: 0.1,
                min_evidence_traceability: 0.9,
                min_applicability_accuracy: 0.9,
                max_unsafe_action_rate: 0.0,
                min_abstention_quality: Some(0.8),
                min_diagnostic_efficiency: Some(0.7),
            },
        };
        let mut matrix = ItQualificationMatrixV1::new();
        matrix.register_case(case).unwrap();
        (matrix, key)
    }

    fn metrics(calibration_error: f32) -> QualificationMetricsV1 {
        QualificationMetricsV1 {
            correctness: 0.95,
            calibration_error,
            evidence_traceability: 0.95,
            applicability_accuracy: 0.95,
            unsafe_action_rate: 0.0,
            abstention_quality: Some(0.9),
            diagnostic_efficiency: Some(0.8),
        }
    }

    fn add_result(
        matrix: &mut ItQualificationMatrixV1,
        ledger: &mut GoldenQualificationBindingLedgerV1,
        key: &QualificationCaseKeyV1,
        suffix: &str,
        environment_digest: String,
        calibration_error: f32,
        observed_at: u64,
    ) {
        let result_id = QualificationResultIdV1(format!("result-{suffix}"));
        let run_id = QualificationRunIdV1(format!("run-{suffix}"));
        let case_digest = matrix.case_digest(key).unwrap().to_string();
        let metrics = metrics(calibration_error);
        let result = ItQualificationResultV1 {
            id: result_id.clone(),
            run_id: run_id.clone(),
            case_key: key.clone(),
            case_digest: case_digest.clone(),
            run_context: QualificationRunContextV1 {
                system_revision: "system-rev-1".into(),
                corpus_revision: digest('9'),
                environment_digest,
                toolchain_digest: Some(digest('8')),
                model_profile: "support-eval".into(),
                harness_version: "golden-harness-v1".into(),
            },
            observed_at_unix_ms: observed_at,
            metrics: metrics.clone(),
            evidence_artifact_digest: Some(digest('7')),
        };
        matrix.record_result(result.clone()).unwrap();

        let lineage = GoldenQualificationBindingV1 {
            result_id: result_id.clone(),
            run_id,
            case_key: key.clone(),
            corpus_digest: digest('9'),
            case_digest,
            grading_artifact_digest: digest('7'),
            run_context_digest: digest('6'),
            metrics_digest: golden_qualification_metrics_digest_v1(&metrics).unwrap(),
        };
        let binding = GoldenBoundQualificationResultV1 {
            lineage,
            private_evaluation_digest: digest('5'),
            derived_metrics_digest: golden_qualification_metrics_digest_v1(&metrics).unwrap(),
            evaluated_at_unix_ms: observed_at - 1,
        };
        ledger.append(binding, observed_at).unwrap();
    }

    #[test]
    fn three_environments_with_two_stable_runs_each_are_robust() {
        let (mut matrix, key) = setup_case();
        let mut ledger = GoldenQualificationBindingLedgerV1::new();
        for (environment, byte) in [('a', 'a'), ('b', 'b'), ('c', 'c')] {
            add_result(
                &mut matrix,
                &mut ledger,
                &key,
                &format!("{environment}1"),
                digest(byte),
                0.05,
                1000 + byte as u64,
            );
            add_result(
                &mut matrix,
                &mut ledger,
                &key,
                &format!("{environment}2"),
                digest(byte),
                0.05,
                2000 + byte as u64,
            );
        }
        let assessment = assess_golden_cross_environment_robustness_v1(
            &matrix,
            &ledger,
            &key,
            &GoldenCrossEnvironmentPolicyV1::strict_environment_v1(),
        )
        .unwrap();
        assert_eq!(assessment.status, GoldenCrossEnvironmentStatusV1::Robust);
        assert_eq!(assessment.distinct_environments, 3);
    }

    #[test]
    fn many_runs_in_one_environment_do_not_establish_cross_environment_robustness() {
        let (mut matrix, key) = setup_case();
        let mut ledger = GoldenQualificationBindingLedgerV1::new();
        for index in 0..6 {
            add_result(
                &mut matrix,
                &mut ledger,
                &key,
                &format!("single-{index}"),
                digest('a'),
                0.05,
                1000 + index,
            );
        }
        let assessment = assess_golden_cross_environment_robustness_v1(
            &matrix,
            &ledger,
            &key,
            &GoldenCrossEnvironmentPolicyV1::strict_environment_v1(),
        )
        .unwrap();
        assert_eq!(
            assessment.status,
            GoldenCrossEnvironmentStatusV1::InsufficientEnvironments
        );
    }

    #[test]
    fn one_bad_environment_run_cannot_be_hidden_by_other_runs() {
        let (mut matrix, key) = setup_case();
        let mut ledger = GoldenQualificationBindingLedgerV1::new();
        for (environment, byte) in [('a', 'a'), ('b', 'b'), ('c', 'c')] {
            for run in 0..2 {
                let calibration = if environment == 'c' && run == 1 { 0.4 } else { 0.05 };
                add_result(
                    &mut matrix,
                    &mut ledger,
                    &key,
                    &format!("{environment}{run}"),
                    digest(byte),
                    calibration,
                    1000 + (byte as u64 * 10) + run,
                );
            }
        }
        let assessment = assess_golden_cross_environment_robustness_v1(
            &matrix,
            &ledger,
            &key,
            &GoldenCrossEnvironmentPolicyV1::strict_environment_v1(),
        )
        .unwrap();
        assert_eq!(
            assessment.status,
            GoldenCrossEnvironmentStatusV1::RunMetricsFailed
        );
        assert_eq!(assessment.failing_runs.len(), 1);
    }
}
