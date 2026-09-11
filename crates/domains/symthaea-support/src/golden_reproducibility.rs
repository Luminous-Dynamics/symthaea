// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Repeated-run reproducibility assessment for Golden Incident qualification.
//!
//! One passing run is not enough to establish stable competence. This module
//! requires ledger-backed repeated results, reuses the case's existing threshold
//! for every run, and then applies explicit context-comparability and metric-range
//! requirements. Averages never erase a failed or unsafe run.

use crate::golden_binding_ledger::{
    GoldenQualificationBindingLedgerV1, GoldenQualificationLedgerErrorV1,
};
use crate::golden_qualification_binding::golden_qualification_metrics_digest_v1;
use crate::it_qualification::{
    qualification_failures_v1, ItQualificationErrorV1, ItQualificationMatrixV1,
    ItQualificationResultV1, QualificationCaseKeyV1, QualificationFailureDimensionV1,
    QualificationResultIdV1, QualificationRunIdV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenReproducibilityPolicyV1 {
    pub minimum_runs: usize,
    pub require_same_system_revision: bool,
    pub require_same_corpus_revision: bool,
    pub require_same_environment_digest: bool,
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

impl GoldenReproducibilityPolicyV1 {
    pub fn strict_deterministic_v1() -> Self {
        Self {
            minimum_runs: 3,
            require_same_system_revision: true,
            require_same_corpus_revision: true,
            require_same_environment_digest: true,
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

    pub fn validate(&self) -> Result<(), GoldenReproducibilityErrorV1> {
        if self.minimum_runs < 2 {
            return Err(GoldenReproducibilityErrorV1::InvalidPolicy(
                "reproducibility requires at least two runs".into(),
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
pub enum GoldenReproducibilityStatusV1 {
    NotEstablished,
    InsufficientRuns,
    MissingLedgerBinding,
    ContextMismatch,
    RunMetricsFailed,
    MetricInstability,
    Reproducible,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenMetricRangesV1 {
    pub correctness: f32,
    pub calibration_error: f32,
    pub evidence_traceability: f32,
    pub applicability_accuracy: f32,
    pub unsafe_action_rate: f32,
    pub abstention_quality: Option<f32>,
    pub diagnostic_efficiency: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenRunFailureV1 {
    pub result_id: QualificationResultIdV1,
    pub run_id: QualificationRunIdV1,
    pub dimensions: Vec<QualificationFailureDimensionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldenContextMismatchV1 {
    SystemRevision,
    CorpusRevision,
    EnvironmentDigest,
    ToolchainDigest,
    ModelProfile,
    HarnessVersion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenReproducibilityAssessmentV1 {
    pub case_key: QualificationCaseKeyV1,
    pub status: GoldenReproducibilityStatusV1,
    pub considered_runs: usize,
    pub distinct_run_ids: usize,
    pub missing_ledger_results: Vec<QualificationResultIdV1>,
    pub context_mismatches: Vec<GoldenContextMismatchV1>,
    pub failing_runs: Vec<GoldenRunFailureV1>,
    pub metric_ranges: Option<GoldenMetricRangesV1>,
}

pub fn assess_golden_reproducibility_v1(
    matrix: &ItQualificationMatrixV1,
    ledger: &GoldenQualificationBindingLedgerV1,
    case_key: &QualificationCaseKeyV1,
    policy: &GoldenReproducibilityPolicyV1,
) -> Result<GoldenReproducibilityAssessmentV1, GoldenReproducibilityErrorV1> {
    policy.validate()?;
    ledger.validate_chain()?;

    let Some(case) = matrix.case(case_key) else {
        return Err(GoldenReproducibilityErrorV1::UnknownCase(case_key.clone()));
    };
    let Some(case_digest) = matrix.case_digest(case_key) else {
        return Err(GoldenReproducibilityErrorV1::UnknownCase(case_key.clone()));
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
        return Ok(GoldenReproducibilityAssessmentV1 {
            case_key: case_key.clone(),
            status: GoldenReproducibilityStatusV1::NotEstablished,
            considered_runs: 0,
            distinct_run_ids: 0,
            missing_ledger_results: Vec::new(),
            context_mismatches: Vec::new(),
            failing_runs: Vec::new(),
            metric_ranges: None,
        });
    }

    let run_ids: BTreeSet<_> = results.iter().map(|result| &result.run_id).collect();
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
            return Err(GoldenReproducibilityErrorV1::LedgerResultMismatch(
                result.id.clone(),
            ));
        }
    }
    missing_ledger_results.sort();

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

    let status = if results.len() < policy.minimum_runs || run_ids.len() < policy.minimum_runs {
        GoldenReproducibilityStatusV1::InsufficientRuns
    } else if !missing_ledger_results.is_empty() {
        GoldenReproducibilityStatusV1::MissingLedgerBinding
    } else if !context_mismatches.is_empty() {
        GoldenReproducibilityStatusV1::ContextMismatch
    } else if !failing_runs.is_empty() {
        GoldenReproducibilityStatusV1::RunMetricsFailed
    } else if unstable {
        GoldenReproducibilityStatusV1::MetricInstability
    } else {
        GoldenReproducibilityStatusV1::Reproducible
    };

    Ok(GoldenReproducibilityAssessmentV1 {
        case_key: case_key.clone(),
        status,
        considered_runs: results.len(),
        distinct_run_ids: run_ids.len(),
        missing_ledger_results,
        context_mismatches,
        failing_runs,
        metric_ranges,
    })
}

fn context_mismatches(
    results: &[&ItQualificationResultV1],
    policy: &GoldenReproducibilityPolicyV1,
) -> Vec<GoldenContextMismatchV1> {
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
        mismatches.push(GoldenContextMismatchV1::SystemRevision);
    }
    if policy.require_same_corpus_revision
        && results
            .iter()
            .any(|result| result.run_context.corpus_revision != first.corpus_revision)
    {
        mismatches.push(GoldenContextMismatchV1::CorpusRevision);
    }
    if policy.require_same_environment_digest
        && results
            .iter()
            .any(|result| result.run_context.environment_digest != first.environment_digest)
    {
        mismatches.push(GoldenContextMismatchV1::EnvironmentDigest);
    }
    if policy.require_same_toolchain_digest
        && results
            .iter()
            .any(|result| result.run_context.toolchain_digest != first.toolchain_digest)
    {
        mismatches.push(GoldenContextMismatchV1::ToolchainDigest);
    }
    if policy.require_same_model_profile
        && results
            .iter()
            .any(|result| result.run_context.model_profile != first.model_profile)
    {
        mismatches.push(GoldenContextMismatchV1::ModelProfile);
    }
    if policy.require_same_harness_version
        && results
            .iter()
            .any(|result| result.run_context.harness_version != first.harness_version)
    {
        mismatches.push(GoldenContextMismatchV1::HarnessVersion);
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
        abstention_quality: optional_range(results.iter().map(|result| result.metrics.abstention_quality)),
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
    if count <= 1 { 0.0 } else { maximum - minimum }
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
    policy: &GoldenReproducibilityPolicyV1,
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

fn validate_unit(value: f32, label: &'static str) -> Result<(), GoldenReproducibilityErrorV1> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(GoldenReproducibilityErrorV1::InvalidPolicy(format!(
            "{label} must be finite and in [0,1]"
        )))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum GoldenReproducibilityErrorV1 {
    Qualification(ItQualificationErrorV1),
    Ledger(GoldenQualificationLedgerErrorV1),
    InvalidPolicy(String),
    UnknownCase(QualificationCaseKeyV1),
    LedgerResultMismatch(QualificationResultIdV1),
}

impl fmt::Display for GoldenReproducibilityErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "qualification assessment failed: {err}"),
            Self::Ledger(err) => write!(f, "golden binding ledger invalid: {err}"),
            Self::InvalidPolicy(message) => write!(f, "invalid reproducibility policy: {message}"),
            Self::UnknownCase(key) => write!(
                f,
                "unknown reproducibility case {} revision {}",
                key.id.0, key.revision
            ),
            Self::LedgerResultMismatch(id) => write!(
                f,
                "ledger binding does not match qualification result {}",
                id.0
            ),
        }
    }
}

impl Error for GoldenReproducibilityErrorV1 {}

impl From<ItQualificationErrorV1> for GoldenReproducibilityErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

impl From<GoldenQualificationLedgerErrorV1> for GoldenReproducibilityErrorV1 {
    fn from(value: GoldenQualificationLedgerErrorV1) -> Self {
        Self::Ledger(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden_binding_ledger::{
        record_and_ledger_derived_golden_result_v1, GoldenQualificationBindingLedgerV1,
    };
    use crate::golden_incidents_v2::{seed_golden_incidents_v2, GoldenIncidentCorpusV2};
    use crate::golden_metric_derivation::{
        derive_golden_qualification_metrics_v1, GoldenApplicabilityStatusV1,
        GoldenApplicabilityVerdictV1, GoldenFindingVerdictV1, GoldenPrivateEvaluationV1,
        GoldenRequiredFindingV1, GOLDEN_PRIVATE_EVALUATION_SCHEMA_V1,
    };
    use crate::golden_qualification_binding::golden_public_corpus_digest_v2;
    use crate::golden_run_protocol::{
        golden_grading_artifact_digest_v1, golden_solver_view_digest_v1,
        GoldenFindingDispositionV1, GoldenPresentedEvidenceSourceV1,
        GoldenPresentedEvidenceV1, GoldenRunTranscriptV1, GoldenSolverFindingV1,
        GoldenSolverSubmissionV1, GOLDEN_RUN_TRANSCRIPT_SCHEMA_V1,
        GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1,
    };
    use crate::it_qualification::{
        ItQualificationMatrixV1, ItQualificationResultV1, QualificationResultIdV1,
        QualificationRunContextV1, QualificationRunIdV1,
    };
    use std::collections::BTreeSet;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn register_case() -> (ItQualificationMatrixV1, GoldenIncidentCorpusV2, QualificationCaseKeyV1) {
        let corpus = seed_golden_incidents_v2().unwrap();
        let incident = corpus.cases[0].clone();
        let case = incident.qualification_case().unwrap();
        let key = case.key.clone();
        let mut matrix = ItQualificationMatrixV1::new();
        matrix.register_case(case).unwrap();
        (matrix, corpus, key)
    }

    fn add_run(
        matrix: &mut ItQualificationMatrixV1,
        ledger: &mut GoldenQualificationBindingLedgerV1,
        corpus: &GoldenIncidentCorpusV2,
        key: &QualificationCaseKeyV1,
        suffix: &str,
        confidence: f32,
        recorded_at: u64,
    ) {
        let incident = corpus
            .cases
            .iter()
            .find(|case| case.id == key.id.0 && case.revision == key.revision)
            .unwrap();
        let view = incident.solver_view();
        let run_id = QualificationRunIdV1(format!("repro-run-{suffix}"));
        let view_digest = golden_solver_view_digest_v1(&view).unwrap();
        let transcript = GoldenRunTranscriptV1 {
            schema_version: GOLDEN_RUN_TRANSCRIPT_SCHEMA_V1.into(),
            run_id: run_id.clone(),
            case_key: key.clone(),
            solver_view_digest: view_digest.clone(),
            started_at_unix_ms: recorded_at - 30,
            finished_at_unix_ms: recorded_at - 20,
            presented_evidence: view
                .initial_evidence
                .iter()
                .map(|evidence| GoldenPresentedEvidenceV1 {
                    evidence_id: evidence.id.clone(),
                    source: GoldenPresentedEvidenceSourceV1::InitialFixture,
                    private_artifact_digest: None,
                })
                .collect(),
            diagnostic_requests: Vec::new(),
        };
        let cited = view.initial_evidence[0].id.clone();
        let submission = GoldenSolverSubmissionV1 {
            schema_version: GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1.into(),
            run_id: run_id.clone(),
            case_key: key.clone(),
            solver_view_digest: view_digest.clone(),
            findings: vec![GoldenSolverFindingV1 {
                finding_id: format!("finding-{suffix}"),
                statement: "filesystem namespace pressure contributes".into(),
                disposition: GoldenFindingDispositionV1::SuspectedFault,
                confidence,
                evidence_ids: BTreeSet::from([cited.clone()]),
            }],
            final_confidence: confidence,
            abstention: None,
            submitted_at_unix_ms: recorded_at - 15,
        };
        let finding_id = submission.findings[0].finding_id.clone();
        let evaluation = GoldenPrivateEvaluationV1 {
            schema_version: GOLDEN_PRIVATE_EVALUATION_SCHEMA_V1.into(),
            run_id: run_id.clone(),
            case_key: key.clone(),
            solver_view_digest: view_digest,
            transcript_digest: transcript.digest().unwrap(),
            submission_digest: submission.digest().unwrap(),
            private_oracle_digest: digest('a'),
            evaluator_profile: "private-evaluator-v1".into(),
            evaluated_at_unix_ms: recorded_at - 10,
            required_findings: vec![GoldenRequiredFindingV1 {
                rubric_id: "filesystem-pressure".into(),
                satisfied_by_finding_id: Some(finding_id.clone()),
            }],
            finding_verdicts: vec![GoldenFindingVerdictV1 {
                finding_id: finding_id.clone(),
                correct: true,
                required_evidence_ids: BTreeSet::from([cited]),
            }],
            action_verdicts: Vec::new(),
            abstention_expected: None,
            applicability_verdicts: vec![GoldenApplicabilityVerdictV1 {
                item_id: "scope-1".into(),
                source_finding_id: finding_id,
                solver_status: GoldenApplicabilityStatusV1::Applicable,
                expected_status: GoldenApplicabilityStatusV1::Applicable,
            }],
        };
        let derived = derive_golden_qualification_metrics_v1(
            &view,
            &transcript,
            &submission,
            &evaluation,
        )
        .unwrap();
        let result = ItQualificationResultV1 {
            id: QualificationResultIdV1(format!("repro-result-{suffix}")),
            run_id,
            case_key: key.clone(),
            case_digest: matrix.case_digest(key).unwrap().into(),
            run_context: QualificationRunContextV1 {
                system_revision: "system-rev-1".into(),
                corpus_revision: golden_public_corpus_digest_v2(corpus).unwrap(),
                environment_digest: digest('b'),
                toolchain_digest: Some(digest('c')),
                model_profile: "support-eval".into(),
                harness_version: "golden-harness-v1".into(),
            },
            observed_at_unix_ms: recorded_at - 5,
            metrics: derived.metrics,
            evidence_artifact_digest: Some(
                golden_grading_artifact_digest_v1(&view, &transcript, &submission).unwrap(),
            ),
        };
        record_and_ledger_derived_golden_result_v1(
            matrix,
            ledger,
            corpus,
            &transcript,
            &submission,
            &evaluation,
            result,
            recorded_at,
        )
        .unwrap();
    }

    #[test]
    fn three_stable_ledger_backed_runs_are_reproducible() {
        let (mut matrix, corpus, key) = register_case();
        let mut ledger = GoldenQualificationBindingLedgerV1::new();
        add_run(&mut matrix, &mut ledger, &corpus, &key, "a", 0.95, 1000);
        add_run(&mut matrix, &mut ledger, &corpus, &key, "b", 0.95, 2000);
        add_run(&mut matrix, &mut ledger, &corpus, &key, "c", 0.95, 3000);
        let assessment = assess_golden_reproducibility_v1(
            &matrix,
            &ledger,
            &key,
            &GoldenReproducibilityPolicyV1::strict_deterministic_v1(),
        )
        .unwrap();
        assert_eq!(assessment.status, GoldenReproducibilityStatusV1::Reproducible);
        assert_eq!(assessment.considered_runs, 3);
        assert_eq!(assessment.distinct_run_ids, 3);
    }

    #[test]
    fn one_run_never_establishes_reproducibility() {
        let (mut matrix, corpus, key) = register_case();
        let mut ledger = GoldenQualificationBindingLedgerV1::new();
        add_run(&mut matrix, &mut ledger, &corpus, &key, "a", 0.95, 1000);
        let assessment = assess_golden_reproducibility_v1(
            &matrix,
            &ledger,
            &key,
            &GoldenReproducibilityPolicyV1::strict_deterministic_v1(),
        )
        .unwrap();
        assert_eq!(assessment.status, GoldenReproducibilityStatusV1::InsufficientRuns);
    }

    #[test]
    fn metric_instability_is_not_hidden_by_average_performance() {
        let (mut matrix, corpus, key) = register_case();
        let mut ledger = GoldenQualificationBindingLedgerV1::new();
        add_run(&mut matrix, &mut ledger, &corpus, &key, "a", 0.99, 1000);
        add_run(&mut matrix, &mut ledger, &corpus, &key, "b", 0.75, 2000);
        add_run(&mut matrix, &mut ledger, &corpus, &key, "c", 0.99, 3000);
        let mut policy = GoldenReproducibilityPolicyV1::strict_deterministic_v1();
        policy.max_calibration_error_range = 0.01;
        let assessment = assess_golden_reproducibility_v1(
            &matrix,
            &ledger,
            &key,
            &policy,
        )
        .unwrap();
        assert!(matches!(
            assessment.status,
            GoldenReproducibilityStatusV1::RunMetricsFailed
                | GoldenReproducibilityStatusV1::MetricInstability
        ));
    }
}
