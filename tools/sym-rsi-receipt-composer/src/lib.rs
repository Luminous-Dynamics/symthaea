#![forbid(unsafe_code)]

//! In-memory composition of canonical benchmark-run receipts for RSI.
//!
//! This crate deliberately does not define a second persisted receipt schema. Each input
//! is validated by the shared benchmark-run receipt validator, then projected into a
//! complete task-universe view. Accounting validation is not producer authentication,
//! runner isolation, grader qualification, or independent regrade.

use benchmark_run_receipt_validator::{
    VerifiedReceipt, canonical_json_bytes, sha256_hex, verify_receipt_json,
};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const EVIDENCE_CLAIM_CEILING: &str =
    "accounting-validated only; producer authentication, runner isolation, grader truth, \
     and independent regrade are not established by this composer";

#[derive(Debug, Clone, PartialEq)]
pub enum ScoreTransform {
    /// Map a frozen numeric grade range to [0, 1].
    Numeric {
        minimum: f64,
        maximum: f64,
        higher_is_better: bool,
    },
    /// Map a binary grade with explicitly declared outcome values.
    Binary {
        true_score: f64,
        false_score: f64,
    },
    /// Map each observed categorical value through an explicit frozen lookup.
    Categorical {
        scores: BTreeMap<String, f64>,
    },
}

impl ScoreTransform {
    fn expected_grade_kind(&self) -> &'static str {
        match self {
            Self::Numeric { .. } => "numeric",
            Self::Binary { .. } => "binary",
            Self::Categorical { .. } => "categorical",
        }
    }

    fn validate(&self) -> Result<(), ComposerError> {
        let valid_score = |score: f64| score.is_finite() && (0.0..=1.0).contains(&score);
        match self {
            Self::Numeric {
                minimum, maximum, ..
            } if !minimum.is_finite() || !maximum.is_finite() || minimum >= maximum => {
                Err(ComposerError::InvalidPlan(
                    "numeric transform requires finite minimum < maximum".into(),
                ))
            }
            Self::Binary {
                true_score,
                false_score,
            } if !valid_score(*true_score) || !valid_score(*false_score) => {
                Err(ComposerError::InvalidPlan(
                    "binary transform scores must be finite and in [0, 1]".into(),
                ))
            }
            Self::Categorical { scores }
                if scores.is_empty()
                    || scores
                        .iter()
                        .any(|(label, score)| label.trim().is_empty() || !valid_score(*score)) =>
            {
                Err(ComposerError::InvalidPlan(
                    "categorical transform requires non-empty labels and scores in [0, 1]"
                        .into(),
                ))
            }
            _ => Ok(()),
        }
    }

    fn apply(&self, grade: &Value) -> Result<f64, ComposerError> {
        let grade_object = grade.as_object().ok_or_else(|| {
            ComposerError::InvalidGrade("task grade must be an object".into())
        })?;
        let kind = required_string(grade_object, "kind", "task grade")?;
        if kind != self.expected_grade_kind() {
            return Err(ComposerError::GradeKindMismatch {
                expected: self.expected_grade_kind().into(),
                actual: kind.into(),
            });
        }
        let value = required_value(grade_object, "value", "task grade")?;
        let score = match self {
            Self::Numeric {
                minimum,
                maximum,
                higher_is_better,
            } => {
                let raw = value.as_f64().ok_or_else(|| {
                    ComposerError::InvalidGrade("numeric grade value is not a number".into())
                })?;
                if !raw.is_finite() || raw < *minimum || raw > *maximum {
                    return Err(ComposerError::InvalidGrade(
                        "numeric grade lies outside its frozen transform range".into(),
                    ));
                }
                let normalized = (raw - minimum) / (maximum - minimum);
                if *higher_is_better {
                    normalized
                } else {
                    1.0 - normalized
                }
            }
            Self::Binary {
                true_score,
                false_score,
            } => match value.as_bool() {
                Some(true) => *true_score,
                Some(false) => *false_score,
                None => {
                    return Err(ComposerError::InvalidGrade(
                        "binary grade value is not a boolean".into(),
                    ));
                }
            },
            Self::Categorical { scores } => {
                let label = value.as_str().ok_or_else(|| {
                    ComposerError::InvalidGrade("categorical grade value is not a string".into())
                })?;
                *scores.get(label).ok_or_else(|| {
                    ComposerError::UnmappedCategory(label.to_owned())
                })?
            }
        };
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Err(ComposerError::InvalidGrade(
                "normalized score is not finite or outside [0, 1]".into(),
            ));
        }
        Ok(score)
    }

    fn manifest_value(&self) -> Value {
        match self {
            Self::Numeric {
                minimum,
                maximum,
                higher_is_better,
            } => json!({
                "kind": "numeric",
                "minimum": minimum,
                "maximum": maximum,
                "higher_is_better": higher_is_better
            }),
            Self::Binary {
                true_score,
                false_score,
            } => json!({
                "kind": "binary",
                "true_score": true_score,
                "false_score": false_score
            }),
            Self::Categorical { scores } => json!({
                "kind": "categorical",
                "scores": scores
            }),
        }
    }
}

/// One capability dimension and its canonical receipt source.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricProjection {
    pub metric_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub benchmark_revision: String,
    pub dataset_manifest_sha256: String,
    pub scoring_formula_id: String,
    pub grader_version: String,
    pub transform: ScoreTransform,
    /// Minimum fraction of eligible tasks that must have a grade before this metric can
    /// enter a candidate capability vector. Missing rows are never removed to satisfy it.
    pub minimum_graded_coverage: f64,
}

impl MetricProjection {
    fn validate(&self) -> Result<(), ComposerError> {
        validate_identifier(&self.metric_id, "metric_id")?;
        validate_identifier(&self.benchmark_id, "benchmark_id")?;
        validate_identifier(&self.benchmark_version, "benchmark_version")?;
        validate_identifier(&self.benchmark_revision, "benchmark_revision")?;
        validate_sha256(&self.dataset_manifest_sha256, "dataset_manifest_sha256")?;
        validate_identifier(&self.scoring_formula_id, "scoring_formula_id")?;
        validate_identifier(&self.grader_version, "grader_version")?;
        self.transform.validate()?;
        if !self.minimum_graded_coverage.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_graded_coverage)
        {
            return Err(ComposerError::InvalidPlan(
                "minimum_graded_coverage must be finite and in [0, 1]".into(),
            ));
        }
        Ok(())
    }
}

/// Frozen in-memory experiment plan. The preregistration digest must identify an
/// independently frozen plan; this type cannot authenticate its origin.
#[derive(Debug, Clone, PartialEq)]
pub struct CompositionPlan {
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub metrics: BTreeMap<String, MetricProjection>,
    /// Exact eligible task ID -> independent cluster ID assignment.
    pub cluster_by_task: BTreeMap<String, String>,
}

impl CompositionPlan {
    pub fn validate(&self) -> Result<(), ComposerError> {
        validate_identifier(&self.experiment_id, "experiment_id")?;
        validate_sha256(&self.preregistration_digest, "preregistration_digest")?;
        if self.metrics.is_empty() {
            return Err(ComposerError::InvalidPlan(
                "at least one metric projection is required".into(),
            ));
        }
        let mut dimension_ids = BTreeSet::new();
        let mut metric_source_ids = BTreeSet::new();
        for (metric_id, projection) in &self.metrics {
            if metric_id != &projection.metric_id {
                return Err(ComposerError::InvalidPlan(
                    "metric map key must equal projection.metric_id".into(),
                ));
            }
            projection.validate()?;
            let source_identity = (
                projection.benchmark_id.clone(),
                projection.benchmark_version.clone(),
                projection.benchmark_revision.clone(),
                projection.dataset_manifest_sha256.clone(),
                projection.scoring_formula_id.clone(),
                projection.grader_version.clone(),
            );
            if !metric_source_ids.insert(source_identity) {
                return Err(ComposerError::InvalidPlan(
                    "one frozen benchmark/scoring/grader source cannot be reused as multiple metrics"
                        .into(),
                ));
            }
            for dimension_id in [
                metric_id.clone(),
                format!("{metric_id}.graded_coverage"),
                format!("{metric_id}.coverage_adjusted_yield"),
            ] {
                if !dimension_ids.insert(dimension_id.clone()) {
                    return Err(ComposerError::InvalidPlan(format!(
                        "derived capability dimension ID collision: {dimension_id}"
                    )));
                }
            }
        }
        if self.cluster_by_task.is_empty()
            || self.cluster_by_task.iter().any(|(task_id, cluster_id)| {
                task_id.trim().is_empty()
                    || cluster_id.trim().is_empty()
                    || task_id.chars().any(char::is_control)
                    || cluster_id.chars().any(char::is_control)
            })
        {
            return Err(ComposerError::InvalidPlan(
                "cluster map must have non-empty, control-free task and cluster IDs".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TaskMetricOutcome {
    pub task_id: String,
    pub input_sha256: String,
    pub attempt_state: String,
    /// None is retained for not-attempted and other non-terminal states.
    pub terminal_disposition: Option<String>,
    pub grade_kind: Option<String>,
    pub raw_grade: Option<Value>,
    pub normalized_score: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricSummary {
    pub eligible_count: usize,
    pub graded_count: usize,
    /// Mean over tasks with grades only. Interpret alongside graded_coverage.
    pub graded_mean: Option<f64>,
    /// graded_count / eligible_count.
    pub graded_coverage: f64,
    /// Sum of normalized grades / eligible_count. Grade-less eligible tasks contribute
    /// zero to this derived yield by formula, without claiming an observed zero grade.
    /// It is not the benchmark owner's official score.
    pub coverage_adjusted_yield: f64,
    /// Non-terminal tasks are counted by their attempt state (for example not_attempted).
    pub disposition_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricEvidence {
    pub projection: MetricProjection,
    pub summary: MetricSummary,
    pub tasks: BTreeMap<String, TaskMetricOutcome>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComposedCandidateEvidence {
    candidate_system_identity_digest: String,
    composite_context_digest: String,
    evidence_set_digest: String,
    eligible_task_ids: Vec<String>,
    cluster_by_task: BTreeMap<String, String>,
    metrics: BTreeMap<String, MetricEvidence>,
}

impl ComposedCandidateEvidence {
    pub fn candidate_system_identity_digest(&self) -> &str {
        &self.candidate_system_identity_digest
    }

    pub fn composite_context_digest(&self) -> &str {
        &self.composite_context_digest
    }

    pub fn evidence_set_digest(&self) -> &str {
        &self.evidence_set_digest
    }

    pub fn eligible_task_ids(&self) -> &[String] {
        &self.eligible_task_ids
    }

    pub fn cluster_by_task(&self) -> &BTreeMap<String, String> {
        &self.cluster_by_task
    }

    pub fn metric(&self, metric_id: &str) -> Option<&MetricEvidence> {
        self.metrics.get(metric_id)
    }

    pub fn metrics(&self) -> impl Iterator<Item = (&str, &MetricEvidence)> {
        self.metrics
            .iter()
            .map(|(metric_id, evidence)| (metric_id.as_str(), evidence))
    }

    /// Produces dimensions suitable for an RSI capability vector. Each metric contributes
    /// three non-collapsed dimensions: graded-only mean, graded coverage, and
    /// coverage-adjusted yield. A metric below its preregistered coverage floor cannot
    /// enter the vector.
    pub fn capability_dimensions(&self) -> Result<BTreeMap<String, f64>, ComposerError> {
        let mut dimensions = BTreeMap::new();
        for (metric_id, evidence) in &self.metrics {
            if evidence.summary.graded_coverage
                < evidence.projection.minimum_graded_coverage
            {
                return Err(ComposerError::CoverageBelowMinimum {
                    metric_id: metric_id.clone(),
                    observed: evidence.summary.graded_coverage,
                    minimum: evidence.projection.minimum_graded_coverage,
                });
            }
            let mean = evidence.summary.graded_mean.ok_or_else(|| {
                ComposerError::NoGradedOutcomes(metric_id.clone())
            })?;
            dimensions.insert(metric_id.clone(), mean);
            dimensions.insert(
                format!("{metric_id}.graded_coverage"),
                evidence.summary.graded_coverage,
            );
            dimensions.insert(
                format!("{metric_id}.coverage_adjusted_yield"),
                evidence.summary.coverage_adjusted_yield,
            );
        }
        Ok(dimensions)
    }

    /// Returns complete per-task score vectors only when every eligible task has a grade
    /// for every metric. It never silently complete-case-filters the task universe.
    pub fn complete_case_scores(&self) -> Result<BTreeMap<String, BTreeMap<String, f64>>, ComposerError> {
        let mut rows = BTreeMap::new();
        for task_id in &self.eligible_task_ids {
            let mut scores = BTreeMap::new();
            for (metric_id, evidence) in &self.metrics {
                let outcome = evidence
                    .tasks
                    .get(task_id)
                    .ok_or_else(|| ComposerError::TaskUniverseMismatch)?;
                let score = outcome.normalized_score.ok_or_else(|| {
                    ComposerError::IncompleteMetricCoverage {
                        metric_id: metric_id.clone(),
                        task_id: task_id.clone(),
                        disposition: outcome
                            .terminal_disposition
                            .clone()
                            .unwrap_or_else(|| outcome.attempt_state.clone()),
                    }
                })?;
                scores.insert(metric_id.clone(), score);
            }
            rows.insert(task_id.clone(), scores);
        }
        Ok(rows)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterDeltaSummary {
    pub paired_task_count: usize,
    pub mean_delta: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedMetricSummary {
    pub eligible_count: usize,
    pub parent_graded_count: usize,
    pub candidate_graded_count: usize,
    pub jointly_graded_count: usize,
    pub parent_only_graded_count: usize,
    pub candidate_only_graded_count: usize,
    pub neither_graded_count: usize,
    pub parent_graded_coverage: f64,
    pub candidate_graded_coverage: f64,
    pub mean_delta_on_jointly_graded_tasks: Option<f64>,
    pub by_cluster: BTreeMap<String, ClusterDeltaSummary>,
}

/// Compare two complete candidate evidence compositions. Missing outcomes remain counted;
/// mean delta is explicitly conditional on jointly graded tasks. No p-value or promotion
/// decision is emitted.
pub fn compare_candidates(
    parent: &ComposedCandidateEvidence,
    candidate: &ComposedCandidateEvidence,
) -> Result<BTreeMap<String, PairedMetricSummary>, ComposerError> {
    if parent.composite_context_digest != candidate.composite_context_digest {
        return Err(ComposerError::ContextMismatch);
    }
    if parent.candidate_system_identity_digest == candidate.candidate_system_identity_digest {
        return Err(ComposerError::SameCandidate);
    }
    if parent.eligible_task_ids != candidate.eligible_task_ids
        || parent.cluster_by_task != candidate.cluster_by_task
    {
        return Err(ComposerError::TaskUniverseMismatch);
    }
    if parent.metrics.keys().ne(candidate.metrics.keys()) {
        return Err(ComposerError::MetricSetMismatch);
    }

    let eligible_count = parent.eligible_task_ids.len();
    let mut comparisons = BTreeMap::new();

    for (metric_id, parent_metric) in &parent.metrics {
        let candidate_metric = candidate
            .metrics
            .get(metric_id)
            .ok_or(ComposerError::MetricSetMismatch)?;

        let mut parent_graded_count = 0usize;
        let mut candidate_graded_count = 0usize;
        let mut jointly_graded_count = 0usize;
        let mut parent_only_graded_count = 0usize;
        let mut candidate_only_graded_count = 0usize;
        let mut neither_graded_count = 0usize;
        let mut delta_sum = 0.0_f64;
        let mut cluster_accumulators: BTreeMap<String, (f64, usize)> = BTreeMap::new();

        for task_id in &parent.eligible_task_ids {
            let parent_task = parent_metric
                .tasks
                .get(task_id)
                .ok_or(ComposerError::TaskUniverseMismatch)?;
            let candidate_task = candidate_metric
                .tasks
                .get(task_id)
                .ok_or(ComposerError::TaskUniverseMismatch)?;

            if parent_task.input_sha256 != candidate_task.input_sha256 {
                return Err(ComposerError::TaskInputMismatch {
                    metric_id: metric_id.clone(),
                    task_id: task_id.clone(),
                });
            }

            match (
                parent_task.normalized_score,
                candidate_task.normalized_score,
            ) {
                (Some(parent_score), Some(candidate_score)) => {
                    parent_graded_count += 1;
                    candidate_graded_count += 1;
                    jointly_graded_count += 1;
                    let delta = candidate_score - parent_score;
                    delta_sum += delta;
                    let cluster_id = parent
                        .cluster_by_task
                        .get(task_id)
                        .ok_or(ComposerError::ClusterMapMismatch)?;
                    let entry = cluster_accumulators
                        .entry(cluster_id.clone())
                        .or_insert((0.0, 0));
                    entry.0 += delta;
                    entry.1 += 1;
                }
                (Some(_), None) => {
                    parent_graded_count += 1;
                    parent_only_graded_count += 1;
                }
                (None, Some(_)) => {
                    candidate_graded_count += 1;
                    candidate_only_graded_count += 1;
                }
                (None, None) => {
                    neither_graded_count += 1;
                }
            }
        }

        let by_cluster = cluster_accumulators
            .into_iter()
            .map(|(cluster_id, (sum, count))| {
                (
                    cluster_id,
                    ClusterDeltaSummary {
                        paired_task_count: count,
                        mean_delta: sum / count as f64,
                    },
                )
            })
            .collect();

        comparisons.insert(
            metric_id.clone(),
            PairedMetricSummary {
                eligible_count,
                parent_graded_count,
                candidate_graded_count,
                jointly_graded_count,
                parent_only_graded_count,
                candidate_only_graded_count,
                neither_graded_count,
                parent_graded_coverage: parent_graded_count as f64 / eligible_count as f64,
                candidate_graded_coverage: candidate_graded_count as f64 / eligible_count as f64,
                mean_delta_on_jointly_graded_tasks: (jointly_graded_count > 0)
                    .then_some(delta_sum / jointly_graded_count as f64),
                by_cluster,
            },
        );
    }

    Ok(comparisons)
}

#[derive(Debug, Clone, PartialEq)]
struct ParsedRun {
    verified: VerifiedReceipt,
    projection: MetricProjection,
    eligible_task_ids: Vec<String>,
    universe_digest: String,
    eligible_digest: String,
    candidate_identity: Value,
    compatibility_context: Value,
    metric_context: Value,
    tasks: BTreeMap<String, TaskMetricOutcome>,
}

pub fn compose_candidate(
    plan: &CompositionPlan,
    receipt_bytes_by_metric: &BTreeMap<String, Vec<u8>>,
) -> Result<ComposedCandidateEvidence, ComposerError> {
    plan.validate()?;
    if receipt_bytes_by_metric.len() != plan.metrics.len()
        || receipt_bytes_by_metric
            .keys()
            .ne(plan.metrics.keys())
    {
        return Err(ComposerError::MetricReceiptSetMismatch);
    }

    let mut parsed_runs = BTreeMap::new();
    let mut seen_receipt_digests = BTreeSet::new();
    let mut shared_eligible_task_ids: Option<Vec<String>> = None;
    let mut shared_universe_digest: Option<String> = None;
    let mut shared_eligible_digest: Option<String> = None;
    let mut shared_candidate_identity: Option<Value> = None;
    let mut shared_compatibility_context: Option<Value> = None;
    let mut shared_input_digests: Option<BTreeMap<String, String>> = None;

    for (metric_id, projection) in &plan.metrics {
        let bytes = receipt_bytes_by_metric
            .get(metric_id)
            .ok_or_else(|| ComposerError::MissingMetricReceipt(metric_id.clone()))?;
        let parsed = parse_run(bytes, projection)?;
        if !seen_receipt_digests.insert(parsed.verified.canonical_receipt_sha256.clone()) {
            return Err(ComposerError::DuplicateEvidenceDigest);
        }

        match &shared_eligible_task_ids {
            Some(task_ids) if task_ids != &parsed.eligible_task_ids => {
                return Err(ComposerError::TaskUniverseMismatch);
            }
            None => shared_eligible_task_ids = Some(parsed.eligible_task_ids.clone()),
            _ => {}
        }
        match &shared_universe_digest {
            Some(digest) if digest != &parsed.universe_digest => {
                return Err(ComposerError::TaskUniverseMismatch);
            }
            None => shared_universe_digest = Some(parsed.universe_digest.clone()),
            _ => {}
        }
        match &shared_eligible_digest {
            Some(digest) if digest != &parsed.eligible_digest => {
                return Err(ComposerError::TaskUniverseMismatch);
            }
            None => shared_eligible_digest = Some(parsed.eligible_digest.clone()),
            _ => {}
        }
        match &shared_candidate_identity {
            Some(identity) if identity != &parsed.candidate_identity => {
                return Err(ComposerError::CandidateIdentityMismatch);
            }
            None => shared_candidate_identity = Some(parsed.candidate_identity.clone()),
            _ => {}
        }
        match &shared_compatibility_context {
            Some(context) if context != &parsed.compatibility_context => {
                return Err(ComposerError::IncompatibleMetricRunContext);
            }
            None => shared_compatibility_context = Some(parsed.compatibility_context.clone()),
            _ => {}
        }
        let input_digests: BTreeMap<String, String> = parsed
            .tasks
            .iter()
            .map(|(task_id, outcome)| (task_id.clone(), outcome.input_sha256.clone()))
            .collect();
        match &shared_input_digests {
            Some(expected) if expected != &input_digests => {
                return Err(ComposerError::TaskInputUniverseMismatch);
            }
            None => shared_input_digests = Some(input_digests),
            _ => {}
        }

        parsed_runs.insert(metric_id.clone(), parsed);
    }

    let eligible_task_ids = shared_eligible_task_ids.ok_or(ComposerError::TaskUniverseMismatch)?;
    if eligible_task_ids.is_empty() {
        return Err(ComposerError::TaskUniverseMismatch);
    }
    let eligible_set: BTreeSet<String> = eligible_task_ids.iter().cloned().collect();
    let cluster_set: BTreeSet<String> = plan.cluster_by_task.keys().cloned().collect();
    if eligible_set != cluster_set {
        return Err(ComposerError::ClusterMapMismatch);
    }

    let mut metric_contexts = Map::new();
    let mut evidence_digests = Map::new();
    let mut metrics = BTreeMap::new();

    for (metric_id, parsed) in parsed_runs {
        metric_contexts.insert(
            metric_id.clone(),
            json!({
                "receipt_context": parsed.metric_context,
                "transform": parsed.projection.transform.manifest_value(),
                "minimum_graded_coverage": parsed.projection.minimum_graded_coverage
            }),
        );
        evidence_digests.insert(
            metric_id.clone(),
            Value::String(parsed.verified.canonical_receipt_sha256.clone()),
        );
        let summary = summarize(&parsed.tasks, eligible_task_ids.len());
        metrics.insert(
            metric_id.clone(),
            MetricEvidence {
                projection: parsed.projection,
                summary,
                tasks: parsed.tasks,
            },
        );
    }

    let context_manifest = json!({
        "experiment_id": plan.experiment_id.clone(),
        "preregistration_digest": plan.preregistration_digest.clone(),
        "compatibility_context": shared_compatibility_context,
        "metric_contexts": Value::Object(metric_contexts),
        "eligible_task_ids": eligible_task_ids.clone(),
        "cluster_by_task": plan.cluster_by_task.clone()
    });
    let evidence_manifest = Value::Object(evidence_digests);
    let context_digest = digest_value(&context_manifest)?;
    let evidence_set_digest = digest_value(&evidence_manifest)?;
    let candidate_identity = shared_candidate_identity.ok_or(
        ComposerError::CandidateIdentityMismatch,
    )?;
    let candidate_system_identity_digest = digest_value(&candidate_identity)?;

    Ok(ComposedCandidateEvidence {
        candidate_system_identity_digest,
        composite_context_digest: context_digest,
        evidence_set_digest,
        eligible_task_ids,
        cluster_by_task: plan.cluster_by_task.clone(),
        metrics,
    })
}

fn parse_run(bytes: &[u8], projection: &MetricProjection) -> Result<ParsedRun, ComposerError> {
    // Structural and semantic validation is owned by the existing canonical validator.
    let verified = verify_receipt_json(bytes)
        .map_err(|error| ComposerError::ReceiptValidation(error.to_string()))?;
    let value = &verified.value;

    let benchmark = required_object(value, "benchmark", "receipt")?;
    let benchmark_id = required_string(benchmark, "id", "benchmark")?;
    let benchmark_version = required_string(benchmark, "version", "benchmark")?;
    let benchmark_revision = required_string(benchmark, "revision", "benchmark")?;
    let dataset_manifest_sha256 =
        required_string(benchmark, "dataset_manifest_sha256", "benchmark")?;
    let scoring_formula_id =
        required_string(benchmark, "scoring_formula_id", "benchmark")?;
    let grader_version = required_string(benchmark, "grader_version", "benchmark")?;
    if benchmark_id != projection.benchmark_id
        || benchmark_version != projection.benchmark_version
        || benchmark_revision != projection.benchmark_revision
        || dataset_manifest_sha256 != projection.dataset_manifest_sha256
        || scoring_formula_id != projection.scoring_formula_id
        || grader_version != projection.grader_version
    {
        return Err(ComposerError::MetricSourceMismatch(
            projection.metric_id.clone(),
        ));
    }

    let grading_mode = required_string_from_value(value, "grading_mode", "receipt")?;
    if grading_mode != projection.transform.expected_grade_kind() {
        return Err(ComposerError::GradeKindMismatch {
            expected: projection.transform.expected_grade_kind().into(),
            actual: grading_mode.into(),
        });
    }

    let comparison = required_object(value, "comparison", "receipt")?;
    let _group_id = required_string(comparison, "group_id", "comparison")?;
    let _protocol_digest = required_string(comparison, "protocol_sha256", "comparison")?;

    let candidate_identity = candidate_identity(value)?;
    let compatibility_context = compatibility_context(value)?;
    let metric_context = metric_context(value)?;

    let task_universe = required_array(value, "task_universe", "receipt")?;
    let mut eligible_task_ids = Vec::new();
    for task in task_universe {
        let task_id = required_string_from_value(task, "task_id", "task_universe item")?;
        let eligibility = required_string_from_value(task, "eligibility", "task_universe item")?;
        if eligibility == "eligible" {
            eligible_task_ids.push(task_id.to_owned());
        }
    }
    eligible_task_ids.sort();
    if eligible_task_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ComposerError::TaskUniverseMismatch);
    }

    let universe_digest =
        required_string_from_value(value, "task_universe_sha256", "receipt")?.to_owned();
    let eligible_digest =
        required_string_from_value(value, "eligible_task_set_sha256", "receipt")?.to_owned();

    let task_results = required_array(value, "task_results", "receipt")?;
    let mut tasks = BTreeMap::new();
    for row in task_results {
        let task_id =
            required_string_from_value(row, "task_id", "task_result")?.to_owned();
        let input_sha256 =
            required_string_from_value(row, "input_sha256", "task_result")?.to_owned();
        let attempt_state =
            required_string_from_value(row, "attempt_state", "task_result")?.to_owned();
        let terminal_disposition = row
            .get("terminal_disposition")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let grade = row.get("grade").filter(|grade| !grade.is_null()).cloned();

        let (grade_kind, normalized_score) = match grade.as_ref() {
            Some(grade_value) => {
                if !matches!(
                    terminal_disposition.as_deref(),
                    Some("passed") | Some("failed")
                ) || attempt_state != "terminal"
                {
                    return Err(ComposerError::InvalidGrade(
                        "a scored task must have a terminal passed/failed disposition".into(),
                    ));
                }
                let kind = required_string_from_value(grade_value, "kind", "task grade")?.to_owned();
                let score = projection.transform.apply(grade_value)?;
                (Some(kind), Some(score))
            }
            None => (None, None),
        };

        let outcome = TaskMetricOutcome {
            task_id: task_id.clone(),
            input_sha256,
            attempt_state,
            terminal_disposition,
            grade_kind,
            raw_grade: grade,
            normalized_score,
        };
        if tasks.insert(task_id, outcome).is_some() {
            return Err(ComposerError::TaskUniverseMismatch);
        }
    }

    let eligible_set: BTreeSet<&str> = eligible_task_ids.iter().map(String::as_str).collect();
    let result_set: BTreeSet<&str> = tasks.keys().map(String::as_str).collect();
    if eligible_set != result_set {
        return Err(ComposerError::TaskUniverseMismatch);
    }

    Ok(ParsedRun {
        verified,
        projection: projection.clone(),
        eligible_task_ids,
        universe_digest,
        eligible_digest,
        candidate_identity,
        compatibility_context,
        metric_context,
        tasks,
    })
}

fn summarize(
    tasks: &BTreeMap<String, TaskMetricOutcome>,
    eligible_count: usize,
) -> MetricSummary {
    let mut score_sum = 0.0_f64;
    let mut graded_count = 0usize;
    let mut disposition_counts = BTreeMap::new();
    for task in tasks.values() {
        let label = task
            .terminal_disposition
            .as_deref()
            .unwrap_or(task.attempt_state.as_str());
        *disposition_counts.entry(label.to_owned()).or_insert(0) += 1;
        if let Some(score) = task.normalized_score {
            score_sum += score;
            graded_count += 1;
        }
    }
    let graded_mean = (graded_count > 0).then_some(score_sum / graded_count as f64);
    MetricSummary {
        eligible_count,
        graded_count,
        graded_mean,
        graded_coverage: graded_count as f64 / eligible_count as f64,
        coverage_adjusted_yield: score_sum / eligible_count as f64,
        disposition_counts,
    }
}

fn candidate_identity(receipt: &Value) -> Result<Value, ComposerError> {
    let system = required_object(receipt, "system", "receipt")?;
    Ok(json!({
        "name": required_value(system, "name", "system")?,
        "arm": required_value(system, "arm", "system")?,
        "version": required_value(system, "version", "system")?,
        "source_revision": required_value(system, "source_revision", "system")?,
        "executable_sha256": required_value(system, "executable_sha256", "system")?,
        "model_id": required_value(system, "model_id", "system")?,
        "config_sha256": required_value(system, "config_sha256", "system")?,
        "prompt_sha256": required_value(system, "prompt_sha256", "system")?
    }))
}

fn compatibility_context(receipt: &Value) -> Result<Value, ComposerError> {
    let system = required_object(receipt, "system", "receipt")?;
    let comparison = required_object(receipt, "comparison", "receipt")?;
    Ok(json!({
        "system_arm": required_value(system, "arm", "system")?,
        "target_scope": required_value_from_value(receipt, "target_scope", "receipt")?,
        "runner": required_value_from_value(receipt, "runner", "receipt")?,
        "comparison": {
            "group_id": required_value(comparison, "group_id", "comparison")?,
            "protocol_sha256": required_value(comparison, "protocol_sha256", "comparison")?,
            "eligible_task_set_sha256": required_value(comparison, "eligible_task_set_sha256", "comparison")?
        },
        "task_universe_sha256": required_value_from_value(receipt, "task_universe_sha256", "receipt")?,
        "eligible_task_set_sha256": required_value_from_value(receipt, "eligible_task_set_sha256", "receipt")?,
        "budget_caps": required_value_from_value(receipt, "budget_caps", "receipt")
    }))
}

fn metric_context(receipt: &Value) -> Result<Value, ComposerError> {
    let benchmark = required_object(receipt, "benchmark", "receipt")?;
    Ok(json!({
        "benchmark_id": required_value(benchmark, "id", "benchmark")?,
        "benchmark_version": required_value(benchmark, "version", "benchmark")?,
        "benchmark_revision": required_value(benchmark, "revision", "benchmark")?,
        "dataset_manifest_sha256": required_value(benchmark, "dataset_manifest_sha256", "benchmark")?,
        "access_class": required_value(benchmark, "access_class", "benchmark")?,
        "scoring_formula_id": required_value(benchmark, "scoring_formula_id", "benchmark")?,
        "grader_version": required_value(benchmark, "grader_version", "benchmark")?,
        "grading_mode": required_value_from_value(receipt, "grading_mode", "receipt")?
    }))
}

fn digest_value(value: &Value) -> Result<String, ComposerError> {
    let bytes = canonical_json_bytes(value)
        .map_err(|error| ComposerError::ReceiptValidation(error.to_string()))?;
    Ok(sha256_hex(&bytes))
}

fn validate_identifier(value: &str, label: &str) -> Result<(), ComposerError> {
    let valid = !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        && !value.starts_with('.')
        && !value.ends_with('.');
    if !valid {
        return Err(ComposerError::InvalidPlan(format!(
            "{label} must use only ASCII letters, digits, '.', '-' or '_' and cannot start/end with a dot"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<(), ComposerError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ComposerError::InvalidPlan(format!(
            "{label} must be lowercase SHA-256 hex"
        )));
    }
    Ok(())
}

fn required_object<'a>(
    value: &'a Value,
    key: &str,
    context: &str,
) -> Result<&'a Map<String, Value>, ComposerError> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ComposerError::InvalidReceiptShape(format!("{context}.{key} must be an object"))
        })
}

fn required_array<'a>(
    value: &'a Value,
    key: &str,
    context: &str,
) -> Result<&'a Vec<Value>, ComposerError> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ComposerError::InvalidReceiptShape(format!("{context}.{key} must be an array"))
        })
}

fn required_value(
    value: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<Value, ComposerError> {
    value
        .get(key)
        .cloned()
        .ok_or_else(|| {
            ComposerError::InvalidReceiptShape(format!("{context}.{key} is required"))
        })
}

fn required_value_from_value(
    value: &Value,
    key: &str,
    context: &str,
) -> Result<Value, ComposerError> {
    value.get(key).cloned().ok_or_else(|| {
        ComposerError::InvalidReceiptShape(format!("{context}.{key} is required"))
    })
}

fn required_string_from_value<'a>(
    value: &'a Value,
    key: &str,
    context: &str,
) -> Result<&'a str, ComposerError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|string| !string.trim().is_empty())
        .ok_or_else(|| {
            ComposerError::InvalidReceiptShape(format!(
                "{context}.{key} must be a non-empty string"
            ))
        })
}

fn required_string<'a>(
    value: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a str, ComposerError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|string| !string.trim().is_empty())
        .ok_or_else(|| {
            ComposerError::InvalidReceiptShape(format!("{context}.{key} must be a non-empty string"))
        })
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComposerError {
    InvalidPlan(String),
    MissingMetricReceipt(String),
    MetricReceiptSetMismatch,
    MetricSourceMismatch(String),
    ReceiptValidation(String),
    InvalidReceiptShape(String),
    InvalidGrade(String),
    GradeKindMismatch { expected: String, actual: String },
    UnmappedCategory(String),
    CandidateIdentityMismatch,
    IncompatibleMetricRunContext,
    TaskUniverseMismatch,
    TaskInputUniverseMismatch,
    ClusterMapMismatch,
    DuplicateEvidenceDigest,
    CoverageBelowMinimum {
        metric_id: String,
        observed: f64,
        minimum: f64,
    },
    NoGradedOutcomes(String),
    IncompleteMetricCoverage {
        metric_id: String,
        task_id: String,
        disposition: String,
    },
    ContextMismatch,
    SameCandidate,
    MetricSetMismatch,
    TaskInputMismatch { metric_id: String, task_id: String },
}

impl fmt::Display for ComposerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(message) => write!(f, "invalid composition plan: {message}"),
            Self::MissingMetricReceipt(metric) => {
                write!(f, "missing receipt for metric {metric}")
            }
            Self::MetricReceiptSetMismatch => {
                write!(f, "receipt metric keys must exactly match the frozen metric plan")
            }
            Self::MetricSourceMismatch(metric) => {
                write!(f, "receipt benchmark identity does not match metric {metric}")
            }
            Self::ReceiptValidation(message) => write!(f, "receipt validation failed: {message}"),
            Self::InvalidReceiptShape(message) => write!(f, "invalid receipt shape: {message}"),
            Self::InvalidGrade(message) => write!(f, "invalid task grade: {message}"),
            Self::GradeKindMismatch { expected, actual } => {
                write!(f, "grade kind mismatch: expected {expected}, received {actual}")
            }
            Self::UnmappedCategory(label) => write!(f, "unmapped categorical grade: {label}"),
            Self::CandidateIdentityMismatch => {
                write!(f, "metric receipts do not share one candidate system identity")
            }
            Self::IncompatibleMetricRunContext => {
                write!(f, "metric receipts have incompatible cohort, runner, protocol or budget context")
            }
            Self::TaskUniverseMismatch => {
                write!(f, "metric receipts do not cover the exact same eligible task universe")
            }
            Self::TaskInputUniverseMismatch => {
                write!(f, "metric receipts use different input bytes for one or more matched task IDs")
            }
            Self::ClusterMapMismatch => {
                write!(f, "cluster map must cover exactly the eligible task universe")
            }
            Self::DuplicateEvidenceDigest => {
                write!(f, "one receipt digest cannot be counted as multiple metrics")
            }
            Self::CoverageBelowMinimum {
                metric_id,
                observed,
                minimum,
            } => write!(
                f,
                "metric {metric_id} graded coverage {observed:.6} is below minimum {minimum:.6}"
            ),
            Self::NoGradedOutcomes(metric) => {
                write!(f, "metric {metric} has no graded task outcomes")
            }
            Self::IncompleteMetricCoverage {
                metric_id,
                task_id,
                disposition,
            } => write!(
                f,
                "metric {metric_id} has no grade for eligible task {task_id} ({disposition})"
            ),
            Self::ContextMismatch => write!(f, "candidate evaluation contexts differ"),
            Self::SameCandidate => write!(f, "cannot compare a candidate with itself"),
            Self::MetricSetMismatch => write!(f, "candidate metric sets differ"),
            Self::TaskInputMismatch { metric_id, task_id } => {
                write!(f, "input digest mismatch for metric {metric_id}, task {task_id}")
            }
        }
    }
}

impl std::error::Error for ComposerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const SYNTHETIC_RECEIPT: &str = include_str!(
        "../../benchmark-run-receipt-harness/examples/benchmark-run-receipt-v1.synthetic.json"
    );

    fn make_receipt(
        benchmark_id: &str,
        source_revision: &str,
        executable_byte: char,
        run_id: &str,
        network_policy: &str,
    ) -> Vec<u8> {
        let mut value: Value = serde_json::from_str(SYNTHETIC_RECEIPT).unwrap();
        value["benchmark"]["id"] = json!(benchmark_id);
        value["system"]["source_revision"] = json!(source_revision);
        value["system"]["executable_sha256"] =
            json!(std::iter::repeat_n(executable_byte, 64).collect::<String>());
        value["run_id"] = json!(run_id);
        value["comparison"] = json!({
            "group_id": "sym-rsi-synthetic-paired-v1",
            "protocol_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "eligible_task_set_sha256": "023cf71d94b1d9fd018da6bfe24fbaa409ddbcfedd77d40750c30990f4495b78"
        });
        value["runner"]["network_policy"] = json!(network_policy);
        serde_json::to_vec(&value).unwrap()
    }

    fn metric(metric_id: &str, benchmark_id: &str, minimum_coverage: f64) -> MetricProjection {
        MetricProjection {
            metric_id: metric_id.into(),
            benchmark_id: benchmark_id.into(),
            benchmark_version: "0.1".into(),
            benchmark_revision: "fixture-1".into(),
            dataset_manifest_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            scoring_formula_id: "binary-pass-rate-v1".into(),
            grader_version: "fixture-grader-1".into(),
            transform: ScoreTransform::Binary {
                true_score: 1.0,
                false_score: 0.0,
            },
            minimum_graded_coverage: minimum_coverage,
        }
    }

    fn plan(metrics: Vec<MetricProjection>) -> CompositionPlan {
        let cluster_by_task = BTreeMap::from([
            ("task-001".into(), "family-a".into()),
            ("task-002".into(), "family-a".into()),
            ("task-003".into(), "family-b".into()),
            ("task-004".into(), "family-b".into()),
            ("task-005".into(), "family-c".into()),
        ]);
        CompositionPlan {
            experiment_id: "SYM-RSI-SYNTHETIC-COMPOSER-001".into(),
            preregistration_digest: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into(),
            metrics: metrics
                .into_iter()
                .map(|projection| (projection.metric_id.clone(), projection))
                .collect(),
            cluster_by_task,
        }
    }

    fn receipt_set(
        source_revision: &str,
        executable_byte: char,
        run_suffix: &str,
        network_policy: &str,
    ) -> BTreeMap<String, Vec<u8>> {
        BTreeMap::from([
            (
                "quality".into(),
                make_receipt(
                    "synthetic-quality",
                    source_revision,
                    executable_byte,
                    &format!("synthetic-quality-{run_suffix}"),
                    network_policy,
                ),
            ),
            (
                "retention".into(),
                make_receipt(
                    "synthetic-retention",
                    source_revision,
                    executable_byte,
                    &format!("synthetic-retention-{run_suffix}"),
                    network_policy,
                ),
            ),
        ])
    }

    #[test]
    fn composes_multiple_metrics_and_preserves_ungraded_dispositions() {
        let plan = plan(vec![
            metric("quality", "synthetic-quality", 0.4),
            metric("retention", "synthetic-retention", 0.4),
        ]);
        let composed = compose_candidate(
            &plan,
            &receipt_set("candidate-a", 'a', "a", "disabled"),
        )
        .unwrap();

        assert_eq!(composed.eligible_task_ids().len(), 5);
        let quality = composed.metric("quality").unwrap();
        assert_eq!(quality.summary.eligible_count, 5);
        assert_eq!(quality.summary.graded_count, 2);
        assert_eq!(quality.summary.graded_mean, Some(1.0));
        assert!((quality.summary.graded_coverage - 0.4).abs() < 1e-12);
        assert!((quality.summary.coverage_adjusted_yield - 0.4).abs() < 1e-12);
        assert_eq!(quality.summary.disposition_counts.get("refused"), Some(&1));
        assert_eq!(quality.summary.disposition_counts.get("unsupported"), Some(&1));
        assert_eq!(quality.summary.disposition_counts.get("not_attempted"), Some(&1));

        let dimensions = composed.capability_dimensions().unwrap();
        assert_eq!(dimensions["quality"], 1.0);
        assert!((dimensions["quality.graded_coverage"] - 0.4).abs() < 1e-12);
        assert!((dimensions["quality.coverage_adjusted_yield"] - 0.4).abs() < 1e-12);
        assert!(matches!(
            composed.complete_case_scores(),
            Err(ComposerError::IncompleteMetricCoverage { .. })
        ));
    }

    #[test]
    fn refuses_capability_vector_below_preregistered_coverage_floor() {
        let plan = plan(vec![metric("quality", "synthetic-quality", 0.5)]);
        let receipts = BTreeMap::from([(
            "quality".into(),
            make_receipt("synthetic-quality", "candidate-a", 'a', "run-a", "disabled"),
        )]);
        let composed = compose_candidate(&plan, &receipts).unwrap();
        assert!(matches!(
            composed.capability_dimensions(),
            Err(ComposerError::CoverageBelowMinimum { .. })
        ));
    }

    #[test]
    fn paired_composition_preserves_missingness_and_cluster_means() {
        let plan = plan(vec![
            metric("quality", "synthetic-quality", 0.4),
            metric("retention", "synthetic-retention", 0.4),
        ]);
        let parent = compose_candidate(
            &plan,
            &receipt_set("parent-rev", 'a', "parent", "disabled"),
        )
        .unwrap();
        let candidate = compose_candidate(
            &plan,
            &receipt_set("candidate-rev", 'b', "candidate", "disabled"),
        )
        .unwrap();
        let summaries = compare_candidates(&parent, &candidate).unwrap();

        let quality = summaries.get("quality").unwrap();
        assert_eq!(quality.eligible_count, 5);
        assert_eq!(quality.parent_graded_count, 2);
        assert_eq!(quality.candidate_graded_count, 2);
        assert_eq!(quality.jointly_graded_count, 2);
        assert_eq!(quality.neither_graded_count, 3);
        assert_eq!(quality.parent_only_graded_count, 0);
        assert_eq!(quality.candidate_only_graded_count, 0);
        assert_eq!(quality.mean_delta_on_jointly_graded_tasks, Some(0.0));
        assert_eq!(quality.by_cluster.get("family-a").unwrap().paired_task_count, 2);
    }

    #[test]
    fn rejects_duplicate_receipt_reuse_as_two_metrics() {
        let plan = plan(vec![
            metric("quality", "synthetic-quality", 0.0),
            metric("retention", "synthetic-quality", 0.0),
        ]);
        let duplicate = make_receipt("synthetic-quality", "candidate-a", 'a', "run-a", "disabled");
        let receipts = BTreeMap::from([
            ("quality".into(), duplicate.clone()),
            ("retention".into(), duplicate),
        ]);
        assert_eq!(
            compose_candidate(&plan, &receipts),
            Err(ComposerError::DuplicateEvidenceDigest)
        );
    }

    #[test]
    fn rejects_cluster_map_that_drops_or_adds_eligible_tasks() {
        let mut plan = plan(vec![metric("quality", "synthetic-quality", 0.0)]);
        plan.cluster_by_task.remove("task-005");
        let receipts = BTreeMap::from([(
            "quality".into(),
            make_receipt("synthetic-quality", "candidate-a", 'a', "run-a", "disabled"),
        )]);
        assert_eq!(
            compose_candidate(&plan, &receipts),
            Err(ComposerError::ClusterMapMismatch)
        );

        plan.cluster_by_task.insert("extra-task".into(), "family-z".into());
        assert_eq!(
            compose_candidate(&plan, &receipts),
            Err(ComposerError::ClusterMapMismatch)
        );
    }

    #[test]
    fn rejects_different_parent_candidate_execution_contexts() {
        let plan = plan(vec![metric("quality", "synthetic-quality", 0.0)]);
        let parent = compose_candidate(
            &plan,
            &BTreeMap::from([(
                "quality".into(),
                make_receipt("synthetic-quality", "parent-rev", 'a', "run-parent", "disabled"),
            )]),
        )
        .unwrap();
        let candidate = compose_candidate(
            &plan,
            &BTreeMap::from([(
                "quality".into(),
                make_receipt("synthetic-quality", "candidate-rev", 'b', "run-child", "allowlisted"),
            )]),
        )
        .unwrap();
        assert_eq!(
            compare_candidates(&parent, &candidate),
            Err(ComposerError::ContextMismatch)
        );
    }

    #[test]
    fn malformed_receipt_is_rejected_by_the_shared_validator() {
        let plan = plan(vec![metric("quality", "synthetic-quality", 0.0)]);
        let receipts = BTreeMap::from([("quality".into(), b"{not-json".to_vec())]);
        assert!(matches!(
            compose_candidate(&plan, &receipts),
            Err(ComposerError::ReceiptValidation(_))
        ));
    }
}
