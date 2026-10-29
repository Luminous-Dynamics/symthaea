//! Paired-cohort evaluation primitives for architecture search.
//!
//! The output is intended as a search reward and an audit receipt, not as a standalone
//! model-quality claim. Raw task scores remain attached to their task/case identities.
//! Every candidate must be evaluated on the same cases, prompt templates, and declared
//! random seeds so task difficulty cannot be confused with architecture merit.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const PLAN_SCHEMA: &str = "symthaea.architecture-engineering.evaluation-plan";
const PLAN_VERSION: &str = "0.1.0";
const MINIMUM_TRIALS: usize = 3;

/// Robust aggregation policy must be fixed before seeing outcomes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrialAggregation {
    /// Median is robust to a single extreme trial and works well for 3+ trials.
    Median,
    /// Removes exactly this many sorted values from each tail before averaging.
    /// Requires at least one value removed per tail and a non-empty retained set.
    TrimmedMean { trim_each_tail: usize },
}

/// Exact case identity, including the prompt variant and scoring semantics.
///
/// Different prompt templates are separate cases so the result can report whether a
/// candidate is template-sensitive. The input digest covers the exact rendered evaluator
/// input, including the prompt template and task content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationCase {
    pub id: String,
    pub task_family: String,
    pub prompt_template_id: String,
    pub metric_id: String,
    pub input_sha256: String,
    pub score_min: f64,
    pub score_max: f64,
    pub higher_is_better: bool,
}

/// Frozen inputs for a paired evaluation cohort.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationPlan {
    pub schema: String,
    pub version: String,
    /// Exact code revision used to run all candidates and score them.
    pub source_commit: String,
    /// SHA-256 of the runtime/hardware/software manifest.
    pub environment_sha256: String,
    /// SHA-256 of the training or inference configuration shared by this campaign.
    pub run_config_sha256: String,
    pub dataset_sha256: String,
    pub evaluator_id: String,
    pub evaluator_sha256: String,
    /// Per-candidate canonical architecture/config digests, keyed by candidate ID.
    pub candidate_config_sha256: BTreeMap<String, String>,
    pub candidates: Vec<String>,
    pub cases: Vec<EvaluationCase>,
    /// Every candidate receives this exact seed set for every case.
    pub trial_seeds: Vec<u64>,
    pub trial_aggregation: TrialAggregation,
}

/// One scored execution. Raw output may live in an artifact store; its digest is mandatory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialObservation {
    pub candidate_id: String,
    pub case_id: String,
    pub seed: u64,
    pub source_commit: String,
    pub environment_sha256: String,
    pub run_config_sha256: String,
    pub dataset_sha256: String,
    pub evaluator_sha256: String,
    pub candidate_config_sha256: String,
    pub input_sha256: String,
    pub output_sha256: String,
    /// Task-native score. Bounds and direction are defined by the case.
    pub score: f64,
}

/// One result per candidate × exact case after the predeclared trial aggregation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseResult {
    pub candidate_id: String,
    pub case_id: String,
    pub task_family: String,
    pub prompt_template_id: String,
    pub metric_id: String,
    pub aggregated_raw_score: f64,
    /// Normalized to [0, 1], with 1 always better, using this case's declared bounds.
    pub normalized_score: f64,
    /// Candidate's normalized score minus cohort mean for this exact case.
    /// This is a search-feedback signal, not a qualification score.
    pub relative_merit: f64,
    pub trial_count: usize,
}

/// Macro summary for one candidate across cases from one task family and metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskMetricSummary {
    pub candidate_id: String,
    pub task_family: String,
    pub metric_id: String,
    pub mean_normalized_score: f64,
    pub mean_relative_merit: f64,
    pub case_count: usize,
}

/// Summary for one candidate under one prompt template and metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateMetricSummary {
    pub candidate_id: String,
    pub task_family: String,
    pub prompt_template_id: String,
    pub metric_id: String,
    pub mean_normalized_score: f64,
    pub mean_relative_merit: f64,
    pub case_count: usize,
}

/// Macro summary across task-family/metric groups, so a larger group cannot dominate merely
/// because it contains more cases. Full per-case and per-template results remain available.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateSummary {
    pub candidate_id: String,
    pub mean_normalized_score: f64,
    pub mean_relative_merit: f64,
    pub case_count: usize,
}

/// Audit receipt. Digest binds the canonical plan, input/output hashes, and derived results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationSummary {
    pub schema: String,
    pub version: String,
    pub candidate_summaries: Vec<CandidateSummary>,
    pub case_results: Vec<CaseResult>,
    pub task_metric_summaries: Vec<TaskMetricSummary>,
    pub template_metric_summaries: Vec<TemplateMetricSummary>,
    pub evaluation_receipt_sha256: String,
}

/// A complete quality/cost point. All dimensions are required; unknown metrics do not
/// silently disappear from dominance comparisons.
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoPoint {
    pub candidate_id: String,
    /// Digest of the fixed comparison profile: workload, hardware/runtime and measurement
    /// protocol must match before any points can be compared.
    pub comparison_profile_sha256: String,
    /// Normalized quality, larger is better.
    pub quality: f64,
    pub latency_ms: f64,
    pub energy_joules_per_token: f64,
    pub peak_memory_bytes: u64,
}

/// Deterministic validation error with a stable path and reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationError {
    pub path: String,
    pub message: String,
}

impl EvaluationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for EvaluationError {}

/// Evaluate a balanced candidate cohort and return a provenance-bound receipt.
///
/// The cohort is rejected if any candidate is missing an exact case/seed combination,
/// receives different prompt/task/metric semantics, or has duplicate/non-finite results.
/// Input and output hashes are included in a canonical digest.
pub fn evaluate_paired_cohort(
    plan: &EvaluationPlan,
    observations: &[TrialObservation],
) -> Result<EvaluationSummary, EvaluationError> {
    validate_plan(plan)?;

    let expected_count = plan
        .candidates
        .len()
        .checked_mul(plan.cases.len())
        .and_then(|value| value.checked_mul(plan.trial_seeds.len()))
        .ok_or_else(|| EvaluationError::new("$", "expected observation count overflow"))?;
    if observations.len() != expected_count {
        return Err(EvaluationError::new(
            "$.observations",
            format!(
                "expected exactly {expected_count} candidate×case×seed observations, found {}",
                observations.len()
            ),
        ));
    }

    let candidate_set: HashSet<String> = plan.candidates.iter().cloned().collect();
    let case_by_id: HashMap<String, &EvaluationCase> = plan
        .cases
        .iter()
        .map(|case| (case.id.clone(), case))
        .collect();
    let seed_set: HashSet<u64> = plan.trial_seeds.iter().copied().collect();
    let mut seen = HashSet::with_capacity(observations.len());
    let mut grouped: HashMap<(String, String), Vec<&TrialObservation>> = HashMap::new();

    for (index, observation) in observations.iter().enumerate() {
        let path = format!("$.observations[{index}]");
        if !candidate_set.contains(&observation.candidate_id) {
            return Err(EvaluationError::new(
                format!("{path}.candidate_id"),
                format!("candidate {:?} is not in the frozen plan", observation.candidate_id),
            ));
        }
        if !case_by_id.contains_key(&observation.case_id) {
            return Err(EvaluationError::new(
                format!("{path}.case_id"),
                format!("case {:?} is not in the frozen plan", observation.case_id),
            ));
        }
        if !seed_set.contains(&observation.seed) {
            return Err(EvaluationError::new(
                format!("{path}.seed"),
                format!("seed {} is not in the frozen seed set", observation.seed),
            ));
        }
        if !is_sha256(&observation.output_sha256) {
            return Err(EvaluationError::new(
                format!("{path}.output_sha256"),
                "expected a lowercase 64-character SHA-256 hex digest",
            ));
        }
        let case = case_by_id[&observation.case_id];
        if observation.source_commit != plan.source_commit {
            return Err(EvaluationError::new(
                format!("{path}.source_commit"),
                "observation source commit does not match the frozen plan",
            ));
        }
        if observation.environment_sha256 != plan.environment_sha256 {
            return Err(EvaluationError::new(
                format!("{path}.environment_sha256"),
                "observation environment does not match the frozen plan",
            ));
        }
        if observation.run_config_sha256 != plan.run_config_sha256 {
            return Err(EvaluationError::new(
                format!("{path}.run_config_sha256"),
                "observation run config does not match the frozen plan",
            ));
        }
        if observation.dataset_sha256 != plan.dataset_sha256 {
            return Err(EvaluationError::new(
                format!("{path}.dataset_sha256"),
                "observation dataset does not match the frozen plan",
            ));
        }
        if observation.evaluator_sha256 != plan.evaluator_sha256 {
            return Err(EvaluationError::new(
                format!("{path}.evaluator_sha256"),
                "observation evaluator does not match the frozen plan",
            ));
        }
        if observation.candidate_config_sha256
            != plan.candidate_config_sha256[&observation.candidate_id]
        {
            return Err(EvaluationError::new(
                format!("{path}.candidate_config_sha256"),
                "observation candidate config does not match the frozen candidate identity",
            ));
        }
        if observation.input_sha256 != case.input_sha256 {
            return Err(EvaluationError::new(
                format!("{path}.input_sha256"),
                "observation input does not match the exact case input in the frozen plan",
            ));
        }
        if !observation.score.is_finite()
            || observation.score < case.score_min
            || observation.score > case.score_max
        {
            return Err(EvaluationError::new(
                format!("{path}.score"),
                format!(
                    "score must be finite and within the declared bounds [{}, {}]",
                    case.score_min, case.score_max
                ),
            ));
        }

        let key = (
            observation.candidate_id.clone(),
            observation.case_id.clone(),
            observation.seed,
        );
        if !seen.insert(key) {
            return Err(EvaluationError::new(
                &path,
                "duplicate candidate/case/seed observation",
            ));
        }
        grouped
            .entry((
                observation.candidate_id.clone(),
                observation.case_id.clone(),
            ))
            .or_default()
            .push(observation);
    }

    let mut cases = plan.cases.clone();
    cases.sort_by(|a, b| a.id.cmp(&b.id));
    let mut candidates = plan.candidates.clone();
    candidates.sort();

    let mut case_results = Vec::with_capacity(candidates.len() * cases.len());
    for candidate_id in &candidates {
        for case in &cases {
            let key = (candidate_id.clone(), case.id.clone());
            let runs = grouped.get(&key).ok_or_else(|| {
                EvaluationError::new(
                    "$.observations",
                    format!("missing observations for candidate {:?}, case {:?}", candidate_id, case.id),
                )
            })?;
            if runs.len() != plan.trial_seeds.len() {
                return Err(EvaluationError::new(
                    "$.observations",
                    format!(
                        "candidate {:?}, case {:?} has {} trials; expected {}",
                        candidate_id,
                        case.id,
                        runs.len(),
                        plan.trial_seeds.len()
                    ),
                ));
            }

            let scores: Vec<f64> = runs.iter().map(|run| run.score).collect();
            let raw_score = aggregate_trials(&scores, &plan.trial_aggregation)?;
            let normalized_score = normalize_score(raw_score, case);
            case_results.push(CaseResult {
                candidate_id: candidate_id.clone(),
                case_id: case.id.clone(),
                task_family: case.task_family.clone(),
                prompt_template_id: case.prompt_template_id.clone(),
                metric_id: case.metric_id.clone(),
                aggregated_raw_score: raw_score,
                normalized_score,
                relative_merit: 0.0,
                trial_count: runs.len(),
            });
        }
    }

    // Compute peer-relative merit only among candidates that saw this exact same case.
    let mut cohort_sum: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for result in &case_results {
        let entry = cohort_sum.entry(result.case_id.clone()).or_insert((0.0, 0));
        entry.0 += result.normalized_score;
        entry.1 += 1;
    }
    let cohort_mean: BTreeMap<String, f64> = cohort_sum
        .into_iter()
        .map(|(case_id, (sum, count))| (case_id, sum / count as f64))
        .collect();
    for result in &mut case_results {
        let mean = cohort_mean[&result.case_id];
        result.relative_merit = result.normalized_score - mean;
    }

    let task_metric_summaries = summarize_task_metrics(&case_results);
    let candidate_summaries =
        summarize_candidates(&case_results, &task_metric_summaries, &candidates);
    let template_metric_summaries = summarize_templates(&case_results);

    // Canonicalize all input sequences before digesting so reordering equivalent input
    // records cannot change the receipt digest.
    let mut canonical_plan = plan.clone();
    canonical_plan.candidates.sort();
    canonical_plan.cases.sort_by(|a, b| a.id.cmp(&b.id));
    canonical_plan.trial_seeds.sort_unstable();

    let mut canonical_observations = observations.to_vec();
    canonical_observations.sort_by(|a, b| {
        (&a.candidate_id, &a.case_id, a.seed).cmp(&(&b.candidate_id, &b.case_id, b.seed))
    });

    #[derive(Serialize)]
    struct DigestPayload<'a> {
        plan: &'a EvaluationPlan,
        observations: &'a [TrialObservation],
        candidate_summaries: &'a [CandidateSummary],
        case_results: &'a [CaseResult],
        task_metric_summaries: &'a [TaskMetricSummary],
        template_metric_summaries: &'a [TemplateMetricSummary],
    }

    let payload = DigestPayload {
        plan: &canonical_plan,
        observations: &canonical_observations,
        candidate_summaries: &candidate_summaries,
        case_results: &case_results,
        task_metric_summaries: &task_metric_summaries,
        template_metric_summaries: &template_metric_summaries,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| EvaluationError::new("$.receipt", format!("serialization failed: {error}")))?;
    let digest = sha256_hex(&bytes);

    Ok(EvaluationSummary {
        schema: PLAN_SCHEMA.to_owned(),
        version: PLAN_VERSION.to_owned(),
        candidate_summaries,
        case_results,
        task_metric_summaries,
        template_metric_summaries,
        evaluation_receipt_sha256: digest,
    })
}

/// Return nondominated candidate IDs with fixed, fully measured metric dimensions.
///
/// Missing/unknown dimensions are intentionally not represented by this type. Do not call
/// Pareto comparison until all four measurements are available under compatible protocols.
pub fn pareto_front(points: &[ParetoPoint]) -> Result<Vec<String>, EvaluationError> {
    if points.is_empty() {
        return Err(EvaluationError::new(
            "$.pareto_points",
            "at least one fully measured point is required",
        ));
    }
    let mut ids = HashSet::new();
    for (index, point) in points.iter().enumerate() {
        let path = format!("$.pareto_points[{index}]");
        if point.candidate_id.trim().is_empty() || !ids.insert(point.candidate_id.as_str()) {
            return Err(EvaluationError::new(
                format!("{path}.candidate_id"),
                "candidate ID must be non-empty and unique",
            ));
        }
        if !is_sha256(&point.comparison_profile_sha256) {
            return Err(EvaluationError::new(
                format!("{path}.comparison_profile_sha256"),
                "expected a lowercase 64-character SHA-256 profile digest",
            ));
        }
        if !point.quality.is_finite()
            || !(0.0..=1.0).contains(&point.quality)
            || !point.latency_ms.is_finite()
            || point.latency_ms < 0.0
            || !point.energy_joules_per_token.is_finite()
            || point.energy_joules_per_token < 0.0
        {
            return Err(EvaluationError::new(
                &path,
                "quality must be in [0,1] and all cost dimensions must be finite/non-negative",
            ));
        }
    }

    if let Some(first) = points.first() {
        if points
            .iter()
            .any(|point| point.comparison_profile_sha256 != first.comparison_profile_sha256)
        {
            return Err(EvaluationError::new(
                "$.pareto_points",
                "cannot compare candidates measured under different workload/hardware profiles",
            ));
        }
    }

    let mut front = Vec::new();
    for point in points {
        let is_dominated = points
            .iter()
            .any(|other| other.candidate_id != point.candidate_id && dominates(other, point));
        if !is_dominated {
            front.push(point.candidate_id.clone());
        }
    }
    front.sort();
    Ok(front)
}

/// Pareto dominance: no worse on every dimension and strictly better on at least one.
pub fn dominates(a: &ParetoPoint, b: &ParetoPoint) -> bool {
    if a.comparison_profile_sha256 != b.comparison_profile_sha256 {
        return false;
    }
    let no_worse = a.quality >= b.quality
        && a.latency_ms <= b.latency_ms
        && a.energy_joules_per_token <= b.energy_joules_per_token
        && a.peak_memory_bytes <= b.peak_memory_bytes;
    let strictly_better = a.quality > b.quality
        || a.latency_ms < b.latency_ms
        || a.energy_joules_per_token < b.energy_joules_per_token
        || a.peak_memory_bytes < b.peak_memory_bytes;
    no_worse && strictly_better
}

fn validate_plan(plan: &EvaluationPlan) -> Result<(), EvaluationError> {
    if plan.schema != PLAN_SCHEMA {
        return Err(EvaluationError::new(
            "$.schema",
            format!("expected {PLAN_SCHEMA:?}"),
        ));
    }
    if plan.version != PLAN_VERSION {
        return Err(EvaluationError::new(
            "$.version",
            format!("expected {PLAN_VERSION:?}"),
        ));
    }
    if !is_git_commit(&plan.source_commit) {
        return Err(EvaluationError::new(
            "$.source_commit",
            "expected a lowercase 40- or 64-character Git commit ID",
        ));
    }
    if !is_sha256(&plan.environment_sha256) {
        return Err(EvaluationError::new(
            "$.environment_sha256",
            "expected a lowercase 64-character SHA-256 hex digest",
        ));
    }
    if !is_sha256(&plan.run_config_sha256) {
        return Err(EvaluationError::new(
            "$.run_config_sha256",
            "expected a lowercase 64-character SHA-256 hex digest",
        ));
    }
    if !is_sha256(&plan.dataset_sha256) {
        return Err(EvaluationError::new(
            "$.dataset_sha256",
            "expected a lowercase 64-character SHA-256 hex digest",
        ));
    }
    if plan.evaluator_id.trim().is_empty() {
        return Err(EvaluationError::new("$.evaluator_id", "must be non-empty"));
    }
    if !is_sha256(&plan.evaluator_sha256) {
        return Err(EvaluationError::new(
            "$.evaluator_sha256",
            "expected a lowercase 64-character SHA-256 hex digest",
        ));
    }
    if plan.candidates.is_empty() {
        return Err(EvaluationError::new("$.candidates", "must not be empty"));
    }
    let mut candidate_ids = HashSet::new();
    for (index, candidate) in plan.candidates.iter().enumerate() {
        if candidate.trim().is_empty() || !candidate_ids.insert(candidate.as_str()) {
            return Err(EvaluationError::new(
                format!("$.candidates[{index}]"),
                "candidate IDs must be non-empty and unique",
            ));
        }
    }
    if plan.candidate_config_sha256.len() != plan.candidates.len() {
        return Err(EvaluationError::new(
            "$.candidate_config_sha256",
            "must contain exactly one configuration digest per candidate",
        ));
    }
    for candidate in &plan.candidates {
        let digest = plan.candidate_config_sha256.get(candidate).ok_or_else(|| {
            EvaluationError::new(
                "$.candidate_config_sha256",
                format!("missing configuration digest for candidate {candidate:?}"),
            )
        })?;
        if !is_sha256(digest) {
            return Err(EvaluationError::new(
                format!("$.candidate_config_sha256.{candidate}"),
                "expected a lowercase 64-character SHA-256 hex digest",
            ));
        }
    }
    if plan.cases.is_empty() {
        return Err(EvaluationError::new("$.cases", "must not be empty"));
    }
    let mut case_ids = HashSet::new();
    for (index, case) in plan.cases.iter().enumerate() {
        let path = format!("$.cases[{index}]");
        if case.id.trim().is_empty() || !case_ids.insert(case.id.as_str()) {
            return Err(EvaluationError::new(
                format!("{path}.id"),
                "case IDs must be non-empty and unique",
            ));
        }
        for (field, value) in [
            ("task_family", case.task_family.as_str()),
            ("prompt_template_id", case.prompt_template_id.as_str()),
            ("metric_id", case.metric_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(EvaluationError::new(
                    format!("{path}.{field}"),
                    "must be non-empty",
                ));
            }
        }
        if !is_sha256(&case.input_sha256) {
            return Err(EvaluationError::new(
                format!("{path}.input_sha256"),
                "expected a lowercase 64-character SHA-256 hex digest",
            ));
        }
        if !case.score_min.is_finite()
            || !case.score_max.is_finite()
            || case.score_min >= case.score_max
            || !(case.score_max - case.score_min).is_finite()
        {
            return Err(EvaluationError::new(
                &path,
                "score bounds must be finite and score_min < score_max",
            ));
        }
    }

    if plan.trial_seeds.len() < MINIMUM_TRIALS {
        return Err(EvaluationError::new(
            "$.trial_seeds",
            format!("at least {MINIMUM_TRIALS} predeclared independent seeds are required"),
        ));
    }
    let mut seeds = HashSet::new();
    for (index, seed) in plan.trial_seeds.iter().enumerate() {
        if !seeds.insert(*seed) {
            return Err(EvaluationError::new(
                format!("$.trial_seeds[{index}]"),
                "trial seeds must be unique",
            ));
        }
    }
    if let TrialAggregation::TrimmedMean { trim_each_tail } = &plan.trial_aggregation {
        if *trim_each_tail == 0 || (*trim_each_tail).saturating_mul(2) >= plan.trial_seeds.len() {
            return Err(EvaluationError::new(
                "$.trial_aggregation",
                "trimmed mean requires trim_each_tail >= 1 and a non-empty retained set",
            ));
        }
    }
    Ok(())
}

fn aggregate_trials(
    scores: &[f64],
    policy: &TrialAggregation,
) -> Result<f64, EvaluationError> {
    if scores.len() < MINIMUM_TRIALS || scores.iter().any(|score| !score.is_finite()) {
        return Err(EvaluationError::new(
            "$.observations",
            "trial aggregate requires at least three finite scores",
        ));
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(f64::total_cmp);
    match policy {
        TrialAggregation::Median => {
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                Ok(sorted[mid - 1] / 2.0 + sorted[mid] / 2.0)
            } else {
                Ok(sorted[mid])
            }
        }
        TrialAggregation::TrimmedMean { trim_each_tail } => {
            if *trim_each_tail == 0 || trim_each_tail.saturating_mul(2) >= sorted.len() {
                return Err(EvaluationError::new(
                    "$.trial_aggregation",
                    "trimmed mean must retain at least one score after trimming both tails",
                ));
            }
            let retained = &sorted[*trim_each_tail..(sorted.len() - trim_each_tail)];
            let divisor = retained.len() as f64;
            Ok(retained.iter().map(|value| value / divisor).sum::<f64>())
        }
    }
}

fn normalize_score(raw: f64, case: &EvaluationCase) -> f64 {
    let span = case.score_max - case.score_min;
    let directional = if case.higher_is_better {
        raw - case.score_min
    } else {
        case.score_max - raw
    };
    (directional / span).clamp(0.0, 1.0)
}

fn summarize_candidates(
    results: &[CaseResult],
    task_metric_summaries: &[TaskMetricSummary],
    candidates: &[String],
) -> Vec<CandidateSummary> {
    candidates
        .iter()
        .map(|candidate_id| {
            let groups: Vec<&TaskMetricSummary> = task_metric_summaries
                .iter()
                .filter(|summary| &summary.candidate_id == candidate_id)
                .collect();
            let case_count = results
                .iter()
                .filter(|result| &result.candidate_id == candidate_id)
                .count();
            CandidateSummary {
                candidate_id: candidate_id.clone(),
                mean_normalized_score: mean(
                    groups.iter().map(|summary| summary.mean_normalized_score),
                ),
                mean_relative_merit: mean(
                    groups.iter().map(|summary| summary.mean_relative_merit),
                ),
                case_count,
            }
        })
        .collect()
}

fn summarize_task_metrics(results: &[CaseResult]) -> Vec<TaskMetricSummary> {
    let mut grouped: BTreeMap<(String, String, String), Vec<&CaseResult>> = BTreeMap::new();
    for result in results {
        grouped
            .entry((
                result.candidate_id.clone(),
                result.task_family.clone(),
                result.metric_id.clone(),
            ))
            .or_default()
            .push(result);
    }
    grouped
        .into_iter()
        .map(|((candidate_id, task_family, metric_id), matching)| TaskMetricSummary {
            candidate_id,
            task_family,
            metric_id,
            mean_normalized_score: mean(matching.iter().map(|result| result.normalized_score)),
            mean_relative_merit: mean(matching.iter().map(|result| result.relative_merit)),
            case_count: matching.len(),
        })
        .collect()
}

fn summarize_templates(results: &[CaseResult]) -> Vec<TemplateMetricSummary> {
    let mut grouped: BTreeMap<(String, String, String, String), Vec<&CaseResult>> = BTreeMap::new();
    for result in results {
        grouped
            .entry((
                result.candidate_id.clone(),
                result.task_family.clone(),
                result.prompt_template_id.clone(),
                result.metric_id.clone(),
            ))
            .or_default()
            .push(result);
    }
    grouped
        .into_iter()
        .map(
            |((candidate_id, task_family, prompt_template_id, metric_id), matching)| {
                TemplateMetricSummary {
                    candidate_id,
                    task_family,
                    prompt_template_id,
                    metric_id,
                    mean_normalized_score: mean(
                        matching.iter().map(|result| result.normalized_score),
                    ),
                    mean_relative_merit: mean(
                        matching.iter().map(|result| result.relative_merit),
                    ),
                    case_count: matching.len(),
                }
            },
        )
        .collect()
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        total += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_git_commit(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn case(id: &str, family: &str, template: &str, offset: f64) -> EvaluationCase {
        EvaluationCase {
            id: id.to_owned(),
            task_family: family.to_owned(),
            prompt_template_id: template.to_owned(),
            metric_id: "normalized_quality".to_owned(),
            input_sha256: hash(if id == "easy-1" { 'a' } else { 'd' }),
            score_min: -10.0 + offset,
            score_max: 10.0 + offset,
            higher_is_better: true,
        }
    }

    fn plan() -> EvaluationPlan {
        EvaluationPlan {
            schema: PLAN_SCHEMA.to_owned(),
            version: PLAN_VERSION.to_owned(),
            source_commit: std::iter::repeat_n('1', 40).collect(),
            environment_sha256: hash('2'),
            run_config_sha256: hash('3'),
            dataset_sha256: hash('b'),
            evaluator_id: "test-evaluator-v1".to_owned(),
            evaluator_sha256: hash('c'),
            candidate_config_sha256: BTreeMap::from([
                ("candidate-a".to_owned(), hash('f')),
                ("candidate-b".to_owned(), hash('0')),
            ]),
            candidates: vec!["candidate-a".to_owned(), "candidate-b".to_owned()],
            cases: vec![
                case("easy-1", "retrieval", "template-1", 0.0),
                case("hard-1", "reasoning", "template-2", 0.0),
            ],
            trial_seeds: vec![11, 22, 33],
            trial_aggregation: TrialAggregation::Median,
        }
    }

    fn observations(plan: &EvaluationPlan) -> Vec<TrialObservation> {
        let scores = [
            ("candidate-a", "easy-1", vec![7.0, 7.5, 6.5]),
            ("candidate-b", "easy-1", vec![6.0, 6.0, 6.5]),
            ("candidate-a", "hard-1", vec![0.0, 1.0, 1.5]),
            ("candidate-b", "hard-1", vec![-1.0, -1.0, 0.0]),
        ];
        let mut result = Vec::new();
        for (candidate, case_id, values) in scores {
            for (seed, score) in plan.trial_seeds.iter().zip(values) {
                let case = plan
                    .cases
                    .iter()
                    .find(|case| case.id == case_id)
                    .expect("test case must exist");
                result.push(TrialObservation {
                    candidate_id: candidate.to_owned(),
                    case_id: case_id.to_owned(),
                    seed: *seed,
                    source_commit: plan.source_commit.clone(),
                    environment_sha256: plan.environment_sha256.clone(),
                    run_config_sha256: plan.run_config_sha256.clone(),
                    dataset_sha256: plan.dataset_sha256.clone(),
                    evaluator_sha256: plan.evaluator_sha256.clone(),
                    candidate_config_sha256: plan.candidate_config_sha256[candidate].clone(),
                    input_sha256: case.input_sha256.clone(),
                    output_sha256: hash(if candidate == "candidate-a" { '8' } else { 'e' }),
                    score,
                });
            }
        }
        result
    }

    #[test]
    fn balanced_paired_cohort_yields_a_digest_and_all_strata() {
        let plan = plan();
        let observations = observations(&plan);
        let summary = evaluate_paired_cohort(&plan, &observations).expect("valid cohort");
        assert_eq!(summary.candidate_summaries.len(), 2);
        assert_eq!(summary.case_results.len(), 4);
        assert_eq!(summary.task_metric_summaries.len(), 4);
        assert_eq!(summary.template_metric_summaries.len(), 4);
        assert_eq!(summary.evaluation_receipt_sha256.len(), 64);
        assert!(summary.case_results.iter().all(|case| case.trial_count == 3));
    }

    #[test]
    fn query_difficulty_offset_cancels_in_paired_relative_merit() {
        let mut original_plan = plan();
        let original_observations = observations(&original_plan);
        let original = evaluate_paired_cohort(&original_plan, &original_observations)
            .expect("original cohort");

        let mut shifted_observations = original_observations.clone();
        for observation in &mut shifted_observations {
            if observation.case_id == "hard-1" {
                observation.score += 2.0;
            }
        }
        let shifted = evaluate_paired_cohort(&original_plan, &shifted_observations)
            .expect("shifted cohort");

        for (before, after) in original.case_results.iter().zip(shifted.case_results.iter()) {
            assert!((before.relative_merit - after.relative_merit).abs() < 1e-12);
        }
    }

    #[test]
    fn median_aggregation_is_robust_to_one_extreme_trial() {
        let ordinary = aggregate_trials(&[0.7, 0.8, 0.9], &TrialAggregation::Median).unwrap();
        let outlier = aggregate_trials(&[0.7, 0.8, 100.0], &TrialAggregation::Median).unwrap();
        assert_eq!(ordinary, outlier);
    }

    #[test]
    fn trimmed_mean_discards_predeclared_tail_outlier() {
        let mean = aggregate_trials(
            &[0.8, 0.8, 0.8, 0.8, 0.0],
            &TrialAggregation::TrimmedMean { trim_each_tail: 1 },
        )
        .unwrap();
        assert!((mean - 0.8).abs() < 1e-12);
    }

    #[test]
    fn missing_case_result_cannot_be_replaced_by_easy_case_repetition() {
        let plan = plan();
        let mut data = observations(&plan);
        data.retain(|observation| observation.case_id != "hard-1");
        let error = evaluate_paired_cohort(&plan, &data).expect_err("missing hard cases must fail");
        assert!(error.to_string().contains("expected exactly"));
    }

    #[test]
    fn duplicate_trial_fails_closed() {
        let plan = plan();
        let mut data = observations(&plan);
        data[1] = data[0].clone();
        let error = evaluate_paired_cohort(&plan, &data).expect_err("duplicate trial must fail");
        assert!(error.to_string().contains("duplicate candidate/case/seed"));
    }

    #[test]
    fn observation_order_does_not_change_receipt_digest() {
        let plan = plan();
        let mut data = observations(&plan);
        let first = evaluate_paired_cohort(&plan, &data).expect("first evaluation");
        data.reverse();
        let second = evaluate_paired_cohort(&plan, &data).expect("reordered evaluation");
        assert_eq!(
            first.evaluation_receipt_sha256,
            second.evaluation_receipt_sha256
        );
    }

    #[test]
    fn pareto_front_preserves_quality_cost_tradeoffs() {
        let points = vec![
            ParetoPoint {
                candidate_id: "balanced".to_owned(),
                comparison_profile_sha256: hash('1'),
                quality: 0.8,
                latency_ms: 10.0,
                energy_joules_per_token: 0.5,
                peak_memory_bytes: 1024,
            },
            ParetoPoint {
                candidate_id: "quality".to_owned(),
                comparison_profile_sha256: hash('1'),
                quality: 0.9,
                latency_ms: 15.0,
                energy_joules_per_token: 0.7,
                peak_memory_bytes: 2048,
            },
            ParetoPoint {
                candidate_id: "dominated".to_owned(),
                comparison_profile_sha256: hash('1'),
                quality: 0.7,
                latency_ms: 20.0,
                energy_joules_per_token: 1.0,
                peak_memory_bytes: 4096,
            },
        ];
        assert_eq!(
            pareto_front(&points).unwrap(),
            vec!["balanced".to_owned(), "quality".to_owned()]
        );
    }

    #[test]
    fn pareto_comparison_rejects_mismatched_workload_profiles() {
        let points = vec![
            ParetoPoint {
                candidate_id: "a".to_owned(),
                comparison_profile_sha256: hash('1'),
                quality: 0.9,
                latency_ms: 10.0,
                energy_joules_per_token: 0.5,
                peak_memory_bytes: 1024,
            },
            ParetoPoint {
                candidate_id: "b".to_owned(),
                comparison_profile_sha256: hash('2'),
                quality: 0.8,
                latency_ms: 20.0,
                energy_joules_per_token: 1.0,
                peak_memory_bytes: 2048,
            },
        ];
        let error = pareto_front(&points).expect_err("different profiles are not comparable");
        assert!(error.to_string().contains("different workload/hardware profiles"));
    }

    #[test]
    fn direct_dominance_check_refuses_different_profiles() {
        let a = ParetoPoint {
            candidate_id: "a".to_owned(),
            comparison_profile_sha256: hash('1'),
            quality: 0.9,
            latency_ms: 5.0,
            energy_joules_per_token: 0.1,
            peak_memory_bytes: 1024,
        };
        let b = ParetoPoint {
            candidate_id: "b".to_owned(),
            comparison_profile_sha256: hash('2'),
            quality: 0.8,
            latency_ms: 10.0,
            energy_joules_per_token: 0.5,
            peak_memory_bytes: 2048,
        };
        assert!(!dominates(&a, &b));
    }

    #[test]
    fn candidate_cannot_dominate_when_quality_latency_trade_off() {
        let a = ParetoPoint {
            candidate_id: "a".to_owned(),
            comparison_profile_sha256: hash('1'),
            quality: 0.9,
            latency_ms: 20.0,
            energy_joules_per_token: 1.0,
            peak_memory_bytes: 2048,
        };
        let b = ParetoPoint {
            candidate_id: "b".to_owned(),
            comparison_profile_sha256: hash('1'),
            quality: 0.8,
            latency_ms: 10.0,
            energy_joules_per_token: 0.5,
            peak_memory_bytes: 1024,
        };
        assert!(!dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn changing_candidate_config_digest_changes_the_receipt() {
        let plan = plan();
        let data = observations(&plan);
        let before = evaluate_paired_cohort(&plan, &data).expect("before receipt");
        let mut changed_plan = plan;
        changed_plan
            .candidate_config_sha256
            .insert("candidate-a".to_owned(), hash('9'));
        let after = evaluate_paired_cohort(&changed_plan, &data).expect("after receipt");
        assert_ne!(
            before.evaluation_receipt_sha256,
            after.evaluation_receipt_sha256
        );
    }

    #[test]
    fn missing_candidate_config_digest_fails_closed() {
        let mut plan = plan();
        plan.candidate_config_sha256.remove("candidate-a");
        let error = evaluate_paired_cohort(&plan, &observations(&plan))
            .expect_err("candidate config must be digest-bound");
        assert!(error.to_string().contains("exactly one configuration digest"));
    }

    #[test]
    fn mismatched_input_digest_fails_closed() {
        let plan = plan();
        let mut data = observations(&plan);
        data[0].input_sha256 = hash('9');
        let error = evaluate_paired_cohort(&plan, &data).expect_err("wrong prompt must fail");
        assert!(error.to_string().contains("does not match the exact case input"));
    }

    #[test]
    fn permuted_case_identity_fails_against_the_exact_input_digest() {
        let plan = plan();
        let mut data = observations(&plan);
        data[0].case_id = "hard-1".to_owned();
        let error = evaluate_paired_cohort(&plan, &data)
            .expect_err("a case label cannot be swapped while retaining the old prompt digest");
        assert!(error.to_string().contains("exact case input"));
    }

    #[test]
    fn mixed_runtime_environment_fails_closed() {
        let plan = plan();
        let mut data = observations(&plan);
        data[0].environment_sha256 = hash('9');
        let error = evaluate_paired_cohort(&plan, &data).expect_err("mixed environments must fail");
        assert!(error.to_string().contains("environment does not match"));
    }

    #[test]
    fn mislabeled_candidate_config_fails_closed() {
        let plan = plan();
        let mut data = observations(&plan);
        data[0].candidate_config_sha256 = hash('9');
        let error = evaluate_paired_cohort(&plan, &data).expect_err("wrong candidate config must fail");
        assert!(error.to_string().contains("candidate config does not match"));
    }

    #[test]
    fn lower_is_better_metrics_are_normalized_in_the_correct_direction() {
        let mut plan = plan();
        for case in &mut plan.cases {
            case.higher_is_better = false;
        }
        let mut data = observations(&plan);
        // Lower raw values are better for this declared metric.
        for observation in &mut data {
            observation.score = if observation.candidate_id == "candidate-a" {
                -2.0
            } else {
                2.0
            };
        }
        let result = evaluate_paired_cohort(&plan, &data).expect("valid lower-is-better cohort");
        let a = result
            .candidate_summaries
            .iter()
            .find(|summary| summary.candidate_id == "candidate-a")
            .unwrap();
        let b = result
            .candidate_summaries
            .iter()
            .find(|summary| summary.candidate_id == "candidate-b")
            .unwrap();
        assert!(a.mean_normalized_score > b.mean_normalized_score);
    }
}
