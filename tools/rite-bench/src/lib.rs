//! RITE-Bench: Research Integrity, Trust & Efficiency.
//!
//! This crate intentionally emits a vector of separately qualified metrics.
//! It does not compute a weighted aggregate score.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

const CALIBRATION_BINS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Supported,
    Contradicted,
    InsufficientEvidence,
}

impl Decision {
    fn is_abstention(self) -> bool {
        self == Self::InsufficientEvidence
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterventionKind {
    Counterfactual,
    SourceRemoval,
    Placebo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CitationRecord {
    pub source_id: String,
    /// Normalized provenance origin, not a URL or citation count.
    pub origin_id: String,
    /// Set by an independent evaluator against the claim-specific rubric.
    pub eligible_for_claim: bool,
    pub supports_expected_decision: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterventionObservation {
    pub kind: InterventionKind,
    pub decision_before: Decision,
    pub decision_after: Decision,
    /// Ground-truth decision after this intervention.
    pub expected_after: Decision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayEvidence {
    pub independent_replay_passed: bool,
    pub contract_digest_matches: bool,
    pub output_manifest_matches: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceUse {
    pub tokens: u64,
    pub tool_calls: u64,
    pub wall_time_ms: u64,
    /// Integer micro-USD to avoid floating point currency accumulation.
    pub cost_micros_usd: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceLimits {
    pub max_tokens: Option<u64>,
    pub max_tool_calls: Option<u64>,
    pub max_wall_time_ms: Option<u64>,
    pub max_cost_micros_usd: Option<u64>,
}

impl ResourceLimits {
    fn has_limit(self) -> bool {
        self.max_tokens.is_some()
            || self.max_tool_calls.is_some()
            || self.max_wall_time_ms.is_some()
            || self.max_cost_micros_usd.is_some()
    }

    fn contains(self, usage: ResourceUse) -> bool {
        self.max_tokens.is_none_or(|limit| usage.tokens <= limit)
            && self.max_tool_calls.is_none_or(|limit| usage.tool_calls <= limit)
            && self
                .max_wall_time_ms
                .is_none_or(|limit| usage.wall_time_ms <= limit)
            && self
                .max_cost_micros_usd
                .is_none_or(|limit| usage.cost_micros_usd <= limit)
    }
}

/// One evaluator-authored task record. Confidence means P(the submitted
/// decision is correct), including when the submitted decision is abstention.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskRecord {
    pub task_id: String,
    pub expected_decision: Decision,
    pub submitted_decision: Decision,
    pub confidence_correct: f64,
    pub citations: Vec<CitationRecord>,
    /// Oracle-defined requirement; mirrored URLs do not increase this count.
    pub minimum_independent_decision_origins: usize,
    pub interventions: Vec<InterventionObservation>,
    pub replay_required: bool,
    pub replay: Option<ReplayEvidence>,
    pub resources: ResourceUse,
    pub limits: Option<ResourceLimits>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rate {
    pub numerator: u64,
    pub denominator: u64,
}

impl Rate {
    pub fn value(self) -> Option<f64> {
        (self.denominator > 0).then(|| self.numerator as f64 / self.denominator as f64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Mean {
    /// Sum of observations (or weighted absolute error for ECE).
    pub numerator: f64,
    pub denominator: u64,
}

impl Mean {
    pub fn value(self) -> Option<f64> {
        (self.denominator > 0).then(|| self.numerator / self.denominator as f64)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scorecard {
    pub task_count: u64,
    pub correct_decisions: u64,
    pub decision_accuracy: Rate,
    pub answer_coverage: Rate,
    pub selective_risk: Rate,
    pub brier_score: Mean,
    pub expected_calibration_error_10_bin: Mean,
    pub citation_precision: Rate,
    pub independent_support_adequacy: Rate,
    pub counterfactual_oracle_accuracy: Rate,
    pub source_removal_oracle_accuracy: Rate,
    pub placebo_invariance: Rate,
    pub replay_pass_rate: Rate,
    pub budget_compliance: Rate,
    pub total_tokens: u64,
    pub total_tool_calls: u64,
    pub total_wall_time_ms: u64,
    pub total_cost_micros_usd: u64,
    pub cost_per_correct_decision_micros_usd: Option<f64>,
}

impl Scorecard {
    /// Export a deterministic CSV scorecard. Every metric includes its raw
    /// numerator and denominator; an empty value denotes an undefined rate.
    pub fn to_csv(&self) -> String {
        let mut output =
            String::from("metric,numerator,denominator,value,unit,direction\\n");
        push_rate_row(
            &mut output,
            "decision_accuracy",
            self.decision_accuracy,
            "fraction",
            "higher_is_better",
        );
        push_rate_row(
            &mut output,
            "answer_coverage",
            self.answer_coverage,
            "fraction",
            "higher_is_better",
        );
        push_rate_row(
            &mut output,
            "selective_risk",
            self.selective_risk,
            "fraction",
            "lower_is_better",
        );
        push_mean_row(
            &mut output,
            "brier_score",
            self.brier_score,
            "mean_squared_error",
            "lower_is_better",
        );
        push_mean_row(
            &mut output,
            "expected_calibration_error_10_bin",
            self.expected_calibration_error_10_bin,
            "fraction",
            "lower_is_better",
        );
        push_rate_row(
            &mut output,
            "citation_precision",
            self.citation_precision,
            "fraction",
            "higher_is_better",
        );
        push_rate_row(
            &mut output,
            "independent_support_adequacy",
            self.independent_support_adequacy,
            "fraction",
            "higher_is_better",
        );
        push_rate_row(
            &mut output,
            "counterfactual_oracle_accuracy",
            self.counterfactual_oracle_accuracy,
            "fraction",
            "higher_is_better",
        );
        push_rate_row(
            &mut output,
            "source_removal_oracle_accuracy",
            self.source_removal_oracle_accuracy,
            "fraction",
            "higher_is_better",
        );
        push_rate_row(
            &mut output,
            "placebo_invariance",
            self.placebo_invariance,
            "fraction",
            "higher_is_better",
        );
        push_rate_row(
            &mut output,
            "replay_pass_rate",
            self.replay_pass_rate,
            "fraction",
            "higher_is_better",
        );
        push_rate_row(
            &mut output,
            "budget_compliance",
            self.budget_compliance,
            "fraction",
            "higher_is_better",
        );
        push_count_row(&mut output, "task_count", self.task_count, "tasks");
        push_count_row(
            &mut output,
            "correct_decisions",
            self.correct_decisions,
            "decisions",
        );
        push_count_row(&mut output, "total_tokens", self.total_tokens, "tokens");
        push_count_row(
            &mut output,
            "total_tool_calls",
            self.total_tool_calls,
            "calls",
        );
        push_count_row(
            &mut output,
            "total_wall_time_ms",
            self.total_wall_time_ms,
            "milliseconds",
        );
        push_count_row(
            &mut output,
            "total_cost_micros_usd",
            self.total_cost_micros_usd,
            "micro_usd",
        );
        let cost_value = self
            .cost_per_correct_decision_micros_usd
            .map(|value| value.to_string())
            .unwrap_or_default();
        output.push_str(&format!(
            "cost_per_correct_decision_micros_usd,{},{},{cost_value},micro_usd_per_correct_decision,lower_is_better\\n",
            self.total_cost_micros_usd, self.correct_decisions
        ));
        output
    }
}

fn push_rate_row(output: &mut String, name: &str, rate: Rate, unit: &str, direction: &str) {
    let value = rate.value().map(|value| value.to_string()).unwrap_or_default();
    output.push_str(&format!(
        "{name},{},{},{value},{unit},{direction}\\n",
        rate.numerator, rate.denominator
    ));
}

fn push_mean_row(output: &mut String, name: &str, mean: Mean, unit: &str, direction: &str) {
    let numerator = mean.numerator.to_string();
    let value = mean.value().map(|value| value.to_string()).unwrap_or_default();
    output.push_str(&format!(
        "{name},{numerator},{},{value},{unit},{direction}\\n",
        mean.denominator
    ));
}

fn push_count_row(output: &mut String, name: &str, value: u64, unit: &str) {
    output.push_str(&format!("{name},{value},1,{value},{unit},neutral\\n"));
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationError {
    EmptyDataset,
    EmptyTaskId,
    DuplicateTaskId(String),
    InvalidConfidence(String),
    EmptyCitationSourceId(String),
    EmptyCitationOriginId(String),
    EmptyBudgetLimits(String),
    UnexpectedReplayEvidence(String),
    InvalidPlaceboOracle(String),
    ResourceTotalsOverflow,
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => f.write_str("evaluation dataset is empty"),
            Self::EmptyTaskId => f.write_str("task identity is empty"),
            Self::DuplicateTaskId(id) => write!(f, "duplicate task identity: {id}"),
            Self::InvalidConfidence(id) => {
                write!(f, "confidence must be finite and in [0, 1]: {id}")
            }
            Self::EmptyCitationSourceId(id) => {
                write!(f, "citation source identity is empty: {id}")
            }
            Self::EmptyCitationOriginId(id) => {
                write!(f, "citation provenance origin is empty: {id}")
            }
            Self::EmptyBudgetLimits(id) => {
                write!(f, "budget record has no declared limits: {id}")
            }
            Self::UnexpectedReplayEvidence(id) => {
                write!(f, "replay evidence supplied when replay is not required: {id}")
            }
            Self::InvalidPlaceboOracle(id) => {
                write!(f, "placebo oracle must preserve the pre-intervention decision: {id}")
            }
            Self::ResourceTotalsOverflow => {
                f.write_str("resource totals overflowed the supported integer range")
            }
        }
    }
}

impl Error for EvaluationError {}

#[derive(Default)]
struct CalibrationBin {
    count: u64,
    confidence_sum: f64,
    accuracy_sum: f64,
}

fn support_origin_count(record: &TaskRecord) -> usize {
    let mut origins = HashSet::new();
    for citation in &record.citations {
        if citation.eligible_for_claim && citation.supports_expected_decision {
            origins.insert(citation.origin_id.as_str());
        }
    }
    origins.len()
}

fn validate(records: &[TaskRecord]) -> Result<(), EvaluationError> {
    if records.is_empty() {
        return Err(EvaluationError::EmptyDataset);
    }

    let mut task_ids = HashSet::new();
    for record in records {
        let task_id = record.task_id.trim();
        if task_id.is_empty() {
            return Err(EvaluationError::EmptyTaskId);
        }
        if !task_ids.insert(task_id) {
            return Err(EvaluationError::DuplicateTaskId(task_id.to_string()));
        }
        if !record.confidence_correct.is_finite()
            || !(0.0..=1.0).contains(&record.confidence_correct)
        {
            return Err(EvaluationError::InvalidConfidence(task_id.to_string()));
        }
        for citation in &record.citations {
            if citation.source_id.trim().is_empty() {
                return Err(EvaluationError::EmptyCitationSourceId(task_id.to_string()));
            }
            if citation.origin_id.trim().is_empty() {
                return Err(EvaluationError::EmptyCitationOriginId(task_id.to_string()));
            }
        }
        if record.limits.is_some_and(|limits| !limits.has_limit()) {
            return Err(EvaluationError::EmptyBudgetLimits(task_id.to_string()));
        }
        if record.replay.is_some() && !record.replay_required {
            return Err(EvaluationError::UnexpectedReplayEvidence(task_id.to_string()));
        }
        if record.interventions.iter().any(|intervention| {
            intervention.kind == InterventionKind::Placebo
                && intervention.expected_after != intervention.decision_before
        }) {
            return Err(EvaluationError::InvalidPlaceboOracle(task_id.to_string()));
        }
    }
    Ok(())
}

/// Evaluate a set of task records with deterministic, separately reported axes.
/// A rate or mean with a zero denominator remains undefined (value() == None).
pub fn evaluate(records: &[TaskRecord]) -> Result<Scorecard, EvaluationError> {
    validate(records)?;

    let mut correct = 0_u64;
    let mut covered = 0_u64;
    let mut selective_errors = 0_u64;
    let mut brier_sum = 0.0_f64;
    let mut bins: [CalibrationBin; CALIBRATION_BINS] =
        std::array::from_fn(|_| CalibrationBin::default());

    let mut citation_count = 0_u64;
    let mut correct_supporting_citations = 0_u64;
    let mut support_required = 0_u64;
    let mut support_adequate = 0_u64;

    let mut counterfactual_count = 0_u64;
    let mut counterfactual_oracle_correct = 0_u64;
    let mut source_removal_count = 0_u64;
    let mut source_removal_oracle_correct = 0_u64;
    let mut placebo_count = 0_u64;
    let mut placebo_oracle_correct = 0_u64;

    let mut replay_required = 0_u64;
    let mut replay_passed = 0_u64;
    let mut budgeted_tasks = 0_u64;
    let mut budget_compliant = 0_u64;

    let mut total_tokens = 0_u64;
    let mut total_tool_calls = 0_u64;
    let mut total_wall_time_ms = 0_u64;
    let mut total_cost_micros_usd = 0_u64;

    for record in records {
        let is_correct = record.submitted_decision == record.expected_decision;
        if is_correct {
            correct += 1;
        }
        let covered_task = !record.submitted_decision.is_abstention();
        if covered_task {
            covered += 1;
            if !is_correct {
                selective_errors += 1;
            }
        }

        let correctness = if is_correct { 1.0 } else { 0.0 };
        let confidence = record.confidence_correct;
        brier_sum += (confidence - correctness).powi(2);

        let bin_index = ((confidence * CALIBRATION_BINS as f64).floor() as usize)
            .min(CALIBRATION_BINS - 1);
        let bin = &mut bins[bin_index];
        bin.count += 1;
        bin.confidence_sum += confidence;
        bin.accuracy_sum += correctness;

        for citation in &record.citations {
            citation_count += 1;
            if citation.eligible_for_claim && citation.supports_expected_decision {
                correct_supporting_citations += 1;
            }
        }

        if record.minimum_independent_decision_origins > 0 {
            support_required += 1;
            if support_origin_count(record) >= record.minimum_independent_decision_origins {
                support_adequate += 1;
            }
        }

        for intervention in &record.interventions {
            let matched_oracle = intervention.decision_after == intervention.expected_after;
            match intervention.kind {
                InterventionKind::Counterfactual => {
                    counterfactual_count += 1;
                    if matched_oracle {
                        counterfactual_oracle_correct += 1;
                    }
                }
                InterventionKind::SourceRemoval => {
                    source_removal_count += 1;
                    if matched_oracle {
                        source_removal_oracle_correct += 1;
                    }
                }
                InterventionKind::Placebo => {
                    placebo_count += 1;
                    if matched_oracle {
                        placebo_oracle_correct += 1;
                    }
                }
            }
        }

        if record.replay_required {
            replay_required += 1;
            let passed = record.replay.is_some_and(|replay| {
                replay.independent_replay_passed
                    && replay.contract_digest_matches
                    && replay.output_manifest_matches
            });
            if passed {
                replay_passed += 1;
            }
        }

        if let Some(limits) = record.limits {
            budgeted_tasks += 1;
            if limits.contains(record.resources) {
                budget_compliant += 1;
            }
        }

        total_tokens = total_tokens
            .checked_add(record.resources.tokens)
            .ok_or(EvaluationError::ResourceTotalsOverflow)?;
        total_tool_calls = total_tool_calls
            .checked_add(record.resources.tool_calls)
            .ok_or(EvaluationError::ResourceTotalsOverflow)?;
        total_wall_time_ms = total_wall_time_ms
            .checked_add(record.resources.wall_time_ms)
            .ok_or(EvaluationError::ResourceTotalsOverflow)?;
        total_cost_micros_usd = total_cost_micros_usd
            .checked_add(record.resources.cost_micros_usd)
            .ok_or(EvaluationError::ResourceTotalsOverflow)?;
    }

    let task_count = records.len() as u64;
    let ece_numerator = bins.iter().fold(0.0, |sum, bin| {
        if bin.count == 0 {
            sum
        } else {
            let average_confidence = bin.confidence_sum / bin.count as f64;
            let average_accuracy = bin.accuracy_sum / bin.count as f64;
            sum + (average_confidence - average_accuracy).abs() * bin.count as f64
        }
    });

    Ok(Scorecard {
        task_count,
        correct_decisions: correct,
        decision_accuracy: Rate {
            numerator: correct,
            denominator: task_count,
        },
        answer_coverage: Rate {
            numerator: covered,
            denominator: task_count,
        },
        selective_risk: Rate {
            numerator: selective_errors,
            denominator: covered,
        },
        brier_score: Mean {
            numerator: brier_sum,
            denominator: task_count,
        },
        expected_calibration_error_10_bin: Mean {
            numerator: ece_numerator,
            denominator: task_count,
        },
        citation_precision: Rate {
            numerator: correct_supporting_citations,
            denominator: citation_count,
        },
        independent_support_adequacy: Rate {
            numerator: support_adequate,
            denominator: support_required,
        },
        counterfactual_oracle_accuracy: Rate {
            numerator: counterfactual_oracle_correct,
            denominator: counterfactual_count,
        },
        source_removal_oracle_accuracy: Rate {
            numerator: source_removal_oracle_correct,
            denominator: source_removal_count,
        },
        placebo_invariance: Rate {
            numerator: placebo_oracle_correct,
            denominator: placebo_count,
        },
        replay_pass_rate: Rate {
            numerator: replay_passed,
            denominator: replay_required,
        },
        budget_compliance: Rate {
            numerator: budget_compliant,
            denominator: budgeted_tasks,
        },
        total_tokens,
        total_tool_calls,
        total_wall_time_ms,
        total_cost_micros_usd,
        cost_per_correct_decision_micros_usd: (correct > 0)
            .then(|| total_cost_micros_usd as f64 / correct as f64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(task_id: &str, expected: Decision, submitted: Decision) -> TaskRecord {
        TaskRecord {
            task_id: task_id.to_string(),
            expected_decision: expected,
            submitted_decision: submitted,
            confidence_correct: 0.9,
            citations: Vec::new(),
            minimum_independent_decision_origins: 0,
            interventions: Vec::new(),
            replay_required: false,
            replay: None,
            resources: ResourceUse {
                tokens: 1_000,
                tool_calls: 4,
                wall_time_ms: 2_000,
                cost_micros_usd: 250,
            },
            limits: None,
        }
    }

    #[test]
    fn csv_export_is_stable_and_preserves_undefined_denominators() {
        let result = evaluate(&[task(
            "csv",
            Decision::Supported,
            Decision::Supported,
        )])
        .unwrap();
        let csv = result.to_csv();
        assert!(csv.starts_with(
            "metric,numerator,denominator,value,unit,direction\\n"
        ));
        assert!(csv.contains("decision_accuracy,1,1,1,fraction,higher_is_better\\n"));
        assert!(csv.contains("citation_precision,0,0,,fraction,higher_is_better\\n"));
        assert!(csv.contains("cost_per_correct_decision_micros_usd,250,1,250,micro_usd_per_correct_decision,lower_is_better\\n"));
    }

    #[test]
    fn reports_independent_axes_without_a_composite_score() {
        let records = vec![
            task("a", Decision::Supported, Decision::Supported),
            task(
                "b",
                Decision::InsufficientEvidence,
                Decision::InsufficientEvidence,
            ),
        ];
        let result = evaluate(&records).unwrap();
        assert_eq!(result.decision_accuracy.value(), Some(1.0));
        assert_eq!(result.answer_coverage.value(), Some(0.5));
        assert_eq!(result.selective_risk.value(), Some(0.0));
        assert_eq!(result.task_count, 2);
        assert_eq!(result.total_tokens, 2_000);
        assert_eq!(result.cost_per_correct_decision_micros_usd, Some(250.0));
    }

    #[test]
    fn overconfident_wrong_decisions_increase_brier_and_selective_risk() {
        let mut record = task("wrong", Decision::Supported, Decision::Contradicted);
        record.confidence_correct = 0.99;
        let result = evaluate(&[record]).unwrap();
        assert_eq!(result.decision_accuracy.value(), Some(0.0));
        assert_eq!(result.selective_risk.value(), Some(1.0));
        assert!((result.brier_score.value().unwrap() - 0.9801).abs() < 1e-9);
    }

    #[test]
    fn abstention_reduces_coverage_without_hiding_accuracy() {
        let record = task(
            "abstain",
            Decision::Supported,
            Decision::InsufficientEvidence,
        );
        let result = evaluate(&[record]).unwrap();
        assert_eq!(result.decision_accuracy.value(), Some(0.0));
        assert_eq!(result.answer_coverage.value(), Some(0.0));
        assert_eq!(result.selective_risk.value(), None);
    }

    #[test]
    fn mirrored_urls_from_one_origin_count_as_one_independent_origin() {
        let mut record = task("mirrors", Decision::Supported, Decision::Supported);
        record.minimum_independent_decision_origins = 2;
        record.citations = vec![
            CitationRecord {
                source_id: "url-a".to_string(),
                origin_id: "publisher-report-17".to_string(),
                eligible_for_claim: true,
                supports_expected_decision: true,
            },
            CitationRecord {
                source_id: "url-b".to_string(),
                origin_id: "publisher-report-17".to_string(),
                eligible_for_claim: true,
                supports_expected_decision: true,
            },
            CitationRecord {
                source_id: "decoy".to_string(),
                origin_id: "independent-decoy".to_string(),
                eligible_for_claim: true,
                supports_expected_decision: false,
            },
        ];
        let result = evaluate(&[record]).unwrap();
        assert_eq!(result.citation_precision.value(), Some(2.0 / 3.0));
        assert_eq!(result.independent_support_adequacy.value(), Some(0.0));
    }

    #[test]
    fn two_eligible_independent_origins_meet_the_support_requirement() {
        let mut record = task("independent", Decision::Supported, Decision::Supported);
        record.minimum_independent_decision_origins = 2;
        record.citations = ["origin-a", "origin-b"]
            .into_iter()
            .enumerate()
            .map(|(index, origin)| CitationRecord {
                source_id: format!("source-{index}"),
                origin_id: origin.to_string(),
                eligible_for_claim: true,
                supports_expected_decision: true,
            })
            .collect();
        let result = evaluate(&[record]).unwrap();
        assert_eq!(result.independent_support_adequacy.value(), Some(1.0));
    }

    #[test]
    fn counterfactual_and_source_removal_should_change_but_placebo_should_not() {
        let mut record = task("interventions", Decision::Supported, Decision::Supported);
        record.interventions = vec![
            InterventionObservation {
                kind: InterventionKind::Counterfactual,
                decision_before: Decision::Supported,
                decision_after: Decision::Contradicted,
                expected_after: Decision::Contradicted,
            },
            InterventionObservation {
                kind: InterventionKind::SourceRemoval,
                decision_before: Decision::Supported,
                decision_after: Decision::InsufficientEvidence,
                expected_after: Decision::InsufficientEvidence,
            },
            InterventionObservation {
                kind: InterventionKind::Placebo,
                decision_before: Decision::Supported,
                decision_after: Decision::Supported,
                expected_after: Decision::Supported,
            },
        ];
        let result = evaluate(&[record]).unwrap();
        assert_eq!(result.counterfactual_oracle_accuracy.value(), Some(1.0));
        assert_eq!(result.source_removal_oracle_accuracy.value(), Some(1.0));
        assert_eq!(result.placebo_invariance.value(), Some(1.0));
    }

    #[test]
    fn placebo_flip_is_a_failure_not_a_successful_intervention() {
        let mut record = task("placebo-flip", Decision::Supported, Decision::Supported);
        record.interventions.push(InterventionObservation {
            kind: InterventionKind::Placebo,
            decision_before: Decision::Supported,
            decision_after: Decision::Contradicted,
            expected_after: Decision::Supported,
        });
        let result = evaluate(&[record]).unwrap();
        assert_eq!(result.placebo_invariance.value(), Some(0.0));
    }

    #[test]
    fn replay_requires_independent_execution_and_exact_digests() {
        let mut record = task("replay", Decision::Supported, Decision::Supported);
        record.replay_required = true;
        record.replay = Some(ReplayEvidence {
            independent_replay_passed: true,
            contract_digest_matches: true,
            output_manifest_matches: false,
        });
        let result = evaluate(&[record]).unwrap();
        assert_eq!(result.replay_pass_rate.value(), Some(0.0));
    }

    #[test]
    fn missing_required_replay_is_counted_as_failure() {
        let mut record = task("missing-replay", Decision::Supported, Decision::Supported);
        record.replay_required = true;
        let result = evaluate(&[record]).unwrap();
        assert_eq!(result.replay_pass_rate.value(), Some(0.0));
    }

    #[test]
    fn resource_caps_are_checked_without_collapsing_into_accuracy() {
        let mut record = task("over-budget", Decision::Supported, Decision::Supported);
        record.limits = Some(ResourceLimits {
            max_tokens: Some(900),
            max_tool_calls: Some(5),
            max_wall_time_ms: Some(3_000),
            max_cost_micros_usd: Some(300),
        });
        let result = evaluate(&[record]).unwrap();
        assert_eq!(result.decision_accuracy.value(), Some(1.0));
        assert_eq!(result.budget_compliance.value(), Some(0.0));
    }

    #[test]
    fn undefined_axes_are_not_silently_reported_as_zero() {
        let result = evaluate(&[task(
            "no-extra-evidence",
            Decision::Supported,
            Decision::Supported,
        )])
        .unwrap();
        assert_eq!(result.citation_precision.value(), None);
        assert_eq!(result.independent_support_adequacy.value(), None);
        assert_eq!(result.counterfactual_oracle_accuracy.value(), None);
        assert_eq!(result.source_removal_oracle_accuracy.value(), None);
        assert_eq!(result.placebo_invariance.value(), None);
        assert_eq!(result.replay_pass_rate.value(), None);
        assert_eq!(result.budget_compliance.value(), None);
    }

    #[test]
    fn calibration_metrics_are_defined_for_all_decisions() {
        let mut record = task("calibration", Decision::Supported, Decision::Supported);
        record.confidence_correct = 0.8;
        let result = evaluate(&[record]).unwrap();
        assert!((result.brier_score.value().unwrap() - 0.04).abs() < 1e-9);
        assert!((result.expected_calibration_error_10_bin.value().unwrap() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn rejects_duplicate_task_ids() {
        let records = vec![
            task("same", Decision::Supported, Decision::Supported),
            task("same", Decision::Supported, Decision::Supported),
        ];
        assert_eq!(
            evaluate(&records).unwrap_err(),
            EvaluationError::DuplicateTaskId("same".to_string())
        );
    }

    #[test]
    fn rejects_non_finite_or_out_of_range_confidence() {
        for confidence in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
            let mut record = task("bad-confidence", Decision::Supported, Decision::Supported);
            record.confidence_correct = confidence;
            assert!(matches!(
                evaluate(&[record]),
                Err(EvaluationError::InvalidConfidence(_))
            ));
        }
    }

    #[test]
    fn rejects_empty_citation_origin() {
        let mut record = task("bad-origin", Decision::Supported, Decision::Supported);
        record.citations.push(CitationRecord {
            source_id: "source-1".to_string(),
            origin_id: "   ".to_string(),
            eligible_for_claim: true,
            supports_expected_decision: true,
        });
        assert!(matches!(
            evaluate(&[record]),
            Err(EvaluationError::EmptyCitationOriginId(_))
        ));
    }

    #[test]
    fn rejects_budget_records_with_no_declared_limits() {
        let mut record = task("empty-limits", Decision::Supported, Decision::Supported);
        record.limits = Some(ResourceLimits::default());
        assert!(matches!(
            evaluate(&[record]),
            Err(EvaluationError::EmptyBudgetLimits(_))
        ));
    }

    #[test]
    fn rejects_unrequested_replay_evidence() {
        let mut record = task("unexpected-replay", Decision::Supported, Decision::Supported);
        record.replay = Some(ReplayEvidence {
            independent_replay_passed: true,
            contract_digest_matches: true,
            output_manifest_matches: true,
        });
        assert!(matches!(
            evaluate(&[record]),
            Err(EvaluationError::UnexpectedReplayEvidence(_))
        ));
    }

    #[test]
    fn rejects_interventions_that_change_to_the_wrong_counterfactual_answer() {
        let mut record = task("wrong-counterfactual", Decision::Supported, Decision::Supported);
        record.interventions.push(InterventionObservation {
            kind: InterventionKind::Counterfactual,
            decision_before: Decision::Supported,
            decision_after: Decision::Contradicted,
            expected_after: Decision::Supported,
        });
        let result = evaluate(&[record]).unwrap();
        assert_eq!(result.counterfactual_oracle_accuracy.value(), Some(0.0));
    }

    #[test]
    fn rejects_placebo_records_with_a_changed_oracle_target() {
        let mut record = task("bad-placebo-oracle", Decision::Supported, Decision::Supported);
        record.interventions.push(InterventionObservation {
            kind: InterventionKind::Placebo,
            decision_before: Decision::Supported,
            decision_after: Decision::Supported,
            expected_after: Decision::Contradicted,
        });
        assert!(matches!(
            evaluate(&[record]),
            Err(EvaluationError::InvalidPlaceboOracle(_))
        ));
    }

    #[test]
    fn rejects_resource_total_overflow_instead_of_saturating() {
        let mut first = task("overflow-a", Decision::Supported, Decision::Supported);
        first.resources.tokens = u64::MAX;
        let mut second = task("overflow-b", Decision::Supported, Decision::Supported);
        second.resources.tokens = 1;
        assert_eq!(
            evaluate(&[first, second]).unwrap_err(),
            EvaluationError::ResourceTotalsOverflow
        );
    }

    #[test]
    fn rejects_empty_dataset() {
        assert_eq!(evaluate(&[]).unwrap_err(), EvaluationError::EmptyDataset);
    }
}
