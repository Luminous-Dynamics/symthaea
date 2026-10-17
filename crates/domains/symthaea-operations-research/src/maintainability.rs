// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed descriptive evaluation for physical serviceability trials.
//!
//! This measures task execution on controlled, non-flight mock-ups. It is not
//! an aircraft maintenance procedure, a legal authority determination, a
//! return-to-service mechanism, or evidence of airworthiness.

use std::collections::{BTreeMap, BTreeSet};

/// Study metadata, not a legal determination of maintenance privilege.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskClass {
    OwnerObservation,
    OwnerPreventiveMaintenance,
    QualifiedMaintenance,
    SpecialistMaintenance,
}

/// Every assigned task gets an outcome, even when work never starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskOutcome {
    CompletedFirstPass,
    CompletedAfterRework,
    Blocked,
    Aborted,
    NotStarted,
    Failed,
}

impl TaskOutcome {
    pub fn is_completed(self) -> bool {
        matches!(self, Self::CompletedFirstPass | Self::CompletedAfterRework)
    }

    pub fn is_first_pass(self) -> bool {
        self == Self::CompletedFirstPass
    }
}

/// Common task intent, boundary, and acceptance criteria across configurations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskDefinition {
    pub task_id: String,
    pub definition_id: String,
    pub task_class: TaskClass,
}

/// Pre-registered task/maintainer matrix. The minimum paired-completion count
/// must be selected before candidate outcomes are inspected; it is not a power
/// calculation or guarantee of statistical significance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkPlan {
    pub plan_id: String,
    pub tasks: Vec<TaskDefinition>,
    /// Use pseudonymous IDs, not personal identifiers.
    pub maintainer_ids: Vec<String>,
    pub repeats_per_task_maintainer: u16,
    pub minimum_completed_pairs_per_task_for_time_claim: usize,
}

/// One observation per scheduled task, maintainer, and repetition.
/// Unsuccessful trials remain in the denominator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrialObservation {
    pub trial_id: String,
    pub design_id: String,
    pub configuration_id: String,
    pub task_id: String,
    pub task_definition_id: String,
    pub task_class: TaskClass,
    pub task_procedure_revision: String,
    pub maintainer_id: String,
    /// One-based repetition number, bounded by the plan.
    pub repetition: u16,
    pub outcome: TaskOutcome,
    pub active_seconds: u64,
    pub elapsed_seconds: u64,
    /// Mutually exclusive non-active time categories.
    pub parts_wait_seconds: u64,
    pub tool_wait_seconds: u64,
    pub documentation_wait_seconds: u64,
    pub approval_wait_seconds: u64,
    pub required_step_count: u32,
    pub completed_step_count: u32,
    pub rework_count: u32,
    pub unrelated_assemblies_removed: u32,
    /// Count errors using a pre-registered coding rubric.
    pub safety_critical_error_count: u32,
    /// Includes source revisions, observer record, and required evidence.
    /// Incomplete evidence stays visible and blocks a time-improvement claim.
    pub evidence_complete: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    EmptyPlanId,
    EmptyTaskSet,
    EmptyMaintainerSet,
    ZeroRepeats,
    ZeroMinimumPairs,
    DuplicateTaskId(String),
    EmptyTaskDefinitionId(String),
    DuplicateMaintainerId(String),
    EmptyTrialSet,
    EmptyField { trial_id: String, field: &'static str },
    DuplicateTrialId(String),
    MixedDesignId,
    MixedConfigurationId,
    UnknownTask { trial_id: String, task_id: String },
    TaskDefinitionMismatch { trial_id: String, task_id: String },
    UnknownMaintainer { trial_id: String, maintainer_id: String },
    RepeatOutOfRange { trial_id: String, repetition: u16 },
    DuplicateExpectedSlot { task_id: String, maintainer_id: String, repetition: u16 },
    MissingExpectedSlot { task_id: String, maintainer_id: String, repetition: u16 },
    ZeroRequiredSteps { trial_id: String },
    CompletedStepsExceedRequired { trial_id: String },
    CompletedOutcomeHasIncompleteSteps { trial_id: String },
    ActiveTimeExceedsElapsed { trial_id: String },
    WaitCategoriesExceedElapsed { trial_id: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComparisonError {
    BaselineInvalid(Vec<ValidationError>),
    CandidateInvalid(Vec<ValidationError>),
    SameDesignId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantileSummary {
    pub sample_count: usize,
    /// Median of completed trials only; absent when no trial completed.
    pub median_seconds: Option<f64>,
    /// Nearest-rank 90th percentile of completed trials only.
    pub p90_seconds: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutcomeCounts {
    pub completed_first_pass: usize,
    pub completed_after_rework: usize,
    pub blocked: usize,
    pub aborted: usize,
    pub not_started: usize,
    pub failed: usize,
}

impl OutcomeCounts {
    fn from_trials(trials: &[TrialObservation]) -> Self {
        let mut counts = Self {
            completed_first_pass: 0,
            completed_after_rework: 0,
            blocked: 0,
            aborted: 0,
            not_started: 0,
            failed: 0,
        };
        for trial in trials {
            match trial.outcome {
                TaskOutcome::CompletedFirstPass => counts.completed_first_pass += 1,
                TaskOutcome::CompletedAfterRework => counts.completed_after_rework += 1,
                TaskOutcome::Blocked => counts.blocked += 1,
                TaskOutcome::Aborted => counts.aborted += 1,
                TaskOutcome::NotStarted => counts.not_started += 1,
                TaskOutcome::Failed => counts.failed += 1,
            }
        }
        counts
    }

    pub fn completed(&self) -> usize {
        self.completed_first_pass + self.completed_after_rework
    }

    pub fn assigned(&self) -> usize {
        self.completed()
            + self.blocked
            + self.aborted
            + self.not_started
            + self.failed
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunMetrics {
    pub plan_id: String,
    pub design_id: String,
    pub configuration_id: String,
    pub assigned_trials: usize,
    pub outcomes: OutcomeCounts,
    pub completion_rate: f64,
    pub first_pass_rate: f64,
    pub evidence_complete_trials: usize,
    pub evidence_coverage_rate: f64,
    pub safety_critical_error_count: u64,
    pub trials_with_safety_critical_errors: usize,
    pub total_rework_count: u64,
    pub trials_requiring_rework: usize,
    pub total_unrelated_assemblies_removed: u64,
    pub trials_with_unrelated_disassembly: usize,
    pub mean_unrelated_assemblies_removed_per_assigned_trial: f64,
    pub total_active_seconds: u128,
    pub total_elapsed_seconds: u128,
    pub total_parts_wait_seconds: u128,
    pub total_tool_wait_seconds: u128,
    pub total_documentation_wait_seconds: u128,
    pub total_approval_wait_seconds: u128,
    /// Successful-only timing summaries; unsuccessful trials remain in outcomes.
    pub completed_active_time: QuantileSummary,
    pub completed_elapsed_time: QuantileSummary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComparisonBlocker {
    BaselineSafetyCriticalErrors { count: u64 },
    CandidateSafetyCriticalErrors { count: u64 },
    BaselineEvidenceIncomplete { count: usize },
    CandidateEvidenceIncomplete { count: usize },
    CompletionRateRegressed { task_id: String, baseline_completed: usize, candidate_completed: usize },
    FirstPassRateRegressed { task_id: String, baseline_first_pass: usize, candidate_first_pass: usize },
    TooFewCompletedPairs { task_id: String, observed: usize, required: usize },
    ActiveMedianNotImproved { task_id: String },
    ActiveP90Regressed { task_id: String },
    ElapsedMedianNotImproved { task_id: String },
    ElapsedP90Regressed { task_id: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct TaskComparison {
    pub task_id: String,
    pub paired_trials: usize,
    pub paired_trials_completed_by_both: usize,
    pub baseline_completed: usize,
    pub candidate_completed: usize,
    pub baseline_first_pass: usize,
    pub candidate_first_pass: usize,
    /// Signed paired delta, candidate minus baseline. Negative is faster.
    pub median_paired_active_delta_seconds: Option<f64>,
    pub p90_paired_active_delta_seconds: Option<f64>,
    pub median_paired_elapsed_delta_seconds: Option<f64>,
    pub p90_paired_elapsed_delta_seconds: Option<f64>,
    pub blockers: Vec<ComparisonBlocker>,
    pub time_improvement_claim_eligible: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonReport {
    pub baseline: RunMetrics,
    pub candidate: RunMetrics,
    pub global_blockers: Vec<ComparisonBlocker>,
    pub task_results: Vec<TaskComparison>,
}

impl ComparisonReport {
    /// True only when every task shows a descriptive time improvement without
    /// worse completion, first-pass, evidence, or safety gates. Not significance
    /// testing, a certification result, or an airworthiness determination.
    pub fn all_tasks_have_eligible_time_improvement(&self) -> bool {
        self.global_blockers.is_empty()
            && !self.task_results.is_empty()
            && self.task_results.iter().all(|task| task.time_improvement_claim_eligible)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SlotKey {
    task_id: String,
    maintainer_id: String,
    repetition: u16,
}

impl SlotKey {
    fn from_trial(trial: &TrialObservation) -> Self {
        Self {
            task_id: trial.task_id.clone(),
            maintainer_id: trial.maintainer_id.clone(),
            repetition: trial.repetition,
        }
    }
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_plan(plan: &BenchmarkPlan) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();
    if !nonempty(&plan.plan_id) { errors.push(ValidationError::EmptyPlanId); }
    if plan.tasks.is_empty() { errors.push(ValidationError::EmptyTaskSet); }
    if plan.maintainer_ids.is_empty() { errors.push(ValidationError::EmptyMaintainerSet); }
    if plan.repeats_per_task_maintainer == 0 { errors.push(ValidationError::ZeroRepeats); }
    if plan.minimum_completed_pairs_per_task_for_time_claim == 0 {
        errors.push(ValidationError::ZeroMinimumPairs);
    }

    let mut task_ids = BTreeSet::new();
    for task in &plan.tasks {
        if !nonempty(&task.task_id) || !nonempty(&task.definition_id) {
            errors.push(ValidationError::EmptyTaskDefinitionId(task.task_id.clone()));
        }
        if !task_ids.insert(task.task_id.clone()) {
            errors.push(ValidationError::DuplicateTaskId(task.task_id.clone()));
        }
    }

    let mut maintainer_ids = BTreeSet::new();
    for maintainer_id in &plan.maintainer_ids {
        if !nonempty(maintainer_id) || !maintainer_ids.insert(maintainer_id.clone()) {
            errors.push(ValidationError::DuplicateMaintainerId(maintainer_id.clone()));
        }
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

fn quantiles_u64(values: &[u64]) -> QuantileSummary {
    if values.is_empty() {
        return QuantileSummary { sample_count: 0, median_seconds: None, p90_seconds: None };
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] as f64 + sorted[n / 2] as f64) / 2.0
    } else {
        sorted[n / 2] as f64
    };
    let p90_index = ((n as f64 * 0.90).ceil() as usize).max(1) - 1;
    QuantileSummary { sample_count: n, median_seconds: Some(median), p90_seconds: Some(sorted[p90_index]) }
}

fn quantiles_f64(values: &[f64]) -> (Option<f64>, Option<f64>) {
    if values.is_empty() { return (None, None); }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    let p90_index = ((n as f64 * 0.90).ceil() as usize).max(1) - 1;
    (Some(median), Some(sorted[p90_index]))
}

fn validate_trial(
    plan: &BenchmarkPlan,
    trial: &TrialObservation,
    task_definitions: &BTreeMap<&str, &TaskDefinition>,
    maintainer_ids: &BTreeSet<&str>,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    for (field, value) in [
        ("trial_id", trial.trial_id.as_str()),
        ("design_id", trial.design_id.as_str()),
        ("configuration_id", trial.configuration_id.as_str()),
        ("task_id", trial.task_id.as_str()),
        ("task_definition_id", trial.task_definition_id.as_str()),
        ("task_procedure_revision", trial.task_procedure_revision.as_str()),
        ("maintainer_id", trial.maintainer_id.as_str()),
    ] {
        if !nonempty(value) {
            errors.push(ValidationError::EmptyField { trial_id: trial.trial_id.clone(), field });
        }
    }

    match task_definitions.get(trial.task_id.as_str()) {
        None => errors.push(ValidationError::UnknownTask {
            trial_id: trial.trial_id.clone(), task_id: trial.task_id.clone(),
        }),
        Some(definition) if definition.definition_id != trial.task_definition_id
            || definition.task_class != trial.task_class =>
        {
            errors.push(ValidationError::TaskDefinitionMismatch {
                trial_id: trial.trial_id.clone(), task_id: trial.task_id.clone(),
            });
        }
        Some(_) => {}
    }

    if !maintainer_ids.contains(trial.maintainer_id.as_str()) {
        errors.push(ValidationError::UnknownMaintainer {
            trial_id: trial.trial_id.clone(), maintainer_id: trial.maintainer_id.clone(),
        });
    }
    if trial.repetition == 0 || trial.repetition > plan.repeats_per_task_maintainer {
        errors.push(ValidationError::RepeatOutOfRange {
            trial_id: trial.trial_id.clone(), repetition: trial.repetition,
        });
    }
    if trial.required_step_count == 0 {
        errors.push(ValidationError::ZeroRequiredSteps { trial_id: trial.trial_id.clone() });
    }
    if trial.completed_step_count > trial.required_step_count {
        errors.push(ValidationError::CompletedStepsExceedRequired { trial_id: trial.trial_id.clone() });
    }
    if trial.outcome.is_completed() && trial.completed_step_count != trial.required_step_count {
        errors.push(ValidationError::CompletedOutcomeHasIncompleteSteps {
            trial_id: trial.trial_id.clone(),
        });
    }
    if trial.active_seconds > trial.elapsed_seconds {
        errors.push(ValidationError::ActiveTimeExceedsElapsed { trial_id: trial.trial_id.clone() });
    }

    let wait_total = [
        trial.parts_wait_seconds,
        trial.tool_wait_seconds,
        trial.documentation_wait_seconds,
        trial.approval_wait_seconds,
    ].iter().try_fold(0_u64, |sum, value| sum.checked_add(*value));
    let active_plus_wait = wait_total.and_then(|wait| trial.active_seconds.checked_add(wait));
    if active_plus_wait.is_none_or(|total| total > trial.elapsed_seconds) {
        errors.push(ValidationError::WaitCategoriesExceedElapsed { trial_id: trial.trial_id.clone() });
    }
    errors
}

/// Evaluate a single design against the pre-registered task/maintainer matrix.
/// Structural errors fail closed; adverse outcomes remain in valid observations.
pub fn evaluate_run(
    plan: &BenchmarkPlan,
    trials: &[TrialObservation],
) -> Result<RunMetrics, Vec<ValidationError>> {
    validate_plan(plan)?;
    if trials.is_empty() { return Err(vec![ValidationError::EmptyTrialSet]); }

    let task_definitions: BTreeMap<&str, &TaskDefinition> = plan.tasks
        .iter().map(|task| (task.task_id.as_str(), task)).collect();
    let maintainer_ids: BTreeSet<&str> = plan.maintainer_ids.iter().map(String::as_str).collect();
    let mut errors = Vec::new();
    let mut trial_ids = BTreeSet::new();
    let mut slots = BTreeSet::new();
    let mut design_ids = BTreeSet::new();
    let mut configuration_ids = BTreeSet::new();

    for trial in trials {
        if !trial_ids.insert(trial.trial_id.clone()) {
            errors.push(ValidationError::DuplicateTrialId(trial.trial_id.clone()));
        }
        design_ids.insert(trial.design_id.clone());
        configuration_ids.insert(trial.configuration_id.clone());
        errors.extend(validate_trial(plan, trial, &task_definitions, &maintainer_ids));
        let key = SlotKey::from_trial(trial);
        if !slots.insert(key.clone()) {
            errors.push(ValidationError::DuplicateExpectedSlot {
                task_id: key.task_id, maintainer_id: key.maintainer_id, repetition: key.repetition,
            });
        }
    }
    if design_ids.len() != 1 { errors.push(ValidationError::MixedDesignId); }
    if configuration_ids.len() != 1 { errors.push(ValidationError::MixedConfigurationId); }

    for task in &plan.tasks {
        for maintainer_id in &plan.maintainer_ids {
            for repetition in 1..=plan.repeats_per_task_maintainer {
                let key = SlotKey {
                    task_id: task.task_id.clone(), maintainer_id: maintainer_id.clone(), repetition,
                };
                if !slots.contains(&key) {
                    errors.push(ValidationError::MissingExpectedSlot {
                        task_id: key.task_id, maintainer_id: key.maintainer_id, repetition: key.repetition,
                    });
                }
            }
        }
    }
    if !errors.is_empty() { return Err(errors); }

    let first_trial = &trials[0];
    let outcomes = OutcomeCounts::from_trials(trials);
    let assigned = trials.len();
    let completed: Vec<&TrialObservation> = trials.iter().filter(|trial| trial.outcome.is_completed()).collect();
    let active_times: Vec<u64> = completed.iter().map(|trial| trial.active_seconds).collect();
    let elapsed_times: Vec<u64> = completed.iter().map(|trial| trial.elapsed_seconds).collect();
    let evidence_complete_trials = trials.iter().filter(|trial| trial.evidence_complete).count();
    let safety_critical_error_count: u64 = trials.iter().map(|t| u64::from(t.safety_critical_error_count)).sum();
    let trials_with_safety_critical_errors = trials.iter().filter(|t| t.safety_critical_error_count > 0).count();
    let total_rework_count: u64 = trials.iter().map(|t| u64::from(t.rework_count)).sum();
    let trials_requiring_rework = trials.iter().filter(|t| t.rework_count > 0).count();
    let total_unrelated_assemblies_removed: u64 =
        trials.iter().map(|t| u64::from(t.unrelated_assemblies_removed)).sum();
    let trials_with_unrelated_disassembly =
        trials.iter().filter(|t| t.unrelated_assemblies_removed > 0).count();

    Ok(RunMetrics {
        plan_id: plan.plan_id.clone(),
        design_id: first_trial.design_id.clone(),
        configuration_id: first_trial.configuration_id.clone(),
        assigned_trials: assigned,
        outcomes: outcomes.clone(),
        completion_rate: outcomes.completed() as f64 / assigned as f64,
        first_pass_rate: outcomes.completed_first_pass as f64 / assigned as f64,
        evidence_complete_trials,
        evidence_coverage_rate: evidence_complete_trials as f64 / assigned as f64,
        safety_critical_error_count,
        trials_with_safety_critical_errors,
        total_rework_count,
        trials_requiring_rework,
        total_unrelated_assemblies_removed,
        trials_with_unrelated_disassembly,
        mean_unrelated_assemblies_removed_per_assigned_trial:
            total_unrelated_assemblies_removed as f64 / assigned as f64,
        total_active_seconds: trials.iter().map(|t| u128::from(t.active_seconds)).sum(),
        total_elapsed_seconds: trials.iter().map(|t| u128::from(t.elapsed_seconds)).sum(),
        total_parts_wait_seconds: trials.iter().map(|t| u128::from(t.parts_wait_seconds)).sum(),
        total_tool_wait_seconds: trials.iter().map(|t| u128::from(t.tool_wait_seconds)).sum(),
        total_documentation_wait_seconds: trials.iter().map(|t| u128::from(t.documentation_wait_seconds)).sum(),
        total_approval_wait_seconds: trials.iter().map(|t| u128::from(t.approval_wait_seconds)).sum(),
        completed_active_time: quantiles_u64(&active_times),
        completed_elapsed_time: quantiles_u64(&elapsed_times),
    })
}

fn paired_time_deltas(
    baseline: &[TrialObservation],
    candidate: &[TrialObservation],
    task_id: &str,
    active: bool,
) -> (Option<f64>, Option<f64>, usize) {
    let base_map: BTreeMap<SlotKey, &TrialObservation> =
        baseline.iter().map(|t| (SlotKey::from_trial(t), t)).collect();
    let cand_map: BTreeMap<SlotKey, &TrialObservation> =
        candidate.iter().map(|t| (SlotKey::from_trial(t), t)).collect();
    let mut deltas = Vec::new();

    for (key, base) in &base_map {
        if key.task_id != task_id || !base.outcome.is_completed() { continue; }
        let Some(cand) = cand_map.get(key) else { continue; };
        if !cand.outcome.is_completed() { continue; }
        let base_seconds = if active { base.active_seconds } else { base.elapsed_seconds };
        let cand_seconds = if active { cand.active_seconds } else { cand.elapsed_seconds };
        deltas.push(cand_seconds as f64 - base_seconds as f64);
    }
    let count = deltas.len();
    let (median, p90) = quantiles_f64(&deltas);
    (median, p90, count)
}

/// Compare two designs against the same pre-registered observation matrix.
/// The report shows per-task deltas and does not collapse safety, evidence,
/// completion, and time into a compensatory score.
pub fn compare_designs(
    plan: &BenchmarkPlan,
    baseline_trials: &[TrialObservation],
    candidate_trials: &[TrialObservation],
) -> Result<ComparisonReport, ComparisonError> {
    let baseline = evaluate_run(plan, baseline_trials).map_err(ComparisonError::BaselineInvalid)?;
    let candidate = evaluate_run(plan, candidate_trials).map_err(ComparisonError::CandidateInvalid)?;
    if baseline.design_id == candidate.design_id { return Err(ComparisonError::SameDesignId); }

    let mut global_blockers = Vec::new();
    if baseline.safety_critical_error_count > 0 {
        global_blockers.push(ComparisonBlocker::BaselineSafetyCriticalErrors {
            count: baseline.safety_critical_error_count,
        });
    }
    if candidate.safety_critical_error_count > 0 {
        global_blockers.push(ComparisonBlocker::CandidateSafetyCriticalErrors {
            count: candidate.safety_critical_error_count,
        });
    }
    if baseline.evidence_complete_trials != baseline.assigned_trials {
        global_blockers.push(ComparisonBlocker::BaselineEvidenceIncomplete {
            count: baseline.assigned_trials - baseline.evidence_complete_trials,
        });
    }
    if candidate.evidence_complete_trials != candidate.assigned_trials {
        global_blockers.push(ComparisonBlocker::CandidateEvidenceIncomplete {
            count: candidate.assigned_trials - candidate.evidence_complete_trials,
        });
    }

    let mut task_results = Vec::new();
    for task in &plan.tasks {
        let base_task: Vec<&TrialObservation> =
            baseline_trials.iter().filter(|t| t.task_id == task.task_id).collect();
        let cand_task: Vec<&TrialObservation> =
            candidate_trials.iter().filter(|t| t.task_id == task.task_id).collect();
        let base_completed = base_task.iter().filter(|t| t.outcome.is_completed()).count();
        let cand_completed = cand_task.iter().filter(|t| t.outcome.is_completed()).count();
        let base_first = base_task.iter().filter(|t| t.outcome.is_first_pass()).count();
        let cand_first = cand_task.iter().filter(|t| t.outcome.is_first_pass()).count();

        let (median_active, p90_active, active_pairs) =
            paired_time_deltas(baseline_trials, candidate_trials, &task.task_id, true);
        let (median_elapsed, p90_elapsed, elapsed_pairs) =
            paired_time_deltas(baseline_trials, candidate_trials, &task.task_id, false);
        debug_assert_eq!(active_pairs, elapsed_pairs);

        let mut blockers = Vec::new();
        if cand_completed < base_completed {
            blockers.push(ComparisonBlocker::CompletionRateRegressed {
                task_id: task.task_id.clone(), baseline_completed: base_completed, candidate_completed: cand_completed,
            });
        }
        if cand_first < base_first {
            blockers.push(ComparisonBlocker::FirstPassRateRegressed {
                task_id: task.task_id.clone(), baseline_first_pass: base_first, candidate_first_pass: cand_first,
            });
        }
        if active_pairs < plan.minimum_completed_pairs_per_task_for_time_claim {
            blockers.push(ComparisonBlocker::TooFewCompletedPairs {
                task_id: task.task_id.clone(),
                observed: active_pairs,
                required: plan.minimum_completed_pairs_per_task_for_time_claim,
            });
        }
        if !median_active.is_some_and(|value| value < 0.0) {
            blockers.push(ComparisonBlocker::ActiveMedianNotImproved { task_id: task.task_id.clone() });
        }
        if !p90_active.is_some_and(|value| value <= 0.0) {
            blockers.push(ComparisonBlocker::ActiveP90Regressed { task_id: task.task_id.clone() });
        }
        if !median_elapsed.is_some_and(|value| value < 0.0) {
            blockers.push(ComparisonBlocker::ElapsedMedianNotImproved { task_id: task.task_id.clone() });
        }
        if !p90_elapsed.is_some_and(|value| value <= 0.0) {
            blockers.push(ComparisonBlocker::ElapsedP90Regressed { task_id: task.task_id.clone() });
        }

        task_results.push(TaskComparison {
            task_id: task.task_id.clone(),
            paired_trials: base_task.len(),
            paired_trials_completed_by_both: active_pairs,
            baseline_completed: base_completed,
            candidate_completed: cand_completed,
            baseline_first_pass: base_first,
            candidate_first_pass: cand_first,
            median_paired_active_delta_seconds: median_active,
            p90_paired_active_delta_seconds: p90_active,
            median_paired_elapsed_delta_seconds: median_elapsed,
            p90_paired_elapsed_delta_seconds: p90_elapsed,
            time_improvement_claim_eligible: global_blockers.is_empty() && blockers.is_empty(),
            blockers,
        });
    }

    Ok(ComparisonReport { baseline, candidate, global_blockers, task_results })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> BenchmarkPlan {
        BenchmarkPlan {
            plan_id: "serviceability-plan-v1".into(),
            tasks: vec![
                TaskDefinition {
                    task_id: "inspection-access".into(),
                    definition_id: "same-boundary-access-v1".into(),
                    task_class: TaskClass::QualifiedMaintenance,
                },
                TaskDefinition {
                    task_id: "routine-service".into(),
                    definition_id: "same-boundary-service-v1".into(),
                    task_class: TaskClass::OwnerObservation,
                },
            ],
            maintainer_ids: vec!["maintainer-a".into(), "maintainer-b".into()],
            repeats_per_task_maintainer: 1,
            minimum_completed_pairs_per_task_for_time_claim: 2,
        }
    }

    fn trial(design: &str, task_id: &str, maintainer_id: &str, active: u64, elapsed: u64) -> TrialObservation {
        let (definition_id, task_class) = match task_id {
            "inspection-access" => ("same-boundary-access-v1", TaskClass::QualifiedMaintenance),
            "routine-service" => ("same-boundary-service-v1", TaskClass::OwnerObservation),
            _ => ("unknown", TaskClass::OwnerObservation),
        };
        TrialObservation {
            trial_id: format!("{design}-{task_id}-{maintainer_id}"),
            design_id: design.into(),
            configuration_id: format!("{design}-configuration-1"),
            task_id: task_id.into(),
            task_definition_id: definition_id.into(),
            task_class,
            task_procedure_revision: format!("{design}-procedure-r1"),
            maintainer_id: maintainer_id.into(),
            repetition: 1,
            outcome: TaskOutcome::CompletedFirstPass,
            active_seconds: active,
            elapsed_seconds: elapsed,
            parts_wait_seconds: 0,
            tool_wait_seconds: 0,
            documentation_wait_seconds: 0,
            approval_wait_seconds: 0,
            required_step_count: 3,
            completed_step_count: 3,
            rework_count: 0,
            unrelated_assemblies_removed: 0,
            safety_critical_error_count: 0,
            evidence_complete: true,
        }
    }

    fn run(design: &str, active_delta: u64, elapsed_delta: u64) -> Vec<TrialObservation> {
        let samples = [
            ("inspection-access", "maintainer-a", 100_u64, 140_u64),
            ("inspection-access", "maintainer-b", 100_u64, 140_u64),
            ("routine-service", "maintainer-a", 120_u64, 160_u64),
            ("routine-service", "maintainer-b", 120_u64, 160_u64),
        ];
        samples.into_iter().map(|(task, maintainer, active, elapsed)| {
            let is_base = design == "baseline";
            trial(
                design,
                task,
                maintainer,
                if is_base { active } else { active.saturating_sub(active_delta) },
                if is_base { elapsed } else { elapsed.saturating_sub(elapsed_delta) },
            )
        }).collect()
    }

    #[test]
    fn all_assigned_outcomes_remain_in_denominator() {
        let mut observations = run("baseline", 0, 0);
        observations[1].outcome = TaskOutcome::Blocked;
        observations[1].completed_step_count = 1;
        observations[1].active_seconds = 10;
        observations[1].elapsed_seconds = 30;
        let metrics = evaluate_run(&plan(), &observations).unwrap();
        assert_eq!(metrics.assigned_trials, 4);
        assert_eq!(metrics.outcomes.assigned(), 4);
        assert_eq!(metrics.outcomes.blocked, 1);
        assert_eq!(metrics.outcomes.completed(), 3);
        assert!((metrics.completion_rate - 0.75).abs() < 1e-12);
    }

    #[test]
    fn completed_only_quantiles_disclose_their_denominator() {
        let metrics = evaluate_run(&plan(), &run("baseline", 0, 0)).unwrap();
        assert_eq!(metrics.completed_active_time.sample_count, 4);
        assert_eq!(metrics.completed_active_time.median_seconds, Some(110.0));
        assert_eq!(metrics.completed_active_time.p90_seconds, Some(120));
        assert_eq!(metrics.completed_elapsed_time.sample_count, 4);
    }

    #[test]
    fn like_for_like_faster_candidate_is_eligible_per_task() {
        let report = compare_designs(&plan(), &run("baseline", 0, 0), &run("candidate", 20, 30)).unwrap();
        assert!(report.global_blockers.is_empty());
        assert_eq!(report.task_results.len(), 2);
        assert!(report.task_results.iter().all(|task| task.time_improvement_claim_eligible));
        assert!(report.all_tasks_have_eligible_time_improvement());
    }

    #[test]
    fn missing_expected_trial_fails_closed() {
        let baseline = run("baseline", 0, 0);
        let mut candidate = run("candidate", 20, 30);
        candidate.pop();
        let result = compare_designs(&plan(), &baseline, &candidate);
        assert!(matches!(
            result,
            Err(ComparisonError::CandidateInvalid(ref errors))
                if errors.iter().any(|e| matches!(e, ValidationError::MissingExpectedSlot { .. }))
        ));
    }

    #[test]
    fn duplicate_trial_slot_fails_closed_even_with_a_new_trial_id() {
        let mut observations = run("baseline", 0, 0);
        let mut duplicate = observations[0].clone();
        duplicate.trial_id = "separate-id-same-task-pair".into();
        observations.push(duplicate);
        let result = evaluate_run(&plan(), &observations);
        assert!(matches!(
            result,
            Err(ref errors) if errors.iter().any(|e| matches!(e, ValidationError::DuplicateExpectedSlot { .. }))
        ));
    }

    #[test]
    fn mismatched_task_definition_fails_closed() {
        let mut observations = run("baseline", 0, 0);
        observations[0].task_definition_id = "different-boundary".into();
        let result = evaluate_run(&plan(), &observations);
        assert!(matches!(
            result,
            Err(ref errors) if errors.iter().any(|e| matches!(e, ValidationError::TaskDefinitionMismatch { .. }))
        ));
    }

    #[test]
    fn wait_time_must_fit_inside_elapsed_time() {
        let mut observations = run("baseline", 0, 0);
        observations[0].parts_wait_seconds = observations[0].elapsed_seconds;
        let result = evaluate_run(&plan(), &observations);
        assert!(matches!(
            result,
            Err(ref errors) if errors.iter().any(|e| matches!(e, ValidationError::WaitCategoriesExceedElapsed { .. }))
        ));
    }

    #[test]
    fn completed_outcome_requires_all_declared_steps() {
        let mut observations = run("baseline", 0, 0);
        observations[0].completed_step_count = 2;
        let result = evaluate_run(&plan(), &observations);
        assert!(matches!(
            result,
            Err(ref errors) if errors.iter().any(|e| matches!(e, ValidationError::CompletedOutcomeHasIncompleteSteps { .. }))
        ));
    }

    #[test]
    fn safety_critical_error_is_non_compensable() {
        let baseline = run("baseline", 0, 0);
        let mut candidate = run("candidate", 20, 30);
        candidate[0].safety_critical_error_count = 1;
        let report = compare_designs(&plan(), &baseline, &candidate).unwrap();
        assert!(report.global_blockers.iter().any(|b| matches!(b, ComparisonBlocker::CandidateSafetyCriticalErrors { count: 1 })));
        assert!(!report.all_tasks_have_eligible_time_improvement());
        assert!(report.task_results.iter().all(|task| !task.time_improvement_claim_eligible));
    }

    #[test]
    fn incomplete_evidence_blocks_time_claim_without_deleting_trial() {
        let baseline = run("baseline", 0, 0);
        let mut candidate = run("candidate", 20, 30);
        candidate[0].evidence_complete = false;
        let report = compare_designs(&plan(), &baseline, &candidate).unwrap();
        assert!(report.global_blockers.iter().any(|b| matches!(b, ComparisonBlocker::CandidateEvidenceIncomplete { count: 1 })));
        assert_eq!(report.candidate.assigned_trials, 4);
        assert!(!report.all_tasks_have_eligible_time_improvement());
    }

    #[test]
    fn faster_times_cannot_offset_worse_completion_rate() {
        let baseline = run("baseline", 0, 0);
        let mut candidate = run("candidate", 40, 60);
        candidate[1].outcome = TaskOutcome::Blocked;
        candidate[1].completed_step_count = 1;
        candidate[1].active_seconds = 5;
        candidate[1].elapsed_seconds = 10;
        let report = compare_designs(&plan(), &baseline, &candidate).unwrap();
        let affected = report.task_results.iter().find(|t| t.task_id == "inspection-access").unwrap();
        assert!(affected.blockers.iter().any(|b| matches!(b, ComparisonBlocker::CompletionRateRegressed { .. })));
        assert!(!affected.time_improvement_claim_eligible);
    }

    #[test]
    fn disassembly_rework_and_evidence_are_separate_metrics() {
        let mut observations = run("baseline", 0, 0);
        observations[0].unrelated_assemblies_removed = 2;
        observations[0].rework_count = 1;
        observations[0].outcome = TaskOutcome::CompletedAfterRework;
        let metrics = evaluate_run(&plan(), &observations).unwrap();
        assert_eq!(metrics.total_unrelated_assemblies_removed, 2);
        assert_eq!(metrics.trials_with_unrelated_disassembly, 1);
        assert_eq!(metrics.total_rework_count, 1);
        assert_eq!(metrics.outcomes.completed_after_rework, 1);
        assert_eq!(metrics.evidence_complete_trials, 4);
    }
}
