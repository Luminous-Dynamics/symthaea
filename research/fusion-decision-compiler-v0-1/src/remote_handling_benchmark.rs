//! Repeatable remote-handling mockup benchmark and statistical summary.
//!
//! Trial records are grouped only under one frozen benchmark profile. Failed,
//! aborted and censored attempts remain in the denominator. Completion-time
//! quantiles are conditional on successful completion and are always reported
//! beside the all-attempt completion rate.

use std::collections::{BTreeMap, BTreeSet};

pub const REMOTE_HANDLING_BENCHMARK_VERSION: &str = "fusion-remote-handling-benchmark/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum BenchmarkEvidenceClass {
    Synthetic,
    KinematicSimulation,
    DryMockup,
    RepresentativeMockup,
    ActivatedEnvironmentMockup,
    IndependentVerification,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum TrialStatus {
    Completed,
    Failed,
    Aborted,
    RightCensored,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum OperationTrialOutcome {
    Completed,
    Recovered,
    Failed,
    Skipped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteHandlingBenchmarkManifest {
    pub benchmark_id: String,
    pub profile_id: String,
    pub profile_sha256: String,
    pub assembly_revision: String,
    pub geometry_sha256: String,
    pub interface_manifest_sha256: String,
    pub procedure_revision: String,
    pub procedure_sha256: String,
    pub robot_revision: String,
    pub tool_revision: String,
    pub control_stack_revision: String,
    pub environment_profile_id: String,
    pub environment_profile_sha256: String,
    pub task_scope_sha256: String,
    pub required_operation_ids: Vec<String>,
    pub operator_competency_stratum: String,
    pub clock_synchronization_ref: String,
    pub metrology_calibration_ref: String,
    pub evidence_class: BenchmarkEvidenceClass,
    /// One-sided confidence used for the exact completion-rate lower bound and
    /// distribution-free completion-time quantile upper confidence bound.
    pub one_sided_confidence_level: f64,
    /// Probability level of the completion-time quantile (e.g. 0.90 for P90).
    pub completion_time_quantile: f64,
    pub alignment_tolerance_mm: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationTrialEvent {
    pub operation_id: String,
    pub start_offset_seconds: u64,
    pub end_offset_seconds: u64,
    pub outcome: OperationTrialOutcome,
    pub retry_count: u32,
    pub intervention_count: u32,
    pub recovery_action_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RemoteHandlingTrial {
    pub trial_id: String,
    pub profile_sha256: String,
    pub assembly_revision: String,
    pub geometry_sha256: String,
    pub interface_manifest_sha256: String,
    pub procedure_revision: String,
    pub procedure_sha256: String,
    pub robot_revision: String,
    pub tool_revision: String,
    pub control_stack_revision: String,
    pub environment_profile_id: String,
    pub environment_profile_sha256: String,
    pub task_scope_sha256: String,
    pub operator_competency_stratum: String,
    pub clock_synchronization_ref: String,
    pub metrology_calibration_ref: Option<String>,
    pub evidence_class: BenchmarkEvidenceClass,
    pub raw_data_ref: String,
    pub raw_data_sha256: String,
    /// Monotonic elapsed duration for the complete attempt, including retries,
    /// recovery, inspection and restart tasks included in the frozen scope.
    pub attempt_duration_seconds: u64,
    pub status: TrialStatus,
    pub safe_terminal_state: bool,
    pub operation_events: Vec<OperationTrialEvent>,
    pub measured_alignment_error_mm: Option<f64>,
    pub dropped_grasp_count: u32,
    pub collision_contact_alarm_count: u32,
    pub connector_error_count: u32,
    pub tool_change_count: u32,
    pub rework_operation_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchmarkDisposition {
    NotEvaluable,
    ValidBenchmarkSummary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchmarkDiagnostic {
    EmptyBenchmarkIdentity,
    InvalidProfileDigest,
    InvalidGeometryDigest,
    InvalidInterfaceDigest,
    InvalidProcedureDigest,
    InvalidEnvironmentDigest,
    InvalidTaskScopeDigest,
    EmptyRequiredOperation,
    DuplicateRequiredOperation,
    EmptyOperatorCompetencyStratum,
    MissingClockSynchronizationReference,
    MissingMetrologyCalibrationReference,
    InvalidConfidenceLevel,
    InvalidCompletionQuantile,
    InvalidAlignmentTolerance,
    NoTrials,
    EmptyTrialIdentity,
    DuplicateTrialIdentity,
    ReusedRawArtifactDigest,
    EmptyRawArtifactReference,
    InvalidRawArtifactDigest,
    ProfileIdentityMismatch,
    AssemblyRevisionMismatch,
    GeometryMismatch,
    InterfaceManifestMismatch,
    ProcedureRevisionMismatch,
    ProcedureDigestMismatch,
    RobotRevisionMismatch,
    ToolRevisionMismatch,
    ControlStackRevisionMismatch,
    EnvironmentIdentityMismatch,
    EnvironmentDigestMismatch,
    TaskScopeMismatch,
    OperatorCompetencyMismatch,
    ClockSynchronizationMismatch,
    EvidenceClassMismatch,
    InvalidAttemptDuration,
    CompletedWithoutSafeTerminalState,
    MissingCompletedOperation,
    DuplicateOperationEvent,
    UnknownOperationEvent,
    OperationOutsideAttempt,
    InvalidOperationDuration,
    FailedOperationInCompletedTrial,
    MissingMetrologyCalibration,
    InvalidAlignmentMeasurement,
    DuplicateMetrologyArtifactUse,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceClassCount {
    pub evidence_class: BenchmarkEvidenceClass,
    pub trial_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RemoteHandlingBenchmarkReport {
    pub contract_version: &'static str,
    pub benchmark_id: String,
    pub profile_id: String,
    pub disposition: BenchmarkDisposition,
    pub diagnostics: Vec<BenchmarkDiagnostic>,
    pub evidence_class: BenchmarkEvidenceClass,
    pub attempted_trials: usize,
    pub completed_trials: usize,
    pub failed_trials: usize,
    pub aborted_trials: usize,
    pub right_censored_trials: usize,
    pub safe_terminal_trials: usize,
    pub unsafe_terminal_trials: usize,
    pub completion_fraction: Option<f64>,
    pub completion_fraction_one_sided_lower_bound: Option<f64>,
    pub one_sided_confidence_level: f64,
    /// P50 and P90 are empirical order statistics conditional on completion.
    pub completed_duration_p50_seconds: Option<u64>,
    pub completed_duration_p90_seconds: Option<u64>,
    /// Distribution-free one-sided upper confidence limit for the declared
    /// conditional completion-time quantile. None means insufficient repeats.
    pub completion_time_quantile_upper_confidence_seconds: Option<u64>,
    pub completed_duration_sample_count: usize,
    pub interventions_total: u64,
    pub retries_total: u64,
    pub recovery_actions_total: u64,
    pub dropped_grasps_total: u64,
    pub collision_contact_alarms_total: u64,
    pub connector_errors_total: u64,
    pub tool_changes_total: u64,
    pub rework_operations_total: u64,
    pub alignment_measurement_count: usize,
    pub alignment_tolerance_failures: usize,
    pub maximum_alignment_error_mm: Option<f64>,
    pub evidence_class_counts: Vec<EvidenceClassCount>,
    pub canonical_trial_ids: Vec<String>,
    pub claim_ceiling: &'static str,
}

pub fn summarize_remote_handling_benchmark(
    manifest: &RemoteHandlingBenchmarkManifest,
    trials: &[RemoteHandlingTrial],
) -> RemoteHandlingBenchmarkReport {
    let mut diagnostics = validate_manifest(manifest);
    if trials.is_empty() {
        diagnostics.push(BenchmarkDiagnostic::NoTrials);
    }
    diagnostics.extend(validate_trials(manifest, trials));
    diagnostics.sort_by_key(|item| format!("{item:?}"));
    diagnostics.dedup();

    let attempted = trials.len();
    let completed_durations: Vec<u64> = trials
        .iter()
        .filter(|trial| trial.status == TrialStatus::Completed)
        .map(|trial| trial.attempt_duration_seconds)
        .collect();
    let mut sorted_durations = completed_durations.clone();
    sorted_durations.sort_unstable();

    let completed = trials
        .iter()
        .filter(|trial| trial.status == TrialStatus::Completed)
        .count();
    let failed = trials
        .iter()
        .filter(|trial| trial.status == TrialStatus::Failed)
        .count();
    let aborted = trials
        .iter()
        .filter(|trial| trial.status == TrialStatus::Aborted)
        .count();
    let censored = trials
        .iter()
        .filter(|trial| trial.status == TrialStatus::RightCensored)
        .count();
    let safe_terminal = trials.iter().filter(|trial| trial.safe_terminal_state).count();
    let unsafe_terminal = attempted.saturating_sub(safe_terminal);

    let completion_fraction = (attempted > 0).then_some(completed as f64 / attempted as f64);
    let completion_fraction_lower_bound = (attempted > 0).then(|| {
        clopper_pearson_one_sided_lower(
            completed,
            attempted,
            1.0 - manifest.one_sided_confidence_level,
        )
    });

    let alignment_values: Vec<f64> = trials
        .iter()
        .filter_map(|trial| trial.measured_alignment_error_mm)
        .collect();
    let maximum_alignment_error_mm = alignment_values
        .iter()
        .copied()
        .reduce(f64::max);
    let alignment_tolerance_failures = alignment_values
        .iter()
        .filter(|value| **value > manifest.alignment_tolerance_mm)
        .count();

    let mut evidence_counts: BTreeMap<BenchmarkEvidenceClass, usize> = BTreeMap::new();
    for trial in trials {
        *evidence_counts.entry(trial.evidence_class).or_default() += 1;
    }
    let evidence_class_counts = evidence_counts
        .into_iter()
        .map(|(evidence_class, trial_count)| EvidenceClassCount {
            evidence_class,
            trial_count,
        })
        .collect();

    let canonical_trial_ids = {
        let mut ids: Vec<String> = trials.iter().map(|trial| trial.trial_id.clone()).collect();
        ids.sort();
        ids
    };

    let sum_u32 = |select: fn(&RemoteHandlingTrial) -> u32| -> u64 {
        trials
            .iter()
            .map(|trial| u64::from(select(trial)))
            .sum()
    };
    let sum_operation = |select: fn(&OperationTrialEvent) -> u64| -> u64 {
        trials
            .iter()
            .flat_map(|trial| &trial.operation_events)
            .map(select)
            .sum()
    };

    let disposition = if diagnostics.is_empty() {
        BenchmarkDisposition::ValidBenchmarkSummary
    } else {
        BenchmarkDisposition::NotEvaluable
    };
    let confidence = manifest.one_sided_confidence_level;
    let quantile = manifest.completion_time_quantile;
    let quantile_upper = if sorted_durations.is_empty() {
        None
    } else {
        upper_order_statistic_index(
            sorted_durations.len(),
            quantile,
            confidence,
        )
        .map(|one_based_index| sorted_durations[one_based_index - 1])
    };

    RemoteHandlingBenchmarkReport {
        contract_version: REMOTE_HANDLING_BENCHMARK_VERSION,
        benchmark_id: manifest.benchmark_id.clone(),
        profile_id: manifest.profile_id.clone(),
        disposition,
        diagnostics,
        evidence_class: manifest.evidence_class,
        attempted_trials: attempted,
        completed_trials: completed,
        failed_trials: failed,
        aborted_trials: aborted,
        right_censored_trials: censored,
        safe_terminal_trials: safe_terminal,
        unsafe_terminal_trials: unsafe_terminal,
        completion_fraction,
        completion_fraction_one_sided_lower_bound: completion_fraction_lower_bound,
        one_sided_confidence_level: confidence,
        completed_duration_p50_seconds: nearest_rank(&sorted_durations, 0.50),
        completed_duration_p90_seconds: nearest_rank(&sorted_durations, quantile),
        completion_time_quantile_upper_confidence_seconds: quantile_upper,
        completed_duration_sample_count: sorted_durations.len(),
        interventions_total: sum_operation(|event| u64::from(event.intervention_count)),
        retries_total: sum_operation(|event| u64::from(event.retry_count)),
        recovery_actions_total: sum_operation(|event| u64::from(event.recovery_action_count)),
        dropped_grasps_total: sum_u32(|trial| trial.dropped_grasp_count),
        collision_contact_alarms_total: sum_u32(|trial| trial.collision_contact_alarm_count),
        connector_errors_total: sum_u32(|trial| trial.connector_error_count),
        tool_changes_total: sum_u32(|trial| trial.tool_change_count),
        rework_operations_total: sum_u32(|trial| trial.rework_operation_count),
        alignment_measurement_count: alignment_values.len(),
        alignment_tolerance_failures,
        maximum_alignment_error_mm,
        evidence_class_counts,
        canonical_trial_ids,
        claim_ceiling: "repeatable benchmark summary over supplied frozen trial records only",
    }
}

fn validate_manifest(manifest: &RemoteHandlingBenchmarkManifest) -> Vec<BenchmarkDiagnostic> {
    let mut diagnostics = Vec::new();
    if manifest.benchmark_id.trim().is_empty() || manifest.profile_id.trim().is_empty() {
        diagnostics.push(BenchmarkDiagnostic::EmptyBenchmarkIdentity);
    }
    if !is_sha256(&manifest.profile_sha256) {
        diagnostics.push(BenchmarkDiagnostic::InvalidProfileDigest);
    }
    if !is_sha256(&manifest.geometry_sha256) {
        diagnostics.push(BenchmarkDiagnostic::InvalidGeometryDigest);
    }
    if !is_sha256(&manifest.interface_manifest_sha256) {
        diagnostics.push(BenchmarkDiagnostic::InvalidInterfaceDigest);
    }
    if !is_sha256(&manifest.procedure_sha256) {
        diagnostics.push(BenchmarkDiagnostic::InvalidProcedureDigest);
    }
    if !is_sha256(&manifest.environment_profile_sha256) {
        diagnostics.push(BenchmarkDiagnostic::InvalidEnvironmentDigest);
    }
    if !is_sha256(&manifest.task_scope_sha256) {
        diagnostics.push(BenchmarkDiagnostic::InvalidTaskScopeDigest);
    }
    if manifest.required_operation_ids.is_empty()
        || manifest.required_operation_ids.iter().any(|id| id.trim().is_empty())
    {
        diagnostics.push(BenchmarkDiagnostic::EmptyRequiredOperation);
    }
    let mut operation_ids = BTreeSet::new();
    for operation_id in &manifest.required_operation_ids {
        if !operation_ids.insert(operation_id.as_str()) {
            diagnostics.push(BenchmarkDiagnostic::DuplicateRequiredOperation);
        }
    }
    if manifest.operator_competency_stratum.trim().is_empty() {
        diagnostics.push(BenchmarkDiagnostic::EmptyOperatorCompetencyStratum);
    }
    if manifest.clock_synchronization_ref.trim().is_empty() {
        diagnostics.push(BenchmarkDiagnostic::MissingClockSynchronizationReference);
    }
    if manifest.metrology_calibration_ref.trim().is_empty() {
        diagnostics.push(BenchmarkDiagnostic::MissingMetrologyCalibrationReference);
    }
    if !manifest.one_sided_confidence_level.is_finite()
        || !(0.0..1.0).contains(&manifest.one_sided_confidence_level)
    {
        diagnostics.push(BenchmarkDiagnostic::InvalidConfidenceLevel);
    }
    if !manifest.completion_time_quantile.is_finite()
        || !(0.0..1.0).contains(&manifest.completion_time_quantile)
    {
        diagnostics.push(BenchmarkDiagnostic::InvalidCompletionQuantile);
    }
    if !manifest.alignment_tolerance_mm.is_finite() || manifest.alignment_tolerance_mm <= 0.0 {
        diagnostics.push(BenchmarkDiagnostic::InvalidAlignmentTolerance);
    }
    diagnostics
}

fn validate_trials(
    manifest: &RemoteHandlingBenchmarkManifest,
    trials: &[RemoteHandlingTrial],
) -> Vec<BenchmarkDiagnostic> {
    let mut diagnostics = Vec::new();
    let mut trial_ids = BTreeSet::new();
    let mut raw_digests = BTreeSet::new();
    let mut metrology_digests = BTreeSet::new();
    let required: BTreeSet<&str> = manifest
        .required_operation_ids
        .iter()
        .map(String::as_str)
        .collect();

    for trial in trials {
        if trial.trial_id.trim().is_empty() {
            diagnostics.push(BenchmarkDiagnostic::EmptyTrialIdentity);
        }
        if !trial_ids.insert(trial.trial_id.as_str()) {
            diagnostics.push(BenchmarkDiagnostic::DuplicateTrialIdentity);
        }
        if trial.raw_data_ref.trim().is_empty() {
            diagnostics.push(BenchmarkDiagnostic::EmptyRawArtifactReference);
        }
        if !is_sha256(&trial.raw_data_sha256) {
            diagnostics.push(BenchmarkDiagnostic::InvalidRawArtifactDigest);
        } else if !raw_digests.insert(trial.raw_data_sha256.as_str()) {
            diagnostics.push(BenchmarkDiagnostic::ReusedRawArtifactDigest);
        }
        if trial.attempt_duration_seconds == 0 {
            diagnostics.push(BenchmarkDiagnostic::InvalidAttemptDuration);
        }
        if trial.profile_sha256 != manifest.profile_sha256 {
            diagnostics.push(BenchmarkDiagnostic::ProfileIdentityMismatch);
        }
        if trial.assembly_revision != manifest.assembly_revision {
            diagnostics.push(BenchmarkDiagnostic::AssemblyRevisionMismatch);
        }
        if trial.geometry_sha256 != manifest.geometry_sha256 {
            diagnostics.push(BenchmarkDiagnostic::GeometryMismatch);
        }
        if trial.interface_manifest_sha256 != manifest.interface_manifest_sha256 {
            diagnostics.push(BenchmarkDiagnostic::InterfaceManifestMismatch);
        }
        if trial.procedure_revision != manifest.procedure_revision {
            diagnostics.push(BenchmarkDiagnostic::ProcedureRevisionMismatch);
        }
        if trial.procedure_sha256 != manifest.procedure_sha256 {
            diagnostics.push(BenchmarkDiagnostic::ProcedureDigestMismatch);
        }
        if trial.robot_revision != manifest.robot_revision {
            diagnostics.push(BenchmarkDiagnostic::RobotRevisionMismatch);
        }
        if trial.tool_revision != manifest.tool_revision {
            diagnostics.push(BenchmarkDiagnostic::ToolRevisionMismatch);
        }
        if trial.control_stack_revision != manifest.control_stack_revision {
            diagnostics.push(BenchmarkDiagnostic::ControlStackRevisionMismatch);
        }
        if trial.environment_profile_id != manifest.environment_profile_id {
            diagnostics.push(BenchmarkDiagnostic::EnvironmentIdentityMismatch);
        }
        if trial.environment_profile_sha256 != manifest.environment_profile_sha256 {
            diagnostics.push(BenchmarkDiagnostic::EnvironmentDigestMismatch);
        }
        if trial.task_scope_sha256 != manifest.task_scope_sha256 {
            diagnostics.push(BenchmarkDiagnostic::TaskScopeMismatch);
        }
        if trial.operator_competency_stratum != manifest.operator_competency_stratum {
            diagnostics.push(BenchmarkDiagnostic::OperatorCompetencyMismatch);
        }
        if trial.clock_synchronization_ref != manifest.clock_synchronization_ref {
            diagnostics.push(BenchmarkDiagnostic::ClockSynchronizationMismatch);
        }
        if trial.evidence_class != manifest.evidence_class {
            diagnostics.push(BenchmarkDiagnostic::EvidenceClassMismatch);
        }
        if trial.status == TrialStatus::Completed && !trial.safe_terminal_state {
            diagnostics.push(BenchmarkDiagnostic::CompletedWithoutSafeTerminalState);
        }
        if let Some(error_mm) = trial.measured_alignment_error_mm {
            if !error_mm.is_finite() || error_mm < 0.0 {
                diagnostics.push(BenchmarkDiagnostic::InvalidAlignmentMeasurement);
            }
            if trial.metrology_calibration_ref.as_deref()
                != Some(manifest.metrology_calibration_ref.as_str())
            {
                diagnostics.push(BenchmarkDiagnostic::MissingMetrologyCalibration);
            } else if !metrology_digests.insert(
                trial.metrology_calibration_ref.as_deref().unwrap_or_default(),
            ) {
                // Repeated use of the same calibration is expected and valid:
                // the manifest fixes one calibration reference for the set.
                // This collection is intentionally not used as a uniqueness gate.
            }
        }

        let mut seen_operations = BTreeSet::new();
        for event in &trial.operation_events {
            if !required.contains(event.operation_id.as_str()) {
                diagnostics.push(BenchmarkDiagnostic::UnknownOperationEvent);
            }
            if !seen_operations.insert(event.operation_id.as_str()) {
                diagnostics.push(BenchmarkDiagnostic::DuplicateOperationEvent);
            }
            if event.end_offset_seconds <= event.start_offset_seconds
                || event.end_offset_seconds > trial.attempt_duration_seconds
            {
                diagnostics.push(BenchmarkDiagnostic::OperationOutsideAttempt);
            }
            if event.outcome == OperationTrialOutcome::Failed
                && trial.status == TrialStatus::Completed
            {
                diagnostics.push(BenchmarkDiagnostic::FailedOperationInCompletedTrial);
            }
        }
        if trial.status == TrialStatus::Completed
            && !required.iter().all(|operation_id| seen_operations.contains(operation_id))
        {
            diagnostics.push(BenchmarkDiagnostic::MissingCompletedOperation);
        }
    }
    diagnostics
}

fn nearest_rank(sorted: &[u64], quantile: f64) -> Option<u64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = ((quantile * sorted.len() as f64).ceil() as usize).max(1);
    sorted.get(rank.saturating_sub(1)).copied()
}

/// Return a 1-based order-statistic index whose value is a distribution-free
/// one-sided upper confidence limit for the population p-quantile.
fn upper_order_statistic_index(n: usize, p: f64, confidence: f64) -> Option<usize> {
    if n == 0 || !(0.0..1.0).contains(&p) || !(0.0..1.0).contains(&confidence) {
        return None;
    }
    if binomial_cdf(n, p, n - 1) < confidence {
        return None;
    }
    let mut low = 1_usize;
    let mut high = n;
    while low < high {
        let middle = low + (high - low) / 2;
        if binomial_cdf(n, p, middle - 1) >= confidence {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    Some(low)
}

/// Exact one-sided Clopper-Pearson lower confidence bound for a binomial
/// completion fraction. The numeric implementation uses log-domain f64.
fn clopper_pearson_one_sided_lower(successes: usize, trials: usize, alpha: f64) -> f64 {
    if trials == 0 || successes == 0 {
        return 0.0;
    }
    if successes == trials {
        return alpha.powf(1.0 / trials as f64);
    }
    let mut low = 0.0_f64;
    let mut high = 1.0_f64;
    for _ in 0..80 {
        let middle = (low + high) / 2.0;
        if binomial_tail(trials, successes, middle) >= alpha {
            high = middle;
        } else {
            low = middle;
        }
    }
    low
}

fn binomial_cdf(n: usize, p: f64, k_max: usize) -> f64 {
    if k_max >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    log_sum_binomial(n, p, 0, k_max)
}

fn binomial_tail(n: usize, k_min: usize, p: f64) -> f64 {
    if k_min == 0 {
        return 1.0;
    }
    if k_min > n {
        return 0.0;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    log_sum_binomial(n, p, k_min, n)
}

fn log_sum_binomial(n: usize, p: f64, start: usize, end: usize) -> f64 {
    let log_choose = |k: usize| {
        let k = k.min(n - k);
        (1..=k)
            .map(|i| ((n - k + i) as f64 / i as f64).ln())
            .sum::<f64>()
    };
    let log_p = p.ln();
    let log_q = (1.0 - p).ln();
    let mut logs = Vec::with_capacity(end - start + 1);
    for k in start..=end {
        logs.push(
            log_choose(k) + k as f64 * log_p + (n - k) as f64 * log_q,
        );
    }
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum + logs.iter().map(|value| (value - maximum).exp()).sum::<f64>().ln()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn manifest() -> RemoteHandlingBenchmarkManifest {
        RemoteHandlingBenchmarkManifest {
            benchmark_id: "rh-blanket-mockup-v1".to_string(),
            profile_id: "blanket-module-profile-v1".to_string(),
            profile_sha256: sha('a'),
            assembly_revision: "blanket-assembly-r4".to_string(),
            geometry_sha256: sha('b'),
            interface_manifest_sha256: sha('c'),
            procedure_revision: "procedure-r3".to_string(),
            procedure_sha256: sha('d'),
            robot_revision: "robot-r2".to_string(),
            tool_revision: "tooling-r5".to_string(),
            control_stack_revision: "controller-r7".to_string(),
            environment_profile_id: "dry-mockup-envelope-v1".to_string(),
            environment_profile_sha256: sha('e'),
            task_scope_sha256: sha('f'),
            required_operation_ids: vec!["isolate".to_string(), "disconnect".to_string(), "extract".to_string(), "install".to_string(), "reconnect".to_string(), "requalify".to_string(), "restart".to_string()],
            operator_competency_stratum: "qualified-remote-handling-team-level-2".to_string(),
            clock_synchronization_ref: "ptp-clock-calibration-v1".to_string(),
            metrology_calibration_ref: "laser-tracker-calibration-v1".to_string(),
            evidence_class: BenchmarkEvidenceClass::DryMockup,
            one_sided_confidence_level: 0.95,
            completion_time_quantile: 0.90,
            alignment_tolerance_mm: 0.5,
        }
    }

    fn trial(id: &str, duration: u64, status: TrialStatus) -> RemoteHandlingTrial {
        let manifest = manifest();
        let operations = manifest
            .required_operation_ids
            .iter()
            .enumerate()
            .map(|(index, operation_id)| OperationTrialEvent {
                operation_id: operation_id.clone(),
                start_offset_seconds: index as u64 * 10,
                end_offset_seconds: index as u64 * 10 + 5,
                outcome: OperationTrialOutcome::Completed,
                retry_count: 0,
                intervention_count: 0,
                recovery_action_count: 0,
            })
            .collect();
        RemoteHandlingTrial {
            trial_id: id.to_string(),
            profile_sha256: manifest.profile_sha256,
            assembly_revision: manifest.assembly_revision,
            geometry_sha256: manifest.geometry_sha256,
            interface_manifest_sha256: manifest.interface_manifest_sha256,
            procedure_revision: manifest.procedure_revision,
            procedure_sha256: manifest.procedure_sha256,
            robot_revision: manifest.robot_revision,
            tool_revision: manifest.tool_revision,
            control_stack_revision: manifest.control_stack_revision,
            environment_profile_id: manifest.environment_profile_id,
            environment_profile_sha256: manifest.environment_profile_sha256,
            task_scope_sha256: manifest.task_scope_sha256,
            operator_competency_stratum: manifest.operator_competency_stratum,
            clock_synchronization_ref: manifest.clock_synchronization_ref,
            metrology_calibration_ref: Some(manifest.metrology_calibration_ref),
            evidence_class: manifest.evidence_class,
            raw_data_ref: format!("raw-trial-{id}"),
            raw_data_sha256: sha(id.chars().next().unwrap_or('9')),
            attempt_duration_seconds: duration,
            status,
            safe_terminal_state: status != TrialStatus::Failed,
            operation_events: if status == TrialStatus::Completed { operations } else { Vec::new() },
            measured_alignment_error_mm: Some(0.2),
            dropped_grasp_count: 0,
            collision_contact_alarm_count: 0,
            connector_error_count: 0,
            tool_change_count: 1,
            rework_operation_count: 0,
        }
    }

    fn thirty_successful_trials() -> Vec<RemoteHandlingTrial> {
        (0..30)
            .map(|index| trial(&format!("trial-{index:02}"), 600 + index as u64, TrialStatus::Completed))
            .collect()
    }

    #[test]
    fn thirty_trials_support_distribution_free_p90_upper_bound() {
        let report = summarize_remote_handling_benchmark(&manifest(), &thirty_successful_trials());
        assert_eq!(report.disposition, BenchmarkDisposition::ValidBenchmarkSummary);
        assert_eq!(report.attempted_trials, 30);
        assert_eq!(report.completed_trials, 30);
        assert_eq!(report.completion_fraction, Some(1.0));
        assert!(report.completion_fraction_one_sided_lower_bound.unwrap() > 0.90);
        assert_eq!(report.completed_duration_p50_seconds, Some(615));
        assert_eq!(report.completed_duration_p90_seconds, Some(626));
        assert_eq!(report.completion_time_quantile_upper_confidence_seconds, Some(629));
    }

    #[test]
    fn failed_and_aborted_trials_remain_in_completion_denominator() {
        let mut trials = thirty_successful_trials();
        trials[0].status = TrialStatus::Failed;
        trials[0].safe_terminal_state = true;
        trials[0].operation_events.clear();
        trials[1].status = TrialStatus::Aborted;
        trials[1].safe_terminal_state = true;
        trials[1].operation_events.clear();
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert_eq!(report.attempted_trials, 30);
        assert_eq!(report.completed_trials, 28);
        assert_eq!(report.failed_trials, 1);
        assert_eq!(report.aborted_trials, 1);
        assert_eq!(report.completion_fraction, Some(28.0 / 30.0));
    }

    #[test]
    fn small_sample_does_not_emit_p90_upper_confidence_bound() {
        let trials: Vec<_> = (0..8)
            .map(|index| trial(&format!("small-{index}"), 100 + index, TrialStatus::Completed))
            .collect();
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert_eq!(report.disposition, BenchmarkDisposition::ValidBenchmarkSummary);
        assert_eq!(report.completed_duration_p90_seconds, Some(107));
        assert_eq!(report.completion_time_quantile_upper_confidence_seconds, None);
    }

    #[test]
    fn mixed_geometry_or_task_scope_is_not_pooled() {
        let mut trials = thirty_successful_trials();
        trials[0].geometry_sha256 = sha('9');
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert_eq!(report.disposition, BenchmarkDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&BenchmarkDiagnostic::GeometryMismatch));
    }

    #[test]
    fn reused_raw_artifact_digest_is_not_a_second_trial() {
        let mut trials = thirty_successful_trials();
        trials[1].raw_data_sha256 = trials[0].raw_data_sha256.clone();
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert!(report.diagnostics.contains(&BenchmarkDiagnostic::ReusedRawArtifactDigest));
    }

    #[test]
    fn duplicate_trial_identity_is_refused() {
        let mut trials = thirty_successful_trials();
        trials[1].trial_id = trials[0].trial_id.clone();
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert!(report.diagnostics.contains(&BenchmarkDiagnostic::DuplicateTrialIdentity));
    }

    #[test]
    fn completed_trial_requires_every_frozen_operation() {
        let mut trials = thirty_successful_trials();
        trials[0].operation_events.pop();
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert!(report.diagnostics.contains(&BenchmarkDiagnostic::MissingCompletedOperation));
    }

    #[test]
    fn completed_trial_requires_safe_terminal_state() {
        let mut trials = thirty_successful_trials();
        trials[0].safe_terminal_state = false;
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert!(report.diagnostics.contains(&BenchmarkDiagnostic::CompletedWithoutSafeTerminalState));
    }

    #[test]
    fn profile_and_procedure_identity_must_match_exactly() {
        let mut trials = thirty_successful_trials();
        trials[0].profile_sha256 = sha('9');
        trials[1].procedure_sha256 = sha('8');
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert!(report.diagnostics.contains(&BenchmarkDiagnostic::ProfileIdentityMismatch));
        assert!(report.diagnostics.contains(&BenchmarkDiagnostic::ProcedureDigestMismatch));
    }

    #[test]
    fn evidence_class_is_not_promoted() {
        let mut trials = thirty_successful_trials();
        trials[0].evidence_class = BenchmarkEvidenceClass::KinematicSimulation;
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert!(report.diagnostics.contains(&BenchmarkDiagnostic::EvidenceClassMismatch));
    }

    #[test]
    fn calibration_and_clock_references_are_mandatory() {
        let mut trials = thirty_successful_trials();
        trials[0].metrology_calibration_ref = None;
        trials[1].clock_synchronization_ref = "different-clock".to_string();
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert!(report.diagnostics.contains(&BenchmarkDiagnostic::MissingMetrologyCalibration));
        assert!(report.diagnostics.contains(&BenchmarkDiagnostic::ClockSynchronizationMismatch));
    }

    #[test]
    fn alignment_tolerance_failures_are_reported_not_hidden() {
        let mut trials = thirty_successful_trials();
        trials[0].measured_alignment_error_mm = Some(0.8);
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert_eq!(report.alignment_tolerance_failures, 1);
        assert_eq!(report.maximum_alignment_error_mm, Some(0.8));
    }

    #[test]
    fn unsafe_failed_attempt_is_retained_and_visible() {
        let mut trials = thirty_successful_trials();
        trials[0].status = TrialStatus::Failed;
        trials[0].safe_terminal_state = false;
        trials[0].operation_events.clear();
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert_eq!(report.unsafe_terminal_trials, 1);
        assert_eq!(report.failed_trials, 1);
    }

    #[test]
    fn event_input_order_does_not_change_summary() {
        let first = thirty_successful_trials();
        let mut second = first.clone();
        second.reverse();
        assert_eq!(
            summarize_remote_handling_benchmark(&manifest(), &first),
            summarize_remote_handling_benchmark(&manifest(), &second)
        );
    }

    #[test]
    fn clopper_pearson_all_successes_30_has_known_answer() {
        let lower = clopper_pearson_one_sided_lower(30, 30, 0.05);
        assert!((lower - 0.90497).abs() < 0.0001);
    }

    #[test]
    fn p90_order_statistic_bound_needs_about_thirty_successful_repeats() {
        assert_eq!(upper_order_statistic_index(30, 0.90, 0.95), Some(30));
        assert_eq!(upper_order_statistic_index(29, 0.90, 0.95), None);
    }

    #[test]
    fn failed_operation_cannot_appear_in_a_completed_trial() {
        let mut trials = thirty_successful_trials();
        trials[0].operation_events[0].outcome = OperationTrialOutcome::Failed;
        let report = summarize_remote_handling_benchmark(&manifest(), &trials);
        assert!(report.diagnostics.contains(&BenchmarkDiagnostic::FailedOperationInCompletedTrial));
    }
}
