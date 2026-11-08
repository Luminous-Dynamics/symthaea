//! Offline HCE epoch-contract validation and asynchronous worker-trace analysis.
//!
//! This crate does not run agents, schedule work, evaluate hidden data, sign
//! receipts, or authorize promotion. It validates synthetic HCE declarations
//! and analyzes already-recorded execution traces. A declared overlap audit is
//! not authenticated by this crate; an independent custodian must inspect the
//! underlying split membership and custody controls.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub mod operator_trace;
pub mod worker_simulation;
pub use operator_trace::{parse_and_validate_operator_trace, OperatorTrace, OperatorTraceReport};
pub use worker_simulation::{
    compare_worker_regimes, simulate_worker_regime, SimulationError, SimulationTask,
    WorkerRegimeComparisonReport, WorkerSimulationPlan,
};

const HCE_SCHEMA: &str = "sym-rsi-hce-epoch-v0";
const HCE_FIXTURE_STATUS: &str = "SYNTHETIC_HCE_FIXTURE_NOT_PREREGISTERED";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HceEpoch {
    pub schema_version: String,
    pub status: String,
    pub epoch_id: String,
    pub task_family_id: String,
    pub candidate_population_sha256: String,
    pub evaluator_artifact_sha256: String,
    pub tool_contract_sha256: String,
    pub search_budget: SearchBudget,
    pub splits: Vec<SplitSpec>,
    pub selection_policy: SelectionPolicy,
    pub overlap_audit: OverlapAudit,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchBudget {
    pub max_candidates: u64,
    pub max_search_evaluations: u64,
    pub max_wall_clock_hours: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SplitRole {
    Train,
    Search,
    Selection,
    Confirmation,
    Transfer,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitSpec {
    pub split_id: String,
    pub role: SplitRole,
    pub task_family_id: String,
    pub case_count: u64,
    pub membership_commitment_sha256: String,
    pub features_visible_to_agent: bool,
    pub labels_visible_to_agent: bool,
    pub score_visible_to_agent: bool,
    pub search_evaluator_access: bool,
    pub final_selection_access: bool,
    pub confirmation_evaluator_access: bool,
    pub transfer_evaluator_access: bool,
    pub fresh_family: bool,
    pub custodian_id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionPolicy {
    pub finalize_search_before_selection: bool,
    pub search_cannot_read_selection: bool,
    pub selection_score_not_returned_to_search: bool,
    pub confirmation_after_candidate_freeze: bool,
    pub transfer_after_candidate_freeze: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlapAudit {
    pub independent_custodian_id: String,
    pub exact_membership_comparison_completed: bool,
    pub zero_overlap_declared: bool,
    pub audited_split_count: usize,
    pub audit_artifact_sha256: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct HceValidationReport {
    pub valid: bool,
    pub split_count: usize,
    pub role_counts: [usize; 5],
    pub issues: Vec<String>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HceError {
    UnsupportedSchema,
    InvalidStatus,
    EmptyIdentifier(&'static str),
    InvalidDigest(&'static str),
    InvalidBudget,
    MissingOrDuplicateRole,
    DuplicateSplitId(String),
    DuplicateMembershipCommitment(String),
    InvalidSplitCount(String),
    InvalidCustody(String),
    InvalidSplitRolePolicy(String),
    WrongTaskFamily(String),
    InvalidSelectionPolicy,
    InvalidOverlapAudit,
}

impl std::fmt::Display for HceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HceError {}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl HceEpoch {
    pub fn parse(contents: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(contents)
    }

    pub fn validate(&self) -> Result<HceValidationReport, HceError> {
        if self.schema_version != HCE_SCHEMA {
            return Err(HceError::UnsupportedSchema);
        }
        if self.status != HCE_FIXTURE_STATUS {
            return Err(HceError::InvalidStatus);
        }
        for (name, value) in [
            ("epoch_id", self.epoch_id.as_str()),
            ("task_family_id", self.task_family_id.as_str()),
        ] {
            if !non_empty(value) {
                return Err(HceError::EmptyIdentifier(name));
            }
        }
        for (name, digest) in [
            ("candidate_population_sha256", self.candidate_population_sha256.as_str()),
            ("evaluator_artifact_sha256", self.evaluator_artifact_sha256.as_str()),
            ("tool_contract_sha256", self.tool_contract_sha256.as_str()),
        ] {
            if !is_sha256(digest) {
                return Err(HceError::InvalidDigest(name));
            }
        }
        if self.search_budget.max_candidates == 0
            || self.search_budget.max_search_evaluations == 0
            || !self.search_budget.max_wall_clock_hours.is_finite()
            || self.search_budget.max_wall_clock_hours <= 0.0
        {
            return Err(HceError::InvalidBudget);
        }

        let mut split_ids = BTreeSet::new();
        let mut commitments = BTreeSet::new();
        let mut custodians = BTreeSet::new();
        let mut roles = [0usize; 5];
        for split in &self.splits {
            if !non_empty(&split.split_id)
                || !non_empty(&split.task_family_id)
                || !non_empty(&split.custodian_id)
            {
                return Err(HceError::EmptyIdentifier("split_id_or_custodian"));
            }
            if split.case_count == 0 {
                return Err(HceError::InvalidSplitCount(split.split_id.clone()));
            }
            if !is_sha256(&split.membership_commitment_sha256) {
                return Err(HceError::InvalidDigest("membership_commitment_sha256"));
            }
            if !split_ids.insert(split.split_id.as_str()) {
                return Err(HceError::DuplicateSplitId(split.split_id.clone()));
            }
            if !commitments.insert(split.membership_commitment_sha256.as_str()) {
                return Err(HceError::DuplicateMembershipCommitment(
                    split.membership_commitment_sha256.clone(),
                ));
            }
            custodians.insert(split.custodian_id.as_str());

            let (role_index, expected_family, policy_valid) = match split.role {
                SplitRole::Train => (
                    0,
                    self.task_family_id.as_str(),
                    split.features_visible_to_agent
                        && split.labels_visible_to_agent
                        && !split.score_visible_to_agent
                        && !split.search_evaluator_access
                        && !split.final_selection_access
                        && !split.confirmation_evaluator_access
                        && !split.transfer_evaluator_access
                        && !split.fresh_family,
                ),
                SplitRole::Search => (
                    1,
                    self.task_family_id.as_str(),
                    !split.features_visible_to_agent
                        && !split.labels_visible_to_agent
                        && split.score_visible_to_agent
                        && split.search_evaluator_access
                        && !split.final_selection_access
                        && !split.confirmation_evaluator_access
                        && !split.transfer_evaluator_access
                        && !split.fresh_family,
                ),
                SplitRole::Selection => (
                    2,
                    self.task_family_id.as_str(),
                    !split.features_visible_to_agent
                        && !split.labels_visible_to_agent
                        && !split.score_visible_to_agent
                        && !split.search_evaluator_access
                        && split.final_selection_access
                        && !split.confirmation_evaluator_access
                        && !split.transfer_evaluator_access
                        && !split.fresh_family,
                ),
                SplitRole::Confirmation => (
                    3,
                    self.task_family_id.as_str(),
                    !split.features_visible_to_agent
                        && !split.labels_visible_to_agent
                        && !split.score_visible_to_agent
                        && !split.search_evaluator_access
                        && !split.final_selection_access
                        && split.confirmation_evaluator_access
                        && !split.transfer_evaluator_access
                        && !split.fresh_family,
                ),
                SplitRole::Transfer => (
                    4,
                    split.task_family_id.as_str(),
                    !split.features_visible_to_agent
                        && !split.labels_visible_to_agent
                        && !split.score_visible_to_agent
                        && !split.search_evaluator_access
                        && !split.final_selection_access
                        && !split.confirmation_evaluator_access
                        && split.transfer_evaluator_access
                        && split.fresh_family,
                ),
            };
            roles[role_index] += 1;
            if !policy_valid {
                return Err(HceError::InvalidSplitRolePolicy(split.split_id.clone()));
            }
            if split.task_family_id != expected_family
                || (role_index == 4 && split.task_family_id == self.task_family_id)
                || (role_index != 4 && split.task_family_id != self.task_family_id)
            {
                return Err(HceError::WrongTaskFamily(split.split_id.clone()));
            }
        }

        if self.splits.len() != 5 || roles != [1, 1, 1, 1, 1] {
            return Err(HceError::MissingOrDuplicateRole);
        }
        if !self.selection_policy.finalize_search_before_selection
            || !self.selection_policy.search_cannot_read_selection
            || !self.selection_policy.selection_score_not_returned_to_search
            || !self.selection_policy.confirmation_after_candidate_freeze
            || !self.selection_policy.transfer_after_candidate_freeze
        {
            return Err(HceError::InvalidSelectionPolicy);
        }
        if !non_empty(&self.overlap_audit.independent_custodian_id)
            || custodians.iter().any(|custodian| *custodian == self.overlap_audit.independent_custodian_id.as_str())
            || !self.overlap_audit.exact_membership_comparison_completed
            || !self.overlap_audit.zero_overlap_declared
            || self.overlap_audit.audited_split_count != self.splits.len()
            || !is_sha256(&self.overlap_audit.audit_artifact_sha256)
        {
            return Err(HceError::InvalidOverlapAudit);
        }

        Ok(HceValidationReport {
            valid: true,
            split_count: self.splits.len(),
            role_counts: roles,
            issues: Vec::new(),
            limitations: vec![
                "Membership digests and a declared overlap audit do not prove set disjointness; an independent custodian must inspect the underlying membership manifests.".to_owned(),
                "This validator checks declared visibility policy, not runtime access controls, mounts, credentials, egress or evaluator isolation.".to_owned(),
                "A valid synthetic HCE contract does not establish model capability or research improvement.".to_owned(),
                "Search, selection, confirmation and transfer scores must remain separate evidence and promotion remains owned by existing authorities.".to_owned(),
            ],
        })
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionRegime {
    SynchronousBarrier,
    AsynchronousSteadyState,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptDisposition {
    Valid,
    Invalid,
    Failed,
    Cancelled,
    TimedOut,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerTrace {
    pub trace_id: String,
    pub regime: ExecutionRegime,
    pub worker_count: u32,
    pub wall_clock_budget_ms: u64,
    pub attempts: Vec<WorkerAttempt>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerAttempt {
    pub attempt_id: String,
    pub candidate_id: String,
    pub worker_id: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    pub disposition: AttemptDisposition,
    pub resource_cost_units: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WorkerTraceReport {
    pub trace_id: String,
    pub regime: ExecutionRegime,
    pub worker_count: u32,
    pub wall_clock_budget_ms: u64,
    pub observed_start_ms: u64,
    pub observed_end_ms: u64,
    pub observed_elapsed_ms: u64,
    pub observed_worker_idle_ms: u64,
    pub observed_worker_utilization: f64,
    pub valid_candidates_per_observed_hour: f64,
    pub total_attempts: usize,
    pub unique_valid_candidates: usize,
    pub valid_attempts: usize,
    pub invalid_attempts: usize,
    pub failed_attempts: usize,
    pub cancelled_attempts: usize,
    pub timed_out_attempts: usize,
    pub total_worker_busy_ms: u64,
    pub total_worker_idle_ms: u64,
    pub worker_utilization: f64,
    pub valid_candidates_per_hour: f64,
    pub p50_attempt_duration_ms: u64,
    pub p95_attempt_duration_ms: u64,
    pub p95_to_p50_straggler_ratio: f64,
    pub total_resource_cost_units: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    EmptyTraceId,
    InvalidWorkerCount,
    InvalidWallClockBudget,
    EmptyAttempts,
    EmptyAttemptId,
    EmptyCandidateId,
    DuplicateAttemptId(String),
    WorkerOutOfRange(u32),
    InvalidAttemptInterval(String),
    AttemptExceedsWallClockBudget(String),
    WorkerOverlap { worker_id: u32, first: String, second: String },
    InvalidResourceCost(String),
    WorkerCapacityOverflow,
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for TraceError {}

fn percentile_nearest_rank(sorted: &[u64], percentile: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (percentile * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

pub fn analyze_worker_trace(trace: &WorkerTrace) -> Result<WorkerTraceReport, TraceError> {
    if !non_empty(&trace.trace_id) {
        return Err(TraceError::EmptyTraceId);
    }
    if trace.worker_count == 0 {
        return Err(TraceError::InvalidWorkerCount);
    }
    if trace.wall_clock_budget_ms == 0 {
        return Err(TraceError::InvalidWallClockBudget);
    }
    if trace.attempts.is_empty() {
        return Err(TraceError::EmptyAttempts);
    }

    let mut attempt_ids = BTreeSet::new();
    let mut per_worker: BTreeMap<u32, Vec<&WorkerAttempt>> = BTreeMap::new();
    let mut durations = Vec::with_capacity(trace.attempts.len());
    let mut total_busy = 0u64;
    let mut observed_start = u64::MAX;
    let mut observed_end = 0u64;
    let mut total_cost = 0.0f64;
    let mut valid_candidates = BTreeSet::new();
    let mut valid_attempts = 0usize;
    let mut invalid_attempts = 0usize;
    let mut failed_attempts = 0usize;
    let mut cancelled_attempts = 0usize;
    let mut timed_out_attempts = 0usize;

    for attempt in &trace.attempts {
        if !non_empty(&attempt.attempt_id) {
            return Err(TraceError::EmptyAttemptId);
        }
        if !non_empty(&attempt.candidate_id) {
            return Err(TraceError::EmptyCandidateId);
        }
        if !attempt_ids.insert(attempt.attempt_id.as_str()) {
            return Err(TraceError::DuplicateAttemptId(attempt.attempt_id.clone()));
        }
        if attempt.worker_id >= trace.worker_count {
            return Err(TraceError::WorkerOutOfRange(attempt.worker_id));
        }
        if attempt.start_ms >= attempt.end_ms {
            return Err(TraceError::InvalidAttemptInterval(attempt.attempt_id.clone()));
        }
        if attempt.end_ms > trace.wall_clock_budget_ms {
            return Err(TraceError::AttemptExceedsWallClockBudget(attempt.attempt_id.clone()));
        }
        if !attempt.resource_cost_units.is_finite() || attempt.resource_cost_units < 0.0 {
            return Err(TraceError::InvalidResourceCost(attempt.attempt_id.clone()));
        }

        let duration = attempt.end_ms - attempt.start_ms;
        observed_start = observed_start.min(attempt.start_ms);
        observed_end = observed_end.max(attempt.end_ms);
        total_busy = total_busy.checked_add(duration).ok_or(TraceError::WorkerCapacityOverflow)?;
        total_cost += attempt.resource_cost_units;
        if !total_cost.is_finite() {
            return Err(TraceError::InvalidResourceCost(attempt.attempt_id.clone()));
        }
        durations.push(duration);
        per_worker.entry(attempt.worker_id).or_default().push(attempt);

        match attempt.disposition {
            AttemptDisposition::Valid => {
                valid_attempts += 1;
                valid_candidates.insert(attempt.candidate_id.as_str());
            }
            AttemptDisposition::Invalid => invalid_attempts += 1,
            AttemptDisposition::Failed => failed_attempts += 1,
            AttemptDisposition::Cancelled => cancelled_attempts += 1,
            AttemptDisposition::TimedOut => timed_out_attempts += 1,
        }
    }

    for (worker_id, attempts) in &mut per_worker {
        attempts.sort_by_key(|attempt| (attempt.start_ms, attempt.end_ms));
        for pair in attempts.windows(2) {
            if pair[1].start_ms < pair[0].end_ms {
                return Err(TraceError::WorkerOverlap {
                    worker_id: *worker_id,
                    first: pair[0].attempt_id.clone(),
                    second: pair[1].attempt_id.clone(),
                });
            }
        }
    }

    let capacity = (trace.worker_count as u64)
        .checked_mul(trace.wall_clock_budget_ms)
        .ok_or(TraceError::WorkerCapacityOverflow)?;
    if total_busy > capacity {
        return Err(TraceError::WorkerCapacityOverflow);
    }
    durations.sort_unstable();
    let p50 = percentile_nearest_rank(&durations, 0.50);
    let p95 = percentile_nearest_rank(&durations, 0.95);
    let utilization = total_busy as f64 / capacity as f64;
    let budget_hours = trace.wall_clock_budget_ms as f64 / 3_600_000.0;
    let observed_elapsed = observed_end.saturating_sub(observed_start);
    if observed_elapsed == 0 {
        return Err(TraceError::WorkerCapacityOverflow);
    }
    let observed_capacity = (trace.worker_count as u64)
        .checked_mul(observed_elapsed)
        .ok_or(TraceError::WorkerCapacityOverflow)?;
    if total_busy > observed_capacity {
        return Err(TraceError::WorkerCapacityOverflow);
    }
    let observed_idle = observed_capacity - total_busy;
    let observed_utilization = total_busy as f64 / observed_capacity as f64;
    let observed_hours = observed_elapsed as f64 / 3_600_000.0;
    let valid_per_observed_hour = valid_candidates.len() as f64 / observed_hours;
    let p95_to_p50 = if p50 == 0 { 0.0 } else { p95 as f64 / p50 as f64 };

    Ok(WorkerTraceReport {
        trace_id: trace.trace_id.clone(),
        regime: trace.regime,
        worker_count: trace.worker_count,
        wall_clock_budget_ms: trace.wall_clock_budget_ms,
        observed_start_ms: observed_start,
        observed_end_ms: observed_end,
        observed_elapsed_ms: observed_elapsed,
        observed_worker_idle_ms: observed_idle,
        observed_worker_utilization: observed_utilization,
        valid_candidates_per_observed_hour: valid_per_observed_hour,
        total_attempts: trace.attempts.len(),
        unique_valid_candidates: valid_candidates.len(),
        valid_attempts,
        invalid_attempts,
        failed_attempts,
        cancelled_attempts,
        timed_out_attempts,
        total_worker_busy_ms: total_busy,
        total_worker_idle_ms: capacity - total_busy,
        worker_utilization: utilization,
        valid_candidates_per_hour: valid_candidates.len() as f64 / budget_hours,
        p50_attempt_duration_ms: p50,
        p95_attempt_duration_ms: p95,
        p95_to_p50_straggler_ratio: p95_to_p50,
        total_resource_cost_units: total_cost,
    })
}

pub fn parse_and_analyze_worker_trace(contents: &str) -> Result<WorkerTraceReport, Box<dyn std::error::Error>> {
    let trace: WorkerTrace = serde_json::from_str(contents)?;
    Ok(analyze_worker_trace(&trace)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const HCE_FIXTURE: &str = include_str!("../fixtures/hce-epoch-v0.json");
    const WORKER_FIXTURE: &str = include_str!("../fixtures/worker-trace-v0.json");

    fn valid_hce() -> HceEpoch {
        HceEpoch::parse(HCE_FIXTURE).expect("HCE fixture parses")
    }

    fn valid_trace() -> WorkerTrace {
        serde_json::from_str(WORKER_FIXTURE).expect("worker trace fixture parses")
    }

    #[test]
    fn accepts_five_role_hce_epoch_contract() {
        let report = valid_hce().validate().expect("HCE contract is valid");
        assert_eq!(report.split_count, 5);
        assert_eq!(report.role_counts, [1, 1, 1, 1, 1]);
    }

    #[test]
    fn rejects_search_labels_exposed_to_agent() {
        let mut value: Value = serde_json::from_str(HCE_FIXTURE).expect("HCE JSON parses");
        value["splits"][1]["labels_visible_to_agent"] = Value::Bool(true);
        let bad: HceEpoch = serde_json::from_value(value).expect("mutated HCE parses");
        assert_eq!(
            bad.validate().expect_err("search labels must remain hidden"),
            HceError::InvalidSplitRolePolicy("search-v0".to_owned())
        );
    }

    #[test]
    fn rejects_search_features_exposed_to_agent() {
        let mut value: Value = serde_json::from_str(HCE_FIXTURE).expect("HCE JSON parses");
        value["splits"][1]["features_visible_to_agent"] = Value::Bool(true);
        let bad: HceEpoch = serde_json::from_value(value).expect("mutated HCE parses");
        assert_eq!(
            bad.validate().expect_err("search inputs must remain hidden"),
            HceError::InvalidSplitRolePolicy("search-v0".to_owned())
        );
    }

    #[test]
    fn rejects_selection_feedback_to_search() {
        let mut value: Value = serde_json::from_str(HCE_FIXTURE).expect("HCE JSON parses");
        value["selection_policy"]["selection_score_not_returned_to_search"] = Value::Bool(false);
        let bad: HceEpoch = serde_json::from_value(value).expect("mutated HCE parses");
        assert_eq!(bad.validate().expect_err("selection must not feed search"), HceError::InvalidSelectionPolicy);
    }

    #[test]
    fn rejects_transfer_split_from_the_original_family() {
        let mut value: Value = serde_json::from_str(HCE_FIXTURE).expect("HCE JSON parses");
        value["splits"][4]["task_family_id"] = Value::String("synthetic-continual-learning-family".to_owned());
        let bad: HceEpoch = serde_json::from_value(value).expect("mutated HCE parses");
        assert_eq!(
            bad.validate().expect_err("transfer must be a fresh family"),
            HceError::WrongTaskFamily("transfer-v0".to_owned())
        );
    }

    #[test]
    fn rejects_missing_independent_overlap_audit() {
        let mut value: Value = serde_json::from_str(HCE_FIXTURE).expect("HCE JSON parses");
        value["overlap_audit"]["independent_custodian_id"] = Value::String("custodian-search".to_owned());
        let bad: HceEpoch = serde_json::from_value(value).expect("mutated HCE parses");
        assert_eq!(bad.validate().expect_err("overlap audit must be independent"), HceError::InvalidOverlapAudit);
    }

    #[test]
    fn analyzes_worker_trace_without_claiming_scheduler_execution() {
        let trace = valid_trace();
        let report = analyze_worker_trace(&trace).expect("synthetic trace is valid");
        assert_eq!(report.total_attempts, 4);
        assert_eq!(report.unique_valid_candidates, 3);
        assert_eq!(report.failed_attempts, 1);
        assert_eq!(report.total_worker_busy_ms, 1700);
        assert_eq!(report.total_worker_idle_ms, 300);
        assert_eq!(report.observed_elapsed_ms, 1000);
        assert_eq!(report.observed_worker_idle_ms, 300);
        assert!((report.observed_worker_utilization - 0.85).abs() < 1e-12);
        assert!((report.valid_candidates_per_observed_hour - 10800.0).abs() < 1e-9);
        assert!((report.worker_utilization - 0.85).abs() < 1e-12);
        assert_eq!(report.p50_attempt_duration_ms, 400);
        assert_eq!(report.p95_attempt_duration_ms, 500);
        assert!((report.p95_to_p50_straggler_ratio - 1.25).abs() < 1e-12);
        assert!((report.total_resource_cost_units - 3.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_overlapping_tasks_on_one_worker() {
        let mut trace = valid_trace();
        trace.attempts[1].worker_id = 0;
        assert!(matches!(
            analyze_worker_trace(&trace),
            Err(TraceError::WorkerOverlap { worker_id: 0, .. })
        ));
    }

    #[test]
    fn rejects_attempt_outside_the_wall_clock_budget() {
        let mut trace = valid_trace();
        trace.attempts[0].end_ms = trace.wall_clock_budget_ms + 1;
        assert_eq!(
            analyze_worker_trace(&trace).expect_err("attempt exceeds trace budget"),
            TraceError::AttemptExceedsWallClockBudget("attempt-001".to_owned())
        );
    }

    #[test]
    fn repeated_valid_candidate_counts_once_in_unique_throughput() {
        let mut trace = valid_trace();
        trace.attempts[3].disposition = AttemptDisposition::Valid;
        trace.attempts[3].candidate_id = "candidate-A".to_owned();
        let report = analyze_worker_trace(&trace).expect("duplicate candidate attempts are retained");
        assert_eq!(report.valid_attempts, 4);
        assert_eq!(report.unique_valid_candidates, 3);
    }

    #[test]
    fn rejects_duplicate_attempt_ids() {
        let mut trace = valid_trace();
        trace.attempts[1].attempt_id = trace.attempts[0].attempt_id.clone();
        assert_eq!(
            analyze_worker_trace(&trace).expect_err("attempt IDs are unique"),
            TraceError::DuplicateAttemptId("attempt-001".to_owned())
        );
    }

    #[test]
    fn rejects_non_finite_resource_cost() {
        let mut trace = valid_trace();
        trace.attempts[0].resource_cost_units = f64::NAN;
        assert_eq!(
            analyze_worker_trace(&trace).expect_err("cost must be finite"),
            TraceError::InvalidResourceCost("attempt-001".to_owned())
        );
    }
}
