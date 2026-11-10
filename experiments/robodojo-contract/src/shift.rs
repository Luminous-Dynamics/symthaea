//! Shift-stratified analysis for paired manipulation-policy evaluations.
//!
//! The manifest defines the intended perturbation identity; this module does not
//! assert that a simulator or physical setup actually applied that perturbation.
use std::collections::{BTreeMap, BTreeSet};
use crate::{compare_paired_trials, TrialRecord};

pub const SHIFT_PLAN_HEADER: &str =
    "task_id\ttrial_id\tcondition_id\tfamily\tseverity\tseed\tperturbation_spec_sha256";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShiftFamily {
    Nominal, Appearance, Geometry, Lighting, Occlusion, Dynamics,
    SensorLatency, InstructionSemantics, ObjectState, ToolFailure,
}
impl ShiftFamily {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "nominal" => Self::Nominal,
            "appearance" => Self::Appearance,
            "geometry" => Self::Geometry,
            "lighting" => Self::Lighting,
            "occlusion" => Self::Occlusion,
            "dynamics" => Self::Dynamics,
            "sensor_latency" => Self::SensorLatency,
            "instruction_semantics" => Self::InstructionSemantics,
            "object_state" => Self::ObjectState,
            "tool_failure" => Self::ToolFailure,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::Appearance => "appearance",
            Self::Geometry => "geometry",
            Self::Lighting => "lighting",
            Self::Occlusion => "occlusion",
            Self::Dynamics => "dynamics",
            Self::SensorLatency => "sensor_latency",
            Self::InstructionSemantics => "instruction_semantics",
            Self::ObjectState => "object_state",
            Self::ToolFailure => "tool_failure",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShiftSeverity { Clean, Mild, Moderate, Severe }
impl ShiftSeverity {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "clean" => Self::Clean,
            "mild" => Self::Mild,
            "moderate" => Self::Moderate,
            "severe" => Self::Severe,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Mild => "mild",
            Self::Moderate => "moderate",
            Self::Severe => "severe",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShiftAssignment {
    pub task_id: String,
    pub trial_id: String,
    pub condition_id: String,
    pub family: ShiftFamily,
    pub severity: ShiftSeverity,
    pub seed: u64,
    pub perturbation_spec_sha256: String,
}
impl ShiftAssignment {
    pub fn validate(&self) -> Result<(), ShiftAuditError> {
        for value in [&self.task_id, &self.trial_id, &self.condition_id] {
            if value.trim().is_empty() || value.chars().any(|c| matches!(c, '\t' | '\r' | '\n')) {
                return Err(ShiftAuditError::InvalidAssignment);
            }
        }
        let digest = self.perturbation_spec_sha256.as_bytes();
        if digest.len() != 64 || !digest.iter().all(|b| b.is_ascii_hexdigit()) {
            return Err(ShiftAuditError::InvalidSpecDigest);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShiftAuditError {
    EmptyPlan, InvalidAssignment, InvalidSpecDigest, DuplicateAssignment,
    TrialPlanMismatch { missing: usize, unmatched: usize },
    InvalidPairedTrials,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShiftStratumSummary {
    pub family: ShiftFamily,
    pub severity: ShiftSeverity,
    pub perturbation_spec_sha256: String,
    pub condition_count: usize,
    pub pairs: usize,
    pub scored_pairs: usize,
    pub mean_progress_delta: Option<f64>,
    pub progress_delta_ci95: Option<(f64, f64)>,
    pub completion_rate_delta: f64,
    pub valid_run_rate_delta: f64,
    pub mean_elapsed_seconds_delta: f64,
    pub mean_cost_usd_delta: f64,
    pub safety_event_count_delta: i64,
    pub timeout_count_delta: i64,
    pub thermal_stop_count_delta: i64,
    pub server_error_count_delta: i64,
    pub invalid_action_count_delta: i64,
    pub safety_stop_count_delta: i64,
    pub other_failure_count_delta: i64,
    /// Complete 3x3 paired outcome matrix. Rows are baseline S/F/U; columns
    /// are candidate S/F/U. Invalid runs are unresolved, not silently failures.
    pub success_to_success: usize,
    pub success_to_failure: usize,
    pub success_to_unresolved: usize,
    pub failure_to_success: usize,
    pub failure_to_failure: usize,
    pub failure_to_unresolved: usize,
    pub unresolved_to_success: usize,
    pub unresolved_to_failure: usize,
    pub unresolved_to_unresolved: usize,
    pub resolved_outcome_pairs: usize,
    pub changed_resolved_outcome_pairs: usize,
    pub unresolved_outcome_pairs: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairedOutcome {
    Success,
    Failure,
    Unresolved,
}

#[derive(Debug, Clone, Copy, Default)]
struct OutcomeTransitionMatrix {
    success_to_success: usize,
    success_to_failure: usize,
    success_to_unresolved: usize,
    failure_to_success: usize,
    failure_to_failure: usize,
    failure_to_unresolved: usize,
    unresolved_to_success: usize,
    unresolved_to_failure: usize,
    unresolved_to_unresolved: usize,
}

impl OutcomeTransitionMatrix {
    fn observe(&mut self, baseline: &TrialRecord, candidate: &TrialRecord) {
        let left = outcome_of(baseline);
        let right = outcome_of(candidate);
        match (left, right) {
            (PairedOutcome::Success, PairedOutcome::Success) => self.success_to_success += 1,
            (PairedOutcome::Success, PairedOutcome::Failure) => self.success_to_failure += 1,
            (PairedOutcome::Success, PairedOutcome::Unresolved) => self.success_to_unresolved += 1,
            (PairedOutcome::Failure, PairedOutcome::Success) => self.failure_to_success += 1,
            (PairedOutcome::Failure, PairedOutcome::Failure) => self.failure_to_failure += 1,
            (PairedOutcome::Failure, PairedOutcome::Unresolved) => self.failure_to_unresolved += 1,
            (PairedOutcome::Unresolved, PairedOutcome::Success) => self.unresolved_to_success += 1,
            (PairedOutcome::Unresolved, PairedOutcome::Failure) => self.unresolved_to_failure += 1,
            (PairedOutcome::Unresolved, PairedOutcome::Unresolved) => self.unresolved_to_unresolved += 1,
        }
    }

    fn resolved_pairs(self) -> usize {
        self.success_to_success
            + self.success_to_failure
            + self.failure_to_success
            + self.failure_to_failure
    }

    fn changed_resolved_pairs(self) -> usize {
        self.success_to_failure + self.failure_to_success
    }

    fn unresolved_pairs(self) -> usize {
        self.success_to_unresolved
            + self.failure_to_unresolved
            + self.unresolved_to_success
            + self.unresolved_to_failure
            + self.unresolved_to_unresolved
    }
}

fn outcome_of(record: &TrialRecord) -> PairedOutcome {
    if !record.valid_run {
        PairedOutcome::Unresolved
    } else if record.completed {
        PairedOutcome::Success
    } else {
        PairedOutcome::Failure
    }
}

/// Manifest task/trial keys must match paired trial keys exactly. Each stratum
/// remains separate by exact configuration digest so one perturbation cannot
/// hide a different configuration's regression.
pub fn audit_shift_robustness(
    baseline: &[TrialRecord],
    candidate: &[TrialRecord],
    assignments: &[ShiftAssignment],
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<Vec<ShiftStratumSummary>, ShiftAuditError> {
    if assignments.is_empty() { return Err(ShiftAuditError::EmptyPlan); }
    compare_paired_trials(baseline, candidate, 0, seed)
        .map_err(|_| ShiftAuditError::InvalidPairedTrials)?;

    let base: BTreeMap<(String, String), &TrialRecord> = baseline.iter()
        .map(|r| ((r.task_id.clone(), r.trial_id.clone()), r)).collect();
    let cand: BTreeMap<(String, String), &TrialRecord> = candidate.iter()
        .map(|r| ((r.task_id.clone(), r.trial_id.clone()), r)).collect();
    let mut plan: BTreeMap<(String, String), &ShiftAssignment> = BTreeMap::new();
    for assignment in assignments {
        assignment.validate()?;
        let key = (assignment.task_id.clone(), assignment.trial_id.clone());
        if plan.insert(key, assignment).is_some() {
            return Err(ShiftAuditError::DuplicateAssignment);
        }
    }
    let trial_keys: BTreeSet<_> = base.keys().cloned().collect();
    let plan_keys: BTreeSet<_> = plan.keys().cloned().collect();
    if trial_keys != plan_keys {
        return Err(ShiftAuditError::TrialPlanMismatch {
            missing: trial_keys.difference(&plan_keys).count(),
            unmatched: plan_keys.difference(&trial_keys).count(),
        });
    }

    let mut groups: BTreeMap<
        (ShiftFamily, ShiftSeverity, String),
        (Vec<TrialRecord>, Vec<TrialRecord>, BTreeSet<String>),
    > = BTreeMap::new();
    for (key, assignment) in plan {
        let b = base.get(&key).expect("manifest keys checked");
        let c = cand.get(&key).expect("paired keys checked");
        let group_key = (assignment.family, assignment.severity, assignment.perturbation_spec_sha256.clone());
        let group = groups.entry(group_key).or_insert_with(|| (Vec::new(), Vec::new(), BTreeSet::new()));
        group.0.push((**b).clone());
        group.1.push((**c).clone());
        group.2.insert(assignment.condition_id.clone());
    }

    let mut summaries = Vec::with_capacity(groups.len());
    for ((family, severity, digest), (b, c, conditions)) in groups {
        let comparison = compare_paired_trials(
            &b, &c, bootstrap_replicates, derived_seed(seed, family, severity, &digest),
        ).map_err(|_| ShiftAuditError::InvalidPairedTrials)?;
        let mut outcome_transitions = OutcomeTransitionMatrix::default();
        for (baseline_row, candidate_row) in b.iter().zip(&c) {
            outcome_transitions.observe(baseline_row, candidate_row);
        }
        let mut scored_by_task: BTreeMap<String, usize> =
            b.iter().map(|row| (row.task_id.clone(), 0)).collect();
        for (base_row, candidate_row) in b.iter().zip(&c) {
            if base_row.progress.is_some() && candidate_row.progress.is_some() {
                *scored_by_task.entry(base_row.task_id.clone()).or_default() += 1;
            }
        }
        // A single observation per task gives a degenerate within-task bootstrap;
        // suppress that interval rather than presenting false precision as certainty.
        let interval_supported = !scored_by_task.is_empty()
            && scored_by_task.values().all(|count| *count >= 2)
            && bootstrap_replicates >= 100;
        summaries.push(ShiftStratumSummary {
            family, severity, perturbation_spec_sha256: digest,
            condition_count: conditions.len(),
            pairs: comparison.total_pairs,
            scored_pairs: comparison.scored_pairs,
            mean_progress_delta: comparison.mean_progress_delta,
            progress_delta_ci95: if interval_supported { comparison.progress_delta_ci95 } else { None },
            completion_rate_delta: comparison.completion_rate_delta,
            valid_run_rate_delta: comparison.valid_run_rate_delta,
            mean_elapsed_seconds_delta: comparison.mean_elapsed_seconds_delta,
            mean_cost_usd_delta: comparison.mean_cost_usd_delta,
            safety_event_count_delta: comparison.safety_event_count_delta,
            timeout_count_delta: comparison.timeout_count_delta,
            thermal_stop_count_delta: comparison.thermal_stop_count_delta,
            server_error_count_delta: comparison.server_error_count_delta,
            invalid_action_count_delta: comparison.invalid_action_count_delta,
            safety_stop_count_delta: comparison.safety_stop_count_delta,
            other_failure_count_delta: comparison.other_failure_count_delta,
            success_to_success: outcome_transitions.success_to_success,
            success_to_failure: outcome_transitions.success_to_failure,
            success_to_unresolved: outcome_transitions.success_to_unresolved,
            failure_to_success: outcome_transitions.failure_to_success,
            failure_to_failure: outcome_transitions.failure_to_failure,
            failure_to_unresolved: outcome_transitions.failure_to_unresolved,
            unresolved_to_success: outcome_transitions.unresolved_to_success,
            unresolved_to_failure: outcome_transitions.unresolved_to_failure,
            unresolved_to_unresolved: outcome_transitions.unresolved_to_unresolved,
            resolved_outcome_pairs: outcome_transitions.resolved_pairs(),
            changed_resolved_outcome_pairs: outcome_transitions.changed_resolved_pairs(),
            unresolved_outcome_pairs: outcome_transitions.unresolved_pairs(),
        });
    }
    Ok(summaries)
}

fn derived_seed(seed: u64, family: ShiftFamily, severity: ShiftSeverity, digest: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ seed;
    for byte in format!("{}:{}:{}", family.as_str(), severity.as_str(), digest).bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record(task: &str, variant: &str, trial: &str, progress: f64) -> TrialRecord {
        TrialRecord {
            task_id: task.to_owned(), variant_id: variant.to_owned(), trial_id: trial.to_owned(),
            progress: Some(progress), completed: progress >= 1.0, valid_run: true, timed_out: false,
            failure_class: crate::FailureClass::None, elapsed_seconds: 1.0, model_calls: 1,
            action_steps: 1, estimated_cost_usd: 0.01, safety_events: 0,
        }
    }
    fn assignment(task: &str, trial: &str) -> ShiftAssignment {
        ShiftAssignment {
            task_id: task.to_owned(), trial_id: trial.to_owned(),
            condition_id: "geometry-offset-a".to_owned(), family: ShiftFamily::Geometry,
            severity: ShiftSeverity::Mild, seed: 42, perturbation_spec_sha256: "a".repeat(64),
        }
    }
    fn rows() -> (Vec<TrialRecord>, Vec<TrialRecord>, Vec<ShiftAssignment>) {
        let (mut b, mut c, mut p) = (Vec::new(), Vec::new(), Vec::new());
        for task in ["stack_bowls", "store_in_safe"] {
            for trial in ["seed-01", "seed-02"] {
                b.push(record(task, "baseline", trial, 0.25));
                c.push(record(task, "candidate", trial, 0.50));
                p.push(assignment(task, trial));
            }
        }
        (b, c, p)
    }
    #[test]
    fn computes_matched_deltas_inside_shift_stratum() {
        let (b, c, p) = rows();
        let s = audit_shift_robustness(&b, &c, &p, 500, 7).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].pairs, 4);
        assert_eq!(s[0].scored_pairs, 4);
        assert_eq!(s[0].condition_count, 1);
        assert!((s[0].mean_progress_delta.unwrap() - 0.25).abs() < 1e-12);
    }
    #[test]
    fn exposes_bidirectional_outcome_flips_when_net_success_delta_is_zero() {
        let (mut baseline, mut candidate, plan) = rows();
        baseline[0].progress = Some(1.0);
        baseline[0].completed = true;
        candidate[0].progress = Some(0.25);
        candidate[0].completed = false;

        baseline[1].progress = Some(0.25);
        baseline[1].completed = false;
        candidate[1].progress = Some(1.0);
        candidate[1].completed = true;

        let summary = audit_shift_robustness(&baseline, &candidate, &plan, 0, 7).unwrap();
        assert_eq!(summary.len(), 1);
        let row = &summary[0];

        // One clean success is lost and one clean failure is recovered:
        // aggregate completion-rate delta is zero, while both episodes flipped.
        assert_eq!(row.completion_rate_delta, 0.0);
        assert_eq!(row.success_to_failure, 1);
        assert_eq!(row.failure_to_success, 1);
        assert_eq!(row.changed_resolved_outcome_pairs, 2);
        assert_eq!(row.resolved_outcome_pairs, 4);
        assert_eq!(row.unresolved_outcome_pairs, 0);
    }

    #[test]
    fn reports_unresolved_outcome_pairs_separately_from_task_failures() {
        let (mut baseline, mut candidate, plan) = rows();
        baseline[0].valid_run = false;
        baseline[0].progress = None;
        baseline[0].completed = false;
        baseline[0].failure_class = crate::FailureClass::ServerError;
        candidate[0].progress = Some(1.0);
        candidate[0].completed = true;

        let summary = audit_shift_robustness(&baseline, &candidate, &plan, 0, 7).unwrap();
        let row = &summary[0];
        assert_eq!(row.unresolved_to_success, 1);
        assert_eq!(row.unresolved_outcome_pairs, 1);
        assert_eq!(row.resolved_outcome_pairs, 3);
        assert_eq!(row.changed_resolved_outcome_pairs, 0);
    }

    #[test]
    fn rejects_missing_manifest_rows() {
        let (b, c, mut p) = rows();
        p.pop();
        assert!(matches!(audit_shift_robustness(&b, &c, &p, 0, 7),
            Err(ShiftAuditError::TrialPlanMismatch { missing: 1, unmatched: 0 })));
    }
    #[test]
    fn rejects_duplicate_manifest_keys() {
        let (b, c, mut p) = rows();
        p.push(p[0].clone());
        assert_eq!(audit_shift_robustness(&b, &c, &p, 0, 7), Err(ShiftAuditError::DuplicateAssignment));
    }
    #[test]
    fn suppresses_degenerate_interval_when_task_has_only_one_scored_pair() {
        let (mut b, mut c, mut p) = rows();
        b.retain(|r| r.trial_id == "seed-01");
        c.retain(|r| r.trial_id == "seed-01");
        p.retain(|r| r.trial_id == "seed-01");
        let s = audit_shift_robustness(&b, &c, &p, 500, 7).unwrap();
        assert_eq!(s[0].pairs, 2);
        assert!(s[0].progress_delta_ci95.is_none());
    }

    #[test]
    fn rejects_bad_spec_digest() {
        let (b, c, mut p) = rows();
        p[0].perturbation_spec_sha256 = "bad".to_owned();
        assert_eq!(audit_shift_robustness(&b, &c, &p, 0, 7), Err(ShiftAuditError::InvalidSpecDigest));
    }
}
