// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference contract for transactional, substep-boundary temporal continuation.
//!
//! This is not a production solver or scheduler. It specifies how a committed
//! temporal state may be yielded and resumed without double-applying time,
//! reusing a revised input trace, or consuming a stale checkpoint generation.
//!
//! The recurrent-state payload itself belongs to the snapshot/restore owner
//! (#1269). This token binds that payload by digest/reference; it does not
//! introduce a second recurrent-state serialization format.
//!
//! From the repository root:
//!   cargo fmt --check
//!   cargo test --locked -p symthaea-silicon --example temporal_resume_contract
//!   cargo run --locked -p symthaea-silicon --example temporal_resume_contract

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const SCHEMA: &str = "symthaea.temporal-solver-continuation";
const SCHEMA_VERSION: &str = "0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SolverMethod {
    Euler,
    RungeKutta4,
    ClosedForm,
    IterativeClosedForm,
    InterpolatedAdaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SafePoint {
    /// State, logical time, work counters and error budget were committed
    /// together after one complete solver substep. No stage is in flight.
    AfterCommittedSubstep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ContinuationState {
    ReadyToResume,
    YieldedAtSafePoint,
    Completed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ErrorBudgetEvidence {
    AnalyticBound,
    EmpiricalEstimate,
    Unspecified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SolverContinuation {
    schema: String,
    schema_version: String,
    continuation_id: String,
    checkpoint_generation: u64,
    model_id: String,
    configuration_id: String,
    source_revision: String,
    solver_profile_id: String,
    solver_method: SolverMethod,
    input_trace_sha256: String,
    interpolation_policy_id: String,
    committed_logical_time_ns: u64,
    target_logical_time_ns: u64,
    max_substeps: u64,
    completed_substeps: u64,
    safe_point: SafePoint,
    state: ContinuationState,
    /// Reference and digest of the snapshot payload owned by #1269.
    recurrent_snapshot_ref: String,
    recurrent_snapshot_sha256: String,
    recurrent_snapshot_bytes: u64,
    error_budget_evidence: ErrorBudgetEvidence,
    error_budget_total_units: u64,
    error_budget_consumed_units: u64,
    /// Direct wall-clock observations, absent when not measured.
    checkpoint_capture_wall_ns: Option<u64>,
    checkpoint_restore_wall_ns: Option<u64>,
}

impl SolverContinuation {
    fn validate(&self) -> Result<(), ResumeError> {
        if self.schema != SCHEMA || self.schema_version != SCHEMA_VERSION {
            return Err(ResumeError::UnsupportedSchema);
        }
        for value in [
            self.continuation_id.as_str(),
            self.model_id.as_str(),
            self.configuration_id.as_str(),
            self.source_revision.as_str(),
            self.solver_profile_id.as_str(),
            self.interpolation_policy_id.as_str(),
            self.recurrent_snapshot_ref.as_str(),
        ] {
            if value.trim().is_empty() {
                return Err(ResumeError::EmptyIdentity);
            }
        }
        if !is_sha256(&self.input_trace_sha256)
            || !is_sha256(&self.recurrent_snapshot_sha256)
        {
            return Err(ResumeError::InvalidDigest);
        }
        if self.target_logical_time_ns <= self.committed_logical_time_ns {
            return Err(ResumeError::InvalidLogicalTime);
        }
        if self.max_substeps == 0 || self.completed_substeps > self.max_substeps {
            return Err(ResumeError::InvalidWorkBudget);
        }
        if self.state == ContinuationState::YieldedAtSafePoint
            && self.completed_substeps == 0
        {
            return Err(ResumeError::InvalidSafePoint);
        }
        if self.state == ContinuationState::Completed {
            if self.committed_logical_time_ns != self.target_logical_time_ns {
                return Err(ResumeError::InvalidCompletion);
            }
        } else if self.committed_logical_time_ns == self.target_logical_time_ns {
            return Err(ResumeError::InvalidCompletion);
        }
        if self.recurrent_snapshot_bytes == 0 {
            return Err(ResumeError::EmptySnapshot);
        }
        if self.error_budget_consumed_units > self.error_budget_total_units {
            return Err(ResumeError::InvalidErrorBudget);
        }
        if self.error_budget_evidence == ErrorBudgetEvidence::Unspecified
            && (self.error_budget_total_units != 0
                || self.error_budget_consumed_units != 0)
        {
            return Err(ResumeError::InvalidErrorBudget);
        }
        if self.error_budget_evidence != ErrorBudgetEvidence::Unspecified
            && self.error_budget_total_units == 0
        {
            return Err(ResumeError::InvalidErrorBudget);
        }
        if self.checkpoint_capture_wall_ns == Some(0)
            || self.checkpoint_restore_wall_ns == Some(0)
        {
            return Err(ResumeError::InvalidMeasurement);
        }
        Ok(())
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ResumeError {
    UnsupportedSchema,
    EmptyIdentity,
    InvalidDigest,
    InvalidLogicalTime,
    InvalidWorkBudget,
    InvalidSafePoint,
    InvalidCompletion,
    EmptySnapshot,
    InvalidErrorBudget,
    InvalidMeasurement,
    DuplicateContinuation,
    UnknownContinuation,
    StaleGeneration,
    IdentityMismatch,
    InputTraceRevised,
    AlreadyCompleted,
    ZeroProgress,
    TimeDidNotAdvance,
    ProgressBeyondTarget,
    WorkBudgetExceeded,
    ErrorBudgetRegressed,
    ErrorBudgetExceeded,
    CounterOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResumeRequest {
    continuation_id: String,
    expected_generation: u64,
    model_id: String,
    configuration_id: String,
    source_revision: String,
    solver_profile_id: String,
    solver_method: SolverMethod,
    input_trace_sha256: String,
    interpolation_policy_id: String,
}

impl std::fmt::Display for ResumeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ResumeError {}

impl From<&SolverContinuation> for ResumeRequest {
    fn from(value: &SolverContinuation) -> Self {
        Self {
            continuation_id: value.continuation_id.clone(),
            expected_generation: value.checkpoint_generation,
            model_id: value.model_id.clone(),
            configuration_id: value.configuration_id.clone(),
            source_revision: value.source_revision.clone(),
            solver_profile_id: value.solver_profile_id.clone(),
            solver_method: value.solver_method,
            input_trace_sha256: value.input_trace_sha256.clone(),
            interpolation_policy_id: value.interpolation_policy_id.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommittedProgress {
    additional_substeps: u64,
    committed_logical_time_ns: u64,
    recurrent_snapshot_ref: String,
    recurrent_snapshot_sha256: String,
    recurrent_snapshot_bytes: u64,
    error_budget_consumed_units: u64,
    checkpoint_capture_wall_ns: Option<u64>,
    checkpoint_restore_wall_ns: Option<u64>,
}

#[derive(Debug, Default)]
struct ContinuationRegistry {
    /// Reference CAS model. A production store must implement an atomic,
    /// durable compare-and-swap; this in-memory map is only a test oracle.
    generations: BTreeMap<String, u64>,
}

impl ContinuationRegistry {
    fn register(&mut self, continuation: &SolverContinuation) -> Result<(), ResumeError> {
        continuation.validate()?;
        if self
            .generations
            .contains_key(&continuation.continuation_id)
        {
            return Err(ResumeError::DuplicateContinuation);
        }
        self.generations.insert(
            continuation.continuation_id.clone(),
            continuation.checkpoint_generation,
        );
        Ok(())
    }

    fn current_generation(&self, continuation_id: &str) -> Option<u64> {
        self.generations.get(continuation_id).copied()
    }

    /// Consume exactly one checkpoint generation and publish its successor.
    /// All validation occurs before the registry generation is advanced.
    fn advance(
        &mut self,
        continuation: &SolverContinuation,
        request: &ResumeRequest,
        progress: CommittedProgress,
    ) -> Result<SolverContinuation, ResumeError> {
        continuation.validate()?;

        let current_generation = self
            .generations
            .get(&continuation.continuation_id)
            .copied()
            .ok_or(ResumeError::UnknownContinuation)?;
        if current_generation != continuation.checkpoint_generation
            || request.expected_generation != current_generation
        {
            return Err(ResumeError::StaleGeneration);
        }
        if continuation.state == ContinuationState::Completed {
            return Err(ResumeError::AlreadyCompleted);
        }
        if request.continuation_id != continuation.continuation_id
            || request.model_id != continuation.model_id
            || request.configuration_id != continuation.configuration_id
            || request.source_revision != continuation.source_revision
            || request.solver_profile_id != continuation.solver_profile_id
            || request.solver_method != continuation.solver_method
            || request.interpolation_policy_id != continuation.interpolation_policy_id
        {
            return Err(ResumeError::IdentityMismatch);
        }
        if request.input_trace_sha256 != continuation.input_trace_sha256
            || !is_sha256(&request.input_trace_sha256)
        {
            return Err(ResumeError::InputTraceRevised);
        }
        if progress.additional_substeps == 0 {
            return Err(ResumeError::ZeroProgress);
        }

        let completed_substeps = continuation
            .completed_substeps
            .checked_add(progress.additional_substeps)
            .ok_or(ResumeError::CounterOverflow)?;
        if completed_substeps > continuation.max_substeps {
            return Err(ResumeError::WorkBudgetExceeded);
        }
        if progress.committed_logical_time_ns <= continuation.committed_logical_time_ns {
            return Err(ResumeError::TimeDidNotAdvance);
        }
        if progress.committed_logical_time_ns > continuation.target_logical_time_ns {
            return Err(ResumeError::ProgressBeyondTarget);
        }
        if progress.error_budget_consumed_units < continuation.error_budget_consumed_units {
            return Err(ResumeError::ErrorBudgetRegressed);
        }
        if progress.error_budget_consumed_units > continuation.error_budget_total_units {
            return Err(ResumeError::ErrorBudgetExceeded);
        }
        if progress.recurrent_snapshot_ref.trim().is_empty()
            || !is_sha256(&progress.recurrent_snapshot_sha256)
            || progress.recurrent_snapshot_bytes == 0
        {
            return Err(ResumeError::EmptySnapshot);
        }
        if progress.checkpoint_capture_wall_ns == Some(0)
            || progress.checkpoint_restore_wall_ns == Some(0)
        {
            return Err(ResumeError::InvalidMeasurement);
        }

        let next_generation = current_generation
            .checked_add(1)
            .ok_or(ResumeError::CounterOverflow)?;
        let completed = progress.committed_logical_time_ns
            == continuation.target_logical_time_ns;
        let next = SolverContinuation {
            schema: continuation.schema.clone(),
            schema_version: continuation.schema_version.clone(),
            continuation_id: continuation.continuation_id.clone(),
            checkpoint_generation: next_generation,
            model_id: continuation.model_id.clone(),
            configuration_id: continuation.configuration_id.clone(),
            source_revision: continuation.source_revision.clone(),
            solver_profile_id: continuation.solver_profile_id.clone(),
            solver_method: continuation.solver_method,
            input_trace_sha256: continuation.input_trace_sha256.clone(),
            interpolation_policy_id: continuation.interpolation_policy_id.clone(),
            committed_logical_time_ns: progress.committed_logical_time_ns,
            target_logical_time_ns: continuation.target_logical_time_ns,
            max_substeps: continuation.max_substeps,
            completed_substeps,
            safe_point: SafePoint::AfterCommittedSubstep,
            state: if completed {
                ContinuationState::Completed
            } else {
                ContinuationState::YieldedAtSafePoint
            },
            recurrent_snapshot_ref: progress.recurrent_snapshot_ref,
            recurrent_snapshot_sha256: progress.recurrent_snapshot_sha256,
            recurrent_snapshot_bytes: progress.recurrent_snapshot_bytes,
            error_budget_evidence: continuation.error_budget_evidence,
            error_budget_total_units: continuation.error_budget_total_units,
            error_budget_consumed_units: progress.error_budget_consumed_units,
            checkpoint_capture_wall_ns: progress.checkpoint_capture_wall_ns,
            checkpoint_restore_wall_ns: progress.checkpoint_restore_wall_ns,
        };
        next.validate()?;

        // Atomicity in this reference model: update only after the successor
        // has passed every check. Production requires durable CAS semantics.
        self.generations
            .insert(continuation.continuation_id.clone(), next_generation);
        Ok(next)
    }
}

fn fixture() -> SolverContinuation {
    SolverContinuation {
        schema: SCHEMA.to_owned(),
        schema_version: SCHEMA_VERSION.to_owned(),
        continuation_id: "trajectory-001".to_owned(),
        checkpoint_generation: 0,
        model_id: "hdc-ltc-model-v1".to_owned(),
        configuration_id: "config-digest-v1".to_owned(),
        source_revision: "source-commit-v1".to_owned(),
        solver_profile_id: "iterative-profile-v1".to_owned(),
        solver_method: SolverMethod::IterativeClosedForm,
        input_trace_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_owned(),
        interpolation_policy_id: "piecewise-linear-causal-v1".to_owned(),
        committed_logical_time_ns: 0,
        target_logical_time_ns: 8,
        max_substeps: 8,
        completed_substeps: 0,
        safe_point: SafePoint::AfterCommittedSubstep,
        state: ContinuationState::ReadyToResume,
        recurrent_snapshot_ref: "snapshot-000".to_owned(),
        recurrent_snapshot_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            .to_owned(),
        recurrent_snapshot_bytes: 4096,
        error_budget_evidence: ErrorBudgetEvidence::EmpiricalEstimate,
        error_budget_total_units: 100,
        error_budget_consumed_units: 0,
        checkpoint_capture_wall_ns: None,
        checkpoint_restore_wall_ns: None,
    }
}

fn progress(
    additional_substeps: u64,
    committed_time_ns: u64,
    step: u64,
    error_consumed: u64,
) -> CommittedProgress {
    CommittedProgress {
        additional_substeps,
        committed_logical_time_ns: committed_time_ns,
        recurrent_snapshot_ref: format!("snapshot-{step:03}"),
        recurrent_snapshot_sha256: format!("{step:064x}"),
        recurrent_snapshot_bytes: 4096,
        error_budget_consumed_units: error_consumed,
        checkpoint_capture_wall_ns: None,
        checkpoint_restore_wall_ns: None,
    }
}

fn scalar_step(value: f64) -> f64 {
    // Synthetic known-answer recurrence used only to demonstrate continuation
    // parity. It is not an HDC-LTC numerical method or performance benchmark.
    value + (1.0 - value) * 0.25
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial = fixture();
    initial.validate()?;
    let mut registry = ContinuationRegistry::default();
    registry.register(&initial)?;
    let next = registry.advance(
        &initial,
        &ResumeRequest::from(&initial),
        progress(2, 2, 2, 10),
    )?;
    println!("{}", serde_json::to_string_pretty(&next)?);
    println!("scope: schema/reference state machine only; no model or hardware claim");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_valid_initial_continuation() {
        assert_eq!(fixture().validate(), Ok(()));
    }

    #[test]
    fn rejects_invalid_schema_identity() {
        let mut token = fixture();
        token.schema_version = "9.0".to_owned();
        assert_eq!(token.validate(), Err(ResumeError::UnsupportedSchema));
    }

    #[test]
    fn rejects_malformed_digests_and_empty_snapshot() {
        let mut token = fixture();
        token.input_trace_sha256 = "not-a-digest".to_owned();
        assert_eq!(token.validate(), Err(ResumeError::InvalidDigest));

        let mut token = fixture();
        token.recurrent_snapshot_bytes = 0;
        assert_eq!(token.validate(), Err(ResumeError::EmptySnapshot));
    }

    #[test]
    fn rejects_invalid_time_and_budget_ranges() {
        let mut token = fixture();
        token.target_logical_time_ns = token.committed_logical_time_ns;
        assert_eq!(token.validate(), Err(ResumeError::InvalidLogicalTime));

        let mut token = fixture();
        token.completed_substeps = token.max_substeps + 1;
        assert_eq!(token.validate(), Err(ResumeError::InvalidWorkBudget));
    }

    #[test]
    fn rejects_yield_without_a_committed_substep() {
        let mut token = fixture();
        token.state = ContinuationState::YieldedAtSafePoint;
        assert_eq!(token.validate(), Err(ResumeError::InvalidSafePoint));
    }

    #[test]
    fn rejects_completion_before_target_time() {
        let mut token = fixture();
        token.state = ContinuationState::Completed;
        assert_eq!(token.validate(), Err(ResumeError::InvalidCompletion));
    }

    #[test]
    fn error_budget_evidence_must_match_declared_units() {
        let mut token = fixture();
        token.error_budget_consumed_units = token.error_budget_total_units + 1;
        assert_eq!(token.validate(), Err(ResumeError::InvalidErrorBudget));

        let mut token = fixture();
        token.error_budget_evidence = ErrorBudgetEvidence::Unspecified;
        assert_eq!(token.validate(), Err(ResumeError::InvalidErrorBudget));
    }

    #[test]
    fn successful_resume_advances_time_state_work_and_generation_together() {
        let token = fixture();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();

        let next = registry
            .advance(
                &token,
                &ResumeRequest::from(&token),
                progress(2, 2, 2, 10),
            )
            .unwrap();

        assert_eq!(next.checkpoint_generation, 1);
        assert_eq!(next.completed_substeps, 2);
        assert_eq!(next.committed_logical_time_ns, 2);
        assert_eq!(next.error_budget_consumed_units, 10);
        assert_eq!(next.state, ContinuationState::YieldedAtSafePoint);
        assert_eq!(registry.current_generation("trajectory-001"), Some(1));
    }

    #[test]
    fn old_checkpoint_generation_cannot_be_consumed_twice() {
        let token = fixture();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        let first = registry
            .advance(
                &token,
                &ResumeRequest::from(&token),
                progress(1, 1, 1, 1),
            )
            .unwrap();
        let error = registry.advance(
            &token,
            &ResumeRequest::from(&token),
            progress(1, 1, 1, 1),
        );
        assert_eq!(error, Err(ResumeError::StaleGeneration));
        assert_eq!(registry.current_generation("trajectory-001"), Some(1));
        assert_eq!(first.checkpoint_generation, 1);
    }

    #[test]
    fn changed_input_trace_invalidates_continuation() {
        let token = fixture();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        let mut request = ResumeRequest::from(&token);
        request.input_trace_sha256 =
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned();
        assert_eq!(
            registry.advance(&token, &request, progress(1, 1, 1, 1)),
            Err(ResumeError::InputTraceRevised)
        );
        assert_eq!(registry.current_generation("trajectory-001"), Some(0));
    }

    #[test]
    fn continuation_id_and_model_configuration_source_and_method_must_match() {
        let token = fixture();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        let mut wrong_id = ResumeRequest::from(&token);
        wrong_id.continuation_id = "different-trajectory".to_owned();
        assert_eq!(
            registry.advance(&token, &wrong_id, progress(1, 1, 1, 1)),
            Err(ResumeError::IdentityMismatch)
        );

        let mut request = ResumeRequest::from(&token);
        request.configuration_id = "different-config".to_owned();
        assert_eq!(
            registry.advance(&token, &request, progress(1, 1, 1, 1)),
            Err(ResumeError::IdentityMismatch)
        );
    }

    #[test]
    fn zero_progress_and_nonmonotonic_time_are_rejected() {
        let token = fixture();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        assert_eq!(
            registry.advance(&token, &ResumeRequest::from(&token), progress(0, 1, 1, 1)),
            Err(ResumeError::ZeroProgress)
        );
        assert_eq!(
            registry.advance(&token, &ResumeRequest::from(&token), progress(1, 0, 1, 1)),
            Err(ResumeError::TimeDidNotAdvance)
        );
    }

    #[test]
    fn cannot_advance_past_target_or_exceed_substep_budget() {
        let token = fixture();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        assert_eq!(
            registry.advance(
                &token,
                &ResumeRequest::from(&token),
                progress(1, 9, 1, 1),
            ),
            Err(ResumeError::ProgressBeyondTarget)
        );
        assert_eq!(
            registry.advance(
                &token,
                &ResumeRequest::from(&token),
                progress(9, 1, 9, 1),
            ),
            Err(ResumeError::WorkBudgetExceeded)
        );
        assert_eq!(registry.current_generation("trajectory-001"), Some(0));
    }

    #[test]
    fn error_budget_cannot_decrease_or_exceed_frozen_total() {
        let token = fixture();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        let partial = registry
            .advance(
                &token,
                &ResumeRequest::from(&token),
                progress(1, 1, 1, 20),
            )
            .unwrap();
        assert_eq!(
            registry.advance(
                &partial,
                &ResumeRequest::from(&partial),
                progress(1, 2, 2, 10),
            ),
            Err(ResumeError::ErrorBudgetRegressed)
        );
        assert_eq!(
            registry.advance(
                &partial,
                &ResumeRequest::from(&partial),
                progress(1, 2, 2, 101),
            ),
            Err(ResumeError::ErrorBudgetExceeded)
        );
        assert_eq!(registry.current_generation("trajectory-001"), Some(1));
    }

    #[test]
    fn completion_occurs_exactly_at_target_and_cannot_resume_again() {
        let token = fixture();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        let completed = registry
            .advance(
                &token,
                &ResumeRequest::from(&token),
                progress(8, 8, 8, 50),
            )
            .unwrap();
        assert_eq!(completed.state, ContinuationState::Completed);
        assert_eq!(completed.committed_logical_time_ns, 8);
        assert_eq!(
            registry.advance(
                &completed,
                &ResumeRequest::from(&completed),
                progress(1, 9, 9, 60),
            ),
            Err(ResumeError::AlreadyCompleted)
        );
    }

    #[test]
    fn cancellation_without_commit_leaves_checkpoint_generation_unchanged() {
        let token = fixture();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        // Cancellation discards only private, uncommitted working state. Since
        // advance is never called, the public checkpoint generation is stable.
        assert_eq!(registry.current_generation("trajectory-001"), Some(0));
        assert_eq!(token.committed_logical_time_ns, 0);
        assert_eq!(token.completed_substeps, 0);
    }

    #[test]
    fn yield_resume_matches_uninterrupted_synthetic_reference() {
        let mut uninterrupted = 0.0_f64;
        for _ in 0..8 {
            uninterrupted = scalar_step(uninterrupted);
        }

        let mut segmented = 0.0_f64;
        let token = fixture();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        let mut current = token;

        for end_step in [2_u64, 4, 6, 8] {
            for _ in current.completed_substeps..end_step {
                segmented = scalar_step(segmented);
            }
            let additional = end_step - current.completed_substeps;
            let next = registry
                .advance(
                    &current,
                    &ResumeRequest::from(&current),
                    progress(additional, end_step, end_step, end_step * 5),
                )
                .unwrap();
            current = next;
        }

        assert_eq!(current.state, ContinuationState::Completed);
        assert_eq!(current.completed_substeps, 8);
        assert!((segmented - uninterrupted).abs() <= f64::EPSILON);
    }

    #[test]
    fn generation_overflow_does_not_advance_checkpoint_store() {
        let mut token = fixture();
        token.checkpoint_generation = u64::MAX;
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        assert_eq!(
            registry.advance(
                &token,
                &ResumeRequest::from(&token),
                progress(1, 1, 1, 1),
            ),
            Err(ResumeError::CounterOverflow)
        );
        assert_eq!(
            registry.current_generation("trajectory-001"),
            Some(u64::MAX)
        );
    }

    #[test]
    fn completed_substep_counter_overflow_fails_before_commit() {
        let mut token = fixture();
        token.max_substeps = u64::MAX;
        token.completed_substeps = u64::MAX;
        token.committed_logical_time_ns = 7;
        token.state = ContinuationState::YieldedAtSafePoint;
        token.recurrent_snapshot_ref = "snapshot-max".to_owned();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        assert_eq!(
            registry.advance(
                &token,
                &ResumeRequest::from(&token),
                progress(1, 8, 8, 1),
            ),
            Err(ResumeError::CounterOverflow)
        );
        assert_eq!(
            registry.current_generation("trajectory-001"),
            Some(0)
        );
    }

    #[test]
    fn zero_checkpoint_timing_is_not_accepted_as_a_measurement() {
        let mut token = fixture();
        token.checkpoint_capture_wall_ns = Some(0);
        assert_eq!(token.validate(), Err(ResumeError::InvalidMeasurement));

        let token = fixture();
        let mut registry = ContinuationRegistry::default();
        registry.register(&token).unwrap();
        let mut observed_progress = progress(1, 1, 1, 1);
        observed_progress.checkpoint_restore_wall_ns = Some(0);
        assert_eq!(
            registry.advance(
                &token,
                &ResumeRequest::from(&token),
                observed_progress,
            ),
            Err(ResumeError::InvalidMeasurement)
        );
        assert_eq!(registry.current_generation("trajectory-001"), Some(0));
    }

    #[test]
    fn deterministic_serialization_of_frozen_continuation() {
        let token = fixture();
        let first = serde_json::to_string(&token).unwrap();
        let second = serde_json::to_string(&token).unwrap();
        assert_eq!(first, second);
        assert!(first.contains("\"checkpoint_generation\":0"));
        assert!(first.contains("\"committed_logical_time_ns\":0"));
    }
}
