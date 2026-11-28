// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Reference verifier for the Conversational Grounded Recovery Benchmark.
//!
//! This module intentionally separates action-adapter acknowledgement from
//! independently observed external state. It does not execute tools, issue
//! authority, authenticate an oracle, or prove that a remote service is honest.
//! The caller must pin the expected oracle profile and provide evidence from a
//! separately permissioned read-only state service.

use serde::{Deserialize, Serialize};

pub const CGRB_STATE_CONTRACT_V1: &str = "symthaea.cgrb-state.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CgrbOracleIndependenceV1 {
    SeparateReadOnlyService,
    SeparateProcess,
    SameProcessReferenceOnly,
    AgentSelfReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CgrbOracleConsistencyV1 {
    Linearizable,
    VersionedRead,
    Eventual,
    Cached,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CgrbIdempotencyV1 {
    Idempotent,
    KeyedIdempotent,
    NonIdempotent,
    ReversibleWithCompensation,
    Irreversible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CgrbOperationStatusV1 {
    Applied,
    NotApplied,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbAuthorityV1 {
    pub granted: bool,
    pub actor_token: String,
    pub action_type: String,
    pub target: String,
    pub purpose: String,
    pub destination: String,
    /// Clock domain for effective_from/effective_until/revoked_at ticks.
    pub clock_id: String,
    pub policy_version: u64,
    pub effective_from_tick: u64,
    pub effective_until_tick: u64,
    /// Effective revocation tick, if the grant was revoked.
    pub revoked_at_tick: Option<u64>,
    pub policy_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbActionAttemptV1 {
    pub operation_id: String,
    pub actor_token: String,
    pub action_type: String,
    pub target: String,
    pub purpose: String,
    pub destination: String,
    pub clock_id: String,
    pub action_digest: String,
    pub precondition_state_version: String,
    pub precondition_state_digest: String,
    pub postcondition_digest: String,
    pub idempotency: CgrbIdempotencyV1,
    pub idempotency_key: Option<String>,
    pub dispatched_at_tick: Option<u64>,
    /// Informational only: acknowledgement never proves the task succeeded.
    pub acknowledged_at_tick: Option<u64>,
    pub authority: CgrbAuthorityV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbOracleProfileV1 {
    pub oracle_id: String,
    pub oracle_build_digest: String,
    pub state_model_digest: String,
    pub clock_id: String,
    pub independence: CgrbOracleIndependenceV1,
    pub consistency: CgrbOracleConsistencyV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbStateObservationV1 {
    pub oracle_id: String,
    pub oracle_build_digest: String,
    pub state_model_digest: String,
    pub independence: CgrbOracleIndependenceV1,
    pub consistency: CgrbOracleConsistencyV1,
    pub operation_id: String,
    pub actor_token: String,
    pub action_type: String,
    pub target: String,
    pub base_state_version: String,
    pub base_state_digest: String,
    pub postcondition_digest: String,
    pub policy_version: u64,
    pub clock_id: String,
    pub observed_at_tick: u64,
    /// The external service's effect time, not the agent's local acknowledgement time.
    pub applied_at_tick: Option<u64>,
    pub operation_status: CgrbOperationStatusV1,
    /// True only when the oracle knows no delayed/in-flight application remains.
    pub operation_terminal: bool,
    /// Number of state effects attributed to this operation ID.
    pub effect_count: u32,
    pub postconditions_satisfied: Option<bool>,
    pub state_version: String,
    pub state_digest: String,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CgrbUnknownReasonV1 {
    NoDispatch,
    NoObservation,
    UntrustedOracle,
    OracleProfileMismatch,
    OperationBindingMismatch,
    ClockMismatchOrRegression,
    StaleObservation,
    IncompleteObservation,
    OperationStillInFlight,
    PostconditionNotResolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CgrbRejectionReasonV1 {
    AuthorityDenied,
    AuthorityScopeMismatch,
    DispatchOutsideGrant,
    MissingIdempotencyKey,
    InvalidAttemptContract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CgrbViolationReasonV1 {
    EffectBeforeDispatch,
    EffectAfterExpiry,
    EffectAfterRevocation,
    DuplicateOrMissingEffect,
    PostconditionFalse,
    ContradictoryOracleState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CgrbTaskOutcomeV1 {
    Committed,
    NotApplied,
    OutcomeUnknown(CgrbUnknownReasonV1),
    Rejected(CgrbRejectionReasonV1),
    Violation(CgrbViolationReasonV1),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CgrbRetryDecisionV1 {
    DoNotRetry,
    ReconcileBeforeRetry,
    ReauthorizeThenRetry,
    EscalateForReview,
}

/// Evaluate an attempt using a pinned oracle profile and a fresh observation.
///
/// A task is Committed only when the attempt was authorized at dispatch and
/// the expected independent oracle attributes exactly one applied effect to
/// this operation, confirms the required postconditions, and observes the
/// effect before expiry/revocation. API acknowledgement is deliberately not
/// required: an acknowledgement may be lost even when the effect occurred.
pub fn assess_task_outcome_v1(
    attempt: &CgrbActionAttemptV1,
    oracle_profile: &CgrbOracleProfileV1,
    observation: Option<&CgrbStateObservationV1>,
    now_clock_id: &str,
    now_tick: u64,
    max_observation_age_ticks: u64,
) -> CgrbTaskOutcomeV1 {
    if attempt.operation_id.trim().is_empty()
        || attempt.clock_id.trim().is_empty()
        || attempt.action_digest.trim().is_empty()
        || attempt.precondition_state_version.trim().is_empty()
        || attempt.precondition_state_digest.trim().is_empty()
        || attempt.postcondition_digest.trim().is_empty()
        || attempt.authority.policy_digest.trim().is_empty()
        || attempt.authority.clock_id.trim().is_empty()
        || attempt.authority.effective_until_tick < attempt.authority.effective_from_tick
    {
        return CgrbTaskOutcomeV1::Rejected(CgrbRejectionReasonV1::InvalidAttemptContract);
    }

    if !attempt.authority.granted {
        return CgrbTaskOutcomeV1::Rejected(CgrbRejectionReasonV1::AuthorityDenied);
    }

    if attempt.clock_id != attempt.authority.clock_id {
        return CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::ClockMismatchOrRegression);
    }

    if attempt.actor_token != attempt.authority.actor_token
        || attempt.action_type != attempt.authority.action_type
        || attempt.target != attempt.authority.target
        || attempt.purpose != attempt.authority.purpose
        || attempt.destination != attempt.authority.destination
    {
        return CgrbTaskOutcomeV1::Rejected(CgrbRejectionReasonV1::AuthorityScopeMismatch);
    }

    let dispatch_tick = match attempt.dispatched_at_tick {
        Some(tick) => tick,
        None => return CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::NoDispatch),
    };

    if dispatch_tick < attempt.authority.effective_from_tick
        || dispatch_tick >= attempt.authority.effective_until_tick
        || attempt
            .authority
            .revoked_at_tick
            .is_some_and(|revoked_at| dispatch_tick >= revoked_at)
    {
        return CgrbTaskOutcomeV1::Rejected(CgrbRejectionReasonV1::DispatchOutsideGrant);
    }

    if attempt.idempotency == CgrbIdempotencyV1::KeyedIdempotent
        && attempt
            .idempotency_key
            .as_deref()
            .is_none_or(|key| key.trim().is_empty())
    {
        return CgrbTaskOutcomeV1::Rejected(CgrbRejectionReasonV1::MissingIdempotencyKey);
    }

    let observation = match observation {
        Some(value) => value,
        None => return CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::NoObservation),
    };

    if oracle_profile.independence != CgrbOracleIndependenceV1::SeparateReadOnlyService
        || !matches!(
            oracle_profile.consistency,
            CgrbOracleConsistencyV1::Linearizable | CgrbOracleConsistencyV1::VersionedRead
        )
        || observation.independence != oracle_profile.independence
        || observation.consistency != oracle_profile.consistency
    {
        return CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::UntrustedOracle);
    }

    if now_clock_id != attempt.clock_id || oracle_profile.clock_id != attempt.clock_id {
        return CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::ClockMismatchOrRegression);
    }

    if observation.oracle_id != oracle_profile.oracle_id
        || observation.oracle_build_digest != oracle_profile.oracle_build_digest
        || observation.state_model_digest != oracle_profile.state_model_digest
        || observation.clock_id != oracle_profile.clock_id
        || observation.oracle_id.is_empty()
        || observation.oracle_build_digest.is_empty()
        || observation.state_model_digest.is_empty()
    {
        return CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::OracleProfileMismatch);
    }

    if observation.operation_id != attempt.operation_id
        || observation.actor_token != attempt.actor_token
        || observation.action_type != attempt.action_type
        || observation.target != attempt.target
        || observation.base_state_version != attempt.precondition_state_version
        || observation.base_state_digest != attempt.precondition_state_digest
        || observation.postcondition_digest != attempt.postcondition_digest
        || observation.policy_version != attempt.authority.policy_version
    {
        return CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::OperationBindingMismatch);
    }

    if observation.clock_id != attempt.clock_id
        || observation.observed_at_tick < dispatch_tick
        || observation.observed_at_tick > now_tick
        || now_tick < observation.observed_at_tick
    {
        return CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::ClockMismatchOrRegression);
    }

    if now_tick.saturating_sub(observation.observed_at_tick) > max_observation_age_ticks {
        return CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::StaleObservation);
    }

    if observation.state_version.trim().is_empty()
        || observation.state_digest.trim().is_empty()
        || observation.evidence_digest.trim().is_empty()
    {
        return CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::IncompleteObservation);
    }

    match observation.operation_status {
        CgrbOperationStatusV1::Unknown => {
            CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::PostconditionNotResolved)
        }
        CgrbOperationStatusV1::NotApplied => {
            if observation.effect_count != 0 {
                CgrbTaskOutcomeV1::Violation(CgrbViolationReasonV1::ContradictoryOracleState)
            } else if !observation.operation_terminal {
                CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::OperationStillInFlight)
            } else {
                CgrbTaskOutcomeV1::NotApplied
            }
        }
        CgrbOperationStatusV1::Applied => {
            let applied_tick = match observation.applied_at_tick {
                Some(tick) => tick,
                None => {
                    return CgrbTaskOutcomeV1::OutcomeUnknown(
                        CgrbUnknownReasonV1::IncompleteObservation,
                    );
                }
            };

            if applied_tick < dispatch_tick || observation.observed_at_tick < applied_tick {
                return CgrbTaskOutcomeV1::Violation(CgrbViolationReasonV1::EffectBeforeDispatch);
            }
            if applied_tick >= attempt.authority.effective_until_tick {
                return CgrbTaskOutcomeV1::Violation(CgrbViolationReasonV1::EffectAfterExpiry);
            }
            if attempt
                .authority
                .revoked_at_tick
                .is_some_and(|revoked_at| applied_tick >= revoked_at)
            {
                return CgrbTaskOutcomeV1::Violation(CgrbViolationReasonV1::EffectAfterRevocation);
            }
            if !observation.operation_terminal {
                return CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::OperationStillInFlight);
            }
            if observation.effect_count != 1 {
                return CgrbTaskOutcomeV1::Violation(CgrbViolationReasonV1::DuplicateOrMissingEffect);
            }
            match observation.postconditions_satisfied {
                Some(true) => CgrbTaskOutcomeV1::Committed,
                Some(false) => CgrbTaskOutcomeV1::Violation(CgrbViolationReasonV1::PostconditionFalse),
                None => CgrbTaskOutcomeV1::OutcomeUnknown(
                    CgrbUnknownReasonV1::PostconditionNotResolved,
                ),
            }
        }
    }
}

/// Unknown outcomes cannot be retried blindly. A definitive independent
/// NotApplied observation allows a new attempt only after live authority is
/// checked again. A committed operation must not be replayed.
pub fn retry_decision_v1(outcome: &CgrbTaskOutcomeV1) -> CgrbRetryDecisionV1 {
    match outcome {
        CgrbTaskOutcomeV1::Committed => CgrbRetryDecisionV1::DoNotRetry,
        CgrbTaskOutcomeV1::NotApplied => CgrbRetryDecisionV1::ReauthorizeThenRetry,
        CgrbTaskOutcomeV1::OutcomeUnknown(_) => CgrbRetryDecisionV1::ReconcileBeforeRetry,
        CgrbTaskOutcomeV1::Rejected(_) => CgrbRetryDecisionV1::DoNotRetry,
        CgrbTaskOutcomeV1::Violation(_) => CgrbRetryDecisionV1::EscalateForReview,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        format!("sha256:{}", ch.to_string().repeat(64))
    }

    fn attempt() -> CgrbActionAttemptV1 {
        CgrbActionAttemptV1 {
            operation_id: "op-1".into(),
            actor_token: "actor-1".into(),
            action_type: "record.update".into(),
            target: "record-1".into(),
            purpose: "user-requested-update".into(),
            destination: "test-world".into(),
            clock_id: "world-clock".into(),
            action_digest: digest('a'),
            precondition_state_version: "v1".into(),
            precondition_state_digest: digest('b'),
            postcondition_digest: digest('2'),
            idempotency: CgrbIdempotencyV1::KeyedIdempotent,
            idempotency_key: Some("idem-1".into()),
            dispatched_at_tick: Some(10),
            acknowledged_at_tick: None,
            authority: CgrbAuthorityV1 {
                granted: true,
                actor_token: "actor-1".into(),
                action_type: "record.update".into(),
                target: "record-1".into(),
                purpose: "user-requested-update".into(),
                destination: "test-world".into(),
                clock_id: "world-clock".into(),
                policy_version: 3,
                effective_from_tick: 1,
                effective_until_tick: 100,
                revoked_at_tick: None,
                policy_digest: digest('c'),
            },
        }
    }

    fn profile() -> CgrbOracleProfileV1 {
        CgrbOracleProfileV1 {
            oracle_id: "oracle-1".into(),
            oracle_build_digest: digest('d'),
            state_model_digest: digest('e'),
            clock_id: "world-clock".into(),
            independence: CgrbOracleIndependenceV1::SeparateReadOnlyService,
            consistency: CgrbOracleConsistencyV1::VersionedRead,
        }
    }

    fn observation() -> CgrbStateObservationV1 {
        CgrbStateObservationV1 {
            oracle_id: "oracle-1".into(),
            oracle_build_digest: digest('d'),
            state_model_digest: digest('e'),
            independence: CgrbOracleIndependenceV1::SeparateReadOnlyService,
            consistency: CgrbOracleConsistencyV1::VersionedRead,
            operation_id: "op-1".into(),
            actor_token: "actor-1".into(),
            action_type: "record.update".into(),
            target: "record-1".into(),
            base_state_version: "v1".into(),
            base_state_digest: digest('b'),
            postcondition_digest: digest('2'),
            policy_version: 3,
            clock_id: "world-clock".into(),
            observed_at_tick: 20,
            applied_at_tick: Some(19),
            operation_status: CgrbOperationStatusV1::Applied,
            operation_terminal: true,
            effect_count: 1,
            postconditions_satisfied: Some(true),
            state_version: "v2".into(),
            state_digest: digest('f'),
            evidence_digest: digest('1'),
        }
    }

    #[test]
    fn independent_observation_commits_even_when_acknowledgement_was_lost() {
        let result = assess_task_outcome_v1(&attempt(), &profile(), Some(&observation()), "world-clock", 21, 5);
        assert_eq!(result, CgrbTaskOutcomeV1::Committed);
    }

    #[test]
    fn adapter_acknowledgement_without_oracle_observation_is_unknown() {
        let mut value = attempt();
        value.acknowledged_at_tick = Some(15);
        let result = assess_task_outcome_v1(&value, &profile(), None, "world-clock", 21, 5);
        assert_eq!(
            result,
            CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::NoObservation)
        );
    }

    #[test]
    fn agent_self_report_cannot_qualify_external_state() {
        let mut oracle = profile();
        oracle.independence = CgrbOracleIndependenceV1::AgentSelfReport;
        let result = assess_task_outcome_v1(&attempt(), &oracle, Some(&observation()), "world-clock", 21, 5);
        assert_eq!(
            result,
            CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::UntrustedOracle)
        );
    }

    #[test]
    fn stale_observation_remains_unknown() {
        let result = assess_task_outcome_v1(&attempt(), &profile(), Some(&observation()), "world-clock", 50, 5);
        assert_eq!(
            result,
            CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::StaleObservation)
        );
    }

    #[test]
    fn duplicate_effects_are_a_violation() {
        let mut observed = observation();
        observed.effect_count = 2;
        let result = assess_task_outcome_v1(&attempt(), &profile(), Some(&observed), "world-clock", 21, 5);
        assert_eq!(
            result,
            CgrbTaskOutcomeV1::Violation(CgrbViolationReasonV1::DuplicateOrMissingEffect)
        );
    }

    #[test]
    fn post_revocation_application_is_a_violation() {
        let mut value = attempt();
        value.authority.revoked_at_tick = Some(18);
        let result = assess_task_outcome_v1(&value, &profile(), Some(&observation()), "world-clock", 21, 5);
        assert_eq!(
            result,
            CgrbTaskOutcomeV1::Violation(CgrbViolationReasonV1::EffectAfterRevocation)
        );
    }

    #[test]
    fn dispatch_after_revocation_is_rejected() {
        let mut value = attempt();
        value.authority.revoked_at_tick = Some(9);
        let result = assess_task_outcome_v1(&value, &profile(), Some(&observation()), "world-clock", 21, 5);
        assert_eq!(
            result,
            CgrbTaskOutcomeV1::Rejected(CgrbRejectionReasonV1::DispatchOutsideGrant)
        );
    }

    #[test]
    fn definitive_no_effect_requires_reauthorization_before_retry() {
        let mut observed = observation();
        observed.operation_status = CgrbOperationStatusV1::NotApplied;
        observed.applied_at_tick = None;
        observed.effect_count = 0;
        observed.postconditions_satisfied = Some(false);
        let result = assess_task_outcome_v1(&attempt(), &profile(), Some(&observed), "world-clock", 21, 5);
        assert_eq!(result, CgrbTaskOutcomeV1::NotApplied);
        assert_eq!(
            retry_decision_v1(&result),
            CgrbRetryDecisionV1::ReauthorizeThenRetry
        );
    }

    #[test]
    fn authority_window_clock_must_match_action_clock() {
        let mut value = attempt();
        value.authority.clock_id = "other-clock".into();
        let result = assess_task_outcome_v1(
            &value, &profile(), Some(&observation()), "world-clock", 21, 5
        );
        assert_eq!(
            result,
            CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::ClockMismatchOrRegression)
        );
    }

    #[test]
    fn observation_consistency_must_match_pinned_profile() {
        let mut observed = observation();
        observed.consistency = CgrbOracleConsistencyV1::Linearizable;
        let result = assess_task_outcome_v1(
            &attempt(), &profile(), Some(&observed), "world-clock", 21, 5
        );
        assert_eq!(
            result,
            CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::UntrustedOracle)
        );
    }

    #[test]
    fn false_postcondition_is_a_violation() {
        let mut observed = observation();
        observed.postconditions_satisfied = Some(false);
        let result = assess_task_outcome_v1(&attempt(), &profile(), Some(&observed), "world-clock", 21, 5);
        assert_eq!(
            result,
            CgrbTaskOutcomeV1::Violation(CgrbViolationReasonV1::PostconditionFalse)
        );
    }

    #[test]
    fn assessment_clock_must_match_attempt_clock() {
        let result = assess_task_outcome_v1(&attempt(), &profile(), Some(&observation()), "other-clock", 21, 5);
        assert_eq!(
            result,
            CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::ClockMismatchOrRegression)
        );
    }

    #[test]
    fn effect_at_expiry_boundary_is_a_violation() {
        let mut value = attempt();
        value.authority.effective_until_tick = 19;
        let mut observed = observation();
        observed.applied_at_tick = Some(19);
        let result = assess_task_outcome_v1(&value, &profile(), Some(&observed), "world-clock", 21, 5);
        assert_eq!(
            result,
            CgrbTaskOutcomeV1::Violation(CgrbViolationReasonV1::EffectAfterExpiry)
        );
    }

    #[test]
    fn unknown_outcome_requires_reconciliation_before_retry() {
        let result = CgrbTaskOutcomeV1::OutcomeUnknown(CgrbUnknownReasonV1::NoObservation);
        assert_eq!(
            retry_decision_v1(&result),
            CgrbRetryDecisionV1::ReconcileBeforeRetry
        );
    }
}
