//! Conservative, train-evidence-bounded recovery specialist selection.
//!
//! This crate selects a policy identity; it does not emit an action, actuate a
//! robot, or grant safety authority. Candidate evidence must be produced by an
//! independently audited, training-only cross-fitting pipeline. Confidence
//! bounds are computed over the supplied mutually exclusive outcome counts.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceScope {
    /// Aggregate evidence is sourced only from the frozen training split and
    /// policy competence estimates were produced out-of-fold.
    TrainingOnlyCrossFitted,
    /// Mixed, held-out, unknown, or otherwise nonconforming evidence.
    OtherOrUnknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbstentionReason {
    NoCandidatePassedEvidenceAndCompatibilityGates,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RejectionReason {
    InvalidPolicyId,
    DuplicatePolicyId,
    InvalidProvenanceDigest,
    ContextMismatch,
    FeatureSpaceMismatch,
    EmbodimentMismatch,
    ActionInterfaceMismatch,
    RuntimeEnvironmentMismatch,
    EvidenceNotTrainingOnlyCrossFitted,
    NoClassifiedTrials,
    OutcomeCountsDoNotPartitionAttempts,
    TooFewTrials,
    SafeRecoveryLowerBoundTooLow,
    AdverseOrUnknownUpperBoundTooHigh,
    LatencyExceedsDeadline,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutcomeCounts {
    /// Independently declared count of valid policy attempts.
    pub declared_attempts: u64,
    /// Policy-completed task success with independently verified safe outcome.
    pub safe_recoveries: u64,
    /// Any adverse event, or completion whose safety/physical continuity is
    /// unknown or unverified. This is deliberately treated as risk-positive.
    pub adverse_or_unverified: u64,
    /// Known policy non-completion without a separately recorded adverse event.
    pub known_non_completions: u64,
    /// Infrastructure-invalid rows excluded from the outcome denominator.
    /// They remain visible and must be covered by the aggregate evidence receipt.
    pub infrastructure_indeterminate: u64,
}

impl OutcomeCounts {
    /// Number of classified policy attempts; infrastructure-indeterminate rows
    /// are excluded and separately retained.
    pub fn attempts(&self) -> Option<u64> {
        let partition = self.safe_recoveries
            .checked_add(self.adverse_or_unverified)?
            .checked_add(self.known_non_completions)?;
        (partition == self.declared_attempts).then_some(partition)
    }

    pub fn is_partitioned(&self) -> bool {
        self.attempts().is_some_and(|n| n > 0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolicyCompetenceProfile {
    pub policy_id: String,
    pub policy_sha256: String,
    /// Canonical context bucket derived from deployable observations/history.
    /// It must not encode evaluator-only Stage/Type or hidden scenario labels.
    pub observation_context_key: String,
    pub feature_space_sha256: String,
    pub embodiment_sha256: String,
    pub action_interface_sha256: String,
    /// Runtime/hardware identity on which p95 inference latency was measured.
    pub runtime_environment_sha256: String,
    pub compatibility_receipt_sha256: String,
    pub training_manifest_sha256: String,
    pub crossfit_receipt_sha256: String,
    pub aggregate_evidence_sha256: String,
    pub evidence_scope: EvidenceScope,
    pub outcomes: OutcomeCounts,
    pub p95_latency_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutingContext {
    pub observation_context_key: String,
    pub feature_space_sha256: String,
    pub embodiment_sha256: String,
    pub action_interface_sha256: String,
    pub runtime_environment_sha256: String,
    pub deadline_ms: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouterConfig {
    pub min_attempts: u64,
    pub min_safe_recovery_lcb: f64,
    pub max_adverse_or_unknown_ucb: f64,
    /// Normal quantile multiplier. 1.96 is the conventional two-sided 95% value.
    pub confidence_z: f64,
}

impl RouterConfig {
    pub fn conservative_v0() -> Self {
        Self {
            min_attempts: 30,
            min_safe_recovery_lcb: 0.60,
            max_adverse_or_unknown_ucb: 0.05,
            confidence_z: 1.96,
        }
    }

    fn validate(&self) -> Result<(), RouterError> {
        if self.min_attempts == 0
            || !unit_interval(self.min_safe_recovery_lcb)
            || !unit_interval(self.max_adverse_or_unknown_ucb)
            || !self.confidence_z.is_finite()
            || self.confidence_z <= 0.0
        {
            return Err(RouterError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RouterError {
    InvalidConfiguration,
    InvalidRoutingContext,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateAssessment {
    pub policy_id: String,
    pub attempts: Option<u64>,
    pub safe_recovery_interval: Option<ConfidenceInterval>,
    pub adverse_or_unknown_interval: Option<ConfidenceInterval>,
    pub p95_latency_ms: u64,
    pub eligible: bool,
    pub selected: bool,
    pub rejection: Option<RejectionReason>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicySelection {
    pub policy_id: String,
    pub policy_sha256: String,
    pub compatibility_receipt_sha256: String,
    pub crossfit_receipt_sha256: String,
    pub safe_recovery_lcb: f64,
    pub adverse_or_unknown_ucb: f64,
    pub p95_latency_ms: u64,
    pub attempts: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouteDecision {
    /// None means abstention. No action or actuator command is returned.
    pub selected: Option<PolicySelection>,
    pub abstention: Option<AbstentionReason>,
    /// One deterministic assessment per candidate input row, sorted by policy ID.
    pub assessments: Vec<CandidateAssessment>,
}

fn unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn valid_token(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'_' | b'-' | b'.' | b':')
        })
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_context(context: &RoutingContext) -> bool {
    valid_token(&context.observation_context_key)
        && valid_sha256(&context.feature_space_sha256)
        && valid_sha256(&context.embodiment_sha256)
        && valid_sha256(&context.action_interface_sha256)
        && valid_sha256(&context.runtime_environment_sha256)
        && context.deadline_ms > 0
}

fn wilson_interval(
    successes: u64,
    attempts: u64,
    z: f64,
) -> Option<ConfidenceInterval> {
    if attempts == 0 || successes > attempts || !z.is_finite() || z <= 0.0 {
        return None;
    }

    let n = attempts as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let radius = z
        * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt())
        / denominator;

    Some(ConfidenceInterval {
        lower: (center - radius).max(0.0),
        upper: (center + radius).min(1.0),
    })
}

fn profile_rejection(
    profile: &PolicyCompetenceProfile,
    context: &RoutingContext,
    config: &RouterConfig,
    duplicate_id: bool,
) -> Result<(u64, ConfidenceInterval, ConfidenceInterval), RejectionReason> {
    if !valid_token(&profile.policy_id) {
        return Err(RejectionReason::InvalidPolicyId);
    }
    if duplicate_id {
        return Err(RejectionReason::DuplicatePolicyId);
    }

    if !valid_sha256(&profile.policy_sha256)
        || !valid_sha256(&profile.feature_space_sha256)
        || !valid_sha256(&profile.embodiment_sha256)
        || !valid_sha256(&profile.action_interface_sha256)
        || !valid_sha256(&profile.runtime_environment_sha256)
        || !valid_sha256(&profile.compatibility_receipt_sha256)
        || !valid_sha256(&profile.training_manifest_sha256)
        || !valid_sha256(&profile.crossfit_receipt_sha256)
        || !valid_sha256(&profile.aggregate_evidence_sha256)
    {
        return Err(RejectionReason::InvalidProvenanceDigest);
    }

    if profile.observation_context_key != context.observation_context_key {
        return Err(RejectionReason::ContextMismatch);
    }
    if profile.feature_space_sha256 != context.feature_space_sha256 {
        return Err(RejectionReason::FeatureSpaceMismatch);
    }
    if profile.embodiment_sha256 != context.embodiment_sha256 {
        return Err(RejectionReason::EmbodimentMismatch);
    }
    if profile.action_interface_sha256 != context.action_interface_sha256 {
        return Err(RejectionReason::ActionInterfaceMismatch);
    }
    if profile.runtime_environment_sha256 != context.runtime_environment_sha256 {
        return Err(RejectionReason::RuntimeEnvironmentMismatch);
    }
    if profile.evidence_scope != EvidenceScope::TrainingOnlyCrossFitted {
        return Err(RejectionReason::EvidenceNotTrainingOnlyCrossFitted);
    }

    let attempts = profile
        .outcomes
        .attempts()
        .ok_or(RejectionReason::OutcomeCountsDoNotPartitionAttempts)?;
    if attempts == 0 {
        return Err(RejectionReason::NoClassifiedTrials);
    }
    if attempts < config.min_attempts {
        return Err(RejectionReason::TooFewTrials);
    }
    if profile.p95_latency_ms > context.deadline_ms {
        return Err(RejectionReason::LatencyExceedsDeadline);
    }

    let safe = wilson_interval(profile.outcomes.safe_recoveries, attempts, config.confidence_z)
        .ok_or(RejectionReason::OutcomeCountsDoNotPartitionAttempts)?;
    let adverse = wilson_interval(
        profile.outcomes.adverse_or_unverified,
        attempts,
        config.confidence_z,
    )
    .ok_or(RejectionReason::OutcomeCountsDoNotPartitionAttempts)?;

    if safe.lower < config.min_safe_recovery_lcb {
        return Err(RejectionReason::SafeRecoveryLowerBoundTooLow);
    }
    if adverse.upper > config.max_adverse_or_unknown_ucb {
        return Err(RejectionReason::AdverseOrUnknownUpperBoundTooHigh);
    }

    Ok((attempts, safe, adverse))
}

/// Select a compatible recovery specialist using only frozen training evidence
/// and the runtime-observation context. This function never receives hidden
/// evaluator labels, scenario outcomes, or a motor/action interface.
pub fn route(
    context: &RoutingContext,
    candidates: &[PolicyCompetenceProfile],
    config: &RouterConfig,
) -> Result<RouteDecision, RouterError> {
    config.validate()?;
    if !valid_context(context) {
        return Err(RouterError::InvalidRoutingContext);
    }

    let mut id_counts = BTreeMap::<String, usize>::new();
    for candidate in candidates {
        *id_counts.entry(candidate.policy_id.clone()).or_insert(0) += 1;
    }

    struct RankedCandidate {
        assessment_index: usize,
        selection: PolicySelection,
    }

    let mut assessments = Vec::with_capacity(candidates.len());
    let mut ranked = Vec::<RankedCandidate>::new();

    for candidate in candidates {
        let duplicate_id = id_counts.get(&candidate.policy_id).copied().unwrap_or(0) > 1;
        match profile_rejection(candidate, context, config, duplicate_id) {
            Ok((attempts, safe, adverse)) => {
                let assessment_index = assessments.len();
                assessments.push(CandidateAssessment {
                    policy_id: candidate.policy_id.clone(),
                    attempts: Some(attempts),
                    safe_recovery_interval: Some(safe),
                    adverse_or_unknown_interval: Some(adverse),
                    p95_latency_ms: candidate.p95_latency_ms,
                    eligible: true,
                    selected: false,
                    rejection: None,
                });
                ranked.push(RankedCandidate {
                    assessment_index,
                    selection: PolicySelection {
                        policy_id: candidate.policy_id.clone(),
                        policy_sha256: candidate.policy_sha256.clone(),
                        compatibility_receipt_sha256: candidate.compatibility_receipt_sha256.clone(),
                        crossfit_receipt_sha256: candidate.crossfit_receipt_sha256.clone(),
                        safe_recovery_lcb: safe.lower,
                        adverse_or_unknown_ucb: adverse.upper,
                        p95_latency_ms: candidate.p95_latency_ms,
                        attempts,
                    },
                });
            }
            Err(reason) => assessments.push(CandidateAssessment {
                policy_id: candidate.policy_id.clone(),
                attempts: candidate.outcomes.attempts(),
                safe_recovery_interval: None,
                adverse_or_unknown_interval: None,
                p95_latency_ms: candidate.p95_latency_ms,
                eligible: false,
                selected: false,
                rejection: Some(reason),
            }),
        }
    }

    ranked.sort_by(|left, right| {
        right
            .selection
            .safe_recovery_lcb
            .total_cmp(&left.selection.safe_recovery_lcb)
            .then_with(|| {
                left.selection
                    .adverse_or_unknown_ucb
                    .total_cmp(&right.selection.adverse_or_unknown_ucb)
            })
            .then_with(|| {
                left.selection
                    .p95_latency_ms
                    .cmp(&right.selection.p95_latency_ms)
            })
            .then_with(|| left.selection.policy_id.cmp(&right.selection.policy_id))
    });

    let selected = ranked.first().map(|winner| {
        assessments[winner.assessment_index].selected = true;
        winner.selection.clone()
    });

    assessments.sort_by(|left, right| left.policy_id.cmp(&right.policy_id));

    Ok(RouteDecision {
        abstention: selected
            .is_none()
            .then_some(AbstentionReason::NoCandidatePassedEvidenceAndCompatibilityGates),
        selected,
        assessments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> RoutingContext {
        RoutingContext {
            observation_context_key: "grasp_object_reorient".to_owned(),
            feature_space_sha256: "a".repeat(64),
            embodiment_sha256: "b".repeat(64),
            action_interface_sha256: "c".repeat(64),
            runtime_environment_sha256: "d".repeat(64),
            deadline_ms: 250,
        }
    }

    fn profile(
        policy_id: &str,
        safe: u64,
        adverse: u64,
        known_failure: u64,
    ) -> PolicyCompetenceProfile {
        PolicyCompetenceProfile {
            policy_id: policy_id.to_owned(),
            policy_sha256: "e".repeat(64),
            observation_context_key: "grasp_object_reorient".to_owned(),
            feature_space_sha256: "a".repeat(64),
            embodiment_sha256: "b".repeat(64),
            action_interface_sha256: "c".repeat(64),
            runtime_environment_sha256: "d".repeat(64),
            compatibility_receipt_sha256: "f".repeat(64),
            training_manifest_sha256: "1".repeat(64),
            crossfit_receipt_sha256: "2".repeat(64),
            aggregate_evidence_sha256: "3".repeat(64),
            evidence_scope: EvidenceScope::TrainingOnlyCrossFitted,
            outcomes: OutcomeCounts {
                declared_attempts: safe + adverse + known_failure,
                safe_recoveries: safe,
                adverse_or_unverified: adverse,
                known_non_completions: known_failure,
                infrastructure_indeterminate: 0,
            },
            p95_latency_ms: 100,
        }
    }

    fn permissive_config() -> RouterConfig {
        RouterConfig {
            min_attempts: 20,
            min_safe_recovery_lcb: 0.60,
            max_adverse_or_unknown_ucb: 0.05,
            confidence_z: 1.96,
        }
    }

    #[test]
    fn wilson_interval_matches_known_boundary_case() {
        let interval = wilson_interval(10, 10, 1.96).unwrap();
        assert!((interval.lower - 0.7225).abs() < 0.001);
        assert_eq!(interval.upper, 1.0);
    }

    #[test]
    fn no_adverse_events_still_needs_enough_samples_for_a_risk_bound() {
        let small = wilson_interval(0, 40, 1.96).unwrap();
        let larger = wilson_interval(0, 100, 1.96).unwrap();
        assert!(small.upper > 0.05);
        assert!(larger.upper < 0.05);
    }

    #[test]
    fn select_highest_safe_recovery_lower_bound_after_risk_gates() {
        let context = context();
        let mut high = profile("specialist-high", 100, 0, 0);
        high.p95_latency_ms = 120;
        let lower = profile("specialist-lower", 90, 0, 10);

        let result = route(&context, &[lower, high], &permissive_config()).unwrap();
        assert_eq!(result.selected.unwrap().policy_id, "specialist-high");
        assert_eq!(result.assessments.iter().filter(|x| x.selected).count(), 1);
    }

    #[test]
    fn unknown_or_unverified_safety_is_risk_positive() {
        let context = context();
        let candidate = profile("specialist", 90, 5, 5);
        let result = route(&context, &[candidate], &permissive_config()).unwrap();
        assert!(result.selected.is_none());
        assert_eq!(
            result.assessments[0].rejection,
            Some(RejectionReason::AdverseOrUnknownUpperBoundTooHigh)
        );
    }

    #[test]
    fn thin_evidence_abstains_even_when_every_observed_trial_was_safe() {
        let context = context();
        let candidate = profile("thin", 40, 0, 0);
        let result = route(&context, &[candidate], &permissive_config()).unwrap();
        assert!(result.selected.is_none());
        assert_eq!(
            result.assessments[0].rejection,
            Some(RejectionReason::AdverseOrUnknownUpperBoundTooHigh)
        );
    }

    #[test]
    fn mixed_or_unknown_data_scope_is_rejected() {
        let context = context();
        let mut candidate = profile("mixed", 100, 0, 0);
        candidate.evidence_scope = EvidenceScope::OtherOrUnknown;
        let result = route(&context, &[candidate], &permissive_config()).unwrap();
        assert!(result.selected.is_none());
        assert_eq!(
            result.assessments[0].rejection,
            Some(RejectionReason::EvidenceNotTrainingOnlyCrossFitted)
        );
    }

    #[test]
    fn held_out_or_wrong_context_cell_is_not_used() {
        let context = context();
        let mut candidate = profile("wrong-context", 100, 0, 0);
        candidate.observation_context_key = "another_observation_bucket".to_owned();
        let result = route(&context, &[candidate], &permissive_config()).unwrap();
        assert!(result.selected.is_none());
        assert_eq!(
            result.assessments[0].rejection,
            Some(RejectionReason::ContextMismatch)
        );
    }

    #[test]
    fn embodiment_action_interface_and_runtime_must_match() {
        let context = context();
        let mut candidate = profile("wrong-robot", 100, 0, 0);
        candidate.embodiment_sha256 = "9".repeat(64);
        let result = route(&context, &[candidate], &permissive_config()).unwrap();
        assert_eq!(
            result.assessments[0].rejection,
            Some(RejectionReason::EmbodimentMismatch)
        );

        let mut candidate = profile("wrong-runtime", 100, 0, 0);
        candidate.runtime_environment_sha256 = "8".repeat(64);
        let result = route(&context, &[candidate], &permissive_config()).unwrap();
        assert_eq!(
            result.assessments[0].rejection,
            Some(RejectionReason::RuntimeEnvironmentMismatch)
        );
    }

    #[test]
    fn malformed_provenance_digest_is_rejected() {
        let context = context();
        let mut candidate = profile("bad-digest", 100, 0, 0);
        candidate.crossfit_receipt_sha256 = "not-a-digest".to_owned();
        let result = route(&context, &[candidate], &permissive_config()).unwrap();
        assert_eq!(
            result.assessments[0].rejection,
            Some(RejectionReason::InvalidProvenanceDigest)
        );
    }

    #[test]
    fn duplicate_policy_ids_are_all_rejected_not_pooled() {
        let context = context();
        let first = profile("same-policy", 100, 0, 0);
        let second = profile("same-policy", 100, 0, 0);
        let result = route(&context, &[first, second], &permissive_config()).unwrap();
        assert!(result.selected.is_none());
        assert_eq!(result.assessments.len(), 2);
        assert!(result.assessments.iter().all(|row| {
            row.rejection == Some(RejectionReason::DuplicatePolicyId)
        }));
    }

    #[test]
    fn latency_over_deadline_is_rejected() {
        let context = context();
        let mut candidate = profile("slow", 100, 0, 0);
        candidate.p95_latency_ms = 251;
        let result = route(&context, &[candidate], &permissive_config()).unwrap();
        assert_eq!(
            result.assessments[0].rejection,
            Some(RejectionReason::LatencyExceedsDeadline)
        );
    }

    #[test]
    fn malformed_outcome_partition_is_rejected() {
        let context = context();
        let mut candidate = profile("malformed", 100, 0, 0);
        candidate.outcomes.known_non_completions = 1;
        let result = route(&context, &[candidate], &permissive_config()).unwrap();
        assert_eq!(
            result.assessments[0].rejection,
            Some(RejectionReason::OutcomeCountsDoNotPartitionAttempts)
        );
    }

    #[test]
    fn invalid_configuration_and_context_fail_closed() {
        let mut config = permissive_config();
        config.max_adverse_or_unknown_ucb = f64::NAN;
        assert_eq!(
            route(&context(), &[], &config),
            Err(RouterError::InvalidConfiguration)
        );

        let mut context = context();
        context.deadline_ms = 0;
        assert_eq!(
            route(&context, &[], &permissive_config()),
            Err(RouterError::InvalidRoutingContext)
        );
    }

    #[test]
    fn ties_are_deterministic_and_lexical_after_risk_and_latency() {
        let context = context();
        let first = profile("policy-b", 100, 0, 0);
        let second = profile("policy-a", 100, 0, 0);
        let result = route(&context, &[first, second], &permissive_config()).unwrap();
        assert_eq!(result.selected.unwrap().policy_id, "policy-a");
    }

    #[test]
    fn empty_portfolio_returns_explicit_abstention() {
        let result = route(&context(), &[], &permissive_config()).unwrap();
        assert!(result.selected.is_none());
        assert_eq!(
            result.abstention,
            Some(AbstentionReason::NoCandidatePassedEvidenceAndCompatibilityGates)
        );
    }

    #[test]
    fn excluded_infrastructure_rows_remain_visible_but_do_not_count_as_attempts() {
        let context = context();
        let mut candidate = profile("infra-heavy", 100, 0, 0);
        candidate.outcomes.infrastructure_indeterminate = 7;
        let result = route(&context, &[candidate], &permissive_config()).unwrap();
        assert_eq!(result.selected.unwrap().policy_id, "infra-heavy");
        assert_eq!(result.assessments[0].attempts, Some(100));
    }
}
