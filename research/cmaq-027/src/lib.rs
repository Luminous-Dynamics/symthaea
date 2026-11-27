#![forbid(unsafe_code)]

//! Finite recoverability-qualified commitment assessment.
//!
//! This crate checks a supplied scalar recovery-profile contract and candidate
//! reachable-prefix bounds. It does not verify physical certificates or reachability.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryProfile {
    pub profile_id: String,
    pub profile_sha256: String,
    pub plant_model_sha256: String,
    pub context_scope_sha256: String,
    pub recovery_controller_sha256: String,
    /// Digest of an independently produced recovery-certificate receipt.
    pub certificate_receipt_sha256: String,
    pub entry_radius_units: u64,
    /// Recovery recurrence: ceil(error * numerator / denominator) + disturbance.
    pub contraction_numerator: u64,
    pub contraction_denominator: u64,
    pub disturbance_per_step_units: u64,
    pub target_radius_units: u64,
    pub max_recovery_steps: u32,
    pub max_high_performance_dwell_steps: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviousOutcome {
    Reconciled,
    ConfirmedNotExecuted,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidatePrefix {
    /// Prefix zero is the currently observed state; subsequent values are predicted states.
    pub step: u32,
    pub predicted_deviation_units: u64,
    pub state_estimation_error_units: u64,
    pub transition_model_error_units: u64,
    pub disturbance_reachability_error_units: u64,
    /// Baseline forward-invariance margin. Negative means an invariance-only shield would stop.
    pub forward_invariance_margin_units: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentRequest {
    pub previous_outcome: PreviousOutcome,
    pub plant_model_sha256: String,
    pub context_scope_sha256: String,
    pub candidate_policy_sha256: String,
    pub action_segment_sha256: String,
    pub prefixes: Vec<CandidatePrefix>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitmentStopReason {
    RecoveryEnvelopeBoundary,
    MaximumDwellReached,
    PlannedSegmentEnded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitmentDisposition {
    BlockUntilReconciliation,
    NoCertifiedRecoveryFromCurrentState,
    SwitchToRecoveryNow,
    CommitPrefix {
        last_safe_commit_step: u32,
        switch_no_later_than_step: u32,
        stop_reason: CommitmentStopReason,
        reassess_before_any_further_commitment: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecoverabilityError {
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    InvalidContraction,
    DisturbanceExceedsEntryMargin,
    RecoveryDeadlineNotMet {
        final_bound_units: u64,
        target_radius_units: u64,
    },
    ZeroRecoveryHorizon,
    ZeroHighPerformanceDwell,
    ScopeMismatch(&'static str),
    EmptyPrefixSequence,
    NonContiguousPrefixSequence,
    ErrorBoundOverflow,
    RecoveryArithmeticOverflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentAssessment {
    pub disposition: CommitmentDisposition,
    pub recovery_bound_at_deadline_units: u64,
    pub recovery_entry_margin_units: u64,
    pub largest_committed_reachable_bound_units: u64,
    /// Last prefix admitted by a conventional forward-invariance-only shield.
    pub invariance_only_last_safe_step: u32,
    pub latest_recoverability_qualified_commit_step: u32,
    pub recovery_profile_id: String,
}

/// Assess a candidate commitment against a supplied recovery profile.
///
/// The profile's declared recurrence is checked exactly using integer arithmetic.
/// Prefix bounds are treated as externally produced upper bounds. Hashes are
/// syntax-checked; an independent trust boundary must authenticate all receipts.
pub fn assess_commitment(
    profile: &RecoveryProfile,
    request: &CommitmentRequest,
) -> Result<CommitmentAssessment, RecoverabilityError> {
    if request.previous_outcome == PreviousOutcome::Unknown {
        return Ok(CommitmentAssessment {
            disposition: CommitmentDisposition::BlockUntilReconciliation,
            recovery_bound_at_deadline_units: 0,
            recovery_entry_margin_units: 0,
            largest_committed_reachable_bound_units: 0,
            invariance_only_last_safe_step: 0,
            latest_recoverability_qualified_commit_step: 0,
            recovery_profile_id: profile.profile_id.clone(),
        });
    }

    let recovery_bound = validate_profile(profile)?;
    validate_request_scope(profile, request)?;

    if request.prefixes.is_empty() {
        return Err(RecoverabilityError::EmptyPrefixSequence);
    }
    for (index, prefix) in request.prefixes.iter().enumerate() {
        if prefix.step as usize != index {
            return Err(RecoverabilityError::NonContiguousPrefixSequence);
        }
    }

    let prefix_bounds = request
        .prefixes
        .iter()
        .map(total_prefix_bound)
        .collect::<Result<Vec<_>, _>>()?;

    let current_bound = prefix_bounds[0];
    if current_bound > profile.entry_radius_units {
        return Ok(CommitmentAssessment {
            disposition: CommitmentDisposition::NoCertifiedRecoveryFromCurrentState,
            recovery_bound_at_deadline_units: recovery_bound,
            recovery_entry_margin_units: 0,
            largest_committed_reachable_bound_units: current_bound,
            invariance_only_last_safe_step: invariance_only_last_safe_step(&request.prefixes),
            latest_recoverability_qualified_commit_step: 0,
            recovery_profile_id: profile.profile_id.clone(),
        });
    }

    let final_prefix_step = request.prefixes.last().expect("non-empty prefix sequence").step;
    let allowed_limit = final_prefix_step.min(profile.max_high_performance_dwell_steps);
    let first_boundary = prefix_bounds
        .iter()
        .enumerate()
        .take(allowed_limit as usize + 1)
        .find(|(_, bound)| **bound > profile.entry_radius_units)
        .map(|(index, _)| index as u32);

    let invariance_last_safe = invariance_only_last_safe_step(&request.prefixes);
    let (last_safe_step, stop_reason, reassess) = match first_boundary {
        Some(0) => {
            return Ok(CommitmentAssessment {
                disposition: CommitmentDisposition::NoCertifiedRecoveryFromCurrentState,
                recovery_bound_at_deadline_units: recovery_bound,
                recovery_entry_margin_units: 0,
                largest_committed_reachable_bound_units: current_bound,
                invariance_only_last_safe_step: invariance_last_safe,
                latest_recoverability_qualified_commit_step: 0,
                recovery_profile_id: profile.profile_id.clone(),
            });
        }
        Some(step) => (
            step - 1,
            CommitmentStopReason::RecoveryEnvelopeBoundary,
            false,
        ),
        None if profile.max_high_performance_dwell_steps <= final_prefix_step => (
            profile.max_high_performance_dwell_steps,
            CommitmentStopReason::MaximumDwellReached,
            false,
        ),
        None => (
            final_prefix_step,
            CommitmentStopReason::PlannedSegmentEnded,
            true,
        ),
    };

    let largest = prefix_bounds
        .iter()
        .take(last_safe_step as usize + 1)
        .copied()
        .max()
        .unwrap_or(current_bound);
    let margin = profile.entry_radius_units - largest;
    let disposition = if last_safe_step == 0
        && stop_reason == CommitmentStopReason::RecoveryEnvelopeBoundary
    {
        CommitmentDisposition::SwitchToRecoveryNow
    } else {
        CommitmentDisposition::CommitPrefix {
            last_safe_commit_step: last_safe_step,
            switch_no_later_than_step: if stop_reason == CommitmentStopReason::RecoveryEnvelopeBoundary {
                last_safe_step
            } else {
                profile.max_high_performance_dwell_steps
            },
            stop_reason,
            reassess_before_any_further_commitment: reassess,
        }
    };

    Ok(CommitmentAssessment {
        disposition,
        recovery_bound_at_deadline_units: recovery_bound,
        recovery_entry_margin_units: margin,
        largest_committed_reachable_bound_units: largest,
        invariance_only_last_safe_step: invariance_last_safe,
        latest_recoverability_qualified_commit_step: last_safe_step,
        recovery_profile_id: profile.profile_id.clone(),
    })
}

fn validate_profile(profile: &RecoveryProfile) -> Result<u64, RecoverabilityError> {
    for (value, name) in [
        (&profile.profile_id, "profile_id"),
        (&profile.profile_sha256, "profile_sha256"),
        (&profile.plant_model_sha256, "plant_model_sha256"),
        (&profile.context_scope_sha256, "context_scope_sha256"),
        (&profile.recovery_controller_sha256, "recovery_controller_sha256"),
        (&profile.certificate_receipt_sha256, "certificate_receipt_sha256"),
    ] {
        if value.trim().is_empty() {
            return Err(RecoverabilityError::EmptyIdentity(name));
        }
    }
    for (name, digest) in [
        ("profile_sha256", &profile.profile_sha256),
        ("plant_model_sha256", &profile.plant_model_sha256),
        ("context_scope_sha256", &profile.context_scope_sha256),
        ("recovery_controller_sha256", &profile.recovery_controller_sha256),
        ("certificate_receipt_sha256", &profile.certificate_receipt_sha256),
    ] {
        if !is_sha256_hex(digest) {
            return Err(RecoverabilityError::InvalidSha256(name));
        }
    }
    if profile.contraction_denominator == 0
        || profile.contraction_numerator >= profile.contraction_denominator
        || profile.entry_radius_units == 0
        || profile.target_radius_units > profile.entry_radius_units
    {
        return Err(RecoverabilityError::InvalidContraction);
    }
    if profile.max_recovery_steps == 0 {
        return Err(RecoverabilityError::ZeroRecoveryHorizon);
    }
    if profile.max_high_performance_dwell_steps == 0 {
        return Err(RecoverabilityError::ZeroHighPerformanceDwell);
    }

    let first_bound = recovery_step_bound(profile, profile.entry_radius_units)?;
    if first_bound > profile.entry_radius_units {
        return Err(RecoverabilityError::DisturbanceExceedsEntryMargin);
    }

    let mut bound = profile.entry_radius_units;
    for _ in 0..profile.max_recovery_steps {
        bound = recovery_step_bound(profile, bound)?;
    }
    if bound > profile.target_radius_units {
        return Err(RecoverabilityError::RecoveryDeadlineNotMet {
            final_bound_units: bound,
            target_radius_units: profile.target_radius_units,
        });
    }
    Ok(bound)
}

fn recovery_step_bound(
    profile: &RecoveryProfile,
    current_bound: u64,
) -> Result<u64, RecoverabilityError> {
    let product = u128::from(current_bound)
        .checked_mul(u128::from(profile.contraction_numerator))
        .ok_or(RecoverabilityError::RecoveryArithmeticOverflow)?;
    let denominator = u128::from(profile.contraction_denominator);
    let quotient = product / denominator;
    let rounded = if product % denominator == 0 {
        quotient
    } else {
        quotient
            .checked_add(1)
            .ok_or(RecoverabilityError::RecoveryArithmeticOverflow)?
    };
    let contracted = u64::try_from(rounded)
        .map_err(|_| RecoverabilityError::RecoveryArithmeticOverflow)?;
    contracted
        .checked_add(profile.disturbance_per_step_units)
        .ok_or(RecoverabilityError::RecoveryArithmeticOverflow)
}

fn validate_request_scope(
    profile: &RecoveryProfile,
    request: &CommitmentRequest,
) -> Result<(), RecoverabilityError> {
    for (value, name) in [
        (&request.plant_model_sha256, "request.plant_model_sha256"),
        (&request.context_scope_sha256, "request.context_scope_sha256"),
        (&request.candidate_policy_sha256, "candidate_policy_sha256"),
        (&request.action_segment_sha256, "action_segment_sha256"),
    ] {
        if !is_sha256_hex(value) {
            return Err(RecoverabilityError::InvalidSha256(name));
        }
    }
    if request.plant_model_sha256 != profile.plant_model_sha256 {
        return Err(RecoverabilityError::ScopeMismatch("plant_model_sha256"));
    }
    if request.context_scope_sha256 != profile.context_scope_sha256 {
        return Err(RecoverabilityError::ScopeMismatch("context_scope_sha256"));
    }
    Ok(())
}

fn total_prefix_bound(prefix: &CandidatePrefix) -> Result<u64, RecoverabilityError> {
    [
        prefix.predicted_deviation_units,
        prefix.state_estimation_error_units,
        prefix.transition_model_error_units,
        prefix.disturbance_reachability_error_units,
    ]
    .into_iter()
    .try_fold(0u64, |sum, term| {
        sum.checked_add(term)
            .ok_or(RecoverabilityError::ErrorBoundOverflow)
    })
}

fn invariance_only_last_safe_step(prefixes: &[CandidatePrefix]) -> u32 {
    prefixes
        .iter()
        .find(|prefix| prefix.forward_invariance_margin_units < 0)
        .map(|prefix| prefix.step.saturating_sub(1))
        .unwrap_or_else(|| prefixes.last().map_or(0, |prefix| prefix.step))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String {
        c.to_string().repeat(64)
    }

    fn profile() -> RecoveryProfile {
        RecoveryProfile {
            profile_id: "recovery-profile-v1".into(),
            profile_sha256: digest('a'),
            plant_model_sha256: digest('b'),
            context_scope_sha256: digest('c'),
            recovery_controller_sha256: digest('d'),
            certificate_receipt_sha256: digest('e'),
            entry_radius_units: 10,
            contraction_numerator: 1,
            contraction_denominator: 2,
            disturbance_per_step_units: 1,
            target_radius_units: 3,
            max_recovery_steps: 3,
            max_high_performance_dwell_steps: 4,
        }
    }

    fn prefix(step: u32, bound: u64, invariance_margin: i64) -> CandidatePrefix {
        CandidatePrefix {
            step,
            predicted_deviation_units: bound,
            state_estimation_error_units: 0,
            transition_model_error_units: 0,
            disturbance_reachability_error_units: 0,
            forward_invariance_margin_units: invariance_margin,
        }
    }

    fn request(prefixes: Vec<CandidatePrefix>) -> CommitmentRequest {
        CommitmentRequest {
            previous_outcome: PreviousOutcome::Reconciled,
            plant_model_sha256: digest('b'),
            context_scope_sha256: digest('c'),
            candidate_policy_sha256: digest('f'),
            action_segment_sha256: digest('1'),
            prefixes,
        }
    }

    #[test]
    fn recoverability_allows_commitment_beyond_forward_invariance_boundary() {
        let p = profile();
        let r = request(vec![
            prefix(0, 4, 1),
            prefix(1, 6, -1),
            prefix(2, 8, -2),
            prefix(3, 9, -3),
        ]);
        let assessment = assess_commitment(&p, &r).unwrap();
        assert_eq!(
            assessment.disposition,
            CommitmentDisposition::CommitPrefix {
                last_safe_commit_step: 3,
                switch_no_later_than_step: 4,
                stop_reason: CommitmentStopReason::PlannedSegmentEnded,
                reassess_before_any_further_commitment: true,
            }
        );
        assert_eq!(assessment.invariance_only_last_safe_step, 0);
        assert_eq!(assessment.latest_recoverability_qualified_commit_step, 3);
        assert_eq!(assessment.recovery_bound_at_deadline_units, 3);
        assert_eq!(assessment.recovery_entry_margin_units, 1);
    }

    #[test]
    fn first_prefix_outside_recovery_envelope_forces_earlier_switch() {
        let p = profile();
        let r = request(vec![
            prefix(0, 4, 1),
            prefix(1, 7, -1),
            prefix(2, 9, -2),
            prefix(3, 11, -3),
        ]);
        let assessment = assess_commitment(&p, &r).unwrap();
        assert_eq!(
            assessment.disposition,
            CommitmentDisposition::CommitPrefix {
                last_safe_commit_step: 2,
                switch_no_later_than_step: 2,
                stop_reason: CommitmentStopReason::RecoveryEnvelopeBoundary,
                reassess_before_any_further_commitment: false,
            }
        );
        assert_eq!(assessment.largest_committed_reachable_bound_units, 9);
        assert_eq!(assessment.recovery_entry_margin_units, 1);
    }

    #[test]
    fn maximum_dwell_caps_commitment_even_when_recovery_margin_remains() {
        let mut p = profile();
        p.max_high_performance_dwell_steps = 2;
        let r = request(vec![
            prefix(0, 2, 1),
            prefix(1, 4, 1),
            prefix(2, 6, 1),
            prefix(3, 7, 1),
            prefix(4, 8, 1),
        ]);
        let assessment = assess_commitment(&p, &r).unwrap();
        assert_eq!(
            assessment.disposition,
            CommitmentDisposition::CommitPrefix {
                last_safe_commit_step: 2,
                switch_no_later_than_step: 2,
                stop_reason: CommitmentStopReason::MaximumDwellReached,
                reassess_before_any_further_commitment: false,
            }
        );
    }

    #[test]
    fn disturbance_larger_than_contraction_margin_rejects_profile() {
        let mut p = profile();
        p.contraction_numerator = 9;
        p.contraction_denominator = 10;
        p.disturbance_per_step_units = 2;
        assert_eq!(
            assess_commitment(&p, &request(vec![prefix(0, 2, 1)])),
            Err(RecoverabilityError::DisturbanceExceedsEntryMargin)
        );
    }

    #[test]
    fn recovery_must_reach_target_within_declared_deadline() {
        let mut p = profile();
        p.max_recovery_steps = 1;
        p.target_radius_units = 5;
        assert_eq!(
            assess_commitment(&p, &request(vec![prefix(0, 2, 1)])),
            Err(RecoverabilityError::RecoveryDeadlineNotMet {
                final_bound_units: 6,
                target_radius_units: 5,
            })
        );
    }

    #[test]
    fn plant_or_context_scope_mismatch_fails_closed() {
        let p = profile();
        let mut r = request(vec![prefix(0, 2, 1)]);
        r.context_scope_sha256 = digest('9');
        assert_eq!(
            assess_commitment(&p, &r),
            Err(RecoverabilityError::ScopeMismatch("context_scope_sha256"))
        );
    }

    #[test]
    fn unknown_previous_execution_blocks_new_commitment() {
        let p = profile();
        let mut r = request(vec![prefix(0, 2, 1), prefix(1, 3, 1)]);
        r.previous_outcome = PreviousOutcome::Unknown;
        let assessment = assess_commitment(&p, &r).unwrap();
        assert_eq!(
            assessment.disposition,
            CommitmentDisposition::BlockUntilReconciliation
        );
    }

    #[test]
    fn non_contiguous_prefixes_are_rejected() {
        let p = profile();
        let r = request(vec![prefix(0, 2, 1), prefix(2, 3, 1)]);
        assert_eq!(
            assess_commitment(&p, &r),
            Err(RecoverabilityError::NonContiguousPrefixSequence)
        );
    }

    #[test]
    fn current_state_outside_recovery_envelope_has_no_certified_recovery() {
        let p = profile();
        let r = request(vec![prefix(0, 11, 1), prefix(1, 9, 1)]);
        let assessment = assess_commitment(&p, &r).unwrap();
        assert_eq!(
            assessment.disposition,
            CommitmentDisposition::NoCertifiedRecoveryFromCurrentState
        );
    }

    #[test]
    fn cumulative_error_bound_overflow_is_rejected() {
        let p = profile();
        let r = request(vec![CandidatePrefix {
            step: 0,
            predicted_deviation_units: u64::MAX,
            state_estimation_error_units: 1,
            transition_model_error_units: 0,
            disturbance_reachability_error_units: 0,
            forward_invariance_margin_units: 1,
        }]);
        assert_eq!(
            assess_commitment(&p, &r),
            Err(RecoverabilityError::ErrorBoundOverflow)
        );
    }

    #[test]
    fn malformed_certificate_receipt_is_rejected() {
        let mut p = profile();
        p.certificate_receipt_sha256 = "unverified".into();
        assert_eq!(
            assess_commitment(&p, &request(vec![prefix(0, 2, 1)])),
            Err(RecoverabilityError::InvalidSha256("certificate_receipt_sha256"))
        );
    }
}
