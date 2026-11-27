#![forbid(unsafe_code)]

//! Finite recovery-option portfolio evaluator.
//!
//! Failure-domain independence and profile qualification are externally attested;
//! this crate validates bindings and counts distinct declared failure domains.

use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryOption {
    pub profile_id: String,
    pub profile_sha256: String,
    pub plant_model_sha256: String,
    pub context_scope_sha256: String,
    pub recovery_controller_sha256: String,
    pub certificate_receipt_sha256: String,
    /// Externally attested common-cause / failure-domain identity.
    pub failure_domain_sha256: String,
    pub entry_radius_units: u64,
    pub max_high_performance_dwell_steps: u32,
    pub recovery_steps: u32,
    pub recovery_energy_units: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviousOutcome {
    Reconciled,
    ConfirmedNotExecuted,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortfolioPrefix {
    pub step: u32,
    pub context_scope_sha256: String,
    /// Upper bound supplied by an independent reachable-set adapter.
    pub reachable_error_bound_units: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortfolioRequest {
    pub previous_outcome: PreviousOutcome,
    pub plant_model_sha256: String,
    pub candidate_policy_sha256: String,
    pub action_segment_sha256: String,
    pub prefixes: Vec<PortfolioPrefix>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortfolioStopReason {
    RecoveryOptionCountBoundary,
    MaximumDwellReached,
    PlannedSegmentEnded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedRecoveryOption {
    pub profile_id: String,
    pub failure_domain_sha256: String,
    pub recovery_steps: u32,
    pub recovery_energy_units: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortfolioDisposition {
    BlockUntilReconciliation,
    NoAdmissibleRecoveryPortfolio,
    SwitchToPortfolioNow,
    CommitPrefix {
        last_safe_commit_step: u32,
        switch_no_later_than_step: u32,
        stop_reason: PortfolioStopReason,
        reassess_before_any_further_commitment: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortfolioError {
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    DuplicateProfileId,
    ZeroRequiredOptions,
    ZeroEntryRadius,
    ZeroDwell,
    ZeroRecoverySteps,
    ScopeMismatch(&'static str),
    EmptyPrefixSequence,
    NonContiguousPrefixSequence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortfolioAssessment {
    pub disposition: PortfolioDisposition,
    pub required_independent_options: usize,
    pub minimum_independent_options_on_commitment: usize,
    /// Minimum option count minus one declared common-cause failure.
    pub minimum_options_after_one_domain_failure: usize,
    pub latest_commitment_step: u32,
    pub weakest_prefix_step: u32,
    pub selected_options_at_switch: Vec<SelectedRecoveryOption>,
    pub profile_count: usize,
    pub distinct_failure_domain_count: usize,
}

/// Scores whether the candidate retains a required number of distinct recovery
/// failure domains at every reachable prefix. Profiles in the same domain count once.
pub fn assess_recovery_portfolio(
    options: &[RecoveryOption],
    request: &PortfolioRequest,
    required_independent_options: usize,
) -> Result<PortfolioAssessment, PortfolioError> {
    if request.previous_outcome == PreviousOutcome::Unknown {
        return Ok(empty_assessment(
            PortfolioDisposition::BlockUntilReconciliation,
            required_independent_options,
            options.len(),
        ));
    }
    if required_independent_options == 0 {
        return Err(PortfolioError::ZeroRequiredOptions);
    }
    validate_request(request)?;
    validate_options(options, &request.plant_model_sha256)?;

    if request.prefixes.is_empty() {
        return Err(PortfolioError::EmptyPrefixSequence);
    }
    for (index, prefix) in request.prefixes.iter().enumerate() {
        if prefix.step as usize != index {
            return Err(PortfolioError::NonContiguousPrefixSequence);
        }
        if !is_sha256_hex(&prefix.context_scope_sha256) {
            return Err(PortfolioError::InvalidSha256("prefix.context_scope_sha256"));
        }
    }

    let mut per_prefix: Vec<Vec<SelectedRecoveryOption>> = Vec::new();
    let mut min_count = usize::MAX;
    let mut weakest_step = 0u32;
    let mut first_deficient_step = None;

    for prefix in &request.prefixes {
        let eligible = best_option_per_failure_domain(
            options,
            &request.plant_model_sha256,
            &prefix.context_scope_sha256,
            prefix.reachable_error_bound_units,
            prefix.step,
        );
        let count = eligible.len();
        if count < min_count {
            min_count = count;
            weakest_step = prefix.step;
        }
        if count < required_independent_options && first_deficient_step.is_none() {
            first_deficient_step = Some(prefix.step);
        }
        per_prefix.push(eligible);
    }

    let current_options = per_prefix.first().expect("non-empty prefixes");
    if current_options.len() < required_independent_options {
        return Ok(assessment_from_options(
            PortfolioDisposition::NoAdmissibleRecoveryPortfolio,
            required_independent_options,
            min_count,
            weakest_step,
            0,
            current_options,
            options,
        ));
    }

    let final_step = request.prefixes.last().expect("non-empty prefixes").step;
    let max_dwell = options
        .iter()
        .filter(|option| {
            option.plant_model_sha256 == request.plant_model_sha256
                && option.entry_radius_units >= 0
        })
        .map(|option| option.max_high_performance_dwell_steps)
        .max()
        .unwrap_or(0);
    let allowed_limit = final_step.min(max_dwell);

    let stop = match first_deficient_step {
        Some(0) => {
            return Ok(assessment_from_options(
                PortfolioDisposition::NoAdmissibleRecoveryPortfolio,
                required_independent_options,
                min_count,
                weakest_step,
                0,
                current_options,
                options,
            ));
        }
        Some(step) if step <= allowed_limit => {
            let last_safe = step - 1;
            let selected = &per_prefix[last_safe as usize];
            let disposition = if last_safe == 0 {
                PortfolioDisposition::SwitchToPortfolioNow
            } else {
                PortfolioDisposition::CommitPrefix {
                    last_safe_commit_step: last_safe,
                    switch_no_later_than_step: last_safe,
                    stop_reason: PortfolioStopReason::RecoveryOptionCountBoundary,
                    reassess_before_any_further_commitment: false,
                }
            };
            return Ok(assessment_from_options(
                disposition,
                required_independent_options,
                min_count,
                weakest_step,
                last_safe,
                selected,
                options,
            ));
        }
        _ => {}
    };

    let (last_safe, reason, reassess) = if max_dwell <= final_step {
        (
            max_dwell,
            PortfolioStopReason::MaximumDwellReached,
            false,
        )
    } else {
        (
            final_step,
            PortfolioStopReason::PlannedSegmentEnded,
            true,
        )
    };
    let disposition = PortfolioDisposition::CommitPrefix {
        last_safe_commit_step: last_safe,
        switch_no_later_than_step: max_dwell,
        stop_reason: reason,
        reassess_before_any_further_commitment: reassess,
    };
    Ok(assessment_from_options(
        disposition,
        required_independent_options,
        min_count,
        weakest_step,
        last_safe,
        &per_prefix[last_safe as usize],
        options,
    ))
}

fn empty_assessment(
    disposition: PortfolioDisposition,
    required: usize,
    profile_count: usize,
) -> PortfolioAssessment {
    PortfolioAssessment {
        disposition,
        required_independent_options: required,
        minimum_independent_options_on_commitment: 0,
        minimum_options_after_one_domain_failure: 0,
        latest_commitment_step: 0,
        weakest_prefix_step: 0,
        selected_options_at_switch: Vec::new(),
        profile_count,
        distinct_failure_domain_count: 0,
    }
}

fn assessment_from_options(
    disposition: PortfolioDisposition,
    required: usize,
    minimum: usize,
    weakest_step: u32,
    latest_commitment_step: u32,
    selected: &[SelectedRecoveryOption],
    options: &[RecoveryOption],
) -> PortfolioAssessment {
    let mut domains = HashSet::new();
    for option in options {
        domains.insert(option.failure_domain_sha256.clone());
    }
    PortfolioAssessment {
        disposition,
        required_independent_options: required,
        minimum_independent_options_on_commitment: minimum,
        minimum_options_after_one_domain_failure: minimum.saturating_sub(1),
        latest_commitment_step,
        weakest_prefix_step: weakest_step,
        selected_options_at_switch: selected.to_vec(),
        profile_count: options.len(),
        distinct_failure_domain_count: domains.len(),
    }
}

fn best_option_per_failure_domain(
    options: &[RecoveryOption],
    plant_model_sha256: &str,
    context_scope_sha256: &str,
    reachable_error_bound_units: u64,
    step: u32,
) -> Vec<SelectedRecoveryOption> {
    let mut best: BTreeMap<String, SelectedRecoveryOption> = BTreeMap::new();
    for option in options {
        if option.plant_model_sha256 != plant_model_sha256
            || option.context_scope_sha256 != context_scope_sha256
            || reachable_error_bound_units > option.entry_radius_units
            || step > option.max_high_performance_dwell_steps
        {
            continue;
        }
        let candidate = SelectedRecoveryOption {
            profile_id: option.profile_id.clone(),
            failure_domain_sha256: option.failure_domain_sha256.clone(),
            recovery_steps: option.recovery_steps,
            recovery_energy_units: option.recovery_energy_units,
        };
        let replace = best
            .get(&option.failure_domain_sha256)
            .is_none_or(|current| option_is_better(&candidate, current));
        if replace {
            best.insert(option.failure_domain_sha256.clone(), candidate);
        }
    }
    let mut values: Vec<_> = best.into_values().collect();
    values.sort_by(|a, b| {
        (a.recovery_steps, a.recovery_energy_units, &a.profile_id).cmp(&(
            b.recovery_steps,
            b.recovery_energy_units,
            &b.profile_id,
        ))
    });
    values
}

fn option_is_better(
    candidate: &SelectedRecoveryOption,
    current: &SelectedRecoveryOption,
) -> bool {
    (candidate.recovery_steps, candidate.recovery_energy_units, &candidate.profile_id)
        < (current.recovery_steps, current.recovery_energy_units, &current.profile_id)
}

fn validate_request(request: &PortfolioRequest) -> Result<(), PortfolioError> {
    for (value, name) in [
        (&request.plant_model_sha256, "request.plant_model_sha256"),
        (&request.candidate_policy_sha256, "candidate_policy_sha256"),
        (&request.action_segment_sha256, "action_segment_sha256"),
    ] {
        if !is_sha256_hex(value) {
            return Err(PortfolioError::InvalidSha256(name));
        }
    }
    Ok(())
}

fn validate_options(
    options: &[RecoveryOption],
    plant_model_sha256: &str,
) -> Result<(), PortfolioError> {
    if options.is_empty() {
        return Err(PortfolioError::EmptyIdentity("recovery_options"));
    }
    let mut profile_ids = HashSet::new();
    for option in options {
        for (value, name) in [
            (&option.profile_id, "profile_id"),
            (&option.profile_sha256, "profile_sha256"),
            (&option.plant_model_sha256, "plant_model_sha256"),
            (&option.context_scope_sha256, "context_scope_sha256"),
            (&option.recovery_controller_sha256, "recovery_controller_sha256"),
            (&option.certificate_receipt_sha256, "certificate_receipt_sha256"),
            (&option.failure_domain_sha256, "failure_domain_sha256"),
        ] {
            if value.trim().is_empty() {
                return Err(PortfolioError::EmptyIdentity(name));
            }
        }
        for (name, digest) in [
            ("profile_sha256", &option.profile_sha256),
            ("plant_model_sha256", &option.plant_model_sha256),
            ("context_scope_sha256", &option.context_scope_sha256),
            ("recovery_controller_sha256", &option.recovery_controller_sha256),
            ("certificate_receipt_sha256", &option.certificate_receipt_sha256),
            ("failure_domain_sha256", &option.failure_domain_sha256),
        ] {
            if !is_sha256_hex(digest) {
                return Err(PortfolioError::InvalidSha256(name));
            }
        }
        if option.plant_model_sha256 != plant_model_sha256 {
            return Err(PortfolioError::ScopeMismatch("plant_model_sha256"));
        }
        if option.entry_radius_units == 0 {
            return Err(PortfolioError::ZeroEntryRadius);
        }
        if option.max_high_performance_dwell_steps == 0 {
            return Err(PortfolioError::ZeroDwell);
        }
        if option.recovery_steps == 0 {
            return Err(PortfolioError::ZeroRecoverySteps);
        }
        if !profile_ids.insert(option.profile_id.clone()) {
            return Err(PortfolioError::DuplicateProfileId);
        }
    }
    Ok(())
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

    fn option(
        id: &str,
        failure_domain: char,
        context: char,
        entry_radius: u64,
        dwell: u32,
        recovery_steps: u32,
        energy: u64,
    ) -> RecoveryOption {
        RecoveryOption {
            profile_id: id.into(),
            profile_sha256: digest('a'),
            plant_model_sha256: digest('b'),
            context_scope_sha256: digest(context),
            recovery_controller_sha256: digest('c'),
            certificate_receipt_sha256: digest('d'),
            failure_domain_sha256: digest(failure_domain),
            entry_radius_units: entry_radius,
            max_high_performance_dwell_steps: dwell,
            recovery_steps,
            recovery_energy_units: energy,
        }
    }

    fn prefix(step: u32, context: char, bound: u64) -> PortfolioPrefix {
        PortfolioPrefix {
            step,
            context_scope_sha256: digest(context),
            reachable_error_bound_units: bound,
        }
    }

    fn request(prefixes: Vec<PortfolioPrefix>) -> PortfolioRequest {
        PortfolioRequest {
            previous_outcome: PreviousOutcome::Reconciled,
            plant_model_sha256: digest('b'),
            candidate_policy_sha256: digest('e'),
            action_segment_sha256: digest('f'),
            prefixes,
        }
    }

    #[test]
    fn one_profile_is_admissible_only_when_one_option_is_required() {
        let options = vec![option("r1", '1', 'c', 10, 4, 2, 3)];
        let r = request(vec![prefix(0, 'c', 5), prefix(1, 'c', 6)]);
        let one = assess_recovery_portfolio(&options, &r, 1).unwrap();
        assert_eq!(one.minimum_independent_options_on_commitment, 1);
        let two = assess_recovery_portfolio(&options, &r, 2).unwrap();
        assert_eq!(two.disposition, PortfolioDisposition::NoAdmissibleRecoveryPortfolio);
    }

    #[test]
    fn two_independent_failure_domains_survive_one_domain_failure() {
        let options = vec![
            option("fast", '1', 'c', 10, 4, 1, 5),
            option("slower-independent", '2', 'c', 10, 4, 3, 2),
        ];
        let r = request(vec![prefix(0, 'c', 4), prefix(1, 'c', 6)]);
        let assessment = assess_recovery_portfolio(&options, &r, 2).unwrap();
        assert_eq!(assessment.minimum_independent_options_on_commitment, 2);
        assert_eq!(assessment.minimum_options_after_one_domain_failure, 1);
        assert_eq!(assessment.selected_options_at_switch.len(), 2);
        assert_eq!(assessment.selected_options_at_switch[0].profile_id, "fast");
    }

    #[test]
    fn multiple_profiles_in_one_failure_domain_count_once() {
        let options = vec![
            option("controller-a", '1', 'c', 10, 4, 2, 2),
            option("controller-b", '1', 'c', 12, 4, 1, 1),
        ];
        let r = request(vec![prefix(0, 'c', 6)]);
        let assessment = assess_recovery_portfolio(&options, &r, 2).unwrap();
        assert_eq!(
            assessment.disposition,
            PortfolioDisposition::NoAdmissibleRecoveryPortfolio
        );
        assert_eq!(assessment.minimum_independent_options_on_commitment, 1);
    }

    #[test]
    fn context_mismatch_removes_an_option_without_averaging() {
        let options = vec![
            option("covers", '1', 'c', 10, 4, 2, 2),
            option("wrong-context", '2', 'x', 10, 4, 1, 1),
        ];
        let r = request(vec![prefix(0, 'c', 5)]);
        let assessment = assess_recovery_portfolio(&options, &r, 2).unwrap();
        assert_eq!(
            assessment.disposition,
            PortfolioDisposition::NoAdmissibleRecoveryPortfolio
        );
        assert_eq!(assessment.minimum_independent_options_on_commitment, 1);
    }

    #[test]
    fn option_loss_at_future_prefix_stops_before_redundancy_is_lost() {
        let options = vec![
            option("wide", '1', 'c', 10, 5, 2, 2),
            option("narrow", '2', 'c', 7, 5, 3, 3),
        ];
        let r = request(vec![
            prefix(0, 'c', 4),
            prefix(1, 'c', 6),
            prefix(2, 'c', 8),
        ]);
        let assessment = assess_recovery_portfolio(&options, &r, 2).unwrap();
        assert_eq!(
            assessment.disposition,
            PortfolioDisposition::CommitPrefix {
                last_safe_commit_step: 1,
                switch_no_later_than_step: 1,
                stop_reason: PortfolioStopReason::RecoveryOptionCountBoundary,
                reassess_before_any_further_commitment: false,
            }
        );
        assert_eq!(assessment.selected_options_at_switch.len(), 2);
    }

    #[test]
    fn common_cause_fast_profile_does_not_replace_independent_slow_option() {
        let options = vec![
            option("fast-a", '1', 'c', 10, 4, 1, 1),
            option("fast-b-same-domain", '1', 'c', 10, 4, 1, 1),
            option("slower-independent", '2', 'c', 10, 4, 3, 2),
        ];
        let r = request(vec![prefix(0, 'c', 5)]);
        let assessment = assess_recovery_portfolio(&options, &r, 2).unwrap();
        assert_eq!(assessment.selected_options_at_switch.len(), 2);
        assert_eq!(assessment.selected_options_at_switch[0].profile_id, "fast-a");
        assert_eq!(
            assessment.selected_options_at_switch[1].profile_id,
            "slower-independent"
        );
        assert_eq!(assessment.minimum_independent_options_on_commitment, 2);
    }

    #[test]
    fn losing_every_recovery_option_at_next_prefix_forces_switch_now() {
        let options = vec![
            option("r1", '1', 'c', 10, 5, 2, 2),
            option("r2", '2', 'c', 10, 5, 2, 2),
        ];
        let r = request(vec![prefix(0, 'c', 5), prefix(1, 'c', 11)]);
        let assessment = assess_recovery_portfolio(&options, &r, 2).unwrap();
        assert_eq!(
            assessment.disposition,
            PortfolioDisposition::SwitchToPortfolioNow
        );
        assert_eq!(assessment.latest_commitment_step, 0);
    }

    #[test]
    fn unknown_previous_outcome_blocks_portfolio_mutation() {
        let options = vec![
            option("r1", '1', 'c', 10, 5, 2, 2),
            option("r2", '2', 'c', 10, 5, 2, 2),
        ];
        let mut r = request(vec![prefix(0, 'c', 5)]);
        r.previous_outcome = PreviousOutcome::Unknown;
        let assessment = assess_recovery_portfolio(&options, &r, 2).unwrap();
        assert_eq!(
            assessment.disposition,
            PortfolioDisposition::BlockUntilReconciliation
        );
    }

    #[test]
    fn invalid_certificate_receipt_is_rejected() {
        let mut options = vec![
            option("r1", '1', 'c', 10, 5, 2, 2),
            option("r2", '2', 'c', 10, 5, 2, 2),
        ];
        options[0].certificate_receipt_sha256 = "missing".into();
        let r = request(vec![prefix(0, 'c', 5)]);
        assert_eq!(
            assess_recovery_portfolio(&options, &r, 2),
            Err(PortfolioError::InvalidSha256("certificate_receipt_sha256"))
        );
    }

    #[test]
    fn duplicate_profile_id_is_rejected() {
        let options = vec![
            option("same", '1', 'c', 10, 5, 2, 2),
            option("same", '2', 'c', 10, 5, 2, 2),
        ];
        let r = request(vec![prefix(0, 'c', 5)]);
        assert_eq!(
            assess_recovery_portfolio(&options, &r, 2),
            Err(PortfolioError::DuplicateProfileId)
        );
    }

    #[test]
    fn non_contiguous_prefix_sequence_is_rejected() {
        let options = vec![
            option("r1", '1', 'c', 10, 5, 2, 2),
            option("r2", '2', 'c', 10, 5, 2, 2),
        ];
        let r = request(vec![prefix(0, 'c', 5), prefix(2, 'c', 6)]);
        assert_eq!(
            assess_recovery_portfolio(&options, &r, 2),
            Err(PortfolioError::NonContiguousPrefixSequence)
        );
    }

    #[test]
    fn required_option_count_must_be_positive() {
        let options = vec![option("r1", '1', 'c', 10, 5, 2, 2)];
        let r = request(vec![prefix(0, 'c', 5)]);
        assert_eq!(
            assess_recovery_portfolio(&options, &r, 0),
            Err(PortfolioError::ZeroRequiredOptions)
        );
    }
}
