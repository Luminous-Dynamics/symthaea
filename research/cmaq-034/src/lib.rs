#![forbid(unsafe_code)]

//! Fail-closed handoff from an anytime-monitor quarantine to a recovery portfolio.
//!
//! This adapter interprets the CMAQ-028 one-prefix assessment as an immediate
//! recovery selection only when the current state is reconciled. A nonzero
//! candidate continuation prefix is never permitted after calibration quarantine.
//! Receipt authentication and physical recovery qualification remain external.

use std::collections::{HashMap, HashSet};

use cmaq_028_recovery_portfolio::{
    PortfolioAssessment, PortfolioDisposition, PortfolioError, PortfolioRequest,
    PreviousOutcome as PortfolioPreviousOutcome, RecoveryOption, SelectedRecoveryOption,
    assess_recovery_portfolio,
};
use cmaq_031_anytime_calibration_monitor::MonitorUpdate;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuarantineBinding {
    pub calibration_profile_sha256: String,
    pub monitor_scope_sha256: String,
    /// The active candidate recovery/controller profile that must not be selected as fallback.
    pub active_recovery_profile_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CurrentStateReconciliation {
    pub outcome: PortfolioPreviousOutcome,
    pub current_state_sha256: String,
    pub current_state_receipt_sha256: String,
    pub plant_model_sha256: String,
    pub context_scope_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandoffDisposition {
    NoMonitorQuarantine,
    BlockUntilReconciliation,
    NoAdmissibleRecoveryPortfolio,
    InsufficientFailureDomainRedundancy {
        required_options: usize,
        options_after_one_domain_failure: usize,
    },
    RejectedNonImmediateAssessment,
    SwitchToRecoveryPortfolio {
        selected_options: Vec<SelectedRecoveryOption>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryHandoffAssessment {
    pub disposition: HandoffDisposition,
    pub calibration_profile_sha256: String,
    pub monitor_scope_sha256: String,
    pub current_state_sha256: Option<String>,
    pub current_state_receipt_sha256: Option<String>,
    pub portfolio_assessment: Option<PortfolioAssessment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandoffError {
    InvalidSha256(&'static str),
    Portfolio(PortfolioError),
    MissingSelectedProfile,
    SelectedProfileMismatch,
    ActiveRecoveryProfileReused,
    DuplicateSelectedProfile,
    DuplicateSelectedFailureDomain,
    ArithmeticOverflow,
}

/// Quarantine the active commitment and evaluate a zero-continuation recovery handoff.
///
/// This function never authorizes the quarantined candidate to continue. It requires:
/// - a CMAQ-031 Quarantined update;
/// - reconciled current execution state and a digest-bound state receipt;
/// - exactly one CMAQ-028 prefix at step zero;
/// - a recovery portfolio assessed by the actual CMAQ-028 evaluator;
/// - at least k+1 independent selected failure domains, so that after one
///   declared common-cause domain is lost at least k remain; and
/// - every selected recovery profile to differ from the quarantined active profile.
///
/// Digest syntax is checked here. A governed evidence-ingestion verifier must
/// authenticate all receipts and authorities before this function is called.
pub fn handoff_after_calibration_quarantine(
    monitor_update: &MonitorUpdate,
    quarantine: &QuarantineBinding,
    reconciliation: &CurrentStateReconciliation,
    options: &[RecoveryOption],
    portfolio_request: &PortfolioRequest,
    required_independent_options: usize,
) -> Result<RecoveryHandoffAssessment, HandoffError> {
    validate_sha256(
        &quarantine.calibration_profile_sha256,
        "calibration_profile_sha256",
    )?;
    validate_sha256(&quarantine.monitor_scope_sha256, "monitor_scope_sha256")?;
    validate_sha256(
        &quarantine.active_recovery_profile_sha256,
        "active_recovery_profile_sha256",
    )?;

    if !matches!(monitor_update, MonitorUpdate::Quarantined { .. }) {
        return Ok(RecoveryHandoffAssessment {
            disposition: HandoffDisposition::NoMonitorQuarantine,
            calibration_profile_sha256: quarantine.calibration_profile_sha256.clone(),
            monitor_scope_sha256: quarantine.monitor_scope_sha256.clone(),
            current_state_sha256: None,
            current_state_receipt_sha256: None,
            portfolio_assessment: None,
        });
    }

    if reconciliation.outcome != PortfolioPreviousOutcome::Reconciled
        || portfolio_request.previous_outcome != PortfolioPreviousOutcome::Reconciled
    {
        return Ok(RecoveryHandoffAssessment {
            disposition: HandoffDisposition::BlockUntilReconciliation,
            calibration_profile_sha256: quarantine.calibration_profile_sha256.clone(),
            monitor_scope_sha256: quarantine.monitor_scope_sha256.clone(),
            current_state_sha256: None,
            current_state_receipt_sha256: None,
            portfolio_assessment: None,
        });
    }

    validate_sha256(&reconciliation.current_state_sha256, "current_state_sha256")?;
    validate_sha256(
        &reconciliation.current_state_receipt_sha256,
        "current_state_receipt_sha256",
    )?;
    validate_sha256(&reconciliation.plant_model_sha256, "plant_model_sha256")?;
    validate_sha256(&reconciliation.context_scope_sha256, "context_scope_sha256")?;

    if portfolio_request.plant_model_sha256 != reconciliation.plant_model_sha256 {
        return Err(HandoffError::Portfolio(PortfolioError::ScopeMismatch(
            "reconciliation.plant_model_sha256",
        )));
    }
    if portfolio_request.prefixes.len() != 1
        || portfolio_request.prefixes[0].step != 0
        || portfolio_request.prefixes[0].context_scope_sha256 != reconciliation.context_scope_sha256
    {
        return Ok(RecoveryHandoffAssessment {
            disposition: HandoffDisposition::RejectedNonImmediateAssessment,
            calibration_profile_sha256: quarantine.calibration_profile_sha256.clone(),
            monitor_scope_sha256: quarantine.monitor_scope_sha256.clone(),
            current_state_sha256: Some(reconciliation.current_state_sha256.clone()),
            current_state_receipt_sha256: Some(reconciliation.current_state_receipt_sha256.clone()),
            portfolio_assessment: None,
        });
    }

    if required_independent_options == 0 {
        return Err(HandoffError::Portfolio(PortfolioError::ZeroRequiredOptions));
    }
    if options.is_empty() {
        return Ok(RecoveryHandoffAssessment {
            disposition: HandoffDisposition::NoAdmissibleRecoveryPortfolio,
            calibration_profile_sha256: quarantine.calibration_profile_sha256.clone(),
            monitor_scope_sha256: quarantine.monitor_scope_sha256.clone(),
            current_state_sha256: Some(reconciliation.current_state_sha256.clone()),
            current_state_receipt_sha256: Some(reconciliation.current_state_receipt_sha256.clone()),
            portfolio_assessment: None,
        });
    }

    let assessment =
        assess_recovery_portfolio(options, portfolio_request, required_independent_options)
            .map_err(HandoffError::Portfolio)?;

    if matches!(
        &assessment.disposition,
        PortfolioDisposition::NoAdmissibleRecoveryPortfolio
    ) {
        return Ok(RecoveryHandoffAssessment {
            disposition: HandoffDisposition::NoAdmissibleRecoveryPortfolio,
            calibration_profile_sha256: quarantine.calibration_profile_sha256.clone(),
            monitor_scope_sha256: quarantine.monitor_scope_sha256.clone(),
            current_state_sha256: Some(reconciliation.current_state_sha256.clone()),
            current_state_receipt_sha256: Some(reconciliation.current_state_receipt_sha256.clone()),
            portfolio_assessment: Some(assessment),
        });
    }

    let immediate = match &assessment.disposition {
        PortfolioDisposition::SwitchToPortfolioNow => true,
        PortfolioDisposition::CommitPrefix {
            last_safe_commit_step: 0,
            ..
        } => true,
        PortfolioDisposition::CommitPrefix { .. }
        | PortfolioDisposition::BlockUntilReconciliation
        | PortfolioDisposition::NoAdmissibleRecoveryPortfolio => false,
    };
    if !immediate || assessment.latest_commitment_step != 0 {
        return Ok(RecoveryHandoffAssessment {
            disposition: HandoffDisposition::RejectedNonImmediateAssessment,
            calibration_profile_sha256: quarantine.calibration_profile_sha256.clone(),
            monitor_scope_sha256: quarantine.monitor_scope_sha256.clone(),
            current_state_sha256: Some(reconciliation.current_state_sha256.clone()),
            current_state_receipt_sha256: Some(reconciliation.current_state_receipt_sha256.clone()),
            portfolio_assessment: Some(assessment),
        });
    }

    let required_after_failure = required_independent_options
        .checked_add(1)
        .ok_or(HandoffError::ArithmeticOverflow)?;
    if assessment.minimum_options_after_one_domain_failure < required_independent_options
        || assessment.selected_options_at_switch.len() < required_after_failure
    {
        return Ok(RecoveryHandoffAssessment {
            disposition: HandoffDisposition::InsufficientFailureDomainRedundancy {
                required_options: required_independent_options,
                options_after_one_domain_failure: assessment
                    .minimum_options_after_one_domain_failure,
            },
            calibration_profile_sha256: quarantine.calibration_profile_sha256.clone(),
            monitor_scope_sha256: quarantine.monitor_scope_sha256.clone(),
            current_state_sha256: Some(reconciliation.current_state_sha256.clone()),
            current_state_receipt_sha256: Some(reconciliation.current_state_receipt_sha256.clone()),
            portfolio_assessment: Some(assessment),
        });
    }

    validate_selected_options(
        &assessment.selected_options_at_switch,
        options,
        &quarantine.active_recovery_profile_sha256,
    )?;

    Ok(RecoveryHandoffAssessment {
        disposition: HandoffDisposition::SwitchToRecoveryPortfolio {
            selected_options: assessment.selected_options_at_switch.clone(),
        },
        calibration_profile_sha256: quarantine.calibration_profile_sha256.clone(),
        monitor_scope_sha256: quarantine.monitor_scope_sha256.clone(),
        current_state_sha256: Some(reconciliation.current_state_sha256.clone()),
        current_state_receipt_sha256: Some(reconciliation.current_state_receipt_sha256.clone()),
        portfolio_assessment: Some(assessment),
    })
}

fn validate_selected_options(
    selected: &[SelectedRecoveryOption],
    inventory: &[RecoveryOption],
    active_profile_sha256: &str,
) -> Result<(), HandoffError> {
    let by_id: HashMap<&str, &RecoveryOption> = inventory
        .iter()
        .map(|option| (option.profile_id.as_str(), option))
        .collect();
    let mut profile_ids = HashSet::new();
    let mut failure_domains = HashSet::new();

    for selection in selected {
        if !profile_ids.insert(selection.profile_id.as_str()) {
            return Err(HandoffError::DuplicateSelectedProfile);
        }
        if !failure_domains.insert(selection.failure_domain_sha256.as_str()) {
            return Err(HandoffError::DuplicateSelectedFailureDomain);
        }
        let option = by_id
            .get(selection.profile_id.as_str())
            .ok_or(HandoffError::MissingSelectedProfile)?;
        if option.profile_sha256 == active_profile_sha256 {
            return Err(HandoffError::ActiveRecoveryProfileReused);
        }
        if option.failure_domain_sha256 != selection.failure_domain_sha256
            || option.recovery_steps != selection.recovery_steps
            || option.recovery_energy_units != selection.recovery_energy_units
        {
            return Err(HandoffError::SelectedProfileMismatch);
        }
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &'static str) -> Result<(), HandoffError> {
    if is_sha256_hex(value) {
        Ok(())
    } else {
        Err(HandoffError::InvalidSha256(field))
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmaq_028_recovery_portfolio::PortfolioPrefix;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn quarantine() -> QuarantineBinding {
        QuarantineBinding {
            calibration_profile_sha256: digest('a'),
            monitor_scope_sha256: digest('b'),
            active_recovery_profile_sha256: digest('c'),
        }
    }

    fn monitor_alarm() -> MonitorUpdate {
        MonitorUpdate::Quarantined {
            rank_one_based: 4,
            log_mixture_wealth: 3.0,
            threshold_log_wealth: 2.9,
            components: Vec::new(),
        }
    }

    fn reconciliation() -> CurrentStateReconciliation {
        CurrentStateReconciliation {
            outcome: PortfolioPreviousOutcome::Reconciled,
            current_state_sha256: digest('d'),
            current_state_receipt_sha256: digest('e'),
            plant_model_sha256: digest('1'),
            context_scope_sha256: digest('2'),
        }
    }

    fn options(include_active_profile: bool, count: usize) -> Vec<RecoveryOption> {
        (0..count)
            .map(|index| RecoveryOption {
                profile_id: format!("recovery-{index}"),
                profile_sha256: if include_active_profile && index == 0 {
                    digest('c')
                } else {
                    digest(char::from(b'3' + index as u8))
                },
                plant_model_sha256: digest('1'),
                context_scope_sha256: digest('2'),
                recovery_controller_sha256: digest('4'),
                certificate_receipt_sha256: digest('5'),
                failure_domain_sha256: digest(char::from(b'a' + index as u8)),
                entry_radius_units: 100,
                max_high_performance_dwell_steps: 10,
                recovery_steps: 3 + index as u32,
                recovery_energy_units: 20 + index as u64,
            })
            .collect()
    }

    fn request(prefix_count: usize) -> PortfolioRequest {
        PortfolioRequest {
            previous_outcome: PortfolioPreviousOutcome::Reconciled,
            plant_model_sha256: digest('1'),
            candidate_policy_sha256: digest('6'),
            action_segment_sha256: digest('7'),
            prefixes: (0..prefix_count)
                .map(|step| PortfolioPrefix {
                    step: step as u32,
                    context_scope_sha256: digest('2'),
                    reachable_error_bound_units: 5,
                })
                .collect(),
        }
    }

    #[test]
    fn alarm_and_unreconciled_execution_block_without_portfolio_assessment() {
        let mut current = reconciliation();
        current.outcome = PortfolioPreviousOutcome::Unknown;
        let result = handoff_after_calibration_quarantine(
            &monitor_alarm(),
            &quarantine(),
            &current,
            &options(false, 3),
            &request(1),
            2,
        )
        .unwrap();
        assert_eq!(
            result.disposition,
            HandoffDisposition::BlockUntilReconciliation
        );
        assert!(result.portfolio_assessment.is_none());
    }

    #[test]
    fn reconciled_alarm_switches_to_step_zero_independent_portfolio() {
        let result = handoff_after_calibration_quarantine(
            &monitor_alarm(),
            &quarantine(),
            &reconciliation(),
            &options(false, 3),
            &request(1),
            2,
        )
        .unwrap();
        match result.disposition {
            HandoffDisposition::SwitchToRecoveryPortfolio { selected_options } => {
                assert_eq!(selected_options.len(), 3);
                assert_eq!(
                    selected_options
                        .iter()
                        .map(|option| option.failure_domain_sha256.clone())
                        .collect::<HashSet<_>>()
                        .len(),
                    3
                );
            }
            other => panic!("expected immediate recovery switch, got {other:?}"),
        }
        assert_eq!(
            result
                .portfolio_assessment
                .as_ref()
                .unwrap()
                .latest_commitment_step,
            0
        );
    }

    #[test]
    fn monitor_without_quarantine_does_not_switch_policy() {
        let accepted = MonitorUpdate::Accepted {
            rank_one_based: 2,
            log_mixture_wealth: 0.1,
            components: Vec::new(),
        };
        let result = handoff_after_calibration_quarantine(
            &accepted,
            &quarantine(),
            &reconciliation(),
            &options(false, 3),
            &request(1),
            2,
        )
        .unwrap();
        assert_eq!(result.disposition, HandoffDisposition::NoMonitorQuarantine);
        assert!(result.portfolio_assessment.is_none());
    }

    #[test]
    fn active_quarantined_controller_cannot_be_selected_as_recovery() {
        let result = handoff_after_calibration_quarantine(
            &monitor_alarm(),
            &quarantine(),
            &reconciliation(),
            &options(true, 3),
            &request(1),
            2,
        );
        assert_eq!(result, Err(HandoffError::ActiveRecoveryProfileReused));
    }

    #[test]
    fn insufficient_domains_after_one_failure_do_not_switch() {
        let result = handoff_after_calibration_quarantine(
            &monitor_alarm(),
            &quarantine(),
            &reconciliation(),
            &options(false, 2),
            &request(1),
            2,
        )
        .unwrap();
        assert_eq!(
            result.disposition,
            HandoffDisposition::InsufficientFailureDomainRedundancy {
                required_options: 2,
                options_after_one_domain_failure: 1,
            }
        );
    }

    #[test]
    fn multiple_prefixes_are_not_allowed_after_quarantine() {
        let result = handoff_after_calibration_quarantine(
            &monitor_alarm(),
            &quarantine(),
            &reconciliation(),
            &options(false, 3),
            &request(2),
            2,
        )
        .unwrap();
        assert_eq!(
            result.disposition,
            HandoffDisposition::RejectedNonImmediateAssessment
        );
        assert!(result.portfolio_assessment.is_none());
    }

    #[test]
    fn current_state_scope_must_match_portfolio_prefix() {
        let mut current = reconciliation();
        current.context_scope_sha256 = digest('f');
        let result = handoff_after_calibration_quarantine(
            &monitor_alarm(),
            &quarantine(),
            &current,
            &options(false, 3),
            &request(1),
            2,
        )
        .unwrap();
        assert_eq!(
            result.disposition,
            HandoffDisposition::RejectedNonImmediateAssessment
        );
    }

    #[test]
    fn reachable_state_outside_all_recovery_entry_envelopes_blocks_switch() {
        let mut portfolio = request(1);
        portfolio.prefixes[0].reachable_error_bound_units = 101;
        let result = handoff_after_calibration_quarantine(
            &monitor_alarm(),
            &quarantine(),
            &reconciliation(),
            &options(false, 3),
            &portfolio,
            2,
        )
        .unwrap();
        assert_eq!(
            result.disposition,
            HandoffDisposition::NoAdmissibleRecoveryPortfolio
        );
    }
}
