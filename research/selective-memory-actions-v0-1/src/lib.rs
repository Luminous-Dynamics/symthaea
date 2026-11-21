#![forbid(unsafe_code)]

// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Research-only selector for choosing memory-related information actions.
//!
//! This crate selects among caller-provided action/loss estimates. It does not
//! compute probabilities, calibrate a model, verify evidence, authorize tools,
//! write memory, or issue source receipts. The caller must derive
//! `AdmissionSnapshot` from the existing authority, source, consent, scope,
//! freshness, retraction and privacy owners. These booleans are not credentials.
//!
//! Keep ingestion decisions (durable commitment) separate from query-time
//! decisions (use, verify, clarify, abstain, probe). No production memory or
//! action path should consume this research crate without independent review.

/// Probability mass is represented in millionths to avoid floating-point
/// comparison and non-deterministic tie behavior.
pub const PROBABILITY_SCALE: u32 = 1_000_000;

/// A probability-weighted loss outcome supplied by a calibrated evaluator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutcomeLoss {
    /// Probability in millionths. A complete distribution sums to SCALE.
    pub probability_micros: u32,
    /// Loss in integer micro-units of the preregistered cost model.
    pub loss_micros: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    Ingestion,
    Query,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Action {
    Persist,
    SessionOnly,
    DiscardCandidate,
    RequestConfirmation,
    Use,
    Verify,
    Clarify,
    Abstain,
    Probe,
}

impl Action {
    pub const fn stage(self) -> Stage {
        match self {
            Self::Persist
            | Self::SessionOnly
            | Self::DiscardCandidate
            | Self::RequestConfirmation => Stage::Ingestion,
            Self::Use | Self::Verify | Self::Clarify | Self::Abstain | Self::Probe => {
                Stage::Query
            }
        }
    }

    const fn stable_code(self) -> u8 {
        match self {
            Self::Persist => 0,
            Self::SessionOnly => 1,
            Self::DiscardCandidate => 2,
            Self::RequestConfirmation => 3,
            Self::Use => 4,
            Self::Verify => 5,
            Self::Clarify => 6,
            Self::Abstain => 7,
            Self::Probe => 8,
        }
    }
}

/// Snapshot of already-computed admission facts from upstream owners.
///
/// This is deliberately not an authorization token or cryptographic receipt.
/// Integrations must not set these fields from model prose or untrusted
/// candidate output. A production integration must bind this snapshot to the
/// exact source, scope, time, consent and policy revisions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdmissionSnapshot {
    pub durable_write_authorized: bool,
    pub session_capture_authorized: bool,
    pub source_qualified: bool,
    pub scope_qualified: bool,
    pub freshness_qualified: bool,
    pub not_revoked_or_erased: bool,
    pub verification_authorized: bool,
    pub verifier_available: bool,
    pub clarification_authorized: bool,
    pub probe_authorized: bool,
}

/// One legal-stage action and its estimated loss distribution.
///
/// `operational_cost_micros` includes direct tool/storage/resource cost.
/// `delay_cost_micros` captures the cost of waiting for the action. Both are
/// added to probability-weighted outcome loss. The rank is a frozen,
/// versioned policy tie-break parameter; lower ranks win exact cost ties.
#[derive(Clone, Copy, Debug)]
pub struct ActionOption<'a> {
    pub action: Action,
    pub outcomes: &'a [OutcomeLoss],
    pub operational_cost_micros: u64,
    pub delay_cost_micros: u64,
    pub tie_break_rank: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolicyDecision {
    pub action: Action,
    pub expected_loss_micros: u64,
    pub tie_break_rank: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionError {
    EmptyOptions,
    WrongStage { action: Action, expected: Stage },
    DuplicateAction { action: Action },
    EmptyOutcomeDistribution { action: Action },
    ProbabilityMassMismatch { action: Action, total_micros: u64 },
    ArithmeticOverflow { action: Action },
    NoAdmissibleAction,
}

/// Select the minimum expected-loss action that passes the stage's hard gates.
///
/// Expected loss is calculated as:
///
/// `sum(probability * outcome_loss) + operational_cost + delay_cost`
///
/// Costs and probabilities use integer fixed-point units. This function does
/// not decide whether the supplied cost model or calibration is valid; those
/// artifacts must be frozen, independently reviewed and digested by caller.
pub fn select_action(
    stage: Stage,
    admission: AdmissionSnapshot,
    options: &[ActionOption<'_>],
) -> Result<PolicyDecision, SelectionError> {
    if options.is_empty() {
        return Err(SelectionError::EmptyOptions);
    }

    let mut seen = [false; 9];
    let mut best: Option<PolicyDecision> = None;

    for option in options {
        if option.action.stage() != stage {
            return Err(SelectionError::WrongStage {
                action: option.action,
                expected: stage,
            });
        }

        let code = usize::from(option.action.stable_code());
        if seen[code] {
            return Err(SelectionError::DuplicateAction {
                action: option.action,
            });
        }
        seen[code] = true;

        let expected_loss = expected_loss_micros(option)?;

        if !is_admissible(option.action, admission) {
            continue;
        }

        let candidate = PolicyDecision {
            action: option.action,
            expected_loss_micros: expected_loss,
            tie_break_rank: option.tie_break_rank,
        };

        let replace = match best {
            None => true,
            Some(current) => {
                (candidate.expected_loss_micros, candidate.tie_break_rank,
                 candidate.action.stable_code())
                    < (current.expected_loss_micros, current.tie_break_rank,
                       current.action.stable_code())
            }
        };

        if replace {
            best = Some(candidate);
        }
    }

    best.ok_or(SelectionError::NoAdmissibleAction)
}

/// Apply non-compensable admission constraints before expected-loss ranking.
pub const fn is_admissible(action: Action, gates: AdmissionSnapshot) -> bool {
    match action {
        Action::Persist => {
            gates.durable_write_authorized
                && gates.source_qualified
                && gates.scope_qualified
                && gates.not_revoked_or_erased
        }
        Action::SessionOnly => {
            gates.session_capture_authorized
                && gates.scope_qualified
                && gates.not_revoked_or_erased
        }
        Action::DiscardCandidate => true,
        Action::RequestConfirmation => gates.clarification_authorized,
        Action::Use => {
            gates.source_qualified
                && gates.scope_qualified
                && gates.freshness_qualified
                && gates.not_revoked_or_erased
        }
        Action::Verify => gates.verification_authorized && gates.verifier_available,
        Action::Clarify => gates.clarification_authorized,
        Action::Abstain => true,
        Action::Probe => gates.probe_authorized,
    }
}

fn expected_loss_micros(option: &ActionOption<'_>) -> Result<u64, SelectionError> {
    if option.outcomes.is_empty() {
        return Err(SelectionError::EmptyOutcomeDistribution {
            action: option.action,
        });
    }

    let mut probability_total: u64 = 0;
    let mut weighted_loss: u128 = 0;

    for outcome in option.outcomes {
        probability_total = probability_total
            .checked_add(u64::from(outcome.probability_micros))
            .ok_or(SelectionError::ArithmeticOverflow {
                action: option.action,
            })?;

        let weighted = u128::from(outcome.probability_micros)
            .checked_mul(u128::from(outcome.loss_micros))
            .ok_or(SelectionError::ArithmeticOverflow {
                action: option.action,
            })?;

        weighted_loss = weighted_loss
            .checked_add(weighted)
            .ok_or(SelectionError::ArithmeticOverflow {
                action: option.action,
            })?;
    }

    if probability_total != u64::from(PROBABILITY_SCALE) {
        return Err(SelectionError::ProbabilityMassMismatch {
            action: option.action,
            total_micros: probability_total,
        });
    }

    let rounded = weighted_loss
        .checked_add(u128::from(PROBABILITY_SCALE / 2))
        .ok_or(SelectionError::ArithmeticOverflow {
            action: option.action,
        })?
        / u128::from(PROBABILITY_SCALE);

    let expected = u64::try_from(rounded).map_err(|_| {
        SelectionError::ArithmeticOverflow {
            action: option.action,
        }
    })?;

    expected
        .checked_add(option.operational_cost_micros)
        .and_then(|value| value.checked_add(option.delay_cost_micros))
        .ok_or(SelectionError::ArithmeticOverflow {
            action: option.action,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOW_LOSS: [OutcomeLoss; 2] = [
        OutcomeLoss {
            probability_micros: 900_000,
            loss_micros: 0,
        },
        OutcomeLoss {
            probability_micros: 100_000,
            loss_micros: 10_000,
        },
    ];

    const HIGH_LOSS: [OutcomeLoss; 2] = [
        OutcomeLoss {
            probability_micros: 500_000,
            loss_micros: 10_000,
        },
        OutcomeLoss {
            probability_micros: 500_000,
            loss_micros: 20_000,
        },
    ];

    fn option(
        action: Action,
        outcomes: &'static [OutcomeLoss],
        operational_cost_micros: u64,
        delay_cost_micros: u64,
        tie_break_rank: u16,
    ) -> ActionOption<'static> {
        ActionOption {
            action,
            outcomes,
            operational_cost_micros,
            delay_cost_micros,
            tie_break_rank,
        }
    }

    #[test]
    fn picks_lower_expected_loss_among_admissible_actions() {
        let gates = AdmissionSnapshot {
            durable_write_authorized: true,
            session_capture_authorized: true,
            source_qualified: true,
            scope_qualified: true,
            not_revoked_or_erased: true,
            ..AdmissionSnapshot::default()
        };
        let options = [
            option(Action::Persist, &LOW_LOSS, 100, 100, 0),
            option(Action::SessionOnly, &HIGH_LOSS, 100, 100, 0),
        ];

        let decision = select_action(Stage::Ingestion, gates, &options).unwrap();
        assert_eq!(decision.action, Action::Persist);
        assert_eq!(decision.expected_loss_micros, 1_200);
    }

    #[test]
    fn hard_gate_rejects_persist_even_when_its_predicted_loss_is_zero() {
        let gates = AdmissionSnapshot {
            session_capture_authorized: true,
            scope_qualified: true,
            not_revoked_or_erased: true,
            ..AdmissionSnapshot::default()
        };
        let options = [
            option(
                Action::Persist,
                &LOW_LOSS,
                0,
                0,
                0,
            ),
            option(
                Action::SessionOnly,
                &HIGH_LOSS,
                0,
                0,
                0,
            ),
            option(
                Action::DiscardCandidate,
                &HIGH_LOSS,
                0,
                0,
                1,
            ),
        ];

        let decision = select_action(Stage::Ingestion, gates, &options).unwrap();
        assert_eq!(decision.action, Action::SessionOnly);
    }

    #[test]
    fn stale_memory_cannot_be_used_but_authorized_verification_can_be_selected() {
        let gates = AdmissionSnapshot {
            verification_authorized: true,
            verifier_available: true,
            source_qualified: true,
            scope_qualified: true,
            freshness_qualified: false,
            not_revoked_or_erased: true,
            ..AdmissionSnapshot::default()
        };
        let options = [
            option(Action::Use, &LOW_LOSS, 0, 0, 0),
            option(Action::Verify, &HIGH_LOSS, 50, 50, 0),
            option(Action::Abstain, &HIGH_LOSS, 0, 0, 10),
        ];

        let decision = select_action(Stage::Query, gates, &options).unwrap();
        assert_eq!(decision.action, Action::Verify);
    }

    #[test]
    fn unavailable_verifier_is_not_selected() {
        let gates = AdmissionSnapshot {
            verification_authorized: true,
            verifier_available: false,
            ..AdmissionSnapshot::default()
        };
        let options = [
            option(Action::Verify, &LOW_LOSS, 0, 0, 0),
            option(Action::Abstain, &HIGH_LOSS, 0, 0, 0),
        ];

        let decision = select_action(Stage::Query, gates, &options).unwrap();
        assert_eq!(decision.action, Action::Abstain);
    }

    #[test]
    fn clarification_requires_explicit_permission() {
        let gates = AdmissionSnapshot::default();
        let options = [
            option(Action::Clarify, &LOW_LOSS, 0, 0, 0),
            option(Action::Abstain, &HIGH_LOSS, 0, 0, 1),
        ];

        let decision = select_action(Stage::Query, gates, &options).unwrap();
        assert_eq!(decision.action, Action::Abstain);
    }

    #[test]
    fn exact_cost_ties_use_frozen_rank_then_stable_action_order() {
        let gates = AdmissionSnapshot {
            clarification_authorized: true,
            ..AdmissionSnapshot::default()
        };
        let options = [
            option(Action::Clarify, &LOW_LOSS, 0, 0, 5),
            option(Action::Abstain, &LOW_LOSS, 0, 0, 1),
        ];

        let decision = select_action(Stage::Query, gates, &options).unwrap();
        assert_eq!(decision.action, Action::Abstain);
    }

    #[test]
    fn rejects_wrong_stage_actions() {
        let options = [option(Action::Persist, &LOW_LOSS, 0, 0, 0)];
        let result = select_action(Stage::Query, AdmissionSnapshot::default(), &options);
        assert_eq!(
            result,
            Err(SelectionError::WrongStage {
                action: Action::Persist,
                expected: Stage::Query,
            })
        );
    }

    #[test]
    fn rejects_duplicate_action_options() {
        let options = [
            option(Action::Abstain, &LOW_LOSS, 0, 0, 0),
            option(Action::Abstain, &HIGH_LOSS, 0, 0, 1),
        ];
        assert_eq!(
            select_action(Stage::Query, AdmissionSnapshot::default(), &options),
            Err(SelectionError::DuplicateAction {
                action: Action::Abstain,
            })
        );
    }

    #[test]
    fn rejects_probability_mass_that_does_not_sum_to_one() {
        const INVALID: [OutcomeLoss; 1] = [OutcomeLoss {
            probability_micros: 999_999,
            loss_micros: 1,
        }];
        let options = [option(Action::Abstain, &INVALID, 0, 0, 0)];

        assert_eq!(
            select_action(Stage::Query, AdmissionSnapshot::default(), &options),
            Err(SelectionError::ProbabilityMassMismatch {
                action: Action::Abstain,
                total_micros: 999_999,
            })
        );
    }

    #[test]
    fn rejects_empty_loss_distribution() {
        let options = [option(Action::Abstain, &[], 0, 0, 0)];
        assert_eq!(
            select_action(Stage::Query, AdmissionSnapshot::default(), &options),
            Err(SelectionError::EmptyOutcomeDistribution {
                action: Action::Abstain,
            })
        );
    }

    #[test]
    fn rejects_arithmetic_overflow() {
        const HUGE: [OutcomeLoss; 1] = [OutcomeLoss {
            probability_micros: PROBABILITY_SCALE,
            loss_micros: u64::MAX,
        }];
        let options = [option(Action::Abstain, &HUGE, 1, 0, 0)];

        assert_eq!(
            select_action(Stage::Query, AdmissionSnapshot::default(), &options),
            Err(SelectionError::ArithmeticOverflow {
                action: Action::Abstain,
            })
        );
    }

    #[test]
    fn discard_and_abstain_are_always_legal_fallbacks() {
        assert!(is_admissible(
            Action::DiscardCandidate,
            AdmissionSnapshot::default()
        ));
        assert!(is_admissible(Action::Abstain, AdmissionSnapshot::default()));
    }
}
