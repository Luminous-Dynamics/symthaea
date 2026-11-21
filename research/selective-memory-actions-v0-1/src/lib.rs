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

/// Use-time freshness assessment derived by the canonical time/evidence
/// owners (currently #9954/#12401/#13091).
///
/// These are durations and deadline slack already computed in one qualified
/// clock contract. This crate does not read a wall clock, compare raw clock
/// ticks, choose valid_at/known_at, or infer freshness from retrieval time.
/// Every required value is optional so an unknown value fails closed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FreshnessAtUse {
    /// Age of evidence when first retrieved; diagnostic only, not the use gate.
    pub evidence_age_at_retrieval_ns: Option<u64>,
    /// Age of the evidence actually supporting this decision at consumption.
    pub evidence_age_at_use_ns: Option<u64>,
    /// Frozen task-specific maximum evidence age, inclusive.
    pub max_evidence_age_ns: Option<u64>,
    /// Whether the frozen task profile declares an enforced deadline.
    /// None means the deadline policy itself is unknown.
    pub deadline_enforced: Option<bool>,
    /// Signed time remaining at consumption. Zero is exactly on deadline;
    /// negative means late. Required only when deadline_enforced is Some(true).
    pub deadline_slack_at_use_ns: Option<i64>,
}

impl FreshnessAtUse {
    /// Freshness is qualified only when age, age budget and deadline policy
    /// are known. An enforced deadline requires known nonnegative slack; a
    /// no-deadline task must explicitly declare false and supply no slack.
    pub const fn is_qualified(self) -> bool {
        match (
            self.evidence_age_at_use_ns,
            self.max_evidence_age_ns,
            self.deadline_enforced,
            self.deadline_slack_at_use_ns,
        ) {
            (Some(age), Some(max_age), Some(false), None) => age <= max_age,
            (Some(age), Some(max_age), Some(true), Some(deadline_slack)) => {
                age <= max_age && deadline_slack >= 0
            }
            _ => false,
        }
    }
}

/// Snapshot of already-computed admission facts from upstream owners.
///
/// This is deliberately not an authorization token or cryptographic receipt.
/// Integrations must not set these fields from model prose or untrusted
/// candidate output. A production integration must bind this snapshot to the
/// exact source, scope, time, consent and policy revisions. Source trust,
/// scope, lifecycle and revocation facts must be rechecked against the
/// canonical record at consumption, not reused from retrieval. Freshness is
/// evaluated at consumption, not merely at retrieval.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdmissionSnapshot {
    pub durable_write_authorized: bool,
    pub session_capture_authorized: bool,
    pub source_qualified: bool,
    pub scope_qualified: bool,
    pub freshness_at_use: FreshnessAtUse,
    /// Lifecycle state as revalidated against the canonical source at use.
    pub not_revoked_or_erased: bool,
    /// True only when a current source/revision receipt was checked at consumption.
    pub canonical_revalidation_at_use: bool,
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
    /// Result of the upstream component-wise hard resource-budget check
    /// (e.g. #9666). This is not a scalarized resource price.
    pub resource_feasible: bool,
    /// Explicit priced cost only; hard resource caps must be enforced above.
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

        // Hard resource feasibility is checked independently of expected loss.
        // A low scalar cost can never compensate for exceeding a resource cap.
        if !option.resource_feasible || !is_admissible(option.action, admission) {
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
                && gates.freshness_at_use.is_qualified()
                && gates.not_revoked_or_erased
                && gates.canonical_revalidation_at_use
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
            resource_feasible: true,
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
    fn hard_resource_infeasibility_overrides_the_lowest_expected_loss() {
        let options = [
            ActionOption {
                action: Action::Use,
                outcomes: &LOW_LOSS,
                resource_feasible: false,
                operational_cost_micros: 0,
                delay_cost_micros: 0,
                tie_break_rank: 0,
            },
            option(Action::Abstain, &HIGH_LOSS, 0, 0, 1),
        ];
        let decision = select_action(Stage::Query, AdmissionSnapshot::default(), &options)
            .unwrap();
        assert_eq!(decision.action, Action::Abstain);
    }

    #[test]
    fn stale_memory_cannot_be_used_but_authorized_verification_can_be_selected() {
        let gates = AdmissionSnapshot {
            verification_authorized: true,
            verifier_available: true,
            source_qualified: true,
            scope_qualified: true,
            freshness_at_use: FreshnessAtUse {
                evidence_age_at_retrieval_ns: None,
                evidence_age_at_use_ns: Some(101),
                max_evidence_age_ns: Some(100),
                deadline_slack_at_use_ns: Some(1_000),
            },
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
    fn use_requires_current_canonical_revalidation_after_retrieval() {
        let freshness = FreshnessAtUse {
            evidence_age_at_retrieval_ns: Some(10),
            evidence_age_at_use_ns: Some(20),
            max_evidence_age_ns: Some(100),
            deadline_enforced: Some(true),
            deadline_slack_at_use_ns: Some(500),
        };
        let gates = AdmissionSnapshot {
            source_qualified: true,
            scope_qualified: true,
            freshness_at_use: freshness,
            not_revoked_or_erased: true,
            // A cached retrieval-time "not retracted" value is not enough.
            canonical_revalidation_at_use: false,
            ..AdmissionSnapshot::default()
        };
        let options = [
            option(Action::Use, &LOW_LOSS, 0, 0, 0),
            option(Action::Verify, &HIGH_LOSS, 0, 0, 1),
            option(Action::Abstain, &HIGH_LOSS, 0, 0, 2),
        ];
        let decision = select_action(Stage::Query, gates, &options).unwrap();
        assert_eq!(decision.action, Action::Verify);
    }

    #[test]
    fn late_correction_blocks_use_even_when_memory_age_is_fresh() {
        let gates = AdmissionSnapshot {
            verification_authorized: true,
            verifier_available: true,
            source_qualified: true,
            scope_qualified: true,
            freshness_at_use: FreshnessAtUse {
                evidence_age_at_retrieval_ns: Some(10),
                evidence_age_at_use_ns: Some(20),
                max_evidence_age_ns: Some(100),
                deadline_slack_at_use_ns: Some(500),
            },
            // The current bitemporal ledger reports a correction/retraction.
            not_revoked_or_erased: false,
            canonical_revalidation_at_use: true,
            ..AdmissionSnapshot::default()
        };
        let options = [
            option(Action::Use, &LOW_LOSS, 0, 0, 0),
            option(Action::Verify, &HIGH_LOSS, 0, 0, 1),
            option(Action::Abstain, &HIGH_LOSS, 0, 0, 2),
        ];
        let decision = select_action(Stage::Query, gates, &options).unwrap();
        assert_eq!(decision.action, Action::Verify);
    }

    #[test]
    fn use_is_available_after_current_source_receipt_revalidation() {
        let gates = AdmissionSnapshot {
            source_qualified: true,
            scope_qualified: true,
            freshness_at_use: FreshnessAtUse {
                evidence_age_at_retrieval_ns: Some(10),
                evidence_age_at_use_ns: Some(20),
                max_evidence_age_ns: Some(100),
                deadline_slack_at_use_ns: Some(500),
            },
            not_revoked_or_erased: true,
            canonical_revalidation_at_use: true,
            ..AdmissionSnapshot::default()
        };
        let options = [
            option(Action::Use, &LOW_LOSS, 0, 0, 0),
            option(Action::Abstain, &HIGH_LOSS, 0, 0, 1),
        ];
        let decision = select_action(Stage::Query, gates, &options).unwrap();
        assert_eq!(decision.action, Action::Use);
    }

    #[test]
    fn evidence_fresh_at_retrieval_but_stale_at_use_is_rejected() {
        // A retrieval at age 20ns can become age 120ns before consumption.
        // Only the use-time age is supplied to the decision gate.
        let fresh_at_retrieval = FreshnessAtUse {
            evidence_age_at_retrieval_ns: Some(20),
            evidence_age_at_use_ns: Some(120),
            max_evidence_age_ns: Some(100),
            deadline_slack_at_use_ns: Some(500),
        };
        assert!(!fresh_at_retrieval.is_qualified());

        let gates = AdmissionSnapshot {
            source_qualified: true,
            scope_qualified: true,
            not_revoked_or_erased: true,
            freshness_at_use: fresh_at_retrieval,
            ..AdmissionSnapshot::default()
        };
        let options = [
            option(Action::Use, &LOW_LOSS, 0, 0, 0),
            option(Action::Abstain, &HIGH_LOSS, 0, 0, 1),
        ];
        let decision = select_action(Stage::Query, gates, &options).unwrap();
        assert_eq!(decision.action, Action::Abstain);
    }

    #[test]
    fn unknown_freshness_fails_closed_for_use() {
        let gates = AdmissionSnapshot {
            source_qualified: true,
            scope_qualified: true,
            not_revoked_or_erased: true,
            freshness_at_use: FreshnessAtUse {
                evidence_age_at_retrieval_ns: None,
                evidence_age_at_use_ns: Some(10),
                max_evidence_age_ns: Some(100),
                deadline_slack_at_use_ns: None,
            },
            ..AdmissionSnapshot::default()
        };
        let options = [
            option(Action::Use, &LOW_LOSS, 0, 0, 0),
            option(Action::Abstain, &HIGH_LOSS, 0, 0, 1),
        ];
        let decision = select_action(Stage::Query, gates, &options).unwrap();
        assert_eq!(decision.action, Action::Abstain);
    }

    #[test]
    fn late_decision_is_not_fresh_even_when_evidence_age_is_within_budget() {
        let freshness = FreshnessAtUse {
            evidence_age_at_retrieval_ns: None,
            evidence_age_at_use_ns: Some(50),
            max_evidence_age_ns: Some(100),
            deadline_enforced: Some(true),
            deadline_slack_at_use_ns: Some(-1),
        };
        assert!(!freshness.is_qualified());
    }

    #[test]
    fn explicit_no_deadline_profile_is_distinct_from_unknown_deadline() {
        let no_deadline = FreshnessAtUse {
            evidence_age_at_retrieval_ns: Some(10),
            evidence_age_at_use_ns: Some(20),
            max_evidence_age_ns: Some(100),
            deadline_enforced: Some(false),
            deadline_slack_at_use_ns: None,
        };
        assert!(no_deadline.is_qualified());

        let unknown_deadline = FreshnessAtUse {
            evidence_age_at_retrieval_ns: Some(10),
            evidence_age_at_use_ns: Some(20),
            max_evidence_age_ns: Some(100),
            deadline_enforced: None,
            deadline_slack_at_use_ns: None,
        };
        assert!(!unknown_deadline.is_qualified());
    }

    #[test]
    fn evidence_at_inclusive_age_limit_and_exact_deadline_is_qualified() {
        let freshness = FreshnessAtUse {
            evidence_age_at_retrieval_ns: None,
            evidence_age_at_use_ns: Some(100),
            max_evidence_age_ns: Some(100),
            deadline_enforced: Some(true),
            deadline_slack_at_use_ns: Some(0),
        };
        assert!(freshness.is_qualified());
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
