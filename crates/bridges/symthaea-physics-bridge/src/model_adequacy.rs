// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Model-adequacy decision semantics for sequential scientific discovery.
//!
//! This module deliberately separates policy-visible evidence from evaluator-only truth.
//! A surviving candidate is not accepted until validation succeeds and a calibrated adequacy
//! assessment on independent audit data is clear. An empty candidate set is not automatically
//! interpreted as proof of misspecification: it may also indicate noise, a bad likelihood,
//! invalid predictions, or an implementation defect.

/// Kind of model-criticism signal that triggered an adequacy alarm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdequacyDiagnostic {
    /// Fresh predictive residuals are inconsistent with the declared observation model.
    PredictiveMisfit,
    /// The selected experiments leave part of the declared operating region unrepresented.
    CoverageGap,
    /// Predictive intervals do not achieve their predeclared calibration target.
    CalibrationFailure,
    /// The queried distribution differs from the declared target distribution.
    QueryDistributionShift,
}

/// Policy-visible adequacy assessment.
///
/// The calibrated flag means the diagnostic threshold was fixed using designated calibration
/// worlds and its false-alarm behavior was checked on clean validation worlds. The
/// independent_audit flag means the assessment used observations not used to fit or tune the
/// candidate under review.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdequacyAssessment {
    /// No adequacy assessment has been completed.
    NotEvaluated,
    /// An adequacy alarm was raised but the evidence is not yet calibrated and independent.
    Alarm {
        diagnostic: AdequacyDiagnostic,
        calibrated: bool,
        independent_audit: bool,
    },
    /// No alarm was observed, with explicit evidence-quality flags.
    Clear {
        calibrated: bool,
        independent_audit: bool,
    },
}

impl AdequacyAssessment {
    fn is_trustworthy_alarm(self) -> bool {
        matches!(
            self,
            Self::Alarm {
                calibrated: true,
                independent_audit: true,
                ..
            }
        )
    }

    fn needs_independent_assessment(self) -> bool {
        match self {
            Self::NotEvaluated => true,
            Self::Alarm {
                calibrated,
                independent_audit,
                ..
            }
            | Self::Clear {
                calibrated,
                independent_audit,
            } => !calibrated || !independent_audit,
        }
    }

    fn is_trustworthy_clear(self) -> bool {
        matches!(
            self,
            Self::Clear {
                calibrated: true,
                independent_audit: true
            }
        )
    }
}

/// Status of the independent validation budget for a singleton candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationStatus {
    /// Validation has not started.
    NotStarted,
    /// Validation has begun but its predeclared criteria are not yet met.
    InProgress,
    /// The candidate passed the predeclared validation criterion.
    Passed,
    /// Fresh validation data contradicted this candidate.
    Contradicted,
    /// The validation budget ended before a decision was supported.
    BudgetExhausted,
}

/// Why the current discovery step is terminating or continuing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationStatus {
    /// Further admissible measurements or validation work remain.
    Active,
    /// The predeclared experiment/query budget has been consumed.
    BudgetExhausted,
    /// The admissible candidate-experiment pool has no remaining entries.
    CandidatePoolExhausted,
    /// No candidate experiment satisfies the declared operational constraints.
    NoAdmissibleExperiment,
    /// A numerical failure prevents interpreting the run.
    NumericalFailure,
    /// A configuration or protocol invariant failed.
    InvalidConfiguration,
}

impl TerminationStatus {
    fn is_invalid(self) -> bool {
        matches!(self, Self::NumericalFailure | Self::InvalidConfiguration)
    }

    fn can_continue(self) -> bool {
        self == Self::Active
    }
}

/// A snapshot of policy-visible state.
///
/// It intentionally contains no hidden ground-truth law, discrepancy amplitude, or evaluator
/// outcome. Those belong in a separate benchmark receipt and must never affect this decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiscoverySnapshot {
    pub live_candidate_count: usize,
    pub adequacy: AdequacyAssessment,
    pub validation: ValidationStatus,
    pub termination: TerminationStatus,
}

/// Next action/outcome selected from the observable discovery state.
///
/// AcceptValidatedCandidate means the candidate passed this protocol's gates. It does not
/// prove that the candidate is the true law; structural recovery is scored independently by
/// the benchmark evaluator on held-out truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscoveryAction {
    /// Continue distinguishing among two or more plausible candidates.
    ContinueDiscrimination,
    /// Keep a singleton provisional and validate it using fresh observations.
    ValidateProvisionalSingleton,
    /// Collect a calibrated adequacy assessment on independent audit observations.
    RunIndependentAdequacyAudit,
    /// Reopen candidate search or generate a declared alternative model family.
    ReopenCandidateSearch,
    /// A calibrated, independent diagnostic supports declaring the candidate space inadequate.
    ReportModelSpaceInadequate,
    /// Accept the candidate as validated under the declared protocol, not as infallibly true.
    AcceptValidatedCandidate,
    /// The budget ended before the evidence supported a resolution.
    ReportUnresolved,
    /// No candidates remain, but no trustworthy adequacy evidence explains why.
    ReportEmptySetWithoutAdequacyEvidence,
    /// A numerical or protocol defect invalidates this run.
    FailClosed,
}

/// Decide the next safe action using only policy-visible state.
///
/// Decision order is intentional:
/// 1. Invalid runs fail closed.
/// 2. Model-space inadequacy requires both a calibrated alarm and independent audit data.
/// 3. An untrusted alarm requests an independent adequacy audit rather than asserting failure.
/// 4. Empty candidate sets without trustworthy evidence remain explicitly unexplained.
/// 5. A singleton is accepted only after both independent adequacy assessment and candidate
///    validation succeed.
///
/// The function is pure and deterministic, which makes its semantics easy to unit test and
/// prevents evaluator-only hidden truth from leaking into selection behavior.
pub fn next_discovery_action(snapshot: DiscoverySnapshot) -> DiscoveryAction {
    if snapshot.termination.is_invalid() {
        return DiscoveryAction::FailClosed;
    }

    if snapshot.adequacy.is_trustworthy_alarm() {
        return DiscoveryAction::ReportModelSpaceInadequate;
    }

    if matches!(snapshot.adequacy, AdequacyAssessment::Alarm { .. }) {
        return if snapshot.termination.can_continue() {
            DiscoveryAction::RunIndependentAdequacyAudit
        } else if snapshot.live_candidate_count == 0 {
            DiscoveryAction::ReportEmptySetWithoutAdequacyEvidence
        } else {
            DiscoveryAction::ReportUnresolved
        };
    }

    if snapshot.live_candidate_count == 0 {
        return if snapshot.termination.can_continue() {
            // An empty candidate set is not itself a diagnosis. When measurements/model
            // expansion remain available, reopen search without asserting misspecification.
            DiscoveryAction::ReopenCandidateSearch
        } else {
            DiscoveryAction::ReportEmptySetWithoutAdequacyEvidence
        };
    }

    if snapshot.live_candidate_count > 1 {
        return if snapshot.termination.can_continue() {
            DiscoveryAction::ContinueDiscrimination
        } else {
            DiscoveryAction::ReportUnresolved
        };
    }

    // Exactly one candidate remains. A contradictory validation result takes precedence over
    // another acceptance attempt: return to search if possible, otherwise stop unresolved.
    if snapshot.validation == ValidationStatus::Contradicted {
        return if snapshot.termination.can_continue() {
            DiscoveryAction::ReopenCandidateSearch
        } else {
            DiscoveryAction::ReportUnresolved
        };
    }

    if snapshot.validation == ValidationStatus::Passed && snapshot.adequacy.is_trustworthy_clear() {
        return DiscoveryAction::AcceptValidatedCandidate;
    }

    // Do not spend a nonexistent budget. A singleton without all required evidence remains
    // unresolved; missing budget must never be silently treated as a validation pass.
    if !snapshot.termination.can_continue() {
        return DiscoveryAction::ReportUnresolved;
    }

    if snapshot.adequacy.needs_independent_assessment() {
        return DiscoveryAction::RunIndependentAdequacyAudit;
    }

    match snapshot.validation {
        ValidationStatus::NotStarted | ValidationStatus::InProgress => {
            DiscoveryAction::ValidateProvisionalSingleton
        }
        ValidationStatus::Passed => DiscoveryAction::AcceptValidatedCandidate,
        ValidationStatus::Contradicted => DiscoveryAction::ReopenCandidateSearch,
        ValidationStatus::BudgetExhausted => DiscoveryAction::ReportUnresolved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        live_candidate_count: usize,
        adequacy: AdequacyAssessment,
        validation: ValidationStatus,
        termination: TerminationStatus,
    ) -> DiscoverySnapshot {
        DiscoverySnapshot {
            live_candidate_count,
            adequacy,
            validation,
            termination,
        }
    }

    const TRUSTWORTHY_CLEAR: AdequacyAssessment = AdequacyAssessment::Clear {
        calibrated: true,
        independent_audit: true,
    };

    const TRUSTWORTHY_ALARM: AdequacyAssessment = AdequacyAssessment::Alarm {
        diagnostic: AdequacyDiagnostic::PredictiveMisfit,
        calibrated: true,
        independent_audit: true,
    };

    #[test]
    fn empty_candidate_set_reopens_search_without_claiming_misspecification() {
        assert_eq!(
            next_discovery_action(snapshot(
                0,
                AdequacyAssessment::NotEvaluated,
                ValidationStatus::NotStarted,
                TerminationStatus::Active,
            )),
            DiscoveryAction::ReopenCandidateSearch
        );
        assert_eq!(
            next_discovery_action(snapshot(
                0,
                AdequacyAssessment::NotEvaluated,
                ValidationStatus::NotStarted,
                TerminationStatus::CandidatePoolExhausted,
            )),
            DiscoveryAction::ReportEmptySetWithoutAdequacyEvidence
        );
    }

    #[test]
    fn calibrated_independent_alarm_reports_model_space_inadequacy() {
        assert_eq!(
            next_discovery_action(snapshot(
                1,
                TRUSTWORTHY_ALARM,
                ValidationStatus::Passed,
                TerminationStatus::Active,
            )),
            DiscoveryAction::ReportModelSpaceInadequate
        );
    }

    #[test]
    fn uncalibrated_alarm_requests_an_independent_audit() {
        assert_eq!(
            next_discovery_action(snapshot(
                1,
                AdequacyAssessment::Alarm {
                    diagnostic: AdequacyDiagnostic::CalibrationFailure,
                    calibrated: false,
                    independent_audit: true,
                },
                ValidationStatus::Passed,
                TerminationStatus::Active,
            )),
            DiscoveryAction::RunIndependentAdequacyAudit
        );
    }

    #[test]
    fn non_independent_alarm_cannot_reject_the_model_space() {
        assert_eq!(
            next_discovery_action(snapshot(
                2,
                AdequacyAssessment::Alarm {
                    diagnostic: AdequacyDiagnostic::CoverageGap,
                    calibrated: true,
                    independent_audit: false,
                },
                ValidationStatus::InProgress,
                TerminationStatus::Active,
            )),
            DiscoveryAction::RunIndependentAdequacyAudit
        );
    }

    #[test]
    fn exhausted_budget_never_requests_an_unavailable_adequacy_audit() {
        let untrusted_alarm = AdequacyAssessment::Alarm {
            diagnostic: AdequacyDiagnostic::CalibrationFailure,
            calibrated: false,
            independent_audit: false,
        };
        assert_eq!(
            next_discovery_action(snapshot(
                1,
                untrusted_alarm,
                ValidationStatus::InProgress,
                TerminationStatus::BudgetExhausted,
            )),
            DiscoveryAction::ReportUnresolved
        );
        assert_eq!(
            next_discovery_action(snapshot(
                0,
                untrusted_alarm,
                ValidationStatus::NotStarted,
                TerminationStatus::CandidatePoolExhausted,
            )),
            DiscoveryAction::ReportEmptySetWithoutAdequacyEvidence
        );
    }

    #[test]
    fn singleton_requires_both_validation_and_trustworthy_clear_assessment() {
        assert_eq!(
            next_discovery_action(snapshot(
                1,
                TRUSTWORTHY_CLEAR,
                ValidationStatus::NotStarted,
                TerminationStatus::Active,
            )),
            DiscoveryAction::ValidateProvisionalSingleton
        );
        assert_eq!(
            next_discovery_action(snapshot(
                1,
                TRUSTWORTHY_CLEAR,
                ValidationStatus::Passed,
                TerminationStatus::Active,
            )),
            DiscoveryAction::AcceptValidatedCandidate
        );
    }

    #[test]
    fn singleton_with_no_adequacy_assessment_runs_audit_first() {
        assert_eq!(
            next_discovery_action(snapshot(
                1,
                AdequacyAssessment::NotEvaluated,
                ValidationStatus::Passed,
                TerminationStatus::Active,
            )),
            DiscoveryAction::RunIndependentAdequacyAudit
        );
    }

    #[test]
    fn contradicted_singleton_reopens_search_when_budget_remains() {
        assert_eq!(
            next_discovery_action(snapshot(
                1,
                TRUSTWORTHY_CLEAR,
                ValidationStatus::Contradicted,
                TerminationStatus::Active,
            )),
            DiscoveryAction::ReopenCandidateSearch
        );
    }

    #[test]
    fn exhausted_budget_never_turns_missing_evidence_into_acceptance() {
        assert_eq!(
            next_discovery_action(snapshot(
                1,
                AdequacyAssessment::NotEvaluated,
                ValidationStatus::InProgress,
                TerminationStatus::BudgetExhausted,
            )),
            DiscoveryAction::ReportUnresolved
        );
    }

    #[test]
    fn multiple_candidates_continue_only_while_budget_and_actions_remain() {
        assert_eq!(
            next_discovery_action(snapshot(
                3,
                TRUSTWORTHY_CLEAR,
                ValidationStatus::NotStarted,
                TerminationStatus::Active,
            )),
            DiscoveryAction::ContinueDiscrimination
        );
        assert_eq!(
            next_discovery_action(snapshot(
                3,
                TRUSTWORTHY_CLEAR,
                ValidationStatus::NotStarted,
                TerminationStatus::CandidatePoolExhausted,
            )),
            DiscoveryAction::ReportUnresolved
        );
    }

    #[test]
    fn invalid_numerical_run_fails_closed_before_acceptance() {
        assert_eq!(
            next_discovery_action(snapshot(
                1,
                TRUSTWORTHY_CLEAR,
                ValidationStatus::Passed,
                TerminationStatus::NumericalFailure,
            )),
            DiscoveryAction::FailClosed
        );
    }
}
