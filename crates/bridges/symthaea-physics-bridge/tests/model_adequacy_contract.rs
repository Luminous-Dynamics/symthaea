// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exhaustive contract tests for the model-adequacy action gate.
//!
//! These tests enumerate the small policy state space to guard against interactions between
//! candidate count, evidence quality, validation, and termination that hand-picked examples miss.

use symthaea_physics_bridge::model_adequacy::{
    AdequacyAssessment, AdequacyDiagnostic, DiscoveryAction, DiscoverySnapshot, TerminationStatus,
    ValidationStatus, next_discovery_action,
};

fn all_assessments() -> Vec<AdequacyAssessment> {
    let diagnostics = [
        AdequacyDiagnostic::PredictiveMisfit,
        AdequacyDiagnostic::CoverageGap,
        AdequacyDiagnostic::CalibrationFailure,
        AdequacyDiagnostic::QueryDistributionShift,
    ];
    let mut assessments = vec![AdequacyAssessment::NotEvaluated];

    for calibrated in [false, true] {
        for independent_audit in [false, true] {
            assessments.push(AdequacyAssessment::Clear {
                calibrated,
                independent_audit,
            });
            for diagnostic in diagnostics {
                assessments.push(AdequacyAssessment::Alarm {
                    diagnostic,
                    calibrated,
                    independent_audit,
                });
            }
        }
    }

    assessments
}

#[test]
fn action_gate_is_safe_for_every_enumerated_state_combination() {
    let assessments = all_assessments();
    let validations = [
        ValidationStatus::NotStarted,
        ValidationStatus::InProgress,
        ValidationStatus::Passed,
        ValidationStatus::Contradicted,
        ValidationStatus::BudgetExhausted,
    ];
    let terminations = [
        TerminationStatus::Active,
        TerminationStatus::BudgetExhausted,
        TerminationStatus::CandidatePoolExhausted,
        TerminationStatus::NoAdmissibleExperiment,
        TerminationStatus::NumericalFailure,
        TerminationStatus::InvalidConfiguration,
    ];
    let mut checked_states = 0usize;

    for assessment in assessments {
        for validation in validations {
            for termination in terminations {
                for live_candidate_count in [0, 1, 2] {
                    let snapshot = DiscoverySnapshot {
                        live_candidate_count,
                        adequacy: assessment,
                        validation,
                        termination,
                    };
                    let action = next_discovery_action(snapshot);
                    checked_states += 1;

                    if matches!(
                        termination,
                        TerminationStatus::NumericalFailure
                            | TerminationStatus::InvalidConfiguration
                    ) {
                        assert_eq!(
                            action,
                            DiscoveryAction::FailClosed,
                            "invalid termination must dominate every other signal: {snapshot:?}"
                        );
                        continue;
                    }

                    if action == DiscoveryAction::ReportModelSpaceInadequate {
                        assert!(
                            matches!(
                                assessment,
                                AdequacyAssessment::Alarm {
                                    calibrated: true,
                                    independent_audit: true,
                                    ..
                                }
                            ),
                            "model-space inadequacy requires calibrated independent alarm: {snapshot:?}"
                        );
                    }

                    if action == DiscoveryAction::AcceptValidatedCandidate {
                        assert_eq!(
                            live_candidate_count, 1,
                            "only a singleton can be accepted: {snapshot:?}"
                        );
                        assert_eq!(
                            validation,
                            ValidationStatus::Passed,
                            "candidate validation must pass: {snapshot:?}"
                        );
                        assert_eq!(
                            assessment,
                            AdequacyAssessment::Clear {
                                calibrated: true,
                                independent_audit: true,
                            },
                            "independent calibrated adequacy must be clear: {snapshot:?}"
                        );
                    }

                    if let AdequacyAssessment::Alarm {
                        calibrated,
                        independent_audit,
                        ..
                    } = assessment
                    {
                        if !(calibrated && independent_audit) {
                            let expected = if termination == TerminationStatus::Active {
                                DiscoveryAction::RunIndependentAdequacyAudit
                            } else if live_candidate_count == 0 {
                                DiscoveryAction::ReportEmptySetWithoutAdequacyEvidence
                            } else {
                                DiscoveryAction::ReportUnresolved
                            };
                            assert_eq!(
                                action, expected,
                                "an untrusted alarm must request an audit only when budget remains: {snapshot:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    assert_eq!(checked_states, 21 * 5 * 6 * 3);
}
