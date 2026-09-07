// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification-only theorem for causal-identification assumption diagnostics.
//!
//! Declared assumption != diagnostic != diagnostic outcome != assumption truth.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AssumptionTestability {
    DirectlyAuditable,
    FalsifiableDiagnostic,
    PartiallyTestable,
    NotTestableFromObservedData,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IdentificationAssumption {
    id: String,
    description: String,
    testability: AssumptionTestability,
}

impl IdentificationAssumption {
    fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        testability: AssumptionTestability,
    ) -> Result<Self, AssumptionError> {
        let id = id.into();
        let description = description.into();
        if id.trim().is_empty() {
            return Err(AssumptionError::EmptyText("assumption id"));
        }
        if description.trim().is_empty() {
            return Err(AssumptionError::EmptyText("assumption description"));
        }
        Ok(Self {
            id,
            description,
            testability,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiagnosticDecisionRule {
    MaximumAbsoluteStatistic { threshold_atoms: i128 },
    MinimumProbabilityMassBps { threshold_bps: u16 },
    ExactArtifactEquality,
}

impl DiagnosticDecisionRule {
    fn validate(self) -> Result<(), AssumptionError> {
        match self {
            Self::MaximumAbsoluteStatistic { threshold_atoms } if threshold_atoms <= 0 => {
                Err(AssumptionError::InvalidDecisionRule)
            }
            Self::MinimumProbabilityMassBps { threshold_bps }
                if threshold_bps == 0 || threshold_bps >= 10_000 =>
            {
                Err(AssumptionError::InvalidDecisionRule)
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiagnosticSpecification {
    id: String,
    assumption_id: String,
    diagnostic_family_id: String,
    implementation_artifact_id: String,
    decision_rule: DiagnosticDecisionRule,
}

impl DiagnosticSpecification {
    fn new(
        id: impl Into<String>,
        assumption: &IdentificationAssumption,
        diagnostic_family_id: impl Into<String>,
        implementation_artifact_id: impl Into<String>,
        decision_rule: DiagnosticDecisionRule,
    ) -> Result<Self, AssumptionError> {
        if assumption.testability == AssumptionTestability::NotTestableFromObservedData {
            return Err(AssumptionError::UntestableAssumptionCannotHaveObservedDiagnostic);
        }
        let id = id.into();
        let diagnostic_family_id = diagnostic_family_id.into();
        let implementation_artifact_id = implementation_artifact_id.into();
        for (field, value) in [
            ("diagnostic id", id.as_str()),
            ("diagnostic family id", diagnostic_family_id.as_str()),
            ("diagnostic implementation artifact", implementation_artifact_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(AssumptionError::EmptyText(field));
            }
        }
        decision_rule.validate()?;
        Ok(Self {
            id,
            assumption_id: assumption.id.clone(),
            diagnostic_family_id,
            implementation_artifact_id,
            decision_rule,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DiagnosticObservation {
    AbsoluteStatisticAtoms(i128),
    ProbabilityMassBps(u16),
    ArtifactEquality(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiagnosticOutcome {
    NoDetectedViolation,
    DetectedViolation,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiagnosticResult {
    diagnostic_id: String,
    assumption_id: String,
    implementation_artifact_id: String,
    observation: DiagnosticObservation,
    outcome: DiagnosticOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AssumptionAssessment {
    NoDetectedViolation {
        assumption_id: String,
        diagnostic_ids: Vec<String>,
    },
    DetectedViolation {
        assumption_id: String,
        violating_diagnostic_ids: Vec<String>,
    },
    Inconclusive {
        assumption_id: String,
        diagnostic_ids: Vec<String>,
    },
    NotTestableFromObservedData {
        assumption_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AssumptionError {
    EmptyText(&'static str),
    InvalidDecisionRule,
    AssumptionMismatch,
    DiagnosticMismatch,
    ObservationShapeMismatch,
    UntestableAssumptionCannotHaveObservedDiagnostic,
    EmptyDiagnosticSet,
}

fn evaluate_diagnostic(
    assumption: &IdentificationAssumption,
    diagnostic: &DiagnosticSpecification,
    observation: DiagnosticObservation,
) -> Result<DiagnosticResult, AssumptionError> {
    if diagnostic.assumption_id != assumption.id {
        return Err(AssumptionError::AssumptionMismatch);
    }

    let outcome = match (diagnostic.decision_rule, &observation) {
        (
            DiagnosticDecisionRule::MaximumAbsoluteStatistic { threshold_atoms },
            DiagnosticObservation::AbsoluteStatisticAtoms(value),
        ) => {
            if value.unsigned_abs() <= threshold_atoms.unsigned_abs() {
                DiagnosticOutcome::NoDetectedViolation
            } else {
                DiagnosticOutcome::DetectedViolation
            }
        }
        (
            DiagnosticDecisionRule::MinimumProbabilityMassBps { threshold_bps },
            DiagnosticObservation::ProbabilityMassBps(value),
        ) => {
            if *value >= threshold_bps {
                DiagnosticOutcome::NoDetectedViolation
            } else {
                DiagnosticOutcome::DetectedViolation
            }
        }
        (
            DiagnosticDecisionRule::ExactArtifactEquality,
            DiagnosticObservation::ArtifactEquality(true),
        ) => DiagnosticOutcome::NoDetectedViolation,
        (
            DiagnosticDecisionRule::ExactArtifactEquality,
            DiagnosticObservation::ArtifactEquality(false),
        ) => DiagnosticOutcome::DetectedViolation,
        _ => return Err(AssumptionError::ObservationShapeMismatch),
    };

    Ok(DiagnosticResult {
        diagnostic_id: diagnostic.id.clone(),
        assumption_id: assumption.id.clone(),
        implementation_artifact_id: diagnostic.implementation_artifact_id.clone(),
        observation,
        outcome,
    })
}

fn assess_assumption(
    assumption: &IdentificationAssumption,
    diagnostics: &[DiagnosticResult],
) -> Result<AssumptionAssessment, AssumptionError> {
    if assumption.testability == AssumptionTestability::NotTestableFromObservedData {
        if !diagnostics.is_empty() {
            return Err(AssumptionError::UntestableAssumptionCannotHaveObservedDiagnostic);
        }
        return Ok(AssumptionAssessment::NotTestableFromObservedData {
            assumption_id: assumption.id.clone(),
        });
    }
    if diagnostics.is_empty() {
        return Err(AssumptionError::EmptyDiagnosticSet);
    }
    if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.assumption_id != assumption.id)
    {
        return Err(AssumptionError::AssumptionMismatch);
    }

    let violating: Vec<String> = diagnostics
        .iter()
        .filter(|item| item.outcome == DiagnosticOutcome::DetectedViolation)
        .map(|item| item.diagnostic_id.clone())
        .collect();
    if !violating.is_empty() {
        return Ok(AssumptionAssessment::DetectedViolation {
            assumption_id: assumption.id.clone(),
            violating_diagnostic_ids: violating,
        });
    }

    let inconclusive = diagnostics
        .iter()
        .any(|item| item.outcome == DiagnosticOutcome::Inconclusive);
    let ids = diagnostics
        .iter()
        .map(|item| item.diagnostic_id.clone())
        .collect();
    if inconclusive {
        Ok(AssumptionAssessment::Inconclusive {
            assumption_id: assumption.id.clone(),
            diagnostic_ids: ids,
        })
    } else {
        Ok(AssumptionAssessment::NoDetectedViolation {
            assumption_id: assumption.id.clone(),
            diagnostic_ids: ids,
        })
    }
}

fn parallel_trends() -> IdentificationAssumption {
    IdentificationAssumption::new(
        "assumption:parallel-trends-v1",
        "Absent treatment, treated and comparison groups would follow the declared counterfactual trend relation.",
        AssumptionTestability::PartiallyTestable,
    )
    .unwrap()
}

fn exclusion_restriction() -> IdentificationAssumption {
    IdentificationAssumption::new(
        "assumption:iv-exclusion-v1",
        "The instrument affects the outcome only through the declared treatment pathway.",
        AssumptionTestability::NotTestableFromObservedData,
    )
    .unwrap()
}

#[test]
fn passing_pretrend_diagnostic_means_no_detected_violation_not_assumption_true() {
    let assumption = parallel_trends();
    let spec = DiagnosticSpecification::new(
        "diagnostic:pretrend-v1",
        &assumption,
        "pretrend-difference",
        "artifact:pretrend-test-v1",
        DiagnosticDecisionRule::MaximumAbsoluteStatistic {
            threshold_atoms: 25,
        },
    )
    .unwrap();
    let result = evaluate_diagnostic(
        &assumption,
        &spec,
        DiagnosticObservation::AbsoluteStatisticAtoms(10),
    )
    .unwrap();
    let assessment = assess_assumption(&assumption, &[result]).unwrap();

    assert!(matches!(
        assessment,
        AssumptionAssessment::NoDetectedViolation { .. }
    ));
}

#[test]
fn failed_diagnostic_records_detected_violation() {
    let assumption = parallel_trends();
    let spec = DiagnosticSpecification::new(
        "diagnostic:pretrend-v1",
        &assumption,
        "pretrend-difference",
        "artifact:pretrend-test-v1",
        DiagnosticDecisionRule::MaximumAbsoluteStatistic {
            threshold_atoms: 25,
        },
    )
    .unwrap();
    let result = evaluate_diagnostic(
        &assumption,
        &spec,
        DiagnosticObservation::AbsoluteStatisticAtoms(100),
    )
    .unwrap();
    let assessment = assess_assumption(&assumption, &[result]).unwrap();

    assert_eq!(
        assessment,
        AssumptionAssessment::DetectedViolation {
            assumption_id: assumption.id,
            violating_diagnostic_ids: vec!["diagnostic:pretrend-v1".into()],
        }
    );
}

#[test]
fn untestable_exclusion_restriction_cannot_be_certified_by_fake_observed_diagnostic() {
    let assumption = exclusion_restriction();
    assert_eq!(
        DiagnosticSpecification::new(
            "diagnostic:fake-exclusion-test",
            &assumption,
            "posthoc-correlation",
            "artifact:fake-test-v1",
            DiagnosticDecisionRule::ExactArtifactEquality,
        ),
        Err(AssumptionError::UntestableAssumptionCannotHaveObservedDiagnostic)
    );
    assert_eq!(
        assess_assumption(&assumption, &[]),
        Ok(AssumptionAssessment::NotTestableFromObservedData {
            assumption_id: assumption.id,
        })
    );
}

#[test]
fn diagnostic_threshold_is_predeclared_in_specification() {
    let assumption = parallel_trends();
    let strict = DiagnosticSpecification::new(
        "diagnostic:strict-v1",
        &assumption,
        "pretrend-difference",
        "artifact:pretrend-test-v1",
        DiagnosticDecisionRule::MaximumAbsoluteStatistic {
            threshold_atoms: 10,
        },
    )
    .unwrap();
    let loose = DiagnosticSpecification::new(
        "diagnostic:loose-v1",
        &assumption,
        "pretrend-difference",
        "artifact:pretrend-test-v1",
        DiagnosticDecisionRule::MaximumAbsoluteStatistic {
            threshold_atoms: 50,
        },
    )
    .unwrap();

    let strict_result = evaluate_diagnostic(
        &assumption,
        &strict,
        DiagnosticObservation::AbsoluteStatisticAtoms(20),
    )
    .unwrap();
    let loose_result = evaluate_diagnostic(
        &assumption,
        &loose,
        DiagnosticObservation::AbsoluteStatisticAtoms(20),
    )
    .unwrap();

    assert_eq!(strict_result.outcome, DiagnosticOutcome::DetectedViolation);
    assert_eq!(loose_result.outcome, DiagnosticOutcome::NoDetectedViolation);
    assert_ne!(strict.id, loose.id);
}

#[test]
fn diagnostic_result_retains_exact_implementation_and_observation() {
    let assumption = IdentificationAssumption::new(
        "assumption:randomization-integrity-v1",
        "The observed assignment artifact matches the preregistered assignment artifact.",
        AssumptionTestability::DirectlyAuditable,
    )
    .unwrap();
    let spec = DiagnosticSpecification::new(
        "diagnostic:assignment-match-v1",
        &assumption,
        "artifact-equality",
        "artifact:assignment-verifier-v1",
        DiagnosticDecisionRule::ExactArtifactEquality,
    )
    .unwrap();
    let result = evaluate_diagnostic(
        &assumption,
        &spec,
        DiagnosticObservation::ArtifactEquality(true),
    )
    .unwrap();

    assert_eq!(result.implementation_artifact_id, "artifact:assignment-verifier-v1");
    assert_eq!(result.observation, DiagnosticObservation::ArtifactEquality(true));
}

#[test]
fn probability_mass_rule_is_validated_and_does_not_imply_truth() {
    let assumption = IdentificationAssumption::new(
        "assumption:overlap-v1",
        "The declared treatment groups retain sufficient support in the measured covariate region.",
        AssumptionTestability::FalsifiableDiagnostic,
    )
    .unwrap();
    assert_eq!(
        DiagnosticSpecification::new(
            "diagnostic:bad-overlap",
            &assumption,
            "overlap-mass",
            "artifact:overlap-v1",
            DiagnosticDecisionRule::MinimumProbabilityMassBps { threshold_bps: 0 },
        ),
        Err(AssumptionError::InvalidDecisionRule)
    );

    let spec = DiagnosticSpecification::new(
        "diagnostic:overlap-v1",
        &assumption,
        "overlap-mass",
        "artifact:overlap-v1",
        DiagnosticDecisionRule::MinimumProbabilityMassBps { threshold_bps: 500 },
    )
    .unwrap();
    let result = evaluate_diagnostic(
        &assumption,
        &spec,
        DiagnosticObservation::ProbabilityMassBps(800),
    )
    .unwrap();
    assert_eq!(result.outcome, DiagnosticOutcome::NoDetectedViolation);
}

#[test]
fn diagnostic_cannot_be_replayed_against_another_assumption() {
    let trends = parallel_trends();
    let assignment = IdentificationAssumption::new(
        "assumption:assignment-v1",
        "Assignment follows the sealed protocol.",
        AssumptionTestability::DirectlyAuditable,
    )
    .unwrap();
    let spec = DiagnosticSpecification::new(
        "diagnostic:pretrend-v1",
        &trends,
        "pretrend-difference",
        "artifact:pretrend-test-v1",
        DiagnosticDecisionRule::MaximumAbsoluteStatistic {
            threshold_atoms: 25,
        },
    )
    .unwrap();

    assert_eq!(
        evaluate_diagnostic(
            &assignment,
            &spec,
            DiagnosticObservation::AbsoluteStatisticAtoms(5),
        ),
        Err(AssumptionError::AssumptionMismatch)
    );
}

#[test]
fn observation_shape_must_match_precommitted_decision_rule() {
    let assumption = parallel_trends();
    let spec = DiagnosticSpecification::new(
        "diagnostic:pretrend-v1",
        &assumption,
        "pretrend-difference",
        "artifact:pretrend-test-v1",
        DiagnosticDecisionRule::MaximumAbsoluteStatistic {
            threshold_atoms: 25,
        },
    )
    .unwrap();
    assert_eq!(
        evaluate_diagnostic(
            &assumption,
            &spec,
            DiagnosticObservation::ArtifactEquality(true),
        ),
        Err(AssumptionError::ObservationShapeMismatch)
    );
}

#[test]
fn no_detected_violation_and_detected_violation_can_coexist_across_assumptions() {
    let trends = parallel_trends();
    let overlap = IdentificationAssumption::new(
        "assumption:overlap-v1",
        "Treatment groups retain support.",
        AssumptionTestability::FalsifiableDiagnostic,
    )
    .unwrap();

    let trends_spec = DiagnosticSpecification::new(
        "diagnostic:pretrend-v1",
        &trends,
        "pretrend-difference",
        "artifact:pretrend-test-v1",
        DiagnosticDecisionRule::MaximumAbsoluteStatistic {
            threshold_atoms: 25,
        },
    )
    .unwrap();
    let overlap_spec = DiagnosticSpecification::new(
        "diagnostic:overlap-v1",
        &overlap,
        "overlap-mass",
        "artifact:overlap-v1",
        DiagnosticDecisionRule::MinimumProbabilityMassBps { threshold_bps: 500 },
    )
    .unwrap();

    let trends_result = evaluate_diagnostic(
        &trends,
        &trends_spec,
        DiagnosticObservation::AbsoluteStatisticAtoms(10),
    )
    .unwrap();
    let overlap_result = evaluate_diagnostic(
        &overlap,
        &overlap_spec,
        DiagnosticObservation::ProbabilityMassBps(200),
    )
    .unwrap();

    assert!(matches!(
        assess_assumption(&trends, &[trends_result]).unwrap(),
        AssumptionAssessment::NoDetectedViolation { .. }
    ));
    assert!(matches!(
        assess_assumption(&overlap, &[overlap_result]).unwrap(),
        AssumptionAssessment::DetectedViolation { .. }
    ));
}
