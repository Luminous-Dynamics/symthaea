//! Read-only comparison primitives for plural ethical-framework assessments.
//!
//! This module is an additive evaluation seam for the existing EthicsEngine.
//! It does not parse scenarios, implement a normative framework, authorize
//! actions, or mutate the existing ethics verdict. Callers supply separately
//! produced framework assessments; this module validates their traceability
//! fields and reports agreement, disagreement, or insufficiency without
//! collapsing framework judgments into a single score.
//!
//! A valid comparison is not proof that any framework is morally correct.

use std::collections::HashSet;

/// Framework-relative position on a candidate action.
///
/// These values intentionally do not map to operational permission. In
/// particular, `SupportsAction` does not authorize execution, and
/// `OpposesAction` does not replace the independent safety/action gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameworkStance {
    /// The selected framework supports the candidate action under its premises.
    SupportsAction,
    /// The selected framework opposes the candidate action under its premises.
    OpposesAction,
    /// The framework identifies material competing considerations.
    Mixed,
    /// The framework's result depends on conditions that are not yet established.
    Conditional,
    /// The available information does not determine a framework-relative result.
    Underdetermined,
}

/// Exact subject binding shared by assessments that may be compared.
///
/// The producer should compute each digest from a canonical serialization of
/// the corresponding context/action. This module only checks that the IDs and
/// digests are non-empty and exactly equal across assessments; it does not
/// recompute digests, authenticate producers, or prove the source bytes match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssessmentSubject {
    /// Stable reference to the normalized scenario/context.
    pub scenario_ref: String,
    /// Opaque digest/fingerprint of the canonical scenario/context.
    pub scenario_digest: String,
    /// Stable reference to the candidate action being assessed.
    pub candidate_action_ref: String,
    /// Opaque digest/fingerprint of the canonical candidate action.
    pub candidate_action_digest: String,
}

/// Freshness of the evaluator result for the reported subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssessmentFreshness {
    /// The evaluator ran on the reported subject for this assessment.
    Fresh,
    /// The result was carried forward from an earlier evaluation.
    ///
    /// The source subject must remain bound to its original scenario/action.
    /// Strict comparison will not treat carried-forward assessments as fresh
    /// evidence of agreement.
    CarriedForward,
    /// The evaluator did not produce an assessment for this requested subject.
    ///
    /// Preserve this as an explicit unavailable slot rather than silently
    /// omitting the framework or manufacturing a neutral result.
    Unavailable,
}

/// Provenance for an ethical-framework assessment.
///
/// These fields are caller-supplied metadata. This module validates their
/// shape and consistency, but does not attest the build identity or prove that
/// the evaluator actually processed the declared source subject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssessmentProvenance {
    /// Exact subject that the evaluator actually processed.
    ///
    /// Must be present for Fresh and CarriedForward, and absent for Unavailable.
    pub source_subject: Option<AssessmentSubject>,
    /// Build/revision identity of the evaluator implementation.
    pub evaluator_build_id: String,
    /// Cycle or monotonic evaluation sequence assigned by the producer.
    ///
    /// Must be present for Fresh and CarriedForward, and absent for Unavailable.
    pub evaluation_cycle: Option<u64>,
    /// Whether the result was computed on the subject or carried forward.
    pub freshness: AssessmentFreshness,
}

/// A single result from one versioned ethical framework.
///
/// `premise_refs` identify normative premises/rules used in the assessment.
/// `evidence_refs` identify relevant empirical support when applicable. A
/// premise is not an empirical fact, and a cited fact is not itself a moral rule.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameworkAssessment {
    /// Stable framework identifier, e.g. `eight_harmonies` or `duty_ethics_demo`.
    pub framework_id: String,
    /// Exact version of the framework that produced this assessment.
    pub framework_version: String,
    /// Scenario and candidate action this assessment claims to describe.
    ///
    /// Must equal provenance.source_subject for a fresh, comparable result.
    /// For carried-forward results, retain the old source subject in provenance
    /// rather than relabeling it to the current input.
    pub subject: AssessmentSubject,
    /// Producer-supplied origin and freshness information.
    pub provenance: AssessmentProvenance,
    /// Framework-relative position; never an execution authorization.
    pub stance: FrameworkStance,
    /// Optional calibrated confidence in this assessment, if defined by the caller.
    pub confidence: Option<f64>,
    /// Human-readable explanation. Must not be empty for a valid result.
    pub rationale: Vec<String>,
    /// IDs of normative premises/rules used. Must contain at least one non-empty ID.
    pub premise_refs: Vec<String>,
    /// Source or observation identifiers supporting empirical claims.
    pub evidence_refs: Vec<String>,
    /// Unresolved questions, conditions, objections, or missing information.
    pub unresolved_questions: Vec<String>,
}

/// Stable identity/version pair for a framework required by a comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameworkIdentity {
    pub framework_id: String,
    pub framework_version: String,
}

impl FrameworkIdentity {
    pub fn new(framework_id: impl Into<String>, framework_version: impl Into<String>) -> Self {
        Self {
            framework_id: framework_id.into(),
            framework_version: framework_version.into(),
        }
    }
}

/// Count of separately produced framework positions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AssessmentCounts {
    pub supports: usize,
    pub opposes: usize,
    pub mixed: usize,
    pub conditional: usize,
    pub underdetermined: usize,
}

/// Result of comparing assessments without deciding which framework is correct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonState {
    /// No framework results were supplied.
    NoAssessments,
    /// Every valid framework assessment supports the candidate action.
    AgreementSupports,
    /// Every valid framework assessment opposes the candidate action.
    AgreementOpposes,
    /// At least one framework supports and at least one opposes the action.
    Disagreement,
    /// At least one result is mixed, conditional, unavailable, or underdetermined,
    /// or the declared framework coverage is incomplete.
    Incomplete,
    /// Fewer than two distinct framework identities are represented.
    ///
    /// Multiple versions of one framework do not count as independent ethical
    /// perspectives and cannot establish plural agreement.
    InsufficientFrameworks,
    /// Valid assessments refer to different scenarios or candidate actions, or
    /// do not match the explicit subject requested by the caller.
    SubjectMismatch,
    /// The reported subject differs from the subject the evaluator actually ran.
    SourceSubjectMismatch,
    /// At least one assessment is carried forward rather than freshly evaluated.
    StaleAssessment,
    /// A result was supplied for a framework/version outside the declared roster.
    FrameworkSetMismatch,
    /// An assessment or roster is malformed, or a duplicate framework/version exists.
    InvalidInput,
}

/// Read-only multi-framework comparison.
///
/// `InvalidInput` is fail-closed: consumers must not treat its counts or
/// underlying assessments as a qualified agreement/disagreement result.
#[derive(Debug, Clone, PartialEq)]
pub struct PluralEthicsComparison {
    pub state: ComparisonState,
    pub counts: AssessmentCounts,
    pub assessments: Vec<FrameworkAssessment>,
    /// Required framework/version identities that had no corresponding result row.
    pub missing_frameworks: Vec<FrameworkIdentity>,
    /// Supplied framework/version identities that were not in the required roster.
    pub unexpected_frameworks: Vec<FrameworkIdentity>,
    pub validation_errors: Vec<String>,
}

/// Compare previously produced framework assessments.
///
/// This function deliberately performs no score averaging, ranking, moral
/// arbitration, or action authorization. Frameworks that disagree remain in
/// disagreement. A malformed assessment invalidates the comparison rather than
/// silently dropping the bad row.
pub fn compare_assessments(assessments: &[FrameworkAssessment]) -> PluralEthicsComparison {
    let expected_subject = assessments.first().map(|assessment| &assessment.subject);
    compare_assessments_inner(expected_subject, assessments)
}

/// Compare assessments for an explicitly requested subject.
///
/// Unlike compare_assessments, this form lets the caller state which exact
/// scenario/action is being compared. It refuses a result whose reported
/// subject differs from that target, whose producer provenance points at a
/// different source subject, or whose result was carried forward. It still
/// does not authenticate provenance metadata or authorize any action.
pub fn compare_assessments_for(
    expected_subject: &AssessmentSubject,
    assessments: &[FrameworkAssessment],
) -> PluralEthicsComparison {
    compare_assessments_inner(Some(expected_subject), assessments)
}

/// Compare assessments against a declared roster of required framework versions.
///
/// The roster is part of the comparison contract: omitting a required framework
/// results in Incomplete, while returning a framework not listed in the roster
/// results in FrameworkSetMismatch. An explicit Unavailable row counts as
/// represented coverage, but still makes the assessment outcome Incomplete.
/// This does not determine which frameworks are ethically sufficient; callers
/// must define and version the roster for their use case.
pub fn compare_assessments_with_roster(
    expected_subject: &AssessmentSubject,
    expected_frameworks: &[FrameworkIdentity],
    assessments: &[FrameworkAssessment],
) -> PluralEthicsComparison {
    let roster_errors = validate_framework_roster(expected_frameworks);
    let mut result = compare_assessments_inner(Some(expected_subject), assessments);
    result.validation_errors.extend(roster_errors);

    if !result.validation_errors.is_empty() {
        result.state = ComparisonState::InvalidInput;
        return result;
    }

    let expected: HashSet<FrameworkIdentity> = expected_frameworks.iter().cloned().collect();
    let actual: HashSet<FrameworkIdentity> = assessments
        .iter()
        .map(|assessment| FrameworkIdentity::new(
            assessment.framework_id.trim(),
            assessment.framework_version.trim(),
        ))
        .collect();

    result.missing_frameworks = expected.difference(&actual).cloned().collect();
    result.missing_frameworks.sort();

    result.unexpected_frameworks = actual.difference(&expected).cloned().collect();
    result.unexpected_frameworks.sort();

    let subject_or_provenance_failure = matches!(
        result.state,
        ComparisonState::SubjectMismatch
            | ComparisonState::SourceSubjectMismatch
            | ComparisonState::StaleAssessment
            | ComparisonState::InvalidInput
    );

    // Coverage issues must never hide a more fundamental subject/provenance
    // failure. Preserve those diagnoses even when the roster is also wrong.
    if !subject_or_provenance_failure && !result.unexpected_frameworks.is_empty() {
        result.state = ComparisonState::FrameworkSetMismatch;
    } else if !subject_or_provenance_failure && !result.missing_frameworks.is_empty() {
        // In particular, a single positive result cannot be called unanimous
        // when the declared roster includes a missing evaluator.
        result.state = ComparisonState::Incomplete;
    }

    result
}

fn validate_framework_roster(roster: &[FrameworkIdentity]) -> Vec<String> {
    let mut errors = Vec::new();
    let mut seen = HashSet::new();

    if roster.is_empty() {
        errors.push("expected_frameworks must not be empty".to_owned());
        return errors;
    }

    for (index, framework) in roster.iter().enumerate() {
        if framework.framework_id.trim().is_empty() {
            errors.push(format!("expected_frameworks[{index}].framework_id must not be empty"));
        }
        if framework.framework_version.trim().is_empty() {
            errors.push(format!("expected_frameworks[{index}].framework_version must not be empty"));
        }
        if !framework.framework_id.trim().is_empty()
            && !framework.framework_version.trim().is_empty()
            && !seen.insert(FrameworkIdentity::new(
                framework.framework_id.trim(),
                framework.framework_version.trim(),
            ))
        {
            errors.push(format!(
                "expected_frameworks[{index}] duplicates {}@{}",
                framework.framework_id.trim(),
                framework.framework_version.trim()
            ));
        }
    }

    errors
}

fn compare_assessments_inner(
    expected_subject: Option<&AssessmentSubject>,
    assessments: &[FrameworkAssessment],
) -> PluralEthicsComparison {
    let counts = AssessmentCounts {
        supports: assessments.iter().filter(|a| a.stance == FrameworkStance::SupportsAction).count(),
        opposes: assessments.iter().filter(|a| a.stance == FrameworkStance::OpposesAction).count(),
        mixed: assessments.iter().filter(|a| a.stance == FrameworkStance::Mixed).count(),
        conditional: assessments.iter().filter(|a| a.stance == FrameworkStance::Conditional).count(),
        underdetermined: assessments.iter().filter(|a| a.stance == FrameworkStance::Underdetermined).count(),
    };
    let validation_errors = validate_assessments(assessments);

    let state = if !validation_errors.is_empty() {
        ComparisonState::InvalidInput
    } else if assessments.is_empty() {
        ComparisonState::NoAssessments
    } else if !subjects_match(assessments)
        || expected_subject.is_some_and(|expected| {
            assessments.iter().any(|assessment| &assessment.subject != expected)
        })
    {
        // Never report agreement between different inputs or a subject other
        // than the caller's explicit target.
        ComparisonState::SubjectMismatch
    } else if assessments.iter().any(|assessment| {
        match assessment.provenance.freshness {
            AssessmentFreshness::Fresh | AssessmentFreshness::CarriedForward => {
                assessment.provenance.source_subject.as_ref() != Some(&assessment.subject)
            }
            AssessmentFreshness::Unavailable => false,
        }
    }) {
        // Never let a cached result inherit the identity of the current input.
        ComparisonState::SourceSubjectMismatch
    } else if assessments
        .iter()
        .any(|assessment| assessment.provenance.freshness == AssessmentFreshness::CarriedForward)
    {
        ComparisonState::StaleAssessment
    } else if assessments
        .iter()
        .any(|assessment| assessment.provenance.freshness == AssessmentFreshness::Unavailable)
    {
        // One or more requested framework results are explicitly unavailable.
        ComparisonState::Incomplete
    } else if assessments
        .iter()
        .map(|assessment| assessment.framework_id.trim())
        .collect::<HashSet<_>>()
        .len()
        < 2
    {
        // A single perspective, or multiple versions of the same perspective,
        // cannot establish plural agreement or disagreement.
        ComparisonState::InsufficientFrameworks
    } else if counts.supports > 0 && counts.opposes > 0 {
        ComparisonState::Disagreement
    } else if counts.mixed > 0 || counts.conditional > 0 || counts.underdetermined > 0 {
        ComparisonState::Incomplete
    } else if counts.supports == assessments.len() {
        ComparisonState::AgreementSupports
    } else if counts.opposes == assessments.len() {
        ComparisonState::AgreementOpposes
    } else {
        // Defensive fail-closed branch if a future stance is added without
        // explicitly defining its comparison semantics.
        ComparisonState::Incomplete
    };

    PluralEthicsComparison {
        state,
        counts,
        assessments: assessments.to_vec(),
        missing_frameworks: Vec::new(),
        unexpected_frameworks: Vec::new(),
        validation_errors,
    }
}

fn subjects_match(assessments: &[FrameworkAssessment]) -> bool {
    let Some(first) = assessments.first() else {
        return true;
    };

    assessments
        .iter()
        .all(|assessment| assessment.subject == first.subject)
}

fn validate_assessments(assessments: &[FrameworkAssessment]) -> Vec<String> {
    let mut errors = Vec::new();
    let mut seen = HashSet::new();

    for (index, assessment) in assessments.iter().enumerate() {
        let prefix = format!("assessment[{index}]");

        if assessment.framework_id.trim().is_empty() {
            errors.push(format!("{prefix}: framework_id must not be empty"));
        }
        if assessment.framework_version.trim().is_empty() {
            errors.push(format!("{prefix}: framework_version must not be empty"));
        }
        if assessment.rationale.iter().all(|reason| reason.trim().is_empty()) {
            errors.push(format!("{prefix}: rationale must contain a non-empty explanation"));
        }
        if assessment.premise_refs.is_empty()
            && assessment.provenance.freshness != AssessmentFreshness::Unavailable
        {
            errors.push(format!("{prefix}: premise_refs must contain at least one premise/rule ID for an evaluated result"));
        }
        for (ref_index, premise_ref) in assessment.premise_refs.iter().enumerate() {
            if premise_ref.trim().is_empty() {
                errors.push(format!(
                    "{prefix}: premise_refs[{ref_index}] must not be empty"
                ));
            }
        }
        for (field, value) in [
            ("subject.scenario_ref", assessment.subject.scenario_ref.as_str()),
            ("subject.scenario_digest", assessment.subject.scenario_digest.as_str()),
            ("subject.candidate_action_ref", assessment.subject.candidate_action_ref.as_str()),
            ("subject.candidate_action_digest", assessment.subject.candidate_action_digest.as_str()),
        ] {
            if value.trim().is_empty() {
                errors.push(format!("{prefix}: {field} must not be empty"));
            }
        }

        match assessment.provenance.freshness {
            AssessmentFreshness::Fresh | AssessmentFreshness::CarriedForward => {
                let Some(source_subject) = assessment.provenance.source_subject.as_ref() else {
                    errors.push(format!("{prefix}: provenance.source_subject is required for an evaluated result"));
                    continue;
                };
                for (field, value) in [
                    ("provenance.source_subject.scenario_ref", source_subject.scenario_ref.as_str()),
                    ("provenance.source_subject.scenario_digest", source_subject.scenario_digest.as_str()),
                    ("provenance.source_subject.candidate_action_ref", source_subject.candidate_action_ref.as_str()),
                    ("provenance.source_subject.candidate_action_digest", source_subject.candidate_action_digest.as_str()),
                ] {
                    if value.trim().is_empty() {
                        errors.push(format!("{prefix}: {field} must not be empty"));
                    }
                }
                if assessment.provenance.evaluation_cycle.is_none() {
                    errors.push(format!("{prefix}: provenance.evaluation_cycle is required for an evaluated result"));
                }
            }
            AssessmentFreshness::Unavailable => {
                if assessment.provenance.source_subject.is_some() {
                    errors.push(format!("{prefix}: unavailable result must not claim a source_subject"));
                }
                if assessment.provenance.evaluation_cycle.is_some() {
                    errors.push(format!("{prefix}: unavailable result must not claim an evaluation_cycle"));
                }
                if assessment.stance != FrameworkStance::Underdetermined {
                    errors.push(format!("{prefix}: unavailable result must use Underdetermined stance"));
                }
                if !assessment.premise_refs.is_empty() {
                    errors.push(format!("{prefix}: unavailable result must not claim used premise_refs"));
                }
                if !assessment.evidence_refs.is_empty() {
                    errors.push(format!("{prefix}: unavailable result must not claim assessment evidence_refs"));
                }
                if assessment.confidence.is_some() {
                    errors.push(format!("{prefix}: unavailable result must not claim confidence"));
                }
            }
        }
        if assessment.provenance.evaluator_build_id.trim().is_empty() {
            errors.push(format!("{prefix}: provenance.evaluator_build_id must not be empty"));
        }
        if let Some(confidence) = assessment.confidence
            && (!confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
        {
            errors.push(format!("{prefix}: confidence must be finite and within [0, 1]"));
        }

        if !assessment.framework_id.trim().is_empty()
            && !assessment.framework_version.trim().is_empty()
            && !seen.insert((
                assessment.framework_id.trim().to_owned(),
                assessment.framework_version.trim().to_owned(),
            ))
        {
            errors.push(format!(
                "{prefix}: duplicate framework/version assessment {}@{}",
                assessment.framework_id.trim(),
                assessment.framework_version.trim()
            ));
        }

        for (ref_index, evidence_ref) in assessment.evidence_refs.iter().enumerate() {
            if evidence_ref.trim().is_empty() {
                errors.push(format!(
                    "{prefix}: evidence_refs[{ref_index}] must not be empty; omit absent references"
                ));
            }
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assessment(
        framework_id: &str,
        stance: FrameworkStance,
    ) -> FrameworkAssessment {
        FrameworkAssessment {
            framework_id: framework_id.to_owned(),
            framework_version: "1.0.0".to_owned(),
            subject: AssessmentSubject {
                scenario_ref: "scenario:case-001".to_owned(),
                scenario_digest: "fixture-digest:scenario-case-001".to_owned(),
                candidate_action_ref: "action:case-001:candidate-a".to_owned(),
                candidate_action_digest: "fixture-digest:candidate-action-a".to_owned(),
            },
            provenance: AssessmentProvenance {
                source_subject: Some(AssessmentSubject {
                    scenario_ref: "scenario:case-001".to_owned(),
                    scenario_digest: "fixture-digest:scenario-case-001".to_owned(),
                    candidate_action_ref: "action:case-001:candidate-a".to_owned(),
                    candidate_action_digest: "fixture-digest:candidate-action-a".to_owned(),
                }),
                evaluator_build_id: "symthaea-test-build:abc123".to_owned(),
                evaluation_cycle: Some(7),
                freshness: AssessmentFreshness::Fresh,
            },
            stance,
            confidence: Some(0.8),
            rationale: vec!["The conclusion follows from the declared profile premises.".to_owned()],
            premise_refs: vec!["premise:declared-principle-1".to_owned()],
            evidence_refs: vec!["scenario:case-001".to_owned()],
            unresolved_questions: Vec::new(),
        }
    }

    fn identity(id: &str, version: &str) -> FrameworkIdentity {
        FrameworkIdentity::new(id, version)
    }

    #[test]
    fn empty_required_roster_is_invalid() {
        let care = assessment("care_ethics", FrameworkStance::SupportsAction);
        let expected_subject = care.subject.clone();
        let result = compare_assessments_with_roster(&expected_subject, &[], &[care]);

        assert_eq!(result.state, ComparisonState::InvalidInput);
        assert!(result.validation_errors.iter().any(|error| error.contains("must not be empty")));
    }

    #[test]
    fn roster_prevents_false_unanimity_when_framework_is_missing() {
        let result = compare_assessments_with_roster(
            &assessment("care_ethics", FrameworkStance::SupportsAction).subject,
            &[
                identity("care_ethics", "1.0.0"),
                identity("rights_ethics", "2.1.0"),
            ],
            &[assessment("care_ethics", FrameworkStance::SupportsAction)],
        );

        assert_eq!(result.state, ComparisonState::Incomplete);
        assert_eq!(
            result.missing_frameworks,
            vec![identity("rights_ethics", "2.1.0")]
        );
        assert!(result.unexpected_frameworks.is_empty());
    }

    #[test]
    fn roster_accepts_explicit_unavailable_row_without_inventing_coverage() {
        let mut unavailable = assessment("rights_ethics", FrameworkStance::Underdetermined);
        unavailable.provenance.source_subject = None;
        unavailable.provenance.evaluation_cycle = None;
        unavailable.provenance.freshness = AssessmentFreshness::Unavailable;
        unavailable.confidence = None;
        unavailable.premise_refs.clear();
        unavailable.evidence_refs.clear();
        unavailable.rationale = vec!["Framework did not produce a fresh assessment.".to_owned()];

        let care = assessment("care_ethics", FrameworkStance::SupportsAction);
        let expected_subject = care.subject.clone();
        let result = compare_assessments_with_roster(
            &expected_subject,
            &[
                identity("care_ethics", "1.0.0"),
                identity("rights_ethics", "1.0.0"),
            ],
            &[care, unavailable],
        );

        assert_eq!(result.state, ComparisonState::Incomplete);
        assert!(result.missing_frameworks.is_empty());
        assert!(result.unexpected_frameworks.is_empty());
        assert!(result.validation_errors.is_empty());
    }

    #[test]
    fn roster_rejects_unexpected_framework_version() {
        let care = assessment("care_ethics", FrameworkStance::SupportsAction);
        let expected_subject = care.subject.clone();
        let result = compare_assessments_with_roster(
            &expected_subject,
            &[identity("care_ethics", "1.0.0")],
            &[
                care,
                assessment("rights_ethics", FrameworkStance::OpposesAction),
            ],
        );

        assert_eq!(result.state, ComparisonState::FrameworkSetMismatch);
        assert_eq!(
            result.unexpected_frameworks,
            vec![identity("rights_ethics", "1.0.0")]
        );
    }

    #[test]
    fn malformed_or_duplicate_roster_is_invalid() {
        let care = assessment("care_ethics", FrameworkStance::SupportsAction);
        let expected_subject = care.subject.clone();
        let result = compare_assessments_with_roster(
            &expected_subject,
            &[
                identity("care_ethics", "1.0.0"),
                identity("care_ethics", "1.0.0"),
            ],
            &[care],
        );
        assert_eq!(result.state, ComparisonState::InvalidInput);
        assert!(result.validation_errors.iter().any(|error| error.contains("duplicates")));
    }

    #[test]
    fn roster_mismatch_does_not_mask_subject_mismatch() {
        let care = assessment("care_ethics", FrameworkStance::SupportsAction);
        let mut rights = assessment("rights_ethics", FrameworkStance::SupportsAction);
        rights.subject.scenario_digest = "fixture-digest:different-context".to_owned();

        let result = compare_assessments_with_roster(
            &care.subject,
            &[identity("care_ethics", "1.0.0")],
            &[care, rights],
        );
        assert_eq!(result.state, ComparisonState::SubjectMismatch);
    }

    #[test]
    fn roster_order_does_not_change_coverage_or_state() {
        let care = assessment("care_ethics", FrameworkStance::SupportsAction);
        let rights = assessment("rights_ethics", FrameworkStance::OpposesAction);
        let subject = care.subject.clone();
        let forward = compare_assessments_with_roster(
            &subject,
            &[identity("care_ethics", "1.0.0"), identity("rights_ethics", "1.0.0")],
            &[care.clone(), rights.clone()],
        );
        let reversed = compare_assessments_with_roster(
            &subject,
            &[identity("rights_ethics", "1.0.0"), identity("care_ethics", "1.0.0")],
            &[rights, care],
        );

        assert_eq!(forward.state, reversed.state);
        assert_eq!(forward.counts, reversed.counts);
        assert_eq!(forward.missing_frameworks, reversed.missing_frameworks);
        assert_eq!(forward.unexpected_frameworks, reversed.unexpected_frameworks);
    }

    #[test]
    fn single_framework_cannot_claim_plural_agreement() {
        let result = compare_assessments(&[
            assessment("care_ethics", FrameworkStance::SupportsAction),
        ]);
        assert_eq!(result.state, ComparisonState::InsufficientFrameworks);
    }

    #[test]
    fn two_versions_of_one_framework_do_not_count_as_plurality() {
        let mut older = assessment("care_ethics", FrameworkStance::SupportsAction);
        let mut newer = assessment("care_ethics", FrameworkStance::OpposesAction);
        older.framework_version = "1.0.0".to_owned();
        newer.framework_version = "2.0.0".to_owned();

        let result = compare_assessments(&[older, newer]);
        assert_eq!(result.state, ComparisonState::InsufficientFrameworks);
    }

    #[test]
    fn empty_input_does_not_claim_agreement() {
        let result = compare_assessments(&[]);
        assert_eq!(result.state, ComparisonState::NoAssessments);
        assert_eq!(result.counts, AssessmentCounts::default());
    }

    #[test]
    fn unanimous_support_is_reported_without_authorizing_action() {
        let result = compare_assessments(&[
            assessment("care_ethics", FrameworkStance::SupportsAction),
            assessment("eight_harmonies", FrameworkStance::SupportsAction),
        ]);
        assert_eq!(result.state, ComparisonState::AgreementSupports);
        assert_eq!(result.counts.supports, 2);
        assert!(result.validation_errors.is_empty());
        // API deliberately has no permission/execute field.
    }

    #[test]
    fn opposing_frameworks_remain_in_explicit_disagreement() {
        let result = compare_assessments(&[
            assessment("consequentialist_demo", FrameworkStance::SupportsAction),
            assessment("rights_demo", FrameworkStance::OpposesAction),
        ]);
        assert_eq!(result.state, ComparisonState::Disagreement);
        assert_eq!(result.counts.supports, 1);
        assert_eq!(result.counts.opposes, 1);
        assert_eq!(result.assessments.len(), 2);
    }

    #[test]
    fn conditional_result_prevents_false_unanimity() {
        let result = compare_assessments(&[
            assessment("care_ethics", FrameworkStance::SupportsAction),
            assessment("duty_ethics", FrameworkStance::Conditional),
        ]);
        assert_eq!(result.state, ComparisonState::Incomplete);
    }

    #[test]
    fn unanimous_opposition_is_preserved() {
        let result = compare_assessments(&[
            assessment("care_ethics", FrameworkStance::OpposesAction),
            assessment("rights_ethics", FrameworkStance::OpposesAction),
        ]);
        assert_eq!(result.state, ComparisonState::AgreementOpposes);
        assert_eq!(result.counts.opposes, 2);
        // Opposition is a framework-relative analysis, not an execution gate.
    }

    #[test]
    fn comparison_state_is_independent_of_assessment_order() {
        let a = assessment("care_ethics", FrameworkStance::SupportsAction);
        let b = assessment("rights_ethics", FrameworkStance::OpposesAction);
        let forward = compare_assessments(&[a.clone(), b.clone()]);
        let reverse = compare_assessments(&[b, a]);

        assert_eq!(forward.state, reverse.state);
        assert_eq!(forward.counts, reverse.counts);
        assert_eq!(forward.state, ComparisonState::Disagreement);
    }

    #[test]
    fn different_scenarios_cannot_create_false_agreement() {
        let a = assessment("care_ethics", FrameworkStance::SupportsAction);
        let mut b = assessment("rights_ethics", FrameworkStance::SupportsAction);
        b.subject.scenario_ref = "scenario:case-002".to_owned();
        b.subject.scenario_digest = "fixture-digest:scenario-case-002".to_owned();

        let result = compare_assessments(&[a, b]);
        assert_eq!(result.state, ComparisonState::SubjectMismatch);
        assert!(result.validation_errors.is_empty());
    }

    #[test]
    fn same_scenario_reference_with_different_context_digest_is_not_comparable() {
        let a = assessment("care_ethics", FrameworkStance::SupportsAction);
        let mut b = assessment("rights_ethics", FrameworkStance::SupportsAction);
        b.subject.scenario_digest = "fixture-digest:changed-context".to_owned();

        let result = compare_assessments(&[a, b]);
        assert_eq!(result.state, ComparisonState::SubjectMismatch);
    }

    #[test]
    fn different_candidate_actions_cannot_create_false_agreement() {
        let a = assessment("care_ethics", FrameworkStance::SupportsAction);
        let mut b = assessment("rights_ethics", FrameworkStance::SupportsAction);
        b.subject.candidate_action_ref = "action:case-001:candidate-b".to_owned();
        b.subject.candidate_action_digest = "fixture-digest:candidate-action-b".to_owned();

        let result = compare_assessments(&[a, b]);
        assert_eq!(result.state, ComparisonState::SubjectMismatch);
    }

    #[test]
    fn blank_subject_binding_invalidates_comparison() {
        let mut malformed = assessment("care_ethics", FrameworkStance::SupportsAction);
        malformed.subject.candidate_action_digest.clear();

        let result = compare_assessments(&[malformed]);
        assert_eq!(result.state, ComparisonState::InvalidInput);
        assert!(result.validation_errors.iter().any(|e| e.contains("candidate_action_digest")));
    }

    #[test]
    fn partially_blank_premise_references_are_rejected() {
        let mut malformed = assessment("care_ethics", FrameworkStance::SupportsAction);
        malformed.premise_refs = vec![
            "premise:declared-principle-1".to_owned(),
            "  ".to_owned(),
        ];

        let result = compare_assessments(&[malformed]);
        assert_eq!(result.state, ComparisonState::InvalidInput);
        assert!(result.validation_errors.iter().any(|e| e.contains("premise_refs[1]")));
    }

    #[test]
    fn explicit_expected_subject_must_match_all_assessments() {
        let result = compare_assessments_for(
            &AssessmentSubject {
                scenario_ref: "scenario:case-999".to_owned(),
                scenario_digest: "fixture-digest:scenario-case-999".to_owned(),
                candidate_action_ref: "action:case-999:candidate-a".to_owned(),
                candidate_action_digest: "fixture-digest:candidate-action-a".to_owned(),
            },
            &[assessment("care_ethics", FrameworkStance::SupportsAction)],
        );
        assert_eq!(result.state, ComparisonState::SubjectMismatch);
    }

    #[test]
    fn carried_forward_assessment_cannot_claim_fresh_agreement() {
        let mut stale = assessment("care_ethics", FrameworkStance::SupportsAction);
        stale.provenance.freshness = AssessmentFreshness::CarriedForward;

        let result = compare_assessments(&[stale]);
        assert_eq!(result.state, ComparisonState::StaleAssessment);
    }

    #[test]
    fn source_subject_mismatch_cannot_be_relabelled_as_current() {
        let mut relabelled = assessment("care_ethics", FrameworkStance::SupportsAction);
        relabelled.provenance.source_subject.as_mut().unwrap().scenario_digest =
            "fixture-digest:old-source-context".to_owned();

        let result = compare_assessments(&[relabelled]);
        assert_eq!(result.state, ComparisonState::SourceSubjectMismatch);
    }

    #[test]
    fn unavailable_framework_is_incomplete_without_fake_premises() {
        let mut unavailable = assessment("eight_harmonies", FrameworkStance::Underdetermined);
        unavailable.provenance.source_subject = None;
        unavailable.provenance.evaluation_cycle = None;
        unavailable.provenance.freshness = AssessmentFreshness::Unavailable;
        unavailable.premise_refs.clear();
        unavailable.evidence_refs.clear();
        unavailable.rationale = vec!["No fresh evaluation was produced on this cycle.".to_owned()];

        let result = compare_assessments(&[
            assessment("care_ethics", FrameworkStance::SupportsAction),
            unavailable,
        ]);
        assert_eq!(result.state, ComparisonState::Incomplete);
        assert_eq!(result.counts.underdetermined, 1);
        assert!(result.validation_errors.is_empty());
    }

    #[test]
    fn unavailable_framework_must_not_claim_a_source_or_cycle() {
        let mut malformed = assessment("eight_harmonies", FrameworkStance::Underdetermined);
        malformed.provenance.freshness = AssessmentFreshness::Unavailable;

        let result = compare_assessments(&[malformed]);
        assert_eq!(result.state, ComparisonState::InvalidInput);
        assert!(result.validation_errors.iter().any(|e| e.contains("unavailable result must not claim a source_subject")));
        assert!(result.validation_errors.iter().any(|e| e.contains("unavailable result must not claim confidence")));
    }

    #[test]
    fn missing_evaluator_build_identity_invalidates_comparison() {
        let mut malformed = assessment("care_ethics", FrameworkStance::SupportsAction);
        malformed.provenance.evaluator_build_id.clear();

        let result = compare_assessments(&[malformed]);
        assert_eq!(result.state, ComparisonState::InvalidInput);
        assert!(result.validation_errors.iter().any(|e| e.contains("evaluator_build_id")));
    }

    #[test]
    fn duplicate_framework_version_invalidates_comparison() {
        let result = compare_assessments(&[
            assessment("care_ethics", FrameworkStance::SupportsAction),
            assessment("care_ethics", FrameworkStance::OpposesAction),
        ]);
        assert_eq!(result.state, ComparisonState::InvalidInput);
        assert!(!result.validation_errors.is_empty());
    }

    #[test]
    fn non_finite_confidence_invalidates_comparison() {
        let mut malformed = assessment("care_ethics", FrameworkStance::SupportsAction);
        malformed.confidence = Some(f64::NAN);
        let result = compare_assessments(&[malformed]);
        assert_eq!(result.state, ComparisonState::InvalidInput);
        assert!(result.validation_errors.iter().any(|e| e.contains("confidence")));
    }

    #[test]
    fn missing_rationale_or_premises_invalidates_comparison() {
        let mut malformed = assessment("care_ethics", FrameworkStance::SupportsAction);
        malformed.rationale.clear();
        malformed.premise_refs.clear();
        let result = compare_assessments(&[malformed]);
        assert_eq!(result.state, ComparisonState::InvalidInput);
        assert_eq!(result.validation_errors.len(), 2);
    }

    #[test]
    fn blank_evidence_reference_is_not_silently_accepted() {
        let mut malformed = assessment("care_ethics", FrameworkStance::SupportsAction);
        malformed.evidence_refs = vec!["  ".to_owned()];
        let result = compare_assessments(&[malformed]);
        assert_eq!(result.state, ComparisonState::InvalidInput);
    }
}
