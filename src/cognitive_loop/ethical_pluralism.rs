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
    /// At least one result is mixed, conditional, or underdetermined, so agreement
    /// cannot be asserted from the supplied results.
    Incomplete,
    /// An assessment is malformed or the same framework/version appears twice.
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
    pub validation_errors: Vec<String>,
}

/// Compare previously produced framework assessments.
///
/// This function deliberately performs no score averaging, ranking, moral
/// arbitration, or action authorization. Frameworks that disagree remain in
/// disagreement. A malformed assessment invalidates the comparison rather than
/// silently dropping the bad row.
pub fn compare_assessments(assessments: &[FrameworkAssessment]) -> PluralEthicsComparison {
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
        validation_errors,
    }
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
        if assessment.premise_refs.iter().all(|premise| premise.trim().is_empty()) {
            errors.push(format!("{prefix}: premise_refs must contain a non-empty premise/rule ID"));
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
            stance,
            confidence: Some(0.8),
            rationale: vec!["The conclusion follows from the declared profile premises.".to_owned()],
            premise_refs: vec!["premise:declared-principle-1".to_owned()],
            evidence_refs: vec!["scenario:case-001".to_owned()],
            unresolved_questions: Vec::new(),
        }
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
