//! Theorem-sandwich controls for statement-fidelity research.
//!
//! Candidate-visible data and evaluator-only controls are separate. A passing
//! result is consistency with selected finite propositional obligations only;
//! it is not full equivalence, a Lean kernel proof, or expert semantic review.

use crate::local_global::{
    BooleanAssignment, BooleanFormula, FiniteTruthTableResult, MAX_FINITE_BOOLEAN_ATOMS,
    check_finite_propositional_entailment,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[cfg(test)]
const FIXTURE: &str =
    include_str!("../../../../docs/research/MATH_PROOF_COMPREHENSION_THEOREM_SANDWICH_V1.json");

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TheoremSandwichReceiptV1 {
    pub schema_version: String,
    pub candidate: CandidateStatementV1,
    pub evaluator_only: Option<EvaluatorOnlyV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateStatementV1 {
    pub task_id: String,
    pub theorem_id: String,
    pub source_repository: String,
    pub source_revision: String,
    pub source_path: String,
    pub declaration: String,
    pub model_provider: String,
    pub model_name: String,
    pub model_version: String,
    pub statement_text: String,
    pub statement_formula: BooleanFormula,
    /// Formal assumptions in the sequent Γ ⊢ statement_formula.
    pub statement_context: Vec<BooleanFormula>,
    /// Human-readable context description only; not used as the logical context.
    pub visible_assumptions: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorOnlyV1 {
    pub target_statement_formula: BooleanFormula,
    /// Hidden formal assumptions for the target sequent Γ* ⊢ P*.
    pub target_statement_context: Vec<BooleanFormula>,
    pub stronger_obligations: Vec<StrongerObligationV1>,
    pub corollary_obligations: Vec<CorollaryObligationV1>,
    pub translation_assessment: TranslationAssessment,
    pub translation_assessor: String,
    pub translation_review_refs: Vec<String>,
    pub adjudication_authority: String,
    pub adjudication_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TranslationAssessment {
    Aligned,
    PartiallyAligned,
    Misaligned,
    Unknown,
    NotReviewed,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StrongerObligationV1 {
    pub id: String,
    pub stronger_formula: BooleanFormula,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorollaryObligationV1 {
    pub id: String,
    pub corollary_formula: BooleanFormula,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObligationDirection {
    StrongerImpliesTarget,
    TargetImpliesCorollary,
    StrongerImpliesCandidate,
    CandidateImpliesCorollary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ObligationStatus {
    PassedFiniteTruthTable {
        assignments_checked: usize,
    },
    CounterexampleFound {
        assignments_checked: usize,
        valuation: Vec<BooleanAssignment>,
    },
    UnknownAtomBudgetExceeded {
        atom_count: usize,
        max_atoms: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ObligationResult {
    pub id: String,
    pub direction: ObligationDirection,
    pub status: ObligationStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SandwichClassification {
    ConsistentWithSelectedControls,
    CandidateMayBeTooStrong,
    CandidateMayBeTooWeak,
    IncomparableOrTranslationMismatch,
    Unknown,
    InvalidEvaluatorControls,
}

impl SandwichClassification {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ConsistentWithSelectedControls => "consistent_with_selected_controls",
            Self::CandidateMayBeTooStrong => "candidate_may_be_too_strong",
            Self::CandidateMayBeTooWeak => "candidate_may_be_too_weak",
            Self::IncomparableOrTranslationMismatch => "incomparable_or_translation_mismatch",
            Self::Unknown => "unknown",
            Self::InvalidEvaluatorControls => "invalid_evaluator_controls",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TheoremSandwichReportV1 {
    pub task_id: String,
    pub evaluator_control_obligations: Vec<ObligationResult>,
    pub upper_obligations: Vec<ObligationResult>,
    pub lower_obligations: Vec<ObligationResult>,
    pub classification: SandwichClassification,
    pub claim_ceiling: String,
    pub candidate_context_assumption_count: usize,
    pub target_context_assumption_count: usize,
    pub translation_assessment: TranslationAssessment,
}

impl TheoremSandwichReceiptV1 {
    /// Only the candidate payload may be sent to a model.
    pub fn candidate_view(&self) -> &CandidateStatementV1 {
        &self.candidate
    }
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn collect_atoms(formula: &BooleanFormula, atoms: &mut HashSet<String>) -> Result<(), String> {
    match formula {
        BooleanFormula::Atom { name } => {
            if !non_empty(name) {
                return Err("formula atom name must be non-empty".to_owned());
            }
            atoms.insert(name.clone());
        }
        BooleanFormula::Not { arg } => collect_atoms(arg, atoms)?,
        BooleanFormula::And { left, right }
        | BooleanFormula::Or { left, right }
        | BooleanFormula::Implies { left, right }
        | BooleanFormula::Iff { left, right } => {
            collect_atoms(left, atoms)?;
            collect_atoms(right, atoms)?;
        }
    }
    Ok(())
}

fn validate_formula(formula: &BooleanFormula, label: &str, errors: &mut Vec<String>) {
    let mut atoms = HashSet::new();
    if let Err(error) = collect_atoms(formula, &mut atoms) {
        errors.push(format!("{label}: {error}"));
    }
}

/// Interpret a formal theorem statement as a sequent Γ ⊢ P, whose classical
/// propositional meaning is (∧ Γ) → P. Human-readable assumption strings are
/// deliberately not parsed or trusted as logical premises.
fn sequent_formula(context: &[BooleanFormula], conclusion: &BooleanFormula) -> BooleanFormula {
    let Some(first) = context.first() else {
        return conclusion.clone();
    };

    let antecedent = context
        .iter()
        .skip(1)
        .cloned()
        .fold(first.clone(), |left, right| BooleanFormula::And {
            left: Box::new(left),
            right: Box::new(right),
        });

    BooleanFormula::Implies {
        left: Box::new(antecedent),
        right: Box::new(conclusion.clone()),
    }
}

fn validate_context(context: &[BooleanFormula], label: &str, errors: &mut Vec<String>) {
    for (index, formula) in context.iter().enumerate() {
        validate_formula(formula, &format!("{label}[{index}]"), errors);
    }
}

pub fn validate_receipt(receipt: &TheoremSandwichReceiptV1) -> Vec<String> {
    let mut errors = Vec::new();
    if receipt.schema_version != "1.0.0" {
        errors.push("schema_version must equal 1.0.0".to_owned());
    }

    let candidate = &receipt.candidate;
    for (field, value) in [
        ("task_id", &candidate.task_id),
        ("theorem_id", &candidate.theorem_id),
        ("source_repository", &candidate.source_repository),
        ("source_revision", &candidate.source_revision),
        ("source_path", &candidate.source_path),
        ("declaration", &candidate.declaration),
        ("model_provider", &candidate.model_provider),
        ("model_name", &candidate.model_name),
        ("model_version", &candidate.model_version),
        ("statement_text", &candidate.statement_text),
    ] {
        if !non_empty(value) {
            errors.push(format!("candidate.{field} must be non-empty"));
        }
    }
    if candidate
        .visible_assumptions
        .iter()
        .any(|value| !non_empty(value))
    {
        errors.push("candidate.visible_assumptions must not contain empty entries".to_owned());
    }
    validate_formula(
        &candidate.statement_formula,
        "candidate.statement_formula",
        &mut errors,
    );
    validate_context(
        &candidate.statement_context,
        "candidate.statement_context",
        &mut errors,
    );

    if let Some(evaluator) = &receipt.evaluator_only {
        if evaluator.stronger_obligations.is_empty() {
            errors.push("evaluator_only requires at least one stronger obligation".to_owned());
        }
        if evaluator.corollary_obligations.is_empty() {
            errors.push("evaluator_only requires at least one corollary obligation".to_owned());
        }
        if !non_empty(&evaluator.adjudication_authority) {
            errors.push("evaluator_only.adjudication_authority must be non-empty".to_owned());
        }
        if evaluator.adjudication_refs.is_empty()
            || evaluator
                .adjudication_refs
                .iter()
                .any(|value| !non_empty(value))
        {
            errors.push(
                "evaluator_only.adjudication_refs must contain non-empty references".to_owned(),
            );
        }

        let mut ids = HashSet::new();
        for obligation in &evaluator.stronger_obligations {
            if !non_empty(&obligation.id) || !ids.insert(obligation.id.as_str()) {
                errors.push(format!(
                    "invalid or duplicate stronger obligation ID: {}",
                    obligation.id
                ));
            }
            validate_formula(
                &obligation.stronger_formula,
                &format!("stronger obligation {}", obligation.id),
                &mut errors,
            );
        }
        for obligation in &evaluator.corollary_obligations {
            if !non_empty(&obligation.id) || !ids.insert(obligation.id.as_str()) {
                errors.push(format!(
                    "invalid or duplicate corollary obligation ID: {}",
                    obligation.id
                ));
            }
            validate_formula(
                &obligation.corollary_formula,
                &format!("corollary obligation {}", obligation.id),
                &mut errors,
            );
        }

        let mut refs = HashSet::new();
        if evaluator
            .adjudication_refs
            .iter()
            .any(|value| !refs.insert(value))
        {
            errors.push("evaluator_only.adjudication_refs must be unique".to_owned());
        }
        validate_formula(
            &evaluator.target_statement_formula,
            "evaluator target statement",
            &mut errors,
        );
        validate_context(
            &evaluator.target_statement_context,
            "evaluator target_statement_context",
            &mut errors,
        );
        if !non_empty(&evaluator.translation_assessor) {
            errors.push("evaluator_only.translation_assessor must be non-empty".to_owned());
        }
        if evaluator.translation_review_refs.is_empty()
            || evaluator
                .translation_review_refs
                .iter()
                .any(|value| !non_empty(value))
        {
            errors.push(
                "evaluator_only.translation_review_refs must contain non-empty references"
                    .to_owned(),
            );
        }
        let mut translation_refs = HashSet::new();
        if evaluator
            .translation_review_refs
            .iter()
            .any(|value| !translation_refs.insert(value))
        {
            errors.push("evaluator_only.translation_review_refs must be unique".to_owned());
        }
    }
    errors
}

fn to_status(result: FiniteTruthTableResult) -> ObligationStatus {
    match result {
        FiniteTruthTableResult::Countermodel {
            valuation,
            assignments_checked,
        } => ObligationStatus::CounterexampleFound {
            assignments_checked,
            valuation,
        },
        FiniteTruthTableResult::NoCountermodelFound {
            assignments_checked,
        } => ObligationStatus::PassedFiniteTruthTable {
            assignments_checked,
        },
        FiniteTruthTableResult::AtomBudgetExceeded {
            atom_count,
            max_atoms,
        } => ObligationStatus::UnknownAtomBudgetExceeded {
            atom_count,
            max_atoms,
        },
    }
}

fn all_pass(results: &[ObligationResult]) -> bool {
    results.iter().all(|item| {
        matches!(
            &item.status,
            ObligationStatus::PassedFiniteTruthTable { .. }
        )
    })
}
fn any_fail(results: &[ObligationResult]) -> bool {
    results
        .iter()
        .any(|item| matches!(&item.status, ObligationStatus::CounterexampleFound { .. }))
}
fn any_unknown(results: &[ObligationResult]) -> bool {
    results.iter().any(|item| {
        matches!(
            &item.status,
            ObligationStatus::UnknownAtomBudgetExceeded { .. }
        )
    })
}

fn evaluate_obligation(
    id: &str,
    direction: ObligationDirection,
    premises: &[BooleanFormula],
    conclusion: &BooleanFormula,
) -> Result<ObligationResult, String> {
    let result =
        check_finite_propositional_entailment(premises, conclusion, MAX_FINITE_BOOLEAN_ATOMS)?;
    Ok(ObligationResult {
        id: id.to_owned(),
        direction,
        status: to_status(result),
    })
}

/// Success means consistency with selected finite controls only.
pub fn evaluate_receipt(
    receipt: &TheoremSandwichReceiptV1,
) -> Result<TheoremSandwichReportV1, Vec<String>> {
    let errors = validate_receipt(receipt);
    if !errors.is_empty() {
        return Err(errors);
    }
    let Some(evaluator) = &receipt.evaluator_only else {
        return Err(vec![
            "post-hoc theorem-sandwich evaluation requires evaluator_only data".to_owned(),
        ]);
    };

    let target_sequent = sequent_formula(
        &evaluator.target_statement_context,
        &evaluator.target_statement_formula,
    );
    let candidate_sequent = sequent_formula(
        &receipt.candidate.statement_context,
        &receipt.candidate.statement_formula,
    );

    let mut evaluator_controls = Vec::new();
    for item in &evaluator.stronger_obligations {
        evaluator_controls.push(
            evaluate_obligation(
                &format!("CONTROL-{}", item.id),
                ObligationDirection::StrongerImpliesTarget,
                std::slice::from_ref(&item.stronger_formula),
                &target_sequent,
            )
            .map_err(|error| vec![format!("evaluator stronger control {}: {error}", item.id)])?,
        );
    }
    for item in &evaluator.corollary_obligations {
        evaluator_controls.push(
            evaluate_obligation(
                &format!("CONTROL-{}", item.id),
                ObligationDirection::TargetImpliesCorollary,
                std::slice::from_ref(&target_sequent),
                &item.corollary_formula,
            )
            .map_err(|error| vec![format!("evaluator corollary control {}: {error}", item.id)])?,
        );
    }

    let mut upper = Vec::new();
    for item in &evaluator.stronger_obligations {
        upper.push(
            evaluate_obligation(
                &item.id,
                ObligationDirection::StrongerImpliesCandidate,
                std::slice::from_ref(&item.stronger_formula),
                &candidate_sequent,
            )
            .map_err(|error| vec![format!("upper obligation {}: {error}", item.id)])?,
        );
    }

    let mut lower = Vec::new();
    for item in &evaluator.corollary_obligations {
        lower.push(
            evaluate_obligation(
                &item.id,
                ObligationDirection::CandidateImpliesCorollary,
                std::slice::from_ref(&candidate_sequent),
                &item.corollary_formula,
            )
            .map_err(|error| vec![format!("lower obligation {}: {error}", item.id)])?,
        );
    }

    let classification = if any_fail(&evaluator_controls) {
        SandwichClassification::InvalidEvaluatorControls
    } else if evaluator.translation_assessment == TranslationAssessment::Misaligned {
        SandwichClassification::IncomparableOrTranslationMismatch
    } else if evaluator.translation_assessment != TranslationAssessment::Aligned
        || any_unknown(&evaluator_controls)
        || any_unknown(&upper)
        || any_unknown(&lower)
    {
        SandwichClassification::Unknown
    } else {
        match (
            all_pass(&upper),
            all_pass(&lower),
            any_fail(&upper),
            any_fail(&lower),
        ) {
            (true, true, false, false) => SandwichClassification::ConsistentWithSelectedControls,
            (false, true, true, false) => SandwichClassification::CandidateMayBeTooStrong,
            (true, false, false, true) => SandwichClassification::CandidateMayBeTooWeak,
            _ => SandwichClassification::IncomparableOrTranslationMismatch,
        }
    };

    Ok(TheoremSandwichReportV1 {
        task_id: receipt.candidate.task_id.clone(),
        evaluator_control_obligations: evaluator_controls,
        upper_obligations: upper,
        lower_obligations: lower,
        classification,
        claim_ceiling:
            "finite_propositional_context_indexed_selected_controls_only_translation_reviewed_not_full_equivalence_or_lean_kernel_proof"
                .to_owned(),
        candidate_context_assumption_count: receipt.candidate.statement_context.len(),
        target_context_assumption_count: evaluator.target_statement_context.len(),
        translation_assessment: evaluator.translation_assessment,
    })
}

pub fn evaluate_receipt_json(
    input: &str,
) -> Result<(TheoremSandwichReceiptV1, TheoremSandwichReportV1), Vec<String>> {
    let receipt: TheoremSandwichReceiptV1 = serde_json::from_str(input)
        .map_err(|error| vec![format!("invalid theorem-sandwich receipt JSON: {error}")])?;
    let report = evaluate_receipt(&receipt)?;
    Ok((receipt, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> TheoremSandwichReceiptV1 {
        evaluate_receipt_json(FIXTURE)
            .expect("valid theorem-sandwich fixture")
            .0
    }
    fn atom(name: &str) -> BooleanFormula {
        BooleanFormula::Atom {
            name: name.to_owned(),
        }
    }
    fn implies(left: BooleanFormula, right: BooleanFormula) -> BooleanFormula {
        BooleanFormula::Implies {
            left: Box::new(left),
            right: Box::new(right),
        }
    }
    fn and(left: BooleanFormula, right: BooleanFormula) -> BooleanFormula {
        BooleanFormula::And {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    #[test]
    fn hidden_stronger_control_must_imply_hidden_target() {
        let mut receipt = fixture();
        receipt
            .evaluator_only
            .as_mut()
            .unwrap()
            .stronger_obligations[0]
            .stronger_formula = atom("p");
        let report = evaluate_receipt(&receipt).expect("syntactically valid receipt");
        assert_eq!(
            report.classification,
            SandwichClassification::InvalidEvaluatorControls
        );
        assert!(any_fail(&report.evaluator_control_obligations));
    }

    #[test]
    fn hidden_target_must_imply_each_selected_corollary() {
        let mut receipt = fixture();
        receipt
            .evaluator_only
            .as_mut()
            .unwrap()
            .corollary_obligations[0]
            .corollary_formula = atom("q");
        let report = evaluate_receipt(&receipt).expect("syntactically valid receipt");
        assert_eq!(
            report.classification,
            SandwichClassification::InvalidEvaluatorControls
        );
        assert!(any_fail(&report.evaluator_control_obligations));
    }

    #[test]
    fn valid_hidden_controls_are_checked_before_candidate_classification() {
        let report = evaluate_receipt(&fixture()).expect("valid synthetic receipt");
        assert!(all_pass(&report.evaluator_control_obligations));
        assert_eq!(
            report.classification,
            SandwichClassification::ConsistentWithSelectedControls
        );
    }

    #[test]
    fn exact_candidate_passes_selected_upper_and_lower_controls() {
        let report = evaluate_receipt(&fixture()).expect("well-formed fixture");
        assert_eq!(
            report.classification,
            SandwichClassification::ConsistentWithSelectedControls
        );
        assert!(all_pass(&report.upper_obligations));
        assert!(all_pass(&report.lower_obligations));
    }

    #[test]
    fn overstrengthened_candidate_fails_upper_obligation() {
        let mut receipt = fixture();
        receipt.candidate.statement_formula = and(implies(atom("p"), atom("q")), atom("s"));
        let report = evaluate_receipt(&receipt).expect("well-formed stronger candidate");
        assert_eq!(
            report.classification,
            SandwichClassification::CandidateMayBeTooStrong
        );
        assert!(any_fail(&report.upper_obligations));
        assert!(all_pass(&report.lower_obligations));
    }

    #[test]
    fn weakened_candidate_fails_a_hidden_corollary() {
        let mut receipt = fixture();
        receipt.candidate.statement_formula = implies(atom("r"), implies(atom("p"), atom("q")));
        let report = evaluate_receipt(&receipt).expect("well-formed weaker candidate");
        assert_eq!(
            report.classification,
            SandwichClassification::CandidateMayBeTooWeak
        );
        assert!(all_pass(&report.upper_obligations));
        assert!(any_fail(&report.lower_obligations));
    }

    #[test]
    fn evaluator_only_obligations_are_absent_from_candidate_projection() {
        let receipt = fixture();
        let visible = serde_json::to_value(receipt.candidate_view()).expect("candidate projection");
        assert!(visible.get("evaluator_only").is_none());
        assert!(visible.get("stronger_obligations").is_none());
        assert!(visible.get("corollary_obligations").is_none());
    }

    #[test]
    fn missing_evaluator_partition_is_not_a_pass() {
        let mut receipt = fixture();
        receipt.evaluator_only = None;
        assert!(
            evaluate_receipt(&receipt)
                .unwrap_err()
                .iter()
                .any(|error| { error.contains("requires evaluator_only") })
        );
    }

    #[test]
    fn malformed_hidden_obligation_is_rejected() {
        let mut receipt = fixture();
        receipt
            .evaluator_only
            .as_mut()
            .unwrap()
            .stronger_obligations[0]
            .id
            .clear();
        assert!(
            validate_receipt(&receipt)
                .iter()
                .any(|error| { error.contains("invalid or duplicate stronger obligation ID") })
        );
    }

    #[test]
    fn moving_a_hypothesis_between_context_and_implication_preserves_sequent_meaning() {
        let mut receipt = fixture();
        receipt.candidate.statement_formula = atom("q");
        receipt.candidate.statement_context = vec![atom("p")];
        let report = evaluate_receipt(&receipt).expect("well-formed candidate sequent");
        assert_eq!(
            report.classification,
            SandwichClassification::ConsistentWithSelectedControls
        );
        assert_eq!(report.candidate_context_assumption_count, 1);
        assert_eq!(report.target_context_assumption_count, 0);
    }

    #[test]
    fn adding_an_extra_candidate_assumption_can_make_a_statement_too_weak() {
        let mut receipt = fixture();
        receipt.candidate.statement_context = vec![atom("r")];
        let report = evaluate_receipt(&receipt).expect("well-formed candidate sequent");
        assert_eq!(
            report.classification,
            SandwichClassification::CandidateMayBeTooWeak
        );
        assert!(any_fail(&report.lower_obligations));
    }

    #[test]
    fn removing_a_required_context_assumption_can_make_a_statement_too_strong() {
        let mut receipt = fixture();
        let evaluator = receipt.evaluator_only.as_mut().unwrap();
        evaluator.target_statement_formula = atom("q");
        evaluator.target_statement_context = vec![atom("p")];
        evaluator.stronger_obligations[0].stronger_formula = implies(atom("p"), atom("q"));
        evaluator.corollary_obligations[0].corollary_formula =
            implies(and(atom("p"), atom("r")), atom("q"));
        evaluator.corollary_obligations[1].corollary_formula = implies(
            and(
                atom("p"),
                BooleanFormula::Not {
                    arg: Box::new(atom("r")),
                },
            ),
            atom("q"),
        );
        receipt.candidate.statement_formula = atom("q");
        receipt.candidate.statement_context.clear();

        let report = evaluate_receipt(&receipt).expect("well-formed context mutation");
        assert_eq!(
            report.classification,
            SandwichClassification::CandidateMayBeTooStrong
        );
        assert!(any_fail(&report.upper_obligations));
    }

    #[test]
    fn a_misaligned_text_to_formula_translation_blocks_semantic_acceptance() {
        let mut receipt = fixture();
        receipt
            .evaluator_only
            .as_mut()
            .unwrap()
            .translation_assessment = TranslationAssessment::Misaligned;
        let report = evaluate_receipt(&receipt).expect("valid but misaligned candidate receipt");
        assert_eq!(
            report.classification,
            SandwichClassification::IncomparableOrTranslationMismatch
        );
    }

    #[test]
    fn partial_or_unreviewed_translation_is_unknown_not_a_pass() {
        for status in [
            TranslationAssessment::PartiallyAligned,
            TranslationAssessment::Unknown,
            TranslationAssessment::NotReviewed,
        ] {
            let mut receipt = fixture();
            receipt
                .evaluator_only
                .as_mut()
                .unwrap()
                .translation_assessment = status;
            let report = evaluate_receipt(&receipt).expect("valid unresolved translation receipt");
            assert_eq!(report.classification, SandwichClassification::Unknown);
        }
    }

    #[test]
    fn malformed_candidate_context_formula_is_rejected() {
        let mut receipt = fixture();
        receipt.candidate.statement_context = vec![atom("  ")];
        assert!(
            validate_receipt(&receipt)
                .iter()
                .any(|error| error.contains("candidate.statement_context[0]"))
        );
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut value: Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
        value["candidate"]["untrusted_gold"] = Value::Bool(true);
        let input = serde_json::to_string(&value).expect("serialize mutation");
        assert!(
            evaluate_receipt_json(&input)
                .unwrap_err()
                .iter()
                .any(|error| { error.contains("unknown field") })
        );
    }
}
