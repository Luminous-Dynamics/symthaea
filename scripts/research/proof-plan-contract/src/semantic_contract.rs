//! Theorem-sandwich controls for statement-fidelity research.
//!
//! Candidate-visible data and evaluator-only controls are separate. A passing
//! result is consistency with selected finite propositional obligations only;
//! it is not full equivalence, a Lean kernel proof, or expert semantic review.

use crate::local_global::{
    BooleanAssignment, BooleanFormula, FiniteTruthTableResult,
    MAX_FINITE_BOOLEAN_ATOMS, check_finite_propositional_entailment,
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
    pub visible_assumptions: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorOnlyV1 {
    pub target_statement_formula: BooleanFormula,
    pub stronger_obligations: Vec<StrongerObligationV1>,
    pub corollary_obligations: Vec<CorollaryObligationV1>,
    pub adjudication_authority: String,
    pub adjudication_refs: Vec<String>,
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
    StrongerImpliesCandidate,
    CandidateImpliesCorollary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ObligationStatus {
    PassedFiniteTruthTable { assignments_checked: usize },
    CounterexampleFound {
        assignments_checked: usize,
        valuation: Vec<BooleanAssignment>,
    },
    UnknownAtomBudgetExceeded { atom_count: usize, max_atoms: usize },
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
}

impl SandwichClassification {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ConsistentWithSelectedControls => "consistent_with_selected_controls",
            Self::CandidateMayBeTooStrong => "candidate_may_be_too_strong",
            Self::CandidateMayBeTooWeak => "candidate_may_be_too_weak",
            Self::IncomparableOrTranslationMismatch => "incomparable_or_translation_mismatch",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TheoremSandwichReportV1 {
    pub task_id: String,
    pub upper_obligations: Vec<ObligationResult>,
    pub lower_obligations: Vec<ObligationResult>,
    pub classification: SandwichClassification,
    pub claim_ceiling: String,
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
    if candidate.visible_assumptions.iter().any(|value| !non_empty(value)) {
        errors.push("candidate.visible_assumptions must not contain empty entries".to_owned());
    }
    validate_formula(
        &candidate.statement_formula,
        "candidate.statement_formula",
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
            || evaluator.adjudication_refs.iter().any(|value| !non_empty(value))
        {
            errors.push("evaluator_only.adjudication_refs must contain non-empty references".to_owned());
        }

        let mut ids = HashSet::new();
        for obligation in &evaluator.stronger_obligations {
            if !non_empty(&obligation.id) || !ids.insert(obligation.id.as_str()) {
                errors.push(format!("invalid or duplicate stronger obligation ID: {}", obligation.id));
            }
            validate_formula(
                &obligation.stronger_formula,
                &format!("stronger obligation {}", obligation.id),
                &mut errors,
            );
        }
        for obligation in &evaluator.corollary_obligations {
            if !non_empty(&obligation.id) || !ids.insert(obligation.id.as_str()) {
                errors.push(format!("invalid or duplicate corollary obligation ID: {}", obligation.id));
            }
            validate_formula(
                &obligation.corollary_formula,
                &format!("corollary obligation {}", obligation.id),
                &mut errors,
            );
        }

        let mut refs = HashSet::new();
        if evaluator.adjudication_refs.iter().any(|value| !refs.insert(value)) {
            errors.push("evaluator_only.adjudication_refs must be unique".to_owned());
        }
        validate_formula(
            &evaluator.target_statement_formula,
            "evaluator target statement",
            &mut errors,
        );
    }
    errors
}

fn to_status(result: FiniteTruthTableResult) -> ObligationStatus {
    match result {
        FiniteTruthTableResult::Countermodel { valuation, assignments_checked } => {
            ObligationStatus::CounterexampleFound { assignments_checked, valuation }
        }
        FiniteTruthTableResult::NoCountermodelFound { assignments_checked } => {
            ObligationStatus::PassedFiniteTruthTable { assignments_checked }
        }
        FiniteTruthTableResult::AtomBudgetExceeded { atom_count, max_atoms } => {
            ObligationStatus::UnknownAtomBudgetExceeded { atom_count, max_atoms }
        }
    }
}

fn all_pass(results: &[ObligationResult]) -> bool {
    results.iter().all(|item| matches!(
        &item.status,
        ObligationStatus::PassedFiniteTruthTable { .. }
    ))
}
fn any_fail(results: &[ObligationResult]) -> bool {
    results.iter().any(|item| matches!(
        &item.status,
        ObligationStatus::CounterexampleFound { .. }
    ))
}
fn any_unknown(results: &[ObligationResult]) -> bool {
    results.iter().any(|item| matches!(
        &item.status,
        ObligationStatus::UnknownAtomBudgetExceeded { .. }
    ))
}

fn evaluate_obligation(
    id: &str,
    direction: ObligationDirection,
    premises: &[BooleanFormula],
    conclusion: &BooleanFormula,
) -> Result<ObligationResult, String> {
    let result = check_finite_propositional_entailment(
        premises,
        conclusion,
        MAX_FINITE_BOOLEAN_ATOMS,
    )?;
    Ok(ObligationResult { id: id.to_owned(), direction, status: to_status(result) })
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
        return Err(vec!["post-hoc theorem-sandwich evaluation requires evaluator_only data".to_owned()]);
    };

    let mut upper = Vec::new();
    for item in &evaluator.stronger_obligations {
        upper.push(evaluate_obligation(
            &item.id,
            ObligationDirection::StrongerImpliesCandidate,
            std::slice::from_ref(&item.stronger_formula),
            &receipt.candidate.statement_formula,
        ).map_err(|error| vec![format!("upper obligation {}: {error}", item.id)])?);
    }

    let mut lower = Vec::new();
    for item in &evaluator.corollary_obligations {
        lower.push(evaluate_obligation(
            &item.id,
            ObligationDirection::CandidateImpliesCorollary,
            std::slice::from_ref(&receipt.candidate.statement_formula),
            &item.corollary_formula,
        ).map_err(|error| vec![format!("lower obligation {}: {error}", item.id)])?);
    }

    let classification = if any_unknown(&upper) || any_unknown(&lower) {
        SandwichClassification::Unknown
    } else {
        match (all_pass(&upper), all_pass(&lower), any_fail(&upper), any_fail(&lower)) {
            (true, true, false, false) => SandwichClassification::ConsistentWithSelectedControls,
            (false, true, true, false) => SandwichClassification::CandidateMayBeTooStrong,
            (true, false, false, true) => SandwichClassification::CandidateMayBeTooWeak,
            _ => SandwichClassification::IncomparableOrTranslationMismatch,
        }
    };

    Ok(TheoremSandwichReportV1 {
        task_id: receipt.candidate.task_id.clone(),
        upper_obligations: upper,
        lower_obligations: lower,
        classification,
        claim_ceiling: "finite_propositional_selected_controls_only_not_full_equivalence_or_lean_kernel_proof".to_owned(),
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
        evaluate_receipt_json(FIXTURE).expect("valid theorem-sandwich fixture").0
    }
    fn atom(name: &str) -> BooleanFormula {
        BooleanFormula::Atom { name: name.to_owned() }
    }
    fn implies(left: BooleanFormula, right: BooleanFormula) -> BooleanFormula {
        BooleanFormula::Implies { left: Box::new(left), right: Box::new(right) }
    }
    fn and(left: BooleanFormula, right: BooleanFormula) -> BooleanFormula {
        BooleanFormula::And { left: Box::new(left), right: Box::new(right) }
    }

    #[test]
    fn exact_candidate_passes_selected_upper_and_lower_controls() {
        let report = evaluate_receipt(&fixture()).expect("well-formed fixture");
        assert_eq!(report.classification, SandwichClassification::ConsistentWithSelectedControls);
        assert!(all_pass(&report.upper_obligations));
        assert!(all_pass(&report.lower_obligations));
    }

    #[test]
    fn overstrengthened_candidate_fails_upper_obligation() {
        let mut receipt = fixture();
        receipt.candidate.statement_formula = and(implies(atom("p"), atom("q")), atom("s"));
        let report = evaluate_receipt(&receipt).expect("well-formed stronger candidate");
        assert_eq!(report.classification, SandwichClassification::CandidateMayBeTooStrong);
        assert!(any_fail(&report.upper_obligations));
        assert!(all_pass(&report.lower_obligations));
    }

    #[test]
    fn weakened_candidate_fails_a_hidden_corollary() {
        let mut receipt = fixture();
        receipt.candidate.statement_formula = implies(atom("r"), implies(atom("p"), atom("q")));
        let report = evaluate_receipt(&receipt).expect("well-formed weaker candidate");
        assert_eq!(report.classification, SandwichClassification::CandidateMayBeTooWeak);
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
        assert!(evaluate_receipt(&receipt).unwrap_err().iter().any(|error| {
            error.contains("requires evaluator_only")
        }));
    }

    #[test]
    fn malformed_hidden_obligation_is_rejected() {
        let mut receipt = fixture();
        receipt.evaluator_only.as_mut().unwrap().stronger_obligations[0].id.clear();
        assert!(validate_receipt(&receipt).iter().any(|error| {
            error.contains("invalid or duplicate stronger obligation ID")
        }));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut value: Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
        value["candidate"]["untrusted_gold"] = Value::Bool(true);
        let input = serde_json::to_string(&value).expect("serialize mutation");
        assert!(evaluate_receipt_json(&input).unwrap_err().iter().any(|error| {
            error.contains("unknown field")
        }));
    }
}
