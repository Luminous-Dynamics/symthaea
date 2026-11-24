//! Hard-mode receipts keep answer, formalization, finite semantics and proof
//! verification as separate evidence stages. The evaluator partition is
//! post-hoc and must never be passed to the candidate.

use crate::local_global::{
    BooleanCountermodel, BooleanSequent, FiniteTruthTableResult, MAX_FINITE_BOOLEAN_ATOMS,
    ModelIdentity, check_finite_boolean_sequent,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

#[cfg(test)]
const FIXTURE: &str = include_str!("../../../../docs/research/MATH_HARDMODE_SYNTHETIC_V1.json");
#[cfg(test)]
const SCHEMA: &str = include_str!("../../../../docs/research/MATH_HARDMODE_RECEIPT_V1.schema.json");

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HardModeReceiptV1 {
    pub schema_version: String,
    pub candidate: HardModeCandidateV1,
    pub evaluator_only: Option<HardModeEvaluatorV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HardModeCandidateV1 {
    pub task: HardModeTaskV1,
    pub model_identity: ModelIdentity,
    pub answer_text: String,
    pub answer_sha256: String,
    pub answer_abstained: bool,
    pub formalization: Option<CandidateFormalizationV1>,
    pub candidate_countermodel: Option<BooleanCountermodel>,
    pub proof_submission: Option<ArtifactSubmissionV1>,
    pub proof_plan_submission: Option<ArtifactSubmissionV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HardModeTaskV1 {
    pub benchmark_id: String,
    pub benchmark_revision: String,
    pub task_id: String,
    pub task_variant: TaskVariant,
    pub problem_text: String,
    pub problem_sha256: String,
    pub prompt_text: String,
    pub prompt_sha256: String,
    pub visible_answer: Option<String>,
    pub formal_source_repository: String,
    pub formal_source_revision: String,
    pub formal_source_path: String,
    pub formal_declaration: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskVariant {
    AnswerProvided,
    NoAnswer,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateFormalizationV1 {
    pub statement_text: String,
    pub statement_sha256: String,
    pub finite_boolean_sequent: Option<BooleanSequent>,
    pub finite_boolean_sequent_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSubmissionV1 {
    pub artifact_ref: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HardModeEvaluatorV1 {
    pub answer_correct: Option<bool>,
    pub answer_evidence_refs: Vec<String>,
    pub formalization_matches_target: Option<bool>,
    pub formalization_evidence_refs: Vec<String>,
    pub gold_countermodel: Option<BooleanCountermodel>,
    pub proof_status: ProofStatus,
    pub proof_verification: Option<ProofVerificationV1>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProofStatus {
    NotRun,
    Accepted,
    Rejected,
    Error,
    Timeout,
    Inconclusive,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProofVerificationV1 {
    pub checker: String,
    pub checker_version: String,
    pub toolchain: String,
    pub environment_manifest_sha256: String,
    pub verified_task_id: String,
    pub verified_formal_source_revision: String,
    pub verified_statement_sha256: String,
    pub verified_proof_sha256: String,
    pub result_artifact_ref: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum FiniteSemanticStatus {
    CountermodelFound {
        assignments_checked: usize,
        witness: Vec<crate::local_global::BooleanAssignment>,
    },
    NoCountermodelFound {
        assignments_checked: usize,
    },
    UnknownAtomBudgetExceeded {
        atom_count: usize,
        max_atoms: usize,
    },
    NotApplicable,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CandidateWitnessStatus {
    NotSubmitted,
    Validated,
    Invalid { reason: String },
    NotApplicable,
}

#[derive(Debug, Clone, Serialize)]
pub struct HardModeEvaluationV1 {
    pub task_id: String,
    pub task_variant: TaskVariant,
    pub answer_correct: Option<bool>,
    pub formalization_matches_target: Option<bool>,
    pub finite_semantics: FiniteSemanticStatus,
    pub candidate_witness: CandidateWitnessStatus,
    pub proof_status: ProofStatus,
    pub claim_ceiling: String,
}

impl HardModeReceiptV1 {
    /// Only the candidate partition is visible to the model or prover.
    pub fn candidate_view(&self) -> &HardModeCandidateV1 {
        &self.candidate
    }

    pub fn candidate_view_sha256(&self) -> Result<String, String> {
        domain_separated_sha256(
            "candidate_payload",
            &(&self.schema_version, &self.candidate),
        )
    }
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn domain_separated_sha256<T: Serialize>(domain: &str, preimage: &T) -> Result<String, String> {
    let canonical = serde_json::to_vec(preimage)
        .map_err(|error| format!("could not serialize {domain} preimage: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.hard_mode_receipt.v1:");
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(canonical);
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn validate_digest<T: Serialize>(
    label: &str,
    actual: &str,
    domain: &str,
    preimage: &T,
    errors: &mut Vec<String>,
) {
    if !valid_sha256(actual) {
        errors.push(format!("{label} must be lowercase SHA-256 hex"));
        return;
    }
    match domain_separated_sha256(domain, preimage) {
        Ok(expected) if expected == actual => {}
        Ok(_) => errors.push(format!("{label} does not match its canonical preimage")),
        Err(error) => errors.push(format!("{label}: {error}")),
    }
}

fn validate_refs(label: &str, refs: &[String], errors: &mut Vec<String>) {
    if refs.is_empty() || refs.iter().any(|value| !non_empty(value)) {
        errors.push(format!(
            "{label} must contain non-empty evidence references"
        ));
        return;
    }
    let mut unique = HashSet::new();
    if refs.iter().any(|value| !unique.insert(value)) {
        errors.push(format!("{label} must be unique"));
    }
}

pub fn validate_receipt(receipt: &HardModeReceiptV1) -> Vec<String> {
    let mut errors = Vec::new();
    if receipt.schema_version != "1.0.0" {
        errors.push("schema_version must equal 1.0.0".to_owned());
    }

    let candidate = &receipt.candidate;
    let task = &candidate.task;
    for (field, value) in [
        ("benchmark_id", &task.benchmark_id),
        ("benchmark_revision", &task.benchmark_revision),
        ("task_id", &task.task_id),
        ("problem_text", &task.problem_text),
        ("prompt_text", &task.prompt_text),
        ("formal_source_repository", &task.formal_source_repository),
        ("formal_source_revision", &task.formal_source_revision),
        ("formal_source_path", &task.formal_source_path),
        ("formal_declaration", &task.formal_declaration),
        (
            "model_identity.provider",
            &candidate.model_identity.provider,
        ),
        ("model_identity.model", &candidate.model_identity.model),
        ("model_identity.version", &candidate.model_identity.version),
        ("answer_text", &candidate.answer_text),
    ] {
        if !non_empty(value) {
            errors.push(format!("{field} must be non-empty"));
        }
    }
    if !valid_sha256(&candidate.model_identity.decoding_config_sha256) {
        errors
            .push("model_identity.decoding_config_sha256 must be lowercase SHA-256 hex".to_owned());
    }

    validate_digest(
        "problem_sha256",
        &task.problem_sha256,
        "problem",
        &task.problem_text,
        &mut errors,
    );
    validate_digest(
        "prompt_sha256",
        &task.prompt_sha256,
        "prompt",
        &(&task.prompt_text, &task.visible_answer),
        &mut errors,
    );
    match task.task_variant {
        TaskVariant::AnswerProvided => {
            if task
                .visible_answer
                .as_ref()
                .is_none_or(|value| !non_empty(value))
            {
                errors.push("answer_provided task requires visible_answer".to_owned());
            }
        }
        TaskVariant::NoAnswer => {
            if task.visible_answer.is_some() {
                errors.push("no_answer task must not expose visible_answer".to_owned());
            }
        }
    }
    validate_digest(
        "answer_sha256",
        &candidate.answer_sha256,
        "answer",
        &candidate.answer_text,
        &mut errors,
    );

    if let Some(formalization) = &candidate.formalization {
        if !non_empty(&formalization.statement_text) {
            errors.push("formalization.statement_text must be non-empty".to_owned());
        }
        validate_digest(
            "formalization.statement_sha256",
            &formalization.statement_sha256,
            "formal_statement",
            &formalization.statement_text,
            &mut errors,
        );
        match (
            &formalization.finite_boolean_sequent,
            &formalization.finite_boolean_sequent_sha256,
        ) {
            (Some(sequent), Some(digest)) => {
                validate_digest(
                    "formalization.finite_boolean_sequent_sha256",
                    digest,
                    "finite_boolean_sequent",
                    sequent,
                    &mut errors,
                );
                if sequent.premises.is_empty() {
                    errors.push("finite_boolean_sequent must have at least one premise".to_owned());
                }
            }
            (None, None) => {}
            _ => errors.push(
                "finite_boolean_sequent and its digest must both be present or both be null"
                    .to_owned(),
            ),
        }
    } else if candidate.candidate_countermodel.is_some()
        || candidate.proof_submission.is_some()
        || candidate.proof_plan_submission.is_some()
    {
        errors.push("countermodel, proof and plan submissions require a formalization".to_owned());
    }

    if let Some(countermodel) = &candidate.candidate_countermodel {
        let sequent = candidate
            .formalization
            .as_ref()
            .and_then(|formalization| formalization.finite_boolean_sequent.as_ref());
        if sequent.is_none() {
            errors.push("candidate_countermodel requires a finite Boolean sequent".to_owned());
        } else {
            // Candidate-witness correctness is scored; keep invalid work visible.
            let _ = countermodel.validate(sequent.unwrap());
        }
    }

    if let Some(proof) = &candidate.proof_submission {
        if !non_empty(&proof.artifact_ref) || !valid_sha256(&proof.sha256) {
            errors
                .push("proof_submission requires an artifact_ref and lowercase SHA-256".to_owned());
        }
    }
    if let Some(plan) = &candidate.proof_plan_submission {
        if !non_empty(&plan.artifact_ref) || !valid_sha256(&plan.sha256) {
            errors.push(
                "proof_plan_submission requires an artifact_ref and lowercase SHA-256".to_owned(),
            );
        }
    }

    if let Some(evaluator) = &receipt.evaluator_only {
        if evaluator.answer_correct.is_some() {
            validate_refs(
                "answer_evidence_refs",
                &evaluator.answer_evidence_refs,
                &mut errors,
            );
        }
        if evaluator.formalization_matches_target.is_some() {
            validate_refs(
                "formalization_evidence_refs",
                &evaluator.formalization_evidence_refs,
                &mut errors,
            );
        }
        if let Some(gold) = &evaluator.gold_countermodel {
            match candidate
                .formalization
                .as_ref()
                .and_then(|formalization| formalization.finite_boolean_sequent.as_ref())
            {
                Some(sequent) => {
                    if let Err(reason) = gold.validate(sequent) {
                        errors.push(format!("gold_countermodel is invalid: {reason}"));
                    }
                }
                None => errors
                    .push("gold_countermodel requires a finite Boolean formalization".to_owned()),
            }
        }

        match (&evaluator.proof_status, &evaluator.proof_verification) {
            (ProofStatus::NotRun, None) => {}
            (ProofStatus::NotRun, Some(_)) => {
                errors.push("not_run proof status must not carry checker evidence".to_owned());
            }
            (status, None) => {
                errors.push(format!(
                    "{status:?} proof status requires verification evidence"
                ));
            }
            (status, Some(verification)) => {
                for (field, value) in [
                    ("checker", &verification.checker),
                    ("checker_version", &verification.checker_version),
                    ("toolchain", &verification.toolchain),
                    ("verified_task_id", &verification.verified_task_id),
                    (
                        "verified_formal_source_revision",
                        &verification.verified_formal_source_revision,
                    ),
                    (
                        "verified_statement_sha256",
                        &verification.verified_statement_sha256,
                    ),
                    ("verified_proof_sha256", &verification.verified_proof_sha256),
                    ("result_artifact_ref", &verification.result_artifact_ref),
                ] {
                    if !non_empty(value) {
                        errors.push(format!("proof_verification.{field} must be non-empty"));
                    }
                }
                if !valid_sha256(&verification.environment_manifest_sha256) {
                    errors.push(
                        "proof_verification.environment_manifest_sha256 must be SHA-256".to_owned(),
                    );
                }
                validate_refs(
                    "proof_verification.evidence_refs",
                    &verification.evidence_refs,
                    &mut errors,
                );
                if candidate.proof_submission.is_none() {
                    errors.push(
                        "proof verification requires a candidate proof submission".to_owned(),
                    );
                }
                if verification.verified_task_id != task.task_id {
                    errors.push("proof verification task identity mismatch".to_owned());
                }
                if verification.verified_formal_source_revision != task.formal_source_revision {
                    errors.push("proof verification source revision mismatch".to_owned());
                }
                let formalization = candidate.formalization.as_ref();
                if formalization.is_none()
                    || verification.verified_statement_sha256
                        != formalization.unwrap().statement_sha256
                {
                    errors.push("proof verification formal statement digest mismatch".to_owned());
                }
                if let Some(proof) = &candidate.proof_submission {
                    if verification.verified_proof_sha256 != proof.sha256 {
                        errors.push("proof verification proof-source digest mismatch".to_owned());
                    }
                }
                if *status == ProofStatus::Accepted && verification.checker != "lean-kernel" {
                    errors.push("accepted proof requires lean-kernel authority".to_owned());
                }
                if *status == ProofStatus::Accepted && evaluator.gold_countermodel.is_some() {
                    errors.push(
                        "accepted proof conflicts with evaluator gold countermodel".to_owned(),
                    );
                }
            }
        }
    }

    errors
}

pub fn parse_receipt(input: &str) -> Result<HardModeReceiptV1, String> {
    serde_json::from_str(input).map_err(|error| format!("invalid HardModeReceiptV1 JSON: {error}"))
}

pub fn evaluate_receipt(receipt: &HardModeReceiptV1) -> Result<HardModeEvaluationV1, Vec<String>> {
    let errors = validate_receipt(receipt);
    if !errors.is_empty() {
        return Err(errors);
    }
    let Some(evaluator) = &receipt.evaluator_only else {
        return Err(vec![
            "evaluator_only partition required for post-hoc evaluation".to_owned(),
        ]);
    };

    let (finite_semantics, candidate_witness) = match receipt
        .candidate
        .formalization
        .as_ref()
        .and_then(|formalization| formalization.finite_boolean_sequent.as_ref())
    {
        Some(sequent) => {
            let result = check_finite_boolean_sequent(sequent, MAX_FINITE_BOOLEAN_ATOMS);
            let semantics = match result {
                Ok(FiniteTruthTableResult::Countermodel {
                    valuation,
                    assignments_checked,
                }) => FiniteSemanticStatus::CountermodelFound {
                    assignments_checked,
                    witness: valuation,
                },
                Ok(FiniteTruthTableResult::NoCountermodelFound {
                    assignments_checked,
                }) => FiniteSemanticStatus::NoCountermodelFound {
                    assignments_checked,
                },
                Ok(FiniteTruthTableResult::AtomBudgetExceeded {
                    atom_count,
                    max_atoms,
                }) => FiniteSemanticStatus::UnknownAtomBudgetExceeded {
                    atom_count,
                    max_atoms,
                },
                Err(_) => FiniteSemanticStatus::NotApplicable,
            };
            let witness = match &receipt.candidate.candidate_countermodel {
                None => CandidateWitnessStatus::NotSubmitted,
                Some(countermodel) => match countermodel.validate(sequent) {
                    Ok(()) => CandidateWitnessStatus::Validated,
                    Err(reason) => CandidateWitnessStatus::Invalid { reason },
                },
            };
            (semantics, witness)
        }
        None => (
            FiniteSemanticStatus::NotApplicable,
            CandidateWitnessStatus::NotApplicable,
        ),
    };

    Ok(HardModeEvaluationV1 {
        task_id: receipt.candidate.task.task_id.clone(),
        task_variant: receipt.candidate.task.task_variant,
        answer_correct: evaluator.answer_correct,
        formalization_matches_target: evaluator.formalization_matches_target,
        finite_semantics,
        candidate_witness,
        proof_status: evaluator.proof_status,
        claim_ceiling: "answer discovery, formalization, finite semantics and proof verification are separate; a finite witness is not a Lean kernel proof".to_owned(),
    })
}

pub fn evaluate_receipt_json(input: &str) -> Result<HardModeEvaluationV1, Vec<String>> {
    let receipt = parse_receipt(input).map_err(|error| vec![error])?;
    evaluate_receipt(&receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> HardModeReceiptV1 {
        let receipt = parse_receipt(FIXTURE).expect("valid hard-mode fixture");
        let errors = validate_receipt(&receipt);
        assert!(errors.is_empty(), "invalid hard-mode fixture: {errors:?}");
        receipt
    }

    #[test]
    fn schema_and_fixture_are_valid_json() {
        let _: Value = serde_json::from_str(SCHEMA).expect("schema JSON");
        let _: Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
    }

    #[test]
    fn answer_and_finite_semantics_do_not_imply_lean_proof() {
        let receipt = fixture();
        let result = evaluate_receipt(&receipt).expect("valid hard-mode evaluation");
        assert_eq!(result.answer_correct, Some(true));
        assert_eq!(result.formalization_matches_target, Some(true));
        assert_eq!(result.proof_status, ProofStatus::NotRun);
        assert!(matches!(
            result.finite_semantics,
            FiniteSemanticStatus::CountermodelFound { .. }
        ));
        assert_eq!(result.candidate_witness, CandidateWitnessStatus::Validated);
    }

    #[test]
    fn candidate_projection_excludes_gold_and_proof_adjudication() {
        let receipt = fixture();
        let visible = serde_json::to_value(receipt.candidate_view()).expect("candidate view");
        assert!(visible.get("evaluator_only").is_none());
        assert!(visible.get("gold_countermodel").is_none());
        assert!(visible.get("answer_correct").is_none());
        assert!(visible.to_string().contains("No; set p=false and q=true"));
    }

    #[test]
    fn candidate_view_digest_ignores_evaluator_only_changes() {
        let receipt = fixture();
        let digest = receipt.candidate_view_sha256().unwrap();
        let mut changed = receipt.clone();
        changed.evaluator_only.as_mut().unwrap().answer_correct = Some(false);
        assert_eq!(digest, changed.candidate_view_sha256().unwrap());
    }

    #[test]
    fn candidate_view_digest_changes_when_candidate_output_changes() {
        let receipt = fixture();
        let digest = receipt.candidate_view_sha256().unwrap();
        let mut changed = receipt.clone();
        changed.candidate.answer_text = "incorrect".to_owned();
        changed.candidate.answer_sha256 =
            domain_separated_sha256("answer", &changed.candidate.answer_text).unwrap();
        assert_ne!(digest, changed.candidate_view_sha256().unwrap());
    }

    #[test]
    fn no_answer_variant_must_not_expose_a_visible_answer() {
        let mut receipt = fixture();
        receipt.candidate.task.visible_answer = Some("hidden answer".to_owned());
        receipt.candidate.task.prompt_sha256 = domain_separated_sha256(
            "prompt",
            &(
                &receipt.candidate.task.prompt_text,
                &receipt.candidate.task.visible_answer,
            ),
        )
        .unwrap();
        assert!(
            validate_receipt(&receipt)
                .iter()
                .any(|error| error.contains("no_answer task must not expose"))
        );
    }

    #[test]
    fn answer_provided_variant_requires_a_visible_answer() {
        let mut receipt = fixture();
        receipt.candidate.task.task_variant = TaskVariant::AnswerProvided;
        assert!(
            validate_receipt(&receipt)
                .iter()
                .any(|error| error.contains("answer_provided task requires"))
        );
    }

    #[test]
    fn invalid_candidate_witness_is_scored_not_silently_accepted() {
        let mut receipt = fixture();
        receipt
            .candidate
            .candidate_countermodel
            .as_mut()
            .unwrap()
            .valuation
            .iter_mut()
            .find(|item| item.atom == "p")
            .unwrap()
            .value = true;
        let result =
            evaluate_receipt(&receipt).expect("invalid candidate witness remains scorable");
        assert!(matches!(
            result.candidate_witness,
            CandidateWitnessStatus::Invalid { .. }
        ));
        assert_eq!(result.answer_correct, Some(true));
    }

    #[test]
    fn completed_proof_check_requires_a_candidate_submission() {
        let mut receipt = fixture();
        receipt.evaluator_only.as_mut().unwrap().proof_status = ProofStatus::Accepted;
        assert!(
            validate_receipt(&receipt)
                .iter()
                .any(|error| error.contains("requires verification evidence"))
        );
    }

    #[test]
    fn accepted_proof_cannot_conflict_with_gold_countermodel() {
        let mut receipt = fixture();
        receipt.candidate.proof_submission = Some(ArtifactSubmissionV1 {
            artifact_ref: "artifact://candidate/lean-proof".to_owned(),
            sha256: "c".repeat(64),
        });
        let evaluator = receipt.evaluator_only.as_mut().unwrap();
        evaluator.proof_status = ProofStatus::Accepted;
        evaluator.proof_verification = Some(ProofVerificationV1 {
            checker: "lean-kernel".to_owned(),
            checker_version: "4.37.0-rc1".to_owned(),
            toolchain: "leanprover/lean4:v4.37.0-rc1".to_owned(),
            environment_manifest_sha256: "b".repeat(64),
            verified_task_id: receipt.candidate.task.task_id.clone(),
            verified_formal_source_revision: receipt.candidate.task.formal_source_revision.clone(),
            verified_statement_sha256: receipt
                .candidate
                .formalization
                .as_ref()
                .unwrap()
                .statement_sha256
                .clone(),
            verified_proof_sha256: "c".repeat(64),
            result_artifact_ref: "artifact://lean-result".to_owned(),
            evidence_refs: vec!["LEAN-KERNEL-CHECK-001".to_owned()],
        });
        assert!(
            validate_receipt(&receipt)
                .iter()
                .any(|error| error.contains("conflicts with evaluator gold countermodel"))
        );
    }

    #[test]
    fn unknown_answer_adjudication_remains_unknown() {
        let mut receipt = fixture();
        let evaluator = receipt.evaluator_only.as_mut().unwrap();
        evaluator.answer_correct = None;
        evaluator.answer_evidence_refs.clear();
        let report = evaluate_receipt(&receipt).unwrap();
        assert_eq!(report.answer_correct, None);
    }


    #[test]
    fn answer_text_mutation_is_rejected_without_digest_update() {
        let mut receipt = fixture();
        receipt.candidate.answer_text.push_str(" This is a mutation.");
        assert!(validate_receipt(&receipt)
            .iter()
            .any(|error| error.contains("answer_sha256 does not match its canonical preimage")));
    }

    #[test]
    fn formal_statement_mutation_is_rejected_without_digest_update() {
        let mut receipt = fixture();
        receipt
            .candidate
            .formalization
            .as_mut()
            .unwrap()
            .statement_text
            .push_str(" and therefore the converse holds.");
        assert!(validate_receipt(&receipt).iter().any(|error| {
            error.contains("formalization.statement_sha256 does not match its canonical preimage")
        }));
    }

    #[test]
    fn unknown_json_fields_are_rejected() {
        let mut value: Value = serde_json::from_str(FIXTURE).unwrap();
        value["candidate"]["untrusted_override"] = Value::Bool(true);
        let input = serde_json::to_string(&value).unwrap();
        assert!(parse_receipt(&input).unwrap_err().contains("unknown field"));
    }
}
