//! Paired local/global evidence receipts for proof-consistency evaluation.
//!
//! Candidate-visible outputs and evaluator-only adjudication are different
//! types. candidate_view intentionally projects away evaluator-only fields.
//! Structural validity is not a Lean-kernel or mathematical correctness claim.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[cfg(test)]
const SYNTHETIC_FIXTURE: &str =
    include_str!("../../../../docs/research/LOCAL_GLOBAL_PAIR_SYNTHETIC_V1.json");
#[cfg(test)]
const SYNTHETIC_SCHEMA: &str =
    include_str!("../../../../docs/research/LOCAL_GLOBAL_PAIR_V1.schema.json");

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LocalGlobalPairV1 {
    pub schema_version: String,
    pub candidate: CandidatePayload,
    pub evaluator_only: Option<EvaluatorReceipt>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidatePayload {
    pub pair_id: String,
    pub task_class: String,
    pub model_identity: ModelIdentity,
    pub local_stage: LocalStage,
    pub global_stage: GlobalStage,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub provider: String,
    pub model: String,
    pub version: String,
    pub decoding_config_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScopeBinding {
    pub source_repository: String,
    pub source_revision: String,
    pub source_path: String,
    pub declaration: String,
    pub theorem_id: String,
    pub root_goal_sha256: String,
    pub hypothesis_set_sha256: String,
    pub premise_set_sha256: String,
    pub conclusion_sha256: String,
    pub warrant_id: String,
    pub warrant_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionRecord {
    pub prompt_sha256: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub output_artifact_ref: String,
    pub token_budget: u64,
    pub tokens_used: u64,
    pub tool_call_budget: u64,
    pub tool_calls_used: u64,
    pub wall_clock_budget_ms: u64,
    pub wall_clock_used_ms: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LocalStage {
    pub scope_binding: ScopeBinding,
    pub execution: ExecutionRecord,
    pub verdict: LocalVerdict,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LocalVerdict {
    Accept,
    Reject,
    Unknown,
    IllFormed,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalStage {
    pub scope_binding: ScopeBinding,
    pub execution: ExecutionRecord,
    pub proof_plan_sha256: String,
    pub warrant_mentions: Vec<WarrantMention>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WarrantMention {
    pub mention_id: String,
    pub warrant_id: String,
    pub plan_node_id: String,
    pub relation: MentionRelation,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MentionRelation {
    UsesAsRule,
    SupportsInference,
    Cites,
    Rebuts,
    ContextOnly,
    Alternative,
}


#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum BooleanFormula {
    Atom { name: String },
    Not { arg: Box<BooleanFormula> },
    And {
        left: Box<BooleanFormula>,
        right: Box<BooleanFormula>,
    },
    Or {
        left: Box<BooleanFormula>,
        right: Box<BooleanFormula>,
    },
    Implies {
        left: Box<BooleanFormula>,
        right: Box<BooleanFormula>,
    },
    Iff {
        left: Box<BooleanFormula>,
        right: Box<BooleanFormula>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BooleanAssignment {
    pub atom: String,
    pub value: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BooleanCountermodel {
    pub premises: Vec<BooleanFormula>,
    pub conclusion: BooleanFormula,
    pub valuation: Vec<BooleanAssignment>,
}

impl BooleanFormula {
    fn collect_atoms(&self, atoms: &mut HashSet<String>) -> Result<(), String> {
        match self {
            Self::Atom { name } => {
                if !non_empty(name) {
                    return Err("Boolean formula atom name must be non-empty".to_owned());
                }
                atoms.insert(name.clone());
            }
            Self::Not { arg } => arg.collect_atoms(atoms)?,
            Self::And { left, right }
            | Self::Or { left, right }
            | Self::Implies { left, right }
            | Self::Iff { left, right } => {
                left.collect_atoms(atoms)?;
                right.collect_atoms(atoms)?;
            }
        }
        Ok(())
    }

    fn evaluate(&self, valuation: &std::collections::HashMap<String, bool>) -> Result<bool, String> {
        match self {
            Self::Atom { name } => valuation
                .get(name)
                .copied()
                .ok_or_else(|| format!("countermodel has no valuation for atom {name}")),
            Self::Not { arg } => Ok(!arg.evaluate(valuation)?),
            Self::And { left, right } => {
                Ok(left.evaluate(valuation)? && right.evaluate(valuation)?)
            }
            Self::Or { left, right } => {
                Ok(left.evaluate(valuation)? || right.evaluate(valuation)?)
            }
            Self::Implies { left, right } => {
                Ok(!left.evaluate(valuation)? || right.evaluate(valuation)?)
            }
            Self::Iff { left, right } => {
                Ok(left.evaluate(valuation)? == right.evaluate(valuation)?)
            }
        }
    }
}

impl BooleanCountermodel {
    /// Confirm that every premise is true and the conclusion is false under
    /// one complete valuation. This is a finite semantic witness, not a Lean
    /// kernel receipt or a proof about non-propositional mathematics.
    pub fn validate(&self) -> Result<(), String> {
        if self.premises.is_empty() {
            return Err("countermodel must contain at least one premise".to_owned());
        }

        let mut values = std::collections::HashMap::new();
        for assignment in &self.valuation {
            if !non_empty(&assignment.atom) {
                return Err("countermodel valuation atom must be non-empty".to_owned());
            }
            if values.insert(assignment.atom.clone(), assignment.value).is_some() {
                return Err(format!(
                    "countermodel contains duplicate valuation for atom {}",
                    assignment.atom
                ));
            }
        }

        let mut atoms = HashSet::new();
        for premise in &self.premises {
            premise.collect_atoms(&mut atoms)?;
        }
        self.conclusion.collect_atoms(&mut atoms)?;
        for atom in atoms {
            if !values.contains_key(&atom) {
                return Err(format!("countermodel has no valuation for atom {atom}"));
            }
        }

        for (index, premise) in self.premises.iter().enumerate() {
            if !premise.evaluate(&values)? {
                return Err(format!(
                    "countermodel premise[{index}] evaluates to false"
                ));
            }
        }
        if self.conclusion.evaluate(&values)? {
            return Err("countermodel conclusion evaluates to true".to_owned());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorReceipt {
    pub eligibility: EvaluationEligibility,
    pub exclusion_reason: Option<String>,
    pub independently_assessed_warrant: WarrantValidity,
    pub local_verdict_correct: Option<bool>,
    pub global_disposition: GlobalDisposition,
    pub adjudication_authority: String,
    pub adjudication_refs: Vec<String>,
    pub finite_boolean_countermodel: Option<BooleanCountermodel>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationEligibility {
    Eligible,
    Excluded,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WarrantValidity {
    Valid,
    Invalid,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GlobalDisposition {
    UsedDirectly,
    UsedImplicitly,
    CitedRejected,
    Omitted,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct CirrSummary {
    pub numerator_definite_reuse: usize,
    pub denominator_eligible_local_rejections: usize,
    pub unresolved_global_outcomes: usize,
    /// Lower bound treats unresolved global outcomes as non-reuse.
    pub lower_bound: Option<f64>,
    /// Upper bound treats unresolved global outcomes as material reuse.
    pub upper_bound: Option<f64>,
    /// Point estimate exists only when the denominator is nonzero and no
    /// eligible global outcomes remain unresolved.
    pub point_estimate: Option<f64>,
}

impl LocalGlobalPairV1 {
    /// Only this projection is suitable for a candidate-visible handoff.
    /// It cannot serialize the evaluator-only partition.
    pub fn candidate_view(&self) -> &CandidatePayload {
        &self.candidate
    }

    pub fn scope_matches(&self) -> bool {
        self.candidate.local_stage.scope_binding == self.candidate.global_stage.scope_binding
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

fn validate_scope(scope: &ScopeBinding, errors: &mut Vec<String>, label: &str) {
    for (field, value) in [
        ("source_repository", &scope.source_repository),
        ("source_revision", &scope.source_revision),
        ("source_path", &scope.source_path),
        ("declaration", &scope.declaration),
        ("theorem_id", &scope.theorem_id),
        ("warrant_id", &scope.warrant_id),
    ] {
        if !non_empty(value) {
            errors.push(format!("{label}.{field} must be non-empty"));
        }
    }

    for (field, digest) in [
        ("root_goal_sha256", &scope.root_goal_sha256),
        ("hypothesis_set_sha256", &scope.hypothesis_set_sha256),
        ("premise_set_sha256", &scope.premise_set_sha256),
        ("conclusion_sha256", &scope.conclusion_sha256),
        ("warrant_sha256", &scope.warrant_sha256),
    ] {
        if !valid_sha256(digest) {
            errors.push(format!("{label}.{field} must be lowercase SHA-256 hex"));
        }
    }
}

fn validate_execution(execution: &ExecutionRecord, errors: &mut Vec<String>, label: &str) {
    for (field, digest) in [
        ("prompt_sha256", &execution.prompt_sha256),
        ("input_sha256", &execution.input_sha256),
        ("output_sha256", &execution.output_sha256),
    ] {
        if !valid_sha256(digest) {
            errors.push(format!("{label}.{field} must be lowercase SHA-256 hex"));
        }
    }
    if !non_empty(&execution.output_artifact_ref) {
        errors.push(format!("{label}.output_artifact_ref must be non-empty"));
    }
    if execution.tokens_used > execution.token_budget {
        errors.push(format!("{label}.tokens_used exceeds token_budget"));
    }
    if execution.tool_calls_used > execution.tool_call_budget {
        errors.push(format!("{label}.tool_calls_used exceeds tool_call_budget"));
    }
    if execution.wall_clock_used_ms > execution.wall_clock_budget_ms {
        errors.push(format!(
            "{label}.wall_clock_used_ms exceeds wall_clock_budget_ms"
        ));
    }
}

pub fn validate_pair(pair: &LocalGlobalPairV1) -> Vec<String> {
    let mut errors = Vec::new();

    if pair.schema_version != "1.0.0" {
        errors.push("schema_version must equal 1.0.0".to_owned());
    }

    let candidate = &pair.candidate;
    for (field, value) in [
        ("pair_id", &candidate.pair_id),
        ("task_class", &candidate.task_class),
        (
            "model_identity.provider",
            &candidate.model_identity.provider,
        ),
        ("model_identity.model", &candidate.model_identity.model),
        ("model_identity.version", &candidate.model_identity.version),
    ] {
        if !non_empty(value) {
            errors.push(format!("candidate.{field} must be non-empty"));
        }
    }
    if !valid_sha256(&candidate.model_identity.decoding_config_sha256) {
        errors.push(
            "candidate.model_identity.decoding_config_sha256 must be lowercase SHA-256 hex"
                .to_owned(),
        );
    }

    validate_scope(
        &candidate.local_stage.scope_binding,
        &mut errors,
        "candidate.local_stage.scope_binding",
    );
    validate_scope(
        &candidate.global_stage.scope_binding,
        &mut errors,
        "candidate.global_stage.scope_binding",
    );
    validate_execution(
        &candidate.local_stage.execution,
        &mut errors,
        "candidate.local_stage.execution",
    );
    validate_execution(
        &candidate.global_stage.execution,
        &mut errors,
        "candidate.global_stage.execution",
    );

    if !valid_sha256(&candidate.global_stage.proof_plan_sha256) {
        errors.push(
            "candidate.global_stage.proof_plan_sha256 must be lowercase SHA-256 hex".to_owned(),
        );
    }

    let mut mention_ids = HashSet::new();
    for mention in &candidate.global_stage.warrant_mentions {
        if !non_empty(&mention.mention_id) {
            errors.push("global warrant mention ID must be non-empty".to_owned());
        }
        if !mention_ids.insert(&mention.mention_id) {
            errors.push(format!(
                "duplicate warrant mention ID: {}",
                mention.mention_id
            ));
        }
        if !non_empty(&mention.warrant_id) || !non_empty(&mention.plan_node_id) {
            errors.push(format!(
                "warrant mention {} has an empty warrant or plan-node ID",
                mention.mention_id
            ));
        }
    }

    if let Some(evaluator) = &pair.evaluator_only {
        if !non_empty(&evaluator.adjudication_authority) {
            errors.push("evaluator_only.adjudication_authority must be non-empty".to_owned());
        }
        if evaluator.adjudication_refs.is_empty()
            || evaluator
                .adjudication_refs
                .iter()
                .any(|reference| !non_empty(reference))
        {
            errors.push(
                "evaluator_only.adjudication_refs must contain non-empty evidence references"
                    .to_owned(),
            );
        }
        let mut adjudication_refs = HashSet::new();
        if evaluator
            .adjudication_refs
            .iter()
            .any(|reference| !adjudication_refs.insert(reference))
        {
            errors.push("evaluator_only.adjudication_refs must be unique".to_owned());
        }

        if let Some(countermodel) = &evaluator.finite_boolean_countermodel {
            if let Err(reason) = countermodel.validate() {
                errors.push(format!("evaluator_only.finite_boolean_countermodel: {reason}"));
            }
            if evaluator.independently_assessed_warrant != WarrantValidity::Invalid {
                errors.push(
                    "finite Boolean countermodel requires independently invalid warrant".to_owned(),
                );
            }
        }
        if evaluator.adjudication_authority == "finite_boolean_countermodel"
            && evaluator.finite_boolean_countermodel.is_none()
        {
            errors.push(
                "finite_boolean_countermodel authority requires a finite Boolean witness".to_owned(),
            );
        }

        match evaluator.eligibility {
            EvaluationEligibility::Eligible => {
                if evaluator
                    .exclusion_reason
                    .as_ref()
                    .is_some_and(|reason| non_empty(reason))
                {
                    errors.push("eligible pair must not have an exclusion_reason".to_owned());
                }
                if !pair.scope_matches() {
                    errors.push("scope mismatch cannot be marked eligible for CIRR".to_owned());
                }
                if evaluator.independently_assessed_warrant != WarrantValidity::Invalid {
                    errors.push(
                        "CIRR-eligible pair requires an independently invalid warrant".to_owned(),
                    );
                }
                if candidate.local_stage.verdict != LocalVerdict::Reject {
                    errors.push("CIRR-eligible pair requires a local reject verdict".to_owned());
                }
                if evaluator.local_verdict_correct != Some(true) {
                    errors.push("CIRR-eligible pair requires a correct local rejection".to_owned());
                }
            }
            EvaluationEligibility::Excluded => {
                if evaluator
                    .exclusion_reason
                    .as_ref()
                    .is_none_or(|reason| !non_empty(reason))
                {
                    errors.push("excluded pair requires a non-empty exclusion_reason".to_owned());
                }
            }
        }
    }

    errors
}

pub fn validate_pair_json(input: &str) -> Result<LocalGlobalPairV1, Vec<String>> {
    let pair: LocalGlobalPairV1 = serde_json::from_str(input)
        .map_err(|error| vec![format!("invalid LocalGlobalPairV1 JSON: {error}")])?;
    let errors = validate_pair(&pair);
    if errors.is_empty() {
        Ok(pair)
    } else {
        Err(errors)
    }
}

/// Compute CIRR only from independently adjudicated, validated receipts.
/// Unknown global dispositions remain in the denominator and widen the interval;
/// they are never silently imputed as reuse or non-reuse.
pub fn compute_cirr(pairs: &[LocalGlobalPairV1]) -> Result<CirrSummary, Vec<String>> {
    let mut errors = Vec::new();
    for (index, pair) in pairs.iter().enumerate() {
        errors.extend(
            validate_pair(pair)
                .into_iter()
                .map(|error| format!("pair[{index}]: {error}")),
        );
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let mut numerator = 0usize;
    let mut denominator = 0usize;
    let mut unresolved = 0usize;

    for pair in pairs {
        let Some(evaluator) = &pair.evaluator_only else {
            continue;
        };
        if evaluator.eligibility != EvaluationEligibility::Eligible
            || !pair.scope_matches()
            || evaluator.independently_assessed_warrant != WarrantValidity::Invalid
            || pair.candidate.local_stage.verdict != LocalVerdict::Reject
        {
            continue;
        }

        denominator += 1;
        match evaluator.global_disposition {
            GlobalDisposition::UsedDirectly | GlobalDisposition::UsedImplicitly => numerator += 1,
            GlobalDisposition::Unknown => unresolved += 1,
            GlobalDisposition::CitedRejected | GlobalDisposition::Omitted => {}
        }
    }

    let lower_bound = (denominator > 0).then(|| numerator as f64 / denominator as f64);
    let upper_bound =
        (denominator > 0).then(|| (numerator + unresolved) as f64 / denominator as f64);
    let point_estimate =
        (denominator > 0 && unresolved == 0).then(|| numerator as f64 / denominator as f64);

    Ok(CirrSummary {
        numerator_definite_reuse: numerator,
        denominator_eligible_local_rejections: denominator,
        unresolved_global_outcomes: unresolved,
        lower_bound,
        upper_bound,
        point_estimate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> LocalGlobalPairV1 {
        validate_pair_json(SYNTHETIC_FIXTURE).expect("valid synthetic receipt fixture")
    }

    #[test]
    fn schema_and_synthetic_fixture_are_valid_json() {
        let _: Value = serde_json::from_str(SYNTHETIC_SCHEMA).expect("schema JSON");
        let _: Value = serde_json::from_str(SYNTHETIC_FIXTURE).expect("fixture JSON");
    }

    #[test]
    fn synthetic_receipt_is_structurally_valid() {
        let pair = fixture();
        assert_eq!(pair.candidate.pair_id, "PAIR-SYNTHETIC-P-TO-Q-001");
        assert!(pair.scope_matches());
        assert_eq!(pair.candidate.local_stage.verdict, LocalVerdict::Reject);
        assert_eq!(
            pair.evaluator_only
                .as_ref()
                .unwrap()
                .independently_assessed_warrant,
            WarrantValidity::Invalid
        );
    }

    #[test]
    fn candidate_projection_excludes_evaluator_only_adjudication() {
        let pair = fixture();
        let visible =
            serde_json::to_value(pair.candidate_view()).expect("serialize candidate view");
        assert!(visible.get("evaluator_only").is_none());
        assert!(visible.get("independently_assessed_warrant").is_none());
        assert!(!visible.to_string().contains("finite_boolean_countermodel"));
        assert!(!visible.to_string().contains("valuation"));
    }


    #[test]
    fn converse_implication_countermodel_is_evaluated_not_trusted_as_text() {
        let pair = fixture();
        let witness = pair
            .evaluator_only
            .as_ref()
            .unwrap()
            .finite_boolean_countermodel
            .as_ref()
            .unwrap();
        assert!(witness.validate().is_ok());
        let premise_value = witness.premises[0]
            .evaluate(
                &witness
                    .valuation
                    .iter()
                    .map(|assignment| (assignment.atom.clone(), assignment.value))
                    .collect(),
            )
            .unwrap();
        assert!(premise_value);
    }

    #[test]
    fn invalid_valuation_is_not_accepted_as_a_countermodel() {
        let mut pair = fixture();
        let witness = pair
            .evaluator_only
            .as_mut()
            .unwrap()
            .finite_boolean_countermodel
            .as_mut()
            .unwrap();
        witness
            .valuation
            .iter_mut()
            .find(|assignment| assignment.atom == "p")
            .unwrap()
            .value = true;
        assert!(
            validate_pair(&pair)
                .iter()
                .any(|error| error.contains("countermodel conclusion evaluates to true"))
        );
    }

    #[test]
    fn duplicate_boolean_atom_assignments_are_rejected() {
        let mut pair = fixture();
        let witness = pair
            .evaluator_only
            .as_mut()
            .unwrap()
            .finite_boolean_countermodel
            .as_mut()
            .unwrap();
        witness.valuation.push(witness.valuation[0].clone());
        assert!(
            validate_pair(&pair)
                .iter()
                .any(|error| error.contains("duplicate valuation for atom"))
        );
    }

    #[test]
    fn missing_boolean_atom_assignment_is_rejected() {
        let mut pair = fixture();
        pair.evaluator_only
            .as_mut()
            .unwrap()
            .finite_boolean_countermodel
            .as_mut()
            .unwrap()
            .valuation
            .retain(|assignment| assignment.atom != "p");
        assert!(
            validate_pair(&pair)
                .iter()
                .any(|error| error.contains("no valuation for atom p"))
        );
    }

    #[test]
    fn synthetic_cirr_is_computed_from_adjudication_not_candidate_claims() {
        let pair = fixture();
        let summary = compute_cirr(&[pair]).expect("valid receipt");
        assert_eq!(summary.numerator_definite_reuse, 1);
        assert_eq!(summary.denominator_eligible_local_rejections, 1);
        assert_eq!(summary.point_estimate, Some(1.0));
    }

    #[test]
    fn scope_changed_control_is_excluded_from_primary_cirr() {
        let mut pair = fixture();
        pair.candidate
            .global_stage
            .scope_binding
            .hypothesis_set_sha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned();
        let evaluator = pair.evaluator_only.as_mut().unwrap();
        evaluator.eligibility = EvaluationEligibility::Excluded;
        evaluator.exclusion_reason = Some("global-stage hypothesis scope differs".to_owned());

        assert!(!pair.scope_matches());
        assert!(validate_pair(&pair).is_empty());
        let summary = compute_cirr(&[pair]).expect("scope-changed control remains auditable");
        assert_eq!(summary.denominator_eligible_local_rejections, 0);
        assert_eq!(summary.point_estimate, None);
    }

    #[test]
    fn scope_changed_pair_cannot_be_marked_eligible() {
        let mut pair = fixture();
        pair.candidate
            .global_stage
            .scope_binding
            .hypothesis_set_sha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned();
        assert!(
            validate_pair(&pair)
                .iter()
                .any(|error| error.contains("scope mismatch cannot be marked eligible"))
        );
    }

    #[test]
    fn unknown_global_outcome_is_reported_as_an_interval() {
        let mut pair = fixture();
        pair.evaluator_only.as_mut().unwrap().global_disposition = GlobalDisposition::Unknown;
        let summary = compute_cirr(&[pair]).expect("valid unresolved receipt");
        assert_eq!(summary.numerator_definite_reuse, 0);
        assert_eq!(summary.denominator_eligible_local_rejections, 1);
        assert_eq!(summary.unresolved_global_outcomes, 1);
        assert_eq!(summary.lower_bound, Some(0.0));
        assert_eq!(summary.upper_bound, Some(1.0));
        assert_eq!(summary.point_estimate, None);
    }

    #[test]
    fn citing_a_rejected_warrant_is_not_material_reuse() {
        let mut pair = fixture();
        pair.evaluator_only.as_mut().unwrap().global_disposition = GlobalDisposition::CitedRejected;
        let summary = compute_cirr(&[pair]).expect("valid cited-rejected receipt");
        assert_eq!(summary.denominator_eligible_local_rejections, 1);
        assert_eq!(summary.numerator_definite_reuse, 0);
        assert_eq!(summary.point_estimate, Some(0.0));
    }

    #[test]
    fn zero_denominator_is_undefined_not_zero() {
        let mut pair = fixture();
        let evaluator = pair.evaluator_only.as_mut().unwrap();
        evaluator.eligibility = EvaluationEligibility::Excluded;
        evaluator.exclusion_reason = Some("not part of the CIRR denominator".to_owned());
        let summary = compute_cirr(&[pair]).expect("valid excluded receipt");
        assert_eq!(summary.denominator_eligible_local_rejections, 0);
        assert_eq!(summary.point_estimate, None);
        assert_eq!(summary.lower_bound, None);
    }

    #[test]
    fn execution_budget_overruns_are_rejected() {
        let mut pair = fixture();
        pair.candidate.local_stage.execution.tokens_used =
            pair.candidate.local_stage.execution.token_budget + 1;
        assert!(
            validate_pair(&pair)
                .iter()
                .any(|error| error.contains("tokens_used exceeds token_budget"))
        );
    }

    #[test]
    fn duplicate_warrant_mention_ids_are_rejected() {
        let mut pair = fixture();
        let duplicate = pair.candidate.global_stage.warrant_mentions[0].clone();
        pair.candidate.global_stage.warrant_mentions.push(duplicate);
        assert!(
            validate_pair(&pair)
                .iter()
                .any(|error| error.contains("duplicate warrant mention ID"))
        );
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut value: Value = serde_json::from_str(SYNTHETIC_FIXTURE).expect("fixture JSON");
        value["candidate"]["untrusted_override"] = Value::Bool(true);
        let input = serde_json::to_string(&value).expect("serialize mutation");
        assert!(
            validate_pair_json(&input)
                .unwrap_err()
                .iter()
                .any(|error| error.contains("unknown field"))
        );
    }
}
