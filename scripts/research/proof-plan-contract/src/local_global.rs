//! Paired local/global evidence receipts for proof-consistency evaluation.
//!
//! Candidate-visible outputs and evaluator-only adjudication are different
//! types. candidate_view intentionally projects away evaluator-only fields.
//! Structural validity is not a Lean-kernel or mathematical correctness claim.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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
    pub root_goal_text: String,
    pub hypotheses: Vec<String>,
    pub premises: Vec<String>,
    pub conclusion_text: String,
    pub warrant_text: String,
    pub root_goal_sha256: String,
    pub hypothesis_set_sha256: String,
    pub premise_set_sha256: String,
    pub conclusion_sha256: String,
    pub warrant_id: String,
    pub warrant_sha256: String,
    pub finite_boolean_sequent: Option<BooleanSequent>,
    pub finite_boolean_sequent_sha256: Option<String>,
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
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum BooleanFormula {
    Atom {
        name: String,
    },
    Not {
        arg: Box<BooleanFormula>,
    },
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BooleanSequent {
    /// Explicit contextual assumptions Γ. These must also match ScopeBinding.hypotheses.
    pub hypotheses: Vec<BooleanFormula>,
    /// Additional inference premises supplied to the local warrant.
    pub premises: Vec<BooleanFormula>,
    pub conclusion: BooleanFormula,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BooleanAssignment {
    pub atom: String,
    pub value: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BooleanCountermodel {
    pub valuation: Vec<BooleanAssignment>,
}

impl BooleanFormula {
    /// Canonical candidate-facing rendering for the finite propositional tier.
    /// The fully parenthesized form is deliberately unambiguous and stable.
    pub fn canonical_surface_text(&self) -> String {
        match self {
            Self::Atom { name } => name.clone(),
            Self::Not { arg } => format!("¬({})", arg.canonical_surface_text()),
            Self::And { left, right } => format!(
                "({} ∧ {})",
                left.canonical_surface_text(),
                right.canonical_surface_text()
            ),
            Self::Or { left, right } => format!(
                "({} ∨ {})",
                left.canonical_surface_text(),
                right.canonical_surface_text()
            ),
            Self::Implies { left, right } => format!(
                "({} → {})",
                left.canonical_surface_text(),
                right.canonical_surface_text()
            ),
            Self::Iff { left, right } => format!(
                "({} ↔ {})",
                left.canonical_surface_text(),
                right.canonical_surface_text()
            ),
        }
    }

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

    fn evaluate(
        &self,
        valuation: &std::collections::HashMap<String, bool>,
    ) -> Result<bool, String> {
        match self {
            Self::Atom { name } => valuation
                .get(name)
                .copied()
                .ok_or_else(|| format!("countermodel has no valuation for atom {name}")),
            Self::Not { arg } => Ok(!arg.evaluate(valuation)?),
            Self::And { left, right } => {
                Ok(left.evaluate(valuation)? && right.evaluate(valuation)?)
            }
            Self::Or { left, right } => Ok(left.evaluate(valuation)? || right.evaluate(valuation)?),
            Self::Implies { left, right } => {
                Ok(!left.evaluate(valuation)? || right.evaluate(valuation)?)
            }
            Self::Iff { left, right } => {
                Ok(left.evaluate(valuation)? == right.evaluate(valuation)?)
            }
        }
    }
}

/// Result of exhaustive enumeration over a finite propositional atom set.
/// NoCountermodelFound is complete only for this Boolean AST and atom set;
/// it is not a general Lean proof or a result about arbitrary mathematical types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteTruthTableResult {
    Countermodel {
        valuation: Vec<BooleanAssignment>,
        assignments_checked: usize,
    },
    NoCountermodelFound {
        assignments_checked: usize,
    },
    AtomBudgetExceeded {
        atom_count: usize,
        max_atoms: usize,
    },
}

pub const MAX_FINITE_BOOLEAN_ATOMS: usize = 16;

/// Decide propositional entailment by exhaustive truth-table enumeration.
///
/// At most max_atoms (and never more than 16) distinct atoms are enumerated.
/// The first countermodel is deterministic because atom names are sorted.
pub fn check_finite_propositional_entailment(
    premises: &[BooleanFormula],
    conclusion: &BooleanFormula,
    max_atoms: usize,
) -> Result<FiniteTruthTableResult, String> {
    let mut atom_set = HashSet::new();
    for premise in premises {
        premise.collect_atoms(&mut atom_set)?;
    }
    conclusion.collect_atoms(&mut atom_set)?;

    let mut atoms = atom_set.into_iter().collect::<Vec<_>>();
    atoms.sort();

    let effective_budget = max_atoms.min(MAX_FINITE_BOOLEAN_ATOMS);
    if atoms.len() > effective_budget {
        return Ok(FiniteTruthTableResult::AtomBudgetExceeded {
            atom_count: atoms.len(),
            max_atoms: effective_budget,
        });
    }

    let assignments_total = 1usize
        .checked_shl(atoms.len() as u32)
        .ok_or_else(|| "Boolean atom count exceeds platform shift width".to_owned())?;
    let mut assignments_checked = 0usize;

    // Prefer the sparsest witness relative to the all-false baseline. Within
    // each support size, ascending bitmask order gives deterministic ties.
    for support_size in 0..=atoms.len() {
        for mask in 0..assignments_total {
            if mask.count_ones() as usize != support_size {
                continue;
            }
            assignments_checked += 1;
            let valuation = atoms
                .iter()
                .enumerate()
                .map(|(index, atom)| BooleanAssignment {
                    atom: atom.clone(),
                    value: (mask & (1usize << index)) != 0,
                })
                .collect::<Vec<_>>();
            let values = valuation
                .iter()
                .map(|assignment| (assignment.atom.clone(), assignment.value))
                .collect::<std::collections::HashMap<_, _>>();

            let mut all_premises_true = true;
            for premise in premises {
                if !premise.evaluate(&values)? {
                    all_premises_true = false;
                    break;
                }
            }

            if all_premises_true && !conclusion.evaluate(&values)? {
                return Ok(FiniteTruthTableResult::Countermodel {
                    valuation,
                    assignments_checked,
                });
            }
        }
    }

    Ok(FiniteTruthTableResult::NoCountermodelFound {
        assignments_checked,
    })
}

/// Check Γ, P₁, ..., Pₙ ⊨ C for a typed Boolean sequent.
pub fn check_finite_boolean_sequent(
    sequent: &BooleanSequent,
    max_atoms: usize,
) -> Result<FiniteTruthTableResult, String> {
    let premises = sequent
        .hypotheses
        .iter()
        .chain(sequent.premises.iter())
        .cloned()
        .collect::<Vec<_>>();
    check_finite_propositional_entailment(&premises, &sequent.conclusion, max_atoms)
}

impl BooleanCountermodel {
    /// Validate the hidden valuation against the exact candidate-visible sequent.
    /// Both contextual hypotheses and inference premises must evaluate to true.
    pub fn validate(&self, sequent: &BooleanSequent) -> Result<(), String> {
        if sequent.premises.is_empty() {
            return Err("candidate finite Boolean sequent must contain at least one inference premise".to_owned());
        }

        let mut values = std::collections::HashMap::new();
        for assignment in &self.valuation {
            if !non_empty(&assignment.atom) {
                return Err("countermodel valuation atom must be non-empty".to_owned());
            }
            if values
                .insert(assignment.atom.clone(), assignment.value)
                .is_some()
            {
                return Err(format!(
                    "countermodel contains duplicate valuation for atom {}",
                    assignment.atom
                ));
            }
        }

        let mut atoms = HashSet::new();
        for hypothesis in &sequent.hypotheses {
            hypothesis.collect_atoms(&mut atoms)?;
        }
        for premise in &sequent.premises {
            premise.collect_atoms(&mut atoms)?;
        }
        sequent.conclusion.collect_atoms(&mut atoms)?;
        for atom in &atoms {
            if !values.contains_key(atom) {
                return Err(format!("countermodel has no valuation for atom {atom}"));
            }
        }
        for atom in values.keys() {
            if !atoms.contains(atom) {
                return Err(format!("countermodel contains unused atom {atom}"));
            }
        }

        for (index, hypothesis) in sequent.hypotheses.iter().enumerate() {
            if !hypothesis.evaluate(&values)? {
                return Err(format!("countermodel hypothesis[{index}] evaluates to false"));
            }
        }
        for (index, premise) in sequent.premises.iter().enumerate() {
            if !premise.evaluate(&values)? {
                return Err(format!("countermodel premise[{index}] evaluates to false"));
            }
        }
        if sequent.conclusion.evaluate(&values)? {
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
    /// Lower Wilson-95 endpoint under the definite-reuse / unknown-as-zero case.
    pub wilson_95_envelope_lower: Option<f64>,
    /// Upper Wilson-95 endpoint under the definite-plus-unknown / unknown-as-one case.
    pub wilson_95_envelope_upper: Option<f64>,
}

impl LocalGlobalPairV1 {
    /// Only this projection is suitable for a candidate-visible handoff.
    /// It cannot serialize the evaluator-only partition.
    pub fn candidate_view(&self) -> &CandidatePayload {
        &self.candidate
    }

    /// Domain-separated identity for the exact versioned candidate-visible
    /// payload. Evaluator-only labels and witnesses are deliberately excluded.
    pub fn candidate_view_sha256(&self) -> Result<String, String> {
        domain_separated_sha256("candidate_payload", &(&self.schema_version, &self.candidate))
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

fn domain_separated_sha256<T: Serialize>(domain: &str, value: &T) -> Result<String, String> {
    let canonical = serde_json::to_vec(value)
        .map_err(|error| format!("could not serialize {domain} preimage: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.local_global_pair.v1:");
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(canonical);
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn canonical_string_set(values: &[String], label: &str) -> Result<Vec<String>, String> {
    if values.iter().any(|value| !non_empty(value)) {
        return Err(format!("{label} entries must be non-empty"));
    }
    let mut canonical = values.to_vec();
    canonical.sort();
    if canonical.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(format!("{label} contains duplicate entries"));
    }
    Ok(canonical)
}

fn validate_preimage_digest<T: Serialize>(
    domain: &str,
    preimage: &T,
    actual: &str,
    label: &str,
    errors: &mut Vec<String>,
) {
    match domain_separated_sha256(domain, preimage) {
        Ok(expected) if expected == actual => {}
        Ok(_) => errors.push(format!("{label} does not match its canonical preimage")),
        Err(error) => errors.push(format!("{label}: {error}")),
    }
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
    for (field, value) in [
        ("root_goal_text", &scope.root_goal_text),
        ("conclusion_text", &scope.conclusion_text),
        ("warrant_text", &scope.warrant_text),
    ] {
        if !non_empty(value) {
            errors.push(format!("{label}.{field} must be non-empty"));
        }
    }

    match canonical_string_set(&scope.hypotheses, "hypotheses") {
        Ok(canonical) => validate_preimage_digest(
            "hypotheses",
            &canonical,
            &scope.hypothesis_set_sha256,
            &format!("{label}.hypothesis_set_sha256"),
            errors,
        ),
        Err(error) => errors.push(format!("{label}.{error}")),
    }
    match canonical_string_set(&scope.premises, "premises") {
        Ok(canonical) => validate_preimage_digest(
            "premises",
            &canonical,
            &scope.premise_set_sha256,
            &format!("{label}.premise_set_sha256"),
            errors,
        ),
        Err(error) => errors.push(format!("{label}.{error}")),
    }
    validate_preimage_digest(
        "root_goal",
        &scope.root_goal_text,
        &scope.root_goal_sha256,
        &format!("{label}.root_goal_sha256"),
        errors,
    );
    validate_preimage_digest(
        "conclusion",
        &scope.conclusion_text,
        &scope.conclusion_sha256,
        &format!("{label}.conclusion_sha256"),
        errors,
    );
    validate_preimage_digest(
        "warrant",
        &scope.warrant_text,
        &scope.warrant_sha256,
        &format!("{label}.warrant_sha256"),
        errors,
    );

    match (&scope.finite_boolean_sequent, &scope.finite_boolean_sequent_sha256) {
        (Some(sequent), Some(digest)) => {
            if !valid_sha256(digest) {
                errors.push(format!(
                    "{label}.finite_boolean_sequent_sha256 must be lowercase SHA-256 hex"
                ));
            }
            if sequent.premises.is_empty() {
                errors.push(format!(
                    "{label}.finite_boolean_sequent must contain at least one inference premise"
                ));
            }
            if scope.hypotheses.len() != sequent.hypotheses.len() {
                errors.push(format!(
                    "{label}.hypotheses length does not match finite_boolean_sequent.hypotheses"
                ));
            } else {
                for (index, (display, formula)) in
                    scope.hypotheses.iter().zip(&sequent.hypotheses).enumerate()
                {
                    if display != &formula.canonical_surface_text() {
                        errors.push(format!(
                            "{label}.hypotheses[{index}] does not match the canonical typed sequent display"
                        ));
                    }
                }
            }
            for (index, hypothesis) in sequent.hypotheses.iter().enumerate() {
                let mut atoms = HashSet::new();
                if let Err(error) = hypothesis.collect_atoms(&mut atoms) {
                    errors.push(format!(
                        "{label}.finite_boolean_sequent.hypotheses[{index}]: {error}"
                    ));
                }
            }
            if scope.premises.len() != sequent.premises.len() {
                errors.push(format!(
                    "{label}.premises length does not match finite_boolean_sequent.premises"
                ));
            } else {
                for (index, (display, formula)) in
                    scope.premises.iter().zip(&sequent.premises).enumerate()
                {
                    if display != &formula.canonical_surface_text() {
                        errors.push(format!(
                            "{label}.premises[{index}] does not match the canonical typed sequent display"
                        ));
                    }
                }
            }
            if scope.conclusion_text != sequent.conclusion.canonical_surface_text() {
                errors.push(format!(
                    "{label}.conclusion_text does not match the canonical typed sequent display"
                ));
            }
            if scope.root_goal_text != sequent.conclusion.canonical_surface_text() {
                errors.push(format!(
                    "{label}.root_goal_text does not match the canonical typed sequent conclusion"
                ));
            }
            let canonical_warrant = format!(
                "From {} infer {}.",
                sequent
                    .premises
                    .iter()
                    .map(BooleanFormula::canonical_surface_text)
                    .collect::<Vec<_>>()
                    .join(", "),
                sequent.conclusion.canonical_surface_text()
            );
            if scope.warrant_text != canonical_warrant {
                errors.push(format!(
                    "{label}.warrant_text does not match the canonical typed sequent warrant"
                ));
            }
            for (index, premise) in sequent.premises.iter().enumerate() {
                let mut atoms = HashSet::new();
                if let Err(error) = premise.collect_atoms(&mut atoms) {
                    errors.push(format!(
                        "{label}.finite_boolean_sequent.premises[{index}]: {error}"
                    ));
                }
            }
            let mut atoms = HashSet::new();
            if let Err(error) = sequent.conclusion.collect_atoms(&mut atoms) {
                errors.push(format!(
                    "{label}.finite_boolean_sequent.conclusion: {error}"
                ));
            }
            validate_preimage_digest(
                "finite_boolean_sequent",
                sequent,
                digest,
                &format!("{label}.finite_boolean_sequent_sha256"),
                errors,
            );
        }
        (Some(_), None) => errors.push(format!(
            "{label}.finite_boolean_sequent_sha256 is required when a finite Boolean sequent is present"
        )),
        (None, Some(_)) => errors.push(format!(
            "{label}.finite_boolean_sequent_sha256 must be absent when no finite Boolean sequent is present"
        )),
        (None, None) => {}
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
            match candidate
                .local_stage
                .scope_binding
                .finite_boolean_sequent
                .as_ref()
            {
                Some(sequent) => {
                    if let Err(reason) = countermodel.validate(sequent) {
                        errors.push(format!(
                            "evaluator_only.finite_boolean_countermodel: {reason}"
                        ));
                    }
                }
                None => errors.push(
                    "finite Boolean countermodel requires a candidate-visible typed sequent"
                        .to_owned(),
                ),
            }
            if evaluator.independently_assessed_warrant != WarrantValidity::Invalid {
                errors.push(
                    "finite Boolean countermodel requires independently invalid warrant".to_owned(),
                );
            }
        }
        if evaluator.adjudication_authority == "finite_boolean_countermodel"
            && (evaluator.finite_boolean_countermodel.is_none()
                || candidate
                    .local_stage
                    .scope_binding
                    .finite_boolean_sequent
                    .is_none())
        {
            errors.push(
                "finite_boolean_countermodel authority requires a typed sequent and a finite valuation"
                    .to_owned(),
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

/// Return the standard 95% Wilson score interval for a binomial proportion.
/// This is a simple unclustered diagnostic; it does not replace cluster-aware
/// inference when observations share theorem families, models or task templates.
fn wilson_95(successes: usize, trials: usize) -> Option<(f64, f64)> {
    if trials == 0 || successes > trials {
        return None;
    }
    const Z: f64 = 1.959_963_984_540_054;
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = Z * Z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = Z * ((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))).sqrt() / denominator;
    Some((
        (center - half_width).max(0.0),
        (center + half_width).min(1.0),
    ))
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
    let wilson_95_envelope_lower = wilson_95(numerator, denominator).map(|interval| interval.0);
    let wilson_95_envelope_upper =
        wilson_95(numerator + unresolved, denominator).map(|interval| interval.1);

    Ok(CirrSummary {
        numerator_definite_reuse: numerator,
        denominator_eligible_local_rejections: denominator,
        unresolved_global_outcomes: unresolved,
        lower_bound,
        upper_bound,
        point_estimate,
        wilson_95_envelope_lower,
        wilson_95_envelope_upper,
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
    fn sha256_implementation_matches_standard_abc_vector() {
        let digest = Sha256::digest(b"abc");
        let rendered: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(
            rendered,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn canonical_scope_preimages_recompute_all_identity_digests() {
        let pair = fixture();
        let scope = &pair.candidate.local_stage.scope_binding;
        assert_eq!(
            domain_separated_sha256("root_goal", &scope.root_goal_text).unwrap(),
            scope.root_goal_sha256
        );
        assert_eq!(
            domain_separated_sha256(
                "hypotheses",
                &canonical_string_set(&scope.hypotheses, "hypotheses").unwrap()
            )
            .unwrap(),
            scope.hypothesis_set_sha256
        );
        assert_eq!(
            domain_separated_sha256(
                "premises",
                &canonical_string_set(&scope.premises, "premises").unwrap()
            )
            .unwrap(),
            scope.premise_set_sha256
        );
        assert_eq!(
            domain_separated_sha256("conclusion", &scope.conclusion_text).unwrap(),
            scope.conclusion_sha256
        );
        assert_eq!(
            domain_separated_sha256("warrant", &scope.warrant_text).unwrap(),
            scope.warrant_sha256
        );
    }

    #[test]
    fn premise_mutation_is_rejected_even_when_digest_shape_is_valid() {
        let mut pair = fixture();
        pair.candidate.local_stage.scope_binding.premises[0] = "q → p".to_owned();
        assert!(validate_pair(&pair).iter().any(|error| {
            error.contains("premise_set_sha256 does not match its canonical preimage")
        }));
    }

    #[test]
    fn hypothesis_and_premise_sets_are_order_canonical_and_duplicate_free() {
        let mut values = vec!["z".to_owned(), "a".to_owned()];
        let first = canonical_string_set(&values, "premises").unwrap();
        values.reverse();
        let second = canonical_string_set(&values, "premises").unwrap();
        assert_eq!(first, second);
        values.push("a".to_owned());
        assert!(canonical_string_set(&values, "premises").is_err());
    }

    #[test]
    fn synthetic_receipt_is_structurally_valid() {
        let pair = fixture();
        assert_eq!(
            pair.candidate.pair_id,
            "PAIR-SYNTHETIC-IMPLICATION-CONVERSE-001"
        );
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
        assert!(visible["local_stage"]["scope_binding"]["finite_boolean_sequent"].is_object());
        assert!(visible.get("valuation").is_none());
    }

    #[test]
    fn countermodel_must_satisfy_typed_hypotheses() {
        let p = BooleanFormula::Atom {
            name: "p".to_owned(),
        };
        let q = BooleanFormula::Atom {
            name: "q".to_owned(),
        };
        let sequent = BooleanSequent {
            hypotheses: vec![p.clone()],
            premises: vec![BooleanFormula::Implies {
                left: Box::new(p),
                right: Box::new(q.clone()),
            }],
            conclusion: q,
        };
        let witness = BooleanCountermodel {
            valuation: vec![
                BooleanAssignment {
                    atom: "p".to_owned(),
                    value: false,
                },
                BooleanAssignment {
                    atom: "q".to_owned(),
                    value: false,
                },
            ],
        };
        assert!(
            witness
                .validate(&sequent)
                .unwrap_err()
                .contains("countermodel hypothesis[0] evaluates to false")
        );
    }

    #[test]
    fn sequent_oracle_includes_contextual_hypotheses() {
        let p = BooleanFormula::Atom {
            name: "p".to_owned(),
        };
        let q = BooleanFormula::Atom {
            name: "q".to_owned(),
        };
        let sequent = BooleanSequent {
            hypotheses: vec![p.clone()],
            premises: vec![BooleanFormula::Implies {
                left: Box::new(p),
                right: Box::new(q.clone()),
            }],
            conclusion: q,
        };
        assert_eq!(
            check_finite_boolean_sequent(&sequent, MAX_FINITE_BOOLEAN_ATOMS).unwrap(),
            FiniteTruthTableResult::NoCountermodelFound {
                assignments_checked: 4,
            }
        );
    }

    #[test]
    fn sequent_scope_rejects_mistranslated_hypothesis_text_after_valid_rehash() {
        let mut pair = fixture();
        let candidate_scope = &mut pair.candidate.local_stage.scope_binding;
        candidate_scope.hypotheses = vec!["q".to_owned()];
        let canonical = canonical_string_set(&candidate_scope.hypotheses, "hypotheses").unwrap();
        candidate_scope.hypothesis_set_sha256 =
            domain_separated_sha256("hypotheses", &canonical).unwrap();

        let sequent = candidate_scope.finite_boolean_sequent.as_mut().unwrap();
        sequent.hypotheses = vec![BooleanFormula::Atom {
            name: "p".to_owned(),
        }];
        candidate_scope.finite_boolean_sequent_sha256 =
            Some(domain_separated_sha256("finite_boolean_sequent", sequent).unwrap());

        pair.candidate.global_stage.scope_binding = candidate_scope.clone();

        assert!(
            validate_pair(&pair).iter().any(|error| {
                error.contains("hypotheses[0] does not match the canonical typed sequent display")
            })
        );
    }

    #[test]
    fn truth_table_finds_converse_implication_countermodel() {
        let p = BooleanFormula::Atom {
            name: "p".to_owned(),
        };
        let q = BooleanFormula::Atom {
            name: "q".to_owned(),
        };
        let result = check_finite_propositional_entailment(
            &[BooleanFormula::Implies {
                left: Box::new(p.clone()),
                right: Box::new(q.clone()),
            }],
            &BooleanFormula::Implies {
                left: Box::new(q),
                right: Box::new(p),
            },
            MAX_FINITE_BOOLEAN_ATOMS,
        )
        .expect("well-formed propositional formula");

        let FiniteTruthTableResult::Countermodel { valuation, .. } = result else {
            panic!("expected a countermodel for p -> q / q -> p");
        };
        assert_eq!(
            valuation,
            vec![
                BooleanAssignment {
                    atom: "p".to_owned(),
                    value: false,
                },
                BooleanAssignment {
                    atom: "q".to_owned(),
                    value: true,
                },
            ]
        );
    }


    #[test]
    fn truth_table_prefers_minimum_support_countermodel() {
        let p = BooleanFormula::Atom {
            name: "p".to_owned(),
        };
        let q = BooleanFormula::Atom {
            name: "q".to_owned(),
        };
        let r = BooleanFormula::Atom {
            name: "r".to_owned(),
        };
        let tautology = BooleanFormula::Or {
            left: Box::new(p.clone()),
            right: Box::new(BooleanFormula::Not {
                arg: Box::new(p.clone()),
            }),
        };
        let premise = BooleanFormula::Or {
            left: Box::new(tautology),
            right: Box::new(BooleanFormula::And {
                left: Box::new(q.clone()),
                right: Box::new(r.clone()),
            }),
        };
        let conclusion = BooleanFormula::Not {
            arg: Box::new(BooleanFormula::Or {
                left: Box::new(BooleanFormula::And {
                    left: Box::new(p),
                    right: Box::new(q),
                }),
                right: Box::new(r),
            }),
        };

        let result = check_finite_propositional_entailment(
            &[premise],
            &conclusion,
            MAX_FINITE_BOOLEAN_ATOMS,
        )
        .expect("well-formed propositional formulas");
        let FiniteTruthTableResult::Countermodel {
            valuation,
            assignments_checked,
        } = result
        else {
            panic!("expected a countermodel");
        };
        assert_eq!(
            valuation,
            vec![
                BooleanAssignment {
                    atom: "p".to_owned(),
                    value: false,
                },
                BooleanAssignment {
                    atom: "q".to_owned(),
                    value: false,
                },
                BooleanAssignment {
                    atom: "r".to_owned(),
                    value: true,
                },
            ]
        );
        assert_eq!(assignments_checked, 4);
    }

    #[test]
    fn truth_table_checks_de_morgan_equivalence_and_biconditional() {
        let p = BooleanFormula::Atom {
            name: "p".to_owned(),
        };
        let q = BooleanFormula::Atom {
            name: "q".to_owned(),
        };
        let left = BooleanFormula::Not {
            arg: Box::new(BooleanFormula::And {
                left: Box::new(p.clone()),
                right: Box::new(q.clone()),
            }),
        };
        let right = BooleanFormula::Or {
            left: Box::new(BooleanFormula::Not {
                arg: Box::new(p),
            }),
            right: Box::new(BooleanFormula::Not {
                arg: Box::new(q),
            }),
        };
        let de_morgan = BooleanFormula::Iff {
            left: Box::new(left),
            right: Box::new(right),
        };

        assert_eq!(
            check_finite_propositional_entailment(
                &[],
                &de_morgan,
                MAX_FINITE_BOOLEAN_ATOMS
            )
            .unwrap(),
            FiniteTruthTableResult::NoCountermodelFound {
                assignments_checked: 4,
            }
        );
    }

    #[test]
    fn truth_table_accepts_modus_ponens_entailment() {
        let p = BooleanFormula::Atom {
            name: "p".to_owned(),
        };
        let q = BooleanFormula::Atom {
            name: "q".to_owned(),
        };
        let result = check_finite_propositional_entailment(
            &[
                p.clone(),
                BooleanFormula::Implies {
                    left: Box::new(p),
                    right: Box::new(q.clone()),
                },
            ],
            &q,
            MAX_FINITE_BOOLEAN_ATOMS,
        )
        .expect("well-formed propositional formula");

        assert_eq!(
            result,
            FiniteTruthTableResult::NoCountermodelFound {
                assignments_checked: 4,
            }
        );
    }

    #[test]
    fn truth_table_abstains_when_atom_budget_is_exceeded() {
        let premise = (0..=MAX_FINITE_BOOLEAN_ATOMS)
            .map(|index| BooleanFormula::Atom {
                name: format!("p{index}"),
            })
            .reduce(|left, right| BooleanFormula::And {
                left: Box::new(left),
                right: Box::new(right),
            })
            .expect("non-empty atom sequence");
        let result = check_finite_propositional_entailment(
            &[premise.clone()],
            &premise,
            MAX_FINITE_BOOLEAN_ATOMS,
        )
        .expect("well-formed propositional formula");

        assert_eq!(
            result,
            FiniteTruthTableResult::AtomBudgetExceeded {
                atom_count: MAX_FINITE_BOOLEAN_ATOMS + 1,
                max_atoms: MAX_FINITE_BOOLEAN_ATOMS,
            }
        );
    }

    #[test]
    fn truth_table_countermodels_are_deterministic() {
        let a = BooleanFormula::Atom {
            name: "a".to_owned(),
        };
        let z = BooleanFormula::Atom {
            name: "z".to_owned(),
        };
        let premise = BooleanFormula::Implies {
            left: Box::new(a.clone()),
            right: Box::new(z.clone()),
        };
        let conclusion = BooleanFormula::Implies {
            left: Box::new(z),
            right: Box::new(a),
        };

        let first = check_finite_propositional_entailment(
            std::slice::from_ref(&premise),
            &conclusion,
            MAX_FINITE_BOOLEAN_ATOMS,
        );
        let second = check_finite_propositional_entailment(
            std::slice::from_ref(&premise),
            &conclusion,
            MAX_FINITE_BOOLEAN_ATOMS,
        );
        assert_eq!(first, second);
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
        let sequent = pair
            .candidate
            .local_stage
            .scope_binding
            .finite_boolean_sequent
            .as_ref()
            .unwrap();
        assert!(witness.validate(sequent).is_ok());
        let premise_value = sequent.premises[0]
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
    fn candidate_view_digest_is_stable_and_excludes_evaluator_only_data() {
        let pair = fixture();
        let original = pair.candidate_view_sha256().expect("candidate digest");
        let mut evaluator_mutation = pair.clone();
        let evaluator = evaluator_mutation.evaluator_only.as_mut().unwrap();
        evaluator.global_disposition = GlobalDisposition::Omitted;
        evaluator.adjudication_refs.push("additional-hidden-reference".to_owned());
        evaluator
            .finite_boolean_countermodel
            .as_mut()
            .unwrap()
            .valuation[0]
            .value = true;
        assert_eq!(
            original,
            evaluator_mutation.candidate_view_sha256().expect("same candidate digest")
        );
    }

    #[test]
    fn candidate_view_digest_changes_when_candidate_payload_changes() {
        let pair = fixture();
        let original = pair.candidate_view_sha256().expect("candidate digest");
        let mut candidate_mutation = pair.clone();
        candidate_mutation.candidate.task_class =
            "modified_candidate_task_class".to_owned();
        assert_ne!(
            original,
            candidate_mutation.candidate_view_sha256().expect("mutated candidate digest")
        );
    }

    #[test]
    fn canonical_surface_renderer_is_unambiguous() {
        let formula = BooleanFormula::Implies {
            left: Box::new(BooleanFormula::Atom { name: "p".to_owned() }),
            right: Box::new(BooleanFormula::Atom { name: "q".to_owned() }),
        };
        assert_eq!(formula.canonical_surface_text(), "(p → q)");
    }


    #[test]
    fn candidate_warrant_text_must_match_typed_sequent_even_after_valid_rehash() {
        let mut pair = fixture();
        {
            let scope = &mut pair.candidate.local_stage.scope_binding;
            scope.warrant_text = "From (q → p) infer (p → q).".to_owned();
            scope.warrant_sha256 =
                domain_separated_sha256("warrant", &scope.warrant_text).unwrap();
        }
        pair.candidate.global_stage.scope_binding =
            pair.candidate.local_stage.scope_binding.clone();
        assert!(
            validate_pair(&pair)
                .iter()
                .any(|error| error.contains("warrant_text does not match the canonical typed sequent warrant"))
        );
    }

    #[test]
    fn candidate_display_text_must_match_typed_sequent_even_after_valid_rehash() {
        let mut pair = fixture();
        {
            let scope = &mut pair.candidate.local_stage.scope_binding;
            scope.premises[0] = "(p ↔ q)".to_owned();
            let canonical = canonical_string_set(&scope.premises, "premises").unwrap();
            scope.premise_set_sha256 =
                domain_separated_sha256("premises", &canonical).unwrap();
        }
        pair.candidate.global_stage.scope_binding =
            pair.candidate.local_stage.scope_binding.clone();
        assert!(
            validate_pair(&pair)
                .iter()
                .any(|error| error.contains("does not match the canonical typed sequent display"))
        );
    }

    #[test]
    fn countermodel_is_bound_to_candidate_visible_sequent() {
        let pair = fixture();
        let scope = &pair.candidate.local_stage.scope_binding;
        let sequent = scope.finite_boolean_sequent.as_ref().unwrap();
        let witness = pair
            .evaluator_only
            .as_ref()
            .unwrap()
            .finite_boolean_countermodel
            .as_ref()
            .unwrap();
        assert!(witness.validate(sequent).is_ok());
        assert_eq!(
            pair.candidate.local_stage.scope_binding.finite_boolean_sequent,
            pair.candidate.global_stage.scope_binding.finite_boolean_sequent
        );
    }

    #[test]
    fn unrelated_sequent_cannot_reuse_a_valid_countermodel_even_with_new_digest() {
        let mut pair = fixture();
        {
            let scope = &mut pair.candidate.local_stage.scope_binding;
            let sequent = scope.finite_boolean_sequent.as_mut().unwrap();
            sequent.conclusion = BooleanFormula::Atom {
                name: "q".to_owned(),
            };
            scope.finite_boolean_sequent_sha256 =
                Some(domain_separated_sha256("finite_boolean_sequent", sequent).unwrap());
        }
        pair.candidate.global_stage.scope_binding =
            pair.candidate.local_stage.scope_binding.clone();

        assert!(
            validate_pair(&pair)
                .iter()
                .any(|error| error.contains("countermodel conclusion evaluates to true"))
        );
    }

    #[test]
    fn changing_only_typed_sequent_digest_is_rejected() {
        let mut pair = fixture();
        pair.candidate
            .local_stage
            .scope_binding
            .finite_boolean_sequent_sha256 = Some(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        );
        assert!(
            validate_pair(&pair)
                .iter()
                .any(|error| error.contains("finite_boolean_sequent_sha256 does not match its canonical preimage"))
        );
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
        {
            let scope = &mut pair.candidate.global_stage.scope_binding;
            scope.hypotheses = vec!["changed assumption".to_owned()];
            let canonical = canonical_string_set(&scope.hypotheses, "hypotheses").unwrap();
            scope.hypothesis_set_sha256 =
                domain_separated_sha256("hypotheses", &canonical).unwrap();
        }
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
    fn wilson_95_for_one_of_one_is_finite_and_bounded() {
        let interval = wilson_95(1, 1).expect("non-empty binomial sample");
        assert!((interval.0 - 0.206_549_314_377_237_45).abs() < 1e-12);
        assert_eq!(interval.1, 1.0);
    }

    #[test]
    fn cirr_wilson_envelope_includes_unknown_outcomes_conservatively() {
        let mut pairs = Vec::new();
        for index in 0..20 {
            let mut pair = fixture();
            pair.candidate.pair_id = format!("PAIR-WILSON-{index:02}");
            let evaluator = pair.evaluator_only.as_mut().unwrap();
            evaluator.global_disposition = if index < 5 {
                GlobalDisposition::UsedDirectly
            } else if index < 15 {
                GlobalDisposition::Omitted
            } else {
                GlobalDisposition::Unknown
            };
            pairs.push(pair);
        }
        let summary = compute_cirr(&pairs).expect("valid independent receipts");
        assert_eq!(summary.numerator_definite_reuse, 5);
        assert_eq!(summary.denominator_eligible_local_rejections, 20);
        assert_eq!(summary.unresolved_global_outcomes, 5);
        assert_eq!(summary.point_estimate, None);
        assert_eq!(summary.lower_bound, Some(0.25));
        assert_eq!(summary.upper_bound, Some(0.5));
        assert!(summary.wilson_95_envelope_lower.unwrap() < 0.25);
        assert!(summary.wilson_95_envelope_upper.unwrap() > 0.5);
        assert!(summary.wilson_95_envelope_upper.unwrap() < 1.0);
    }

    #[test]
    fn cirr_wilson_envelope_is_undefined_for_zero_denominator() {
        let mut pair = fixture();
        let evaluator = pair.evaluator_only.as_mut().unwrap();
        evaluator.eligibility = EvaluationEligibility::Excluded;
        evaluator.exclusion_reason = Some("zero denominator control".to_owned());
        let summary = compute_cirr(&[pair]).expect("valid excluded receipt");
        assert_eq!(summary.wilson_95_envelope_lower, None);
        assert_eq!(summary.wilson_95_envelope_upper, None);
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
