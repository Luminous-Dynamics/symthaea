//! Finite metamorphic-pair validator for mathematical invariance/sensitivity.
//!
 //! The finite oracle classifies equivalence relations only for the declared
 //! Boolean AST. It is not a Lean proof, theorem prover, or authority for
 //! arbitrary mathematical propositions.

use crate::local_global::{
    BooleanFormula, BooleanSequent, FiniteTruthTableResult, MAX_FINITE_BOOLEAN_ATOMS,
    check_finite_propositional_entailment,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(test)]
const FIXTURE: &str = include_str!("../../../../docs/research/MATH_METAMORPHIC_PAIR_V1.json");

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MathMetamorphicDatasetV1 {
    pub schema_version: String,
    pub pairs: Vec<MathMetamorphicPairV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MathMetamorphicPairV1 {
    pub candidate: CandidatePairV1,
    pub evaluator_only: Option<EvaluatorPairV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidatePairV1 {
    pub pair_id: String,
    pub source: SourceIdentityV1,
    pub base: StatementBindingV1,
    pub variant: StatementBindingV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceIdentityV1 {
    pub repository: String,
    pub revision: String,
    pub declaration: String,
    pub task_family: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StatementBindingV1 {
    pub sequent: BooleanSequent,
    pub sequent_sha256: String,
    pub proof_route_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorPairV1 {
    pub transformation_class: TransformationClassV1,
    pub oracle_relation: OracleRelationV1,
    pub adjudication_authority: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransformationClassV1 {
    SemanticsPreserving,
    SemanticsChanging,
    ProofRouteOnly,
    Unsupported,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OracleRelationV1 {
    Equivalent,
    BaseEntailsVariant,
    VariantEntailsBase,
    Incomparable,
    Unknown,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PairValidationV1 {
    pub pair_id: String,
    pub oracle_relation: OracleRelationV1,
    pub transformation_class: TransformationClassV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Entailment {
    Yes,
    No,
    Unknown,
}

fn domain_separated_sha256<T: Serialize>(domain: &str, value: &T) -> Result<String, String> {
    let canonical = serde_json::to_vec(value)
        .map_err(|error| format!("could not serialize {domain} preimage: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.math_metamorphic_pair.v1:");
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(canonical);
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub fn sequent_sha256(sequent: &BooleanSequent) -> Result<String, String> {
    domain_separated_sha256("boolean_sequent", sequent)
}

fn as_implication(sequent: &BooleanSequent) -> BooleanFormula {
    let antecedents = sequent
        .hypotheses
        .iter()
        .chain(sequent.premises.iter())
        .cloned()
        .collect::<Vec<_>>();
    match antecedents
        .into_iter()
        .reduce(|left, right| BooleanFormula::And {
            left: Box::new(left),
            right: Box::new(right),
        }) {
        Some(antecedent) => BooleanFormula::Implies {
            left: Box::new(antecedent),
            right: Box::new(sequent.conclusion.clone()),
        },
        None => sequent.conclusion.clone(),
    }
}

fn check_entailment(
    premise: &BooleanFormula,
    conclusion: &BooleanFormula,
) -> Result<Entailment, String> {
    match check_finite_propositional_entailment(
        std::slice::from_ref(premise),
        conclusion,
        MAX_FINITE_BOOLEAN_ATOMS,
    )? {
        FiniteTruthTableResult::NoCountermodelFound { .. } => Ok(Entailment::Yes),
        FiniteTruthTableResult::Countermodel { .. } => Ok(Entailment::No),
        FiniteTruthTableResult::AtomBudgetExceeded { .. } => Ok(Entailment::Unknown),
    }
}

fn derive_relation(
    base: &BooleanSequent,
    variant: &BooleanSequent,
) -> Result<OracleRelationV1, String> {
    let base_formula = as_implication(base);
    let variant_formula = as_implication(variant);
    let base_to_variant = check_entailment(&base_formula, &variant_formula)?;
    let variant_to_base = check_entailment(&variant_formula, &base_formula)?;

    Ok(match (base_to_variant, variant_to_base) {
        (Entailment::Yes, Entailment::Yes) => OracleRelationV1::Equivalent,
        (Entailment::Yes, Entailment::No) => OracleRelationV1::BaseEntailsVariant,
        (Entailment::No, Entailment::Yes) => OracleRelationV1::VariantEntailsBase,
        (Entailment::No, Entailment::No) => OracleRelationV1::Incomparable,
        _ => OracleRelationV1::Unknown,
    })
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_binding(binding: &StatementBindingV1, label: &str, errors: &mut Vec<String>) {
    if !valid_sha256(&binding.sequent_sha256) {
        errors.push(format!(
            "{label}.sequent_sha256 must be lowercase SHA-256 hex"
        ));
    } else {
        match sequent_sha256(&binding.sequent) {
            Ok(expected) if expected == binding.sequent_sha256 => {}
            Ok(_) => errors.push(format!(
                "{label}.sequent_sha256 does not match the canonical typed sequent"
            )),
            Err(error) => errors.push(format!("{label}: {error}")),
        }
    }
    if let Some(route) = &binding.proof_route_sha256 {
        if !valid_sha256(route) {
            errors.push(format!(
                "{label}.proof_route_sha256 must be lowercase SHA-256 hex"
            ));
        }
    }
}

pub fn validate_pair(pair: &MathMetamorphicPairV1) -> Vec<String> {
    let mut errors = Vec::new();
    let candidate = &pair.candidate;
    if candidate.pair_id.trim().is_empty() {
        errors.push("candidate.pair_id must be non-empty".to_owned());
    }
    for (field, value) in [
        ("source.repository", &candidate.source.repository),
        ("source.revision", &candidate.source.revision),
        ("source.declaration", &candidate.source.declaration),
        ("source.task_family", &candidate.source.task_family),
    ] {
        if value.trim().is_empty() {
            errors.push(format!("candidate.{field} must be non-empty"));
        }
    }
    validate_binding(&candidate.base, "candidate.base", &mut errors);
    validate_binding(&candidate.variant, "candidate.variant", &mut errors);

    if let Some(evaluator) = &pair.evaluator_only {
        if evaluator.adjudication_authority != "finite_boolean_truth_table" {
            errors.push("finite-only v1 requires finite_boolean_truth_table authority".to_owned());
        }
        if evaluator.evidence_refs.is_empty()
            || evaluator
                .evidence_refs
                .iter()
                .any(|value| value.trim().is_empty())
        {
            errors.push("evaluator_only.evidence_refs must be non-empty".to_owned());
        }
        let derived = match derive_relation(&candidate.base.sequent, &candidate.variant.sequent) {
            Ok(relation) => relation,
            Err(error) => {
                errors.push(format!("finite semantic check failed: {error}"));
                OracleRelationV1::Unknown
            }
        };
        if derived != evaluator.oracle_relation {
            errors.push(format!(
                "evaluator_only.oracle_relation mismatch: declared {:?}, derived {:?}",
                evaluator.oracle_relation, derived
            ));
        }

        let same_statement = candidate.base.sequent_sha256 == candidate.variant.sequent_sha256;
        let distinct_route_receipts = matches!(
            (
                candidate.base.proof_route_sha256.as_deref(),
                candidate.variant.proof_route_sha256.as_deref()
            ),
            (Some(base), Some(variant)) if base != variant
        );
        let class_matches = match evaluator.transformation_class {
            TransformationClassV1::SemanticsPreserving => {
                evaluator.oracle_relation == OracleRelationV1::Equivalent
            }
            TransformationClassV1::SemanticsChanging => {
                evaluator.oracle_relation != OracleRelationV1::Equivalent
                    && evaluator.oracle_relation != OracleRelationV1::Unknown
            }
            TransformationClassV1::ProofRouteOnly => {
                evaluator.oracle_relation == OracleRelationV1::Equivalent
                    && same_statement
                    && distinct_route_receipts
            }
            TransformationClassV1::Unsupported => {
                evaluator.oracle_relation == OracleRelationV1::Unknown
            }
        };
        if !class_matches {
            errors.push(
                "evaluator_only.transformation_class conflicts with finite semantic relation"
                    .to_owned(),
            );
        }
    }

    errors
}

pub fn validate_dataset(dataset: &MathMetamorphicDatasetV1) -> Vec<String> {
    let mut errors = Vec::new();
    if dataset.schema_version != "1.0.0" {
        errors.push("schema_version must equal 1.0.0".to_owned());
    }
    if dataset.pairs.is_empty() {
        errors.push("pairs must contain at least one pair".to_owned());
    }
    let mut ids = std::collections::HashSet::new();
    for (index, pair) in dataset.pairs.iter().enumerate() {
        if !ids.insert(&pair.candidate.pair_id) {
            errors.push(format!("duplicate pair_id: {}", pair.candidate.pair_id));
        }
        errors.extend(
            validate_pair(pair)
                .into_iter()
                .map(|error| format!("pairs[{index}]: {error}")),
        );
    }
    errors
}

pub fn parse_dataset(input: &str) -> Result<MathMetamorphicDatasetV1, String> {
    let dataset: MathMetamorphicDatasetV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid MathMetamorphicDatasetV1 JSON: {error}"))?;
    let errors = validate_dataset(&dataset);
    if errors.is_empty() {
        Ok(dataset)
    } else {
        Err(errors.join("; "))
    }
}

pub fn validation_receipts(
    dataset: &MathMetamorphicDatasetV1,
) -> Result<Vec<PairValidationV1>, String> {
    let errors = validate_dataset(dataset);
    if !errors.is_empty() {
        return Err(errors.join("; "));
    }
    dataset
        .pairs
        .iter()
        .map(|pair| {
            let evaluator = pair.evaluator_only.as_ref().ok_or_else(|| {
                format!(
                    "pair {} has no evaluator-only adjudication",
                    pair.candidate.pair_id
                )
            })?;
            Ok(PairValidationV1 {
                pair_id: pair.candidate.pair_id.clone(),
                oracle_relation: evaluator.oracle_relation,
                transformation_class: evaluator.transformation_class,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> MathMetamorphicDatasetV1 {
        parse_dataset(FIXTURE).expect("valid finite metamorphic fixture")
    }

    #[test]
    fn fixture_has_equivalence_semantic_change_and_route_only_controls() {
        let dataset = fixture();
        let receipts = validation_receipts(&dataset).expect("validated receipts");
        assert_eq!(receipts.len(), 3);
        assert_eq!(receipts[0].oracle_relation, OracleRelationV1::Equivalent);
        assert_eq!(
            receipts[1].oracle_relation,
            OracleRelationV1::VariantEntailsBase
        );
        assert_eq!(
            receipts[2].transformation_class,
            TransformationClassV1::ProofRouteOnly
        );
    }

    #[test]
    fn false_equivalence_claim_is_rejected() {
        let mut dataset = fixture();
        let evaluator = dataset.pairs[1].evaluator_only.as_mut().unwrap();
        evaluator.transformation_class = TransformationClassV1::SemanticsPreserving;
        evaluator.oracle_relation = OracleRelationV1::Equivalent;
        assert!(
            validate_dataset(&dataset)
                .iter()
                .any(|error| error.contains("oracle_relation mismatch"))
        );
    }

    #[test]
    fn changed_typed_sequent_with_stale_digest_is_rejected() {
        let mut dataset = fixture();
        dataset.pairs[0].candidate.variant.sequent.conclusion = BooleanFormula::Atom {
            name: "p".to_owned(),
        };
        assert!(
            validate_dataset(&dataset)
                .iter()
                .any(|error| error.contains("does not match the canonical typed sequent"))
        );
    }

    #[test]
    fn route_only_change_requires_same_statement_and_distinct_route_receipts() {
        let mut dataset = fixture();
        let pair = &mut dataset.pairs[2];
        pair.candidate.variant.proof_route_sha256 = pair.candidate.base.proof_route_sha256.clone();
        assert!(
            validate_dataset(&dataset)
                .iter()
                .any(|error| error.contains("transformation_class conflicts"))
        );
    }

    #[test]
    fn candidate_inputs_do_not_serialize_evaluator_gold() {
        let dataset = fixture();
        let candidate_json = serde_json::to_value(&dataset.pairs[0].candidate)
            .expect("candidate-only serialization");
        assert!(!candidate_json.to_string().contains("oracle_relation"));
        assert!(
            !candidate_json
                .to_string()
                .contains("adjudication_authority")
        );
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut value: Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
        value["unexpected"] = Value::Bool(true);
        let input = serde_json::to_string(&value).expect("serialize mutation");
        assert!(parse_dataset(&input).unwrap_err().contains("unknown field"));
    }
}
