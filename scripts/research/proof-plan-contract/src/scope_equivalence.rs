//! Bounded Tier-1 scope comparability for unordered propositional hypotheses.
//!
//! This normalizer only sorts the explicitly unordered hypothesis collection.
//! It preserves premise order and multiplicity, conclusion structure, atom names,
//! and the exact Tier-0 LocalGlobalPairV1 contract. It does not implement
//! quantified alpha-renaming, definitional equality, or general logical equivalence.

use crate::local_global::BooleanSequent;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

#[cfg(test)]
const FIXTURE: &str =
    include_str!("../../../../docs/research/MATH_SCOPE_EQUIVALENCE_TIER1_V1.json");

pub const TIER1_NORMALIZER_VERSION: &str = "scope-tier1-unordered-hypotheses-v1";

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeEquivalenceDatasetV1 {
    pub schema_version: String,
    pub pairs: Vec<ScopeEquivalencePairV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeEquivalencePairV1 {
    pub candidate: CandidateScopePairV1,
    pub evaluator_only: Option<EvaluatorScopePairV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateScopePairV1 {
    pub pair_id: String,
    pub source: ScopeSourceIdentityV1,
    pub base: BooleanSequent,
    pub variant: BooleanSequent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeSourceIdentityV1 {
    pub repository: String,
    pub revision: String,
    pub declaration: String,
    pub task_family: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorScopePairV1 {
    pub normalizer_version: String,
    pub expected_relation: ScopeRelationV1,
    pub adjudication_authority: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScopeRelationV1 {
    ExactIdentity,
    Tier1HypothesisPermutationEquivalent,
    NotTier1Equivalent,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ScopeEquivalenceReportV1 {
    pub pair_id: String,
    pub normalizer_version: String,
    pub base_raw_sha256: String,
    pub variant_raw_sha256: String,
    pub base_normalized_sha256: String,
    pub variant_normalized_sha256: String,
    pub relation: ScopeRelationV1,
}

fn domain_separated_sha256<T: Serialize>(domain: &str, value: &T) -> Result<String, String> {
    let canonical = serde_json::to_vec(value)
        .map_err(|error| format!("could not serialize {domain} preimage: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.scope_equivalence.tier1.v1:");
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(canonical);
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn normalize_hypothesis_set(sequent: &BooleanSequent) -> Result<BooleanSequent, String> {
    let mut normalized = sequent.clone();
    let mut keyed = normalized
        .hypotheses
        .iter()
        .map(|formula| {
            serde_json::to_string(formula)
                .map(|key| (key, formula.clone()))
                .map_err(|error| format!("could not canonicalize hypothesis AST: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Sorting is stable and does not deduplicate: multiplicity is retained.
    keyed.sort_by(|left, right| left.0.cmp(&right.0));
    normalized.hypotheses = keyed.into_iter().map(|(_, formula)| formula).collect();
    Ok(normalized)
}

/// Hash a typed sequent without normalization. This is not the existing
/// LocalGlobalPairV1 digest; it is a domain-separated Tier-1 fixture digest.
pub fn raw_sequent_sha256(sequent: &BooleanSequent) -> Result<String, String> {
    domain_separated_sha256("raw_sequent", sequent)
}

/// Hash a typed sequent after sorting only the explicitly unordered hypotheses.
/// Premises retain their input order and multiplicity; conclusion structure and
/// all atom identities are preserved exactly.
pub fn normalized_sequent_sha256(sequent: &BooleanSequent) -> Result<String, String> {
    domain_separated_sha256("normalized_sequent", &normalize_hypothesis_set(sequent)?)
}

fn derive_relation(
    base: &BooleanSequent,
    variant: &BooleanSequent,
) -> Result<(ScopeRelationV1, String, String, String, String), String> {
    let base_raw = raw_sequent_sha256(base)?;
    let variant_raw = raw_sequent_sha256(variant)?;
    let base_normalized = normalized_sequent_sha256(base)?;
    let variant_normalized = normalized_sequent_sha256(variant)?;

    let relation = if base_raw == variant_raw {
        ScopeRelationV1::ExactIdentity
    } else if base_normalized == variant_normalized {
        ScopeRelationV1::Tier1HypothesisPermutationEquivalent
    } else {
        ScopeRelationV1::NotTier1Equivalent
    };

    Ok((
        relation,
        base_raw,
        variant_raw,
        base_normalized,
        variant_normalized,
    ))
}

pub fn validate_pair(pair: &ScopeEquivalencePairV1) -> Vec<String> {
    let mut errors = Vec::new();
    let candidate = &pair.candidate;

    for (field, value) in [
        ("pair_id", &candidate.pair_id),
        ("source.repository", &candidate.source.repository),
        ("source.revision", &candidate.source.revision),
        ("source.declaration", &candidate.source.declaration),
        ("source.task_family", &candidate.source.task_family),
    ] {
        if value.trim().is_empty() {
            errors.push(format!("candidate.{field} must be non-empty"));
        }
    }

    let Some(evaluator) = pair.evaluator_only.as_ref() else {
        errors.push("evaluator_only adjudication is required for dataset validation".to_owned());
        return errors;
    };

    if evaluator.normalizer_version != TIER1_NORMALIZER_VERSION {
        errors.push(format!(
            "evaluator_only.normalizer_version must equal {TIER1_NORMALIZER_VERSION}"
        ));
    }
    if evaluator.adjudication_authority.trim().is_empty() {
        errors.push("evaluator_only.adjudication_authority must be non-empty".to_owned());
    }
    if evaluator.evidence_refs.is_empty()
        || evaluator.evidence_refs.iter().any(|reference| reference.trim().is_empty())
    {
        errors.push("evaluator_only.evidence_refs must be non-empty".to_owned());
    }
    let unique_refs: HashSet<&str> = evaluator.evidence_refs.iter().map(String::as_str).collect();
    if unique_refs.len() != evaluator.evidence_refs.len() {
        errors.push("evaluator_only.evidence_refs must be unique".to_owned());
    }

    match derive_relation(&candidate.base, &candidate.variant) {
        Ok((derived, _, _, _, _)) => {
            if derived != evaluator.expected_relation {
                errors.push(format!(
                    "evaluator_only.expected_relation mismatch: declared {:?}, derived {:?}",
                    evaluator.expected_relation, derived
                ));
            }
        }
        Err(error) => errors.push(format!("Tier-1 normalization failed: {error}")),
    }

    errors
}

pub fn validate_dataset(dataset: &ScopeEquivalenceDatasetV1) -> Vec<String> {
    let mut errors = Vec::new();
    if dataset.schema_version != "1.0.0" {
        errors.push("schema_version must equal 1.0.0".to_owned());
    }
    if dataset.pairs.is_empty() {
        errors.push("pairs must contain at least one pair".to_owned());
    }

    let mut ids = HashSet::new();
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

pub fn parse_dataset(input: &str) -> Result<ScopeEquivalenceDatasetV1, String> {
    let dataset: ScopeEquivalenceDatasetV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid ScopeEquivalenceDatasetV1 JSON: {error}"))?;
    let errors = validate_dataset(&dataset);
    if errors.is_empty() {
        Ok(dataset)
    } else {
        Err(errors.join("; "))
    }
}

pub fn validation_reports(
    dataset: &ScopeEquivalenceDatasetV1,
) -> Result<Vec<ScopeEquivalenceReportV1>, String> {
    let errors = validate_dataset(dataset);
    if !errors.is_empty() {
        return Err(errors.join("; "));
    }

    dataset
        .pairs
        .iter()
        .map(|pair| {
            let (relation, base_raw, variant_raw, base_normalized, variant_normalized) =
                derive_relation(&pair.candidate.base, &pair.candidate.variant)?;
            Ok(ScopeEquivalenceReportV1 {
                pair_id: pair.candidate.pair_id.clone(),
                normalizer_version: TIER1_NORMALIZER_VERSION.to_owned(),
                base_raw_sha256: base_raw,
                variant_raw_sha256: variant_raw,
                base_normalized_sha256: base_normalized,
                variant_normalized_sha256: variant_normalized,
                relation,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_global::BooleanFormula;
    use serde_json::Value;

    fn fixture() -> ScopeEquivalenceDatasetV1 {
        parse_dataset(FIXTURE).expect("valid Tier-1 scope-equivalence fixture")
    }

    fn atom(name: &str) -> BooleanFormula {
        BooleanFormula::Atom {
            name: name.to_owned(),
        }
    }

    #[test]
    fn fixture_separates_exact_tier1_and_non_equivalent_relations() {
        let dataset = fixture();
        let reports = validation_reports(&dataset).expect("validated reports");
        assert_eq!(reports.len(), 5);
        assert_eq!(reports[0].relation, ScopeRelationV1::ExactIdentity);
        assert_eq!(
            reports[1].relation,
            ScopeRelationV1::Tier1HypothesisPermutationEquivalent
        );
        assert_eq!(reports[2].relation, ScopeRelationV1::NotTier1Equivalent);
        assert_eq!(reports[3].relation, ScopeRelationV1::NotTier1Equivalent);
        assert_eq!(reports[4].relation, ScopeRelationV1::NotTier1Equivalent);
    }

    #[test]
    fn hypothesis_permutation_changes_raw_digest_but_not_normalized_digest() {
        let dataset = fixture();
        let report = validation_reports(&dataset).expect("validated reports")
            .into_iter()
            .find(|report| report.pair_id == "SCOPE-T1-HYPOTHESIS-PERMUTATION-001")
            .expect("hypothesis permutation pair");
        assert_ne!(report.base_raw_sha256, report.variant_raw_sha256);
        assert_eq!(report.base_normalized_sha256, report.variant_normalized_sha256);
        assert_eq!(
            report.relation,
            ScopeRelationV1::Tier1HypothesisPermutationEquivalent
        );
    }

    #[test]
    fn premise_order_is_not_normalized() {
        let dataset = fixture();
        let report = validation_reports(&dataset).expect("validated reports")
            .into_iter()
            .find(|report| report.pair_id == "SCOPE-T1-PREMISE-ORDER-001")
            .expect("premise order pair");
        assert_ne!(report.base_normalized_sha256, report.variant_normalized_sha256);
    }

    #[test]
    fn hypothesis_multiplicity_is_not_deduplicated() {
        let dataset = fixture();
        let report = validation_reports(&dataset).expect("validated reports")
            .into_iter()
            .find(|report| report.pair_id == "SCOPE-T1-HYPOTHESIS-MULTIPLICITY-001")
            .expect("hypothesis multiplicity pair");
        assert_ne!(report.base_normalized_sha256, report.variant_normalized_sha256);
    }

    #[test]
    fn conclusion_structure_is_not_normalized() {
        let dataset = fixture();
        let report = validation_reports(&dataset).expect("validated reports")
            .into_iter()
            .find(|report| report.pair_id == "SCOPE-T1-CONCLUSION-CHANGE-001")
            .expect("conclusion change pair");
        assert_ne!(report.base_normalized_sha256, report.variant_normalized_sha256);
    }

    #[test]
    fn false_equivalence_adjudication_is_rejected() {
        let mut dataset = fixture();
        let evaluator = dataset.pairs[2].evaluator_only.as_mut().unwrap();
        evaluator.expected_relation = ScopeRelationV1::Tier1HypothesisPermutationEquivalent;
        assert!(validate_dataset(&dataset)
            .iter()
            .any(|error| error.contains("expected_relation mismatch")));
    }

    #[test]
    fn candidate_serialization_excludes_evaluator_gold() {
        let dataset = fixture();
        let candidate_json = serde_json::to_value(&dataset.pairs[1].candidate)
            .expect("candidate-only serialization");
        assert!(!candidate_json.to_string().contains("expected_relation"));
        assert!(!candidate_json.to_string().contains("adjudication_authority"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut value: Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
        value["unexpected"] = Value::Bool(true);
        let input = serde_json::to_string(&value).expect("serialize mutation");
        assert!(parse_dataset(&input).unwrap_err().contains("unknown field"));
    }

    #[test]
    fn raw_and_normalized_hashes_are_domain_separated() {
        let dataset = fixture();
        let sequent = dataset.pairs[1].candidate.base.clone();
        assert_ne!(
            raw_sequent_sha256(&sequent).unwrap(),
            normalized_sequent_sha256(&sequent).unwrap()
        );
    }

    #[test]
    fn all_permutations_of_five_hypotheses_share_one_normalized_digest() {
        fn permutations(
            items: &mut [BooleanFormula],
            start: usize,
            output: &mut Vec<Vec<BooleanFormula>>,
        ) {
            if start == items.len() {
                output.push(items.to_vec());
                return;
            }
            for index in start..items.len() {
                items.swap(start, index);
                permutations(items, start + 1, output);
                items.swap(start, index);
            }
        }

        let hypotheses = vec![atom("A"), atom("B"), atom("C"), atom("D"), atom("E")];
        let baseline = BooleanSequent {
            hypotheses: hypotheses.clone(),
            premises: vec![atom("R")],
            conclusion: atom("Z"),
        };
        let expected_normalized = normalized_sequent_sha256(&baseline).unwrap();
        let baseline_raw = raw_sequent_sha256(&baseline).unwrap();
        let mut variants = Vec::new();
        let mut mutable = hypotheses;
        permutations(&mut mutable, 0, &mut variants);
        assert_eq!(variants.len(), 120);

        let mut raw_digests = HashSet::new();
        for variant_hypotheses in variants {
            let variant = BooleanSequent {
                hypotheses: variant_hypotheses,
                premises: baseline.premises.clone(),
                conclusion: baseline.conclusion.clone(),
            };
            assert_eq!(
                normalized_sequent_sha256(&variant).unwrap(),
                expected_normalized
            );
            let variant_raw = raw_sequent_sha256(&variant).unwrap();
            let expected_relation = if variant_raw == baseline_raw {
                ScopeRelationV1::ExactIdentity
            } else {
                ScopeRelationV1::Tier1HypothesisPermutationEquivalent
            };
            assert_eq!(derive_relation(&baseline, &variant).unwrap().0, expected_relation);
            raw_digests.insert(variant_raw);
        }
        assert_eq!(raw_digests.len(), 120);
    }

    #[test]
    fn normalization_preserves_atom_names() {
        let mut sequent = fixture().pairs[1].candidate.base.clone();
        sequent.hypotheses.push(atom("R"));
        let normalized = normalize_hypothesis_set(&sequent).unwrap();
        assert!(normalized.hypotheses.iter().any(|formula| {
            matches!(formula, BooleanFormula::Atom { name } if name.as_str() == "R")
        }));
    }
}
