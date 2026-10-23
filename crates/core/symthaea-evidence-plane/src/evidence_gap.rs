//! Evidence-gap decomposition for scientific challenge criteria.
//!
//! This module is deliberately criterion-agnostic: it tracks which externally
//! authored predicates have evidence, rather than deciding whether the science
//! is true.

use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidencePredicate {
    pub id: String,
    pub required_event_kinds: BTreeSet<String>,
    pub satisfied: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceGap {
    pub predicate_id: String,
    pub missing_event_kinds: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceGapError {
    EmptyPredicateId,
    NoRequiredEvidenceKinds,
    DuplicatePredicate(String),
}

pub fn validate_predicates(predicates: &[EvidencePredicate]) -> Result<(), EvidenceGapError> {
    let mut ids = BTreeSet::new();
    for predicate in predicates {
        if predicate.id.trim().is_empty() {
            return Err(EvidenceGapError::EmptyPredicateId);
        }
        if predicate.required_event_kinds.is_empty() {
            return Err(EvidenceGapError::NoRequiredEvidenceKinds);
        }
        if !ids.insert(predicate.id.as_str()) {
            return Err(EvidenceGapError::DuplicatePredicate(predicate.id.clone()));
        }
    }
    Ok(())
}

/// Returns only unmet evidence requirements. This is a gap report, not a
/// confidence score or scientific-truth judgment.
pub fn identify_gaps(predicates: &[EvidencePredicate]) -> Result<Vec<EvidenceGap>, EvidenceGapError> {
    validate_predicates(predicates)?;
    Ok(predicates.iter()
        .filter(|p| !p.satisfied)
        .map(|p| EvidenceGap {
            predicate_id: p.id.clone(),
            missing_event_kinds: p.required_event_kinds.clone(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn reports_only_unmet_predicates() {
        let predicates = vec![
            EvidencePredicate { id: "observation".into(), required_event_kinds: set(&["ExternalExperimentalObservation"]), satisfied: true },
            EvidencePredicate { id: "replication".into(), required_event_kinds: set(&["IndependentReplication"]), satisfied: false },
        ];
        let gaps = identify_gaps(&predicates).unwrap();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].predicate_id, "replication");
    }

    #[test]
    fn preserves_all_required_evidence_kinds() {
        let predicates = vec![
            EvidencePredicate { id: "completion".into(), required_event_kinds: set(&["Observation", "Replication", "CriterionEvidence"]), satisfied: false },
        ];
        let gaps = identify_gaps(&predicates).unwrap();
        assert_eq!(gaps[0].missing_event_kinds, set(&["Observation", "Replication", "CriterionEvidence"]));
    }

    #[test]
    fn rejects_duplicate_predicates() {
        let predicates = vec![
            EvidencePredicate { id: "x".into(), required_event_kinds: set(&["Observation"]), satisfied: false },
            EvidencePredicate { id: "x".into(), required_event_kinds: set(&["Replication"]), satisfied: false },
        ];
        assert_eq!(validate_predicates(&predicates), Err(EvidenceGapError::DuplicatePredicate("x".into())));
    }

    #[test]
    fn rejects_empty_requirements() {
        let predicates = vec![
            EvidencePredicate { id: "x".into(), required_event_kinds: BTreeSet::new(), satisfied: false },
        ];
        assert_eq!(validate_predicates(&predicates), Err(EvidenceGapError::NoRequiredEvidenceKinds));
    }
}
