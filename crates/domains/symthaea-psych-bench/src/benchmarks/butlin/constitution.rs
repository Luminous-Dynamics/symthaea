//! Canonical epistemic constitution for the Butlin 14-indicator program.
//!
//! This module does not establish empirical support for any indicator. It
//! freezes the admissibility rules that evidence-producing code must obey so
//! future experiments cannot silently regress to score blending or self-
//! authorization.

use std::collections::BTreeSet;

/// The canonical Butlin et al. (2023) indicator identifiers used by Symthaea.
pub const CANONICAL_BUTLIN_14: [&str; 14] = [
    "RPT-1", "RPT-2", "GWT-1", "GWT-2", "GWT-3", "GWT-4", "HOT-1", "HOT-2",
    "HOT-3", "HOT-4", "AST-1", "PP-1", "AE-1", "AE-2",
];

/// Evidence invariants that are intentionally stronger than a simple
/// "14/14" score. These are policy identifiers, not empirical findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceInvariant {
    CanonicalTaxonomyOnly,
    NoScalarConsciousnessScore,
    ArchitectureCannotSelfPromote,
    SharedSignalIsNotIndependentEvidence,
    SharedInterventionIsNotIndependentReplication,
    FailedPositiveControlIsInconclusive,
    QualifiedNullIsNotDemonstrated,
    WrongDirectionIsContradicted,
    NegativeRunsAreRetained,
    ThresholdsFrozenBeforeTargetInspection,
    FunctionalSupportRequiresIndependentDownstreamTask,
    EvidenceBindsExecutionIdentity,
}

pub const REQUIRED_INVARIANTS: [EvidenceInvariant; 12] = [
    EvidenceInvariant::CanonicalTaxonomyOnly,
    EvidenceInvariant::NoScalarConsciousnessScore,
    EvidenceInvariant::ArchitectureCannotSelfPromote,
    EvidenceInvariant::SharedSignalIsNotIndependentEvidence,
    EvidenceInvariant::SharedInterventionIsNotIndependentReplication,
    EvidenceInvariant::FailedPositiveControlIsInconclusive,
    EvidenceInvariant::QualifiedNullIsNotDemonstrated,
    EvidenceInvariant::WrongDirectionIsContradicted,
    EvidenceInvariant::NegativeRunsAreRetained,
    EvidenceInvariant::ThresholdsFrozenBeforeTargetInspection,
    EvidenceInvariant::FunctionalSupportRequiresIndependentDownstreamTask,
    EvidenceInvariant::EvidenceBindsExecutionIdentity,
];

/// Validate an externally supplied indicator-id set against the exact Butlin
/// 14 taxonomy. Order is irrelevant; duplicates and substitutions fail.
pub fn validate_indicator_ids<'a>(ids: impl IntoIterator<Item = &'a str>) -> Result<(), String> {
    let supplied: Vec<&str> = ids.into_iter().collect();
    if supplied.len() != CANONICAL_BUTLIN_14.len() {
        return Err(format!(
            "expected exactly {} indicator ids, got {}",
            CANONICAL_BUTLIN_14.len(),
            supplied.len()
        ));
    }

    let unique: BTreeSet<&str> = supplied.iter().copied().collect();
    if unique.len() != supplied.len() {
        return Err("duplicate Butlin indicator id".into());
    }

    let canonical: BTreeSet<&str> = CANONICAL_BUTLIN_14.into_iter().collect();
    if unique != canonical {
        let missing: Vec<_> = canonical.difference(&unique).copied().collect();
        let unexpected: Vec<_> = unique.difference(&canonical).copied().collect();
        return Err(format!(
            "non-canonical Butlin taxonomy: missing={missing:?}, unexpected={unexpected:?}"
        ));
    }

    Ok(())
}

/// Returns true only when every required evidence invariant is explicitly
/// present. Useful for persisted campaign manifests and CI contract tests.
pub fn validate_invariant_set(
    invariants: impl IntoIterator<Item = EvidenceInvariant>,
) -> Result<(), String> {
    let supplied: BTreeSet<_> = invariants.into_iter().collect();
    let required: BTreeSet<_> = REQUIRED_INVARIANTS.into_iter().collect();
    if supplied != required {
        let missing: Vec<_> = required.difference(&supplied).copied().collect();
        let unexpected: Vec<_> = supplied.difference(&required).copied().collect();
        return Err(format!(
            "Butlin evidence constitution mismatch: missing={missing:?}, unexpected={unexpected:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_taxonomy_is_exactly_fourteen_unique_ids() {
        validate_indicator_ids(CANONICAL_BUTLIN_14).unwrap();
    }

    #[test]
    fn rejects_legacy_iit_substitution() {
        let mut ids = CANONICAL_BUTLIN_14;
        ids[13] = "IIT-1";
        let err = validate_indicator_ids(ids).unwrap_err();
        assert!(err.contains("non-canonical"));
        assert!(err.contains("AE-2"));
        assert!(err.contains("IIT-1"));
    }

    #[test]
    fn rejects_duplicate_ids_even_when_count_is_fourteen() {
        let mut ids = CANONICAL_BUTLIN_14;
        ids[13] = "AE-1";
        assert!(validate_indicator_ids(ids).is_err());
    }

    #[test]
    fn required_invariant_set_is_closed() {
        validate_invariant_set(REQUIRED_INVARIANTS).unwrap();
    }

    #[test]
    fn missing_fail_closed_rule_is_rejected() {
        let truncated = REQUIRED_INVARIANTS
            .into_iter()
            .filter(|i| *i != EvidenceInvariant::FailedPositiveControlIsInconclusive);
        assert!(validate_invariant_set(truncated).is_err());
    }
}
