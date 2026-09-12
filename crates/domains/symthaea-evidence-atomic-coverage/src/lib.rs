// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Atomic sub-claim coverage assurance for safety-evidence receipts.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_formal_safety::{
    SafetyCase, SafetyEvidenceReceipt, StrictSafetyCaseReport, StrictSafetyCaseStatus,
    assess_strict_safety_case,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtomicEvidenceFacet {
    pub facet_id: String,
    pub obligation_key: String,
    pub controlled_claim: String,
    pub minimum_distinct_receipts: usize,
    pub minimum_distinct_evidence_objects: usize,
    pub evidence_refs: Vec<String>,
}

impl AtomicEvidenceFacet {
    pub fn validate(&self) -> bool {
        !self.facet_id.trim().is_empty()
            && !self.obligation_key.trim().is_empty()
            && !self.controlled_claim.trim().is_empty()
            && self.minimum_distinct_receipts >= 1
            && self.minimum_distinct_evidence_objects >= 1
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtomicCoveragePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub facets: Vec<AtomicEvidenceFacet>,
    pub evidence_refs: Vec<String>,
}

impl AtomicCoveragePolicy {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.policy_id.trim().is_empty()
            && !self.facets.is_empty()
            && self.facets.iter().all(AtomicEvidenceFacet::validate)
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FacetEvidenceBinding {
    pub binding_id: String,
    pub receipt_id: String,
    pub facet_id: String,
    /// Exact sub-artifact or result relevant to this facet.
    pub facet_evidence_ref: String,
    /// Content digest over the exact facet-level evidence object.
    pub facet_evidence_digest: String,
    pub rationale_ref: String,
}

impl FacetEvidenceBinding {
    pub fn validate(&self) -> bool {
        !self.binding_id.trim().is_empty()
            && !self.receipt_id.trim().is_empty()
            && !self.facet_id.trim().is_empty()
            && !self.facet_evidence_ref.trim().is_empty()
            && valid_digest(&self.facet_evidence_digest)
            && !self.rationale_ref.trim().is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtomicCoverageIssue {
    InvalidPolicy,
    InvalidFacet(String),
    DuplicateFacetId(String),
    UnknownObligation(String),
    InvalidBinding(String),
    DuplicateBindingId(String),
    DuplicateReceiptFacetBinding {
        receipt_id: String,
        facet_id: String,
    },
    UnknownReceipt(String),
    UnknownFacet(String),
    ReceiptObligationMismatch {
        receipt_id: String,
        facet_id: String,
    },
    MissingFacetCoverage(String),
    InsufficientDistinctReceipts {
        facet_id: String,
        observed: usize,
        required: usize,
    },
    InsufficientDistinctEvidenceObjects {
        facet_id: String,
        observed: usize,
        required: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FacetCoverageReport {
    pub facet_id: String,
    pub obligation_key: String,
    pub binding_count: usize,
    pub distinct_receipts: usize,
    pub distinct_evidence_objects: usize,
    pub satisfied: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtomicCoverageReport {
    pub policy_id: String,
    pub status: StrictSafetyCaseStatus,
    pub issues: Vec<AtomicCoverageIssue>,
    pub facet_reports: Vec<FacetCoverageReport>,
    pub strict_report: StrictSafetyCaseReport,
}

impl AtomicCoverageReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Assess reviewed atomic sub-claim coverage on top of ordinary strict readiness.
///
/// Parent receipts do not implicitly cover any facet. Only explicit valid bindings
/// count, and this layer can only retain or reduce strict readiness.
pub fn assess_atomic_coverage(
    safety_case: &SafetyCase,
    receipts: &[SafetyEvidenceReceipt],
    policy: &AtomicCoveragePolicy,
    bindings: &[FacetEvidenceBinding],
) -> AtomicCoverageReport {
    let strict_report = assess_strict_safety_case(safety_case, receipts);
    let mut issues = Vec::new();
    let mut facet_reports = Vec::new();

    if !policy.validate() {
        issues.push(AtomicCoverageIssue::InvalidPolicy);
    }

    let obligation_keys = safety_case
        .obligations
        .iter()
        .map(|obligation| obligation.stable_key())
        .collect::<BTreeSet<_>>();

    let mut facets = BTreeMap::<String, &AtomicEvidenceFacet>::new();
    for facet in &policy.facets {
        if !facet.validate() {
            issues.push(AtomicCoverageIssue::InvalidFacet(facet.facet_id.clone()));
            continue;
        }
        if !obligation_keys.contains(&facet.obligation_key) {
            issues.push(AtomicCoverageIssue::UnknownObligation(
                facet.obligation_key.clone(),
            ));
            continue;
        }
        if facets.insert(facet.facet_id.clone(), facet).is_some() {
            issues.push(AtomicCoverageIssue::DuplicateFacetId(
                facet.facet_id.clone(),
            ));
        }
    }

    let mut receipts_by_id = BTreeMap::<String, &SafetyEvidenceReceipt>::new();
    for receipt in receipts {
        if receipt.validate() {
            receipts_by_id.entry(receipt.receipt_id.clone()).or_insert(receipt);
        }
    }

    let mut binding_ids = BTreeSet::new();
    let mut receipt_facet_pairs = BTreeSet::new();
    let mut accepted = BTreeMap::<String, Vec<&FacetEvidenceBinding>>::new();

    for binding in bindings {
        if !binding.validate() {
            issues.push(AtomicCoverageIssue::InvalidBinding(
                binding.binding_id.clone(),
            ));
            continue;
        }
        if !binding_ids.insert(binding.binding_id.clone()) {
            issues.push(AtomicCoverageIssue::DuplicateBindingId(
                binding.binding_id.clone(),
            ));
            continue;
        }
        let pair = (binding.receipt_id.clone(), binding.facet_id.clone());
        if !receipt_facet_pairs.insert(pair.clone()) {
            issues.push(AtomicCoverageIssue::DuplicateReceiptFacetBinding {
                receipt_id: pair.0,
                facet_id: pair.1,
            });
            continue;
        }
        let Some(receipt) = receipts_by_id.get(&binding.receipt_id) else {
            issues.push(AtomicCoverageIssue::UnknownReceipt(
                binding.receipt_id.clone(),
            ));
            continue;
        };
        let Some(facet) = facets.get(&binding.facet_id) else {
            issues.push(AtomicCoverageIssue::UnknownFacet(binding.facet_id.clone()));
            continue;
        };
        if receipt.obligation_key != facet.obligation_key {
            issues.push(AtomicCoverageIssue::ReceiptObligationMismatch {
                receipt_id: binding.receipt_id.clone(),
                facet_id: binding.facet_id.clone(),
            });
            continue;
        }
        accepted
            .entry(binding.facet_id.clone())
            .or_default()
            .push(binding);
    }

    for (facet_id, facet) in &facets {
        let facet_bindings = accepted.get(facet_id).cloned().unwrap_or_default();
        let distinct_receipts = facet_bindings
            .iter()
            .map(|binding| binding.receipt_id.as_str())
            .collect::<BTreeSet<_>>();
        let distinct_evidence_objects = facet_bindings
            .iter()
            .map(|binding| binding.facet_evidence_digest.as_str())
            .collect::<BTreeSet<_>>();

        let mut satisfied = true;
        if facet_bindings.is_empty() {
            satisfied = false;
            issues.push(AtomicCoverageIssue::MissingFacetCoverage(
                facet_id.clone(),
            ));
        }
        if distinct_receipts.len() < facet.minimum_distinct_receipts {
            satisfied = false;
            issues.push(AtomicCoverageIssue::InsufficientDistinctReceipts {
                facet_id: facet_id.clone(),
                observed: distinct_receipts.len(),
                required: facet.minimum_distinct_receipts,
            });
        }
        if distinct_evidence_objects.len() < facet.minimum_distinct_evidence_objects {
            satisfied = false;
            issues.push(AtomicCoverageIssue::InsufficientDistinctEvidenceObjects {
                facet_id: facet_id.clone(),
                observed: distinct_evidence_objects.len(),
                required: facet.minimum_distinct_evidence_objects,
            });
        }

        facet_reports.push(FacetCoverageReport {
            facet_id: facet_id.clone(),
            obligation_key: facet.obligation_key.clone(),
            binding_count: facet_bindings.len(),
            distinct_receipts: distinct_receipts.len(),
            distinct_evidence_objects: distinct_evidence_objects.len(),
            satisfied,
        });
    }

    facet_reports.sort_by(|a, b| a.facet_id.cmp(&b.facet_id));

    let structurally_invalid = issues.iter().any(|issue| {
        matches!(
            issue,
            AtomicCoverageIssue::InvalidPolicy
                | AtomicCoverageIssue::InvalidFacet(_)
                | AtomicCoverageIssue::DuplicateFacetId(_)
                | AtomicCoverageIssue::UnknownObligation(_)
                | AtomicCoverageIssue::InvalidBinding(_)
                | AtomicCoverageIssue::DuplicateBindingId(_)
                | AtomicCoverageIssue::DuplicateReceiptFacetBinding { .. }
                | AtomicCoverageIssue::UnknownReceipt(_)
                | AtomicCoverageIssue::UnknownFacet(_)
                | AtomicCoverageIssue::ReceiptObligationMismatch { .. }
        )
    });
    let coverage_blocked = issues.iter().any(|issue| {
        matches!(
            issue,
            AtomicCoverageIssue::MissingFacetCoverage(_)
                | AtomicCoverageIssue::InsufficientDistinctReceipts { .. }
                | AtomicCoverageIssue::InsufficientDistinctEvidenceObjects { .. }
        )
    });

    let status = if structurally_invalid || strict_report.status == StrictSafetyCaseStatus::Invalid {
        StrictSafetyCaseStatus::Invalid
    } else if strict_report.status != StrictSafetyCaseStatus::Ready || coverage_blocked {
        StrictSafetyCaseStatus::Blocked
    } else {
        StrictSafetyCaseStatus::Ready
    };

    AtomicCoverageReport {
        policy_id: policy.policy_id.clone(),
        status,
        issues,
        facet_reports,
        strict_report,
    }
}

fn valid_digest(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && trimmed
            .split_once(':')
            .is_some_and(|(algorithm, digest)| !algorithm.is_empty() && !digest.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_formal_safety::{EvidenceKind, ObligationStatus, ProofObligation};

    fn case() -> SafetyCase {
        let mut case = SafetyCase::new("subject");
        case.add_obligation(ProofObligation::new("composite claim", EvidenceKind::Test).discharge("test:x"));
        case
    }

    fn receipt(case: &SafetyCase, id: &str, digest: &str) -> SafetyEvidenceReceipt {
        let obligation = &case.obligations[0];
        SafetyEvidenceReceipt {
            receipt_id: id.into(),
            obligation_key: obligation.stable_key(),
            evidence_kind: obligation.expected_evidence,
            evidence_ref: format!("bundle:{id}"),
            evidence_digest: digest.into(),
            verifier_ref: format!("verifier:{id}"),
            verified_at_ms: 100,
        }
    }

    fn facet(case: &SafetyCase, id: &str, min_receipts: usize, min_objects: usize) -> AtomicEvidenceFacet {
        AtomicEvidenceFacet {
            facet_id: id.into(),
            obligation_key: case.obligations[0].stable_key(),
            controlled_claim: format!("controlled facet {id}"),
            minimum_distinct_receipts: min_receipts,
            minimum_distinct_evidence_objects: min_objects,
            evidence_refs: vec![format!("facet-review:{id}")],
        }
    }

    fn policy(case: &SafetyCase) -> AtomicCoveragePolicy {
        AtomicCoveragePolicy {
            schema_version: "1".into(),
            policy_id: "atomic-v1".into(),
            facets: vec![facet(case, "facet:a", 1, 1), facet(case, "facet:b", 1, 1)],
            evidence_refs: vec!["policy:atomic-v1".into()],
        }
    }

    fn binding(id: &str, receipt: &str, facet: &str, digest: &str) -> FacetEvidenceBinding {
        FacetEvidenceBinding {
            binding_id: id.into(),
            receipt_id: receipt.into(),
            facet_id: facet.into(),
            facet_evidence_ref: format!("result:{id}"),
            facet_evidence_digest: digest.into(),
            rationale_ref: format!("rationale:{id}"),
        }
    }

    #[test]
    fn one_parent_receipt_can_cover_multiple_facets_only_with_explicit_bindings() {
        let case = case();
        let receipts = vec![receipt(&case, "r1", "blake3:bundle-r1")];
        let bindings = vec![
            binding("b1", "r1", "facet:a", "blake3:facet-a"),
            binding("b2", "r1", "facet:b", "blake3:facet-b"),
        ];
        let report = assess_atomic_coverage(&case, &receipts, &policy(&case), &bindings);
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn parent_receipt_does_not_implicitly_cover_any_facet() {
        let case = case();
        let receipts = vec![receipt(&case, "r1", "blake3:bundle-r1")];
        let report = assess_atomic_coverage(&case, &receipts, &policy(&case), &[]);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.facet_reports.iter().filter(|item| !item.satisfied).count(), 2);
    }

    #[test]
    fn covering_one_facet_does_not_cover_sibling_facet() {
        let case = case();
        let receipts = vec![receipt(&case, "r1", "blake3:bundle-r1")];
        let bindings = vec![binding("b1", "r1", "facet:a", "blake3:facet-a")];
        let report = assess_atomic_coverage(&case, &receipts, &policy(&case), &bindings);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert!(report.facet_reports.iter().any(|item| item.facet_id == "facet:b" && !item.satisfied));
    }

    #[test]
    fn minimum_distinct_evidence_objects_is_enforced() {
        let case = case();
        let mut policy = policy(&case);
        policy.facets = vec![facet(&case, "facet:a", 2, 2)];
        let receipts = vec![
            receipt(&case, "r1", "blake3:bundle-r1"),
            receipt(&case, "r2", "blake3:bundle-r2"),
        ];
        let same_object = vec![
            binding("b1", "r1", "facet:a", "blake3:same-result"),
            binding("b2", "r2", "facet:a", "blake3:same-result"),
        ];
        let report = assess_atomic_coverage(&case, &receipts, &policy, &same_object);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.facet_reports[0].distinct_receipts, 2);
        assert_eq!(report.facet_reports[0].distinct_evidence_objects, 1);

        let distinct = vec![
            binding("b1", "r1", "facet:a", "blake3:result-a"),
            binding("b2", "r2", "facet:a", "blake3:result-b"),
        ];
        let report = assess_atomic_coverage(&case, &receipts, &policy, &distinct);
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
    }

    #[test]
    fn receipt_from_wrong_obligation_is_invalid_binding() {
        let mut case = case();
        case.add_obligation(ProofObligation::new("other claim", EvidenceKind::Test).discharge("test:y"));
        let other = &case.obligations[1];
        let receipt = SafetyEvidenceReceipt {
            receipt_id: "r-other".into(),
            obligation_key: other.stable_key(),
            evidence_kind: other.expected_evidence,
            evidence_ref: "bundle:other".into(),
            evidence_digest: "blake3:other".into(),
            verifier_ref: "verifier:other".into(),
            verified_at_ms: 100,
        };
        let mut policy = policy(&case);
        policy.facets = vec![facet(&case, "facet:a", 1, 1)];
        let report = assess_atomic_coverage(
            &case,
            &[receipt],
            &policy,
            &[binding("b1", "r-other", "facet:a", "blake3:facet-a")],
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
    }

    #[test]
    fn duplicate_receipt_facet_binding_is_invalid() {
        let case = case();
        let receipts = vec![receipt(&case, "r1", "blake3:bundle-r1")];
        let bindings = vec![
            binding("b1", "r1", "facet:a", "blake3:facet-a"),
            binding("b2", "r1", "facet:a", "blake3:facet-a-2"),
        ];
        let report = assess_atomic_coverage(&case, &receipts, &policy(&case), &bindings);
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
    }

    #[test]
    fn atomic_gate_cannot_upgrade_blocked_strict_case() {
        let mut case = case();
        case.obligations[0].status = ObligationStatus::Open;
        let receipts = vec![receipt(&case, "r1", "blake3:bundle-r1")];
        let bindings = vec![
            binding("b1", "r1", "facet:a", "blake3:facet-a"),
            binding("b2", "r1", "facet:b", "blake3:facet-b"),
        ];
        let report = assess_atomic_coverage(&case, &receipts, &policy(&case), &bindings);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
    }
}
