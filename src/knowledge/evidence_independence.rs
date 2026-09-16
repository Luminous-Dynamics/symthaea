// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence source-ancestry analysis.
//!
//! Multiple evidence records are not automatically independent corroboration.
//! Two reports may be copies of the same paper, two datasets may descend from
//! the same collection, and a synthesis may combine already-counted sources.
//!
//! This module performs a deliberately narrow analysis: it traces provenance
//! ancestry and reports shared roots. A distinct provenance root is **not** a
//! proof of statistical or institutional independence; it is only evidence that
//! two records do not share a declared source ancestor.

use super::claim_evidence::{
    ClaimId, EpistemicLedger, EvidenceId, EvidencePolarity, ProvenanceId,
};
use std::collections::{BTreeSet, HashSet};

/// Provenance roots reached from one evidence record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceLineage {
    pub evidence_id: EvidenceId,
    pub root_ids: Vec<ProvenanceId>,
}

/// A pair of evidence records with one or more declared source roots in common.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedAncestry {
    pub left: EvidenceId,
    pub right: EvidenceId,
    pub shared_root_ids: Vec<ProvenanceId>,
}

/// Descriptive source-diversity report for one claim and one evidence polarity.
///
/// This intentionally does not emit a truth score or confidence multiplier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceDiversityReport {
    pub claim_id: ClaimId,
    pub polarity: EvidencePolarity,
    pub evidence_count: usize,
    /// Number of distinct declared ultimate provenance roots across the records.
    pub distinct_root_count: usize,
    pub lineages: Vec<EvidenceLineage>,
    pub shared_ancestry: Vec<SharedAncestry>,
}

impl ProvenanceDiversityReport {
    /// True when at least one pair of evidence records descends from a common
    /// declared provenance root.
    pub fn has_shared_ancestry(&self) -> bool {
        !self.shared_ancestry.is_empty()
    }

    /// Ratio of distinct declared roots to evidence records, bounded to [0, 1].
    ///
    /// This is a provenance-diversity diagnostic only. It must not be treated as
    /// a probability of correctness or as proof of source independence.
    pub fn root_diversity_ratio(&self) -> f64 {
        if self.evidence_count == 0 {
            return 0.0;
        }
        (self.distinct_root_count as f64 / self.evidence_count as f64).min(1.0)
    }
}

/// Trace declared provenance ancestry for evidence attached to a claim.
#[derive(Debug, Default, Clone, Copy)]
pub struct EvidenceIndependenceAnalyzer;

impl EvidenceIndependenceAnalyzer {
    /// Build a provenance-diversity report for evidence with the requested
    /// polarity. Supporting and contradicting evidence are therefore analyzed
    /// separately rather than accidentally cancelling or corroborating each other.
    pub fn analyze(
        ledger: &EpistemicLedger,
        claim_id: ClaimId,
        polarity: EvidencePolarity,
    ) -> ProvenanceDiversityReport {
        let evidence = ledger
            .evidence_for_claim(claim_id)
            .into_iter()
            .filter(|record| record.polarity == polarity)
            .collect::<Vec<_>>();

        let mut lineages = Vec::with_capacity(evidence.len());
        let mut all_roots = BTreeSet::new();

        for record in evidence {
            let roots = provenance_roots(ledger, record.provenance_id);
            all_roots.extend(roots.iter().copied());
            lineages.push(EvidenceLineage {
                evidence_id: record.id,
                root_ids: roots.into_iter().collect(),
            });
        }

        let mut shared_ancestry = Vec::new();
        for i in 0..lineages.len() {
            for j in (i + 1)..lineages.len() {
                let left_roots: BTreeSet<_> = lineages[i].root_ids.iter().copied().collect();
                let right_roots: BTreeSet<_> = lineages[j].root_ids.iter().copied().collect();
                let shared = left_roots
                    .intersection(&right_roots)
                    .copied()
                    .collect::<Vec<_>>();
                if !shared.is_empty() {
                    shared_ancestry.push(SharedAncestry {
                        left: lineages[i].evidence_id,
                        right: lineages[j].evidence_id,
                        shared_root_ids: shared,
                    });
                }
            }
        }

        ProvenanceDiversityReport {
            claim_id,
            polarity,
            evidence_count: lineages.len(),
            distinct_root_count: all_roots.len(),
            lineages,
            shared_ancestry,
        }
    }
}

/// Resolve ultimate declared provenance roots.
///
/// The ledger API makes provenance ancestry acyclic by requiring parents to
/// exist before children are created. `visited` is retained as a defensive guard
/// for future imported/persisted provenance records that may bypass that API.
fn provenance_roots(
    ledger: &EpistemicLedger,
    start: ProvenanceId,
) -> BTreeSet<ProvenanceId> {
    fn walk(
        ledger: &EpistemicLedger,
        current: ProvenanceId,
        visited: &mut HashSet<ProvenanceId>,
        roots: &mut BTreeSet<ProvenanceId>,
    ) {
        if !visited.insert(current) {
            return;
        }

        let Some(record) = ledger.provenance(current) else {
            return;
        };

        if record.parent_ids.is_empty() {
            roots.insert(current);
            return;
        }

        for parent in &record.parent_ids {
            walk(ledger, *parent, visited, roots);
        }
    }

    let mut visited = HashSet::new();
    let mut roots = BTreeSet::new();
    walk(ledger, start, &mut visited, &mut roots);
    roots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::claim_evidence::{ClaimKind, EvidenceKind};

    fn root(ledger: &mut EpistemicLedger, name: &str) -> ProvenanceId {
        ledger
            .add_provenance(name, None, None, 1, vec![])
            .expect("root source")
    }

    fn supporting_report(
        ledger: &mut EpistemicLedger,
        claim: ClaimId,
        provenance: ProvenanceId,
    ) -> EvidenceId {
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Report,
                EvidencePolarity::Supports,
                provenance,
                2,
                None,
                None,
            )
            .expect("supporting report")
    }

    #[test]
    fn copied_reports_do_not_count_as_distinct_roots() {
        let mut ledger = EpistemicLedger::new();
        let original = root(&mut ledger, "original-paper");
        let copy_a = ledger
            .add_provenance("news-a", None, None, 2, vec![original])
            .unwrap();
        let copy_b = ledger
            .add_provenance("news-b", None, None, 2, vec![original])
            .unwrap();
        let claim = ledger.add_claim("X is true", ClaimKind::Descriptive, None, None, 1);

        supporting_report(&mut ledger, claim, copy_a);
        supporting_report(&mut ledger, claim, copy_b);

        let report = EvidenceIndependenceAnalyzer::analyze(
            &ledger,
            claim,
            EvidencePolarity::Supports,
        );
        assert_eq!(report.evidence_count, 2);
        assert_eq!(report.distinct_root_count, 1);
        assert!(report.has_shared_ancestry());
        assert_eq!(report.shared_ancestry.len(), 1);
        assert_eq!(report.shared_ancestry[0].shared_root_ids, vec![original]);
        assert!((report.root_diversity_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn separately_rooted_reports_have_no_declared_shared_ancestry() {
        let mut ledger = EpistemicLedger::new();
        let source_a = root(&mut ledger, "lab-a");
        let source_b = root(&mut ledger, "lab-b");
        let claim = ledger.add_claim("X is true", ClaimKind::Descriptive, None, None, 1);

        supporting_report(&mut ledger, claim, source_a);
        supporting_report(&mut ledger, claim, source_b);

        let report = EvidenceIndependenceAnalyzer::analyze(
            &ledger,
            claim,
            EvidencePolarity::Supports,
        );
        assert_eq!(report.distinct_root_count, 2);
        assert!(!report.has_shared_ancestry());
        assert_eq!(report.root_diversity_ratio(), 1.0);
    }

    #[test]
    fn synthesis_exposes_all_ultimate_roots() {
        let mut ledger = EpistemicLedger::new();
        let source_a = root(&mut ledger, "dataset-a");
        let source_b = root(&mut ledger, "dataset-b");
        let synthesis = ledger
            .add_provenance("meta-analysis", None, None, 2, vec![source_a, source_b])
            .unwrap();
        let claim = ledger.add_claim("X is true", ClaimKind::Descriptive, None, None, 1);
        let evidence_id = supporting_report(&mut ledger, claim, synthesis);

        let report = EvidenceIndependenceAnalyzer::analyze(
            &ledger,
            claim,
            EvidencePolarity::Supports,
        );
        assert_eq!(report.evidence_count, 1);
        assert_eq!(report.distinct_root_count, 2);
        assert_eq!(report.lineages[0].evidence_id, evidence_id);
        assert_eq!(report.lineages[0].root_ids, vec![source_a, source_b]);
        // Ratio is capped: two roots do not make one record "200% independent".
        assert_eq!(report.root_diversity_ratio(), 1.0);
    }

    #[test]
    fn supporting_and_contradicting_lineages_are_not_mixed() {
        let mut ledger = EpistemicLedger::new();
        let support_source = root(&mut ledger, "support");
        let contradiction_source = root(&mut ledger, "contradiction");
        let claim = ledger.add_claim("X is true", ClaimKind::Descriptive, None, None, 1);

        supporting_report(&mut ledger, claim, support_source);
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Contradicts,
                contradiction_source,
                3,
                None,
                None,
            )
            .unwrap();

        let supports = EvidenceIndependenceAnalyzer::analyze(
            &ledger,
            claim,
            EvidencePolarity::Supports,
        );
        let contradictions = EvidenceIndependenceAnalyzer::analyze(
            &ledger,
            claim,
            EvidencePolarity::Contradicts,
        );

        assert_eq!(supports.evidence_count, 1);
        assert_eq!(supports.lineages[0].root_ids, vec![support_source]);
        assert_eq!(contradictions.evidence_count, 1);
        assert_eq!(
            contradictions.lineages[0].root_ids,
            vec![contradiction_source]
        );
    }

    #[test]
    fn empty_claim_report_is_neutral_not_perfect_diversity() {
        let mut ledger = EpistemicLedger::new();
        let claim = ledger.add_claim("untested", ClaimKind::Descriptive, None, None, 1);
        let report = EvidenceIndependenceAnalyzer::analyze(
            &ledger,
            claim,
            EvidencePolarity::Supports,
        );
        assert_eq!(report.evidence_count, 0);
        assert_eq!(report.distinct_root_count, 0);
        assert_eq!(report.root_diversity_ratio(), 0.0);
    }
}
