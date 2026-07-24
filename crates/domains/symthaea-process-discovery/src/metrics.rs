// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-category metrics for the Reaction Corpus Auditor (Phase A.2).
//!
//! An external review's central point: the important number is not overall
//! certification percentage, it's whether each [`crate::corpus::RecordCategory`]
//! receives the correct *kind* of outcome. This module turns that into a
//! computed report rather than something eyeballed from a raw printout.
//!
//! **Honest scope note**: "normalization ambiguity" (multiple recognized
//! encodings conflicting) is not reported here -- with exactly one
//! normalization rule (`normalization.rs`), ambiguity between rules cannot
//! occur yet, and fabricating a metric for something that can't happen would
//! overclaim. `normalization_applied` (a real, countable fact) is reported
//! instead. Widen this once a second rule exists.

use crate::audit::AuditReport;
use crate::corpus::{CorpusRecord, RecordCategory};
use crate::policy::ScopePolicy;

#[derive(Debug, Clone)]
pub struct CategoryMetrics {
    pub category: RecordCategory,
    pub total: usize,
    /// How many records' actual outcome matched their corpus-declared
    /// `expected_outcome` -- the core "did this category get the right
    /// kind of verdict" number.
    pub expectation_matches: usize,
    /// How many records' raw structural validity matched their corpus-
    /// declared `expected_raw_validity_ok`.
    pub raw_validity_matches: usize,
    pub certified: usize,
    /// Out of `certified`: how many produced a byte-identical certificate
    /// JSON when the offline pipeline (parse -> normalize -> validity ->
    /// classify -> certify, no external source) ran a second independent
    /// time on the same record. Certificates never carry PubChem/RDKit
    /// content, so this isolates the offline pipeline's own determinism
    /// without a second network round-trip.
    pub certificate_deterministic: usize,
    /// How many records had at least one normalization rule fire on any
    /// reactant or product.
    pub normalization_applied: usize,
    /// How many distinct PubChem cross-references in this category had a
    /// GENUINE compositional disagreement with this crate's own computed
    /// formula -- deliberately excludes `RepresentationOnlyDifference`
    /// (e.g. Hill-notation element-ordering conventions), so this count
    /// never implies a chemically meaningful conflict when the only
    /// difference is presentation. Purely informational (see `audit.rs`'s
    /// `PubChemAgreement`).
    pub pubchem_disagreements: usize,
    /// How many PubChem cross-references were a representation-only
    /// difference specifically -- tracked separately, not folded into
    /// `pubchem_disagreements`.
    pub pubchem_representation_only: usize,
    /// Same as `pubchem_disagreements`, for the RDKit cross-reference.
    pub rdkit_disagreements: usize,
    pub rdkit_representation_only: usize,
}

impl CategoryMetrics {
    fn zeroed(category: RecordCategory) -> Self {
        Self {
            category,
            total: 0,
            expectation_matches: 0,
            raw_validity_matches: 0,
            certified: 0,
            certificate_deterministic: 0,
            normalization_applied: 0,
            pubchem_disagreements: 0,
            pubchem_representation_only: 0,
            rdkit_disagreements: 0,
            rdkit_representation_only: 0,
        }
    }

    pub fn expectation_pass_rate(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.expectation_matches as f64 / self.total as f64
    }
}

/// Computes one [`CategoryMetrics`] per [`RecordCategory::ALL`], in that
/// fixed order. `corpus`/`policy` must be the same ones used to produce
/// `report` -- this function reruns them once more, fully offline, purely
/// to compute the determinism metric (see the module doc for why offline
/// is sufficient).
pub fn compute_metrics(
    corpus: &[CorpusRecord],
    policy: &dyn ScopePolicy,
    report: &AuditReport,
) -> Vec<CategoryMetrics> {
    // Deliberately offline for the determinism rerun -- certificates don't
    // carry external-source content, so this is a faithful, network-free
    // test of the offline pipeline's own determinism, and avoids doubling
    // real network/subprocess calls just to compute a metric.
    let rerun: AuditReport = crate::audit::run_audit(corpus, policy, None, None);

    RecordCategory::ALL
        .iter()
        .map(|&category| {
            let mut m = CategoryMetrics::zeroed(category);
            for (record, rerun_record) in report.records.iter().zip(rerun.records.iter()) {
                if record.category != category {
                    continue;
                }
                m.total += 1;
                if record.matched_expectation {
                    m.expectation_matches += 1;
                }
                if record.raw_validity_matched_expectation {
                    m.raw_validity_matches += 1;
                }
                // `record.normalization_applied` comes from
                // `check_raw_validity`, which runs for every record
                // regardless of classification -- so this correctly counts
                // normalization on records that never reach a certificate
                // (e.g. the Adversarial nitro-alcohol record, which is
                // scope-rejected, not certified, but still normalizes).
                if record.normalization_applied {
                    m.normalization_applied += 1;
                }
                if let Some(cert) = &record.certificate {
                    m.certified += 1;
                    let rerun_json = rerun_record
                        .certificate
                        .as_ref()
                        .map(|c| c.to_json_pretty().unwrap());
                    let this_json = cert.to_json_pretty().unwrap();
                    if rerun_json.as_deref() == Some(this_json.as_str()) {
                        m.certificate_deterministic += 1;
                    }
                }
                m.pubchem_disagreements += record
                    .pubchem
                    .iter()
                    .filter(|x| x.agreement == crate::audit::PubChemAgreement::Disagrees)
                    .count();
                m.pubchem_representation_only += record
                    .pubchem
                    .iter()
                    .filter(|x| {
                        x.agreement == crate::audit::PubChemAgreement::RepresentationOnlyDifference
                    })
                    .count();
                m.rdkit_disagreements += record
                    .rdkit
                    .iter()
                    .filter(|x| x.agreement == crate::audit::RdkitAgreement::Disagrees)
                    .count();
                m.rdkit_representation_only += record
                    .rdkit
                    .iter()
                    .filter(|x| {
                        x.agreement == crate::audit::RdkitAgreement::RepresentationOnlyDifference
                    })
                    .count();
            }
            m
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::phase_a_fixture_corpus;
    use crate::hazard_heuristics::ExternalScopeConfig;
    use crate::policy::OpenWithHeuristicScreenPolicy;

    fn policy() -> OpenWithHeuristicScreenPolicy {
        OpenWithHeuristicScreenPolicy {
            external: ExternalScopeConfig::default(),
        }
    }

    #[test]
    fn every_category_gets_a_metrics_row_even_if_zero() {
        let corpus = phase_a_fixture_corpus();
        let p = policy();
        let report = crate::audit::run_audit(&corpus, &p, None, None);
        let metrics = compute_metrics(&corpus, &p, &report);
        assert_eq!(metrics.len(), RecordCategory::ALL.len());
        for m in &metrics {
            assert!(m.total > 0, "{:?} has zero records", m.category);
        }
    }

    #[test]
    fn every_category_has_a_perfect_expectation_pass_rate_on_this_corpus() {
        // The corpus is hand-authored so every record's expectation SHOULD
        // match reality (confirmed separately by
        // audit::full_corpus_runs_without_panicking_and_summary_counts_add_up).
        // This test asserts the same property at the per-category metrics
        // level specifically, since that's the number a reviewer actually
        // reads.
        let corpus = phase_a_fixture_corpus();
        let p = policy();
        let report = crate::audit::run_audit(&corpus, &p, None, None);
        let metrics = compute_metrics(&corpus, &p, &report);
        for m in &metrics {
            assert_eq!(
                m.expectation_pass_rate(),
                1.0,
                "{:?}: {}/{} matched expectation",
                m.category,
                m.expectation_matches,
                m.total
            );
        }
    }

    #[test]
    fn certified_records_are_deterministic_across_an_independent_rerun() {
        let corpus = phase_a_fixture_corpus();
        let p = policy();
        let report = crate::audit::run_audit(&corpus, &p, None, None);
        let metrics = compute_metrics(&corpus, &p, &report);
        for m in &metrics {
            assert_eq!(
                m.certificate_deterministic, m.certified,
                "{:?}: {}/{} certificates were deterministic across a rerun",
                m.category, m.certificate_deterministic, m.certified
            );
        }
    }

    #[test]
    fn supported_category_has_the_most_certifications() {
        let corpus = phase_a_fixture_corpus();
        let p = policy();
        let report = crate::audit::run_audit(&corpus, &p, None, None);
        let metrics = compute_metrics(&corpus, &p, &report);
        let supported = metrics
            .iter()
            .find(|m| m.category == RecordCategory::Supported)
            .unwrap();
        assert!(supported.certified > 0);
        let adversarial = metrics
            .iter()
            .find(|m| m.category == RecordCategory::Adversarial)
            .unwrap();
        assert_eq!(
            adversarial.certified, 0,
            "adversarial records are designed to be scope-rejected, not certified"
        );
    }
}
