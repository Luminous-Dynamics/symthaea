// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative public authority surface for evidence-corpus coverage.
//!
//! The raw coverage engine can deterministically establish that every retrieved
//! item was accounted for under exact source/query/index commitments. But its
//! include/exclude/duplicate decisions are caller-declared semantics. Therefore
//! a complete accounting pass can bind retrieval accounting, while claim-level
//! relevance coverage remains only declared until a later semantic decision
//! review gate authenticates those classifications.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::{
    AuthorityLevel, ClaimRelationBindingReport, EvidenceRecord, ResearchId, ScientificClaim,
    Sha256Digest,
    evidence_coverage::{
        CorpusItemDecision, EvidenceCoverageClosure, EvidenceCoverageFinding,
        FrozenEvidenceCoverageProtocol, SourceRetrievalReceipt, audit_evidence_coverage,
    },
};

const COVERAGE_AUTHORITY_REPORT_DOMAIN: &str =
    "symthaea.evidence-coverage-authority-report.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceCoverageAuthorityReport {
    report_sha256: Sha256Digest,
    raw_accounting_report_sha256: Sha256Digest,
    protocol_sha256: Sha256Digest,
    claim_binding_sha256: Sha256Digest,
    closure: EvidenceCoverageClosure,
    retrieval_accounting_level: AuthorityLevel,
    relevance_decision_level: AuthorityLevel,
    claim_coverage_level: AuthorityLevel,
    retrieval_accounting_established: bool,
    protocol_local_coverage_established: bool,
    global_exhaustiveness_established: bool,
    included_evidence_ids: BTreeSet<ResearchId>,
    relevant_evidence_omitted_from_claim: BTreeSet<ResearchId>,
    claim_evidence_not_covered: BTreeSet<ResearchId>,
    unresolved_items: BTreeSet<(ResearchId, Sha256Digest)>,
    findings: Vec<EvidenceCoverageFinding>,
    qualification_established: bool,
}

impl EvidenceCoverageAuthorityReport {
    pub fn report_sha256(&self) -> &Sha256Digest {
        &self.report_sha256
    }

    pub fn raw_accounting_report_sha256(&self) -> &Sha256Digest {
        &self.raw_accounting_report_sha256
    }

    pub fn protocol_sha256(&self) -> &Sha256Digest {
        &self.protocol_sha256
    }

    pub fn claim_binding_sha256(&self) -> &Sha256Digest {
        &self.claim_binding_sha256
    }

    pub fn closure(&self) -> EvidenceCoverageClosure {
        self.closure
    }

    pub fn retrieval_accounting_level(&self) -> AuthorityLevel {
        self.retrieval_accounting_level
    }

    pub fn relevance_decision_level(&self) -> AuthorityLevel {
        self.relevance_decision_level
    }

    pub fn claim_coverage_level(&self) -> AuthorityLevel {
        self.claim_coverage_level
    }

    pub fn retrieval_accounting_established(&self) -> bool {
        self.retrieval_accounting_established
    }

    pub fn protocol_local_coverage_established(&self) -> bool {
        self.protocol_local_coverage_established
    }

    pub fn global_exhaustiveness_established(&self) -> bool {
        self.global_exhaustiveness_established
    }

    pub fn included_evidence_ids(&self) -> &BTreeSet<ResearchId> {
        &self.included_evidence_ids
    }

    pub fn relevant_evidence_omitted_from_claim(&self) -> &BTreeSet<ResearchId> {
        &self.relevant_evidence_omitted_from_claim
    }

    pub fn claim_evidence_not_covered(&self) -> &BTreeSet<ResearchId> {
        &self.claim_evidence_not_covered
    }

    pub fn unresolved_items(&self) -> &BTreeSet<(ResearchId, Sha256Digest)> {
        &self.unresolved_items
    }

    pub fn findings(&self) -> &[EvidenceCoverageFinding] {
        &self.findings
    }

    pub fn qualification_established(&self) -> bool {
        self.qualification_established
    }
}

/// Public evidence-coverage audit.
///
/// `retrieval_accounting_level=Bound` means only that the exact frozen retrieval
/// surface was completely accounted for. Because corpus relevance decisions are
/// not yet semantically reviewed in SCI-011A, `claim_coverage_level` cannot exceed
/// `Declared` and `protocol_local_coverage_established` is always false here.
pub fn audit_claim_evidence_coverage(
    claim: &ScientificClaim,
    binding: &ClaimRelationBindingReport,
    frozen_protocol: &FrozenEvidenceCoverageProtocol,
    known_evidence: &[EvidenceRecord],
    receipts: &[SourceRetrievalReceipt],
    decisions: &[CorpusItemDecision],
) -> EvidenceCoverageAuthorityReport {
    let raw = audit_evidence_coverage(
        claim,
        binding,
        frozen_protocol,
        known_evidence,
        receipts,
        decisions,
    );

    let (retrieval_accounting_level, relevance_decision_level, claim_coverage_level) =
        authority_levels(raw.closure());
    let retrieval_accounting_established =
        raw.closure() == EvidenceCoverageClosure::CompleteWithinProtocol;

    let report_sha256 = authority_report_digest(
        raw.report_sha256(),
        frozen_protocol.protocol_sha256(),
        binding.binding_sha256(),
        raw.closure(),
        retrieval_accounting_level,
        relevance_decision_level,
        claim_coverage_level,
        retrieval_accounting_established,
    );

    EvidenceCoverageAuthorityReport {
        report_sha256,
        raw_accounting_report_sha256: raw.report_sha256().clone(),
        protocol_sha256: frozen_protocol.protocol_sha256().clone(),
        claim_binding_sha256: binding.binding_sha256().clone(),
        closure: raw.closure(),
        retrieval_accounting_level,
        relevance_decision_level,
        claim_coverage_level,
        retrieval_accounting_established,
        protocol_local_coverage_established: false,
        global_exhaustiveness_established: false,
        included_evidence_ids: raw.included_evidence_ids().clone(),
        relevant_evidence_omitted_from_claim: raw.relevant_evidence_omitted_from_claim().clone(),
        claim_evidence_not_covered: raw.claim_evidence_not_covered().clone(),
        unresolved_items: raw.unresolved_items().clone(),
        findings: raw.findings().to_vec(),
        qualification_established: false,
    }
}

fn authority_levels(
    closure: EvidenceCoverageClosure,
) -> (AuthorityLevel, AuthorityLevel, AuthorityLevel) {
    match closure {
        EvidenceCoverageClosure::CompleteWithinProtocol => (
            AuthorityLevel::Bound,
            AuthorityLevel::Declared,
            AuthorityLevel::Declared,
        ),
        EvidenceCoverageClosure::Incomplete => (
            AuthorityLevel::Declared,
            AuthorityLevel::Declared,
            AuthorityLevel::Declared,
        ),
        EvidenceCoverageClosure::Invalid => (
            AuthorityLevel::None,
            AuthorityLevel::None,
            AuthorityLevel::None,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn authority_report_digest(
    raw_report_sha256: &Sha256Digest,
    protocol_sha256: &Sha256Digest,
    claim_binding_sha256: &Sha256Digest,
    closure: EvidenceCoverageClosure,
    retrieval_level: AuthorityLevel,
    relevance_level: AuthorityLevel,
    coverage_level: AuthorityLevel,
    retrieval_accounting_established: bool,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(COVERAGE_AUTHORITY_REPORT_DOMAIN);
    digest.text(raw_report_sha256.as_str());
    digest.text(protocol_sha256.as_str());
    digest.text(claim_binding_sha256.as_str());
    digest.text(closure_tag(closure));
    digest.text(authority_level_tag(retrieval_level));
    digest.text(authority_level_tag(relevance_level));
    digest.text(authority_level_tag(coverage_level));
    digest.text(if retrieval_accounting_established {
        "retrieval-accounting-established"
    } else {
        "retrieval-accounting-not-established"
    });
    digest.text("protocol-local-coverage-not-established");
    digest.text("global-exhaustiveness-not-established");
    digest.text("qualification-not-established");
    digest.finish()
}

const fn closure_tag(closure: EvidenceCoverageClosure) -> &'static str {
    match closure {
        EvidenceCoverageClosure::CompleteWithinProtocol => "complete-within-protocol",
        EvidenceCoverageClosure::Incomplete => "incomplete",
        EvidenceCoverageClosure::Invalid => "invalid",
    }
}

const fn authority_level_tag(level: AuthorityLevel) -> &'static str {
    match level {
        AuthorityLevel::None => "none",
        AuthorityLevel::Declared => "declared",
        AuthorityLevel::Bound => "bound",
    }
}

struct FramedDigest {
    bytes: Vec<u8>,
}

impl FramedDigest {
    fn new(domain: &str) -> Self {
        let mut digest = Self { bytes: Vec::new() };
        digest.text(domain);
        digest
    }

    fn text(&mut self, value: &str) {
        self.bytes
            .extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn finish(self) -> Sha256Digest {
        Sha256Digest::of_bytes(&self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_accounting_does_not_establish_relevance_coverage() {
        let (retrieval, relevance, coverage) =
            authority_levels(EvidenceCoverageClosure::CompleteWithinProtocol);
        assert_eq!(retrieval, AuthorityLevel::Bound);
        assert_eq!(relevance, AuthorityLevel::Declared);
        assert_eq!(coverage, AuthorityLevel::Declared);
    }

    #[test]
    fn invalid_accounting_carries_no_positive_coverage_authority() {
        let (retrieval, relevance, coverage) = authority_levels(EvidenceCoverageClosure::Invalid);
        assert_eq!(retrieval, AuthorityLevel::None);
        assert_eq!(relevance, AuthorityLevel::None);
        assert_eq!(coverage, AuthorityLevel::None);
    }
}
