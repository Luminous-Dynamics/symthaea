// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Protocol-scoped evidence-corpus coverage.
//!
//! Strong evidence-to-claim semantics are still vulnerable to selective evidence
//! assembly. This layer audits a narrower, reproducible proposition: under an
//! exact frozen source/query/retrieval protocol and exact index snapshots, every
//! retrieved item was accounted for and every evidence record used by the claim
//! was represented in the audited corpus.
//!
//! `CompleteWithinProtocol` is deliberately **not** global exhaustiveness. New
//! sources, later publications, inaccessible records, indexing failures, or a
//! poorly chosen search protocol can still matter.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    AuthorityLevel, ClaimRelationBindingReport, EvidenceRecord, ResearchId, ScientificClaim,
    Sha256Digest,
};

pub const EVIDENCE_COVERAGE_SCHEMA: &str = "symthaea.evidence-coverage.v1";
const COVERAGE_PROTOCOL_DIGEST_DOMAIN: &str = "symthaea.evidence-coverage-policy.identity.v1";
const RETRIEVAL_SET_DIGEST_DOMAIN: &str = "symthaea.evidence-coverage-retrieval-set.identity.v1";
const DECISION_SET_DIGEST_DOMAIN: &str = "symthaea.evidence-coverage-decision-set.identity.v1";
const COVERAGE_REPORT_DIGEST_DOMAIN: &str = "symthaea.evidence-coverage-report.identity.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvidenceCoverageIntent {
    SystematicClaimAudit,
    SupplementalSearch,
    ExploratorySearch,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvidenceSourceClass {
    LiteratureIndex,
    PreprintIndex,
    TrialRegistry,
    DatasetRegistry,
    FormalLibrary,
    InternalEvidenceGraph,
    ExperimentalArchive,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceSourceSpec {
    pub source_id: ResearchId,
    pub source_class: EvidenceSourceClass,
    pub source_scope_sha256: Sha256Digest,
    pub query_sha256: Sha256Digest,
    pub retrieval_protocol_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCoverageProtocol {
    pub schema_version: String,
    pub coverage_id: ResearchId,
    pub claim_binding_sha256: Sha256Digest,
    pub subject_sha256: Sha256Digest,
    pub intent: EvidenceCoverageIntent,
    /// Declared timing only. A timestamp in this object does not authenticate
    /// prospective registration; provenance/qualification must do that later.
    pub declared_frozen_at_unix_ms: u64,
    pub declared_search_cutoff_unix_ms: u64,
    pub decision_policy_sha256: Sha256Digest,
    pub deduplication_policy_sha256: Sha256Digest,
    pub sources: Vec<EvidenceSourceSpec>,
    /// Lineage only; authority never transfers across supersession.
    pub supersedes_protocol_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceCoverageProtocolIssue {
    WrongSchemaVersion { found: String },
    MissingSources,
    DuplicateSourceId { source_id: ResearchId },
}

impl EvidenceCoverageProtocol {
    pub fn validate(&self) -> Vec<EvidenceCoverageProtocolIssue> {
        let mut issues = Vec::new();
        if self.schema_version != EVIDENCE_COVERAGE_SCHEMA {
            issues.push(EvidenceCoverageProtocolIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if self.sources.is_empty() {
            issues.push(EvidenceCoverageProtocolIssue::MissingSources);
        }
        let mut source_ids = BTreeSet::new();
        for source in &self.sources {
            if !source_ids.insert(source.source_id.clone()) {
                issues.push(EvidenceCoverageProtocolIssue::DuplicateSourceId {
                    source_id: source.source_id.clone(),
                });
            }
        }
        issues
    }

    pub fn freeze(
        self,
    ) -> Result<FrozenEvidenceCoverageProtocol, Vec<EvidenceCoverageProtocolIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        let protocol_sha256 = protocol_digest(&self);
        Ok(FrozenEvidenceCoverageProtocol {
            protocol: self,
            protocol_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenEvidenceCoverageProtocol {
    protocol: EvidenceCoverageProtocol,
    protocol_sha256: Sha256Digest,
}

impl FrozenEvidenceCoverageProtocol {
    pub fn protocol(&self) -> &EvidenceCoverageProtocol {
        &self.protocol
    }

    pub fn protocol_sha256(&self) -> &Sha256Digest {
        &self.protocol_sha256
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum RetrievalOutcome {
    Complete,
    Partial,
    Failed,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRetrievalReceipt {
    pub source_id: ResearchId,
    pub source_scope_sha256: Sha256Digest,
    pub query_sha256: Sha256Digest,
    pub retrieval_protocol_sha256: Sha256Digest,
    pub index_snapshot_sha256: Sha256Digest,
    pub retrieval_artifact_sha256: Sha256Digest,
    pub outcome: RetrievalOutcome,
    pub retrieved_item_sha256s: BTreeSet<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorpusDecisionKind {
    Include {
        evidence_id: ResearchId,
        evidence_artifact_sha256: Sha256Digest,
    },
    Exclude,
    Duplicate {
        canonical_item_sha256: Sha256Digest,
    },
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusItemDecision {
    pub source_id: ResearchId,
    pub item_sha256: Sha256Digest,
    pub decision_policy_sha256: Sha256Digest,
    pub rationale_sha256: Sha256Digest,
    pub decision: CorpusDecisionKind,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum EvidenceCoverageFinding {
    ClaimBindingMismatch,
    ClaimSubjectMismatch,
    DuplicateKnownEvidence { evidence_id: ResearchId },
    InvalidKnownEvidence { evidence_id: ResearchId },
    KnownEvidenceSubjectMismatch { evidence_id: ResearchId },
    ClaimEvidenceMissingRecord { evidence_id: ResearchId },
    DuplicateSourceReceipt { source_id: ResearchId },
    UnknownSourceReceipt { source_id: ResearchId },
    SourceScopeSubstitution { source_id: ResearchId },
    SourceQuerySubstitution { source_id: ResearchId },
    RetrievalProtocolSubstitution { source_id: ResearchId },
    MissingSourceReceipt { source_id: ResearchId },
    PartialRetrieval { source_id: ResearchId },
    FailedRetrieval { source_id: ResearchId },
    InvalidRetrieval { source_id: ResearchId },
    DuplicateDecision {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
    },
    UnknownDecisionSource { source_id: ResearchId },
    DecisionForUnretrievedItem {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
    },
    MissingItemDecision {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
    },
    DecisionPolicySubstitution {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
    },
    IncludedUnknownEvidence { evidence_id: ResearchId },
    IncludedEvidenceArtifactMismatch { evidence_id: ResearchId },
    DuplicateIncludedEvidence { evidence_id: ResearchId },
    DuplicateTargetIsSelf { item_sha256: Sha256Digest },
    DuplicateTargetUnknown { item_sha256: Sha256Digest },
    UnresolvedCorpusItem {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
    },
    RelevantEvidenceOmittedFromClaim { evidence_id: ResearchId },
    ClaimEvidenceNotCovered { evidence_id: ResearchId },
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvidenceCoverageClosure {
    CompleteWithinProtocol,
    Incomplete,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceCoverageReport {
    report_sha256: Sha256Digest,
    protocol_sha256: Sha256Digest,
    claim_binding_sha256: Sha256Digest,
    retrieval_set_sha256: Sha256Digest,
    decision_set_sha256: Sha256Digest,
    closure: EvidenceCoverageClosure,
    coverage_level: AuthorityLevel,
    protocol_local_coverage_established: bool,
    global_exhaustiveness_established: bool,
    included_evidence_ids: BTreeSet<ResearchId>,
    relevant_evidence_omitted_from_claim: BTreeSet<ResearchId>,
    claim_evidence_not_covered: BTreeSet<ResearchId>,
    unresolved_items: BTreeSet<(ResearchId, Sha256Digest)>,
    findings: Vec<EvidenceCoverageFinding>,
    retrieval_receipts: Vec<SourceRetrievalReceipt>,
    decisions: Vec<CorpusItemDecision>,
}

impl EvidenceCoverageReport {
    pub fn report_sha256(&self) -> &Sha256Digest {
        &self.report_sha256
    }

    pub fn protocol_sha256(&self) -> &Sha256Digest {
        &self.protocol_sha256
    }

    pub fn closure(&self) -> EvidenceCoverageClosure {
        self.closure
    }

    pub fn coverage_level(&self) -> AuthorityLevel {
        self.coverage_level
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
}

pub fn audit_evidence_coverage(
    claim: &ScientificClaim,
    binding: &ClaimRelationBindingReport,
    frozen_protocol: &FrozenEvidenceCoverageProtocol,
    known_evidence: &[EvidenceRecord],
    receipts: &[SourceRetrievalReceipt],
    decisions: &[CorpusItemDecision],
) -> EvidenceCoverageReport {
    let protocol = frozen_protocol.protocol();
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;

    if &protocol.claim_binding_sha256 != binding.binding_sha256() {
        findings.push(EvidenceCoverageFinding::ClaimBindingMismatch);
        invalid = true;
    }
    if &protocol.subject_sha256 != claim.subject_sha256() {
        findings.push(EvidenceCoverageFinding::ClaimSubjectMismatch);
        invalid = true;
    }

    let mut evidence_by_id = BTreeMap::new();
    for record in known_evidence {
        if evidence_by_id
            .insert(record.evidence_id().clone(), record)
            .is_some()
        {
            findings.push(EvidenceCoverageFinding::DuplicateKnownEvidence {
                evidence_id: record.evidence_id().clone(),
            });
            invalid = true;
        }
        if !record.validate().is_empty() {
            findings.push(EvidenceCoverageFinding::InvalidKnownEvidence {
                evidence_id: record.evidence_id().clone(),
            });
            invalid = true;
        }
        if record.subject_sha256() != claim.subject_sha256() {
            findings.push(EvidenceCoverageFinding::KnownEvidenceSubjectMismatch {
                evidence_id: record.evidence_id().clone(),
            });
            invalid = true;
        }
    }
    for evidence_id in claim.evidence_ids() {
        if !evidence_by_id.contains_key(evidence_id) {
            findings.push(EvidenceCoverageFinding::ClaimEvidenceMissingRecord {
                evidence_id: evidence_id.clone(),
            });
            invalid = true;
        }
    }

    let source_specs: BTreeMap<_, _> = protocol
        .sources
        .iter()
        .map(|source| (source.source_id.clone(), source))
        .collect();
    let mut receipt_by_source = BTreeMap::new();
    let mut retrieved_pairs = BTreeSet::new();
    let mut all_retrieved_item_digests = BTreeSet::new();

    for receipt in receipts {
        if receipt_by_source
            .insert(receipt.source_id.clone(), receipt)
            .is_some()
        {
            findings.push(EvidenceCoverageFinding::DuplicateSourceReceipt {
                source_id: receipt.source_id.clone(),
            });
            invalid = true;
            continue;
        }
        let Some(spec) = source_specs.get(&receipt.source_id) else {
            findings.push(EvidenceCoverageFinding::UnknownSourceReceipt {
                source_id: receipt.source_id.clone(),
            });
            invalid = true;
            continue;
        };
        if receipt.source_scope_sha256 != spec.source_scope_sha256 {
            findings.push(EvidenceCoverageFinding::SourceScopeSubstitution {
                source_id: receipt.source_id.clone(),
            });
            invalid = true;
        }
        if receipt.query_sha256 != spec.query_sha256 {
            findings.push(EvidenceCoverageFinding::SourceQuerySubstitution {
                source_id: receipt.source_id.clone(),
            });
            invalid = true;
        }
        if receipt.retrieval_protocol_sha256 != spec.retrieval_protocol_sha256 {
            findings.push(EvidenceCoverageFinding::RetrievalProtocolSubstitution {
                source_id: receipt.source_id.clone(),
            });
            invalid = true;
        }
        match receipt.outcome {
            RetrievalOutcome::Complete => {}
            RetrievalOutcome::Partial => {
                findings.push(EvidenceCoverageFinding::PartialRetrieval {
                    source_id: receipt.source_id.clone(),
                });
                incomplete = true;
            }
            RetrievalOutcome::Failed => {
                findings.push(EvidenceCoverageFinding::FailedRetrieval {
                    source_id: receipt.source_id.clone(),
                });
                incomplete = true;
            }
            RetrievalOutcome::Invalid => {
                findings.push(EvidenceCoverageFinding::InvalidRetrieval {
                    source_id: receipt.source_id.clone(),
                });
                invalid = true;
            }
        }
        for item_sha256 in &receipt.retrieved_item_sha256s {
            retrieved_pairs.insert((receipt.source_id.clone(), item_sha256.clone()));
            all_retrieved_item_digests.insert(item_sha256.clone());
        }
    }

    for source in &protocol.sources {
        if !receipt_by_source.contains_key(&source.source_id) {
            findings.push(EvidenceCoverageFinding::MissingSourceReceipt {
                source_id: source.source_id.clone(),
            });
            incomplete = true;
        }
    }

    let mut decision_by_item = BTreeMap::new();
    let mut included_evidence_ids = BTreeSet::new();
    let mut unresolved_items = BTreeSet::new();

    for decision in decisions {
        let key = (decision.source_id.clone(), decision.item_sha256.clone());
        if decision_by_item.insert(key.clone(), decision).is_some() {
            findings.push(EvidenceCoverageFinding::DuplicateDecision {
                source_id: key.0,
                item_sha256: key.1,
            });
            invalid = true;
            continue;
        }
        if !source_specs.contains_key(&decision.source_id) {
            findings.push(EvidenceCoverageFinding::UnknownDecisionSource {
                source_id: decision.source_id.clone(),
            });
            invalid = true;
            continue;
        }
        if !retrieved_pairs.contains(&key) {
            findings.push(EvidenceCoverageFinding::DecisionForUnretrievedItem {
                source_id: decision.source_id.clone(),
                item_sha256: decision.item_sha256.clone(),
            });
            invalid = true;
            continue;
        }
        if decision.decision_policy_sha256 != protocol.decision_policy_sha256 {
            findings.push(EvidenceCoverageFinding::DecisionPolicySubstitution {
                source_id: decision.source_id.clone(),
                item_sha256: decision.item_sha256.clone(),
            });
            invalid = true;
            continue;
        }

        match &decision.decision {
            CorpusDecisionKind::Include {
                evidence_id,
                evidence_artifact_sha256,
            } => {
                let Some(record) = evidence_by_id.get(evidence_id) else {
                    findings.push(EvidenceCoverageFinding::IncludedUnknownEvidence {
                        evidence_id: evidence_id.clone(),
                    });
                    invalid = true;
                    continue;
                };
                if record.artifact_sha256() != evidence_artifact_sha256 {
                    findings.push(EvidenceCoverageFinding::IncludedEvidenceArtifactMismatch {
                        evidence_id: evidence_id.clone(),
                    });
                    invalid = true;
                    continue;
                }
                if !included_evidence_ids.insert(evidence_id.clone()) {
                    findings.push(EvidenceCoverageFinding::DuplicateIncludedEvidence {
                        evidence_id: evidence_id.clone(),
                    });
                    invalid = true;
                }
            }
            CorpusDecisionKind::Exclude => {}
            CorpusDecisionKind::Duplicate {
                canonical_item_sha256,
            } => {
                if canonical_item_sha256 == &decision.item_sha256 {
                    findings.push(EvidenceCoverageFinding::DuplicateTargetIsSelf {
                        item_sha256: decision.item_sha256.clone(),
                    });
                    invalid = true;
                } else if !all_retrieved_item_digests.contains(canonical_item_sha256) {
                    findings.push(EvidenceCoverageFinding::DuplicateTargetUnknown {
                        item_sha256: canonical_item_sha256.clone(),
                    });
                    invalid = true;
                }
            }
            CorpusDecisionKind::Unresolved => {
                unresolved_items.insert(key.clone());
                findings.push(EvidenceCoverageFinding::UnresolvedCorpusItem {
                    source_id: key.0,
                    item_sha256: key.1,
                });
                incomplete = true;
            }
        }
    }

    for (source_id, item_sha256) in &retrieved_pairs {
        if !decision_by_item.contains_key(&(source_id.clone(), item_sha256.clone())) {
            findings.push(EvidenceCoverageFinding::MissingItemDecision {
                source_id: source_id.clone(),
                item_sha256: item_sha256.clone(),
            });
            incomplete = true;
        }
    }

    let mut relevant_evidence_omitted_from_claim = BTreeSet::new();
    for evidence_id in included_evidence_ids.difference(claim.evidence_ids()) {
        relevant_evidence_omitted_from_claim.insert(evidence_id.clone());
        findings.push(EvidenceCoverageFinding::RelevantEvidenceOmittedFromClaim {
            evidence_id: evidence_id.clone(),
        });
        incomplete = true;
    }

    let mut claim_evidence_not_covered = BTreeSet::new();
    for evidence_id in claim.evidence_ids().difference(&included_evidence_ids) {
        claim_evidence_not_covered.insert(evidence_id.clone());
        findings.push(EvidenceCoverageFinding::ClaimEvidenceNotCovered {
            evidence_id: evidence_id.clone(),
        });
        incomplete = true;
    }

    let closure = if invalid {
        EvidenceCoverageClosure::Invalid
    } else if incomplete {
        EvidenceCoverageClosure::Incomplete
    } else {
        EvidenceCoverageClosure::CompleteWithinProtocol
    };

    let systematic_complete = closure == EvidenceCoverageClosure::CompleteWithinProtocol
        && protocol.intent == EvidenceCoverageIntent::SystematicClaimAudit;
    let coverage_level = if invalid {
        AuthorityLevel::None
    } else if systematic_complete {
        AuthorityLevel::Bound
    } else {
        AuthorityLevel::Declared
    };

    let mut retained_receipts = receipts.to_vec();
    retained_receipts.sort_by(|left, right| left.source_id.cmp(&right.source_id));
    let mut retained_decisions = decisions.to_vec();
    retained_decisions.sort_by(|left, right| {
        (&left.source_id, &left.item_sha256).cmp(&(&right.source_id, &right.item_sha256))
    });
    findings.sort();

    let retrieval_set_sha256 = retrieval_set_digest(&retained_receipts);
    let decision_set_sha256 = decision_set_digest(&retained_decisions);
    let report_sha256 = coverage_report_digest(
        frozen_protocol.protocol_sha256(),
        binding.binding_sha256(),
        &retrieval_set_sha256,
        &decision_set_sha256,
        closure,
        coverage_level,
        systematic_complete,
        &included_evidence_ids,
        &relevant_evidence_omitted_from_claim,
        &claim_evidence_not_covered,
        &unresolved_items,
        &findings,
    );

    EvidenceCoverageReport {
        report_sha256,
        protocol_sha256: frozen_protocol.protocol_sha256().clone(),
        claim_binding_sha256: binding.binding_sha256().clone(),
        retrieval_set_sha256,
        decision_set_sha256,
        closure,
        coverage_level,
        protocol_local_coverage_established: systematic_complete,
        global_exhaustiveness_established: false,
        included_evidence_ids,
        relevant_evidence_omitted_from_claim,
        claim_evidence_not_covered,
        unresolved_items,
        findings,
        retrieval_receipts: retained_receipts,
        decisions: retained_decisions,
    }
}

fn protocol_digest(protocol: &EvidenceCoverageProtocol) -> Sha256Digest {
    let mut digest = FramedDigest::new(COVERAGE_PROTOCOL_DIGEST_DOMAIN);
    digest.text(EVIDENCE_COVERAGE_SCHEMA);
    digest.text(protocol.coverage_id.as_str());
    digest.text(protocol.claim_binding_sha256.as_str());
    digest.text(protocol.subject_sha256.as_str());
    digest.text(coverage_intent_tag(protocol.intent));
    digest.text(&protocol.declared_frozen_at_unix_ms.to_string());
    digest.text(&protocol.declared_search_cutoff_unix_ms.to_string());
    digest.text(protocol.decision_policy_sha256.as_str());
    digest.text(protocol.deduplication_policy_sha256.as_str());
    let mut sources = protocol.sources.clone();
    sources.sort_by(|left, right| left.source_id.cmp(&right.source_id));
    for source in sources {
        digest.text("source");
        digest.text(source.source_id.as_str());
        digest.text(source_class_tag(source.source_class));
        digest.text(source.source_scope_sha256.as_str());
        digest.text(source.query_sha256.as_str());
        digest.text(source.retrieval_protocol_sha256.as_str());
    }
    digest.optional_sha(protocol.supersedes_protocol_sha256.as_ref());
    digest.finish()
}

fn retrieval_set_digest(receipts: &[SourceRetrievalReceipt]) -> Sha256Digest {
    let mut digest = FramedDigest::new(RETRIEVAL_SET_DIGEST_DOMAIN);
    for receipt in receipts {
        digest.text("receipt");
        digest.text(receipt.source_id.as_str());
        digest.text(receipt.source_scope_sha256.as_str());
        digest.text(receipt.query_sha256.as_str());
        digest.text(receipt.retrieval_protocol_sha256.as_str());
        digest.text(receipt.index_snapshot_sha256.as_str());
        digest.text(receipt.retrieval_artifact_sha256.as_str());
        digest.text(retrieval_outcome_tag(receipt.outcome));
        for item in &receipt.retrieved_item_sha256s {
            digest.text("item");
            digest.text(item.as_str());
        }
    }
    digest.finish()
}

fn decision_set_digest(decisions: &[CorpusItemDecision]) -> Sha256Digest {
    let mut digest = FramedDigest::new(DECISION_SET_DIGEST_DOMAIN);
    for decision in decisions {
        digest.text("decision");
        digest.text(decision.source_id.as_str());
        digest.text(decision.item_sha256.as_str());
        digest.text(decision.decision_policy_sha256.as_str());
        digest.text(decision.rationale_sha256.as_str());
        match &decision.decision {
            CorpusDecisionKind::Include {
                evidence_id,
                evidence_artifact_sha256,
            } => {
                digest.text("include");
                digest.text(evidence_id.as_str());
                digest.text(evidence_artifact_sha256.as_str());
            }
            CorpusDecisionKind::Exclude => digest.text("exclude"),
            CorpusDecisionKind::Duplicate {
                canonical_item_sha256,
            } => {
                digest.text("duplicate");
                digest.text(canonical_item_sha256.as_str());
            }
            CorpusDecisionKind::Unresolved => digest.text("unresolved"),
        }
    }
    digest.finish()
}

#[allow(clippy::too_many_arguments)]
fn coverage_report_digest(
    protocol_sha256: &Sha256Digest,
    claim_binding_sha256: &Sha256Digest,
    retrieval_set_sha256: &Sha256Digest,
    decision_set_sha256: &Sha256Digest,
    closure: EvidenceCoverageClosure,
    coverage_level: AuthorityLevel,
    local_established: bool,
    included: &BTreeSet<ResearchId>,
    omitted: &BTreeSet<ResearchId>,
    not_covered: &BTreeSet<ResearchId>,
    unresolved: &BTreeSet<(ResearchId, Sha256Digest)>,
    findings: &[EvidenceCoverageFinding],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(COVERAGE_REPORT_DIGEST_DOMAIN);
    digest.text(protocol_sha256.as_str());
    digest.text(claim_binding_sha256.as_str());
    digest.text(retrieval_set_sha256.as_str());
    digest.text(decision_set_sha256.as_str());
    digest.text(coverage_closure_tag(closure));
    digest.text(authority_level_tag(coverage_level));
    digest.text(if local_established { "local-established" } else { "local-not-established" });
    digest.text("global-exhaustiveness-not-established");
    digest_ids(&mut digest, "included", included);
    digest_ids(&mut digest, "omitted", omitted);
    digest_ids(&mut digest, "not-covered", not_covered);
    for (source_id, item_sha256) in unresolved {
        digest.text("unresolved");
        digest.text(source_id.as_str());
        digest.text(item_sha256.as_str());
    }
    for finding in findings {
        digest.text("finding");
        digest.text(&finding_tag(finding));
    }
    digest.finish()
}

fn digest_ids(digest: &mut FramedDigest, tag: &str, ids: &BTreeSet<ResearchId>) {
    for id in ids {
        digest.text(tag);
        digest.text(id.as_str());
    }
}

fn finding_tag(finding: &EvidenceCoverageFinding) -> String {
    match finding {
        EvidenceCoverageFinding::ClaimBindingMismatch => "claim-binding-mismatch".into(),
        EvidenceCoverageFinding::ClaimSubjectMismatch => "claim-subject-mismatch".into(),
        EvidenceCoverageFinding::DuplicateKnownEvidence { evidence_id } => format!("duplicate-known-evidence:{}", evidence_id.as_str()),
        EvidenceCoverageFinding::InvalidKnownEvidence { evidence_id } => format!("invalid-known-evidence:{}", evidence_id.as_str()),
        EvidenceCoverageFinding::KnownEvidenceSubjectMismatch { evidence_id } => format!("known-evidence-subject-mismatch:{}", evidence_id.as_str()),
        EvidenceCoverageFinding::ClaimEvidenceMissingRecord { evidence_id } => format!("claim-evidence-missing-record:{}", evidence_id.as_str()),
        EvidenceCoverageFinding::DuplicateSourceReceipt { source_id } => format!("duplicate-source-receipt:{}", source_id.as_str()),
        EvidenceCoverageFinding::UnknownSourceReceipt { source_id } => format!("unknown-source-receipt:{}", source_id.as_str()),
        EvidenceCoverageFinding::SourceScopeSubstitution { source_id } => format!("source-scope-substitution:{}", source_id.as_str()),
        EvidenceCoverageFinding::SourceQuerySubstitution { source_id } => format!("source-query-substitution:{}", source_id.as_str()),
        EvidenceCoverageFinding::RetrievalProtocolSubstitution { source_id } => format!("retrieval-protocol-substitution:{}", source_id.as_str()),
        EvidenceCoverageFinding::MissingSourceReceipt { source_id } => format!("missing-source-receipt:{}", source_id.as_str()),
        EvidenceCoverageFinding::PartialRetrieval { source_id } => format!("partial-retrieval:{}", source_id.as_str()),
        EvidenceCoverageFinding::FailedRetrieval { source_id } => format!("failed-retrieval:{}", source_id.as_str()),
        EvidenceCoverageFinding::InvalidRetrieval { source_id } => format!("invalid-retrieval:{}", source_id.as_str()),
        EvidenceCoverageFinding::DuplicateDecision { source_id, item_sha256 } => format!("duplicate-decision:{}:{}", source_id.as_str(), item_sha256.as_str()),
        EvidenceCoverageFinding::UnknownDecisionSource { source_id } => format!("unknown-decision-source:{}", source_id.as_str()),
        EvidenceCoverageFinding::DecisionForUnretrievedItem { source_id, item_sha256 } => format!("decision-for-unretrieved-item:{}:{}", source_id.as_str(), item_sha256.as_str()),
        EvidenceCoverageFinding::MissingItemDecision { source_id, item_sha256 } => format!("missing-item-decision:{}:{}", source_id.as_str(), item_sha256.as_str()),
        EvidenceCoverageFinding::DecisionPolicySubstitution { source_id, item_sha256 } => format!("decision-policy-substitution:{}:{}", source_id.as_str(), item_sha256.as_str()),
        EvidenceCoverageFinding::IncludedUnknownEvidence { evidence_id } => format!("included-unknown-evidence:{}", evidence_id.as_str()),
        EvidenceCoverageFinding::IncludedEvidenceArtifactMismatch { evidence_id } => format!("included-evidence-artifact-mismatch:{}", evidence_id.as_str()),
        EvidenceCoverageFinding::DuplicateIncludedEvidence { evidence_id } => format!("duplicate-included-evidence:{}", evidence_id.as_str()),
        EvidenceCoverageFinding::DuplicateTargetIsSelf { item_sha256 } => format!("duplicate-target-is-self:{}", item_sha256.as_str()),
        EvidenceCoverageFinding::DuplicateTargetUnknown { item_sha256 } => format!("duplicate-target-unknown:{}", item_sha256.as_str()),
        EvidenceCoverageFinding::UnresolvedCorpusItem { source_id, item_sha256 } => format!("unresolved-corpus-item:{}:{}", source_id.as_str(), item_sha256.as_str()),
        EvidenceCoverageFinding::RelevantEvidenceOmittedFromClaim { evidence_id } => format!("relevant-evidence-omitted-from-claim:{}", evidence_id.as_str()),
        EvidenceCoverageFinding::ClaimEvidenceNotCovered { evidence_id } => format!("claim-evidence-not-covered:{}", evidence_id.as_str()),
    }
}

const fn coverage_intent_tag(intent: EvidenceCoverageIntent) -> &'static str {
    match intent {
        EvidenceCoverageIntent::SystematicClaimAudit => "systematic-claim-audit",
        EvidenceCoverageIntent::SupplementalSearch => "supplemental-search",
        EvidenceCoverageIntent::ExploratorySearch => "exploratory-search",
    }
}

const fn source_class_tag(class: EvidenceSourceClass) -> &'static str {
    match class {
        EvidenceSourceClass::LiteratureIndex => "literature-index",
        EvidenceSourceClass::PreprintIndex => "preprint-index",
        EvidenceSourceClass::TrialRegistry => "trial-registry",
        EvidenceSourceClass::DatasetRegistry => "dataset-registry",
        EvidenceSourceClass::FormalLibrary => "formal-library",
        EvidenceSourceClass::InternalEvidenceGraph => "internal-evidence-graph",
        EvidenceSourceClass::ExperimentalArchive => "experimental-archive",
        EvidenceSourceClass::Other => "other",
    }
}

const fn retrieval_outcome_tag(outcome: RetrievalOutcome) -> &'static str {
    match outcome {
        RetrievalOutcome::Complete => "complete",
        RetrievalOutcome::Partial => "partial",
        RetrievalOutcome::Failed => "failed",
        RetrievalOutcome::Invalid => "invalid",
    }
}

const fn coverage_closure_tag(closure: EvidenceCoverageClosure) -> &'static str {
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
        self.bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn optional_sha(&mut self, value: Option<&Sha256Digest>) {
        match value {
            Some(value) => {
                self.text("some");
                self.text(value.as_str());
            }
            None => self.text("none"),
        }
    }

    fn finish(self) -> Sha256Digest {
        Sha256Digest::of_bytes(&self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AdjudicatedScientificClaim, AuthorityFacet, AuthorityProfile, ClaimEvidenceLink,
        ClaimEvidenceRelation, ClaimRelationBindingPolicy, EvidenceKind, EvidenceState,
        RelationReviewRequirement, SemanticReviewDimension, SemanticReviewOutcome,
        RelationSemanticReview, CLAIM_RELATION_BINDING_SCHEMA, adjudicate_scientific_claim,
        bind_claim_relations,
    };

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn evidence(subject: &Sha256Digest, evidence_id: &str) -> EvidenceRecord {
        EvidenceRecord::new(
            id(evidence_id),
            subject.clone(),
            EvidenceKind::LiteratureEvidence,
            EvidenceState::Pass,
            sha(&format!("artifact-{evidence_id}")),
            [],
            None,
            AuthorityProfile::empty().with(AuthorityFacet::Provenance, AuthorityLevel::Bound),
        )
        .unwrap()
    }

    fn bound_claim(
        evidence: &[EvidenceRecord],
    ) -> (ScientificClaim, AdjudicatedScientificClaim, ClaimRelationBindingReport) {
        let subject = evidence[0].subject_sha256().clone();
        let claim = ScientificClaim::from_evidence(id("CLAIM-1"), subject, evidence).unwrap();
        let links: Vec<_> = evidence
            .iter()
            .map(|record| ClaimEvidenceLink {
                evidence_id: record.evidence_id().clone(),
                evidence_artifact_sha256: record.artifact_sha256().clone(),
                relation: ClaimEvidenceRelation::Supports,
                relation_rationale_sha256: sha(&format!("rationale-{}", record.evidence_id().as_str())),
            })
            .collect();
        let adjudicated = adjudicate_scientific_claim(&claim, evidence, &links).unwrap();
        let policy = ClaimRelationBindingPolicy {
            schema_version: CLAIM_RELATION_BINDING_SCHEMA.into(),
            policy_id: id("BIND-POLICY"),
            adjudication_sha256: adjudicated.adjudication_sha256().clone(),
            requirements: vec![RelationReviewRequirement {
                requirement_id: id("REVIEW-1"),
                relation: ClaimEvidenceRelation::Supports,
                dimension: SemanticReviewDimension::EvidenceApplicability,
                reviewer_identity_sha256: sha("reviewer"),
                review_protocol_sha256: sha("review-protocol"),
            }],
            supersedes_policy_sha256: None,
        }
        .freeze()
        .unwrap();
        let reviews: Vec<_> = links
            .iter()
            .map(|link| RelationSemanticReview {
                requirement_id: id("REVIEW-1"),
                evidence_id: link.evidence_id.clone(),
                evidence_artifact_sha256: link.evidence_artifact_sha256.clone(),
                relation: link.relation,
                relation_rationale_sha256: link.relation_rationale_sha256.clone(),
                dimension: SemanticReviewDimension::EvidenceApplicability,
                reviewer_identity_sha256: sha("reviewer"),
                review_protocol_sha256: sha("review-protocol"),
                review_artifact_sha256: sha(&format!("review-{}", link.evidence_id.as_str())),
                outcome: SemanticReviewOutcome::Accepted,
            })
            .collect();
        let binding = bind_claim_relations(
            &claim,
            evidence,
            &links,
            &adjudicated,
            &policy,
            &reviews,
        )
        .unwrap();
        (claim, adjudicated, binding)
    }

    fn protocol(binding: &ClaimRelationBindingReport, subject: &Sha256Digest) -> FrozenEvidenceCoverageProtocol {
        EvidenceCoverageProtocol {
            schema_version: EVIDENCE_COVERAGE_SCHEMA.into(),
            coverage_id: id("COVERAGE-1"),
            claim_binding_sha256: binding.binding_sha256().clone(),
            subject_sha256: subject.clone(),
            intent: EvidenceCoverageIntent::SystematicClaimAudit,
            declared_frozen_at_unix_ms: 1,
            declared_search_cutoff_unix_ms: 2,
            decision_policy_sha256: sha("decision-policy"),
            deduplication_policy_sha256: sha("dedup-policy"),
            sources: vec![EvidenceSourceSpec {
                source_id: id("SOURCE-1"),
                source_class: EvidenceSourceClass::LiteratureIndex,
                source_scope_sha256: sha("scope"),
                query_sha256: sha("query"),
                retrieval_protocol_sha256: sha("retrieval-protocol"),
            }],
            supersedes_protocol_sha256: None,
        }
        .freeze()
        .unwrap()
    }

    fn receipt(items: impl IntoIterator<Item = Sha256Digest>) -> SourceRetrievalReceipt {
        SourceRetrievalReceipt {
            source_id: id("SOURCE-1"),
            source_scope_sha256: sha("scope"),
            query_sha256: sha("query"),
            retrieval_protocol_sha256: sha("retrieval-protocol"),
            index_snapshot_sha256: sha("snapshot"),
            retrieval_artifact_sha256: sha("retrieval-artifact"),
            outcome: RetrievalOutcome::Complete,
            retrieved_item_sha256s: items.into_iter().collect(),
        }
    }

    fn include(item: &str, record: &EvidenceRecord) -> CorpusItemDecision {
        CorpusItemDecision {
            source_id: id("SOURCE-1"),
            item_sha256: sha(item),
            decision_policy_sha256: sha("decision-policy"),
            rationale_sha256: sha(&format!("decision-{item}")),
            decision: CorpusDecisionKind::Include {
                evidence_id: record.evidence_id().clone(),
                evidence_artifact_sha256: record.artifact_sha256().clone(),
            },
        }
    }

    #[test]
    fn complete_systematic_protocol_establishes_only_local_coverage() {
        let subject = sha("subject");
        let evidence = vec![evidence(&subject, "E-1")];
        let (claim, _adjudicated, binding) = bound_claim(&evidence);
        let report = audit_evidence_coverage(
            &claim,
            &binding,
            &protocol(&binding, &subject),
            &evidence,
            &[receipt([sha("item-1")])],
            &[include("item-1", &evidence[0])],
        );
        assert_eq!(report.closure(), EvidenceCoverageClosure::CompleteWithinProtocol);
        assert_eq!(report.coverage_level(), AuthorityLevel::Bound);
        assert!(report.protocol_local_coverage_established());
        assert!(!report.global_exhaustiveness_established());
    }

    #[test]
    fn relevant_item_omitted_from_claim_is_incomplete() {
        let subject = sha("subject");
        let claim_evidence = evidence(&subject, "E-CLAIM");
        let omitted = evidence(&subject, "E-OMITTED");
        let (claim, _adjudicated, binding) = bound_claim(std::slice::from_ref(&claim_evidence));
        let known = vec![claim_evidence.clone(), omitted.clone()];
        let report = audit_evidence_coverage(
            &claim,
            &binding,
            &protocol(&binding, &subject),
            &known,
            &[receipt([sha("item-claim"), sha("item-omitted")])],
            &[
                include("item-claim", &claim_evidence),
                include("item-omitted", &omitted),
            ],
        );
        assert_eq!(report.closure(), EvidenceCoverageClosure::Incomplete);
        assert!(report.relevant_evidence_omitted_from_claim().contains(omitted.evidence_id()));
    }

    #[test]
    fn missing_corpus_decision_is_incomplete() {
        let subject = sha("subject");
        let evidence = vec![evidence(&subject, "E-1")];
        let (claim, _adjudicated, binding) = bound_claim(&evidence);
        let report = audit_evidence_coverage(
            &claim,
            &binding,
            &protocol(&binding, &subject),
            &evidence,
            &[receipt([sha("item-1"), sha("item-2")])],
            &[include("item-1", &evidence[0])],
        );
        assert_eq!(report.closure(), EvidenceCoverageClosure::Incomplete);
        assert!(report.findings().iter().any(|finding| matches!(
            finding,
            EvidenceCoverageFinding::MissingItemDecision { .. }
        )));
    }

    #[test]
    fn query_substitution_is_invalid() {
        let subject = sha("subject");
        let evidence = vec![evidence(&subject, "E-1")];
        let (claim, _adjudicated, binding) = bound_claim(&evidence);
        let mut changed = receipt([sha("item-1")]);
        changed.query_sha256 = sha("different-query");
        let report = audit_evidence_coverage(
            &claim,
            &binding,
            &protocol(&binding, &subject),
            &evidence,
            &[changed],
            &[include("item-1", &evidence[0])],
        );
        assert_eq!(report.closure(), EvidenceCoverageClosure::Invalid);
    }

    #[test]
    fn supplemental_search_never_claims_bound_coverage() {
        let subject = sha("subject");
        let evidence = vec![evidence(&subject, "E-1")];
        let (claim, _adjudicated, binding) = bound_claim(&evidence);
        let mut draft = protocol(&binding, &subject).protocol().clone();
        draft.intent = EvidenceCoverageIntent::SupplementalSearch;
        let report = audit_evidence_coverage(
            &claim,
            &binding,
            &draft.freeze().unwrap(),
            &evidence,
            &[receipt([sha("item-1")])],
            &[include("item-1", &evidence[0])],
        );
        assert_eq!(report.closure(), EvidenceCoverageClosure::CompleteWithinProtocol);
        assert_eq!(report.coverage_level(), AuthorityLevel::Declared);
        assert!(!report.protocol_local_coverage_established());
    }
}
