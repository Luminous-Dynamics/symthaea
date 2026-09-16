// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim / Evidence / Provenance data model.
//!
//! This module separates *what is asserted* from *what supports or contradicts
//! the assertion* and *where that evidence came from*. It is deliberately a
//! passive data model: storing a claim or attaching evidence does not change the
//! knowledge graph, causal DAG, confidence policy, or action selection.
//!
//! The key epistemic invariant is:
//!
//! ```text
//! reported causal assertion != interventional causal evidence
//! ```
//!
//! A source saying "A causes B" can therefore be represented faithfully as a
//! causal `KnowledgeClaim` supported by `EvidenceKind::Report` without silently
//! promoting A -> B into an experimentally supported causal relation.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Stable identifier for a claim in an [`EpistemicLedger`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClaimId(pub u64);

/// Stable identifier for an evidence record in an [`EpistemicLedger`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvidenceId(pub u64);

/// Stable identifier for a provenance record in an [`EpistemicLedger`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProvenanceId(pub u64);

/// Semantic category of a claim.
///
/// The category describes the *content* of the assertion, not how strongly it
/// is supported. In particular, `Causal` does not imply causal evidence exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClaimKind {
    Descriptive,
    Predictive,
    Causal,
    Counterfactual,
    Procedural,
    Normative,
}

/// What kind of observation or derivation an evidence record represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceKind {
    /// A person, document, model, or other source asserted the proposition.
    Report,
    /// Passive observation without controlled intervention.
    Observation,
    /// Instrumented or otherwise quantified observation.
    Measurement,
    /// A controlled intervention intended to distinguish causal hypotheses.
    Intervention,
    /// Independent or repeated intervention testing an earlier result.
    Replication,
    /// Evidence generated inside a model or simulation rather than the target world.
    Simulation,
    /// Evidence derived by a formal or logical inference chain.
    Deduction,
    /// Evidence returned by an external tool or executable procedure.
    ToolResult,
}

/// Relationship between an evidence record and its target claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidencePolarity {
    Supports,
    Contradicts,
    Contextualizes,
}

/// Provenance for evidence.
///
/// `parent_ids` describe derivation ancestry. A copied article, transformed
/// dataset, or synthesized report can point back to the provenance records it
/// depends on. Because parents must already exist when a provenance record is
/// added, the ledger builds an acyclic ancestry graph by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceRecord {
    pub id: ProvenanceId,
    /// Human-readable source/provider label.
    pub source_label: String,
    /// URI or other source locator when available.
    pub source_uri: Option<String>,
    /// Content-addressed digest when one is available.
    pub content_hash: Option<String>,
    /// Cognitive/runtime cycle at which this provenance was registered.
    pub recorded_at_cycle: u64,
    /// Direct source ancestry for derived/copy/synthesis detection.
    pub parent_ids: Vec<ProvenanceId>,
}

/// A proposition Symthaea may reason about.
///
/// Evidence is linked by identifier instead of embedded so contradictory
/// evidence can coexist without overwriting history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeClaim {
    pub id: ClaimId,
    pub statement: String,
    pub kind: ClaimKind,
    pub domain: Option<String>,
    /// Optional applicability boundary such as an environment, population, or regime.
    pub scope: Option<String>,
    pub created_at_cycle: u64,
    pub evidence_ids: Vec<EvidenceId>,
}

/// One piece of evidence attached to a claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub id: EvidenceId,
    pub claim_id: ClaimId,
    pub kind: EvidenceKind,
    pub polarity: EvidencePolarity,
    pub provenance_id: ProvenanceId,
    pub observed_at_cycle: u64,
    /// Optional environment/population/experimental context.
    pub context: Option<String>,
    /// Optional method/protocol identifier or short description.
    pub method: Option<String>,
}

/// Referential-integrity failures at the epistemic ledger boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    UnknownClaim(ClaimId),
    UnknownProvenance(ProvenanceId),
    UnknownParentProvenance(ProvenanceId),
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownClaim(id) => write!(f, "unknown claim id {}", id.0),
            Self::UnknownProvenance(id) => write!(f, "unknown provenance id {}", id.0),
            Self::UnknownParentProvenance(id) => {
                write!(f, "unknown parent provenance id {}", id.0)
            }
        }
    }
}

impl Error for LedgerError {}

/// Passive in-memory ledger connecting claims, evidence, and provenance.
///
/// The ledger intentionally exposes descriptive queries rather than a single
/// truth score. Calibration, evidence independence, confidence aggregation,
/// and causal qualification are separate policies layered on top later.
#[derive(Debug, Clone)]
pub struct EpistemicLedger {
    claims: HashMap<ClaimId, KnowledgeClaim>,
    evidence: HashMap<EvidenceId, EvidenceRecord>,
    provenance: HashMap<ProvenanceId, ProvenanceRecord>,
    next_claim_id: u64,
    next_evidence_id: u64,
    next_provenance_id: u64,
}

impl Default for EpistemicLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicLedger {
    pub fn new() -> Self {
        Self {
            claims: HashMap::new(),
            evidence: HashMap::new(),
            provenance: HashMap::new(),
            next_claim_id: 1,
            next_evidence_id: 1,
            next_provenance_id: 1,
        }
    }

    /// Register a provenance node.
    ///
    /// Parents must already exist, which prevents dangling ancestry and makes
    /// cycles impossible through this API.
    pub fn add_provenance(
        &mut self,
        source_label: impl Into<String>,
        source_uri: Option<String>,
        content_hash: Option<String>,
        recorded_at_cycle: u64,
        parent_ids: Vec<ProvenanceId>,
    ) -> Result<ProvenanceId, LedgerError> {
        if let Some(missing) = parent_ids
            .iter()
            .find(|id| !self.provenance.contains_key(id))
            .copied()
        {
            return Err(LedgerError::UnknownParentProvenance(missing));
        }

        let id = ProvenanceId(self.next_provenance_id);
        self.next_provenance_id += 1;
        self.provenance.insert(
            id,
            ProvenanceRecord {
                id,
                source_label: source_label.into(),
                source_uri,
                content_hash,
                recorded_at_cycle,
                parent_ids,
            },
        );
        Ok(id)
    }

    /// Register a proposition without assigning it a truth status.
    pub fn add_claim(
        &mut self,
        statement: impl Into<String>,
        kind: ClaimKind,
        domain: Option<String>,
        scope: Option<String>,
        created_at_cycle: u64,
    ) -> ClaimId {
        let id = ClaimId(self.next_claim_id);
        self.next_claim_id += 1;
        self.claims.insert(
            id,
            KnowledgeClaim {
                id,
                statement: statement.into(),
                kind,
                domain,
                scope,
                created_at_cycle,
                evidence_ids: Vec::new(),
            },
        );
        id
    }

    /// Attach an evidence record to an existing claim and provenance node.
    ///
    /// This mutates only ledger linkage. It does not change a claim status,
    /// knowledge confidence, or causal graph.
    pub fn add_evidence(
        &mut self,
        claim_id: ClaimId,
        kind: EvidenceKind,
        polarity: EvidencePolarity,
        provenance_id: ProvenanceId,
        observed_at_cycle: u64,
        context: Option<String>,
        method: Option<String>,
    ) -> Result<EvidenceId, LedgerError> {
        if !self.claims.contains_key(&claim_id) {
            return Err(LedgerError::UnknownClaim(claim_id));
        }
        if !self.provenance.contains_key(&provenance_id) {
            return Err(LedgerError::UnknownProvenance(provenance_id));
        }

        let id = EvidenceId(self.next_evidence_id);
        self.next_evidence_id += 1;
        self.evidence.insert(
            id,
            EvidenceRecord {
                id,
                claim_id,
                kind,
                polarity,
                provenance_id,
                observed_at_cycle,
                context,
                method,
            },
        );
        self.claims
            .get_mut(&claim_id)
            .expect("claim existence checked above")
            .evidence_ids
            .push(id);
        Ok(id)
    }

    pub fn claim(&self, id: ClaimId) -> Option<&KnowledgeClaim> {
        self.claims.get(&id)
    }

    pub fn evidence(&self, id: EvidenceId) -> Option<&EvidenceRecord> {
        self.evidence.get(&id)
    }

    pub fn provenance(&self, id: ProvenanceId) -> Option<&ProvenanceRecord> {
        self.provenance.get(&id)
    }

    /// Evidence attached to a claim, preserving contradictory/contextual records.
    pub fn evidence_for_claim(&self, claim_id: ClaimId) -> Vec<&EvidenceRecord> {
        self.claims
            .get(&claim_id)
            .map(|claim| {
                claim
                    .evidence_ids
                    .iter()
                    .filter_map(|id| self.evidence.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Number of *supporting* intervention/replication records for a claim.
    ///
    /// Reports, observations, simulations, deductions, and tool results do not
    /// satisfy this predicate. This is deliberately a descriptive count, not a
    /// causal-qualification policy.
    pub fn interventional_support_count(&self, claim_id: ClaimId) -> usize {
        self.evidence_for_claim(claim_id)
            .into_iter()
            .filter(|record| {
                record.polarity == EvidencePolarity::Supports
                    && matches!(
                        record.kind,
                        EvidenceKind::Intervention | EvidenceKind::Replication
                    )
            })
            .count()
    }

    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }

    pub fn evidence_count(&self) -> usize {
        self.evidence.len()
    }

    pub fn provenance_count(&self) -> usize {
        self.provenance.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root_source(ledger: &mut EpistemicLedger, label: &str) -> ProvenanceId {
        ledger
            .add_provenance(label, None, None, 1, vec![])
            .expect("root provenance should be accepted")
    }

    #[test]
    fn reported_causal_claim_is_not_interventional_support() {
        let mut ledger = EpistemicLedger::new();
        let source = root_source(&mut ledger, "paper-a");
        let claim = ledger.add_claim(
            "A causes B",
            ClaimKind::Causal,
            Some("test".into()),
            None,
            1,
        );

        ledger
            .add_evidence(
                claim,
                EvidenceKind::Report,
                EvidencePolarity::Supports,
                source,
                1,
                None,
                None,
            )
            .unwrap();

        assert_eq!(ledger.claim(claim).unwrap().kind, ClaimKind::Causal);
        assert_eq!(ledger.interventional_support_count(claim), 0);
    }

    #[test]
    fn intervention_and_replication_are_counted_explicitly() {
        let mut ledger = EpistemicLedger::new();
        let experiment = root_source(&mut ledger, "experiment-a");
        let replication = root_source(&mut ledger, "experiment-b");
        let claim = ledger.add_claim("A causes B", ClaimKind::Causal, None, None, 1);

        ledger
            .add_evidence(
                claim,
                EvidenceKind::Intervention,
                EvidencePolarity::Supports,
                experiment,
                2,
                Some("environment-a".into()),
                Some("randomized intervention".into()),
            )
            .unwrap();
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Replication,
                EvidencePolarity::Supports,
                replication,
                3,
                Some("environment-b".into()),
                Some("independent replication".into()),
            )
            .unwrap();

        assert_eq!(ledger.interventional_support_count(claim), 2);
    }

    #[test]
    fn contradictory_evidence_coexists_without_overwriting_support() {
        let mut ledger = EpistemicLedger::new();
        let source_a = root_source(&mut ledger, "source-a");
        let source_b = root_source(&mut ledger, "source-b");
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);

        ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                source_a,
                2,
                None,
                None,
            )
            .unwrap();
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Contradicts,
                source_b,
                3,
                None,
                None,
            )
            .unwrap();

        let evidence = ledger.evidence_for_claim(claim);
        assert_eq!(evidence.len(), 2);
        assert!(evidence
            .iter()
            .any(|e| e.polarity == EvidencePolarity::Supports));
        assert!(evidence
            .iter()
            .any(|e| e.polarity == EvidencePolarity::Contradicts));
    }

    #[test]
    fn evidence_requires_existing_claim_and_provenance() {
        let mut ledger = EpistemicLedger::new();
        let source = root_source(&mut ledger, "source");
        let claim = ledger.add_claim("test", ClaimKind::Descriptive, None, None, 1);

        let missing_claim = ClaimId(999);
        assert_eq!(
            ledger.add_evidence(
                missing_claim,
                EvidenceKind::Observation,
                EvidencePolarity::Supports,
                source,
                1,
                None,
                None,
            ),
            Err(LedgerError::UnknownClaim(missing_claim))
        );

        let missing_source = ProvenanceId(999);
        assert_eq!(
            ledger.add_evidence(
                claim,
                EvidenceKind::Observation,
                EvidencePolarity::Supports,
                missing_source,
                1,
                None,
                None,
            ),
            Err(LedgerError::UnknownProvenance(missing_source))
        );
    }

    #[test]
    fn provenance_ancestry_requires_existing_parents() {
        let mut ledger = EpistemicLedger::new();
        let root = root_source(&mut ledger, "root");
        let derived = ledger
            .add_provenance(
                "derived",
                Some("https://example.invalid/derived".into()),
                Some("sha256:test".into()),
                2,
                vec![root],
            )
            .unwrap();
        assert_eq!(ledger.provenance(derived).unwrap().parent_ids, vec![root]);

        let missing = ProvenanceId(500);
        assert_eq!(
            ledger.add_provenance("bad", None, None, 3, vec![missing]),
            Err(LedgerError::UnknownParentProvenance(missing))
        );
    }
}
