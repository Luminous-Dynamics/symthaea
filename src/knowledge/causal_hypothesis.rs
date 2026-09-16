// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic causal hypotheses and evidence profiles.
//!
//! A causal sentence is represented here as a *hypothesis about persistent
//! entities*, not as an edge in the executable causal DAG. Evidence channels
//! remain separate so reports, passive observations, simulations, controlled
//! interventions, replications, and contradictions cannot be silently collapsed.

use super::claim_evidence::{
    ClaimId, ClaimKind, EpistemicLedger, EvidenceKind, EvidencePolarity,
};
use super::entity_event::{EntityEventStore, EntityId};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CausalHypothesisId(pub u64);

/// Direction/sign of the proposed effect, independent of confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalSign {
    Increases,
    Decreases,
    Enables,
    Inhibits,
    Unknown,
}

/// Explicit semantic causal proposition connecting persistent entities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalHypothesis {
    pub id: CausalHypothesisId,
    pub cause: EntityId,
    pub effect: EntityId,
    /// The epistemic claim containing the causal assertion.
    pub claim_id: ClaimId,
    pub sign: CausalSign,
    /// Optional context/regime in which the causal assertion is proposed.
    pub context: Option<String>,
    /// Claims describing a proposed mechanism. These are references, not proof.
    pub mechanism_claim_ids: Vec<ClaimId>,
    pub created_at_cycle: u64,
}

/// Descriptive census of evidence attached to a causal claim.
///
/// No weighted sum is provided intentionally. Different evidence classes have
/// different epistemic meaning and should remain inspectable independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CausalEvidenceProfile {
    pub supporting_reports: usize,
    pub supporting_observations: usize,
    pub supporting_measurements: usize,
    pub supporting_interventions: usize,
    pub supporting_replications: usize,
    pub supporting_simulations: usize,
    pub supporting_deductions: usize,
    pub supporting_tool_results: usize,
    pub contradicting_reports: usize,
    pub contradicting_observations: usize,
    pub contradicting_measurements: usize,
    pub contradicting_interventions: usize,
    pub contradicting_replications: usize,
    pub contradicting_simulations: usize,
    pub contradicting_deductions: usize,
    pub contradicting_tool_results: usize,
    pub contextual_records: usize,
}

impl CausalEvidenceProfile {
    pub fn has_interventional_support(&self) -> bool {
        self.supporting_interventions > 0 || self.supporting_replications > 0
    }

    pub fn has_replication_support(&self) -> bool {
        self.supporting_replications > 0
    }

    pub fn has_interventional_contradiction(&self) -> bool {
        self.contradicting_interventions > 0 || self.contradicting_replications > 0
    }

    pub fn supporting_record_count(&self) -> usize {
        self.supporting_reports
            + self.supporting_observations
            + self.supporting_measurements
            + self.supporting_interventions
            + self.supporting_replications
            + self.supporting_simulations
            + self.supporting_deductions
            + self.supporting_tool_results
    }

    pub fn contradicting_record_count(&self) -> usize {
        self.contradicting_reports
            + self.contradicting_observations
            + self.contradicting_measurements
            + self.contradicting_interventions
            + self.contradicting_replications
            + self.contradicting_simulations
            + self.contradicting_deductions
            + self.contradicting_tool_results
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CausalHypothesisError {
    UnknownCause(EntityId),
    UnknownEffect(EntityId),
    UnknownClaim(ClaimId),
    ClaimIsNotCausal(ClaimId),
    UnknownMechanismClaim(ClaimId),
}

impl fmt::Display for CausalHypothesisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCause(id) => write!(f, "unknown causal-hypothesis cause entity {}", id.0),
            Self::UnknownEffect(id) => write!(f, "unknown causal-hypothesis effect entity {}", id.0),
            Self::UnknownClaim(id) => write!(f, "unknown causal claim {}", id.0),
            Self::ClaimIsNotCausal(id) => write!(f, "claim {} is not typed as causal", id.0),
            Self::UnknownMechanismClaim(id) => write!(f, "unknown mechanism claim {}", id.0),
        }
    }
}

impl Error for CausalHypothesisError {}

/// Stores semantic causal candidates without promoting them into a causal DAG.
#[derive(Debug, Clone)]
pub struct CausalHypothesisStore {
    hypotheses: HashMap<CausalHypothesisId, CausalHypothesis>,
    next_id: u64,
}

impl Default for CausalHypothesisStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalHypothesisStore {
    pub fn new() -> Self {
        Self {
            hypotheses: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn add_hypothesis(
        &mut self,
        entities: &EntityEventStore,
        ledger: &EpistemicLedger,
        cause: EntityId,
        effect: EntityId,
        claim_id: ClaimId,
        sign: CausalSign,
        context: Option<String>,
        mechanism_claim_ids: Vec<ClaimId>,
        created_at_cycle: u64,
    ) -> Result<CausalHypothesisId, CausalHypothesisError> {
        if entities.entity(cause).is_none() {
            return Err(CausalHypothesisError::UnknownCause(cause));
        }
        if entities.entity(effect).is_none() {
            return Err(CausalHypothesisError::UnknownEffect(effect));
        }
        let claim = ledger
            .claim(claim_id)
            .ok_or(CausalHypothesisError::UnknownClaim(claim_id))?;
        if claim.kind != ClaimKind::Causal {
            return Err(CausalHypothesisError::ClaimIsNotCausal(claim_id));
        }
        if let Some(missing) = mechanism_claim_ids
            .iter()
            .find(|id| ledger.claim(**id).is_none())
            .copied()
        {
            return Err(CausalHypothesisError::UnknownMechanismClaim(missing));
        }

        let id = CausalHypothesisId(self.next_id);
        self.next_id += 1;
        self.hypotheses.insert(
            id,
            CausalHypothesis {
                id,
                cause,
                effect,
                claim_id,
                sign,
                context,
                mechanism_claim_ids,
                created_at_cycle,
            },
        );
        Ok(id)
    }

    pub fn hypothesis(&self, id: CausalHypothesisId) -> Option<&CausalHypothesis> {
        self.hypotheses.get(&id)
    }

    /// Classify the evidence currently attached to the hypothesis's claim.
    pub fn evidence_profile(
        &self,
        ledger: &EpistemicLedger,
        id: CausalHypothesisId,
    ) -> Option<CausalEvidenceProfile> {
        let hypothesis = self.hypotheses.get(&id)?;
        let mut profile = CausalEvidenceProfile::default();

        for evidence in ledger.evidence_for_claim(hypothesis.claim_id) {
            match evidence.polarity {
                EvidencePolarity::Contextualizes => {
                    profile.contextual_records += 1;
                }
                EvidencePolarity::Supports => increment_support(&mut profile, evidence.kind),
                EvidencePolarity::Contradicts => {
                    increment_contradiction(&mut profile, evidence.kind)
                }
            }
        }

        Some(profile)
    }

    pub fn len(&self) -> usize {
        self.hypotheses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hypotheses.is_empty()
    }
}

fn increment_support(profile: &mut CausalEvidenceProfile, kind: EvidenceKind) {
    match kind {
        EvidenceKind::Report => profile.supporting_reports += 1,
        EvidenceKind::Observation => profile.supporting_observations += 1,
        EvidenceKind::Measurement => profile.supporting_measurements += 1,
        EvidenceKind::Intervention => profile.supporting_interventions += 1,
        EvidenceKind::Replication => profile.supporting_replications += 1,
        EvidenceKind::Simulation => profile.supporting_simulations += 1,
        EvidenceKind::Deduction => profile.supporting_deductions += 1,
        EvidenceKind::ToolResult => profile.supporting_tool_results += 1,
    }
}

fn increment_contradiction(profile: &mut CausalEvidenceProfile, kind: EvidenceKind) {
    match kind {
        EvidenceKind::Report => profile.contradicting_reports += 1,
        EvidenceKind::Observation => profile.contradicting_observations += 1,
        EvidenceKind::Measurement => profile.contradicting_measurements += 1,
        EvidenceKind::Intervention => profile.contradicting_interventions += 1,
        EvidenceKind::Replication => profile.contradicting_replications += 1,
        EvidenceKind::Simulation => profile.contradicting_simulations += 1,
        EvidenceKind::Deduction => profile.contradicting_deductions += 1,
        EvidenceKind::ToolResult => profile.contradicting_tool_results += 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::claim_evidence::EvidenceKind;
    use super::super::extraction::EntityType;

    fn fixture() -> (EntityEventStore, EpistemicLedger, EntityId, EntityId, ClaimId) {
        let mut entities = EntityEventStore::new();
        let cause = entities
            .create_entity("temperature", EntityType::Quantity, 1)
            .unwrap();
        let effect = entities
            .create_entity("evaporation", EntityType::Process, 1)
            .unwrap();
        let mut ledger = EpistemicLedger::new();
        let claim = ledger.add_claim(
            "higher temperature increases evaporation",
            ClaimKind::Causal,
            Some("physics".into()),
            None,
            1,
        );
        (entities, ledger, cause, effect, claim)
    }

    fn add_evidence(
        ledger: &mut EpistemicLedger,
        claim: ClaimId,
        kind: EvidenceKind,
        polarity: EvidencePolarity,
        label: &str,
    ) {
        let provenance = ledger
            .add_provenance(label, None, None, 2, vec![])
            .unwrap();
        ledger
            .add_evidence(claim, kind, polarity, provenance, 2, None, None)
            .unwrap();
    }

    #[test]
    fn reported_causal_claim_remains_report_only() {
        let (entities, mut ledger, cause, effect, claim) = fixture();
        add_evidence(
            &mut ledger,
            claim,
            EvidenceKind::Report,
            EvidencePolarity::Supports,
            "paper",
        );

        let mut store = CausalHypothesisStore::new();
        let hypothesis = store
            .add_hypothesis(
                &entities,
                &ledger,
                cause,
                effect,
                claim,
                CausalSign::Increases,
                None,
                vec![],
                2,
            )
            .unwrap();
        let profile = store.evidence_profile(&ledger, hypothesis).unwrap();

        assert_eq!(profile.supporting_reports, 1);
        assert!(!profile.has_interventional_support());
        assert!(!profile.has_replication_support());
    }

    #[test]
    fn simulation_does_not_masquerade_as_intervention() {
        let (entities, mut ledger, cause, effect, claim) = fixture();
        add_evidence(
            &mut ledger,
            claim,
            EvidenceKind::Simulation,
            EvidencePolarity::Supports,
            "simulator",
        );

        let mut store = CausalHypothesisStore::new();
        let hypothesis = store
            .add_hypothesis(
                &entities,
                &ledger,
                cause,
                effect,
                claim,
                CausalSign::Increases,
                None,
                vec![],
                2,
            )
            .unwrap();
        let profile = store.evidence_profile(&ledger, hypothesis).unwrap();
        assert_eq!(profile.supporting_simulations, 1);
        assert!(!profile.has_interventional_support());
    }

    #[test]
    fn intervention_support_and_contradiction_coexist() {
        let (entities, mut ledger, cause, effect, claim) = fixture();
        add_evidence(
            &mut ledger,
            claim,
            EvidenceKind::Intervention,
            EvidencePolarity::Supports,
            "experiment-a",
        );
        add_evidence(
            &mut ledger,
            claim,
            EvidenceKind::Replication,
            EvidencePolarity::Contradicts,
            "experiment-b",
        );

        let mut store = CausalHypothesisStore::new();
        let hypothesis = store
            .add_hypothesis(
                &entities,
                &ledger,
                cause,
                effect,
                claim,
                CausalSign::Increases,
                None,
                vec![],
                3,
            )
            .unwrap();
        let profile = store.evidence_profile(&ledger, hypothesis).unwrap();

        assert!(profile.has_interventional_support());
        assert!(profile.has_interventional_contradiction());
        assert_eq!(profile.supporting_record_count(), 1);
        assert_eq!(profile.contradicting_record_count(), 1);
    }

    #[test]
    fn non_causal_claim_cannot_back_a_causal_hypothesis() {
        let mut entities = EntityEventStore::new();
        let cause = entities.create_entity("A", EntityType::Concept, 1).unwrap();
        let effect = entities.create_entity("B", EntityType::Concept, 1).unwrap();
        let mut ledger = EpistemicLedger::new();
        let descriptive = ledger.add_claim("A resembles B", ClaimKind::Descriptive, None, None, 1);

        let mut store = CausalHypothesisStore::new();
        assert_eq!(
            store.add_hypothesis(
                &entities,
                &ledger,
                cause,
                effect,
                descriptive,
                CausalSign::Unknown,
                None,
                vec![],
                1,
            ),
            Err(CausalHypothesisError::ClaimIsNotCausal(descriptive))
        );
    }

    #[test]
    fn unknown_entities_fail_closed() {
        let (entities, ledger, cause, _effect, claim) = fixture();
        let mut store = CausalHypothesisStore::new();
        assert_eq!(
            store.add_hypothesis(
                &entities,
                &ledger,
                cause,
                EntityId(999),
                claim,
                CausalSign::Unknown,
                None,
                vec![],
                1,
            ),
            Err(CausalHypothesisError::UnknownEffect(EntityId(999)))
        );
    }
}
