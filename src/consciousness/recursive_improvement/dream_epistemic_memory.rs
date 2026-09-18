// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Runtime-only aggregate for dream feedback, semantic retrieval, and derived narrative memory.
//!
//! This groups the non-authorizing dream-memory mechanisms behind one field so live
//! cognition can adopt them without increasing `CognitiveLoopService` field count.
//! The aggregate intentionally does not derive serde traits and intentionally has no
//! `NarrativeSelfModel` API: dream/replay material cannot mutate lived autobiography
//! through this capability surface.

use std::collections::HashSet;

use super::dream_feedback::{DreamFeedbackBridge, DreamInsight, hash_context};
use super::dream_semantic_coordinator::{
    DreamSemanticCoordinatorError, DreamSemanticOutcome, process_semantic_insight_atomically,
};
use super::dream_wisdom_provenance::{
    DreamWisdomBatchIdentity, DreamWisdomIdentity, DreamWisdomProposalIdentity,
    DreamWisdomProvenanceError, dream_wisdom_batch_identity, dream_wisdom_identity,
    dream_wisdom_proposal_identity,
};
use super::mce_narrative_boundary::{
    MceDerivedNarrativeError, MceDerivedNarrativeInput, MceDerivedNarrativeOutcome,
    MceDerivedNarrativeRouter, MceNarrativeTemporalClass,
};
use super::narrative_evidence::{
    NarrativeDerivedMemory, NarrativeDerivedRetention, NarrativeEvidenceError,
    NarrativeEvidenceInput, retain_derived_narrative,
};
use super::semantic_context::{SemanticContextEncoder, SemanticContextError};
use super::semantic_prior::{SemanticPriorCandidate, SemanticPriorError, SemanticPriorMemory};
use symthaea_consciousness_equation::NarrativeCoherence;
use symthaea_dream::Wisdom;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DreamWisdomItemDisposition {
    /// A previously unseen proposal was evaluated by the exact+semantic prior gate.
    ProposalEvaluated {
        semantic_outcome: DreamSemanticOutcome,
        narrative_inserted: bool,
        mce_outcome: MceDerivedNarrativeOutcome,
    },
    /// A distinct generated record rescored an already indexed context+action proposal.
    /// The record may be retained for provenance/imagination, but prior support is not recounted.
    ExistingProposalRecord {
        narrative_inserted: bool,
        mce_outcome: MceDerivedNarrativeOutcome,
    },
    /// The complete generated record was already processed during this runtime.
    DuplicateRecordIgnored,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamWisdomItemReceipt {
    pub wisdom_identity: DreamWisdomIdentity,
    pub proposal_identity: DreamWisdomProposalIdentity,
    pub disposition: DreamWisdomItemDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamWisdomBatchReceipt {
    pub batch_identity: DreamWisdomBatchIdentity,
    pub items: Vec<DreamWisdomItemReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DreamWisdomIngestionError {
    Provenance(DreamWisdomProvenanceError),
    SemanticContext(SemanticContextError),
    SemanticCoordinator(DreamSemanticCoordinatorError),
    Narrative(NarrativeEvidenceError),
    Mce(MceDerivedNarrativeError),
}

impl From<DreamWisdomProvenanceError> for DreamWisdomIngestionError {
    fn from(value: DreamWisdomProvenanceError) -> Self {
        Self::Provenance(value)
    }
}

impl From<SemanticContextError> for DreamWisdomIngestionError {
    fn from(value: SemanticContextError) -> Self {
        Self::SemanticContext(value)
    }
}

impl From<DreamSemanticCoordinatorError> for DreamWisdomIngestionError {
    fn from(value: DreamSemanticCoordinatorError) -> Self {
        Self::SemanticCoordinator(value)
    }
}

impl From<NarrativeEvidenceError> for DreamWisdomIngestionError {
    fn from(value: NarrativeEvidenceError) -> Self {
        Self::Narrative(value)
    }
}

impl From<MceDerivedNarrativeError> for DreamWisdomIngestionError {
    fn from(value: MceDerivedNarrativeError) -> Self {
        Self::Mce(value)
    }
}

#[derive(Clone, Default)]
pub struct DreamEpistemicMemory {
    feedback: DreamFeedbackBridge,
    semantic_priors: SemanticPriorMemory,
    narrative_derived: NarrativeDerivedMemory,
    mce_derived: MceDerivedNarrativeRouter,
    /// Successfully indexed context+action proposals. Full record-score changes do
    /// not create additional support counts for the same generated proposal.
    processed_proposals: HashSet<[u8; 32]>,
}

impl DreamEpistemicMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read-only access to exact dream-feedback state.
    ///
    /// Mutable access is intentionally not exposed: generated-prior mutation must
    /// pass through `process_semantic_insight` / `ingest_generated_wisdom_batch` so
    /// exact and semantic state cannot diverge.
    pub fn feedback(&self) -> &DreamFeedbackBridge {
        &self.feedback
    }

    pub fn semantic_priors(&self) -> &SemanticPriorMemory {
        &self.semantic_priors
    }

    pub fn narrative_derived(&self) -> &NarrativeDerivedMemory {
        &self.narrative_derived
    }

    pub fn mce_derived(&self) -> &MceDerivedNarrativeRouter {
        &self.mce_derived
    }

    pub fn processed_proposal_count(&self) -> usize {
        self.processed_proposals.len()
    }

    /// Atomically update exact dream feedback and semantic retrieval state.
    ///
    /// This low-level API does not perform proposal-level deduplication because it
    /// accepts a generic `DreamInsight`, not a provenance-bound `Wisdom` record.
    /// Live dream-wisdom ingestion should prefer `ingest_generated_wisdom_batch`.
    pub fn process_semantic_insight(
        &mut self,
        context: &[f32],
        insight: DreamInsight,
    ) -> Result<DreamSemanticOutcome, DreamSemanticCoordinatorError> {
        process_semantic_insight_atomically(
            &mut self.feedback,
            &mut self.semantic_priors,
            context,
            insight,
        )
    }

    /// Retrieve generated action-prior candidates by semantic context similarity.
    /// Returned strengths are exploration/search weights only.
    pub fn retrieve_semantic_priors(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
    ) -> Result<Vec<SemanticPriorCandidate>, SemanticPriorError> {
        self.semantic_priors.retrieve(query, min_similarity, limit)
    }

    /// Retain generated/replay narrative without possessing an autobiographical model.
    pub fn retain_derived_narrative(
        &mut self,
        input: NarrativeEvidenceInput<'_>,
    ) -> Result<NarrativeDerivedRetention, NarrativeEvidenceError> {
        retain_derived_narrative(&mut self.narrative_derived, input)
    }

    /// Route non-lived narrative material into the MCE simulation/recollection boundary.
    pub fn route_mce_derived(
        &mut self,
        narrative: &mut NarrativeCoherence,
        input: MceDerivedNarrativeInput<'_>,
    ) -> Result<MceDerivedNarrativeOutcome, MceDerivedNarrativeError> {
        self.mce_derived.route(narrative, input)
    }

    /// Atomically ingest generated dream wisdom without epistemic self-inflation.
    ///
    /// The complete record identity and proposal identity are separate. Exact record
    /// duplicates are ignored. A rescored version of an already indexed context+action
    /// proposal may be retained as a distinct provenance/imagination record, but it
    /// cannot increment generated-prior support again. A proposal is marked processed
    /// only after the semantic coordinator actually indexes it; a bridge-policy rejection
    /// therefore does not block a later stronger score for the same proposal.
    ///
    /// Current `Wisdom` is retrospective counterfactual material: it receives no
    /// autobiographical access and is MCE-neutral.
    pub fn ingest_generated_wisdom_batch(
        &mut self,
        narrative: &mut NarrativeCoherence,
        wisdom: &[Wisdom<Vec<f32>>],
    ) -> Result<DreamWisdomBatchReceipt, DreamWisdomIngestionError> {
        let batch_identity = dream_wisdom_batch_identity(wisdom)?;
        let encoder = SemanticContextEncoder::default();
        let mut staged_memory = self.clone();
        let mut staged_narrative = narrative.clone();
        let mut items = Vec::with_capacity(wisdom.len());

        for item in wisdom {
            let wisdom_identity = dream_wisdom_identity(item)?;
            let proposal_identity = dream_wisdom_proposal_identity(item)?;
            let source_digest = format!("blake3:{}", wisdom_identity.to_hex());

            if staged_memory.mce_derived.contains_source(&source_digest) {
                items.push(DreamWisdomItemReceipt {
                    wisdom_identity,
                    proposal_identity,
                    disposition: DreamWisdomItemDisposition::DuplicateRecordIgnored,
                });
                continue;
            }

            let proposal_already_indexed = staged_memory
                .processed_proposals
                .contains(&proposal_identity.digest);
            let semantic_outcome = if proposal_already_indexed {
                None
            } else {
                let outcome = staged_memory.process_semantic_insight(
                    &item.context_state,
                    DreamInsight::new(
                        hash_context(&item.context_state),
                        item.context_state.clone(),
                        item.better_action.clone(),
                        f64::from(item.phi_improvement),
                    ),
                )?;
                if matches!(outcome, DreamSemanticOutcome::Indexed { .. }) {
                    staged_memory
                        .processed_proposals
                        .insert(proposal_identity.digest);
                }
                Some(outcome)
            };

            let representation = encoder.encode(&item.context_state)?;
            let wisdom_hex = wisdom_identity.to_hex();
            let description = format!("dream_wisdom_{}", &wisdom_hex[..12]);
            let retained = staged_memory.retain_derived_narrative(NarrativeEvidenceInput {
                representation: &representation,
                description: &description,
                evidence_kind: super::epistemic_world::WorldEvidenceKind::Counterfactual,
                source_digest: &source_digest,
                appraisal_positive: item.phi_improvement > 0.0,
                effort: 0.0,
                significance: f64::from(item.phi_improvement).clamp(0.0, 1.0),
            })?;

            let mce_outcome = staged_memory.route_mce_derived(
                &mut staged_narrative,
                MceDerivedNarrativeInput {
                    evidence_kind: super::epistemic_world::WorldEvidenceKind::Counterfactual,
                    temporal_class: MceNarrativeTemporalClass::Retrospective,
                    source_digest: &source_digest,
                    description: &description,
                    horizon_steps: 1,
                    probability: 0.5,
                    desirability: f64::from(item.phi_improvement).clamp(-1.0, 1.0),
                },
            )?;

            let disposition = match semantic_outcome {
                Some(semantic_outcome) => DreamWisdomItemDisposition::ProposalEvaluated {
                    semantic_outcome,
                    narrative_inserted: retained.inserted,
                    mce_outcome,
                },
                None => DreamWisdomItemDisposition::ExistingProposalRecord {
                    narrative_inserted: retained.inserted,
                    mce_outcome,
                },
            };
            items.push(DreamWisdomItemReceipt {
                wisdom_identity,
                proposal_identity,
                disposition,
            });
        }

        *self = staged_memory;
        *narrative = staged_narrative;

        Ok(DreamWisdomBatchReceipt {
            batch_identity,
            items,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wisdom() -> Wisdom<Vec<f32>> {
        Wisdom {
            context_state: vec![0.2, -0.1, 0.7],
            better_action: vec![0.4, 0.5, 0.6],
            phi_improvement: 0.4,
            effective_information: 0.2,
            confidence: 0.8,
        }
    }

    #[test]
    fn semantic_dream_update_commits_into_grouped_memory() {
        let context = [0.2, -0.1, 0.7];
        let mut memory = DreamEpistemicMemory::new();
        let outcome = memory
            .process_semantic_insight(
                &context,
                DreamInsight::new(
                    hash_context(&context),
                    vec![0.1, 0.2, 0.3],
                    vec![0.4, 0.5, 0.6],
                    0.4,
                ),
            )
            .unwrap();

        assert!(matches!(outcome, DreamSemanticOutcome::Indexed { .. }));
        assert_eq!(memory.feedback().num_priors(), 1);
        assert_eq!(memory.semantic_priors().len(), 1);
    }

    #[test]
    fn direct_derived_retention_stays_outside_autobiography_by_type() {
        let mut memory = DreamEpistemicMemory::new();
        let representation = SemanticContextEncoder::default()
            .encode(&[0.2, -0.1, 0.7])
            .unwrap();
        let retained = memory
            .retain_derived_narrative(NarrativeEvidenceInput {
                representation: &representation,
                description: "dream alternative",
                evidence_kind: super::epistemic_world::WorldEvidenceKind::Counterfactual,
                source_digest: "blake3:dream-wisdom",
                appraisal_positive: true,
                effort: 0.3,
                significance: 0.8,
            })
            .unwrap();

        assert!(retained.inserted);
        assert_eq!(memory.narrative_derived().len(), 1);
    }

    #[test]
    fn retrospective_counterfactual_is_mce_neutral() {
        let mut memory = DreamEpistemicMemory::new();
        let mut narrative = NarrativeCoherence::new();
        let episodes_before = narrative.episode_count();
        let scenarios_before = narrative.scenario_count();

        assert_eq!(
            memory.route_mce_derived(
                &mut narrative,
                MceDerivedNarrativeInput {
                    evidence_kind: super::epistemic_world::WorldEvidenceKind::Counterfactual,
                    temporal_class: MceNarrativeTemporalClass::Retrospective,
                    source_digest: "blake3:dream-mce",
                    description: "dream alternative past",
                    horizon_steps: 1,
                    probability: 0.5,
                    desirability: 0.4,
                },
            ),
            Ok(MceDerivedNarrativeOutcome::NonProspectiveDerivedIgnored)
        );
        assert_eq!(narrative.episode_count(), episodes_before);
        assert_eq!(narrative.scenario_count(), scenarios_before);
        assert_eq!(memory.mce_derived().routed_count(), 1);
    }

    #[test]
    fn wisdom_batch_updates_search_and_imagination_without_mce_self_inflation() {
        let mut memory = DreamEpistemicMemory::new();
        let mut narrative = NarrativeCoherence::new();
        let receipt = memory
            .ingest_generated_wisdom_batch(&mut narrative, &[wisdom()])
            .unwrap();

        assert_eq!(receipt.items.len(), 1);
        assert!(matches!(
            receipt.items[0].disposition,
            DreamWisdomItemDisposition::ProposalEvaluated {
                semantic_outcome: DreamSemanticOutcome::Indexed { .. },
                mce_outcome: MceDerivedNarrativeOutcome::NonProspectiveDerivedIgnored,
                ..
            }
        ));
        assert_eq!(memory.feedback().num_priors(), 1);
        assert_eq!(memory.semantic_priors().len(), 1);
        assert_eq!(memory.processed_proposal_count(), 1);
        assert_eq!(memory.narrative_derived().len(), 1);
        assert_eq!(narrative.episode_count(), 0);
        assert_eq!(narrative.scenario_count(), 0);
    }

    #[test]
    fn exact_duplicate_record_is_ignored_before_prior_recount() {
        let mut memory = DreamEpistemicMemory::new();
        let mut narrative = NarrativeCoherence::new();
        let item = wisdom();
        memory
            .ingest_generated_wisdom_batch(&mut narrative, std::slice::from_ref(&item))
            .unwrap();
        let evidence_count = memory
            .feedback()
            .get_prior(hash_context(&item.context_state))
            .unwrap()
            .evidence_count;

        let second = memory
            .ingest_generated_wisdom_batch(&mut narrative, &[item])
            .unwrap();
        assert!(matches!(
            second.items[0].disposition,
            DreamWisdomItemDisposition::DuplicateRecordIgnored
        ));
        assert_eq!(
            memory
                .feedback()
                .get_prior(hash_context(&[0.2, -0.1, 0.7]))
                .unwrap()
                .evidence_count,
            evidence_count
        );
    }

    #[test]
    fn rescored_same_proposal_does_not_manufacture_support_count() {
        let mut memory = DreamEpistemicMemory::new();
        let mut narrative = NarrativeCoherence::new();
        let first = wisdom();
        memory
            .ingest_generated_wisdom_batch(&mut narrative, std::slice::from_ref(&first))
            .unwrap();
        let context_hash = hash_context(&first.context_state);
        let count_after_first = memory.feedback().get_prior(context_hash).unwrap().evidence_count;

        let mut rescored = first;
        rescored.phi_improvement = 0.7;
        rescored.effective_information = 0.3;
        rescored.confidence = 0.6;
        let second = memory
            .ingest_generated_wisdom_batch(&mut narrative, &[rescored])
            .unwrap();

        assert!(matches!(
            second.items[0].disposition,
            DreamWisdomItemDisposition::ExistingProposalRecord { .. }
        ));
        assert_eq!(
            memory.feedback().get_prior(context_hash).unwrap().evidence_count,
            count_after_first
        );
        assert_eq!(memory.processed_proposal_count(), 1);
        assert_eq!(memory.narrative_derived().len(), 2);
    }

    #[test]
    fn rejected_proposal_can_be_reconsidered_when_rescored() {
        let mut memory = DreamEpistemicMemory::new();
        let mut narrative = NarrativeCoherence::new();
        let mut weak = wisdom();
        weak.phi_improvement = 0.01;
        let first = memory
            .ingest_generated_wisdom_batch(&mut narrative, std::slice::from_ref(&weak))
            .unwrap();
        assert!(matches!(
            first.items[0].disposition,
            DreamWisdomItemDisposition::ProposalEvaluated {
                semantic_outcome: DreamSemanticOutcome::RejectedByDreamPolicy,
                ..
            }
        ));
        assert_eq!(memory.processed_proposal_count(), 0);

        weak.phi_improvement = 0.4;
        let second = memory
            .ingest_generated_wisdom_batch(&mut narrative, &[weak])
            .unwrap();
        assert!(matches!(
            second.items[0].disposition,
            DreamWisdomItemDisposition::ProposalEvaluated {
                semantic_outcome: DreamSemanticOutcome::Indexed { .. },
                ..
            }
        ));
        assert_eq!(memory.processed_proposal_count(), 1);
    }

    #[test]
    fn malformed_batch_rolls_back_all_staged_memory() {
        let mut memory = DreamEpistemicMemory::new();
        let mut narrative = NarrativeCoherence::new();
        let mut invalid = wisdom();
        invalid.context_state[0] = f32::NAN;

        assert!(memory
            .ingest_generated_wisdom_batch(&mut narrative, &[wisdom(), invalid])
            .is_err());
        assert_eq!(memory.feedback().num_priors(), 0);
        assert_eq!(memory.semantic_priors().len(), 0);
        assert_eq!(memory.processed_proposal_count(), 0);
        assert_eq!(memory.narrative_derived().len(), 0);
        assert_eq!(narrative.episode_count(), 0);
        assert_eq!(narrative.scenario_count(), 0);
    }

    #[test]
    fn grouped_memory_still_cannot_promote_unvalidated_confidence() {
        let context = [0.2, -0.1, 0.7];
        let context_hash = hash_context(&context);
        let mut memory = DreamEpistemicMemory::new();
        memory
            .process_semantic_insight(
                &context,
                DreamInsight::new(context_hash, vec![0.1], vec![0.9], 0.8),
            )
            .unwrap();

        let (adjusted, informed) = memory.feedback().adjust_confidence(0.7, context_hash);
        assert!(informed);
        assert!(adjusted <= 0.7);
    }
}
