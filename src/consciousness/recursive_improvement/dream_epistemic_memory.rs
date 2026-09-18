// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Runtime-only aggregate for dream feedback, semantic retrieval, and derived narrative memory.
//!
//! This groups the three non-authorizing dream-memory mechanisms behind one field so
//! live cognition can adopt them without increasing `CognitiveLoopService` field count.
//! The aggregate intentionally does not derive serde traits: restart/persistence must
//! never turn semantic analogy or generated narrative into empirical authority.

use super::dream_feedback::{DreamFeedbackBridge, DreamInsight};
use super::dream_semantic_coordinator::{
    DreamSemanticCoordinatorError, DreamSemanticOutcome, process_semantic_insight_atomically,
};
use super::narrative_evidence::{
    NarrativeDerivedMemory, NarrativeEvidenceError, NarrativeEvidenceInput,
    NarrativeEvidenceReceipt, ingest_narrative_evidence_with_derived_memory,
};
use super::semantic_prior::{SemanticPriorCandidate, SemanticPriorError, SemanticPriorMemory};
use crate::consciousness::narrative_self::NarrativeSelfModel;

#[derive(Clone, Default)]
pub struct DreamEpistemicMemory {
    feedback: DreamFeedbackBridge,
    semantic_priors: SemanticPriorMemory,
    narrative_derived: NarrativeDerivedMemory,
}

impl DreamEpistemicMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feedback(&self) -> &DreamFeedbackBridge {
        &self.feedback
    }

    pub fn feedback_mut(&mut self) -> &mut DreamFeedbackBridge {
        &mut self.feedback
    }

    pub fn semantic_priors(&self) -> &SemanticPriorMemory {
        &self.semantic_priors
    }

    pub fn narrative_derived(&self) -> &NarrativeDerivedMemory {
        &self.narrative_derived
    }

    /// Atomically update exact dream feedback and semantic retrieval state.
    ///
    /// The coordinator guarantees that an accepted generated prior cannot land in
    /// only one of the two stores. This operation still grants no confidence or
    /// evidence authority.
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
        self.semantic_priors
            .retrieve(query, min_similarity, limit)
    }

    /// Route narrative material through the lived/recollection/imagination boundary.
    pub fn ingest_narrative(
        &mut self,
        model: &mut NarrativeSelfModel,
        input: NarrativeEvidenceInput<'_>,
    ) -> Result<NarrativeEvidenceReceipt, NarrativeEvidenceError> {
        ingest_narrative_evidence_with_derived_memory(
            model,
            &mut self.narrative_derived,
            input,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::dream_feedback::hash_context;
    use super::super::epistemic_world::WorldEvidenceKind;
    use crate::consciousness::narrative_self::NarrativeSelfConfig;
    use crate::hdc::binary_hv::BinaryHV;

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
    fn generated_narrative_stays_out_of_lived_autobiography() {
        let mut memory = DreamEpistemicMemory::new();
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());
        let representation = BinaryHV::random(9001);

        let receipt = memory
            .ingest_narrative(
                &mut model,
                NarrativeEvidenceInput {
                    representation: &representation,
                    description: "dream alternative",
                    evidence_kind: WorldEvidenceKind::Counterfactual,
                    source_digest: "blake3:dream-wisdom",
                    appraisal_positive: true,
                    effort: 0.3,
                    significance: 0.8,
                },
            )
            .unwrap();

        assert!(!receipt.recorded_episode());
        assert_eq!(model.autobio.life_story.len(), 0);
        assert_eq!(memory.narrative_derived().len(), 1);
    }

    #[test]
    fn grouped_memory_still_cannot_promote_unvalidated_confidence() {
        let context = [0.2, -0.1, 0.7];
        let context_hash = hash_context(&context);
        let mut memory = DreamEpistemicMemory::new();
        memory
            .process_semantic_insight(
                &context,
                DreamInsight::new(
                    context_hash,
                    vec![0.1],
                    vec![0.9],
                    0.8,
                ),
            )
            .unwrap();

        let (adjusted, informed) = memory.feedback().adjust_confidence(0.7, context_hash);
        assert!(informed);
        assert!(adjusted <= 0.7);
    }
}
