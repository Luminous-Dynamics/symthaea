// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dreaming Service — Candidate Generation and Gated Distillation
//!
//! Parser-accepted mutations are staged as proposals. They are not eligible
//! for distillation until an independent task-level verifier establishes that
//! they preserve the original task contract.

use crate::compiler_trainer::CompilerGroundedTrainer;
use crate::evolutionary_scaffolder::EvolutionaryScaffolder;
use crate::training::{TrainingDataset, TrainingPair};
use parking_lot::Mutex;
use std::sync::Arc;

pub struct DreamingService {
    scaffolder: EvolutionaryScaffolder,
    trainer: CompilerGroundedTrainer,
    /// Syntax-accepted candidates awaiting independent task-level verification.
    proposal_pool: Arc<Mutex<TrainingDataset>>,
    /// Only independently task-verified candidates may enter this pool.
    /// The current built-in path intentionally does not populate it.
    dream_pool: Arc<Mutex<TrainingDataset>>,
}

impl DreamingService {
    pub fn new() -> Self {
        Self {
            scaffolder: EvolutionaryScaffolder::default(),
            trainer: CompilerGroundedTrainer::default(),
            proposal_pool: Arc::new(Mutex::new(TrainingDataset::default())),
            dream_pool: Arc::new(Mutex::new(TrainingDataset::default())),
        }
    }

    /// Generate candidate mutations. Parser acceptance is not task verification.
    pub fn dream_cycle(&self, baseline_pairs: &[TrainingPair]) {
        for pair in baseline_pairs {
            if let Some(mutated) = self.scaffolder.scaffold_pair(pair) {
                // This checks Nix syntax, not preservation of task semantics.
                let result = self.trainer.train_step("dream", &mutated.target_text);
                if result.converged {
                    println!(
                        "Staged syntax-accepted proposal; independent task verification is required: reward={:.2}",
                        result.reward
                    );
                    self.stage_proposal(mutated);
                }
            }
        }
    }

    fn stage_proposal(&self, candidate: TrainingPair) {
        let mut proposals = self.proposal_pool.lock();
        let duplicate = proposals
            .pairs
            .iter()
            .any(|existing| same_training_pair(existing, &candidate));
        if !duplicate {
            proposals.push(candidate);
        }
    }

    /// Get syntax-accepted proposals awaiting an independent task-level oracle.
    pub fn proposed_dreams(&self) -> TrainingDataset {
        self.proposal_pool.lock().clone()
    }

    /// Get only independently task-verified dreams for distillation.
    /// The built-in dreaming path has no semantic task verifier yet, so this remains empty.
    pub fn internalized_dreams(&self) -> TrainingDataset {
        self.dream_pool.lock().clone()
    }
}

fn same_training_pair(left: &TrainingPair, right: &TrainingPair) -> bool {
    left.target_text == right.target_text
        && left.target_ids == right.target_ids
        && left.valence.to_bits() == right.valence.to_bits()
        && left.arousal.to_bits() == right.arousal.to_bits()
        && same_f32_slice(&left.channels, &right.channels)
        && match (&left.semantic_hv, &right.semantic_hv) {
            (Some(left), Some(right)) => same_f32_slice(left, right),
            (None, None) => true,
            _ => false,
        }
}

fn same_f32_slice(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

impl Default for DreamingService {
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax_accepted_proposals_are_not_internalized() {
        let service = DreamingService::new();
        let candidate = TrainingPair {
            channels: vec![0.0, 0.5],
            target_text: "{ value = 1; }".to_string(),
            ..TrainingPair::default()
        };

        service.stage_proposal(candidate);

        assert_eq!(service.proposed_dreams().len(), 1);
        assert!(service.internalized_dreams().is_empty());
    }

    #[test]
    fn replaying_the_same_proposal_is_idempotent() {
        let service = DreamingService::new();
        let candidate = TrainingPair {
            channels: vec![0.0, 0.5],
            target_text: "{ value = 1; }".to_string(),
            ..TrainingPair::default()
        };

        service.stage_proposal(candidate.clone());
        service.stage_proposal(candidate);

        assert_eq!(service.proposed_dreams().len(), 1);
        assert!(service.internalized_dreams().is_empty());
    }

    #[test]
    fn different_channel_contexts_remain_separate() {
        let service = DreamingService::new();
        let first = TrainingPair {
            channels: vec![0.0, 0.5],
            target_text: "{ value = 1; }".to_string(),
            ..TrainingPair::default()
        };
        let second = TrainingPair {
            channels: vec![1.0, 0.5],
            target_text: "{ value = 1; }".to_string(),
            ..TrainingPair::default()
        };

        service.stage_proposal(first);
        service.stage_proposal(second);

        assert_eq!(service.proposed_dreams().len(), 2);
        assert!(service.internalized_dreams().is_empty());
    }
}


#[cfg(test)]
mod proposal_identity_tests {
    use super::*;

    fn candidate() -> TrainingPair {
        TrainingPair {
            channels: vec![0.0, 0.5],
            target_text: "{ value = 1; }".to_string(),
            target_ids: vec![1, 2, 3],
            valence: 0.25,
            arousal: 0.5,
            semantic_hv: Some(vec![0.1, 0.2]),
        }
    }

    #[test]
    fn different_valence_is_not_deduplicated() {
        let service = DreamingService::new();
        let first = candidate();
        let mut second = candidate();
        second.valence = 0.75;

        service.stage_proposal(first);
        service.stage_proposal(second);

        assert_eq!(service.proposed_dreams().len(), 2);
    }

    #[test]
    fn different_semantic_conditioning_is_not_deduplicated() {
        let service = DreamingService::new();
        let first = candidate();
        let mut second = candidate();
        second.semantic_hv = Some(vec![0.1, 0.9]);

        service.stage_proposal(first);
        service.stage_proposal(second);

        assert_eq!(service.proposed_dreams().len(), 2);
    }

    #[test]
    fn different_tokenizer_ids_are_not_deduplicated() {
        let service = DreamingService::new();
        let first = candidate();
        let mut second = candidate();
        second.target_ids = vec![9, 8, 7];

        service.stage_proposal(first);
        service.stage_proposal(second);

        assert_eq!(service.proposed_dreams().len(), 2);
    }

    #[test]
    fn identical_complete_training_pairs_are_deduplicated() {
        let service = DreamingService::new();
        let first = candidate();
        let second = candidate();

        service.stage_proposal(first);
        service.stage_proposal(second);

        assert_eq!(service.proposed_dreams().len(), 1);
    }
}
