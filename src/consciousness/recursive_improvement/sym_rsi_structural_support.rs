// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quality-independent structural support for SYM-RSI-001D dream v7.
//!
//! This module implements only the preregistered structural support substrate.
//! It does not execute dream verification, fresh, or OOD seeds and does not alter
//! arm-D action selection yet.

use super::experience_tree::ExperienceNodeId;
use super::sym_rsi_candidate_family::{
    canonical_candidate_family_digest, canonical_fixed_hash_candidate_family, FrozenReplayCorpus,
    SYM_RSI_001_CANDIDATE_FAMILY_SCHEMA, SYM_RSI_001_REPLAY_CORPUS_SCHEMA,
};
use super::sym_rsi_experiment::{EvaluationSplit, SymRsiExperimentManifest};
use super::sym_rsi_fixtures::{
    canonical_sym_rsi_001_fixture_manifest, FixtureDomainKind, FixtureState,
};
use super::sym_rsi_runner::fixture_action_digest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SYM_RSI_STRUCTURAL_SUPPORT_SCHEMA: &str =
    "symthaea.sym-rsi.structural-support-index.v1";
pub const STRUCTURAL_SUPPORT_DIM: usize = 7;

/// The preregistered quality-independent state representation used only for
/// locality/support estimation. Task quality and terminal status are intentionally
/// absent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralStateVector(pub [f32; STRUCTURAL_SUPPORT_DIM]);

impl StructuralStateVector {
    pub fn cosine_similarity(self, other: Self) -> f32 {
        let mut dot = 0.0_f32;
        let mut left_norm = 0.0_f32;
        let mut right_norm = 0.0_f32;
        for index in 0..STRUCTURAL_SUPPORT_DIM {
            let left = self.0[index];
            let right = other.0[index];
            dot += left * right;
            left_norm += left * left;
            right_norm += right * right;
        }
        if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
            return 0.0;
        }
        (dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(-1.0, 1.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralSupportEntry {
    pub domain: FixtureDomainKind,
    pub action: u8,
    pub parent_state_digest: String,
    pub vector: StructuralStateVector,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralSupportIndex {
    pub schema: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub training_corpus_evidence_digest: String,
    pub entries: Vec<StructuralSupportEntry>,
    pub evidence_digest: String,
}

impl StructuralSupportIndex {
    /// Maximum same-domain/same-action structural cosine support, clamped to [0,1].
    /// Missing support returns None rather than silently invoking a fallback model.
    pub fn nearest_support(
        &self,
        domain: FixtureDomainKind,
        action: u8,
        query: StructuralStateVector,
    ) -> Option<f32> {
        self.entries
            .iter()
            .filter(|entry| entry.domain == domain && entry.action == action)
            .map(|entry| query.cosine_similarity(entry.vector).clamp(0.0, 1.0))
            .filter(|similarity| similarity.is_finite())
            .max_by(|left, right| {
                left.partial_cmp(right)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    pub fn support_count(&self, domain: FixtureDomainKind, action: u8) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.domain == domain && entry.action == action)
            .count()
    }
}

/// Encode only structural state. Quality and terminal status are deliberately
/// excluded so support cannot become circular with the task outcome being predicted.
pub fn structural_state_vector(
    domain: FixtureDomainKind,
    split: EvaluationSplit,
    state: &FixtureState,
) -> StructuralStateVector {
    let mut vector = [0.0_f32; STRUCTURAL_SUPPORT_DIM];
    vector[domain_rank(domain)] = 1.0;
    vector[3] = state.step as f32 / domain.horizon(split).max(1) as f32;
    vector[4] = signed_unit(state.a);
    vector[5] = signed_unit(state.b);
    vector[6] = state.aux as f32 / (state.aux as f32 + 10.0);
    StructuralStateVector(vector)
}

/// Build a deterministic support index solely from observed TrainingReplay parent
/// states and their recorded outgoing actions.
pub fn build_structural_support_index(
    manifest: &SymRsiExperimentManifest,
    training_corpus: &FrozenReplayCorpus,
) -> Result<StructuralSupportIndex, StructuralSupportError> {
    ensure_canonical_parent_manifest(manifest)?;
    validate_training_corpus(manifest, training_corpus)?;

    let mut entries = Vec::new();
    for world in training_corpus.worlds() {
        let mut stack = vec![world.root_node_id];
        let mut visited = BTreeSet::new();
        while let Some(parent_id) = stack.pop() {
            if !visited.insert(parent_id) {
                continue;
            }
            let parent_state = world
                .state(parent_id)
                .ok_or(StructuralSupportError::MissingRawState(parent_id))?;
            let legal_actions = world.domain.legal_actions(parent_state, world.split);
            let parent_node = world
                .experience_tree
                .node(parent_id)
                .ok_or(StructuralSupportError::MissingReplayNode(parent_id))?;
            let vector = structural_state_vector(world.domain, world.split, parent_state);

            for &child_id in world.experience_tree.children_of(parent_id) {
                let child = world
                    .experience_tree
                    .node(child_id)
                    .ok_or(StructuralSupportError::MissingReplayNode(child_id))?;
                let action = legal_actions
                    .iter()
                    .copied()
                    .find(|candidate| {
                        fixture_action_digest(world.domain, *candidate) == child.action_digest
                    })
                    .ok_or(StructuralSupportError::UnrecoverableObservedAction {
                        node_id: child_id,
                    })?;
                entries.push(StructuralSupportEntry {
                    domain: world.domain,
                    action,
                    parent_state_digest: parent_node.world_state_digest.clone(),
                    vector,
                });
                stack.push(child_id);
            }
        }
    }

    if entries.is_empty() {
        return Err(StructuralSupportError::EmptyObservedSupport);
    }
    entries.sort_by(compare_entries);

    let evidence_digest = structural_support_digest(manifest, training_corpus, &entries);
    Ok(StructuralSupportIndex {
        schema: SYM_RSI_STRUCTURAL_SUPPORT_SCHEMA.into(),
        experiment_id: manifest.experiment_id.clone(),
        preregistration_digest: manifest.preregistration_digest.clone(),
        subject_digest: manifest.subject_digest.clone(),
        environment_digest: manifest.environment_digest.clone(),
        training_corpus_evidence_digest: training_corpus.receipt.evidence_digest.clone(),
        entries,
        evidence_digest,
    })
}

fn validate_training_corpus(
    manifest: &SymRsiExperimentManifest,
    corpus: &FrozenReplayCorpus,
) -> Result<(), StructuralSupportError> {
    let receipt = &corpus.receipt;
    let canonical_ids = canonical_fixed_hash_candidate_family()
        .into_iter()
        .map(|candidate| candidate.policy_id)
        .collect::<Vec<_>>();
    if receipt.schema != SYM_RSI_001_REPLAY_CORPUS_SCHEMA
        || receipt.candidate_family_schema != SYM_RSI_001_CANDIDATE_FAMILY_SCHEMA
        || receipt.split != EvaluationSplit::TrainingReplay
        || receipt.experiment_id != manifest.experiment_id
        || receipt.preregistration_digest != manifest.preregistration_digest
        || receipt.subject_digest != manifest.subject_digest
        || receipt.environment_digest != manifest.environment_digest
        || receipt.candidate_family_digest != canonical_candidate_family_digest()
        || receipt.collector_policy_ids != canonical_ids
        || receipt.evidence_digest.trim().is_empty()
        || corpus
            .worlds()
            .iter()
            .any(|world| world.split != EvaluationSplit::TrainingReplay)
    {
        return Err(StructuralSupportError::TrainingCorpusBindingMismatch);
    }

    let mut expected = BTreeSet::new();
    let mut expected_seed_count_per_domain = None;
    for domain in &manifest.domains {
        match expected_seed_count_per_domain {
            None => expected_seed_count_per_domain = Some(domain.seeds.training_replay.len()),
            Some(expected_count) if expected_count != domain.seeds.training_replay.len() => {
                return Err(StructuralSupportError::TrainingCorpusIncomplete);
            }
            Some(_) => {}
        }
        for &seed in &domain.seeds.training_replay {
            expected.insert((domain.domain_id.clone(), seed));
        }
    }
    let mut observed = BTreeSet::new();
    for world in corpus.worlds() {
        let key = (world.domain.id().to_owned(), world.seed);
        if !observed.insert(key) {
            return Err(StructuralSupportError::DuplicateTrainingWorld);
        }
    }

    let expected_seed_count_per_domain = expected_seed_count_per_domain.unwrap_or(0);
    let expected_trajectory_count = expected.len() * canonical_ids.len();
    if observed != expected
        || receipt.domain_count != manifest.domains.len()
        || receipt.seed_count_per_domain != expected_seed_count_per_domain
        || receipt.world_count != expected.len()
        || receipt.world_count != corpus.worlds().len()
        || receipt.trajectory_count != expected_trajectory_count
    {
        return Err(StructuralSupportError::TrainingCorpusIncomplete);
    }
    Ok(())
}

fn ensure_canonical_parent_manifest(
    manifest: &SymRsiExperimentManifest,
) -> Result<(), StructuralSupportError> {
    manifest
        .validate()
        .map_err(|_| StructuralSupportError::ManifestInvalid)?;
    let expected = canonical_sym_rsi_001_fixture_manifest(
        manifest.preregistration_digest.clone(),
        manifest.subject_digest.clone(),
        manifest.environment_digest.clone(),
    );
    if manifest != &expected {
        return Err(StructuralSupportError::ManifestNotCanonical);
    }
    Ok(())
}

fn compare_entries(
    left: &StructuralSupportEntry,
    right: &StructuralSupportEntry,
) -> std::cmp::Ordering {
    domain_rank(left.domain)
        .cmp(&domain_rank(right.domain))
        .then_with(|| left.action.cmp(&right.action))
        .then_with(|| left.parent_state_digest.cmp(&right.parent_state_digest))
        .then_with(|| {
            for index in 0..STRUCTURAL_SUPPORT_DIM {
                let ordering = left.vector.0[index]
                    .to_bits()
                    .cmp(&right.vector.0[index].to_bits());
                if ordering != std::cmp::Ordering::Equal {
                    return ordering;
                }
            }
            std::cmp::Ordering::Equal
        })
}

fn structural_support_digest(
    manifest: &SymRsiExperimentManifest,
    corpus: &FrozenReplayCorpus,
    entries: &[StructuralSupportEntry],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi.structural-support-index.v1\0");
    for value in [
        manifest.experiment_id.as_str(),
        manifest.preregistration_digest.as_str(),
        manifest.subject_digest.as_str(),
        manifest.environment_digest.as_str(),
        corpus.receipt.evidence_digest.as_str(),
        SYM_RSI_STRUCTURAL_SUPPORT_SCHEMA,
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&(STRUCTURAL_SUPPORT_DIM as u64).to_le_bytes());
    hasher.update(&(entries.len() as u64).to_le_bytes());
    for entry in entries {
        hasher.update(&[domain_rank(entry.domain) as u8, entry.action]);
        hasher.update(&(entry.parent_state_digest.len() as u64).to_le_bytes());
        hasher.update(entry.parent_state_digest.as_bytes());
        for value in entry.vector.0 {
            hasher.update(&value.to_bits().to_le_bytes());
        }
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn domain_rank(domain: FixtureDomainKind) -> usize {
    match domain {
        FixtureDomainKind::BranchingSearch => 0,
        FixtureDomainKind::DelayedNavigation => 1,
        FixtureDomainKind::RuggedOptimization => 2,
    }
}

fn signed_unit(value: i32) -> f32 {
    let value = value as f32;
    value / (1.0 + value.abs())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralSupportError {
    ManifestInvalid,
    ManifestNotCanonical,
    TrainingCorpusBindingMismatch,
    DuplicateTrainingWorld,
    TrainingCorpusIncomplete,
    MissingRawState(ExperienceNodeId),
    MissingReplayNode(ExperienceNodeId),
    UnrecoverableObservedAction { node_id: ExperienceNodeId },
    EmptyObservedSupport,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::acquire_canonical_replay_corpus;

    #[test]
    fn structural_vector_is_exactly_seven_dimensions_and_quality_independent() {
        let domain = FixtureDomainKind::RuggedOptimization;
        let split = EvaluationSplit::TrainingReplay;
        let mut first = domain.reset(1, split);
        first.step = 4;
        first.a = 7;
        first.b = -2;
        first.aux = 3;
        first.quality = 0.1;
        first.terminal = false;

        let mut second = first.clone();
        second.quality = 0.95;
        second.terminal = true;

        let left = structural_state_vector(domain, split, &first);
        let right = structural_state_vector(domain, split, &second);
        assert_eq!(left.0.len(), STRUCTURAL_SUPPORT_DIM);
        assert_eq!(left, right);
    }

    #[test]
    fn changing_structural_coordinates_changes_vector() {
        let domain = FixtureDomainKind::DelayedNavigation;
        let split = EvaluationSplit::TrainingReplay;
        let first = domain.reset(1, split);
        let mut second = first.clone();
        second.a += 1;
        assert_ne!(
            structural_state_vector(domain, split, &first),
            structural_state_vector(domain, split, &second)
        );
    }

    #[test]
    fn exact_structural_match_has_unit_similarity() {
        let domain = FixtureDomainKind::BranchingSearch;
        let split = EvaluationSplit::TrainingReplay;
        let state = domain.reset(1, split);
        let vector = structural_state_vector(domain, split, &state);
        assert!((vector.cosine_similarity(vector) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn support_index_is_training_only_and_evidence_bound() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay).unwrap();
        let index = build_structural_support_index(&manifest, &training).unwrap();

        assert_eq!(index.schema, SYM_RSI_STRUCTURAL_SUPPORT_SCHEMA);
        assert_eq!(
            index.training_corpus_evidence_digest,
            training.receipt.evidence_digest
        );
        assert!(!index.entries.is_empty());
        assert!(index.evidence_digest.starts_with("blake3:"));
        assert!(index.entries.iter().all(|entry| {
            entry.parent_state_digest.starts_with("blake3:")
                && entry.vector.0.iter().all(|value| value.is_finite())
        }));
    }

    #[test]
    fn nearest_support_is_same_domain_same_action_only() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay).unwrap();
        let index = build_structural_support_index(&manifest, &training).unwrap();
        let entry = index.entries.first().unwrap();
        let exact = index
            .nearest_support(entry.domain, entry.action, entry.vector)
            .unwrap();
        assert!((exact - 1.0).abs() < 1e-6);

        let other_domain = FixtureDomainKind::ALL
            .into_iter()
            .find(|domain| *domain != entry.domain)
            .unwrap();
        let cross_domain = index.nearest_support(other_domain, entry.action, entry.vector);
        if index.support_count(other_domain, entry.action) == 0 {
            assert!(cross_domain.is_none());
        }
    }
}
