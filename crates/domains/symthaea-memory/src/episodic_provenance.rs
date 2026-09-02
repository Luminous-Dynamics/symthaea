// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic provenance sidecar for episodic memory.
//!
//! This module deliberately leaves [`Episode`](crate::episodic_replay::Episode)
//! serialization unchanged. Provenance is stored in a separate index keyed by a
//! stable digest of immutable episode content. Existing/legacy episodic stores
//! therefore remain readable and are treated as epistemically `Unknown` until
//! explicit provenance is attached.

use crate::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use symthaea_epistemic_types::{ProvenanceEnvelope, ProvenanceRetrievalMode, RealityDomain};

const EPISODE_DIGEST_DOMAIN: &[u8] = b"SYMTHAEA-EPISODIC-PROVENANCE-v1\0";

/// Stable SHA-256 identity for the immutable epistemic content of an episode.
///
/// Mutable replay/consolidation counters are intentionally excluded. The digest
/// changes when perception/response content, encoding-time metadata, or semantic
/// embedding changes, but remains stable across replay and reconsolidation.
pub fn episode_subject_sha256(episode: &Episode) -> String {
    let mut hasher = Sha256::new();
    hasher.update(EPISODE_DIGEST_DOMAIN);
    hash_f32_slice(&mut hasher, &episode.input.values);
    hash_f32_slice(&mut hasher, &episode.output.values);
    hasher.update(episode.psi.to_bits().to_be_bytes());
    hasher.update(episode.timestamp.to_be_bytes());
    hash_opt_f32(&mut hasher, episode.prediction_error);
    hash_opt_f32(&mut hasher, episode.valence);
    hash_opt_f32(&mut hasher, episode.coherence);
    hash_opt_f32(&mut hasher, episode.dopamine_at_encoding);
    hash_opt_f32_array_9(&mut hasher, episode.bath_state_at_encoding.as_ref());
    match episode.semantic_embedding.as_ref() {
        Some(values) => {
            hasher.update([1]);
            hash_f32_slice(&mut hasher, values);
        }
        None => hasher.update([0]),
    }
    let digest = hasher.finalize();
    format!("{digest:x}")
}

fn hash_f32_slice(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_be_bytes());
    for value in values {
        hasher.update(value.to_bits().to_be_bytes());
    }
}

fn hash_opt_f32(hasher: &mut Sha256, value: Option<f32>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.to_bits().to_be_bytes());
        }
        None => hasher.update([0]),
    }
}

fn hash_opt_f32_array_9(hasher: &mut Sha256, value: Option<&[f32; 9]>) {
    match value {
        Some(values) => {
            hasher.update([1]);
            for value in values {
                hasher.update(value.to_bits().to_be_bytes());
            }
        }
        None => hasher.update([0]),
    }
}

/// Persistable provenance metadata kept outside the historical `Episode` schema.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpisodicProvenanceIndex {
    by_subject_sha256: HashMap<String, ProvenanceEnvelope>,
}

impl EpisodicProvenanceIndex {
    pub fn len(&self) -> usize {
        self.by_subject_sha256.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_subject_sha256.is_empty()
    }

    pub fn get(&self, subject_sha256: &str) -> Option<&ProvenanceEnvelope> {
        self.by_subject_sha256.get(subject_sha256)
    }

    /// Bind provenance to an exact immutable episode identity.
    ///
    /// Existing bindings are immutable: attempting to overwrite the same episode
    /// with different provenance fails closed. A later grounding event should be
    /// represented as a new epistemic object/episode rather than laundering the
    /// historical record in place.
    pub fn attach(
        &mut self,
        episode: &Episode,
        envelope: ProvenanceEnvelope,
    ) -> Result<(), EpisodicProvenanceError> {
        let expected = episode_subject_sha256(episode);
        if envelope.subject_sha256 != expected {
            return Err(EpisodicProvenanceError::SubjectDigestMismatch {
                expected,
                got: envelope.subject_sha256,
            });
        }

        if let Some(existing) = self.by_subject_sha256.get(&expected) {
            if existing != &envelope {
                return Err(EpisodicProvenanceError::ConflictingImmutableBinding {
                    subject_sha256: expected,
                });
            }
            return Ok(());
        }

        self.by_subject_sha256.insert(expected, envelope);
        Ok(())
    }

    /// Return explicit provenance for an episode, synthesizing an ungrounded
    /// `Unknown` envelope for legacy/unannotated episodes.
    pub fn effective_provenance(&self, episode: &Episode) -> ProvenanceEnvelope {
        let subject_sha256 = episode_subject_sha256(episode);
        self.by_subject_sha256
            .get(&subject_sha256)
            .cloned()
            .unwrap_or_else(|| {
                ProvenanceEnvelope::new(
                    subject_sha256,
                    RealityDomain::Unknown,
                    Vec::new(),
                    None,
                    0.0,
                )
                .expect("episode digest is always valid lowercase SHA-256")
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodicProvenanceError {
    SubjectDigestMismatch { expected: String, got: String },
    ConflictingImmutableBinding { subject_sha256: String },
}

impl fmt::Display for EpisodicProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SubjectDigestMismatch { expected, got } => write!(
                f,
                "episode provenance subject mismatch: expected {expected}, got {got}"
            ),
            Self::ConflictingImmutableBinding { subject_sha256 } => write!(
                f,
                "episode provenance binding is immutable for subject {subject_sha256}"
            ),
        }
    }
}

impl std::error::Error for EpisodicProvenanceError {}

/// Audit record that distinguishes epistemic exclusion from ranking truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EpisodicRetrievalAudit {
    /// Episodes that passed the underlying similarity/vector-shape eligibility checks.
    pub similarity_eligible: usize,
    /// Similarity-eligible episodes admitted by the provenance view before top-k.
    pub provenance_admitted: usize,
    /// Final matches returned after top-k truncation.
    pub returned: usize,
    /// Excluded because active counterfactual taint was incompatible with the view.
    pub excluded_taint: usize,
    /// Excluded because a known non-tainted domain was incompatible with the view.
    pub excluded_domain: usize,
    /// Excluded because the episode had no attached provenance / effective Unknown domain.
    pub excluded_unknown: usize,
    /// Provenance-admitted matches omitted only because of top-k ranking.
    pub truncated_by_top_k: usize,
}

#[derive(Debug, Clone)]
pub struct ProvenancedEpisodeMatch {
    pub episode: Episode,
    pub similarity: f32,
    pub subject_sha256: String,
    /// Always present, including explicit `Unknown` for legacy episodes.
    pub provenance: ProvenanceEnvelope,
}

#[derive(Debug, Clone, Default)]
pub struct AuditedEpisodicRecall {
    pub matches: Vec<ProvenancedEpisodeMatch>,
    pub audit: EpisodicRetrievalAudit,
}

/// Compatibility wrapper that adds epistemic provenance without changing the
/// serialized `Episode` layout or existing `EpisodicMemory` behavior.
pub struct ProvenanceAwareEpisodicMemory {
    memory: EpisodicMemory,
    provenance: EpisodicProvenanceIndex,
}

impl ProvenanceAwareEpisodicMemory {
    pub fn new(config: EpisodicReplayConfig) -> Self {
        Self {
            memory: EpisodicMemory::new(config),
            provenance: EpisodicProvenanceIndex::default(),
        }
    }

    /// Wrap an existing episodic store. Existing episodes have no sidecar entry
    /// and therefore resolve to explicit `Unknown` provenance.
    pub fn from_legacy(memory: EpisodicMemory) -> Self {
        Self {
            memory,
            provenance: EpisodicProvenanceIndex::default(),
        }
    }

    pub fn with_index(memory: EpisodicMemory, provenance: EpisodicProvenanceIndex) -> Self {
        Self { memory, provenance }
    }

    pub fn memory(&self) -> &EpisodicMemory {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut EpisodicMemory {
        &mut self.memory
    }

    pub fn provenance_index(&self) -> &EpisodicProvenanceIndex {
        &self.provenance
    }

    pub fn into_parts(self) -> (EpisodicMemory, EpisodicProvenanceIndex) {
        (self.memory, self.provenance)
    }

    /// Preserve the old storage path. The stored episode remains epistemically
    /// Unknown until provenance is attached through a provenance-aware path.
    pub fn store_if_significant(&mut self, episode: Episode) -> bool {
        self.memory.store_if_significant(episode)
    }

    /// Store an episode only after provenance is proven to bind to its exact digest.
    pub fn store_if_significant_with_provenance(
        &mut self,
        episode: Episode,
        envelope: ProvenanceEnvelope,
    ) -> Result<bool, EpisodicProvenanceError> {
        let expected = episode_subject_sha256(&episode);
        if envelope.subject_sha256 != expected {
            return Err(EpisodicProvenanceError::SubjectDigestMismatch {
                expected,
                got: envelope.subject_sha256,
            });
        }
        if let Some(existing) = self.provenance.get(&expected) {
            if existing != &envelope {
                return Err(EpisodicProvenanceError::ConflictingImmutableBinding {
                    subject_sha256: expected,
                });
            }
        }

        let stored = self.memory.store_if_significant(episode.clone());
        if stored {
            self.provenance.attach(&episode, envelope)?;
        }
        Ok(stored)
    }

    pub fn retrieve_by_input_similarity_with_provenance(
        &self,
        query: &[f32],
        top_k: usize,
        mode: ProvenanceRetrievalMode,
    ) -> AuditedEpisodicRecall {
        let candidates = self
            .memory
            .retrieve_by_input_similarity(query, self.memory.len());
        self.apply_provenance_view(candidates, top_k, mode)
    }

    pub fn retrieve_by_embedding_similarity_with_provenance(
        &self,
        query: &[f32],
        top_k: usize,
        mode: ProvenanceRetrievalMode,
    ) -> AuditedEpisodicRecall {
        let candidates = self
            .memory
            .retrieve_by_embedding_similarity(query, self.memory.len());
        self.apply_provenance_view(candidates, top_k, mode)
    }

    fn apply_provenance_view(
        &self,
        candidates: Vec<(Episode, f32)>,
        top_k: usize,
        mode: ProvenanceRetrievalMode,
    ) -> AuditedEpisodicRecall {
        let mut audit = EpisodicRetrievalAudit {
            similarity_eligible: candidates.len(),
            ..EpisodicRetrievalAudit::default()
        };
        let mut admitted = Vec::new();

        for (episode, similarity) in candidates {
            let subject_sha256 = episode_subject_sha256(&episode);
            let provenance = self.provenance.effective_provenance(&episode);
            if mode.allows(&provenance) {
                audit.provenance_admitted += 1;
                admitted.push(ProvenancedEpisodeMatch {
                    episode,
                    similarity,
                    subject_sha256,
                    provenance,
                });
                continue;
            }

            if provenance.domain == RealityDomain::Unknown {
                audit.excluded_unknown += 1;
            } else if provenance.counterfactual_taint {
                audit.excluded_taint += 1;
            } else {
                audit.excluded_domain += 1;
            }
        }

        let admitted_count = admitted.len();
        admitted.truncate(top_k);
        audit.returned = admitted.len();
        audit.truncated_by_top_k = admitted_count.saturating_sub(audit.returned);

        AuditedEpisodicRecall {
            matches: admitted,
            audit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_epistemic_types::GroundingEvidence;

    fn episode(seed: u64) -> Episode {
        Episode::new(
            ContinuousHV::random(256, seed),
            ContinuousHV::random(256, seed + 1),
            0.8,
            seed,
        )
    }

    #[test]
    fn stable_subject_digest_ignores_mutable_replay_state() {
        let base = episode(10);
        let digest = episode_subject_sha256(&base);
        let mut replayed = base.clone();
        replayed.replay_count = 99;
        replayed.retrieval_count = 42;
        replayed.consolidation_strength = 4.5;
        assert_eq!(digest, episode_subject_sha256(&replayed));
    }

    #[test]
    fn subject_digest_changes_when_epistemic_content_changes() {
        let base = episode(10);
        let mut changed = base.clone();
        changed.input.values[0] += 0.25;
        assert_ne!(episode_subject_sha256(&base), episode_subject_sha256(&changed));
    }

    #[test]
    fn mismatched_provenance_binding_fails_closed() {
        let ep = episode(11);
        let wrong = ProvenanceEnvelope::new(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            RealityDomain::Imported,
            vec!["source".into()],
            None,
            0.5,
        )
        .unwrap();
        let mut index = EpisodicProvenanceIndex::default();
        assert!(matches!(
            index.attach(&ep, wrong),
            Err(EpisodicProvenanceError::SubjectDigestMismatch { .. })
        ));
    }

    #[test]
    fn legacy_episode_is_unknown_and_excluded_from_grounded_history() {
        let ep = episode(12);
        let query = ep.input.values.clone();
        let mut memory = ProvenanceAwareEpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        assert!(memory.store_if_significant(ep));

        let grounded = memory.retrieve_by_input_similarity_with_provenance(
            &query,
            4,
            ProvenanceRetrievalMode::GroundedHistory,
        );
        assert!(grounded.matches.is_empty());
        assert_eq!(grounded.audit.similarity_eligible, 1);
        assert_eq!(grounded.audit.excluded_unknown, 1);

        let all = memory.retrieve_by_input_similarity_with_provenance(
            &query,
            4,
            ProvenanceRetrievalMode::AllWithProvenance,
        );
        assert_eq!(all.matches.len(), 1);
        assert_eq!(all.matches[0].provenance.domain, RealityDomain::Unknown);
    }

    #[test]
    fn grounded_episode_enters_history_only_with_matching_evidence() {
        let ep = episode(13);
        let query = ep.input.values.clone();
        let subject = episode_subject_sha256(&ep);
        let evidence = GroundingEvidence::direct_observation(
            subject,
            "obs-13",
            "sensor-13",
            Some(13),
            0.95,
        )
        .unwrap();
        let provenance = ProvenanceEnvelope::from_grounding(evidence);
        let mut memory = ProvenanceAwareEpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        assert!(memory
            .store_if_significant_with_provenance(ep, provenance)
            .unwrap());

        let recall = memory.retrieve_by_input_similarity_with_provenance(
            &query,
            4,
            ProvenanceRetrievalMode::GroundedHistory,
        );
        assert_eq!(recall.matches.len(), 1);
        assert_eq!(recall.audit.provenance_admitted, 1);
        assert_eq!(
            recall.matches[0].provenance.domain,
            RealityDomain::PhysicalGrounded
        );
    }

    #[test]
    fn counterfactual_memory_is_audited_as_taint_exclusion() {
        let ep = episode(14);
        let query = ep.input.values.clone();
        let provenance = ProvenanceEnvelope::new(
            episode_subject_sha256(&ep),
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            None,
            0.6,
        )
        .unwrap();
        let mut memory = ProvenanceAwareEpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        assert!(memory
            .store_if_significant_with_provenance(ep, provenance)
            .unwrap());

        let recall = memory.retrieve_by_input_similarity_with_provenance(
            &query,
            4,
            ProvenanceRetrievalMode::GroundedHistory,
        );
        assert!(recall.matches.is_empty());
        assert_eq!(recall.audit.excluded_taint, 1);
    }

    #[test]
    fn top_k_truncation_is_not_misreported_as_epistemic_rejection() {
        let ep1 = episode(20);
        let mut ep2 = ep1.clone();
        ep2.timestamp = 21;
        let query = ep1.input.values.clone();
        let mut memory = ProvenanceAwareEpisodicMemory::new(EpisodicReplayConfig::broad_capture());

        for ep in [ep1, ep2] {
            let evidence = GroundingEvidence::direct_observation(
                episode_subject_sha256(&ep),
                format!("obs-{}", ep.timestamp),
                "sensor",
                Some(ep.timestamp),
                0.9,
            )
            .unwrap();
            memory
                .store_if_significant_with_provenance(
                    ep,
                    ProvenanceEnvelope::from_grounding(evidence),
                )
                .unwrap();
        }

        let recall = memory.retrieve_by_input_similarity_with_provenance(
            &query,
            1,
            ProvenanceRetrievalMode::GroundedHistory,
        );
        assert_eq!(recall.audit.similarity_eligible, 2);
        assert_eq!(recall.audit.provenance_admitted, 2);
        assert_eq!(recall.audit.returned, 1);
        assert_eq!(recall.audit.truncated_by_top_k, 1);
        assert_eq!(
            recall.audit.excluded_taint
                + recall.audit.excluded_domain
                + recall.audit.excluded_unknown,
            0
        );
    }
}
