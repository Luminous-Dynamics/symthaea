// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Memory subsystems for Symthaea.
//!
//! Extracted modules:
//! - Phase A: semantic memory (HDC-based similarity lookup), coherence tracking
//! - Phase B: hippocampus (episodic memory), conversation memory (SQLite)
//! - Phase C: episodic replay (Phi-prioritized consolidation), memory coordinator

pub mod coherence;
pub mod conversation;
pub mod coordinator;
pub mod episodic_component_index;
pub mod episodic_components;
pub mod episodic_occurrence;
pub mod episodic_provenance;
pub mod episodic_replay;
pub mod episodic_shadow;
pub mod hippocampus;
pub mod semantic;

pub use coherence::{CoherenceTrajectoryStats, ConversationCoherenceTracker};
pub use conversation::{
    CausalLearning, ConversationMemory, ConversationMemoryStats, ConversationSummary,
    ConversationTurn,
};
pub use coordinator::{
    CoordinatorConfig, CoordinatorStats, GraduationEvent, MemoryCoordinator, MemorySignals,
    MemorySource, content_hash,
};
pub use episodic_component_index::{
    EpisodicComponentIndexError, EpisodicComponentProvenanceIndex,
};
pub use episodic_components::{
    EpisodicComponentProvenance, EpisodicComponentProvenanceError,
    episode_cognition_subject_sha256, episode_perception_subject_sha256,
};
pub use episodic_occurrence::{
    EpisodicOccurrenceError, EpisodicOccurrenceIndex, EpisodicOccurrenceRecord,
    episode_occurrence_subject_sha256,
};
pub use episodic_provenance::{
    AuditedEpisodicRecall, EpisodicProvenanceError, EpisodicProvenanceIndex,
    EpisodicRetrievalAudit, ProvenanceAwareEpisodicMemory, ProvenancedEpisodeMatch,
    episode_subject_sha256,
};
pub use episodic_replay::{
    Episode, EpisodicMemory, EpisodicMemoryStats, EpisodicReplayConfig, ReplaySessionResult,
    bath_cosine_similarity,
};
pub use episodic_shadow::{
    EpisodicShadowRecallAudit, ShadowRecallModeAudit, shadow_audit_embedding_similarity,
    shadow_audit_input_similarity,
};
#[allow(deprecated)]
pub use hippocampus::{
    EmotionalValence, HippocampusActor, HippocampusStats, MemoryTrace, RecallQuery, RecallResult,
};
pub use semantic::{SemanticEntry, SemanticMemory, SemanticMemoryStats};

/// Trait abstraction for trainable temporal networks.
///
/// Allows episodic replay to train any compatible network without
/// depending on the concrete CfCNetwork type from the main crate.
pub trait TrainableNetwork {
    fn train_step(
        &mut self,
        input: &ndarray::Array1<f32>,
        target: &ndarray::Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> anyhow::Result<f32>;
}
