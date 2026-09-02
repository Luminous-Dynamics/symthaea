// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Episodic Persistence Manager — episodic replay + memory database.
//!
//! Groups episodic memory replay, persistent SQLite database, flush guard, reasoning
//! context, composite provenance, and component-level epistemic provenance into one
//! owned persistence boundary.
//!
//! Provenance remains sidecar state: the historical `Episode` serialization format is
//! unchanged. Existing episodic stores therefore remain readable and resolve to explicit
//! `Unknown` provenance until subject-bound metadata is attached.

use std::fmt;

/// Consolidated episodic persistence subsystem.
pub struct EpisodicPersistenceManager {
    /// Canonical production episodic replay store.
    pub(crate) replay: Option<crate::memory::episodic_replay::EpisodicMemory>,

    /// Composite epistemic provenance keyed by immutable episode subject digest.
    pub(crate) provenance: crate::memory::EpisodicProvenanceIndex,

    /// Rich component-level provenance for perception + internal cognition + composite
    /// derivation. This index is additive and must agree with `provenance` whenever a
    /// component record exists.
    pub(crate) component_provenance: crate::memory::EpisodicComponentProvenanceIndex,

    /// Persistent memory database for cross-session episode storage.
    pub(crate) db: Option<std::sync::Arc<crate::databases::SqliteMemory>>,

    /// Optional write-behind runtime for non-blocking durable memory writes.
    pub(crate) storage_runtime: Option<crate::databases::storage_runtime::StorageRuntimeHandle>,

    /// Worker backing the storage runtime when it was spawned from this manager.
    storage_worker: Option<std::thread::JoinHandle<()>>,

    /// Guard to prevent overlapping memory flushes.
    pub(crate) flush_in_progress: std::sync::Arc<std::sync::atomic::AtomicBool>,

    /// Last assembled reasoning context from the knowledge engine.
    pub(crate) last_reasoning_context: Option<crate::knowledge::ReasoningContext>,
}

impl EpisodicPersistenceManager {
    /// Create a new persistence manager. Existing episodes inside `replay` have no
    /// sidecar bindings and therefore resolve to explicit `Unknown` provenance.
    pub fn new(replay: Option<crate::memory::episodic_replay::EpisodicMemory>) -> Self {
        Self {
            replay,
            provenance: crate::memory::EpisodicProvenanceIndex::default(),
            component_provenance: crate::memory::EpisodicComponentProvenanceIndex::default(),
            db: None,
            storage_runtime: None,
            storage_worker: None,
            flush_in_progress: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            last_reasoning_context: None,
        }
    }

    /// Attach subject-bound composite provenance to an exact episodic record.
    pub fn attach_episode_provenance(
        &mut self,
        episode: &crate::memory::Episode,
        envelope: symthaea_epistemic_types::ProvenanceEnvelope,
    ) -> Result<(), crate::memory::EpisodicProvenanceError> {
        self.provenance.attach(episode, envelope)
    }

    /// Transactionally attach component provenance and publish its composite envelope
    /// through the existing provenance index.
    ///
    /// Both indexes are preflighted before either is mutated. This prevents the richer
    /// component record and legacy composite view from diverging.
    pub fn attach_episode_component_provenance(
        &mut self,
        episode: &crate::memory::Episode,
        record: crate::memory::EpisodicComponentProvenance,
    ) -> Result<(), EpisodicComponentBindingError> {
        self.component_provenance.preflight(episode, &record)?;
        let episode_sha = crate::memory::episode_subject_sha256(episode);
        if let Some(existing) = self.provenance.get(&episode_sha) {
            if existing != &record.episode {
                return Err(EpisodicComponentBindingError::CompositeConflict {
                    episode_sha256: episode_sha,
                });
            }
        }

        // After both preflight checks, these writes are deterministic/idempotent.
        self.component_provenance.attach(episode, record.clone())?;
        self.provenance
            .attach(episode, record.episode)
            .map_err(EpisodicComponentBindingError::CompositeProvenance)?;
        Ok(())
    }

    /// Return explicit composite provenance for an episode. Legacy/unannotated episodes
    /// resolve to `RealityDomain::Unknown` with zero confidence.
    pub fn effective_episode_provenance(
        &self,
        episode: &crate::memory::Episode,
    ) -> symthaea_epistemic_types::ProvenanceEnvelope {
        self.provenance.effective_provenance(episode)
    }

    pub fn provenance_index(&self) -> &crate::memory::EpisodicProvenanceIndex {
        &self.provenance
    }

    pub fn component_provenance_index(
        &self,
    ) -> &crate::memory::EpisodicComponentProvenanceIndex {
        &self.component_provenance
    }

    /// Behavior-neutral shadow audit of compressed-input episodic recall.
    pub fn shadow_audit_input_recall(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Option<crate::memory::EpisodicShadowRecallAudit> {
        self.replay.as_ref().map(|replay| {
            crate::memory::shadow_audit_input_similarity(replay, &self.provenance, query, top_k)
        })
    }

    /// Behavior-neutral shadow audit of semantic-embedding episodic recall.
    pub fn shadow_audit_embedding_recall(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Option<crate::memory::EpisodicShadowRecallAudit> {
        self.replay.as_ref().map(|replay| {
            crate::memory::shadow_audit_embedding_similarity(replay, &self.provenance, query, top_k)
        })
    }

    /// Replace the replay store and deliberately reset *all* provenance bindings.
    /// Restoring persisted replay+provenance must use a future paired restore API.
    pub fn replace_replay(
        &mut self,
        replay: Option<crate::memory::episodic_replay::EpisodicMemory>,
    ) {
        self.replay = replay;
        self.provenance = crate::memory::EpisodicProvenanceIndex::default();
        self.component_provenance = crate::memory::EpisodicComponentProvenanceIndex::default();
    }

    pub fn attach_sqlite_db<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> crate::databases::DbResult<()> {
        let db = crate::databases::SqliteMemory::new(path)?;
        let db = std::sync::Arc::new(db);
        let backend: std::sync::Arc<dyn crate::databases::ConsciousnessDatabase> = db.clone();
        let (runtime, worker) = crate::databases::storage_runtime::spawn_storage_runtime_threaded(
            backend,
            crate::databases::storage_runtime::DEFAULT_STORAGE_QUEUE_CAPACITY,
        );
        self.stop_owned_storage_worker();
        self.db = Some(db);
        self.storage_runtime = Some(runtime);
        self.storage_worker = Some(worker);
        Ok(())
    }

    pub fn attach_storage_runtime(
        &mut self,
        runtime: crate::databases::storage_runtime::StorageRuntimeHandle,
    ) {
        self.stop_owned_storage_worker();
        self.storage_runtime = Some(runtime);
    }

    fn stop_owned_storage_worker(&mut self) {
        self.storage_runtime.take();
        if let Some(worker) = self.storage_worker.take() {
            if worker.join().is_err() {
                tracing::warn!("episodic storage runtime worker panicked during shutdown");
            }
        }
    }
}

#[derive(Debug)]
pub enum EpisodicComponentBindingError {
    Component(crate::memory::EpisodicComponentIndexError),
    CompositeConflict { episode_sha256: String },
    CompositeProvenance(crate::memory::EpisodicProvenanceError),
}

impl fmt::Display for EpisodicComponentBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Component(error) => write!(f, "component provenance preflight failed: {error}"),
            Self::CompositeConflict { episode_sha256 } => write!(
                f,
                "composite provenance conflicts with component record for episode {episode_sha256}"
            ),
            Self::CompositeProvenance(error) => {
                write!(f, "composite provenance attachment failed: {error}")
            }
        }
    }
}

impl std::error::Error for EpisodicComponentBindingError {}

impl From<crate::memory::EpisodicComponentIndexError> for EpisodicComponentBindingError {
    fn from(value: crate::memory::EpisodicComponentIndexError) -> Self {
        Self::Component(value)
    }
}

impl Drop for EpisodicPersistenceManager {
    fn drop(&mut self) {
        self.stop_owned_storage_worker();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_epistemic_types::{GroundingEvidence, ProvenanceEnvelope, RealityDomain};

    fn episode(seed: u64) -> crate::memory::Episode {
        crate::memory::Episode::new(
            ContinuousHV::random(64, seed),
            ContinuousHV::random(64, seed + 1),
            0.8,
            seed,
        )
    }

    fn component_record(ep: &crate::memory::Episode) -> crate::memory::EpisodicComponentProvenance {
        let perception = ProvenanceEnvelope::from_grounding(
            GroundingEvidence::direct_observation(
                crate::memory::episode_perception_subject_sha256(ep),
                "obs",
                "sensor",
                None,
                0.95,
            )
            .unwrap(),
        );
        let cognition = ProvenanceEnvelope::new(
            crate::memory::episode_cognition_subject_sha256(ep),
            RealityDomain::Unknown,
            vec!["cognitive-loop".into()],
            None,
            0.8,
        )
        .unwrap();
        crate::memory::EpisodicComponentProvenance::compose(
            ep,
            perception,
            cognition,
            "symthaea.episode-compose",
            "v1",
        )
        .unwrap()
    }

    #[test]
    fn test_episodic_persistence_manager_new() {
        let mgr = EpisodicPersistenceManager::new(None);
        assert!(mgr.replay.is_none());
        assert!(mgr.provenance_index().is_empty());
        assert!(mgr.component_provenance_index().is_empty());
        assert!(mgr.db.is_none());
        assert!(mgr.storage_runtime.is_none());
        assert!(mgr.storage_worker.is_none());
        assert!(!mgr.flush_in_progress.load(std::sync::atomic::Ordering::Relaxed));
        assert!(mgr.last_reasoning_context.is_none());
    }

    #[test]
    fn legacy_episode_resolves_to_unknown_not_grounded() {
        let mgr = EpisodicPersistenceManager::new(None);
        let ep = episode(7);
        let provenance = mgr.effective_episode_provenance(&ep);
        assert_eq!(provenance.domain, RealityDomain::Unknown);
        assert!(!provenance.may_enter_grounded_history());
        assert_eq!(provenance.confidence, 0.0);
    }

    #[test]
    fn manager_rejects_provenance_for_wrong_episode_digest() {
        let mut mgr = EpisodicPersistenceManager::new(None);
        let ep = episode(8);
        let wrong = ProvenanceEnvelope::new(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            RealityDomain::Imported,
            vec!["test-source".into()],
            None,
            0.5,
        )
        .unwrap();
        assert!(mgr.attach_episode_provenance(&ep, wrong).is_err());
        assert!(mgr.provenance_index().is_empty());
    }

    #[test]
    fn component_attachment_publishes_consistent_composite_view() {
        let mut mgr = EpisodicPersistenceManager::new(None);
        let ep = episode(9);
        let record = component_record(&ep);
        let expected = record.episode.clone();
        mgr.attach_episode_component_provenance(&ep, record).unwrap();
        assert_eq!(mgr.component_provenance_index().len(), 1);
        assert_eq!(mgr.effective_episode_provenance(&ep), expected);
    }

    #[test]
    fn composite_conflict_rejects_before_component_index_mutation() {
        let mut mgr = EpisodicPersistenceManager::new(None);
        let ep = episode(10);
        let imported = ProvenanceEnvelope::new(
            crate::memory::episode_subject_sha256(&ep),
            RealityDomain::Imported,
            vec!["external".into()],
            None,
            0.5,
        )
        .unwrap();
        mgr.attach_episode_provenance(&ep, imported).unwrap();
        let result = mgr.attach_episode_component_provenance(&ep, component_record(&ep));
        assert!(matches!(result, Err(EpisodicComponentBindingError::CompositeConflict { .. })));
        assert!(mgr.component_provenance_index().is_empty());
        assert_eq!(mgr.provenance_index().len(), 1);
    }

    #[test]
    fn replacing_replay_resets_all_provenance_sidecars() {
        let mut mgr = EpisodicPersistenceManager::new(None);
        let ep = episode(11);
        mgr.attach_episode_component_provenance(&ep, component_record(&ep))
            .unwrap();
        assert_eq!(mgr.provenance_index().len(), 1);
        assert_eq!(mgr.component_provenance_index().len(), 1);
        mgr.replace_replay(None);
        assert!(mgr.provenance_index().is_empty());
        assert!(mgr.component_provenance_index().is_empty());
    }

    #[test]
    fn shadow_audit_is_behavior_neutral_for_legacy_unknown_memory() {
        let ep = episode(12);
        let query = ep.input.values.clone();
        let mut replay = crate::memory::EpisodicMemory::new(
            crate::memory::EpisodicReplayConfig::broad_capture(),
        );
        assert!(replay.store_if_significant(ep));
        let before_len = replay.len();
        let mgr = EpisodicPersistenceManager::new(Some(replay));
        let audit = mgr.shadow_audit_input_recall(&query, 1).unwrap();
        assert_eq!(audit.raw_returned, 1);
        assert_eq!(audit.grounded_history.would_return, 0);
        assert_eq!(audit.grounded_history.excluded_unknown, 1);
        assert!(audit.grounded_history.would_change_selection);
        assert_eq!(mgr.replay.as_ref().unwrap().len(), before_len);
        assert!(mgr.provenance_index().is_empty());
        assert!(mgr.component_provenance_index().is_empty());
    }

    #[test]
    fn shadow_audit_returns_none_when_replay_is_disabled() {
        let mgr = EpisodicPersistenceManager::new(None);
        assert!(mgr.shadow_audit_input_recall(&[0.0, 1.0], 3).is_none());
        assert!(mgr.shadow_audit_embedding_recall(&[0.0, 1.0], 3).is_none());
    }

    #[test]
    fn test_attach_sqlite_db_initializes_persistence() {
        let mut mgr = EpisodicPersistenceManager::new(None);
        let path = std::env::temp_dir().join(format!(
            "symthaea_episodic_persistence_{}.sqlite",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        mgr.attach_sqlite_db(&path).unwrap();
        assert!(mgr.db.is_some());
        assert!(mgr.storage_runtime.is_some());
        assert!(mgr.storage_worker.is_some());
        let _ = std::fs::remove_file(path);
    }
}
