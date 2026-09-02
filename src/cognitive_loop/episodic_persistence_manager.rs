// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Episodic Persistence Manager — episodic replay + memory database.
//!
//! Groups episodic memory replay, persistent SQLite database,
//! flush guard, reasoning context, and epistemic provenance sidecar into a
//! single data-holder manager.
//!
//! The provenance sidecar is deliberately separate from the historical
//! `Episode` serialization format. Existing episodic stores therefore remain
//! readable and resolve to explicit `Unknown` provenance until a subject-bound
//! provenance envelope is attached.
//!
//! Science: Tulving (1972) — episodic memory; Stickgold (2005) — replay consolidation.

/// Consolidated episodic persistence subsystem.
///
/// Holds episodic replay, memory database, flush guard, reasoning context, and
/// provenance state that were previously scattered across `CognitiveLoopService`.
pub struct EpisodicPersistenceManager {
    /// Episodic memory replay for high-Phi moment consolidation.
    /// When enabled via `config.memory_graduation`, stores high-consciousness episodes
    /// and periodically replays them to reinforce important patterns.
    pub(crate) replay: Option<crate::memory::episodic_replay::EpisodicMemory>,

    /// Epistemic provenance sidecar keyed by immutable episode subject digest.
    ///
    /// This does not alter `Episode` serialization. An episode without an entry is
    /// explicitly `Unknown`, never silently grounded.
    pub(crate) provenance: crate::memory::EpisodicProvenanceIndex,

    /// Persistent memory database for cross-session episode storage.
    /// Created when `config.memory_db_path` is `Some`. Episodes are periodically
    /// flushed (every 199 cycles) via the storage runtime or a background thread.
    pub(crate) db: Option<std::sync::Arc<crate::databases::SqliteMemory>>,

    /// Optional write-behind runtime for non-blocking durable memory writes.
    pub(crate) storage_runtime: Option<crate::databases::storage_runtime::StorageRuntimeHandle>,

    /// Worker backing the storage runtime when it was spawned from this manager.
    storage_worker: Option<std::thread::JoinHandle<()>>,

    /// Guard to prevent overlapping memory flushes. When true, a flush is in progress.
    pub(crate) flush_in_progress: std::sync::Arc<std::sync::atomic::AtomicBool>,

    /// Last assembled reasoning context from the knowledge engine.
    /// Contains grounded facts, causal chains, and epistemic state.
    /// Populated after knowledge engine processes each cycle's input.
    pub(crate) last_reasoning_context: Option<crate::knowledge::ReasoningContext>,
}

impl EpisodicPersistenceManager {
    /// Create a new EpisodicPersistenceManager.
    ///
    /// `replay` is the optional episodic memory (built from config). Existing
    /// episodes inside it begin with no sidecar bindings and therefore resolve
    /// to explicit `Unknown` provenance.
    pub fn new(replay: Option<crate::memory::episodic_replay::EpisodicMemory>) -> Self {
        Self {
            replay,
            provenance: crate::memory::EpisodicProvenanceIndex::default(),
            db: None,
            storage_runtime: None,
            storage_worker: None,
            flush_in_progress: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            last_reasoning_context: None,
        }
    }

    /// Attach subject-bound provenance to an exact episodic record.
    ///
    /// The sidecar rejects digest mismatches and conflicting rewrites. This method
    /// intentionally does not infer provenance from episode contents.
    pub fn attach_episode_provenance(
        &mut self,
        episode: &crate::memory::Episode,
        envelope: symthaea_epistemic_types::ProvenanceEnvelope,
    ) -> Result<(), crate::memory::EpisodicProvenanceError> {
        self.provenance.attach(episode, envelope)
    }

    /// Return explicit provenance for an episode.
    ///
    /// Legacy/unannotated episodes resolve to `RealityDomain::Unknown` with zero
    /// confidence. This is the fail-closed compatibility behavior.
    pub fn effective_episode_provenance(
        &self,
        episode: &crate::memory::Episode,
    ) -> symthaea_epistemic_types::ProvenanceEnvelope {
        self.provenance.effective_provenance(episode)
    }

    /// Expose the provenance index read-only for persistence/audit code.
    pub fn provenance_index(&self) -> &crate::memory::EpisodicProvenanceIndex {
        &self.provenance
    }

    /// Replace the replay store and deliberately reset provenance bindings.
    ///
    /// This prevents stale sidecar entries from surviving a wholesale episodic
    /// store replacement. Callers that restore a persisted sidecar should use a
    /// future paired restore API rather than this reset path.
    pub fn replace_replay(
        &mut self,
        replay: Option<crate::memory::episodic_replay::EpisodicMemory>,
    ) {
        self.replay = replay;
        self.provenance = crate::memory::EpisodicProvenanceIndex::default();
    }

    /// Attach a persistent SQLite memory database for periodic episode flushes.
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

    /// Attach a write-behind runtime for periodic episode flushes.
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

impl Drop for EpisodicPersistenceManager {
    fn drop(&mut self) {
        self.stop_owned_storage_worker();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_epistemic_types::{ProvenanceEnvelope, RealityDomain};

    fn episode(seed: u64) -> crate::memory::Episode {
        crate::memory::Episode::new(
            ContinuousHV::random(64, seed),
            ContinuousHV::random(64, seed + 1),
            0.8,
            seed,
        )
    }

    #[test]
    fn test_episodic_persistence_manager_new() {
        let mgr = EpisodicPersistenceManager::new(None);
        assert!(mgr.replay.is_none());
        assert!(mgr.provenance_index().is_empty());
        assert!(mgr.db.is_none());
        assert!(mgr.storage_runtime.is_none());
        assert!(mgr.storage_worker.is_none());
        assert!(
            !mgr.flush_in_progress
                .load(std::sync::atomic::Ordering::Relaxed)
        );
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
    fn replacing_replay_resets_provenance_sidecar() {
        let mut mgr = EpisodicPersistenceManager::new(None);
        let ep = episode(9);
        let subject = crate::memory::episode_subject_sha256(&ep);
        let imported = ProvenanceEnvelope::new(
            subject,
            RealityDomain::Imported,
            vec!["test-source".into()],
            None,
            0.6,
        )
        .unwrap();
        mgr.attach_episode_provenance(&ep, imported).unwrap();
        assert_eq!(mgr.provenance_index().len(), 1);

        mgr.replace_replay(None);
        assert!(mgr.provenance_index().is_empty());
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