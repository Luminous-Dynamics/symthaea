// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Approved assurance adapter from validated restart materialization into canonical episodic memory.
//!
//! `CanonicalEpisodicImportBatch` can only be minted by the persisted-envelope + restart-activation
//! theorem. This adapter is therefore the intended production bridge into the mechanism-only
//! exact-UUID constructor in `symthaea-memory`.

use symthaea_memory::episodic_replay::{
    EpisodicMemory, EpisodicReplayConfig,
    persisted_import::PersistedEpisodicImportError,
};

use crate::persisted_episode_envelope::CanonicalEpisodicImportBatch;

/// Reconstruct the canonical active replay heap from an already validated active-only batch.
///
/// Withheld/reconciliation-required occurrences are not materialized as `Episode` values by the
/// batch and therefore cannot enter this call accidentally. The exact occurrence UUID carried by
/// every active episode is preserved by the memory-layer mechanism.
pub fn reconstruct_canonical_episodic_memory(
    config: EpisodicReplayConfig,
    recovery_cycle: u64,
    batch: &CanonicalEpisodicImportBatch,
) -> Result<EpisodicMemory, PersistedEpisodicImportError> {
    let active = batch
        .active()
        .iter()
        .cloned()
        .map(|entry| entry.into_episode())
        .collect();
    EpisodicMemory::from_validated_persisted_active_state(config, recovery_cycle, active)
}
