// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-identity import mechanism for already-validated persisted episodic state.
//!
//! This module is a child of `episodic_replay` so it can reconstruct the private canonical heap
//! without widening normal insertion internals. It contains mechanism only: callers in higher
//! assurance layers are responsible for proving that an episode is authorized to become active.
//!
//! Ordinary cognition insertion remains deliberately separate and continues to mint a fresh
//! `EpisodeInstanceId` regardless of any caller-supplied/deserialized identity.

use super::*;

/// Mechanism-level failure while reconstructing or restoring already-validated persisted state.
///
/// This is not an authorization or consent decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistedEpisodicImportError {
    CapacityExceeded { active: usize, capacity: usize },
    MissingInstanceId { index: usize },
    DuplicateInstanceId(EpisodeInstanceId),
    ActiveInstanceCollision(EpisodeInstanceId),
    QuarantinedInstanceCollision(EpisodeInstanceId),
    RecoveryCycleBeforeEpisode {
        instance_id: EpisodeInstanceId,
        episode_cycle: u64,
        recovery_cycle: u64,
    },
    BelowConfiguredPsiThreshold {
        instance_id: EpisodeInstanceId,
    },
    NonFiniteEpisodeState {
        instance_id: EpisodeInstanceId,
        field: &'static str,
    },
    NonFinitePriority(EpisodeInstanceId),
    NonFiniteConfig(&'static str),
}

impl fmt::Display for PersistedEpisodicImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityExceeded { active, capacity } => write!(
                f,
                "persisted active episodic state exceeds configured capacity: active={active}, capacity={capacity}"
            ),
            Self::MissingInstanceId { index } => {
                write!(f, "persisted active episode at index {index} has no instance ID")
            }
            Self::DuplicateInstanceId(id) => {
                write!(f, "duplicate persisted active episodic instance ID: {id}")
            }
            Self::ActiveInstanceCollision(id) => {
                write!(f, "persisted restore collides with active episodic instance: {id}")
            }
            Self::QuarantinedInstanceCollision(id) => write!(
                f,
                "persisted restore collides with in-memory quarantined episodic instance: {id}"
            ),
            Self::RecoveryCycleBeforeEpisode {
                instance_id,
                episode_cycle,
                recovery_cycle,
            } => write!(
                f,
                "recovery cycle predates persisted episode {instance_id}: episode={episode_cycle}, recovery={recovery_cycle}"
            ),
            Self::BelowConfiguredPsiThreshold { instance_id } => write!(
                f,
                "persisted episode {instance_id} is below the configured active Psi threshold"
            ),
            Self::NonFiniteEpisodeState { instance_id, field } => write!(
                f,
                "persisted episode {instance_id} contains non-finite numeric state in `{field}`"
            ),
            Self::NonFinitePriority(id) => {
                write!(f, "persisted episode {id} produces non-finite replay priority")
            }
            Self::NonFiniteConfig(field) => {
                write!(f, "episodic replay config contains non-finite `{field}`")
            }
        }
    }
}

impl std::error::Error for PersistedEpisodicImportError {}

impl EpisodicMemory {
    /// Construct canonical active replay state from an already-validated active-only restart batch.
    ///
    /// This is the only canonical restart mechanism that preserves caller-supplied occurrence IDs.
    /// It deliberately accepts **active episodes only**. Quarantined/inactive episode payloads must
    /// remain outside live memory until a separately governed point restore calls
    /// [`Self::restore_validated_persisted_occurrence`].
    ///
    /// The entire batch is checked before a live value is returned:
    /// - configured numeric parameters are finite;
    /// - active count fits capacity exactly (no eviction/drop policy is applied);
    /// - every episode has one exact UUID and UUIDs are unique;
    /// - all persisted numeric state and the recomputed priority are finite;
    /// - recovery time does not predate any imported episode;
    /// - each episode remains compatible with the supplied active Psi threshold.
    ///
    /// Process-local counters such as replay sessions, demand triggers, evictions and new-store
    /// count restart at zero. Episode-local replay/retrieval/consolidation state is preserved.
    pub fn from_validated_persisted_active_state(
        config: EpisodicReplayConfig,
        recovery_cycle: u64,
        active: Vec<Episode>,
    ) -> Result<Self, PersistedEpisodicImportError> {
        validate_import_config(&config)?;
        if active.len() > config.capacity {
            return Err(PersistedEpisodicImportError::CapacityExceeded {
                active: active.len(),
                capacity: config.capacity,
            });
        }

        let mut seen = HashSet::with_capacity(active.len());
        let mut heap = BinaryHeap::with_capacity(active.len());
        for (index, episode) in active.into_iter().enumerate() {
            let instance_id = episode
                .instance_id
                .ok_or(PersistedEpisodicImportError::MissingInstanceId { index })?;
            if !seen.insert(instance_id) {
                return Err(PersistedEpisodicImportError::DuplicateInstanceId(
                    instance_id,
                ));
            }
            validate_import_episode(&episode, instance_id, recovery_cycle, &config)?;
            let score = episode.priority_score(recovery_cycle, config.recency_weight);
            if !score.is_finite() {
                return Err(PersistedEpisodicImportError::NonFinitePriority(
                    instance_id,
                ));
            }
            heap.push(PrioritizedEpisode { episode, score });
        }

        let mut memory = Self {
            config,
            episodes: heap,
            quarantined: HashMap::new(),
            current_cycle: recovery_cycle,
            cycles_since_replay: 0,
            total_stored: 0,
            total_evicted: 0,
            total_replay_steps: 0,
            average_psi: 0.0,
            min_psi_in_buffer: f64::MAX,
            sum_replay_loss: 0.0,
            demand_replay_triggered: false,
            demand_replay_count: 0,
        };
        memory.recompute_active_psi_stats();
        Ok(memory)
    }

    /// Restore one exact persisted occurrence after higher-layer governed escrow validation.
    ///
    /// The caller must have already performed the point lookup and authorization checks. This
    /// mechanism validates the episode again and preserves its exact persisted UUID. It never
    /// accepts a mixed/bulk quarantine set and never increments `total_stored`.
    ///
    /// Validation completes before mutation, so every error leaves the live heap unchanged.
    pub fn restore_validated_persisted_occurrence(
        &mut self,
        episode: Episode,
    ) -> Result<EpisodeInstanceId, PersistedEpisodicImportError> {
        validate_import_config(&self.config)?;
        let instance_id = episode
            .instance_id
            .ok_or(PersistedEpisodicImportError::MissingInstanceId { index: 0 })?;

        if self
            .episodes
            .iter()
            .any(|value| value.episode.instance_id == Some(instance_id))
        {
            return Err(PersistedEpisodicImportError::ActiveInstanceCollision(
                instance_id,
            ));
        }
        if self.quarantined.contains_key(&instance_id) {
            return Err(PersistedEpisodicImportError::QuarantinedInstanceCollision(
                instance_id,
            ));
        }
        if self.episodes.len() >= self.config.capacity {
            return Err(PersistedEpisodicImportError::CapacityExceeded {
                active: self.episodes.len().saturating_add(1),
                capacity: self.config.capacity,
            });
        }

        validate_import_episode(&episode, instance_id, self.current_cycle, &self.config)?;
        let score = episode.priority_score(self.current_cycle, self.config.recency_weight);
        if !score.is_finite() {
            return Err(PersistedEpisodicImportError::NonFinitePriority(
                instance_id,
            ));
        }

        self.episodes.push(PrioritizedEpisode { episode, score });
        self.recompute_active_psi_stats();
        Ok(instance_id)
    }
}

fn validate_import_config(
    config: &EpisodicReplayConfig,
) -> Result<(), PersistedEpisodicImportError> {
    for (field, finite) in [
        ("psi_threshold", config.psi_threshold.is_finite()),
        ("recency_weight", config.recency_weight.is_finite()),
        (
            "replay_learning_rate_multiplier",
            config.replay_learning_rate_multiplier.is_finite(),
        ),
        ("replay_dt", config.replay_dt.is_finite()),
        (
            "sampling_temperature",
            config.sampling_temperature.is_finite(),
        ),
    ] {
        if !finite {
            return Err(PersistedEpisodicImportError::NonFiniteConfig(field));
        }
    }
    Ok(())
}

fn validate_import_episode(
    episode: &Episode,
    instance_id: EpisodeInstanceId,
    recovery_cycle: u64,
    config: &EpisodicReplayConfig,
) -> Result<(), PersistedEpisodicImportError> {
    if episode.instance_id != Some(instance_id) {
        return Err(PersistedEpisodicImportError::MissingInstanceId { index: 0 });
    }
    if episode.timestamp > recovery_cycle {
        return Err(PersistedEpisodicImportError::RecoveryCycleBeforeEpisode {
            instance_id,
            episode_cycle: episode.timestamp,
            recovery_cycle,
        });
    }
    if episode.psi < config.psi_threshold {
        return Err(PersistedEpisodicImportError::BelowConfiguredPsiThreshold {
            instance_id,
        });
    }

    if !episode.psi.is_finite() {
        return non_finite(instance_id, "psi");
    }
    for (field, value) in [
        ("prediction_error", episode.prediction_error),
        ("valence", episode.valence),
        ("coherence", episode.coherence),
        ("dopamine_at_encoding", episode.dopamine_at_encoding),
    ] {
        if value.is_some_and(|number| !number.is_finite()) {
            return non_finite(instance_id, field);
        }
    }
    if !episode.consolidation_strength.is_finite() {
        return non_finite(instance_id, "consolidation_strength");
    }
    if episode
        .input
        .values
        .iter()
        .chain(episode.output.values.iter())
        .any(|number| !number.is_finite())
    {
        return non_finite(instance_id, "input_or_output");
    }
    if episode
        .semantic_embedding
        .as_ref()
        .is_some_and(|values| values.iter().any(|number| !number.is_finite()))
    {
        return non_finite(instance_id, "semantic_embedding");
    }
    if episode
        .bath_state_at_encoding
        .is_some_and(|values| values.iter().any(|number| !number.is_finite()))
    {
        return non_finite(instance_id, "bath_state_at_encoding");
    }
    Ok(())
}

fn non_finite(
    instance_id: EpisodeInstanceId,
    field: &'static str,
) -> Result<(), PersistedEpisodicImportError> {
    Err(PersistedEpisodicImportError::NonFiniteEpisodeState {
        instance_id,
        field,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn episode(psi: f64, timestamp: u64) -> Episode {
        Episode::new(
            ContinuousHV::from_vec(vec![timestamp as f32; 8]),
            ContinuousHV::from_vec(vec![psi as f32; 8]),
            psi,
            timestamp,
        )
    }

    fn two_persisted_duplicate_content() -> (Episode, Episode, EpisodeInstanceId, EpisodeInstanceId) {
        let config = EpisodicReplayConfig {
            psi_threshold: 0.0,
            capacity: 8,
            ..EpisodicReplayConfig::default()
        };
        let mut original = EpisodicMemory::new(config);
        let template = episode(0.8, 10);
        let a = original.store_if_significant_with_id(template.clone()).unwrap();
        let b = original.store_if_significant_with_id(template).unwrap();
        let mut values = original.get_top_episode_instances(8);
        values.sort_by_key(|(id, _)| *id);
        let a_episode = values.iter().find(|(id, _)| *id == a).unwrap().1.clone();
        let b_episode = values.iter().find(|(id, _)| *id == b).unwrap().1.clone();
        (a_episode, b_episode, a, b)
    }

    #[test]
    fn restart_import_preserves_exact_ids_and_withholds_absent_occurrence() {
        let (a_episode, b_episode, a, b) = two_persisted_duplicate_content();
        let config = EpisodicReplayConfig {
            psi_threshold: 0.0,
            capacity: 8,
            ..EpisodicReplayConfig::default()
        };

        // A is withheld by the higher activation theorem: canonical restart sees B only.
        let mut restarted = EpisodicMemory::from_validated_persisted_active_state(
            config,
            20,
            vec![b_episode.clone()],
        )
        .unwrap();
        let ids: HashSet<_> = restarted
            .get_top_episode_instances(8)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(ids, HashSet::from([b]));
        assert!(!ids.contains(&a));
        assert_eq!(restarted.quarantined_len(), 0);

        // Governed exact point-restore of A preserves its old UUID.
        let restored = restarted
            .restore_validated_persisted_occurrence(a_episode.clone())
            .unwrap();
        assert_eq!(restored, a);
        let ids: HashSet<_> = restarted
            .get_top_episode_instances(8)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(ids, HashSet::from([a, b]));

        // A second process death/restart can reconstruct A+B with the same identities.
        let restarted_again = EpisodicMemory::from_validated_persisted_active_state(
            EpisodicReplayConfig {
                psi_threshold: 0.0,
                capacity: 8,
                ..EpisodicReplayConfig::default()
            },
            30,
            vec![a_episode, b_episode],
        )
        .unwrap();
        let ids: HashSet<_> = restarted_again
            .get_top_episode_instances(8)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(ids, HashSet::from([a, b]));
    }

    #[test]
    fn ordinary_insertion_still_remints_caller_supplied_identity() {
        let (a_episode, _, a, _) = two_persisted_duplicate_content();
        let mut ordinary = EpisodicMemory::new(EpisodicReplayConfig {
            psi_threshold: 0.0,
            ..EpisodicReplayConfig::default()
        });
        assert_eq!(a_episode.instance_id, Some(a));
        let new_id = ordinary.store_if_significant_with_id(a_episode).unwrap();
        assert_ne!(new_id, a);
    }

    #[test]
    fn duplicate_uuid_fails_batch() {
        let (a_episode, _, a, _) = two_persisted_duplicate_content();
        let result = EpisodicMemory::from_validated_persisted_active_state(
            EpisodicReplayConfig {
                psi_threshold: 0.0,
                capacity: 8,
                ..EpisodicReplayConfig::default()
            },
            20,
            vec![a_episode.clone(), a_episode],
        );
        assert_eq!(
            result.unwrap_err(),
            PersistedEpisodicImportError::DuplicateInstanceId(a)
        );
    }

    #[test]
    fn missing_uuid_fails_batch() {
        let result = EpisodicMemory::from_validated_persisted_active_state(
            EpisodicReplayConfig {
                psi_threshold: 0.0,
                capacity: 8,
                ..EpisodicReplayConfig::default()
            },
            20,
            vec![episode(0.8, 10)],
        );
        assert_eq!(
            result.unwrap_err(),
            PersistedEpisodicImportError::MissingInstanceId { index: 0 }
        );
    }

    #[test]
    fn capacity_disagreement_fails_instead_of_evicting() {
        let (a_episode, b_episode, _, _) = two_persisted_duplicate_content();
        let result = EpisodicMemory::from_validated_persisted_active_state(
            EpisodicReplayConfig {
                psi_threshold: 0.0,
                capacity: 1,
                ..EpisodicReplayConfig::default()
            },
            20,
            vec![a_episode, b_episode],
        );
        assert_eq!(
            result.unwrap_err(),
            PersistedEpisodicImportError::CapacityExceeded {
                active: 2,
                capacity: 1
            }
        );
    }

    #[test]
    fn non_finite_episode_state_fails() {
        let (mut a_episode, _, a, _) = two_persisted_duplicate_content();
        a_episode.consolidation_strength = f64::NAN;
        let result = EpisodicMemory::from_validated_persisted_active_state(
            EpisodicReplayConfig {
                psi_threshold: 0.0,
                capacity: 8,
                ..EpisodicReplayConfig::default()
            },
            20,
            vec![a_episode],
        );
        assert_eq!(
            result.unwrap_err(),
            PersistedEpisodicImportError::NonFiniteEpisodeState {
                instance_id: a,
                field: "consolidation_strength"
            }
        );
    }

    #[test]
    fn recovery_cycle_regression_fails() {
        let (a_episode, _, a, _) = two_persisted_duplicate_content();
        let result = EpisodicMemory::from_validated_persisted_active_state(
            EpisodicReplayConfig {
                psi_threshold: 0.0,
                capacity: 8,
                ..EpisodicReplayConfig::default()
            },
            9,
            vec![a_episode],
        );
        assert_eq!(
            result.unwrap_err(),
            PersistedEpisodicImportError::RecoveryCycleBeforeEpisode {
                instance_id: a,
                episode_cycle: 10,
                recovery_cycle: 9
            }
        );
    }

    #[test]
    fn failed_point_restore_is_atomic() {
        let (_, b_episode, _, b) = two_persisted_duplicate_content();
        let mut restarted = EpisodicMemory::from_validated_persisted_active_state(
            EpisodicReplayConfig {
                psi_threshold: 0.0,
                capacity: 1,
                ..EpisodicReplayConfig::default()
            },
            20,
            vec![b_episode.clone()],
        )
        .unwrap();
        let before = restarted.get_top_episode_instances(8);
        let err = restarted
            .restore_validated_persisted_occurrence(b_episode)
            .unwrap_err();
        assert_eq!(err, PersistedEpisodicImportError::ActiveInstanceCollision(b));
        assert_eq!(restarted.get_top_episode_instances(8).len(), before.len());
        assert_eq!(restarted.get_top_episode_instances(8)[0].0, b);
    }

    #[test]
    fn imported_occurrences_are_not_counted_as_new_stores() {
        let (a_episode, _, _, _) = two_persisted_duplicate_content();
        let restarted = EpisodicMemory::from_validated_persisted_active_state(
            EpisodicReplayConfig {
                psi_threshold: 0.0,
                capacity: 8,
                ..EpisodicReplayConfig::default()
            },
            20,
            vec![a_episode],
        )
        .unwrap();
        assert_eq!(restarted.stats().total_stored, 0);
        assert_eq!(restarted.stats().current_count, 1);
        assert_eq!(restarted.stats().total_evicted, 0);
    }
}
