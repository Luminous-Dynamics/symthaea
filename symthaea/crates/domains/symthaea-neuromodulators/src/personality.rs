// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::collections::VecDeque;

/// Neurochemical personality derived from receptor sensitivities.
///
/// Science: Cloninger (1987) — psychobiological model of temperament.
///   DA receptor → Novelty Seeking
///   NE receptor → Harm Avoidance (inverse)
///   5-HT receptor → Reward Dependence
///   ACh receptor → Persistence
#[derive(Debug, Clone)]
pub struct NeuromodulatorProfile {
    /// DA sensitivity → novelty seeking
    pub novelty_seeking: f32,
    /// Inverse NE sensitivity → harm avoidance (high NE sens = low harm avoidance)
    pub harm_avoidance: f32,
    /// 5-HT sensitivity → reward dependence
    pub reward_dependence: f32,
    /// ACh sensitivity → persistence
    pub persistence: f32,
}

/// Tracks personality profile drift over time for metacognitive anomaly detection.
///
/// Records `NeuromodulatorProfile` snapshots and computes the maximum
/// per-trait delta rate. Rapid drift signals destabilization (e.g. receptor
/// adaptation runaway).
#[derive(Debug, Clone)]
pub struct PersonalityDriftTracker {
    pub(crate) history: VecDeque<NeuromodulatorProfile>,
    capacity: usize,
}

impl Default for PersonalityDriftTracker {
    fn default() -> Self {
        Self::new(16)
    }
}

impl PersonalityDriftTracker {
    pub fn new(capacity: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Record a personality profile snapshot. Evicts oldest if at capacity.
    pub fn record(&mut self, profile: &NeuromodulatorProfile) {
        if self.history.len() >= self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(profile.clone());
    }

    /// Maximum absolute trait delta per snapshot across all 4 traits.
    /// Returns 0.0 if fewer than 2 snapshots recorded.
    pub fn drift_rate(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let (Some(first), Some(last)) = (self.history.front(), self.history.back()) else {
            return 0.0;
        };
        let n = (self.history.len() - 1) as f32;
        let deltas = [
            (last.novelty_seeking - first.novelty_seeking).abs() / n,
            (last.harm_avoidance - first.harm_avoidance).abs() / n,
            (last.reward_dependence - first.reward_dependence).abs() / n,
            (last.persistence - first.persistence).abs() / n,
        ];
        deltas.into_iter().fold(0.0_f32, f32::max)
    }

    /// Whether drift exceeds the anomaly threshold (0.005 per snapshot).
    pub fn is_anomalous(&self) -> bool {
        self.drift_rate() > 0.005
    }
}
