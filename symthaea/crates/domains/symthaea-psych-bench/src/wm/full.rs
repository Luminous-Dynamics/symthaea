// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Full working memory backend wrapping Symthaea's `ContinuousMind`.
//!
//! Provides the same API as the lightweight backend but delegates to the
//! real cognitive pipeline: FIFO eviction, dream consolidation (merges
//! similar items > 0.8 similarity), social coherence, and the full
//! tick pipeline.
//!
//! Enabled via `--features symthaea-backend`.

use symthaea::mind::{ContinuousMind, MindConfig};
use symthaea_core::hdc::ContinuousHV;

/// Default activation decay rate per tick.
const DEFAULT_DECAY_RATE: f32 = 0.95;

/// Configuration for the full working memory backend.
#[derive(Debug, Clone)]
pub struct WmConfig {
    /// HDC dimension.
    pub dimension: usize,
    /// Maximum items in working memory.
    pub capacity: usize,
    /// Activation decay rate per tick (default 0.95).
    /// Used for `activation_weighted_similarity()` computation.
    pub decay_rate: f32,
}

impl Default for WmConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            capacity: 7,
            decay_rate: DEFAULT_DECAY_RATE,
        }
    }
}

/// Full working memory backed by Symthaea's `ContinuousMind`.
///
/// Delegates `perceive()` and `tick()` to the real cognitive pipeline,
/// which includes dream consolidation and optional social coherence.
/// Activation is computed from arrival ticks using exponential decay.
pub struct WorkingMemory {
    mind: ContinuousMind,
    decay_rate: f32,
}

impl WorkingMemory {
    /// Create a new working memory backed by `ContinuousMind`.
    pub fn new(config: WmConfig) -> Self {
        let mind_config = MindConfig {
            dimension: config.dimension,
            working_memory_capacity: config.capacity,
            ..Default::default()
        };
        let mut mind = ContinuousMind::new(mind_config);
        mind.activate();
        Self {
            mind,
            decay_rate: config.decay_rate,
        }
    }

    /// Add an item to working memory via `ContinuousMind::perceive()`.
    ///
    /// Auto-ticks after perceive so items are immediately available in
    /// `contents()`, matching the lightweight backend's API contract.
    pub fn perceive(&mut self, hv: ContinuousHV) {
        self.mind.perceive(hv);
        let _ = self.mind.tick();
    }

    /// Advance one tick via `ContinuousMind::tick()`.
    ///
    /// This runs the full cognitive pipeline including:
    /// - Input processing with FIFO eviction
    /// - Dream consolidation (merges similar items > 0.8 similarity)
    /// - Social coherence processing (if enabled)
    pub fn tick(&mut self) {
        let _ = self.mind.tick();
    }

    /// Get current working memory contents.
    pub fn contents(&self) -> &[ContinuousHV] {
        self.mind.working_memory()
    }

    /// Get activation levels for each item, computed from arrival ticks.
    ///
    /// `activation = decay_rate^(current_tick - arrival_tick)`
    pub fn activations(&self) -> Vec<f32> {
        let current = self.mind.state().tick;
        self.mind
            .working_memory_ticks_slice()
            .iter()
            .map(|&arrival| {
                let age = current.saturating_sub(arrival) as f32;
                self.decay_rate.powf(age)
            })
            .collect()
    }

    /// Compute the maximum activation-weighted similarity to `probe`.
    pub fn activation_weighted_similarity(&self, probe: &ContinuousHV) -> f32 {
        let activations = self.activations();
        self.mind
            .working_memory()
            .iter()
            .zip(&activations)
            .map(|(item, &act)| item.similarity(probe) * act)
            .fold(0.0f32, f32::max)
    }

    /// Current tick count.
    pub fn current_tick(&self) -> u64 {
        self.mind.state().tick
    }

    /// Drain evicted items since last call.
    pub fn take_evicted(&mut self) -> Vec<(ContinuousHV, u64)> {
        self.mind
            .take_evicted()
            .into_iter()
            .map(|(hv, tick, _source, _consolidated)| (hv, tick))
            .collect()
    }

    /// Number of items currently in WM.
    pub fn len(&self) -> usize {
        self.mind.working_memory().len()
    }

    /// Whether WM is empty.
    pub fn is_empty(&self) -> bool {
        self.mind.working_memory().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_wm_basic() {
        let mut wm = WorkingMemory::new(WmConfig {
            dimension: 64,
            capacity: 3,
            ..Default::default()
        });

        for i in 0..5 {
            wm.perceive(ContinuousHV::random(64, i + 1));
            wm.tick();
        }

        assert!(wm.len() <= 3);
    }

    #[test]
    fn test_full_wm_activation() {
        let mut wm = WorkingMemory::new(WmConfig {
            dimension: 64,
            capacity: 7,
            ..Default::default()
        });

        wm.perceive(ContinuousHV::random(64, 1));
        // ContinuousMind processes input during tick, so after 1 tick
        // the item is in WM. After several more ticks, activation decays.
        for _ in 0..5 {
            wm.tick();
        }

        let activations = wm.activations();
        if !activations.is_empty() {
            // After multiple ticks, activation should be below 1.0
            assert!(
                activations[0] < 1.0,
                "activation should decay: got {}",
                activations[0]
            );
            assert!(activations[0] > 0.0);
        }
    }
}
