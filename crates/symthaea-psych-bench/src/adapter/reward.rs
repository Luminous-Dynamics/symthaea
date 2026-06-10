// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reward/bandit outcome encoding.
//!
//! Encodes arm identities scaled by reward magnitude for bandit tasks
//! and instrumental learning experiments.

use super::StimulusAdapter;
use symthaea_core::hdc::ContinuousHV;

/// A reward outcome from a bandit arm or instrumental trial.
#[derive(Debug, Clone, Copy)]
pub struct RewardOutcome {
    /// Which arm/action was chosen (0-indexed).
    pub arm: u32,
    /// Reward magnitude (can be negative for losses).
    pub reward: f32,
    /// Trial number for temporal context.
    pub trial: u32,
}

impl RewardOutcome {
    pub fn new(arm: u32, reward: f32, trial: u32) -> Self {
        Self { arm, reward, trial }
    }
}

/// Encodes reward outcomes as scaled arm vectors with temporal tags.
///
/// `arm_hv * reward_magnitude` bundled with `trial_tag.permute(trial % dim)`.
pub struct RewardAdapter {
    pub arm_seed_offset: u64,
    pub trial_seed_offset: u64,
}

impl Default for RewardAdapter {
    fn default() -> Self {
        Self {
            arm_seed_offset: 60_000,
            trial_seed_offset: 70_000,
        }
    }
}

impl StimulusAdapter for RewardAdapter {
    type Stimulus = RewardOutcome;

    fn encode(&self, outcome: &RewardOutcome, dim: usize) -> ContinuousHV {
        let arm_hv = ContinuousHV::random(dim, self.arm_seed_offset + outcome.arm as u64);
        let scaled = arm_hv.scale(outcome.reward);
        let trial_tag =
            ContinuousHV::random(dim, self.trial_seed_offset).permute(outcome.trial as usize % dim);
        ContinuousHV::bundle_owned(&[scaled, trial_tag.scale(0.1)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_outcome_deterministic() {
        let adapter = RewardAdapter::default();
        let o = RewardOutcome::new(0, 1.0, 5);
        let a = adapter.encode(&o, 512);
        let b = adapter.encode(&o, 512);
        assert!((a.similarity(&b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_different_arms_dissimilar() {
        let adapter = RewardAdapter::default();
        let a = adapter.encode(&RewardOutcome::new(0, 1.0, 0), 512);
        let b = adapter.encode(&RewardOutcome::new(1, 1.0, 0), 512);
        assert!(a.similarity(&b) < 0.5);
    }

    #[test]
    fn test_reward_polarity_reflected() {
        let adapter = RewardAdapter::default();
        let positive = adapter.encode(&RewardOutcome::new(0, 1.0, 0), 512);
        let negative = adapter.encode(&RewardOutcome::new(0, -1.0, 0), 512);
        // Opposite reward polarity should yield low or negative similarity
        assert!(positive.similarity(&negative) < 0.3);
    }
}
