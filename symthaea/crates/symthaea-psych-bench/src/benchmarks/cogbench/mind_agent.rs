// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CogBench agent backed by Symthaea's `ContinuousMind`.
//!
//! Provides a decision-making agent that uses the full cognitive pipeline
//! (working memory, consciousness monitoring, dream consolidation) instead
//! of FEP active inference. Action selection is similarity-driven: the
//! agent computes cosine similarity between its `current_thought` and
//! each arm's prototype HV, applies softmax, and samples.
//!
//! This tests a fundamentally different hypothesis about decision-making:
//! **WM-based** (emergent value from memory persistence) vs **FEP-based**
//! (explicit free energy minimization).
//!
//! Enabled via `--features symthaea-backend`.

use symthaea::mind::{ContinuousMind, MindConfig};
use symthaea_core::hdc::ContinuousHV;

use crate::adapter::StimulusAdapter;
use crate::adapter::reward::{RewardAdapter, RewardOutcome};

/// A CogBench decision agent backed by `ContinuousMind`.
///
/// Each arm has a deterministic prototype HV. Action selection uses
/// cosine similarity between `current_thought` and arm prototypes.
/// Reward feedback is perceived as scaled arm HVs, allowing rewarded
/// arms to accumulate in working memory and drive exploitation.
pub struct CogBenchMindAgent {
    mind: ContinuousMind,
    arm_prototypes: Vec<ContinuousHV>,
    reward_adapter: RewardAdapter,
    temperature: f64,
    trial_counter: u32,
}

/// Result of action selection, mirroring the FEP agent's interface.
pub struct MindActionResult {
    /// Softmax probability for each arm.
    pub action_probabilities: Vec<f64>,
}

impl CogBenchMindAgent {
    /// Create a new mind-backed agent.
    ///
    /// - `num_arms`: number of actions/arms
    /// - `dim`: HDC dimension
    /// - `wm_capacity`: working memory capacity
    /// - `temperature`: softmax temperature for action selection
    /// - `seed`: deterministic seed for arm prototype generation
    pub fn new(
        num_arms: usize,
        dim: usize,
        wm_capacity: usize,
        temperature: f64,
        seed: u64,
    ) -> Self {
        let mind_config = MindConfig {
            dimension: dim,
            working_memory_capacity: wm_capacity,
            learning_enabled: true,
            ..Default::default()
        };
        let mut mind = ContinuousMind::new(mind_config);
        mind.activate();

        // Create deterministic arm prototypes
        let arm_prototypes: Vec<ContinuousHV> = (0..num_arms)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(1000 + i as u64)))
            .collect();

        Self {
            mind,
            arm_prototypes,
            reward_adapter: RewardAdapter::default(),
            temperature,
            trial_counter: 0,
        }
    }

    /// Select an action via similarity-based softmax over arm prototypes.
    ///
    /// Computes `softmax(sim(current_thought, arm_i) / temperature)` for
    /// each arm. When WM is empty (first tick), returns uniform distribution.
    pub fn select_action(&self) -> MindActionResult {
        let thought = &self.mind.state().current_thought;
        let wm = self.mind.working_memory();

        if wm.is_empty() {
            // No memory yet — uniform random
            let n = self.arm_prototypes.len();
            return MindActionResult {
                action_probabilities: vec![1.0 / n as f64; n],
            };
        }

        // Compute similarity of current thought to each arm prototype
        let sims: Vec<f64> = self
            .arm_prototypes
            .iter()
            .map(|proto| {
                // Use both current thought AND max WM similarity to this arm
                let thought_sim = thought.similarity(proto) as f64;
                let wm_sim = wm
                    .iter()
                    .map(|item| item.similarity(proto) as f32)
                    .fold(0.0f32, f32::max) as f64;
                // Blend: thought captures recency, WM captures accumulated experience
                0.4 * thought_sim + 0.6 * wm_sim
            })
            .collect();

        // Softmax with temperature
        let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = sims
            .iter()
            .map(|s| ((s - max_sim) / self.temperature).exp())
            .collect();
        let sum: f64 = exps.iter().sum();

        let probs = if sum > 0.0 {
            exps.iter().map(|e| e / sum).collect()
        } else {
            let n = self.arm_prototypes.len();
            vec![1.0 / n as f64; n]
        };

        MindActionResult {
            action_probabilities: probs,
        }
    }

    /// Perceive a reward outcome, encoding it as a scaled arm HV.
    ///
    /// High rewards produce HVs similar to the arm prototype (reinforcing it
    /// in WM). Low/negative rewards produce dissimilar signals.
    pub fn perceive_reward(&mut self, arm: usize, reward: f64) {
        let outcome = RewardOutcome::new(arm as u32, reward as f32, self.trial_counter);
        let hv = self
            .reward_adapter
            .encode(&outcome, self.mind.config().dimension);
        self.mind.perceive(hv);
        let _ = self.mind.tick();
        self.trial_counter += 1;
    }

    /// Get the current consciousness level (confidence proxy).
    ///
    /// Replaces FEP's `perceptual_precision` for metacognitive sensitivity.
    pub fn consciousness_level(&self) -> f64 {
        self.mind.state().consciousness_level
    }

    /// Get working memory utilization [0, 1].
    pub fn memory_utilization(&self) -> f32 {
        self.mind.state().memory_utilization
    }

    /// Get meta-awareness level.
    pub fn meta_awareness(&self) -> f64 {
        self.mind.snapshot().meta_awareness
    }
}
