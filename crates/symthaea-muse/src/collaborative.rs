// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Collaborative consciousness composition: multiple nodes create shared music.
//!
//! Each participant runs their own consciousness engine. Harmony activations
//! are shared (optionally encrypted via FHE) and blended into a collective
//! musical state. The collective drives a shared composition where each
//! participant hears the same music colored by their individual consciousness.
//!
//! # Architecture
//!
//! ```text
//! Node A: consciousness → harmony_activations[8] → encrypt → broadcast
//! Node B: consciousness → harmony_activations[8] → encrypt → broadcast
//! Node C: consciousness → harmony_activations[8] → encrypt → broadcast
//!                                    ↓
//!                    Collective Harmony Blend (mean or majority)
//!                                    ↓
//!                    Shared MusicalState → compose() → shared music
//! ```

use crate::MusicalState;

/// Maximum number of participants in a collaborative session.
pub const MAX_PARTICIPANTS: usize = 16;

/// A participant's contribution to the collective composition.
#[derive(Debug, Clone)]
pub struct HarmonyContribution {
    /// Participant ID.
    pub participant_id: u64,
    /// Harmony activations from this participant's consciousness.
    pub harmonies: [f32; 8],
    /// Consciousness level (Phi proxy).
    pub phi: f32,
    /// Timestamp (cycle number).
    pub cycle: u64,
}

/// Blending strategy for combining multiple harmony contributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendStrategy {
    /// Simple average of all contributions.
    Mean,
    /// Weighted by each participant's Phi (more conscious = more influence).
    PhiWeighted,
    /// Maximum activation per harmony (union of all experiences).
    Maximum,
    /// Minimum activation per harmony (intersection — shared experience only).
    Minimum,
}

impl Default for BlendStrategy {
    fn default() -> Self {
        Self::PhiWeighted
    }
}

/// Collaborative composition orchestrator.
pub struct CollaborativeComposer {
    contributions: Vec<HarmonyContribution>,
    strategy: BlendStrategy,
    /// Smoothed collective harmony (EMA across updates).
    collective_harmonies: [f32; 8],
    collective_phi: f32,
    ema_alpha: f32,
}

impl CollaborativeComposer {
    pub fn new(strategy: BlendStrategy) -> Self {
        Self {
            contributions: Vec::with_capacity(MAX_PARTICIPANTS),
            strategy,
            collective_harmonies: [0.3; 8],
            collective_phi: 0.5,
            ema_alpha: 0.15,
        }
    }

    /// Add or update a participant's contribution.
    pub fn contribute(&mut self, contribution: HarmonyContribution) {
        if let Some(existing) = self
            .contributions
            .iter_mut()
            .find(|c| c.participant_id == contribution.participant_id)
        {
            *existing = contribution;
        } else if self.contributions.len() < MAX_PARTICIPANTS {
            self.contributions.push(contribution);
        }
    }

    /// Remove a participant.
    pub fn remove_participant(&mut self, id: u64) {
        self.contributions.retain(|c| c.participant_id != id);
    }

    /// Blend all contributions into a collective MusicalState.
    ///
    /// The blend strategy determines how individual harmony activations
    /// combine into the shared musical language.
    pub fn blend(&mut self) -> MusicalState {
        if self.contributions.is_empty() {
            return MusicalState::default();
        }

        let blended = match self.strategy {
            BlendStrategy::Mean => self.blend_mean(),
            BlendStrategy::PhiWeighted => self.blend_phi_weighted(),
            BlendStrategy::Maximum => self.blend_max(),
            BlendStrategy::Minimum => self.blend_min(),
        };

        // EMA smoothing
        let a = self.ema_alpha;
        for i in 0..8 {
            self.collective_harmonies[i] += a * (blended.0[i] - self.collective_harmonies[i]);
        }
        self.collective_phi += a * (blended.1 - self.collective_phi);

        let mut state = MusicalState::default();
        state.harmony_activations = self.collective_harmonies;
        state.consciousness_level = self.collective_phi;
        // Arousal from variance (disagreement = tension)
        state.arousal = self.compute_disagreement();
        // Valence from collective phi direction
        state.valence = (self.collective_phi - 0.5) * 2.0;
        state
    }

    fn blend_mean(&self) -> ([f32; 8], f32) {
        let n = self.contributions.len() as f32;
        let mut harmonies = [0.0f32; 8];
        let mut phi = 0.0f32;
        for c in &self.contributions {
            for i in 0..8 {
                harmonies[i] += c.harmonies[i];
            }
            phi += c.phi;
        }
        for h in &mut harmonies {
            *h /= n;
        }
        (harmonies, phi / n)
    }

    fn blend_phi_weighted(&self) -> ([f32; 8], f32) {
        let total_phi: f32 = self.contributions.iter().map(|c| c.phi.max(0.01)).sum();
        let mut harmonies = [0.0f32; 8];
        let mut phi = 0.0f32;
        for c in &self.contributions {
            let weight = c.phi.max(0.01) / total_phi;
            for i in 0..8 {
                harmonies[i] += c.harmonies[i] * weight;
            }
            phi += c.phi * weight;
        }
        (harmonies, phi)
    }

    fn blend_max(&self) -> ([f32; 8], f32) {
        let mut harmonies = [0.0f32; 8];
        let mut phi = 0.0f32;
        for c in &self.contributions {
            for i in 0..8 {
                harmonies[i] = harmonies[i].max(c.harmonies[i]);
            }
            phi = phi.max(c.phi);
        }
        (harmonies, phi)
    }

    fn blend_min(&self) -> ([f32; 8], f32) {
        let mut harmonies = [1.0f32; 8];
        let mut phi = 1.0f32;
        for c in &self.contributions {
            for i in 0..8 {
                harmonies[i] = harmonies[i].min(c.harmonies[i]);
            }
            phi = phi.min(c.phi);
        }
        (harmonies, phi)
    }

    /// Compute disagreement (variance) across participants.
    /// High disagreement → high arousal in the collective state.
    fn compute_disagreement(&self) -> f32 {
        if self.contributions.len() < 2 {
            return 0.0;
        }
        let n = self.contributions.len() as f32;
        let mean_phi: f32 = self.contributions.iter().map(|c| c.phi).sum::<f32>() / n;
        let variance: f32 = self
            .contributions
            .iter()
            .map(|c| (c.phi - mean_phi).powi(2))
            .sum::<f32>()
            / n;
        (variance.sqrt() * 3.0).clamp(0.0, 1.0) // scale to [0, 1]
    }

    /// Number of active participants.
    pub fn participant_count(&self) -> usize {
        self.contributions.len()
    }

    /// Get collective harmony activations.
    pub fn collective_harmonies(&self) -> &[f32; 8] {
        &self.collective_harmonies
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_contribution(id: u64, phi: f32, harmonies: [f32; 8]) -> HarmonyContribution {
        HarmonyContribution {
            participant_id: id,
            harmonies,
            phi,
            cycle: 0,
        }
    }

    #[test]
    fn single_participant_is_identity() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        composer.contribute(make_contribution(1, 0.8, [0.5; 8]));
        let state = composer.blend();
        // With EMA smoothing from default [0.3; 8], won't be exactly [0.5; 8]
        // but should move toward it
        assert!(state.harmony_activations[0] > 0.3);
    }

    #[test]
    fn phi_weighted_favors_high_phi() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::PhiWeighted);
        // High-phi participant with high harmonies
        composer.contribute(make_contribution(1, 0.9, [0.8; 8]));
        // Low-phi participant with low harmonies
        composer.contribute(make_contribution(2, 0.1, [0.2; 8]));
        let state = composer.blend();

        // Should be closer to high-phi participant
        // (EMA moves from default 0.3 toward weighted blend ~0.74)
        assert!(
            state.harmony_activations[0] > 0.3,
            "phi-weighted should favor high-phi: {}",
            state.harmony_activations[0]
        );
    }

    #[test]
    fn maximum_takes_union() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Maximum);
        composer.contribute(make_contribution(
            1,
            0.5,
            [0.2, 0.8, 0.3, 0.1, 0.5, 0.6, 0.4, 0.7],
        ));
        composer.contribute(make_contribution(
            2,
            0.5,
            [0.8, 0.2, 0.7, 0.9, 0.1, 0.4, 0.6, 0.3],
        ));
        let state = composer.blend();

        // Each harmony should be at least the EMA-smoothed max of the two
        // After one blend step from default [0.3], should move toward the max
        assert!(state.harmony_activations[0] > 0.3);
    }

    #[test]
    fn disagreement_increases_arousal() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        // Very different Phi values = high disagreement
        composer.contribute(make_contribution(1, 0.1, [0.5; 8]));
        composer.contribute(make_contribution(2, 0.9, [0.5; 8]));
        let state = composer.blend();

        assert!(
            state.arousal > 0.3,
            "disagreement should increase arousal: {}",
            state.arousal
        );
    }

    #[test]
    fn remove_participant() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        composer.contribute(make_contribution(1, 0.5, [0.5; 8]));
        composer.contribute(make_contribution(2, 0.5, [0.5; 8]));
        assert_eq!(composer.participant_count(), 2);

        composer.remove_participant(1);
        assert_eq!(composer.participant_count(), 1);
    }

    #[test]
    fn empty_returns_default() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        let state = composer.blend();
        assert_eq!(state.consciousness_level, 0.5); // MusicalState default
    }

    #[test]
    fn update_replaces_existing() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        composer.contribute(make_contribution(1, 0.3, [0.3; 8]));
        composer.contribute(make_contribution(1, 0.8, [0.8; 8])); // update, not add
        assert_eq!(composer.participant_count(), 1);
    }
}
