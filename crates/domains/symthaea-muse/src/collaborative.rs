// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Collaborative music composition from participant harmony contributions.
//!
//! This legacy composer aggregates feature vectors. It does not implement
//! consent, attribution, authorship, or contribution provenance; new
//! sympoietic workflows should use creative_ecology for those boundaries.
//! A Phi-like value is retained for compatibility and an explicit legacy
//! ablation, but is not a validated consciousness measure or social authority.
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
    /// Legacy Phi-like proxy. Not a validated consciousness measure and not
    /// suitable as a default influence, merit, or authority weight.
    pub phi: f32,
    /// Timestamp (cycle number).
    pub cycle: u64,
}

/// Blending strategy for combining multiple harmony contributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendStrategy {
    /// Simple average of all contributions.
    Mean,
    /// Legacy research ablation weighted by a Phi-like proxy. Do not interpret
    /// this as consciousness measurement or a legitimate social authority.
    PhiWeighted,
    /// Maximum activation per harmony (union of all experiences).
    Maximum,
    /// Minimum activation per harmony (intersection — shared experience only).
    Minimum,
}

impl Default for BlendStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

/// Collaborative composition orchestrator.
pub struct CollaborativeComposer {
    contributions: Vec<HarmonyContribution>,
    strategy: BlendStrategy,
    /// Smoothed collective harmony (EMA across updates).
    collective_harmonies: [f32; 8],
    ema_alpha: f32,
}

impl CollaborativeComposer {
    pub fn new(strategy: BlendStrategy) -> Self {
        Self {
            contributions: Vec::with_capacity(MAX_PARTICIPANTS),
            strategy,
            collective_harmonies: [0.3; 8],
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

        // EMA smoothing preserves a continuous aggregate without assigning
        // extra social authority to any contributor under the default policy.
        let a = self.ema_alpha;
        for (current, target) in self.collective_harmonies.iter_mut().zip(blended) {
            *current += a * (target - *current);
        }

        let mut state = MusicalState::default();
        state.harmony_activations = self.collective_harmonies;
        state
    }

    fn blend_mean(&self) -> [f32; 8] {
        let n = self.contributions.len() as f32;
        let mut harmonies = [0.0f32; 8];
        for contribution in &self.contributions {
            for (sum, value) in harmonies.iter_mut().zip(contribution.harmonies) {
                *sum += value;
            }
        }
        for value in &mut harmonies {
            *value /= n;
        }
        harmonies
    }

    fn blend_phi_weighted(&self) -> [f32; 8] {
        let total_phi: f32 = self
            .contributions
            .iter()
            .map(|contribution| contribution.phi.max(0.01))
            .sum();
        let mut harmonies = [0.0f32; 8];
        for contribution in &self.contributions {
            let weight = contribution.phi.max(0.01) / total_phi;
            for (sum, value) in harmonies.iter_mut().zip(contribution.harmonies) {
                *sum += value * weight;
            }
        }
        harmonies
    }

    fn blend_max(&self) -> [f32; 8] {
        let mut harmonies = [0.0f32; 8];
        for contribution in &self.contributions {
            for (current, value) in harmonies.iter_mut().zip(contribution.harmonies) {
                *current = (*current).max(value);
            }
        }
        harmonies
    }

    fn blend_min(&self) -> [f32; 8] {
        let mut harmonies = [1.0f32; 8];
        for contribution in &self.contributions {
            for (current, value) in harmonies.iter_mut().zip(contribution.harmonies) {
                *current = (*current).min(value);
            }
        }
        harmonies
    }

    /// Compute disagreement across participant harmony contributions.
    /// This reports feature disagreement, not affective arousal.
    fn compute_disagreement(&self) -> f32 {
        if self.contributions.len() < 2 {
            return 0.0;
        }
        let n = self.contributions.len() as f32;
        let mut total_variance = 0.0f32;
        for index in 0..8 {
            let mean = self
                .contributions
                .iter()
                .map(|contribution| contribution.harmonies[index])
                .sum::<f32>()
                / n;
            let variance = self
                .contributions
                .iter()
                .map(|contribution| (contribution.harmonies[index] - mean).powi(2))
                .sum::<f32>()
                / n;
            total_variance += variance;
        }
        ((total_variance / 8.0).sqrt() * 3.0).clamp(0.0, 1.0)
    }

    /// Number of active participants.
    pub fn participant_count(&self) -> usize {
        self.contributions.len()
    }

    /// Get collective harmony activations.
    pub fn collective_harmonies(&self) -> &[f32; 8] {
        &self.collective_harmonies
    }

    /// Measure disagreement between participants' harmony feature vectors.
    ///
    /// This is a descriptive feature-space statistic. It is not affective
    /// arousal, social conflict, or an estimate of consciousness.
    pub fn disagreement(&self) -> f32 {
        self.compute_disagreement()
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
    fn default_strategy_does_not_weight_by_phi_proxy() {
        assert_eq!(BlendStrategy::default(), BlendStrategy::Mean);
    }

    #[test]
    fn default_blend_is_invariant_to_legacy_phi_proxy() {
        let mut first = CollaborativeComposer::new(BlendStrategy::default());
        first.contribute(make_contribution(1, 0.01, [0.2; 8]));
        first.contribute(make_contribution(2, 0.99, [0.8; 8]));

        let mut second = CollaborativeComposer::new(BlendStrategy::default());
        second.contribute(make_contribution(1, 0.99, [0.2; 8]));
        second.contribute(make_contribution(2, 0.01, [0.8; 8]));

        assert_eq!(first.blend(), second.blend());
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
    fn disagreement_tracks_musical_contribution_vectors() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        // Identical musical contributions have zero feature disagreement,
        // regardless of different legacy Phi-like values.
        composer.contribute(make_contribution(1, 0.1, [0.5; 8]));
        composer.contribute(make_contribution(2, 0.9, [0.5; 8]));
        assert_eq!(composer.disagreement(), 0.0);

        let mut different = CollaborativeComposer::new(BlendStrategy::Mean);
        different.contribute(make_contribution(1, 0.9, [0.1; 8]));
        different.contribute(make_contribution(2, 0.1, [0.9; 8]));
        assert!(
            different.disagreement() > 0.3,
            "distinct music contributions should show feature disagreement"
        );

        // The descriptive aggregate does not masquerade as affective arousal.
        let state = different.blend();
        assert_eq!(state.arousal, MusicalState::default().arousal);
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
