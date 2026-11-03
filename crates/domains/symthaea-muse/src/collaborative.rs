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
/// A rejected contribution. Invalid or stale input is never silently admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContributionError {
    /// The Phi proxy is not a finite number.
    NonFinitePhi,
    /// The Phi proxy must be in the normalized interval [0, 1].
    PhiOutOfRange,
    /// A harmony activation is not a finite number.
    NonFiniteHarmony { index: usize },
    /// A harmony activation must be in the normalized interval [0, 1].
    HarmonyOutOfRange { index: usize },
    /// An update would replace a newer contribution from the same participant.
    StaleCycle { incoming_cycle: u64, current_cycle: u64 },
    /// The session has reached its participant limit.
    CapacityExceeded,
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
    ///
    /// Rejects non-finite/out-of-range values, stale per-participant cycles,
    /// and capacity overflow. Callers must handle the result: no contribution
    /// is silently dropped or allowed to overwrite newer state.
    pub fn contribute(
        &mut self,
        contribution: HarmonyContribution,
    ) -> Result<(), ContributionError> {
        if !contribution.phi.is_finite() {
            return Err(ContributionError::NonFinitePhi);
        }
        if !(0.0..=1.0).contains(&contribution.phi) {
            return Err(ContributionError::PhiOutOfRange);
        }
        for (index, activation) in contribution.harmonies.iter().enumerate() {
            if !activation.is_finite() {
                return Err(ContributionError::NonFiniteHarmony { index });
            }
            if !(0.0..=1.0).contains(activation) {
                return Err(ContributionError::HarmonyOutOfRange { index });
            }
        }

        if let Some(existing) = self
            .contributions
            .iter_mut()
            .find(|c| c.participant_id == contribution.participant_id)
        {
            if contribution.cycle < existing.cycle {
                return Err(ContributionError::StaleCycle {
                    incoming_cycle: contribution.cycle,
                    current_cycle: existing.cycle,
                });
            }
            *existing = contribution;
            return Ok(());
        }

        if self.contributions.len() >= MAX_PARTICIPANTS {
            return Err(ContributionError::CapacityExceeded);
        }

        self.contributions.push(contribution);
        Ok(())
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
        // This contribution schema contains no participant-level affect signal.
        // Do not infer valence or arousal from Phi proxies or profile divergence.
        state.arousal = 0.0;
        state.valence = 0.0;
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

    /// Normalized RMS spread of participant harmony-activation vectors.
    ///
    /// This is a descriptive contribution-divergence statistic, not a measure
    /// of conflict, musical quality, merit, authority, or emotional arousal.
    /// It is normalized to [0, 1] using the maximum possible standard
    /// deviation of values constrained to [0, 1].
    pub fn contribution_divergence(&self) -> f32 {
        let participant_count = self.contributions.len();
        if participant_count < 2 {
            return 0.0;
        }

        let mut means = [0.0f32; 8];
        for contribution in &self.contributions {
            for (mean, activation) in means.iter_mut().zip(contribution.harmonies.iter()) {
                *mean += activation;
            }
        }
        for mean in &mut means {
            *mean /= participant_count as f32;
        }

        let squared_error = self
            .contributions
            .iter()
            .flat_map(|contribution| {
                contribution
                    .harmonies
                    .iter()
                    .zip(means.iter())
                    .map(|(activation, mean)| (activation - mean).powi(2))
            })
            .sum::<f32>();
        let mean_squared_error = squared_error / (participant_count * 8) as f32;

        (mean_squared_error.sqrt() * 2.0).clamp(0.0, 1.0)
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
        composer
            .contribute(make_contribution(1, 0.8, [0.5; 8]))
            .expect("valid test contribution");
        let state = composer.blend();
        // With EMA smoothing from default [0.3; 8], won't be exactly [0.5; 8]
        // but should move toward it
        assert!(state.harmony_activations[0] > 0.3);
    }

    #[test]
    fn phi_weighted_favors_high_phi() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::PhiWeighted);
        // High-phi participant with high harmonies
        composer
            .contribute(make_contribution(1, 0.9, [0.8; 8]))
            .expect("valid test contribution");
        // Low-phi participant with low harmonies
        composer
            .contribute(make_contribution(2, 0.1, [0.2; 8]))
            .expect("valid test contribution");
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
    fn rejects_non_finite_phi() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        assert_eq!(
            composer.contribute(make_contribution(1, f32::NAN, [0.5; 8])),
            Err(ContributionError::NonFinitePhi)
        );
        assert_eq!(composer.participant_count(), 0);
    }

    #[test]
    fn rejects_out_of_range_phi_and_harmony_values() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        assert_eq!(
            composer.contribute(make_contribution(1, 1.01, [0.5; 8])),
            Err(ContributionError::PhiOutOfRange)
        );
        assert_eq!(
            composer.contribute(make_contribution(2, 0.5, [-0.01; 8])),
            Err(ContributionError::HarmonyOutOfRange { index: 0 })
        );
        assert_eq!(composer.participant_count(), 0);
    }

    #[test]
    fn rejects_non_finite_harmony_values() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        let mut harmonies = [0.5; 8];
        harmonies[3] = f32::INFINITY;
        assert_eq!(
            composer.contribute(make_contribution(1, 0.5, harmonies)),
            Err(ContributionError::NonFiniteHarmony { index: 3 })
        );
        assert_eq!(composer.participant_count(), 0);
    }

    #[test]
    fn stale_cycle_cannot_replace_newer_contribution() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        composer
            .contribute(HarmonyContribution {
                participant_id: 1,
                harmonies: [0.4; 8],
                phi: 0.5,
                cycle: 10,
            })
            .expect("cycle-10 contribution is valid");
        assert_eq!(
            composer.contribute(HarmonyContribution {
                participant_id: 1,
                harmonies: [0.8; 8],
                phi: 0.9,
                cycle: 9,
            }),
            Err(ContributionError::StaleCycle {
                incoming_cycle: 9,
                current_cycle: 10,
            })
        );

        let state = composer.blend();
        assert!(
            (state.harmony_activations[0] - 0.315).abs() < 1e-6,
            "stale contribution must not replace the cycle-10 state: {}",
            state.harmony_activations[0]
        );
    }

    #[test]
    fn participant_capacity_overflow_is_reported() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        for id in 0..MAX_PARTICIPANTS as u64 {
            composer
                .contribute(make_contribution(id, 0.5, [0.5; 8]))
                .expect("participant should fit within capacity");
        }

        assert_eq!(
            composer.contribute(make_contribution(999, 0.5, [0.5; 8])),
            Err(ContributionError::CapacityExceeded)
        );
        assert_eq!(composer.participant_count(), MAX_PARTICIPANTS);
    }

    #[test]
    fn same_or_newer_cycle_can_update_participant() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        composer
            .contribute(make_contribution(1, 0.3, [0.3; 8]))
            .expect("initial contribution is valid");
        composer
            .contribute(make_contribution(1, 0.8, [0.8; 8]))
            .expect("newer-or-equal contribution is valid");
        assert_eq!(composer.participant_count(), 1);
    }

    #[test]
    fn maximum_takes_union() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Maximum);
        composer.contribute(make_contribution(
            1,
            0.5,
            [0.2, 0.8, 0.3, 0.1, 0.5, 0.6, 0.4, 0.7],
        )).expect("valid test contribution");
        composer.contribute(make_contribution(
            2,
            0.5,
            [0.8, 0.2, 0.7, 0.9, 0.1, 0.4, 0.6, 0.3],
        )).expect("valid test contribution");
        let state = composer.blend();

        // Each harmony should be at least the EMA-smoothed max of the two
        // After one blend step from default [0.3], should move toward the max
        assert!(state.harmony_activations[0] > 0.3);
    }

    #[test]
    fn contribution_divergence_tracks_harmony_vector_spread() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        composer
            .contribute(make_contribution(1, 0.5, [0.1; 8]))
            .expect("valid test contribution");
        composer
            .contribute(make_contribution(2, 0.5, [0.9; 8]))
            .expect("valid test contribution");

        assert!(
            (composer.contribution_divergence() - 0.8).abs() < 1e-6,
            "divergence should reflect the spread in musical contribution vectors"
        );
    }

    #[test]
    fn phi_difference_does_not_change_contribution_divergence() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        composer
            .contribute(make_contribution(1, 0.1, [0.5; 8]))
            .expect("valid test contribution");
        composer
            .contribute(make_contribution(2, 0.9, [0.5; 8]))
            .expect("valid test contribution");

        assert_eq!(composer.contribution_divergence(), 0.0);
    }

    #[test]
    fn missing_affect_inputs_remain_neutral() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        composer
            .contribute(make_contribution(1, 0.1, [0.5; 8]))
            .expect("valid test contribution");
        composer
            .contribute(make_contribution(2, 0.9, [0.5; 8]))
            .expect("valid test contribution");
        let state = composer.blend();

        assert_eq!(state.valence, 0.0);
        assert_eq!(state.arousal, 0.0);
    }

    #[test]
    fn remove_participant() {
        let mut composer = CollaborativeComposer::new(BlendStrategy::Mean);
        composer
            .contribute(make_contribution(1, 0.5, [0.5; 8]))
            .expect("valid test contribution");
        composer
            .contribute(make_contribution(2, 0.5, [0.5; 8]))
            .expect("valid test contribution");
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
        composer
            .contribute(make_contribution(1, 0.3, [0.3; 8]))
            .expect("valid test contribution");
        composer
            .contribute(make_contribution(1, 0.8, [0.8; 8]))
            .expect("valid test contribution"); // update, not add
        assert_eq!(composer.participant_count(), 1);
    }
}
