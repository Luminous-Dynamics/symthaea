// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bidirectional affect co-regulation.
//!
//! Co-regulation is the process by which a calm caregiver helps modulate an
//! infant's arousal through physical contact and attunement. The mechanism is
//! oxytocin-mediated: skin-to-skin contact promotes oxytocin release, which
//! dampens the HPA stress axis.
//!
//! Key dynamics:
//! - Contact + calm caregiver -> reduced infant arousal
//! - Synchrony builds with sustained contact (slow increase)
//! - Synchrony decays without contact (slower decrease)
//! - Oxytocin mediates the coupling strength
//!
//! Science: Tronick (1989) Mutual Regulation Model; Feldman (2012) Bio-behavioral
//! synchrony; Moberg (2003) Oxytocin and calm-and-connection system.

use serde::{Deserialize, Serialize};

use crate::types::AttachmentNeuromodulation;

/// Co-regulation state between caregiver and infant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoRegulationState {
    /// Degree of affect synchrony between caregiver and infant (0-1).
    pub synchrony: f64,
    /// How much contact reduces arousal (0-1).
    pub damping_factor: f64,
    /// Oxytocin-mediated coupling strength (0-1).
    pub oxytocin_mediated_coupling: f64,
    /// Number of contact episodes (builds synchrony capacity).
    pub contact_episodes: u64,
}

impl CoRegulationState {
    /// Create a new co-regulation state.
    pub fn new() -> Self {
        Self {
            synchrony: 0.1,
            damping_factor: 0.3,
            oxytocin_mediated_coupling: 0.5,
            contact_episodes: 0,
        }
    }

    /// Regulate infant arousal via caregiver contact.
    ///
    /// When contact is present, the caregiver's calmness dampens the infant's
    /// arousal proportionally to synchrony and oxytocin coupling. Synchrony
    /// builds slowly (+0.01 per contact step, max 1.0).
    ///
    /// When contact is absent, synchrony slowly decays (-0.005 per step).
    ///
    /// Returns the neuromodulatory signature of the co-regulation event.
    ///
    /// # Arguments
    /// * `infant_arousal` - Mutable reference to infant's arousal level (0-1)
    /// * `caregiver_calm` - Caregiver's calmness level (0-1, where 1 = perfectly calm)
    /// * `contact` - Whether physical contact is occurring
    pub fn regulate(
        &mut self,
        infant_arousal: &mut f64,
        caregiver_calm: f64,
        contact: bool,
    ) -> AttachmentNeuromodulation {
        if contact {
            self.contact_episodes += 1;

            // Damping: reduce arousal based on how calm the caregiver is
            // The formula: damping * synchrony * oxytocin_coupling * (calm - 0.5)
            // When caregiver_calm > 0.5, infant arousal decreases
            // When caregiver_calm < 0.5 (anxious caregiver), arousal can increase
            let damping = self.damping_factor * self.synchrony * self.oxytocin_mediated_coupling;
            *infant_arousal -= damping * (caregiver_calm - 0.5);
            *infant_arousal = infant_arousal.clamp(0.0, 1.0);

            // Synchrony builds with contact
            self.synchrony = (self.synchrony + 0.01).min(1.0);

            // Oxytocin coupling strengthens with repeated contact
            self.oxytocin_mediated_coupling = (self.oxytocin_mediated_coupling + 0.002).min(1.0);

            AttachmentNeuromodulation::coregulation(self.synchrony)
        } else {
            // Synchrony decays without contact (slower than buildup)
            self.synchrony = (self.synchrony - 0.005).max(0.0);

            AttachmentNeuromodulation::neutral()
        }
    }

    /// Get the current regulation capacity (how effective co-regulation can be).
    pub fn regulation_capacity(&self) -> f64 {
        self.damping_factor * self.synchrony * self.oxytocin_mediated_coupling
    }
}

impl Default for CoRegulationState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contact_reduces_arousal() {
        let mut state = CoRegulationState::new();
        state.synchrony = 0.8; // Established synchrony
        state.oxytocin_mediated_coupling = 0.8;

        let mut arousal = 0.8;
        let calm_caregiver = 0.9; // Very calm

        state.regulate(&mut arousal, calm_caregiver, true);

        assert!(
            arousal < 0.8,
            "Contact with calm caregiver should reduce arousal, got {arousal}"
        );
    }

    #[test]
    fn test_no_contact_no_damping() {
        let mut state = CoRegulationState::new();
        let mut arousal = 0.8;

        state.regulate(&mut arousal, 0.9, false);

        assert!(
            (arousal - 0.8).abs() < 1e-10,
            "No contact should not change arousal, got {arousal}"
        );
    }

    #[test]
    fn test_synchrony_builds_with_contact() {
        let mut state = CoRegulationState::new();
        let initial_sync = state.synchrony;
        let mut arousal = 0.5;

        for _ in 0..100 {
            state.regulate(&mut arousal, 0.7, true);
        }

        assert!(
            state.synchrony > initial_sync,
            "Synchrony should build with contact: {} -> {}",
            initial_sync,
            state.synchrony
        );
    }

    #[test]
    fn test_synchrony_decays_without_contact() {
        let mut state = CoRegulationState::new();
        state.synchrony = 0.8;
        let mut arousal = 0.5;

        for _ in 0..100 {
            state.regulate(&mut arousal, 0.7, false);
        }

        assert!(
            state.synchrony < 0.8,
            "Synchrony should decay without contact, got {}",
            state.synchrony
        );
    }

    #[test]
    fn test_calm_caregiver_more_effective() {
        let mut state1 = CoRegulationState::new();
        state1.synchrony = 0.5;
        state1.oxytocin_mediated_coupling = 0.7;

        let mut state2 = state1.clone();

        let mut arousal_calm = 0.8;
        let mut arousal_anxious = 0.8;

        state1.regulate(&mut arousal_calm, 0.95, true); // Very calm
        state2.regulate(&mut arousal_anxious, 0.55, true); // Barely calm

        assert!(
            arousal_calm < arousal_anxious,
            "Calm caregiver should reduce more: calm={arousal_calm}, anxious={arousal_anxious}"
        );
    }

    #[test]
    fn test_anxious_caregiver_increases_arousal() {
        let mut state = CoRegulationState::new();
        state.synchrony = 0.8;
        state.oxytocin_mediated_coupling = 0.8;

        let mut arousal = 0.5;
        // Caregiver below 0.5 = anxious -> should increase infant arousal
        state.regulate(&mut arousal, 0.2, true);

        assert!(
            arousal > 0.5,
            "Anxious caregiver should increase arousal, got {arousal}"
        );
    }

    #[test]
    fn test_arousal_clamped_0_1() {
        let mut state = CoRegulationState::new();
        state.synchrony = 1.0;
        state.damping_factor = 1.0;
        state.oxytocin_mediated_coupling = 1.0;

        let mut arousal = 0.1;
        state.regulate(&mut arousal, 1.0, true); // Maximum calming
        assert!(arousal >= 0.0, "Arousal should not go below 0");

        arousal = 0.9;
        state.regulate(&mut arousal, 0.0, true); // Maximum agitation
        assert!(arousal <= 1.0, "Arousal should not go above 1");
    }

    #[test]
    fn test_regulation_capacity() {
        let state = CoRegulationState::new();
        let cap = state.regulation_capacity();
        assert!(cap > 0.0, "Should have some regulation capacity");
        assert!(cap < 1.0, "Shouldn't be at full capacity initially");
    }

    #[test]
    fn test_repeated_contact_improves_coupling() {
        let mut state = CoRegulationState::new();
        let initial_coupling = state.oxytocin_mediated_coupling;
        let mut arousal = 0.5;

        for _ in 0..200 {
            state.regulate(&mut arousal, 0.7, true);
        }

        assert!(
            state.oxytocin_mediated_coupling > initial_coupling,
            "Oxytocin coupling should strengthen: {} -> {}",
            initial_coupling,
            state.oxytocin_mediated_coupling
        );
    }

    #[test]
    fn test_contact_episodes_count() {
        let mut state = CoRegulationState::new();
        let mut arousal = 0.5;

        for _ in 0..10 {
            state.regulate(&mut arousal, 0.7, true);
        }
        for _ in 0..5 {
            state.regulate(&mut arousal, 0.7, false);
        }

        assert_eq!(state.contact_episodes, 10);
    }

    #[test]
    fn test_coregulation_neuromod_returns_oxytocin() {
        let mut state = CoRegulationState::new();
        state.synchrony = 0.8;
        let mut arousal = 0.5;

        let neuromod = state.regulate(&mut arousal, 0.8, true);
        assert!(
            neuromod.oxytocin_burst > 0.0,
            "Co-regulation with contact should produce oxytocin"
        );
    }
}
