// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nurture/attachment bridge for the cognitive loop.
//!
//! Integrates Bowlby attachment theory into the neuromodulator cycle:
//! caregiver presence/absence modulates oxytocin, NE, 5-HT, DA, adenosine.
//!
//! Enabled via `CognitiveLoopConfig::enable_nurture_attachment`.

use symthaea_nurture::{
    AttachmentNeuromodulation, AttachmentStyle, AttachmentSystem, CaregiverAction,
};

/// Bridge between the attachment system and the cognitive loop.
#[derive(Debug, Clone)]
pub struct NurtureAttachmentBridge {
    pub system: AttachmentSystem,
    /// Whether a caregiver is currently present (drives separation/reunion).
    pub caregiver_present: bool,
    /// Accumulated cycle count for age advancement.
    cycles_since_age_update: u64,
    /// Cycles per simulated month (configurable).
    cycles_per_month: u64,
}

impl NurtureAttachmentBridge {
    pub fn new() -> Self {
        Self {
            system: AttachmentSystem::new(),
            caregiver_present: true,
            cycles_since_age_update: 0,
            cycles_per_month: 234 * 60 * 60 * 24 * 30, // ~234Hz * seconds_per_month
        }
    }

    /// Process one cognitive cycle's worth of attachment dynamics.
    /// Returns the neuromodulatory signature to apply to the bath.
    pub fn cycle_step(&mut self, arousal: f32, valence: f32) -> AttachmentNeuromodulation {
        // Advance developmental age
        self.cycles_since_age_update += 1;
        if self.cycles_since_age_update >= self.cycles_per_month {
            self.cycles_since_age_update = 0;
            self.system.advance_age(1.0); // 1 month
        }

        // Infer caregiver presence from arousal/valence
        // High arousal + negative valence = distress (separation-like)
        // Low arousal + positive valence = calm (caregiver present)
        let was_present = self.caregiver_present;
        let distress_signal = arousal > 0.7 && valence < 0.3;
        let comfort_signal = arousal < 0.4 && valence > 0.6;

        if distress_signal && was_present {
            self.caregiver_present = false;
        } else if comfort_signal && !was_present {
            self.caregiver_present = true;
        }

        // Generate neuromodulation based on state transition
        if !was_present && self.caregiver_present {
            // Reunion
            let action = CaregiverAction::reliable();
            self.system.process_reunion(&action)
        } else if was_present && !self.caregiver_present {
            // Separation onset
            self.system.process_separation(0.07); // ~4 seconds at 234Hz
            AttachmentNeuromodulation::separation()
        } else if !self.caregiver_present {
            // Ongoing separation
            self.system.process_separation(0.07);
            self.system.current_neuromodulation()
        } else {
            // Caregiver present, normal interaction
            // Scale co-regulation by inverse arousal (calmer = better sync)
            let synchrony = (1.0 - arousal as f64).max(0.0);
            AttachmentNeuromodulation::coregulation(synchrony)
        }
    }

    /// Get the current attachment style.
    pub fn style(&self) -> AttachmentStyle {
        self.system.style
    }

    /// Get the current security score.
    pub fn security_score(&self) -> f64 {
        self.system.security_score
    }
}

impl Default for NurtureAttachmentBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_neuromodulators::NeuromodulatorBath;

    #[test]
    fn test_bridge_creation() {
        let bridge = NurtureAttachmentBridge::new();
        assert!(bridge.caregiver_present);
        assert_eq!(bridge.system.style, AttachmentStyle::Forming);
    }

    #[test]
    fn test_calm_cycle_produces_coregulation() {
        let mut bridge = NurtureAttachmentBridge::new();
        // Calm, positive state should produce co-regulation oxytocin
        let neuromod = bridge.cycle_step(0.3, 0.7);
        assert!(neuromod.oxytocin_burst > 0.0);
    }

    #[test]
    fn test_distress_triggers_separation() {
        let mut bridge = NurtureAttachmentBridge::new();
        assert!(bridge.caregiver_present);
        let neuromod = bridge.cycle_step(0.8, 0.2); // high arousal, negative
        assert!(!bridge.caregiver_present);
        assert!(neuromod.noradrenaline_separation > 0.0);
    }

    #[test]
    fn test_comfort_triggers_reunion() {
        let mut bridge = NurtureAttachmentBridge::new();
        // First, trigger separation
        bridge.cycle_step(0.8, 0.2);
        assert!(!bridge.caregiver_present);
        // Then comfort -> reunion
        let neuromod = bridge.cycle_step(0.3, 0.7);
        assert!(bridge.caregiver_present);
        assert!(neuromod.oxytocin_burst > 0.0);
    }

    #[test]
    fn test_neuromod_applies_to_bath() {
        let mut bridge = NurtureAttachmentBridge::new();
        let mut bath = NeuromodulatorBath::default();
        let initial_oxy = bath.oxytocin.level;

        // Trigger separation then reunion for maximum neuromod effect
        bridge.cycle_step(0.8, 0.2); // separation
        let reunion_mod = bridge.cycle_step(0.3, 0.7); // reunion
        reunion_mod.apply_to_bath(&mut bath);

        assert!(bath.oxytocin.level > initial_oxy);
    }

    /// Simulates a Strange Situation-inspired protocol:
    /// 100 calm cycles (caregiver present) -> 50 distress cycles -> 50 reunion cycles.
    /// Verifies that attachment style and security score evolve appropriately.
    #[test]
    fn test_strange_situation_100_calm_50_distress_50_reunion() {
        let mut bridge = NurtureAttachmentBridge::new();
        let initial_security = bridge.security_score();

        // Phase 1: 100 calm cycles — caregiver present, low arousal, positive valence
        // Co-regulation only (no process_interaction), so security_score stays at baseline.
        for _ in 0..100 {
            let neuromod = bridge.cycle_step(0.3, 0.7);
            // Every calm cycle should produce positive oxytocin co-regulation
            assert!(
                neuromod.oxytocin_burst >= 0.0,
                "calm cycle should not produce negative oxytocin"
            );
        }
        assert!(
            bridge.caregiver_present,
            "caregiver should remain present during calm phase"
        );
        assert_eq!(
            bridge.style(),
            AttachmentStyle::Forming,
            "style should still be Forming after calm-only cycles"
        );
        let post_calm_security = bridge.security_score();
        assert!(
            (post_calm_security - initial_security).abs() < f64::EPSILON,
            "security score should not change during calm (no interactions processed)"
        );

        // Phase 2: 50 distress cycles — high arousal, negative valence
        // First cycle triggers separation onset; subsequent cycles are ongoing separation.
        for i in 0..50 {
            let neuromod = bridge.cycle_step(0.8, 0.2);
            if i == 0 {
                // First distress cycle triggers separation onset -> NE spike
                assert!(
                    neuromod.noradrenaline_separation > 0.0,
                    "separation onset should produce NE spike"
                );
            }
        }
        assert!(
            !bridge.caregiver_present,
            "caregiver should be absent after distress phase"
        );
        assert_eq!(
            bridge.style(),
            AttachmentStyle::Forming,
            "style should remain Forming (no reunion interactions yet)"
        );
        // Separation doesn't call process_interaction, so security_score unchanged
        let post_distress_security = bridge.security_score();
        assert!(
            (post_distress_security - initial_security).abs() < f64::EPSILON,
            "security score should not change during separation (no interactions processed)"
        );

        // Phase 3: 50 reunion cycles — low arousal, positive valence
        // First cycle triggers reunion (process_reunion with reliable action);
        // subsequent cycles are calm co-regulation (caregiver already present).
        for i in 0..50 {
            let neuromod = bridge.cycle_step(0.3, 0.7);
            if i == 0 {
                // First reunion cycle: oxytocin burst from reliable reunion
                assert!(
                    neuromod.oxytocin_burst > 0.0,
                    "reunion should produce oxytocin burst"
                );
            }
        }
        assert!(
            bridge.caregiver_present,
            "caregiver should be present after reunion phase"
        );

        // After reunion: security score should have increased from the reliable interaction
        let post_reunion_security = bridge.security_score();
        assert!(
            post_reunion_security > initial_security,
            "security score should increase after reliable reunion: {} > {}",
            post_reunion_security,
            initial_security
        );

        // Style remains Forming because only 1 interaction was processed
        // (style requires >= 20 interactions to differentiate)
        assert_eq!(
            bridge.style(),
            AttachmentStyle::Forming,
            "style should remain Forming after a single reunion interaction"
        );
    }
}
