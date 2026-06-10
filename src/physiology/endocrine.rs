// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Endocrine System - Hormone modeling for consciousness
//!
//! Models the hormonal influences on consciousness and cognition:
//! - Cortisol (stress) - scatters attention
//! - Dopamine (reward) - enhances engagement
//! - Acetylcholine (attention) - focuses processing
//! - Oxytocin (connection) - enhances coherence

use serde::{Deserialize, Serialize};

/// Hormone state representation for consciousness modeling
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HormoneState {
    /// Cortisol level (0.0-1.0) - stress hormone
    /// High cortisol impairs learning and scatters attention
    pub cortisol: f64,

    /// Dopamine level (0.0-1.0) - reward hormone
    /// High dopamine enhances engagement and Φ gain
    pub dopamine: f64,

    /// Acetylcholine level (0.0-1.0) - attention neurotransmitter
    /// High acetylcholine enhances focus and centering
    pub acetylcholine: f64,

    /// Oxytocin level (0.0-1.0) - connection hormone
    /// High oxytocin enhances relational coherence
    pub oxytocin: f64,

    /// Norepinephrine level (0.0-1.0) - arousal neurotransmitter
    /// High norepinephrine increases alertness
    pub norepinephrine: f64,

    /// Serotonin level (0.0-1.0) - mood neurotransmitter
    /// High serotonin enhances well-being
    pub serotonin: f64,
}

impl Default for HormoneState {
    fn default() -> Self {
        Self::neutral()
    }
}

impl HormoneState {
    /// Create a neutral/baseline hormone state
    pub fn neutral() -> Self {
        Self {
            cortisol: 0.3,
            dopamine: 0.5,
            acetylcholine: 0.5,
            oxytocin: 0.5,
            norepinephrine: 0.5,
            serotonin: 0.5,
        }
    }

    /// Create a stressed hormone state (high cortisol)
    pub fn stressed() -> Self {
        Self {
            cortisol: 0.8,
            dopamine: 0.3,
            acetylcholine: 0.3,
            oxytocin: 0.3,
            norepinephrine: 0.8,
            serotonin: 0.3,
        }
    }

    /// Create a relaxed hormone state (low cortisol, high oxytocin)
    pub fn relaxed() -> Self {
        Self {
            cortisol: 0.2,
            dopamine: 0.6,
            acetylcholine: 0.5,
            oxytocin: 0.7,
            norepinephrine: 0.3,
            serotonin: 0.7,
        }
    }

    /// Create a focused hormone state (high acetylcholine)
    pub fn focused() -> Self {
        Self {
            cortisol: 0.4,
            dopamine: 0.6,
            acetylcholine: 0.8,
            oxytocin: 0.5,
            norepinephrine: 0.6,
            serotonin: 0.5,
        }
    }

    /// Create a rewarded hormone state (high dopamine)
    pub fn rewarded() -> Self {
        Self {
            cortisol: 0.2,
            dopamine: 0.9,
            acetylcholine: 0.6,
            oxytocin: 0.6,
            norepinephrine: 0.5,
            serotonin: 0.7,
        }
    }

    /// Create a connected hormone state (high oxytocin)
    pub fn connected() -> Self {
        Self {
            cortisol: 0.2,
            dopamine: 0.7,
            acetylcholine: 0.5,
            oxytocin: 0.9,
            norepinephrine: 0.4,
            serotonin: 0.7,
        }
    }

    /// Calculate overall stress level (0.0-1.0)
    pub fn stress_level(&self) -> f64 {
        // Stress is high cortisol + high norepinephrine, reduced by serotonin
        ((self.cortisol + self.norepinephrine * 0.5) / 1.5 - self.serotonin * 0.3).clamp(0.0, 1.0)
    }

    /// Calculate overall engagement level (0.0-1.0)
    pub fn engagement_level(&self) -> f64 {
        (self.dopamine + self.acetylcholine + self.norepinephrine) / 3.0
    }

    /// Calculate overall connection level (0.0-1.0)
    pub fn connection_level(&self) -> f64 {
        (self.oxytocin + self.serotonin * 0.5) / 1.5
    }

    /// Apply temporal decay toward baseline
    pub fn decay_toward_neutral(&mut self, rate: f64) {
        let neutral = Self::neutral();
        self.cortisol += (neutral.cortisol - self.cortisol) * rate;
        self.dopamine += (neutral.dopamine - self.dopamine) * rate;
        self.acetylcholine += (neutral.acetylcholine - self.acetylcholine) * rate;
        self.oxytocin += (neutral.oxytocin - self.oxytocin) * rate;
        self.norepinephrine += (neutral.norepinephrine - self.norepinephrine) * rate;
        self.serotonin += (neutral.serotonin - self.serotonin) * rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_state() {
        let state = HormoneState::neutral();
        assert!(state.cortisol < 0.5);
        assert!((state.dopamine - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_stress_level() {
        let stressed = HormoneState::stressed();
        let relaxed = HormoneState::relaxed();
        assert!(stressed.stress_level() > relaxed.stress_level());
    }

    #[test]
    fn test_engagement_level() {
        let focused = HormoneState::focused();
        let stressed = HormoneState::stressed();
        assert!(focused.engagement_level() > stressed.engagement_level());
    }
}
