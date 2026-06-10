// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Community-configurable dimension weights for the 8D Sovereign Profile.
//!
//! Different communities value different axes. An energy cooperative weights
//! Thermodynamic Yield higher; a knowledge commons weights Epistemic Integrity.
//! These presets provide sensible defaults; communities can override via governance.

/// Weights for the 8 sovereign dimensions. Must sum to 1.0 (±0.01).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DimensionWeights {
    /// Weights in canonical dimension order (D0-D7).
    pub weights: [f64; 8],
}

impl DimensionWeights {
    /// Equal weights: each dimension contributes 12.5%.
    pub fn equal() -> Self {
        Self {
            weights: [0.125; 8],
        }
    }

    /// Default governance weights — balanced with slight emphasis on
    /// epistemic integrity and civic participation.
    pub fn governance() -> Self {
        Self {
            weights: [
                0.15, // D0: Epistemic Integrity (elevated)
                0.10, // D1: Thermodynamic Yield
                0.10, // D2: Network Resilience
                0.12, // D3: Economic Velocity
                0.18, // D4: Civic Participation (elevated)
                0.13, // D5: Stewardship & Care
                0.12, // D6: Semantic Resonance
                0.10, // D7: Domain Competence
            ],
        }
    }

    /// Energy cooperative — elevated thermodynamic + network resilience.
    pub fn energy_cooperative() -> Self {
        Self {
            weights: [
                0.10, // D0: Epistemic Integrity
                0.22, // D1: Thermodynamic Yield (elevated)
                0.18, // D2: Network Resilience (elevated)
                0.10, // D3: Economic Velocity
                0.10, // D4: Civic Participation
                0.12, // D5: Stewardship & Care
                0.10, // D6: Semantic Resonance
                0.08, // D7: Domain Competence
            ],
        }
    }

    /// Knowledge commons — elevated epistemic integrity + domain competence.
    pub fn knowledge_commons() -> Self {
        Self {
            weights: [
                0.22, // D0: Epistemic Integrity (elevated)
                0.06, // D1: Thermodynamic Yield
                0.08, // D2: Network Resilience
                0.08, // D3: Economic Velocity
                0.12, // D4: Civic Participation
                0.10, // D5: Stewardship & Care
                0.14, // D6: Semantic Resonance
                0.20, // D7: Domain Competence (elevated)
            ],
        }
    }

    /// Care community — elevated stewardship + semantic resonance.
    pub fn care_community() -> Self {
        Self {
            weights: [
                0.08, // D0: Epistemic Integrity
                0.08, // D1: Thermodynamic Yield
                0.08, // D2: Network Resilience
                0.10, // D3: Economic Velocity
                0.14, // D4: Civic Participation
                0.22, // D5: Stewardship & Care (elevated)
                0.20, // D6: Semantic Resonance (elevated)
                0.10, // D7: Domain Competence
            ],
        }
    }

    /// Check that weights sum to 1.0 within tolerance.
    pub fn is_normalized(&self) -> bool {
        let sum: f64 = self.weights.iter().sum();
        (sum - 1.0).abs() < 0.01
    }

    /// Normalize weights to sum to 1.0.
    pub fn normalize(&mut self) {
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }
}

impl Default for DimensionWeights {
    fn default() -> Self {
        Self::governance()
    }
}
