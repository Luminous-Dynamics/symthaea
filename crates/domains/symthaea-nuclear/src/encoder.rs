// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC encoder for nuclear states.
//!
//! Encodes nuclear properties (Z, N, binding energy, shell correction, deformation)
//! into ContinuousHV for integration with the physics-bridge catalog and cognitive loop.

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// Deterministic seeds for basis vectors
const SEED_Z: u64 = 0xA0C1_0001_0001;
const SEED_N: u64 = 0xA0C1_0002_0002;
const SEED_BINDING: u64 = 0xA0C1_0003_0003;
const SEED_SHELL: u64 = 0xA0C1_0004_0004;
const SEED_DEFORMATION: u64 = 0xA0C1_0005_0005;

/// Nuclear state for HDC encoding.
#[derive(Debug, Clone)]
pub struct NuclearState {
    /// Atomic number (protons)
    pub z: u16,
    /// Neutron number
    pub n: u16,
    /// Binding energy (MeV), normalized to ~0-2000 range
    pub binding_energy: f64,
    /// Shell correction energy (MeV), typically -10 to +10
    pub shell_correction: f64,
    /// Deformation parameter β₂ (0 = spherical), typically 0-0.5
    pub deformation: f64,
}

/// Encodes nuclear states as 16,384-dimensional hypervectors.
pub struct NuclearStateEncoder {
    basis_z: ContinuousHV,
    basis_n: ContinuousHV,
    basis_binding: ContinuousHV,
    basis_shell: ContinuousHV,
    basis_deformation: ContinuousHV,
}

impl NuclearStateEncoder {
    pub fn new() -> Self {
        Self {
            basis_z: ContinuousHV::random(HDC_DIMENSION, SEED_Z),
            basis_n: ContinuousHV::random(HDC_DIMENSION, SEED_N),
            basis_binding: ContinuousHV::random(HDC_DIMENSION, SEED_BINDING),
            basis_shell: ContinuousHV::random(HDC_DIMENSION, SEED_SHELL),
            basis_deformation: ContinuousHV::random(HDC_DIMENSION, SEED_DEFORMATION),
        }
    }

    /// Encode a nuclear state as a ContinuousHV.
    ///
    /// Uses level-encoding (thermometer) for each property, then binds
    /// with the property basis vector, and bundles all into one HV.
    pub fn encode(&self, state: &NuclearState) -> ContinuousHV {
        // Normalize each property to [0, 1] range
        let z_norm = (state.z as f64 / 130.0).min(1.0); // Z up to ~130
        let n_norm = (state.n as f64 / 200.0).min(1.0); // N up to ~200
        let b_norm = (state.binding_energy / 2000.0).clamp(0.0, 1.0);
        let shell_norm = ((state.shell_correction + 10.0) / 20.0).clamp(0.0, 1.0); // -10 to +10
        let def_norm = (state.deformation / 0.5).clamp(0.0, 1.0);

        // Bind each normalized value with its basis vector (scale = value encoding)
        let z_hv = self.basis_z.scale(z_norm as f32);
        let n_hv = self.basis_n.scale(n_norm as f32);
        let b_hv = self.basis_binding.scale(b_norm as f32);
        let shell_hv = self.basis_shell.scale(shell_norm as f32);
        let def_hv = self.basis_deformation.scale(def_norm as f32);

        // Bundle all components
        ContinuousHV::bundle(&[&z_hv, &n_hv, &b_hv, &shell_hv, &def_hv])
    }
}

impl Default for NuclearStateEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_dimension() {
        let encoder = NuclearStateEncoder::new();
        let state = NuclearState {
            z: 82,
            n: 126,
            binding_energy: 1636.0,
            shell_correction: -5.0,
            deformation: 0.0,
        };
        let hv = encoder.encode(&state);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_encoder_determinism() {
        let encoder = NuclearStateEncoder::new();
        let state = NuclearState {
            z: 26,
            n: 30,
            binding_energy: 492.0,
            shell_correction: -1.0,
            deformation: 0.05,
        };
        let hv1 = encoder.encode(&state);
        let hv2 = encoder.encode(&state);
        assert!(
            hv1.similarity(&hv2) > 0.999,
            "Same state should produce identical HV"
        );
    }

    #[test]
    fn test_similar_nuclei_cluster() {
        let encoder = NuclearStateEncoder::new();

        // Two neighboring isotopes should be more similar than distant ones
        let fe56 = encoder.encode(&NuclearState {
            z: 26,
            n: 30,
            binding_energy: 492.0,
            shell_correction: -1.0,
            deformation: 0.05,
        });
        let fe58 = encoder.encode(&NuclearState {
            z: 26,
            n: 32,
            binding_energy: 509.0,
            shell_correction: -0.5,
            deformation: 0.05,
        });
        let u238 = encoder.encode(&NuclearState {
            z: 92,
            n: 146,
            binding_energy: 1802.0,
            shell_correction: 2.0,
            deformation: 0.28,
        });

        let sim_near = fe56.similarity(&fe58);
        let sim_far = fe56.similarity(&u238);

        assert!(
            sim_near > sim_far,
            "Fe-56/Fe-58 similarity ({}) should exceed Fe-56/U-238 ({})",
            sim_near,
            sim_far
        );
    }
}
