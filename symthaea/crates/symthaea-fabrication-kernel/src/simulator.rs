// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Simulator Abstraction Layer (SAL)
//!
//! Symthaea doesn't send coordinates — she sends Force-HVs.
//! The physics backend translates HDC-encoded forces into physical simulation,
//! and the surprise between expected and actual response feeds back into FEP.

use crate::primitives::FAB_KERNEL_DIM;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;

// Force type seeds
const TENSION_SEED: u64 = 0x5445_4E53_1001;
const COMPRESSION_SEED: u64 = 0x434F_4D50_1002;
const SHEAR_SEED: u64 = 0x5348_4541_1003;
const TORSION_SEED: u64 = 0x544F_5253_1004;
const BENDING_SEED: u64 = 0x4245_4E44_1005;

// Axis seeds
const X_AXIS_SEED: u64 = 0x5841_5849_2001;
const Y_AXIS_SEED: u64 = 0x5941_5849_2002;
const Z_AXIS_SEED: u64 = 0x5A41_5849_2003;

/// Force type prototype HVs
pub fn tension_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, TENSION_SEED)
}
pub fn compression_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, COMPRESSION_SEED)
}
pub fn shear_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, SHEAR_SEED)
}
pub fn torsion_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, TORSION_SEED)
}
pub fn bending_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, BENDING_SEED)
}

/// Axis prototype HVs
pub fn x_axis_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, X_AXIS_SEED)
}
pub fn y_axis_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, Y_AXIS_SEED)
}
pub fn z_axis_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, Z_AXIS_SEED)
}

/// Compose a directional force by binding force type with direction
pub fn compose_force(force_type: &ContinuousHV, direction: &ContinuousHV) -> ContinuousHV {
    force_type.bind(direction)
}

/// An HDC-encoded force application
#[derive(Debug, Clone)]
pub struct ForceHV {
    /// HDC encoding of force intent (type ⊗ direction)
    pub force_vector: ContinuousHV,
    /// Force magnitude in Newtons
    pub magnitude: f32,
    /// Application point in body coordinates
    pub application_point: [f32; 3],
    /// Expected resistance (what she expects to "feel")
    pub expected_resistance: f32,
}

impl ForceHV {
    /// FEP surprise: |expected - actual| / expected
    pub fn surprise(&self, actual_resistance: f32) -> f32 {
        if self.expected_resistance.abs() < 1e-10 {
            return actual_resistance.abs();
        }
        ((self.expected_resistance - actual_resistance).abs() / self.expected_resistance.abs())
            .min(10.0) // Cap at 10x surprise
    }
}

/// Simulation state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimState {
    pub time: f32,
    pub positions: Vec<[f32; 3]>,
    pub velocities: Vec<[f32; 3]>,
    pub total_energy: f32,
}

/// Contact point between bodies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactPoint {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub force_magnitude: f32,
}

/// Deformation field for stress analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeformationField {
    pub displacements: Vec<[f32; 3]>,
    pub strains: Vec<f32>,
}

impl DeformationField {
    pub fn max_strain(&self) -> f32 {
        self.strains.iter().copied().fold(0.0f32, f32::max)
    }

    pub fn avg_displacement(&self) -> f32 {
        if self.displacements.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .displacements
            .iter()
            .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
            .sum();
        sum / self.displacements.len() as f32
    }
}

/// Physics simulation backend trait
pub trait PhysicsBackend: Send + Sync {
    /// Step the simulation forward by dt seconds
    fn step(&mut self, dt: f32, forces: &[ForceHV]) -> SimState;
    /// Get current contact points
    fn get_contacts(&self) -> Vec<ContactPoint>;
    /// Get current deformation field (if available)
    fn get_deformation(&self) -> Option<DeformationField>;
    /// Get reaction force at a point (for surprise calculation)
    fn get_reaction_force(&self, point: [f32; 3]) -> f32;
    /// Reset to initial state
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surprise_zero() {
        let force = ForceHV {
            force_vector: tension_hv(),
            magnitude: 100.0,
            application_point: [0.0, 0.0, 0.0],
            expected_resistance: 50.0,
        };
        let surprise = force.surprise(50.0);
        assert!(surprise < 0.001, "No surprise when actual matches expected");
    }

    #[test]
    fn test_surprise_double() {
        let force = ForceHV {
            force_vector: tension_hv(),
            magnitude: 100.0,
            application_point: [0.0, 0.0, 0.0],
            expected_resistance: 50.0,
        };
        let surprise = force.surprise(100.0);
        assert!(
            (surprise - 1.0).abs() < 0.001,
            "Double expected = 1.0 surprise"
        );
    }

    #[test]
    fn test_force_type_orthogonality() {
        let t = tension_hv();
        let c = compression_hv();
        let s = shear_hv();
        assert!(t.similarity(&c).abs() < 0.1);
        assert!(t.similarity(&s).abs() < 0.1);
        assert!(c.similarity(&s).abs() < 0.1);
    }

    #[test]
    fn test_axis_orthogonality() {
        let x = x_axis_hv();
        let y = y_axis_hv();
        let z = z_axis_hv();
        assert!(x.similarity(&y).abs() < 0.1);
        assert!(x.similarity(&z).abs() < 0.1);
        assert!(y.similarity(&z).abs() < 0.1);
    }

    #[test]
    fn test_compose_force() {
        let f = compose_force(&tension_hv(), &x_axis_hv());
        // Composed force should be dissimilar to both components
        assert!(f.similarity(&tension_hv()).abs() < 0.15);
        assert!(f.similarity(&x_axis_hv()).abs() < 0.15);
    }

    #[test]
    fn test_deformation_field() {
        let df = DeformationField {
            displacements: vec![[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
            strains: vec![0.001, 0.005, 0.002],
        };
        assert!((df.max_strain() - 0.005).abs() < 1e-6);
        let avg = df.avg_displacement();
        assert!(avg > 0.0);
    }

    #[test]
    fn test_surprise_capped() {
        let force = ForceHV {
            force_vector: tension_hv(),
            magnitude: 100.0,
            application_point: [0.0, 0.0, 0.0],
            expected_resistance: 1.0,
        };
        let surprise = force.surprise(1000.0);
        assert!(surprise <= 10.0, "Surprise should be capped at 10.0");
    }
}
