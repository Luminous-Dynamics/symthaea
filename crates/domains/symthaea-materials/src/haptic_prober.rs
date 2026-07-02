// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Haptic Material Prober — Atomic Composition Identification.
//!
//! Bridges the 64-DOF humanoid's finger metrology (Torque, Displacement, Temp)
//! to the MaterialProperty database, enabling autonomous identification
//! of scrap materials for sovereign recycling.

use crate::properties::{MaterialCategory, MaterialProperty};
use anyhow::Result;

/// Estimates material properties from haptic sensory input.
pub struct HapticMaterialProber;

impl HapticMaterialProber {
    /// Estimate engineering properties from a calibrated haptic pinch.
    /// - torque_nm: Peak motor torque applied before deformation
    /// - displacement_m: Displacement of the joint
    /// - temp_delta_k: Rate of temperature change during contact
    pub fn estimate_properties(
        &self,
        torque_nm: f32,
        displacement_m: f32,
        temp_delta_k: f32,
    ) -> Result<MaterialProperty> {
        // Simple heuristic mapping for Phase 25 v0
        // Stiffness (E) ~ Torque / Displacement
        let youngs_modulus_estimate = (torque_nm / displacement_m.max(1e-6)) / 1e9; // Simplified GPa

        // Thermal conductivity (K) ~ 1 / Temp Delta (High conductivity = low delta)
        let thermal_cond_estimate = 1.0 / temp_delta_k.max(0.01);

        Ok(MaterialProperty {
            name: "Haptic-Estimated-Asset".into(),
            category: if youngs_modulus_estimate > 50.0 {
                MaterialCategory::Metal
            } else {
                MaterialCategory::Polymer
            },
            density_kg_m3: 0.0, // Cannot be estimated via pinch alone (requires volume/mass)
            youngs_modulus_gpa: youngs_modulus_estimate,
            yield_strength_mpa: 0.0,
            thermal_conductivity_w_mk: thermal_cond_estimate,
            specific_heat_j_kgk: 0.0,
            melting_point_c: 0.0,
            corrosion_resistance: 0.5,
            fatigue_limit_mpa: 0.0,
        })
    }

    /// Identify the material by searching the local database for the nearest neighbor.
    pub fn identify(
        &self,
        estimated: &MaterialProperty,
        database: &[MaterialProperty],
    ) -> Option<MaterialProperty> {
        let est_vals = estimated.normalized_values();

        database
            .iter()
            .min_by(|a, b| {
                let da = self.euclidean_distance(&est_vals, &a.normalized_values());
                let db = self.euclidean_distance(&est_vals, &b.normalized_values());
                da.total_cmp(&db)
            })
            .cloned()
    }

    fn euclidean_distance(&self, a: &[f32; 8], b: &[f32; 8]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}
