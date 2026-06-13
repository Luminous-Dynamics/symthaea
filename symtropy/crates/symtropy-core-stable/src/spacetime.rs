// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Holographic spacetime physics field.

use nalgebra::Vector3;
use crate::hdc::{ContinuousHV, HDC_DIM};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacetimeCrystalField {
    pub global_cosmic_vector: ContinuousHV,
    pub slope: f64,
}

impl Default for SpacetimeCrystalField {
    fn default() -> Self {
        Self {
            global_cosmic_vector: ContinuousHV::zero(HDC_DIM),
            slope: 1.61803398875,
        }
    }
}

impl SpacetimeCrystalField {
    pub fn inject_mass(&mut self, pos: Vector3<f64>, mass: f64) {
        let p = pos.x * self.slope + pos.y * self.slope.sqrt() + pos.z;
        let sig = ContinuousHV::random(HDC_DIM, (p.abs() * 1e6) as u64);
        let w = (mass.log10() / 30.0).clamp(0.0, 1.0) as f32;
        self.global_cosmic_vector.lerp_in_place(&sig, 1.0 - w, w);
        self.global_cosmic_vector.l2_normalize();
    }

    pub fn probe(&self, pos: Vector3<f64>) -> f32 {
        let p = pos.x * self.slope + pos.y * self.slope.sqrt() + pos.z;
        let vec = ContinuousHV::random(HDC_DIM, (p.abs() * 1e6) as u64);
        self.global_cosmic_vector.similarity(&vec)
    }

    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = serde_json::to_string_pretty(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let decoded = serde_json::from_str(&content)?;
        Ok(decoded)
    }
}
