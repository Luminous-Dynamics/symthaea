// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Biome Tensor — Goodhart-resistant ecological health monitoring.
//!
//! Replaces single-scalar metrics with a high-dimensional state vector
//! representing the homeostasis of a living ecosystem.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

/// Holistic state of a bioregion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemState {
    /// Vegetation health (NDVI: 0.0 - 1.0) - From Sentinel-2 (Skin)
    pub canopy_cover: f64,
    /// Structural volume/surface roughness - From Sentinel-1 (SAR)
    pub structural_biomass: f64,
    /// Soil moisture (0.0 - 1.0) - Fused SAR/Probe
    pub soil_moisture: f64,
    /// Ground water pH (0.0 - 1.0 normalized)
    pub water_ph: f64,
    /// Ambient temperature variance (0.0 - 1.0 normalized)
    pub temp_stability: f64,
    /// Bio-acoustic entropy (0.0 - 1.0: measure of biodiversity)
    pub acoustic_entropy: f64,

    // --- Fractal Boundary Flows (Phase 13) ---
    /// Water received from upstream (normalized)
    pub upstream_flow_in: f64,
    /// Water passed to downstream (normalized)
    pub downstream_flow_out: f64,
}

/// The Biome Tensor: A 16,384-bit HDC representation of ecological health.
pub struct BiomeTensor {
    pub vector: ContinuousHV,
}

pub struct BiomeEncoder {
    dimension: usize,
}

impl BiomeEncoder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Encode the multi-variable EcosystemState into a single BiomeTensor.
    /// Uses ContinuousHV mapping to ensure similar environmental states
    /// produce similar vectors.
    pub fn encode(&self, state: &EcosystemState) -> BiomeTensor {
        let mut bundle = Vec::new();

        // 1. Encode Canopy (Optical Skin)
        let canopy_hv =
            ContinuousHV::from_vec(vec![state.canopy_cover as f32; 64]).dilate(self.dimension);
        bundle.push(canopy_hv);

        // 2. Encode Structural Biomass (Radar Structure)
        let structural_hv = ContinuousHV::from_vec(vec![state.structural_biomass as f32; 64])
            .dilate(self.dimension);
        bundle.push(structural_hv);

        // 3. Encode Soil Moisture
        let moisture_hv =
            ContinuousHV::from_vec(vec![state.soil_moisture as f32; 64]).dilate(self.dimension);
        bundle.push(moisture_hv);

        // 4. Encode Water pH
        let ph_hv = ContinuousHV::from_vec(vec![state.water_ph as f32; 64]).dilate(self.dimension);
        bundle.push(ph_hv);

        // 5. Encode Temperature Stability
        let temp_hv =
            ContinuousHV::from_vec(vec![state.temp_stability as f32; 64]).dilate(self.dimension);
        bundle.push(temp_hv);

        // 6. Encode Acoustic Entropy (The Biodiversity Anchor)
        let audio_hv =
            ContinuousHV::from_vec(vec![state.acoustic_entropy as f32; 64]).dilate(self.dimension);
        bundle.push(audio_hv);

        let refs: Vec<&ContinuousHV> = bundle.iter().collect();
        let bundled = ContinuousHV::bundle(&refs);

        BiomeTensor { vector: bundled }
    }

    /// Calculate the geometric similarity (Hamming distance proxy) between
    /// the current state and a target healthy biome.
    pub fn calculate_restoration_progress(
        &self,
        current: &BiomeTensor,
        target: &BiomeTensor,
    ) -> f64 {
        // High similarity = close to target homeostasis
        current.vector.similarity(&target.vector) as f64
    }
}
