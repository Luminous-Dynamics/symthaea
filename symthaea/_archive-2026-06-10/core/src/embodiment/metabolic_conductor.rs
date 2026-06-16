// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metabolic Conductor — Homeostatic coordination of body metabolism.
//!
//! Orchestrates the energy (Helios) and material (Detritivore) bridges
//! to ensure the Mk0 Bootstrapper Protocol maintains physiological grounding.

use crate::embodiment::detritivore_bridge::DetritivoreEmbodiment;
use crate::embodiment::helios_bridge::HeliosEmbodiment;
use symthaea_core::embodiment::{EmbodimentBridge, MotorSafetyLevel};
use symthaea_core::hdc::ContinuousHV;

/// Physiological state of the Mk0 organism.
#[derive(Debug, Clone, Default)]
pub struct MetabolicHomeostasis {
    /// Normalized energy level [0, 1]
    pub energy_level: f32,
    /// Normalized material feedstock level [0, 1]
    pub material_level: f32,
    /// Thermal stability [0, 1] (1.0 = optimal)
    pub thermal_stability: f32,
    /// Combined metabolic surprise (prediction error)
    pub surprise: f32,
}

/// Conductor for coordinating multi-bridge metabolic grounding.
pub struct MetabolicConductor {
    pub helios: HeliosEmbodiment,
    pub detritivore: DetritivoreEmbodiment,
    last_homeostasis: MetabolicHomeostasis,
}

impl MetabolicConductor {
    /// Create a new conductor from its constituent bridges.
    pub fn new(helios: HeliosEmbodiment, detritivore: DetritivoreEmbodiment) -> Self {
        Self {
            helios,
            detritivore,
            last_homeostasis: MetabolicHomeostasis::default(),
        }
    }

    /// Update homeostasis by polling constituent bridges.
    pub fn tick(&mut self) -> MetabolicHomeostasis {
        let _helios_perception = self.helios.encode_perception();
        let _detritivore_perception = self.detritivore.encode_perception();

        // 1. Calculate Energy Level from Helios
        let energy = self.helios.telemetry.battery_soc;

        // 2. Calculate Material Level from Detritivore
        // For simplicity, we normalize a 5kg bin capacity.
        let material = (self.detritivore.telemetry.feedstock_mass / 5000.0).min(1.0);

        // 3. Thermal Stability (Optimal = 200C for extrusion, roughly)
        let temp = self.detritivore.telemetry.extruder_temp;
        let thermal = if temp > 180.0 && temp < 240.0 {
            1.0
        } else {
            0.5
        };

        let homeostasis = MetabolicHomeostasis {
            energy_level: energy,
            material_level: material,
            thermal_stability: thermal,
            surprise: 0.0, // Calculated via FEP agent elsewhere
        };

        self.last_homeostasis = homeostasis.clone();
        homeostasis
    }

    /// Determine if the organism has enough "metabolic surplus" to perform
    /// a high-energy material shredding cycle.
    pub fn should_perform_shredding(&self) -> bool {
        // Requirements: Battery > 40%, Temp stable, Feedstock > 100g
        self.last_homeostasis.energy_level > 0.4
            && self.last_homeostasis.thermal_stability > 0.8
            && self.detritivore.telemetry.feedstock_mass > 100.0
    }

    /// Produce a unified "Physiological Surprise" hypervector for the cognitive core.
    ///
    /// This vector encodes the delta between current homeostasis and optimal
    /// baseline, allowing the Brain to "feel" metabolic stress.
    pub fn encode_metabolic_stress(&mut self) -> ContinuousHV {
        let dim = 16384;
        let mut bundle = Vec::new();

        // 1. Energy Stress
        if self.last_homeostasis.energy_level < 0.3 {
            bundle.push(ContinuousHV::random(dim, 1111)); // Role: Energy Hunger
        }

        // 2. Material Hunger
        if self.last_homeostasis.material_level < 0.2 {
            bundle.push(ContinuousHV::random(dim, 2222)); // Role: Feedstock Hunger
        }

        if bundle.is_empty() {
            return ContinuousHV::zero(dim);
        }

        let refs: Vec<&ContinuousHV> = bundle.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Get current safety level across all metabolic systems.
    pub fn safety_level(&self) -> MotorSafetyLevel {
        let h_safety = self.helios.safety_level();
        let d_safety = self.detritivore.safety_level();

        match (h_safety, d_safety) {
            (MotorSafetyLevel::Red, _) | (_, MotorSafetyLevel::Red) => MotorSafetyLevel::Red,
            (MotorSafetyLevel::Yellow, _) | (_, MotorSafetyLevel::Yellow) => {
                MotorSafetyLevel::Yellow
            }
            _ => MotorSafetyLevel::Green,
        }
    }
}
