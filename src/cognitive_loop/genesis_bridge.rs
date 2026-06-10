// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Genesis Bridge — From Genomics to Cognition
//!
//! Connects the `symthaea-population` genomics data to the core cognitive loop.
//! Maps genetic traits to "innate" hypervectors and neuromodulator baselines.

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
#[cfg(feature = "population")]
use symthaea_population::types::Individual;

/// Innate cognitive traits derived from an individual's genome.
#[derive(Debug, Clone)]
pub struct InnateTraits {
    /// Innate curiosity bias (derived from Dopamine-related loci).
    pub curiosity_baseline: f32,
    /// Innate social trust bias (derived from Oxytocin/Serotonin loci).
    pub trust_baseline: f32,
    /// Innate resilience (derived from Cortisol/Stress-response loci).
    pub resilience_factor: f32,
    /// Bundled genome hypervector representing the "self" identity.
    pub genome_self_hv: ContinuousHV,
}

#[cfg(feature = "population")]
impl InnateTraits {
    /// Derive innate traits from an individual genome.
    pub fn from_individual(individual: &Individual) -> Self {
        // Map genome similarity to baseline shifts
        // (Simplified for Phase 11: uses checksum of genome_hv as proxy for specific alleles)
        let genome_sum: f32 = individual.genome_hv.values.iter().sum();
        let genome_abs_sum: f32 = individual.genome_hv.values.iter().map(|v| v.abs()).sum();

        // Pseudo-random but deterministic mapping [0, 1]
        let h_val = (genome_sum.cos() * 0.5 + 0.5).clamp(0.0, 1.0);
        let s_val = (genome_abs_sum.sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let r_val = ((genome_sum + genome_abs_sum).cos() * 0.5 + 0.5).clamp(0.0, 1.0);

        Self {
            curiosity_baseline: 0.1 + h_val * 0.4, // [0.1, 0.5]
            trust_baseline: 0.3 + s_val * 0.4,     // [0.3, 0.7]
            resilience_factor: 0.5 + r_val * 0.5,  // [0.5, 1.0]
            genome_self_hv: individual.genome_hv.clone(),
        }
    }

    /// Apply innate traits to the CognitiveLoopService.
    pub fn apply(&self, service: &mut crate::cognitive_loop::CognitiveLoopService) {
        // Set innate curiosity (boredom baseline)
        service.behavior.curiosity_drive.boredom = self.curiosity_baseline;

        // Set innate social trust
        service.behavior.social_mgr.social.social_trust = self.trust_baseline;
        service.behavior.social_mgr.social.social_mean_trust = self.trust_baseline;

        // Injected genome_self_hv could be used for identity-based similarity gating
        // in future phases.
    }
}

impl InnateTraits {
    /// Emit a developmental signal based on current cognitive state.
    #[cfg(feature = "cell-foundry")]
    pub fn emit_developmental_signal(
        &self,
        allostatic_load: f32,
        integration_peak: f32,
        dt_hours: f32,
    ) -> symthaea_cell_foundry::types::DevelopmentalSignal {
        symthaea_cell_foundry::types::DevelopmentalSignal {
            allostatic_load,
            integration_peak,
            dt_hours,
        }
    }
}

impl Default for InnateTraits {
    fn default() -> Self {
        Self {
            curiosity_baseline: 0.3,
            trust_baseline: 0.5,
            resilience_factor: 0.7,
            genome_self_hv: ContinuousHV::random(HDC_DIMENSION, 0x6E65_7369_73), // "genesis"
        }
    }
}
