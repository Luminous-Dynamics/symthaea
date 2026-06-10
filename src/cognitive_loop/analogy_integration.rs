// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Analogy engine integration for the cognitive loop.
//!
//! Wraps `symthaea_core::physics::AnalogyEngine` into a cycle-compatible
//! interface with interval-gated queries and compressed_state blending.
//! Follows the same pattern as `physics_integration.rs`.

#[cfg(feature = "analogy-engine")]
use symthaea_core::hdc::ContinuousHV;
#[cfg(feature = "analogy-engine")]
use symthaea_core::physics::{Analogy, AnalogyEngine};

/// Telemetry snapshot from the analogy engine.
#[derive(Debug, Clone, Default)]
pub struct AnalogyTelemetry {
    pub concepts_count: usize,
    pub analogies_found: usize,
    pub top_similarity: f32,
    pub top_concept_a: String,
    pub top_concept_b: String,
    pub queried_this_cycle: bool,
    pub effective_interval: usize,
    pub effective_blend_weight: f32,
}

#[cfg(feature = "analogy-engine")]
pub(crate) struct AnalogyIntegration {
    engine: AnalogyEngine,
    last_analogy_hv: Option<ContinuousHV>,
    last_analogies: Vec<Analogy>,
    queried_this_cycle: bool,
    last_effective_interval: usize,
    last_effective_blend: f32,
    pub(crate) last_top_similarity: f32,
}

#[cfg(feature = "analogy-engine")]
impl std::fmt::Debug for AnalogyIntegration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnalogyIntegration")
            .field("queried_this_cycle", &self.queried_this_cycle)
            .field("has_last_hv", &self.last_analogy_hv.is_some())
            .field("last_effective_interval", &self.last_effective_interval)
            .field("last_effective_blend", &self.last_effective_blend)
            .field("last_top_similarity", &self.last_top_similarity)
            .field("analogies_cached", &self.last_analogies.len())
            .finish()
    }
}

#[cfg(feature = "analogy-engine")]
impl AnalogyIntegration {
    pub fn new(genesis: &symthaea_core::genesis::GenesisSeed) -> Self {
        Self {
            engine: AnalogyEngine::from_genesis(genesis),
            last_analogy_hv: None,
            last_analogies: Vec::new(),
            queried_this_cycle: false,
            last_effective_interval: 0,
            last_effective_blend: 0.0,
            last_top_similarity: 0.0,
        }
    }

    /// Collect telemetry and reset per-cycle flag.
    pub fn telemetry(&mut self) -> AnalogyTelemetry {
        let (top_a, top_b, top_sim) = self
            .last_analogies
            .first()
            .map(|a| {
                (
                    a.concept_a.clone(),
                    a.concept_b.clone(),
                    a.similarity as f32,
                )
            })
            .unwrap_or_default();
        let telem = AnalogyTelemetry {
            concepts_count: self.engine.list_patterns().len(),
            analogies_found: self.last_analogies.len(),
            top_similarity: top_sim,
            top_concept_a: top_a,
            top_concept_b: top_b,
            queried_this_cycle: self.queried_this_cycle,
            effective_interval: self.last_effective_interval,
            effective_blend_weight: self.last_effective_blend,
        };
        self.queried_this_cycle = false;
        telem
    }

    /// Per-cycle analogy query with interval checking and state blending.
    ///
    /// Discovers cross-domain analogies from the input thought vector and
    /// blends the strongest analogy's concept vector into compressed_state.
    ///
    /// Returns true if a query was issued.
    pub fn query_cycle(
        &mut self,
        cycle_number: usize,
        base_interval: usize,
        base_blend_weight: f32,
        substrate_tau_factor: f32,
        substrate_scale_pressure: f32,
        hv16: &symthaea_core::hdc::BinaryHV,
        compressed_state: &mut [f32],
    ) -> bool {
        let tau = substrate_tau_factor.max(0.01);
        let effective_interval = ((base_interval as f32 / tau).round() as usize).max(1);
        let effective_blend =
            (base_blend_weight * tau * (1.0 + substrate_scale_pressure.clamp(0.0, 1.0) * 0.1))
                .clamp(0.0, 1.0);

        self.last_effective_interval = effective_interval;
        self.last_effective_blend = effective_blend;

        if effective_interval == 0 || cycle_number % effective_interval != 0 {
            return false;
        }

        self.queried_this_cycle = true;

        // Find the nearest concept to the input thought
        let query_hv = hv16.to_continuous();
        let nearest = self.engine.nearest_concept(&query_hv);
        let (concept, concept_sim) = match nearest {
            Some((c, sim)) => (c, sim),
            None => return true, // queried but no concepts available
        };
        let concept_name = concept.name.clone();
        let concept_vector = concept.vector.clone();

        // Discover cross-domain analogies for the nearest concept
        let analogies = self.engine.find_analogies(&concept_name, 0.3);
        self.last_analogies = analogies;
        self.last_top_similarity = concept_sim as f32;

        // Blend the nearest concept's HV into compressed_state
        let w = effective_blend * (concept_sim as f32).clamp(0.0, 1.0);
        self.last_effective_blend = w;
        let concept_vals = &concept_vector.values;
        let stride = if concept_vals.len() > compressed_state.len() {
            concept_vals.len() / compressed_state.len()
        } else {
            1
        };
        for (i, state_val) in compressed_state.iter_mut().enumerate() {
            let analogy_component = concept_vals.get(i * stride).copied().unwrap_or(0.0);
            *state_val = (1.0 - w) * *state_val + w * analogy_component;
        }
        self.last_analogy_hv = Some(concept_vector);

        true
    }
}

#[cfg(all(test, feature = "analogy-engine"))]
mod tests {
    use super::*;
    use symthaea_core::genesis::GenesisSeed;

    #[test]
    fn test_analogy_integration_creates() {
        let genesis = GenesisSeed::from_phrase("test analogy");
        let ai = AnalogyIntegration::new(&genesis);
        assert!(!ai.queried_this_cycle);
        assert!(ai.last_analogy_hv.is_none());
    }

    #[test]
    fn test_analogy_telemetry_default() {
        let genesis = GenesisSeed::from_phrase("test analogy");
        let mut ai = AnalogyIntegration::new(&genesis);
        let telem = ai.telemetry();
        assert!(!telem.queried_this_cycle);
        assert!(telem.concepts_count > 0); // pre-populated concepts
    }

    #[test]
    fn test_analogy_interval_skips() {
        let genesis = GenesisSeed::from_phrase("test analogy");
        let mut ai = AnalogyIntegration::new(&genesis);
        let hv = symthaea_core::hdc::BinaryHV::random();
        let mut state = vec![0.0f32; 128];
        // Cycle 1 with interval 11 should not query
        let queried = ai.query_cycle(1, 11, 0.1, 1.0, 0.0, &hv, &mut state);
        assert!(!queried);
    }

    #[test]
    fn test_analogy_fires_on_interval() {
        let genesis = GenesisSeed::from_phrase("test analogy");
        let mut ai = AnalogyIntegration::new(&genesis);
        let hv = symthaea_core::hdc::BinaryHV::random();
        let mut state = vec![0.0f32; 128];
        // Cycle 11 with interval 11 should query
        let queried = ai.query_cycle(11, 11, 0.1, 1.0, 0.0, &hv, &mut state);
        assert!(queried);
        assert!(ai.queried_this_cycle);
    }
}
