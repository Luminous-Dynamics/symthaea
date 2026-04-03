// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HDC Consciousness Context — 16,384D holographic thought vectors for NPCs.
//!
//! Replaces the scalar MasterConsciousnessEquation with genuine cognitive
//! dynamics: each NPC carries a high-dimensional thought vector evolved by
//! a unified HDC-LTC neural network. Phi becomes an emergent property of
//! thought-harmony resonance, not a hand-tuned equation.
//!
//! # Architecture
//!
//! ```text
//! 7 scalar consciousness components
//!   ↓ encode as weighted basis HVs
//! input_hv (16,384D)
//!   ↓ HdcLtcUnifiedNetwork::evolve_closed_form(dt)
//! thought_hv (16,384D) ← has temporal memory via LTC dynamics
//!   ↓ similarity with harmony superposition
//! phi ∈ [0, 1] (emergent, not computed from equation)
//! ```
//!
//! # Integration
//!
//! Feature-gated: `consciousness-hdc`. When enabled, `EntityConsciousness`
//! carries an optional `HdcConsciousnessContext`. When present, `phi()`
//! returns the HDC-derived value instead of the scalar equation.

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};

/// Number of consciousness components (phi, broadcast, working_memory,
/// attention, recurrence, embodiment, knowledge).
const NUM_COMPONENTS: usize = 7;

/// Number of harmonies.
const NUM_HARMONIES: usize = 8;

/// HDC dimension for thought vectors.
const HDC_DIM: usize = 16_384;

/// Holographic consciousness context for a single entity.
///
/// Carries a 16,384D thought vector evolved by a multi-layer LTC network.
/// The thought vector's similarity with the harmony superposition produces
/// an emergent Phi value.
pub struct HdcConsciousnessContext {
    /// Current thought hypervector (output of encoder network).
    pub thought_hv: ContinuousHV,
    /// Multi-layer encoder network (7 → 8 → 4 neurons).
    encoder: HdcLtcUnifiedNetwork,
    /// Basis HVs for the 7 consciousness components.
    consciousness_basis: Vec<ContinuousHV>,
    /// Basis HVs for the 8 harmonies.
    harmony_basis: Vec<ContinuousHV>,
}

impl HdcConsciousnessContext {
    /// Create a new context with deterministic random basis vectors.
    ///
    /// Different seeds produce different "cognitive personalities" — each NPC
    /// develops unique thought patterns from the same inputs.
    pub fn new(seed: u64) -> Self {
        // Generate basis HVs for consciousness components.
        let consciousness_basis: Vec<ContinuousHV> = (0..NUM_COMPONENTS)
            .map(|i| ContinuousHV::random(HDC_DIM, seed.wrapping_add(i as u64 * 7919)))
            .collect();

        // Generate basis HVs for harmonies.
        let harmony_basis: Vec<ContinuousHV> = (0..NUM_HARMONIES)
            .map(|i| ContinuousHV::random(HDC_DIM, seed.wrapping_add(1000 + i as u64 * 6271)))
            .collect();

        // Create encoder network: 7 input neurons → 8 hidden → 4 output.
        let config = UnifiedNetworkConfig {
            neuron_config: UnifiedConfig {
                dimension: HDC_DIM,
                tau_base: 0.1,       // 100ms time constant
                backbone_tau: 0.5,   // state-dependent scaling
                ..UnifiedConfig::default()
            },
            layer_sizes: vec![7, 8, 4],
            use_layer_binding: true,
            skip_connections: true,
        };
        let encoder = HdcLtcUnifiedNetwork::new(config, seed.wrapping_add(9999));

        Self {
            thought_hv: ContinuousHV::zero(HDC_DIM),
            encoder,
            consciousness_basis,
            harmony_basis,
        }
    }

    /// Step the consciousness context: encode inputs, evolve network, update thought HV.
    ///
    /// `inputs`: 7 scalar consciousness components [phi, broadcast, wm, attention, recurrence, embodiment, knowledge]
    /// `harmony`: 8 harmony activations [0, 1]
    /// `dt`: time delta in seconds
    pub fn step(&mut self, inputs: &[f64; NUM_COMPONENTS], harmony: &[f64; NUM_HARMONIES], dt: f32) {
        // Encode scalar inputs as weighted basis HVs bundled together.
        let scaled: Vec<ContinuousHV> = self
            .consciousness_basis
            .iter()
            .zip(inputs.iter())
            .map(|(basis, &val)| basis.scale(val as f32))
            .collect();
        let refs: Vec<&ContinuousHV> = scaled.iter().collect();
        let input_hv = ContinuousHV::bundle(&refs);

        // Evolve the encoder network with closed-form LTC dynamics.
        self.encoder.evolve_closed_form(dt, &input_hv);

        // Extract thought vector from network output.
        self.thought_hv = self.encoder.output();
    }

    /// Extract emergent Phi from the thought vector.
    ///
    /// Phi = similarity(thought_hv, harmony_superposition) × complexity
    /// where complexity = thought_hv.norm() (bounded to [0, 2]).
    ///
    /// This makes Phi a genuine emergent property: it measures how well
    /// the agent's cognitive state resonates with its harmony activations.
    pub fn phi_from_thought(&self, harmony: &[f64; NUM_HARMONIES]) -> f64 {
        // Build harmony superposition: weighted bundle of harmony basis HVs.
        let scaled: Vec<ContinuousHV> = self
            .harmony_basis
            .iter()
            .zip(harmony.iter())
            .map(|(basis, &act)| basis.scale(act as f32))
            .collect();
        let refs: Vec<&ContinuousHV> = scaled.iter().collect();
        let harmony_hv = ContinuousHV::bundle(&refs);

        // Similarity between thought and harmony superposition.
        let resonance = self.thought_hv.similarity(&harmony_hv);

        // Complexity: how rich is the thought state? (norm as proxy)
        let complexity = (self.thought_hv.norm().min(2.0) / 2.0) as f64;

        // Emergent Phi: resonance × complexity.
        (resonance as f64 * complexity).clamp(0.0, 1.0)
    }

    /// Get the thought hypervector for inter-agent resonance computation.
    pub fn thought_hv(&self) -> &ContinuousHV {
        &self.thought_hv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_context() {
        let ctx = HdcConsciousnessContext::new(42);
        assert_eq!(ctx.thought_hv.values.len(), HDC_DIM);
        assert_eq!(ctx.consciousness_basis.len(), NUM_COMPONENTS);
        assert_eq!(ctx.harmony_basis.len(), NUM_HARMONIES);
    }

    #[test]
    fn phi_in_valid_range() {
        let mut ctx = HdcConsciousnessContext::new(42);
        let inputs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let harmony = [0.5; 8];

        // Step a few times to build state.
        for _ in 0..10 {
            ctx.step(&inputs, &harmony, 0.016);
        }

        let phi = ctx.phi_from_thought(&harmony);
        assert!(phi >= 0.0 && phi <= 1.0, "Phi should be in [0,1]: {phi}");
    }

    #[test]
    fn different_seeds_different_thoughts() {
        let mut ctx_a = HdcConsciousnessContext::new(42);
        let mut ctx_b = HdcConsciousnessContext::new(99);

        let inputs = [0.7; 7];
        let harmony = [0.5; 8];

        for _ in 0..5 {
            ctx_a.step(&inputs, &harmony, 0.016);
            ctx_b.step(&inputs, &harmony, 0.016);
        }

        // Different seeds should produce different thought vectors.
        let sim = ctx_a.thought_hv.similarity(&ctx_b.thought_hv);
        assert!(
            sim < 0.9,
            "Different seeds should produce distinct thoughts: similarity={sim}"
        );
    }

    #[test]
    fn zero_inputs_low_phi() {
        let mut ctx = HdcConsciousnessContext::new(42);
        let inputs = [0.0; 7];
        let harmony = [0.0; 8];

        for _ in 0..10 {
            ctx.step(&inputs, &harmony, 0.016);
        }

        let phi = ctx.phi_from_thought(&harmony);
        // Zero inputs + zero harmony = minimal thought-harmony resonance.
        assert!(phi < 0.5, "Zero inputs should produce low phi: {phi}");
    }

    #[test]
    fn temporal_memory() {
        let mut ctx = HdcConsciousnessContext::new(42);
        let harmony = [0.5; 8];

        // Step with high inputs.
        for _ in 0..20 {
            ctx.step(&[0.9; 7], &harmony, 0.016);
        }
        let thought_after_high = ctx.thought_hv.clone();

        // Step with zero inputs — thought should change but retain some memory.
        ctx.step(&[0.0; 7], &harmony, 0.016);
        let thought_after_zero = &ctx.thought_hv;

        let sim = thought_after_high.similarity(thought_after_zero);
        // LTC dynamics: state should retain memory (not instantly reset).
        assert!(
            sim > 0.3,
            "Thought should retain temporal memory: similarity={sim}"
        );
    }
}
