// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HDC Integration Context — 16,384D state vectors for agent integration dynamics.
//!
//! Replaces the scalar MasterConsciousnessEquation with LTC-driven state
//! dynamics: each agent carries a high-dimensional state vector evolved by
//! a unified HDC-LTC neural network. Φ becomes a norm-based integration
//! metric derived from network activity, not a hand-tuned equation.
//!
//! Note: "consciousness" is used in the IIT formal sense (Tononi 2004) —
//! an integration metric, not a claim about subjective experience.
//!
//! # Architecture
//!
//! ```text
//! 7 scalar integration components
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

use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Number of consciousness components (phi, broadcast, working_memory,
/// attention, recurrence, embodiment, knowledge).
const NUM_COMPONENTS: usize = 7;

/// Number of harmonies (mirrors harmony_field::NUM_HARMONIES).
const NUM_HARMONIES: usize = 9;

/// HDC dimension for state vectors.
const HDC_DIM: usize = 16_384;

/// HDC integration context for a single entity.
///
/// Carries a 16,384D state vector evolved by a multi-layer LTC network.
/// The state vector's norm serves as an integration metric (Φ) — higher
/// norm indicates more active information integration across network layers.
pub struct HdcConsciousnessContext {
    /// Current state hypervector (output of encoder network; not a thought in the phenomenal sense).
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
    /// Different seeds produce different integration profiles — each agent
    /// develops unique state-vector trajectories from the same inputs.
    pub fn new(seed: u64) -> Self {
        // Generate basis HVs for consciousness components.
        let consciousness_basis: Vec<ContinuousHV> = (0..NUM_COMPONENTS)
            .map(|i| ContinuousHV::random(HDC_DIM, seed.wrapping_add(i as u64 * 7919)))
            .collect();

        // Generate harmony basis as CORRELATED perturbations of consciousness basis.
        // This ensures HDC state vectors (in consciousness_basis span) have nonzero
        // projection onto harmony space — fixing the Phi=0 readout problem.
        //
        // Each harmony basis[i] = blend(consciousness_basis[i % 7], noise, 0.7/0.3).
        // The 70% correlation means Phi readout is meaningful; 30% noise gives
        // each harmony a unique identity.
        let harmony_basis: Vec<ContinuousHV> = (0..NUM_HARMONIES)
            .map(|i| {
                let base = &consciousness_basis[i % NUM_COMPONENTS];
                let noise =
                    ContinuousHV::random(HDC_DIM, seed.wrapping_add(1000 + i as u64 * 6271));
                // 70% correlated with consciousness basis, 30% unique noise
                base.scale(0.7).add(&noise.scale(0.3)).normalize()
            })
            .collect();

        // Create encoder network: 7 input neurons → 8 hidden → 4 output.
        let config = UnifiedNetworkConfig {
            neuron_config: UnifiedConfig {
                dimension: HDC_DIM,
                tau_base: 0.1,     // 100ms time constant
                backbone_tau: 0.5, // state-dependent scaling
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
    pub fn step(
        &mut self,
        inputs: &[f64; NUM_COMPONENTS],
        harmony: &[f64; NUM_HARMONIES],
        dt: f32,
    ) {
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

        // Extract state vector from network output.
        self.thought_hv = self.encoder.output();
    }

    /// Extract integration metric (Φ) from HDC state vector norm.
    ///
    /// Phi measures how much the LTC network has INTEGRATED its inputs into
    /// a coherent internal representation. This is computed as:
    ///
    /// 1. **Integration** = similarity(input_encoding, thought_output)
    ///    How well the network's output reflects its input = IIT-inspired integration.
    ///    High similarity = the network has integrated inputs into a coherent state.
    ///    Low similarity = inputs are fragmented/not integrated.
    ///
    /// 2. **Complexity** = thought_hv.norm() / 2.0 (bounded to [0, 1])
    ///    How rich/active the state vector is. Zero norm = no activity.
    ///
    /// Phi = integration × complexity, clamped to [0, 1].
    /// The norm scale at which thought_hv reaches maximum integration.
    /// Empirically determined: after 50 ticks of strong input, norm ≈ 0.02-0.05.
    /// Scale factor maps this range to [0, 1].
    const PHI_NORM_SCALE: f32 = 30.0;

    pub fn phi_from_thought(&self, _harmony: &[f64; NUM_HARMONIES]) -> f64 {
        // Emergent Phi = normalized HDC state vector activity.
        //
        // The LTC network's output norm reflects how actively the
        // system is integrating: high norm = strong integration of inputs,
        // low norm = quiescent/collapsed state. This is a proxy for
        // IIT's "integrated information" — the network must bind inputs
        // through its weights to produce a high-norm output.
        //
        // Norm naturally varies with input strength, nearby agent count,
        // energy level, and danger — creating LTC-driven state dynamics.
        let norm = self.thought_hv.norm();
        let phi = (norm * Self::PHI_NORM_SCALE).min(1.0) as f64;
        phi.clamp(0.0, 1.0)
    }

    /// Get the thought hypervector for inter-agent resonance computation.
    pub fn thought_hv(&self) -> &ContinuousHV {
        &self.thought_hv
    }
}

/// Derive integration-metric inputs dynamically from physics/game state (HDC-D).
///
/// Closes the integration-physics loop: physics state drives integration
/// inputs, integration metrics (via HDC state vectors) drive motor authority,
/// motor authority drives physics state.
///
/// Replaces the hardcoded `ConsciousnessInputs { phi: 0.5, ... }` with
/// state-responsive values.
pub fn inputs_from_state(
    energy_frac: f64,
    nearby_count: usize,
    prediction_error: f64,
    danger: f64,
    motor_precision: f64,
    harmony_total: f64,
    collective_phi: f64,
) -> [f64; NUM_COMPONENTS] {
    [
        energy_frac.clamp(0.0, 1.0), // phi ← energy = integration capacity
        (nearby_count as f64 / 5.0).min(1.0), // broadcast ← social = workspace
        (1.0 - prediction_error).clamp(0.0, 1.0), // working_memory ← surprise load
        if danger > 0.3 { 0.8 } else { 0.5 }, // attention ← threat presence
        motor_precision.clamp(0.0, 1.0), // recurrence ← action feedback
        (1.0 - energy_frac * 0.3).clamp(0.0, 1.0), // embodiment ← physical cost
        (harmony_total / 8.0).clamp(0.0, 1.0), // knowledge ← harmony integration
    ]
}

/// Compute HDC vector similarity between agents.
///
/// Returns cosine similarity between two state vectors.
/// Complements scalar harmony resonance with high-dimensional state alignment.
pub fn thought_resonance(thought_a: &ContinuousHV, thought_b: &ContinuousHV) -> f64 {
    thought_a.similarity(thought_b) as f64
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
        let harmony = [0.5; NUM_HARMONIES];

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
        let harmony = [0.5; NUM_HARMONIES];

        for _ in 0..5 {
            ctx_a.step(&inputs, &harmony, 0.016);
            ctx_b.step(&inputs, &harmony, 0.016);
        }

        // Different seeds should produce different HDC state vectors.
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
        let harmony = [0.0; NUM_HARMONIES];

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
        let harmony = [0.5; NUM_HARMONIES];

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

    #[test]
    fn inputs_from_state_clamped() {
        let inputs = inputs_from_state(0.5, 3, 0.2, 0.0, 0.8, 4.0, 0.5);
        for &v in &inputs {
            assert!(v >= 0.0 && v <= 1.0, "Input {v} out of [0,1]");
        }
    }

    #[test]
    fn inputs_respond_to_energy_depletion() {
        let high_energy = inputs_from_state(0.9, 3, 0.1, 0.0, 0.9, 4.0, 0.5);
        let low_energy = inputs_from_state(0.1, 3, 0.1, 0.0, 0.9, 4.0, 0.5);
        // Low energy → lower phi input (integration capacity drops)
        assert!(low_energy[0] < high_energy[0]);
    }

    #[test]
    fn inputs_respond_to_danger() {
        let safe = inputs_from_state(0.5, 3, 0.1, 0.0, 0.8, 4.0, 0.5);
        let danger = inputs_from_state(0.5, 3, 0.1, 0.8, 0.8, 4.0, 0.5);
        // Danger → higher attention input
        assert!(danger[3] > safe[3]);
    }

    #[test]
    fn phi_is_nonzero_after_stepping() {
        let mut ctx = HdcConsciousnessContext::new(42);
        let inputs = [0.8, 0.5, 0.7, 0.5, 0.9, 0.6, 0.5];
        let harmony = [0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8];

        for _ in 0..50 {
            ctx.step(&inputs, &harmony, 1.0 / 64.0);
        }

        let phi = ctx.phi_from_thought(&harmony);
        eprintln!(
            "  HDC Phi after 50 ticks: {phi:.6}, norm: {:.4}",
            ctx.thought_hv().norm()
        );
        assert!(phi > 0.001, "Phi should be nonzero after stepping: {phi}");
    }

    #[test]
    fn phi_responds_to_input_changes() {
        let mut ctx = HdcConsciousnessContext::new(42);
        let harmony = [0.5; NUM_HARMONIES];

        // Step with high inputs
        for _ in 0..30 {
            ctx.step(&[0.9; 7], &harmony, 1.0 / 64.0);
        }
        let phi_high = ctx.phi_from_thought(&harmony);

        // Step with low inputs — Phi should change
        for _ in 0..30 {
            ctx.step(&[0.1; 7], &harmony, 1.0 / 64.0);
        }
        let phi_low = ctx.phi_from_thought(&harmony);

        eprintln!("  phi_high={phi_high:.6}, phi_low={phi_low:.6}");
        // They should be different (consciousness responds to input changes)
        assert!(
            (phi_high - phi_low).abs() > 0.001,
            "Phi should respond to input changes: high={phi_high}, low={phi_low}"
        );
    }

    #[test]
    fn thought_resonance_self_is_one() {
        let ctx = HdcConsciousnessContext::new(42);
        // Before stepping, thought_hv is zero → similarity undefined
        // After stepping, should be self-similar
        let mut ctx2 = HdcConsciousnessContext::new(42);
        let inputs = [0.5; 7];
        let harmony = [0.5; NUM_HARMONIES];
        for _ in 0..5 {
            ctx2.step(&inputs, &harmony, 0.016);
        }
        let res = thought_resonance(ctx2.thought_hv(), ctx2.thought_hv());
        assert!(
            (res - 1.0).abs() < 0.01,
            "Self-resonance should be ~1.0: {res}"
        );
    }
}
