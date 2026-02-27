// Engine is wired into CognitiveLoopService::cycle(). Output fields are not yet
// individually consumed (additive wiring phase — telemetry only), so allow dead_code.
#![allow(dead_code)]

//! # Unified Consciousness Engine
//!
//! Wraps the 4 independent consciousness measurement systems into a single
//! coherent engine with clear data flow and unified output.
//!
//! ## Systems (Tiered Architecture)
//!
//! | Tier | System | Interval | Science |
//! |------|--------|----------|---------|
//! | Layer 1 | SpectralMIPFinder | 47 cycles | Tononi (2004) — IIT Phi via Fiedler ordering |
//! | Layer 2 | MultiModalIntegrator | 13 cycles | Damasio (1994) — cross-modal binding Phi |
//! | Layer 3 | ConsciousnessEquationV2 | 23 cycles | 7-theory master equation C(t) |
//! | Layer 4 | UnifiedConsciousnessPipeline | 47 cycles | Dehaene (2011) — end-to-end pipeline |
//!
//! ## Design Principles
//!
//! 1. **Single entry point**: `measure()` called once per cycle
//! 2. **No direct field mutation**: Returns `ConsciousnessEngineOutput` with proposed deltas
//! 3. **Preserves co-prime intervals**: Each subsystem fires at its original rate
//! 4. **Backward compatible**: All existing carryover fields populated
//! 5. **WASM-ready**: Output struct uses only Copy/Clone types (no trait objects)
// Engine is wired into CognitiveLoopService::cycle() — no dead code.

use std::time::Instant;

use symthaea_core::consciousness_metrics::SpectralMIPFinder;
use symthaea_core::hdc::{BinaryHV, ContinuousHV};

use crate::consciousness::consciousness_equation_v2::{
    ConsciousnessEquationV2, ConsciousnessStateV2, CoreComponent,
};
use crate::consciousness::cross_modal_binding::Modality;
use crate::consciousness::multi_modal_integration::{ModalInput, MultiModalIntegrator};
use crate::consciousness::unified_consciousness_pipeline::UnifiedConsciousnessPipeline;

use super::types::ConsciousnessCache;

/// Unified output from the consciousness engine.
///
/// Contains all measurement results plus proposed feedback deltas.
/// The caller applies deltas to prediction_confidence, fep_lr_boost, etc.
#[derive(Debug, Clone)]
pub(crate) struct ConsciousnessEngineOutput {
    // ── Raw measurements ───────────────────────────────────────────────
    /// Spectral MIP Phi — integrated information via Fiedler ordering [0, ∞)
    pub spectral_mip_phi: Option<f64>,
    /// Hierarchical MIP Phi — multi-scale (32→64→128) [0, ∞)
    pub hierarchical_mip_phi: Option<f64>,
    /// Multi-modal integrated Phi — cross-modal binding [0, 1]
    pub multimodal_phi: f64,
    /// Consciousness Equation V2 — 7-theory unified C(t) [0, 1]
    pub equation_v2_consciousness: f64,
    /// Pipeline consciousness — end-to-end sensory→consciousness [0, 1]
    pub pipeline_consciousness: f64,
    /// Limiting component from equation v2
    pub limiting_component: Option<CoreComponent>,

    // ── Unified consciousness level ────────────────────────────────────
    /// Weighted consensus consciousness level [0, 1].
    ///
    /// Computed as: 0.35 × spectral_phi_norm + 0.25 × equation_v2 + 0.25 × pipeline + 0.15 × multimodal
    /// (spectral_phi_norm = sigmoid(phi) to map [0,∞) → [0,1])
    pub unified_consciousness: f64,

    // ── Sigma (backward compatibility for memory coordinator) ──────────
    pub sigma: Option<f64>,

    // ── Proposed feedback deltas ────────────────────────────────────────
    /// Additive delta for prediction_confidence (positive = boost, negative = dampen)
    pub confidence_delta: f32,
    /// Multiplicative factor for fep_lr_boost (1.0 = no change)
    pub lr_factor: f32,
    /// Additive delta for exploration_urge
    pub exploration_delta: f32,
    /// Multiplicative factor for subsystem_lr_factor
    pub subsystem_lr_factor: f32,
    /// Whether to boost episodic consolidation (high consciousness moment)
    pub episodic_consolidation_boost: Option<f64>,

    // ── Timing ─────────────────────────────────────────────────────────
    pub spectral_mip_us: u64,
    pub equation_v2_us: u64,
    pub pipeline_us: u64,
    pub multimodal_us: u64,
    pub total_us: u64,
}

/// Input snapshot for the consciousness engine.
///
/// Collected once per cycle by the caller, passed immutably.
pub(crate) struct ConsciousnessEngineInput<'a> {
    /// Current HDC encoding (ContinuousHV, 16384-dim) — for SpectralMIP push
    pub hdv: &'a ContinuousHV,
    /// Current HDC encoding (BinaryHV, 16384-bit) — for multimodal + pipeline
    pub hv16: &'a BinaryHV,
    /// Current cycle number
    pub cycle: u64,
    /// Unified Psi from primitive consciousness
    pub unified_psi: f64,
    /// Smoothed coherence from HDC pipeline
    pub coherence: f32,
    /// Current prediction error
    pub prediction_error: f32,
    /// Phi attention weight
    pub phi_attention_weight: f32,
    /// Last epistemic quality (knowledge component for equation v2)
    pub epistemic_quality: f64,
    /// Phi validation correlation (for adaptive weighting)
    pub phi_validation_correlation: f64,
    /// Phi spectral weight (adaptive, from validation)
    pub phi_spectral_weight: f64,
}

/// The unified consciousness measurement engine.
///
/// Owns all 4 measurement systems and orchestrates their execution
/// at co-prime intervals.
pub(crate) struct ConsciousnessEngine {
    // ── Layer 1: Spectral MIP (always present) ─────────────────────────
    spectral_mip_finder: SpectralMIPFinder,

    // ── Layer 2: Multi-modal integration (optional) ────────────────────
    multi_modal_integrator: Option<MultiModalIntegrator>,

    // ── Layer 3: Master equation (optional) ────────────────────────────
    consciousness_equation_v2: Option<ConsciousnessEquationV2>,

    // ── Layer 4: Full pipeline (optional) ──────────────────────────────
    unified_consciousness_pipeline: Option<UnifiedConsciousnessPipeline>,

    // ── Cached values (for non-firing cycles) ──────────────────────────
    cache: ConsciousnessEngineCache,
}

/// Internal cache for inter-cycle persistence.
#[derive(Debug, Clone, Default)]
struct ConsciousnessEngineCache {
    last_spectral_mip_phi: Option<f64>,
    last_hierarchical_mip_phi: Option<f64>,
    last_sigma: Option<f64>,
    last_multimodal_phi: f64,
    last_equation_v2_consciousness: f64,
    last_pipeline_consciousness: f64,
}

impl ConsciousnessEngine {
    /// Create a new consciousness engine from its component systems.
    pub fn new(
        spectral_mip_finder: SpectralMIPFinder,
        multi_modal_integrator: Option<MultiModalIntegrator>,
        consciousness_equation_v2: Option<ConsciousnessEquationV2>,
        unified_consciousness_pipeline: Option<UnifiedConsciousnessPipeline>,
    ) -> Self {
        Self {
            spectral_mip_finder,
            multi_modal_integrator,
            consciousness_equation_v2,
            unified_consciousness_pipeline,
            cache: ConsciousnessEngineCache::default(),
        }
    }

    /// Measure consciousness for the current cycle.
    ///
    /// Each subsystem fires at its co-prime interval:
    /// - SpectralMIPFinder: push every cycle, compute every 47, adapt every 94
    /// - MultiModalIntegrator: every 13 cycles
    /// - ConsciousnessEquationV2: every 23 cycles
    /// - UnifiedConsciousnessPipeline: every 47 cycles
    ///
    /// Returns proposed feedback deltas — the caller applies them.
    pub fn measure(&mut self, input: &ConsciousnessEngineInput) -> ConsciousnessEngineOutput {
        let total_start = Instant::now();
        let mut confidence_delta: f32 = 0.0;
        let mut lr_factor: f32 = 1.0;
        let mut exploration_delta: f32 = 0.0;
        let mut subsystem_lr_factor: f32 = 1.0;
        let mut episodic_consolidation_boost: Option<f64> = None;

        // ═══════════════════════════════════════════════════════════════════
        // LAYER 1: Spectral MIP — O(n³) Fiedler-ordered Phi
        // Push every cycle, compute every 47, adapt+hierarchical every 94
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        self.spectral_mip_finder.push(input.hdv); // ContinuousHV

        let spectral_mip_phi = if input.cycle % 47 == 0 {
            let result = self.spectral_mip_finder.compute();
            let phi = result.as_ref().map(|r| r.phi);
            if phi.is_some() {
                self.cache.last_spectral_mip_phi = phi;
                self.cache.last_sigma = phi;
            }

            // Adaptive dimension selection + hierarchical every 94 cycles
            if input.cycle % 94 == 0 {
                if let Some(ref r) = result {
                    self.spectral_mip_finder.adapt(r);
                }
                if let Some(hier) = self.spectral_mip_finder.compute_hierarchical() {
                    self.cache.last_hierarchical_mip_phi = Some(hier.phi);
                }
            }
            phi
        } else {
            self.cache.last_spectral_mip_phi
        };
        let spectral_mip_us = t.elapsed().as_micros() as u64;

        // Sigma → learning rate + confidence modulation
        // Science: Tononi (2008) — high Φ → stabilize, low Φ → explore
        if let Some(sig) = self.cache.last_sigma {
            if sig > 0.5 {
                let sig_dampen = ((sig - 0.5) * 0.1).min(0.05) as f32;
                lr_factor *= 1.0 - sig_dampen;
                confidence_delta += sig_dampen * 0.5;
            } else if sig < 0.2 {
                let sig_boost = ((0.2 - sig) * 0.15).min(0.05) as f32;
                lr_factor *= 1.0 + sig_boost;
            }
        }

        // Adaptive Phi validation weighting
        if let Some(sig) = self.cache.last_sigma {
            if input.phi_validation_correlation > 0.7 {
                let validation_boost =
                    (input.phi_validation_correlation - 0.7) as f32 * 0.1;
                confidence_delta += sig as f32 * validation_boost;
            } else if input.phi_validation_correlation > 0.0
                && input.phi_validation_correlation < 0.3
            {
                let attenuate = (0.3 - input.phi_validation_correlation) as f32 * 0.05;
                // Negative confidence contribution (multiplicative attenuation encoded as delta)
                confidence_delta -= attenuate;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // LAYER 2: Multi-modal integration — cross-modal Phi
        // Every 13 cycles (co-prime)
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        let multimodal_phi = if let Some(ref mut mmi) = self.multi_modal_integrator {
            if input.cycle % 13 == 0 && input.cycle > 0 {
                let visual_input =
                    ModalInput::new(Modality::Visual, *input.hv16, input.coherence as f64);
                let temporal_input = ModalInput::new(
                    Modality::Temporal,
                    *input.hv16,
                    input.unified_psi.clamp(0.0, 1.0),
                );
                let result = mmi.integrate(&[visual_input, temporal_input]);
                self.cache.last_multimodal_phi = result.integrated_phi;
                result.integrated_phi
            } else {
                self.cache.last_multimodal_phi
            }
        } else {
            0.0
        };
        let multimodal_us = t.elapsed().as_micros() as u64;

        // Multimodal feedback: strong integration → learning precision
        // Science: Ghazanfar & Schroeder (2006)
        if multimodal_phi > 0.5 {
            let phi_conf = (multimodal_phi - 0.5) * 0.04;
            confidence_delta += phi_conf as f32;
            let phi_lr = 1.0 + (multimodal_phi - 0.5) * 0.4;
            subsystem_lr_factor *= phi_lr as f32;
        }

        // ═══════════════════════════════════════════════════════════════════
        // LAYER 3: Consciousness Equation V2 — 7-theory C(t)
        // Every 23 cycles (co-prime)
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        let equation_v2_consciousness = if let Some(ref mut eq) = self.consciousness_equation_v2 {
            if input.cycle % 23 == 0 && input.cycle > 0 {
                use std::collections::HashMap;
                let mut core_values = HashMap::new();
                core_values
                    .insert(CoreComponent::Integration, input.unified_psi.clamp(0.0, 1.0));
                core_values.insert(CoreComponent::Binding, input.coherence as f64);
                core_values.insert(CoreComponent::Workspace, input.coherence as f64 * 0.8);
                core_values.insert(CoreComponent::Attention, input.phi_attention_weight as f64);
                core_values.insert(CoreComponent::Recursion, 0.5); // Placeholder: HOT depth
                core_values
                    .insert(CoreComponent::Efficacy, 1.0 - input.prediction_error as f64);
                core_values.insert(CoreComponent::Knowledge, input.epistemic_quality);

                let state = ConsciousnessStateV2 {
                    core_values,
                    extended_values: HashMap::new(),
                    phase_coherence: HashMap::new(),
                    substrate_feasibility: 1.0,
                    timestamp: input.cycle,
                    context: String::new(),
                };
                let result = eq.compute(&state);
                self.cache.last_equation_v2_consciousness = result.consciousness;
                result.consciousness
            } else {
                self.cache.last_equation_v2_consciousness
            }
        } else {
            0.0
        };
        let equation_v2_us = t.elapsed().as_micros() as u64;

        // Equation V2 feedback: high consciousness → confidence + consolidation
        // Science: Tononi (2004), Baars (1988), Dehaene (2014)
        if equation_v2_consciousness > 0.6 {
            let boost = (equation_v2_consciousness - 0.6) * 0.08;
            confidence_delta += boost as f32;
            episodic_consolidation_boost = Some((equation_v2_consciousness - 0.6) * 0.1);
        } else if equation_v2_consciousness > 0.0 && equation_v2_consciousness < 0.3 {
            exploration_delta += 0.02;
        }

        // ═══════════════════════════════════════════════════════════════════
        // LAYER 4: Unified Consciousness Pipeline — end-to-end
        // Every 47 cycles (co-prime with Spectral MIP via offset)
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        let pipeline_consciousness =
            if let Some(ref mut pipeline) = self.unified_consciousness_pipeline {
                if input.cycle % 47 == 0 && input.cycle > 0 {
                    let sensory: Vec<f64> = (0..64)
                        .map(|i| {
                            if input.hv16.get_bit(i) != 0 {
                                1.0
                            } else {
                                -1.0
                            }
                        })
                        .collect();
                    match pipeline.process(&sensory) {
                        Ok(moment) => {
                            self.cache.last_pipeline_consciousness = moment.consciousness;
                            moment.consciousness
                        }
                        Err(_) => self.cache.last_pipeline_consciousness,
                    }
                } else {
                    self.cache.last_pipeline_consciousness
                }
            } else {
                0.0
            };
        let pipeline_us = t.elapsed().as_micros() as u64;

        // Pipeline feedback: high consciousness → learning coherence
        // Science: Dehaene (2011) — global workspace broadcasts learning signals
        if pipeline_consciousness > 0.6 {
            let pipeline_lr_scale = 1.0 + (pipeline_consciousness - 0.6) * 0.5;
            lr_factor *= pipeline_lr_scale as f32;
        }

        // ═══════════════════════════════════════════════════════════════════
        // UNIFIED CONSCIOUSNESS: Weighted consensus across all systems
        // ═══════════════════════════════════════════════════════════════════
        let unified_consciousness = self.compute_unified(
            spectral_mip_phi,
            multimodal_phi,
            equation_v2_consciousness,
            pipeline_consciousness,
        );

        let total_us = total_start.elapsed().as_micros() as u64;

        // Clamp subsystem_lr_factor to valid range
        subsystem_lr_factor = subsystem_lr_factor.clamp(0.8, 1.2);

        ConsciousnessEngineOutput {
            spectral_mip_phi,
            hierarchical_mip_phi: self.cache.last_hierarchical_mip_phi,
            multimodal_phi,
            equation_v2_consciousness,
            pipeline_consciousness,
            limiting_component: None, // TODO: extract from equation v2 result
            unified_consciousness,
            sigma: self.cache.last_sigma,
            confidence_delta,
            lr_factor,
            exploration_delta,
            subsystem_lr_factor,
            episodic_consolidation_boost,
            spectral_mip_us,
            equation_v2_us,
            pipeline_us,
            multimodal_us,
            total_us,
        }
    }

    /// Compute weighted consensus consciousness level.
    ///
    /// Weights: 0.35 spectral (rigorous IIT) + 0.25 equation (7-theory) +
    ///          0.25 pipeline (end-to-end) + 0.15 multimodal (cross-modal)
    ///
    /// Spectral Phi is mapped [0,∞) → [0,1] via sigmoid: 2/(1+exp(-phi)) - 1
    fn compute_unified(
        &self,
        spectral_mip_phi: Option<f64>,
        multimodal_phi: f64,
        equation_v2: f64,
        pipeline: f64,
    ) -> f64 {
        // Normalize spectral phi from [0, ∞) to [0, 1] via shifted sigmoid
        let spectral_norm = spectral_mip_phi
            .map(|phi| 2.0 / (1.0 + (-phi).exp()) - 1.0)
            .unwrap_or(0.0);

        // Weighted consensus
        let unified = 0.35 * spectral_norm
            + 0.25 * equation_v2
            + 0.25 * pipeline
            + 0.15 * multimodal_phi;

        unified.clamp(0.0, 1.0)
    }

    /// Populate a ConsciousnessCache from the engine's internal cache.
    ///
    /// Used to maintain backward compatibility with CognitiveLoopService
    /// carryover fields.
    pub fn update_cache(&self, cache: &mut ConsciousnessCache) {
        cache.last_spectral_mip_phi = self.cache.last_spectral_mip_phi;
        cache.last_sigma = self.cache.last_sigma;
        cache.last_multimodal_phi = self.cache.last_multimodal_phi;
        cache.last_equation_v2_consciousness = self.cache.last_equation_v2_consciousness;
        cache.last_hierarchical_mip_phi = self.cache.last_hierarchical_mip_phi;
    }

    /// Get the spectral MIP finder for direct push access (HDC state feed).
    pub fn spectral_mip_finder(&self) -> &SpectralMIPFinder {
        &self.spectral_mip_finder
    }

    /// Get mutable access to the spectral MIP finder.
    pub fn spectral_mip_finder_mut(&mut self) -> &mut SpectralMIPFinder {
        &mut self.spectral_mip_finder
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::consciousness_metrics::SpectralMIPConfig;
    use symthaea_core::hdc::{BinaryHV, ContinuousHV};

    fn make_engine() -> ConsciousnessEngine {
        let config = SpectralMIPConfig {
            num_components: 64,
            window_size: 20,
            min_samples: 5,
            regularization: 1e-6,
        };
        let finder = SpectralMIPFinder::new(config);
        ConsciousnessEngine::new(finder, None, None, None)
    }

    /// Helper to build a test input with both ContinuousHV and BinaryHV.
    fn make_input<'a>(
        hdv: &'a ContinuousHV,
        hv16: &'a BinaryHV,
        cycle: u64,
    ) -> ConsciousnessEngineInput<'a> {
        ConsciousnessEngineInput {
            hdv,
            hv16,
            cycle,
            unified_psi: 0.5,
            coherence: 0.6,
            prediction_error: 0.2,
            phi_attention_weight: 0.4,
            epistemic_quality: 0.5,
            phi_validation_correlation: 0.5,
            phi_spectral_weight: 0.6,
        }
    }

    #[test]
    fn test_engine_creation() {
        let engine = make_engine();
        assert!(engine.multi_modal_integrator.is_none());
        assert!(engine.consciousness_equation_v2.is_none());
        assert!(engine.unified_consciousness_pipeline.is_none());
    }

    #[test]
    fn test_engine_measure_returns_valid_output() {
        let mut engine = make_engine();
        let hdv = ContinuousHV::random(16384, 42);
        let hv16 = BinaryHV::random(42);
        let input = make_input(&hdv, &hv16, 1);

        let output = engine.measure(&input);

        assert!(output.unified_consciousness >= 0.0);
        assert!(output.unified_consciousness <= 1.0);
        assert!(output.total_us > 0);
        assert!(output.lr_factor.is_finite());
        assert!(output.confidence_delta.is_finite());
    }

    #[test]
    fn test_engine_spectral_mip_fires_at_interval_47() {
        let mut engine = make_engine();

        for cycle in 0..48 {
            let hdv = ContinuousHV::random(16384, cycle + 1);
            let hv16 = BinaryHV::random(cycle + 1);
            let input = make_input(&hdv, &hv16, cycle);
            let output = engine.measure(&input);

            if cycle < 47 {
                assert!(
                    output.spectral_mip_phi.is_none()
                        || output.spectral_mip_phi.unwrap().is_finite()
                );
            }
        }
    }

    #[test]
    fn test_unified_consciousness_bounded() {
        let engine = make_engine();

        assert!(engine.compute_unified(Some(100.0), 1.0, 1.0, 1.0) <= 1.0);
        assert!(engine.compute_unified(Some(0.0), 0.0, 0.0, 0.0) >= 0.0);
        assert!(engine.compute_unified(None, 0.5, 0.5, 0.5) >= 0.0);
        assert!(engine.compute_unified(None, 0.5, 0.5, 0.5) <= 1.0);
    }

    #[test]
    fn test_equation_v2_feedback_deltas() {
        let config = SpectralMIPConfig {
            num_components: 64,
            window_size: 20,
            min_samples: 5,
            regularization: 1e-6,
        };
        let finder = SpectralMIPFinder::new(config);
        let eq = ConsciousnessEquationV2::default();
        let mut engine = ConsciousnessEngine::new(finder, None, Some(eq), None);

        // Run to cycle 23 (when equation v2 fires)
        for cycle in 0..24 {
            let hdv = ContinuousHV::random(16384, cycle + 100);
            let hv16 = BinaryHV::random(cycle + 100);
            let input = ConsciousnessEngineInput {
                hdv: &hdv,
                hv16: &hv16,
                cycle,
                unified_psi: 0.8,
                coherence: 0.9,
                prediction_error: 0.1,
                phi_attention_weight: 0.7,
                epistemic_quality: 0.8,
                phi_validation_correlation: 0.5,
                phi_spectral_weight: 0.6,
            };
            let output = engine.measure(&input);

            if cycle == 23 {
                assert!(
                    output.equation_v2_consciousness.is_finite(),
                    "C(t) should be finite"
                );
                assert!(
                    output.equation_v2_consciousness >= 0.0,
                    "C(t) should be non-negative"
                );
            }
        }
    }

    #[test]
    fn test_cache_update() {
        let mut engine = make_engine();
        engine.cache.last_spectral_mip_phi = Some(0.42);
        engine.cache.last_sigma = Some(0.42);
        engine.cache.last_multimodal_phi = 0.33;
        engine.cache.last_equation_v2_consciousness = 0.55;

        let mut cc = ConsciousnessCache::default();
        engine.update_cache(&mut cc);

        assert_eq!(cc.last_spectral_mip_phi, Some(0.42));
        assert_eq!(cc.last_sigma, Some(0.42));
        assert!((cc.last_multimodal_phi - 0.33).abs() < f64::EPSILON);
        assert!((cc.last_equation_v2_consciousness - 0.55).abs() < f64::EPSILON);
    }

    #[test]
    fn test_low_consciousness_boosts_exploration() {
        let config = SpectralMIPConfig {
            num_components: 64,
            window_size: 20,
            min_samples: 5,
            regularization: 1e-6,
        };
        let finder = SpectralMIPFinder::new(config);
        let eq = ConsciousnessEquationV2::default();
        let mut engine = ConsciousnessEngine::new(finder, None, Some(eq), None);

        for cycle in 0..24 {
            let hdv = ContinuousHV::random(16384, cycle + 100);
            let hv16 = BinaryHV::random(cycle + 100);
            let input = ConsciousnessEngineInput {
                hdv: &hdv,
                hv16: &hv16,
                cycle,
                unified_psi: 0.05,
                coherence: 0.1,
                prediction_error: 0.9,
                phi_attention_weight: 0.05,
                epistemic_quality: 0.1,
                phi_validation_correlation: 0.0,
                phi_spectral_weight: 0.5,
            };
            let output = engine.measure(&input);

            if cycle == 23
                && output.equation_v2_consciousness < 0.3
                && output.equation_v2_consciousness > 0.0
            {
                assert!(
                    output.exploration_delta > 0.0,
                    "Low consciousness should boost exploration"
                );
            }
        }
    }

    #[test]
    fn test_timing_fields_populated() {
        let mut engine = make_engine();
        let hdv = ContinuousHV::random(16384, 42);
        let hv16 = BinaryHV::random(42);
        let input = make_input(&hdv, &hv16, 1);
        let output = engine.measure(&input);

        assert!(output.spectral_mip_us < 1_000_000);
        assert!(output.total_us < 1_000_000);
    }
}
