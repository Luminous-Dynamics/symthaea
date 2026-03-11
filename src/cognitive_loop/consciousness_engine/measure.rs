//! Core measurement method for the consciousness engine.

use std::time::Instant;

use crate::consciousness::consciousness_equation_v2::{ConsciousnessStateV2, CoreComponent};
use crate::consciousness::cross_modal_binding::Modality;
use crate::consciousness::multi_modal_integration::ModalInput;

use super::types::{ConsciousnessEngineInput, ConsciousnessEngineOutput};
use super::ConsciousnessEngine;

impl ConsciousnessEngine {
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
        // Push every cycle, compute every 97, adapt+hierarchical every 194
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        self.spectral_mip_finder.push(input.hdv); // ContinuousHV

        let spectral_mip_phi = if input.cycle % 97 == 0 {
            let result = self.spectral_mip_finder.compute();
            let phi = result.as_ref().map(|r| r.phi);
            if phi.is_some() {
                self.cache.last_spectral_mip_phi = phi;
                self.cache.last_sigma = phi;
            }

            // Structural hierarchy: compute on every spectral MIP pass (every 97 cycles).
            // Ensures structural Phi is available within the first 100 cycles for
            // short benchmarks and cold-start validation.
            if let Some(ref r) = result {
                if let Some(structural) = self.spectral_mip_finder.compute_structural_hierarchy(r) {
                    self.cache.last_structural_phi = Some(structural);
                }
            }

            // Adaptive dimension selection + hierarchical MIP every 194 cycles
            // (adapt() is expensive — keep at half the structural rate)
            if input.cycle % 194 == 0 {
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
                let sig_dampen = ((sig - 0.5) * 0.1).min(0.05_f64) as f32;
                lr_factor *= 1.0 - sig_dampen;
                confidence_delta += sig_dampen * 0.5;
            } else if sig < 0.2 {
                let sig_boost = ((0.2 - sig) * 0.15).min(0.05_f64) as f32;
                lr_factor *= 1.0 + sig_boost;
            }
        }

        // Update dynamic consciousness weights from structural Phi.
        // Extract emergence_ratio to avoid cloning the full struct (Vec allocs).
        if let Some(er) = self
            .cache
            .last_structural_phi
            .as_ref()
            .map(|s| s.emergence_ratio)
        {
            self.update_weights_from_emergence(er);
        }

        // Structural Phi feedback: cross-region binding diagnostics
        // Science: Mediano et al. (2022) — multi-scale integrated information
        if let Some(ref structural) = self.cache.last_structural_phi {
            if structural.num_clusters >= 2 {
                // Weak global binding: local regions integrate but don't unify
                if structural.emergence_ratio < 0.8 && structural.micro_phi > 0.01 {
                    // Nudge toward exploration to discover cross-region associations
                    exploration_delta += 0.01;
                }
                // Strong emergence: the whole exceeds the sum of parts
                if structural.emergence_ratio > 1.2 {
                    confidence_delta += 0.01;
                }
                // Bottleneck: large gap between global and inter-cluster integration
                if structural.bottleneck_score > 0.3 {
                    // Boost learning rate to strengthen weak inter-cluster connections
                    lr_factor *= 1.02;
                }
            }
        }

        // Adaptive Phi validation weighting
        if let Some(sig) = self.cache.last_sigma {
            if input.phi_validation_correlation > 0.7 {
                let validation_boost = (input.phi_validation_correlation - 0.7) as f32 * 0.1;
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
                core_values.insert(
                    CoreComponent::Integration,
                    input.unified_psi.clamp(0.0, 1.0),
                );
                // Substrate-modulated: binding/workspace/attention capabilities
                // scale the respective consciousness components.
                core_values.insert(
                    CoreComponent::Binding,
                    input.coherence as f64 * input.binding_capability,
                );
                core_values.insert(
                    CoreComponent::Workspace,
                    input.coherence as f64 * 0.8 * input.workspace_capability,
                );
                core_values.insert(
                    CoreComponent::Attention,
                    input.phi_attention_weight as f64 * input.attention_capability,
                );
                core_values.insert(CoreComponent::Recursion, input.hot_depth);
                core_values.insert(CoreComponent::Efficacy, 1.0 - input.prediction_error as f64);

                // Approach C: Drift-driven epistemic humility
                // High moral drift → attenuate Knowledge component in EquationV2.
                // Science: Epistemic humility during value shifts — if your moral
                // stance is changing rapidly, "knowledge" claims carry less weight.
                let effective_epistemic = if self.moral_coupling.enabled {
                    let drift_ratio =
                        (input.moral_drift / self.moral_coupling.drift_saturation).min(1.0);
                    let attenuation =
                        1.0 - drift_ratio * self.moral_coupling.drift_epistemic_attenuation;
                    input.epistemic_quality * attenuation
                } else {
                    input.epistemic_quality
                };
                core_values.insert(CoreComponent::Knowledge, effective_epistemic);

                let state = ConsciousnessStateV2 {
                    core_values,
                    extended_values: HashMap::new(),
                    phase_coherence: HashMap::new(),
                    substrate_feasibility: input.substrate_feasibility,
                    timestamp: input.cycle,
                    context: String::new(),
                };
                let result = eq.compute(&state);
                self.cache.last_equation_v2_consciousness = result.consciousness;
                self.cache.last_limiting_component = Some(result.limiting_factor);
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
        // advance() every cycle (builds state), process() every 97 (with binding)
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        let pipeline_consciousness =
            if let Some(ref mut pipeline) = self.unified_consciousness_pipeline {
                let sensory: Vec<f64> = (0..64)
                    .map(|i| {
                        if input.hv16.get_bit(i) != 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect();
                if input.cycle % 97 == 0 && input.cycle > 0 {
                    // Full process with oscillatory binding
                    match pipeline.process(&sensory) {
                        Ok(moment) => {
                            self.cache.last_pipeline_consciousness = moment.consciousness;
                            moment.consciousness
                        }
                        Err(_) => self.cache.last_pipeline_consciousness,
                    }
                } else {
                    // Lightweight advance: HDC + LTC + state (no binding)
                    if let Ok(c) = pipeline.advance(&sensory) {
                        self.cache.last_pipeline_consciousness = c;
                    }
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
        let mut unified_consciousness = self.compute_unified(
            spectral_mip_phi,
            multimodal_phi,
            equation_v2_consciousness,
            pipeline_consciousness,
        );

        // ═══════════════════════════════════════════════════════════════════
        // BATH-CONSCIOUSNESS COUPLING (Seth 2013 — interoceptive inference)
        // ═══════════════════════════════════════════════════════════════════
        // High 5-HT2A amplifies perceptual richness → consciousness boost
        let sht_2a_boost = (input.sht_2a_signal - 0.5) * 0.1; // ±5% from baseline 0.5
                                                              // GABA-A dampens global gain → consciousness reduction
        let gaba_a_dampen = -(input.gaba_a_signal - 0.4) * 0.08; // baseline GABA=0.4
                                                                 // Low entropy (stuck attractor) → consciousness depression
        let entropy_factor = if input.attractor_detected { -0.05 } else { 0.0 };
        // Approach B: Anomaly dampens unified consciousness
        // High moral anomaly score → reduce consciousness coherence.
        // Science: Moral incoherence as a form of cognitive dissonance
        // (Festinger 1957) — unresolved moral conflict reduces unified experience.
        let moral_dampen = if self.moral_coupling.enabled {
            -input.moral_anomaly_score * self.moral_coupling.anomaly_dampening_strength
        } else {
            0.0
        };
        // Cantor metacognitive depth → consciousness coupling
        // High self-similarity (deep strange loops) boosts consciousness ±3%.
        // Neutral at depth 0.5 — below dampens, above amplifies.
        // Science: Hofstadter (1979) — strange loops; Metzinger (2003) — self-model richness.
        let cantor_depth_factor = (input.cantor_metacognitive_depth - 0.5)
            * super::super::thresholds::CANTOR_CONSCIOUSNESS_MODULATION;

        unified_consciousness = (unified_consciousness
            + sht_2a_boost as f64
            + gaba_a_dampen as f64
            + entropy_factor
            + moral_dampen
            + cantor_depth_factor)
            .clamp(0.0, 1.0);

        let total_us = total_start.elapsed().as_micros() as u64;

        // Clamp subsystem_lr_factor to valid range
        subsystem_lr_factor = subsystem_lr_factor.clamp(0.8, 1.2);

        ConsciousnessEngineOutput {
            spectral_mip_phi,
            hierarchical_mip_phi: self.cache.last_hierarchical_mip_phi,
            structural_phi: self.cache.last_structural_phi.clone(),
            multimodal_phi,
            equation_v2_consciousness,
            pipeline_consciousness,
            limiting_component: self.cache.last_limiting_component,
            unified_consciousness,
            sigma: self.cache.last_sigma,
            confidence_delta,
            lr_factor,
            exploration_delta,
            subsystem_lr_factor,
            episodic_consolidation_boost,
            current_weights: self.cache.weights.as_array(),
            weight_variance: self.weight_variance(),
            convergence_state: self.convergence_state(),
            spectral_mip_us,
            equation_v2_us,
            pipeline_us,
            multimodal_us,
            total_us,
        }
    }
}
