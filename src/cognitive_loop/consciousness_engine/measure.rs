// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core measurement method for the consciousness engine.

use std::time::Instant;

use crate::consciousness::consciousness_equation_v2::{ConsciousnessStateV2, CoreComponent};
use crate::consciousness::cross_modal_binding::Modality;
use crate::consciousness::multi_modal_integration::ModalInput;

use super::ConsciousnessEngine;
use super::types::{ConsciousnessEngineInput, ConsciousnessEngineOutput};

impl ConsciousnessEngine {
    /// Measure consciousness for the current cycle.
    ///
    /// Each subsystem fires at its co-prime interval:
    /// - SpectralMIPFinder: push every 2 cycles, compute every 67, adapt every 134
    /// - MultiModalIntegrator: every 13 cycles
    /// - ConsciousnessEquationV2: every 23 cycles
    /// - UnifiedConsciousnessPipeline: every 67 cycles
    ///
    /// CAVEAT (found 2026-07-04): only SpectralMIPFinder is live in the shipped binary.
    /// `ConsciousnessEngine::new()`'s single production call site (`constructor.rs`)
    /// passes `None` for the other three, with no setter anywhere in production code —
    /// their branches below always take the `else { 0.0 }` arm. `compute_unified()`
    /// therefore collapses to `max(spectral_weight * sigmoid(spectral_mip_phi), 0.05)`,
    /// not the four-way weighted consensus this comment describes. The other three are
    /// real, compiled, non-stub code (constructed only in `tests.rs`), not dead ends —
    /// just dormant.
    ///
    /// This `consciousness_level`/Φ is a *different* quantity from `unified_psi`/Ψ
    /// (`cognitive_loop/helpers/cycle_extracted.rs::compute_unified_psi`, a weighted sum
    /// of CfC coherence/voice/flow/relational/body/embodied contributions). Φ gates
    /// motor safety; Ψ feeds ethics evaluation and Broca's generation trigger. See
    /// `compute_unified_psi`'s doc comment for the deliberate-split rationale.
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
        // Push every 2 cycles, compute every 67, adapt+hierarchical every 134
        // Interval raised from 47→67 (co-prime): reduces O(n³) Fiedler compute
        // frequency by ~30%, lifting sustained throughput from ~12 Hz to ≥20 Hz.
        // Phi window still gets ~24 samples between computes (push every 2).
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        // Push every 2 cycles — halves per-cycle overhead while maintaining ~24 samples
        // in the window before first compute.
        if input.cycle % 2 == 0 {
            self.spectral_mip_finder.push(input.hdv); // ContinuousHV
        }

        let spectral_mip_phi = if input.cycle % 67 == 0 {
            let result = self.spectral_mip_finder.compute();
            let phi = result.as_ref().map(|r| r.phi);
            if phi.is_some() {
                self.cache.last_spectral_mip_phi = phi;
                self.cache.last_sigma = phi;
            }

            // Structural hierarchy: compute on every spectral MIP pass (every 67 cycles).
            // Ensures structural Phi is available within the first 100 cycles for
            // short benchmarks and cold-start validation.
            if let Some(ref r) = result {
                if let Some(structural) = self.spectral_mip_finder.compute_structural_hierarchy(r) {
                    self.cache.last_structural_phi = Some(structural);
                }
            }

            // Adaptive dimension selection + hierarchical MIP every 134 cycles
            // (adapt() is expensive — keep at 2× the base compute interval)
            if input.cycle % 134 == 0 {
                if let Some(ref r) = result {
                    self.spectral_mip_finder.adapt(r);
                    self.cache.last_spectral_mip_adapted = self.spectral_mip_finder.is_adapted();
                    self.cache.last_spectral_mip_active_dim_count = self
                        .spectral_mip_finder
                        .active_dim_indices()
                        .map(|d| d.len());
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
        if let Some(sig) = self.cache.last_sigma.filter(|s| s.is_finite()) {
            if sig > super::super::thresholds::SIGMA_HIGH_THRESHOLD {
                let sig_dampen = ((sig - super::super::thresholds::SIGMA_HIGH_THRESHOLD)
                    * super::super::thresholds::SIGMA_DAMPEN_SCALE)
                    .min(super::super::thresholds::SIGMA_DAMPEN_MAX)
                    as f32;
                lr_factor *= 1.0 - sig_dampen;
                confidence_delta += sig_dampen * super::super::thresholds::SIGMA_CONFIDENCE_SCALE;
            } else if sig < super::super::thresholds::SIGMA_LOW_THRESHOLD {
                let sig_boost = ((super::super::thresholds::SIGMA_LOW_THRESHOLD - sig)
                    * super::super::thresholds::SIGMA_BOOST_SCALE)
                    .min(super::super::thresholds::SIGMA_BOOST_MAX)
                    as f32;
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
                if structural.emergence_ratio
                    < super::super::thresholds::STRUCTURAL_WEAK_EMERGENCE_THRESHOLD
                    && structural.micro_phi
                        > super::super::thresholds::STRUCTURAL_MICRO_PHI_THRESHOLD
                {
                    exploration_delta += super::super::thresholds::STRUCTURAL_EXPLORATION_NUDGE;
                }
                // Strong emergence: the whole exceeds the sum of parts
                if structural.emergence_ratio
                    > super::super::thresholds::STRUCTURAL_STRONG_EMERGENCE_THRESHOLD
                {
                    confidence_delta += super::super::thresholds::STRUCTURAL_CONFIDENCE_NUDGE;
                }
                // Bottleneck: large gap between global and inter-cluster integration
                if structural.bottleneck_score
                    > super::super::thresholds::STRUCTURAL_BOTTLENECK_THRESHOLD
                {
                    lr_factor *= super::super::thresholds::STRUCTURAL_BOTTLENECK_LR_BOOST;
                }
            }
        }

        // Adaptive Phi validation weighting
        if let Some(sig) = self.cache.last_sigma {
            if input.phi_validation_correlation
                > super::super::thresholds::PHI_VALIDATION_HIGH_THRESHOLD
            {
                let validation_boost = (input.phi_validation_correlation
                    - super::super::thresholds::PHI_VALIDATION_HIGH_THRESHOLD)
                    as f32
                    * super::super::thresholds::PHI_VALIDATION_BOOST_SCALE;
                confidence_delta += sig as f32 * validation_boost;
            } else if input.phi_validation_correlation > 0.0
                && input.phi_validation_correlation
                    < super::super::thresholds::PHI_VALIDATION_LOW_THRESHOLD
            {
                let attenuate = (super::super::thresholds::PHI_VALIDATION_LOW_THRESHOLD
                    - input.phi_validation_correlation) as f32
                    * super::super::thresholds::PHI_VALIDATION_ATTENUATION_SCALE;
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
        if multimodal_phi > super::super::thresholds::MULTIMODAL_PHI_THRESHOLD {
            let phi_conf = (multimodal_phi - super::super::thresholds::MULTIMODAL_PHI_THRESHOLD)
                * super::super::thresholds::MULTIMODAL_CONFIDENCE_SCALE;
            confidence_delta += phi_conf as f32;
            let phi_lr = 1.0
                + (multimodal_phi - super::super::thresholds::MULTIMODAL_PHI_THRESHOLD)
                    * super::super::thresholds::MULTIMODAL_LR_SCALE;
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
                let mut core_values = HashMap::with_capacity(7);

                // Integration: prefer spectral MIP Phi (actual IIT computation from
                // Layer 1) over unified_psi (primitive consciousness proxy).
                // Spectral Phi measures genuine integrated information via Fiedler
                // ordering; unified_psi is a simpler pre-IIT estimate.
                // Fallback to unified_psi when spectral Phi hasn't been computed yet.
                let integration = self
                    .cache
                    .last_spectral_mip_phi
                    .map(|phi| {
                        // Normalize spectral phi [0,∞) → [0,1] via sigmoid
                        let normalized = 2.0 / (1.0 + (-phi).exp()) - 1.0;
                        // Blend: 70% spectral + 30% primitive for stability
                        normalized * 0.7 + input.unified_psi * 0.3
                    })
                    .unwrap_or(input.unified_psi)
                    .clamp(0.0, 1.0);
                core_values.insert(CoreComponent::Integration, integration);

                // Binding: prefer pipeline's oscillatory binding coherence (actual
                // gamma-band PLV from OscillatoryBinding) over scaled coherence proxy.
                // Substrate binding_capability still modulates as a ceiling.
                //
                // SAFETY: When binding subsystem is disabled or binding_capability=0,
                // fall back to a degraded coherence floor (0.15) from CfC temporal
                // dynamics. This prevents consciousness death from binding failure.
                // Discovery: Test B proved binding was a single point of failure.
                const BINDING_FALLBACK_FLOOR: f64 = 0.15;
                let raw_binding = if self.cache.last_pipeline_consciousness > 0.0 {
                    let osc_binding = self.cache.last_pipeline_consciousness.min(1.0);
                    (osc_binding * 0.6 + input.coherence as f64 * 0.4) * input.binding_capability
                } else {
                    input.coherence as f64 * input.binding_capability
                };
                // Ensure binding never drops below the coherence-derived floor.
                // Even without phenomenal binding hardware, temporal coherence
                // provides a minimal form of information integration.
                // Use max of: coherence-proportional floor OR absolute minimum.
                let coherence_floor = BINDING_FALLBACK_FLOOR * input.coherence as f64;
                let binding = raw_binding
                    .max(coherence_floor)
                    .max(BINDING_FALLBACK_FLOOR * 0.5);
                core_values.insert(CoreComponent::Binding, binding.clamp(0.0, 1.0));

                // Workspace: use GWT broadcast success as primary signal.
                // gwt_broadcast_occurred from carryover indicates actual workspace
                // ignition (Dehaene 2011), not just scaled coherence.
                // Coalition size reflects workspace breadth.
                let workspace = {
                    let gwt_signal = if input.gwt_broadcast_occurred {
                        // Broadcast happened: workspace is active.
                        // Scale by coalition size (1-8 members typical)
                        (0.5 + 0.5 * (input.gwt_coalition_size as f64 / 4.0).min(1.0))
                            .clamp(0.0, 1.0)
                    } else {
                        // No broadcast: use coherence as fallback
                        input.coherence as f64 * 0.6
                    };
                    gwt_signal * input.workspace_capability
                };
                core_values.insert(CoreComponent::Workspace, workspace.clamp(0.0, 1.0));

                // Substrate-modulated attention capability.
                core_values.insert(
                    CoreComponent::Attention,
                    input.phi_attention_weight as f64 * input.attention_capability,
                );
                core_values.insert(CoreComponent::Recursion, input.hot_depth);

                // Efficacy: precision-weighted prediction error.
                // Raw (1-PE) conflates low error with high efficacy — a sleeping
                // system also has low PE. Precision weighting (inverse variance of
                // recent PE) distinguishes genuine predictive success from inactivity.
                let pe = input.prediction_error as f64;
                let precision = if input.prediction_precision > 0.0 {
                    input.prediction_precision as f64
                } else {
                    1.0 // fallback: unweighted
                };
                // High precision + low PE = genuine efficacy
                // Low precision + low PE = uncertain (dampen efficacy)
                let efficacy = ((1.0 - pe) * precision.sqrt().min(2.0)).clamp(0.0, 1.0);
                core_values.insert(CoreComponent::Efficacy, efficacy);

                // Approach C: Drift-driven epistemic humility + knowledge grounding + coherence
                // High moral drift → attenuate Knowledge component in EquationV2.
                // Knowledge grounding blends in verified factual content as epistemic anchor.
                // Knowledge coherence further modulates by graph quality (size, calibration,
                // contradiction-freeness) — a large, well-calibrated, consistent knowledge
                // base strengthens the epistemic contribution to consciousness.
                // Science: Epistemic humility during value shifts — if your moral
                // stance is changing rapidly, "knowledge" claims carry less weight.
                // Science: Mercier & Sperber (2017) — grounded knowledge strengthens
                // epistemic claims by anchoring them in verified factual content.
                // Science: Stanovich (2009) — epistemic rationality; Guo et al. (2017) — calibration.
                let blend = super::super::thresholds::KNOWLEDGE_GROUNDING_EPISTEMIC_BLEND;
                let coherence_weight =
                    super::super::thresholds::KNOWLEDGE_COHERENCE_CONSCIOUSNESS_WEIGHT;
                let safe_grounding = if input.knowledge_grounding.is_finite() {
                    input.knowledge_grounding
                } else {
                    0.5
                };
                let safe_coherence = if input.knowledge_coherence.is_finite() {
                    input.knowledge_coherence
                } else {
                    0.0
                };
                let effective_epistemic = if self.moral_coupling.enabled {
                    let drift_ratio =
                        (input.moral_drift / self.moral_coupling.drift_saturation).min(1.0);
                    let attenuation =
                        1.0 - drift_ratio * self.moral_coupling.drift_epistemic_attenuation;
                    let drift_attenuated = input.epistemic_quality * attenuation;
                    drift_attenuated * (1.0 - blend) + safe_grounding * blend
                } else {
                    input.epistemic_quality * (1.0 - blend) + safe_grounding * blend
                };
                // Blend in knowledge coherence: a small additive nudge (capped at weight)
                // based on graph quality metrics, applied after drift attenuation.
                let effective_epistemic =
                    (effective_epistemic + safe_coherence * coherence_weight).clamp(0.0, 1.0);

                // CfC temporal coherence → consciousness (Clark 2013: temporal integration
                // supports unified experience). Small additive nudge to epistemic quality.
                let safe_temporal = if input.temporal_coherence_phi.is_finite() {
                    input.temporal_coherence_phi as f64
                } else {
                    0.0
                };
                let effective_epistemic = (effective_epistemic
                    + safe_temporal
                        * super::super::thresholds::TEMPORAL_COHERENCE_CONSCIOUSNESS_WEIGHT)
                    .clamp(0.0, 1.0);

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
                // Cache PAC modulation and binding for CTC wiring
                #[cfg(feature = "ctc_wiring")]
                {
                    self.cache.last_pac_modulation = result.pac_modulation;
                    self.cache.last_binding_coherence =
                        *result.core_breakdown.get(&crate::consciousness::consciousness_equation_v2::CoreComponent::Binding).unwrap_or(&0.0);
                }
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
        if equation_v2_consciousness > super::super::thresholds::EQ_V2_HIGH_THRESHOLD {
            let boost = (equation_v2_consciousness
                - super::super::thresholds::EQ_V2_HIGH_THRESHOLD)
                * super::super::thresholds::EQ_V2_CONFIDENCE_SCALE;
            confidence_delta += boost as f32;
            episodic_consolidation_boost = Some(
                (equation_v2_consciousness - super::super::thresholds::EQ_V2_HIGH_THRESHOLD)
                    * super::super::thresholds::EQ_V2_CONSOLIDATION_SCALE,
            );
        } else if equation_v2_consciousness > 0.0
            && equation_v2_consciousness < super::super::thresholds::EQ_V2_LOW_THRESHOLD
        {
            exploration_delta += super::super::thresholds::EQ_V2_EXPLORATION_NUDGE;
        }

        // ═══════════════════════════════════════════════════════════════════
        // LAYER 4: Unified Consciousness Pipeline — end-to-end
        // advance() every cycle (builds state), process() every 97 (with binding)
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        // Fill sensory buffer (reused across cycles to avoid per-cycle Vec<f64> alloc).
        // Take ownership temporarily to satisfy borrow checker (pipeline needs &mut self field).
        let mut sensory_buf = std::mem::take(&mut self.sensory_buffer);
        sensory_buf.resize(64, 0.0);
        for i in 0..64 {
            sensory_buf[i] = if input.hv16.get_bit(i) != 0 {
                1.0
            } else {
                -1.0
            };
        }
        let pipeline_consciousness =
            if let Some(ref mut pipeline) = self.unified_consciousness_pipeline {
                if input.cycle % 97 == 0 && input.cycle > 0 {
                    // Full process with oscillatory binding
                    match pipeline.process(&sensory_buf) {
                        Ok(moment) => {
                            self.cache.last_pipeline_consciousness = moment.consciousness;
                            moment.consciousness
                        }
                        Err(_) => self.cache.last_pipeline_consciousness,
                    }
                } else {
                    // Lightweight advance: HDC + LTC + state (no binding)
                    if let Ok(c) = pipeline.advance(&sensory_buf) {
                        self.cache.last_pipeline_consciousness = c;
                    }
                    self.cache.last_pipeline_consciousness
                }
            } else {
                0.0
            };
        self.sensory_buffer = sensory_buf;
        let pipeline_us = t.elapsed().as_micros() as u64;

        // Pipeline feedback: high consciousness → learning coherence
        // Science: Dehaene (2011) — global workspace broadcasts learning signals
        if pipeline_consciousness > super::super::thresholds::PIPELINE_CONSCIOUSNESS_THRESHOLD {
            let pipeline_lr_scale = 1.0
                + (pipeline_consciousness
                    - super::super::thresholds::PIPELINE_CONSCIOUSNESS_THRESHOLD)
                    * super::super::thresholds::PIPELINE_LR_SCALE;
            lr_factor *= pipeline_lr_scale as f32;
        }

        // ═══════════════════════════════════════════════════════════════════
        // LAYER 5: IIT 4.0 Concept Structure Analysis (Albantakis 2023)
        // Co-prime interval 101 (~3x/sec at 31Hz). O(2^n) per mechanism,
        // but n = active modal channels (typically 3-6), so fast enough.
        // ═══════════════════════════════════════════════════════════════════
        #[cfg(feature = "iit4")]
        if input.cycle % 101 == 0 {
            if let Some(ref mmi) = self.multi_modal_integrator {
                let components = mmi.component_hvs_for_iit4();
                if components.len() >= 2 {
                    let calc = symthaea_core::consciousness_metrics::IIT4Calculator::new();
                    let result = calc.analyze(&components);
                    // Feed IIT4 Big Phi into learning modulation
                    // More integrated concept structure → sharper learning
                    if result.big_phi > 0.0 {
                        lr_factor *= 1.0 + (result.big_phi * 0.05) as f32;
                    }
                }
            }
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
        let sht_2a_boost = (input.sht_2a_signal as f64
            - super::super::thresholds::BATH_5HT2A_BASELINE)
            * super::super::thresholds::BATH_5HT2A_SCALE;
        // GABA-A dampens global gain → consciousness reduction
        let gaba_a_dampen = -(input.gaba_a_signal as f64
            - super::super::thresholds::BATH_GABA_BASELINE)
            * super::super::thresholds::BATH_GABA_SCALE;
        // Low entropy (stuck attractor) → consciousness depression
        let entropy_factor = if input.attractor_detected {
            super::super::thresholds::BATH_ENTROPY_ATTRACTOR_PENALTY
        } else {
            0.0
        };
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

        // Governance collective Phi → consciousness coupling
        // High collective integration during governance → social consciousness boost ±2%.
        // Neutral at 0.0 (no governance data) — only applies when mycelix feature is active.
        // Science: Woolley et al. (2010) — collective intelligence factor from social sensitivity.
        let governance_phi_factor = if input.governance_collective_phi > 0.01 {
            (input.governance_collective_phi - 0.5)
                * super::super::thresholds::GOV_CONSCIOUSNESS_MODULATION
        } else {
            0.0
        };

        // Glyph coherence → consciousness coupling
        // High symbolic integration across 11 Field Modalities → consciousness boost ±2%.
        // Neutral at 0.0 (no glyph data). Only active with feature `glyph_codex`.
        // Science: Jung (1959) — archetypal integration deepens conscious awareness.
        let glyph_coherence_factor = if input.glyph_coherence > 0.01 {
            (input.glyph_coherence - 0.5) * super::super::thresholds::GLYPH_CONSCIOUSNESS_MODULATION
        } else {
            0.0
        };

        // CPG sync → consciousness coupling (Varela et al. 2001)
        // Full synchrony (1.0) → +5% consciousness, no sync (0.0) → −5%.
        // Science: Varela et al. (2001) — large-scale neural synchrony correlates
        // with conscious awareness; Engel & Singer (2001) — binding-by-synchrony.
        let cpg_sync_factor = (input.cpg_sync_index - 0.5)
            * 2.0
            * super::super::thresholds::CPG_SYNC_PHI_MODULATION_AMPLITUDE as f64;

        unified_consciousness = (unified_consciousness
            + sht_2a_boost as f64
            + gaba_a_dampen as f64
            + entropy_factor
            + moral_dampen
            + cantor_depth_factor
            + governance_phi_factor
            + glyph_coherence_factor
            + cpg_sync_factor)
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
