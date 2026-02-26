//! Advanced subsystem updates extracted from cycle.rs.
//!
//! Contains: hierarchical LTC, evolution coordinator, holographic analyzer,
//! differentiable consciousness, affective consciousness, unified pipeline,
//! multi-modal integration, synthetic grounding, epistemic gate, primitive validation,
//! cross-module feedback, meta-cognitive reasoner, code primitive router,
//! empathic unification, multi-objective evolution, resonator codebook growth.

use std::time::Instant;

use super::CognitiveLoopService;
use crate::consciousness::cross_modal_binding::Modality;

/// Values computed by the advanced subsystems phase.
pub(crate) struct SubsystemMetrics {
    pub hierarchical_ltc_phi: f32,
    pub evolution_generation: usize,
    pub evolution_phi_delta: f64,
    pub holographic_unity: f64,
    pub holographic_binding: f64,
    pub consciousness_gradient_magnitude: f64,
    pub consciousness_limiting_component: String,
    pub affect_cons_valence: f32,
    pub affect_cons_arousal: f32,
    pub pipeline_consciousness: f64,
    pub multimodal_integrated_phi: f64,
    pub consciousness_state_label: String,
    pub consciousness_state_level: f64,
    pub epistemic_gate_confidence: f32,
    pub epistemic_gate_approved: bool,
    pub primitive_validation_phi_gain: f64,
    pub primitive_validation_p_value: f64,
    pub meta_reasoning_confidence: f64,
    pub meta_reasoning_insights: usize,
    pub code_primitives_selected: usize,
    pub empathic_compassion: f64,
    pub empathic_tone_adj: f64,
    pub multi_obj_frontier_size: usize,
    pub grid_encoding_norm: f32,
    pub grid_spatial_complexity: f32,
}

impl CognitiveLoopService {
    /// Run advanced subsystem updates.
    ///
    /// This is extracted from cycle.rs lines ~2372-2900.
    /// All logic and behavior is preserved exactly.
    pub(crate) fn run_advanced_subsystems(
        &mut self,
        state: &super::CycleState<'_>,
        active_primitive_names: &[String],
        module_timings: &mut super::ModuleTimings,
    ) -> SubsystemMetrics {
        let hv16_cached = *state.hv16_cached;
        let unified_psi = state.unified_psi;
        let coherence = state.coherence;
        let prediction_error = state.prediction_error;
        let phi_attention_weight = state.phi_attention_weight;
        let _compressed_state = state.compressed_state;

        // ── Phase 19+20: Attention budget gating (reactive + predictive) ──────
        // Science: Kahneman (1973) + Botvinick & Braver (2015) — reactive gating
        // kicks in after 3+ consecutive exceeded cycles; predictive gating
        // preemptively doubles intervals when >80% budget consumed at midpoint.
        let budget_gated = (state.attention_budget_exceeded
            && self.stats.attention_budget_exceeded_count > 3)
            || state.predictive_budget_gated;
        let budget_interval_mult: usize = if budget_gated { 2 } else { 1 };
        if budget_gated {
            self.stats.attention_budget_gated_count += 1;
        }
        let input = state.input;
        // ═══════════════════════════════════════════════════════════════════════
        // HIERARCHICAL LTC: Distributed temporal processing with local circuits
        // Local circuits + global integrator. Step propagates temporal dynamics;
        // read consciousness metrics (phi, workspace access, binding coherence).
        // Science: Hasani et al. (2021), Dehaene et al. (2003).
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let hierarchical_ltc_phi = if let Some(ref mut hltc) = self.hierarchical_ltc {
            if self.stats.total_cycles % 11 == 0 {
                let input_vec: Vec<f32> = (0..64)
                    .map(|i| {
                        if hv16_cached.get_bit(i) != 0 {
                            1.0f32
                        } else {
                            -1.0f32
                        }
                    })
                    .collect();
                hltc.inject_distributed(&input_vec);
                let _ = hltc.step();
                hltc.estimate_phi()
            } else {
                0.0
            }
        } else {
            0.0
        };
        module_timings.hierarchical_ltc = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Hierarchical LTC Phi cross-validates spectral MIP
        // Science: Independent Phi estimates should converge; divergence signals instability.
        // When they agree, boost confidence. When they diverge, increase exploration —
        // the system is uncertain about its own integration level (Tononi 2015, §3.1).
        if hierarchical_ltc_phi > 0.1 {
            let spectral_phi = unified_psi as f32;
            let phi_divergence = (hierarchical_ltc_phi - spectral_phi).abs();
            if phi_divergence < 0.2 {
                // Phi estimates converge → strong confidence in integration measure
                let convergence_boost = (0.2 - phi_divergence) * 0.05;
                self.prediction_confidence =
                    (self.prediction_confidence + convergence_boost).clamp(0.0, 1.0);
            } else if phi_divergence > 0.4 {
                // Significant divergence → epistemic uncertainty about integration
                // Attenuated 50%: NE exploration_delta covers surprise-driven exploration
                let divergence_penalty = (phi_divergence - 0.4).min(0.3) * 0.015;
                self.prediction_confidence =
                    (self.prediction_confidence - divergence_penalty).max(0.0);
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + divergence_penalty).clamp(0.0, 1.0);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // EVOLUTION COORDINATOR: Stateful co-evolution of primitives + architecture
        // Replaces one-shot PrimitiveEvolution with cross-generation Thompson sampling.
        // The coordinator manages its own Interleaved schedule internally.
        // EXPENSIVE — called every 199 cycles (actual evolution runs every 5th step, co-prime).
        // Science: Holland (1975), Kauffman (1993), Thompson (1933).
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (evolution_generation, evolution_phi_delta) =
            if let Some(ref mut coordinator) = self.evolution_coordinator {
                if self.stats.total_cycles % 199 == 0 && self.stats.total_cycles > 0 {
                    match coordinator.step() {
                        Ok(result) => (result.generation, result.primitive_psi_delta),
                        Err(_) => (coordinator.generation(), 0.0),
                    }
                } else {
                    (coordinator.generation(), 0.0)
                }
            } else {
                (0, 0.0)
            };
        module_timings.primitive_evolution = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Evolution delta → confidence/exploration + LR (Phase 18 closure)
        // Science: Holland (1975) — evolutionary fitness signals drive adaptive behavior.
        // Positive delta: evolution improving → boost confidence + LR (exploit).
        // Negative delta: evolution regressing → boost exploration (search harder).
        if evolution_phi_delta > 0.01 {
            let evo_boost = 1.0 + (evolution_phi_delta * 0.1).min(0.05) as f32; // up to +5% LR
            self.carryover.learning.subsystem_lr_factor *= evo_boost;
            // Phase 18: Positive delta → boost confidence (evolution is working)
            let conf_boost = (evolution_phi_delta * 0.05).min(0.03) as f32;
            self.prediction_confidence = (self.prediction_confidence + conf_boost).clamp(0.0, 1.0);
            self.stats.evolution_feedback_count += 1;
        } else if evolution_phi_delta < -0.01 {
            // Phase 18: Negative delta → boost exploration urge (need to search harder)
            let explore_boost = ((-evolution_phi_delta) * 0.08).min(0.04) as f32;
            self.curiosity_drive.exploration_urge =
                (self.curiosity_drive.exploration_urge + explore_boost).clamp(0.0, 1.0);
            self.stats.evolution_feedback_count += 1;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS HOLOGRAPHY: Interference-based binding and holographic recall
        // Encodes current experience as holographic pattern; analyzes coherence,
        // unity score, and binding strength via interference patterns.
        // Science: Pribram (1971), Gabor (1946), Bohm (1980).
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (holographic_unity, holographic_binding) =
            if let Some(ref mut ha) = self.holographic_analyzer {
                if self.stats.total_cycles % (19 * budget_interval_mult) == 0 {
                    let content: Vec<f64> = (0..64)
                        .map(|i| {
                            if hv16_cached.get_bit(i) != 0 {
                                1.0
                            } else {
                                -1.0
                            }
                        })
                        .collect();
                    ha.encode_experience(&content, &format!("cycle_{}", self.stats.total_cycles));
                    let analysis = ha.analyze();
                    self.carryover.consciousness.last_holographic_unity = analysis.unity_score;
                    (analysis.unity_score, analysis.binding_strength)
                } else {
                    (self.carryover.consciousness.last_holographic_unity, 0.0)
                }
            } else {
                (0.0, 0.0)
            };
        module_timings.consciousness_holography = _t.elapsed().as_micros() as u64;

        // FEEDBACK: High holographic unity boosts prediction confidence
        // Science: Pribram (1991) — holographic encoding enables stable predictions
        if holographic_unity > 0.7 {
            let unity_boost = (holographic_unity - 0.7) * 0.03;
            self.prediction_confidence =
                (self.prediction_confidence + unity_boost as f32).clamp(0.0, 1.0);
        }
        // FEEDBACK: Binding strength modulates learning rate
        // Strong binding = coherent representations → safe to learn faster
        if holographic_binding > 0.7 {
            self.carryover.learning.subsystem_lr_factor *= 1.01;
        } else if holographic_binding > 0.0 && holographic_binding < 0.3 {
            // Weak binding = fragmented representations → dampen learning
            self.carryover.learning.subsystem_lr_factor *= 0.99;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // DIFFERENTIABLE CONSCIOUSNESS: Gradient-based consciousness optimization
        // Computes ∂C/∂component to identify which factor limits consciousness most.
        // Science: Bengio (2017), Tononi (2004), Oizumi et al. (2014).
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (consciousness_gradient_magnitude, consciousness_limiting_component) =
            if let Some(ref dc) = self.differentiable_consciousness {
                if self.stats.total_cycles % (23 * budget_interval_mult) == 0
                    && self.stats.total_cycles > 0
                {
                    use crate::consciousness::consciousness_equation_v2::{
                        ConsciousnessStateV2, CoreComponent,
                    };
                    use std::collections::HashMap;
                    let mut core_values = HashMap::new();
                    core_values.insert(CoreComponent::Integration, unified_psi.clamp(0.0, 1.0));
                    core_values.insert(CoreComponent::Binding, coherence as f64);
                    core_values.insert(CoreComponent::Workspace, coherence as f64 * 0.8);
                    core_values.insert(CoreComponent::Attention, phi_attention_weight as f64);
                    core_values.insert(CoreComponent::Recursion, 0.5);
                    core_values.insert(CoreComponent::Efficacy, 1.0 - prediction_error as f64);
                    core_values.insert(
                        CoreComponent::Knowledge,
                        self.carryover.quality.last_epistemic_quality,
                    );
                    let state = ConsciousnessStateV2 {
                        core_values,
                        extended_values: HashMap::new(),
                        phase_coherence: HashMap::new(),
                        substrate_feasibility: 1.0,
                        timestamp: self.stats.total_cycles as u64,
                        context: String::new(),
                    };
                    let (_value, gradient) = dc.forward(&state);
                    let (component, _grad_val, _suggestion) = dc.suggest_improvement(&state);
                    self.carryover.quality.last_gradient_magnitude = gradient.magnitude;
                    (gradient.magnitude, format!("{:?}", component))
                } else {
                    (
                        self.carryover.quality.last_gradient_magnitude,
                        String::new(),
                    )
                }
            } else {
                (0.0, String::new())
            };
        module_timings.differentiable_consciousness = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Large consciousness gradients drive exploration
        // Science: Bengio (2017) — gradient information guides search
        if consciousness_gradient_magnitude > 0.5 {
            // Attenuated 50%: NE exploration_delta handles consciousness gradient exploration
            let gradient_explore = (consciousness_gradient_magnitude - 0.5).clamp(0.0, 0.5) * 0.05;
            self.curiosity_drive.exploration_urge =
                (self.curiosity_drive.exploration_urge + gradient_explore as f32).clamp(0.0, 1.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // AFFECTIVE CONSCIOUSNESS: Valence-arousal-dominance affect tracking
        // Lightweight: decay every cycle, process stimulus every 10 cycles.
        // Science: Russell (2003), Barrett (2017), Colombetti (2014).
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (affect_cons_valence, affect_cons_arousal) =
            if let Some(ref mut ac) = self.affective_consciousness {
                ac.decay(0.05);
                if self.stats.total_cycles % (11 * budget_interval_mult) == 0 {
                    let valence = 1.0 - 2.0 * prediction_error;
                    let base_affect = crate::consciousness::affective_consciousness::CoreAffect {
                        valence,
                        arousal: prediction_error.abs().clamp(0.0, 1.0),
                        dominance: self.prediction_confidence * 2.0 - 1.0,
                    };
                    let affect = ac.process_stimulus(
                        &format!("cycle_{}", self.stats.total_cycles),
                        Some(base_affect),
                    );
                    self.carryover.quality.last_affective_valence = affect.valence;
                    (affect.valence, affect.arousal)
                } else {
                    let affect = ac.current_affect();
                    (affect.valence, affect.arousal)
                }
            } else {
                (0.0, 0.0)
            };
        module_timings.affective_consciousness = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Negative affect strengthens caution (lower confidence)
        if let Some(ref ac) = self.affective_consciousness {
            let affect = ac.current_affect();
            if affect.valence < -0.3 {
                self.prediction_confidence =
                    (self.prediction_confidence + affect.valence * 0.02).max(0.0);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // UNIFIED CONSCIOUSNESS PIPELINE: End-to-end sensory→consciousness
        // EXPENSIVE — runs every 47 cycles (co-prime). Combines HDC, LTC, binding, equation.
        // Science: Dehaene (2011), Tononi (2004), Hasani et al. (2021).
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let pipeline_consciousness = if let Some(ref mut pipeline) =
            self.unified_consciousness_pipeline
        {
            if self.stats.total_cycles % 47 == 0 && self.stats.total_cycles > 0 {
                let sensory: Vec<f64> = (0..64)
                    .map(|i| {
                        if hv16_cached.get_bit(i) != 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect();
                match pipeline.process(&sensory) {
                    Ok(moment) => {
                        self.carryover.quality.last_pipeline_consciousness = moment.consciousness;
                        moment.consciousness
                    }
                    Err(_) => self.carryover.quality.last_pipeline_consciousness,
                }
            } else {
                self.carryover.quality.last_pipeline_consciousness
            }
        } else {
            0.0
        };
        module_timings.unified_consciousness_pipeline = _t.elapsed().as_micros() as u64;

        // FEEDBACK: High pipeline consciousness strengthens learning toward coherence
        // Science: Dehaene (2011) — global workspace broadcasts learning signals
        if pipeline_consciousness > 0.6 {
            let pipeline_lr_scale = 1.0 + (pipeline_consciousness - 0.6) * 0.5;
            self.fep_lr_boost = (self.fep_lr_boost * pipeline_lr_scale as f32).clamp(1.0, 2.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // MULTI-MODAL INTEGRATION: Phi-guided cross-modal binding
        // Science: Damasio (1994), Mesulam (1998), Ghazanfar & Schroeder (2006).
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let multimodal_integrated_phi = if let Some(ref mut mmi) = self.multi_modal_integrator {
            if self.stats.total_cycles % 13 == 0 && self.stats.total_cycles > 0 {
                use crate::consciousness::multi_modal_integration::ModalInput;
                let visual_input = ModalInput::new(Modality::Visual, hv16_cached, coherence as f64);
                let temporal_input =
                    ModalInput::new(Modality::Temporal, hv16_cached, unified_psi.clamp(0.0, 1.0));
                let result = mmi.integrate(&[visual_input, temporal_input]);
                self.carryover.consciousness.last_multimodal_phi = result.integrated_phi;
                result.integrated_phi
            } else {
                self.carryover.consciousness.last_multimodal_phi
            }
        } else {
            0.0
        };
        module_timings.multi_modal_integration = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Strong multimodal integration improves learning precision
        // Science: Ghazanfar & Schroeder (2006) — cross-modal binding enables precise learning
        if multimodal_integrated_phi > 0.5 {
            let phi_confidence = (multimodal_integrated_phi - 0.5) * 0.04;
            self.prediction_confidence =
                (self.prediction_confidence + phi_confidence as f32).clamp(0.0, 1.0);
            let phi_subsystem_lr = 1.0 + (multimodal_integrated_phi - 0.5) * 0.4;
            self.carryover.learning.subsystem_lr_factor *= phi_subsystem_lr as f32;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(0.8, 1.2);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // SYNTHETIC STATES NSM GROUNDING: Classify current consciousness state
        // Maps current BinaryHV to closest consciousness state via NSM primitives.
        // Science: Wierzbicka (1996) — Natural Semantic Metalanguage.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (consciousness_state_label, consciousness_state_level_val) =
            if let Some(ref sg) = self.synthetic_grounding {
                if self.stats.total_cycles % 97 == 0 {
                    let similar = sg.find_similar(&hv16_cached, 0.1);
                    if let Some((state_type, _sim)) = similar.first() {
                        let label = format!("{:?}", state_type);
                        let level = state_type.consciousness_level();
                        self.carryover.quality.last_consciousness_state = label.clone();
                        (label, level)
                    } else {
                        (self.carryover.quality.last_consciousness_state.clone(), 0.0)
                    }
                } else {
                    (self.carryover.quality.last_consciousness_state.clone(), 0.0)
                }
            } else {
                (String::new(), 0.0)
            };
        module_timings.synthetic_grounding = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // EPISTEMIC DECISION GATE: Evaluate input through Graceful Ignorance
        // Provides confidence-based gating before actions.
        // Science: Kruger & Dunning (1999), Schwartz (2004) — epistemic humility.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (epistemic_gate_confidence, epistemic_gate_approved) = if let Some(ref mut gate) =
            self.epistemic_gate
        {
            if self.stats.total_cycles % 7 == 0 {
                let action_risk = (1.0 - self.prediction_confidence).clamp(0.0, 1.0);
                let decision = gate.evaluate(input, action_risk);
                let (confidence, approved) = match &decision {
                        crate::consciousness::gis_integration::EpistemicDecision::Proceed { confidence } => (*confidence, true),
                        crate::consciousness::gis_integration::EpistemicDecision::ProceedWithCaveat { confidence, .. } => (*confidence, true),
                        crate::consciousness::gis_integration::EpistemicDecision::Defer { .. } => (0.0, false),
                        crate::consciousness::gis_integration::EpistemicDecision::RequestGuidance { .. } => (0.0, false),
                        crate::consciousness::gis_integration::EpistemicDecision::OutOfDomain { .. } => (0.0, false),
                    };
                self.carryover.quality.last_epistemic_confidence = confidence;
                (confidence, approved)
            } else {
                (self.carryover.quality.last_epistemic_confidence, true)
            }
        } else {
            (0.5, true)
        };
        module_timings.epistemic_gate = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Low epistemic confidence reduces prediction confidence
        if epistemic_gate_confidence < 0.3 && !epistemic_gate_approved {
            self.prediction_confidence = (self.prediction_confidence - 0.03).max(0.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // GRID ENCODER: Spatial reasoning via HDC grid encoding.
        // Encodes the current input as a 2D grid (treating bytes as color indices)
        // to extract spatial structure metrics (norm + complexity).
        // Science: Chollet (2019) — Abstraction and Reasoning Corpus.
        // Amortized: every 13 cycles (lightweight but not needed every tick).
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (grid_encoding_norm, grid_spatial_complexity) =
            if let Some(ref encoder) = self.grid_encoder {
                if self.stats.total_cycles % 13 == 0 {
                    // Interpret input bytes as a small grid (up to 8x8)
                    let input_bytes = input.as_bytes();
                    let side = (input_bytes.len() as f32).sqrt().ceil() as usize;
                    let side = side.clamp(1, 8);
                    let num_colors = encoder.num_colors();
                    let mut grid = vec![vec![0u8; side]; side];
                    for (i, &b) in input_bytes.iter().take(side * side).enumerate() {
                        grid[i / side][i % side] = b % num_colors as u8;
                    }
                    let hv = encoder.encode_grid(&grid);
                    let norm = hv.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();

                    // Spatial complexity: ratio of unique colors used to total possible
                    let mut seen = [false; 16];
                    for row in &grid {
                        for &c in row {
                            if (c as usize) < seen.len() {
                                seen[c as usize] = true;
                            }
                        }
                    }
                    let unique = seen.iter().filter(|&&x| x).count() as f32;
                    let complexity = (unique / num_colors.max(1) as f32).clamp(0.0, 1.0);

                    self.carryover.quality.last_grid_norm = norm;
                    self.carryover.quality.last_grid_complexity = complexity;
                    (norm, complexity)
                } else {
                    (
                        self.carryover.quality.last_grid_norm,
                        self.carryover.quality.last_grid_complexity,
                    )
                }
            } else {
                (0.0, 0.0)
            };
        module_timings.grid_encoder = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PRIMITIVE VALIDATION: One-shot empirical Φ validation at cycle 500
        // Runs StandardExperiments::tier1_mathematical() once to validate that
        // primitives genuinely improve consciousness (Φ) vs baseline.
        // Science: Popper (1959) — falsifiability, scientific method.
        // ═══════════════════════════════════════════════════════════════════════
        let (primitive_validation_phi_gain, primitive_validation_p_value) =
            if self.primitive_validation_result.is_none()
                && self.primitive_processor.is_some()
                && self.stats.total_cycles == 500
            {
                let mut experiment =
                crate::consciousness::primitive_validation::StandardExperiments::tier1_mathematical(
                );
                match experiment.run() {
                    Ok(results) => {
                        let gain = results.statistics.mean_phi_gain;
                        let p = results.statistics.p_value;
                        self.primitive_validation_result = Some((gain, p));
                        (gain, p)
                    }
                    Err(_) => (0.0, 1.0),
                }
            } else {
                self.primitive_validation_result.unwrap_or((0.0, 1.0))
            };

        // FEEDBACK: Validated primitives boost LR; falsified primitives dampen it
        // Science: Popper (1959) — if primitives don't improve Φ, reduce their influence
        if let Some((phi_gain, p_value)) = self.primitive_validation_result {
            if p_value < 0.05 && phi_gain > 0.0 {
                // Significant positive effect → boost primitive subsystem LR
                self.carryover.learning.subsystem_lr_factor *=
                    1.0 + (phi_gain * 0.02).min(0.03) as f32;
            } else if p_value < 0.05 && phi_gain < 0.0 {
                // Significant negative effect → dampen primitive processing
                self.carryover.learning.subsystem_lr_factor *= 0.98;
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // CROSS-MODULE FEEDBACK: Modules inform each other for emergent behavior
        // Science: Varela (1991) — autopoietic coupling, Beer (2000) — circular
        // causality in cognitive systems.
        // ═══════════════════════════════════════════════════════════════════════

        // 1. Consciousness state level modulates urgency: low consciousness → Critical
        //    (run all subsystems to diagnose), high consciousness → can tolerate Cruise
        if consciousness_state_level_val > 0.0
            && consciousness_state_level_val < 0.3
            && self.carryover.urgency.urgency != super::types::CycleUrgency::Critical
        {
            self.carryover.urgency.urgency = super::types::CycleUrgency::Normal;
        }

        // 2. Gradient analysis → adaptive exploration: large gradients mean the system
        //    has clear direction for improvement → focus rather than explore
        if consciousness_gradient_magnitude > 1.0 {
            // Strong gradient = clear optimization direction → reduce random exploration
            self.curiosity_drive.boredom = (self.curiosity_drive.boredom - 0.05).max(0.0);
        } else if consciousness_gradient_magnitude > 0.0 && consciousness_gradient_magnitude < 0.1 {
            // Near-zero gradient = plateau → boost exploration to escape
            self.curiosity_drive.boredom = (self.curiosity_drive.boredom + 0.03).min(1.0);
        }

        // 3. Holographic unity gates learning: high unity = coherent representation →
        //    safe to learn aggressively. Low unity = fragmented → be conservative.
        if holographic_unity > 0.8 {
            self.carryover.learning.subsystem_lr_factor *= 1.02;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(0.8, 1.2);
        } else if holographic_unity < 0.2 && holographic_unity > 0.0 {
            self.carryover.learning.subsystem_lr_factor *= 0.98;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(0.8, 1.2);
        }

        // 4. Pipeline consciousness → epistemic gating: high pipeline consciousness
        //    means the system has strong global workspace → relax epistemic threshold
        if pipeline_consciousness > 0.7 {
            self.carryover.quality.last_epistemic_confidence =
                (self.carryover.quality.last_epistemic_confidence + 0.02).min(1.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // META-COGNITIVE REASONER: Self-reflective reasoning about reasoning
        // Reflects on context detection confidence, strategy effectiveness, and
        // learns meta-patterns across reasoning episodes.
        // Amortized: every 47 cycles (heavy — creates CandidatePrimitives + chain, co-prime).
        // Science: Flavell (1979), Nelson & Narens (1990) — metacognition hierarchy.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (meta_reasoning_confidence, meta_reasoning_insights) = if let Some(ref mut reasoner) =
            self.meta_cognitive_reasoner
        {
            if self.stats.total_cycles % 47 == 0 && self.stats.total_cycles > 0 {
                // Build lightweight candidate primitives from active set
                let candidates: Vec<crate::consciousness::primitive_evolution::CandidatePrimitive> =
                    active_primitive_names
                        .iter()
                        .take(3)
                        .map(
                            |name| crate::consciousness::primitive_evolution::CandidatePrimitive {
                                name: name.clone(),
                                tier: symthaea_core::hdc::PrimitiveTier::NSM,
                                definition: name.clone(),
                                fitness: unified_psi,
                                encoding: symthaea_core::hdc::BinaryHV::random(42),
                                epistemic_coordinate: Default::default(),
                                harmonic_alignment: 0.5,
                            },
                        )
                        .collect();
                let mut chain =
                    crate::consciousness::primitive_reasoning::ReasoningChain::new(hv16_cached);
                match reasoner.meta_reason(input, candidates, &mut chain) {
                    Ok(result) => (result.meta_confidence, result.meta_insights.len()),
                    Err(_) => (0.5, 0),
                }
            } else {
                (0.5, 0)
            }
        } else {
            (0.5, 0)
        };
        module_timings.meta_cognitive_reasoning = _t.elapsed().as_micros() as u64;

        // FEEDBACK: High meta-cognitive confidence boosts learning rate
        // The MetaCognitiveReasoner path is fully deterministic (ContextAwareOptimizer
        // uses weighted selection, not RNG). Safe for genesis determinism.
        // Science: Nelson & Narens (1990) — monitoring-control loop
        if meta_reasoning_confidence > 0.7 {
            let meta_boost = (meta_reasoning_confidence - 0.7) * 0.1;
            self.fep_lr_boost = (self.fep_lr_boost + meta_boost as f32).clamp(1.0, 2.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // CODE PRIMITIVE ROUTER: Consciousness-aware code reasoning
        // Selects optimal code-tier primitives when input looks code-related.
        // Amortized: every 11 cycles (lightweight O(1) lookup, co-prime).
        // Science: Plate (2003) — VSA for structured representations.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let code_primitives_selected = if let Some(ref router) = self.code_primitive_router {
            if self.stats.total_cycles % 11 == 0 {
                // Heuristic: detect code-related input
                let code_related = input.contains("code")
                    || input.contains("function")
                    || input.contains("debug")
                    || input.contains("refactor")
                    || input.contains("parse")
                    || input.contains("compile");
                if code_related {
                    let operation = if input.contains("debug") {
                        crate::consciousness::code_primitives::CodeOperation::Debug
                    } else if input.contains("refactor") {
                        crate::consciousness::code_primitives::CodeOperation::Refactor
                    } else if input.contains("parse") {
                        crate::consciousness::code_primitives::CodeOperation::Parse
                    } else {
                        crate::consciousness::code_primitives::CodeOperation::Explain
                    };
                    router.select_primitives(operation).len()
                } else {
                    0
                }
            } else {
                0
            }
        } else {
            0
        };
        module_timings.code_primitive_routing = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // EMPATHIC UNIFICATION: Resonant empathy via user state inference
        // Senses user emotional state from input, generates compassion response.
        // Amortized: every 11 cycles (lightweight keyword + inference, co-prime).
        // Science: Decety & Jackson (2004) — shared neural representations.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (empathic_compassion, empathic_tone_adj) =
            if let Some(ref mut empathy) = self.empathic_unification {
                if self.stats.total_cycles % 11 == 0 {
                    let context = crate::user_state_inference::ContextKind::Task;
                    let response = empathy.process(input, context);
                    (response.compassion, response.patience_adjustment)
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };
        module_timings.empathic_unification = _t.elapsed().as_micros() as u64;

        // FEEDBACK: High compassion slightly boosts LR (empathic learning bias)
        // The EmpathicUnification path is deterministic (text-based emotion detection,
        // ContextKind input). Instant::now() timestamps are internal only.
        // Science: Decety & Jackson (2004) — shared representations enhance learning
        if empathic_compassion > 0.7 {
            self.carryover.learning.subsystem_lr_factor *=
                1.0 + (empathic_compassion as f32 - 0.7) * 0.02;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(0.8, 1.2);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // MULTI-OBJECTIVE EVOLUTION: Pareto-frontier consciousness optimization
        // Very expensive — runs once every 1000 cycles. Evolves primitives across
        // 5 dimensions (Φ, ∇Φ, Entropy, Complexity, Coherence).
        // Science: Deb et al. (2002) — NSGA-II multi-objective optimization.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let multi_obj_frontier_size = if let Some(ref mut moe) = self.multi_objective_evolution {
            if self.stats.total_cycles % 997 == 0 && self.stats.total_cycles > 0 {
                match moe.evolve() {
                    Ok(result) => result.frontier_size,
                    Err(_) => 0,
                }
            } else {
                0
            }
        } else {
            0
        };
        module_timings.multi_objective_evolution = _t.elapsed().as_micros() as u64;

        SubsystemMetrics {
            hierarchical_ltc_phi,
            evolution_generation,
            evolution_phi_delta,
            holographic_unity,
            holographic_binding,
            consciousness_gradient_magnitude,
            consciousness_limiting_component,
            affect_cons_valence,
            affect_cons_arousal,
            pipeline_consciousness,
            multimodal_integrated_phi,
            consciousness_state_label,
            consciousness_state_level: consciousness_state_level_val,
            epistemic_gate_confidence,
            epistemic_gate_approved,
            primitive_validation_phi_gain,
            primitive_validation_p_value,
            meta_reasoning_confidence,
            meta_reasoning_insights,
            code_primitives_selected,
            empathic_compassion,
            empathic_tone_adj,
            multi_obj_frontier_size,
            grid_encoding_norm,
            grid_spatial_complexity,
        }
    }
}
