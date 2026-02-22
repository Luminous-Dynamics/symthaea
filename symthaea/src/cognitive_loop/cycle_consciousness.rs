//! Consciousness metrics computation extracted from cycle.rs.
//!
//! Contains: primitive consciousness, temporal primitives, lattice, compositionality,
//! value evaluator, consciousness profile, context-aware evolution, semantic value embedder,
//! harmonies integration, composition rules, fiduciary harmonics, primitive reasoning,
//! causal self-explanation, adaptive reasoning, epistemic tiers, phi validation,
//! dissipative consciousness, epistemic conflict, consciousness equation v2.

use std::time::Instant;

use super::CognitiveLoopService;

/// Values computed by the consciousness metrics phase.
/// Passed to later phases that need these results.
pub(crate) struct ConsciousnessMetrics {
    pub primitive_psi: f64,
    pub active_primitive_names: Vec<String>,
    pub temporal_causal_chains: usize,
    pub temporal_continuity: f64,
    pub temporal_max_chain_length: usize,
    pub chain_cycle_numbers: Vec<u64>,
    pub causal_codebook_entries: Vec<(String, Vec<f32>)>,
    pub continuity_replay_needed: bool,
    pub lattice_height: usize,
    pub lattice_width: usize,
    pub lattice_join_concept: Option<String>,
    pub compositionality_total: usize,
    pub value_evaluator_score: f64,
    pub value_evaluator_decision: String,
    pub consciousness_profile_composite: f64,
    pub synergy_enhanced_composite: f64,
    pub emergent_properties_count: usize,
    pub reasoning_context: String,
    pub context_phi_weight: f64,
    pub value_embeddings_created: u64,
    pub value_cache_hit_rate: f32,
    pub harmonies_alignment: f32,
    pub harmonies_approved: bool,
    pub composition_rule_applied: String,
    pub harmonic_field_coherence: f64,
    pub harmonic_love_resonance: f64,
    pub harmonic_interferences: usize,
    pub reasoning_chain_confidence: f32,
    pub reasoning_chain_depth: usize,
    pub causal_relations_count: usize,
    pub causal_avg_confidence: f64,
    pub adaptive_reasoning_phi: f64,
    pub epistemic_quality: f64,
    pub phi_validation_correlation: f64,
    pub dissipative_health: f64,
    pub dissipative_regime: String,
    pub dissipative_entropy_rate: f64,
    pub epistemic_phi_eff: f64,
    pub epistemic_conflict_count: usize,
    pub equation_v2_consciousness: f64,
}

impl CognitiveLoopService {
    /// Compute consciousness metrics from the current cycle state.
    ///
    /// This is Phase "consciousness metrics" — extracted from cycle.rs lines ~1634-2370.
    /// All logic and behavior is preserved exactly.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_consciousness_metrics(
        &mut self,
        hv16_cached: symthaea_core::hdc::BinaryHV,
        unified_psi: f64,
        coherence: f32,
        prediction_error: f32,
        phi_attention_weight: f32,
        compressed_state: &[f32],
        input: &str,
        urgency: super::CycleUrgency,
        module_timings: &mut super::ModuleTimings,
    ) -> ConsciousnessMetrics {
        // ═══════════════════════════════════════════════════════════════════════
        // PRIMITIVE CONSCIOUSNESS: Decompose consciousness state into primitives
        // Provides explainable consciousness by mapping HDC encodings to the
        // 9-tier primitive system with activation tracking and binding.
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // Science: Tononi & Koch (2015) — primitives of consciousness experience
        // ═══════════════════════════════════════════════════════════════════════
        let (primitive_psi, active_primitive_names) =
            if urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
                if let Some(ref mut processor) = self.primitive_processor {
                    let timestamp = self.stats.total_cycles as f64 * 0.02; // 50Hz → seconds
                    let state = processor.process_input(&hv16_cached, timestamp);
                    let names: Vec<String> = state
                        .all_active()
                        .iter()
                        .take(4)
                        .map(|ap| ap.primitive.name.clone())
                        .collect();
                    (state.phi, names)
                } else {
                    (0.0, Vec::new())
                }
            } else {
                (0.0, Vec::new())
            };

        // ═══════════════════════════════════════════════════════════════════════
        // TEMPORAL PRIMITIVES: Allen's Interval Algebra on conscious states
        // [extracted to compute_temporal_primitives_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let (
            temporal_causal_chains,
            temporal_continuity,
            temporal_max_chain_length,
            chain_cycle_numbers,
            causal_codebook_entries,
            continuity_replay_needed,
        ) = self.compute_temporal_primitives_phase(
            hv16_cached,
            unified_psi,
            coherence,
            module_timings,
        );

        // ═══════════════════════════════════════════════════════════════════════
        // PRIMITIVE LATTICE: Structural metrics from tier system
        // [extracted to compute_lattice_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let (lattice_height, lattice_width, lattice_join_concept) =
            self.compute_lattice_phase(&active_primitive_names, module_timings);

        // ═══════════════════════════════════════════════════════════════════════
        // COMPOSITIONALITY ENGINE: Algebraic composition of primitives
        // Tracks composition stats; actual compositions are demand-driven.
        // Science: Category Theory (Mac Lane 1998), HDC algebraic operators.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let compositionality_total =
            if let Some(ref compositionality) = self.compositionality_engine {
                compositionality.get_stats().total_compositions
            } else {
                0
            };
        module_timings.compositionality = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // UNIFIED VALUE EVALUATOR: Seven Harmonies alignment scoring
        // [extracted to compute_value_evaluator_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let (value_evaluator_score, value_evaluator_decision) =
            self.compute_value_evaluator_phase(unified_psi, module_timings);

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS PROFILE: Multi-dimensional consciousness assessment
        // Computes 5-axis profile (Phi, gradient, entropy, complexity, coherence).
        // Amortized: every 10 cycles (involves Phi computation over HDC state).
        // Science: Tononi (2004), Koch (2012) — multi-dimensional consciousness.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        // Maintain ring buffer of last 4 BinaryHVs for multi-component profile
        // Capacity bound: 4 elements — evict before push to prevent transient over-capacity
        if self.carryover.history.recent_hvs.len() >= 4 {
            self.carryover.history.recent_hvs.pop_front();
        }
        self.carryover.history.recent_hvs.push_back(hv16_cached);
        let (
            consciousness_profile_composite,
            synergy_enhanced_composite,
            emergent_properties_count,
        ) = if self.stats.total_cycles % 23 == 0 && self.primitive_processor.is_some() {
            let profile =
                crate::consciousness::consciousness_profile::ConsciousnessProfile::from_components(
                    self.carryover.history.recent_hvs.make_contiguous(),
                );
            let composite = profile.composite;
            // Dimension synergies: discover non-linear interactions between consciousness dims
            let synergy =
                crate::consciousness::dimension_synergies::SynergyProfile::from_base(profile);
            self.carryover.history.last_profile_composite = composite;
            self.carryover.history.last_synergy_composite = synergy.enhanced_composite;
            self.carryover.history.last_emergent_count = synergy.emergent_properties.len();
            (
                composite,
                synergy.enhanced_composite,
                synergy.emergent_properties.len(),
            )
        } else {
            // Non-compute cycle: return cached values
            (
                self.carryover.history.last_profile_composite,
                self.carryover.history.last_synergy_composite,
                self.carryover.history.last_emergent_count,
            )
        };
        module_timings.consciousness_profile = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CONTEXT-AWARE EVOLUTION: Dynamic Φ/Harmonic/Epistemic weighting
        // Detects reasoning context from input text and adjusts objective weights.
        // Runs every cycle (keyword match is O(1) in input length).
        // Science: Gigerenzer (2007) — ecological rationality, context-adaptive reasoning.
        // ═══════════════════════════════════════════════════════════════════════
        let (reasoning_context, context_phi_weight) =
            if let Some(ref optimizer) = self.context_optimizer {
                let ctx = optimizer.detect_context(input, None);
                let weights = optimizer.get_weights_for_context(&ctx);
                (ctx.description().to_string(), weights.phi_weight)
            } else {
                (String::new(), 0.0)
            };

        // ═══════════════════════════════════════════════════════════════════════
        // SEMANTIC VALUE EMBEDDER: Value-aligned embeddings grounded in primitives
        // Projects compressed state into value-aware space using primitive-tier
        // harmony bases. Cached — repeated inputs hit O(1) lookup.
        // Science: Schwartz (2012) — value theory, Kanerva (2009) — HDC semantics.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (value_embeddings_created, value_cache_hit_rate) =
            if let Some(ref mut embedder) = self.semantic_value_embedder {
                if self.stats.total_cycles % 11 == 0 {
                    let continuous = symthaea_core::hdc::ContinuousHV::from_slice(compressed_state);
                    let _concept =
                        embedder.embed(format!("cycle_{}", self.stats.total_cycles), continuous);
                    (
                        embedder.stats().embeddings_created,
                        embedder.cache_hit_rate(),
                    )
                } else {
                    (
                        embedder.stats().embeddings_created,
                        embedder.cache_hit_rate(),
                    )
                }
            } else {
                (0, 0.0)
            };
        module_timings.semantic_value_embedder = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // HARMONIES INTEGRATOR: Per-action ethical alignment via Seven Harmonies
        // Evaluates the current cycle's compressed state as a ValuedAction and
        // scores it against harmony embeddings for approval/rejection.
        // Amortized: every 19 cycles (embedding similarity + scoring, co-prime).
        // Science: Schwartz (2012) — basic human values, Deci & Ryan (2000).
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (harmonies_alignment, harmonies_approved) =
            if let Some(ref mut integrator) = self.harmonies_integrator {
                if self.stats.total_cycles % 19 == 0 {
                    let embedding = symthaea_core::hdc::ContinuousHV::from_slice(compressed_state);
                    let action = crate::consciousness::harmonies_integration::ValuedAction::new(
                        format!("cycle_{}", self.stats.total_cycles),
                        input,
                        embedding,
                    );
                    let eval = integrator.evaluate(&action);
                    (eval.overall_alignment, eval.approved)
                } else {
                    (integrator.stats().avg_alignment, true)
                }
            } else {
                (0.0, true)
            };
        module_timings.harmonies_integration = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Low harmony alignment → reduce confidence (ethical uncertainty)
        if harmonies_alignment > 0.0 && !harmonies_approved {
            self.prediction_confidence = (self.prediction_confidence - 0.02).max(0.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // COMPOSITION RULES: Domain-specific HDC binding operator selection
        // Selects the best composition rule (temporal-physical, mathematical,
        // consciousness, cross-tier) for the top-2 active primitives.
        // Stateless — O(1) rule lookup per cycle.
        // Science: Plate (2003) — holographic reduced representations.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let composition_rule_applied = if let Some(ref rule_engine) = self.composition_rule_engine {
            if active_primitive_names.len() >= 2 {
                let system = symthaea_core::hdc::primitive_system::PrimitiveSystem::global();
                let tier1 = system
                    .get(&active_primitive_names[0])
                    .map(|p| p.tier)
                    .unwrap_or(symthaea_core::hdc::primitive_system::PrimitiveTier::NSM);
                let tier2 = system
                    .get(&active_primitive_names[1])
                    .map(|p| p.tier)
                    .unwrap_or(symthaea_core::hdc::primitive_system::PrimitiveTier::NSM);
                rule_engine.matching_rule_name(tier1, tier2).to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        module_timings.composition_rules = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // FIDUCIARY HARMONICS: Seven Harmonies field coherence + interference
        // [extracted to compute_fiduciary_harmonics_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let (harmonic_field_coherence, harmonic_love_resonance, harmonic_interferences) = self
            .compute_fiduciary_harmonics_phase(
                coherence,
                prediction_error,
                unified_psi,
                module_timings,
            );

        // ═══════════════════════════════════════════════════════════════════════
        // PRIMITIVE REASONING: HDC-based analogical reasoning
        // Runs a quick reasoning chain on the current input for concept binding.
        // Amortized: every 23 cycles (reasoning chains have some compute cost, co-prime).
        // Science: Kanerva (2009) — hyperdimensional analogical reasoning.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (reasoning_chain_confidence, reasoning_chain_depth) =
            if let Some(ref mut reasoner) = self.primitive_reasoner {
                if self.stats.total_cycles % 23 == 0 && self.stats.total_cycles > 0 {
                    let result = reasoner.reason("cognitive_state", &[]);
                    (result.confidence, result.reasoning_chain.len())
                } else {
                    (0.0, 0)
                }
            } else {
                (0.0, 0)
            };
        module_timings.primitive_reasoning = _t.elapsed().as_micros() as u64;

        // FEEDBACK: High reasoning confidence → boost prediction confidence
        if reasoning_chain_confidence > 0.7 {
            let reason_boost = (reasoning_chain_confidence - 0.7) * 0.03; // up to +0.9%
            self.prediction_confidence = (self.prediction_confidence + reason_boost).min(1.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // CAUSAL SELF-EXPLANATION: Pearl causal model of primitive→Phi effects
        // [extracted to compute_causal_self_explanation_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let (causal_relations_count, causal_avg_confidence) = self
            .compute_causal_self_explanation_phase(
                hv16_cached,
                &active_primitive_names,
                primitive_psi,
                module_timings,
            );

        // ═══════════════════════════════════════════════════════════════════════
        // ADAPTIVE REASONING: Q-learning-guided primitive selection
        // Builds reasoning chains with RL-optimized primitive selection.
        // Amortized: every 47 cycles (Q-learning step + chain construction, co-prime).
        // Science: Sutton & Barto (2018) — reinforcement learning + HDC.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let adaptive_reasoning_phi = if let Some(ref mut reasoner) = self.adaptive_reasoner {
            if self.stats.total_cycles % 47 == 0 && self.stats.total_cycles > 0 {
                match reasoner.reason_adaptive(hv16_cached, 5) {
                    Ok(chain) => chain.total_phi,
                    Err(_) => 0.0,
                }
            } else {
                0.0
            }
        } else {
            0.0
        };
        module_timings.adaptive_reasoning = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // EPISTEMIC TIERS: 3-axis epistemic classification of Phi measurements
        // Classifies the current Phi measurement's empirical, normative, and
        // materiality status. Lightweight — computed each cycle.
        // Science: Mycelix Epistemic Charter v2.0.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let epistemic_quality = if self.primitive_processor.is_some() {
            if self.stats.total_cycles % 47 == 0 {
                use crate::consciousness::epistemic_tiers::*;
                // Classify based on cycle count (more cycles = higher empirical tier)
                let empirical = if self.stats.total_cycles > 1000 {
                    EmpiricalTier::E3CryptographicallyProven
                } else if self.stats.total_cycles > 100 {
                    EmpiricalTier::E2PrivatelyVerifiable
                } else if self.stats.total_cycles > 10 {
                    EmpiricalTier::E1Testimonial
                } else {
                    EmpiricalTier::E0Null
                };
                let coord = EpistemicCoordinate::new(
                    empirical,
                    NormativeTier::N0Personal,
                    MaterialityTier::M1Temporal,
                );
                let q = coord.quality_score();
                self.carryover.quality.last_epistemic_quality = q;
                q
            } else {
                self.carryover.quality.last_epistemic_quality
            }
        } else {
            0.0
        };
        module_timings.epistemic_tiers = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PHI VALIDATION: Empirical validation of Phi against synthetic states
        // EXPENSIVE — runs a validation study very rarely (every 500 cycles).
        // Results cached as correlation metric for telemetry.
        // Science: IIT empirical validation (Casali et al. 2013).
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let phi_validation_correlation = if let Some(ref mut validator) = self.phi_validation {
            if self.stats.total_cycles == 500 {
                // Run once at cycle 500 (enough history, one-shot validation)
                let results = validator.run_validation_study(10); // small sample for speed
                results.pearson_r
            } else {
                0.0
            }
        } else {
            0.0
        };
        module_timings.phi_validation = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // DISSIPATIVE CONSCIOUSNESS: Prigogine thermodynamic self-organization
        // [extracted to compute_dissipative_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let (dissipative_health, dissipative_regime, dissipative_entropy_rate) = self
            .compute_dissipative_phase(prediction_error, coherence, unified_psi, module_timings);

        // ═══════════════════════════════════════════════════════════════════════
        // EPISTEMIC CONFLICT: Multi-theory conflict detection + Φ_eff reliability weighting
        // Compares IIT, GWT, AST, PP, RPT, 4E scores; computes Φ_eff = Φ × R^γ.
        // Science: IIT (Tononi 2015), GWT (Baars 1988), AST (Graziano 2013).
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (epistemic_phi_eff, epistemic_conflict_count) =
            if let (Some(ref mut detector), Some(ref calibrator)) = (
                &mut self.epistemic_conflict_detector,
                &self.theory_calibrator,
            ) {
                if self.stats.total_cycles % 47 == 0 && self.stats.total_cycles > 0 {
                    use crate::consciousness::epistemic_conflict::{
                        compute_phi_eff, ConflictMatrix, MultiTheoryMetrics,
                    };
                    let metrics = MultiTheoryMetrics {
                        phi: unified_psi,
                        gwt: coherence as f64 * 0.8,
                        ast: coherence as f64,
                        pp: 1.0 - prediction_error as f64,
                        rpt: coherence as f64 * 0.9,
                        embodiment: self.carryover.consciousness.body_phi_modulation,
                        unified: unified_psi,
                    };
                    let matrix: ConflictMatrix = detector.detect(&metrics);
                    let phi_eff_result = compute_phi_eff(&metrics, calibrator);
                    self.carryover.quality.last_phi_eff = phi_eff_result.phi_eff;
                    (phi_eff_result.phi_eff, matrix.conflicts.len())
                } else {
                    (self.carryover.quality.last_phi_eff, 0)
                }
            } else {
                (0.0, 0)
            };
        module_timings.epistemic_conflict = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS EQUATION V2: Unified 7-theory formula
        // [extracted to compute_equation_v2_phase]
        // ═══════════════════════════════════════════════════════════════════════
        let equation_v2_consciousness = self.compute_equation_v2_phase(
            unified_psi,
            coherence,
            prediction_error,
            phi_attention_weight,
            module_timings,
        );

        ConsciousnessMetrics {
            primitive_psi,
            active_primitive_names,
            temporal_causal_chains,
            temporal_continuity,
            temporal_max_chain_length,
            chain_cycle_numbers,
            causal_codebook_entries,
            continuity_replay_needed,
            lattice_height,
            lattice_width,
            lattice_join_concept,
            compositionality_total,
            value_evaluator_score,
            value_evaluator_decision,
            consciousness_profile_composite,
            synergy_enhanced_composite,
            emergent_properties_count,
            reasoning_context,
            context_phi_weight,
            value_embeddings_created,
            value_cache_hit_rate,
            harmonies_alignment,
            harmonies_approved,
            composition_rule_applied,
            harmonic_field_coherence,
            harmonic_love_resonance,
            harmonic_interferences,
            reasoning_chain_confidence,
            reasoning_chain_depth,
            causal_relations_count,
            causal_avg_confidence,
            adaptive_reasoning_phi,
            epistemic_quality,
            phi_validation_correlation,
            dissipative_health,
            dissipative_regime,
            dissipative_entropy_rate,
            epistemic_phi_eff,
            epistemic_conflict_count,
            equation_v2_consciousness,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Extracted subsystem methods — compute + apply feedback + return values
    // ═══════════════════════════════════════════════════════════════════════════

    /// Temporal Primitives: Allen's Interval Algebra on conscious states.
    ///
    /// Records conscious intervals each cycle; amortized causal chain detection
    /// (every 47 cycles) and continuity analysis (every 97 cycles). Applies feedback:
    /// temporal continuity and causal chains boost prediction confidence; causal chains
    /// boost episodic consolidation; continuity gaps trigger demand replay.
    ///
    /// Science: Allen (1983), Varela (1999).
    #[allow(clippy::type_complexity)]
    fn compute_temporal_primitives_phase(
        &mut self,
        hv16_cached: symthaea_core::hdc::BinaryHV,
        unified_psi: f64,
        coherence: f32,
        module_timings: &mut super::ModuleTimings,
    ) -> (usize, f64, usize, Vec<u64>, Vec<(String, Vec<f32>)>, bool) {
        let _t = Instant::now();
        let (
            temporal_causal_chains,
            temporal_continuity,
            temporal_max_chain_length,
            chain_cycle_numbers,
            causal_codebook_entries,
            continuity_replay_needed,
        ) = if let Some(ref mut analyzer) = self.temporal_analyzer {
            // Record this cycle as a conscious interval
            let timestamp = self.stats.total_cycles as f64 * 0.02; // 50Hz → seconds
            use crate::consciousness::temporal_primitives::{
                ConsciousInterval, PhiTrend, TemporalInterval,
            };
            let mut ti = TemporalInterval::new(
                format!("c{}", self.stats.total_cycles),
                timestamp,
                timestamp + 0.02,
            )
            .unwrap_or_else(|_| {
                // Fallback: 0.0..0.02 is always valid (end > start)
                TemporalInterval::new("c_fallback", 0.0, 0.02)
                    .expect("hardcoded valid interval 0.0..0.02")
            });
            ti.phi = Some(unified_psi);
            let mut interval = ConsciousInterval::new(
                ti,
                unified_psi,
                coherence as f64,
                if self.stats.total_cycles > 0 {
                    0.5
                } else {
                    0.0
                },
            );
            interval.phi_trend = if unified_psi > self.carryover.history.consciousness_level + 0.01
            {
                PhiTrend::Rising
            } else if unified_psi < self.carryover.history.consciousness_level - 0.01 {
                PhiTrend::Falling
            } else {
                PhiTrend::Stable
            };
            interval.content = Some(hv16_cached);
            analyzer.add_interval(interval);

            // Amortized analysis: causal chains every 47 cycles (co-prime)
            let (chains, ccn, cce) = if self.stats.total_cycles % 47 == 0
                && self.stats.total_cycles > 0
            {
                let detected = analyzer.detect_causal_chains(3);
                let count = detected.len();
                let max_len = detected
                    .iter()
                    .map(|c| c.intervals.len())
                    .max()
                    .unwrap_or(0);
                self.carryover.quality.causal_chain_count = count;

                // Track A: Extract cycle numbers from genuine chains for episodic consolidation
                let cycle_nums: Vec<u64> = analyzer.genuine_chain_cycle_numbers();

                // Track A: Build causal codebook entries by binding content BinaryHVs
                let codebook_entries: Vec<(String, Vec<f32>)> = detected
                    .iter()
                    .filter(|c| c.genuine_causation && c.causal_strength > 0.5)
                    .filter_map(|c| {
                        let contents: Vec<&crate::hdc::binary_hv::BinaryHV> = c
                            .intervals
                            .iter()
                            .filter_map(|id| analyzer.interval_content(id))
                            .collect();
                        if contents.len() >= 2 {
                            let mut bound = *contents[0];
                            for hv in &contents[1..] {
                                bound = bound.bind(hv);
                            }
                            // Compress to resonator dim via same pipeline as compressed_state
                            let continuous = bound.to_continuous();
                            let compressed = self
                                .encoder
                                .compress_for_ltc(&continuous, self.config.cfc_config.input_dim);
                            Some((
                                format!("causal_{}_{}", self.stats.total_cycles, c.intervals.len()),
                                compressed,
                            ))
                        } else {
                            None
                        }
                    })
                    .collect();

                ((count, max_len), cycle_nums, codebook_entries)
            } else {
                (
                    (self.carryover.quality.causal_chain_count, 0),
                    Vec::new(),
                    Vec::new(),
                )
            };

            // Amortized analysis: continuity every 97 cycles (co-prime)
            let (continuity, cont_replay) =
                if self.stats.total_cycles % 97 == 0 && self.stats.total_cycles > 0 {
                    let analysis = analyzer.analyze_continuity();
                    self.carryover.quality.temporal_continuity = analysis.continuity_score;
                    // Track A: Detect continuity gaps that warrant demand replay
                    let replay_needed = analysis.continuity_score < 0.3 || analysis.gap_count > 5;
                    (analysis.continuity_score, replay_needed)
                } else {
                    (self.carryover.quality.temporal_continuity, false)
                };

            (chains.0, continuity, chains.1, ccn, cce, cont_replay)
        } else {
            (0, 0.0, 0, Vec::new(), Vec::new(), false)
        };
        module_timings.temporal_analyzer = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Temporal continuity → prediction confidence (stable time-axis = reliable predictions)
        if temporal_continuity > 0.7 {
            let boost = ((temporal_continuity - 0.7) * 0.05) as f32; // up to +1.5%
            self.prediction_confidence = (self.prediction_confidence + boost).min(1.0);
        }

        // FEEDBACK: Causal chain detection → confidence boost (the system found real structure)
        if temporal_causal_chains > 2 {
            let chain_boost = (temporal_causal_chains.min(10) as f32 - 2.0) * 0.005; // +0.5% per chain, up to +4%
            self.prediction_confidence = (self.prediction_confidence + chain_boost).min(1.0);
        }

        // Track A-1: Causal chain → episodic memory consolidation boost
        if !chain_cycle_numbers.is_empty() {
            if let Some(ref mut replay) = self.phi_episodic_replay {
                replay.boost_causal_consolidation(&chain_cycle_numbers, 0.15);
            }
        }

        // Track A-3: Continuity gaps → demand replay
        if continuity_replay_needed {
            if let Some(ref mut replay) = self.phi_episodic_replay {
                replay.trigger_demand_replay();
            }
        }

        (
            temporal_causal_chains,
            temporal_continuity,
            temporal_max_chain_length,
            chain_cycle_numbers,
            causal_codebook_entries,
            continuity_replay_needed,
        )
    }

    /// Dissipative Consciousness: Prigogine thermodynamic self-organization.
    ///
    /// Tracks entropy production, order parameter, and edge-of-chaos criticality.
    /// Applies feedback: thermodynamic regime drives cognitive adjustments via
    /// `recommend_action()` — exploration, coherence, differentiation, integration.
    ///
    /// Science: Prigogine (1977), Kauffman (1993), England (2013).
    fn compute_dissipative_phase(
        &mut self,
        prediction_error: f32,
        coherence: f32,
        unified_psi: f64,
        module_timings: &mut super::ModuleTimings,
    ) -> (f64, String, f64) {
        let _t = Instant::now();
        let (dissipative_health, dissipative_regime, dissipative_entropy_rate) =
            if let Some(ref mut dc) = self.dissipative_consciousness {
                let energy = prediction_error as f64;
                let info = coherence as f64 * unified_psi;
                dc.update(unified_psi, energy, info, coherence as f64);
                let health = dc.health_score();
                self.carryover.quality.last_dissipative_health = health;
                (
                    health,
                    format!("{:?}", dc.current_regime()),
                    dc.entropy_production_rate,
                )
            } else {
                (0.0, String::new(), 0.0)
            };
        module_timings.dissipative_consciousness = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Dissipative regime → exploration + learning rate modulation
        // Science: Prigogine (1977) — dissipative structures self-organize at edge of chaos.
        // Use recommend_action() to translate thermodynamic state into cognitive adjustments.
        if let Some(ref dc) = self.dissipative_consciousness {
            use crate::consciousness::dissipative_consciousness::DissipativeAction;
            match dc.recommend_action() {
                DissipativeAction::Maintain { .. } => {
                    // Optimal regime: slight confidence boost (system is well-organized)
                    self.prediction_confidence =
                        (self.prediction_confidence + 0.005).clamp(0.0, 1.0);
                }
                DissipativeAction::IncreaseActivity {
                    suggested_increase, ..
                } => {
                    // Near equilibrium: boost exploration proportional to suggested increase
                    let explore_boost = (suggested_increase * 0.15).min(0.05) as f32;
                    self.curiosity_drive.exploration_urge =
                        (self.curiosity_drive.exploration_urge + explore_boost).clamp(0.0, 1.0);
                    self.prediction_confidence = (self.prediction_confidence - 0.01).max(0.0);
                }
                DissipativeAction::IncreaseCoherence { .. } => {
                    // Chaotic: suppress exploration, boost learning to restore coherence
                    self.curiosity_drive.exploration_urge =
                        (self.curiosity_drive.exploration_urge * 0.9).max(0.0);
                    self.fep_lr_boost = (self.fep_lr_boost * 1.05).clamp(1.0, 2.0);
                }
                DissipativeAction::IncreaseDifferentiation { .. } => {
                    // Too ordered: nudge exploration up, learning down slightly
                    self.curiosity_drive.exploration_urge =
                        (self.curiosity_drive.exploration_urge + 0.02).clamp(0.0, 1.0);
                    self.prediction_confidence = (self.prediction_confidence - 0.005).max(0.0);
                }
                DissipativeAction::IncreaseIntegration { .. } => {
                    // Too differentiated: boost learning rate for better integration
                    self.fep_lr_boost = (self.fep_lr_boost * 1.03).clamp(1.0, 2.0);
                }
            }
        }

        (
            dissipative_health,
            dissipative_regime,
            dissipative_entropy_rate,
        )
    }

    /// Consciousness Equation V2: Unified 7-theory formula C(t) = σ × Σ × S × ρ.
    ///
    /// Combines Integration, Binding, Workspace, Attention, Recursion, Efficacy,
    /// Knowledge into a single consciousness score with PAC modulation.
    /// Applies feedback: high consciousness boosts confidence + episodic consolidation;
    /// low consciousness boosts exploration.
    ///
    /// Science: Tononi (2004), Baars (1988), Friston (2010), Graziano (2013).
    fn compute_equation_v2_phase(
        &mut self,
        unified_psi: f64,
        coherence: f32,
        prediction_error: f32,
        phi_attention_weight: f32,
        module_timings: &mut super::ModuleTimings,
    ) -> f64 {
        let _t = Instant::now();
        let equation_v2_consciousness = if let Some(ref mut eq) = self.consciousness_equation_v2 {
            if self.stats.total_cycles % 23 == 0 && self.stats.total_cycles > 0 {
                use crate::consciousness::consciousness_equation_v2::{
                    ConsciousnessStateV2, CoreComponent,
                };
                use std::collections::HashMap;
                let mut core_values = HashMap::new();
                core_values.insert(CoreComponent::Integration, unified_psi.clamp(0.0, 1.0));
                core_values.insert(CoreComponent::Binding, coherence as f64);
                core_values.insert(CoreComponent::Workspace, coherence as f64 * 0.8); // GWT proxy
                core_values.insert(CoreComponent::Attention, phi_attention_weight as f64);
                core_values.insert(CoreComponent::Recursion, 0.5); // Placeholder: HOT depth requires higher-order thought tracking (deferred — see W2-A in consolidation plan)
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
                let result = eq.compute(&state);
                self.carryover.consciousness.last_equation_v2_consciousness = result.consciousness;
                result.consciousness
            } else {
                self.carryover.consciousness.last_equation_v2_consciousness
            }
        } else {
            0.0
        };
        module_timings.consciousness_equation_v2 = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Unified consciousness score modulates confidence + exploration + consolidation
        // Science: High C(t) = strong integration across theories → confident, less exploration needed
        // Additionally: high consciousness moments are episodically significant (Baars 2005 —
        // GWT predicts conscious moments are preferentially consolidated into long-term memory)
        if equation_v2_consciousness > 0.6 {
            let boost = (equation_v2_consciousness - 0.6) * 0.08; // up to +3.2%
            self.prediction_confidence =
                (self.prediction_confidence + boost as f32).clamp(0.0, 1.0);
            // High-consciousness moments → boost episodic consolidation priority
            // Science: Conscious access correlates with memory formation (Dehaene 2014, ch.4)
            if let Some(ref mut replay) = self.phi_episodic_replay {
                let consolidation_boost = (equation_v2_consciousness - 0.6) * 0.1;
                replay.boost_recent_consolidation(consolidation_boost);
            }
        } else if equation_v2_consciousness > 0.0 && equation_v2_consciousness < 0.3 {
            // Low consciousness → boost exploration to find better integration
            self.curiosity_drive.exploration_urge =
                (self.curiosity_drive.exploration_urge + 0.02).clamp(0.0, 1.0);
        }

        equation_v2_consciousness
    }

    /// Primitive Lattice: Structural metrics from the consciousness tier system.
    ///
    /// Reads height/width from the precomputed lattice (cached after first cycle).
    /// Computes lattice join for active primitives every 7 cycles.
    /// Applies feedback: deep lattice (height > 5) reduces LR on first cycle.
    ///
    /// Science: Davey & Priestley (2002) — lattice theory for knowledge systems.
    fn compute_lattice_phase(
        &mut self,
        active_primitive_names: &[String],
        module_timings: &mut super::ModuleTimings,
    ) -> (usize, usize, Option<String>) {
        let _t = Instant::now();
        let (lattice_height, lattice_width, lattice_join_concept) =
            if let Some(ref lattice) = self.primitive_lattice {
                // Properties (height/width/modularity) are O(n²–n³) on the lattice graph.
                // The lattice is immutable after construction → compute once on first cycle,
                // cache in stats, and reuse. This eliminates ~31ms/cycle overhead.
                let (height, width) = if self.stats.lattice_height_cached == 0 {
                    let props = lattice.properties();
                    self.stats.lattice_height_cached = props.height;
                    self.stats.lattice_width_cached = props.width;
                    (props.height, props.width)
                } else {
                    (
                        self.stats.lattice_height_cached,
                        self.stats.lattice_width_cached,
                    )
                };

                // FEEDBACK: Lattice height (integration depth) → LR modulation (once)
                if height > 5 && self.stats.total_cycles == 0 {
                    let depth_factor = 1.0 - (height.min(9) as f32 - 5.0) * 0.01;
                    self.carryover.learning.subsystem_lr_factor *= depth_factor;
                }

                // Track B: Lattice join for concept composition (every 7 cycles — join is O(1) via precomputed table)
                let join_concept =
                    if active_primitive_names.len() >= 2 && self.stats.total_cycles % 7 == 0 {
                        let mut best_join: Option<usize> = None;
                        for i in 0..active_primitive_names.len() {
                            for j in (i + 1)..active_primitive_names.len() {
                                if let (Some(a), Some(b)) = (
                                    lattice.element_index_by_name(&active_primitive_names[i]),
                                    lattice.element_index_by_name(&active_primitive_names[j]),
                                ) {
                                    if let Some(idx) = lattice.join(a, b) {
                                        match best_join {
                                            Some(prev)
                                                if lattice.elements[idx].tier
                                                    < lattice.elements[prev].tier =>
                                            {
                                                best_join = Some(idx)
                                            }
                                            None => best_join = Some(idx),
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                        best_join.map(|idx| lattice.elements[idx].name.clone())
                    } else {
                        None
                    };

                (height, width, join_concept)
            } else {
                (0, 0, None)
            };
        module_timings.primitive_lattice = _t.elapsed().as_micros() as u64;

        (lattice_height, lattice_width, lattice_join_concept)
    }

    /// Unified Value Evaluator: Seven Harmonies alignment scoring.
    ///
    /// Evaluates cognitive action against fiduciary harmonics every 19 cycles.
    /// Applies feedback: Veto decision drastically reduces learning rate.
    ///
    /// Science: Panksepp (1998) affective neuroscience + value alignment.
    fn compute_value_evaluator_phase(
        &mut self,
        unified_psi: f64,
        module_timings: &mut super::ModuleTimings,
    ) -> (f64, String) {
        let _t = Instant::now();
        let (value_evaluator_score, value_evaluator_decision) =
            if let Some(ref mut evaluator) = self.value_evaluator {
                if self.stats.total_cycles % 19 == 0 {
                    let ctx = crate::consciousness::unified_value_evaluator::EvaluationContext {
                        consciousness_level: unified_psi,
                        ..Default::default()
                    };
                    let result = evaluator.evaluate("cognitive_cycle", ctx);
                    let decision_str = match &result.decision {
                        crate::consciousness::unified_value_evaluator::Decision::Allow => "Allow",
                        crate::consciousness::unified_value_evaluator::Decision::Warn(_) => "Warn",
                        crate::consciousness::unified_value_evaluator::Decision::Veto(_) => "Veto",
                    };
                    self.carryover.quality.last_value_score = result.overall_score;
                    (result.overall_score, decision_str.to_string())
                } else {
                    (self.carryover.quality.last_value_score, String::new())
                }
            } else {
                (0.0, String::new())
            };
        module_timings.value_evaluator = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Value evaluator Veto → suppress learning for this cycle
        if value_evaluator_decision == "Veto" {
            self.carryover.learning.subsystem_lr_factor *= 0.1; // Drastically reduce LR on value violation
        }

        (value_evaluator_score, value_evaluator_decision)
    }

    /// Fiduciary Harmonics: Seven Harmonies field coherence + interference detection.
    ///
    /// Drives harmonic levels from consciousness metrics every 11 cycles, detects
    /// and resolves value tensions. Applies feedback: high coherence boosts LR;
    /// interferences reduce prediction confidence.
    ///
    /// Science: Whitehead (1929), Deci & Ryan (2000) — value coherence theory.
    fn compute_fiduciary_harmonics_phase(
        &mut self,
        coherence: f32,
        prediction_error: f32,
        unified_psi: f64,
        module_timings: &mut super::ModuleTimings,
    ) -> (f64, f64, usize) {
        let _t = Instant::now();
        let (harmonic_field_coherence, harmonic_love_resonance, harmonic_interferences) =
            if let Some(ref mut field) = self.harmonic_field {
                if self.stats.total_cycles % 11 == 0 {
                    // Drive harmonic levels from consciousness metrics
                    use crate::consciousness::harmonics::FiduciaryHarmonic;
                    field.set_level(FiduciaryHarmonic::ResonantCoherence, coherence as f64);
                    field.set_level(
                        FiduciaryHarmonic::EvolutionaryProgression,
                        (prediction_error as f64 * 2.0).clamp(0.0, 1.0),
                    ); // high error = high evolution pressure
                    field.set_level(
                        FiduciaryHarmonic::IntegralWisdom,
                        self.prediction_confidence as f64,
                    );
                    field.set_level(
                        FiduciaryHarmonic::PanSentientFlourishing,
                        unified_psi.clamp(0.0, 1.0),
                    );
                    field.detect_interferences();
                    // Resolve interferences if any were detected
                    if !field.interferences.is_empty() {
                        if let Some(ref resolver) = self.harmonic_resolver {
                            let _resolution = resolver.resolve(field);
                        }
                    }
                    self.carryover.consciousness.last_harmonic_coherence = field.field_coherence;
                    (
                        field.field_coherence,
                        field.infinite_love_resonance,
                        field.interferences.len(),
                    )
                } else {
                    (self.carryover.consciousness.last_harmonic_coherence, 0.0, 0)
                }
            } else {
                (0.0, 0.0, 0)
            };
        module_timings.harmonics = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Harmonic coherence → LR stability (coherent values = stable learning)
        if harmonic_field_coherence > 0.6 {
            let harmony_boost = 1.0 + ((harmonic_field_coherence - 0.6) * 0.05) as f32; // up to +2%
            self.carryover.learning.subsystem_lr_factor *= harmony_boost;
        }
        // FEEDBACK: Harmonic interferences → reduce confidence (value tensions = uncertainty)
        if harmonic_interferences > 0 {
            let interference_penalty = (harmonic_interferences.min(3) as f32) * 0.01; // -1% per interference
            self.prediction_confidence =
                (self.prediction_confidence - interference_penalty).max(0.0);
        }

        (
            harmonic_field_coherence,
            harmonic_love_resonance,
            harmonic_interferences,
        )
    }

    /// Causal Self-Explanation: Pearl causal model of primitive→Phi effects.
    ///
    /// Learns which primitives cause which Phi changes every 23 cycles by constructing
    /// reasoning chains from active primitives and feeding them to the causal explainer.
    ///
    /// Science: Pearl (2009) — causal inference, Woodward (2003) — interventionism.
    fn compute_causal_self_explanation_phase(
        &mut self,
        hv16_cached: symthaea_core::hdc::BinaryHV,
        active_primitive_names: &[String],
        primitive_psi: f64,
        module_timings: &mut super::ModuleTimings,
    ) -> (usize, f64) {
        let _t = Instant::now();
        let (causal_relations_count, causal_avg_confidence) = if let Some(ref mut explainer) =
            self.causal_explainer
        {
            if self.stats.total_cycles % 23 == 0
                && self.stats.total_cycles > 0
                && !active_primitive_names.is_empty()
            {
                // Construct PrimitiveExecution entries from active primitives
                if let Some(ref mut processor) = self.primitive_processor {
                    let timestamp = self.stats.total_cycles as f64 * 0.02;
                    let state = processor.process_input(&hv16_cached, timestamp);
                    let chain = {
                        let mut c = crate::consciousness::primitive_reasoning::ReasoningChain::new(
                            hv16_cached,
                        );
                        for ap in state.all_active().iter().take(4) {
                            let exec = crate::consciousness::primitive_reasoning::PrimitiveExecution {
                                    primitive: ap.primitive.clone(),
                                    input: hv16_cached,
                                    output: hv16_cached.bind(&crate::hdc::BinaryHV::random(
                                        self.stats.total_cycles as u64,
                                    )),
                                    transformation: crate::consciousness::primitive_reasoning::TransformationType::Bind,
                                    phi_contribution: primitive_psi * ap.activation,
                                    timestamp,
                                };
                            c.executions.push(exec);
                        }
                        c
                    };
                    explainer.learn_from_chain(&chain, "cognitive_cycle");
                }
                // Only call summarize_understanding on learning cycles (expensive)
                let summary = explainer.summarize_understanding();
                self.carryover.history.last_causal_relations = summary.total_causal_relations;
                self.carryover.history.last_causal_confidence = summary.average_confidence;
                (summary.total_causal_relations, summary.average_confidence)
            } else {
                // Non-learning cycle: return cached summary (avoids ~880µs/cycle)
                (
                    self.carryover.history.last_causal_relations,
                    self.carryover.history.last_causal_confidence,
                )
            }
        } else {
            (0, 0.0)
        };
        module_timings.causal_explanation = _t.elapsed().as_micros() as u64;

        (causal_relations_count, causal_avg_confidence)
    }
}
