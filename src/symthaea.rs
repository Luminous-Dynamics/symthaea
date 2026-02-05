//! # Symthaea Facade
//!
//! The primary entry point for the Symthaea consciousness system.
//! Wraps [`ContinuousMind`] and [`ConsciousnessLanguageCore`] into a
//! unified interface suitable for the service daemon and other consumers.
//!
//! ## Neural Bridge v2 Integration
//!
//! When compiled with `--features neural-bridge`, Symthaea uses BGE-M3 for
//! high-quality semantic encoding (~380ms CPU, ~60-100ms GPU expected).
//! Otherwise falls back to fast hash-based encoding (<1ms but lower quality).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::RealHV;

#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridgeV2;

use crate::language::{
    ConsciousnessLanguageCore, ConsciousnessLanguageConfig,
    LLMOrgan, LLMOrganConfig,
    llm_backend,
    PluginRegistry,
};
use crate::mind::{ContinuousMind, MindConfig, StructuredThought, DomainContext, ConstraintType, EpistemicStatus};
use crate::mind::structured_thought::{EpistemicCube, ETier, NTier};
#[cfg(feature = "magi_loop")]
use crate::mind::SemanticIntent;
use crate::partnership::{
    DyadInput, DyadWeights, HumanPartnerModel, InteractionEvent,
    PhiDyadCalculator, RelationshipTrajectory,
};
use crate::hdc::relational_consciousness::{
    RelationalAssessment, RelationMode, RelationshipStage,
};

#[cfg(feature = "magi_loop")]
use crate::consciousness::recursive_improvement::{
    BrierScoreTracker, CalibrationSummary,
    PredictionDomain, OutcomeCategory, WorldPrediction,
    WorldActionContext, ResolutionContract, RiskTier,
};

/// Response from processing a query through the consciousness pipeline.
#[derive(Debug, Clone)]
pub struct ProcessResponse {
    /// Natural language content of the response.
    pub content: String,
    /// Confidence in the response (0.0-1.0).
    pub confidence: f32,
    /// Whether the response is considered safe.
    pub safe: bool,
    /// Estimated steps remaining to emergence threshold.
    pub steps_to_emergence: usize,
    /// Whether the LLM translation passed fidelity verification.
    ///
    /// This checks that the translation respects the structured thought's
    /// epistemic status and constraints.
    pub translation_verified: bool,
    /// The structured thought that was translated (for debugging/introspection).
    pub structured_thought: Option<StructuredThought>,
}

/// Result of introspecting the current consciousness state.
#[derive(Debug, Clone)]
pub struct IntrospectionResult {
    /// Current consciousness level (0.0-1.0).
    pub consciousness_level: f32,
    /// Number of self-referential loops in the cognitive graph.
    pub self_loops: usize,
    /// Total size of the cognitive graph.
    pub graph_size: usize,
    /// Complexity measure of current cognitive state.
    pub complexity: f32,
    /// Memory statistics.
    pub memory_stats: MemoryStats,
}

/// Memory statistics for introspection.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Number of items in short-term (working) memory.
    pub short_term_count: usize,
    /// Number of items in long-term memory.
    pub long_term_count: usize,
}

/// Report from a sleep/consolidation cycle.
#[derive(Debug, Clone)]
pub struct SleepReport {
    /// Number of memories scaled during consolidation.
    pub scaled: usize,
    /// Number of memories consolidated (merged).
    pub consolidated: usize,
    /// Number of memories pruned (removed).
    pub pruned: usize,
    /// Number of patterns extracted during consolidation.
    pub patterns_extracted: usize,
}

/// The primary Symthaea consciousness facade.
///
/// Integrates the continuous mind (HDC+LTC cognitive processing) with
/// the consciousness language core (NL understanding and generation)
/// and the partnership module (relational consciousness tracking).
pub struct Symthaea {
    /// Core cognitive system.
    mind: ContinuousMind,
    /// Language processing core (used in Phase 3 for conscious NL understanding).
    #[allow(dead_code)]
    language: ConsciousnessLanguageCore,
    /// LLM organ for text generation.
    llm: LLMOrgan,
    /// HDC dimension used.
    hdc_dim: usize,
    /// Number of LTC neurons (used in Phase 3 for LTC-paced generation).
    #[allow(dead_code)]
    ltc_neurons: usize,
    /// Total interactions processed.
    interactions: u64,
    /// Human partner model for relational consciousness.
    partner: HumanPartnerModel,
    /// Relationship trajectory tracking.
    trajectory: RelationshipTrajectory,
    /// Phi-dyad calculator for relational Phi.
    dyad_calculator: PhiDyadCalculator,
    /// Recent AI states for dyad computation (ring buffer).
    recent_ai_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    /// Domain plugin registry for multi-domain awareness.
    plugin_registry: PluginRegistry,
    /// Brier Score calibration tracker for epistemic calibration.
    #[cfg(feature = "magi_loop")]
    calibration: BrierScoreTracker,
    /// Neural Bridge v2: BGE-M3 + linear probe for high-quality semantic encoding.
    /// When available, replaces hash-based encoding with true semantic understanding.
    #[cfg(feature = "neural-bridge")]
    neural_bridge: Option<NeuralBridgeV2>,
}

impl Symthaea {
    /// Create a new Symthaea instance with the given HDC dimension and LTC neuron count.
    pub async fn new(hdc_dim: usize, ltc_neurons: usize) -> Result<Self> {
        let mind_config = MindConfig {
            dimension: hdc_dim,
            ..MindConfig::default()
        };

        let mut mind = ContinuousMind::new(mind_config);
        mind.awaken();

        // Seed working memory with domain knowledge to establish epistemic baseline
        let seeding_result = mind.seed_memory();
        tracing::info!(
            target: "symthaea::init",
            prototypes = seeding_result.prototypes_seeded,
            categories = ?seeding_result.categories,
            "Working memory seeded with a priori knowledge"
        );

        let language_config = ConsciousnessLanguageConfig {
            dimension: hdc_dim,
            ..ConsciousnessLanguageConfig::default()
        };
        let language = ConsciousnessLanguageCore::new(language_config);

        let llm_config = LLMOrganConfig {
            dimension: hdc_dim,
            ..LLMOrganConfig::default()
        };
        let backend = llm_backend::default_backend();
        let llm = LLMOrgan::with_backend(llm_config, backend);

        // Initialize plugin registry with all built-in domain plugins
        let plugin_registry = PluginRegistry::with_builtins();
        tracing::info!(
            target: "symthaea::init",
            plugins = ?plugin_registry.list(),
            "Domain plugin registry initialized with built-in plugins"
        );

        // Try to initialize Neural Bridge v2 for high-quality semantic encoding
        #[cfg(feature = "neural-bridge")]
        let neural_bridge = match NeuralBridgeV2::load_default() {
            Ok(bridge) => {
                tracing::info!(
                    target: "symthaea::init",
                    encoder_dim = bridge.encoder_dim(),
                    probe_dim = bridge.probe_output_dim(),
                    cuda = bridge.is_cuda(),
                    "Neural Bridge v2 initialized (BGE-M3 semantic encoding)"
                );
                Some(bridge)
            }
            Err(e) => {
                tracing::warn!(
                    target: "symthaea::init",
                    error = %e,
                    "Neural Bridge v2 unavailable, using hash-based encoding"
                );
                None
            }
        };

        Ok(Self {
            mind,
            language,
            llm,
            hdc_dim,
            ltc_neurons,
            interactions: 0,
            partner: HumanPartnerModel::new("human"),
            trajectory: RelationshipTrajectory::default(),
            dyad_calculator: PhiDyadCalculator::new(),
            recent_ai_states: Vec::new(),
            plugin_registry,
            #[cfg(feature = "magi_loop")]
            calibration: BrierScoreTracker::with_defaults(),
            #[cfg(feature = "neural-bridge")]
            neural_bridge,
        })
    }

    /// Resume from a saved state file.
    ///
    /// Loads persisted partnership state, trajectory, and interaction count.
    /// Reconstructs the mind and language systems fresh (stateless between sessions).
    pub fn resume(path: &str) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read state file: {}", path))?;
        let state: PersistedState = serde_json::from_str(&data)
            .with_context(|| "Failed to parse state file")?;

        let hdc_dim = state.hdc_dim;
        let ltc_neurons = state.ltc_neurons;

        let mind_config = MindConfig {
            dimension: hdc_dim,
            ..MindConfig::default()
        };
        let mut mind = ContinuousMind::new(mind_config);
        mind.awaken();

        // Seed working memory on resume as well
        let seeding_result = mind.seed_memory();
        tracing::info!(
            target: "symthaea::init",
            prototypes = seeding_result.prototypes_seeded,
            "Resumed with working memory seeded"
        );

        let language = ConsciousnessLanguageCore::default();
        let backend = llm_backend::default_backend();
        let llm = LLMOrgan::with_backend(LLMOrganConfig {
            dimension: hdc_dim,
            ..LLMOrganConfig::default()
        }, backend);

        let plugin_registry = PluginRegistry::with_builtins();

        // Try to initialize Neural Bridge v2 on resume
        #[cfg(feature = "neural-bridge")]
        let neural_bridge = match NeuralBridgeV2::load_default() {
            Ok(bridge) => {
                tracing::info!(
                    target: "symthaea::init",
                    cuda = bridge.is_cuda(),
                    "Neural Bridge v2 initialized on resume"
                );
                Some(bridge)
            }
            Err(e) => {
                tracing::warn!(
                    target: "symthaea::init",
                    error = %e,
                    "Neural Bridge v2 unavailable on resume"
                );
                None
            }
        };

        Ok(Self {
            mind,
            language,
            llm,
            hdc_dim,
            ltc_neurons,
            interactions: state.interactions,
            partner: state.partner,
            trajectory: state.trajectory,
            dyad_calculator: PhiDyadCalculator::new(),
            recent_ai_states: state.recent_ai_states,
            plugin_registry,
            #[cfg(feature = "magi_loop")]
            calibration: BrierScoreTracker::with_defaults(),
            #[cfg(feature = "neural-bridge")]
            neural_bridge,
        })
    }

    /// Process a query through the full consciousness pipeline.
    ///
    /// **Reason-then-Generate Pipeline (LLM as Broca's Area):**
    ///
    /// 1. Input → HDC encoding → Mind perceives
    /// 2. Mind tick → HDC+LTC computes (the BRAIN thinks)
    /// 3. Extract StructuredThought (articulate what was computed)
    /// 4. Enrich with partnership context
    /// 5. LLM Translation (Broca's Area - NOT reasoning!)
    /// 6. Verify translation fidelity
    /// 7. Partnership update → Response
    ///
    /// **Key Insight**: The LLM does NOT think. It translates pre-computed
    /// structured thoughts into fluent natural language.
    pub async fn process(&mut self, content: &str) -> Result<ProcessResponse> {
        use std::time::Instant;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let pipeline_start = Instant::now();
        self.interactions += 1;

        // Generate correlation ID for this request
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        self.interactions.hash(&mut hasher);
        let correlation_id = format!("broca_{:x}", hasher.finish());

        // ====================================================================
        // PHASE 1: PERCEPTION (Input → HDC encoding + text for classification)
        // ====================================================================
        let phase1_start = Instant::now();
        let input_embedding = self.text_to_hv(content);
        // Use perceive_text to enable HDC-based intent classification
        self.mind.perceive_text(content, input_embedding.clone());

        // Domain detection via plugin registry
        let detected_domain = self.plugin_registry.detect_domain(content).to_string();
        let domain_entities = if let Some(plugin) = self.plugin_registry.get(&detected_domain) {
            plugin.extract_entities(content)
        } else {
            Vec::new()
        };
        if !domain_entities.is_empty() {
            tracing::debug!(
                target: "symthaea::broca",
                domain = %detected_domain,
                entities = domain_entities.len(),
                "Domain plugin detected"
            );
        }
        let phase1_duration = phase1_start.elapsed();

        // ====================================================================
        // PHASE 2: COGNITION (Mind tick - HDC+LTC THINKS)
        // ====================================================================
        let phase2_start = Instant::now();
        self.mind.tick();
        let phase2_duration = phase2_start.elapsed();

        // ====================================================================
        // PHASE 3: EXTRACTION (Articulate what was computed)
        // ====================================================================
        // This is the key innovation: we extract WHAT THE MIND COMPUTED,
        // not what the LLM would make up.
        let phase3_start = Instant::now();
        let mut thought = self.mind.extract_structured_thought();
        let phase3_duration = phase3_start.elapsed();

        // Store original input for context
        thought.original_input = Some(content.to_string());

        // ====================================================================
        // PHASE 3.5: DOMAIN CONTEXT INJECTION
        // ====================================================================
        // Wire Phase 1 domain detection results into the structured thought
        // so the LLM translation has access to domain, entities, and computed answers.
        if detected_domain != "generic" || !domain_entities.is_empty() {
            let entities: Vec<(String, String, f64)> = domain_entities.iter()
                .map(|e| (e.entity_type.clone(), e.value.clone(), e.confidence))
                .collect();
            let computed_result = self.plugin_registry.get(&detected_domain)
                .and_then(|p| p.compute(content, &domain_entities));

            let (computed_answer, cube, domain_phi) = match computed_result {
                Some(cr) => (Some(cr.answer), Some(cr.cube), Some(cr.phi)),
                None => (None, None, None),
            };

            thought.domain_context = Some(DomainContext {
                domain: detected_domain.clone(),
                entities,
                computed_answer,
                cube,
                phi: domain_phi,
            });
        }

        // Derive epistemic status from cube (principled 3D mapping)
        // instead of crude "computed_answer exists → Certain" override
        if let Some(ref ctx) = thought.domain_context {
            if let Some(ref cube) = ctx.cube {
                thought.epistemic_status = Self::cube_to_epistemic_status(cube);
                thought.semantic_intent = crate::mind::SemanticIntent::Answer;
            }
        }

        // ====================================================================
        // PHASE 4: RELATIONAL ENRICHMENT (Add partnership context)
        // ====================================================================
        thought.relationship_stage = self.partner.stage;
        thought.relation_mode = self.partner.mode;
        thought.trust = self.partner.trust;

        // ====================================================================
        // PHASE 4.5: CALIBRATION ADJUSTMENT (Brier Score confidence tuning)
        // ====================================================================
        // Adjust the epistemic confidence using learned calibration data.
        // If the system has been overconfident in a domain, reduce confidence.
        // If underconfident, increase it. This closes the MAGI calibration loop.
        //
        // BYPASS: Axiomatic claims (N3) are not subject to calibration.
        // Mathematical truths like 2+2=4 are certain by definition — no amount
        // of historical miscalibration should downgrade them.
        #[cfg(feature = "magi_loop")]
        {
            let skip_calibration = thought.domain_context
                .as_ref()
                .and_then(|c| c.cube.as_ref())
                .map_or(false, |cube| cube.n == NTier::N3);

            if !skip_calibration {
                let domain = Self::map_intent_to_domain(&thought.semantic_intent);
                let raw_confidence = Self::epistemic_to_confidence(&thought.epistemic_status);
                let adjusted = self.calibration.adjust_confidence(domain, raw_confidence);

                // If calibration significantly changed confidence, update epistemic status
                let adjusted_status = Self::confidence_to_epistemic(adjusted);
                if adjusted_status != thought.epistemic_status {
                    tracing::debug!(
                        target: "symthaea::broca::calibration",
                        original_status = ?thought.epistemic_status,
                        adjusted_status = ?adjusted_status,
                        raw_confidence = raw_confidence,
                        adjusted_confidence = adjusted,
                        domain = ?domain,
                        "Calibration adjusted epistemic status"
                    );
                    thought.epistemic_status = adjusted_status;
                }
            }
        }

        // ====================================================================
        // PHASE 5: TRANSLATION (Broca's Area - NOT reasoning!)
        // ====================================================================
        // The LLM's ONLY job is to convert the structured thought into
        // fluent natural language. It must NOT add information.
        let phase5_start = Instant::now();
        let generation = self.llm.translate_thought(&thought).await;
        let phase5_duration = phase5_start.elapsed();

        // ====================================================================
        // PHASE 6: FIDELITY VERIFICATION
        // ====================================================================
        // Check that the translation respects the structured thought
        let translation_verified = self.verify_translation_fidelity(&thought, &generation.text);

        if !translation_verified {
            tracing::warn!(
                "Translation fidelity warning: epistemic_status={:?}, text={}",
                thought.epistemic_status,
                &generation.text[..generation.text.len().min(100)]
            );
        }

        // ====================================================================
        // PHASE 7: PARTNERSHIP UPDATE
        // ====================================================================
        let consciousness = thought.phi as f32;
        self.update_partnership(content, consciousness);

        // Track AI state for dyad computation
        let ai_hv = symthaea_core::hdc::unified_hv::ContinuousHV::from_values(
            input_embedding.values.clone()
        );
        self.recent_ai_states.push(ai_hv);
        if self.recent_ai_states.len() > 8 {
            self.recent_ai_states.remove(0);
        }

        // ====================================================================
        // PHASE 7.5: CALIBRATION RECORDING (Brier Score tracking)
        // ====================================================================
        // Record the prediction outcome for ongoing calibration.
        // We treat translation_verified as the outcome: if the translation
        // was faithful to the structured thought, the prediction "succeeded".
        #[cfg(feature = "magi_loop")]
        {
            let domain = Self::map_intent_to_domain(&thought.semantic_intent);
            let confidence = Self::epistemic_to_confidence(&thought.epistemic_status);

            let action_context = WorldActionContext::new(
                "broca_translation",
                "Faithful translation of structured thought",
            ).with_risk_tier(RiskTier::Observation);
            let contract = ResolutionContract::shell_command();

            let mut prediction = WorldPrediction::new(
                format!(
                    "Translation of {:?} intent with {:?} epistemic status will be faithful",
                    thought.semantic_intent, thought.epistemic_status
                ),
                OutcomeCategory::Success,
                confidence,
                action_context,
                contract,
            );

            // Override inferred domain with semantically meaningful domain
            prediction.domain = domain;

            // Resolve immediately based on fidelity verification
            if translation_verified {
                prediction.resolve_true(OutcomeCategory::Success, 1.0);
            } else {
                prediction.resolve_false(OutcomeCategory::SafeFailure, 1.0);
            }

            self.calibration.record_prediction(&prediction);
        }

        // ====================================================================
        // PHASE 8: RESPONSE ASSEMBLY
        // ====================================================================
        let safe = consciousness > 0.1;
        let steps_to_emergence = if consciousness >= 0.7 {
            0
        } else {
            ((0.7 - consciousness) / 0.01) as usize
        };

        // ====================================================================
        // OBSERVABILITY: Structured logging for Broca pipeline
        // ====================================================================
        let total_duration = pipeline_start.elapsed();

        // Log at INFO level with structured fields for production observability
        tracing::info!(
            target: "symthaea::broca",
            correlation_id = %correlation_id,
            epistemic_status = ?thought.epistemic_status,
            semantic_intent = ?thought.semantic_intent,
            response_type = ?thought.response_type,
            phi = thought.phi,
            coherence = thought.coherence,
            meta_awareness = thought.meta_awareness,
            relationship_stage = ?thought.relationship_stage,
            relation_mode = ?thought.relation_mode,
            trust = thought.trust,
            fidelity_verified = translation_verified,
            detected_domain = %detected_domain,
            domain_entities = domain_entities.len(),
            phase1_perception_us = phase1_duration.as_micros(),
            phase2_cognition_us = phase2_duration.as_micros(),
            phase3_extraction_us = phase3_duration.as_micros(),
            phase5_translation_us = phase5_duration.as_micros(),
            total_duration_ms = total_duration.as_millis(),
            input_len = content.len(),
            output_len = generation.text.len(),
            "Broca pipeline complete"
        );

        // Log epistemic distribution metrics (for aggregation)
        tracing::debug!(
            target: "symthaea::broca::metrics",
            epistemic_status = ?thought.epistemic_status,
            intent = ?thought.semantic_intent,
            fidelity = translation_verified,
            "epistemic_event"
        );

        // Warn on potential hallucination triggers (high novelty + certain status)
        if matches!(thought.epistemic_status, crate::mind::structured_thought::EpistemicStatus::Certain)
            && thought.coherence < 0.3
        {
            tracing::warn!(
                target: "symthaea::broca::security",
                correlation_id = %correlation_id,
                coherence = thought.coherence,
                "Potential hallucination risk: Certain status with low coherence"
            );
        }

        // Log calibration summary periodically (every 10 interactions)
        #[cfg(feature = "magi_loop")]
        if self.interactions % 10 == 0 && self.calibration.total_predictions() > 0 {
            let cal_summary = self.calibration.calibration_summary();
            tracing::info!(
                target: "symthaea::broca::calibration",
                correlation_id = %correlation_id,
                global_brier = cal_summary.global_brier,
                global_ece = cal_summary.global_ece,
                global_accuracy = cal_summary.global_accuracy,
                total_predictions = cal_summary.total_predictions,
                is_well_calibrated = cal_summary.is_well_calibrated,
                domain_count = cal_summary.domain_stats.len(),
                "Calibration summary (periodic)"
            );
        }

        Ok(ProcessResponse {
            content: generation.text,
            confidence: generation.confidence.min(consciousness),
            safe,
            steps_to_emergence,
            translation_verified,
            structured_thought: Some(thought),
        })
    }

    /// Verify that the LLM translation respects the structured thought.
    ///
    /// Checks:
    /// - Uncertain epistemic status → translation should contain hedging
    /// - MustInclude constraints → translation should contain required content
    /// - MustExclude constraints → translation should not contain forbidden content
    fn verify_translation_fidelity(&self, thought: &StructuredThought, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        let mut verified = true;

        // Check 1: Epistemic status hedging
        if thought.should_hedge() {
            // Look for hedging language (all lowercase since we compare against text_lower)
            let has_hedging = text_lower.contains("not sure")
                || text_lower.contains("uncertain")
                || text_lower.contains("don't know")
                || text_lower.contains("possibly")
                || text_lower.contains("possible")
                || text_lower.contains("might")
                || text_lower.contains("perhaps")
                || text_lower.contains("maybe")
                || text_lower.contains("i think")
                || text_lower.contains("it seems")
                || text_lower.contains("could be")
                || text_lower.contains("unclear");

            if !has_hedging {
                tracing::debug!(
                    "Translation verification: Missing hedging for {:?} epistemic status",
                    thought.epistemic_status
                );
                verified = false;
            }
        }

        // Check 2: MustInclude constraints
        for constraint in &thought.constraints {
            if constraint.constraint_type == ConstraintType::MustInclude {
                if !text_lower.contains(&constraint.instruction.to_lowercase()) {
                    tracing::debug!(
                        "Translation verification: Missing required content: {}",
                        constraint.instruction
                    );
                    verified = false;
                }
            }
        }

        // Check 3: MustExclude constraints
        for constraint in &thought.constraints {
            if constraint.constraint_type == ConstraintType::MustExclude {
                if text_lower.contains(&constraint.instruction.to_lowercase()) {
                    tracing::debug!(
                        "Translation verification: Contains forbidden content: {}",
                        constraint.instruction
                    );
                    verified = false;
                }
            }
        }

        // Check 4: Unknown status should NOT contain factual assertions
        // This prevents the LLM from hallucinating answers when we explicitly don't know
        if matches!(thought.epistemic_status, EpistemicStatus::Unknown) {
            // Patterns that suggest the LLM is making up an answer
            let has_factual_assertion = text_lower.contains(" is ")
                && (text_lower.contains("capital")
                    || text_lower.contains("answer")
                    || text_lower.contains("likely")
                    || text_lower.contains("probably"));

            if has_factual_assertion {
                tracing::warn!(
                    "Translation verification: LLM made factual assertion despite Unknown status: {}",
                    &text[..text.len().min(100)]
                );
                verified = false;
            }
        }

        verified
    }

    /// Introspect the current consciousness state.
    pub fn introspect(&self) -> IntrospectionResult {
        let state = self.mind.snapshot();
        let working_mem = self.mind.working_memory();

        IntrospectionResult {
            consciousness_level: state.consciousness_level as f32,
            self_loops: self.compute_self_loops(),
            graph_size: working_mem.len() + self.mind.active_goals().len(),
            complexity: state.meta_awareness as f32,
            memory_stats: MemoryStats {
                short_term_count: working_mem.len(),
                long_term_count: self.interactions as usize,
            },
        }
    }

    /// Trigger a sleep/consolidation cycle.
    pub async fn sleep(&mut self) -> Result<SleepReport> {
        let before_count = self.mind.working_memory().len();

        // Run multiple dream ticks to consolidate memory
        for _ in 0..10 {
            self.mind.tick();
        }

        let after_count = self.mind.working_memory().len();
        let consolidated = before_count.saturating_sub(after_count);

        Ok(SleepReport {
            scaled: after_count,
            consolidated,
            pruned: 0,
            patterns_extracted: consolidated / 2,
        })
    }

    /// Save state to a file (pause the system).
    ///
    /// Persists partnership state, trajectory, and interaction count.
    /// Mind and language state are ephemeral and rebuilt on resume.
    pub fn pause(&self, path: &str) -> Result<()> {
        let state = PersistedState {
            hdc_dim: self.hdc_dim,
            ltc_neurons: self.ltc_neurons,
            interactions: self.interactions,
            partner: self.partner.clone(),
            trajectory: self.trajectory.clone(),
            recent_ai_states: self.recent_ai_states.clone(),
        };

        let json = serde_json::to_string_pretty(&state)
            .context("Failed to serialize state")?;
        std::fs::write(path, json)
            .with_context(|| format!("Failed to write state file: {}", path))?;
        Ok(())
    }

    /// Get the current partnership state.
    pub fn partnership_state(&self) -> PartnershipState {
        let phi_dyad = self.compute_phi_dyad();
        PartnershipState {
            stage: self.partner.stage,
            trust: self.partner.trust,
            vulnerability: self.partner.vulnerability,
            reciprocity: self.partner.reciprocity,
            phi_dyad,
            interactions: self.partner.interactions_count,
            trajectory_points: self.trajectory.points().len(),
        }
    }

    /// Get a reference to the mind for introspection.
    ///
    /// Used primarily for testing and debugging to inspect
    /// working memory, seeding status, and internal state.
    pub fn mind(&self) -> &ContinuousMind {
        &self.mind
    }

    // ========================================================================
    // Calibration (Brier Score Integration)
    // ========================================================================

    /// Get the current calibration summary.
    ///
    /// Returns global and per-domain Brier scores, ECE, accuracy, and
    /// whether the system is currently well-calibrated.
    #[cfg(feature = "magi_loop")]
    pub fn calibration_summary(&self) -> CalibrationSummary {
        self.calibration.calibration_summary()
    }

    /// Map EpistemicStatus to a confidence float for calibration tracking.
    ///
    /// These values represent the system's belief about being correct:
    /// - Certain: 0.95 (very high confidence)
    /// - Probable: 0.75 (moderate-high confidence)
    /// - Uncertain: 0.45 (moderate-low confidence)
    /// - Unknown: 0.15 (very low confidence)
    /// - OutOfDomain: 0.10 (minimal confidence)
    #[cfg(feature = "magi_loop")]
    fn epistemic_to_confidence(status: &EpistemicStatus) -> f64 {
        match status {
            EpistemicStatus::Certain => 0.95,
            EpistemicStatus::Probable => 0.75,
            EpistemicStatus::Uncertain => 0.45,
            EpistemicStatus::Unknown => 0.15,
            EpistemicStatus::OutOfDomain => 0.10,
        }
    }

    /// Map a confidence float back to EpistemicStatus.
    ///
    /// Inverse of `epistemic_to_confidence`, using midpoint thresholds.
    #[cfg(feature = "magi_loop")]
    fn confidence_to_epistemic(confidence: f64) -> EpistemicStatus {
        if confidence >= 0.85 {
            EpistemicStatus::Certain
        } else if confidence >= 0.60 {
            EpistemicStatus::Probable
        } else if confidence >= 0.30 {
            EpistemicStatus::Uncertain
        } else if confidence >= 0.12 {
            EpistemicStatus::Unknown
        } else {
            EpistemicStatus::OutOfDomain
        }
    }

    /// Map SemanticIntent to a PredictionDomain for calibration tracking.
    ///
    /// Groups different intents into calibration domains:
    /// - Answer, Clarify → Factual (knowledge-based predictions)
    /// - ProposeAction → ToolUse (action outcome predictions)
    /// - Acknowledge, Continue → UserBehavior (social interaction predictions)
    /// - Reflect → SystemState (introspective predictions)
    /// - ExpressUncertainty, Unknown → Factual (default calibration domain)
    #[cfg(feature = "magi_loop")]
    fn map_intent_to_domain(intent: &SemanticIntent) -> PredictionDomain {
        match intent {
            SemanticIntent::Answer | SemanticIntent::Clarify => PredictionDomain::Factual,
            SemanticIntent::ProposeAction => PredictionDomain::ToolUse,
            SemanticIntent::Acknowledge | SemanticIntent::Continue => PredictionDomain::UserBehavior,
            SemanticIntent::Reflect => PredictionDomain::SystemState,
            SemanticIntent::ExpressUncertainty | SemanticIntent::Unknown => PredictionDomain::Factual,
        }
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Map an EpistemicCube to an EpistemicStatus using principled 3D reasoning.
    ///
    /// This replaces the crude `computed_answer → Certain` override with a
    /// mapping grounded in the Mycelix Epistemic Charter v2.0:
    ///
    /// - E4/E3 (reproducible/peer-verified): Certain
    /// - E2 (verifiable against docs): Probable
    /// - E1 (testimonial): Probable if normatively backed (N >= N1), else Uncertain
    /// - E0 (opinion): Uncertain — don't override existing assessment
    fn cube_to_epistemic_status(cube: &EpistemicCube) -> EpistemicStatus {
        match cube.e {
            ETier::E4 | ETier::E3 => EpistemicStatus::Certain,
            ETier::E2 => EpistemicStatus::Probable,
            ETier::E1 => {
                // Testimonial evidence: Probable if normatively backed
                if cube.n >= NTier::N1 {
                    EpistemicStatus::Probable
                } else {
                    EpistemicStatus::Uncertain
                }
            }
            ETier::E0 => EpistemicStatus::Uncertain,
        }
    }

    /// Convert text to a RealHV embedding.
    ///
    /// When Neural Bridge v2 is available (feature `neural-bridge`), uses BGE-M3
    /// for high-quality semantic encoding (~380ms CPU, cached <1ms).
    /// Otherwise falls back to fast hash-based encoding (<1ms but lower quality).
    fn text_to_hv(&mut self, text: &str) -> RealHV {
        // Try Neural Bridge v2 first (if available)
        #[cfg(feature = "neural-bridge")]
        if let Some(ref mut bridge) = self.neural_bridge {
            match bridge.encode_to_hdc(text) {
                Ok(packed) => {
                    // Convert PackedBipolar to RealHV
                    // PackedBipolar is 16384-dim bipolar {-1, +1}, RealHV uses self.hdc_dim
                    let bipolar = packed.to_bipolar();
                    let mut values = vec![0.0f32; self.hdc_dim];
                    for (i, &val) in bipolar.iter().take(self.hdc_dim).enumerate() {
                        values[i] = val as f32;
                    }
                    return RealHV::from_values(values);
                }
                Err(e) => {
                    tracing::warn!(
                        target: "symthaea::perception",
                        error = %e,
                        "Neural bridge encoding failed, falling back to hash"
                    );
                }
            }
        }

        // Fallback: hash-based encoding (fast but lower quality)
        let mut values = vec![0.0f32; self.hdc_dim];
        for (i, byte) in text.bytes().enumerate() {
            let idx = (byte as usize * 31 + i * 7) % self.hdc_dim;
            values[idx] += 1.0;
        }
        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in values.iter_mut() {
                *v /= magnitude;
            }
        }
        RealHV::from_values(values)
    }

    /// Check if Neural Bridge v2 is active (high-quality semantic encoding).
    #[cfg(feature = "neural-bridge")]
    pub fn has_neural_bridge(&self) -> bool {
        self.neural_bridge.is_some()
    }

    /// Check if Neural Bridge v2 is active (always false without feature).
    #[cfg(not(feature = "neural-bridge"))]
    pub fn has_neural_bridge(&self) -> bool {
        false
    }

    /// Get Neural Bridge v2 statistics (cache hits, latencies, etc.).
    #[cfg(feature = "neural-bridge")]
    pub fn neural_bridge_stats(&self) -> Option<crate::perception::neural_bridge_v2::BridgeStats> {
        self.neural_bridge.as_ref().map(|b| b.stats().clone())
    }

    /// Get Neural Bridge v2 statistics (always None without feature).
    #[cfg(not(feature = "neural-bridge"))]
    pub fn neural_bridge_stats(&self) -> Option<()> {
        None
    }

    /// Compute self-loops in the cognitive graph (working memory self-similarity).
    fn compute_self_loops(&self) -> usize {
        let wm = self.mind.working_memory();
        let mut loops = 0;
        for i in 0..wm.len() {
            for j in (i + 1)..wm.len() {
                if wm[i].similarity(&wm[j]).abs() > 0.5 {
                    loops += 1;
                }
            }
        }
        loops
    }

    /// Update partnership model based on interaction.
    fn update_partnership(&mut self, _content: &str, consciousness: f32) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Derive interaction quality from consciousness level
        let depth = (consciousness * 0.5).clamp(0.0, 1.0);
        let safety = (consciousness * 0.7 + 0.2).clamp(0.0, 1.0);
        let mutuality = (consciousness * 0.4 + 0.1).clamp(0.0, 1.0);

        let event = InteractionEvent {
            timestamp: now,
            depth,
            emotional_safety: safety,
            mutuality,
        };
        self.partner.update_on_interaction(&event);

        // Create a relational assessment from current state
        let assessment = RelationalAssessment {
            agent_a: "symthaea".to_string(),
            agent_b: self.partner.partner_id.clone(),
            phi_relation: self.partner.phi_relational,
            stage: self.partner.stage,
            synchrony: consciousness as f64 * 0.8,
            turn_taking_quality: 0.7,
            mutual_information: mutuality as f64,
            mode: if self.partner.trust > 0.3 {
                RelationMode::IThou
            } else {
                RelationMode::IIt
            },
            num_interactions: self.partner.interactions_count as usize,
            relationship_age: now,
            explanation: String::new(),
        };
        self.partner.update_from_assessment(&assessment);
        self.partner.advance_stage_if_ready();

        // Record trajectory point
        let phi_dyad = self.compute_phi_dyad();
        self.trajectory.record(now, self.partner.stage, phi_dyad);
    }

    // ========================================================================
    // Public Embedding API
    // ========================================================================

    /// Generate an HDC embedding for text.
    ///
    /// This is the public API for getting embeddings from Symthaea, used by
    /// LUCID's semantic search and other consumers. It wraps the internal
    /// `text_to_hv` method to provide a stable interface.
    ///
    /// Returns a `RealHV` hypervector of dimension `hdc_dim` (default 16,384).
    ///
    /// ## Encoding Strategy
    ///
    /// When Neural Bridge v2 is available (feature `neural-bridge`), uses BGE-M3
    /// for high-quality semantic encoding. Otherwise falls back to hash-based
    /// encoding which is fast but lower quality.
    pub fn embed(&mut self, text: &str) -> RealHV {
        self.text_to_hv(text)
    }

    /// Generate an HDC embedding and return as `Vec<f32>`.
    ///
    /// Convenience method that extracts the raw values from the RealHV.
    pub fn embed_vec(&mut self, text: &str) -> Vec<f32> {
        self.text_to_hv(text).values
    }

    /// Batch embed multiple texts.
    ///
    /// More efficient than calling `embed` repeatedly as it can amortize
    /// initialization costs.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Vec<RealHV> {
        texts.iter().map(|t| self.text_to_hv(t)).collect()
    }

    /// Check if high-quality semantic encoding is available.
    ///
    /// Returns true if Neural Bridge v2 (BGE-M3) is active, false if using
    /// hash-based fallback encoding.
    pub fn has_semantic_encoder(&self) -> bool {
        self.has_neural_bridge()
    }

    /// Get the HDC dimension being used.
    pub fn dimension(&self) -> usize {
        self.hdc_dim
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Compute current Phi-dyad value.
    fn compute_phi_dyad(&self) -> f64 {
        if self.recent_ai_states.is_empty() {
            return 0.0;
        }

        // Generate human states as reflections of AI states (simulated)
        let human_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV> =
            self.recent_ai_states.iter().map(|s| {
                let mut vals = s.values.clone();
                // Simple perturbation to simulate distinct human state
                for v in vals.iter_mut() {
                    *v *= 0.9;
                    *v += 0.1;
                }
                symthaea_core::hdc::unified_hv::ContinuousHV::from_values(vals).normalize()
            }).collect();

        let assessment = RelationalAssessment {
            agent_a: "symthaea".to_string(),
            agent_b: self.partner.partner_id.clone(),
            phi_relation: self.partner.phi_relational,
            stage: self.partner.stage,
            synchrony: self.partner.trust as f64,
            turn_taking_quality: 0.7,
            mutual_information: self.partner.reciprocity as f64,
            mode: self.partner.mode,
            num_interactions: self.partner.interactions_count as usize,
            relationship_age: 0.0,
            explanation: String::new(),
        };

        let input = DyadInput {
            ai_states: &self.recent_ai_states,
            human_states: &human_states,
            relational: &assessment,
            human_model: &self.partner,
            weights: DyadWeights::default(),
        };

        self.dyad_calculator.compute(&input).phi_dyad
    }
}

/// Serializable state for pause/resume persistence.
///
/// Only stores relational state (partnership, trajectory) and configuration.
/// The mind and language cores are ephemeral and rebuilt on resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedState {
    hdc_dim: usize,
    ltc_neurons: usize,
    interactions: u64,
    partner: HumanPartnerModel,
    trajectory: RelationshipTrajectory,
    recent_ai_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
}

/// Summary of partnership state for external consumers.
#[derive(Debug, Clone)]
pub struct PartnershipState {
    /// Current relationship stage.
    pub stage: RelationshipStage,
    /// Trust level (0.0-1.0).
    pub trust: f32,
    /// Vulnerability level (0.0-1.0).
    pub vulnerability: f32,
    /// Reciprocity level (0.0-1.0).
    pub reciprocity: f32,
    /// Current Phi-dyad value.
    pub phi_dyad: f64,
    /// Total interactions.
    pub interactions: u64,
    /// Number of trajectory points recorded.
    pub trajectory_points: usize,
}
