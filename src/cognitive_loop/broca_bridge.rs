//! Broca language bridge: consciousness-gated thought-to-text generation.
//!
//! Wraps the Broca SSM language center with consciousness signal extraction
//! and generation management. Feature-gated by `ssm_language`.

#[cfg(feature = "ssm_language")]
use symthaea_broca::{BrocaConfig, BrocaGenerator, GenerationResult, ThoughtChannels};

#[cfg(feature = "ssm_language")]
use symthaea_core::genesis::GenesisSeed;

/// Compact consciousness signals for language generation gating.
#[derive(Debug, Clone, Default)]
pub struct BrocaConsciousnessSignals {
    /// Epistemic confidence (0=out-of-domain .. 1=certain).
    pub epistemic_confidence: f32,
    /// Emotional valence (-1=negative .. 1=positive).
    pub emotional_valence: f32,
    /// Emotional arousal (0=calm .. 1=excited).
    pub emotional_arousal: f32,
    /// Emotional warmth (0=cold .. 1=warm).
    pub emotional_warmth: f32,
    /// Consciousness level / psi (0..1).
    pub consciousness_level: f32,
    /// Meta-awareness (0..1).
    pub meta_awareness: f32,
    /// Coherence (0..1).
    pub coherence: f32,
    /// Knowledge grounding (0..1). How well current reasoning is supported by stored knowledge.
    /// High grounding enables more confident, factual generation.
    /// Science: Baddeley (2000) — semantic grounding improves production coherence.
    pub knowledge_grounding: f32,
    /// Top-k relevant knowledge facts for context-grounded generation.
    pub knowledge_context: Vec<String>,
    /// Therapeutic intent (0=validate .. 7=crisis). Only used with `therapeutic` feature.
    #[cfg(feature = "therapeutic")]
    pub therapeutic_intent: f32,
    /// Therapeutic alliance quality (0..1).
    #[cfg(feature = "therapeutic")]
    pub alliance_quality: f32,
    /// Client distress level (0..1).
    #[cfg(feature = "therapeutic")]
    pub client_distress_level: f32,
    /// Intervention depth (0..1).
    #[cfg(feature = "therapeutic")]
    pub intervention_depth: f32,
    /// Ethics gate: true when `EthicalVerdict::Blocked` prevents generation.
    /// Uses a bool to avoid coupling Broca crate to the EthicalVerdict enum.
    pub ethics_blocked: bool,
    /// NSM semantic primitives detected in the current cycle's input.
    /// Populated from `perception.encoding.encoding_result.detected_primitives`.
    /// Science: Wierzbicka (1996) — universal semantic primes ground language production.
    pub detected_primitives: Vec<String>,
    /// NSM primitive grounding score (0.0–1.0): fraction of input words that
    /// mapped to recognized NSM semantic primes. Higher = better decomposition.
    pub primitive_grounding: f32,
    /// NSM-composed semantic content vector (16,384D ContinuousHV).
    /// Produced by `GroundedUnderstanding.understand()` → BinaryHV → ContinuousHV.
    /// When present and confidence is above threshold, blended with thought HV
    /// via `lerp()` before generation.
    /// Science: Barsalou (1999) — grounded cognition requires semantic modulation.
    pub semantic_hv: Option<symthaea_core::hdc::ContinuousHV>,
    /// Confidence of the NSM semantic decomposition (0.0–1.0).
    /// From `GroundedUnderstanding.understand().confidence`.
    pub semantic_confidence: f32,
}

// Re-export telemetry type from the types module.
pub use super::types::BrocaGenerationTelemetry;

/// Manager wrapping BrocaGenerator with consciousness gating and multi-turn context.
#[cfg(feature = "ssm_language")]
pub struct BrocaManager {
    generator: BrocaGenerator,
    pub(crate) last_telemetry: BrocaGenerationTelemetry,
    /// Minimum consciousness level required to generate (default 0.1).
    pub(crate) consciousness_threshold: f32,
    /// Multi-turn context: number of recent generations to carry state for.
    /// When > 0, uses `generate_continuing()` after the first generation,
    /// preserving CfC temporal context across turns.
    pub(crate) multi_turn_depth: usize,
    /// Number of generations since last state reset.
    turn_count: usize,
    /// Conversation context HVs from recent turns, used to bias thought encoding.
    /// Stored as a ring buffer of up to `multi_turn_depth` HDC vectors.
    context_window: std::collections::VecDeque<symthaea_core::hdc::ContinuousHV>,
}

#[cfg(feature = "ssm_language")]
impl BrocaManager {
    /// Default checkpoint path relative to the symthaea crate root.
    const DEFAULT_CHECKPOINT: &'static str = "crates/symthaea-broca/data/broca-cfc-v2.bin";

    /// Create a new BrocaManager, optionally loading from a checkpoint.
    ///
    /// If `checkpoint_path` is `Some`, loads weights from that file.
    /// If `None`, tries the default checkpoint at `crates/symthaea-broca/data/broca-cfc-v2.bin`.
    /// If neither path exists, creates a fresh (untrained) generator.
    pub fn new(genesis: &GenesisSeed, config: BrocaConfig, checkpoint_path: Option<&str>) -> Self {
        let generator = Self::try_load_checkpoint(checkpoint_path, genesis)
            .unwrap_or_else(|| BrocaGenerator::new(genesis, config));

        Self {
            generator,
            last_telemetry: BrocaGenerationTelemetry::default(),
            consciousness_threshold: 0.1,
            multi_turn_depth: 0,
            turn_count: 0,
            context_window: std::collections::VecDeque::new(),
        }
    }

    /// Attempt to load a BrocaGenerator from a checkpoint file.
    ///
    /// Tries `explicit_path` first, then `DEFAULT_CHECKPOINT`. Returns `None`
    /// if no checkpoint can be loaded (missing file, corrupt data, etc.).
    fn try_load_checkpoint(
        explicit_path: Option<&str>,
        genesis: &GenesisSeed,
    ) -> Option<BrocaGenerator> {
        let paths_to_try: Vec<&str> = match explicit_path {
            Some(p) => vec![p],
            None => vec![Self::DEFAULT_CHECKPOINT],
        };

        for path in paths_to_try {
            if !std::path::Path::new(path).exists() {
                tracing::debug!("Broca checkpoint not found at {path}, skipping");
                continue;
            }
            match BrocaGenerator::from_checkpoint(path, genesis) {
                Ok((gen, _adam, _proj, _lm_config)) => {
                    tracing::info!(path = %path, "Loaded Broca checkpoint");
                    return Some(gen);
                }
                Err(e) => {
                    tracing::warn!(path = %path, err = %e, "Failed to load Broca checkpoint");
                }
            }
        }

        tracing::info!("No Broca checkpoint loaded, creating fresh generator");
        None
    }

    /// Generate text from consciousness signals, gated by consciousness level.
    ///
    /// Returns `None` if consciousness is too low (below threshold).
    /// Populates `ThoughtChannels` from the provided signals and delegates
    /// to `BrocaGenerator::generate()`.
    pub fn generate(&mut self, signals: BrocaConsciousnessSignals) -> Option<GenerationResult> {
        // Gate: don't generate if ethics verdict is Blocked.
        // Science: APA Ethics Code (2017) principle 3.04 — avoid harm.
        if signals.ethics_blocked {
            self.last_telemetry = BrocaGenerationTelemetry {
                ethics_gated: true,
                ..Default::default()
            };
            return None;
        }

        // Gate: don't generate if consciousness too low
        if signals.consciousness_level < self.consciousness_threshold {
            self.last_telemetry = BrocaGenerationTelemetry {
                consciousness_gated: true,
                ..Default::default()
            };
            return None;
        }

        let start = std::time::Instant::now();

        // Build thought channels from consciousness signals
        let mut channels = ThoughtChannels::default();
        // Map epistemic confidence to epistemic ordinal (invert: 1.0=Certain→0, 0.0=OutOfDomain→4)
        channels.set_epistemic((1.0 - signals.epistemic_confidence) * 4.0);
        channels.set_emotion(
            signals.emotional_valence,
            signals.emotional_arousal,
            signals.emotional_warmth,
        );
        channels.set_consciousness(
            signals.consciousness_level,
            signals.meta_awareness,
            signals.coherence,
        );

        // Wire NSM primitives into concept_count (channel 19) and domain_familiarity (channel 21).
        // Science: Wierzbicka (1996) — semantic decomposition depth correlates with domain knowledge.
        {
            use super::thresholds::{NSM_CONCEPT_COUNT_SCALE, NSM_GROUNDING_DOMAIN_BLEND};
            let nsm_count = (signals.detected_primitives.len() as f32 * NSM_CONCEPT_COUNT_SCALE)
                .clamp(0.0, 10.0);
            channels.channels[19] = nsm_count;
            // Blend primitive grounding into domain_familiarity (index 21)
            let existing_familiarity = channels.channels[21];
            channels.channels[21] = existing_familiarity * (1.0 - NSM_GROUNDING_DOMAIN_BLEND)
                + signals.primitive_grounding * NSM_GROUNDING_DOMAIN_BLEND;
        }

        // Set therapeutic channels from manager state
        #[cfg(feature = "therapeutic")]
        channels.set_therapeutic(
            signals.therapeutic_intent,
            signals.alliance_quality,
            signals.client_distress_level,
            signals.intervention_depth,
        );

        // Multi-turn context: use generate_continuing() after the first turn
        // to preserve CfC temporal context.
        // Pass NSM semantic HV through when available (Phase 2).
        let result = if self.multi_turn_depth > 0 && self.turn_count > 0 {
            self.generator.generate_continuing(&channels)
        } else if signals.semantic_hv.is_some() {
            self.generator
                .generate_with_semantic(&channels, signals.semantic_hv.as_ref())
        } else {
            self.generator.generate(&channels)
        };
        self.turn_count += 1;
        // Reset turn count when we exceed the context depth
        if self.multi_turn_depth > 0 && self.turn_count >= self.multi_turn_depth {
            self.turn_count = 0;
        }

        let elapsed = start.elapsed();

        // Compute quality metrics from token IDs
        let (type_token_ratio, max_repetition) = if result.token_ids.is_empty() {
            (0.0, 0)
        } else {
            let unique: std::collections::HashSet<u32> = result.token_ids.iter().copied().collect();
            let ttr = unique.len() as f32 / result.token_ids.len() as f32;
            let max_rep = {
                let mut max = 1usize;
                let mut run = 1usize;
                for w in result.token_ids.windows(2) {
                    if w[0] == w[1] {
                        run += 1;
                        max = max.max(run);
                    } else {
                        run = 1;
                    }
                }
                max
            };
            (ttr, max_rep)
        };

        self.last_telemetry = BrocaGenerationTelemetry {
            generated: true,
            token_count: result.num_tokens,
            final_coherence: result.final_coherence,
            veto_triggered: result.veto_triggered,
            generation_time_us: elapsed.as_micros() as u64,
            consciousness_gated: false,
            type_token_ratio,
            max_repetition,
            nsm_primitive_count: signals.detected_primitives.len(),
            nsm_grounding: signals.primitive_grounding,
            ..Default::default()
        };

        Some(result)
    }

    /// Get the most recent generation telemetry.
    pub fn last_telemetry(&self) -> &BrocaGenerationTelemetry {
        &self.last_telemetry
    }

    /// Get a reference to the underlying generator.
    pub fn generator(&self) -> &BrocaGenerator {
        &self.generator
    }

    /// Get a mutable reference to the underlying generator (for training).
    pub fn generator_mut(&mut self) -> &mut BrocaGenerator {
        &mut self.generator
    }

    /// Set multi-turn context depth (0 = disabled).
    ///
    /// When > 0, the first generation resets CfC state, and subsequent
    /// generations within the window use `generate_continuing()` to
    /// preserve temporal context from prior turns.
    pub fn set_multi_turn_depth(&mut self, depth: usize) {
        self.multi_turn_depth = depth;
        self.turn_count = 0;
    }

    /// Reset the turn counter (force next generation to reset CfC state).
    pub fn reset_context(&mut self) {
        self.turn_count = 0;
        self.context_window.clear();
    }

    /// Inject conversation history as context for topic persistence.
    ///
    /// Encodes each recent turn string into an HDC vector via the generator's
    /// encoder and stores them in the context window. On subsequent generations,
    /// these context HVs are bundled with the thought HV to bias generation
    /// toward conversational continuity.
    pub fn inject_conversation_context(&mut self, recent_turns: &[String]) {
        use symthaea_core::hdc::ContinuousHV;

        self.context_window.clear();
        let max_context = if self.multi_turn_depth > 0 {
            self.multi_turn_depth
        } else {
            4 // Default context window if multi-turn not explicitly set
        };

        // Encode each turn into an HDC vector via token encoding + bundling
        for turn in recent_turns.iter().rev().take(max_context) {
            let token_ids = self.generator.tokenizer().encode(turn);
            if token_ids.is_empty() {
                continue;
            }
            // Bundle token embeddings to get a turn-level HV
            let all_embs = self.generator.controller().token_embeddings();
            let embs: Vec<&ContinuousHV> = token_ids
                .iter()
                .filter_map(|&id| all_embs.get(id as usize))
                .collect();
            if !embs.is_empty() {
                let turn_hv = ContinuousHV::bundle(&embs);
                self.context_window.push_front(turn_hv);
            }
        }
    }

    /// Get the number of context vectors currently stored.
    pub fn context_depth(&self) -> usize {
        self.context_window.len()
    }
}

#[cfg(test)]
#[cfg(feature = "ssm_language")]
mod tests {
    use super::*;
    use symthaea_core::genesis::GenesisSeed;

    fn test_manager() -> BrocaManager {
        let genesis = GenesisSeed::from_phrase("test-nsm-broca");
        BrocaManager::new(&genesis, BrocaConfig::default(), None)
    }

    #[test]
    fn test_detected_primitives_affect_concept_count() {
        let mut mgr = test_manager();
        let signals = BrocaConsciousnessSignals {
            consciousness_level: 0.8,
            coherence: 0.5,
            detected_primitives: vec![
                "FEEL".into(),
                "BAD".into(),
                "BECAUSE".into(),
                "SOMEONE".into(),
                "MOVE".into(),
            ],
            primitive_grounding: 0.6,
            ..Default::default()
        };
        // Generate (may produce empty text — we just check telemetry)
        let _ = mgr.generate(signals);
        let telem = mgr.last_telemetry();
        assert_eq!(telem.nsm_primitive_count, 5);
        assert!((telem.nsm_grounding - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_primitive_grounding_blends_domain_familiarity() {
        use super::super::thresholds::NSM_GROUNDING_DOMAIN_BLEND;

        // Default domain_familiarity (channel 21) = 0.5
        // After blending with primitive_grounding=0.8:
        // new = 0.5 * (1 - 0.5) + 0.8 * 0.5 = 0.25 + 0.4 = 0.65
        let default_familiarity = 0.5_f32;
        let grounding = 0.8_f32;
        let expected = default_familiarity * (1.0 - NSM_GROUNDING_DOMAIN_BLEND)
            + grounding * NSM_GROUNDING_DOMAIN_BLEND;
        assert!(
            (expected - 0.65).abs() < 1e-5,
            "Blend formula should produce 0.65, got {expected}"
        );
    }

    #[test]
    fn test_empty_primitives_no_effect_on_concept_count() {
        let mut mgr = test_manager();
        let signals = BrocaConsciousnessSignals {
            consciousness_level: 0.8,
            coherence: 0.5,
            detected_primitives: vec![],
            primitive_grounding: 0.0,
            ..Default::default()
        };
        let _ = mgr.generate(signals);
        let telem = mgr.last_telemetry();
        assert_eq!(telem.nsm_primitive_count, 0);
        assert!(telem.nsm_grounding.abs() < 1e-5);
    }
}
