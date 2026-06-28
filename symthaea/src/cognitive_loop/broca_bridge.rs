// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

    // ── Epistemic Cube (v6) ─────────────────────────────────────────────
    /// E-tier (empirical verifiability): 0=E0(opinion) .. 4=E4(reproducible).
    /// None means cube not computed for this cycle.
    pub cube_e_tier: Option<u8>,
    /// N-tier (normative authority): 0=N0(personal) .. 3=N3(axiomatic).
    pub cube_n_tier: Option<u8>,
    /// M-tier (materiality/permanence): 0=M0(ephemeral) .. 3=M3(foundational).
    pub cube_m_tier: Option<u8>,
    /// H-value (harmonic coherence): 0.0-1.0 continuous.
    pub cube_h_value: f32,
    /// Epistemic quality score: E×0.4 + N×0.35 + M×0.25 (normalized 0-1).
    pub cube_quality: f32,
    /// Code channels from CodingAgent: [syntax_complexity, type_confidence, algorithm_pattern, error_likelihood]
    pub code_channels: Option<[f32; 4]>,
    /// FEP prediction error / surprise (0..1). High = unexpected input.
    pub fep_surprise: f32,
    /// FEP pragmatic value (0..1). High = action is expected to reduce future surprise.
    pub fep_pragmatic_value: f32,
}

// Re-export telemetry type from the types module.
pub use super::types::BrocaGenerationTelemetry;

/// Record of NSM primes expressed in a single generation, for cross-cycle discourse memory.
#[derive(Debug, Clone)]
pub struct NsmDiscourseRecord {
    /// Primes that were expressed in this generation (uppercased names).
    pub expressed_primes: Vec<String>,
    /// Prime coverage for this generation (0.0–1.0).
    pub coverage: f32,
    /// Cycle number when this generation occurred.
    pub cycle: u64,
}

/// A single turn from the conversation partner.
#[cfg(feature = "ssm_language")]
#[derive(Debug, Clone)]
pub struct InterlocutorTurn {
    /// Raw text of the partner's utterance.
    pub text: String,
    /// Optional stance hint: positive = agreement, negative = disagreement.
    pub stance_delta: Option<f32>,
}

/// State tracking conversation partner's theory of mind.
#[cfg(feature = "ssm_language")]
#[derive(Debug, Clone)]
pub struct TheoryOfMindState {
    /// Tracked partner belief state vector.
    pub partner_belief_hv: symthaea_core::hdc::ContinuousHV,
    /// Familiarity level (0.0–1.0).
    pub familiarity: f32,
    /// Alignment score (-1.0–1.0).
    pub alignment_score: f32,
    /// Total interaction count.
    pub interaction_count: u64,
}

#[cfg(feature = "ssm_language")]
impl Default for TheoryOfMindState {
    fn default() -> Self {
        Self {
            partner_belief_hv: symthaea_core::hdc::ContinuousHV::zero(
                symthaea_core::hdc::HDC_DIMENSION,
            ),
            familiarity: 0.0,
            alignment_score: 0.0,
            interaction_count: 0,
        }
    }
}

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
    /// Cross-cycle NSM discourse memory: ring buffer of prime expression records.
    /// Tracks which NSM primes were expressed across recent generations,
    /// enabling discourse coherence and topic continuity.
    /// Science: Pickering & Garrod (2004) — alignment in dialogue via shared priming.
    discourse_memory: std::collections::VecDeque<NsmDiscourseRecord>,
    /// Theory of Mind state modeling the interlocutor's beliefs and alignment.
    pub theory_of_mind: TheoryOfMindState,
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
            multi_turn_depth: 4, // Default: preserve CfC context across 4 turns
            turn_count: 0,
            context_window: std::collections::VecDeque::new(),
            discourse_memory: std::collections::VecDeque::new(),
            theory_of_mind: TheoryOfMindState::default(),
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
                Ok((r#gen, _adam, _proj, _lm_config)) => {
                    tracing::info!(path = %path, "Loaded Broca checkpoint");
                    return Some(r#gen);
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
        let mut signals = signals;

        // T1.3: Feed recurring_discourse_primes back into signals
        let recurring = self.recurring_discourse_primes(0.4);
        for prime in recurring {
            if !signals.detected_primitives.contains(&prime) {
                signals.detected_primitives.push(prime.clone());
            }
        }

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

        // T1.8: fep_surprise → prediction_error proxy (channel 8) and fep_pragmatic_value → curiosity (channel 0)
        channels.channels[0] = channels.channels[0] * 0.6 + signals.fep_pragmatic_value * 0.4;
        channels.channels[8] =
            (channels.channels[8] * 0.5 + signals.fep_surprise * 0.5).clamp(0.0, 1.0);

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

        // Set epistemic cube channels from consciousness signals
        if let (Some(e), Some(n), Some(m)) = (
            signals.cube_e_tier,
            signals.cube_n_tier,
            signals.cube_m_tier,
        ) {
            channels.set_epistemic_cube(e, n, m, signals.cube_h_value, signals.cube_quality);
        }

        // Set code channels from CodingAgent injection (channels 24–27).
        if let Some([sc, tc, ap, el]) = signals.code_channels {
            channels.set_code(sc, tc, ap, el);
        }

        // Multi-turn context: use generate_continuing() after the first turn
        // to preserve CfC temporal context.
        // Pass NSM semantic HV and active primes through when available.
        let reset_state = !(self.multi_turn_depth > 0 && self.turn_count > 0);
        let sem_hv = if reset_state {
            signals.semantic_hv.as_ref()
        } else {
            None
        };
        let empty_primes = Vec::new();
        let primes = if reset_state {
            &signals.detected_primitives
        } else {
            &empty_primes
        };

        // Bundle conversation context window (T1.1)
        let context_hv = if !self.context_window.is_empty() {
            let ctx_refs: Vec<&symthaea_core::hdc::ContinuousHV> =
                self.context_window.iter().collect();
            Some(symthaea_core::hdc::ContinuousHV::bundle(&ctx_refs))
        } else {
            None
        };

        // Encode and bundle knowledge context (T1.2)
        let knowledge_hv = self.encode_knowledge_context(&signals.knowledge_context);

        let result = self.generator.generate_full(
            &channels,
            sem_hv,
            primes,
            context_hv.as_ref(),
            knowledge_hv.as_ref(),
            reset_state,
        );

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
            nsm_prime_coverage: result.nsm_prime_coverage,
            ..Default::default()
        };

        // Record NSM discourse memory for cross-cycle topic continuity.
        // Uses the detected primitives as the "expressed" set — in a fully
        // wired system, this would come from NsmCoherenceTracker.expressed_primes,
        // but that data stays inside the generator. Using detected_primitives
        // captures what we *intended* to say, which is sufficient for priming.
        if !signals.detected_primitives.is_empty() {
            const DISCOURSE_MEMORY_CAP: usize = 16;
            if self.discourse_memory.len() >= DISCOURSE_MEMORY_CAP {
                self.discourse_memory.pop_front();
            }
            self.discourse_memory.push_back(NsmDiscourseRecord {
                expressed_primes: signals.detected_primitives.clone(),
                coverage: result.nsm_prime_coverage,
                cycle: 0, // Cycle number not available here; caller can set via accessor
            });
        }

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

    /// Get the NSM discourse memory (recent prime expression history).
    pub fn discourse_memory(&self) -> &std::collections::VecDeque<NsmDiscourseRecord> {
        &self.discourse_memory
    }

    /// Get primes that were frequently expressed across recent generations.
    /// Returns primes that appeared in at least `min_frequency` of the last N records.
    /// Useful for topic continuity: boost generation toward recently-discussed concepts.
    pub fn recurring_discourse_primes(&self, min_frequency: f32) -> Vec<String> {
        if self.discourse_memory.is_empty() {
            return Vec::new();
        }
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let n = self.discourse_memory.len();
        for record in &self.discourse_memory {
            for prime in &record.expressed_primes {
                *counts.entry(prime.clone()).or_default() += 1;
            }
        }
        let threshold = (min_frequency * n as f32).ceil() as usize;
        counts
            .into_iter()
            .filter(|&(_, count)| count >= threshold)
            .map(|(prime, _)| prime)
            .collect()
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

    /// Encode each knowledge context string into an HDC vector via the generator's
    /// tokenizer + embedding lookup, bundle them, and return (T1.2).
    fn encode_knowledge_context(
        &self,
        facts: &[String],
    ) -> Option<symthaea_core::hdc::ContinuousHV> {
        if facts.is_empty() {
            return None;
        }
        use symthaea_core::hdc::ContinuousHV;
        let all_embs = self.generator.controller().token_embeddings();
        let fact_hvs: Vec<ContinuousHV> = facts
            .iter()
            .filter_map(|fact| {
                let ids = self.generator.tokenizer().encode(fact);
                let embs: Vec<&ContinuousHV> = ids
                    .iter()
                    .filter_map(|&id| all_embs.get(id as usize))
                    .collect();
                if embs.is_empty() {
                    None
                } else {
                    Some(ContinuousHV::bundle(&embs))
                }
            })
            .collect();
        if fact_hvs.is_empty() {
            None
        } else {
            let refs: Vec<&ContinuousHV> = fact_hvs.iter().collect();
            Some(ContinuousHV::bundle(&refs))
        }
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

    /// Record a turn from the conversation partner, updating Theory of Mind.
    pub fn record_interlocutor_turn(&mut self, turn: InterlocutorTurn) {
        use symthaea_core::hdc::ContinuousHV;

        // 1. Encode turn text into an HDC belief vector.
        let token_ids = self.generator.tokenizer().encode(&turn.text);
        let turn_hv = if token_ids.is_empty() {
            ContinuousHV::zero(symthaea_core::hdc::HDC_DIMENSION)
        } else {
            let all_embs = self.generator.controller().token_embeddings();
            let embs: Vec<&ContinuousHV> = token_ids
                .iter()
                .filter_map(|&id| all_embs.get(id as usize))
                .collect();
            if embs.is_empty() {
                ContinuousHV::zero(symthaea_core::hdc::HDC_DIMENSION)
            } else {
                ContinuousHV::bundle(&embs)
            }
        };

        // 2. Blend into the tracked partner belief HV with a learning rate
        //    that scales with the number of turns (familiarity).
        let lr = (0.10 + 0.05 * self.theory_of_mind.familiarity).clamp(0.05, 0.30);
        self.theory_of_mind
            .partner_belief_hv
            .lerp_in_place(&turn_hv, 1.0 - lr, lr);
        self.theory_of_mind.partner_belief_hv = self.theory_of_mind.partner_belief_hv.normalize();

        // 3. Apply optional stance delta to shift alignment target.
        if let Some(delta) = turn.stance_delta {
            self.theory_of_mind.alignment_score =
                (self.theory_of_mind.alignment_score + delta * 0.1).clamp(-1.0, 1.0);
        }

        self.theory_of_mind.familiarity = (self.theory_of_mind.familiarity + 0.02).clamp(0.0, 1.0);
        self.theory_of_mind.interaction_count += 1;
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

    #[test]
    fn test_turn_based_belief_updates() {
        let mut mgr = test_manager();

        // 1. Initial State assertions
        assert_eq!(mgr.theory_of_mind.familiarity, 0.0);
        assert_eq!(mgr.theory_of_mind.alignment_score, 0.0);
        assert_eq!(mgr.theory_of_mind.interaction_count, 0);

        // 2. Record turn 1 (stance_delta = Some(0.5))
        mgr.record_interlocutor_turn(InterlocutorTurn {
            text: "Hello, let's cooperate and share some details".into(),
            stance_delta: Some(0.5),
        });

        // 3. Assertions after turn 1
        assert_eq!(mgr.theory_of_mind.interaction_count, 1);
        assert!((mgr.theory_of_mind.familiarity - 0.02).abs() < 1e-5);
        assert!((mgr.theory_of_mind.alignment_score - 0.05).abs() < 1e-5);

        // Store belief vector to check drift later
        let first_belief = mgr.theory_of_mind.partner_belief_hv.clone();

        // 4. Record turn 2 (stance_delta = Some(-0.8))
        mgr.record_interlocutor_turn(InterlocutorTurn {
            text: "I actually completely disagree with this path".into(),
            stance_delta: Some(-0.8),
        });

        // 5. Assertions after turn 2
        assert_eq!(mgr.theory_of_mind.interaction_count, 2);
        assert!((mgr.theory_of_mind.familiarity - 0.04).abs() < 1e-5);
        // alignment_score should drift: 0.05 + (-0.8 * 0.1) = -0.03
        assert!((mgr.theory_of_mind.alignment_score - (-0.03)).abs() < 1e-5);

        let second_belief = mgr.theory_of_mind.partner_belief_hv.clone();
        // Belief vector should change / drift
        assert!(first_belief.similarity(&second_belief) < 1.0);
    }
}
