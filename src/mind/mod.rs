//! # Continuous Mind: The Integrated Consciousness System
//!
//! Provides the main orchestration layer for the conscious AI system,
//! integrating perception, reasoning, memory, and action into a unified
//! continuous-time cognitive architecture.

mod config;
mod goals;
pub mod intent;
pub mod knowledge;
pub mod structured_thought;
mod tick;
mod utils;

pub use config::*;
pub use intent::{
    ConceptLabel, ConceptPrototype, EpistemicAssessment, IntentClassification, IntentClassifier,
    IntentScores,
};
pub use knowledge::{DomainKnowledge, KnowledgeEntry, SeedingResult};
pub use structured_thought::*;
pub use utils::{
    float_eq, float_eq_f32, is_nonzero, is_nonzero_f32, is_zero, is_zero_f32, EPSILON, EPSILON_F32,
};

use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;

/// The continuous mind system
pub struct ContinuousMind {
    /// Configuration
    pub(crate) config: MindConfig,
    /// Current state
    pub(crate) state: MindState,
    /// Working memory
    pub(crate) working_memory: Vec<ContinuousHV>,
    /// Arrival tick for each working memory item (parallel array).
    /// Used to compute accurate `steps_survived` on eviction.
    pub(crate) working_memory_ticks: Vec<u64>,
    /// Goal stack
    pub(crate) goals: Vec<Goal>,
    /// Input queue
    pub(crate) input_queue: Vec<MindInput>,
    /// Statistics
    pub(crate) stats: MindStats,
    /// Time of awakening
    pub(crate) awaken_time: std::time::Instant,
    /// Shutdown has been requested
    shutdown_requested: bool,
    /// HDC-based intent classifier for algebraic intuition
    intent_classifier: IntentClassifier,
    /// Most recent input text (for classification)
    last_input_text: Option<String>,
    /// Optional genesis-seeded RNG for deterministic dream processing
    seeded_rng: Option<symthaea_core::genesis::ShakeRng>,
    /// Optional federated learning aggregator.
    /// When enabled, the tick loop participates in distributed gradient exchange.
    pub(crate) federated: Option<crate::swarm::FederatedAggregator>,
    /// Incoming gradient messages from network peers.
    pub(crate) federated_inbox: Vec<crate::swarm::GradientMessage>,
    /// Outgoing gradient messages to broadcast to peers.
    pub(crate) federated_outbox: Vec<crate::swarm::GradientMessage>,
    /// Buffer of items evicted from working memory when capacity is exceeded.
    /// Each entry is `(hypervector, steps_survived)` where `steps_survived`
    /// is `current_tick - arrival_tick` at the moment of eviction.
    /// Consuming code can drain this via `take_evicted()` and route items
    /// to episodic memory or the MemoryCoordinator for graduation.
    evicted_items: Vec<(ContinuousHV, u64)>,
    /// Optional social coherence (theory of mind) system.
    /// When enabled, the mind models other agents' mental states and
    /// uses social reasoning to inform cooperation decisions.
    pub(crate) social_coherence: Option<crate::brain::SocialCoherence>,
    /// Incoming social messages from network peers.
    pub(crate) social_inbox: Vec<SocialMessage>,
    /// Outgoing social messages to broadcast to peers.
    pub(crate) social_outbox: Vec<SocialMessage>,
    /// Optional Iroh P2P bridge for real-time social message exchange.
    /// When set, the tick loop flushes `social_outbox` to the network
    /// and drains inbound messages into `social_inbox` after each
    /// `process_social()` call.
    pub(crate) iroh_bridge: Option<crate::swarm::IrohBridgeHandle>,
}

impl ContinuousMind {
    /// Create a new continuous mind
    pub fn new(config: MindConfig) -> Self {
        let dim = config.dimension;
        let social = if config.enable_social_coherence {
            Some(crate::brain::SocialCoherence::new(
                crate::brain::SocialCoherenceConfig {
                    dimension: dim,
                    ..Default::default()
                },
            ))
        } else {
            None
        };
        Self {
            intent_classifier: IntentClassifier::new(dim),
            config,
            state: MindState {
                current_thought: ContinuousHV::zero(dim),
                ..Default::default()
            },
            working_memory: Vec::new(),
            working_memory_ticks: Vec::new(),
            goals: Vec::new(),
            input_queue: Vec::new(),
            stats: MindStats::default(),
            awaken_time: std::time::Instant::now(),
            shutdown_requested: false,
            last_input_text: None,
            seeded_rng: None,
            federated: None,
            federated_inbox: Vec::new(),
            federated_outbox: Vec::new(),
            evicted_items: Vec::new(),
            social_coherence: social,
            social_inbox: Vec::new(),
            social_outbox: Vec::new(),
            iroh_bridge: None,
        }
    }

    /// Create a continuous mind with deterministic RNG from a genesis seed.
    pub fn from_genesis(
        config: MindConfig,
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> Self {
        let mut mind = Self::new(config);
        mind.seeded_rng = Some(genesis.domain(&format!("{label}::mind")));
        mind
    }

    /// Add input to the mind
    pub fn input(&mut self, input: MindInput) {
        self.input_queue.push(input);
    }

    /// Add a perception input
    pub fn perceive(&mut self, content: ContinuousHV) {
        self.input(MindInput {
            input_type: InputType::Perception,
            content,
            priority: 0.5,
            metadata: HashMap::new(),
        });
    }

    /// Set the original input text for intent classification.
    ///
    /// Call this before `tick()` to enable HDC-based intent inference.
    pub fn set_input_text(&mut self, text: impl Into<String>) {
        self.last_input_text = Some(text.into());
    }

    /// Perceive with text context for better classification.
    ///
    /// Combines HDC encoding with text-based intent classification.
    pub fn perceive_text(&mut self, text: &str, embedding: ContinuousHV) {
        self.last_input_text = Some(text.to_string());
        self.perceive(embedding);
    }

    /// Set a goal
    pub fn set_goal(
        &mut self,
        description: impl Into<String>,
        embedding: ContinuousHV,
        priority: f32,
    ) {
        let mut metadata = HashMap::new();
        metadata.insert("description".to_string(), description.into());

        self.input(MindInput {
            input_type: InputType::Goal,
            content: embedding,
            priority,
            metadata,
        });
    }

    /// Activate the mind
    pub fn activate(&mut self) {
        self.state.is_active = true;
    }

    /// Deactivate the mind
    pub fn deactivate(&mut self) {
        self.state.is_active = false;
    }

    /// Get current state
    pub fn state(&self) -> &MindState {
        &self.state
    }

    /// Get configuration
    pub fn config(&self) -> &MindConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &MindStats {
        &self.stats
    }

    /// Get working memory contents
    pub fn working_memory(&self) -> &[ContinuousHV] {
        &self.working_memory
    }

    /// Drain items evicted from working memory since the last call.
    ///
    /// Returns `(hypervector, steps_survived)` pairs. `steps_survived` is the
    /// number of ticks the item spent in working memory before eviction.
    /// These can be routed to the MemoryCoordinator for graduation.
    pub fn take_evicted(&mut self) -> Vec<(ContinuousHV, u64)> {
        std::mem::take(&mut self.evicted_items)
    }

    /// Get active goals
    pub fn active_goals(&self) -> Vec<&Goal> {
        self.goals.iter().filter(|g| g.is_active).collect()
    }

    /// Awaken the mind - start consciousness processing
    pub fn awaken(&mut self) {
        self.state.is_active = true;
        self.state.is_conscious = true;
        self.awaken_time = std::time::Instant::now();
    }

    /// Get a snapshot of the current mind state
    pub fn snapshot(&self) -> MindState {
        let mut state = self.state.clone();
        state.phi = state.consciousness_level;
        state.total_cycles = state.tick;
        state.time_awake_ms = self.awaken_time.elapsed().as_millis() as u64;
        state.meta_awareness =
            (state.consciousness_level * 0.7 + state.memory_utilization as f64 * 0.3).min(1.0);
        state.cognitive_load = state.memory_utilization as f64;
        state.is_conscious = state.consciousness_level >= self.config.min_consciousness;
        state
    }

    /// Request graceful shutdown of the mind
    pub fn request_shutdown(&mut self) {
        self.state.is_active = false;
        self.state.is_conscious = false;
        self.shutdown_requested = true;
    }

    /// Check if shutdown was requested
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    // ========================================================================
    // Federated Learning Interface
    // ========================================================================

    /// Enable federated learning with initial weights.
    pub fn enable_federated(&mut self, weights: Vec<f32>) {
        use crate::swarm::FederatedAggregator;
        self.federated = Some(FederatedAggregator::new(weights).with_byzantine_tolerance(0.1));
    }

    /// Receive a gradient message from a network peer.
    pub fn receive_gradient(&mut self, msg: crate::swarm::GradientMessage) {
        self.federated_inbox.push(msg);
    }

    /// Drain outgoing gradient messages (for network broadcast).
    pub fn drain_outbox(&mut self) -> Vec<crate::swarm::GradientMessage> {
        std::mem::take(&mut self.federated_outbox)
    }

    /// Check if federated learning is enabled.
    pub fn is_federated(&self) -> bool {
        self.federated.is_some()
    }

    // ========================================================================
    // Social Coherence Interface
    // ========================================================================

    /// Enable social coherence after construction.
    pub fn enable_social_coherence(&mut self) {
        if self.social_coherence.is_none() {
            self.social_coherence = Some(crate::brain::SocialCoherence::new(
                crate::brain::SocialCoherenceConfig {
                    dimension: self.config.dimension,
                    ..Default::default()
                },
            ));
        }
    }

    /// Receive a social message from a network peer.
    pub fn receive_social(&mut self, msg: SocialMessage) {
        self.social_inbox.push(msg);
    }

    /// Drain outgoing social messages (for network broadcast).
    pub fn drain_social_outbox(&mut self) -> Vec<SocialMessage> {
        std::mem::take(&mut self.social_outbox)
    }

    /// Check if social coherence is enabled.
    pub fn is_social(&self) -> bool {
        self.social_coherence.is_some()
    }

    /// Get a reference to the social coherence system (if enabled).
    pub fn social_coherence(&self) -> Option<&crate::brain::SocialCoherence> {
        self.social_coherence.as_ref()
    }

    // ========================================================================
    // Iroh P2P Bridge Interface
    // ========================================================================

    /// Attach an Iroh P2P bridge handle for real-time social message exchange.
    ///
    /// Once attached, each `tick()` will automatically:
    /// 1. Flush `social_outbox` messages to the network via the bridge
    /// 2. Drain inbound network messages into `social_inbox`
    ///
    /// The bridge actor must be spawned separately on a tokio runtime.
    pub fn set_iroh_bridge(&mut self, handle: crate::swarm::IrohBridgeHandle) {
        self.iroh_bridge = Some(handle);
    }

    /// Check if an Iroh P2P bridge is attached and alive.
    pub fn has_iroh_bridge(&self) -> bool {
        self.iroh_bridge.as_ref().is_some_and(|h| h.is_alive())
    }

    // ========================================================================
    // Working Memory Seeding (Epistemic Baseline)
    // ========================================================================

    /// Seed working memory with a priori domain knowledge.
    ///
    /// This establishes the epistemic baseline - concepts the system knows
    /// from "birth". Without seeding, the classifier defaults to Unknown
    /// for everything because working memory is empty.
    ///
    /// # Returns
    ///
    /// A `SeedingResult` containing statistics about the seeding operation.
    pub fn seed_memory(&mut self) -> SeedingResult {
        use knowledge::DomainKnowledge;

        let entries = DomainKnowledge::get_initial_seeding();
        let total = entries.len();

        tracing::info!(
            target: "symthaea::mind",
            count = total,
            "Seeding working memory with domain prototypes"
        );

        let mut total_magnitude = 0.0f32;
        let mut categories_seen = std::collections::HashSet::new();

        for entry in &entries {
            // Encode the knowledge entry into a hypervector
            let hv = self.encode_knowledge_entry(entry);

            // Track statistics
            let magnitude: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
            total_magnitude += magnitude;
            categories_seen.insert(entry.category.to_string());

            // Store in working memory (tick 0 = genesis seeding)
            self.working_memory.push(hv);
            self.working_memory_ticks.push(0);

            tracing::debug!(
                target: "symthaea::mind::seeding",
                label = entry.label,
                category = entry.category,
                magnitude = magnitude,
                "Seeded knowledge prototype"
            );
        }

        let avg_magnitude = if total > 0 {
            total_magnitude / total as f32
        } else {
            0.0
        };

        tracing::info!(
            target: "symthaea::mind",
            prototypes = total,
            categories = categories_seen.len(),
            avg_magnitude = avg_magnitude,
            "Seeding complete - epistemic baseline established"
        );

        SeedingResult {
            prototypes_seeded: total,
            categories: categories_seen.into_iter().collect(),
            avg_magnitude,
        }
    }

    /// Encode a knowledge entry into a hypervector.
    ///
    /// Uses the same encoding as text perception to ensure alignment
    /// between seeded knowledge and runtime inputs.
    fn encode_knowledge_entry(&self, entry: &knowledge::KnowledgeEntry) -> ContinuousHV {
        // Combine label and content for richer encoding
        let combined = format!("{} {}", entry.label.replace('_', " "), entry.content);

        // Use the same encoding method as text_to_hv_internal in IntentClassifier
        let dim = self.config.dimension;
        let mut values = vec![0.0f32; dim];
        let text_lower = combined.to_lowercase();

        for (i, byte) in text_lower.bytes().enumerate() {
            let idx = (byte as usize * 31 + i * 7) % dim;
            values[idx] += entry.confidence; // Weight by confidence
        }

        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if is_nonzero_f32(magnitude) {
            for v in values.iter_mut() {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_values(values)
    }

    /// Check if memory has been seeded
    pub fn is_seeded(&self) -> bool {
        // We consider memory seeded if it has at least 10 entries
        // (the minimum from domain knowledge)
        self.working_memory.len() >= 10
    }

    /// Get the number of seeded prototypes
    pub fn seeded_count(&self) -> usize {
        self.working_memory.len()
    }

    // ========================================================================
    // Structured Thought Extraction (Broca's Area Interface)
    // ========================================================================

    /// Extract a structured thought from the current mind state.
    ///
    /// This is the key interface between the HDC+LTC cognitive system and the
    /// LLM translation layer. The mind computes; this method articulates what
    /// was computed into a structured format for faithful translation.
    ///
    /// **Critical Insight**: The LLM should NOT add reasoning - only translate
    /// what this method returns.
    pub fn extract_structured_thought(&self) -> StructuredThought {
        use symthaea_core::hdc::relational_consciousness::{RelationMode, RelationshipStage};

        let state = self.snapshot();

        // Determine epistemic status from consciousness metrics
        let epistemic_status = self.determine_epistemic_status(&state);

        // Infer semantic intent from goals and working memory state
        let semantic_intent = self.infer_semantic_intent();

        // Infer response type
        let response_type = self.infer_response_type();

        // Extract top concepts from working memory
        let activated_concepts = self.extract_top_concepts(5);

        // Calculate working memory coherence
        let coherence = self.calculate_coherence();

        // Calculate relational warmth from emotional state
        let warmth = self.calculate_relational_warmth(&state);

        StructuredThought {
            semantic_intent,
            response_type,
            activated_concepts,
            emotional_tone: EmotionalTone {
                valence: state.emotional_valence as f64,
                arousal: state.arousal as f64,
                warmth,
            },
            structured_data: None,
            domain_context: None,
            phi: state.consciousness_level,
            meta_awareness: state.meta_awareness,
            coherence,
            epistemic_status,
            // Relational fields are filled by the Symthaea facade
            // from the partnership module
            relationship_stage: RelationshipStage::NoRelation,
            relation_mode: RelationMode::IIt,
            trust: 0.0,
            code_context: None,
            constraints: Vec::new(),
            original_input: None,
            primitive_tiers: Vec::new(), // Populated by Symthaea facade from language grounding
        }
    }

    /// Determine epistemic status using HDC algebraic assessment.
    ///
    /// Combines:
    /// 1. **HDC Resonance**: How familiar is the input to our prototypes?
    /// 2. **Memory Resonance**: Do we have relevant context in working memory?
    /// 3. **Consciousness Metrics**: Phi and meta-awareness modulate certainty
    ///
    /// This is the KEY function for hallucination prevention:
    /// - High familiarity + high phi → Certain
    /// - Low familiarity + empty memory → Unknown (triggers hedging)
    fn determine_epistemic_status(&self, state: &MindState) -> EpistemicStatus {
        // If we have input text, use HDC classification
        if let Some(ref text) = self.last_input_text {
            let assessment = self
                .intent_classifier
                .assess_epistemic_text(text, &self.working_memory);

            // Modulate by consciousness level
            let phi = state.consciousness_level;
            let meta = state.meta_awareness;

            // High consciousness can upgrade Uncertain → Probable
            // Low consciousness can downgrade Probable → Uncertain
            match assessment.status {
                EpistemicStatus::Certain => {
                    if phi > 0.7 && meta > 0.6 {
                        EpistemicStatus::Certain
                    } else if phi > 0.5 {
                        EpistemicStatus::Probable
                    } else {
                        EpistemicStatus::Uncertain
                    }
                }
                EpistemicStatus::Probable => {
                    if phi > 0.8 && meta > 0.7 && assessment.familiarity > 0.7 {
                        EpistemicStatus::Certain
                    } else if phi > 0.4 {
                        EpistemicStatus::Probable
                    } else {
                        EpistemicStatus::Uncertain
                    }
                }
                EpistemicStatus::Uncertain => {
                    if phi > 0.8 && meta > 0.8 && assessment.familiarity > 0.6 {
                        EpistemicStatus::Probable
                    } else {
                        EpistemicStatus::Uncertain
                    }
                }
                EpistemicStatus::Unknown => {
                    // Unknown stays unknown - we don't have the knowledge
                    // This is the hallucination prevention mechanism
                    EpistemicStatus::Unknown
                }
                EpistemicStatus::OutOfDomain => EpistemicStatus::OutOfDomain,
            }
        } else {
            // Fallback to pure consciousness metrics if no text available
            let phi = state.consciousness_level;
            let meta = state.meta_awareness;

            if phi > 0.8 && meta > 0.7 {
                EpistemicStatus::Certain
            } else if phi > 0.6 && meta > 0.5 {
                EpistemicStatus::Probable
            } else if phi > 0.3 || meta > 0.3 {
                EpistemicStatus::Uncertain
            } else {
                EpistemicStatus::Unknown
            }
        }
    }

    /// Infer semantic intent using HDC prototype resonance.
    ///
    /// Computes cosine similarity between input and intent prototypes:
    /// - **Greeting**: "hello", "hi", etc.
    /// - **Question**: "what", "why", "?", etc.
    /// - **Command**: "do", "make", "create", etc.
    /// - **Reflection**: "think", "feel", etc.
    ///
    /// Falls back to goal/memory heuristics if no text is available.
    fn infer_semantic_intent(&self) -> SemanticIntent {
        // If we have input text, use HDC classification
        if let Some(ref text) = self.last_input_text {
            let classification = self.intent_classifier.classify_text(text);

            // If confidence is high enough, use the HDC classification
            if classification.confidence > 0.3 {
                return classification.intent;
            }
            // Fall through to heuristics for low confidence
        }

        // Fallback: Goal and memory-based heuristics
        let has_goals = !self.goals.is_empty();
        let has_memory = !self.working_memory.is_empty();
        let is_conscious = self.state.is_conscious;

        if !is_conscious {
            return SemanticIntent::Acknowledge;
        }

        // Check if any goal suggests clarification need
        let needs_clarification = self.goals.iter().any(|g| {
            g.description.to_lowercase().contains("clarify")
                || g.description.to_lowercase().contains("question")
        });

        if needs_clarification {
            return SemanticIntent::Clarify;
        }

        // Check for action-oriented goals
        let action_oriented = self.goals.iter().any(|g| {
            g.description.to_lowercase().contains("do")
                || g.description.to_lowercase().contains("execute")
                || g.description.to_lowercase().contains("action")
        });

        if action_oriented {
            return SemanticIntent::ProposeAction;
        }

        // If we have working memory content, we likely have an answer
        if has_memory && has_goals {
            return SemanticIntent::Answer;
        }

        // Low consciousness suggests uncertainty
        if self.state.consciousness_level < 0.3 {
            return SemanticIntent::ExpressUncertainty;
        }

        // Default to acknowledgment if nothing specific
        if has_memory {
            SemanticIntent::Answer
        } else {
            SemanticIntent::Acknowledge
        }
    }

    /// Infer response type using HDC classification.
    ///
    /// Uses the response_type from intent classification when available.
    fn infer_response_type(&self) -> ResponseType {
        // If we have input text, use HDC classification
        if let Some(ref text) = self.last_input_text {
            let classification = self.intent_classifier.classify_text(text);
            if classification.confidence > 0.3 {
                return classification.response_type;
            }
        }

        // Fallback to heuristics

        // If high arousal and positive valence, might be empathic
        if self.state.arousal > 0.7 && self.state.emotional_valence > 0.5 {
            return ResponseType::Empathic;
        }

        // If goals suggest questions
        let asking_question = self.goals.iter().any(|g| g.description.ends_with('?'));

        if asking_question {
            return ResponseType::Question;
        }

        // Default to statement
        ResponseType::Statement
    }

    /// Extract top N activated concepts from working memory.
    ///
    /// Uses the HDC concept vocabulary to label working memory contents
    /// via nearest-neighbor lookup against concept prototypes.
    fn extract_top_concepts(&self, n: usize) -> Vec<ActivatedConcept> {
        if self.working_memory.is_empty() {
            return Vec::new();
        }

        // Label all working memory contents
        let labels = self.intent_classifier.label_concepts(&self.working_memory);

        // Convert to ActivatedConcepts, taking top N by confidence
        labels
            .into_iter()
            .take(n)
            .enumerate()
            .map(|(i, label)| {
                // Combine label confidence with position-based decay
                let position_factor = 1.0 - (i as f32 * 0.1); // Decay by 10% per position
                let activation = label.confidence * position_factor;

                ActivatedConcept {
                    // Use semantic label instead of placeholder
                    name: if label.confidence > 0.3 {
                        format!("{}:{}", label.category, label.name)
                    } else {
                        // Fall back to generic if confidence too low
                        format!("unknown:concept_{i}")
                    },
                    activation,
                    relevance: label.similarity.max(0.0) * activation,
                }
            })
            .collect()
    }

    /// Calculate coherence of working memory.
    ///
    /// Measures how well-integrated the current thoughts are by computing
    /// average pairwise similarity in working memory.
    fn calculate_coherence(&self) -> f64 {
        if self.working_memory.len() < 2 {
            return 0.5; // Neutral coherence for insufficient data
        }

        let mut total_similarity = 0.0;
        let mut count = 0;

        for i in 0..self.working_memory.len() {
            for j in (i + 1)..self.working_memory.len() {
                let sim = self.working_memory[i]
                    .similarity(&self.working_memory[j])
                    .abs() as f64;
                total_similarity += sim;
                count += 1;
            }
        }

        if count > 0 {
            total_similarity / count as f64
        } else {
            0.5
        }
    }

    /// Calculate relational warmth from emotional state.
    fn calculate_relational_warmth(&self, state: &MindState) -> f64 {
        // Warmth increases with positive valence and moderate arousal
        let valence_contrib = (state.emotional_valence as f64 + 1.0) / 2.0; // Normalize to 0-1
        let arousal_contrib = 1.0 - (state.arousal as f64 - 0.5).abs(); // Peak at 0.5

        (valence_contrib * 0.7 + arousal_contrib * 0.3).clamp(0.0, 1.0)
    }
}

impl Default for ContinuousMind {
    fn default() -> Self {
        Self::new(MindConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mind_creation() {
        let mind = ContinuousMind::default();
        assert_eq!(mind.state.tick, 0);
        assert!(!mind.state.is_active);
    }

    #[test]
    fn test_mind_tick() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        mind.tick();
        assert_eq!(mind.state.tick, 1);
    }

    #[test]
    fn test_perception() {
        let mut mind = ContinuousMind::default();
        mind.perceive(ContinuousHV::random(512, 42));
        mind.tick();
        assert_eq!(mind.working_memory.len(), 1);
    }

    #[test]
    fn test_goal_setting() {
        let mut mind = ContinuousMind::default();
        mind.set_goal("Test goal", ContinuousHV::random(512, 42), 1.0);
        mind.tick();
        assert!(!mind.active_goals().is_empty());
    }

    #[test]
    fn test_consciousness_update() {
        let mut mind = ContinuousMind::default();

        for i in 0..5 {
            mind.perceive(ContinuousHV::random(512, 42 + i as u64));
        }

        for _ in 0..5 {
            mind.tick();
        }

        assert!(mind.state.consciousness_level > 0.0);
    }

    // ====================================================================
    // Social Coherence Integration Tests
    // ====================================================================

    #[test]
    fn test_social_coherence_disabled_by_default() {
        let mind = ContinuousMind::default();
        assert!(!mind.is_social());
        assert!(mind.social_coherence().is_none());
    }

    #[test]
    fn test_social_coherence_enabled_via_config() {
        let mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        assert!(mind.is_social());
        assert!(mind.social_coherence().is_some());
    }

    #[test]
    fn test_social_coherence_enable_after_construction() {
        let mut mind = ContinuousMind::default();
        assert!(!mind.is_social());
        mind.enable_social_coherence();
        assert!(mind.is_social());
    }

    #[test]
    fn test_social_inbox_processed_on_tick() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();

        // Send a social message
        mind.receive_social(SocialMessage {
            agent_id: "peer_1".to_string(),
            behavior: ContinuousHV::random(512, 0xBEEF_0001),
            context: ContinuousHV::random(512, 0xBEEF_0002),
            interaction_outcome: None,
        });

        assert_eq!(mind.social_inbox.len(), 1);
        mind.tick();
        // Inbox should be drained after tick
        assert_eq!(mind.social_inbox.len(), 0);
        // Agent should be modeled now
        let sc = mind.social_coherence().unwrap();
        assert!(sc.get_mental_model("peer_1").is_some());
    }

    #[test]
    fn test_social_interaction_builds_relationship() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();

        // Send cooperative interaction
        mind.receive_social(SocialMessage {
            agent_id: "ally_1".to_string(),
            behavior: ContinuousHV::random(512, 0xA11E_0001),
            context: ContinuousHV::random(512, 0xA11E_0002),
            interaction_outcome: Some(0.9),
        });
        mind.tick();

        let sc = mind.social_coherence().unwrap();
        let rel = sc.get_relationship("ally_1");
        assert!(rel.is_some(), "Relationship should be created");
        assert!(rel.unwrap().trust > 0.5, "Trust should increase from cooperation");
    }

    #[test]
    fn test_social_outbox_populated_on_tick() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();

        // Tick 5 times to trigger outbox export (every 5 ticks)
        for _ in 0..5 {
            mind.tick();
        }

        let outbox = mind.drain_social_outbox();
        assert!(!outbox.is_empty(), "Outbox should have messages after 5 ticks");
        assert_eq!(outbox[0].agent_id, "self");
    }

    #[test]
    fn test_social_no_processing_when_disabled() {
        let mut mind = ContinuousMind::default();
        mind.activate();

        // Inbox messages should remain when social is disabled
        // (actually they get drained regardless but social coherence isn't updated)
        mind.receive_social(SocialMessage {
            agent_id: "ghost".to_string(),
            behavior: ContinuousHV::random(512, 0xDEAD),
            context: ContinuousHV::random(512, 0xDEAD),
            interaction_outcome: None,
        });
        mind.tick();

        // Social coherence is None, so no models are built
        assert!(mind.social_coherence().is_none());
        // Outbox should be empty since social is disabled
        let outbox = mind.drain_social_outbox();
        assert!(outbox.is_empty());
    }

    // ====================================================================
    // Iroh P2P Bridge Integration Tests
    // ====================================================================

    #[test]
    fn test_iroh_bridge_not_set_by_default() {
        let mind = ContinuousMind::default();
        assert!(!mind.has_iroh_bridge());
    }

    #[test]
    fn test_iroh_bridge_attach() {
        let mut mind = ContinuousMind::default();
        let (handle, _actor) = crate::swarm::IrohBridgeHandle::new(4, 4);
        mind.set_iroh_bridge(handle);
        assert!(mind.has_iroh_bridge());
    }

    #[test]
    fn test_iroh_bridge_flushes_outbox_on_tick() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();
        let (handle, _actor) = crate::swarm::IrohBridgeHandle::new(64, 128);
        mind.set_iroh_bridge(handle);

        // Tick 5 times — social coherence exports on tick 5
        for _ in 0..5 {
            mind.tick();
        }

        // Outbox should be empty because the bridge flushed it
        assert!(
            mind.social_outbox.is_empty(),
            "Bridge should have flushed the outbox"
        );
    }

    #[test]
    fn test_iroh_bridge_drains_inbox_on_tick() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();
        let (handle, actor) = crate::swarm::IrohBridgeHandle::new(64, 128);

        // We need the actor's inbound_tx to inject messages.
        // Instead, manually push to inbox and verify tick processes it.
        // The bridge integration is: bridge drains → inbox, tick processes inbox → social coherence.
        // We can verify the bridge wiring by checking that when bridge is attached,
        // outbox messages get sent to the bridge channel.
        mind.set_iroh_bridge(handle);

        // Manually inject into inbox (simulating what bridge.drain_inbox would return)
        mind.receive_social(SocialMessage {
            agent_id: "network_peer".to_string(),
            behavior: ContinuousHV::random(512, 0xCAFE),
            context: ContinuousHV::random(512, 0xCAFE),
            interaction_outcome: None,
        });

        mind.tick();

        // The message should have been processed by social coherence
        let sc = mind.social_coherence().unwrap();
        assert!(
            sc.get_mental_model("network_peer").is_some(),
            "Network peer should be modeled after tick"
        );

        // Suppress unused variable warning
        drop(actor);
    }
}
