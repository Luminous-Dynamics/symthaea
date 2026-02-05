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
pub use utils::{EPSILON, EPSILON_F32, float_eq, float_eq_f32, is_zero, is_zero_f32, is_nonzero, is_nonzero_f32};
pub use intent::{IntentClassifier, IntentClassification, EpistemicAssessment, IntentScores, ConceptLabel, ConceptPrototype};
pub use knowledge::{DomainKnowledge, KnowledgeEntry, SeedingResult};
pub use structured_thought::*;

use std::collections::HashMap;
use symthaea_core::hdc::RealHV;

/// The continuous mind system
pub struct ContinuousMind {
    /// Configuration
    pub(crate) config: MindConfig,
    /// Current state
    pub(crate) state: MindState,
    /// Working memory
    pub(crate) working_memory: Vec<RealHV>,
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
}

impl ContinuousMind {
    /// Create a new continuous mind
    pub fn new(config: MindConfig) -> Self {
        let dim = config.dimension;
        Self {
            intent_classifier: IntentClassifier::new(dim),
            config,
            state: MindState {
                current_thought: RealHV::zero(dim),
                ..Default::default()
            },
            working_memory: Vec::new(),
            goals: Vec::new(),
            input_queue: Vec::new(),
            stats: MindStats::default(),
            awaken_time: std::time::Instant::now(),
            shutdown_requested: false,
            last_input_text: None,
            seeded_rng: None,
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
    pub fn perceive(&mut self, content: RealHV) {
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
    pub fn perceive_text(&mut self, text: &str, embedding: RealHV) {
        self.last_input_text = Some(text.to_string());
        self.perceive(embedding);
    }

    /// Set a goal
    pub fn set_goal(&mut self, description: impl Into<String>, embedding: RealHV, priority: f32) {
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
    pub fn working_memory(&self) -> &[RealHV] {
        &self.working_memory
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
        state.meta_awareness = (state.consciousness_level * 0.7
            + state.memory_utilization as f64 * 0.3).min(1.0);
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

            // Store in working memory
            self.working_memory.push(hv);

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
    fn encode_knowledge_entry(&self, entry: &knowledge::KnowledgeEntry) -> RealHV {
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

        RealHV::from_values(values)
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
            constraints: Vec::new(),
            original_input: None,
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
            let assessment = self.intent_classifier.assess_epistemic_text(text, &self.working_memory);

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
        let asking_question = self.goals.iter().any(|g| {
            g.description.ends_with('?')
        });

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
                        format!("unknown:concept_{}", i)
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
        mind.perceive(RealHV::random(512, 42));
        mind.tick();
        assert_eq!(mind.working_memory.len(), 1);
    }

    #[test]
    fn test_goal_setting() {
        let mut mind = ContinuousMind::default();
        mind.set_goal("Test goal", RealHV::random(512, 42), 1.0);
        mind.tick();
        assert!(!mind.active_goals().is_empty());
    }

    #[test]
    fn test_consciousness_update() {
        let mut mind = ContinuousMind::default();

        for i in 0..5 {
            mind.perceive(RealHV::random(512, 42 + i as u64));
        }

        for _ in 0..5 {
            mind.tick();
        }

        assert!(mind.state.consciousness_level > 0.0);
    }
}
