// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Understanding Pipeline
//!
//! Integrates all understanding capabilities into a coherent pipeline:
//!
//! 1. **Grounded Understanding** - Semantic primes + embodiment
//! 2. **Predictive Processing** - Expectations + surprise + learning
//! 3. **Narrative Memory** - Temporal coherence + autobiographical self
//! 4. **Theory of Mind** - Speaker intention + perspective-taking
//! 5. **Learning Lexicon** - Dynamic vocabulary expansion
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED UNDERSTANDING PIPELINE                       │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │                                                                        │
//! │  Text Input ───┬───────────────────────────────────────────────────    │
//! │                │                                                       │
//! │                ▼                                                       │
//! │  ┌─────────────────────────────┐                                       │
//! │  │   LexicalExpander           │◄─── Unknown words → WordLearner      │
//! │  │   (Dynamic Vocabulary)      │     Learn from context/internet       │
//! │  └─────────────┬───────────────┘                                       │
//! │                │                                                       │
//! │                ▼                                                       │
//! │  ┌─────────────────────────────┐                                       │
//! │  │   GroundedUnderstanding     │◄─── 65 Wierzbicka primes             │
//! │  │   (Semantic Decomposition)  │     Embodied grounding                │
//! │  └─────────────┬───────────────┘                                       │
//! │                │                                                       │
//! │                ▼                                                       │
//! │  ┌─────────────────────────────┐                                       │
//! │  │   PredictiveIntegration     │◄─── Generate expectations            │
//! │  │   (Surprise → Learning)     │     Compute prediction error          │
//! │  └─────────────┬───────────────┘                                       │
//! │                │                                                       │
//! │                ▼                                                       │
//! │  ┌─────────────────────────────┐                                       │
//! │  │   NarrativeIntegration      │◄─── Update autobiographical memory   │
//! │  │   (Temporal Coherence)      │     Thread causal story               │
//! │  └─────────────┬───────────────┘                                       │
//! │                │                                                       │
//! │                ▼                                                       │
//! │  ┌─────────────────────────────┐                                       │
//! │  │   TheoryOfMindIntegration   │◄─── Model speaker's intentions       │
//! │  │   (Perspective-Taking)      │     Empathic understanding            │
//! │  └─────────────┬───────────────┘                                       │
//! │                │                                                       │
//! │                ▼                                                       │
//! │  ┌─────────────────────────────┐                                       │
//! │  │   DeepUnderstanding         │◄─── Unified conscious comprehension  │
//! │  │   (Φ-Integrated Meaning)    │     Ready for conscious processing    │
//! │  └─────────────────────────────┘                                       │
//! │                                                                        │
//! └────────────────────────────────────────────────────────────────────────┘
//! ```

use super::binary_hv::BinaryHV;
use super::grounded_understanding::{
    CausalRelation, EmbodiedGrounding, GroundedUnderstanding, UnderstoodMeaning,
};
use super::predictive_coding::PredictiveCoding;
use super::universal_semantics::SemanticPrime;
use std::collections::{HashMap, VecDeque};

// =============================================================================
// DEEP UNDERSTANDING: The complete result of unified processing
// =============================================================================

/// Complete understanding with all integration layers
#[derive(Debug, Clone)]
pub struct DeepUnderstanding {
    /// Original text
    pub text: String,

    /// Grounded meaning (semantic primes + embodiment)
    pub grounded: UnderstoodMeaning,

    /// Predictive processing results
    pub predictive: PredictiveResult,

    /// Narrative integration
    pub narrative: NarrativeContext,

    /// Theory of mind inference
    pub speaker_model: SpeakerModel,

    /// Words that were learned during processing
    pub learned_words: Vec<LearnedWordInfo>,

    /// Overall understanding confidence (0-1)
    pub confidence: f64,

    /// Φ-weighted integration score
    pub integration_phi: f64,
}

/// Results from predictive processing
#[derive(Debug, Clone)]
pub struct PredictiveResult {
    /// Was this expected or surprising?
    pub surprise: f32,

    /// Prediction error magnitude
    pub prediction_error: f32,

    /// What was predicted vs what occurred
    pub predicted_primes: Vec<SemanticPrime>,
    pub actual_primes: Vec<SemanticPrime>,

    /// Free energy (lower = better fit with model)
    pub free_energy: f64,

    /// Did this update our world model?
    pub model_updated: bool,
}

/// Narrative context from autobiographical integration
#[derive(Debug, Clone)]
pub struct NarrativeContext {
    /// How this fits into the ongoing story
    pub story_role: StoryRole,

    /// Entities mentioned (tracked across conversation)
    pub entities: Vec<TrackedEntity>,

    /// Causal links to previous statements
    pub causal_links: Vec<CausalLink>,

    /// Temporal relationship to narrative
    pub temporal_position: TemporalPosition,

    /// Identity coherence with speaker's self-model
    pub identity_coherence: f64,
}

/// Role this utterance plays in the narrative
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoryRole {
    /// Introduces new information
    Introduction,
    /// Continues existing thread
    Continuation,
    /// Provides explanation for previous events
    Explanation,
    /// Contradicts or corrects previous information
    Correction,
    /// Expresses emotional reaction
    Reaction,
    /// Asks for information
    Query,
    /// Concludes a narrative arc
    Resolution,
}

/// An entity being tracked in the narrative
#[derive(Debug, Clone)]
pub struct TrackedEntity {
    /// Entity identifier
    pub name: String,
    /// Semantic representation
    pub semantic_hv: BinaryHV,
    /// Role in narrative (agent, patient, etc.)
    pub role: EntityRole,
    /// Emotional valence associated with entity
    pub valence: f32,
    /// Number of mentions
    pub mention_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntityRole {
    Agent,       // Does actions
    Patient,     // Receives actions
    Experiencer, // Feels/perceives
    Location,    // Where things happen
    Instrument,  // How things are done
    Topic,       // What is being discussed
}

/// Causal link between narrative elements
#[derive(Debug, Clone)]
pub struct CausalLink {
    /// What caused this
    pub cause: String,
    /// What resulted
    pub effect: String,
    /// Link strength
    pub strength: f32,
    /// Type of causation
    pub link_type: CausalLinkType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CausalLinkType {
    Physical,      // Physical causation
    Psychological, // Mental causation (beliefs, desires)
    Social,        // Social causation (norms, expectations)
    Temporal,      // Temporal sequence (before/after)
}

/// Temporal position in narrative
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemporalPosition {
    Past,
    Present,
    Future,
    Hypothetical,
    Atemporal,
}

/// Model of the speaker's mental state (Theory of Mind)
#[derive(Debug, Clone)]
pub struct SpeakerModel {
    /// What the speaker believes
    pub beliefs: Vec<InferredBelief>,

    /// What the speaker wants/intends
    pub intentions: Vec<InferredIntention>,

    /// Speaker's emotional state
    pub emotional_state: EmotionalInference,

    /// What the speaker knows (epistemic state)
    pub knowledge_state: EpistemicState,

    /// Speaker's perspective (different from listener's?)
    pub perspective_divergence: f32,
}

/// An inferred belief
#[derive(Debug, Clone)]
pub struct InferredBelief {
    /// What is believed
    pub content: String,
    /// Confidence in this inference
    pub confidence: f32,
    /// Evidence for this belief
    pub evidence: String,
}

/// An inferred intention
#[derive(Debug, Clone)]
pub struct InferredIntention {
    /// What the speaker intends
    pub goal: String,
    /// Type of speech act
    pub speech_act: SpeechAct,
    /// Confidence
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeechAct {
    Assert,   // Stating a fact
    Question, // Seeking information
    Command,  // Requesting action
    Promise,  // Committing to action
    Express,  // Expressing emotion
    Declare,  // Making something the case
}

/// Inferred emotional state of speaker
#[derive(Debug, Clone)]
pub struct EmotionalInference {
    /// Primary emotion
    pub primary: String,
    /// Valence
    pub valence: f32,
    /// Arousal
    pub arousal: f32,
    /// Confidence
    pub confidence: f32,
}

/// What the speaker knows/doesn't know
#[derive(Debug, Clone)]
pub struct EpistemicState {
    /// Things speaker explicitly knows
    pub known: Vec<String>,
    /// Things speaker is uncertain about
    pub uncertain: Vec<String>,
    /// Things speaker is ignorant of (that we know)
    pub ignorant: Vec<String>,
}

/// Information about a word that was learned
#[derive(Debug, Clone)]
pub struct LearnedWordInfo {
    /// The word
    pub word: String,
    /// Inferred semantic primes
    pub primes: Vec<SemanticPrime>,
    /// Learning method
    pub method: LearningMethod,
    /// Confidence
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearningMethod {
    Context,  // Inferred from surrounding words
    Internet, // Looked up online
    User,     // User provided definition
    Analogy,  // Inferred by analogy
}

// =============================================================================
// UNIFIED UNDERSTANDING PIPELINE
// =============================================================================

/// The complete unified understanding pipeline
pub struct UnifiedUnderstandingPipeline {
    /// Grounded understanding engine
    grounded: GroundedUnderstanding,

    /// Predictive coding system
    predictive: PredictiveCoding,

    /// Conversation history for narrative tracking
    narrative_history: VecDeque<NarrativeFrame>,

    /// Entity tracking across conversation
    tracked_entities: HashMap<String, TrackedEntity>,

    /// Previous predictions for surprise calculation
    previous_predictions: Vec<SemanticPrime>,

    /// Learning configuration
    learning_enabled: bool,

    /// Learned words this session
    learned_words: Vec<LearnedWordInfo>,

    /// Maximum history length
    max_history: usize,
}

/// A single frame in the narrative history
#[derive(Debug, Clone)]
struct NarrativeFrame {
    /// The text
    text: String,
    /// Key semantic primes
    primes: Vec<SemanticPrime>,
    /// Entities mentioned
    entities: Vec<String>,
    /// Timestamp (frame number)
    frame_id: u64,
}

impl UnifiedUnderstandingPipeline {
    /// Create a new unified understanding pipeline
    pub fn new() -> Self {
        Self {
            grounded: GroundedUnderstanding::new(),
            predictive: PredictiveCoding::new(4), // 4-layer hierarchy
            narrative_history: VecDeque::with_capacity(100),
            tracked_entities: HashMap::new(),
            previous_predictions: Vec::new(),
            learning_enabled: true,
            learned_words: Vec::new(),
            max_history: 100,
        }
    }

    /// Enable or disable learning
    pub fn set_learning(&mut self, enabled: bool) {
        self.learning_enabled = enabled;
    }

    /// Process text through the complete pipeline
    pub fn understand(&mut self, text: &str) -> DeepUnderstanding {
        // 1. Check for unknown words and attempt learning
        let learned_words = if self.learning_enabled {
            self.attempt_word_learning(text)
        } else {
            Vec::new()
        };

        // 2. Grounded understanding (semantic primes + embodiment)
        let grounded = self.grounded.understand(text);

        // 3. Predictive processing (surprise + prediction error)
        let predictive = self.compute_predictive(&grounded);

        // 4. Narrative integration (story coherence + entity tracking)
        let narrative = self.integrate_narrative(text, &grounded);

        // 5. Theory of mind (speaker model)
        let speaker_model = self.infer_speaker_model(&grounded, &narrative);

        // 6. Compute overall integration
        let confidence = self.compute_confidence(&grounded, &predictive, &narrative);
        let integration_phi = self.compute_integration_phi(&grounded, &predictive);

        // 7. Update history
        self.update_narrative_history(text, &grounded);

        DeepUnderstanding {
            text: text.to_string(),
            grounded,
            predictive,
            narrative,
            speaker_model,
            learned_words,
            confidence,
            integration_phi,
        }
    }

    /// Attempt to learn unknown words from context
    fn attempt_word_learning(&mut self, text: &str) -> Vec<LearnedWordInfo> {
        let mut learned = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        for word in words {
            let word_lower = word.to_lowercase();
            // Check if word is in lexicon
            if self.grounded.lexicon().decompose(&word_lower).is_none() {
                // Unknown word - try to infer from context
                if let Some(info) = self.infer_word_from_context(&word_lower, text) {
                    learned.push(info);
                }
            }
        }

        // Store learned words
        self.learned_words.extend(learned.clone());

        learned
    }

    /// Infer word meaning from surrounding context
    fn infer_word_from_context(&self, word: &str, context: &str) -> Option<LearnedWordInfo> {
        // Simple heuristic: look at surrounding known words
        let words: Vec<&str> = context.split_whitespace().collect();
        let word_idx = words.iter().position(|w| w.to_lowercase() == *word)?;

        // Get primes from surrounding words
        let mut context_primes = Vec::new();

        // Look at 2 words before and after
        for offset in -2i32..=2 {
            if offset == 0 {
                continue;
            }
            let idx = word_idx as i32 + offset;
            if idx >= 0 && (idx as usize) < words.len() {
                if let Some(primes) = self.grounded.lexicon().decompose(words[idx as usize]) {
                    context_primes.extend(primes.clone());
                }
            }
        }

        if context_primes.is_empty() {
            return None;
        }

        // Infer primes (simple: use most common from context)
        let mut prime_counts: HashMap<SemanticPrime, usize> = HashMap::new();
        for prime in &context_primes {
            *prime_counts.entry(*prime).or_insert(0) += 1;
        }

        let mut inferred_primes: Vec<SemanticPrime> = prime_counts.keys().copied().collect();
        inferred_primes.sort_by_key(|p| std::cmp::Reverse(prime_counts.get(p).unwrap_or(&0)));
        inferred_primes.truncate(3); // Top 3 primes

        let confidence = if inferred_primes.len() >= 2 { 0.5 } else { 0.3 };

        Some(LearnedWordInfo {
            word: word.to_string(),
            primes: inferred_primes,
            method: LearningMethod::Context,
            confidence,
        })
    }

    /// Compute predictive processing results
    fn compute_predictive(&mut self, grounded: &UnderstoodMeaning) -> PredictiveResult {
        // Calculate surprise based on prediction vs actual
        let actual_primes = &grounded.primes;

        let (surprise, prediction_error) = if self.previous_predictions.is_empty() {
            (0.5, 0.5) // Neutral for first input
        } else {
            // Count overlap between predicted and actual
            let overlap: usize = actual_primes
                .iter()
                .filter(|p| self.previous_predictions.contains(p))
                .count();
            let max_len = actual_primes
                .len()
                .max(self.previous_predictions.len())
                .max(1);
            let similarity = overlap as f32 / max_len as f32;

            let surprise = 1.0 - similarity;
            let prediction_error = surprise;
            (surprise, prediction_error)
        };

        // Update predictions for next time
        let predicted_primes = self.previous_predictions.clone();
        self.previous_predictions = actual_primes.clone();

        // Use predictive coding to process
        let (_, _) = self.predictive.predict_and_update(&grounded.semantic_hv);
        let free_energy = self.predictive.current_free_energy();

        PredictiveResult {
            surprise,
            prediction_error,
            predicted_primes,
            actual_primes: actual_primes.clone(),
            free_energy,
            model_updated: surprise > 0.3,
        }
    }

    /// Integrate with narrative history
    fn integrate_narrative(
        &mut self,
        text: &str,
        grounded: &UnderstoodMeaning,
    ) -> NarrativeContext {
        // Determine story role based on content
        let story_role = self.determine_story_role(text, grounded);

        // Track entities
        let entities = self.extract_and_track_entities(text, grounded);

        // Find causal links to previous statements
        let causal_links = self.find_causal_links(grounded);

        // Determine temporal position
        let temporal_position = self.determine_temporal_position(grounded);

        // Compute identity coherence
        let identity_coherence = self.compute_identity_coherence();

        NarrativeContext {
            story_role,
            entities,
            causal_links,
            temporal_position,
            identity_coherence,
        }
    }

    fn determine_story_role(&self, text: &str, grounded: &UnderstoodMeaning) -> StoryRole {
        let text_lower = text.to_lowercase();

        // Check for question markers
        if text_lower.contains('?')
            || text_lower.starts_with("what")
            || text_lower.starts_with("why")
            || text_lower.starts_with("how")
            || text_lower.starts_with("when")
            || text_lower.starts_with("where")
        {
            return StoryRole::Query;
        }

        // Check for causal explanation
        if grounded.causal_structure.is_some() {
            return StoryRole::Explanation;
        }

        // Check for emotional expression
        if grounded.embodied.arousal > 0.6 && grounded.embodied.valence.abs() > 0.4 {
            return StoryRole::Reaction;
        }

        // Check for contradiction markers
        if text_lower.contains("but")
            || text_lower.contains("however")
            || text_lower.contains("actually")
            || text_lower.contains("no,")
        {
            return StoryRole::Correction;
        }

        // Check if continues previous topic
        if !self.narrative_history.is_empty() {
            // Check for shared entities
            let current_words: std::collections::HashSet<_> =
                text_lower.split_whitespace().collect();
            if let Some(last) = self.narrative_history.back() {
                let shared = last
                    .entities
                    .iter()
                    .any(|e| current_words.contains(e.as_str()));
                if shared {
                    return StoryRole::Continuation;
                }
            }
        }

        // Default: introduction of new information
        StoryRole::Introduction
    }

    fn extract_and_track_entities(
        &mut self,
        text: &str,
        grounded: &UnderstoodMeaning,
    ) -> Vec<TrackedEntity> {
        let mut entities = Vec::new();

        // Simple entity extraction based on primes
        let words: Vec<&str> = text.split_whitespace().collect();

        for word in words {
            let word_lower = word.to_lowercase();

            // Check if it's a person/entity reference
            if let Some(primes) = self.grounded.lexicon().decompose(&word_lower) {
                if primes.contains(&SemanticPrime::Someone)
                    || primes.contains(&SemanticPrime::I)
                    || primes.contains(&SemanticPrime::You)
                {
                    // It's an entity reference
                    let entity = if let Some(existing) = self.tracked_entities.get_mut(&word_lower)
                    {
                        existing.mention_count += 1;
                        existing.clone()
                    } else {
                        let new_entity = TrackedEntity {
                            name: word_lower.clone(),
                            semantic_hv: grounded.semantic_hv,
                            role: EntityRole::Agent, // Default
                            valence: grounded.embodied.valence,
                            mention_count: 1,
                        };
                        self.tracked_entities.insert(word_lower, new_entity.clone());
                        new_entity
                    };
                    entities.push(entity);
                }
            }
        }

        entities
    }

    fn find_causal_links(&self, grounded: &UnderstoodMeaning) -> Vec<CausalLink> {
        let mut links = Vec::new();

        // If there's a causal structure, create a link
        if let Some(ref causal) = grounded.causal_structure {
            links.push(CausalLink {
                cause: causal.cause.clone(),
                effect: causal.effect.clone(),
                strength: causal.strength as f32,
                link_type: match causal.relation {
                    CausalRelation::Causes | CausalRelation::Prevents => CausalLinkType::Physical,
                    CausalRelation::Enables => CausalLinkType::Psychological,
                    CausalRelation::Temporal => CausalLinkType::Temporal,
                    CausalRelation::Conditional => CausalLinkType::Psychological,
                },
            });
        }

        links
    }

    fn determine_temporal_position(&self, grounded: &UnderstoodMeaning) -> TemporalPosition {
        // Check for temporal primes
        if grounded.primes.contains(&SemanticPrime::Now) {
            TemporalPosition::Present
        } else if grounded.primes.contains(&SemanticPrime::Before) {
            TemporalPosition::Past
        } else if grounded.primes.contains(&SemanticPrime::After) {
            TemporalPosition::Future
        } else if grounded.primes.contains(&SemanticPrime::If)
            || grounded.primes.contains(&SemanticPrime::Maybe)
        {
            TemporalPosition::Hypothetical
        } else {
            TemporalPosition::Atemporal
        }
    }

    fn compute_identity_coherence(&self) -> f64 {
        // Simple coherence based on entity consistency
        if self.tracked_entities.is_empty() {
            return 1.0;
        }

        let total_mentions: u32 = self
            .tracked_entities
            .values()
            .map(|e| e.mention_count)
            .sum();
        let unique_entities = self.tracked_entities.len();

        // More mentions of same entities = higher coherence
        let avg_mentions = total_mentions as f64 / unique_entities as f64;
        (avg_mentions / 5.0).min(1.0) // Normalize
    }

    /// Infer speaker's mental state (Theory of Mind)
    fn infer_speaker_model(
        &self,
        grounded: &UnderstoodMeaning,
        narrative: &NarrativeContext,
    ) -> SpeakerModel {
        // Infer beliefs from content
        let beliefs = self.infer_beliefs(grounded);

        // Infer intentions from speech act
        let intentions = self.infer_intentions(grounded, narrative);

        // Infer emotional state
        let emotional_state = EmotionalInference {
            primary: self.emotion_from_embodied(&grounded.embodied),
            valence: grounded.embodied.valence,
            arousal: grounded.embodied.arousal,
            confidence: grounded.confidence as f32,
        };

        // Epistemic state
        let knowledge_state = EpistemicState {
            known: Vec::new(),
            uncertain: Vec::new(),
            ignorant: Vec::new(),
        };

        // Perspective divergence (how different is speaker's view from ours)
        let perspective_divergence = if narrative.story_role == StoryRole::Correction {
            0.7 // High divergence if correcting
        } else {
            0.2 // Low by default
        };

        SpeakerModel {
            beliefs,
            intentions,
            emotional_state,
            knowledge_state,
            perspective_divergence,
        }
    }

    fn infer_beliefs(&self, grounded: &UnderstoodMeaning) -> Vec<InferredBelief> {
        let mut beliefs = Vec::new();

        // If speaker uses KNOW or THINK primes, they're expressing a belief
        if grounded.primes.contains(&SemanticPrime::Know)
            || grounded.primes.contains(&SemanticPrime::Think)
        {
            beliefs.push(InferredBelief {
                content: grounded.text.clone(),
                confidence: 0.8,
                evidence: "Contains KNOW/THINK primes".to_string(),
            });
        }

        beliefs
    }

    fn infer_intentions(
        &self,
        grounded: &UnderstoodMeaning,
        narrative: &NarrativeContext,
    ) -> Vec<InferredIntention> {
        let speech_act = match narrative.story_role {
            StoryRole::Query => SpeechAct::Question,
            StoryRole::Reaction => SpeechAct::Express,
            StoryRole::Introduction | StoryRole::Continuation | StoryRole::Explanation => {
                SpeechAct::Assert
            }
            StoryRole::Correction => SpeechAct::Assert,
            StoryRole::Resolution => SpeechAct::Declare,
        };

        let goal = match speech_act {
            SpeechAct::Question => "Seek information".to_string(),
            SpeechAct::Assert => "Share information".to_string(),
            SpeechAct::Express => "Express feelings".to_string(),
            SpeechAct::Command => "Request action".to_string(),
            SpeechAct::Promise => "Commit to action".to_string(),
            SpeechAct::Declare => "Establish fact".to_string(),
        };

        vec![InferredIntention {
            goal,
            speech_act,
            confidence: 0.7,
        }]
    }

    fn emotion_from_embodied(&self, embodied: &EmbodiedGrounding) -> String {
        if embodied.valence > 0.5 && embodied.arousal > 0.5 {
            "excited".to_string()
        } else if embodied.valence > 0.5 {
            "happy".to_string()
        } else if embodied.valence < -0.5 && embodied.arousal > 0.5 {
            "angry".to_string()
        } else if embodied.valence < -0.5 {
            "sad".to_string()
        } else if embodied.arousal < 0.3 {
            "calm".to_string()
        } else {
            "neutral".to_string()
        }
    }

    fn compute_confidence(
        &self,
        grounded: &UnderstoodMeaning,
        predictive: &PredictiveResult,
        narrative: &NarrativeContext,
    ) -> f64 {
        // Weighted combination of different confidence sources
        let grounded_conf = grounded.confidence;
        let predictive_conf = 1.0 - predictive.surprise as f64;
        let narrative_conf = narrative.identity_coherence;

        (grounded_conf * 0.4 + predictive_conf * 0.3 + narrative_conf * 0.3).min(1.0)
    }

    fn compute_integration_phi(
        &self,
        grounded: &UnderstoodMeaning,
        predictive: &PredictiveResult,
    ) -> f64 {
        // Integration = grounded Φ adjusted by prediction quality
        let base_phi = grounded.integration_phi;
        let prediction_factor = 1.0 - (predictive.free_energy / 10.0).min(0.5);

        base_phi * prediction_factor
    }

    fn update_narrative_history(&mut self, text: &str, grounded: &UnderstoodMeaning) {
        let frame_id = self.narrative_history.len() as u64;

        // Extract entity names from tracked entities
        let entity_names: Vec<String> = self
            .tracked_entities
            .keys()
            .filter(|name| text.to_lowercase().contains(name.as_str()))
            .cloned()
            .collect();

        let frame = NarrativeFrame {
            text: text.to_string(),
            primes: grounded.primes.clone(),
            entities: entity_names,
            frame_id,
        };

        self.narrative_history.push_back(frame);

        // Trim history if too long
        while self.narrative_history.len() > self.max_history {
            self.narrative_history.pop_front();
        }
    }

    /// Get conversation history length
    pub fn history_len(&self) -> usize {
        self.narrative_history.len()
    }

    /// Get tracked entities
    pub fn entities(&self) -> &HashMap<String, TrackedEntity> {
        &self.tracked_entities
    }

    /// Get learned words this session
    pub fn learned_words(&self) -> &[LearnedWordInfo] {
        &self.learned_words
    }

    /// Clear conversation history (new conversation)
    pub fn clear_history(&mut self) {
        self.narrative_history.clear();
        self.tracked_entities.clear();
        self.previous_predictions.clear();
    }
}

impl Default for UnifiedUnderstandingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = UnifiedUnderstandingPipeline::new();
        assert_eq!(pipeline.history_len(), 0);
    }

    #[test]
    fn test_basic_understanding() {
        let mut pipeline = UnifiedUnderstandingPipeline::new();
        let understanding = pipeline.understand("I feel happy");

        assert!(!understanding.grounded.primes.is_empty());
        assert!(understanding.confidence > 0.0);
    }

    #[test]
    fn test_narrative_continuity() {
        let mut pipeline = UnifiedUnderstandingPipeline::new();

        let u1 = pipeline.understand("I met my friend today");
        assert_eq!(u1.narrative.story_role, StoryRole::Introduction);

        let u2 = pipeline.understand("My friend was happy");
        // Should continue the narrative about "friend"
        assert!(
            pipeline.entities().contains_key("friend") || pipeline.entities().contains_key("my")
        );
    }

    #[test]
    fn test_surprise_calculation() {
        let mut pipeline = UnifiedUnderstandingPipeline::new();

        // First input - neutral surprise
        let u1 = pipeline.understand("I feel happy");
        assert!(u1.predictive.surprise >= 0.0);

        // Similar input - low surprise
        let u2 = pipeline.understand("I feel joyful");
        // Should be somewhat expected

        // Different input - higher surprise
        let u3 = pipeline.understand("The big dog moved quickly");
        // Should be more surprising
    }

    #[test]
    fn test_theory_of_mind() {
        let mut pipeline = UnifiedUnderstandingPipeline::new();

        let understanding = pipeline.understand("Why did you do that?");

        // Should detect as question
        assert_eq!(understanding.narrative.story_role, StoryRole::Query);
        assert!(!understanding.speaker_model.intentions.is_empty());
        assert_eq!(
            understanding.speaker_model.intentions[0].speech_act,
            SpeechAct::Question
        );
    }
}
