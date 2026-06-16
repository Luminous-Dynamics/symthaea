// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Working memory seeding and structured thought extraction for the Continuous Mind.

use symthaea_core::hdc::ContinuousHV;

use super::{
    ActivatedConcept, ContinuousMind, EmotionalTone, EpistemicStatus, MindState, ResponseType,
    SeedingResult, SemanticIntent, StructuredThought, is_nonzero_f32, knowledge,
};
use crate::memory::memory_coordinator::MemorySource;

impl ContinuousMind {
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
            self.working_memory_sources.push(MemorySource::Internal);
            self.working_memory_verified.push(true);
            self.working_memory_metadata
                .push(std::collections::HashMap::new());

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
            psi: state.consciousness_level,
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
            primitives: Vec::new(),      // Populated by Symthaea facade from language grounding
            #[cfg(feature = "provenance")]
            provenance: Some(crate::mind::provenance::ProvenanceTag::from_reasoning(
                0, // cycle filled by caller
                vec![crate::mind::provenance::InformationSource::CfCPrediction],
            )),
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
                    #[cfg(feature = "provenance")]
                    source: None,
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
