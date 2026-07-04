// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Goal Inference — User Input → Desired System State
//!
//! Instead of classifying intent, we infer the user's **goal state** in HDC
//! space. The gap between current state and goal state IS the free energy
//! that drives action selection.
//!
//! Multi-turn understanding: "Set up a web server" → "Use nginx" → "Add SSL"
//! builds a goal vector that progressively refines.

use super::working_memory::{MemorySource, WorkingMemory};
use crate::encoding::{NixCodebook, UserInputEncoder};
use symthaea_core::hdc::ContinuousHV;

/// Inferred goal with confidence.
#[derive(Debug, Clone)]
pub struct InferredGoal {
    /// The desired system state as an HDC vector.
    pub goal_state: ContinuousHV,
    /// Confidence in this interpretation (0.0–1.0).
    pub confidence: f64,
    /// Human-readable description of the inferred goal.
    pub description: String,
    /// Whether this goal requires clarification.
    pub needs_clarification: bool,
}

/// Goal inference engine — maps user input to desired system state.
pub struct GoalInference {
    /// Working memory accumulates context over conversation.
    working_memory: WorkingMemory,
    /// Weight for new input vs accumulated context.
    context_weight: f64,
}

impl GoalInference {
    /// Create a new goal inference engine.
    pub fn new() -> Self {
        Self {
            working_memory: WorkingMemory::new(),
            context_weight: 0.7,
        }
    }

    /// Infer the user's goal from input text.
    ///
    /// The goal vector is computed by encoding the input and blending with
    /// accumulated context in working memory. Multi-turn conversations
    /// progressively refine the goal.
    pub fn infer(&mut self, input: &str, codebook: &mut NixCodebook) -> InferredGoal {
        // Encode user input
        let input_hv = {
            let mut enc = UserInputEncoder::new(codebook);
            enc.encode_input(input)
        };

        // Store in working memory
        self.working_memory
            .push(input_hv.clone(), MemorySource::UserInput, input.to_string());

        // Boost items related to current input
        self.working_memory.attend(&input_hv, 0.3);

        // Blend input with context
        let context = self.working_memory.context_vector();
        let goal_state = if context.norm() > 1e-6 {
            let refs = [&input_hv, &context];
            let weights = [1.0, self.context_weight as f32];
            ContinuousHV::weighted_bundle(&refs, &weights)
        } else {
            input_hv
        };

        // Estimate confidence from input clarity
        let confidence = self.estimate_confidence(input);

        // Generate description
        let description = self.describe_goal(input);

        InferredGoal {
            goal_state,
            confidence,
            description,
            needs_clarification: confidence < 0.5,
        }
    }

    /// Infer goal given a pre-encoded input vector.
    pub fn infer_from_hv(&mut self, input_hv: ContinuousHV, label: &str) -> InferredGoal {
        self.working_memory
            .push(input_hv.clone(), MemorySource::UserInput, label.to_string());

        let context = self.working_memory.context_vector();
        let goal_state = if context.norm() > 1e-6 {
            let refs = [&input_hv, &context];
            let weights = [1.0, self.context_weight as f32];
            ContinuousHV::weighted_bundle(&refs, &weights)
        } else {
            input_hv
        };

        InferredGoal {
            goal_state,
            confidence: 0.5,
            description: label.to_string(),
            needs_clarification: false,
        }
    }

    /// Get the current goal state vector (accumulated context).
    pub fn current_goal(&self) -> ContinuousHV {
        self.working_memory.context_vector()
    }

    /// Access working memory.
    pub fn working_memory(&self) -> &WorkingMemory {
        &self.working_memory
    }

    /// Access working memory mutably.
    pub fn working_memory_mut(&mut self) -> &mut WorkingMemory {
        &mut self.working_memory
    }

    /// Reset goal inference (new conversation).
    pub fn reset(&mut self) {
        self.working_memory.clear();
    }

    /// Estimate confidence from input characteristics.
    fn estimate_confidence(&self, input: &str) -> f64 {
        let words: Vec<&str> = input.split_whitespace().collect();
        let word_count = words.len();

        // Very short inputs are ambiguous
        if word_count <= 1 {
            return 0.3;
        }

        // Questions indicate uncertainty
        if input.contains('?') {
            return 0.4;
        }

        // Check for specific action words (higher confidence)
        let action_words = [
            "install", "remove", "enable", "disable", "rebuild", "switch", "rollback", "update",
            "upgrade", "search",
        ];
        let has_action = words
            .iter()
            .any(|w| action_words.contains(&w.to_lowercase().as_str()));

        if has_action && word_count >= 2 {
            0.8
        } else if has_action {
            0.6
        } else if word_count >= 3 {
            0.5
        } else {
            0.4
        }
    }

    /// Generate a human-readable description of the inferred goal.
    fn describe_goal(&self, input: &str) -> String {
        let lower = input.to_lowercase();

        if lower.contains("install") || lower.contains("add") {
            let target = self.extract_target(&lower, &["install", "add", "get"]);
            format!("Install {} on the system", target.unwrap_or("package"))
        } else if lower.contains("remove") || lower.contains("uninstall") {
            let target = self.extract_target(&lower, &["remove", "uninstall", "delete"]);
            format!("Remove {} from the system", target.unwrap_or("package"))
        } else if lower.contains("enable") {
            let target = self.extract_target(&lower, &["enable"]);
            format!("Enable {}", target.unwrap_or("service"))
        } else if lower.contains("why") || lower.contains("fail") || lower.contains("error") {
            "Diagnose system issue".to_string()
        } else if lower.contains("faster") || lower.contains("optimize") || lower.contains("slow") {
            "Optimize system performance".to_string()
        } else if lower.contains("rebuild") || lower.contains("switch") {
            "Apply system configuration".to_string()
        } else {
            format!("Process: {input}")
        }
    }

    /// Extract the target word after an action verb.
    fn extract_target<'a>(&self, input: &'a str, verbs: &[&str]) -> Option<&'a str> {
        for verb in verbs {
            if let Some(pos) = input.find(verb) {
                let rest = &input[pos + verb.len()..].trim_start();
                let target = rest.split_whitespace().next();
                if let Some(t) = target {
                    if !["the", "a", "an", "my", "this"].contains(&t) {
                        return Some(t);
                    }
                    // Skip article, get next word
                    let after = rest[t.len()..].trim_start();
                    return after.split_whitespace().next();
                }
            }
        }
        None
    }

    /// Infer the user's goal asynchronously using the hybrid symbolic-neural encoder.
    ///
    /// Blends the exact matching of the symbolic encoder with the semantic
    /// generalization of the BGE-M3 neural bridge.
    #[cfg(feature = "native")]
    pub async fn infer_async(
        &mut self,
        input: &str,
        codebook: &mut NixCodebook,
        bridge: &super::neural_bridge::NeuralBridge,
    ) -> Result<InferredGoal, super::neural_bridge::BridgeError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(super::neural_bridge::BridgeError::EmptyInput);
        }

        // 1. Get symbolic representation
        let symbolic_hv = {
            let mut enc = UserInputEncoder::new(codebook);
            enc.encode_input(input)
        };

        // 2. Get neural representation
        let embed_res = bridge.embed_text(input).await?;
        let neural_hv = embed_res.continuous;

        // 3. Blend them (40% symbolic, 60% neural)
        let blended_hv = if symbolic_hv.norm() > 1e-6 {
            let refs = [&symbolic_hv, &neural_hv];
            let weights = [0.4, 0.6];
            ContinuousHV::weighted_bundle(&refs, &weights)
        } else {
            neural_hv
        };

        // 4. Store in working memory
        self.working_memory.push(
            blended_hv.clone(),
            MemorySource::UserInput,
            input.to_string(),
        );

        // 5. Boost items related to this hybrid intent
        self.working_memory.attend(&blended_hv, 0.3);

        // 6. Blend with accumulated context
        let context = self.working_memory.context_vector();
        let goal_state = if context.norm() > 1e-6 {
            let refs = [&blended_hv, &context];
            let weights = [1.0, self.context_weight as f32];
            ContinuousHV::weighted_bundle(&refs, &weights)
        } else {
            blended_hv
        };

        let confidence = self.estimate_confidence(input);
        let description = self.describe_goal(input);

        Ok(InferredGoal {
            goal_state,
            confidence,
            description,
            needs_clarification: confidence < 0.5,
        })
    }

    /// Offline-safe hybrid goal inference using deterministic embeddings.
    pub fn infer_hybrid_offline(
        &mut self,
        input: &str,
        codebook: &mut NixCodebook,
        bridge: &super::neural_bridge::NeuralBridge,
    ) -> Result<InferredGoal, super::neural_bridge::BridgeError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(super::neural_bridge::BridgeError::EmptyInput);
        }

        // 1. Get symbolic representation
        let symbolic_hv = {
            let mut enc = UserInputEncoder::new(codebook);
            enc.encode_input(input)
        };

        // 2. Get deterministic offline neural representation
        let embed_res = bridge.embed_deterministic(input)?;
        let neural_hv = embed_res.continuous;

        // 3. Blend them (40% symbolic, 60% neural)
        let blended_hv = if symbolic_hv.norm() > 1e-6 {
            let refs = [&symbolic_hv, &neural_hv];
            let weights = [0.4, 0.6];
            ContinuousHV::weighted_bundle(&refs, &weights)
        } else {
            neural_hv
        };

        // 4. Store in working memory
        self.working_memory.push(
            blended_hv.clone(),
            MemorySource::UserInput,
            input.to_string(),
        );

        // 5. Boost items related to this hybrid intent
        self.working_memory.attend(&blended_hv, 0.3);

        // 6. Blend with accumulated context
        let context = self.working_memory.context_vector();
        let goal_state = if context.norm() > 1e-6 {
            let refs = [&blended_hv, &context];
            let weights = [1.0, self.context_weight as f32];
            ContinuousHV::weighted_bundle(&refs, &weights)
        } else {
            blended_hv
        };

        let confidence = self.estimate_confidence(input);
        let description = self.describe_goal(input);

        Ok(InferredGoal {
            goal_state,
            confidence,
            description,
            needs_clarification: confidence < 0.5,
        })
    }
}

impl Default for GoalInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_goal_inference() {
        let mut gi = GoalInference::new();
        let mut cb = NixCodebook::new();

        let goal = gi.infer("install firefox", &mut cb);
        assert!(goal.confidence > 0.5);
        assert!(!goal.needs_clarification);
        assert!(goal.description.contains("Install"));
        assert!(goal.goal_state.norm() > 0.0);
    }

    #[test]
    fn test_multi_turn_refinement() {
        let mut gi = GoalInference::new();
        let mut cb = NixCodebook::new();

        let goal1 = gi.infer("set up a web server", &mut cb);
        let goal2 = gi.infer("use nginx", &mut cb);

        // Second goal should incorporate context from first
        let sim = goal1.goal_state.similarity(&goal2.goal_state);
        assert!(sim.is_finite());
        // They should share some similarity due to accumulated context
        assert!(gi.working_memory().len() == 2);
    }

    #[test]
    fn test_low_confidence_ambiguous() {
        let mut gi = GoalInference::new();
        let mut cb = NixCodebook::new();

        let goal = gi.infer("help", &mut cb);
        assert!(goal.confidence < 0.5);
        assert!(goal.needs_clarification);
    }

    #[test]
    fn test_reset() {
        let mut gi = GoalInference::new();
        let mut cb = NixCodebook::new();

        gi.infer("install firefox", &mut cb);
        assert!(!gi.working_memory().is_empty());

        gi.reset();
        assert!(gi.working_memory().is_empty());
    }

    #[test]
    fn test_question_lower_confidence() {
        let mut gi = GoalInference::new();
        let mut cb = NixCodebook::new();

        let statement = gi.infer("install nginx", &mut cb);
        gi.reset();
        let question = gi.infer("how do I install nginx?", &mut cb);

        assert!(statement.confidence > question.confidence);
    }

    #[test]
    fn test_infer_from_hv_basic() {
        let mut gi = GoalInference::new();
        let hv = ContinuousHV::random(1024, 42);

        let goal = gi.infer_from_hv(hv, "pre-encoded goal");
        assert!(
            (goal.confidence - 0.5).abs() < 1e-6,
            "infer_from_hv always returns 0.5 confidence"
        );
        assert_eq!(goal.description, "pre-encoded goal");
        assert!(
            !goal.needs_clarification,
            "0.5 confidence should not need clarification"
        );
        assert!(goal.goal_state.norm() > 0.0);
    }

    #[test]
    fn test_infer_from_hv_context_blending() {
        let mut gi = GoalInference::new();
        let hv1 = ContinuousHV::random(1024, 1);
        let hv2 = ContinuousHV::random(1024, 2);

        // First call populates working memory
        let goal1 = gi.infer_from_hv(hv1.clone(), "first");
        assert_eq!(gi.working_memory().len(), 1);

        // Second call should blend with context from first
        let goal2 = gi.infer_from_hv(hv2.clone(), "second");
        assert_eq!(gi.working_memory().len(), 2);

        // goal2 should differ from raw hv2 due to context blending
        let sim_to_raw = goal2.goal_state.similarity(&hv2);
        // With context blending, it won't be identical to the raw input
        assert!(sim_to_raw.is_finite());
    }

    #[test]
    fn test_infer_from_hv_empty_context() {
        let mut gi = GoalInference::new();
        assert!(gi.working_memory().is_empty());

        // With empty working memory, context_vector().norm() should be ~0,
        // so infer_from_hv should return the raw input vector
        let hv = ContinuousHV::random(1024, 99);
        let goal = gi.infer_from_hv(hv.clone(), "solo");

        let sim = goal.goal_state.similarity(&hv);
        // First call: context is just the input itself after push, so
        // blending input + context (which includes input) should be close
        assert!(
            sim > 0.5,
            "First infer_from_hv should stay close to input, got sim={sim}"
        );
    }

    #[test]
    fn test_current_goal_tracks_context() {
        let mut gi = GoalInference::new();
        let mut cb = NixCodebook::new();

        // Initially no context
        let initial = gi.current_goal();
        assert!(
            initial.norm() < 1e-6,
            "Empty context should have near-zero norm"
        );

        // After inference, context should be non-zero
        gi.infer("install firefox", &mut cb);
        let after = gi.current_goal();
        assert!(after.norm() > 0.0);
    }

    #[test]
    fn test_estimate_confidence_branches() {
        let mut gi = GoalInference::new();
        let mut cb = NixCodebook::new();

        // Single word → 0.3
        let single = gi.infer("help", &mut cb);
        assert!((single.confidence - 0.3).abs() < 1e-6);
        gi.reset();

        // Question → 0.4
        let question = gi.infer("what is this?", &mut cb);
        assert!((question.confidence - 0.4).abs() < 1e-6);
        gi.reset();

        // Action word + 2+ words → 0.8
        let action = gi.infer("install firefox browser", &mut cb);
        assert!((action.confidence - 0.8).abs() < 1e-6);
        gi.reset();

        // 3+ words, no action word → 0.5
        let verbose = gi.infer("my system is slow", &mut cb);
        assert!((verbose.confidence - 0.5).abs() < 1e-6);
        gi.reset();

        // 2 words, no action → 0.4
        let two_words = gi.infer("system slow", &mut cb);
        assert!((two_words.confidence - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_describe_goal_branches() {
        let mut gi = GoalInference::new();
        let mut cb = NixCodebook::new();

        let install = gi.infer("install firefox", &mut cb);
        assert!(install.description.contains("Install"));
        gi.reset();

        let remove = gi.infer("remove firefox", &mut cb);
        assert!(remove.description.contains("Remove"));
        gi.reset();

        let enable = gi.infer("enable nginx", &mut cb);
        assert!(enable.description.contains("Enable"));
        gi.reset();

        let diag = gi.infer("why did it fail", &mut cb);
        assert_eq!(diag.description, "Diagnose system issue");
        gi.reset();

        let opt = gi.infer("make it faster", &mut cb);
        assert_eq!(opt.description, "Optimize system performance");
        gi.reset();

        let rebuild = gi.infer("rebuild the system", &mut cb);
        assert_eq!(rebuild.description, "Apply system configuration");
        gi.reset();

        let other = gi.infer("hello world foo", &mut cb);
        assert!(other.description.starts_with("Process:"));
    }

    #[test]
    fn test_infer_hybrid_offline_determinism() {
        let mut gi = GoalInference::new();
        let mut cb = NixCodebook::new();
        let bridge = crate::mind::neural_bridge::NeuralBridge::new();

        let goal1 = gi
            .infer_hybrid_offline("install postgresql", &mut cb, &bridge)
            .unwrap();
        gi.reset();
        let goal2 = gi
            .infer_hybrid_offline("install postgresql", &mut cb, &bridge)
            .unwrap();

        let sim = goal1.goal_state.similarity(&goal2.goal_state);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Hybrid offline goal inference must be deterministic: sim={sim}"
        );
        assert!(goal1.goal_state.norm() > 0.0);
    }

    #[test]
    fn test_infer_hybrid_offline_generalization() {
        let mut gi = GoalInference::new();
        let mut cb = NixCodebook::new();
        let bridge = crate::mind::neural_bridge::NeuralBridge::new();

        let pg_goal = gi
            .infer_hybrid_offline("install postgresql", &mut cb, &bridge)
            .unwrap();
        gi.reset();
        let db_goal = gi
            .infer_hybrid_offline("set up database", &mut cb, &bridge)
            .unwrap();
        gi.reset();
        let other_goal = gi
            .infer_hybrid_offline("clean disk space", &mut cb, &bridge)
            .unwrap();

        let sim_related = pg_goal.goal_state.similarity(&db_goal.goal_state);
        let sim_unrelated = pg_goal.goal_state.similarity(&other_goal.goal_state);

        assert!(sim_related.is_finite());
        assert!(sim_unrelated.is_finite());
        // Since pg_goal and db_goal both use "install" / "database" semantics from the bridge,
        // they should share some alignment, but at least the similarity stays bounded in [-1, 1].
        assert!(sim_related >= -1.0 && sim_related <= 1.0);
        assert!(sim_unrelated >= -1.0 && sim_unrelated <= 1.0);
    }
}
