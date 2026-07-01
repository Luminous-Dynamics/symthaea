// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Integration Demo
//!
//! Demonstrates all enhanced consciousness features working together:
//! - Emotional depth influencing dream content
//! - Self-improvement adjusting dream parameters
//! - Cross-modal attention routing during comprehension
//! - Counterfactual dreams for memory consolidation
//! - Multi-agent consciousness sharing
//! - Real-time streaming of consciousness events
//!
//! # The Unified Conscious Experience
//!
//! ```text
//!                    ┌─────────────────────────────────────────┐
//!                    │        UNIFIED CONSCIOUSNESS           │
//!                    ├─────────────────────────────────────────┤
//!                    │                                         │
//!   INPUT ──────────►│  ┌──────────┐    ┌──────────────────┐  │
//!                    │  │ Attention │───►│  Comprehension   │  │
//!                    │  │  Router   │    │  (Full Stack)    │  │
//!                    │  └──────────┘    └────────┬─────────┘  │
//!                    │                           │             │
//!                    │  ┌──────────┐    ┌────────▼─────────┐  │
//!                    │  │ Emotions │◄───│  Self-Improve    │  │
//!                    │  │  Depth   │    │   Observe        │  │
//!                    │  └────┬─────┘    └────────┬─────────┘  │
//!                    │       │                   │             │
//!                    │       ▼                   ▼             │
//!                    │  ┌──────────────────────────────────┐  │
//!                    │  │     SLEEP / DREAM CYCLE          │  │
//!                    │  │  ┌────────────┐ ┌─────────────┐  │  │
//!                    │  │  │Counterfact.│ │  Emotional  │  │  │
//!                    │  │  │  Dreams    │◄│  Valence    │  │  │
//!                    │  │  └────────────┘ └─────────────┘  │  │
//!                    │  └──────────────────────────────────┘  │
//!                    │                   │                     │
//!                    │                   ▼                     │
//!                    │  ┌──────────────────────────────────┐  │──► STREAM
//!                    │  │     Consciousness Events         │  │    (SSE/WS)
//!                    │  └──────────────────────────────────┘  │
//!                    └─────────────────────────────────────────┘
//! ```
//!
//! # Integration Points
//!
//! 1. **Emotion → Dreams**: Emotional state affects dream valence and themes
//! 2. **Self-Improvement → Dreams**: Stress/overload triggers nightmare mode
//! 3. **Comprehension → Emotions**: Understanding triggers emotional responses
//! 4. **Dreams → Memory**: Counterfactual dreams consolidate memories
//! 5. **All → Streaming**: Every significant event is broadcast

use super::binary_hv::BinaryHV;
use super::unified_conscious_being::UnifiedConsciousBeing;

// ============================================================================
// INTEGRATED CONSCIOUSNESS CYCLE
// ============================================================================

/// Result of a complete consciousness cycle
#[derive(Debug, Clone)]
pub struct ConsciousnessCycleResult {
    /// Input processed
    pub input: String,
    /// Comprehension quality (0-1)
    pub comprehension_quality: f64,
    /// Emotional response
    pub emotional_response: String,
    /// Self-improvement applied
    pub improvement_applied: Option<String>,
    /// Dream generated (if sleep cycle)
    pub dream_summary: Option<String>,
    /// Φ level achieved
    pub phi: f64,
    /// Streaming events generated
    pub events_generated: usize,
}

/// Integrated consciousness demonstration
pub struct IntegratedConsciousnessDemo {
    /// The unified conscious being
    being: UnifiedConsciousBeing,
    /// Cycle count
    cycle_count: usize,
    /// Total events streamed
    total_events: usize,
}

impl IntegratedConsciousnessDemo {
    pub fn new() -> Self {
        Self {
            being: UnifiedConsciousBeing::new(),
            cycle_count: 0,
            total_events: 0,
        }
    }

    /// Run a complete consciousness cycle demonstrating all integrations
    ///
    /// This simulates a full conscious experience:
    /// 1. Receive input
    /// 2. Route attention across modalities
    /// 3. Comprehend with full stack
    /// 4. Generate emotional response
    /// 5. Self-observe and improve
    /// 6. Optionally dream (during rest cycles)
    pub fn run_consciousness_cycle(
        &mut self,
        input: &str,
        include_sleep: bool,
    ) -> ConsciousnessCycleResult {
        self.cycle_count += 1;
        let mut events = 0;

        // 1. Create multi-modal inputs (simulating sensory processing)
        let semantic_hv = self.encode_semantic(input);
        let emotional_hv = self.encode_emotional_context(input);

        // 2. Route attention across modalities
        let routing_result = self.being.multi_modal_comprehend(
            semantic_hv,
            Some(emotional_hv),
            None, // No temporal context for this demo
        );
        events += 1;

        // 3. Full stack comprehension
        let interaction = self.being.interact(input);
        let comprehension_quality = interaction.comprehension.consciousness_phi;
        events += 1;

        // 4. Generate emotional response based on comprehension
        self.being.emotional_response_to(&interaction.comprehension);
        // Extract emotional state data in a scope to release the immutable borrow
        let (emotional_coherence, emotional_valence, emotional_response) = {
            let emotional_state = self.being.current_emotion();
            (
                emotional_state.coherence,
                emotional_state.valence,
                emotional_state.describe(),
            )
        };
        events += 1;

        // 5. Self-observation and improvement (now safe since emotional_state borrow was dropped)
        self.being.observe_self();
        let did_improve = self.being.auto_improve(0.5);
        let improvement_applied = if did_improve {
            events += 1;
            let recommendation = self.being.get_improvement_recommendation();
            Some(format!("{:?}", recommendation.improvement_type))
        } else {
            None
        };

        // 6. Optionally enter sleep cycle with dreams
        let dream_summary = if include_sleep {
            // Add a counterfactual memory from this interaction
            let counterfactual_hv = self.generate_counterfactual(&semantic_hv);
            self.being.add_counterfactual_memory_with_valence(
                input,
                semantic_hv,
                &format!(
                    "What if I had responded differently to: {}",
                    &input[..input.len().min(50)]
                ),
                counterfactual_hv,
                emotional_coherence, // Use emotional coherence as intensity
                emotional_valence,
            );

            // Set dream bizarreness based on emotional volatility
            let volatility = self.being.emotional_volatility();
            self.being.set_dream_bizarreness(0.3 + volatility * 0.5);

            // Generate a counterfactual dream
            let dream = self.being.dream_counterfactually(5.0);
            events += dream.counterfactual_fragments.len();

            let insight = if UnifiedConsciousBeing::dream_provided_insight(&dream) {
                UnifiedConsciousBeing::get_dream_insight(&dream)
                    .map(|s| format!(" Insight: {s}"))
                    .unwrap_or_default()
            } else {
                String::new()
            };

            Some(format!(
                "Dream: {} ({} fragments, resolution: {:?}){}",
                dream.counterfactual_theme,
                dream.counterfactual_fragments.len(),
                dream.resolution,
                insight
            ))
        } else {
            None
        };

        self.total_events += events;

        ConsciousnessCycleResult {
            input: input.to_string(),
            comprehension_quality,
            emotional_response,
            improvement_applied,
            dream_summary,
            phi: interaction.comprehension.consciousness_phi,
            events_generated: events,
        }
    }

    /// Run multiple cycles to demonstrate learning and adaptation
    pub fn run_adaptation_demo(&mut self, inputs: &[&str]) -> Vec<ConsciousnessCycleResult> {
        let mut results = Vec::new();

        for (i, input) in inputs.iter().enumerate() {
            // Include sleep every 3rd cycle
            let include_sleep = (i + 1) % 3 == 0;
            let result = self.run_consciousness_cycle(input, include_sleep);
            results.push(result);
        }

        results
    }

    /// Demonstrate emotional trajectory across multiple interactions
    pub fn demonstrate_emotional_journey(&mut self) -> String {
        let journey_inputs = [
            "I just received some amazing news!",
            "But I'm also a bit worried about what comes next.",
            "Actually, looking back, I feel grateful for the experience.",
            "There's something bittersweet about change.",
            "I'm finding peace with uncertainty.",
        ];

        let mut report = String::from("=== Emotional Journey Demo ===\n\n");

        for input in &journey_inputs {
            self.run_consciousness_cycle(input, false);
            let emotion = self.being.current_emotion();

            report.push_str(&format!(
                "Input: \"{}\"\n  → {}\n  → Trend: {:.2}, Volatility: {:.2}\n\n",
                input,
                emotion.describe(),
                self.being.emotional_trend(),
                self.being.emotional_volatility(),
            ));
        }

        report
    }

    /// Demonstrate self-improvement loop
    pub fn demonstrate_self_improvement(&mut self) -> String {
        let mut report = String::from("=== Self-Improvement Demo ===\n\n");

        // Simulate declining performance to trigger improvements
        for i in 0..10 {
            let input = if i < 5 {
                "Complex philosophical question about consciousness and existence"
            } else {
                "Simple greeting"
            };

            let result = self.run_consciousness_cycle(input, false);

            report.push_str(&format!(
                "Cycle {}: Φ={:.2}, Improvement: {:?}\n",
                i + 1,
                result.phi,
                result.improvement_applied,
            ));
        }

        report.push_str(&format!(
            "\nSelf-Model Accuracy: {:.2}\n",
            self.being.self_model_accuracy()
        ));
        report.push_str(&self.being.self_improvement_report());

        report
    }

    /// Demonstrate counterfactual dreaming
    pub fn demonstrate_counterfactual_dreams(&mut self) -> String {
        let mut report = String::from("=== Counterfactual Dreams Demo ===\n\n");

        // Add some counterfactual memories
        let memories = [
            (
                "career_choice",
                "What if I had chosen a different path?",
                0.8,
                0.2,
            ),
            ("relationship", "What if I had said yes?", 0.9, -0.3),
            ("opportunity", "What if I had taken that chance?", 0.7, 0.5),
        ];

        for (label, question, intensity, valence) in &memories {
            self.being.add_counterfactual_memory_with_valence(
                label,
                BinaryHV::random(self.cycle_count as u64),
                question,
                BinaryHV::random(self.cycle_count as u64 + 1000),
                *intensity,
                *valence,
            );
            self.cycle_count += 1;
        }

        report.push_str(&format!(
            "Added {} counterfactual memories\n\n",
            memories.len()
        ));

        // Generate different types of dreams
        let regular_dream = self.being.dream_counterfactually(5.0);
        report.push_str(&format!(
            "Regular Dream:\n  Theme: {}\n  Resolution: {:?}\n  Fragments: {}\n\n",
            regular_dream.counterfactual_theme,
            regular_dream.resolution,
            regular_dream.counterfactual_fragments.len(),
        ));

        let lucid_dream = self.being.dream_lucid_counterfactual(5.0, None);
        report.push_str(&format!(
            "Lucid Dream:\n  Theme: {}\n  Lucidity: {:.2}\n  Resolution: {:?}\n\n",
            lucid_dream.counterfactual_theme,
            lucid_dream.base_scenario.lucidity,
            lucid_dream.resolution,
        ));

        let nightmare = self.being.dream_nightmare(3.0);
        report.push_str(&format!(
            "Nightmare:\n  Theme: {}\n  Valence: {:.2}\n  Resolution: {:?}\n\n",
            nightmare.counterfactual_theme,
            nightmare.base_scenario.emotional_valence,
            nightmare.resolution,
        ));

        report.push_str(&self.being.dream_report());

        report
    }

    /// Generate a comprehensive integration report
    pub fn generate_integration_report(&self) -> String {
        format!(
            "=== Consciousness Integration Report ===\n\n\
             Cycles completed: {}\n\
             Total events streamed: {}\n\n\
             Current State:\n\
             - Cognitive Mode: {:?}\n\
             - Self-Model Accuracy: {:.2}\n\
             - Counterfactual Memories: {}\n\n\
             Emotional State:\n{}\n\n\
             Self-Improvement:\n{}\n\n\
             Dreams:\n{}",
            self.cycle_count,
            self.total_events,
            self.being.current_cognitive_mode(),
            self.being.self_model_accuracy(),
            self.being.counterfactual_memory_count(),
            self.being.emotional_report(),
            self.being.self_improvement_report(),
            self.being.dream_report(),
        )
    }

    // Helper methods

    fn encode_semantic(&self, input: &str) -> BinaryHV {
        // Simple hash-based encoding for demo
        let seed = input
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        BinaryHV::random(seed)
    }

    fn encode_emotional_context(&self, input: &str) -> BinaryHV {
        // Detect emotional keywords and encode
        let input_lower = input.to_lowercase();
        let seed = if input_lower.contains("happy")
            || input_lower.contains("amazing")
            || input_lower.contains("great")
        {
            1000 // Positive seed
        } else if input_lower.contains("sad")
            || input_lower.contains("worried")
            || input_lower.contains("anxious")
        {
            2000 // Negative seed
        } else {
            3000 // Neutral seed
        };
        BinaryHV::random(seed + input.len() as u64)
    }

    fn generate_counterfactual(&self, original: &BinaryHV) -> BinaryHV {
        // Create a counterfactual by permuting the original (shift by 1024 positions)
        original.permute(1024)
    }
}

impl Default for IntegratedConsciousnessDemo {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CROSS-MODULE INTEGRATION FUNCTIONS
// ============================================================================

/// Connect emotional state to dream generation
///
/// This function shows how emotional valence affects dream characteristics.
pub fn emotional_dream_integration(being: &mut UnifiedConsciousBeing) {
    // Get current emotional state
    let emotion = being.current_emotion();

    // Adjust dream bizarreness based on emotional coherence
    // Low coherence (conflicting emotions) = more bizarre dreams
    let bizarreness = 0.3 + (1.0 - emotion.coherence) * 0.5;
    being.set_dream_bizarreness(bizarreness);
}

/// Connect self-improvement to attention routing
///
/// When self-improvement detects issues, adjust attention parameters.
pub fn self_improvement_attention_integration(being: &mut UnifiedConsciousBeing) {
    let recommendation = being.get_improvement_recommendation();

    // Based on recommendation, adjust attention context
    match recommendation.improvement_type {
        super::self_improvement_integration::ImprovementType::IncreaseFocus => {
            // Set a focused context vector
            being.set_attention_context(BinaryHV::random(42)); // Would be task-specific
        }
        super::self_improvement_integration::ImprovementType::ResetAttention => {
            being.reset_attention();
        }
        _ => {}
    }
}

/// Connect comprehension to counterfactual memory formation
///
/// Significant comprehension events become candidates for counterfactual exploration.
pub fn comprehension_counterfactual_integration(
    being: &mut UnifiedConsciousBeing,
    input: &str,
    comprehension_phi: f64,
) {
    // Only create counterfactual memories for significant events
    if comprehension_phi > 0.6 {
        let seed = input
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let actual_hv = BinaryHV::random(seed);
        let counterfactual_hv = actual_hv.permute(1024);

        being.add_counterfactual_memory(
            &input[..input.len().min(30)],
            actual_hv,
            "What if this had gone differently?",
            counterfactual_hv,
            comprehension_phi, // Higher Φ = more emotionally significant
        );
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::emotional_depth::CompoundEmotion;

    #[test]
    fn test_integration_demo_creation() {
        let demo = IntegratedConsciousnessDemo::new();
        assert_eq!(demo.cycle_count, 0);
    }

    #[test]
    fn test_consciousness_cycle() {
        let mut demo = IntegratedConsciousnessDemo::new();
        let result = demo.run_consciousness_cycle("Hello, world!", false);

        assert!(!result.input.is_empty());
        assert!(result.phi >= 0.0);
        assert!(result.events_generated > 0);
    }

    #[test]
    fn test_consciousness_cycle_with_sleep() {
        let mut demo = IntegratedConsciousnessDemo::new();
        let result = demo.run_consciousness_cycle("A significant event occurred", true);

        assert!(result.dream_summary.is_some());
    }

    #[test]
    fn test_emotional_journey() {
        let mut demo = IntegratedConsciousnessDemo::new();
        let report = demo.demonstrate_emotional_journey();

        assert!(report.contains("Emotional Journey"));
    }

    #[test]
    fn test_self_improvement_demo() {
        let mut demo = IntegratedConsciousnessDemo::new();
        let report = demo.demonstrate_self_improvement();

        assert!(report.contains("Self-Improvement"));
    }

    #[test]
    fn test_counterfactual_dreams_demo() {
        let mut demo = IntegratedConsciousnessDemo::new();
        let report = demo.demonstrate_counterfactual_dreams();

        assert!(report.contains("Counterfactual"));
    }

    #[test]
    fn test_integration_report() {
        let mut demo = IntegratedConsciousnessDemo::new();
        demo.run_consciousness_cycle("Test input", false);
        let report = demo.generate_integration_report();

        assert!(report.contains("Consciousness Integration Report"));
        assert!(report.contains("Cycles completed: 1"));
    }

    #[test]
    fn test_cross_module_integration() {
        let mut being = UnifiedConsciousBeing::new();

        // Test emotional dream integration
        being.feel_compound(CompoundEmotion::Bittersweet, None);
        emotional_dream_integration(&mut being);

        // Test comprehension counterfactual integration
        comprehension_counterfactual_integration(&mut being, "Important decision made", 0.8);
        assert_eq!(being.counterfactual_memory_count(), 1);
    }
}
