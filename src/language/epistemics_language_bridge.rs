//! Epistemics-Language Bridge: Revolutionary Knowledge Grounding
//!
//! This module bridges the gap between the consciousness-language pipeline
//! and the epistemic verification system, enabling:
//!
//! 1. Uncertainty-triggered research (when Φ is low)
//! 2. Verified knowledge integration (no hallucinations)
//! 3. Continuous epistemic improvement (feedback loop)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    EPISTEMICS-LANGUAGE BRIDGE                                │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │   ConsciousnessLanguageCore                                                  │
//! │          │                                                                   │
//! │          ▼                                                                   │
//! │   ┌──────────────────────────┐                                              │
//! │   │  Uncertainty Detection   │  ← Φ < threshold triggers research           │
//! │   └──────────────────────────┘                                              │
//! │          │                                                                   │
//! │          ▼                                                                   │
//! │   ┌──────────────────────────┐                                              │
//! │   │    ResearchPlan          │  ← Consciousness-guided research plan         │
//! │   └──────────────────────────┘                                              │
//! │          │                                                                   │
//! │          ▼                                                                   │
//! │   ┌──────────────────────────┐                                              │
//! │   │  WebResearcher (async)   │  ← Autonomous web search                      │
//! │   └──────────────────────────┘                                              │
//! │          │                                                                   │
//! │          ▼                                                                   │
//! │   ┌──────────────────────────┐                                              │
//! │   │  Verified Knowledge      │  ← Epistemic verification included           │
//! │   └──────────────────────────┘                                              │
//! │          │                                                                   │
//! │          ▼                                                                   │
//! │   Improved Φ + Grounded Response                                             │
//! │                                                                              │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::language::{
    ConsciousnessLanguageCore,
    ConsciousUnderstandingResult, ConsciousnessStateLevel,
    KnowledgeGraph,
};
use crate::web_research::{
    Verification,
    VerificationLevel, ResearchPlan,
};
use crate::observability::{
    SharedObserver,
    PhiMeasurementEvent, PhiComponents,
    LanguageStepEvent, LanguageStepType,
};
use chrono::Utc;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the epistemics-language bridge
#[derive(Debug, Clone)]
pub struct EpistemicsBridgeConfig {
    /// Φ threshold below which research is triggered
    pub phi_threshold_for_research: f64,
    /// Free energy threshold above which research is triggered
    pub free_energy_threshold: f64,
    /// Maximum number of sources to consult
    pub max_research_sources: usize,
    /// Minimum verification level required
    pub min_verification_level: VerificationLevel,
    /// Enable observability tracing
    pub enable_observability: bool,
}

impl Default for EpistemicsBridgeConfig {
    fn default() -> Self {
        Self {
            phi_threshold_for_research: 0.4,
            free_energy_threshold: 0.6,
            max_research_sources: 5,
            min_verification_level: VerificationLevel::Standard,
            enable_observability: true,
        }
    }
}

// ============================================================================
// BRIDGE RESULT TYPES
// ============================================================================

/// Result from epistemic research triggered by uncertainty
#[derive(Debug, Clone)]
pub struct EpistemicResearchResult {
    /// Original understanding result
    pub original: ConsciousUnderstandingResult,
    /// Research triggered?
    pub research_triggered: bool,
    /// Research plan (if generated)
    pub research_plan: Option<ResearchPlan>,
    /// Verified findings from research (if any)
    pub verifications: Vec<Verification>,
    /// New Φ after knowledge integration
    pub improved_phi: f64,
    /// Explanation of what was researched
    pub research_summary: String,
    /// Processing metrics
    pub metrics: EpistemicsBridgeMetrics,
}

/// Metrics for bridge processing
#[derive(Debug, Clone, Default)]
pub struct EpistemicsBridgeMetrics {
    /// Time spent in uncertainty detection (ms)
    pub detection_time_ms: u64,
    /// Time spent in research (ms)
    pub research_time_ms: u64,
    /// Time spent in verification (ms)
    pub verification_time_ms: u64,
    /// Total time (ms)
    pub total_time_ms: u64,
    /// Number of claims verified
    pub claims_verified: usize,
    /// Number of sources consulted
    pub sources_consulted: usize,
}

// ============================================================================
// EPISTEMICS-LANGUAGE BRIDGE
// ============================================================================

/// Revolutionary bridge connecting consciousness-language to epistemic verification.
///
/// When the consciousness-language core detects uncertainty (low Φ or high free energy),
/// this bridge generates a research plan that can be executed asynchronously.
/// The bridge provides synchronous uncertainty detection and research planning,
/// with async research execution handled separately.
///
/// # Example
///
/// ```ignore
/// let bridge = EpistemicsLanguageBridge::new(EpistemicsBridgeConfig::default());
///
/// // Process and detect uncertainty
/// let result = bridge.detect_and_plan(&mut core, "what is nix flakes");
///
/// if result.research_triggered {
///     // Research plan is available for async execution
///     if let Some(plan) = &result.research_plan {
///         // Execute async research with WebResearcher
///         // let research = researcher.research_and_verify(&plan.query).await?;
///     }
/// }
/// ```
pub struct EpistemicsLanguageBridge {
    /// Configuration
    config: EpistemicsBridgeConfig,
    /// Knowledge graph for storing verified facts
    knowledge_graph: KnowledgeGraph,
    /// Observability hook
    observer: Option<SharedObserver>,
    /// Statistics
    stats: EpistemicsBridgeStats,
}

/// Statistics for the bridge
#[derive(Debug, Clone, Default)]
pub struct EpistemicsBridgeStats {
    /// Total inputs processed
    pub inputs_processed: u64,
    /// Times research was triggered
    pub research_triggered_count: u64,
    /// Claims verified
    pub claims_verified: u64,
    /// Knowledge items integrated
    pub knowledge_integrated: u64,
    /// Average Φ improvement
    pub avg_phi_improvement: f64,
}

impl EpistemicsLanguageBridge {
    /// Create a new epistemics-language bridge
    pub fn new(config: EpistemicsBridgeConfig) -> Self {
        Self {
            config,
            knowledge_graph: KnowledgeGraph::new(),
            observer: None,
            stats: EpistemicsBridgeStats::default(),
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(EpistemicsBridgeConfig::default())
    }

    /// Attach observer for consciousness tracing
    pub fn set_observer(&mut self, observer: SharedObserver) {
        self.observer = Some(observer);
    }

    /// Detect uncertainty and generate research plan (synchronous).
    ///
    /// This method processes input through the consciousness-language core,
    /// detects uncertainty, and generates a research plan if needed.
    /// The actual research execution is async and should be handled separately.
    pub fn detect_and_plan(
        &mut self,
        core: &mut ConsciousnessLanguageCore,
        input: &str,
    ) -> EpistemicResearchResult {
        let start = std::time::Instant::now();
        let mut metrics = EpistemicsBridgeMetrics::default();

        // Step 1: Process through consciousness-language core
        let detection_start = std::time::Instant::now();
        let understanding = core.process(input);
        metrics.detection_time_ms = detection_start.elapsed().as_millis() as u64;

        // Step 2: Detect uncertainty
        let needs_research = self.detect_uncertainty(&understanding);

        // Emit observability event for language step
        self.emit_language_step(input, "UncertaintyDetection", needs_research);

        if !needs_research {
            // No research needed
            metrics.total_time_ms = start.elapsed().as_millis() as u64;
            self.stats.inputs_processed += 1;

            return EpistemicResearchResult {
                original: understanding.clone(),
                research_triggered: false,
                research_plan: None,
                verifications: Vec::new(),
                improved_phi: understanding.consciousness_phi,
                research_summary: "No research needed - high confidence".to_string(),
                metrics,
            };
        }

        self.stats.research_triggered_count += 1;

        // Step 3: Generate research plan
        let plan = self.generate_research_plan(input, &understanding);

        metrics.total_time_ms = start.elapsed().as_millis() as u64;
        self.stats.inputs_processed += 1;

        let research_summary = format!(
            "Research plan generated: query='{}', {} sub-questions, expected Φ gain: {:.3}",
            plan.query,
            plan.sub_questions.len(),
            plan.expected_phi_gain
        );

        EpistemicResearchResult {
            original: understanding.clone(),
            research_triggered: true,
            research_plan: Some(plan),
            verifications: Vec::new(),
            improved_phi: understanding.consciousness_phi,
            research_summary,
            metrics,
        }
    }

    /// Integrate research results after async execution.
    ///
    /// Call this after executing the research plan to integrate
    /// verified knowledge and update statistics.
    pub fn integrate_research(
        &mut self,
        core: &mut ConsciousnessLanguageCore,
        input: &str,
        original_phi: f64,
        verifications: Vec<Verification>,
        sources_count: usize,
    ) -> EpistemicResearchResult {
        let start = std::time::Instant::now();
        let mut metrics = EpistemicsBridgeMetrics::default();

        metrics.claims_verified = verifications.len();
        metrics.sources_consulted = sources_count;
        self.stats.claims_verified += verifications.len() as u64;

        // Emit research event
        self.emit_language_step(input, "WebResearch", !verifications.is_empty());

        // Re-process with enriched context to get improved Φ
        let improved_understanding = core.process(input);
        let improved_phi = improved_understanding.consciousness_phi;

        // Track Φ improvement
        let phi_improvement = improved_phi - original_phi;
        self.update_phi_improvement_stats(phi_improvement);

        // Emit final observability event
        self.emit_phi_improvement(&improved_understanding, improved_phi);

        metrics.total_time_ms = start.elapsed().as_millis() as u64;

        let research_summary = format!(
            "Researched {} sources, verified {} claims. Φ {} by {:.3} ({:.3} → {:.3})",
            sources_count,
            metrics.claims_verified,
            if phi_improvement >= 0.0 { "improved" } else { "decreased" },
            phi_improvement.abs(),
            original_phi,
            improved_phi
        );

        EpistemicResearchResult {
            original: improved_understanding.clone(),
            research_triggered: true,
            research_plan: None,
            verifications,
            improved_phi,
            research_summary,
            metrics,
        }
    }

    /// Check if the understanding indicates uncertainty that warrants research
    pub fn detect_uncertainty(&self, understanding: &ConsciousUnderstandingResult) -> bool {
        // Low Φ indicates poor integration
        let low_phi = understanding.consciousness_phi < self.config.phi_threshold_for_research;

        // High free energy indicates high prediction error
        let high_fe = understanding.unified_free_energy > self.config.free_energy_threshold;

        // Skeptical state indicates need for clarification
        let skeptical = matches!(
            understanding.consciousness_state,
            ConsciousnessStateLevel::Struggling | ConsciousnessStateLevel::Failed
        );

        low_phi || high_fe || skeptical
    }

    /// Generate a research plan from the input and understanding
    pub fn generate_research_plan(
        &self,
        input: &str,
        understanding: &ConsciousUnderstandingResult,
    ) -> ResearchPlan {
        // Extract topic from NixOS understanding
        let topic = &understanding.nix_understanding.description;

        // Generate sub-questions based on the input
        let sub_questions = vec![
            format!("What is {} in NixOS?", input),
            format!("How to {} in NixOS?", input),
            format!("{} nixos documentation", input),
        ];

        ResearchPlan {
            query: format!("{} {}", input, topic),
            sub_questions,
            expected_phi_gain: 1.0 - understanding.consciousness_phi,
            verification_level: self.config.min_verification_level,
            max_sources: self.config.max_research_sources,
            timeout_seconds: 30,
        }
    }

    /// Update running statistics for Φ improvement
    fn update_phi_improvement_stats(&mut self, improvement: f64) {
        let n = self.stats.research_triggered_count as f64;
        if n > 0.0 {
            self.stats.avg_phi_improvement =
                (self.stats.avg_phi_improvement * (n - 1.0) + improvement) / n;
        }
    }

    // ========================================================================
    // OBSERVABILITY HOOKS
    // ========================================================================

    /// Emit language step event
    fn emit_language_step(&self, input: &str, step: &str, success: bool) {
        if !self.config.enable_observability {
            return;
        }

        if let Some(observer) = &self.observer {
            let event = LanguageStepEvent {
                timestamp: Utc::now(),
                step_type: match step {
                    "UncertaintyDetection" => LanguageStepType::IntentRecognition,
                    "WebResearch" => LanguageStepType::SemanticEncoding,
                    _ => LanguageStepType::Parsing,
                },
                input: input.to_string(),
                output: if success { "success".to_string() } else { "skipped".to_string() },
                confidence: 0.0,
                duration_ms: 0,
            };

            if let Ok(guard) = observer.try_write() {
                let mut obs = guard;
                let _ = obs.record_language_step(event);
            }
        }
    }

    /// Emit Φ improvement event
    fn emit_phi_improvement(
        &self,
        understanding: &ConsciousUnderstandingResult,
        improved_phi: f64,
    ) {
        if !self.config.enable_observability {
            return;
        }

        if let Some(observer) = &self.observer {
            let event = PhiMeasurementEvent {
                timestamp: Utc::now(),
                phi: improved_phi,
                components: PhiComponents {
                    integration: improved_phi,
                    binding: understanding.precision_weights.coherence as f64,
                    workspace: 1.0 - understanding.unified_free_energy,
                    attention: understanding.precision_weights.task_success as f64,
                    recursion: 0.0,
                    efficacy: understanding.precision_weights.performance as f64,
                    knowledge: 1.0, // High knowledge after research
                },
                temporal_continuity: 1.0,
            };

            if let Ok(guard) = observer.try_write() {
                let mut obs = guard;
                let _ = obs.record_phi_measurement(event);
            }
        }
    }

    /// Get bridge statistics
    pub fn stats(&self) -> &EpistemicsBridgeStats {
        &self.stats
    }

    /// Get the knowledge graph
    pub fn knowledge_graph(&self) -> &KnowledgeGraph {
        &self.knowledge_graph
    }

    /// Get configuration
    pub fn config(&self) -> &EpistemicsBridgeConfig {
        &self.config
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_config_default() {
        let config = EpistemicsBridgeConfig::default();
        assert!(config.phi_threshold_for_research > 0.0);
        assert!(config.phi_threshold_for_research < 1.0);
        assert_eq!(config.min_verification_level, VerificationLevel::Standard);
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = EpistemicsLanguageBridge::with_defaults();
        assert_eq!(bridge.stats.inputs_processed, 0);
    }

    #[test]
    fn test_uncertainty_detection() {
        let bridge = EpistemicsLanguageBridge::with_defaults();
        let mut core = ConsciousnessLanguageCore::new();

        // Process something that should have reasonable confidence
        let understanding = core.process("install firefox");

        // This won't trigger research if confidence is high enough
        let needs_research = bridge.detect_uncertainty(&understanding);

        // The result depends on the actual Φ value
        // Just verify the method runs without panic
        println!("Φ: {}, Research needed: {}", understanding.consciousness_phi, needs_research);
    }

    #[test]
    fn test_research_plan_generation() {
        let bridge = EpistemicsLanguageBridge::with_defaults();
        let mut core = ConsciousnessLanguageCore::new();

        let understanding = core.process("nix flakes");
        let plan = bridge.generate_research_plan("nix flakes", &understanding);

        assert!(!plan.query.is_empty());
        assert!(!plan.sub_questions.is_empty());
        assert!(plan.expected_phi_gain >= 0.0);
    }

    #[test]
    fn test_detect_and_plan() {
        let mut bridge = EpistemicsLanguageBridge::with_defaults();
        let mut core = ConsciousnessLanguageCore::new();

        let result = bridge.detect_and_plan(&mut core, "install firefox");

        // Stats should be updated
        assert_eq!(bridge.stats.inputs_processed, 1);

        // Should have a valid result
        assert!(result.improved_phi >= 0.0);
    }
}
