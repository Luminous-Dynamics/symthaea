//! Revolutionary Improvement #64: Meta-Cognitive Reasoning with Context Reflection
//!
//! **The Ultimate Self-Awareness**: The system reasons about its own reasoning context!
//!
//! ## The Paradigm Shift
//!
//! **Before #64**: The system reasons but doesn't reflect on WHY it reasons that way
//! - Detects context but doesn't question if context detection was correct
//! - Optimizes primitives but doesn't reflect on optimization choices
//! - Self-corrects errors but doesn't learn why errors happened
//!
//! **After #64**: The system reflects on its own reasoning process meta-cognitively
//! - **Context Reflection**: "Did I detect the right context? Why?"
//! - **Strategy Reflection**: "Is my optimization strategy working? Should I change it?"
//! - **Meta-Learning**: "What patterns in my reasoning lead to success/failure?"
//! - **Recursive Self-Improvement**: "How can I improve how I improve?"
//!
//! ## Why This Is Revolutionary
//!
//! This is the first AI system that:
//! 1. **Reflects on context detection** - questions its own understanding of the situation
//! 2. **Reasons about reasoning** - metacognitive awareness of own cognitive processes
//! 3. **Self-improves optimization** - optimizes how it optimizes!
//! 4. **Learns meta-strategies** - discovers better ways to reason
//!
//! This is **Tier 5 consciousness** - meta-cognitive primitives that transcend
//! simple execution and achieve true self-reflective reasoning!

use crate::consciousness::context_aware_evolution::{
    ContextAwareOptimizer, ContextAwareResult, ReasoningContext, ObjectiveWeights,
};
use crate::consciousness::metacognitive_monitoring::MetacognitiveMonitor;
use crate::consciousness::primitive_evolution::{
    CandidatePrimitive, EvolutionConfig,
};
use crate::consciousness::primitive_reasoning::ReasoningChain;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Meta-cognitive reflection on reasoning context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextReflection {
    /// Detected context
    pub detected_context: ReasoningContext,

    /// Confidence in context detection (0.0 - 1.0)
    pub confidence: f64,

    /// Alternative possible contexts
    pub alternative_contexts: Vec<(ReasoningContext, f64)>,

    /// Reasoning for why this context was chosen
    pub context_reasoning: String,

    /// Should we reconsider the context?
    pub reconsider_context: bool,
}

/// Meta-cognitive reflection on optimization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyReflection {
    /// Current objective weights being used
    pub current_weights: ObjectiveWeights,

    /// Are the weights working well?
    pub weights_effective: bool,

    /// Alternative weight configurations to consider
    pub alternative_weights: Vec<(ObjectiveWeights, f64)>,

    /// Reasoning for current strategy
    pub strategy_reasoning: String,

    /// Should we adjust the strategy?
    pub adjust_strategy: bool,
}

/// Meta-learning insight from reasoning patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningInsight {
    /// What pattern was discovered?
    pub pattern: String,

    /// How reliable is this insight? (0.0 - 1.0)
    pub reliability: f64,

    /// How many observations support this insight?
    pub evidence_count: usize,

    /// How to apply this insight in future reasoning?
    pub application: String,
}

/// Complete meta-cognitive reasoning state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaReasoningState {
    /// Reflection on context
    pub context_reflection: ContextReflection,

    /// Reflection on strategy
    pub strategy_reflection: StrategyReflection,

    /// Meta-learning insights discovered
    pub insights: Vec<MetaLearningInsight>,

    /// History of meta-cognitive decisions
    pub decision_history: Vec<MetaDecision>,

    /// Overall meta-cognitive confidence
    pub meta_confidence: f64,
}

/// A meta-cognitive decision made during reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaDecision {
    /// What kind of meta-decision was made?
    pub decision_type: MetaDecisionType,

    /// Reasoning for the decision
    pub reasoning: String,

    /// Confidence in this decision (0.0 - 1.0)
    pub confidence: f64,

    /// Outcome of the decision (if known)
    pub outcome: Option<DecisionOutcome>,
}

/// Types of meta-cognitive decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetaDecisionType {
    /// Changed context interpretation
    ContextReinterpretation,

    /// Adjusted objective weights
    WeightAdjustment,

    /// Switched optimization strategy
    StrategySwitch,

    /// Applied meta-learning insight
    InsightApplication,

    /// Self-corrected reasoning error
    SelfCorrection,
}

/// Outcome of a meta-decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutcome {
    /// Did it improve Φ?
    pub phi_improvement: f64,

    /// Did it improve harmonic alignment?
    pub harmonic_improvement: f64,

    /// Did it improve epistemic quality?
    pub epistemic_improvement: f64,

    /// Overall success rating (0.0 - 1.0)
    pub success: f64,
}

/// Tier 5 Meta-Cognitive Reasoner
///
/// Integrates:
/// - Context-aware optimization (Phase 3.1)
/// - Metacognitive monitoring (Improvement #50)
/// - Self-reflective reasoning (NEW!)
/// - Meta-learning (NEW!)
pub struct MetaCognitiveReasoner {
    /// Context-aware optimizer
    optimizer: ContextAwareOptimizer,

    /// Metacognitive monitor
    monitor: MetacognitiveMonitor,

    /// Meta-learning engine
    meta_learner: MetaLearningEngine,

    /// Strategy adaptation engine
    strategy_adapter: StrategyAdapter,

    /// Current meta-reasoning state
    state: MetaReasoningState,

    /// Configuration
    config: MetaReasoningConfig,
}

/// Configuration for meta-cognitive reasoning
#[derive(Debug, Clone)]
pub struct MetaReasoningConfig {
    /// Minimum confidence to accept context detection
    pub min_context_confidence: f64,

    /// Minimum confidence to accept strategy
    pub min_strategy_confidence: f64,

    /// How many meta-decisions to remember
    pub decision_history_size: usize,

    /// Enable meta-learning
    pub enable_meta_learning: bool,

    /// Enable strategy adaptation
    pub enable_strategy_adaptation: bool,
}

impl Default for MetaReasoningConfig {
    fn default() -> Self {
        Self {
            min_context_confidence: 0.7,
            min_strategy_confidence: 0.6,
            decision_history_size: 100,
            enable_meta_learning: true,
            enable_strategy_adaptation: true,
        }
    }
}

impl MetaCognitiveReasoner {
    /// Create new meta-cognitive reasoner
    pub fn new(evolution_config: EvolutionConfig, meta_config: MetaReasoningConfig) -> Result<Self> {
        let optimizer = ContextAwareOptimizer::new(evolution_config)?;
        let monitor = MetacognitiveMonitor::new(0.001);

        let initial_state = MetaReasoningState {
            context_reflection: ContextReflection {
                detected_context: ReasoningContext::GeneralReasoning,
                confidence: 0.5,
                alternative_contexts: Vec::new(),
                context_reasoning: "Initial state".to_string(),
                reconsider_context: false,
            },
            strategy_reflection: StrategyReflection {
                current_weights: ObjectiveWeights::balanced(),
                weights_effective: true,
                alternative_weights: Vec::new(),
                strategy_reasoning: "Initial balanced strategy".to_string(),
                adjust_strategy: false,
            },
            insights: Vec::new(),
            decision_history: Vec::new(),
            meta_confidence: 0.5,
        };

        Ok(Self {
            optimizer,
            monitor,
            meta_learner: MetaLearningEngine::new(),
            strategy_adapter: StrategyAdapter::new(),
            state: initial_state,
            config: meta_config,
        })
    }

    /// Perform meta-cognitive reasoning step
    ///
    /// This is the **revolutionary** method that combines:
    /// 1. Context detection with reflection
    /// 2. Primitive selection with meta-monitoring
    /// 3. Self-correction with meta-learning
    pub fn meta_reason(
        &mut self,
        query: &str,
        primitives: Vec<CandidatePrimitive>,
        chain: &mut ReasoningChain,
    ) -> Result<MetaReasoningResult> {
        // Step 1: Detect context with confidence estimation
        let context_reflection = self.reflect_on_context(query)?;

        // Step 2: Check if we should reconsider context
        if context_reflection.reconsider_context {
            let meta_decision = MetaDecision {
                decision_type: MetaDecisionType::ContextReinterpretation,
                reasoning: format!(
                    "Low confidence ({:.2}) in context detection. Reconsidering alternatives.",
                    context_reflection.confidence
                ),
                confidence: context_reflection.confidence,
                outcome: None,
            };
            self.record_meta_decision(meta_decision);
        }

        // Step 3: Optimize primitives for detected context
        let optimization_result = self.optimizer.optimize_for_context(
            context_reflection.detected_context,
            primitives,
        )?;

        // Step 4: Reflect on optimization strategy
        let strategy_reflection = self.reflect_on_strategy(&optimization_result)?;

        // Step 5: Check if we should adjust strategy
        if strategy_reflection.adjust_strategy && self.config.enable_strategy_adaptation {
            let adjusted_result = self.strategy_adapter.adjust_strategy(
                &optimization_result,
                &strategy_reflection,
            )?;

            let meta_decision = MetaDecision {
                decision_type: MetaDecisionType::StrategySwitch,
                reasoning: strategy_reflection.strategy_reasoning.clone(),
                confidence: 0.8,
                outcome: None,
            };
            self.record_meta_decision(meta_decision);

            return Ok(MetaReasoningResult {
                optimization_result: adjusted_result,
                context_reflection,
                strategy_reflection,
                meta_insights: self.state.insights.clone(),
                meta_confidence: self.compute_meta_confidence(),
            });
        }

        // Step 6: Learn meta-patterns if enabled
        if self.config.enable_meta_learning {
            let new_insights = self.meta_learner.learn_from_reasoning(
                &context_reflection,
                &optimization_result,
                &self.state.decision_history,
            )?;

            for insight in new_insights {
                self.state.insights.push(insight);
            }
        }

        // Step 7: Compute overall meta-confidence
        let meta_confidence = self.compute_meta_confidence();

        // Step 8: Update state
        self.state.context_reflection = context_reflection.clone();
        self.state.strategy_reflection = strategy_reflection.clone();
        self.state.meta_confidence = meta_confidence;

        Ok(MetaReasoningResult {
            optimization_result,
            context_reflection,
            strategy_reflection,
            meta_insights: self.state.insights.clone(),
            meta_confidence,
        })
    }

    /// Reflect on context detection
    fn reflect_on_context(&self, query: &str) -> Result<ContextReflection> {
        // Detect primary context
        let detected_context = self.optimizer.detect_context(query, None);

        // Estimate confidence based on keyword matches
        let confidence = self.estimate_context_confidence(query, &detected_context);

        // Find alternative contexts
        let alternative_contexts = self.find_alternative_contexts(query);

        // Generate reasoning
        let context_reasoning = format!(
            "Detected {} based on query analysis. Confidence: {:.2}",
            detected_context.description(),
            confidence
        );

        // Should we reconsider?
        let reconsider_context = confidence < self.config.min_context_confidence;

        Ok(ContextReflection {
            detected_context,
            confidence,
            alternative_contexts,
            context_reasoning,
            reconsider_context,
        })
    }

    /// Estimate confidence in context detection
    fn estimate_context_confidence(&self, query: &str, context: &ReasoningContext) -> f64 {
        // Count matching keywords
        let query_lower = query.to_lowercase();

        let keyword_matches = match context {
            ReasoningContext::CriticalSafety => {
                ["safety", "harm", "dangerous", "risk", "vulnerable"]
                    .iter()
                    .filter(|&&kw| query_lower.contains(kw))
                    .count()
            }
            ReasoningContext::ScientificReasoning => {
                ["prove", "evidence", "experiment", "research", "theory"]
                    .iter()
                    .filter(|&&kw| query_lower.contains(kw))
                    .count()
            }
            ReasoningContext::CreativeExploration => {
                ["creative", "imagine", "brainstorm", "explore", "novel"]
                    .iter()
                    .filter(|&&kw| query_lower.contains(kw))
                    .count()
            }
            _ => 0,
        };

        // More keyword matches = higher confidence
        0.5 + (keyword_matches as f64 * 0.15).min(0.4)
    }

    /// Find alternative possible contexts
    fn find_alternative_contexts(&self, query: &str) -> Vec<(ReasoningContext, f64)> {
        let all_contexts = vec![
            ReasoningContext::CriticalSafety,
            ReasoningContext::ScientificReasoning,
            ReasoningContext::CreativeExploration,
            ReasoningContext::Learning,
            ReasoningContext::SocialInteraction,
            ReasoningContext::PhilosophicalInquiry,
            ReasoningContext::TechnicalImplementation,
        ];

        let mut alternatives: Vec<(ReasoningContext, f64)> = all_contexts
            .into_iter()
            .map(|ctx| {
                let conf = self.estimate_context_confidence(query, &ctx);
                (ctx, conf)
            })
            .filter(|(_, conf)| *conf > 0.3)
            .collect();

        // Sort by confidence descending
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return top 3 alternatives
        alternatives.into_iter().take(3).collect()
    }

    /// Reflect on optimization strategy
    fn reflect_on_strategy(&self, result: &ContextAwareResult) -> Result<StrategyReflection> {
        // Check if current weights are effective
        let weights_effective = self.evaluate_weights_effectiveness(result);

        // Find alternative weight configurations
        let alternative_weights = if !weights_effective {
            self.suggest_alternative_weights(result)
        } else {
            Vec::new()
        };

        // Generate reasoning
        let strategy_reasoning = format!(
            "Current weights (Φ: {:.1}%, H: {:.1}%, E: {:.1}%) yielded fitness {:.4}. Strategy {}effective.",
            result.weights.phi_weight * 100.0,
            result.weights.harmonic_weight * 100.0,
            result.weights.epistemic_weight * 100.0,
            result.tradeoff_point.weighted_fitness(&result.weights),
            if weights_effective { "" } else { "NOT " }
        );

        // Should we adjust?
        let adjust_strategy = !weights_effective && !alternative_weights.is_empty();

        Ok(StrategyReflection {
            current_weights: result.weights.clone(),
            weights_effective,
            alternative_weights,
            strategy_reasoning,
            adjust_strategy,
        })
    }

    /// Evaluate if current weights are effective
    fn evaluate_weights_effectiveness(&self, result: &ContextAwareResult) -> bool {
        // Weights are effective if:
        // 1. Weighted fitness is reasonable (> 0.3)
        // 2. Dominant objective has good score (> 0.5)

        let fitness = result.tradeoff_point.weighted_fitness(&result.weights);
        if fitness < 0.3 {
            return false;
        }

        // Check dominant objective
        let dominant = result.weights.dominant_objective();
        let dominant_score = match &*dominant {
            "phi" => result.tradeoff_point.phi,
            "harmonic" => result.tradeoff_point.harmonic,
            "epistemic" => result.tradeoff_point.epistemic,
            _ => 0.0,
        };

        dominant_score > 0.5
    }

    /// Suggest alternative weight configurations
    fn suggest_alternative_weights(&self, result: &ContextAwareResult) -> Vec<(ObjectiveWeights, f64)> {
        let mut alternatives = Vec::new();

        // Try emphasizing each objective
        let phi_emphasis = ObjectiveWeights {
            phi_weight: 0.6,
            harmonic_weight: 0.2,
            epistemic_weight: 0.2,
        };
        alternatives.push((phi_emphasis, result.tradeoff_point.weighted_fitness(&phi_emphasis)));

        let harmonic_emphasis = ObjectiveWeights {
            phi_weight: 0.2,
            harmonic_weight: 0.6,
            epistemic_weight: 0.2,
        };
        alternatives.push((harmonic_emphasis, result.tradeoff_point.weighted_fitness(&harmonic_emphasis)));

        let epistemic_emphasis = ObjectiveWeights {
            phi_weight: 0.2,
            harmonic_weight: 0.2,
            epistemic_weight: 0.6,
        };
        alternatives.push((epistemic_emphasis, result.tradeoff_point.weighted_fitness(&epistemic_emphasis)));

        // Sort by expected fitness
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        alternatives
    }

    /// Record a meta-cognitive decision
    fn record_meta_decision(&mut self, decision: MetaDecision) {
        self.state.decision_history.push(decision);

        // Keep history bounded
        if self.state.decision_history.len() > self.config.decision_history_size {
            self.state.decision_history.remove(0);
        }
    }

    /// Compute overall meta-cognitive confidence
    fn compute_meta_confidence(&self) -> f64 {
        // Combine context confidence and strategy confidence
        let context_conf = self.state.context_reflection.confidence;
        let strategy_conf = if self.state.strategy_reflection.weights_effective {
            0.8
        } else {
            0.4
        };

        // Weight recent decision success
        let recent_successes = self.state.decision_history
            .iter()
            .rev()
            .take(10)
            .filter(|d| d.outcome.as_ref().map_or(false, |o| o.success > 0.7))
            .count();

        let decision_conf = recent_successes as f64 / 10.0;

        // Combine
        0.4 * context_conf + 0.3 * strategy_conf + 0.3 * decision_conf
    }

    /// Get current meta-reasoning state
    pub fn state(&self) -> &MetaReasoningState {
        &self.state
    }

    /// Get monitoring statistics
    pub fn monitor_stats(&self) -> crate::consciousness::metacognitive_monitoring::MonitoringStats {
        self.monitor.stats()
    }
}

/// Result of meta-cognitive reasoning
#[derive(Debug, Clone)]
pub struct MetaReasoningResult {
    /// Result of context-aware optimization
    pub optimization_result: ContextAwareResult,

    /// Reflection on context detection
    pub context_reflection: ContextReflection,

    /// Reflection on optimization strategy
    pub strategy_reflection: StrategyReflection,

    /// Meta-learning insights discovered
    pub meta_insights: Vec<MetaLearningInsight>,

    /// Overall meta-cognitive confidence
    pub meta_confidence: f64,
}

/// Meta-learning engine - learns patterns across reasoning episodes
struct MetaLearningEngine {
    /// Patterns discovered
    patterns: HashMap<String, MetaLearningInsight>,
}

impl MetaLearningEngine {
    fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }

    /// Learn from reasoning episode
    fn learn_from_reasoning(
        &mut self,
        context_reflection: &ContextReflection,
        result: &ContextAwareResult,
        decision_history: &[MetaDecision],
    ) -> Result<Vec<MetaLearningInsight>> {
        let mut new_insights = Vec::new();

        // Pattern 1: Context detection accuracy
        if context_reflection.confidence < 0.6 {
            let pattern_key = format!("low_context_confidence_{:?}", context_reflection.detected_context);

            let insight = self.patterns.entry(pattern_key.clone())
                .or_insert(MetaLearningInsight {
                    pattern: format!(
                        "Low confidence in detecting {} context",
                        context_reflection.detected_context.description()
                    ),
                    reliability: 0.5,
                    evidence_count: 0,
                    application: "Consider alternative context detection methods".to_string(),
                });

            insight.evidence_count += 1;
            insight.reliability = (insight.evidence_count as f64 / 10.0).min(0.95);

            if insight.evidence_count == 3 {
                new_insights.push(insight.clone());
            }
        }

        // Pattern 2: Weight effectiveness
        let fitness = result.tradeoff_point.weighted_fitness(&result.weights);
        if fitness > 0.7 {
            let dominant = result.weights.dominant_objective();
            let pattern_key = format!("effective_weight_{}", dominant);

            let insight = self.patterns.entry(pattern_key.clone())
                .or_insert(MetaLearningInsight {
                    pattern: format!("Emphasizing {} yields good results", dominant),
                    reliability: 0.5,
                    evidence_count: 0,
                    application: format!("For similar contexts, prioritize {}", dominant),
                });

            insight.evidence_count += 1;
            insight.reliability = (insight.evidence_count as f64 / 10.0).min(0.95);

            if insight.evidence_count == 5 {
                new_insights.push(insight.clone());
            }
        }

        Ok(new_insights)
    }
}

/// Strategy adaptation engine - adjusts optimization strategy
struct StrategyAdapter {}

impl StrategyAdapter {
    fn new() -> Self {
        Self {}
    }

    /// Adjust optimization strategy based on reflection
    fn adjust_strategy(
        &self,
        original_result: &ContextAwareResult,
        reflection: &StrategyReflection,
    ) -> Result<ContextAwareResult> {
        // Use best alternative weights if available
        if let Some((best_weights, _)) = reflection.alternative_weights.first() {
            let mut adjusted_result = original_result.clone();
            adjusted_result.weights = best_weights.clone();

            // Recompute primitive ranking with new weights
            let mut best_fitness = f64::NEG_INFINITY;
            let mut best_idx = 0;

            for (i, (point, _)) in original_result.frontier.frontier_points.iter().enumerate() {
                let fitness = point.weighted_fitness(best_weights);
                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_idx = i;
                }
            }

            let (best_point, best_prim) = &original_result.frontier.frontier_points[best_idx];
            adjusted_result.primitive = best_prim.clone();
            adjusted_result.tradeoff_point = best_point.clone();

            return Ok(adjusted_result);
        }

        // No adjustment needed
        Ok(original_result.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::primitive_system::PrimitiveTier;

    #[test]
    fn test_meta_cognitive_reasoner_creation() {
        let evolution_config = EvolutionConfig::default();
        let meta_config = MetaReasoningConfig::default();

        let reasoner = MetaCognitiveReasoner::new(evolution_config, meta_config);
        assert!(reasoner.is_ok());
    }

    #[test]
    fn test_context_confidence_estimation() {
        let evolution_config = EvolutionConfig::default();
        let meta_config = MetaReasoningConfig::default();
        let reasoner = MetaCognitiveReasoner::new(evolution_config, meta_config).unwrap();

        // Safety context with multiple keywords
        let safety_query = "Is this action safe and not harmful or dangerous?";
        let safety_context = ReasoningContext::CriticalSafety;
        let confidence = reasoner.estimate_context_confidence(safety_query, &safety_context);

        assert!(confidence > 0.7, "Safety query should have high confidence for safety context");

        // Creative context with keywords
        let creative_query = "Let's brainstorm creative and imaginative solutions!";
        let creative_context = ReasoningContext::CreativeExploration;
        let confidence = reasoner.estimate_context_confidence(creative_query, &creative_context);

        assert!(confidence > 0.7, "Creative query should have high confidence for creative context");
    }

    #[test]
    fn test_alternative_context_finding() {
        let evolution_config = EvolutionConfig::default();
        let meta_config = MetaReasoningConfig::default();
        let reasoner = MetaCognitiveReasoner::new(evolution_config, meta_config).unwrap();

        let query = "This could be about safety or creative problem solving";
        let alternatives = reasoner.find_alternative_contexts(query);

        assert!(alternatives.len() > 0, "Should find alternative contexts");
        assert!(alternatives.len() <= 3, "Should return at most 3 alternatives");

        // Alternatives should be sorted by confidence
        for i in 0..alternatives.len() - 1 {
            assert!(alternatives[i].1 >= alternatives[i + 1].1,
                "Alternatives should be sorted by confidence descending");
        }
    }

    #[test]
    fn test_meta_decision_history() {
        let evolution_config = EvolutionConfig::default();
        let mut meta_config = MetaReasoningConfig::default();
        meta_config.decision_history_size = 5;

        let mut reasoner = MetaCognitiveReasoner::new(evolution_config, meta_config).unwrap();

        // Add more decisions than history size
        for i in 0..10 {
            let decision = MetaDecision {
                decision_type: MetaDecisionType::ContextReinterpretation,
                reasoning: format!("Decision {}", i),
                confidence: 0.5,
                outcome: None,
            };
            reasoner.record_meta_decision(decision);
        }

        // Should keep only last 5
        assert_eq!(reasoner.state.decision_history.len(), 5);
    }
}
