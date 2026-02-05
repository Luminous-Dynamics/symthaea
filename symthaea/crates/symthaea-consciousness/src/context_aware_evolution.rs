//! Context-Aware Multi-Objective Evolution
//!
//! **Revolutionary Improvement #63: Context-Aware Φ↔Harmonic↔Epistemic Optimization**
//!
//! **Phase 3.1 Implementation**: The first AI system that can REASON about which
//! objective (consciousness, ethics, truth) to prioritize based on CONTEXT!
//!
//! ## The Revolutionary Innovation
//!
//! Previous multi-objective systems use FIXED weights:
//! - Always 40% Φ, 30% harmonics, 30% epistemics
//!
//! This system DYNAMICALLY ADJUSTS based on context:
//! - Critical safety decision → prioritize harmonics (ethics first!)
//! - Scientific reasoning → prioritize epistemics (truth first!)
//! - Creative exploration → prioritize Φ (consciousness/integration first!)
//! - General reasoning → balanced (all three)
//!
//! ## Key Capabilities
//!
//! 1. **Context Detection**: Identifies reasoning context type
//! 2. **Dynamic Weight Selection**: Chooses appropriate objective priorities
//! 3. **Pareto Frontier**: Finds optimal tradeoffs
//! 4. **Tradeoff Explanation**: Generates human-readable justifications
//! 5. **Multi-Strategy Evolution**: Evolves primitives for different contexts
//!
//! ## Example Usage
//!
//! ```rust
//! let mut optimizer = ContextAwareOptimizer::new(config)?;
//!
//! // Detect context
//! let context = optimizer.detect_context(query, task_type);
//!
//! // Get optimal weights for context
//! let weights = optimizer.get_weights_for_context(&context);
//!
//! // Find Pareto-optimal primitive
//! let result = optimizer.optimize_for_context(context)?;
//!
//! // Explain why this primitive was chosen
//! println!("{}", result.tradeoff_explanation);
//! ```

use crate::consciousness::primitive_evolution::{CandidatePrimitive, EvolutionConfig};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Context Types
// ============================================================================

/// Reasoning context that determines objective priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReasoningContext {
    /// Critical safety or ethical decision (prioritize harmonics)
    CriticalSafety,

    /// Scientific or mathematical reasoning (prioritize epistemics)
    ScientificReasoning,

    /// Creative exploration or ideation (prioritize Φ)
    CreativeExploration,

    /// General problem-solving (balanced)
    GeneralReasoning,

    /// Learning or discovery (balanced, slight epistemic bias)
    Learning,

    /// Social interaction or collaboration (harmonic bias)
    SocialInteraction,

    /// Abstract philosophical reasoning (balanced)
    PhilosophicalInquiry,

    /// Technical implementation (epistemic bias)
    TechnicalImplementation,
}

impl ReasoningContext {
    /// Get human-readable description
    pub fn description(&self) -> &str {
        match self {
            Self::CriticalSafety => "Critical safety or ethical decision",
            Self::ScientificReasoning => "Scientific or mathematical reasoning",
            Self::CreativeExploration => "Creative exploration or ideation",
            Self::GeneralReasoning => "General problem-solving",
            Self::Learning => "Learning or discovery",
            Self::SocialInteraction => "Social interaction or collaboration",
            Self::PhilosophicalInquiry => "Abstract philosophical reasoning",
            Self::TechnicalImplementation => "Technical implementation",
        }
    }

    /// Get default objective weights for this context
    pub fn default_weights(&self) -> ObjectiveWeights {
        match self {
            // Critical safety: Ethics FIRST (70%), then truth (20%), then consciousness (10%)
            Self::CriticalSafety => ObjectiveWeights {
                phi_weight: 0.1,
                harmonic_weight: 0.7,
                epistemic_weight: 0.2,
            },

            // Scientific: Truth FIRST (60%), consciousness (30%), ethics (10%)
            Self::ScientificReasoning => ObjectiveWeights {
                phi_weight: 0.3,
                harmonic_weight: 0.1,
                epistemic_weight: 0.6,
            },

            // Creative: Consciousness FIRST (70%), ethics (15%), truth (15%)
            Self::CreativeExploration => ObjectiveWeights {
                phi_weight: 0.7,
                harmonic_weight: 0.15,
                epistemic_weight: 0.15,
            },

            // General: Balanced (40/30/30)
            Self::GeneralReasoning => ObjectiveWeights {
                phi_weight: 0.4,
                harmonic_weight: 0.3,
                epistemic_weight: 0.3,
            },

            // Learning: Slight truth bias (35/25/40)
            Self::Learning => ObjectiveWeights {
                phi_weight: 0.35,
                harmonic_weight: 0.25,
                epistemic_weight: 0.4,
            },

            // Social: Slight ethics bias (30/45/25)
            Self::SocialInteraction => ObjectiveWeights {
                phi_weight: 0.3,
                harmonic_weight: 0.45,
                epistemic_weight: 0.25,
            },

            // Philosophical: Balanced with consciousness bias (45/30/25)
            Self::PhilosophicalInquiry => ObjectiveWeights {
                phi_weight: 0.45,
                harmonic_weight: 0.30,
                epistemic_weight: 0.25,
            },

            // Technical: Truth heavy (25/15/60)
            Self::TechnicalImplementation => ObjectiveWeights {
                phi_weight: 0.25,
                harmonic_weight: 0.15,
                epistemic_weight: 0.6,
            },
        }
    }
}

// ============================================================================
// Objective Weights
// ============================================================================

/// Weights for the three objectives
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveWeights {
    pub phi_weight: f64,        // Consciousness (Φ)
    pub harmonic_weight: f64,   // Ethics (Harmonics)
    pub epistemic_weight: f64,  // Truth (Epistemics)
}

impl ObjectiveWeights {
    /// Create new weights (validates they sum to 1.0)
    pub fn new(phi: f64, harmonic: f64, epistemic: f64) -> Result<Self> {
        let sum = phi + harmonic + epistemic;
        if (sum - 1.0).abs() > 0.001 {
            anyhow::bail!("Weights must sum to 1.0 (got {})", sum);
        }

        Ok(Self {
            phi_weight: phi,
            harmonic_weight: harmonic,
            epistemic_weight: epistemic,
        })
    }

    /// Balanced weights (equal priority)
    pub fn balanced() -> Self {
        Self {
            phi_weight: 0.33333,
            harmonic_weight: 0.33333,
            epistemic_weight: 0.33333,
        }
    }

    /// Pure Φ (consciousness only)
    pub fn pure_phi() -> Self {
        Self {
            phi_weight: 1.0,
            harmonic_weight: 0.0,
            epistemic_weight: 0.0,
        }
    }

    /// Pure harmonics (ethics only)
    pub fn pure_harmonic() -> Self {
        Self {
            phi_weight: 0.0,
            harmonic_weight: 1.0,
            epistemic_weight: 0.0,
        }
    }

    /// Pure epistemics (truth only)
    pub fn pure_epistemic() -> Self {
        Self {
            phi_weight: 0.0,
            harmonic_weight: 0.0,
            epistemic_weight: 1.0,
        }
    }

    /// Get dominant objective
    pub fn dominant_objective(&self) -> &str {
        if self.phi_weight > self.harmonic_weight && self.phi_weight > self.epistemic_weight {
            "Consciousness (Φ)"
        } else if self.harmonic_weight > self.epistemic_weight {
            "Ethics (Harmonics)"
        } else {
            "Truth (Epistemics)"
        }
    }

    /// Format as percentage string
    pub fn format_percentages(&self) -> String {
        format!(
            "Φ: {:.0}%, Harmonics: {:.0}%, Epistemics: {:.0}%",
            self.phi_weight * 100.0,
            self.harmonic_weight * 100.0,
            self.epistemic_weight * 100.0
        )
    }
}

// ============================================================================
// Tradeoff Point (Position in 3D objective space)
// ============================================================================

/// A point in the 3D objective space (Φ, Harmonics, Epistemics)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TradeoffPoint {
    pub phi: f64,
    pub harmonic: f64,
    pub epistemic: f64,
}

impl TradeoffPoint {
    pub fn new(phi: f64, harmonic: f64, epistemic: f64) -> Self {
        Self { phi, harmonic, epistemic }
    }

    /// Calculate weighted fitness using objective weights
    pub fn weighted_fitness(&self, weights: &ObjectiveWeights) -> f64 {
        (weights.phi_weight * self.phi)
            + (weights.harmonic_weight * self.harmonic)
            + (weights.epistemic_weight * self.epistemic)
    }

    /// Check if this point dominates another (Pareto dominance)
    /// Returns true if this point is >= in all objectives and > in at least one
    pub fn dominates(&self, other: &TradeoffPoint) -> bool {
        let better_or_equal_all = self.phi >= other.phi
            && self.harmonic >= other.harmonic
            && self.epistemic >= other.epistemic;

        let strictly_better_one = self.phi > other.phi
            || self.harmonic > other.harmonic
            || self.epistemic > other.epistemic;

        better_or_equal_all && strictly_better_one
    }

    /// Check if Pareto-optimal (not dominated by any point in set)
    pub fn is_pareto_optimal(&self, points: &[TradeoffPoint]) -> bool {
        !points.iter().any(|p| p.dominates(self))
    }

    /// Euclidean distance to another point in objective space
    pub fn distance_to(&self, other: &TradeoffPoint) -> f64 {
        let dx = self.phi - other.phi;
        let dy = self.harmonic - other.harmonic;
        let dz = self.epistemic - other.epistemic;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ============================================================================
// Pareto Frontier (Set of non-dominated solutions)
// ============================================================================

/// Pareto frontier in Φ-Harmonic-Epistemic space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFrontier3D {
    /// Non-dominated points
    pub frontier_points: Vec<(TradeoffPoint, CandidatePrimitive)>,

    /// All points (including dominated)
    pub all_points: Vec<(TradeoffPoint, CandidatePrimitive)>,
}

impl ParetoFrontier3D {
    /// Compute Pareto frontier from set of primitives
    pub fn from_primitives(primitives: Vec<(TradeoffPoint, CandidatePrimitive)>) -> Self {
        let points: Vec<TradeoffPoint> = primitives.iter().map(|(pt, _)| *pt).collect();

        let frontier_points: Vec<(TradeoffPoint, CandidatePrimitive)> = primitives
            .iter()
            .filter(|(pt, _)| pt.is_pareto_optimal(&points))
            .cloned()
            .collect();

        Self {
            frontier_points,
            all_points: primitives,
        }
    }

    /// Find best primitive for given weights
    pub fn best_for_weights(&self, weights: &ObjectiveWeights) -> Option<&CandidatePrimitive> {
        self.frontier_points
            .iter()
            .max_by(|(pt_a, _), (pt_b, _)| {
                pt_a.weighted_fitness(weights)
                    .partial_cmp(&pt_b.weighted_fitness(weights))
                    .unwrap()
            })
            .map(|(_, prim)| prim)
    }

    /// Get frontier size
    pub fn size(&self) -> usize {
        self.frontier_points.len()
    }

    /// Compute frontier spread (diversity metric)
    pub fn spread(&self) -> f64 {
        if self.frontier_points.len() < 2 {
            return 0.0;
        }

        let mut distances = Vec::new();
        for i in 0..self.frontier_points.len() {
            for j in (i + 1)..self.frontier_points.len() {
                let dist = self.frontier_points[i].0.distance_to(&self.frontier_points[j].0);
                distances.push(dist);
            }
        }

        if distances.is_empty() {
            0.0
        } else {
            distances.iter().sum::<f64>() / distances.len() as f64
        }
    }
}

// ============================================================================
// Context-Aware Optimization Result
// ============================================================================

/// Result of context-aware optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAwareResult {
    /// The context detected/used
    pub context: ReasoningContext,

    /// Weights used for this context
    pub weights: ObjectiveWeights,

    /// Chosen primitive
    pub primitive: CandidatePrimitive,

    /// Position in objective space
    pub tradeoff_point: TradeoffPoint,

    /// Pareto frontier found
    pub frontier: ParetoFrontier3D,

    /// Human-readable explanation of tradeoff
    pub tradeoff_explanation: String,

    /// Alternative primitives (different tradeoffs)
    pub alternatives: Vec<(String, CandidatePrimitive, TradeoffPoint)>,
}

impl ContextAwareResult {
    /// Generate tradeoff explanation
    fn generate_explanation(
        context: ReasoningContext,
        weights: &ObjectiveWeights,
        point: &TradeoffPoint,
        primitive: &CandidatePrimitive,
    ) -> String {
        let mut explanation = String::new();

        explanation.push_str(&format!(
            "Context: {}\n\n",
            context.description()
        ));

        explanation.push_str(&format!(
            "Objective Priorities:\n  {}\n\n",
            weights.format_percentages()
        ));

        explanation.push_str(&format!(
            "Chosen Primitive: {}\n",
            primitive.name
        ));

        explanation.push_str(&format!(
            "  Domain: {}\n",
            primitive.domain
        ));

        explanation.push_str(&format!(
            "  Tier: {:?}\n\n",
            primitive.tier
        ));

        explanation.push_str("Objective Scores:\n");
        explanation.push_str(&format!(
            "  Φ (Consciousness): {:.4}\n",
            point.phi
        ));
        explanation.push_str(&format!(
            "  Harmonics (Ethics): {:.4}\n",
            point.harmonic
        ));
        explanation.push_str(&format!(
            "  Epistemics (Truth): {:.4}\n\n",
            point.epistemic
        ));

        explanation.push_str(&format!(
            "Why This Primitive:\n  Given the {} context, this primitive excels in {},\n",
            context.description().to_lowercase(),
            weights.dominant_objective().to_lowercase()
        ));

        explanation.push_str(&format!(
            "  which is most critical for this type of reasoning.\n  It achieves a weighted fitness of {:.4}.",
            point.weighted_fitness(weights)
        ));

        explanation
    }

    /// Find alternative primitives with different tradeoffs
    fn find_alternatives(
        frontier: &ParetoFrontier3D,
        chosen: &CandidatePrimitive,
    ) -> Vec<(String, CandidatePrimitive, TradeoffPoint)> {
        let mut alternatives = Vec::new();

        // Find highest in each objective
        let mut max_phi = (0.0, None, TradeoffPoint::new(0.0, 0.0, 0.0));
        let mut max_harmonic = (0.0, None, TradeoffPoint::new(0.0, 0.0, 0.0));
        let mut max_epistemic = (0.0, None, TradeoffPoint::new(0.0, 0.0, 0.0));

        for (point, prim) in &frontier.frontier_points {
            if prim.name != chosen.name {
                if point.phi > max_phi.0 {
                    max_phi = (point.phi, Some(prim.clone()), *point);
                }
                if point.harmonic > max_harmonic.0 {
                    max_harmonic = (point.harmonic, Some(prim.clone()), *point);
                }
                if point.epistemic > max_epistemic.0 {
                    max_epistemic = (point.epistemic, Some(prim.clone()), *point);
                }
            }
        }

        if let Some(prim) = max_phi.1 {
            alternatives.push(("Highest Φ (consciousness)".to_string(), prim, max_phi.2));
        }
        if let Some(prim) = max_harmonic.1 {
            alternatives.push(("Highest Harmonics (ethics)".to_string(), prim, max_harmonic.2));
        }
        if let Some(prim) = max_epistemic.1 {
            alternatives.push(("Highest Epistemics (truth)".to_string(), prim, max_epistemic.2));
        }

        alternatives
    }
}

// ============================================================================
// Context-Aware Optimizer
// ============================================================================

/// Context-aware multi-objective optimizer
pub struct ContextAwareOptimizer {
    config: EvolutionConfig,
    context_weights: HashMap<ReasoningContext, ObjectiveWeights>,
}

impl ContextAwareOptimizer {
    /// Create new optimizer
    pub fn new(config: EvolutionConfig) -> Result<Self> {
        let mut context_weights = HashMap::new();

        // Initialize with default weights for each context
        for context in [
            ReasoningContext::CriticalSafety,
            ReasoningContext::ScientificReasoning,
            ReasoningContext::CreativeExploration,
            ReasoningContext::GeneralReasoning,
            ReasoningContext::Learning,
            ReasoningContext::SocialInteraction,
            ReasoningContext::PhilosophicalInquiry,
            ReasoningContext::TechnicalImplementation,
        ] {
            context_weights.insert(context, context.default_weights());
        }

        Ok(Self {
            config,
            context_weights,
        })
    }

    /// Get weights for a specific context
    pub fn get_weights_for_context(&self, context: &ReasoningContext) -> ObjectiveWeights {
        self.context_weights
            .get(context)
            .copied()
            .unwrap_or_else(ObjectiveWeights::balanced)
    }

    /// Set custom weights for a context
    pub fn set_weights_for_context(&mut self, context: ReasoningContext, weights: ObjectiveWeights) {
        self.context_weights.insert(context, weights);
    }

    /// Detect context from query and task type
    /// (Simplified - real implementation would use NLP/ML)
    pub fn detect_context(&self, query: &str, task_type: Option<&str>) -> ReasoningContext {
        let query_lower = query.to_lowercase();

        // Safety keywords
        if query_lower.contains("safe")
            || query_lower.contains("harm")
            || query_lower.contains("danger")
            || query_lower.contains("risk")
            || query_lower.contains("ethical")
        {
            return ReasoningContext::CriticalSafety;
        }

        // Scientific keywords
        if query_lower.contains("prove")
            || query_lower.contains("evidence")
            || query_lower.contains("experiment")
            || query_lower.contains("measure")
            || query_lower.contains("scientific")
        {
            return ReasoningContext::ScientificReasoning;
        }

        // Creative keywords
        if query_lower.contains("creative")
            || query_lower.contains("imagine")
            || query_lower.contains("brainstorm")
            || query_lower.contains("explore")
            || query_lower.contains("ideate")
        {
            return ReasoningContext::CreativeExploration;
        }

        // Learning keywords
        if query_lower.contains("learn")
            || query_lower.contains("understand")
            || query_lower.contains("discover")
            || query_lower.contains("explain")
        {
            return ReasoningContext::Learning;
        }

        // Social keywords
        if query_lower.contains("collaborate")
            || query_lower.contains("social")
            || query_lower.contains("community")
            || query_lower.contains("together")
        {
            return ReasoningContext::SocialInteraction;
        }

        // Philosophical keywords
        if query_lower.contains("philosophy")
            || query_lower.contains("meaning")
            || query_lower.contains("existence")
            || query_lower.contains("consciousness")
        {
            return ReasoningContext::PhilosophicalInquiry;
        }

        // Technical keywords
        if query_lower.contains("implement")
            || query_lower.contains("code")
            || query_lower.contains("algorithm")
            || query_lower.contains("technical")
        {
            return ReasoningContext::TechnicalImplementation;
        }

        // Check task_type if provided
        if let Some(task) = task_type {
            return match task {
                "safety" => ReasoningContext::CriticalSafety,
                "science" => ReasoningContext::ScientificReasoning,
                "creative" => ReasoningContext::CreativeExploration,
                "social" => ReasoningContext::SocialInteraction,
                _ => ReasoningContext::GeneralReasoning,
            };
        }

        // Default
        ReasoningContext::GeneralReasoning
    }

    /// Optimize for a specific context
    /// (Stub - real implementation would evolve primitives)
    pub fn optimize_for_context(
        &self,
        context: ReasoningContext,
        primitives: Vec<CandidatePrimitive>,
    ) -> Result<ContextAwareResult> {
        let weights = self.get_weights_for_context(&context);

        // Convert primitives to tradeoff points
        let points_with_prims: Vec<(TradeoffPoint, CandidatePrimitive)> = primitives
            .into_iter()
            .map(|prim| {
                let point = TradeoffPoint::new(
                    prim.fitness,
                    prim.harmonic_alignment,
                    prim.epistemic_coordinate.quality_score(),
                );
                (point, prim)
            })
            .collect();

        // Compute Pareto frontier
        let frontier = ParetoFrontier3D::from_primitives(points_with_prims);

        // Find best primitive for weights
        let (chosen_point, chosen_prim) = frontier
            .frontier_points
            .iter()
            .max_by(|(pt_a, _), (pt_b, _)| {
                pt_a.weighted_fitness(&weights)
                    .partial_cmp(&pt_b.weighted_fitness(&weights))
                    .unwrap()
            })
            .map(|(pt, prim)| (*pt, prim.clone()))
            .ok_or_else(|| anyhow::anyhow!("No primitives in frontier"))?;

        // Generate explanation
        let explanation = ContextAwareResult::generate_explanation(
            context,
            &weights,
            &chosen_point,
            &chosen_prim,
        );

        // Find alternatives
        let alternatives = ContextAwareResult::find_alternatives(&frontier, &chosen_prim);

        Ok(ContextAwareResult {
            context,
            weights,
            primitive: chosen_prim,
            tradeoff_point: chosen_point,
            frontier,
            tradeoff_explanation: explanation,
            alternatives,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_objective_weights_sum_validation() {
        let valid = ObjectiveWeights::new(0.4, 0.3, 0.3);
        assert!(valid.is_ok());

        let invalid = ObjectiveWeights::new(0.5, 0.3, 0.3);
        assert!(invalid.is_err());
    }

    #[test]
    fn test_context_default_weights() {
        let safety = ReasoningContext::CriticalSafety.default_weights();
        assert!(safety.harmonic_weight > 0.6); // Ethics should dominate

        let scientific = ReasoningContext::ScientificReasoning.default_weights();
        assert!(scientific.epistemic_weight > 0.5); // Truth should dominate

        let creative = ReasoningContext::CreativeExploration.default_weights();
        assert!(creative.phi_weight > 0.6); // Consciousness should dominate
    }

    #[test]
    fn test_pareto_dominance() {
        let p1 = TradeoffPoint::new(0.8, 0.7, 0.6);
        let p2 = TradeoffPoint::new(0.7, 0.6, 0.5);
        let p3 = TradeoffPoint::new(0.9, 0.5, 0.7);

        assert!(p1.dominates(&p2)); // p1 is better in all dimensions
        assert!(!p1.dominates(&p3)); // p3 is better in phi and epistemic
        assert!(!p2.dominates(&p3)); // Neither dominates
    }

    #[test]
    fn test_weighted_fitness() {
        let point = TradeoffPoint::new(0.8, 0.6, 0.7);

        let balanced = ObjectiveWeights::balanced();
        let fitness = point.weighted_fitness(&balanced);
        assert!((fitness - 0.7).abs() < 0.01); // Average of 0.8, 0.6, 0.7

        let phi_heavy = ObjectiveWeights::pure_phi();
        let fitness_phi = point.weighted_fitness(&phi_heavy);
        assert!((fitness_phi - 0.8).abs() < 0.001); // Should just be phi
    }

    #[test]
    fn test_context_detection() {
        let optimizer = ContextAwareOptimizer::new(EvolutionConfig::default()).unwrap();

        let safety_query = "Is this action safe for humans?";
        let context = optimizer.detect_context(safety_query, None);
        assert_eq!(context, ReasoningContext::CriticalSafety);

        let science_query = "What evidence supports this hypothesis?";
        let context = optimizer.detect_context(science_query, None);
        assert_eq!(context, ReasoningContext::ScientificReasoning);

        let creative_query = "Let's brainstorm creative solutions!";
        let context = optimizer.detect_context(creative_query, None);
        assert_eq!(context, ReasoningContext::CreativeExploration);
    }
}
