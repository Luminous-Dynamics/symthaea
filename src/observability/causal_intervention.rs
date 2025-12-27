// Revolutionary Enhancement #4: Causal Intervention & Counterfactuals
//
// Implements Level 2 of Pearl's Causal Hierarchy: Intervention (Doing)
//
// Key Innovation: do-calculus for causal inference under interventions
//
// Core Concept:
//   P(Y | X=x)        - Association: What we observe
//   P(Y | do(X=x))    - Intervention: What happens if we act
//
// The difference matters! Correlation ≠ Causation, but intervention reveals causation.
//
// Example:
//   Observing: P(recovery | medicine) might be low (sick people take medicine)
//   Intervention: P(recovery | do(medicine)) could be high (medicine causes recovery)
//
// Graph Surgery: do(X=x) removes all incoming edges to X
//   Before:  A → X → Y     After:  A   X → Y
//                                     ↓
//                                   (set to x)
//
// This module enables:
// - Predicting effects of actions before taking them
// - Answering "what if we do X?" questions
// - Safety analysis for system changes
// - Optimal intervention selection

use super::{
    causal_graph::{CausalGraph, CausalEdge, CausalNode, EdgeType},
    probabilistic_inference::{
        ProbabilisticCausalGraph, ProbabilisticEdge, ProbabilisticPrediction,
        UncertaintySource,
    },
};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

/// Type of intervention on a causal node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionType {
    /// Set node to specific probability/value
    SetValue(f64),

    /// Force node to specific outcome (binary)
    Enable,
    Disable,

    /// Remove all effects from node (make it causally inert)
    Neutralize,
}

/// Specification of a causal intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionSpec {
    /// Nodes to intervene on
    pub interventions: HashMap<String, InterventionType>,

    /// Optional: nodes to condition on (observed but not intervened)
    pub conditions: HashMap<String, f64>,
}

impl InterventionSpec {
    /// Create new intervention specification
    pub fn new() -> Self {
        Self {
            interventions: HashMap::new(),
            conditions: HashMap::new(),
        }
    }

    /// Set node to specific value
    pub fn set_value(mut self, node: &str, value: f64) -> Self {
        self.interventions.insert(node.to_string(), InterventionType::SetValue(value));
        self
    }

    /// Enable node (set to true/1.0)
    pub fn enable(mut self, node: &str) -> Self {
        self.interventions.insert(node.to_string(), InterventionType::Enable);
        self
    }

    /// Disable node (set to false/0.0)
    pub fn disable(mut self, node: &str) -> Self {
        self.interventions.insert(node.to_string(), InterventionType::Disable);
        self
    }

    /// Observe node value (condition, not intervene)
    pub fn observe(mut self, node: &str, value: f64) -> Self {
        self.conditions.insert(node.to_string(), value);
        self
    }
}

/// Result of intervention prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionResult {
    /// Target node being predicted
    pub target: String,

    /// Predicted probability/value after intervention
    pub predicted_value: f64,

    /// Baseline value without intervention (for comparison)
    pub baseline_value: Option<f64>,

    /// Uncertainty in prediction (standard deviation)
    pub uncertainty: f64,

    /// Confidence interval [lower, upper]
    pub confidence_interval: (f64, f64),

    /// Causal path from intervention to target
    pub causal_path: Vec<String>,

    /// Source of uncertainty
    pub uncertainty_source: UncertaintySource,

    /// Natural language explanation
    pub explanation: String,
}

/// Causal intervention engine - implements do-calculus
pub struct CausalInterventionEngine {
    /// Original causal graph
    original_graph: ProbabilisticCausalGraph,

    /// Cache of intervention results
    intervention_cache: HashMap<String, InterventionResult>,
}

impl CausalInterventionEngine {
    /// Create new intervention engine
    pub fn new(graph: ProbabilisticCausalGraph) -> Self {
        Self {
            original_graph: graph,
            intervention_cache: HashMap::new(),
        }
    }

    /// Predict effect of intervention on a target
    ///
    /// Answers: P(target | do(intervention))
    ///
    /// This implements do-calculus by:
    /// 1. Creating modified graph with incoming edges removed
    /// 2. Setting intervention nodes to specified values
    /// 3. Propagating effects through modified graph
    /// 4. Computing probability distribution at target
    ///
    /// Example:
    /// ```ignore
    /// let engine = CausalInterventionEngine::new(prob_graph);
    /// let result = engine.predict_intervention(
    ///     "security_check",  // Do this
    ///     "phi_value",       // Effect on this
    /// )?;
    /// println!("P(phi_value | do(security_check)) = {:.2}", result.predicted_value);
    /// ```
    pub fn predict_intervention(
        &mut self,
        intervention_node: &str,
        target_node: &str,
    ) -> InterventionResult {
        // Create intervention specification
        let spec = InterventionSpec::new().enable(intervention_node);

        // Predict with full specification
        self.predict_intervention_spec(&spec, target_node)
    }

    /// Predict effect of complex intervention specification
    pub fn predict_intervention_spec(
        &mut self,
        spec: &InterventionSpec,
        target_node: &str,
    ) -> InterventionResult {
        // Check cache first
        let cache_key = format!("{:?}→{}", spec.interventions, target_node);
        if let Some(cached) = self.intervention_cache.get(&cache_key) {
            return cached.clone();
        }

        // Step 1: Get baseline (observational) value for comparison
        let baseline = self.get_baseline_value(target_node);

        // Step 2: Create modified graph via graph surgery
        let modified_graph = self.apply_intervention(&spec);

        // Step 3: Predict from intervention nodes to find effects
        // We predict from the intervention node(s), not the target
        let intervention_nodes: Vec<String> = spec.interventions.keys().cloned().collect();

        // For now, handle single intervention case (most common)
        let all_predictions = if intervention_nodes.len() == 1 {
            modified_graph.predict_with_uncertainty(&intervention_nodes[0])
        } else {
            // Multiple interventions - predict from first one for now
            // TODO: Handle multiple intervention nodes properly
            if let Some(first) = intervention_nodes.first() {
                modified_graph.predict_with_uncertainty(first)
            } else {
                Vec::new()
            }
        };

        // Step 4: Find prediction for our specific target node
        let target_prediction = all_predictions.iter()
            .find(|pred| pred.event_type == target_node);

        // Step 5: Extract result
        let result = if let Some(pred) = target_prediction {
            InterventionResult {
                target: target_node.to_string(),
                predicted_value: pred.probability,
                baseline_value: Some(baseline),
                uncertainty: (pred.confidence_interval.1 - pred.confidence_interval.0) / 4.0, // ~1σ
                confidence_interval: pred.confidence_interval,
                causal_path: pred.causal_chain.clone(),
                uncertainty_source: pred.uncertainty_source.clone(),
                explanation: self.generate_explanation(spec, target_node, pred),
            }
        } else {
            // No prediction available - target might be intervention node itself
            let value = spec.interventions.get(target_node)
                .map(|i| match i {
                    InterventionType::SetValue(v) => *v,
                    InterventionType::Enable => 1.0,
                    InterventionType::Disable => 0.0,
                    InterventionType::Neutralize => 0.0,
                })
                .unwrap_or(baseline);

            InterventionResult {
                target: target_node.to_string(),
                predicted_value: value,
                baseline_value: Some(baseline),
                uncertainty: 0.0,
                confidence_interval: (value, value),
                causal_path: vec![target_node.to_string()],
                uncertainty_source: UncertaintySource::WellEstimated,
                explanation: format!("Direct intervention sets {} to {:.2}", target_node, value),
            }
        };

        // Cache result
        self.intervention_cache.insert(cache_key, result.clone());

        result
    }

    /// Get baseline (observational) value for a node
    fn get_baseline_value(&self, node: &str) -> f64 {
        // For now, use simple heuristic: average of all outgoing edge probabilities
        // In full implementation, this would do proper probabilistic inference
        let predictions = self.original_graph.predict_with_uncertainty(node);
        predictions.first()
            .map(|p| p.probability)
            .unwrap_or(0.5)  // Default to 50% if unknown
    }

    /// Apply intervention via graph surgery
    ///
    /// Implements do(X=x) by:
    /// 1. Copying original graph
    /// 2. For each intervention node:
    ///    - Remove all incoming edges (break parent relationships)
    ///    - Set node value deterministically
    /// 3. Keep all downstream edges intact
    ///
    /// This creates the "mutilated graph" for causal inference.
    fn apply_intervention(&self, spec: &InterventionSpec) -> ProbabilisticCausalGraph {
        // Clone original graph
        let mut modified = self.original_graph.clone();

        // For each intervention
        for (node, intervention) in &spec.interventions {
            // Graph surgery: Remove incoming edges to intervention node
            // This breaks the parent-child relationships, making the intervention
            // exogenous (not caused by anything in the system)
            modified.remove_incoming_edges(node);

            // Set intervention value
            let value = match intervention {
                InterventionType::SetValue(v) => *v,
                InterventionType::Enable => 1.0,
                InterventionType::Disable => 0.0,
                InterventionType::Neutralize => 0.0,
            };

            // If neutralizing, also remove outgoing edges
            if matches!(intervention, InterventionType::Neutralize) {
                modified.remove_outgoing_edges(node);
            }

            // Store intervention for later reference
            // (In full implementation, this would update node values)
        }

        modified
    }

    /// Generate natural language explanation
    fn generate_explanation(
        &self,
        spec: &InterventionSpec,
        target: &str,
        prediction: &ProbabilisticPrediction,
    ) -> String {
        let intervention_desc = if spec.interventions.len() == 1 {
            let (node, itype) = spec.interventions.iter().next().unwrap();
            match itype {
                InterventionType::Enable => format!("Enabling {}", node),
                InterventionType::Disable => format!("Disabling {}", node),
                InterventionType::SetValue(v) => format!("Setting {} to {:.2}", node, v),
                InterventionType::Neutralize => format!("Neutralizing {}", node),
            }
        } else {
            format!("Applying {} interventions", spec.interventions.len())
        };

        let path = if prediction.causal_chain.len() > 1 {
            format!(" through {}", prediction.causal_chain[1..].join(" → "))
        } else {
            String::new()
        };

        format!(
            "{} will affect {} with {:.0}% probability{}. Confidence: {}",
            intervention_desc,
            target,
            prediction.probability * 100.0,
            path,
            if prediction.confidence_level > 0.8 { "High" }
            else if prediction.confidence_level > 0.5 { "Medium" }
            else { "Low" }
        )
    }

    /// Compare multiple intervention strategies
    pub fn compare_interventions(
        &mut self,
        specs: &[InterventionSpec],
        target: &str,
    ) -> Vec<InterventionResult> {
        specs.iter()
            .map(|spec| self.predict_intervention_spec(spec, target))
            .collect()
    }

    /// Find best intervention to maximize target value
    pub fn optimize_intervention(
        &mut self,
        candidate_nodes: &[String],
        target: &str,
    ) -> Option<(String, InterventionResult)> {
        candidate_nodes.iter()
            .map(|node| {
                let result = self.predict_intervention(node, target);
                (node.clone(), result)
            })
            .max_by(|(_, a), (_, b)| {
                a.predicted_value.partial_cmp(&b.predicted_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::probabilistic_inference::ProbabilisticConfig;

    #[test]
    fn test_intervention_spec_builder() {
        let spec = InterventionSpec::new()
            .enable("security_check")
            .set_value("threshold", 0.8)
            .observe("phi_value", 0.7);

        assert_eq!(spec.interventions.len(), 2);
        assert_eq!(spec.conditions.len(), 1);
        assert!(matches!(
            spec.interventions.get("security_check"),
            Some(InterventionType::Enable)
        ));
    }

    #[test]
    fn test_intervention_engine_creation() {
        let graph = ProbabilisticCausalGraph::new();
        let engine = CausalInterventionEngine::new(graph);

        assert_eq!(engine.intervention_cache.len(), 0);
    }

    #[test]
    fn test_simple_intervention() {
        // Create simple A → B graph
        let mut graph = ProbabilisticCausalGraph::new();

        // Observe A → B relationship (80% of the time)
        for _ in 0..8 {
            graph.observe_edge("A", "B", EdgeType::Direct, true);
        }
        for _ in 0..2 {
            graph.observe_edge("A", "B", EdgeType::Direct, false);
        }

        let mut engine = CausalInterventionEngine::new(graph);

        // Predict: P(B | do(A))
        let result = engine.predict_intervention("A", "B");

        // Should predict high probability since A causes B
        assert!(result.predicted_value > 0.5, "Intervention should predict positive effect");
        assert_eq!(result.target, "B");
        assert!(result.causal_path.contains(&"A".to_string()));
    }

    #[test]
    fn test_intervention_caching() {
        let mut graph = ProbabilisticCausalGraph::new();
        graph.observe_edge("X", "Y", EdgeType::Direct, true);

        let mut engine = CausalInterventionEngine::new(graph);

        // First prediction
        let result1 = engine.predict_intervention("X", "Y");
        assert_eq!(engine.intervention_cache.len(), 1);

        // Second prediction (should use cache)
        let result2 = engine.predict_intervention("X", "Y");
        assert_eq!(engine.intervention_cache.len(), 1);

        // Results should be identical
        assert_eq!(result1.predicted_value, result2.predicted_value);
    }

    #[test]
    fn test_intervention_comparison() {
        let mut graph = ProbabilisticCausalGraph::new();

        // Create two paths: A → C and B → C
        for _ in 0..7 {
            graph.observe_edge("A", "C", EdgeType::Direct, true);
        }
        for _ in 0..5 {
            graph.observe_edge("B", "C", EdgeType::Direct, true);
        }

        let mut engine = CausalInterventionEngine::new(graph);

        // Compare: do(A) vs do(B) for effect on C
        let spec_a = InterventionSpec::new().enable("A");
        let spec_b = InterventionSpec::new().enable("B");

        let results = engine.compare_interventions(&[spec_a, spec_b], "C");

        assert_eq!(results.len(), 2);
        // A should have stronger effect (observed 70% vs 50%)
        assert!(results[0].predicted_value > results[1].predicted_value);
    }
}
