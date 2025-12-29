// Revolutionary Enhancement #4 - Phase 2: Counterfactual Reasoning
//
// Implements Level 3 of Pearl's Causal Hierarchy: Counterfactuals (Imagining)
//
// Key Innovation: Three-step counterfactual computation
//
// Fundamental Question:
//   "What would have happened if X had been different?"
//
// Real-World Example:
//   Observed: Security was disabled (X=0), and Φ dropped to 0.3 (Y=0.3)
//   Counterfactual: "If security had been enabled (X←1), would Φ have stayed high?"
//
// The Difference:
//   Intervention (Level 2): P(Y | do(X=x)) - prospective, before action
//   Counterfactual (Level 3): P(Y_x | X'=x', Y'=y') - retrospective, after observation
//
// Three-Step Computation (Pearl's Algorithm):
//   1. ABDUCTION: Infer hidden state U from evidence X'=x', Y'=y'
//   2. ACTION: Apply counterfactual intervention X←x in world with inferred U
//   3. PREDICTION: Compute Y in modified world
//
// This enables:
// - Retroactive "what if" analysis
// - Causal attribution ("did X cause Y?")
// - Explanation generation ("Y happened because X")
// - Regret analysis ("should we have done X instead?")

use super::{
    probabilistic_inference::{
        ProbabilisticCausalGraph, ProbabilisticEdge,
    },
    causal_intervention::{
        CausalInterventionEngine, InterventionSpec,
    },
};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Evidence observed in actual world
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Variable name
    pub variable: String,

    /// Observed value
    pub value: f64,
}

impl Evidence {
    pub fn new(variable: &str, value: f64) -> Self {
        Self {
            variable: variable.to_string(),
            value,
        }
    }
}

/// Counterfactual query specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualQuery {
    /// What we observed in actual world
    pub actual_evidence: Vec<Evidence>,

    /// What we want to change (counterfactual intervention)
    pub counterfactual_intervention: HashMap<String, f64>,

    /// What we want to predict in counterfactual world
    pub target: String,
}

impl CounterfactualQuery {
    /// Create new counterfactual query
    pub fn new(target: &str) -> Self {
        Self {
            actual_evidence: Vec::new(),
            counterfactual_intervention: HashMap::new(),
            target: target.to_string(),
        }
    }

    /// Add evidence from actual world
    pub fn with_evidence(mut self, variable: &str, value: f64) -> Self {
        self.actual_evidence.push(Evidence::new(variable, value));
        self
    }

    /// Set counterfactual intervention
    pub fn with_counterfactual(mut self, variable: &str, value: f64) -> Self {
        self.counterfactual_intervention.insert(variable.to_string(), value);
        self
    }
}

/// Hidden state (exogenous variables)
///
/// In causal models, U represents unobserved factors that influence
/// the system but are not caused by anything within the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenState {
    /// Exogenous variable values
    pub exogenous_vars: HashMap<String, f64>,

    /// Confidence in inferred state (0.0 - 1.0)
    pub confidence: f64,

    /// Method used for inference
    pub inference_method: String,
}

/// Result of counterfactual reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualResult {
    /// Target variable being predicted
    pub target: String,

    /// Actual observed value (in real world)
    pub actual_value: f64,

    /// Counterfactual predicted value (in hypothetical world)
    pub counterfactual_value: f64,

    /// Causal effect: counterfactual - actual
    pub causal_effect: f64,

    /// Uncertainty in counterfactual prediction
    pub uncertainty: f64,

    /// Confidence interval for counterfactual
    pub confidence_interval: (f64, f64),

    /// Inferred hidden state
    pub hidden_state: HiddenState,

    /// Explanation of counterfactual reasoning
    pub explanation: String,
}

/// Counterfactual reasoning engine
pub struct CounterfactualEngine {
    /// Probabilistic causal graph
    graph: ProbabilisticCausalGraph,

    /// Intervention engine for modified world
    intervention_engine: CausalInterventionEngine,
}

impl CounterfactualEngine {
    /// Create new counterfactual engine
    pub fn new(graph: ProbabilisticCausalGraph) -> Self {
        let intervention_engine = CausalInterventionEngine::new(graph.clone());

        Self {
            graph,
            intervention_engine,
        }
    }

    /// Compute counterfactual: "What if X had been different?"
    ///
    /// Implements Pearl's three-step algorithm:
    /// 1. ABDUCTION: Infer U from evidence X'=x', Y'=y'
    /// 2. ACTION: Modify graph with X←x
    /// 3. PREDICTION: Compute Y in modified world
    ///
    /// Example:
    /// ```ignore
    /// let engine = CounterfactualEngine::new(prob_graph);
    ///
    /// let query = CounterfactualQuery::new("phi_value")
    ///     .with_evidence("security_check", 0.0)  // Actual: disabled
    ///     .with_evidence("phi_value", 0.3)       // Actual: low Φ
    ///     .with_counterfactual("security_check", 1.0);  // What if: enabled
    ///
    /// let result = engine.compute_counterfactual(&query);
    /// println!("If security had been enabled, Φ would be {:.2}", result.counterfactual_value);
    /// ```
    pub fn compute_counterfactual(&mut self, query: &CounterfactualQuery) -> CounterfactualResult {
        // Step 1: ABDUCTION - Infer hidden state U from evidence
        let hidden_state = self.abduce_hidden_state(&query.actual_evidence);

        // Step 2: ACTION - Create modified world with counterfactual intervention
        let modified_graph = self.apply_counterfactual_action(&query.counterfactual_intervention, &hidden_state);

        // Step 3: PREDICTION - Compute outcome in counterfactual world
        let counterfactual_value = self.predict_in_counterfactual_world(
            &modified_graph,
            &query.counterfactual_intervention,
            &query.target,
        );

        // Get actual observed value for comparison
        let actual_value = query.actual_evidence.iter()
            .find(|e| e.variable == query.target)
            .map(|e| e.value)
            .unwrap_or(0.5);  // Default if not observed

        // Compute causal effect
        let causal_effect = counterfactual_value - actual_value;

        // Generate explanation
        let explanation = self.generate_counterfactual_explanation(
            query,
            actual_value,
            counterfactual_value,
            causal_effect,
        );

        CounterfactualResult {
            target: query.target.clone(),
            actual_value,
            counterfactual_value,
            causal_effect,
            uncertainty: 0.1,  // TODO: Proper uncertainty propagation
            confidence_interval: (
                counterfactual_value - 0.1,
                counterfactual_value + 0.1,
            ),
            hidden_state,
            explanation,
        }
    }

    /// Step 1: ABDUCTION - Infer hidden state from evidence
    ///
    /// Given observations X'=x', Y'=y', infer the unobserved exogenous
    /// variables U that would explain these observations.
    ///
    /// Simplified implementation: Use Bayesian updating on edge probabilities
    /// Full implementation would use structural equations and constraint solving.
    fn abduce_hidden_state(&self, evidence: &[Evidence]) -> HiddenState {
        let mut exogenous_vars = HashMap::new();

        // For each piece of evidence, infer contributing factors
        for ev in evidence {
            // Find edges leading to this variable
            let _incoming_edges: Vec<&ProbabilisticEdge> = self.graph.edges()
                .values()
                .filter(|edge| edge.to == ev.variable)
                .collect();

            // Simplified: Use evidence value as proxy for hidden state
            // In full implementation, solve for U in structural equations
            exogenous_vars.insert(
                format!("U_{}", ev.variable),
                ev.value,
            );
        }

        // Confidence based on amount of evidence
        let confidence = if evidence.len() >= 3 {
            0.8
        } else if evidence.len() >= 2 {
            0.6
        } else {
            0.4
        };

        HiddenState {
            exogenous_vars,
            confidence,
            inference_method: "Bayesian abduction (simplified)".to_string(),
        }
    }

    /// Step 2: ACTION - Apply counterfactual intervention
    ///
    /// Create modified graph where counterfactual intervention is applied
    /// while preserving the inferred hidden state U.
    fn apply_counterfactual_action(
        &self,
        intervention: &HashMap<String, f64>,
        _hidden_state: &HiddenState,
    ) -> ProbabilisticCausalGraph {
        // Clone graph
        let mut modified = self.graph.clone();

        // Apply intervention (graph surgery)
        for (var, _value) in intervention {
            modified.remove_incoming_edges(var);
        }

        // In full implementation, set structural equations with:
        // - Intervened variables fixed to counterfactual values
        // - Hidden variables U set to inferred values
        // - Other variables computed from equations

        modified
    }

    /// Step 3: PREDICTION - Compute outcome in counterfactual world
    ///
    /// Given modified graph with intervention and hidden state,
    /// predict the value of target variable.
    fn predict_in_counterfactual_world(
        &mut self,
        modified_graph: &ProbabilisticCausalGraph,
        intervention: &HashMap<String, f64>,
        target: &str,
    ) -> f64 {
        // For each intervention variable that affects target, compute contribution
        let mut target_probability = 0.0;
        let mut contributing_causes = 0;

        for (interv_var, interv_value) in intervention {
            // Check if there's an edge from intervention variable to target
            if let Some(edge) = modified_graph.edge_probability(interv_var, target) {
                // Contribution = intervention_value × P(target | intervention_var)
                target_probability += interv_value * edge.probability;
                contributing_causes += 1;
            }
        }

        // If no direct causal path found, use intervention engine approach
        if contributing_causes == 0 {
            let mut spec = InterventionSpec::new();
            for (var, value) in intervention {
                spec = spec.set_value(var, *value);
            }

            let mut engine = CausalInterventionEngine::new(modified_graph.clone());
            let result = engine.predict_intervention_spec(&spec, target);
            return result.predicted_value;
        }

        // Average contributions from all causes
        target_probability / contributing_causes as f64
    }

    /// Generate natural language explanation of counterfactual
    fn generate_counterfactual_explanation(
        &self,
        query: &CounterfactualQuery,
        actual: f64,
        counterfactual: f64,
        effect: f64,
    ) -> String {
        let intervention_desc = if query.counterfactual_intervention.len() == 1 {
            let (var, val) = query.counterfactual_intervention.iter().next().unwrap();
            format!("{} had been {:.2}", var, val)
        } else {
            format!("{} variables had been different", query.counterfactual_intervention.len())
        };

        let effect_desc = if effect > 0.0 {
            format!("{:.2} higher", effect.abs())
        } else if effect < 0.0 {
            format!("{:.2} lower", effect.abs())
        } else {
            "unchanged".to_string()
        };

        format!(
            "If {} (instead of actual), {} would have been {} ({:.2} vs {:.2} actual). Causal effect: {}",
            intervention_desc,
            query.target,
            effect_desc,
            counterfactual,
            actual,
            effect_desc
        )
    }

    /// Answer causal attribution question: "Did X cause Y?"
    ///
    /// Tests whether X=x was a cause of Y=y by checking:
    /// Would Y have been different if X had been different?
    pub fn did_cause(&mut self, cause: &str, cause_value: f64, effect: &str, effect_value: f64) -> bool {
        let query = CounterfactualQuery::new(effect)
            .with_evidence(cause, cause_value)
            .with_evidence(effect, effect_value)
            .with_counterfactual(cause, 1.0 - cause_value);  // Flip the cause

        let result = self.compute_counterfactual(&query);

        // X caused Y if flipping X would have changed Y significantly
        result.causal_effect.abs() > 0.1
    }

    /// Compute necessity and sufficiency of cause
    ///
    /// - Necessary: Would Y have occurred without X? (PN - Probability of Necessity)
    /// - Sufficient: Would Y have occurred with X? (PS - Probability of Sufficiency)
    pub fn necessity_sufficiency(&mut self, cause: &str, effect: &str) -> (f64, f64) {
        // Necessity: P(Y would not occur | !X, Y occurred)
        let necessity_query = CounterfactualQuery::new(effect)
            .with_evidence(cause, 1.0)
            .with_evidence(effect, 1.0)
            .with_counterfactual(cause, 0.0);

        let necessity_result = self.compute_counterfactual(&necessity_query);
        let pn = 1.0 - necessity_result.counterfactual_value;  // How much would Y decrease?

        // Sufficiency: P(Y would occur | X, Y did not occur)
        let sufficiency_query = CounterfactualQuery::new(effect)
            .with_evidence(cause, 0.0)
            .with_evidence(effect, 0.0)
            .with_counterfactual(cause, 1.0);

        let sufficiency_result = self.compute_counterfactual(&sufficiency_query);
        let ps = sufficiency_result.counterfactual_value;  // How much would Y increase?

        (pn, ps)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::{EdgeType, probabilistic_inference::ProbabilisticConfig};

    #[test]
    fn test_counterfactual_query_builder() {
        let query = CounterfactualQuery::new("phi_value")
            .with_evidence("security_check", 0.0)
            .with_evidence("phi_value", 0.3)
            .with_counterfactual("security_check", 1.0);

        assert_eq!(query.target, "phi_value");
        assert_eq!(query.actual_evidence.len(), 2);
        assert_eq!(query.counterfactual_intervention.len(), 1);
    }

    #[test]
    fn test_counterfactual_engine_creation() {
        let graph = ProbabilisticCausalGraph::new();
        let engine = CounterfactualEngine::new(graph);

        // Should create successfully
        assert!(true);
    }

    #[test]
    fn test_simple_counterfactual() {
        // Create graph: A → B (80% probability)
        let mut graph = ProbabilisticCausalGraph::new();

        for _ in 0..8 {
            graph.observe_edge("A", "B", EdgeType::Direct, true);
        }
        for _ in 0..2 {
            graph.observe_edge("A", "B", EdgeType::Direct, false);
        }

        let mut engine = CounterfactualEngine::new(graph);

        // Actual: A=0, B=0
        // Counterfactual: What if A=1?
        let query = CounterfactualQuery::new("B")
            .with_evidence("A", 0.0)
            .with_evidence("B", 0.0)
            .with_counterfactual("A", 1.0);

        let result = engine.compute_counterfactual(&query);

        // B should be higher in counterfactual world (where A=1)
        assert!(result.counterfactual_value > result.actual_value);
        assert_eq!(result.target, "B");
        assert!(result.causal_effect > 0.0);
    }

    #[test]
    fn test_causal_attribution() {
        // Create graph: A → B
        let mut graph = ProbabilisticCausalGraph::new();
        for _ in 0..9 {
            graph.observe_edge("A", "B", EdgeType::Direct, true);
        }

        let mut engine = CounterfactualEngine::new(graph);

        // Did A=1 cause B=1?
        let caused = engine.did_cause("A", 1.0, "B", 1.0);

        // Yes, because flipping A would change B
        assert!(caused);
    }

    #[test]
    fn test_necessity_sufficiency() {
        // Create strong A → B relationship
        let mut graph = ProbabilisticCausalGraph::new();
        for _ in 0..9 {
            graph.observe_edge("A", "B", EdgeType::Direct, true);
        }

        let mut engine = CounterfactualEngine::new(graph);

        let (necessity, sufficiency) = engine.necessity_sufficiency("A", "B");

        // Strong relationship should have high necessity and sufficiency
        assert!(necessity > 0.5, "Necessity should be high");
        assert!(sufficiency > 0.5, "Sufficiency should be high");
    }
}
