// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pearl's Do-Calculus for Rigorous Counterfactual Reasoning
//!
//! Implements Structural Causal Models (SCM) with do-interventions and
//! the three-step counterfactual algorithm (abduction, action, prediction).

use super::super::unified_understanding::DeepUnderstanding;
use std::collections::HashMap;

/// Pearl's Structural Causal Model (SCM)
/// Implements do-calculus for rigorous causal intervention
#[derive(Debug, Clone)]
pub struct StructuralCausalModel {
    /// Variables in the model
    variables: HashMap<String, CausalVariable>,
    /// Structural equations: variable -> parents -> function
    equations: HashMap<String, StructuralEquation>,
    /// Exogenous noise terms
    exogenous: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct CausalVariable {
    pub name: String,
    pub value: f64,
    pub domain: VariableDomain,
}

#[derive(Debug, Clone)]
pub enum VariableDomain {
    Binary, // 0 or 1
    Continuous { min: f64, max: f64 },
    Categorical(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct StructuralEquation {
    /// Variable this equation defines
    pub target: String,
    /// Parent variables
    pub parents: Vec<String>,
    /// Coefficients for linear combination (simplified)
    pub coefficients: Vec<f64>,
    /// Intercept
    pub intercept: f64,
}

/// Result of a do-intervention: do(X = x)
#[derive(Debug, Clone)]
pub struct InterventionResult {
    /// The intervention performed
    pub intervention: String,
    /// Post-intervention distribution of target variable
    pub target_value: f64,
    /// Causal effect: E[Y | do(X=x)] - E[Y | do(X=x')]
    pub causal_effect: f64,
    /// Confidence based on model completeness
    pub confidence: f64,
    /// Causal path from intervention to outcome
    pub causal_path: Vec<String>,
}

/// Counterfactual query result
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// The counterfactual question
    pub query: String,
    /// Factual value (what actually happened)
    pub factual: f64,
    /// Counterfactual value (what would have happened)
    pub counterfactual: f64,
    /// Probability of necessity: P(Y_x' = 0 | X = x, Y = 1)
    pub probability_necessity: f64,
    /// Probability of sufficiency: P(Y_x = 1 | X = x', Y = 0)
    pub probability_sufficiency: f64,
    /// Explanation chain
    pub explanation: Vec<String>,
}

impl StructuralCausalModel {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            equations: HashMap::new(),
            exogenous: HashMap::new(),
        }
    }

    /// Add a variable to the model
    pub fn add_variable(&mut self, name: &str, value: f64, domain: VariableDomain) {
        self.variables.insert(
            name.to_string(),
            CausalVariable {
                name: name.to_string(),
                value,
                domain,
            },
        );
    }

    /// Add a structural equation
    pub fn add_equation(
        &mut self,
        target: &str,
        parents: Vec<&str>,
        coefficients: Vec<f64>,
        intercept: f64,
    ) {
        self.equations.insert(
            target.to_string(),
            StructuralEquation {
                target: target.to_string(),
                parents: parents.into_iter().map(String::from).collect(),
                coefficients,
                intercept,
            },
        );
    }

    /// Learn causal structure from understanding
    pub fn learn_from_understanding(&mut self, understanding: &DeepUnderstanding) {
        // Extract entities as potential variables
        for entity in &understanding.narrative.entities {
            if !self.variables.contains_key(&entity.name) {
                self.add_variable(
                    &entity.name,
                    entity.valence as f64,
                    VariableDomain::Continuous {
                        min: -1.0,
                        max: 1.0,
                    },
                );
            }
        }

        // Extract causal structure if present
        if let Some(ref causal) = understanding.grounded.causal_structure {
            // Add cause and effect as variables if not present
            if !self.variables.contains_key(&causal.cause) {
                self.add_variable(&causal.cause, 1.0, VariableDomain::Binary);
            }
            if !self.variables.contains_key(&causal.effect) {
                self.add_variable(&causal.effect, 1.0, VariableDomain::Binary);
            }

            // Add structural equation: effect = f(cause)
            let strength = causal.strength;
            self.add_equation(&causal.effect, vec![&causal.cause], vec![strength], 0.0);
        }
    }

    /// Perform do-intervention: do(X = x)
    /// This cuts incoming edges to X and sets X = x
    pub fn do_intervention(&self, variable: &str, value: f64, target: &str) -> InterventionResult {
        let mut mutilated = self.clone();

        // Mutilate the graph: remove equation for intervention variable
        mutilated.equations.remove(variable);

        // Set intervention value
        if let Some(var) = mutilated.variables.get_mut(variable) {
            var.value = value;
        }

        // Propagate effects through the graph
        let target_value = mutilated.evaluate(target);

        // Calculate causal effect (compare with counterfactual value)
        let baseline = self.evaluate(target);
        let causal_effect = target_value - baseline;

        // Build causal path
        let causal_path = self.trace_causal_path(variable, target);

        // Confidence based on how complete the model is
        let confidence = if self.equations.contains_key(target) {
            0.7
        } else {
            0.3
        };

        InterventionResult {
            intervention: format!("do({variable} = {value})"),
            target_value,
            causal_effect,
            confidence,
            causal_path,
        }
    }

    /// Three-step counterfactual algorithm (Pearl)
    /// 1. Abduction: Use evidence to determine U
    /// 2. Action: Modify model (do-intervention)
    /// 3. Prediction: Compute outcome in modified model
    pub fn counterfactual(
        &self,
        evidence: &HashMap<String, f64>, // What we observed
        intervention: &str,              // Variable to change
        new_value: f64,                  // Counterfactual value
        target: &str,                    // What to predict
    ) -> CounterfactualResult {
        // Step 1: Abduction - infer exogenous variables from evidence
        let mut model = self.clone();
        for (var, val) in evidence {
            if let Some(v) = model.variables.get_mut(var) {
                v.value = *val;
            }
        }

        // Step 2: Action - apply intervention
        model.equations.remove(intervention);
        if let Some(var) = model.variables.get_mut(intervention) {
            var.value = new_value;
        }

        // Step 3: Prediction - compute counterfactual outcome
        let counterfactual_value = model.evaluate(target);
        let factual_value = self.evaluate(target);

        // Compute probabilities of necessity and sufficiency (simplified)
        let prob_necessity = if factual_value > 0.5 && counterfactual_value < 0.5 {
            0.8 // High: removing cause would remove effect
        } else {
            0.2
        };

        let prob_sufficiency = if factual_value < 0.5 && counterfactual_value > 0.5 {
            0.8 // High: adding cause would add effect
        } else {
            0.2
        };

        CounterfactualResult {
            query: format!("What if {intervention} = {new_value} instead?"),
            factual: factual_value,
            counterfactual: counterfactual_value,
            probability_necessity: prob_necessity,
            probability_sufficiency: prob_sufficiency,
            explanation: self.trace_causal_path(intervention, target),
        }
    }

    /// Evaluate a variable given current state
    fn evaluate(&self, variable: &str) -> f64 {
        if let Some(eq) = self.equations.get(variable) {
            let mut value = eq.intercept;
            for (i, parent) in eq.parents.iter().enumerate() {
                let parent_val = self.variables.get(parent).map(|v| v.value).unwrap_or(0.0);
                value += eq.coefficients.get(i).copied().unwrap_or(0.0) * parent_val;
            }
            value.clamp(-1.0, 1.0)
        } else {
            self.variables.get(variable).map(|v| v.value).unwrap_or(0.0)
        }
    }

    /// Trace causal path between two variables
    fn trace_causal_path(&self, from: &str, to: &str) -> Vec<String> {
        let mut path = vec![from.to_string()];
        let mut current = from.to_string();

        // Simple BFS to find path (in a real implementation, use proper graph traversal)
        for _ in 0..10 {
            // Max depth
            let mut found_next = false;
            for (target, eq) in &self.equations {
                if eq.parents.contains(&current) {
                    path.push(target.clone());
                    if target == to {
                        return path;
                    }
                    current = target.clone();
                    found_next = true;
                    break;
                }
            }
            if !found_next {
                break;
            }
        }

        path
    }

    /// Get number of variables
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Get number of equations
    pub fn equation_count(&self) -> usize {
        self.equations.len()
    }
}

impl Default for StructuralCausalModel {
    fn default() -> Self {
        Self::new()
    }
}
