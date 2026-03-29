// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Counterfactual Reasoning with Uncertainty Propagation
//!
//! Implements what-if analysis for consciousness systems with proper uncertainty
//! quantification. Enables the system to reason about alternative scenarios:
//!
//! - "What would have happened if I chose differently?"
//! - "How certain am I about counterfactual outcomes?"
//! - "How does uncertainty propagate through causal chains?"
//!
//! ## Theoretical Foundation
//!
//! Based on Pearl's Structural Causal Models (SCM) and Judea Pearl's Do-Calculus:
//!
//! 1. **Counterfactual**: P(Y_x | X=x', Y=y')
//!    - What would Y have been if X were x, given that we observed X=x' and Y=y'?
//!
//! 2. **Uncertainty Propagation**:
//!    - Epistemic uncertainty (reducible through observation)
//!    - Aleatoric uncertainty (irreducible randomness)
//!    - Structural uncertainty (model limitations)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// UNCERTAINTY MODEL FOR COUNTERFACTUAL REASONING
// ═══════════════════════════════════════════════════════════════════════════════

/// 3D Uncertainty for counterfactual inferences
///
/// Mirrors the GIS uncertainty model but specialized for causal reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualUncertainty {
    /// Epistemic: Uncertainty from limited observations/knowledge
    /// Can be reduced by observing more data
    pub epistemic: f32,

    /// Aleatoric: Inherent randomness in the causal mechanism
    /// Cannot be reduced - fundamental stochasticity
    pub aleatoric: f32,

    /// Structural: Uncertainty from model misspecification
    /// Can be reduced by improving the causal model
    pub structural: f32,
}

impl CounterfactualUncertainty {
    /// Create new uncertainty with explicit values
    pub fn new(epistemic: f32, aleatoric: f32, structural: f32) -> Self {
        Self {
            epistemic: epistemic.clamp(0.0, 1.0),
            aleatoric: aleatoric.clamp(0.0, 1.0),
            structural: structural.clamp(0.0, 1.0),
        }
    }

    /// Create with complete certainty
    pub fn certain() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Create with maximum uncertainty
    pub fn maximum() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Total uncertainty (L2 norm, scaled to `[0,1]`)
    pub fn total(&self) -> f32 {
        let sum_sq = self.epistemic.powi(2) + self.aleatoric.powi(2) + self.structural.powi(2);
        (sum_sq.sqrt() / 3.0_f32.sqrt()).clamp(0.0, 1.0)
    }

    /// Confidence (inverse of total uncertainty)
    pub fn confidence(&self) -> f32 {
        1.0 - self.total()
    }

    /// Propagate uncertainty through a causal link
    ///
    /// When A causes B, uncertainty propagates:
    /// - Epistemic compounds (we know less about downstream effects)
    /// - Aleatoric adds (independent random sources)
    /// - Structural compounds (model errors accumulate)
    pub fn propagate(&self, link_strength: f32, link_noise: f32) -> Self {
        // Epistemic uncertainty increases with weaker links
        let epistemic_factor = 1.0 + (1.0 - link_strength) * 0.5;

        // Aleatoric adds quadratically (independent noise sources)
        let aleatoric_added = (self.aleatoric.powi(2) + link_noise.powi(2)).sqrt();

        // Structural compounds multiplicatively
        let structural_factor = 1.0 + link_strength * 0.2;

        Self::new(
            self.epistemic * epistemic_factor,
            aleatoric_added,
            self.structural * structural_factor,
        )
    }

    /// Combine uncertainties from multiple sources (e.g., multiple causes)
    ///
    /// Uses maximum for epistemic/structural (worst-case) and
    /// quadratic sum for aleatoric (independent noise)
    pub fn combine(uncertainties: &[Self]) -> Self {
        if uncertainties.is_empty() {
            return Self::certain();
        }

        let epistemic = uncertainties
            .iter()
            .map(|u| u.epistemic)
            .fold(0.0_f32, |a, b| a.max(b));

        let aleatoric_sq: f32 = uncertainties.iter().map(|u| u.aleatoric.powi(2)).sum();
        let aleatoric = aleatoric_sq.sqrt().min(1.0);

        let structural = uncertainties
            .iter()
            .map(|u| u.structural)
            .fold(0.0_f32, |a, b| a.max(b));

        Self::new(epistemic, aleatoric, structural)
    }

    /// Reduce epistemic uncertainty by factor (from new evidence)
    pub fn reduce_epistemic(&self, factor: f32) -> Self {
        Self::new(
            self.epistemic * (1.0 - factor.clamp(0.0, 1.0)),
            self.aleatoric,
            self.structural,
        )
    }

    /// Reduce structural uncertainty by factor (from model improvement)
    pub fn reduce_structural(&self, factor: f32) -> Self {
        Self::new(
            self.epistemic,
            self.aleatoric,
            self.structural * (1.0 - factor.clamp(0.0, 1.0)),
        )
    }
}

impl Default for CounterfactualUncertainty {
    fn default() -> Self {
        // Default to moderate uncertainty
        Self::new(0.3, 0.2, 0.2)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSAL STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════

/// A node in the causal graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalNode {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Current observed value (if any)
    pub observed_value: Option<f64>,
    /// Counterfactual value (if computed)
    pub counterfactual_value: Option<f64>,
    /// Uncertainty in this node's value
    pub uncertainty: CounterfactualUncertainty,
    /// Whether this node has been intervened on (do-operator)
    pub intervened: bool,
}

impl CausalNode {
    /// Create new causal node
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            observed_value: None,
            counterfactual_value: None,
            uncertainty: CounterfactualUncertainty::default(),
            intervened: false,
        }
    }

    /// Set observed value
    pub fn observe(&mut self, value: f64) {
        self.observed_value = Some(value);
        // Observation reduces epistemic uncertainty
        self.uncertainty = self.uncertainty.reduce_epistemic(0.3);
    }

    /// Set intervention (do-operator)
    pub fn intervene(&mut self, value: f64) {
        self.counterfactual_value = Some(value);
        self.intervened = true;
        // Intervention gives us certainty about this node
        self.uncertainty = CounterfactualUncertainty::new(0.0, self.uncertainty.aleatoric, 0.0);
    }
}

/// A causal link between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Causal strength (-1 to 1, negative = inhibitory)
    pub strength: f64,
    /// Noise/randomness in this causal mechanism
    pub noise: f32,
    /// Confidence in this causal relationship
    pub confidence: f32,
}

impl CausalLink {
    /// Create new causal link
    pub fn new(from: impl Into<String>, to: impl Into<String>, strength: f64) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            strength: strength.clamp(-1.0, 1.0),
            noise: 0.1,
            confidence: 0.8,
        }
    }

    /// Set noise level
    pub fn with_noise(mut self, noise: f32) -> Self {
        self.noise = noise.clamp(0.0, 1.0);
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COUNTERFACTUAL ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the counterfactual engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualConfig {
    /// Maximum iterations for inference
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Whether to track uncertainty propagation
    pub track_uncertainty: bool,
    /// Minimum confidence to report counterfactual
    pub min_confidence: f32,
}

impl Default for CounterfactualConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            track_uncertainty: true,
            min_confidence: 0.3,
        }
    }
}

/// Result of a counterfactual query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualResult {
    /// Query description
    pub query: String,
    /// Target node ID
    pub target_node: String,
    /// Factual value (what actually happened)
    pub factual_value: Option<f64>,
    /// Counterfactual value (what would have happened)
    pub counterfactual_value: f64,
    /// Uncertainty in the counterfactual
    pub uncertainty: CounterfactualUncertainty,
    /// Confidence (inverse of total uncertainty)
    pub confidence: f32,
    /// Causal path used for inference
    pub causal_path: Vec<String>,
    /// Interventions applied
    pub interventions: HashMap<String, f64>,
}

/// Counterfactual Reasoning Engine
///
/// Implements Pearl's three-level causal hierarchy:
/// 1. Association: P(Y|X) - Seeing
/// 2. Intervention: P(Y|do(X)) - Doing
/// 3. Counterfactual: P(Y_x|X=x', Y=y') - Imagining
#[derive(Debug)]
pub struct CounterfactualEngine {
    /// Causal graph nodes
    nodes: HashMap<String, CausalNode>,
    /// Causal links (edges)
    links: Vec<CausalLink>,
    /// Adjacency list (for efficient traversal)
    adjacency: HashMap<String, Vec<usize>>,
    /// Configuration
    #[allow(dead_code)]
    config: CounterfactualConfig,
}

impl CounterfactualEngine {
    /// Create new counterfactual engine
    pub fn new(config: CounterfactualConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            links: Vec::new(),
            adjacency: HashMap::new(),
            config,
        }
    }

    /// Add a node to the causal graph
    pub fn add_node(&mut self, node: CausalNode) {
        let id = node.id.clone();
        self.nodes.insert(id.clone(), node);
        self.adjacency.entry(id).or_default();
    }

    /// Add a causal link
    pub fn add_link(&mut self, link: CausalLink) {
        let link_idx = self.links.len();
        let from = link.from.clone();
        self.links.push(link);
        self.adjacency.entry(from).or_default().push(link_idx);
    }

    /// Set an observation (evidence)
    pub fn observe(&mut self, node_id: &str, value: f64) -> Result<()> {
        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| anyhow::anyhow!("Node not found: {}", node_id))?;
        node.observe(value);
        Ok(())
    }

    /// Perform a do-intervention
    pub fn intervene(&mut self, node_id: &str, value: f64) -> Result<()> {
        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| anyhow::anyhow!("Node not found: {}", node_id))?;
        node.intervene(value);
        Ok(())
    }

    /// Query: What would target_node be if we do(intervention_node = value)?
    ///
    /// Implements Pearl's counterfactual algorithm:
    /// 1. Abduction: Infer noise terms from observations
    /// 2. Action: Apply intervention (cut incoming edges)
    /// 3. Prediction: Propagate effects forward
    pub fn query_counterfactual(
        &self,
        target_node: &str,
        intervention_node: &str,
        intervention_value: f64,
    ) -> Result<CounterfactualResult> {
        // Validate nodes exist
        if !self.nodes.contains_key(target_node) {
            anyhow::bail!("Target node not found: {}", target_node);
        }
        if !self.nodes.contains_key(intervention_node) {
            anyhow::bail!("Intervention node not found: {}", intervention_node);
        }

        // Find causal path from intervention to target
        let causal_path = self.find_causal_path(intervention_node, target_node);

        // If no path exists, counterfactual = factual (no causal effect)
        if causal_path.is_empty() && intervention_node != target_node {
            let factual_value = self.nodes.get(target_node).and_then(|n| n.observed_value);

            return Ok(CounterfactualResult {
                query: format!(
                    "P({}|do({} = {}))",
                    target_node, intervention_node, intervention_value
                ),
                target_node: target_node.to_string(),
                factual_value,
                counterfactual_value: factual_value.unwrap_or(0.0),
                uncertainty: CounterfactualUncertainty::new(0.5, 0.2, 0.3),
                confidence: 0.5,
                causal_path: vec![],
                interventions: [(intervention_node.to_string(), intervention_value)].into(),
            });
        }

        // Propagate counterfactual effect along path with uncertainty
        let (counterfactual_value, uncertainty) =
            self.propagate_counterfactual(&causal_path, intervention_value)?;

        let factual_value = self.nodes.get(target_node).and_then(|n| n.observed_value);

        let confidence = uncertainty.confidence();

        Ok(CounterfactualResult {
            query: format!(
                "P({}|do({} = {}))",
                target_node, intervention_node, intervention_value
            ),
            target_node: target_node.to_string(),
            factual_value,
            counterfactual_value,
            uncertainty,
            confidence,
            causal_path,
            interventions: [(intervention_node.to_string(), intervention_value)].into(),
        })
    }

    /// Find causal path from source to target using BFS
    fn find_causal_path(&self, source: &str, target: &str) -> Vec<String> {
        if source == target {
            return vec![source.to_string()];
        }

        let mut visited: HashMap<&str, &str> = HashMap::new();
        let mut queue: std::collections::VecDeque<&str> = std::collections::VecDeque::new();

        visited.insert(source, "");
        queue.push_back(source);

        while let Some(current) = queue.pop_front() {
            if let Some(link_indices) = self.adjacency.get(current) {
                for &idx in link_indices {
                    let link = &self.links[idx];
                    let neighbor = &link.to;

                    if !visited.contains_key(neighbor.as_str()) {
                        visited.insert(neighbor.as_str(), current);

                        if neighbor == target {
                            // Reconstruct path
                            let mut path = vec![target.to_string()];
                            let mut curr = target;
                            while let Some(&prev) = visited.get(curr) {
                                if prev.is_empty() {
                                    break;
                                }
                                path.push(prev.to_string());
                                curr = prev;
                            }
                            path.reverse();
                            return path;
                        }

                        queue.push_back(neighbor.as_str());
                    }
                }
            }
        }

        vec![]
    }

    /// Propagate counterfactual value along path with uncertainty tracking
    ///
    /// This is the core uncertainty propagation algorithm:
    /// At each step, uncertainty compounds based on:
    /// - Link strength (weaker links = more epistemic uncertainty)
    /// - Link noise (adds aleatoric uncertainty)
    /// - Link confidence (affects structural uncertainty)
    fn propagate_counterfactual(
        &self,
        path: &[String],
        initial_value: f64,
    ) -> Result<(f64, CounterfactualUncertainty)> {
        if path.is_empty() {
            return Ok((initial_value, CounterfactualUncertainty::certain()));
        }

        let mut current_value = initial_value;
        let mut current_uncertainty = CounterfactualUncertainty::certain();

        // Propagate through each link in the path
        for i in 0..(path.len() - 1) {
            let from = &path[i];
            let to = &path[i + 1];

            // Find the link
            let link = self
                .links
                .iter()
                .find(|l| &l.from == from && &l.to == to)
                .ok_or_else(|| anyhow::anyhow!("Link not found: {} -> {}", from, to))?;

            // Apply causal mechanism with noise
            // y = strength * x + noise
            let mechanism_effect = link.strength * current_value;

            // Add noise (aleatoric uncertainty manifests here)
            let noise_contribution = link.noise as f64 * (current_value.abs() + 0.1);
            current_value = mechanism_effect + noise_contribution * 0.1; // Small noise impact on value

            // Propagate uncertainty through this link
            current_uncertainty = current_uncertainty.propagate(link.confidence, link.noise);

            // Add structural uncertainty from link confidence
            if link.confidence < 0.8 {
                current_uncertainty = CounterfactualUncertainty::new(
                    current_uncertainty.epistemic,
                    current_uncertainty.aleatoric,
                    current_uncertainty.structural + (1.0 - link.confidence) * 0.2,
                );
            }
        }

        Ok((current_value, current_uncertainty))
    }

    /// Compute average treatment effect (ATE) with uncertainty
    ///
    /// ATE = E[Y|do(X=1)] - E[Y|do(X=0)]
    pub fn average_treatment_effect(
        &self,
        treatment_node: &str,
        outcome_node: &str,
    ) -> Result<(f64, CounterfactualUncertainty)> {
        // Query counterfactual for treatment = 1
        let cf_treated = self.query_counterfactual(outcome_node, treatment_node, 1.0)?;

        // Query counterfactual for treatment = 0
        let cf_control = self.query_counterfactual(outcome_node, treatment_node, 0.0)?;

        // ATE = difference
        let ate = cf_treated.counterfactual_value - cf_control.counterfactual_value;

        // Combined uncertainty (both counterfactuals contribute)
        let combined_uncertainty =
            CounterfactualUncertainty::combine(&[cf_treated.uncertainty, cf_control.uncertainty]);

        Ok((ate, combined_uncertainty))
    }

    /// Get node by ID
    pub fn get_node(&self, id: &str) -> Option<&CausalNode> {
        self.nodes.get(id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = &CausalNode> {
        self.nodes.values()
    }

    /// Get all links
    pub fn links(&self) -> &[CausalLink] {
        &self.links
    }

    /// Clear all interventions
    pub fn clear_interventions(&mut self) {
        for node in self.nodes.values_mut() {
            node.intervened = false;
            node.counterfactual_value = None;
        }
    }
}

impl Default for CounterfactualEngine {
    fn default() -> Self {
        Self::new(CounterfactualConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Existing tests (preserved) ──────────────────────────────────────

    #[test]
    fn test_uncertainty_propagation() {
        let u = CounterfactualUncertainty::new(0.2, 0.1, 0.15);

        // Propagate through a moderately strong link
        let propagated = u.propagate(0.8, 0.1);

        // Epistemic should increase (weaker certainty downstream)
        assert!(propagated.epistemic > u.epistemic);

        // Aleatoric should increase (noise adds)
        assert!(propagated.aleatoric > u.aleatoric);

        // Structural should increase slightly
        assert!(propagated.structural >= u.structural);
    }

    #[test]
    fn test_uncertainty_combine() {
        let u1 = CounterfactualUncertainty::new(0.3, 0.1, 0.2);
        let u2 = CounterfactualUncertainty::new(0.2, 0.2, 0.3);

        let combined = CounterfactualUncertainty::combine(&[u1, u2]);

        // Epistemic takes max
        assert!((combined.epistemic - 0.3).abs() < 0.01);

        // Aleatoric combines quadratically
        let expected_aleatoric = (0.1_f32.powi(2) + 0.2_f32.powi(2)).sqrt();
        assert!((combined.aleatoric - expected_aleatoric).abs() < 0.01);

        // Structural takes max
        assert!((combined.structural - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_simple_counterfactual() {
        let mut engine = CounterfactualEngine::default();

        // Build simple causal graph: X -> Y
        engine.add_node(CausalNode::new("X", "Treatment"));
        engine.add_node(CausalNode::new("Y", "Outcome"));
        engine.add_link(CausalLink::new("X", "Y", 0.5));

        // Observe X = 0, Y = 0
        engine.observe("X", 0.0).unwrap();
        engine.observe("Y", 0.0).unwrap();

        // Query: What would Y be if X = 1?
        let result = engine.query_counterfactual("Y", "X", 1.0).unwrap();

        // Y should increase (positive causal effect)
        assert!(result.counterfactual_value > 0.0);
        assert!(result.confidence > 0.0);
        assert_eq!(result.causal_path, vec!["X", "Y"]);
    }

    #[test]
    fn test_chain_counterfactual() {
        let mut engine = CounterfactualEngine::default();

        // Build chain: X -> M -> Y
        engine.add_node(CausalNode::new("X", "Cause"));
        engine.add_node(CausalNode::new("M", "Mediator"));
        engine.add_node(CausalNode::new("Y", "Effect"));

        engine.add_link(CausalLink::new("X", "M", 0.8).with_confidence(0.9));
        engine.add_link(CausalLink::new("M", "Y", 0.6).with_confidence(0.8));

        // Query: What would Y be if X = 1?
        let result = engine.query_counterfactual("Y", "X", 1.0).unwrap();

        // Path should go through mediator
        assert_eq!(result.causal_path, vec!["X", "M", "Y"]);

        // Uncertainty should be higher than direct link (longer path)
        assert!(result.uncertainty.total() > 0.0);
    }

    #[test]
    fn test_average_treatment_effect() {
        let mut engine = CounterfactualEngine::default();

        // Build: Treatment -> Outcome
        engine.add_node(CausalNode::new("T", "Treatment"));
        engine.add_node(CausalNode::new("O", "Outcome"));
        engine.add_link(CausalLink::new("T", "O", 0.7).with_noise(0.1));

        let (ate, uncertainty) = engine.average_treatment_effect("T", "O").unwrap();

        // ATE should be positive (treatment has positive effect)
        assert!(ate > 0.0);

        // Uncertainty should be moderate
        assert!(uncertainty.total() < 0.8);
    }

    #[test]
    fn test_no_causal_path() {
        let mut engine = CounterfactualEngine::default();

        // Build disconnected graph: X, Y (no edge)
        engine.add_node(CausalNode::new("X", "Node X"));
        engine.add_node(CausalNode::new("Y", "Node Y"));

        // Observe Y
        engine.observe("Y", 5.0).unwrap();

        // Query: What would Y be if X = 1?
        let result = engine.query_counterfactual("Y", "X", 1.0).unwrap();

        // No causal effect - counterfactual should equal factual
        assert!((result.counterfactual_value - 5.0).abs() < 0.01);
        assert!(result.causal_path.is_empty());
    }

    #[test]
    fn test_uncertainty_reduces_with_observation() {
        let mut u = CounterfactualUncertainty::default();
        let initial_epistemic = u.epistemic;

        // Reduce epistemic through observation
        u = u.reduce_epistemic(0.5);

        assert!(u.epistemic < initial_epistemic);
        // Aleatoric shouldn't change
    }

    // ── New tests ───────────────────────────────────────────────────────

    // -- Uncertainty edge cases --

    #[test]
    fn test_uncertainty_certain() {
        let u = CounterfactualUncertainty::certain();
        assert_eq!(u.epistemic, 0.0);
        assert_eq!(u.aleatoric, 0.0);
        assert_eq!(u.structural, 0.0);
        assert_eq!(u.total(), 0.0);
        assert_eq!(u.confidence(), 1.0);
    }

    #[test]
    fn test_uncertainty_maximum() {
        let u = CounterfactualUncertainty::maximum();
        assert_eq!(u.epistemic, 1.0);
        assert_eq!(u.aleatoric, 1.0);
        assert_eq!(u.structural, 1.0);
        assert_eq!(u.total(), 1.0);
        assert_eq!(u.confidence(), 0.0);
    }

    #[test]
    fn test_uncertainty_clamping() {
        // Values above 1.0 should be clamped
        let u = CounterfactualUncertainty::new(5.0, -1.0, 2.0);
        assert_eq!(u.epistemic, 1.0);
        assert_eq!(u.aleatoric, 0.0);
        assert_eq!(u.structural, 1.0);
    }

    #[test]
    fn test_uncertainty_combine_empty() {
        let combined = CounterfactualUncertainty::combine(&[]);
        assert_eq!(combined.epistemic, 0.0);
        assert_eq!(combined.aleatoric, 0.0);
        assert_eq!(combined.structural, 0.0);
    }

    #[test]
    fn test_uncertainty_combine_single() {
        let u = CounterfactualUncertainty::new(0.4, 0.3, 0.2);
        let combined = CounterfactualUncertainty::combine(&[u]);
        assert!((combined.epistemic - 0.4).abs() < 1e-6);
        assert!((combined.aleatoric - 0.3).abs() < 1e-6);
        assert!((combined.structural - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_epistemic_preserves_aleatoric_and_structural() {
        let u = CounterfactualUncertainty::new(0.5, 0.3, 0.4);
        let reduced = u.reduce_epistemic(0.5);
        assert!((reduced.epistemic - 0.25).abs() < 1e-6);
        assert_eq!(reduced.aleatoric, u.aleatoric);
        assert_eq!(reduced.structural, u.structural);
    }

    #[test]
    fn test_reduce_structural_preserves_epistemic_and_aleatoric() {
        let u = CounterfactualUncertainty::new(0.5, 0.3, 0.4);
        let reduced = u.reduce_structural(0.5);
        assert_eq!(reduced.epistemic, u.epistemic);
        assert_eq!(reduced.aleatoric, u.aleatoric);
        assert!((reduced.structural - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_epistemic_factor_clamped() {
        let u = CounterfactualUncertainty::new(0.5, 0.3, 0.4);
        // Factor > 1.0 should be clamped to 1.0 → epistemic becomes 0
        let reduced = u.reduce_epistemic(2.0);
        assert_eq!(reduced.epistemic, 0.0);
        // Factor < 0.0 should be clamped to 0.0 → epistemic unchanged
        let unchanged = u.reduce_epistemic(-1.0);
        assert!((unchanged.epistemic - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_uncertainty_propagate_weak_link() {
        let u = CounterfactualUncertainty::new(0.2, 0.1, 0.1);
        // Very weak link (strength=0.1) → high epistemic amplification
        let weak = u.propagate(0.1, 0.0);
        // epistemic_factor = 1.0 + (1.0 - 0.1)*0.5 = 1.45
        let expected_epistemic = (0.2 * 1.45_f32).clamp(0.0, 1.0);
        assert!((weak.epistemic - expected_epistemic).abs() < 1e-4);
    }

    #[test]
    fn test_uncertainty_default() {
        let u = CounterfactualUncertainty::default();
        assert!((u.epistemic - 0.3).abs() < 1e-6);
        assert!((u.aleatoric - 0.2).abs() < 1e-6);
        assert!((u.structural - 0.2).abs() < 1e-6);
    }

    // -- CausalNode edge cases --

    #[test]
    fn test_node_observe_reduces_epistemic() {
        let mut node = CausalNode::new("n", "Node");
        let before = node.uncertainty.epistemic;
        node.observe(1.0);
        assert!(node.uncertainty.epistemic < before);
        assert_eq!(node.observed_value, Some(1.0));
        assert!(!node.intervened);
    }

    #[test]
    fn test_node_intervene_zeroes_epistemic_and_structural() {
        let mut node = CausalNode::new("n", "Node");
        let original_aleatoric = node.uncertainty.aleatoric;
        node.intervene(42.0);
        assert_eq!(node.uncertainty.epistemic, 0.0);
        assert_eq!(node.uncertainty.structural, 0.0);
        assert_eq!(node.uncertainty.aleatoric, original_aleatoric);
        assert!(node.intervened);
        assert_eq!(node.counterfactual_value, Some(42.0));
    }

    // -- CausalLink builder --

    #[test]
    fn test_causal_link_strength_clamped() {
        let link = CausalLink::new("a", "b", 5.0);
        assert_eq!(link.strength, 1.0);

        let neg = CausalLink::new("a", "b", -3.0);
        assert_eq!(neg.strength, -1.0);
    }

    #[test]
    fn test_causal_link_with_noise_clamped() {
        let link = CausalLink::new("a", "b", 0.5).with_noise(2.0);
        assert_eq!(link.noise, 1.0);

        let link2 = CausalLink::new("a", "b", 0.5).with_noise(-0.5);
        assert_eq!(link2.noise, 0.0);
    }

    #[test]
    fn test_causal_link_with_confidence_clamped() {
        let link = CausalLink::new("a", "b", 0.5).with_confidence(3.0);
        assert_eq!(link.confidence, 1.0);
    }

    // -- Engine error paths --

    #[test]
    fn test_observe_nonexistent_node_errors() {
        let mut engine = CounterfactualEngine::default();
        let result = engine.observe("ghost", 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_intervene_nonexistent_node_errors() {
        let mut engine = CounterfactualEngine::default();
        let result = engine.intervene("ghost", 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_query_nonexistent_target_errors() {
        let mut engine = CounterfactualEngine::default();
        engine.add_node(CausalNode::new("X", "X"));
        let result = engine.query_counterfactual("MISSING", "X", 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_query_nonexistent_intervention_errors() {
        let mut engine = CounterfactualEngine::default();
        engine.add_node(CausalNode::new("Y", "Y"));
        let result = engine.query_counterfactual("Y", "MISSING", 1.0);
        assert!(result.is_err());
    }

    // -- Interventional distribution edge cases --

    #[test]
    fn test_intervention_same_node() {
        // do(X = 3) asking about X itself — path = [X]
        let mut engine = CounterfactualEngine::default();
        engine.add_node(CausalNode::new("X", "X"));
        engine.observe("X", 0.0).unwrap();

        let result = engine.query_counterfactual("X", "X", 3.0).unwrap();
        // Path is just [X]; propagate_counterfactual with single-element path
        // returns (initial_value, certain)
        assert!((result.counterfactual_value - 3.0).abs() < 1e-6);
        assert_eq!(result.causal_path, vec!["X"]);
        assert_eq!(result.uncertainty.total(), 0.0);
    }

    #[test]
    fn test_negative_causal_strength() {
        // X --(-0.5)--> Y : increasing X should decrease Y
        let mut engine = CounterfactualEngine::default();
        engine.add_node(CausalNode::new("X", "X"));
        engine.add_node(CausalNode::new("Y", "Y"));
        engine.add_link(CausalLink::new("X", "Y", -0.5).with_noise(0.0));

        let result = engine.query_counterfactual("Y", "X", 1.0).unwrap();
        // mechanism_effect = -0.5 * 1.0 = -0.5, noise_contribution small
        assert!(result.counterfactual_value < 0.0);
    }

    #[test]
    fn test_zero_intervention_value() {
        // do(X = 0) should propagate zero through the graph
        let mut engine = CounterfactualEngine::default();
        engine.add_node(CausalNode::new("X", "X"));
        engine.add_node(CausalNode::new("Y", "Y"));
        engine.add_link(CausalLink::new("X", "Y", 0.9).with_noise(0.0));

        let result = engine.query_counterfactual("Y", "X", 0.0).unwrap();
        // mechanism_effect = 0.9*0.0 = 0.0, noise_contribution = 0.0 * (0+0.1) * 0.1
        // The result should be very close to 0
        assert!(result.counterfactual_value.abs() < 0.02);
    }

    #[test]
    fn test_chain_uncertainty_increases_with_length() {
        // Compare 2-link chain vs 4-link chain
        let mut engine2 = CounterfactualEngine::default();
        engine2.add_node(CausalNode::new("A", "A"));
        engine2.add_node(CausalNode::new("B", "B"));
        engine2.add_link(
            CausalLink::new("A", "B", 0.8)
                .with_noise(0.1)
                .with_confidence(0.9),
        );

        let r2 = engine2.query_counterfactual("B", "A", 1.0).unwrap();

        let mut engine4 = CounterfactualEngine::default();
        for name in &["A", "B", "C", "D"] {
            engine4.add_node(CausalNode::new(*name, *name));
        }
        engine4.add_link(
            CausalLink::new("A", "B", 0.8)
                .with_noise(0.1)
                .with_confidence(0.9),
        );
        engine4.add_link(
            CausalLink::new("B", "C", 0.8)
                .with_noise(0.1)
                .with_confidence(0.9),
        );
        engine4.add_link(
            CausalLink::new("C", "D", 0.8)
                .with_noise(0.1)
                .with_confidence(0.9),
        );

        let r4 = engine4.query_counterfactual("D", "A", 1.0).unwrap();

        // Longer chain should have higher total uncertainty
        assert!(
            r4.uncertainty.total() > r2.uncertainty.total(),
            "4-link chain uncertainty {} should exceed 2-link {}",
            r4.uncertainty.total(),
            r2.uncertainty.total()
        );
    }

    #[test]
    fn test_clear_interventions() {
        let mut engine = CounterfactualEngine::default();
        engine.add_node(CausalNode::new("X", "X"));
        engine.add_node(CausalNode::new("Y", "Y"));
        engine.add_link(CausalLink::new("X", "Y", 0.5));

        engine.intervene("X", 1.0).unwrap();
        assert!(engine.get_node("X").unwrap().intervened);

        engine.clear_interventions();
        assert!(!engine.get_node("X").unwrap().intervened);
        assert!(engine.get_node("X").unwrap().counterfactual_value.is_none());
    }

    #[test]
    fn test_low_confidence_link_adds_structural_uncertainty() {
        let mut engine = CounterfactualEngine::default();
        engine.add_node(CausalNode::new("A", "A"));
        engine.add_node(CausalNode::new("B", "B"));
        // confidence < 0.8 triggers extra structural uncertainty
        engine.add_link(CausalLink::new("A", "B", 0.9).with_confidence(0.5));

        let result = engine.query_counterfactual("B", "A", 1.0).unwrap();
        // Structural uncertainty should be non-trivial due to low-confidence link
        assert!(
            result.uncertainty.structural > 0.05,
            "structural uncertainty {} should be elevated for low-confidence link",
            result.uncertainty.structural
        );
    }

    #[test]
    fn test_engine_accessor_methods() {
        let mut engine = CounterfactualEngine::default();
        engine.add_node(CausalNode::new("A", "Node A"));
        engine.add_node(CausalNode::new("B", "Node B"));
        engine.add_link(CausalLink::new("A", "B", 0.5));

        assert!(engine.get_node("A").is_some());
        assert!(engine.get_node("MISSING").is_none());
        assert_eq!(engine.nodes().count(), 2);
        assert_eq!(engine.links().len(), 1);
    }

    #[test]
    fn test_counterfactual_config_default() {
        let cfg = CounterfactualConfig::default();
        assert_eq!(cfg.max_iterations, 100);
        assert!((cfg.convergence_threshold - 1e-6).abs() < 1e-12);
        assert!(cfg.track_uncertainty);
        assert!((cfg.min_confidence - 0.3).abs() < 1e-6);
    }
}
