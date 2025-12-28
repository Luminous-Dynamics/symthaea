// Revolutionary Enhancement #3: Probabilistic Inference
//
// Transforms causal reasoning from deterministic to probabilistic, enabling:
// - Quantified uncertainty in causal relationships: P(effect|cause) = 0.85 ± 0.10
// - Bayesian learning: Update probabilities as evidence accumulates
// - Confidence intervals: "80% likely with 95% CI: 70%-90%"
// - Robust to noise and missing data
//
// Key Innovations:
// - Probabilistic edges with Beta-distributed probabilities
// - Bayesian belief updates (conjugate priors)
// - Monte Carlo uncertainty propagation through causal chains
// - Automatic uncertainty source diagnosis
//
// Integration with Previous Enhancements:
// - Enhancement #1 (Streaming): Incremental probability updates
// - Enhancement #2 (Patterns): Probabilistic pattern matching

use super::{
    causal_graph::{CausalGraph, CausalEdge, EdgeType},
    types::Event,
    correlation::EventMetadata,
};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use rand::Rng;

/// Configuration for probabilistic inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticConfig {
    /// Prior probability for unknown edges (default: 0.5)
    pub prior_probability: f64,

    /// Confidence in prior (default: 1.0 = weak prior)
    pub prior_confidence: f64,

    /// Minimum observations before trusting probability
    pub min_observations: usize,

    /// Enable Bayesian learning
    pub enable_bayesian_learning: bool,

    /// Monte Carlo samples for uncertainty propagation
    pub monte_carlo_samples: usize,

    /// Confidence level for intervals (default: 0.95)
    pub confidence_level: f64,
}

impl Default for ProbabilisticConfig {
    fn default() -> Self {
        Self {
            prior_probability: 0.5,
            prior_confidence: 1.0,
            min_observations: 3,
            enable_bayesian_learning: true,
            monte_carlo_samples: 1000,
            confidence_level: 0.95,
        }
    }
}

/// Probabilistic edge with learned probabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticEdge {
    /// Edge ID
    pub id: String,

    /// From event ID
    pub from: String,

    /// To event ID
    pub to: String,

    /// P(to | from) - conditional probability
    pub probability: f64,

    /// Confidence in probability estimate (0.0 - 1.0)
    pub confidence: f64,

    /// Total observations
    pub observations: usize,

    /// Times "from" occurred
    pub from_count: usize,

    /// Times "to" followed "from"
    pub to_given_from_count: usize,

    /// Beta distribution alpha (successes + prior)
    pub alpha: f64,

    /// Beta distribution beta (failures + prior)
    pub beta: f64,

    /// Edge type
    pub edge_type: EdgeType,
}

impl ProbabilisticEdge {
    /// Create new probabilistic edge with prior
    pub fn new(
        from: String,
        to: String,
        edge_type: EdgeType,
        prior_prob: f64,
        prior_strength: f64,
    ) -> Self {
        // Convert prior probability and strength to Beta parameters
        let alpha = prior_prob * prior_strength;
        let beta = (1.0 - prior_prob) * prior_strength;

        Self {
            id: format!("{}->{}", from, to),
            from,
            to,
            probability: prior_prob,
            confidence: 0.0,  // Low confidence initially
            observations: 0,
            from_count: 0,
            to_given_from_count: 0,
            alpha,
            beta,
            edge_type,
        }
    }

    /// Update with new observation
    pub fn update(&mut self, from_occurred: bool, to_followed: bool) {
        if from_occurred {
            self.from_count += 1;

            if to_followed {
                self.to_given_from_count += 1;
                self.alpha += 1.0;
            } else {
                self.beta += 1.0;
            }

            self.observations += 1;
            self.update_probability();
        }
    }

    /// Update probability and confidence from Beta distribution
    fn update_probability(&mut self) {
        // MAP estimate: (α - 1) / (α + β - 2)
        // For α, β > 1, otherwise use mean: α / (α + β)
        if self.alpha > 1.0 && self.beta > 1.0 {
            self.probability = (self.alpha - 1.0) / (self.alpha + self.beta - 2.0);
        } else {
            self.probability = self.alpha / (self.alpha + self.beta);
        }

        // Confidence: inverse of variance
        let variance = (self.alpha * self.beta) /
                      ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.0));

        // Normalize to [0, 1]: higher observations = higher confidence
        self.confidence = 1.0 / (1.0 + variance * 10.0);
    }

    /// Get confidence interval
    pub fn confidence_interval(&self, level: f64) -> (f64, f64) {
        // For Beta distribution, use quantiles
        // Approximate with mean ± z * std for now
        let mean = self.alpha / (self.alpha + self.beta);
        let variance = (self.alpha * self.beta) /
                      ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.0));
        let std = variance.sqrt();

        // z-score for confidence level (e.g., 1.96 for 95%)
        let z = match level {
            l if l >= 0.99 => 2.576,
            l if l >= 0.95 => 1.96,
            l if l >= 0.90 => 1.645,
            _ => 1.0,
        };

        let margin = z * std;
        ((mean - margin).max(0.0), (mean + margin).min(1.0))
    }
}

/// Result of probabilistic computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticResult {
    /// Mean probability
    pub mean: f64,

    /// Standard deviation
    pub std_dev: f64,

    /// Confidence interval
    pub confidence_interval: (f64, f64),

    /// Number of samples used
    pub samples: usize,
}

/// Source of uncertainty in predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintySource {
    /// Insufficient observations (< min_observations)
    SmallSampleSize,

    /// High variance in observations
    HighVariance,

    /// Long causal chain compounds uncertainty
    LongChain,

    /// Well-estimated with sufficient data
    WellEstimated,
}

impl std::fmt::Display for UncertaintySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SmallSampleSize => write!(f, "Small Sample Size"),
            Self::HighVariance => write!(f, "High Variance"),
            Self::LongChain => write!(f, "Long Chain"),
            Self::WellEstimated => write!(f, "Well Estimated"),
        }
    }
}

/// Prediction with uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticPrediction {
    /// Predicted event type
    pub event_type: String,

    /// Probability (point estimate)
    pub probability: f64,

    /// Confidence interval (lower, upper)
    pub confidence_interval: (f64, f64),

    /// Confidence level (e.g., 0.95)
    pub confidence_level: f64,

    /// Causal chain leading to prediction
    pub causal_chain: Vec<String>,

    /// Source of uncertainty
    pub uncertainty_source: UncertaintySource,

    /// Number of observations supporting this
    pub observations: usize,
}

/// Bayesian inference engine
#[derive(Clone)]
pub struct BayesianInference {
    /// Configuration
    config: ProbabilisticConfig,
}

impl BayesianInference {
    /// Create new Bayesian inference engine
    pub fn new(config: ProbabilisticConfig) -> Self {
        Self {
            config,
        }
    }

    /// Sample from Beta distribution (simplified approximation)
    /// For production, use rand_distr crate's Beta distribution
    pub fn sample_beta(&self, alpha: f64, beta: f64) -> f64 {
        // Simplified: use mean with random perturbation
        // This is a placeholder - for production, use proper Beta sampling
        let mean = alpha / (alpha + beta);
        let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
        let std = variance.sqrt();

        // Add random perturbation (rough approximation)
        let mut rng = rand::thread_rng();
        let perturbation = (rng.gen::<f64>() - 0.5) * 2.0 * std * 2.0;
        (mean + perturbation).max(0.0).min(1.0)
    }

    /// Diagnose uncertainty source
    pub fn diagnose_uncertainty(&self, edge: &ProbabilisticEdge) -> UncertaintySource {
        if edge.observations < self.config.min_observations {
            UncertaintySource::SmallSampleSize
        } else if edge.confidence < 0.5 {
            UncertaintySource::HighVariance
        } else {
            UncertaintySource::WellEstimated
        }
    }
}

/// Probabilistic causal graph
#[derive(Clone)]
pub struct ProbabilisticCausalGraph {
    /// Underlying deterministic graph
    graph: CausalGraph,

    /// Probabilistic edges (keyed by "from->to")
    probabilistic_edges: HashMap<String, ProbabilisticEdge>,

    /// Bayesian inference engine
    inference: BayesianInference,

    /// Configuration
    config: ProbabilisticConfig,
}

impl ProbabilisticCausalGraph {
    /// Create new probabilistic graph
    pub fn new() -> Self {
        Self::with_config(ProbabilisticConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: ProbabilisticConfig) -> Self {
        let inference = BayesianInference::new(config.clone());

        Self {
            graph: CausalGraph::new(),
            probabilistic_edges: HashMap::new(),
            inference,
            config,
        }
    }

    /// Observe an event pair (from → to)
    pub fn observe_edge(
        &mut self,
        from: &str,
        to: &str,
        edge_type: EdgeType,
        to_actually_followed: bool,
    ) {
        let key = format!("{}->{}", from, to);

        // Get or create probabilistic edge
        let edge = self.probabilistic_edges.entry(key.clone()).or_insert_with(|| {
            ProbabilisticEdge::new(
                from.to_string(),
                to.to_string(),
                edge_type,
                self.config.prior_probability,
                self.config.prior_confidence,
            )
        });

        // Update with observation
        edge.update(true, to_actually_followed);
    }

    /// Get edge probability
    pub fn edge_probability(&self, from: &str, to: &str) -> Option<&ProbabilisticEdge> {
        let key = format!("{}->{}", from, to);
        self.probabilistic_edges.get(&key)
    }

    /// Find all edges from an event
    pub fn outgoing_edges(&self, from: &str) -> Vec<&ProbabilisticEdge> {
        self.probabilistic_edges
            .values()
            .filter(|e| e.from == from)
            .collect()
    }

    /// Predict next events with uncertainty
    pub fn predict_with_uncertainty(&self, current_event: &str) -> Vec<ProbabilisticPrediction> {
        let outgoing = self.outgoing_edges(current_event);

        outgoing
            .iter()
            .map(|edge| {
                let ci = edge.confidence_interval(self.config.confidence_level);
                let uncertainty = self.inference.diagnose_uncertainty(edge);

                ProbabilisticPrediction {
                    event_type: edge.to.clone(),
                    probability: edge.probability,
                    confidence_interval: ci,
                    confidence_level: self.config.confidence_level,
                    causal_chain: vec![current_event.to_string(), edge.to.clone()],
                    uncertainty_source: uncertainty,
                    observations: edge.observations,
                }
            })
            .collect()
    }

    /// Propagate uncertainty through a chain
    pub fn propagate_chain_uncertainty(
        &self,
        chain: &[&ProbabilisticEdge],
    ) -> ProbabilisticResult {
        let mut samples = Vec::with_capacity(self.config.monte_carlo_samples);

        for _ in 0..self.config.monte_carlo_samples {
            let mut chain_prob = 1.0;
            for edge in chain {
                let p = self.inference.sample_beta(edge.alpha, edge.beta);
                chain_prob *= p;
            }
            samples.push(chain_prob);
        }

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / samples.len() as f64;
        let std_dev = variance.sqrt();

        // Compute 95% confidence interval
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lower_idx = ((1.0 - self.config.confidence_level) / 2.0 * sorted.len() as f64) as usize;
        let upper_idx = ((1.0 + self.config.confidence_level) / 2.0 * sorted.len() as f64) as usize;

        ProbabilisticResult {
            mean,
            std_dev,
            confidence_interval: (sorted[lower_idx], sorted[upper_idx.min(sorted.len() - 1)]),
            samples: samples.len(),
        }
    }

    /// Get underlying deterministic graph
    pub fn graph(&self) -> &CausalGraph {
        &self.graph
    }

    /// Get all probabilistic edges
    pub fn edges(&self) -> &HashMap<String, ProbabilisticEdge> {
        &self.probabilistic_edges
    }

    /// Remove all incoming edges to a node (for graph surgery in interventions)
    ///
    /// This is used in do-calculus to simulate interventions: do(X=x) removes
    /// all edges A → X, making X exogenous (not caused by anything in the system).
    ///
    /// # Example
    /// ```ignore
    /// // Before: A → X → Y
    /// graph.remove_incoming_edges("X");
    /// // After:  A   X → Y  (X is now exogenous)
    /// ```
    pub fn remove_incoming_edges(&mut self, node: &str) {
        // Remove all edges where "to" equals node
        self.probabilistic_edges.retain(|_key, edge| edge.to != node);
    }

    /// Remove all outgoing edges from a node (for neutralization interventions)
    ///
    /// This is used when we want to make a node causally inert, removing all its
    /// downstream effects.
    ///
    /// # Example
    /// ```ignore
    /// // Before: X → Y → Z
    /// graph.remove_outgoing_edges("X");
    /// // After:  X   Y → Z  (X no longer affects anything)
    /// ```
    pub fn remove_outgoing_edges(&mut self, node: &str) {
        // Remove all edges where "from" equals node
        self.probabilistic_edges.retain(|_key, edge| edge.from != node);
    }
}

impl Default for ProbabilisticCausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probabilistic_edge_creation() {
        let edge = ProbabilisticEdge::new(
            "A".to_string(),
            "B".to_string(),
            EdgeType::Direct,
            0.5,
            1.0,
        );

        assert_eq!(edge.from, "A");
        assert_eq!(edge.to, "B");
        assert_eq!(edge.probability, 0.5);
        assert_eq!(edge.observations, 0);
    }

    #[test]
    fn test_bayesian_update_converges() {
        let mut edge = ProbabilisticEdge::new(
            "A".to_string(),
            "B".to_string(),
            EdgeType::Direct,
            0.5,  // Prior: 50%
            1.0,
        );

        // Observe A → B 9 times, A → !B 1 time
        for _ in 0..9 {
            edge.update(true, true);  // A occurred, B followed
        }
        edge.update(true, false);  // A occurred, B didn't follow

        // Should converge to ~90%
        assert!(edge.probability > 0.85 && edge.probability < 0.95);
        assert_eq!(edge.observations, 10);
        assert_eq!(edge.to_given_from_count, 9);
    }

    #[test]
    fn test_confidence_increases_with_observations() {
        let mut edge = ProbabilisticEdge::new(
            "A".to_string(),
            "B".to_string(),
            EdgeType::Direct,
            0.5,
            1.0,
        );

        let initial_confidence = edge.confidence;

        // Add observations
        for _ in 0..20 {
            edge.update(true, true);
        }

        // Confidence should increase
        assert!(edge.confidence > initial_confidence);
    }

    #[test]
    fn test_confidence_interval() {
        let mut edge = ProbabilisticEdge::new(
            "A".to_string(),
            "B".to_string(),
            EdgeType::Direct,
            0.5,
            1.0,
        );

        // Add observations
        for _ in 0..50 {
            edge.update(true, true);
        }

        let (lower, upper) = edge.confidence_interval(0.95);

        // Should be tight interval around ~1.0
        assert!(lower > 0.8);
        assert!(upper <= 1.0);  // Upper bound can be exactly 1.0 for high confidence
        assert!(lower < upper);
    }

    #[test]
    fn test_probabilistic_graph_learning() {
        let mut graph = ProbabilisticCausalGraph::new();

        // Observe pattern: A → B (90% of time)
        for _ in 0..9 {
            graph.observe_edge("A", "B", EdgeType::Direct, true);
        }
        graph.observe_edge("A", "B", EdgeType::Direct, false);

        let edge = graph.edge_probability("A", "B").unwrap();
        assert!(edge.probability > 0.85 && edge.probability < 0.95);
    }

    #[test]
    fn test_prediction_with_uncertainty() {
        let mut graph = ProbabilisticCausalGraph::new();

        // Learn: A → B (high probability)
        for _ in 0..20 {
            graph.observe_edge("A", "B", EdgeType::Direct, true);
        }

        // Learn: A → C (low probability)
        for _ in 0..5 {
            graph.observe_edge("A", "C", EdgeType::Direct, true);
        }
        for _ in 0..15 {
            graph.observe_edge("A", "C", EdgeType::Direct, false);
        }

        let predictions = graph.predict_with_uncertainty("A");

        // Should have 2 predictions
        assert_eq!(predictions.len(), 2);

        // Find B prediction
        let b_pred = predictions.iter().find(|p| p.event_type == "B").unwrap();
        assert!(b_pred.probability > 0.9);
        assert_eq!(b_pred.observations, 20);
    }

    #[test]
    fn test_uncertainty_propagation_single_edge() {
        let mut graph = ProbabilisticCausalGraph::new();

        // Learn A → B
        for _ in 0..10 {
            graph.observe_edge("A", "B", EdgeType::Direct, true);
        }

        let edge = graph.edge_probability("A", "B").unwrap();
        let chain = vec![edge];

        let result = graph.propagate_chain_uncertainty(&chain);

        // Mean should be close to 1.0
        assert!(result.mean > 0.9);
        // Should have done monte carlo samples
        assert_eq!(result.samples, 1000);
    }

    #[test]
    fn test_uncertainty_source_diagnosis() {
        let config = ProbabilisticConfig::default();
        let inference = BayesianInference::new(config);

        // Edge with few observations
        let mut edge_small = ProbabilisticEdge::new(
            "A".to_string(),
            "B".to_string(),
            EdgeType::Direct,
            0.5,
            1.0,
        );
        edge_small.update(true, true);

        assert_eq!(
            inference.diagnose_uncertainty(&edge_small),
            UncertaintySource::SmallSampleSize
        );

        // Edge with many observations
        for _ in 0..20 {
            edge_small.update(true, true);
        }

        assert_eq!(
            inference.diagnose_uncertainty(&edge_small),
            UncertaintySource::WellEstimated
        );
    }
}
