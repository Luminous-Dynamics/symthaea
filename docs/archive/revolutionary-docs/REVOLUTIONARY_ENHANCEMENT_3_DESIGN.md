# 🎲 Revolutionary Enhancement #3: Probabilistic Inference - Design Document

**Date**: December 25, 2025
**Status**: 🎯 **DESIGN PHASE**
**Estimated Effort**: 600-800 lines + 8-10 tests

---

## Vision

Transform Symthaea's causal understanding from **deterministic** to **probabilistic**, enabling:
- Quantified uncertainty in causal relationships
- Confidence intervals for predictions
- Bayesian learning from observations
- Graceful handling of noisy/missing data

### The Gap We're Filling

**Current State** (After Enhancements #1 & #2):
- ✅ Real-time causal graph construction
- ✅ Pattern recognition with confidence scores
- ❌ All edges are deterministic (binary: exists or doesn't exist)
- ❌ No uncertainty quantification in predictions
- ❌ Cannot handle missing or contradictory evidence

**After Enhancement #3**:
- ✅ Probabilistic causal edges: P(effect|cause) = 0.85
- ✅ Uncertainty propagation through chains
- ✅ Confidence intervals: "80% likely with ±10% margin"
- ✅ Bayesian learning: Update probabilities as evidence accumulates
- ✅ Robust to noise and incomplete data

---

## Core Innovations

### Innovation #1: Probabilistic Causal Edges

**Concept**: Instead of "A causes B", we say "A causes B with probability P"

**Implementation**:
```rust
pub struct ProbabilisticEdge {
    pub from: String,
    pub to: String,
    pub probability: f64,           // P(to | from) ∈ [0, 1]
    pub confidence: f64,            // How certain we are about this probability
    pub observations: usize,        // Number of times observed
    pub edge_type: EdgeType,        // Parent, Temporal, etc.
}
```

**Example**:
```
security_check → phi_measurement (P=0.92, confidence=0.85, n=150)
```
Meaning: "When security_check occurs, phi_measurement follows 92% of the time, and we're 85% confident in this estimate based on 150 observations"

### Innovation #2: Bayesian Belief Propagation

**Concept**: Update probabilities as new evidence arrives

**Algorithm**:
```rust
// Prior: P(effect | cause) = 0.5 (unknown)
// Observe: cause → effect (positive evidence)
// Posterior: P(effect | cause) = update_belief(prior, evidence)

fn update_belief(prior: f64, observed: bool, confidence: f64) -> f64 {
    // Bayesian update with Beta distribution
    let alpha = prior * confidence;
    let beta = (1.0 - prior) * confidence;

    if observed {
        (alpha + 1.0) / (alpha + beta + 2.0)
    } else {
        alpha / (alpha + beta + 2.0)
    }
}
```

### Innovation #3: Uncertainty Propagation

**Concept**: Compute uncertainty through causal chains

**Problem**: Given chain A → B → C with probabilities:
- P(B|A) = 0.9 ± 0.05
- P(C|B) = 0.8 ± 0.10

What is P(C|A)?

**Solution**: Monte Carlo uncertainty propagation
```rust
fn propagate_uncertainty(chain: &[ProbabilisticEdge]) -> (f64, f64) {
    let samples = 1000;
    let mut results = Vec::new();

    for _ in 0..samples {
        let mut prob = 1.0;
        for edge in chain {
            // Sample from distribution
            let p = sample_beta(edge.probability, edge.confidence);
            prob *= p;
        }
        results.push(prob);
    }

    let mean = results.iter().sum::<f64>() / samples as f64;
    let std = compute_std(&results);

    (mean, std)  // P(C|A) = mean ± std
}
```

### Innovation #4: Confidence Intervals for Predictions

**Concept**: Instead of "router_selection will happen next", say "router_selection will happen next with 85% probability (CI: 75%-95%)"

**Implementation**:
```rust
pub struct ProbabilisticPrediction {
    pub event_type: String,
    pub probability: f64,           // Point estimate
    pub confidence_interval: (f64, f64),  // (lower, upper)
    pub confidence_level: f64,      // e.g., 0.95 for 95% CI
    pub causal_chain: Vec<String>,
    pub uncertainty_source: UncertaintySource,
}

pub enum UncertaintySource {
    SmallSampleSize,    // Not enough data
    HighVariance,       // Inconsistent observations
    LongChain,          // Many edges compound uncertainty
    MissingData,        // Incomplete evidence
}
```

---

## Architecture

### Module Structure

```
src/observability/
├── probabilistic_graph.rs (NEW)
│   ├── ProbabilisticCausalGraph
│   ├── ProbabilisticEdge
│   └── EdgeProbability
│
├── bayesian_inference.rs (NEW)
│   ├── BayesianInference
│   ├── BeliefUpdate
│   └── PriorDistribution
│
├── uncertainty_propagation.rs (NEW)
│   ├── UncertaintyPropagator
│   ├── MonteCarloSampler
│   └── ConfidenceInterval
│
└── streaming_causal.rs (EXTEND)
    └── Add probabilistic mode
```

### Integration with Enhancements #1 & #2

**Enhancement #1 (Streaming)**:
- Collect observations in real-time
- Update edge probabilities incrementally

**Enhancement #2 (Patterns)**:
- Probabilistic pattern matching
- "Pattern X occurs with probability P ± σ"

**Enhancement #3 (This)**:
- Add probability layer to streaming graph
- Quantify uncertainty in pattern matches

---

## API Design

### Core Types

```rust
/// Probabilistic extension of CausalGraph
pub struct ProbabilisticCausalGraph {
    /// Underlying deterministic graph
    graph: CausalGraph,

    /// Probabilistic edges (parallel to graph.edges)
    probabilistic_edges: Vec<ProbabilisticEdge>,

    /// Bayesian inference engine
    inference: BayesianInference,

    /// Configuration
    config: ProbabilisticConfig,
}

pub struct ProbabilisticEdge {
    /// Edge ID (matches CausalEdge)
    pub id: String,

    /// From and to event IDs
    pub from: String,
    pub to: String,

    /// P(to | from) - conditional probability
    pub probability: f64,

    /// Confidence in this probability estimate
    pub confidence: f64,

    /// Number of observations
    pub observations: usize,

    /// Number of times "from" occurred
    pub from_count: usize,

    /// Number of times "to" followed "from"
    pub to_given_from_count: usize,

    /// Beta distribution parameters (for Bayesian updates)
    pub alpha: f64,  // Successes
    pub beta: f64,   // Failures

    /// Edge type
    pub edge_type: EdgeType,
}

pub struct ProbabilisticConfig {
    /// Prior for unknown edges (default: 0.5)
    pub prior_probability: f64,

    /// Confidence in prior (default: 1.0)
    pub prior_confidence: f64,

    /// Minimum observations before trusting probability
    pub min_observations: usize,

    /// Enable Bayesian updates
    pub enable_bayesian_learning: bool,

    /// Monte Carlo samples for uncertainty propagation
    pub monte_carlo_samples: usize,
}
```

### Bayesian Inference Engine

```rust
pub struct BayesianInference {
    /// Prior distributions for edge types
    priors: HashMap<EdgeType, BetaDistribution>,

    /// Learning rate (how quickly to update beliefs)
    learning_rate: f64,
}

pub struct BetaDistribution {
    pub alpha: f64,  // Shape parameter (successes)
    pub beta: f64,   // Shape parameter (failures)
}

impl BayesianInference {
    /// Update edge probability with new observation
    pub fn update_edge(
        &self,
        edge: &mut ProbabilisticEdge,
        observed: bool,
    ) {
        if observed {
            edge.alpha += 1.0;
            edge.to_given_from_count += 1;
        } else {
            edge.beta += 1.0;
        }
        edge.from_count += 1;
        edge.observations += 1;

        // Update probability (MAP estimate)
        edge.probability = (edge.alpha - 1.0) / (edge.alpha + edge.beta - 2.0);

        // Update confidence (inverse variance)
        let variance = (edge.alpha * edge.beta) /
                      ((edge.alpha + edge.beta).powi(2) * (edge.alpha + edge.beta + 1.0));
        edge.confidence = 1.0 / (1.0 + variance);
    }

    /// Get confidence interval for edge probability
    pub fn confidence_interval(
        &self,
        edge: &ProbabilisticEdge,
        level: f64,  // e.g., 0.95
    ) -> (f64, f64) {
        // Use beta distribution quantiles
        let alpha = (1.0 - level) / 2.0;
        let lower = beta_quantile(edge.alpha, edge.beta, alpha);
        let upper = beta_quantile(edge.alpha, edge.beta, 1.0 - alpha);
        (lower, upper)
    }
}
```

### Uncertainty Propagation

```rust
pub struct UncertaintyPropagator {
    /// Monte Carlo sampler
    sampler: MonteCarloSampler,

    /// Number of samples
    num_samples: usize,
}

impl UncertaintyPropagator {
    /// Propagate uncertainty through causal chain
    pub fn propagate_chain(
        &self,
        edges: &[ProbabilisticEdge],
    ) -> ProbabilisticResult {
        let mut samples = Vec::with_capacity(self.num_samples);

        for _ in 0..self.num_samples {
            let mut chain_prob = 1.0;
            for edge in edges {
                // Sample from beta distribution
                let p = self.sampler.sample_beta(edge.alpha, edge.beta);
                chain_prob *= p;
            }
            samples.push(chain_prob);
        }

        ProbabilisticResult {
            mean: mean(&samples),
            std_dev: std_dev(&samples),
            confidence_interval: percentile_interval(&samples, 0.95),
            samples: samples.len(),
        }
    }

    /// Predict next event with uncertainty
    pub fn predict_with_uncertainty(
        &self,
        graph: &ProbabilisticCausalGraph,
        current_event: &str,
    ) -> Vec<ProbabilisticPrediction> {
        // Find all edges from current_event
        let outgoing = graph.find_outgoing_edges(current_event);

        let mut predictions = Vec::new();
        for edge in outgoing {
            // Compute uncertainty
            let (mean, std) = self.propagate_chain(&[edge.clone()]);
            let ci = self.confidence_interval(&edge, 0.95);

            predictions.push(ProbabilisticPrediction {
                event_type: edge.to.clone(),
                probability: mean,
                confidence_interval: ci,
                confidence_level: 0.95,
                causal_chain: vec![current_event.to_string(), edge.to.clone()],
                uncertainty_source: self.diagnose_uncertainty(&edge),
            });
        }

        predictions
    }

    fn diagnose_uncertainty(&self, edge: &ProbabilisticEdge) -> UncertaintySource {
        if edge.observations < 10 {
            UncertaintySource::SmallSampleSize
        } else if edge.confidence < 0.5 {
            UncertaintySource::HighVariance
        } else {
            UncertaintySource::WellEstimated
        }
    }
}
```

---

## Use Cases

### Use Case #1: Learning Causal Probabilities

```rust
let mut graph = ProbabilisticCausalGraph::new();

// Observe events over time
for event in event_stream {
    graph.observe_event(event);

    // Graph automatically learns:
    // - Which events follow others
    // - How frequently
    // - With what probability
}

// Query learned probabilities
let prob = graph.edge_probability("security_check", "phi_measurement");
println!("P(phi_measurement | security_check) = {:.2} ± {:.2}",
         prob.mean, prob.std_dev);
// Output: P(phi_measurement | security_check) = 0.92 ± 0.08
```

### Use Case #2: Predictions with Confidence Intervals

```rust
let predictions = graph.predict_with_uncertainty("security_check");

for pred in predictions {
    println!("{}: {:.1}% (95% CI: {:.1}%-{:.1}%)",
             pred.event_type,
             pred.probability * 100.0,
             pred.confidence_interval.0 * 100.0,
             pred.confidence_interval.1 * 100.0);
}

// Output:
// phi_measurement: 92.0% (95% CI: 85.0%-97.0%)
// router_selection: 15.0% (95% CI: 8.0%-25.0%)
```

### Use Case #3: Uncertainty-Aware Pattern Detection

```rust
// Combine with Enhancement #2
let pattern_match = library.match_sequence_probabilistic(&events);

for m in pattern_match {
    println!("Pattern: {} (P={:.1}%, uncertainty={})",
             m.motif.name,
             m.probability * 100.0,
             m.uncertainty_source);
}

// Output:
// Pattern: Normal Consciousness Flow (P=87.5%, uncertainty=SmallSampleSize)
// Pattern: Degraded Consciousness (P=12.3%, uncertainty=HighVariance)
```

### Use Case #4: Anomaly Detection via Probability

```rust
// Current event: phi_measurement
// Expected next: router_selection with P=0.9

let actual = "error_event";
let expected_prob = graph.edge_probability("phi_measurement", "router_selection");
let actual_prob = graph.edge_probability("phi_measurement", actual);

if actual_prob.mean < 0.1 {
    println!("ANOMALY: {} is very unlikely after phi_measurement (P={:.1}%)",
             actual, actual_prob.mean * 100.0);
}
```

---

## Implementation Plan

### Phase 1: Core Data Structures (200 lines)
- [ ] ProbabilisticEdge struct
- [ ] ProbabilisticCausalGraph struct
- [ ] ProbabilisticConfig
- [ ] Basic edge probability calculation

### Phase 2: Bayesian Inference (150 lines)
- [ ] BayesianInference engine
- [ ] Beta distribution support
- [ ] Belief update algorithm
- [ ] Confidence interval calculation

### Phase 3: Uncertainty Propagation (200 lines)
- [ ] UncertaintyPropagator
- [ ] Monte Carlo sampling
- [ ] Chain probability computation
- [ ] Uncertainty source diagnosis

### Phase 4: Integration with Streaming (100 lines)
- [ ] Extend StreamingCausalAnalyzer
- [ ] Real-time probability updates
- [ ] Probabilistic insights generation

### Phase 5: Testing (100 lines, 8-10 tests)
- [ ] Test Bayesian updates
- [ ] Test uncertainty propagation
- [ ] Test confidence intervals
- [ ] Test integration with existing code

---

## Success Metrics

### Correctness Metrics
- ✅ Probabilities converge to true values with enough observations
- ✅ Confidence intervals contain true value 95% of the time (for 95% CI)
- ✅ Bayesian updates follow correct posterior distributions
- ✅ All tests passing (8-10 comprehensive tests)

### Performance Metrics
- ✅ Edge probability update: <0.1ms per observation
- ✅ Uncertainty propagation: <5ms for 10-edge chain
- ✅ Monte Carlo sampling: <10ms for 1000 samples
- ✅ Memory overhead: ~100 bytes per probabilistic edge

### Integration Metrics
- ✅ Zero breaking changes to Enhancements #1 & #2
- ✅ Can be disabled (fallback to deterministic mode)
- ✅ Seamless with existing CausalGraph API
- ✅ Enhances pattern matching with uncertainty

---

## Risks and Mitigations

### Risk #1: Computational Complexity

**Problem**: Monte Carlo sampling is expensive
**Mitigation**:
- Cache results for common queries
- Use analytical solutions where possible (e.g., single edge)
- Adjustable sample count (trade accuracy for speed)

### Risk #2: Cold Start Problem

**Problem**: No data initially, so all probabilities are prior (0.5)
**Mitigation**:
- Informed priors based on event types
- Rapid learning (Bayesian updates converge quickly)
- Indicate uncertainty clearly ("based on 3 observations")

### Risk #3: Overfitting with Small Data

**Problem**: High confidence with few observations
**Mitigation**:
- Regularization (minimum sample size)
- Confidence decays with small sample size
- Explicit uncertainty quantification

---

## Future Extensions (Not in Scope)

### Extension #1: Causal Discovery
Learn causal structure (which edges exist) from data using constraint-based or score-based algorithms

### Extension #2: Conditional Probabilities
P(C | A, B) - multiple causes

### Extension #3: Temporal Dynamics
P(B | A, t) - probability changes over time

### Extension #4: Counterfactual Reasoning
"What would have happened if A didn't occur?"

---

## Conclusion

Revolutionary Enhancement #3 will transform Symthaea from deterministic causal reasoning to **probabilistic causal inference**, enabling:

1. **Quantified Uncertainty**: Know HOW certain we are
2. **Bayesian Learning**: Update beliefs as evidence accumulates
3. **Confidence Intervals**: Predictions with error bars
4. **Robust to Noise**: Handle incomplete/contradictory data gracefully

This is the foundation for truly **intelligent causal reasoning** - not just detecting patterns, but understanding the **probability** of those patterns and the **uncertainty** in our knowledge.

**Estimated Implementation**: 650-800 lines + 8-10 tests
**Expected Integration**: Seamless with Enhancements #1 & #2
**Revolutionary Impact**: HIGH - enables scientific rigor in consciousness analysis

---

**Ready to implement!** 🚀

*Designed with rigor. Architected for elegance. Ready to revolutionize.*
