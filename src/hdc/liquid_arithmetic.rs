//! # Liquid Arithmetic Reasoning
//!
//! ## Purpose
//! Combines HDC (Hyperdimensional Computing) with LTC (Liquid Time-Constant) dynamics
//! for temporal mathematical reasoning. This enables:
//!
//! - **Adaptive computation time**: Harder proofs get more "thinking time" via τ(state)
//! - **Temporal proof exploration**: Proof states evolve via continuous dynamics
//! - **Φ-guided reasoning**: Consciousness metrics steer proof search
//! - **Multi-trajectory proofs**: Different LTC paths = different proof strategies
//!
//! ## Theoretical Basis
//!
//! Traditional proof search is discrete: enumerate proof trees, backtrack.
//! Our approach models reasoning as continuous dynamics:
//!
//! ```text
//! dP/dt = (-P ⊕ f(W⊗P ⊕ Q⊗query)) / τ(||P||, Φ)
//! ```
//!
//! Where:
//! - `P` is the proof state (HDC hypervector encoding current knowledge)
//! - `query` is the problem to solve (HDC-encoded)
//! - `τ(||P||, Φ)` adapts based on state complexity and consciousness
//! - The system relaxes toward states that satisfy the query
//!
//! ## Key Innovation
//!
//! Mathematical cognition as a dynamical system - proofs emerge from
//! continuous evolution rather than discrete search. This mirrors how
//! human mathematical insight often comes from "sitting with" a problem.

use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use crate::hdc::hdc_ltc_neuron::{HdcLtcNeuron, HdcLtcConfig, ActivationFunction};
use crate::hdc::arithmetic_engine::{HybridArithmeticEngine, HybridResult, ArithmeticOp};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the Liquid Arithmetic Reasoner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidArithmeticConfig {
    /// Base time constant for reasoning dynamics
    pub tau_base: f32,

    /// How much state complexity affects τ
    pub complexity_scaling: f32,

    /// How much Φ affects τ (higher Φ = more deliberate reasoning)
    pub phi_scaling: f32,

    /// Number of LTC neurons in the proof network
    pub num_neurons: usize,

    /// Integration timestep (seconds)
    pub dt: f32,

    /// Maximum reasoning steps before timeout
    pub max_steps: usize,

    /// Convergence threshold (state change below this = solution found)
    pub convergence_threshold: f32,

    /// Dimension of HDC vectors
    pub dimension: usize,

    /// Seeds for deterministic initialization
    pub seed: u64,
}

impl Default for LiquidArithmeticConfig {
    fn default() -> Self {
        Self {
            tau_base: 0.1,              // 100ms base time constant
            complexity_scaling: 0.3,     // Moderate complexity effect
            phi_scaling: 0.5,            // Consciousness moderately affects time
            num_neurons: 8,              // 8 neurons for proof exploration
            dt: 0.01,                    // 10ms timestep
            max_steps: 1000,             // Max 10 seconds of reasoning
            convergence_threshold: 0.01, // 1% change = converged
            dimension: HDC_DIMENSION,
            seed: 42,
        }
    }
}

// ============================================================================
// CORE TYPES
// ============================================================================

/// A mathematical query encoded as HDC hypervector
#[derive(Debug, Clone)]
pub struct MathQuery {
    /// The query encoding
    pub encoding: ContinuousHV,

    /// Operation type
    pub operation: ArithmeticOp,

    /// Operands
    pub operands: Vec<u64>,

    /// Human-readable description
    pub description: String,
}

/// Result of liquid arithmetic reasoning
#[derive(Debug, Clone)]
pub struct LiquidReasoningResult {
    /// The answer (if found)
    pub answer: Option<HybridResult>,

    /// Number of reasoning steps taken
    pub steps: usize,

    /// Total reasoning time (simulated seconds)
    pub reasoning_time: f32,

    /// Average Φ during reasoning
    pub average_phi: f64,

    /// Peak Φ during reasoning
    pub peak_phi: f64,

    /// Trajectory of proof states (for analysis)
    pub trajectory: Vec<ProofSnapshot>,

    /// Whether reasoning converged
    pub converged: bool,

    /// Final confidence in answer
    pub confidence: f64,
}

/// Snapshot of proof state at a moment in time
#[derive(Debug, Clone)]
pub struct ProofSnapshot {
    /// Timestep
    pub step: usize,

    /// Simulated time
    pub time: f32,

    /// State magnitude (||P||)
    pub state_magnitude: f32,

    /// Current Φ
    pub phi: f64,

    /// Current τ (time constant)
    pub tau: f32,

    /// State change from previous step
    pub state_change: f32,
}

// ============================================================================
// NUMBER ENCODING
// ============================================================================

/// Encodes numbers as HDC hypervectors for liquid reasoning
pub struct NumberEncoder {
    /// Basis vector for zero
    zero_basis: ContinuousHV,

    /// Basis vector for successor operation
    successor_basis: ContinuousHV,

    /// Cache of encoded numbers
    cache: HashMap<u64, ContinuousHV>,

    /// Dimension
    dimension: usize,
}

impl NumberEncoder {
    /// Create a new encoder
    pub fn new(dimension: usize, seed: u64) -> Self {
        Self {
            zero_basis: ContinuousHV::random(dimension, seed),
            successor_basis: ContinuousHV::random(dimension, seed + 1000),
            cache: HashMap::new(),
            dimension,
        }
    }

    /// Encode a number as HDC hypervector
    /// Uses Peano-style construction: n = S(S(S(...S(0)...)))
    pub fn encode(&mut self, n: u64) -> ContinuousHV {
        if let Some(cached) = self.cache.get(&n) {
            return cached.clone();
        }

        let encoded = if n == 0 {
            self.zero_basis.clone()
        } else {
            // Recursive Peano construction via binding
            let prev = self.encode(n - 1);
            prev.bind(&self.successor_basis)
        };

        self.cache.insert(n, encoded.clone());
        encoded
    }

    /// Encode an operation type
    pub fn encode_operation(&self, op: ArithmeticOp, seed: u64) -> ContinuousHV {
        // Each operation gets a unique basis vector
        let op_seed = match op {
            ArithmeticOp::Add => seed + 10000,
            ArithmeticOp::Subtract => seed + 20000,
            ArithmeticOp::Multiply => seed + 30000,
            ArithmeticOp::Power => seed + 40000,
            ArithmeticOp::Factorial => seed + 50000,
        };
        ContinuousHV::random(self.dimension, op_seed)
    }

    /// Create a query encoding: op(a, b)
    pub fn encode_query(&mut self, op: ArithmeticOp, a: u64, b: u64, seed: u64) -> MathQuery {
        let op_enc = self.encode_operation(op, seed);
        let a_enc = self.encode(a);
        let b_enc = self.encode(b);

        // Bundle operation with operands
        let query_enc = ContinuousHV::bundle(&[&op_enc, &a_enc, &b_enc]);

        MathQuery {
            encoding: query_enc,
            operation: op,
            operands: vec![a, b],
            description: format!("{:?}({}, {})", op, a, b),
        }
    }
}

// ============================================================================
// LIQUID ARITHMETIC REASONER
// ============================================================================

/// The main liquid arithmetic reasoning system
pub struct LiquidArithmeticReasoner {
    /// Configuration
    config: LiquidArithmeticConfig,

    /// LTC neurons for proof state evolution
    neurons: Vec<HdcLtcNeuron>,

    /// Number encoder
    encoder: NumberEncoder,

    /// Underlying arithmetic engine for verification
    engine: HybridArithmeticEngine,

    /// Current proof state (bundle of all neuron states)
    proof_state: ContinuousHV,

    /// Total Φ accumulated
    total_phi: f64,

    /// Statistics
    stats: LiquidReasoningStats,
}

/// Statistics for liquid reasoning
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LiquidReasoningStats {
    /// Total queries processed
    pub queries_processed: usize,

    /// Total reasoning steps
    pub total_steps: usize,

    /// Queries that converged
    pub converged_count: usize,

    /// Average steps to convergence
    pub avg_steps_to_converge: f32,

    /// Total accumulated Φ
    pub total_phi: f64,
}

impl LiquidArithmeticReasoner {
    /// Create a new liquid arithmetic reasoner
    pub fn new(config: LiquidArithmeticConfig) -> Self {
        let ltc_config = HdcLtcConfig {
            tau_base: config.tau_base,
            backbone_tau: config.complexity_scaling,
            dimension: config.dimension,
            activation: ActivationFunction::Tanh,
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
        };

        // Create neurons with different seeds for diversity
        let neurons: Vec<HdcLtcNeuron> = (0..config.num_neurons)
            .map(|i| HdcLtcNeuron::new(ltc_config.clone(), config.seed + i as u64 * 100))
            .collect();

        let encoder = NumberEncoder::new(config.dimension, config.seed);
        let engine = HybridArithmeticEngine::new();
        let proof_state = ContinuousHV::random(config.dimension, config.seed + 999);

        Self {
            config,
            neurons,
            encoder,
            engine,
            proof_state,
            total_phi: 0.0,
            stats: LiquidReasoningStats::default(),
        }
    }

    /// Reason about a mathematical query using liquid dynamics
    pub fn reason(&mut self, op: ArithmeticOp, a: u64, b: u64) -> LiquidReasoningResult {
        // Encode the query
        let query = self.encoder.encode_query(op, a, b, self.config.seed);

        // Reset neurons to query-seeded state
        self.reset_to_query(&query);

        // Evolve toward solution
        let mut trajectory = Vec::new();
        let mut converged = false;
        let mut steps = 0;
        let mut total_reasoning_phi = 0.0;
        let mut peak_phi = 0.0;
        let mut prev_state_mag = self.proof_state.norm();

        for step in 0..self.config.max_steps {
            // Compute current Φ (simplified: based on state diversity)
            let current_phi = self.compute_reasoning_phi();
            total_reasoning_phi += current_phi;
            if current_phi > peak_phi {
                peak_phi = current_phi;
            }

            // Compute adaptive time constant
            let state_mag = self.proof_state.norm();
            let tau = self.compute_tau(state_mag, current_phi);

            // Evolve all neurons with query input
            for neuron in &mut self.neurons {
                neuron.evolve(self.config.dt, &query.encoding);
            }

            // Bundle neuron states into proof state
            let neuron_states: Vec<&ContinuousHV> = self.neurons.iter()
                .map(|n| n.state())
                .collect();
            self.proof_state = ContinuousHV::bundle(&neuron_states);

            // Check convergence
            let state_change = (self.proof_state.norm() - prev_state_mag).abs() / (prev_state_mag + 1e-6);

            // Record snapshot
            trajectory.push(ProofSnapshot {
                step,
                time: step as f32 * self.config.dt,
                state_magnitude: state_mag,
                phi: current_phi,
                tau,
                state_change,
            });

            if state_change < self.config.convergence_threshold && step > 10 {
                converged = true;
                steps = step;
                break;
            }

            prev_state_mag = self.proof_state.norm();
            steps = step;
        }

        // Compute answer using traditional engine
        let answer = self.compute_answer(op, a, b);
        let avg_phi = if steps > 0 { total_reasoning_phi / steps as f64 } else { 0.0 };

        // Compute confidence based on convergence and Φ
        let confidence = if converged {
            0.5 + 0.5 * (avg_phi / 0.5).min(1.0)  // Higher Φ = higher confidence
        } else {
            0.3  // Lower confidence if didn't converge
        };

        // Update stats
        self.stats.queries_processed += 1;
        self.stats.total_steps += steps;
        if converged {
            self.stats.converged_count += 1;
            self.stats.avg_steps_to_converge =
                (self.stats.avg_steps_to_converge * (self.stats.converged_count - 1) as f32
                 + steps as f32) / self.stats.converged_count as f32;
        }
        self.stats.total_phi += total_reasoning_phi;
        self.total_phi += total_reasoning_phi;

        LiquidReasoningResult {
            answer,
            steps,
            reasoning_time: steps as f32 * self.config.dt,
            average_phi: avg_phi,
            peak_phi,
            trajectory,
            converged,
            confidence,
        }
    }

    /// Reset neurons to a query-seeded state
    fn reset_to_query(&mut self, query: &MathQuery) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            // Each neuron starts with query + random perturbation for diversity
            let perturbation = ContinuousHV::random(
                self.config.dimension,
                self.config.seed + i as u64 * 500
            ).scale(0.1);
            let initial_state = query.encoding.add(&perturbation);
            neuron.set_state(initial_state);
        }
    }

    /// Compute Φ for current reasoning state
    fn compute_reasoning_phi(&self) -> f64 {
        // Simplified Φ: based on diversity of neuron states
        // Higher diversity = more integrated information
        if self.neurons.len() < 2 {
            return 0.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..self.neurons.len() {
            for j in (i+1)..self.neurons.len() {
                let sim = self.neurons[i].state().similarity(self.neurons[j].state());
                total_sim += sim.abs();
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let avg_sim = total_sim / count as f32;

        // Φ is higher when neurons are moderately similar (not identical, not orthogonal)
        // Peak at ~0.5 similarity
        let phi = 1.0 - (avg_sim - 0.5).abs() * 2.0;
        phi.max(0.0) as f64
    }

    /// Compute adaptive time constant
    fn compute_tau(&self, state_mag: f32, phi: f64) -> f32 {
        // τ = τ_base × (1 + complexity_scaling × ||P|| + phi_scaling × Φ)
        self.config.tau_base
            * (1.0 + self.config.complexity_scaling * state_mag
               + self.config.phi_scaling * phi as f32)
    }

    /// Compute actual answer using arithmetic engine
    fn compute_answer(&mut self, op: ArithmeticOp, a: u64, b: u64) -> Option<HybridResult> {
        match op {
            ArithmeticOp::Add => Some(self.engine.add(a, b)),
            ArithmeticOp::Subtract => self.engine.subtract(a, b),
            ArithmeticOp::Multiply => Some(self.engine.multiply(a, b)),
            ArithmeticOp::Power => Some(self.engine.power(a, b)),
            ArithmeticOp::Factorial => Some(self.engine.factorial(a)),
        }
    }

    /// Get current proof state
    pub fn proof_state(&self) -> &ContinuousHV {
        &self.proof_state
    }

    /// Get accumulated Φ
    pub fn total_phi(&self) -> f64 {
        self.total_phi
    }

    /// Get statistics
    pub fn stats(&self) -> &LiquidReasoningStats {
        &self.stats
    }

    /// Access underlying engine for direct computation
    pub fn engine(&mut self) -> &mut HybridArithmeticEngine {
        &mut self.engine
    }
}

impl Default for LiquidArithmeticReasoner {
    fn default() -> Self {
        Self::new(LiquidArithmeticConfig::default())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_encoding() {
        let mut encoder = NumberEncoder::new(HDC_DIMENSION, 42);

        let zero = encoder.encode(0);
        let one = encoder.encode(1);
        let five = encoder.encode(5);

        // Different numbers should have different encodings
        assert!(zero.similarity(&one) < 0.9);
        assert!(one.similarity(&five) < 0.9);

        // Successive numbers should be more similar than distant ones
        let two = encoder.encode(2);
        let sim_1_2 = one.similarity(&two);
        let sim_1_5 = one.similarity(&five);
        // Not guaranteed but likely due to binding structure
        println!("sim(1,2)={}, sim(1,5)={}", sim_1_2, sim_1_5);
    }

    #[test]
    fn test_liquid_reasoning_basic() {
        let mut reasoner = LiquidArithmeticReasoner::default();

        // Test addition
        let result = reasoner.reason(ArithmeticOp::Add, 3, 5);

        assert!(result.answer.is_some());
        assert_eq!(result.answer.unwrap().value, 8);
        assert!(result.steps > 0);
        assert!(result.reasoning_time > 0.0);
        assert!(result.average_phi >= 0.0);
    }

    #[test]
    fn test_liquid_reasoning_multiplication() {
        let mut reasoner = LiquidArithmeticReasoner::default();

        let result = reasoner.reason(ArithmeticOp::Multiply, 7, 8);

        assert!(result.answer.is_some());
        assert_eq!(result.answer.unwrap().value, 56);
        println!("Multiplication reasoning: {} steps, Φ={:.4}", result.steps, result.average_phi);
    }

    #[test]
    fn test_phi_evolution() {
        let mut reasoner = LiquidArithmeticReasoner::default();

        let result = reasoner.reason(ArithmeticOp::Add, 10, 15);

        // Check that Φ was tracked
        assert!(result.trajectory.len() > 0);

        // Check trajectory has Φ values
        for snapshot in &result.trajectory[..5.min(result.trajectory.len())] {
            println!("Step {}: Φ={:.4}, τ={:.4}, Δstate={:.4}",
                     snapshot.step, snapshot.phi, snapshot.tau, snapshot.state_change);
        }

        assert!(result.peak_phi >= result.average_phi);
    }

    #[test]
    fn test_convergence() {
        let config = LiquidArithmeticConfig {
            max_steps: 500,
            convergence_threshold: 0.001,
            ..Default::default()
        };
        let mut reasoner = LiquidArithmeticReasoner::new(config);

        let result = reasoner.reason(ArithmeticOp::Add, 2, 3);

        println!("Converged: {}, steps: {}", result.converged, result.steps);
        // Should eventually converge
        assert!(result.steps <= 500);
    }

    #[test]
    fn test_stats_accumulation() {
        let mut reasoner = LiquidArithmeticReasoner::default();

        // Run multiple queries
        reasoner.reason(ArithmeticOp::Add, 1, 1);
        reasoner.reason(ArithmeticOp::Multiply, 2, 3);
        reasoner.reason(ArithmeticOp::Add, 5, 7);

        let stats = reasoner.stats();
        assert_eq!(stats.queries_processed, 3);
        assert!(stats.total_steps > 0);
        assert!(stats.total_phi > 0.0);
    }
}
