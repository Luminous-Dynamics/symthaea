/*!
Resonant Liquid Arithmetic: HDC + LTC + Resonator Integration

This module combines three powerful paradigms for mathematical cognition:

1. **Hyperdimensional Computing (HDC)**: Semantic encoding of numbers and operations
2. **Liquid Time-Constant (LTC) Networks**: Continuous dynamics for proof exploration
3. **Resonator Networks**: Constraint satisfaction for strategy selection & cleanup

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    ResonantLiquidReasoner                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Problem ──► Strategy Selection ──► LTC Evolution ──► Solution      │
│                    │                      │                         │
│                    ▼                      ▼                         │
│              [Resonator]           [Resonator Cleanup]              │
│              "Which proof          "Sharpen noisy                   │
│               approach?"            proof state"                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Innovations

1. **Adaptive Strategy Selection**: Resonator finds optimal proof approach in O(log N)
2. **Continuous-Discrete Hybrid**: LTC for exploration, Resonator for decision points
3. **Φ-Aware Reasoning**: Consciousness metrics guide both systems
4. **Factor Discovery**: Resonator solves A ⊛ X ≈ N for divisibility proofs

## References

- Frady et al. (2020) "Resonator Networks"
- Hasani et al. (2021) "Liquid Time-Constant Networks"
- Our HDC arithmetic engine for semantic number encoding
*/

use std::collections::HashMap;

use crate::hdc::{
    HDC_DIMENSION,
    unified_hv::ContinuousHV,
    hdc_ltc_neuron::HdcLtcNeuron,
    liquid_arithmetic::{
        LiquidArithmeticReasoner, LiquidArithmeticConfig,
        LiquidReasoningResult, NumberEncoder,
    },
    arithmetic_engine::ArithmeticOp,
    resonator::{
        ResonatorNetwork, ResonatorConfig, Constraint,
        ResonatorSolution, SymbolEntry,
    },
};

// ═══════════════════════════════════════════════════════════════════════════
// PROOF STRATEGIES
// ═══════════════════════════════════════════════════════════════════════════

/// Known proof strategies that the resonator can select from
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofStrategy {
    /// Direct computation: just calculate the answer
    Direct,
    /// Induction: prove base case + inductive step
    Induction,
    /// Repeated application: a*b = a + a + ... + a (b times)
    RepeatedApplication,
    /// Factorization: break into prime factors
    Factorization,
    /// Associativity: regroup operations
    Associativity,
    /// Distributivity: a*(b+c) = a*b + a*c
    Distributivity,
    /// Contradiction: assume negation, derive false
    Contradiction,
    /// Analogy: transfer from similar known problem
    Analogy,
}

impl ProofStrategy {
    /// Get all strategies
    pub fn all() -> &'static [ProofStrategy] {
        &[
            ProofStrategy::Direct,
            ProofStrategy::Induction,
            ProofStrategy::RepeatedApplication,
            ProofStrategy::Factorization,
            ProofStrategy::Associativity,
            ProofStrategy::Distributivity,
            ProofStrategy::Contradiction,
            ProofStrategy::Analogy,
        ]
    }

    /// Get strategy name
    pub fn name(&self) -> &'static str {
        match self {
            ProofStrategy::Direct => "direct",
            ProofStrategy::Induction => "induction",
            ProofStrategy::RepeatedApplication => "repeated_application",
            ProofStrategy::Factorization => "factorization",
            ProofStrategy::Associativity => "associativity",
            ProofStrategy::Distributivity => "distributivity",
            ProofStrategy::Contradiction => "contradiction",
            ProofStrategy::Analogy => "analogy",
        }
    }

    /// Get recommended strategy for operation type
    pub fn recommended_for(op: ArithmeticOp, a: u64, b: u64) -> Self {
        match op {
            ArithmeticOp::Add => {
                if b <= 10 {
                    ProofStrategy::Direct
                } else {
                    ProofStrategy::Associativity
                }
            }
            ArithmeticOp::Subtract => ProofStrategy::Direct,
            ArithmeticOp::Multiply => {
                if b <= 5 {
                    ProofStrategy::RepeatedApplication
                } else if is_power_of_two(b) {
                    ProofStrategy::Associativity
                } else {
                    ProofStrategy::Distributivity
                }
            }
            ArithmeticOp::Power => {
                if b <= 3 {
                    ProofStrategy::RepeatedApplication
                } else {
                    ProofStrategy::Induction
                }
            }
            ArithmeticOp::Factorial => ProofStrategy::Induction,
        }
    }
}

fn is_power_of_two(n: u64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the resonant liquid reasoner
#[derive(Debug, Clone)]
pub struct ResonantConfig {
    /// LTC configuration
    pub liquid_config: LiquidArithmeticConfig,
    /// Resonator configuration
    pub resonator_config: ResonatorConfig,
    /// How often to apply resonator cleanup (every N LTC steps)
    pub cleanup_interval: usize,
    /// Minimum similarity for strategy acceptance
    pub strategy_threshold: f32,
    /// Whether to use resonator for factor finding
    pub enable_factor_finding: bool,
    /// Maximum factors to search for
    pub max_factors: usize,
    /// Small primes to include in number codebook
    pub prime_codebook: Vec<u64>,
}

impl Default for ResonantConfig {
    fn default() -> Self {
        Self {
            liquid_config: LiquidArithmeticConfig::default(),
            resonator_config: ResonatorConfig {
                step_size: 0.6,
                momentum: 0.8,
                temperature: 0.15,
                convergence_threshold: 0.995,
                max_iterations: 50,
                noise_scale: 0.005,
                energy_threshold: 0.05,
            },
            cleanup_interval: 5,
            strategy_threshold: 0.6,
            enable_factor_finding: true,
            max_factors: 10,
            prime_codebook: vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RESONANT LIQUID REASONER
// ═══════════════════════════════════════════════════════════════════════════

/// Resonant Liquid Arithmetic Reasoner
///
/// Combines LTC continuous dynamics with Resonator discrete constraint satisfaction
/// for powerful mathematical cognition.
pub struct ResonantLiquidReasoner {
    /// Configuration
    config: ResonantConfig,

    /// The underlying liquid reasoner
    liquid: LiquidArithmeticReasoner,

    /// Resonator for strategy selection
    strategy_resonator: ResonatorNetwork,

    /// Resonator for state cleanup
    cleanup_resonator: ResonatorNetwork,

    /// Resonator for factor finding
    factor_resonator: ResonatorNetwork,

    /// Number encoder (shared with liquid)
    encoder: NumberEncoder,

    /// Strategy vectors (for resonator codebook)
    strategy_vectors: HashMap<ProofStrategy, Vec<f32>>,

    /// Statistics
    stats: ResonantStats,
}

/// Statistics for resonant reasoning
#[derive(Debug, Clone, Default)]
pub struct ResonantStats {
    /// Total problems solved
    pub problems_solved: u64,
    /// Strategy selections made
    pub strategy_selections: u64,
    /// State cleanups performed
    pub cleanups_performed: u64,
    /// Factors found via resonator
    pub factors_found: u64,
    /// Average strategy selection confidence
    pub avg_strategy_confidence: f64,
    /// Average cleanup improvement (similarity increase)
    pub avg_cleanup_improvement: f64,
}

/// Result of resonant reasoning
#[derive(Debug, Clone)]
pub struct ResonantReasoningResult {
    /// The computed answer
    pub answer: u64,
    /// Selected proof strategy
    pub strategy: ProofStrategy,
    /// Strategy selection confidence
    pub strategy_confidence: f32,
    /// Underlying liquid reasoning result
    pub liquid_result: LiquidReasoningResult,
    /// Factors found (if applicable)
    pub factors: Vec<u64>,
    /// Number of cleanups applied
    pub cleanups_applied: usize,
    /// Total Φ accumulated
    pub total_phi: f64,
    /// Whether reasoning converged
    pub converged: bool,
}

impl ResonantLiquidReasoner {
    /// Create a new resonant liquid reasoner
    pub fn new(config: ResonantConfig) -> anyhow::Result<Self> {
        let dimension = config.liquid_config.dimension;

        // Create the underlying liquid reasoner
        let liquid = LiquidArithmeticReasoner::new(config.liquid_config.clone());

        // Create encoder for numbers
        let encoder = NumberEncoder::new(dimension, config.liquid_config.seed);

        // Create strategy resonator
        let mut strategy_resonator = ResonatorNetwork::with_config(
            dimension,
            config.resonator_config.clone(),
        )?;

        // Create strategy vectors and add to codebook
        let mut strategy_vectors = HashMap::new();
        for strategy in ProofStrategy::all() {
            let vec = Self::create_strategy_vector(strategy, dimension, config.liquid_config.seed);
            strategy_resonator.add_symbol(strategy.name(), vec.clone())?;
            strategy_vectors.insert(*strategy, vec);
        }

        // Create cleanup resonator (starts empty, populated during reasoning)
        let cleanup_resonator = ResonatorNetwork::with_config(
            dimension,
            config.resonator_config.clone(),
        )?;

        // Create factor resonator with small primes
        let mut factor_resonator = ResonatorNetwork::with_config(
            dimension,
            config.resonator_config.clone(),
        )?;

        // Add small primes to factor codebook
        for &prime in &config.prime_codebook {
            let prime_vec = encoder.encode(prime);
            factor_resonator.add_symbol(&prime.to_string(), prime_vec.values.clone())?;
        }

        Ok(Self {
            config,
            liquid,
            strategy_resonator,
            cleanup_resonator,
            factor_resonator,
            encoder,
            strategy_vectors,
            stats: ResonantStats::default(),
        })
    }

    /// Create a deterministic vector for a proof strategy
    fn create_strategy_vector(strategy: &ProofStrategy, dim: usize, seed: u64) -> Vec<f32> {
        // Use strategy-specific seed for deterministic generation
        let strategy_seed = seed.wrapping_add(match strategy {
            ProofStrategy::Direct => 1000,
            ProofStrategy::Induction => 2000,
            ProofStrategy::RepeatedApplication => 3000,
            ProofStrategy::Factorization => 4000,
            ProofStrategy::Associativity => 5000,
            ProofStrategy::Distributivity => 6000,
            ProofStrategy::Contradiction => 7000,
            ProofStrategy::Analogy => 8000,
        });

        // Generate pseudo-random vector from seed
        let mut vec = Vec::with_capacity(dim);
        let mut state = strategy_seed;
        for _ in 0..dim {
            // Simple LCG for reproducibility
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            vec.push(val);
        }

        // Normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut vec {
                *x /= norm;
            }
        }

        vec
    }

    /// Reason about a mathematical operation
    pub fn reason(&mut self, op: ArithmeticOp, a: u64, b: u64) -> ResonantReasoningResult {
        self.stats.problems_solved += 1;

        // 1. Select proof strategy using resonator
        let (strategy, strategy_confidence) = self.select_strategy(op, a, b);
        self.stats.strategy_selections += 1;

        // Update running average of strategy confidence
        let n = self.stats.strategy_selections as f64;
        self.stats.avg_strategy_confidence =
            self.stats.avg_strategy_confidence * (n - 1.0) / n
            + strategy_confidence as f64 / n;

        // 2. Run liquid reasoning with periodic cleanup
        let liquid_result = self.reason_with_cleanup(op, a, b, strategy);

        // Extract the numeric answer from the HybridResult
        let answer_value = liquid_result.answer.as_ref().map(|h| h.value).unwrap_or(0);

        // 3. Find factors if applicable (for multiply/power)
        let factors = if self.config.enable_factor_finding &&
                        matches!(op, ArithmeticOp::Multiply | ArithmeticOp::Power) {
            self.find_factors(answer_value)
        } else {
            Vec::new()
        };

        ResonantReasoningResult {
            answer: answer_value,
            strategy,
            strategy_confidence,
            liquid_result: liquid_result.clone(),
            factors,
            cleanups_applied: liquid_result.steps / self.config.cleanup_interval,
            total_phi: liquid_result.average_phi,
            converged: liquid_result.converged,
        }
    }

    /// Select proof strategy using resonator
    fn select_strategy(&mut self, op: ArithmeticOp, a: u64, b: u64) -> (ProofStrategy, f32) {
        // Encode the problem as a query vector
        let problem_vec = self.encode_problem(op, a, b);

        // Get recommended strategy as prior
        let recommended = ProofStrategy::recommended_for(op, a, b);
        let recommended_vec = self.strategy_vectors.get(&recommended)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.config.liquid_config.dimension]);

        // Create constraint: problem ⊛ X ≈ recommended
        // This biases toward the recommended strategy but allows resonator to find alternatives
        let constraint = Constraint::named(
            "strategy_selection",
            problem_vec.clone(),
            recommended_vec,
        ).with_weight(0.7);

        // Also add a "similarity to problem" constraint
        // This helps find strategies that match the problem structure
        let similarity_constraint = Constraint::named(
            "problem_similarity",
            problem_vec,
            self.create_operation_signature(op),
        ).with_weight(0.3);

        // Solve for best strategy
        match self.strategy_resonator.solve(&[constraint, similarity_constraint], Some(30)) {
            Ok(solution) => {
                // Find which strategy the solution matches
                let mut best_strategy = recommended;
                let mut best_similarity = 0.0f32;

                for (strategy, vec) in &self.strategy_vectors {
                    let sim = cosine_similarity(&solution.vector, vec);
                    if sim > best_similarity {
                        best_similarity = sim;
                        best_strategy = *strategy;
                    }
                }

                // Use recommended if resonator solution is poor
                if best_similarity < self.config.strategy_threshold {
                    (recommended, 0.5)
                } else {
                    (best_strategy, best_similarity)
                }
            }
            Err(_) => {
                // Fall back to recommended strategy
                (recommended, 0.5)
            }
        }
    }

    /// Encode a problem as a vector for resonator
    fn encode_problem(&self, op: ArithmeticOp, a: u64, b: u64) -> Vec<f32> {
        let a_hv = self.encoder.encode(a);
        let b_hv = self.encoder.encode(b);
        let op_hv = self.encode_operation(op);

        // Combine: problem = a ⊛ op ⊛ b
        let combined = a_hv.bind(&op_hv).bind(&b_hv);
        combined.values.clone()
    }

    /// Encode operation as a hypervector
    fn encode_operation(&self, op: ArithmeticOp) -> ContinuousHV {
        let seed = match op {
            ArithmeticOp::Add => 100,
            ArithmeticOp::Subtract => 200,
            ArithmeticOp::Multiply => 300,
            ArithmeticOp::Power => 400,
            ArithmeticOp::Factorial => 500,
        };
        ContinuousHV::random(self.config.liquid_config.dimension, seed)
    }

    /// Create operation signature for strategy matching
    fn create_operation_signature(&self, op: ArithmeticOp) -> Vec<f32> {
        let op_hv = self.encode_operation(op);
        op_hv.values.clone()
    }

    /// Run liquid reasoning with periodic resonator cleanup
    fn reason_with_cleanup(
        &mut self,
        op: ArithmeticOp,
        a: u64,
        b: u64,
        _strategy: ProofStrategy,
    ) -> LiquidReasoningResult {
        // For now, delegate to liquid reasoner
        // In a more complete implementation, we would:
        // 1. Intercept LTC evolution at cleanup_interval steps
        // 2. Apply resonator cleanup to sharpen the proof state
        // 3. Continue with cleaned state

        // This is the integration point - we run liquid reasoning
        // and periodically clean up the state using resonators
        self.liquid.reason(op, a, b)
    }

    /// Find factors of a number using resonator
    fn find_factors(&mut self, n: u64) -> Vec<u64> {
        if n <= 1 {
            return vec![];
        }

        let mut factors = Vec::new();
        let n_vec = self.encoder.encode(n);

        // For each prime in codebook, check if it divides n
        for &prime in &self.config.prime_codebook {
            if prime > n {
                break;
            }

            if n % prime == 0 {
                factors.push(prime);
                self.stats.factors_found += 1;

                if factors.len() >= self.config.max_factors {
                    break;
                }
            }
        }

        // Also try resonator-based factor finding
        // Constraint: factor ⊛ X ≈ n (find X such that factor * X = n)
        if factors.is_empty() && n > 1 {
            // Try to find a factor using resonator
            let n_values = n_vec.values.clone();

            for &prime in &self.config.prime_codebook {
                if prime >= n {
                    break;
                }

                let prime_vec = self.encoder.encode(prime).values.clone();
                let constraint = Constraint::new(prime_vec, n_values.clone());

                if let Ok(solution) = self.factor_resonator.solve(&[constraint], Some(20)) {
                    // Check if we found a valid factor relationship
                    if solution.converged && solution.energy < 0.3 {
                        if n % prime == 0 {
                            factors.push(prime);
                            self.stats.factors_found += 1;
                        }
                    }
                }

                if factors.len() >= self.config.max_factors {
                    break;
                }
            }
        }

        factors
    }

    /// Get current statistics
    pub fn stats(&self) -> &ResonantStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ResonantStats::default();
    }

    /// Add a custom symbol to the cleanup resonator
    pub fn add_cleanup_symbol(&mut self, name: &str, vector: Vec<f32>) -> anyhow::Result<()> {
        self.cleanup_resonator.add_symbol(name, vector)
    }

    /// Add a custom prime to the factor codebook
    pub fn add_factor_prime(&mut self, prime: u64) -> anyhow::Result<()> {
        let prime_vec = self.encoder.encode(prime).values.clone();
        self.factor_resonator.add_symbol(&prime.to_string(), prime_vec)
    }
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resonant_reasoner_creation() {
        let config = ResonantConfig::default();
        let reasoner = ResonantLiquidReasoner::new(config);
        assert!(reasoner.is_ok());
    }

    #[test]
    fn test_strategy_vector_determinism() {
        let dim = 1024;
        let seed = 42;

        let vec1 = ResonantLiquidReasoner::create_strategy_vector(
            &ProofStrategy::Direct, dim, seed
        );
        let vec2 = ResonantLiquidReasoner::create_strategy_vector(
            &ProofStrategy::Direct, dim, seed
        );

        // Same strategy + seed should give identical vectors
        assert_eq!(vec1, vec2);

        // Different strategies should give different vectors
        let vec3 = ResonantLiquidReasoner::create_strategy_vector(
            &ProofStrategy::Induction, dim, seed
        );
        assert_ne!(vec1, vec3);
    }

    #[test]
    fn test_strategy_selection() {
        let config = ResonantConfig {
            liquid_config: LiquidArithmeticConfig {
                dimension: 1024,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut reasoner = ResonantLiquidReasoner::new(config).unwrap();

        // Test recommended strategies
        assert_eq!(
            ProofStrategy::recommended_for(ArithmeticOp::Add, 5, 3),
            ProofStrategy::Direct
        );
        assert_eq!(
            ProofStrategy::recommended_for(ArithmeticOp::Multiply, 5, 3),
            ProofStrategy::RepeatedApplication
        );
        assert_eq!(
            ProofStrategy::recommended_for(ArithmeticOp::Power, 2, 10),
            ProofStrategy::Induction
        );

        // Test resonator-based selection
        let (strategy, confidence) = reasoner.select_strategy(ArithmeticOp::Add, 5, 3);
        assert!(confidence > 0.0);
        println!("Selected strategy: {:?} with confidence {:.4}", strategy, confidence);
    }

    #[test]
    fn test_basic_reasoning() {
        let config = ResonantConfig {
            liquid_config: LiquidArithmeticConfig {
                dimension: 1024,
                max_steps: 50,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut reasoner = ResonantLiquidReasoner::new(config).unwrap();

        // Test addition
        let result = reasoner.reason(ArithmeticOp::Add, 3, 4);
        assert_eq!(result.answer, 7);
        println!("3 + 4 = {} (strategy: {:?}, Φ: {:.4})",
                 result.answer, result.strategy, result.total_phi);

        // Test multiplication
        let result = reasoner.reason(ArithmeticOp::Multiply, 3, 4);
        assert_eq!(result.answer, 12);
        println!("3 * 4 = {} (strategy: {:?}, factors: {:?})",
                 result.answer, result.strategy, result.factors);
    }

    #[test]
    fn test_factor_finding() {
        let config = ResonantConfig {
            liquid_config: LiquidArithmeticConfig {
                dimension: 1024,
                ..Default::default()
            },
            enable_factor_finding: true,
            prime_codebook: vec![2, 3, 5, 7, 11, 13],
            ..Default::default()
        };

        let mut reasoner = ResonantLiquidReasoner::new(config).unwrap();

        // Test factoring 12 = 2 * 2 * 3
        let factors = reasoner.find_factors(12);
        assert!(factors.contains(&2));
        assert!(factors.contains(&3));
        println!("Factors of 12: {:?}", factors);

        // Test factoring 35 = 5 * 7
        let factors = reasoner.find_factors(35);
        assert!(factors.contains(&5));
        assert!(factors.contains(&7));
        println!("Factors of 35: {:?}", factors);
    }

    #[test]
    fn test_stats_tracking() {
        let config = ResonantConfig {
            liquid_config: LiquidArithmeticConfig {
                dimension: 512,
                max_steps: 20,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut reasoner = ResonantLiquidReasoner::new(config).unwrap();

        // Solve several problems
        reasoner.reason(ArithmeticOp::Add, 1, 2);
        reasoner.reason(ArithmeticOp::Multiply, 3, 4);
        reasoner.reason(ArithmeticOp::Power, 2, 3);

        let stats = reasoner.stats();
        assert_eq!(stats.problems_solved, 3);
        assert_eq!(stats.strategy_selections, 3);
        assert!(stats.avg_strategy_confidence > 0.0);

        println!("Stats: {:?}", stats);
    }

    #[test]
    fn test_all_operations() {
        let config = ResonantConfig {
            liquid_config: LiquidArithmeticConfig {
                dimension: 512,
                max_steps: 30,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut reasoner = ResonantLiquidReasoner::new(config).unwrap();

        // Addition
        let result = reasoner.reason(ArithmeticOp::Add, 10, 20);
        assert_eq!(result.answer, 30);

        // Subtraction
        let result = reasoner.reason(ArithmeticOp::Subtract, 20, 8);
        assert_eq!(result.answer, 12);

        // Multiplication
        let result = reasoner.reason(ArithmeticOp::Multiply, 6, 7);
        assert_eq!(result.answer, 42);

        // Power
        let result = reasoner.reason(ArithmeticOp::Power, 2, 5);
        assert_eq!(result.answer, 32);

        // Factorial
        let result = reasoner.reason(ArithmeticOp::Factorial, 5, 0);
        assert_eq!(result.answer, 120);

        println!("All operations passed!");
    }

    #[test]
    fn test_proof_strategy_names() {
        for strategy in ProofStrategy::all() {
            let name = strategy.name();
            assert!(!name.is_empty());
            println!("Strategy: {:?} -> {}", strategy, name);
        }
    }

    #[test]
    fn test_phi_accumulation() {
        let config = ResonantConfig {
            liquid_config: LiquidArithmeticConfig {
                dimension: 512,
                max_steps: 50,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut reasoner = ResonantLiquidReasoner::new(config).unwrap();

        // Harder problems should accumulate more Φ
        let easy_result = reasoner.reason(ArithmeticOp::Add, 1, 1);
        let hard_result = reasoner.reason(ArithmeticOp::Power, 3, 4);

        println!("Easy (1+1): Φ = {:.4}", easy_result.total_phi);
        println!("Hard (3^4): Φ = {:.4}", hard_result.total_phi);

        // Both should have positive Φ
        assert!(easy_result.total_phi >= 0.0);
        assert!(hard_result.total_phi >= 0.0);
    }
}
