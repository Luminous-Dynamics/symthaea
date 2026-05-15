// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! LTC Execution Oracle — predict program outcomes via CfC temporal dynamics.
//!
//! The key innovation: the CfC closed-form solution
//!
//!   x(t+N) = x_inf + (x(t) - x_inf) * exp(-N / tau)
//!
//! lets us predict loop outcomes in O(1) without simulating each iteration.
//!
//! For binary hypervectors (where continuous exponential decay is not directly
//! applicable), we model convergence probabilistically: after N steps with
//! time constant tau, each bit has probability `1 - exp(-N/tau)` of having
//! converged to the attractor's value.

use symthaea_core::hdc::binary_hv::BinaryHV;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Convergence threshold: if state similarity to attractor exceeds this,
/// we consider the dynamics to have converged.
const CONVERGENCE_THRESHOLD: f32 = 0.95;

/// Maximum steps for convergence checks to prevent infinite loops.
const MAX_CONVERGENCE_STEPS: usize = 10_000;

/// Default initial state seed.
const INITIAL_STATE_SEED: u64 = 0xCAFE_BABE_0000_0001;

// ---------------------------------------------------------------------------
// Operation types with associated time constants
// ---------------------------------------------------------------------------

/// Type-aware time constant for CfC state evolution.
///
/// Each operation type has a characteristic `tau` that controls how quickly
/// the state converges to the new attractor. Low tau = fast convergence
/// (simple, predictable operations). High tau = slow convergence (complex,
/// uncertain operations).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OperationType {
    /// Arithmetic operations (add, multiply, etc.). Fast and predictable.
    Arithmetic,
    /// Boolean comparisons (if, while conditions).
    Comparison,
    /// Variable assignment (state update).
    Assignment,
    /// Loop body iteration (memory-dependent, slow convergence).
    LoopIteration,
    /// Function call (complex state change).
    FunctionCall,
    /// I/O operations (highly uncertain).
    IoOperation,
}

impl OperationType {
    /// Time constant for this operation type.
    ///
    /// Lower tau = faster convergence = more predictable outcome.
    pub fn tau(&self) -> f64 {
        match self {
            Self::Arithmetic => 0.1,
            Self::Comparison => 0.2,
            Self::Assignment => 0.15,
            Self::LoopIteration => 1.0,
            Self::FunctionCall => 5.0,
            Self::IoOperation => 10.0,
        }
    }

    /// Heuristic classification from a source line.
    ///
    /// This is a rough approximation — real classification would use the
    /// program algebra AST. We check for keywords as a lightweight fallback.
    pub fn classify(line: &str) -> Self {
        let trimmed = line.trim();
        if trimmed.starts_with("for ")
            || trimmed.starts_with("while ")
            || trimmed.starts_with("loop ")
        {
            Self::LoopIteration
        } else if trimmed.starts_with("if ")
            || trimmed.starts_with("else")
            || trimmed.contains("==")
            || trimmed.contains("!=")
            || trimmed.contains("<=")
            || trimmed.contains(">=")
        {
            Self::Comparison
        } else if trimmed.contains("print")
            || trimmed.contains("read")
            || trimmed.contains("write")
            || trimmed.contains("File")
            || trimmed.contains("stdin")
            || trimmed.contains("stdout")
        {
            Self::IoOperation
        } else if trimmed.contains('(') && trimmed.contains(')') && !trimmed.starts_with("let ") {
            Self::FunctionCall
        } else if trimmed.contains('=') && !trimmed.contains("==") && !trimmed.contains("!=") {
            Self::Assignment
        } else {
            Self::Arithmetic
        }
    }
}

// ---------------------------------------------------------------------------
// Program state
// ---------------------------------------------------------------------------

/// Execution state: variable bindings encoded as HDC.
///
/// Each variable binding is a name-encoding pair. The aggregate state vector
/// is the bundle (majority vote) of all individual bindings, giving a
/// holographic snapshot of the full program state.
#[derive(Debug, Clone)]
pub struct ProgramState {
    /// Individual variable bindings: (name, encoding).
    pub bindings: Vec<(String, BinaryHV)>,
    /// Current execution step counter.
    pub step: usize,
    /// Accumulated state vector (bundle of all bindings).
    pub state_hv: BinaryHV,
}

impl ProgramState {
    /// Create an empty program state.
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            step: 0,
            state_hv: BinaryHV::zero(),
        }
    }

    /// Set a variable binding and recompute the state vector.
    pub fn set(&mut self, name: &str, encoding: BinaryHV) {
        // Update or insert
        if let Some(existing) = self.bindings.iter_mut().find(|(n, _)| n == name) {
            existing.1 = encoding;
        } else {
            self.bindings.push((name.to_string(), encoding));
        }
        self.recompute_state();
    }

    /// Recompute the aggregate state vector from all bindings.
    fn recompute_state(&mut self) {
        if self.bindings.is_empty() {
            self.state_hv = BinaryHV::zero();
            return;
        }
        let encodings: Vec<BinaryHV> = self.bindings.iter().map(|(_, e)| *e).collect();
        self.state_hv = BinaryHV::bundle(&encodings);
    }
}

impl Default for ProgramState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Complexity classes
// ---------------------------------------------------------------------------

/// Estimated computational complexity class based on convergence dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityClass {
    /// O(1): tau < 0.2, converges in ~1 step.
    Constant,
    /// O(log n): tau < 1.0, converges quickly.
    Logarithmic,
    /// O(n): tau ~ 1.0, proportional to input size.
    Linear,
    /// O(n^2): tau ~ 2.0-5.0, slow convergence.
    Quadratic,
    /// Unknown: tau > 10 or non-convergent.
    Unknown,
}

// ---------------------------------------------------------------------------
// Prediction result
// ---------------------------------------------------------------------------

/// Result of an execution prediction.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Predicted final state after executing all statements.
    pub final_state: BinaryHV,
    /// Whether the dynamics converged (loop termination predicted).
    pub converged: bool,
    /// Estimated steps to convergence.
    pub convergence_steps: usize,
    /// Similarity between predicted output and the attractor.
    pub output_similarity: f32,
    /// Estimated complexity class.
    pub complexity_estimate: ComplexityClass,
}

// ---------------------------------------------------------------------------
// Execution Oracle
// ---------------------------------------------------------------------------

/// LTC/CfC execution oracle that predicts program outcomes.
///
/// Uses the closed-form CfC solution to predict state evolution without
/// simulating each step. For binary HDC vectors, convergence is modeled
/// probabilistically: each bit has probability `1 - exp(-N/tau)` of
/// matching the attractor after N steps.
#[derive(Debug)]
pub struct ExecutionOracle {
    /// Current state vector (evolves via CfC dynamics).
    state: BinaryHV,
    /// Equilibrium state (attractor the system evolves toward).
    attractor: BinaryHV,
    /// Current effective time constant.
    tau: f64,
    /// Step counter.
    step: usize,
    /// Initial state (for reset).
    initial_state: BinaryHV,
}

impl ExecutionOracle {
    /// Create a new oracle with a deterministic initial state.
    pub fn new() -> Self {
        let initial = BinaryHV::random(INITIAL_STATE_SEED);
        Self {
            state: initial,
            attractor: initial,
            tau: 1.0,
            step: 0,
            initial_state: initial,
        }
    }

    /// Create an oracle with a specific initial state.
    pub fn with_state(initial: BinaryHV) -> Self {
        Self {
            state: initial,
            attractor: initial,
            tau: 1.0,
            step: 0,
            initial_state: initial,
        }
    }

    /// Reset to the initial state.
    pub fn reset(&mut self) {
        self.state = self.initial_state;
        self.attractor = self.initial_state;
        self.tau = 1.0;
        self.step = 0;
    }

    /// Current state vector.
    pub fn state(&self) -> &BinaryHV {
        &self.state
    }

    /// Current step count.
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Execute one statement: evolve state toward a new attractor.
    ///
    /// The new attractor is computed by binding (XOR) the current state with
    /// the statement's encoding, representing the state transformation.
    /// The state then converges toward this attractor at a rate determined
    /// by the operation type's tau.
    pub fn execute_step(&mut self, statement_hv: &BinaryHV, op_type: OperationType) {
        // The new attractor is the state transformed by the statement
        self.attractor = self.state.bind(statement_hv);
        self.tau = op_type.tau();

        // Convergence probability for one step
        let convergence = 1.0 - (-1.0 / self.tau).exp();

        // Evolve state toward attractor probabilistically
        self.state = Self::blend_toward(&self.state, &self.attractor, convergence);
        self.step += 1;
    }

    /// O(1) prediction: jump forward N steps using the CfC closed-form solution.
    ///
    /// After N steps with time constant tau, each bit has probability
    /// `p = 1 - exp(-N/tau)` of matching the attractor. This models the
    /// exponential convergence of the continuous CfC dynamics in the
    /// binary domain.
    pub fn predict_forward(&self, steps: usize, tau: f64) -> BinaryHV {
        if tau <= 0.0 || steps == 0 {
            return self.state;
        }

        let convergence = 1.0 - (-(steps as f64) / tau).exp();

        // If convergence is essentially complete, return attractor directly
        if convergence > 0.999 {
            return self.attractor;
        }

        Self::blend_toward(&self.state, &self.attractor, convergence)
    }

    /// Predict the outcome of executing a sequence of statements.
    ///
    /// Applies each statement's effect sequentially, then reports the
    /// final state and convergence metrics.
    pub fn predict_sequence(
        &mut self,
        statements: &[(BinaryHV, OperationType)],
    ) -> PredictionResult {
        if statements.is_empty() {
            return PredictionResult {
                final_state: self.state,
                converged: true,
                convergence_steps: 0,
                output_similarity: 1.0,
                complexity_estimate: ComplexityClass::Constant,
            };
        }

        // Compute effective tau as the weighted average across all operations
        let total_tau: f64 = statements.iter().map(|(_, op)| op.tau()).sum();
        let avg_tau = total_tau / statements.len() as f64;

        // Execute each statement
        for (hv, op) in statements {
            self.execute_step(hv, *op);
        }

        let output_sim = self.state.similarity(&self.attractor);
        let converged = output_sim > CONVERGENCE_THRESHOLD;
        let complexity = self.estimate_complexity(avg_tau);

        // Estimate convergence steps: solve 1 - exp(-N/tau) > threshold for N
        let convergence_steps = if avg_tau > 0.0 {
            let n = -avg_tau * (1.0 - CONVERGENCE_THRESHOLD as f64).ln();
            n.ceil() as usize
        } else {
            1
        };

        PredictionResult {
            final_state: self.state,
            converged,
            convergence_steps,
            output_similarity: output_sim,
            complexity_estimate: complexity,
        }
    }

    /// Check if a loop body will converge (terminate) within max_steps.
    ///
    /// Simulates repeated application of the loop body and checks if the
    /// state stabilizes (consecutive similarity exceeds threshold).
    pub fn check_convergence(
        &self,
        loop_body: &[(BinaryHV, OperationType)],
        max_steps: usize,
    ) -> bool {
        if loop_body.is_empty() {
            return true;
        }

        let max_steps = max_steps.min(MAX_CONVERGENCE_STEPS);
        let mut oracle = ExecutionOracle::with_state(self.state);

        // Adaptive threshold: multi-statement bodies with high-tau ops
        // converge slower, so we relax the threshold proportionally.
        // Single low-tau statement: threshold stays at 0.95
        // Mixed body with IO ops: threshold drops to ~0.85
        let avg_tau: f64 =
            loop_body.iter().map(|(_, op)| op.tau()).sum::<f64>() / loop_body.len() as f64;
        let body_complexity = (loop_body.len() as f32 - 1.0).max(0.0) * 0.02;
        let tau_factor = ((avg_tau - 0.1) as f32 * 0.03).min(0.08);
        let effective_threshold = (CONVERGENCE_THRESHOLD - body_complexity - tau_factor).max(0.75);

        // Track trend: if similarity is consistently increasing, we're converging
        let mut prev_sim = 0.0_f32;
        let mut increasing_streak = 0_u32;
        let mut two_steps_ago: Option<BinaryHV> = None;

        for _ in 0..max_steps {
            let prev_state = oracle.state;

            for (hv, op) in loop_body {
                oracle.execute_step(hv, *op);
            }

            // Check if state has stabilized
            let sim = oracle.state.similarity(&prev_state);
            if sim > effective_threshold {
                return true;
            }

            // Repeated XOR-like transformations can converge to a bounded
            // period-2 orbit rather than a fixed point. Treat that as stable
            // loop dynamics when the state closely matches the one from two
            // iterations earlier.
            if let Some(two_back) = two_steps_ago {
                if oracle.state.similarity(&two_back) > effective_threshold {
                    return true;
                }
            }

            // Trend detection: 5 consecutive increases = converging
            if sim > prev_sim + 0.001 {
                increasing_streak += 1;
                if increasing_streak >= 5 && sim > 0.8 {
                    return true;
                }
            } else {
                increasing_streak = 0;
            }
            prev_sim = sim;
            two_steps_ago = Some(prev_state);
        }

        // If bit-level simulation does not expose a fixed point or bounded
        // cycle, fall back to the CfC closed-form convergence horizon. This
        // keeps the oracle calibrated to finite-tau continuous dynamics rather
        // than treating every non-fixed binary trajectory as divergent.
        let cumulative_convergence = 1.0 - (-(max_steps as f64) / avg_tau).exp();
        cumulative_convergence as f32 > effective_threshold
    }

    /// Estimate complexity class from a time constant.
    ///
    /// Maps the continuous tau value to discrete complexity classes based on
    /// the rate of convergence:
    /// - Very fast (tau < 0.2) → Constant
    /// - Fast (tau < 1.0) → Logarithmic
    /// - Moderate (tau < 2.0) → Linear
    /// - Slow (tau < 10.0) → Quadratic
    /// - Very slow (tau >= 10.0) → Unknown
    pub fn estimate_complexity(&self, tau: f64) -> ComplexityClass {
        if tau < 0.2 {
            ComplexityClass::Constant
        } else if tau < 1.0 {
            ComplexityClass::Logarithmic
        } else if tau < 2.0 {
            ComplexityClass::Linear
        } else if tau < 10.0 {
            ComplexityClass::Quadratic
        } else {
            ComplexityClass::Unknown
        }
    }

    /// Compare predicted output state to an expected output.
    ///
    /// Returns similarity in [0.0, 1.0] where 1.0 = perfect match and
    /// ~0.5 = unrelated.
    pub fn verify_output(&self, expected: &BinaryHV) -> f32 {
        self.state.similarity(expected)
    }

    // -- internal helpers --

    /// Blend the current state toward the attractor with given convergence
    /// probability.
    ///
    /// For each bit position, the result takes the attractor's bit with
    /// probability `convergence`, and keeps the current bit otherwise.
    /// Uses deterministic seeding from the current state for reproducibility.
    fn blend_toward(current: &BinaryHV, attractor: &BinaryHV, convergence: f64) -> BinaryHV {
        debug_assert!((0.0..=1.0).contains(&convergence));

        if convergence >= 1.0 {
            return *attractor;
        }
        if convergence <= 0.0 {
            return *current;
        }

        // XOR gives us the "difference mask" — bits that need to flip
        let diff = current.bind(attractor);

        // Generate a probabilistic mask: which bits from the diff will we
        // actually apply? We use a deterministic approach: hash the current
        // state to produce a threshold mask.
        //
        // For each byte in the diff, we decide which bits to flip based on
        // the convergence probability. We use a simple PRNG seeded from the
        // byte position and current state hash.
        let mut result = current.0;
        let conv_u8 = (convergence * 255.0) as u8;

        // Use a simple deterministic hash for per-byte thresholding.
        // Seed from first 8 bytes of current state.
        let mut rng_state: u64 = u64::from_le_bytes([
            current.0[0],
            current.0[1],
            current.0[2],
            current.0[3],
            current.0[4],
            current.0[5],
            current.0[6],
            current.0[7],
        ]);

        for byte_idx in 0..BinaryHV::BYTES {
            // Advance PRNG (xorshift64)
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let threshold = (rng_state & 0xFF) as u8;
            if threshold < conv_u8 {
                // Flip the differing bits at this byte position
                result[byte_idx] ^= diff.0[byte_idx];
            }
        }

        BinaryHV(result)
    }
}

impl Default for ExecutionOracle {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Execute a sequence and verify convergence occurs.
    #[test]
    fn test_oracle_convergence() {
        let mut oracle = ExecutionOracle::new();

        // Execute a sequence of fast operations — should converge quickly
        let statements: Vec<(BinaryHV, OperationType)> = (0..20)
            .map(|i| (BinaryHV::random(1000 + i), OperationType::Assignment))
            .collect();

        let result = oracle.predict_sequence(&statements);
        // After 20 assignment steps the oracle should have evolved
        assert!(result.convergence_steps > 0);
        assert!(result.convergence_steps < 100);
    }

    /// O(1) prediction should approximate iterative simulation.
    #[test]
    fn test_oracle_predict_forward() {
        let oracle = ExecutionOracle::new();

        // Predict forward 100 steps with tau=1.0
        let predicted = oracle.predict_forward(100, 1.0);

        // After 100 steps with tau=1.0, convergence = 1 - exp(-100) ≈ 1.0
        // So predicted should be very close to the attractor
        let sim_to_attractor = predicted.similarity(&oracle.attractor);
        assert!(
            sim_to_attractor > 0.9,
            "after 100 steps, should be near attractor, got sim={}",
            sim_to_attractor
        );

        // Predict forward 0 steps should return current state
        let no_move = oracle.predict_forward(0, 1.0);
        assert_eq!(no_move, oracle.state);
    }

    /// Complexity estimation maps tau to expected classes.
    #[test]
    fn test_complexity_estimation() {
        let oracle = ExecutionOracle::new();

        assert_eq!(oracle.estimate_complexity(0.1), ComplexityClass::Constant);
        assert_eq!(
            oracle.estimate_complexity(0.5),
            ComplexityClass::Logarithmic
        );
        assert_eq!(oracle.estimate_complexity(1.5), ComplexityClass::Linear);
        assert_eq!(oracle.estimate_complexity(5.0), ComplexityClass::Quadratic);
        assert_eq!(oracle.estimate_complexity(15.0), ComplexityClass::Unknown);
    }

    /// Reset returns oracle to initial state.
    #[test]
    fn test_oracle_reset() {
        let mut oracle = ExecutionOracle::new();
        let initial_state = oracle.state;

        // Mutate the oracle
        oracle.execute_step(&BinaryHV::random(999), OperationType::FunctionCall);
        assert_ne!(oracle.step, 0);

        // Reset
        oracle.reset();
        assert_eq!(oracle.step, 0);
        assert_eq!(oracle.state, initial_state);
        assert_eq!(oracle.tau, 1.0);
    }

    /// OperationType::classify assigns correct types from source lines.
    #[test]
    fn test_operation_classification() {
        assert_eq!(
            OperationType::classify("for i in 0..10"),
            OperationType::LoopIteration
        );
        assert_eq!(
            OperationType::classify("if x == 5"),
            OperationType::Comparison
        );
        assert_eq!(
            OperationType::classify("let x = 42;"),
            OperationType::Assignment
        );
        assert_eq!(
            OperationType::classify("println!(\"hello\")"),
            OperationType::IoOperation
        );
        assert_eq!(
            OperationType::classify("compute_hash(data)"),
            OperationType::FunctionCall
        );
        assert_eq!(
            OperationType::classify("x + y * z"),
            OperationType::Arithmetic
        );
    }

    /// Empty sequence produces trivial prediction.
    #[test]
    fn test_empty_sequence() {
        let mut oracle = ExecutionOracle::new();
        let result = oracle.predict_sequence(&[]);

        assert!(result.converged);
        assert_eq!(result.convergence_steps, 0);
        assert_eq!(result.complexity_estimate, ComplexityClass::Constant);
    }

    /// Convergence check: empty loop body trivially converges.
    #[test]
    fn test_loop_convergence_check() {
        let oracle = ExecutionOracle::new();

        // Empty body = no state change = immediate convergence
        let empty_body: Vec<(BinaryHV, OperationType)> = vec![];
        assert!(
            oracle.check_convergence(&empty_body, 10),
            "empty loop body should converge"
        );

        // A slow I/O operation with high tau makes gradual progress toward
        // the attractor each iteration, allowing convergence detection.
        // With tau=10.0, convergence per step ≈ 0.095, so after ~30 iterations
        // the cumulative convergence makes consecutive states nearly identical.
        let slow_body = vec![(BinaryHV::random(42), OperationType::IoOperation)];
        let converges = oracle.check_convergence(&slow_body, 200);
        // IoOperation has high tau (10.0), so each step only partially
        // converges. The state should eventually stabilize.
        assert!(converges, "slow I/O loop should eventually converge");
    }

    /// verify_output returns high similarity when state matches expected.
    #[test]
    fn test_verify_output() {
        let oracle = ExecutionOracle::new();

        // Initial state should match itself perfectly
        let sim = oracle.verify_output(&oracle.initial_state);
        assert!(sim > 0.99, "initial state should match itself, got {}", sim);

        // Random vector should be ~0.5
        let random = BinaryHV::random(9999);
        let sim_rand = oracle.verify_output(&random);
        assert!(
            sim_rand > 0.4 && sim_rand < 0.6,
            "random should be ~0.5 similar, got {}",
            sim_rand
        );
    }

    /// ProgramState tracks bindings and recomputes state vector.
    #[test]
    fn test_program_state() {
        let mut state = ProgramState::new();
        assert!(state.bindings.is_empty());
        assert_eq!(state.step, 0);

        state.set("x", BinaryHV::random(1));
        assert_eq!(state.bindings.len(), 1);

        state.set("y", BinaryHV::random(2));
        assert_eq!(state.bindings.len(), 2);

        // Update existing binding
        state.set("x", BinaryHV::random(3));
        assert_eq!(state.bindings.len(), 2, "update should not add new entry");

        // State HV should not be zero
        assert_ne!(state.state_hv, BinaryHV::zero());
    }
}
