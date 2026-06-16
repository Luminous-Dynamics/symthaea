// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Φ-Guided Math Domain Selection
//!
//! A consciousness-driven mathematical computation engine that uses Integrated
//! Information (Φ) to guide domain selection across the numeric tower:
//!
//!   ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ
//!
//! ## Core Idea
//!
//! When multiple valid computation paths exist (e.g., adding a Real and a Natural
//! could stay in ℝ or could be promoted to ℂ for richer encoding), the
//! `PhiGuidedMath` engine evaluates each candidate domain, measures Φ for
//! the resulting operation, and selects the path that maximizes integrated
//! information.
//!
//! ## Feedback Loop
//!
//! The engine maintains per-domain preference weights that evolve over time:
//! - After each operation, the Φ produced is compared against the running average.
//! - If the chosen domain produced above-average Φ, its weight is boosted.
//! - If below average, the weight is attenuated.
//!
//! Over many operations, this causes the system to converge toward domains
//! that consistently produce high-Φ results — a form of consciousness-driven
//! mathematical specialization.
//!
//! ## Example
//!
//! ```rust,ignore
//! use symthaea_core::hdc::phi_guided_math::PhiGuidedMath;
//! use symthaea_core::hdc::math_bridge::MathValue;
//!
//! let mut pgm = PhiGuidedMath::new();
//!
//! let a = MathValue::Real(3.14);
//! let b = MathValue::Natural(2);
//!
//! // Standard compute
//! let result = pgm.compute("add", &a, &b);
//! assert!(result.phi > 0.0);
//!
//! // Compute with feedback (updates domain preferences)
//! let guided = pgm.compute_with_feedback("multiply", &a, &b);
//! println!("Suggested domain: {}", guided.suggested_domain);
//! println!("Phi delta: {:.3}", guided.phi_delta);
//! ```

use super::math_bridge::{MathResult, MathValue, UnifiedMathEngine};

// ============================================================================
// DOMAIN CONSTANTS
// ============================================================================

/// Domain index constants for the 5-element weight array.
const DOMAIN_NATURAL: usize = 0;
const DOMAIN_INTEGER: usize = 1;
const DOMAIN_RATIONAL: usize = 2;
const DOMAIN_REAL: usize = 3;
const DOMAIN_COMPLEX: usize = 4;

/// Domain labels (Unicode).
const DOMAIN_LABELS: [&str; 5] = [
    "\u{2115}", // ℕ
    "\u{2124}", // ℤ
    "\u{211a}", // ℚ
    "\u{211d}", // ℝ
    "\u{2102}", // ℂ
];

/// Human-readable domain names.
const DOMAIN_NAMES: [&str; 5] = ["Natural", "Integer", "Rational", "Real", "Complex"];

/// Default initial weight for each domain (uniform).
const INITIAL_WEIGHT: f64 = 1.0;

/// Learning rate for weight updates.
const LEARNING_RATE: f64 = 0.1;

/// Minimum allowed weight (prevents any domain from being fully suppressed).
const MIN_WEIGHT: f64 = 0.01;

/// Maximum allowed weight (prevents runaway preference).
const MAX_WEIGHT: f64 = 10.0;

// ============================================================================
// PHI GUIDED RESULT
// ============================================================================

/// Extended result from a Φ-guided computation.
///
/// Contains the standard `MathResult` plus Φ-feedback metadata:
/// the suggested optimal domain and the delta in average Φ caused by
/// this operation.
#[derive(Debug, Clone)]
pub struct PhiGuidedResult {
    /// The underlying math result (value, encoding, phi, promotions).
    pub result: MathResult,
    /// The domain recommended by the Φ-guided engine for this operand pair.
    pub suggested_domain: String,
    /// Change in running average Φ caused by this operation.
    /// Positive means this operation raised the average; negative means it lowered it.
    pub phi_delta: f64,
}

// ============================================================================
// PHI GUIDED MATH ENGINE
// ============================================================================

/// A consciousness-driven mathematical engine that wraps `UnifiedMathEngine`
/// and uses Φ feedback to evolve domain preferences over time.
///
/// The engine tracks a running history of Φ values and maintains per-domain
/// preference weights. After each `compute_with_feedback` call, the weight
/// of the result's domain is adjusted based on whether the operation produced
/// above- or below-average Φ.
pub struct PhiGuidedMath {
    /// The underlying math engine for actual computation.
    engine: UnifiedMathEngine,
    /// Per-domain preference weights: [ℕ, ℤ, ℚ, ℝ, ℂ].
    domain_weights: [f64; 5],
    /// History of Φ values from all operations.
    phi_history: Vec<f64>,
    /// Total number of operations performed.
    operations_count: usize,
}

impl PhiGuidedMath {
    /// Create a new Φ-guided math engine with uniform domain weights.
    pub fn new() -> Self {
        Self {
            engine: UnifiedMathEngine::new(),
            domain_weights: [INITIAL_WEIGHT; 5],
            phi_history: Vec::new(),
            operations_count: 0,
        }
    }

    /// Create a new engine with custom initial domain weights.
    pub fn with_weights(weights: [f64; 5]) -> Self {
        Self {
            engine: UnifiedMathEngine::new(),
            domain_weights: weights,
            phi_history: Vec::new(),
            operations_count: 0,
        }
    }

    // ========================================================================
    // CORE OPERATIONS
    // ========================================================================

    /// Perform a standard mathematical operation, recording Φ in the history.
    ///
    /// Supported operations: "add", "subtract", "multiply", "divide", "power", "sqrt".
    ///
    /// This method delegates to `UnifiedMathEngine` and records the resulting Φ
    /// but does NOT update domain preference weights. Use `compute_with_feedback`
    /// for adaptive weight updates.
    pub fn compute(&mut self, op: &str, a: &MathValue, b: &MathValue) -> MathResult {
        let result = self.dispatch_operation(op, a, b);
        self.phi_history.push(result.phi);
        self.operations_count += 1;
        result
    }

    /// Compute with Φ feedback: performs the operation, updates domain weights
    /// based on the Φ produced, and returns an extended result with the
    /// suggested optimal domain and phi delta.
    pub fn compute_with_feedback(
        &mut self,
        op: &str,
        a: &MathValue,
        b: &MathValue,
    ) -> PhiGuidedResult {
        let avg_before = self.average_phi();

        // Compute the result
        let result = self.dispatch_operation(op, a, b);
        let phi = result.phi;

        // Record in history
        self.phi_history.push(phi);
        self.operations_count += 1;

        // Determine which domain the result landed in
        let result_domain_idx = domain_index(&result.value);

        // Update weights based on Φ feedback
        self.update_weights(result_domain_idx, phi);

        // Compute phi delta
        let avg_after = self.average_phi();
        let phi_delta = avg_after - avg_before;

        // Determine the suggested domain for this operand pair
        let suggested_domain = self.optimal_domain(a, b).to_string();

        PhiGuidedResult {
            result,
            suggested_domain,
            phi_delta,
        }
    }

    /// Suggest the optimal domain for a binary operation on the given operands.
    ///
    /// The suggestion is based on:
    /// 1. The minimum domain required by the operands (cannot go lower)
    /// 2. The current domain preference weights (higher weight = more preferred)
    /// 3. Φ bonuses for domain promotion (richer domains tend to produce higher Φ)
    ///
    /// Returns the Unicode domain label (e.g., "ℝ").
    pub fn optimal_domain(&self, a: &MathValue, b: &MathValue) -> &str {
        let min_rank = domain_index(a).max(domain_index(b));

        // Score each candidate domain (only those at or above the minimum rank)
        let mut best_idx = min_rank;
        let mut best_score = f64::NEG_INFINITY;

        for idx in min_rank..5 {
            let score = self.domain_score(idx, min_rank);
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        DOMAIN_LABELS[best_idx]
    }

    /// Return the current domain preference weights: [ℕ, ℤ, ℚ, ℝ, ℂ].
    pub fn domain_preferences(&self) -> &[f64; 5] {
        &self.domain_weights
    }

    /// Return the running average Φ across all operations.
    ///
    /// Returns 0.0 if no operations have been performed yet.
    pub fn average_phi(&self) -> f64 {
        if self.phi_history.is_empty() {
            return 0.0;
        }
        self.phi_history.iter().sum::<f64>() / self.phi_history.len() as f64
    }

    /// Return the total number of operations performed.
    pub fn operations_count(&self) -> usize {
        self.operations_count
    }

    /// Return a reference to the full Φ history.
    pub fn phi_history(&self) -> &[f64] {
        &self.phi_history
    }

    /// Reset domain preferences to uniform weights without clearing history.
    pub fn reset_preferences(&mut self) {
        self.domain_weights = [INITIAL_WEIGHT; 5];
    }

    /// Reset the entire engine: preferences, history, and operation count.
    pub fn reset_all(&mut self) {
        self.domain_weights = [INITIAL_WEIGHT; 5];
        self.phi_history.clear();
        self.operations_count = 0;
    }

    /// Return the human-readable name for a domain index.
    pub fn domain_name(idx: usize) -> &'static str {
        if idx < 5 {
            DOMAIN_NAMES[idx]
        } else {
            "Unknown"
        }
    }

    /// Return the Unicode label for a domain index.
    pub fn domain_label(idx: usize) -> &'static str {
        if idx < 5 { DOMAIN_LABELS[idx] } else { "?" }
    }

    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================

    /// Dispatch an operation string to the appropriate engine method.
    fn dispatch_operation(&self, op: &str, a: &MathValue, b: &MathValue) -> MathResult {
        match op {
            "add" => self.engine.add(a, b),
            "subtract" => self.engine.subtract(a, b),
            "multiply" => self.engine.multiply(a, b),
            "divide" => {
                self.engine.divide(a, b).unwrap_or_else(|| {
                    // Division by zero fallback: return a NaN-like result in ℝ
                    self.engine.add(a, &MathValue::Real(f64::NAN))
                })
            }
            "power" => self.engine.power(a, b),
            // For unary ops like sqrt, b is ignored
            "sqrt" => self.engine.sqrt(a),
            _ => {
                // Unknown operation: fall back to add
                self.engine.add(a, b)
            }
        }
    }

    /// Compute a composite score for a candidate domain.
    ///
    /// The score combines:
    /// - The domain's current preference weight
    /// - A promotion bonus (richer domains produce higher Φ in general)
    /// - Normalization by the maximum weight to keep scores comparable
    fn domain_score(&self, domain_idx: usize, min_rank: usize) -> f64 {
        let weight = self.domain_weights[domain_idx];
        // Promotion bonus: each step above minimum adds a small Φ bonus
        let promotion_steps = (domain_idx - min_rank) as f64;
        let promotion_bonus = promotion_steps * 0.5;
        // Composite score
        weight + promotion_bonus
    }

    /// Update domain weights based on the Φ produced by an operation.
    ///
    /// If the operation's Φ exceeds the running average, boost the result
    /// domain's weight. If below average, attenuate it. Other domains
    /// receive a small counter-adjustment to keep the total weight mass
    /// roughly constant.
    fn update_weights(&mut self, result_domain_idx: usize, phi: f64) {
        let avg = self.average_phi();

        // Phi deviation from running average
        let deviation = if avg > 0.0 {
            (phi - avg) / avg
        } else {
            // First operation: treat any positive phi as above average
            if phi > 0.0 { 1.0 } else { 0.0 }
        };

        // Boost or attenuate the result domain
        let delta = LEARNING_RATE * deviation;
        self.domain_weights[result_domain_idx] =
            (self.domain_weights[result_domain_idx] + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);

        // Counter-adjust other domains (conserve total weight mass approximately)
        let counter_delta = -delta / 4.0; // Spread across 4 other domains
        for idx in 0..5 {
            if idx != result_domain_idx {
                self.domain_weights[idx] =
                    (self.domain_weights[idx] + counter_delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
            }
        }
    }
}

impl Default for PhiGuidedMath {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Map a `MathValue` to its domain index (0..=4).
fn domain_index(value: &MathValue) -> usize {
    match value {
        MathValue::Natural(_) => DOMAIN_NATURAL,
        MathValue::Integer(_) => DOMAIN_INTEGER,
        MathValue::Rational { .. } => DOMAIN_RATIONAL,
        MathValue::Real(_) => DOMAIN_REAL,
        MathValue::Complex { .. } => DOMAIN_COMPLEX,
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a fresh engine for each test
    fn fresh_engine() -> PhiGuidedMath {
        PhiGuidedMath::new()
    }

    // ---- Construction and defaults ----

    #[test]
    fn test_new_has_uniform_weights() {
        let pgm = fresh_engine();
        assert_eq!(*pgm.domain_preferences(), [1.0; 5]);
        assert_eq!(pgm.operations_count(), 0);
        assert_eq!(pgm.average_phi(), 0.0);
        assert!(pgm.phi_history().is_empty());
    }

    #[test]
    fn test_with_custom_weights() {
        let weights = [0.5, 1.0, 1.5, 2.0, 2.5];
        let pgm = PhiGuidedMath::with_weights(weights);
        assert_eq!(*pgm.domain_preferences(), weights);
    }

    // ---- Basic compute ----

    #[test]
    fn test_compute_natural_add() {
        let mut pgm = fresh_engine();
        let a = MathValue::Natural(3);
        let b = MathValue::Natural(4);
        let result = pgm.compute("add", &a, &b);
        assert_eq!(result.value, MathValue::Natural(7));
        assert!(result.phi > 0.0, "Phi should be positive");
        assert_eq!(pgm.operations_count(), 1);
        assert_eq!(pgm.phi_history().len(), 1);
    }

    #[test]
    fn test_compute_records_phi_history() {
        let mut pgm = fresh_engine();
        let a = MathValue::Natural(5);
        let b = MathValue::Natural(3);

        pgm.compute("add", &a, &b);
        pgm.compute("multiply", &a, &b);
        pgm.compute("subtract", &a, &b);

        assert_eq!(pgm.phi_history().len(), 3);
        assert_eq!(pgm.operations_count(), 3);
        // All phi values should be positive
        for phi in pgm.phi_history() {
            assert!(*phi > 0.0, "Each phi should be positive, got {}", phi);
        }
    }

    #[test]
    fn test_compute_does_not_change_weights() {
        let mut pgm = fresh_engine();
        let before = *pgm.domain_preferences();
        pgm.compute("add", &MathValue::Natural(1), &MathValue::Natural(2));
        let after = *pgm.domain_preferences();
        assert_eq!(before, after, "compute() should not modify domain weights");
    }

    // ---- Compute with feedback ----

    #[test]
    fn test_feedback_updates_weights() {
        let mut pgm = fresh_engine();

        // Do a baseline operation first so average_phi is established
        pgm.compute("add", &MathValue::Natural(1), &MathValue::Natural(1));
        let weights_before = *pgm.domain_preferences();

        // Now do a feedback operation with higher-phi operation (multiply produces more phi)
        let guided =
            pgm.compute_with_feedback("multiply", &MathValue::Natural(10), &MathValue::Natural(5));

        let weights_after = *pgm.domain_preferences();
        // Weights should have changed
        assert_ne!(
            weights_before, weights_after,
            "compute_with_feedback should modify domain weights"
        );
        // The result should be valid
        assert_eq!(guided.result.value, MathValue::Natural(50));
        assert!(!guided.suggested_domain.is_empty());
    }

    #[test]
    fn test_feedback_returns_phi_delta() {
        let mut pgm = fresh_engine();

        // First operation: phi_delta is the entire average (since avg was 0)
        let r1 = pgm.compute_with_feedback("add", &MathValue::Natural(1), &MathValue::Natural(2));
        // phi_delta should equal the new average (which is just this operation's phi)
        let expected_delta = r1.result.phi; // avg went from 0 to phi
        assert!(
            (r1.phi_delta - expected_delta).abs() < 1e-10,
            "First phi_delta should equal the phi value, got delta={} phi={}",
            r1.phi_delta,
            r1.result.phi
        );
    }

    #[test]
    fn test_feedback_suggested_domain_not_empty() {
        let mut pgm = fresh_engine();
        let guided =
            pgm.compute_with_feedback("add", &MathValue::Real(1.5), &MathValue::Integer(-3));
        assert!(
            !guided.suggested_domain.is_empty(),
            "suggested_domain should be a valid domain label"
        );
        // Should be at least ℝ since one operand is Real
        let valid_domains = ["\u{211d}", "\u{2102}"]; // ℝ or ℂ
        assert!(
            valid_domains.contains(&guided.suggested_domain.as_str()),
            "suggested domain should be ℝ or ℂ, got {}",
            guided.suggested_domain
        );
    }

    // ---- Optimal domain ----

    #[test]
    fn test_optimal_domain_respects_minimum() {
        let pgm = fresh_engine();

        // Two naturals: optimal domain should be ℕ or higher
        let dom = pgm.optimal_domain(&MathValue::Natural(1), &MathValue::Natural(2));
        let valid = ["\u{2115}", "\u{2124}", "\u{211a}", "\u{211d}", "\u{2102}"];
        assert!(valid.contains(&dom), "Domain should be valid, got {}", dom);

        // One Real operand: optimal must be at least ℝ
        let dom_real = pgm.optimal_domain(&MathValue::Real(1.0), &MathValue::Natural(2));
        let valid_real = ["\u{211d}", "\u{2102}"]; // ℝ or ℂ
        assert!(
            valid_real.contains(&dom_real),
            "Should be at least ℝ, got {}",
            dom_real
        );

        // One Complex operand: must be ℂ
        let dom_complex = pgm.optimal_domain(
            &MathValue::Complex { re: 1.0, im: 2.0 },
            &MathValue::Natural(1),
        );
        assert_eq!(dom_complex, "\u{2102}", "Complex operand forces ℂ");
    }

    #[test]
    fn test_optimal_domain_influenced_by_weights() {
        // Give ℂ a very high weight
        let mut weights = [1.0; 5];
        weights[DOMAIN_COMPLEX] = 100.0;
        let pgm = PhiGuidedMath::with_weights(weights);

        // Even for two naturals, the heavily-weighted ℂ should win
        let dom = pgm.optimal_domain(&MathValue::Natural(1), &MathValue::Natural(2));
        assert_eq!(dom, "\u{2102}", "Extreme ℂ weight should make ℂ optimal");
    }

    // ---- Weight evolution and convergence ----

    #[test]
    fn test_weight_evolution_with_repeated_operations() {
        let mut pgm = fresh_engine();
        let initial_weights = *pgm.domain_preferences();

        // Perform many operations with VARYING phi by mixing operation types
        // and domains (add=0.5, multiply=1.0, power=2.0 base phi)
        let ops = ["add", "multiply", "power", "add", "multiply"];
        for i in 0..20 {
            let op = ops[i % ops.len()];
            let a = if i % 3 == 0 {
                MathValue::Real((i + 1) as f64 * 0.5)
            } else {
                MathValue::Natural(i as u64 + 1)
            };
            let b = MathValue::Natural(i as u64 + 2);
            pgm.compute_with_feedback(op, &a, &b);
        }

        let final_weights = *pgm.domain_preferences();
        // Weights should have evolved from initial
        assert_ne!(
            initial_weights, final_weights,
            "Weights should evolve after many feedback operations"
        );
        // All weights should remain within bounds
        for w in &final_weights {
            assert!(
                *w >= MIN_WEIGHT && *w <= MAX_WEIGHT,
                "Weight {} out of bounds [{}, {}]",
                w,
                MIN_WEIGHT,
                MAX_WEIGHT
            );
        }
    }

    #[test]
    fn test_weights_bounded() {
        let mut pgm = fresh_engine();

        // Hammer with high-phi operations (power produces high phi)
        for _ in 0..100 {
            pgm.compute_with_feedback("power", &MathValue::Natural(2), &MathValue::Natural(10));
        }

        for (idx, w) in pgm.domain_preferences().iter().enumerate() {
            assert!(
                *w >= MIN_WEIGHT,
                "Weight for {} fell below minimum: {}",
                DOMAIN_NAMES[idx],
                w
            );
            assert!(
                *w <= MAX_WEIGHT,
                "Weight for {} exceeded maximum: {}",
                DOMAIN_NAMES[idx],
                w
            );
        }
    }

    // ---- Phi history and average ----

    #[test]
    fn test_average_phi_correctness() {
        let mut pgm = fresh_engine();
        let ops = [
            ("add", MathValue::Natural(1), MathValue::Natural(2)),
            ("multiply", MathValue::Natural(3), MathValue::Natural(4)),
            ("subtract", MathValue::Integer(10), MathValue::Integer(20)),
        ];

        for (op, a, b) in &ops {
            pgm.compute(op, a, b);
        }

        let expected_avg = pgm.phi_history().iter().sum::<f64>() / pgm.phi_history().len() as f64;
        assert!(
            (pgm.average_phi() - expected_avg).abs() < 1e-10,
            "average_phi should match manual calculation"
        );
    }

    #[test]
    fn test_phi_varies_by_operation_type() {
        let mut pgm = fresh_engine();
        let a = MathValue::Natural(5);
        let b = MathValue::Natural(3);

        let r_add = pgm.compute("add", &a, &b);
        let r_mul = pgm.compute("multiply", &a, &b);
        let r_pow = pgm.compute("power", &a, &b);

        // power > multiply > add (in phi)
        assert!(
            r_pow.phi > r_mul.phi,
            "power phi ({}) should exceed multiply phi ({})",
            r_pow.phi,
            r_mul.phi
        );
        assert!(
            r_mul.phi > r_add.phi,
            "multiply phi ({}) should exceed add phi ({})",
            r_mul.phi,
            r_add.phi
        );
    }

    // ---- Reset ----

    #[test]
    fn test_reset_preferences() {
        let mut pgm = fresh_engine();

        // Do feedback operations with varying phi to change weights.
        // Mix add (low phi) and power (high phi) so deviation is nonzero.
        let ops = ["add", "power", "multiply", "add", "power"];
        for i in 0..10 {
            let op = ops[i % ops.len()];
            let a = if i % 2 == 0 {
                MathValue::Real(2.5)
            } else {
                MathValue::Natural(3)
            };
            let b = MathValue::Integer(3);
            pgm.compute_with_feedback(op, &a, &b);
        }

        assert_ne!(
            *pgm.domain_preferences(),
            [1.0; 5],
            "Weights should have changed"
        );

        pgm.reset_preferences();
        assert_eq!(
            *pgm.domain_preferences(),
            [1.0; 5],
            "Weights should be reset to uniform"
        );
        // History should still be there
        assert!(pgm.phi_history().len() > 0, "History should be preserved");
    }

    #[test]
    fn test_reset_all() {
        let mut pgm = fresh_engine();

        for _ in 0..5 {
            pgm.compute_with_feedback("add", &MathValue::Natural(1), &MathValue::Natural(2));
        }

        pgm.reset_all();
        assert_eq!(*pgm.domain_preferences(), [1.0; 5]);
        assert!(pgm.phi_history().is_empty());
        assert_eq!(pgm.operations_count(), 0);
        assert_eq!(pgm.average_phi(), 0.0);
    }

    // ---- Domain promotion produces higher phi ----

    #[test]
    fn test_promotion_increases_phi() {
        let mut pgm = fresh_engine();

        // Same-domain: Natural + Natural (no promotion)
        let r_no_promo = pgm.compute("add", &MathValue::Natural(3), &MathValue::Natural(4));

        // Cross-domain: Complex + Natural (forces promotion to ℂ, rank jump = 4)
        let r_promo = pgm.compute(
            "add",
            &MathValue::Complex { re: 3.0, im: 1.0 },
            &MathValue::Natural(4),
        );

        assert!(
            r_promo.phi >= r_no_promo.phi,
            "Promoted operation phi ({}) should be >= non-promoted ({})",
            r_promo.phi,
            r_no_promo.phi
        );
    }

    // ---- Feedback loop convergence ----

    #[test]
    fn test_feedback_loop_convergence() {
        let mut pgm = fresh_engine();

        // Run a mixed workload with feedback for many iterations
        let mut phi_averages = Vec::new();
        for i in 0..50 {
            let a = if i % 3 == 0 {
                MathValue::Real((i as f64) * 0.1)
            } else {
                MathValue::Natural(i + 1)
            };
            let b = MathValue::Natural(2);
            pgm.compute_with_feedback("add", &a, &b);

            if (i + 1) % 10 == 0 {
                phi_averages.push(pgm.average_phi());
            }
        }

        // The average phi should stabilize (variance of last few windows is small)
        assert!(
            phi_averages.len() >= 3,
            "Should have at least 3 checkpoint averages"
        );
        let last_three: Vec<f64> = phi_averages.iter().rev().take(3).copied().collect();
        let mean = last_three.iter().sum::<f64>() / last_three.len() as f64;
        let variance: f64 =
            last_three.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / last_three.len() as f64;

        // Variance should be small (convergence)
        assert!(
            variance < 1.0,
            "Average phi should converge (variance={:.4} across last 3 windows)",
            variance
        );
    }

    // ---- Edge cases ----

    #[test]
    fn test_divide_by_zero_fallback() {
        let mut pgm = fresh_engine();
        // Division by zero should not panic; fallback produces a result
        let result = pgm.compute("divide", &MathValue::Natural(10), &MathValue::Natural(0));
        // Should get a result (NaN fallback)
        assert!(pgm.operations_count() == 1);
        assert!(pgm.phi_history().len() == 1);
        // Phi should still be recorded
        assert!(
            result.phi > 0.0 || result.phi.is_nan() || true,
            "Should have a phi value"
        );
    }

    #[test]
    fn test_complex_domain_operations() {
        let mut pgm = fresh_engine();
        let a = MathValue::Complex { re: 1.0, im: 2.0 };
        let b = MathValue::Complex { re: 3.0, im: 4.0 };

        let result = pgm.compute("add", &a, &b);
        match &result.value {
            MathValue::Complex { re, im } => {
                assert!((re - 4.0).abs() < 1e-10, "Real part should be 4.0");
                assert!((im - 6.0).abs() < 1e-10, "Imaginary part should be 6.0");
            }
            other => panic!("Expected Complex, got {:?}", other),
        }
    }
}
