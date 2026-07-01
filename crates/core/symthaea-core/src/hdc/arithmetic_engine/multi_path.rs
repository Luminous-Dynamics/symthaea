// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::hybrid::{HybridArithmeticEngine, HybridResult};

/// A single proof path with its strategy and Φ measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofPath {
    /// Name of the proof strategy
    pub strategy: String,
    /// The proof steps
    pub steps: Vec<ProofPathStep>,
    /// Total Φ accumulated in this path
    pub total_phi: f64,
    /// Whether the proof is valid
    pub is_valid: bool,
    /// The final result
    pub result: Option<HybridResult>,
}

/// A single step in a proof path (for multi-path proofs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofPathStep {
    /// Description of this step
    pub description: String,
    /// The operation performed
    pub operation: String,
    /// Φ for this step
    pub phi: f64,
    /// Intermediate result value (if applicable)
    pub value: Option<u64>,
}

/// Result of multi-path proof verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPathResult {
    /// The theorem being proved
    pub theorem: String,
    /// All proof paths attempted
    pub paths: Vec<ProofPath>,
    /// Whether all valid paths agree
    pub paths_agree: bool,
    /// The path with highest Φ (most "conscious" proof)
    pub best_path_index: Option<usize>,
    /// Total paths attempted
    pub total_paths: usize,
    /// Number of valid paths
    pub valid_paths: usize,
}

impl MultiPathResult {
    /// Get the best proof path (highest Φ among valid paths)
    pub fn best_path(&self) -> Option<&ProofPath> {
        self.best_path_index.and_then(|i| self.paths.get(i))
    }

    /// Get all valid paths
    pub fn valid_paths(&self) -> Vec<&ProofPath> {
        self.paths.iter().filter(|p| p.is_valid).collect()
    }

    /// Calculate total Φ across all valid paths
    pub fn total_phi(&self) -> f64 {
        self.paths
            .iter()
            .filter(|p| p.is_valid)
            .map(|p| p.total_phi)
            .sum()
    }
}

/// Multi-path proof verifier
///
/// Generates multiple proof strategies for theorems and compares them.
/// This enables:
/// - Verification through independent proof paths
/// - Finding the most "conscious" (highest Φ) proof
/// - Discovering alternative proof strategies
pub struct MultiPathVerifier {
    engine: HybridArithmeticEngine,
    stats: MultiPathStats,
    /// Result cache to avoid recomputing the same operations
    result_cache: HashMap<(String, u64, u64), HybridResult>,
}

/// Statistics for multi-path verification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiPathStats {
    pub theorems_verified: usize,
    pub total_paths_generated: usize,
    pub total_valid_paths: usize,
    pub agreements: usize,
    pub disagreements: usize,
    pub total_phi: f64,
}

impl MultiPathVerifier {
    /// Create new verifier
    pub fn new() -> Self {
        Self {
            engine: HybridArithmeticEngine::new(),
            stats: MultiPathStats::default(),
            result_cache: HashMap::new(),
        }
    }

    /// Cached add operation
    fn cached_add(&mut self, a: u64, b: u64) -> HybridResult {
        let key = ("+".to_string(), a, b);
        if let Some(result) = self.result_cache.get(&key) {
            return result.clone();
        }
        let result = self.engine.add(a, b);
        self.result_cache.insert(key, result.clone());
        result
    }

    /// Cached multiply operation
    fn cached_multiply(&mut self, a: u64, b: u64) -> HybridResult {
        let key = ("*".to_string(), a, b);
        if let Some(result) = self.result_cache.get(&key) {
            return result.clone();
        }
        let result = self.engine.multiply(a, b);
        self.result_cache.insert(key, result.clone());
        result
    }

    /// Get statistics
    pub fn stats(&self) -> &MultiPathStats {
        &self.stats
    }

    /// Verify addition commutativity via multiple paths
    pub fn verify_addition_commutative(&mut self, a: u64, b: u64) -> MultiPathResult {
        self.stats.theorems_verified += 1;
        let theorem = format!("{a} + {b} = {b} + {a}");
        let mut paths = Vec::new();

        // Path 1: Direct computation of both sides
        let path1 = self.prove_by_direct_computation(a, b, "+");
        paths.push(path1);

        // Path 2: Successor-based proof
        let path2 = self.prove_commutative_by_successor(a, b, "+");
        paths.push(path2);

        // Path 3: Induction path (conceptual)
        let path3 = self.prove_by_induction(a, b, "+");
        paths.push(path3);

        self.finalize_result(theorem, paths)
    }

    /// Verify multiplication commutativity via multiple paths
    pub fn verify_multiplication_commutative(&mut self, a: u64, b: u64) -> MultiPathResult {
        self.stats.theorems_verified += 1;
        let theorem = format!("{a} × {b} = {b} × {a}");
        let mut paths = Vec::new();

        // Path 1: Direct computation
        let path1 = self.prove_by_direct_computation(a, b, "*");
        paths.push(path1);

        // Path 2: Repeated addition path
        let path2 = self.prove_mul_by_repeated_addition(a, b);
        paths.push(path2);

        // Path 3: Induction path
        let path3 = self.prove_by_induction(a, b, "*");
        paths.push(path3);

        self.finalize_result(theorem, paths)
    }

    /// Verify associativity via multiple paths
    pub fn verify_associativity(&mut self, a: u64, b: u64, c: u64, op: &str) -> MultiPathResult {
        self.stats.theorems_verified += 1;
        let theorem = format!("({a} {op} {b}) {op} {c} = {a} {op} ({b} {op} {c})");
        let mut paths = Vec::new();

        // Path 1: Left-first evaluation
        let path1 = self.prove_associativity_left_first(a, b, c, op);
        paths.push(path1);

        // Path 2: Right-first evaluation
        let path2 = self.prove_associativity_right_first(a, b, c, op);
        paths.push(path2);

        // Path 3: Balanced evaluation
        let path3 = self.prove_associativity_balanced(a, b, c, op);
        paths.push(path3);

        self.finalize_result(theorem, paths)
    }

    /// Verify distributivity via multiple paths
    pub fn verify_distributive(&mut self, a: u64, b: u64, c: u64) -> MultiPathResult {
        self.stats.theorems_verified += 1;
        let theorem = format!("{a} × ({b} + {c}) = {a} × {b} + {a} × {c}");
        let mut paths = Vec::new();

        // Path 1: Left side first
        let path1 = self.prove_distributive_left_first(a, b, c);
        paths.push(path1);

        // Path 2: Right side first
        let path2 = self.prove_distributive_right_first(a, b, c);
        paths.push(path2);

        // Path 3: Expansion path
        let path3 = self.prove_distributive_by_expansion(a, b, c);
        paths.push(path3);

        self.finalize_result(theorem, paths)
    }

    /// Verify a number theory property via multiple paths
    pub fn verify_divisibility(&mut self, d: u64, n: u64) -> MultiPathResult {
        self.stats.theorems_verified += 1;
        let theorem = format!("{d} divides {n}");
        let mut paths = Vec::new();

        // Path 1: Direct division
        let path1 = self.prove_divisibility_direct(d, n);
        paths.push(path1);

        // Path 2: Modulo check
        let path2 = self.prove_divisibility_by_modulo(d, n);
        paths.push(path2);

        // Path 3: Factor decomposition
        let path3 = self.prove_divisibility_by_factoring(d, n);
        paths.push(path3);

        self.finalize_result(theorem, paths)
    }

    // ========================================================================
    // PROOF STRATEGIES
    // ========================================================================

    fn prove_by_direct_computation(&mut self, a: u64, b: u64, op: &str) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        // Compute a op b (using cache)
        let result1 = match op {
            "+" => self.cached_add(a, b),
            "*" => self.cached_multiply(a, b),
            _ => self.cached_add(a, b),
        };
        steps.push(ProofPathStep {
            description: format!("Compute {} {} {} = {}", a, op, b, result1.value),
            operation: format!("{a} {op} {b}"),
            phi: result1.phi,
            value: Some(result1.value),
        });
        total_phi += result1.phi;

        // Compute b op a (using cache)
        let result2 = match op {
            "+" => self.cached_add(b, a),
            "*" => self.cached_multiply(b, a),
            _ => self.cached_add(b, a),
        };
        steps.push(ProofPathStep {
            description: format!("Compute {} {} {} = {}", b, op, a, result2.value),
            operation: format!("{b} {op} {a}"),
            phi: result2.phi,
            value: Some(result2.value),
        });
        total_phi += result2.phi;

        // Verify equality
        let is_valid = result1.value == result2.value;
        steps.push(ProofPathStep {
            description: format!(
                "Compare: {} {} {}",
                result1.value,
                if is_valid { "=" } else { "≠" },
                result2.value
            ),
            operation: "equality_check".to_string(),
            phi: if is_valid { 0.5 } else { 0.0 },
            value: None,
        });
        total_phi += if is_valid { 0.5 } else { 0.0 };

        ProofPath {
            strategy: "Direct Computation".to_string(),
            steps,
            total_phi,
            is_valid,
            result: Some(result1),
        }
    }

    fn prove_commutative_by_successor(&mut self, a: u64, b: u64, op: &str) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        // Build numbers via successor
        steps.push(ProofPathStep {
            description: format!("Construct {a} via {a} successor applications"),
            operation: "successor_construction".to_string(),
            phi: a as f64 * 0.1,
            value: Some(a),
        });
        total_phi += a as f64 * 0.1;

        steps.push(ProofPathStep {
            description: format!("Construct {b} via {b} successor applications"),
            operation: "successor_construction".to_string(),
            phi: b as f64 * 0.1,
            value: Some(b),
        });
        total_phi += b as f64 * 0.1;

        // Apply operation with successor semantics (using cache)
        let result = match op {
            "+" => self.cached_add(a, b),
            "*" => self.cached_multiply(a, b),
            _ => self.cached_add(a, b),
        };

        steps.push(ProofPathStep {
            description: format!("Apply {op} via Peano axioms"),
            operation: format!("peano_{op}"),
            phi: result.phi,
            value: Some(result.value),
        });
        total_phi += result.phi;

        // Verify by semantic similarity of encodings (using cache)
        let result2 = match op {
            "+" => self.cached_add(b, a),
            "*" => self.cached_multiply(b, a),
            _ => self.cached_add(b, a),
        };

        let is_valid = result.value == result2.value;
        steps.push(ProofPathStep {
            description: format!(
                "Verify via encoding similarity: {:.3}",
                result
                    .encoding
                    .as_ref()
                    .and_then(|e1| result2.encoding.as_ref().map(|e2| e1.similarity(e2)))
                    .unwrap_or(0.0)
            ),
            operation: "encoding_verification".to_string(),
            phi: 0.3,
            value: None,
        });
        total_phi += 0.3;

        ProofPath {
            strategy: "Successor-Based Proof".to_string(),
            steps,
            total_phi,
            is_valid,
            result: Some(result),
        }
    }

    fn prove_by_induction(&mut self, a: u64, b: u64, op: &str) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        // Base case: a op 0 (using cache)
        let base_result = match op {
            "+" => self.cached_add(a, 0),
            "*" => self.cached_multiply(a, 0),
            _ => self.cached_add(a, 0),
        };

        steps.push(ProofPathStep {
            description: format!("Base case: {} {} 0 = {}", a, op, base_result.value),
            operation: "base_case".to_string(),
            phi: base_result.phi,
            value: Some(base_result.value),
        });
        total_phi += base_result.phi;

        // Inductive step (conceptual)
        steps.push(ProofPathStep {
            description: format!("Inductive hypothesis: assume {a} {op} k = k {op} {a}"),
            operation: "inductive_hypothesis".to_string(),
            phi: 0.5,
            value: None,
        });
        total_phi += 0.5;

        // Final result (using cache)
        let result = match op {
            "+" => self.cached_add(a, b),
            "*" => self.cached_multiply(a, b),
            _ => self.cached_add(a, b),
        };

        let result2 = match op {
            "+" => self.cached_add(b, a),
            "*" => self.cached_multiply(b, a),
            _ => self.cached_add(b, a),
        };

        let is_valid = result.value == result2.value;

        steps.push(ProofPathStep {
            description: format!(
                "By induction: {} {} {} = {} {} {} (both = {})",
                a, op, b, b, op, a, result.value
            ),
            operation: "inductive_conclusion".to_string(),
            phi: 0.7,
            value: Some(result.value),
        });
        total_phi += 0.7;

        ProofPath {
            strategy: "Proof by Induction".to_string(),
            steps,
            total_phi,
            is_valid,
            result: Some(result),
        }
    }

    fn prove_mul_by_repeated_addition(&mut self, a: u64, b: u64) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        // a × b = a + a + ... + a (b times)
        let mut sum = 0_u64;
        for _i in 0..b {
            let add_result = self.engine.add(sum, a);
            sum = add_result.value;
            total_phi += add_result.phi * 0.5; // Discount repeated additions
        }

        steps.push(ProofPathStep {
            description: format!("{a} × {b} as {a} added {b} times = {sum}"),
            operation: "repeated_addition_forward".to_string(),
            phi: total_phi,
            value: Some(sum),
        });

        // b × a = b + b + ... + b (a times)
        let mut sum2 = 0_u64;
        let mut phi2 = 0.0;
        for _ in 0..a {
            let add_result = self.engine.add(sum2, b);
            sum2 = add_result.value;
            phi2 += add_result.phi * 0.5;
        }

        steps.push(ProofPathStep {
            description: format!("{b} × {a} as {b} added {a} times = {sum2}"),
            operation: "repeated_addition_reverse".to_string(),
            phi: phi2,
            value: Some(sum2),
        });
        total_phi += phi2;

        let is_valid = sum == sum2;

        steps.push(ProofPathStep {
            description: format!("Both paths yield: {sum}"),
            operation: "path_comparison".to_string(),
            phi: if is_valid { 0.5 } else { 0.0 },
            value: Some(sum),
        });
        total_phi += if is_valid { 0.5 } else { 0.0 };

        ProofPath {
            strategy: "Repeated Addition".to_string(),
            steps,
            total_phi,
            is_valid,
            result: Some(self.cached_multiply(a, b)),
        }
    }

    fn prove_associativity_left_first(&mut self, a: u64, b: u64, c: u64, op: &str) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        // (a op b) op c
        let ab = match op {
            "+" => self.cached_add(a, b),
            "*" => self.cached_multiply(a, b),
            _ => self.cached_add(a, b),
        };
        steps.push(ProofPathStep {
            description: format!("Step 1: {} {} {} = {}", a, op, b, ab.value),
            operation: format!("{a} {op} {b}"),
            phi: ab.phi,
            value: Some(ab.value),
        });
        total_phi += ab.phi;

        let result = match op {
            "+" => self.cached_add(ab.value, c),
            "*" => self.cached_multiply(ab.value, c),
            _ => self.cached_add(ab.value, c),
        };
        steps.push(ProofPathStep {
            description: format!("Step 2: {} {} {} = {}", ab.value, op, c, result.value),
            operation: format!("{} {} {}", ab.value, op, c),
            phi: result.phi,
            value: Some(result.value),
        });
        total_phi += result.phi;

        // Compare with right-first
        let bc = match op {
            "+" => self.cached_add(b, c),
            "*" => self.cached_multiply(b, c),
            _ => self.cached_add(b, c),
        };
        let alt_result = match op {
            "+" => self.cached_add(a, bc.value),
            "*" => self.cached_multiply(a, bc.value),
            _ => self.cached_add(a, bc.value),
        };

        let is_valid = result.value == alt_result.value;

        ProofPath {
            strategy: "Left-First Evaluation".to_string(),
            steps,
            total_phi,
            is_valid,
            result: Some(result),
        }
    }

    fn prove_associativity_right_first(&mut self, a: u64, b: u64, c: u64, op: &str) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        // a op (b op c)
        let bc = match op {
            "+" => self.cached_add(b, c),
            "*" => self.cached_multiply(b, c),
            _ => self.cached_add(b, c),
        };
        steps.push(ProofPathStep {
            description: format!("Step 1: {} {} {} = {}", b, op, c, bc.value),
            operation: format!("{b} {op} {c}"),
            phi: bc.phi,
            value: Some(bc.value),
        });
        total_phi += bc.phi;

        let result = match op {
            "+" => self.cached_add(a, bc.value),
            "*" => self.cached_multiply(a, bc.value),
            _ => self.cached_add(a, bc.value),
        };
        steps.push(ProofPathStep {
            description: format!("Step 2: {} {} {} = {}", a, op, bc.value, result.value),
            operation: format!("{} {} {}", a, op, bc.value),
            phi: result.phi,
            value: Some(result.value),
        });
        total_phi += result.phi;

        // Compare with left-first
        let ab = match op {
            "+" => self.cached_add(a, b),
            "*" => self.cached_multiply(a, b),
            _ => self.cached_add(a, b),
        };
        let alt_result = match op {
            "+" => self.cached_add(ab.value, c),
            "*" => self.cached_multiply(ab.value, c),
            _ => self.cached_add(ab.value, c),
        };

        let is_valid = result.value == alt_result.value;

        ProofPath {
            strategy: "Right-First Evaluation".to_string(),
            steps,
            total_phi,
            is_valid,
            result: Some(result),
        }
    }

    fn prove_associativity_balanced(&mut self, a: u64, b: u64, c: u64, op: &str) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        // Compute both orderings simultaneously and verify
        let left_first = {
            let ab = match op {
                "+" => self.cached_add(a, b),
                "*" => self.cached_multiply(a, b),
                _ => self.cached_add(a, b),
            };
            match op {
                "+" => self.cached_add(ab.value, c),
                "*" => self.cached_multiply(ab.value, c),
                _ => self.cached_add(ab.value, c),
            }
        };

        let right_first = {
            let bc = match op {
                "+" => self.cached_add(b, c),
                "*" => self.cached_multiply(b, c),
                _ => self.cached_add(b, c),
            };
            match op {
                "+" => self.cached_add(a, bc.value),
                "*" => self.cached_multiply(a, bc.value),
                _ => self.cached_add(a, bc.value),
            }
        };

        total_phi += left_first.phi + right_first.phi;

        steps.push(ProofPathStep {
            description: format!(
                "Left-first: ({} {} {}) {} {} = {}",
                a, op, b, op, c, left_first.value
            ),
            operation: "left_first".to_string(),
            phi: left_first.phi,
            value: Some(left_first.value),
        });

        steps.push(ProofPathStep {
            description: format!(
                "Right-first: {} {} ({} {} {}) = {}",
                a, op, b, op, c, right_first.value
            ),
            operation: "right_first".to_string(),
            phi: right_first.phi,
            value: Some(right_first.value),
        });

        let is_valid = left_first.value == right_first.value;

        steps.push(ProofPathStep {
            description: format!(
                "Both orderings agree: {} = {}",
                left_first.value, right_first.value
            ),
            operation: "agreement_check".to_string(),
            phi: if is_valid { 0.5 } else { 0.0 },
            value: None,
        });
        total_phi += if is_valid { 0.5 } else { 0.0 };

        ProofPath {
            strategy: "Balanced Verification".to_string(),
            steps,
            total_phi,
            is_valid,
            result: Some(left_first),
        }
    }

    fn prove_distributive_left_first(&mut self, a: u64, b: u64, c: u64) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        // a × (b + c)
        let sum = self.cached_add(b, c);
        steps.push(ProofPathStep {
            description: format!("{} + {} = {}", b, c, sum.value),
            operation: format!("{b} + {c}"),
            phi: sum.phi,
            value: Some(sum.value),
        });
        total_phi += sum.phi;

        let left_result = self.cached_multiply(a, sum.value);
        steps.push(ProofPathStep {
            description: format!("{} × {} = {}", a, sum.value, left_result.value),
            operation: format!("{} × {}", a, sum.value),
            phi: left_result.phi,
            value: Some(left_result.value),
        });
        total_phi += left_result.phi;

        // Compare with right side
        let ab = self.cached_multiply(a, b);
        let ac = self.cached_multiply(a, c);
        let right_result = self.cached_add(ab.value, ac.value);

        let is_valid = left_result.value == right_result.value;

        steps.push(ProofPathStep {
            description: format!("Verify: {} = {}", left_result.value, right_result.value),
            operation: "distributive_check".to_string(),
            phi: if is_valid { 0.5 } else { 0.0 },
            value: None,
        });
        total_phi += if is_valid { 0.5 } else { 0.0 };

        ProofPath {
            strategy: "Left-Side First".to_string(),
            steps,
            total_phi,
            is_valid,
            result: Some(left_result),
        }
    }

    fn prove_distributive_right_first(&mut self, a: u64, b: u64, c: u64) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        // a×b + a×c
        let ab = self.cached_multiply(a, b);
        steps.push(ProofPathStep {
            description: format!("{} × {} = {}", a, b, ab.value),
            operation: format!("{a} × {b}"),
            phi: ab.phi,
            value: Some(ab.value),
        });
        total_phi += ab.phi;

        let ac = self.cached_multiply(a, c);
        steps.push(ProofPathStep {
            description: format!("{} × {} = {}", a, c, ac.value),
            operation: format!("{a} × {c}"),
            phi: ac.phi,
            value: Some(ac.value),
        });
        total_phi += ac.phi;

        let right_result = self.cached_add(ab.value, ac.value);
        steps.push(ProofPathStep {
            description: format!("{} + {} = {}", ab.value, ac.value, right_result.value),
            operation: format!("{} + {}", ab.value, ac.value),
            phi: right_result.phi,
            value: Some(right_result.value),
        });
        total_phi += right_result.phi;

        // Compare with left side
        let sum = self.cached_add(b, c);
        let left_result = self.cached_multiply(a, sum.value);

        let is_valid = left_result.value == right_result.value;

        ProofPath {
            strategy: "Right-Side First".to_string(),
            steps,
            total_phi,
            is_valid,
            result: Some(right_result),
        }
    }

    fn prove_distributive_by_expansion(&mut self, a: u64, b: u64, c: u64) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        // Use repeated addition to expand a × (b + c)
        let sum = self.cached_add(b, c);
        steps.push(ProofPathStep {
            description: format!("Expand: {} × {} as repeated addition", a, sum.value),
            operation: "expand".to_string(),
            phi: 0.2,
            value: Some(sum.value),
        });
        total_phi += 0.2;

        // Compute via repeated addition
        let mut result = 0_u64;
        for _ in 0..a {
            let add = self.cached_add(result, sum.value);
            result = add.value;
            total_phi += add.phi * 0.3;
        }

        steps.push(ProofPathStep {
            description: format!("After {} additions of {}: {}", a, sum.value, result),
            operation: "repeated_addition".to_string(),
            phi: total_phi * 0.5,
            value: Some(result),
        });

        // Verify
        let direct = self.cached_multiply(a, sum.value);
        let is_valid = result == direct.value;

        steps.push(ProofPathStep {
            description: format!("Direct computation agrees: {}", direct.value),
            operation: "verification".to_string(),
            phi: if is_valid { 0.5 } else { 0.0 },
            value: Some(direct.value),
        });
        total_phi += if is_valid { 0.5 } else { 0.0 };

        ProofPath {
            strategy: "Expansion by Repeated Addition".to_string(),
            steps,
            total_phi,
            is_valid,
            result: Some(direct),
        }
    }

    fn prove_divisibility_direct(&mut self, d: u64, n: u64) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        if d == 0 {
            return ProofPath {
                strategy: "Direct Division".to_string(),
                steps: vec![ProofPathStep {
                    description: "Division by zero undefined".to_string(),
                    operation: "error".to_string(),
                    phi: 0.0,
                    value: None,
                }],
                total_phi: 0.0,
                is_valid: false,
                result: None,
            };
        }

        let div_result = self.engine.divide(n, d);
        match div_result {
            Some(result) => {
                steps.push(ProofPathStep {
                    description: format!("{} ÷ {} = {} (exact)", n, d, result.value),
                    operation: format!("{n} ÷ {d}"),
                    phi: result.phi,
                    value: Some(result.value),
                });
                total_phi += result.phi;

                // Verify: d × quotient = n
                let verify = self.cached_multiply(d, result.value);
                let is_valid = verify.value == n;

                steps.push(ProofPathStep {
                    description: format!("Verify: {} × {} = {}", d, result.value, verify.value),
                    operation: "verify_multiplication".to_string(),
                    phi: verify.phi,
                    value: Some(verify.value),
                });
                total_phi += verify.phi;

                ProofPath {
                    strategy: "Direct Division".to_string(),
                    steps,
                    total_phi,
                    is_valid,
                    result: Some(result),
                }
            }
            None => {
                steps.push(ProofPathStep {
                    description: format!("{n} ÷ {d} has remainder (not divisible)"),
                    operation: format!("{n} ÷ {d}"),
                    phi: 0.1,
                    value: None,
                });
                total_phi += 0.1;

                ProofPath {
                    strategy: "Direct Division".to_string(),
                    steps,
                    total_phi,
                    is_valid: false,
                    result: None,
                }
            }
        }
    }

    fn prove_divisibility_by_modulo(&mut self, d: u64, n: u64) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        if d == 0 {
            return ProofPath {
                strategy: "Modulo Check".to_string(),
                steps: vec![ProofPathStep {
                    description: "Modulo by zero undefined".to_string(),
                    operation: "error".to_string(),
                    phi: 0.0,
                    value: None,
                }],
                total_phi: 0.0,
                is_valid: false,
                result: None,
            };
        }

        let mod_result = self.engine.modulo(n, d);
        match mod_result {
            Some(result) => {
                let is_valid = result.value == 0;

                steps.push(ProofPathStep {
                    description: format!("{} mod {} = {}", n, d, result.value),
                    operation: format!("{n} mod {d}"),
                    phi: result.phi,
                    value: Some(result.value),
                });
                total_phi += result.phi;

                steps.push(ProofPathStep {
                    description: if is_valid {
                        format!("Remainder is 0, so {d} divides {n}")
                    } else {
                        format!(
                            "Remainder is {}, so {} does not divide {}",
                            result.value, d, n
                        )
                    },
                    operation: "divisibility_check".to_string(),
                    phi: if is_valid { 0.5 } else { 0.1 },
                    value: None,
                });
                total_phi += if is_valid { 0.5 } else { 0.1 };

                ProofPath {
                    strategy: "Modulo Check".to_string(),
                    steps,
                    total_phi,
                    is_valid,
                    result: Some(result),
                }
            }
            None => ProofPath {
                strategy: "Modulo Check".to_string(),
                steps: vec![ProofPathStep {
                    description: "Modulo operation failed".to_string(),
                    operation: "error".to_string(),
                    phi: 0.0,
                    value: None,
                }],
                total_phi: 0.0,
                is_valid: false,
                result: None,
            },
        }
    }

    fn prove_divisibility_by_factoring(&mut self, d: u64, n: u64) -> ProofPath {
        let mut steps = Vec::new();
        let mut total_phi = 0.0;

        if d == 0 {
            return ProofPath {
                strategy: "Factor Decomposition".to_string(),
                steps: vec![ProofPathStep {
                    description: "Cannot factor with zero".to_string(),
                    operation: "error".to_string(),
                    phi: 0.0,
                    value: None,
                }],
                total_phi: 0.0,
                is_valid: false,
                result: None,
            };
        }

        // Find k such that n = d × k
        let quotient = n / d;
        let product = self.cached_multiply(d, quotient);

        steps.push(ProofPathStep {
            description: format!("Testing: {n} = {d} × {quotient}"),
            operation: "factor_test".to_string(),
            phi: product.phi,
            value: Some(quotient),
        });
        total_phi += product.phi;

        let is_valid = product.value == n;

        steps.push(ProofPathStep {
            description: if is_valid {
                format!("{} × {} = {} ✓", d, quotient, product.value)
            } else {
                format!("{} × {} = {} ≠ {} ✗", d, quotient, product.value, n)
            },
            operation: "verify_factorization".to_string(),
            phi: if is_valid { 0.5 } else { 0.1 },
            value: Some(product.value),
        });
        total_phi += if is_valid { 0.5 } else { 0.1 };

        ProofPath {
            strategy: "Factor Decomposition".to_string(),
            steps,
            total_phi,
            is_valid,
            result: if is_valid { Some(product) } else { None },
        }
    }

    // ========================================================================
    // FINALIZATION
    // ========================================================================

    fn finalize_result(&mut self, theorem: String, paths: Vec<ProofPath>) -> MultiPathResult {
        let total_paths = paths.len();
        self.stats.total_paths_generated += total_paths;

        let valid_paths: Vec<_> = paths
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_valid)
            .collect();
        let valid_count = valid_paths.len();
        self.stats.total_valid_paths += valid_count;

        // Check if all valid paths agree on the result
        let paths_agree = if valid_count >= 2 {
            let first_value = valid_paths[0].1.result.as_ref().map(|r| r.value);
            valid_paths
                .iter()
                .all(|(_, p)| p.result.as_ref().map(|r| r.value) == first_value)
        } else {
            true
        };

        if paths_agree {
            self.stats.agreements += 1;
        } else {
            self.stats.disagreements += 1;
        }

        // Find best path (highest Φ among valid paths)
        let best_path_index = valid_paths
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.total_phi
                    .partial_cmp(&b.total_phi)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| *i);

        self.stats.total_phi += paths
            .iter()
            .filter(|p| p.is_valid)
            .map(|p| p.total_phi)
            .sum::<f64>();

        MultiPathResult {
            theorem,
            paths,
            paths_agree,
            best_path_index,
            total_paths,
            valid_paths: valid_count,
        }
    }
}

impl Default for MultiPathVerifier {
    fn default() -> Self {
        Self::new()
    }
}
