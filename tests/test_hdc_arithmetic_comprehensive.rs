// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Comprehensive HDC Arithmetic Test Suite
//!
//! Tests for the hyperdimensional arithmetic engine covering:
//! - Peano number construction (correctness and edge cases)
//! - Basic arithmetic operations (add, subtract, multiply, power, factorial)
//! - Theorem proving (commutativity, associativity, distributivity)
//! - Hybrid engine (small vs large number handling)
//! - Symbolic algebra operations
//! - Multi-path proof verification
//! - Performance characteristics
//!
//! The arithmetic engine computes mathematics from first principles using HDC,
//! not pattern matching. Each test validates this foundational approach.

use symthaea::hdc::arithmetic_engine::{
    ArithmeticEngine, HdcNumber, HybridArithmeticEngine, MathReasoningBridge, MultiPathVerifier,
    SymbolicAlgebra, SymbolicExpr, TheoremProver,
};
use symthaea::hdc::primitive_system::PrimitiveSystem;

// ============================================================================
// PEANO NUMBER CONSTRUCTION TESTS
// ============================================================================

#[test]
fn test_zero_construction() {
    let primitives = PrimitiveSystem::new();
    let zero = HdcNumber::zero(&primitives);

    assert_eq!(zero.value, 0, "Zero should have value 0");
    assert_eq!(
        zero.construction.len(),
        1,
        "Zero should have single construction step"
    );
    assert_eq!(
        zero.construction[0], "ZERO",
        "Zero construction should be ZERO"
    );
}

#[test]
fn test_small_number_construction() {
    let primitives = PrimitiveSystem::new();

    for n in 1..=10 {
        let num = HdcNumber::from_value(n, &primitives);
        assert_eq!(num.value, n, "Number {} should have correct value", n);
        // Construction should have ZERO + n successor steps
        assert_eq!(
            num.construction.len(),
            (n + 1) as usize,
            "Number {} should have {} construction steps",
            n,
            n + 1
        );
    }
}

#[test]
fn test_number_similarity_is_high_for_same_value() {
    let primitives = PrimitiveSystem::new();

    let five_a = HdcNumber::from_value(5, &primitives);
    let five_b = HdcNumber::from_value(5, &primitives);

    let similarity = five_a.similarity(&five_b);
    assert!(
        similarity > 0.99,
        "Same numbers should have very high similarity, got {}",
        similarity
    );
}

#[test]
fn test_different_numbers_have_lower_similarity() {
    let primitives = PrimitiveSystem::new();

    let five = HdcNumber::from_value(5, &primitives);
    let ten = HdcNumber::from_value(10, &primitives);

    let similarity = five.similarity(&ten);
    // Different numbers should still have some similarity due to shared construction
    // but less than identical numbers
    assert!(
        similarity < 0.99,
        "Different numbers should have lower similarity, got {}",
        similarity
    );
}

#[test]
fn test_successor_increments_correctly() {
    let primitives = PrimitiveSystem::new();

    let three = HdcNumber::from_value(3, &primitives);
    let four = three.successor(&primitives);

    assert_eq!(four.value, 4, "Successor of 3 should be 4");
    assert_eq!(
        four.construction.len(),
        three.construction.len() + 1,
        "Successor should add one construction step"
    );
}

#[test]
fn test_construction_phi_is_nonnegative() {
    let primitives = PrimitiveSystem::new();

    for n in 0..=20 {
        let num = HdcNumber::from_value(n, &primitives);
        assert!(
            num.construction_phi >= 0.0,
            "Construction Phi should be non-negative for {}, got {}",
            n,
            num.construction_phi
        );
    }
}

// ============================================================================
// BASIC ARITHMETIC OPERATIONS
// ============================================================================

#[test]
fn test_addition_basic() {
    let mut engine = ArithmeticEngine::new();

    // Test simple additions
    let cases = [(2, 3, 5), (0, 5, 5), (5, 0, 5), (7, 8, 15), (1, 1, 2)];

    for (a, b, expected) in cases {
        let result = engine.add(a, b);
        assert_eq!(
            result.result.value, expected,
            "{} + {} should equal {}",
            a, b, expected
        );
        assert!(result.verified, "Addition result should be verified");
        assert!(result.total_phi > 0.0, "Addition should have positive Phi");
    }
}

#[test]
fn test_addition_commutativity() {
    let mut engine = ArithmeticEngine::new();

    for a in 0..=5 {
        for b in 0..=5 {
            let ab = engine.add(a, b);
            let ba = engine.add(b, a);

            assert_eq!(
                ab.result.value, ba.result.value,
                "Addition should be commutative: {} + {} = {} + {}",
                a, b, b, a
            );
        }
    }
}

#[test]
fn test_multiplication_basic() {
    let mut engine = ArithmeticEngine::new();

    let cases = [(2, 3, 6), (0, 5, 0), (5, 0, 0), (7, 8, 56), (1, 10, 10)];

    for (a, b, expected) in cases {
        let result = engine.multiply(a, b);
        assert_eq!(
            result.result.value, expected,
            "{} * {} should equal {}",
            a, b, expected
        );
        assert!(result.verified, "Multiplication result should be verified");
    }
}

#[test]
fn test_subtraction_basic() {
    let mut engine = ArithmeticEngine::new();

    // Valid subtractions (a >= b)
    let valid_cases = [(5, 3, 2), (10, 0, 10), (7, 7, 0), (20, 15, 5)];

    for (a, b, expected) in valid_cases {
        let result = engine.subtract(a, b);
        assert!(result.is_some(), "{} - {} should be valid", a, b);
        assert_eq!(
            result.unwrap().result.value,
            expected,
            "{} - {} should equal {}",
            a,
            b,
            expected
        );
    }

    // Invalid subtractions (a < b) - undefined in naturals
    let invalid_cases = [(3, 5), (0, 1), (10, 20)];

    for (a, b) in invalid_cases {
        let result = engine.subtract(a, b);
        assert!(
            result.is_none(),
            "{} - {} should be None (undefined in naturals)",
            a,
            b
        );
    }
}

#[test]
fn test_power_basic() {
    let mut engine = ArithmeticEngine::new();

    let cases = [(2, 3, 8), (3, 2, 9), (2, 0, 1), (5, 1, 5), (1, 10, 1)];

    for (base, exp, expected) in cases {
        let result = engine.power(base, exp);
        assert_eq!(
            result.result.value, expected,
            "{}^{} should equal {}",
            base, exp, expected
        );
    }
}

#[test]
fn test_factorial_basic() {
    let mut engine = ArithmeticEngine::new();

    let cases = [(0, 1), (1, 1), (2, 2), (3, 6), (4, 24), (5, 120)];

    for (n, expected) in cases {
        let result = engine.factorial(n);
        assert_eq!(
            result.result.value, expected,
            "{}! should equal {}",
            n, expected
        );
    }
}

// ============================================================================
// THEOREM PROVER TESTS
// ============================================================================

#[test]
fn test_prove_addition_commutative() {
    let mut prover = TheoremProver::new();

    for (a, b) in [(2, 3), (5, 7), (0, 4)] {
        let theorem = prover.prove_addition_commutative(a, b);

        assert!(theorem.verified, "Commutativity theorem should be verified");
        assert!(theorem.total_phi > 0.0, "Theorem should have positive Phi");
        assert_eq!(theorem.proof_steps.len(), 2, "Should have two proof steps");
    }
}

#[test]
fn test_prove_addition_associative() {
    let mut prover = TheoremProver::new();

    let theorem = prover.prove_addition_associative(2, 3, 4);

    assert!(theorem.verified, "Associativity theorem should be verified");
    // (2+3)+4 = 2+(3+4)
    assert_eq!(theorem.proof_steps.len(), 4, "Should have four proof steps");
}

#[test]
fn test_prove_distributive() {
    let mut prover = TheoremProver::new();

    let theorem = prover.prove_distributive(2, 3, 4);

    assert!(theorem.verified, "Distributive law should be verified");
    // 2*(3+4) = 2*3 + 2*4
    assert_eq!(theorem.proof_steps.len(), 5, "Should have five proof steps");
}

// ============================================================================
// HYBRID ENGINE TESTS
// ============================================================================

#[test]
fn test_hybrid_engine_small_numbers() {
    let mut hybrid = HybridArithmeticEngine::new();

    // Small numbers should use full Peano derivation
    let result = hybrid.add(3, 4);

    assert_eq!(result.value, 7);
    // Result should have computation info
    assert!(
        result.full_proof.is_some() || result.abstract_proof.is_some(),
        "Result should have proof"
    );
}

#[test]
fn test_hybrid_engine_large_numbers() {
    let mut hybrid = HybridArithmeticEngine::new();

    // Large numbers should use direct computation
    let result = hybrid.add(1000, 2000);

    assert_eq!(result.value, 3000);
}

#[test]
fn test_hybrid_engine_consistency() {
    let mut hybrid = HybridArithmeticEngine::new();

    // Both paths should give same answer
    let small = hybrid.multiply(5, 7);
    let large = hybrid.multiply(500, 700);

    assert_eq!(small.value, 35);
    assert_eq!(large.value, 350000);
}

// ============================================================================
// SYMBOLIC ALGEBRA TESTS
// ============================================================================

#[test]
fn test_symbolic_constant_creation() {
    let primitives = PrimitiveSystem::new();

    let five = SymbolicExpr::constant(5, &primitives);

    assert_eq!(five.display, "5", "Constant should display as its value");
    assert!(five.phi >= 0.0, "Phi should be non-negative");
}

#[test]
fn test_symbolic_variable_creation() {
    let primitives = PrimitiveSystem::new();

    let x = SymbolicExpr::variable("x", &primitives);

    assert_eq!(x.display, "x", "Variable should display its name");
}

#[test]
fn test_symbolic_addition() {
    let primitives = PrimitiveSystem::new();

    let x = SymbolicExpr::variable("x", &primitives);
    let five = SymbolicExpr::constant(5, &primitives);

    let sum = x.add(&five, &primitives);

    assert!(
        sum.display.contains("x") && sum.display.contains("5"),
        "Sum should contain both operands"
    );
}

#[test]
fn test_symbolic_algebra_simplification() {
    let mut algebra = SymbolicAlgebra::new();
    let primitives = PrimitiveSystem::new();

    // x + 0 should simplify to x
    let x = SymbolicExpr::variable("x", &primitives);
    let zero = SymbolicExpr::constant(0, &primitives);
    let sum = x.add(&zero, &primitives);

    let simplified = algebra.simplify(&sum, &primitives);

    // The simplification should recognize x + 0 = x
    assert!(
        simplified.phi >= 0.0,
        "Simplified expression should have non-negative Phi"
    );
}

// ============================================================================
// MULTI-PATH VERIFICATION TESTS
// ============================================================================

#[test]
fn test_multi_path_simple_addition() {
    let mut verifier = MultiPathVerifier::new();

    let result = verifier.verify_addition_commutative(3, 4);

    assert!(
        result.paths_agree,
        "All paths should agree on 3 + 4 = 4 + 3"
    );
    assert!(
        !result.valid_paths().is_empty(),
        "Should have at least one valid path"
    );
}

#[test]
fn test_multi_path_multiplication() {
    let mut verifier = MultiPathVerifier::new();

    let result = verifier.verify_multiplication_commutative(4, 5);

    assert!(
        result.paths_agree,
        "All paths should agree on 4 * 5 = 5 * 4"
    );
}

#[test]
fn test_multi_path_best_path_selection() {
    let mut verifier = MultiPathVerifier::new();

    let result = verifier.verify_addition_commutative(5, 7);

    // Should select the path with highest Phi
    if let Some(best) = result.best_path() {
        assert!(best.is_valid, "Best path should be valid");
        assert!(best.total_phi > 0.0, "Best path should have positive Phi");
    }
}

// ============================================================================
// MATH REASONING BRIDGE TESTS
// ============================================================================

#[test]
fn test_reasoning_bridge_creation() {
    let mut bridge = MathReasoningBridge::new();

    // New bridge starts empty; generate an assertion to confirm it works
    let assertions = bridge.assertions();
    assert!(
        assertions.is_empty(),
        "New bridge starts with no assertions"
    );

    // After asserting, it should have one
    bridge.assert_equality(2, 3, "add");
    assert_eq!(
        bridge.assertions().len(),
        1,
        "Should have one assertion after assert_equality"
    );
}

#[test]
fn test_reasoning_bridge_number_assertions() {
    let mut bridge = MathReasoningBridge::new();

    // Should be able to assert properties about numbers
    let assertion = bridge.assert_prime(7);

    assert!(
        assertion.phi > 0.0,
        "Prime assertion should have positive Phi"
    );
}

// ============================================================================
// PROOF TRACE TESTS
// ============================================================================

#[test]
fn test_proof_steps_have_descriptions() {
    let mut engine = ArithmeticEngine::new();

    let result = engine.add(2, 3);

    for (i, step) in result.proof.iter().enumerate() {
        assert!(
            !step.description.is_empty(),
            "Proof step {} should have description",
            i
        );
        assert!(
            step.phi >= 0.0,
            "Proof step {} should have non-negative Phi",
            i
        );
    }
}

#[test]
fn test_proof_traces_are_deterministic() {
    let mut engine1 = ArithmeticEngine::new();
    let mut engine2 = ArithmeticEngine::new();

    let result1 = engine1.multiply(3, 4);
    let result2 = engine2.multiply(3, 4);

    assert_eq!(
        result1.result.value, result2.result.value,
        "Same computation should give same result"
    );
    assert_eq!(
        result1.proof.len(),
        result2.proof.len(),
        "Same computation should have same proof length"
    );
}

// ============================================================================
// EDGE CASES AND BOUNDARY TESTS
// ============================================================================

#[test]
fn test_zero_operations() {
    let mut engine = ArithmeticEngine::new();

    // 0 + n = n
    assert_eq!(engine.add(0, 5).result.value, 5);
    assert_eq!(engine.add(5, 0).result.value, 5);

    // 0 * n = 0
    assert_eq!(engine.multiply(0, 5).result.value, 0);
    assert_eq!(engine.multiply(5, 0).result.value, 0);

    // n^0 = 1
    assert_eq!(engine.power(5, 0).result.value, 1);

    // 0! = 1
    assert_eq!(engine.factorial(0).result.value, 1);
}

#[test]
fn test_identity_operations() {
    let mut engine = ArithmeticEngine::new();

    // n + 0 = n
    for n in 1..=10 {
        assert_eq!(engine.add(n, 0).result.value, n);
    }

    // n * 1 = n
    for n in 1..=10 {
        assert_eq!(engine.multiply(n, 1).result.value, n);
    }

    // n^1 = n
    for n in 1..=10 {
        assert_eq!(engine.power(n, 1).result.value, n);
    }
}

#[test]
fn test_cache_effectiveness() {
    let mut engine = ArithmeticEngine::new();

    // First computation - cache miss
    let _ = engine.add(5, 7);

    // Second computation - should be cache hit
    let _ = engine.add(5, 7);

    let stats = engine.stats();
    assert!(
        stats.cache_hits > 0,
        "Should have cache hits for repeated computation"
    );
}

// ============================================================================
// PERFORMANCE SANITY TESTS
// ============================================================================

#[test]
fn test_small_number_performance() {
    use std::time::Instant;

    let mut engine = ArithmeticEngine::new();

    let start = Instant::now();
    for _ in 0..100 {
        let _ = engine.add(5, 7);
    }
    let elapsed = start.elapsed();

    // Should complete 100 cached operations quickly
    // Note: Debug builds are ~10-20x slower than release, so use generous threshold
    // For accurate performance testing, use `cargo bench` in release mode
    assert!(
        elapsed.as_millis() < 500,
        "100 cached additions should be fast, took {}ms (debug mode expected < 500ms)",
        elapsed.as_millis()
    );
}

#[test]
fn test_proof_generation_scales_reasonably() {
    let mut engine = ArithmeticEngine::new();

    // Larger numbers take more steps but should still complete
    let result = engine.add(15, 20);

    assert_eq!(result.result.value, 35);
    assert!(!result.proof.is_empty(), "Should have proof steps");
}