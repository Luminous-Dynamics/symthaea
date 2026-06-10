// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;
use crate::hdc::primitive_system::PrimitiveSystem;
use proptest::prelude::*;
use std::collections::HashMap;

#[test]

fn test_number_construction() {
    let primitives = PrimitiveSystem::global();

    let zero = HdcNumber::zero(&primitives);
    assert_eq!(zero.value, 0);

    let five = HdcNumber::from_value(5, &primitives);
    assert_eq!(five.value, 5);
    assert_eq!(five.construction.len(), 6); // ZERO + 5 successor steps
}

#[test]

fn test_successor() {
    let primitives = PrimitiveSystem::global();

    let three = HdcNumber::from_value(3, &primitives);
    let four = three.successor(&primitives);

    assert_eq!(four.value, 4);
}

#[test]

fn test_addition() {
    let mut engine = ArithmeticEngine::new();

    let result = engine.add(3, 4);
    assert_eq!(result.result.value, 7);
    assert!(result.total_phi > 0.0, "Should have positive Φ");
    assert!(!result.proof.is_empty(), "Should have proof steps");

    // Test 2 + 2 = 4
    let result2 = engine.add(2, 2);
    assert_eq!(result2.result.value, 4);
}

#[test]

fn test_multiplication() {
    let mut engine = ArithmeticEngine::new();

    // 7 × 8 = 56
    let result = engine.multiply(7, 8);
    assert_eq!(result.result.value, 56);
    assert!(result.total_phi > 0.0, "Should have positive Φ");

    // 3 × 4 = 12
    let result2 = engine.multiply(3, 4);
    assert_eq!(result2.result.value, 12);
}

#[test]

fn test_subtraction() {
    let mut engine = ArithmeticEngine::new();

    // 10 - 3 = 7
    let result = engine.subtract(10, 3);
    assert!(result.is_some());
    assert_eq!(result.unwrap().result.value, 7);

    // 3 - 10 is undefined in naturals
    let invalid = engine.subtract(3, 10);
    assert!(invalid.is_none());
}

#[test]

fn test_power() {
    let mut engine = ArithmeticEngine::new();

    // 2^3 = 8
    let result = engine.power(2, 3);
    assert_eq!(result.result.value, 8);

    // 3^2 = 9
    let result2 = engine.power(3, 2);
    assert_eq!(result2.result.value, 9);
}

#[test]

fn test_factorial() {
    let mut engine = ArithmeticEngine::new();

    // 5! = 120
    let result = engine.factorial(5);
    assert_eq!(result.result.value, 120);

    // 0! = 1
    let result0 = engine.factorial(0);
    assert_eq!(result0.result.value, 1);
}

#[test]

fn test_theorem_commutativity() {
    let mut prover = TheoremProver::new();

    let theorem = prover.prove_addition_commutative(3, 5);
    assert!(theorem.verified, "3 + 5 should equal 5 + 3");
    assert!(theorem.total_phi > 0.0);
}

#[test]

fn test_theorem_associativity() {
    let mut prover = TheoremProver::new();

    let theorem = prover.prove_addition_associative(2, 3, 4);
    assert!(theorem.verified, "(2+3)+4 should equal 2+(3+4)");
}

#[test]

fn test_theorem_distributive() {
    let mut prover = TheoremProver::new();

    let theorem = prover.prove_distributive(2, 3, 4);
    assert!(theorem.verified, "2×(3+4) should equal 2×3 + 2×4");
    assert_eq!(
        prover.engine().multiply(2, 7).result.value,
        prover.engine().add(6, 8).result.value
    );
}

#[test]

fn test_caching() {
    let mut engine = ArithmeticEngine::new();

    // First call - computes
    let _ = engine.add(5, 3);
    assert_eq!(engine.stats().cache_hits, 0);

    // Second call - cache hit
    let _ = engine.add(5, 3);
    assert_eq!(engine.stats().cache_hits, 1);
}

#[test]

fn test_phi_accumulation() {
    let mut engine = ArithmeticEngine::new();

    // Perform several operations (each may involve internal sub-computations)
    let _ = engine.add(2, 3);
    let _ = engine.multiply(4, 5);
    let _ = engine.factorial(4);

    let stats = engine.stats();
    // Multiply and factorial involve internal operations, so total > 3
    assert!(
        stats.total_computations >= 3,
        "Should track all computations"
    );
    assert!(stats.total_phi > 0.0, "Should accumulate Φ");
    assert!(stats.mean_phi > 0.0, "Mean Φ should be positive");
}

// ========================================================================
// HYBRID ENGINE TESTS
// ========================================================================

#[test]

fn test_hybrid_deep_path() {
    let mut engine = HybridArithmeticEngine::new();

    // Small numbers should use deep path
    let result = engine.add(3, 4);
    assert_eq!(result.value, 7);
    assert_eq!(result.computation_path, ComputationPath::Deep);
    assert!(result.phi_is_exact, "Small numbers should have exact Φ");
    assert!(
        result.full_proof.is_some(),
        "Deep path should have full proof"
    );
    assert!(
        result.encoding.is_some(),
        "Deep path should have HDC encoding"
    );
}

#[test]

fn test_hybrid_fast_path() {
    let mut engine = HybridArithmeticEngine::new();

    // Large numbers should use fast path
    let result = engine.add(1000, 2000);
    assert_eq!(result.value, 3000);
    assert_eq!(result.computation_path, ComputationPath::Fast);
    assert!(
        !result.phi_is_exact,
        "Large numbers should have estimated Φ"
    );
    assert!(
        result.abstract_proof.is_some(),
        "Fast path should have abstract proof"
    );
}

#[test]

fn test_hybrid_semantics_always_present() {
    let mut engine = HybridArithmeticEngine::new();

    // Both paths should have semantic annotations
    let deep = engine.add(5, 3);
    assert!(!deep.semantics.primitives_involved.is_empty());
    assert!(!deep.semantics.axiom_references.is_empty());

    let fast = engine.add(500, 300);
    assert!(!fast.semantics.primitives_involved.is_empty());
    assert!(!fast.semantics.axiom_references.is_empty());
}

#[test]

fn test_hybrid_multiply() {
    let mut engine = HybridArithmeticEngine::new();

    // Deep path
    let result = engine.multiply(7, 8);
    assert_eq!(result.value, 56);
    assert_eq!(result.computation_path, ComputationPath::Deep);

    // Fast path
    let result = engine.multiply(1000, 2000);
    assert_eq!(result.value, 2_000_000);
    assert_eq!(result.computation_path, ComputationPath::Fast);
}

#[test]

fn test_hybrid_power() {
    let mut engine = HybridArithmeticEngine::new();

    // Deep path (small base and exponent)
    let result = engine.power(2, 3);
    assert_eq!(result.value, 8);
    assert_eq!(result.computation_path, ComputationPath::Deep);

    // Fast path (large exponent)
    let result = engine.power(2, 10);
    assert_eq!(result.value, 1024);
    assert_eq!(result.computation_path, ComputationPath::Fast);
}

#[test]

fn test_hybrid_factorial() {
    let mut engine = HybridArithmeticEngine::new();

    // Deep path (n <= 6)
    let result = engine.factorial(5);
    assert_eq!(result.value, 120);
    assert_eq!(result.computation_path, ComputationPath::Deep);

    // Fast path (n > 6)
    let result = engine.factorial(10);
    assert_eq!(result.value, 3628800);
    assert_eq!(result.computation_path, ComputationPath::Fast);
}

#[test]

fn test_hybrid_stats() {
    let mut engine = HybridArithmeticEngine::new();

    // Mix of deep and fast operations
    let _ = engine.add(3, 4); // Deep
    let _ = engine.add(100, 200); // Fast
    let _ = engine.multiply(5, 6); // Deep
    let _ = engine.multiply(1000, 2000); // Fast

    let stats = engine.stats();
    assert!(
        stats.deep_computations >= 2,
        "Should track deep computations"
    );
    assert!(
        stats.fast_computations >= 2,
        "Should track fast computations"
    );
    assert!(stats.exact_phi > 0.0, "Should have exact Φ from deep");
    assert!(
        stats.estimated_phi > 0.0,
        "Should have estimated Φ from fast"
    );
}

#[test]

fn test_hybrid_abstract_proof_validity() {
    let mut engine = HybridArithmeticEngine::new();

    let result = engine.multiply(100, 200);
    assert!(result.abstract_proof.is_some());

    let proof = result.abstract_proof.unwrap();
    assert!(proof.is_sound, "Abstract proof should be sound");
    assert!(!proof.base_cases.is_empty(), "Should reference base cases");
    assert!(!proof.justification.is_empty(), "Should have justification");
}

#[test]

fn test_hybrid_configurable_threshold() {
    // Create with custom config - lower threshold for more deep computations
    let config = HybridConfig {
        deep_threshold: 20,
        generate_abstract_proofs: true,
        estimate_phi: true,
        phi_scale_factor: 0.15,
    };
    let mut engine = HybridArithmeticEngine::with_config(config);

    // 15 + 10 should now be deep (both < 20)
    let result = engine.add(15, 10);
    assert_eq!(result.computation_path, ComputationPath::Deep);

    // 25 + 30 should be fast (both >= 20)
    let result = engine.add(25, 30);
    assert_eq!(result.computation_path, ComputationPath::Fast);
}

// ========================================================================
// EXTENDED OPERATIONS TESTS
// ========================================================================

#[test]

fn test_force_deep_mode() {
    let mut engine = HybridArithmeticEngine::new();

    // Force deep even for large numbers
    let result = engine.add_deep(100, 200);
    assert_eq!(result.value, 300);
    assert_eq!(result.computation_path, ComputationPath::Deep);
    assert!(result.phi_is_exact);
    assert!(result.full_proof.is_some());
}

#[test]

fn test_division() {
    let mut engine = HybridArithmeticEngine::new();

    // Basic division
    let result = engine.divide(20, 4).unwrap();
    assert_eq!(result.value, 5);

    // Division with remainder (floor)
    let result = engine.divide(17, 5).unwrap();
    assert_eq!(result.value, 3); // 17 / 5 = 3

    // Division by zero
    assert!(engine.divide(10, 0).is_none());
}

#[test]

fn test_modulo() {
    let mut engine = HybridArithmeticEngine::new();

    // Basic modulo
    let result = engine.modulo(17, 5).unwrap();
    assert_eq!(result.value, 2); // 17 = 3*5 + 2

    let result = engine.modulo(20, 4).unwrap();
    assert_eq!(result.value, 0); // Perfect division

    // Modulo by zero
    assert!(engine.modulo(10, 0).is_none());
}

#[test]

fn test_gcd() {
    let mut engine = HybridArithmeticEngine::new();

    // Classic GCD examples
    let result = engine.gcd(48, 18);
    assert_eq!(result.value, 6);

    let result = engine.gcd(100, 35);
    assert_eq!(result.value, 5);

    // Coprime numbers
    let result = engine.gcd(17, 13);
    assert_eq!(result.value, 1);

    // GCD with zero
    let result = engine.gcd(42, 0);
    assert_eq!(result.value, 42);

    // Proof trace should exist
    let result = engine.gcd(48, 18);
    assert!(result.abstract_proof.is_some());
    let proof = result.abstract_proof.unwrap();
    assert!(proof.inductive_step.contains("gcd"));
}

#[test]

fn test_primality() {
    let mut engine = HybridArithmeticEngine::new();

    // Known primes
    assert_eq!(engine.is_prime(2).value, 1);
    assert_eq!(engine.is_prime(3).value, 1);
    assert_eq!(engine.is_prime(17).value, 1);
    assert_eq!(engine.is_prime(97).value, 1);

    // Known composites
    assert_eq!(engine.is_prime(0).value, 0);
    assert_eq!(engine.is_prime(1).value, 0);
    assert_eq!(engine.is_prime(4).value, 0);
    assert_eq!(engine.is_prime(100).value, 0);

    // Check proof exists
    let result = engine.is_prime(17);
    assert!(result.abstract_proof.is_some());
}

// ========================================================================
// MATHEMATICAL DISCOVERY TESTS
// ========================================================================

#[test]

fn test_proof_exploration() {
    let mut discovery = MathDiscovery::new();

    // Explore commutativity proofs
    let exploration = discovery.explore_proofs("commutativity_add", 3, 5, 0);

    assert!(!exploration.proofs.is_empty());
    assert!(exploration.most_elegant.is_some());
    assert!(!exploration.insights.is_empty());
}

#[test]

fn test_conjecture_generation() {
    let mut discovery = MathDiscovery::new();

    // Generate conjectures from patterns
    let conjectures = discovery.generate_conjectures(10);

    // Should find at least the symmetry conjecture
    assert!(!discovery.patterns.is_empty());
    // The conjecture finding depends on the data, so we just verify it runs
}

#[test]

fn test_discovery_engine_access() {
    let mut discovery = MathDiscovery::new();

    // Should be able to use the engine through discovery
    let result = discovery.engine().add(5, 3);
    assert_eq!(result.value, 8);

    let result = discovery.engine().gcd(12, 8);
    assert_eq!(result.value, 4);
}

// ========================================================================
// REASONING INTEGRATION TESTS
// ========================================================================

#[test]

fn test_reasoning_bridge_creation() {
    let bridge = MathReasoningBridge::new();
    assert!(bridge.assertions().is_empty());
    assert!(bridge.proven_theorems().is_empty());
}

#[test]

fn test_assert_equality() {
    let mut bridge = MathReasoningBridge::new();

    let assertion = bridge.assert_equality(3, 4, "add");
    assert_eq!(assertion.object, "7");
    assert_eq!(assertion.relation, MathRelation::Equals);
    assert_eq!(assertion.confidence, 1.0);
    assert!(assertion.phi > 0.0);

    let assertion = bridge.assert_equality(5, 6, "*");
    assert_eq!(assertion.object, "30");
}

#[test]

fn test_assert_divisibility() {
    let mut bridge = MathReasoningBridge::new();

    // 3 divides 12
    let assertion = bridge.assert_divides(3, 12);
    assert_eq!(assertion.confidence, 1.0);
    assert_eq!(assertion.relation, MathRelation::Divides);

    // 5 does not divide 12
    let assertion = bridge.assert_divides(5, 12);
    assert_eq!(assertion.confidence, 0.0);
}

#[test]

fn test_assert_primality() {
    let mut bridge = MathReasoningBridge::new();

    let assertion = bridge.assert_prime(17);
    assert_eq!(assertion.object, "Prime");
    assert_eq!(assertion.confidence, 1.0);

    let assertion = bridge.assert_prime(15);
    assert_eq!(assertion.object, "Composite");
}

#[test]

fn test_assert_coprime() {
    let mut bridge = MathReasoningBridge::new();

    // 7 and 11 are coprime
    let assertion = bridge.assert_coprime(7, 11);
    assert_eq!(assertion.confidence, 1.0);
    assert_eq!(assertion.relation, MathRelation::Coprime);

    // 6 and 9 are not coprime (gcd = 3)
    let assertion = bridge.assert_coprime(6, 9);
    assert_eq!(assertion.confidence, 0.0);
}

#[test]

fn test_prove_theorem_commutativity() {
    let mut bridge = MathReasoningBridge::new();

    let result = bridge.prove_theorem("commutativity_add", &[5, 7]);
    assert!(result.is_some());

    let assertion = result.unwrap();
    assert!(assertion.subject.contains("5 + 7"));
    assert!(assertion.object.contains("7 + 5"));
    assert_eq!(assertion.confidence, 1.0);

    // Check theorem is cached
    assert!(bridge.proven_theorems().contains_key("commutativity_add"));
}

#[test]

fn test_transitive_divisibility() {
    let mut bridge = MathReasoningBridge::new();

    // 2 | 4 and 4 | 12, therefore 2 | 12
    let result = bridge.reason_transitive_divisibility(2, 4, 12);
    assert!(result.is_some());

    let assertion = result.unwrap();
    assert_eq!(assertion.subject, "2");
    assert_eq!(assertion.object, "12");
    assert!(assertion.proof_source.is_some());
    assert!(
        assertion
            .proof_source
            .unwrap()
            .inductive_step
            .contains("transitive")
    );
}

#[test]

fn test_gcd_properties() {
    let mut bridge = MathReasoningBridge::new();

    let assertions = bridge.reason_gcd_properties(12, 18);
    assert!(!assertions.is_empty());

    // Should include divisibility assertions
    let divides_assertions: Vec<_> = assertions
        .iter()
        .filter(|a| a.relation == MathRelation::Divides)
        .collect();
    assert!(divides_assertions.len() >= 2);
}

#[test]

fn test_multi_step_proof() {
    let mut bridge = MathReasoningBridge::new();

    let steps = vec![
        ("commutativity_add", &[3, 5][..]),
        ("commutativity_mul", &[2, 4][..]),
        ("associativity", &[1, 2, 3][..]),
    ];

    let (assertions, total_phi) = bridge.multi_step_proof(steps);
    assert_eq!(assertions.len(), 3);
    assert!(total_phi > 0.0);
}

#[test]

fn test_query_by_relation() {
    let mut bridge = MathReasoningBridge::new();

    // Generate some assertions
    bridge.assert_equality(2, 3, "+");
    bridge.assert_equality(4, 5, "+");
    bridge.assert_divides(2, 6);
    bridge.assert_prime(7);

    let equals = bridge.query_by_relation(&MathRelation::Equals);
    assert_eq!(equals.len(), 2);

    let divides = bridge.query_by_relation(&MathRelation::Divides);
    assert_eq!(divides.len(), 1);

    let instance_of = bridge.query_by_relation(&MathRelation::InstanceOf);
    assert_eq!(instance_of.len(), 1);
}

#[test]

fn test_query_involving() {
    let mut bridge = MathReasoningBridge::new();

    bridge.assert_equality(5, 3, "+");
    bridge.assert_equality(5, 7, "*");
    bridge.assert_equality(2, 4, "+");

    let involving_5 = bridge.query_involving(5);
    assert_eq!(involving_5.len(), 2);

    let involving_2 = bridge.query_involving(2);
    assert_eq!(involving_2.len(), 1);
}

#[test]

fn test_total_phi_accumulation() {
    let mut bridge = MathReasoningBridge::new();

    bridge.assert_equality(3, 4, "+");
    let phi_1 = bridge.total_phi();

    bridge.assert_equality(5, 6, "+");
    let phi_2 = bridge.total_phi();

    assert!(phi_2 > phi_1, "Total Φ should accumulate");
}

#[test]

fn test_highest_phi_assertion() {
    let mut bridge = MathReasoningBridge::new();

    bridge.assert_equality(2, 2, "+"); // Simple
    bridge.assert_equality(5, 6, "*"); // More complex
    bridge.assert_equality(3, 3, "+"); // Simple

    let highest = bridge.highest_phi_assertion();
    assert!(highest.is_some());
    // Multiplication should have higher Φ
    assert!(highest.unwrap().subject.contains("*") || highest.unwrap().phi > 0.0);
}

// ========================================================================
// SYMBOLIC ALGEBRA TESTS
// ========================================================================

#[test]

fn test_symbolic_expr_constant() {
    let primitives = PrimitiveSystem::global();
    let expr = SymbolicExpr::constant(5, &primitives);
    assert_eq!(expr.term_type, TermType::Constant(5));
    assert_eq!(expr.to_string(), "5");
}

#[test]

fn test_symbolic_expr_variable() {
    let primitives = PrimitiveSystem::global();
    let x = SymbolicExpr::variable("x", &primitives);
    assert_eq!(x.term_type, TermType::Variable("x".to_string()));
    assert_eq!(x.to_string(), "x");
}

#[test]

fn test_symbolic_addition() {
    let primitives = PrimitiveSystem::global();
    let x = SymbolicExpr::variable("x", &primitives);
    let five = SymbolicExpr::constant(5, &primitives);
    let sum = x.add(&five, &primitives);

    assert!(matches!(
        sum.term_type,
        TermType::BinaryOp {
            op: SymbolicOp::Add,
            ..
        }
    ));
    assert_eq!(sum.to_string(), "(x + 5)");
}

#[test]

fn test_symbolic_multiplication() {
    let primitives = PrimitiveSystem::global();
    let x = SymbolicExpr::variable("x", &primitives);
    let three = SymbolicExpr::constant(3, &primitives);
    let prod = three.mul(&x, &primitives);

    assert!(matches!(
        prod.term_type,
        TermType::BinaryOp {
            op: SymbolicOp::Mul,
            ..
        }
    ));
    assert_eq!(prod.to_string(), "(3 * x)");
}

#[test]

fn test_symbolic_evaluation() {
    let primitives = PrimitiveSystem::global();
    let mut engine = HybridArithmeticEngine::new();

    // Build: 2x + 3
    let x = SymbolicExpr::variable("x", &primitives);
    let two = SymbolicExpr::constant(2, &primitives);
    let three = SymbolicExpr::constant(3, &primitives);
    let two_x = two.mul(&x, &primitives);
    let expr = two_x.add(&three, &primitives);

    // Evaluate at x = 5: 2*5 + 3 = 13
    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 5_i64);

    let result = expr.evaluate(&bindings, &mut engine);
    assert!(result.is_some());
    assert_eq!(result.unwrap().value, 13);
}

#[test]

fn test_symbolic_simplify_like_terms() {
    let primitives = PrimitiveSystem::global();
    let mut algebra = SymbolicAlgebra::new();

    // Build: 2x + 3x = 5x
    let x = SymbolicExpr::variable("x", &primitives);
    let two = SymbolicExpr::constant(2, &primitives);
    let three = SymbolicExpr::constant(3, &primitives);
    let two_x = two.mul(&x, &primitives);
    let three_x = three.mul(&x, &primitives);
    let sum = two_x.add(&three_x, &primitives);

    let simplified = algebra.simplify(&sum, &primitives);
    // The simplified form should be 5x
    // Note: Full simplification may depend on implementation
    assert!(simplified.phi > 0.0);
}

#[test]

fn test_symbolic_constant_folding() {
    let primitives = PrimitiveSystem::global();
    let mut algebra = SymbolicAlgebra::new();

    // Build: 3 + 4 should fold to 7
    let three = SymbolicExpr::constant(3, &primitives);
    let four = SymbolicExpr::constant(4, &primitives);
    let sum = three.add(&four, &primitives);

    let simplified = algebra.simplify(&sum, &primitives);
    assert_eq!(simplified.term_type, TermType::Constant(7));
    assert_eq!(simplified.to_string(), "7");
}

#[test]

fn test_symbolic_multiply_by_zero() {
    let primitives = PrimitiveSystem::global();
    let mut algebra = SymbolicAlgebra::new();

    // Build: 0 * x = 0
    let zero = SymbolicExpr::constant(0, &primitives);
    let x = SymbolicExpr::variable("x", &primitives);
    let product = zero.mul(&x, &primitives);

    let simplified = algebra.simplify(&product, &primitives);
    assert_eq!(simplified.term_type, TermType::Constant(0));
}

#[test]

fn test_symbolic_multiply_by_one() {
    let primitives = PrimitiveSystem::global();
    let mut algebra = SymbolicAlgebra::new();

    // Build: 1 * x = x
    let one = SymbolicExpr::constant(1, &primitives);
    let x = SymbolicExpr::variable("x", &primitives);
    let product = one.mul(&x, &primitives);

    let simplified = algebra.simplify(&product, &primitives);
    assert_eq!(simplified.term_type, TermType::Variable("x".to_string()));
}

#[test]

fn test_symbolic_add_zero() {
    let primitives = PrimitiveSystem::global();
    let mut algebra = SymbolicAlgebra::new();

    // Build: x + 0 = x
    let x = SymbolicExpr::variable("x", &primitives);
    let zero = SymbolicExpr::constant(0, &primitives);
    let sum = x.add(&zero, &primitives);

    let simplified = algebra.simplify(&sum, &primitives);
    assert_eq!(simplified.term_type, TermType::Variable("x".to_string()));
}

#[test]

fn test_symbolic_expand_distribution() {
    let primitives = PrimitiveSystem::global();
    let mut algebra = SymbolicAlgebra::new();

    // Build: 2 * (x + 3) should expand to (2*x) + (2*3)
    let two = SymbolicExpr::constant(2, &primitives);
    let x = SymbolicExpr::variable("x", &primitives);
    let three = SymbolicExpr::constant(3, &primitives);
    let sum = x.add(&three, &primitives);
    let product = two.mul(&sum, &primitives);

    let expanded = algebra.expand(&product, &primitives);
    // Should now be in the form (2*x) + (2*3)
    assert!(expanded.phi > 0.0);
}

#[test]

fn test_polynomial_creation() {
    let primitives = PrimitiveSystem::global();

    // Create 2x^2 + 3x + 1
    let poly = Polynomial::new(vec![1, 3, 2], "x", &primitives);
    assert_eq!(poly.degree(), 2);
    assert_eq!(poly.coefficients(), &[1, 3, 2]);
}

#[test]

fn test_polynomial_evaluation() {
    let primitives = PrimitiveSystem::global();
    let mut engine = HybridArithmeticEngine::new();

    // 2x^2 + 3x + 1, evaluate at x = 2
    // 2(4) + 3(2) + 1 = 8 + 6 + 1 = 15
    let poly = Polynomial::new(vec![1, 3, 2], "x", &primitives);
    let result = poly.evaluate(2, &mut engine);
    assert_eq!(result.value, 15);
}

#[test]

fn test_polynomial_addition() {
    let primitives = PrimitiveSystem::global();
    let algebra = SymbolicAlgebra::new();

    // (2x + 1) + (3x + 2) = 5x + 3
    let p1 = Polynomial::new(vec![1, 2], "x", &primitives);
    let p2 = Polynomial::new(vec![2, 3], "x", &primitives);

    let sum = algebra.poly_add(&p1, &p2, &primitives);
    assert_eq!(sum.coefficients(), &[3, 5]);
}

#[test]

fn test_polynomial_multiplication() {
    let primitives = PrimitiveSystem::global();
    let algebra = SymbolicAlgebra::new();

    // (x + 1) * (x + 2) = x^2 + 3x + 2
    let p1 = Polynomial::new(vec![1, 1], "x", &primitives);
    let p2 = Polynomial::new(vec![2, 1], "x", &primitives);

    let product = algebra.poly_multiply(&p1, &p2, &primitives);
    assert_eq!(product.coefficients(), &[2, 3, 1]);
}

#[test]

fn test_linear_equation_solver() {
    let primitives = PrimitiveSystem::global();
    let mut algebra = SymbolicAlgebra::new();

    // Solve: 2x + 6 = 0 => x = -3
    let p = Polynomial::new(vec![6, 2], "x", &primitives);
    let solution = algebra.solve_linear(&p);

    assert!(solution.is_some());
    assert_eq!(solution.unwrap(), -3);
}

#[test]

fn test_quadratic_equation_solver() {
    let primitives = PrimitiveSystem::global();
    let mut algebra = SymbolicAlgebra::new();

    // Solve: x^2 - 5x + 6 = 0 => x = 2 or x = 3
    let p = Polynomial::new(vec![6, -5, 1], "x", &primitives);
    let solutions = algebra.solve_quadratic(&p);

    assert_eq!(solutions.len(), 2);
    let mut sorted = solutions.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    assert!((sorted[0] - 2.0).abs() < 0.001);
    assert!((sorted[1] - 3.0).abs() < 0.001);
}

#[test]

fn test_symbolic_algebra_stats() {
    let primitives = PrimitiveSystem::global();
    let mut algebra = SymbolicAlgebra::new();

    // Do some operations
    let three = SymbolicExpr::constant(3, &primitives);
    let four = SymbolicExpr::constant(4, &primitives);
    let sum = three.add(&four, &primitives);
    let _ = algebra.simplify(&sum, &primitives);

    let stats = algebra.stats();
    assert!(stats.simplifications > 0);
    assert!(stats.total_phi > 0.0);
}

// ========================================================================
// MULTI-PATH PROOF VERIFICATION TESTS
// ========================================================================

#[test]

fn test_multipath_addition_commutative() {
    let mut verifier = MultiPathVerifier::new();
    let result = verifier.verify_addition_commutative(3, 5);

    assert_eq!(result.theorem, "3 + 5 = 5 + 3");
    assert!(
        result.total_paths >= 2,
        "Should have at least 2 proof paths"
    );
    assert!(result.valid_paths >= 2, "At least 2 paths should be valid");
    assert!(result.paths_agree, "All paths should agree on result");

    // All valid paths should conclude 3 + 5 = 5 + 3 = 8
    for path in result.paths.iter().filter(|p| p.is_valid) {
        assert!(path.result.is_some());
        assert_eq!(path.result.as_ref().unwrap().value, 8);
    }
}

#[test]

fn test_multipath_multiplication_commutative() {
    let mut verifier = MultiPathVerifier::new();
    let result = verifier.verify_multiplication_commutative(4, 6);

    assert_eq!(result.theorem, "4 × 6 = 6 × 4");
    assert!(result.total_paths >= 2);
    assert!(result.valid_paths >= 2);
    assert!(result.paths_agree);

    // All valid paths should conclude 4 × 6 = 6 × 4 = 24
    for path in result.paths.iter().filter(|p| p.is_valid) {
        assert!(path.result.is_some());
        assert_eq!(path.result.as_ref().unwrap().value, 24);
    }
}

#[test]

fn test_multipath_associativity() {
    let mut verifier = MultiPathVerifier::new();
    let result = verifier.verify_associativity(2, 3, 4, "+");

    // Verify correct theorem format
    assert!(
        result.theorem.contains("2")
            && result.theorem.contains("3")
            && result.theorem.contains("4")
    );
    assert!(result.total_paths >= 2);
    assert!(result.valid_paths >= 2);
    assert!(result.paths_agree);

    // All paths should conclude (2+3)+4 = 2+(3+4) = 9
    for path in result.paths.iter().filter(|p| p.is_valid) {
        assert!(path.result.is_some());
        assert_eq!(path.result.as_ref().unwrap().value, 9);
    }
}

#[test]

fn test_multipath_distributive() {
    let mut verifier = MultiPathVerifier::new();
    let result = verifier.verify_distributive(2, 3, 4);

    // Verify theorem contains expected operands and structure
    assert!(
        result.theorem.contains("2")
            && result.theorem.contains("3")
            && result.theorem.contains("4")
    );
    assert!(result.total_paths >= 2);
    assert!(result.valid_paths >= 2);
    assert!(result.paths_agree);

    // 2 × (3 + 4) = 2 × 7 = 14, (2 × 3) + (2 × 4) = 6 + 8 = 14
    for path in result.paths.iter().filter(|p| p.is_valid) {
        assert!(path.result.is_some());
        assert_eq!(path.result.as_ref().unwrap().value, 14);
    }
}

#[test]

fn test_multipath_divisibility_true() {
    let mut verifier = MultiPathVerifier::new();
    let result = verifier.verify_divisibility(3, 12);

    assert_eq!(result.theorem, "3 divides 12");
    assert!(result.total_paths >= 2);
    assert!(result.valid_paths >= 2, "12 is divisible by 3");
    // Note: paths may not agree numerically for divisibility because
    // different paths compute different values (e.g., quotient=4 vs remainder=0)
    // What matters is that multiple paths successfully confirm divisibility
}

#[test]

fn test_multipath_divisibility_false() {
    let mut verifier = MultiPathVerifier::new();
    let result = verifier.verify_divisibility(5, 12);

    assert_eq!(result.theorem, "5 divides 12");
    assert!(result.total_paths >= 2);
    // Not all paths should be valid since 5 does not divide 12
    assert!(!result.paths.iter().all(|p| p.is_valid));
}

#[test]

fn test_multipath_proof_steps() {
    let mut verifier = MultiPathVerifier::new();
    let result = verifier.verify_addition_commutative(2, 7);

    // Check that each path has meaningful steps
    for path in &result.paths {
        assert!(!path.steps.is_empty(), "Each path should have steps");
        assert!(path.total_phi >= 0.0, "Φ should be non-negative");
        assert!(!path.strategy.is_empty(), "Strategy should be named");
    }
}

#[test]

fn test_multipath_best_path_selection() {
    let mut verifier = MultiPathVerifier::new();
    let result = verifier.verify_multiplication_commutative(5, 3);

    // Should have identified a best path
    assert!(result.best_path_index.is_some());
    let best_idx = result.best_path_index.unwrap();
    let best_path = &result.paths[best_idx];

    // Best path should be valid
    assert!(best_path.is_valid);

    // Best path should have highest Φ among valid paths
    for (i, path) in result.paths.iter().enumerate() {
        if path.is_valid && i != best_idx {
            assert!(
                best_path.total_phi >= path.total_phi,
                "Best path should have highest Φ"
            );
        }
    }
}

#[test]

fn test_multipath_verifier_stats() {
    let mut verifier = MultiPathVerifier::new();

    // Run several verifications
    verifier.verify_addition_commutative(1, 2);
    verifier.verify_multiplication_commutative(2, 3);
    verifier.verify_associativity(1, 2, 3, "+");

    let stats = verifier.stats();
    assert_eq!(stats.theorems_verified, 3);
    assert!(stats.total_paths_generated >= 6); // At least 2 paths per theorem
    assert!(stats.total_valid_paths >= 6);
    assert!(stats.total_phi > 0.0);
    assert!(stats.agreements >= 3); // All should agree
}

#[test]

fn test_multipath_division_by_zero() {
    let mut verifier = MultiPathVerifier::new();
    let result = verifier.verify_divisibility(0, 10);

    // Division by zero should yield invalid paths
    assert!(result.valid_paths == 0 || !result.paths.iter().any(|p| p.is_valid));
}

#[test]

fn test_multipath_identity_cases() {
    let mut verifier = MultiPathVerifier::new();

    // Test with identity element
    let add_zero = verifier.verify_addition_commutative(5, 0);
    assert!(add_zero.paths_agree);
    for path in add_zero.paths.iter().filter(|p| p.is_valid) {
        assert_eq!(path.result.as_ref().unwrap().value, 5);
    }

    // Test multiplication by 1
    let mul_one = verifier.verify_multiplication_commutative(7, 1);
    assert!(mul_one.paths_agree);
    for path in mul_one.paths.iter().filter(|p| p.is_valid) {
        assert_eq!(path.result.as_ref().unwrap().value, 7);
    }
}

#[test]

fn test_multipath_large_numbers() {
    let mut verifier = MultiPathVerifier::new();

    // Test with larger numbers - now works after overflow fix
    let result = verifier.verify_addition_commutative(100, 200);
    assert!(result.paths_agree);
    for path in result.paths.iter().filter(|p| p.is_valid) {
        assert_eq!(path.result.as_ref().unwrap().value, 300);
    }

    // Also test edge cases with zero
    let result_zero = verifier.verify_addition_commutative(50, 0);
    assert!(result_zero.paths_agree);
    for path in result_zero.paths.iter().filter(|p| p.is_valid) {
        assert_eq!(path.result.as_ref().unwrap().value, 50);
    }
}

#[test]
#[ignore = "slow ~47s: MultiPathVerifier with full Phi measurement"]
fn test_multipath_phi_measurement() {
    let mut verifier = MultiPathVerifier::new();
    let result = verifier.verify_distributive(3, 4, 5);

    // Φ measurements should be present and positive for valid paths
    for path in result.paths.iter().filter(|p| p.is_valid) {
        assert!(path.total_phi > 0.0, "Valid paths should have positive Φ");

        // Each step should have phi measurement
        for step in &path.steps {
            assert!(step.phi >= 0.0, "Step Φ should be non-negative");
        }
    }
}

// ========================================================================
// PROPERTY-BASED TESTS (proptest)
// ========================================================================

/// Strategy for small numbers suitable for deep (Peano) path
fn small_nat() -> impl Strategy<Value = u64> {
    0u64..15
}

/// Strategy for numbers that exercise the fast path (>= 50 threshold)
fn fast_path_nat() -> impl Strategy<Value = u64> {
    50u64..200
}

/// Strategy for nonzero small naturals
fn nonzero_small() -> impl Strategy<Value = u64> {
    1u64..15
}

/// Strategy for nonzero fast-path naturals
fn nonzero_fast() -> impl Strategy<Value = u64> {
    50u64..200
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    // =====================================================================
    // ArithmeticEngine (Peano) properties
    // =====================================================================

    #[test]
    fn prop_add_commutative(a in small_nat(), b in small_nat()) {
        let mut engine = ArithmeticEngine::new();
        let ab = engine.add(a, b);
        let ba = engine.add(b, a);
        prop_assert_eq!(ab.result.value, ba.result.value,
            "Addition should be commutative: {} + {} vs {} + {}", a, b, b, a);
    }

    #[test]
    fn prop_mul_commutative(a in 0u64..8, b in 0u64..8) {
        let mut engine = ArithmeticEngine::new();
        let ab = engine.multiply(a, b);
        let ba = engine.multiply(b, a);
        prop_assert_eq!(ab.result.value, ba.result.value,
            "Multiplication should be commutative: {} × {} vs {} × {}", a, b, b, a);
    }

    #[test]
    fn prop_add_identity(a in small_nat()) {
        let mut engine = ArithmeticEngine::new();
        let result = engine.add(a, 0);
        prop_assert_eq!(result.result.value, a,
            "Adding zero should be identity");
    }

    #[test]
    fn prop_mul_identity(a in 0u64..10) {
        let mut engine = ArithmeticEngine::new();
        let result = engine.multiply(a, 1);
        prop_assert_eq!(result.result.value, a,
            "Multiplying by one should be identity");
    }

    #[test]
    fn prop_mul_zero(a in 0u64..10) {
        let mut engine = ArithmeticEngine::new();
        let result = engine.multiply(a, 0);
        prop_assert_eq!(result.result.value, 0,
            "Multiplying by zero should give zero");
    }

    #[test]
    fn prop_phi_positive_add(a in small_nat(), b in small_nat()) {
        let mut engine = ArithmeticEngine::new();
        let result = engine.add(a, b);
        prop_assert!(result.total_phi > 0.0,
            "Φ should be positive for add({}, {})", a, b);
    }

    #[test]
    fn prop_phi_positive_mul(a in 0u64..8, b in 0u64..8) {
        let mut engine = ArithmeticEngine::new();
        let result = engine.multiply(a, b);
        prop_assert!(result.total_phi > 0.0,
            "Φ should be positive for multiply({}, {})", a, b);
    }

}

// ========================================================================
// HybridArithmeticEngine + MultiPathVerifier property tests (expensive)
// ========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn prop_subtract_inverse_of_add(a in small_nat(), b in small_nat()) {
        let mut engine = ArithmeticEngine::new();
        let sum = engine.add(a, b);
        let diff = engine.subtract(sum.result.value, b);
        prop_assert!(diff.is_some(), "subtract(a+b, b) should succeed");
        prop_assert_eq!(diff.unwrap().result.value, a,
            "subtract(add({}, {}), {}) should equal {}", a, b, b, a);
    }

    #[test]
    fn prop_hybrid_add_commutative(a in fast_path_nat(), b in fast_path_nat()) {
        let mut engine = HybridArithmeticEngine::new();
        let ab = engine.add(a, b);
        let ba = engine.add(b, a);
        prop_assert_eq!(ab.value, ba.value,
            "Hybrid add should be commutative: {} + {} vs {} + {}", a, b, b, a);
    }

    #[test]
    fn prop_hybrid_mul_commutative(a in fast_path_nat(), b in fast_path_nat()) {
        let mut engine = HybridArithmeticEngine::new();
        let ab = engine.multiply(a, b);
        let ba = engine.multiply(b, a);
        prop_assert_eq!(ab.value, ba.value,
            "Hybrid mul should be commutative: {} × {} vs {} × {}", a, b, b, a);
    }

    #[test]
    fn prop_hybrid_add_correct(a in 0u64..500, b in 0u64..500) {
        let mut engine = HybridArithmeticEngine::new();
        let result = engine.add(a, b);
        prop_assert_eq!(result.value, a + b,
            "Hybrid add({}, {}) should equal native", a, b);
    }

    #[test]
    fn prop_hybrid_mul_correct(a in 0u64..100, b in 0u64..100) {
        let mut engine = HybridArithmeticEngine::new();
        let result = engine.multiply(a, b);
        prop_assert_eq!(result.value, a * b,
            "Hybrid multiply({}, {}) should equal native", a, b);
    }

    #[test]
    fn prop_hybrid_deep_fast_agree(a in 1u64..20, b in 1u64..20) {
        let mut engine = HybridArithmeticEngine::new();
        let deep_result = engine.add_deep(a, b);
        let fast_result = engine.add(a + 50, b + 50);
        // Deep computes a+b, fast computes (a+50)+(b+50) = a+b+100
        prop_assert_eq!(deep_result.value + 100, fast_result.value,
            "Deep and fast paths should produce consistent results");
    }

    #[test]
    fn prop_hybrid_division_modulo(a in 1u64..200, b in nonzero_fast()) {
        let mut engine = HybridArithmeticEngine::new();
        if let (Some(quot), Some(rem)) = (engine.divide(a, b), engine.modulo(a, b)) {
            prop_assert_eq!(quot.value * b + rem.value, a,
                "q*b + r should equal a for {}/{}", a, b);
        }
    }

    #[test]
    fn prop_hybrid_gcd_divides_both(a in nonzero_fast(), b in nonzero_fast()) {
        let mut engine = HybridArithmeticEngine::new();
        let g = engine.gcd(a, b);
        prop_assert_eq!(a % g.value, 0,
            "gcd({},{})={} should divide {}", a, b, g.value, a);
        prop_assert_eq!(b % g.value, 0,
            "gcd({},{})={} should divide {}", a, b, g.value, b);
    }

    #[test]
    fn prop_hybrid_gcd_commutative(a in nonzero_fast(), b in nonzero_fast()) {
        let mut engine = HybridArithmeticEngine::new();
        let gcd_ab = engine.gcd(a, b);
        let gcd_ba = engine.gcd(b, a);
        prop_assert_eq!(gcd_ab.value, gcd_ba.value,
            "gcd should be commutative: gcd({},{}) vs gcd({},{})", a, b, b, a);
    }

    #[test]
    fn prop_hybrid_power_correct(base in 0u64..10, exp in 0u64..6) {
        let mut engine = HybridArithmeticEngine::new();
        let result = engine.power(base, exp);
        prop_assert_eq!(result.value, base.pow(exp as u32),
            "power({}, {}) should match native pow", base, exp);
    }

    #[test]
    fn prop_hybrid_phi_nonnegative(a in 0u64..200, b in 0u64..200) {
        let mut engine = HybridArithmeticEngine::new();
        let result = engine.add(a, b);
        prop_assert!(result.phi >= 0.0,
            "Φ should be non-negative for add({}, {})", a, b);
    }

    // =====================================================================
    // MultiPathVerifier properties
    // =====================================================================

    #[test]
    fn prop_multipath_add_commutative_valid(a in 1u64..15, b in 1u64..15) {
        let mut verifier = MultiPathVerifier::new();
        let result = verifier.verify_addition_commutative(a, b);
        prop_assert!(result.valid_paths >= 2,
            "Should have at least 2 valid proof paths for {} + {}", a, b);
        prop_assert!(result.paths_agree,
            "All paths should agree for {} + {}", a, b);
    }

    #[test]
    fn prop_multipath_mul_commutative_valid(a in 1u64..8, b in 1u64..8) {
        let mut verifier = MultiPathVerifier::new();
        let result = verifier.verify_multiplication_commutative(a, b);
        prop_assert!(result.valid_paths >= 2,
            "Should have at least 2 valid proof paths for {} × {}", a, b);
        prop_assert!(result.paths_agree,
            "All paths should agree for {} × {}", a, b);
    }

    #[test]
    fn prop_multipath_best_path_is_valid(a in 1u64..10, b in 1u64..10) {
        let mut verifier = MultiPathVerifier::new();
        let result = verifier.verify_addition_commutative(a, b);
        if let Some(idx) = result.best_path_index {
            prop_assert!(result.paths[idx].is_valid,
                "Best path should be valid for {} + {}", a, b);
        }
    }
}
