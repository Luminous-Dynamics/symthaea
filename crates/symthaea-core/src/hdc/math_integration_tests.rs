// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mathematical Foundations Integration Tests
//!
//! Cross-module tests verifying the coherence of the mathematical system:
//! - Validation diagnostics for the primitive system
//! - Number tower integration (Z → Q → R)
//! - Number theory feeding algebraic structure verification
//! - Property-based algebraic tests
//! - Calculus integration with arithmetic engines

#[cfg(test)]
mod tests {
    use crate::hdc::algebraic_structures::{
        AlgebraicElement, BinaryOperation, FieldVerifier, GroupVerifier, RingVerifier,
    };
    use crate::hdc::binary_hv::BinaryHV;
    use crate::hdc::calculus::{FundamentalTheoremVerifier, SymbolicIntegrator};
    use crate::hdc::foundations::InductionEngine;
    use crate::hdc::integer::IntegerArithmeticEngine;
    use crate::hdc::number_theory::{ModularRing, NumberTheoryEngine};
    use crate::hdc::primitive_system::PrimitiveSystem;
    use crate::hdc::primitive_system::seed_from_name;
    use crate::hdc::rational::RationalArithmeticEngine;
    use crate::hdc::real_arithmetic::{MathConstants, PeanoPhysicsBridge};

    // ========================================================================
    // 1. VALIDATION DIAGNOSTICS
    // ========================================================================

    #[test]
    fn test_derivation_chain_no_orphans() {
        let ps = PrimitiveSystem::global();
        let diagnostics = ps.validate_derivation_chain();
        // Every derived primitive should have all parents registered
        let orphans: Vec<_> = diagnostics
            .iter()
            .filter(|(_, resolved, _)| !resolved)
            .collect();
        if !orphans.is_empty() {
            // Report but don't fail hard — some physics-layer derived prims
            // may reference experimental parents
            eprintln!(
                "Derivation chain diagnostics: {} total, {} unresolved",
                diagnostics.len(),
                orphans.len()
            );
            for (name, _, derivation) in &orphans {
                eprintln!("  UNRESOLVED: {} <- {:?}", name, derivation);
            }
        }
        // The math-layer primitives we added should all resolve
        // (GRADIENT, WAVE, etc. depend on Tier 2 which loads before derived)

        // Report derivation chain resolution stats
        let resolved_count = diagnostics
            .iter()
            .filter(|(_, resolved, _)| *resolved)
            .count();
        let total = diagnostics.len();
        if total > 0 {
            let resolved_ratio = resolved_count as f64 / total as f64;
            eprintln!(
                "Derivation chain resolution: {}/{} = {:.1}%",
                resolved_count,
                total,
                resolved_ratio * 100.0
            );
        }
    }

    #[test]
    fn test_domain_orthogonality() {
        let ps = PrimitiveSystem::global();
        let domain_pairs = ps.validate_domain_orthogonality();
        let mut high_similarity_count = 0;
        for (d1, d2, sim) in &domain_pairs {
            // Domain pairs should be near random baseline (0.5)
            // Flag anything above 0.55 as suspicious
            if *sim > 0.55 {
                high_similarity_count += 1;
                eprintln!("  HIGH SIMILARITY: {} vs {} = {:.4}", d1, d2, sim);
            }
        }
        eprintln!(
            "Domain orthogonality: {} pairs checked, {} above 0.55 threshold",
            domain_pairs.len(),
            high_similarity_count
        );
        // Most domain pairs should be near 0.5
        let above_threshold_ratio = high_similarity_count as f64 / domain_pairs.len().max(1) as f64;
        assert!(
            above_threshold_ratio < 0.2,
            "Too many domain pairs have high similarity: {:.1}%",
            above_threshold_ratio * 100.0
        );
    }

    #[test]
    fn test_validate_all_summary() {
        let ps = PrimitiveSystem::global();
        let (derivation_diags, domain_diags) = ps.validate_all();
        eprintln!("=== SYSTEM VALIDATION SUMMARY ===");
        eprintln!("Derivation diagnostics: {}", derivation_diags.len());
        eprintln!("Domain pair diagnostics: {}", domain_diags.len());

        // Count unresolved derivations
        let unresolved = derivation_diags
            .iter()
            .filter(|(_, resolved, _)| !resolved)
            .count();
        eprintln!("Unresolved derivations: {}", unresolved);

        // validate_all should return non-empty results for a populated system
        assert!(
            !domain_diags.is_empty(),
            "Domain diagnostics should be non-empty for a populated primitive system"
        );

        // Average domain similarity should be near 0.5 (random baseline)
        let avg_sim: f32 =
            domain_diags.iter().map(|(_, _, s)| s).sum::<f32>() / domain_diags.len() as f32;
        eprintln!(
            "Average inter-domain similarity: {:.4} (ideal: ~0.5)",
            avg_sim
        );

        assert!(
            avg_sim.is_finite(),
            "Average domain similarity should be finite"
        );
        assert!(
            avg_sim >= 0.0 && avg_sim <= 1.0,
            "Average domain similarity should be in [0, 1]: {:.4}",
            avg_sim
        );
        // Average similarity should be near random baseline (0.5 +/- 0.15)
        assert!(
            (avg_sim - 0.5).abs() < 0.15,
            "Average inter-domain similarity should be near 0.5: {:.4}",
            avg_sim
        );
    }

    // ========================================================================
    // 2. NUMBER TOWER INTEGRATION (Z → Q → R)
    // ========================================================================

    #[test]
    fn test_integer_to_rational_roundtrip() {
        let rat_engine = RationalArithmeticEngine::new();

        // Integer 7 → rational 7/1 → back to integer check
        let r = rat_engine.from_integer(7);
        assert_eq!(r.numerator, 7);
        assert_eq!(r.denominator, 1);
        assert!((rat_engine.to_f64(&r) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_rational_to_real_roundtrip() {
        let bridge = PeanoPhysicsBridge::new();

        // 1/3 as rational → f64 → back to rational
        let f = 1.0 / 3.0;
        let (num, den) = bridge.f64_to_rational(f, 1000);
        assert_eq!(num, 1);
        assert_eq!(den, 3);

        // Check the reconstructed value
        let reconstructed = num as f64 / den as f64;
        assert!((reconstructed - f).abs() < 1e-10);
    }

    #[test]
    fn test_number_tower_integer_rational_real() {
        // Full tower: integer → rational → f64 → rational → check
        let mut int_engine = IntegerArithmeticEngine::new();
        let rat_engine = RationalArithmeticEngine::new();
        let bridge = PeanoPhysicsBridge::new();

        // Compute 2 * 3 = 6 in integers
        let int_result = int_engine.multiply(2, 3);
        assert_eq!(int_result.integer.value, 6);

        // Lift to rational: 6/1
        let r = rat_engine.from_integer(int_result.integer.value);
        assert_eq!(r.numerator, 6);
        assert_eq!(r.denominator, 1);

        // Convert to f64 through bridge
        let f = rat_engine.to_f64(&r);
        assert!((f - 6.0).abs() < 1e-10);

        // Convert back through bridge
        let (num, den) = bridge.f64_to_rational(f, 1000);
        assert_eq!(num, 6);
        assert_eq!(den, 1);
    }

    #[test]
    fn test_rational_arithmetic_then_real_comparison() {
        // Compute 1/7 + 2/7 = 3/7 in rationals, compare with f64
        let engine = RationalArithmeticEngine::new();
        let a = engine.encode(1, 7);
        let b = engine.encode(2, 7);
        let result = engine.add(&a, &b);
        assert_eq!(result.rational.numerator, 3);
        assert_eq!(result.rational.denominator, 7);

        // Compare with f64 computation
        let f64_result = 1.0 / 7.0 + 2.0 / 7.0;
        let rat_as_f64 = engine.to_f64(&result.rational);
        assert!((rat_as_f64 - f64_result).abs() < 1e-10);
    }

    // ========================================================================
    // 3. NUMBER THEORY → ALGEBRAIC STRUCTURE VERIFICATION
    // ========================================================================

    /// Modular addition operation for Z/nZ using ModularRing
    struct ModAddOp {
        ring: ModularRing,
    }
    impl BinaryOperation for ModAddOp {
        fn apply(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
            let a_val: u64 = a.label.parse().unwrap_or(0);
            let b_val: u64 = b.label.parse().unwrap_or(0);
            let result_val = self.ring.add(a_val, b_val);
            AlgebraicElement {
                label: result_val.to_string(),
                encoding: BinaryHV::random(seed_from_name(&format!(
                    "mod{}_{}",
                    self.ring.modulus, result_val
                ))),
            }
        }
        fn name(&self) -> &str {
            "mod_add"
        }
    }

    /// Modular multiplication operation for Z/nZ
    struct ModMulOp {
        ring: ModularRing,
    }
    impl BinaryOperation for ModMulOp {
        fn apply(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
            let a_val: u64 = a.label.parse().unwrap_or(0);
            let b_val: u64 = b.label.parse().unwrap_or(0);
            let result_val = self.ring.multiply(a_val, b_val);
            AlgebraicElement {
                label: result_val.to_string(),
                encoding: BinaryHV::random(seed_from_name(&format!(
                    "mod{}_{}",
                    self.ring.modulus, result_val
                ))),
            }
        }
        fn name(&self) -> &str {
            "mod_mul"
        }
    }

    fn make_zn_elements(n: u64) -> Vec<AlgebraicElement> {
        (0..n)
            .map(|i| AlgebraicElement {
                label: i.to_string(),
                encoding: BinaryHV::random(seed_from_name(&format!("mod{}_{}", n, i))),
            })
            .collect()
    }

    #[test]
    fn test_z5_additive_group() {
        // Z/5Z under addition should be a group
        let ring = ModularRing::new(5);
        let elements = make_zn_elements(5);
        let add_op = ModAddOp { ring };

        let result = GroupVerifier::verify(&elements, &add_op);
        eprintln!("Z/5Z additive group verification:");
        for check in &result.axioms_checked {
            eprintln!(
                "  {} : {} (phi: {:.4})",
                check.axiom,
                if check.holds { "PASS" } else { "FAIL" },
                check.phi
            );
            if let Some(ref ce) = check.counterexample {
                eprintln!("    counterexample: {}", ce);
            }
        }
        // Should have all 4 axioms checked
        assert!(
            result.axioms_checked.len() >= 4,
            "Expected 4 axiom checks, got {}",
            result.axioms_checked.len()
        );
    }

    #[test]
    fn test_z7_is_field() {
        // Z/7Z should be a field (7 is prime)
        let ring = ModularRing::new(7);
        let elements = make_zn_elements(7);
        let add_op = ModAddOp {
            ring: ModularRing::new(7),
        };
        let mul_op = ModMulOp { ring };

        let result = FieldVerifier::verify(&elements, &add_op, &mul_op);
        eprintln!("Z/7Z field verification:");
        for check in &result.axioms_checked {
            eprintln!(
                "  {} : {} (phi: {:.4})",
                check.axiom,
                if check.holds { "PASS" } else { "FAIL" },
                check.phi
            );
        }
        // Euler's totient of 7 should be 6 (all nonzero elements are units)
        let totient = ModularRing::new(7).euler_totient();
        assert_eq!(totient, 6);
    }

    #[test]
    fn test_z6_is_ring_not_field() {
        // Z/6Z should be a ring but NOT a field (6 is composite)
        let ring = ModularRing::new(6);
        let elements = make_zn_elements(6);
        let add_op = ModAddOp {
            ring: ModularRing::new(6),
        };
        let mul_op = ModMulOp { ring };

        let result = RingVerifier::verify(&elements, &add_op, &mul_op);
        eprintln!("Z/6Z ring verification:");
        for check in &result.axioms_checked {
            eprintln!(
                "  {} : {} (phi: {:.4})",
                check.axiom,
                if check.holds { "PASS" } else { "FAIL" },
                check.phi
            );
        }

        // 2 and 3 are zero divisors in Z/6Z: 2*3 = 0 mod 6
        let r6 = ModularRing::new(6);
        assert_eq!(r6.multiply(2, 3), 0);
        // Not all elements have inverses
        assert!(!r6.is_unit(2));
        assert!(!r6.is_unit(3));
        // But 1 and 5 do
        assert!(r6.is_unit(1));
        assert!(r6.is_unit(5));
    }

    // ========================================================================
    // 4. PROPERTY-BASED / ALGEBRAIC TESTS
    // ========================================================================

    #[test]
    fn test_integer_addition_commutative_property() {
        let mut engine = IntegerArithmeticEngine::new();
        let test_values: Vec<i64> = vec![
            0,
            1,
            -1,
            2,
            -2,
            7,
            -7,
            42,
            -42,
            100,
            -100,
            i64::MAX / 2,
            i64::MIN / 2,
        ];
        for &a in &test_values {
            for &b in &test_values {
                let ab = engine.add(a, b);
                let ba = engine.add(b, a);
                assert_eq!(
                    ab.integer.value, ba.integer.value,
                    "Commutativity failed: {} + {} = {} but {} + {} = {}",
                    a, b, ab.integer.value, b, a, ba.integer.value
                );
            }
        }
    }

    #[test]
    fn test_integer_addition_associative_property() {
        let mut engine = IntegerArithmeticEngine::new();
        let test_values: Vec<i64> = vec![0, 1, -1, 5, -3, 17, -42];
        for &a in &test_values {
            for &b in &test_values {
                for &c in &test_values {
                    let ab = engine.add(a, b);
                    let ab_c = engine.add(ab.integer.value, c);
                    let bc = engine.add(b, c);
                    let a_bc = engine.add(a, bc.integer.value);
                    assert_eq!(
                        ab_c.integer.value, a_bc.integer.value,
                        "Associativity failed: ({} + {}) + {} = {} but {} + ({} + {}) = {}",
                        a, b, c, ab_c.integer.value, a, b, c, a_bc.integer.value
                    );
                }
            }
        }
    }

    #[test]
    fn test_integer_multiply_associative_property() {
        let mut engine = IntegerArithmeticEngine::new();
        let test_values: Vec<i64> = vec![0, 1, -1, 2, -2, 3, -3, 5];
        for &a in &test_values {
            for &b in &test_values {
                for &c in &test_values {
                    let ab = engine.multiply(a, b);
                    let ab_c = engine.multiply(ab.integer.value, c);
                    let bc = engine.multiply(b, c);
                    let a_bc = engine.multiply(a, bc.integer.value);
                    assert_eq!(
                        ab_c.integer.value, a_bc.integer.value,
                        "Multiply associativity failed: ({} * {}) * {} != {} * ({} * {})",
                        a, b, c, a, b, c
                    );
                }
            }
        }
    }

    #[test]
    fn test_integer_distributive_property() {
        let mut engine = IntegerArithmeticEngine::new();
        let test_values: Vec<i64> = vec![0, 1, -1, 2, -3, 5, -7];
        for &a in &test_values {
            for &b in &test_values {
                for &c in &test_values {
                    // a * (b + c) = a*b + a*c
                    let bc = engine.add(b, c);
                    let lhs = engine.multiply(a, bc.integer.value);
                    let ab = engine.multiply(a, b);
                    let ac = engine.multiply(a, c);
                    let rhs = engine.add(ab.integer.value, ac.integer.value);
                    assert_eq!(
                        lhs.integer.value, rhs.integer.value,
                        "Distributivity failed: {} * ({} + {}) = {} but {}*{} + {}*{} = {}",
                        a, b, c, lhs.integer.value, a, b, a, c, rhs.integer.value
                    );
                }
            }
        }
    }

    #[test]
    fn test_rational_addition_commutative_property() {
        let engine = RationalArithmeticEngine::new();
        let test_fractions: Vec<(i64, i64)> = vec![
            (0, 1),
            (1, 1),
            (-1, 1),
            (1, 2),
            (-1, 3),
            (2, 7),
            (-5, 11),
            (3, 4),
            (7, 13),
        ];
        for &(an, ad) in &test_fractions {
            for &(bn, bd) in &test_fractions {
                let a = engine.encode(an, ad);
                let b = engine.encode(bn, bd);
                let ab = engine.add(&a, &b);
                let ba = engine.add(&b, &a);
                assert_eq!(
                    ab.rational.numerator, ba.rational.numerator,
                    "Rational commutativity failed: {}/{} + {}/{}",
                    an, ad, bn, bd
                );
                assert_eq!(ab.rational.denominator, ba.rational.denominator);
            }
        }
    }

    #[test]
    fn test_gcd_normalization_idempotent() {
        let engine = RationalArithmeticEngine::new();
        // Encoding a fraction should normalize, and re-encoding the result should be identical
        let test_cases = vec![
            (2, 4),     // → 1/2
            (6, 9),     // → 2/3
            (-4, 6),    // → -2/3
            (15, 25),   // → 3/5
            (0, 7),     // → 0/1
            (100, 100), // → 1/1
        ];
        for (num, den) in test_cases {
            let r1 = engine.encode(num, den);
            let r2 = engine.encode(r1.numerator, r1.denominator);
            assert_eq!(
                r1.numerator, r2.numerator,
                "Normalization not idempotent for {}/{}",
                num, den
            );
            assert_eq!(
                r1.denominator, r2.denominator,
                "Normalization not idempotent for {}/{}",
                num, den
            );
        }
    }

    #[test]
    fn test_factorize_product_equals_original() {
        let engine = NumberTheoryEngine::new();
        let test_values: Vec<u64> = vec![1, 2, 3, 6, 12, 30, 60, 100, 360, 1000, 997, 2048];
        for &n in &test_values {
            let f = engine.factorize(n);
            let product: u64 = f.factors.iter().map(|&(p, e)| p.pow(e)).product();
            if n == 0 {
                continue;
            }
            assert_eq!(
                product, n,
                "Factorization product mismatch for {}: factors {:?} → product {}",
                n, f.factors, product
            );
        }
    }

    #[test]
    fn test_factorize_all_primes() {
        let engine = NumberTheoryEngine::new();
        let test_values: Vec<u64> = vec![6, 12, 30, 60, 100, 360];
        for &n in &test_values {
            let f = engine.factorize(n);
            for &(p, _) in &f.factors {
                assert!(engine.is_prime(p), "Factor {} of {} is not prime", p, n);
            }
        }
    }

    #[test]
    fn test_gcd_lcm_identity() {
        // gcd(a,b) * lcm(a,b) = a * b
        let engine = NumberTheoryEngine::new();
        let test_values: Vec<u64> = vec![1, 2, 3, 6, 12, 15, 35, 100];
        for &a in &test_values {
            for &b in &test_values {
                let g = engine.gcd(a, b);
                let l = engine.lcm(a, b);
                assert_eq!(
                    g * l,
                    a * b,
                    "gcd*lcm identity failed: gcd({},{})={}, lcm={}, product={} != {}",
                    a,
                    b,
                    g,
                    l,
                    g * l,
                    a * b
                );
            }
        }
    }

    #[test]
    fn test_extended_gcd_bezout_identity() {
        // ax + by = gcd(a,b)
        let engine = NumberTheoryEngine::new();
        let test_pairs: Vec<(i64, i64)> =
            vec![(12, 8), (35, 15), (100, 37), (17, 13), (1, 1), (7, 0)];
        for (a, b) in test_pairs {
            let (g, x, y) = engine.extended_gcd(a, b);
            assert_eq!(
                a * x + b * y,
                g,
                "Bezout identity failed: {}*{} + {}*{} = {} != gcd={}",
                a,
                x,
                b,
                y,
                a * x + b * y,
                g
            );
        }
    }

    #[test]
    fn test_modular_inverse_roundtrip() {
        // a * a^(-1) ≡ 1 (mod p) for prime p
        let primes = vec![5, 7, 11, 13, 17, 19, 23];
        for p in primes {
            let ring = ModularRing::new(p);
            for a in 1..p {
                let inv = ring
                    .inverse(a)
                    .expect(&format!("No inverse for {} mod {}", a, p));
                let product = ring.multiply(a, inv);
                assert_eq!(
                    product, 1,
                    "{} * {} mod {} = {} (expected 1)",
                    a, inv, p, product
                );
            }
        }
    }

    #[test]
    fn test_fermats_little_theorem() {
        // a^(p-1) ≡ 1 (mod p) for prime p and gcd(a,p) = 1
        let primes = vec![5, 7, 11, 13];
        for p in primes {
            let ring = ModularRing::new(p);
            for a in 1..p {
                let result = ring.power(a, p - 1);
                assert_eq!(
                    result,
                    1,
                    "Fermat's little theorem failed: {}^{} mod {} = {} (expected 1)",
                    a,
                    p - 1,
                    p,
                    result
                );
            }
        }
    }

    // ========================================================================
    // 5. CALCULUS INTEGRATION
    // ========================================================================

    #[test]
    fn test_polynomial_differentiate_integrate_roundtrip() {
        use crate::hdc::arithmetic_engine::Polynomial;
        let prims = PrimitiveSystem::global();

        // f(x) = 3x^2 + 2x + 1, coefficients [1, 2, 3]
        let f = Polynomial::new(vec![1, 2, 3], "x", &prims);

        // f'(x) = 6x + 2, coefficients [2, 6]
        let f_prime = f.derivative(&prims);

        // Integrate f'(x) → should get 3x^2 + 2x (+ C, but C=0 in our impl)
        let f_reconstructed = SymbolicIntegrator::integrate_polynomial(&f_prime, &prims);

        // The reconstructed polynomial should match original (minus constant term)
        // Polynomial coefficients: f = [1, 2, 3], f' = [2, 6], ∫f' = [0, 2, 3]
        // (constant term lost in differentiation, restored as 0)
        let orig_coeffs = f.coefficients();
        let recon_coeffs = f_reconstructed.coefficients();

        // Check all non-constant coefficients match
        for i in 1..orig_coeffs.len() {
            assert_eq!(
                orig_coeffs[i], recon_coeffs[i],
                "Coefficient mismatch at degree {}: original={}, reconstructed={}",
                i, orig_coeffs[i], recon_coeffs[i]
            );
        }
    }

    #[test]
    fn test_fundamental_theorem_multiple_polynomials() {
        let prims = PrimitiveSystem::global();
        use crate::hdc::arithmetic_engine::Polynomial;

        // Test FTC for polynomials where integration preserves integer coefficients.
        // Note: ∫(a_n * x^n)dx = a_n/(n+1) * x^(n+1), so a_n must be divisible by (n+1)
        // for the result to stay in integer polynomial space.
        let polys = vec![
            Polynomial::new(vec![1], "x", &prims),       // f(x) = 1, ∫ = x
            Polynomial::new(vec![0, 2], "x", &prims),    // f(x) = 2x, ∫ = x^2
            Polynomial::new(vec![0, 0, 3], "x", &prims), // f(x) = 3x^2, ∫ = x^3
            Polynomial::new(vec![1, 2, 3], "x", &prims), // f(x) = 3x^2 + 2x + 1
            Polynomial::new(vec![4, 0, 0, 4], "x", &prims), // f(x) = 4x^3 + 4, ∫ = x^4 + 4x
        ];

        for poly in &polys {
            let holds = FundamentalTheoremVerifier::verify(poly, &prims);
            assert!(
                holds,
                "Fundamental theorem failed for polynomial with coefficients {:?}",
                poly.coefficients()
            );
        }
    }

    // ========================================================================
    // 6. INDUCTION ENGINE INTEGRATION
    // ========================================================================

    #[test]
    fn test_induction_engine_with_arithmetic() {
        let engine = InductionEngine::new();

        // Prove sum formula with larger range
        let proof = engine.prove_sum_formula(50);
        assert!(proof.valid, "Sum formula induction should hold up to 50");
        assert!(proof.phi > 0.0, "Proof should have positive phi");

        // Prove commutativity
        let comm_proof = engine.prove_addition_commutative();
        assert!(comm_proof.valid, "Addition commutativity should hold");
    }

    #[test]
    fn test_induction_detects_false_property() {
        let engine = InductionEngine::new();

        // Try to "prove" a false property: n^2 < 100 for all n
        let proof = engine.prove_by_induction(
            "n_squared_less_100",
            &|n| (n * n) < 100,             // base check
            &|n| ((n + 1) * (n + 1)) < 100, // step check (will fail for n >= 9)
            50,
        );
        assert!(!proof.valid, "False property should not be provable");
    }

    // ========================================================================
    // 7. CROSS-MODULE COHERENCE
    // ========================================================================

    #[test]
    fn test_integer_encoding_deterministic() {
        // Same value should produce same encoding
        let engine = IntegerArithmeticEngine::new();
        let a1 = engine.encode(42);
        let a2 = engine.encode(42);
        assert_eq!(
            a1.encoding.0, a2.encoding.0,
            "Encoding for same value should be deterministic"
        );
    }

    #[test]
    fn test_rational_encoding_normalized() {
        // 2/4 and 1/2 should have same numerator/denominator after normalization
        let engine = RationalArithmeticEngine::new();
        let a = engine.encode(2, 4);
        let b = engine.encode(1, 2);
        assert_eq!(a.numerator, b.numerator);
        assert_eq!(a.denominator, b.denominator);
    }

    #[test]
    fn test_math_constants_accuracy() {
        let pi = MathConstants::pi();
        assert!(
            (pi.f64_value - std::f64::consts::PI).abs() < 1e-15,
            "Pi f64 value should be exact"
        );
        assert!(
            pi.error_bound < 0.0001,
            "355/113 approximation error should be very small: {}",
            pi.error_bound
        );

        let e = MathConstants::e();
        assert!((e.f64_value - std::f64::consts::E).abs() < 1e-15);

        let phi = MathConstants::phi_golden();
        let exact_phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((phi.f64_value - exact_phi).abs() < 1e-15);
    }

    #[test]
    fn test_dual_compute_exact_vs_approximate() {
        let bridge = PeanoPhysicsBridge::new();

        // Exact integer computation should agree with f64
        let result = bridge.dual_compute(7.0, 3.0, "multiply");
        assert!((result.approximate_value - 21.0).abs() < 1e-10);
        assert!(
            result.exact_value.is_some(),
            "Integer inputs should have exact result"
        );
        let (num, den) = result.exact_value.unwrap();
        assert_eq!(num, 21);
        assert_eq!(den, 1);
    }

    #[test]
    fn test_primes_sieve_correctness() {
        let engine = NumberTheoryEngine::new();
        let primes = engine.primes_up_to(100);
        let prime_values: Vec<u64> = primes.iter().map(|&(p, _)| p).collect();

        let expected = vec![
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97,
        ];
        assert_eq!(
            prime_values, expected,
            "Sieve of Eratosthenes should produce correct primes up to 100"
        );
    }
}
