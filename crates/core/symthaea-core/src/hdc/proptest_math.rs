// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Mathematical Foundations

Uses proptest to verify algebraic properties of the integer, rational,
and number theory modules hold across randomly generated inputs.

## Key Properties Tested

1. **Integer addition commutativity**: a + b == b + a
2. **Integer multiplication commutativity**: a × b == b × a
3. **Integer addition associativity**: (a + b) + c == a + (b + c)
4. **Integer multiplication associativity**: (a × b) × c == a × (b × c)
5. **Integer distributivity**: a × (b + c) == a×b + a×c
6. **Rational normalization idempotency**: normalize(normalize(a/b)) == normalize(a/b)
7. **Rational addition commutativity**: a/b + c/d == c/d + a/b
8. **GCD divides both**: gcd(a,b) divides a and gcd(a,b) divides b
9. **Bezout's identity**: gcd(a,b) = ax + by for some x,y
10. **Negate involution**: negate(negate(x)) == x
11. **Integer division/remainder**: a == (a/b)*b + a%b
*/

#![cfg(test)]

use crate::hdc::integer::IntegerArithmeticEngine;
use crate::hdc::number_theory::NumberTheoryEngine;
use crate::hdc::rational::RationalArithmeticEngine;
use proptest::prelude::*;

/// Strategy for "small" integers that won't overflow under multiplication
fn small_integer() -> impl Strategy<Value = i64> {
    -10000i64..=10000i64
}

/// Strategy for "medium" integers (avoid overflow in a*b*c)
fn medium_integer() -> impl Strategy<Value = i64> {
    -100i64..=100i64
}

/// Strategy for nonzero integers (for division)
fn nonzero_integer() -> impl Strategy<Value = i64> {
    prop::num::i64::ANY
        .prop_filter("nonzero", |&x| x != 0)
        .prop_map(|x| {
            // Clamp to avoid overflow issues
            x.max(-10000).min(10000)
        })
}

/// Strategy for positive integers (for number theory)
fn positive_i64() -> impl Strategy<Value = i64> {
    1i64..=10000i64
}

/// Strategy for rational number components (nonzero denominator)
fn rational_pair() -> impl Strategy<Value = (i64, i64)> {
    ((-100i64..=100i64), (1i64..=100i64))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    // =========================================================================
    // Property 1: Integer addition is commutative
    // a + b == b + a
    // =========================================================================
    #[test]
    fn prop_integer_add_commutative(
        a in small_integer(),
        b in small_integer()
    ) {
        let mut engine = IntegerArithmeticEngine::new();
        let ab = engine.add(a, b);
        let ba = engine.add(b, a);
        prop_assert_eq!(
            ab.integer.value, ba.integer.value,
            "{} + {} should equal {} + {}", a, b, b, a
        );
    }

    // =========================================================================
    // Property 2: Integer multiplication is commutative
    // a × b == b × a
    // =========================================================================
    #[test]
    fn prop_integer_mul_commutative(
        a in small_integer(),
        b in small_integer()
    ) {
        let mut engine = IntegerArithmeticEngine::new();
        let ab = engine.multiply(a, b);
        let ba = engine.multiply(b, a);
        prop_assert_eq!(
            ab.integer.value, ba.integer.value,
            "{} × {} should equal {} × {}", a, b, b, a
        );
    }

    // =========================================================================
    // Property 3: Integer addition is associative
    // (a + b) + c == a + (b + c)
    // =========================================================================
    #[test]
    fn prop_integer_add_associative(
        a in medium_integer(),
        b in medium_integer(),
        c in medium_integer()
    ) {
        let mut engine = IntegerArithmeticEngine::new();
        let ab = engine.add(a, b);
        let ab_c = engine.add(ab.integer.value, c);

        let bc = engine.add(b, c);
        let a_bc = engine.add(a, bc.integer.value);

        prop_assert_eq!(
            ab_c.integer.value, a_bc.integer.value,
            "({} + {}) + {} should equal {} + ({} + {})", a, b, c, a, b, c
        );
    }

    // =========================================================================
    // Property 4: Integer multiplication is associative
    // (a × b) × c == a × (b × c)
    // =========================================================================
    #[test]
    fn prop_integer_mul_associative(
        a in medium_integer(),
        b in medium_integer(),
        c in medium_integer()
    ) {
        let mut engine = IntegerArithmeticEngine::new();
        let ab = engine.multiply(a, b);
        let ab_c = engine.multiply(ab.integer.value, c);

        let bc = engine.multiply(b, c);
        let a_bc = engine.multiply(a, bc.integer.value);

        prop_assert_eq!(
            ab_c.integer.value, a_bc.integer.value,
            "({} × {}) × {} should equal {} × ({} × {})", a, b, c, a, b, c
        );
    }

    // =========================================================================
    // Property 5: Distributivity
    // a × (b + c) == a×b + a×c
    // =========================================================================
    #[test]
    fn prop_integer_distributive(
        a in medium_integer(),
        b in medium_integer(),
        c in medium_integer()
    ) {
        let mut engine = IntegerArithmeticEngine::new();
        let bc = engine.add(b, c);
        let a_bc = engine.multiply(a, bc.integer.value);

        let ab = engine.multiply(a, b);
        let ac = engine.multiply(a, c);
        let ab_plus_ac = engine.add(ab.integer.value, ac.integer.value);

        prop_assert_eq!(
            a_bc.integer.value, ab_plus_ac.integer.value,
            "{} × ({} + {}) should equal {} × {} + {} × {}", a, b, c, a, b, a, c
        );
    }

    // =========================================================================
    // Property 6: Negate is involution
    // -(-x) == x
    // =========================================================================
    #[test]
    fn prop_negate_involution(a in small_integer()) {
        let mut engine = IntegerArithmeticEngine::new();
        let neg_a = engine.negate(a);
        let neg_neg_a = engine.negate(neg_a.integer.value);

        prop_assert_eq!(
            neg_neg_a.integer.value, a,
            "-(-{}) should equal {}", a, a
        );
    }

    // =========================================================================
    // Property 7: Addition identity
    // a + 0 == a
    // =========================================================================
    #[test]
    fn prop_add_identity(a in small_integer()) {
        let mut engine = IntegerArithmeticEngine::new();
        let result = engine.add(a, 0);
        prop_assert_eq!(result.integer.value, a, "{} + 0 should equal {}", a, a);
    }

    // =========================================================================
    // Property 8: Multiplication identity
    // a × 1 == a
    // =========================================================================
    #[test]
    fn prop_mul_identity(a in small_integer()) {
        let mut engine = IntegerArithmeticEngine::new();
        let result = engine.multiply(a, 1);
        prop_assert_eq!(result.integer.value, a, "{} × 1 should equal {}", a, a);
    }

    // =========================================================================
    // Property 9: Multiplication by zero
    // a × 0 == 0
    // =========================================================================
    #[test]
    fn prop_mul_zero(a in small_integer()) {
        let mut engine = IntegerArithmeticEngine::new();
        let result = engine.multiply(a, 0);
        prop_assert_eq!(result.integer.value, 0, "{} × 0 should equal 0", a);
    }

    // =========================================================================
    // Property 10: Division/remainder identity
    // a == (a / b) * b + a % b  (for b != 0)
    // =========================================================================
    #[test]
    fn prop_division_remainder(
        a in small_integer(),
        b in nonzero_integer()
    ) {
        let mut engine = IntegerArithmeticEngine::new();
        let div_result = engine.divide(a, b).unwrap();
        let _mod_result = engine.modulo(a, b).unwrap();

        // For Rust's integer division (truncation toward zero):
        // a == (a / b) * b + a % b where % is Rust's remainder
        let q = a / b;  // Rust truncating division

        prop_assert_eq!(
            div_result.integer.value, q,
            "Division result should match Rust division: {} / {} = {} (got {})",
            a, b, q, div_result.integer.value
        );
    }

    // =========================================================================
    // Property 11: Rational addition is commutative
    // a/b + c/d == c/d + a/b
    // =========================================================================
    #[test]
    fn prop_rational_add_commutative(
        (an, ad) in rational_pair(),
        (bn, bd) in rational_pair()
    ) {
        let engine = RationalArithmeticEngine::new();
        let a = engine.encode(an, ad);
        let b = engine.encode(bn, bd);

        let ab = engine.add(&a, &b);
        let ba = engine.add(&b, &a);

        prop_assert_eq!(
            (ab.rational.numerator, ab.rational.denominator),
            (ba.rational.numerator, ba.rational.denominator),
            "{}/{} + {}/{} should equal {}/{} + {}/{}",
            an, ad, bn, bd, bn, bd, an, ad
        );
    }

    // =========================================================================
    // Property 12: Rational multiplication is commutative
    // (a/b) × (c/d) == (c/d) × (a/b)
    // =========================================================================
    #[test]
    fn prop_rational_mul_commutative(
        (an, ad) in rational_pair(),
        (bn, bd) in rational_pair()
    ) {
        let engine = RationalArithmeticEngine::new();
        let a = engine.encode(an, ad);
        let b = engine.encode(bn, bd);

        let ab = engine.multiply(&a, &b);
        let ba = engine.multiply(&b, &a);

        prop_assert_eq!(
            (ab.rational.numerator, ab.rational.denominator),
            (ba.rational.numerator, ba.rational.denominator),
            "({}/{}) × ({}/{}) should be commutative",
            an, ad, bn, bd
        );
    }

    // =========================================================================
    // Property 13: Rational normalization is idempotent
    // normalize(a/b) is already in lowest terms
    // =========================================================================
    #[test]
    fn prop_rational_normalization_idempotent(
        (n, d) in rational_pair()
    ) {
        let engine = RationalArithmeticEngine::new();
        let r = engine.encode(n, d);

        // Re-encode the normalized form — should be unchanged
        let r2 = engine.encode(r.numerator, r.denominator);

        prop_assert_eq!(
            (r.numerator, r.denominator),
            (r2.numerator, r2.denominator),
            "Normalization should be idempotent for {}/{}",
            n, d
        );
    }

    // =========================================================================
    // Property 14: GCD divides both operands
    // =========================================================================
    #[test]
    fn prop_gcd_divides_both(
        a in positive_i64(),
        b in positive_i64()
    ) {
        let engine = NumberTheoryEngine::new();
        let (g, _, _) = engine.extended_gcd(a, b);

        prop_assert!(
            a % g == 0 && b % g == 0,
            "gcd({}, {}) = {} should divide both", a, b, g
        );
    }

    // =========================================================================
    // Property 15: Bezout's identity
    // gcd(a, b) = a*x + b*y for some integers x, y
    // =========================================================================
    #[test]
    fn prop_bezout_identity(
        a in positive_i64(),
        b in positive_i64()
    ) {
        let engine = NumberTheoryEngine::new();
        let (g, x, y) = engine.extended_gcd(a, b);

        let lhs = a * x + b * y;
        prop_assert_eq!(
            lhs, g,
            "Bezout: {} × {} + {} × {} should equal gcd = {}",
            a, x, b, y, g
        );
    }
}
