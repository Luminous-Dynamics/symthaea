// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests for motor safety invariants.
//!
//! Verifies that the consciousness-gated safety mechanism can never
//! produce unbounded motor output, regardless of input Phi values.

use proptest::prelude::*;
use symthaea_core::embodiment::MotorSafetyLevel;

// ============================================================================
// Property 1: motor_gain is always in [0.0, 1.0] for any f64 phi
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    #[test]
    fn motor_gain_always_bounded(phi in prop::num::f64::ANY) {
        let level = MotorSafetyLevel::from_phi(phi);
        let gain = level.motor_gain();
        prop_assert!(
            gain >= 0.0 && gain <= 1.0,
            "motor_gain {gain} out of [0.0, 1.0] for phi={phi}"
        );
    }
}

// ============================================================================
// Property 2: motor_gain is always bounded for the typical phi range
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    #[test]
    fn motor_gain_bounded_in_typical_range(phi in -1e6f64..1e6) {
        let level = MotorSafetyLevel::from_phi(phi);
        let gain = level.motor_gain();
        prop_assert!(
            gain >= 0.0 && gain <= 1.0,
            "motor_gain {gain} out of [0.0, 1.0] for phi={phi}"
        );
    }
}

// ============================================================================
// Property 3: from_phi monotonicity — higher phi never produces stricter level
// The MotorSafetyLevel ordering is Green < Yellow < Orange < Red,
// so higher Phi should produce a LOWER (less restrictive) level.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    #[test]
    fn from_phi_monotonic(a in 0.0f64..1.0, b in 0.0f64..1.0) {
        let level_a = MotorSafetyLevel::from_phi(a);
        let level_b = MotorSafetyLevel::from_phi(b);
        let gain_a = level_a.motor_gain();
        let gain_b = level_b.motor_gain();

        if a > b {
            // Higher Phi → same or higher gain (less restrictive)
            prop_assert!(
                gain_a >= gain_b,
                "Monotonicity violated: phi={a} → gain={gain_a}, phi={b} → gain={gain_b}"
            );
        } else if b > a {
            prop_assert!(
                gain_b >= gain_a,
                "Monotonicity violated: phi={b} → gain={gain_b}, phi={a} → gain={gain_a}"
            );
        }
    }
}

// ============================================================================
// Property 4: Non-finite phi always maps to Red (emergency stop)
// ============================================================================

#[test]
fn non_finite_phi_always_red() {
    let cases = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -f64::NAN];
    for phi in cases {
        let level = MotorSafetyLevel::from_phi(phi);
        assert_eq!(
            level,
            MotorSafetyLevel::Red,
            "Non-finite phi={phi} should map to Red, got {level:?}"
        );
        assert_eq!(level.motor_gain(), 0.0);
    }
}

// ============================================================================
// Property 5: Safety level threshold boundaries are exact
// ============================================================================

#[test]
fn threshold_boundaries_exact() {
    // At threshold values (using > not >=):
    // phi=0.6 → Yellow (not Green, because > 0.6 needed for Green)
    assert_eq!(MotorSafetyLevel::from_phi(0.6), MotorSafetyLevel::Yellow);
    assert_eq!(
        MotorSafetyLevel::from_phi(0.6 + f64::EPSILON),
        MotorSafetyLevel::Yellow
    );
    assert_eq!(MotorSafetyLevel::from_phi(0.60001), MotorSafetyLevel::Green);

    assert_eq!(MotorSafetyLevel::from_phi(0.3), MotorSafetyLevel::Orange);
    assert_eq!(
        MotorSafetyLevel::from_phi(0.30001),
        MotorSafetyLevel::Yellow
    );

    assert_eq!(MotorSafetyLevel::from_phi(0.1), MotorSafetyLevel::Red);
    assert_eq!(
        MotorSafetyLevel::from_phi(0.10001),
        MotorSafetyLevel::Orange
    );
}

// ============================================================================
// Property 6: Gain values match documented spec
// ============================================================================

#[test]
fn gain_values_match_spec() {
    assert_eq!(MotorSafetyLevel::Green.motor_gain(), 1.0);
    assert_eq!(MotorSafetyLevel::Yellow.motor_gain(), 0.6);
    assert_eq!(MotorSafetyLevel::Orange.motor_gain(), 0.3);
    assert_eq!(MotorSafetyLevel::Red.motor_gain(), 0.0);

    // Verify ordering: Green > Yellow > Orange > Red (gain decreasing)
    assert!(MotorSafetyLevel::Green.motor_gain() > MotorSafetyLevel::Yellow.motor_gain());
    assert!(MotorSafetyLevel::Yellow.motor_gain() > MotorSafetyLevel::Orange.motor_gain());
    assert!(MotorSafetyLevel::Orange.motor_gain() > MotorSafetyLevel::Red.motor_gain());
}
