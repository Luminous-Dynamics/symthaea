// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Validation Utilities
//!
//! Pure functions for validating data integrity. Used both in tests AND
//! at conductor sync time to verify Trial Mode data hasn't been tampered with.
//!
//! ## BKT Integrity Validation
//!
//! When a Trial user syncs their localStorage ProgressStore to the conductor,
//! the conductor MUST verify that BKT state transitions are mathematically
//! consistent. This prevents localStorage tampering (DevTools attack vector).
//!
//! ## TEND Formula Validation
//!
//! Property checks ensuring TEND credits are computed correctly:
//! never exceed cap, never negative, reputation scaling monotonic.

use serde::{Deserialize, Serialize};

// ============== BKT Integrity Validation ==============

/// BKT parameters (must match curriculum.rs exactly)
const P_TRANSIT: f32 = 0.1;
const P_SLIP: f32 = 0.1;
const P_GUESS: f32 = 0.25;
const P_INITIAL: f32 = 0.1; // P(L₀) prior

/// Result of BKT integrity validation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BktValidationResult {
    /// Whether the state is consistent with the claimed attempts
    pub valid: bool,
    /// Expected p_mastery from replaying attempts
    pub expected_p_mastery: f32,
    /// Claimed p_mastery from the imported state
    pub claimed_p_mastery: f32,
    /// Absolute difference
    pub difference: f32,
    /// Number of attempts replayed
    pub attempts_replayed: u32,
    /// Reason for rejection (if invalid)
    pub rejection_reason: Option<String>,
}

/// Validate that a BKT state is consistent with its claimed attempts.
///
/// Replays the BKT update formula from P(L₀) = 0.1 through the given
/// sequence of correct/incorrect responses. If the final p_mastery doesn't
/// match the claimed value within tolerance, the state was tampered with.
///
/// ## Security
///
/// This is the primary defense against localStorage tampering in Trial Mode.
/// The conductor MUST call this before accepting imported progress data.
pub fn validate_bkt_integrity(
    claimed_p_mastery: f32,
    attempts: u32,
    correct: u32,
    tolerance: f32,
) -> BktValidationResult {
    // Basic sanity checks
    if correct > attempts {
        return BktValidationResult {
            valid: false,
            expected_p_mastery: 0.0,
            claimed_p_mastery,
            difference: 0.0,
            attempts_replayed: 0,
            rejection_reason: Some(format!(
                "Correct ({}) exceeds attempts ({})",
                correct, attempts
            )),
        };
    }

    if claimed_p_mastery < 0.0 || claimed_p_mastery > 1.0 {
        return BktValidationResult {
            valid: false,
            expected_p_mastery: 0.0,
            claimed_p_mastery,
            difference: 0.0,
            attempts_replayed: 0,
            rejection_reason: Some("p_mastery out of [0,1] range".to_string()),
        };
    }

    // Replay BKT from prior
    // We don't know the exact order of correct/incorrect, so we compute
    // bounds: best case (all correct first) and worst case (all incorrect first).
    // The claimed value must fall within these bounds ± tolerance.
    let case_a = replay_bkt(attempts, correct, true);
    let case_b = replay_bkt(attempts, correct, false);

    // BKT ordering matters — take the actual min/max of both cases
    let min_case = case_a.min(case_b);
    let max_case = case_a.max(case_b);

    let lower_bound = min_case - tolerance;
    let upper_bound = max_case + tolerance;

    let valid = claimed_p_mastery >= lower_bound && claimed_p_mastery <= upper_bound;

    // Also check: with 0 attempts, p_mastery should be near P_INITIAL
    if attempts == 0 && (claimed_p_mastery - P_INITIAL).abs() > tolerance {
        return BktValidationResult {
            valid: false,
            expected_p_mastery: P_INITIAL,
            claimed_p_mastery,
            difference: (claimed_p_mastery - P_INITIAL).abs(),
            attempts_replayed: 0,
            rejection_reason: Some(format!(
                "Zero attempts but p_mastery={:.3} (expected ~{:.1})",
                claimed_p_mastery, P_INITIAL
            )),
        };
    }

    let midpoint = (min_case + max_case) / 2.0;

    BktValidationResult {
        valid,
        expected_p_mastery: midpoint,
        claimed_p_mastery,
        difference: (claimed_p_mastery - midpoint).abs(),
        attempts_replayed: attempts,
        rejection_reason: if valid {
            None
        } else {
            Some(format!(
                "p_mastery={:.3} outside valid range [{:.3}, {:.3}] for {} attempts ({} correct)",
                claimed_p_mastery, lower_bound, upper_bound, attempts, correct
            ))
        },
    }
}

/// Replay BKT updates to compute expected p_mastery.
///
/// If `correct_first` is true, processes all correct responses before incorrect
/// (best case for p_mastery). If false, incorrect first (worst case).
fn replay_bkt(attempts: u32, correct: u32, correct_first: bool) -> f32 {
    let mut p = P_INITIAL;
    let incorrect = attempts - correct;

    let (first_count, first_correct, second_count, second_correct) = if correct_first {
        (correct, true, incorrect, false)
    } else {
        (incorrect, false, correct, true)
    };

    for _ in 0..first_count {
        p = bkt_update_step(p, first_correct);
    }
    for _ in 0..second_count {
        p = bkt_update_step(p, second_correct);
    }

    p.clamp(0.01, 0.99)
}

/// Single BKT update step (must match curriculum.rs exactly).
fn bkt_update_step(p_l: f32, correct: bool) -> f32 {
    let posterior = if correct {
        let p_correct = p_l * (1.0 - P_SLIP) + (1.0 - p_l) * P_GUESS;
        p_l * (1.0 - P_SLIP) / p_correct
    } else {
        let p_incorrect = p_l * P_SLIP + (1.0 - p_l) * (1.0 - P_GUESS);
        p_l * P_SLIP / p_incorrect
    };
    // Apply transition
    let p_new = posterior + (1.0 - posterior) * P_TRANSIT;
    p_new.clamp(0.01, 0.99)
}

// ============== TEND Formula Validation ==============

/// TEND computation constants (must match praxis-bridge coordinator)
const MAX_TEND_PER_EVENT: f32 = 2.0;
const BASE_TEND_RATE: f32 = 0.25;
const MIN_QUALITY_FOR_TEND: u16 = 600;
const MIN_DURATION_FOR_TEND: u32 = 900;

/// Validate TEND computation properties.
///
/// Pure function — verifies that a computed TEND credit amount satisfies
/// all invariants: non-negative, capped, quality gated, duration gated.
pub fn validate_tend_properties(
    tend_credits: f32,
    quality_permille: u16,
    duration_seconds: u32,
) -> Result<(), String> {
    // Non-negative
    if tend_credits < 0.0 {
        return Err(format!("TEND credits cannot be negative: {}", tend_credits));
    }

    // Capped
    if tend_credits > MAX_TEND_PER_EVENT {
        return Err(format!(
            "TEND credits {} exceed cap {}",
            tend_credits, MAX_TEND_PER_EVENT
        ));
    }

    // Quality gate
    if quality_permille < MIN_QUALITY_FOR_TEND && tend_credits > 0.0 {
        return Err(format!(
            "Quality {} below minimum {} but TEND > 0",
            quality_permille, MIN_QUALITY_FOR_TEND
        ));
    }

    // Duration gate
    if duration_seconds < MIN_DURATION_FOR_TEND && tend_credits > 0.0 {
        return Err(format!(
            "Duration {}s below minimum {}s but TEND > 0",
            duration_seconds, MIN_DURATION_FOR_TEND
        ));
    }

    Ok(())
}

/// Compute TEND credits using the same formula as the coordinator.
/// Pure function for testing — no Holochain dependencies.
pub fn compute_tend_pure(
    quality_permille: u16,
    duration_seconds: u32,
    reputation_permille: u16,
    phi_permille: Option<u16>,
) -> f32 {
    if quality_permille < MIN_QUALITY_FOR_TEND || duration_seconds < MIN_DURATION_FOR_TEND {
        return 0.0;
    }

    let hours = duration_seconds as f32 / 3600.0;
    let quality_factor = quality_permille as f32 / 1000.0;
    let base_credit = hours * BASE_TEND_RATE;
    let quality_bonus = hours * quality_factor * 0.5;
    let reputation = reputation_permille as f32 / 1000.0;
    let reputation_multiplier = 0.5 + reputation * 0.5;
    let phi_bonus = phi_permille
        .map(|p| (p as f32 / 1000.0) * 0.1 * hours)
        .unwrap_or(0.0);

    ((base_credit + quality_bonus + phi_bonus) * reputation_multiplier).min(MAX_TEND_PER_EVENT)
}

// ============== Application State Machine Validation ==============

/// All valid ApplicationStatus transitions.
/// Returns true if (from, to) is a legal transition.
pub fn is_valid_application_transition(from: &str, to: &str) -> bool {
    matches!(
        (from, to),
        ("Draft", "Submitted")
            | ("Draft", "Withdrawn")
            | ("Submitted", "UnderReview")
            | ("Submitted", "Withdrawn")
            | ("UnderReview", "Interview")
            | ("UnderReview", "Rejected")
            | ("UnderReview", "Withdrawn")
            | ("Interview", "Offered")
            | ("Interview", "Rejected")
            | ("Interview", "Withdrawn")
            | ("Offered", "Accepted")
            | ("Offered", "Rejected")
            | ("Offered", "Withdrawn")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- BKT Integrity Tests ----

    #[test]
    fn test_bkt_zero_attempts_valid() {
        let result = validate_bkt_integrity(0.1, 0, 0, 0.05);
        assert!(result.valid, "Zero attempts should be valid at P_INITIAL");
    }

    #[test]
    fn test_bkt_zero_attempts_tampered() {
        let result = validate_bkt_integrity(0.95, 0, 0, 0.05);
        assert!(!result.valid, "Zero attempts with high mastery = tampered");
        assert!(result.rejection_reason.is_some());
    }

    #[test]
    fn test_bkt_correct_exceeds_attempts() {
        let result = validate_bkt_integrity(0.5, 5, 10, 0.1);
        assert!(!result.valid, "Correct > attempts is impossible");
    }

    #[test]
    fn test_bkt_all_correct_high_mastery_valid() {
        // 20 correct answers should push mastery very high
        let expected = replay_bkt(20, 20, true);
        let result = validate_bkt_integrity(expected, 20, 20, 0.05);
        assert!(result.valid, "All correct with matching mastery should be valid");
    }

    #[test]
    fn test_bkt_all_incorrect_low_mastery_valid() {
        let expected = replay_bkt(10, 0, false);
        let result = validate_bkt_integrity(expected, 10, 0, 0.05);
        assert!(result.valid, "All incorrect with matching mastery should be valid");
    }

    #[test]
    fn test_bkt_impossible_jump_detected() {
        // 3 attempts, 2 correct — can't reach 0.95
        let result = validate_bkt_integrity(0.95, 3, 2, 0.05);
        assert!(!result.valid, "0.95 mastery in 3 attempts is impossible");
    }

    #[test]
    fn test_bkt_mixed_responses_within_bounds() {
        // 10 attempts, 7 correct — mastery should be moderate-high
        let best = replay_bkt(10, 7, true);
        let worst = replay_bkt(10, 7, false);
        eprintln!("BKT 10 attempts, 7 correct: best={:.4} worst={:.4}", best, worst);
        // Both endpoints should validate with generous tolerance
        let result_best = validate_bkt_integrity(best, 10, 7, 0.15);
        assert!(result_best.valid, "Best-case replay should be valid: {:?}", result_best.rejection_reason);
        let result_worst = validate_bkt_integrity(worst, 10, 7, 0.15);
        assert!(result_worst.valid, "Worst-case replay should be valid: {:?}", result_worst.rejection_reason);
    }

    #[test]
    fn test_bkt_replay_monotonic_with_correct() {
        // More correct answers = higher mastery
        let m5 = replay_bkt(10, 5, true);
        let m8 = replay_bkt(10, 8, true);
        let m10 = replay_bkt(10, 10, true);
        assert!(m5 < m8, "More correct should mean higher mastery");
        assert!(m8 < m10);
    }

    #[test]
    fn test_bkt_out_of_range() {
        let result = validate_bkt_integrity(1.5, 5, 3, 0.1);
        assert!(!result.valid, "p_mastery > 1.0 is invalid");
    }

    // ---- TEND Property Tests ----

    #[test]
    fn test_tend_never_negative() {
        for q in (0..=1000).step_by(100) {
            for d in [0, 100, 900, 3600, 7200] {
                for r in (0..=1000).step_by(200) {
                    let tend = compute_tend_pure(q, d, r, None);
                    assert!(tend >= 0.0, "TEND must never be negative: q={} d={} r={}", q, d, r);
                }
            }
        }
    }

    #[test]
    fn test_tend_never_exceeds_cap() {
        for q in (0..=1000).step_by(100) {
            for d in [900, 3600, 7200, 36000] {
                for r in (0..=1000).step_by(200) {
                    let tend = compute_tend_pure(q, d, r, Some(1000));
                    assert!(
                        tend <= MAX_TEND_PER_EVENT,
                        "TEND {} exceeds cap {}: q={} d={} r={}",
                        tend, MAX_TEND_PER_EVENT, q, d, r
                    );
                }
            }
        }
    }

    #[test]
    fn test_tend_quality_gate() {
        let tend = compute_tend_pure(500, 3600, 500, None);
        assert_eq!(tend, 0.0, "Quality below 600 should yield zero TEND");
    }

    #[test]
    fn test_tend_duration_gate() {
        let tend = compute_tend_pure(800, 600, 500, None);
        assert_eq!(tend, 0.0, "Duration below 900s should yield zero TEND");
    }

    #[test]
    fn test_tend_reputation_monotonic() {
        let low_rep = compute_tend_pure(800, 3600, 100, None);
        let mid_rep = compute_tend_pure(800, 3600, 500, None);
        let high_rep = compute_tend_pure(800, 3600, 1000, None);
        assert!(low_rep <= mid_rep, "Higher reputation should yield >= TEND");
        assert!(mid_rep <= high_rep);
    }

    #[test]
    fn test_tend_quality_monotonic() {
        let low_q = compute_tend_pure(600, 3600, 500, None);
        let mid_q = compute_tend_pure(800, 3600, 500, None);
        let high_q = compute_tend_pure(1000, 3600, 500, None);
        assert!(low_q <= mid_q, "Higher quality should yield >= TEND");
        assert!(mid_q <= high_q);
    }

    #[test]
    fn test_tend_properties_validator() {
        // Valid computation
        assert!(validate_tend_properties(0.5, 800, 3600).is_ok());

        // Negative
        assert!(validate_tend_properties(-0.1, 800, 3600).is_err());

        // Exceeds cap
        assert!(validate_tend_properties(3.0, 800, 3600).is_err());

        // Quality gate violation
        assert!(validate_tend_properties(0.5, 400, 3600).is_err());

        // Duration gate violation
        assert!(validate_tend_properties(0.5, 800, 600).is_err());
    }

    // ---- Application State Machine Exhaustive Tests ----

    #[test]
    fn test_state_machine_exhaustive() {
        let states = [
            "Draft", "Submitted", "UnderReview", "Interview",
            "Offered", "Accepted", "Rejected", "Withdrawn",
        ];

        // All 64 combinations (8×8)
        let expected_valid = [
            ("Draft", "Submitted"),
            ("Draft", "Withdrawn"),
            ("Submitted", "UnderReview"),
            ("Submitted", "Withdrawn"),
            ("UnderReview", "Interview"),
            ("UnderReview", "Rejected"),
            ("UnderReview", "Withdrawn"),
            ("Interview", "Offered"),
            ("Interview", "Rejected"),
            ("Interview", "Withdrawn"),
            ("Offered", "Accepted"),
            ("Offered", "Rejected"),
            ("Offered", "Withdrawn"),
        ];

        let mut valid_count = 0;
        let mut invalid_count = 0;

        for from in &states {
            for to in &states {
                let is_valid = is_valid_application_transition(from, to);
                let should_be_valid = expected_valid.contains(&(from, to));

                assert_eq!(
                    is_valid, should_be_valid,
                    "Transition {} -> {}: expected {}, got {}",
                    from, to, should_be_valid, is_valid
                );

                if is_valid {
                    valid_count += 1;
                } else {
                    invalid_count += 1;
                }
            }
        }

        assert_eq!(valid_count, 13, "Should have exactly 13 valid transitions");
        assert_eq!(invalid_count, 51, "Should have exactly 51 invalid transitions");
    }

    #[test]
    fn test_terminal_states_have_no_exits() {
        let terminals = ["Accepted", "Rejected", "Withdrawn"];
        let all_states = [
            "Draft", "Submitted", "UnderReview", "Interview",
            "Offered", "Accepted", "Rejected", "Withdrawn",
        ];

        for terminal in &terminals {
            for target in &all_states {
                assert!(
                    !is_valid_application_transition(terminal, target),
                    "Terminal state {} should not transition to {}",
                    terminal, target
                );
            }
        }
    }

    #[test]
    fn test_withdrawal_from_all_active_states() {
        let active = ["Draft", "Submitted", "UnderReview", "Interview", "Offered"];
        for state in &active {
            assert!(
                is_valid_application_transition(state, "Withdrawn"),
                "{} should allow withdrawal",
                state
            );
        }
    }

    #[test]
    fn test_no_backward_transitions() {
        // Can't go backward in the pipeline
        assert!(!is_valid_application_transition("Submitted", "Draft"));
        assert!(!is_valid_application_transition("UnderReview", "Submitted"));
        assert!(!is_valid_application_transition("Interview", "UnderReview"));
        assert!(!is_valid_application_transition("Offered", "Interview"));
    }

    #[test]
    fn test_no_skip_transitions() {
        // Can't skip stages
        assert!(!is_valid_application_transition("Draft", "Interview"));
        assert!(!is_valid_application_transition("Draft", "Offered"));
        assert!(!is_valid_application_transition("Submitted", "Offered"));
        assert!(!is_valid_application_transition("Draft", "Accepted"));
    }
}
