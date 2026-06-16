// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Integrity Framework

Verifies structural invariants of the integrity subsystem:
- Consecutive failure counting is monotonic until reset
- Integrity confidence is constrained to {0.1, 0.5, 1.0}
- Clean registries always yield confidence 1.0
- N≥3 consecutive failures always yield Critical severity
- verify_live_thresholds is symmetric (match → None, mismatch → Some)
- Event log capacity is bounded

Feature-gated: `integrity`.
*/

#![cfg(feature = "integrity")]

use proptest::prelude::*;
use symthaea::integrity::{
    IntegrityManager,
    attestation::{AttestationRegistry, blake3_hash, blake3_hash_f32_slice},
};

// ═══════════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════════

fn arb_f32_slice() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(any::<f32>().prop_filter("finite", |v| v.is_finite()), 1..20)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Properties
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Consecutive failures monotonically increase until a pass resets them.
    #[test]
    fn prop_consecutive_failures_monotonic(n_failures in 1usize..10) {
        let mut reg = AttestationRegistry::new();
        let baseline = blake3_hash(b"original");
        reg.register_tampered(
            "test",
            baseline,
            Box::new(|| blake3_hash(b"tampered")),
        );
        let mut prev_streak = 0;
        for i in 0..n_failures {
            reg.verify_all(i + 1);
            let streak = reg.records()[0].consecutive_failures;
            prop_assert!(streak > prev_streak, "streak should increase: {} > {}", streak, prev_streak);
            prev_streak = streak;
        }
        prop_assert_eq!(prev_streak, n_failures);
    }

    /// Consecutive failures reset to 0 when verification passes.
    #[test]
    fn prop_consecutive_failures_reset_on_pass(n_failures in 1usize..10) {
        let mut reg = AttestationRegistry::new();
        let hash = blake3_hash(b"stable");
        // Start with a tampered hasher
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = call_count.clone();
        let threshold = n_failures;
        reg.register_tampered(
            "switchable",
            hash,
            Box::new(move || {
                let c = cc.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if c < threshold {
                    blake3_hash(b"tampered")
                } else {
                    blake3_hash(b"stable")
                }
            }),
        );
        // Run failures
        for i in 0..n_failures {
            reg.verify_all(i + 1);
        }
        prop_assert_eq!(reg.records()[0].consecutive_failures, n_failures);
        // One more pass → reset
        reg.verify_all(n_failures + 1);
        prop_assert_eq!(reg.records()[0].consecutive_failures, 0);
    }

    /// Integrity confidence is always in {0.1, 0.5, 1.0}.
    #[test]
    fn prop_integrity_confidence_constrained(cycle in 1usize..500) {
        let mut mgr = IntegrityManager::new();
        mgr.tick(cycle, 0.02, false);
        let c = mgr.integrity_confidence;
        prop_assert!(
            c == 1.0 || c == 0.5 || c == 0.1,
            "integrity_confidence must be 0.1, 0.5, or 1.0, got {}",
            c
        );
    }

    /// Clean registry always yields confidence 1.0.
    #[test]
    fn prop_clean_registry_confidence_one(cycle in 1usize..1000) {
        let mut mgr = IntegrityManager::new();
        // Register only passing attestations
        let hash = blake3_hash(b"clean");
        mgr.attestation.register("clean", hash, Box::new(|| blake3_hash(b"clean")));
        // Only tick on attestation intervals to actually trigger verification
        mgr.tick(cycle, 0.02, false);
        // If attestation ran (cycle % 101 == 0), it should pass
        prop_assert_eq!(mgr.integrity_confidence, 1.0);
    }

    /// N≥3 consecutive attestation failures always yield Critical severity.
    #[test]
    fn prop_three_plus_failures_critical(n in 3usize..15) {
        let mut mgr = IntegrityManager::new();
        let baseline = blake3_hash(b"original");
        mgr.attestation.register_tampered(
            "tampered",
            baseline,
            Box::new(|| blake3_hash(b"modified")),
        );
        // Run n failures with full_sweep to bypass jittered intervals
        for i in 1..=n {
            mgr.tick(i * 101, 0.02, true);
        }
        prop_assert!(mgr.has_critical_anomaly());
        prop_assert_eq!(mgr.integrity_confidence, 0.1);
    }

    /// verify_live_thresholds: matching hash → None.
    #[test]
    fn prop_live_verify_match_is_none(values in arb_f32_slice()) {
        let mut mgr = IntegrityManager::new();
        mgr.register_safety_thresholds(&values);
        let live = blake3_hash_f32_slice(&values);
        prop_assert!(mgr.verify_live_thresholds("safety_thresholds", live).is_none());
    }

    /// verify_live_thresholds: mismatched hash → Some.
    #[test]
    fn prop_live_verify_mismatch_is_some(
        values in arb_f32_slice(),
        delta_idx in 0usize..20,
    ) {
        let mut mgr = IntegrityManager::new();
        mgr.register_safety_thresholds(&values);
        let mut tampered = values.clone();
        let idx = delta_idx % tampered.len();
        // Use bit-level manipulation to guarantee a different value
        // (adding 1.0 to large floats can be a no-op due to ULP)
        let bits = tampered[idx].to_bits();
        tampered[idx] = f32::from_bits(bits ^ 1); // flip LSB — always different bytes
        let live = blake3_hash_f32_slice(&tampered);
        prop_assert!(mgr.verify_live_thresholds("safety_thresholds", live).is_some());
    }

    /// Event log never exceeds capacity.
    #[test]
    fn prop_event_log_bounded(n_ticks in 1usize..200) {
        let mut mgr = IntegrityManager::new();
        let baseline = blake3_hash(b"original");
        mgr.attestation.register_tampered(
            "tampered",
            baseline,
            Box::new(|| blake3_hash(b"modified")),
        );
        for i in 1..=n_ticks {
            mgr.tick(i * 101, 0.02, true); // full_sweep to bypass jittered interval
        }
        prop_assert!(mgr.event_log.len() <= 64, "event log exceeded capacity: {}", mgr.event_log.len());
    }
}