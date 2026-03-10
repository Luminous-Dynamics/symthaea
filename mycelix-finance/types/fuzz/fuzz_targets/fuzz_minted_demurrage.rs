#![no_main]
//! Fuzz target for `compute_minted_demurrage` (mutual credit demurrage).
//!
//! Invariants:
//! 1. Never panics
//! 2. Returns 0 for non-positive balances
//! 3. Deduction never exceeds balance
//! 4. Returns 0 when rate is non-positive or non-finite

use libfuzzer_sys::fuzz_target;
use mycelix_finance_types::compute_minted_demurrage;

fuzz_target!(|data: &[u8]| {
    if data.len() < 20 {
        return;
    }

    let balance = i32::from_le_bytes(data[0..4].try_into().unwrap());
    let rate_bits = u64::from_le_bytes(data[4..12].try_into().unwrap());
    let rate = f64::from_bits(rate_bits);
    let seconds_elapsed = u64::from_le_bytes(data[12..20].try_into().unwrap());

    let deduction = compute_minted_demurrage(balance, rate, seconds_elapsed);

    // Invariant: deduction is always non-negative
    assert!(deduction >= 0, "deduction must be >= 0, got {}", deduction);

    // Invariant: deduction never exceeds balance
    if balance > 0 {
        assert!(
            deduction <= balance,
            "deduction {} > balance {} for rate={}, secs={}",
            deduction, balance, rate, seconds_elapsed
        );
    }

    // Invariant: non-positive balance means zero deduction
    if balance <= 0 {
        assert_eq!(deduction, 0, "deduction must be 0 for non-positive balance {}", balance);
    }

    // Invariant: non-positive or non-finite rate means zero deduction
    if rate <= 0.0 || !rate.is_finite() {
        assert_eq!(deduction, 0, "deduction must be 0 for rate {}", rate);
    }
});
