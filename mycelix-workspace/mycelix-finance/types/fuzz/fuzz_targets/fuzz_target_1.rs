#![no_main]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fuzz target for `compute_demurrage_deduction`.
//!
//! Invariants being checked:
//! 1. Never panics for any input combination
//! 2. Deduction never exceeds eligible balance (balance - exempt_floor)
//! 3. Returns 0 when balance <= exempt_floor
//! 4. Returns 0 when seconds_elapsed == 0

use libfuzzer_sys::fuzz_target;
use mycelix_finance_types::compute_demurrage_deduction;

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }

    let balance = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let exempt_floor = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let rate_bits = u64::from_le_bytes(data[16..24].try_into().unwrap());
    let rate = f64::from_bits(rate_bits);
    let seconds_elapsed = if data.len() >= 32 {
        u64::from_le_bytes(data[24..32].try_into().unwrap())
    } else {
        0
    };

    let deduction = compute_demurrage_deduction(balance, exempt_floor, rate, seconds_elapsed);

    // Invariant: deduction never exceeds eligible balance
    if balance > exempt_floor {
        let eligible = balance - exempt_floor;
        assert!(
            deduction <= eligible,
            "deduction {} > eligible {} for balance={}, floor={}, rate={}, secs={}",
            deduction, eligible, balance, exempt_floor, rate, seconds_elapsed
        );
    } else {
        assert_eq!(deduction, 0, "deduction must be 0 when balance <= exempt_floor");
    }

    // Invariant: deduction is 0 when seconds_elapsed is 0
    if seconds_elapsed == 0 {
        assert_eq!(deduction, 0, "deduction must be 0 when seconds_elapsed == 0");
    }
});
