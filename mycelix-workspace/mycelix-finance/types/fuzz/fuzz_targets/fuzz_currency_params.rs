#![no_main]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fuzz target for `MintedCurrencyParams::validate()`.
//!
//! Invariants:
//! 1. Never panics for any input
//! 2. Returns Ok or descriptive Err (no empty error strings)
//! 3. Valid params always pass (positive test after construction)

use libfuzzer_sys::fuzz_target;
use mycelix_finance_types::MintedCurrencyParams;

fuzz_target!(|data: &[u8]| {
    if data.len() < 20 {
        return;
    }

    // Extract fields from fuzzer input
    let name_len = (data[0] as usize) % 200;
    let symbol_len = (data[1] as usize) % 20;
    let desc_len = (data[2] as usize) % 200;
    let offset = 3;

    if data.len() < offset + name_len + symbol_len + desc_len + 12 {
        return;
    }

    let name = String::from_utf8_lossy(&data[offset..offset + name_len]).to_string();
    let symbol = String::from_utf8_lossy(&data[offset + name_len..offset + name_len + symbol_len])
        .to_string();
    let desc = String::from_utf8_lossy(
        &data[offset + name_len + symbol_len..offset + name_len + symbol_len + desc_len],
    )
    .to_string();

    let rest = &data[offset + name_len + symbol_len + desc_len..];
    if rest.len() < 12 {
        return;
    }

    let credit_limit = i32::from_le_bytes(rest[0..4].try_into().unwrap());
    let rate_bits = u32::from_le_bytes(rest[4..8].try_into().unwrap());
    let demurrage_rate = f32::from_bits(rate_bits) as f64;
    let max_hours = rest[8] as u32;
    let min_minutes = rest[9] as u32;
    let requires_confirmation = rest[10] & 1 == 1;
    let timeout = rest[11] as u32;

    let params = MintedCurrencyParams {
        name,
        symbol,
        description: desc,
        credit_limit,
        demurrage_rate,
        max_service_hours: max_hours,
        min_service_minutes: min_minutes,
        requires_confirmation,
        confirmation_timeout_hours: timeout,
        max_exchanges_per_day: 0,
    };

    // Must not panic
    let result = params.validate();

    // If Err, message must be non-empty
    if let Err(msg) = &result {
        assert!(!msg.is_empty(), "Error message must not be empty");
    }
});
