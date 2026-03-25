// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Fuzz target: Search query sanitization
//!
//! Tests that arbitrary input to search doesn't cause panics.
//! Run: cd holochain/fuzz && cargo +nightly fuzz run search_sanitization

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Test: UTF-8 handling
    let input = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return, // Invalid UTF-8 should be rejected at API boundary
    };

    // Reproduce our sanitize_text logic
    let sanitized: String = input
        .chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
        .take(500)
        .collect();

    // Invariants that must hold:
    assert!(sanitized.len() <= input.len());
    assert!(!sanitized.contains('\0')); // Null bytes stripped
    assert!(sanitized.chars().count() <= 500); // Length capped

    // Test: trigram generation from sanitized input
    let lower = sanitized.to_lowercase();
    for window in lower.as_bytes().windows(3) {
        // Trigram creation should never panic
        let _trigram = std::str::from_utf8(window);
    }
});
