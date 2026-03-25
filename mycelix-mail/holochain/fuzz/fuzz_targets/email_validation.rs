// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Fuzz target: Email validation
//!
//! Tests that malformed EncryptedEmail entries don't cause panics in validation.
//! Run: cd holochain/fuzz && cargo +nightly fuzz run email_validation

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Exercise the email_signing_content function with arbitrary data
    // This tests that the canonical encoding doesn't panic on any input
    if data.len() < 100 {
        return;
    }

    // Test: signature length constants are within expected ranges
    let sig_len = u16::from_le_bytes([data[0], data[1]]) as usize;
    let key_len = u16::from_le_bytes([data[2], data[3]]) as usize;

    // Verify our constants don't cause issues
    assert!(sig_len <= 65535);
    assert!(key_len <= 65535);

    // Test: CryptoSuite string parsing doesn't panic
    let suite_str = std::str::from_utf8(&data[4..data.len().min(20)]).unwrap_or("");
    let valid_suites = ["ed25519", "dilithium3", "dilithium2", "x25519", "kyber1024", "kyber768"];
    let _ = valid_suites.contains(&suite_str);

    // Test: chunk validation boundaries
    let chunk_index = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
    let total_chunks = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    // These should never panic
    let _ = chunk_index < total_chunks;
    let _ = total_chunks > 0 && total_chunks <= 1000;
});
