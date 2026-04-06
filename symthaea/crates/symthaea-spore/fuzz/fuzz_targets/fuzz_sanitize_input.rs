// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![no_main]
use libfuzzer_sys::fuzz_target;

/// Exact copy of sanitize_input from ssh_relay.rs.
fn sanitize_input(s: &str, field_name: &str, allow_slashes: bool) -> Result<String, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err(format!("{} cannot be empty", field_name));
    }
    for ch in s.chars() {
        match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => {}
            '/' if allow_slashes => {}
            _ => return Err(format!("{} contains invalid character '{}'", field_name, ch)),
        }
    }
    Ok(s.to_string())
}

fuzz_target!(|data: &str| {
    // Test both with and without slash allowance
    for allow_slashes in [false, true] {
        match sanitize_input(data, "fuzz_field", allow_slashes) {
            Ok(result) => {
                // Invariants for valid output:
                assert!(!result.contains(';'), "Semicolon must be rejected");
                assert!(!result.contains('`'), "Backtick must be rejected");
                assert!(!result.contains('$'), "Dollar sign must be rejected");
                assert!(!result.contains('\''), "Single quote must be rejected");
                assert!(!result.contains('"'), "Double quote must be rejected");
                assert!(!result.contains('\\'), "Backslash must be rejected");
                assert!(!result.contains('|'), "Pipe must be rejected");
                assert!(!result.contains('&'), "Ampersand must be rejected");
                assert!(!result.contains('\n'), "Newline must be rejected");
                if !allow_slashes {
                    assert!(!result.contains('/'), "Slash must be rejected when not allowed");
                }
            }
            Err(_) => {
                // Rejection is always safe
            }
        }
    }
});
