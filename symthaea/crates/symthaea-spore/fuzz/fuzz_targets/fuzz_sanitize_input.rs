// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![no_main]
use libfuzzer_sys::fuzz_target;
use symthaea_spore::security::sanitize_input;

fuzz_target!(|data: &str| {
    for allow_slashes in [false, true] {
        match sanitize_input(data, "fuzz_field", allow_slashes) {
            Ok(result) => {
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
                    assert!(
                        !result.contains('/'),
                        "Slash must be rejected when not allowed"
                    );
                }
            }
            Err(_) => {}
        }
    }
});