// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![no_main]
use libfuzzer_sys::fuzz_target;
use symthaea_spore::security::sanitize_heredoc;

fuzz_target!(|data: &str| {
    let delimiters = ["NIXCONF", "FLAKEEOF", "SCRIPTEOF", "SYSPATCH"];
    for delim in &delimiters {
        let result = sanitize_heredoc(data, delim);
        // Invariant: no line in the output should be exactly the delimiter
        for line in result.lines() {
            assert_ne!(
                line.trim(),
                *delim,
                "sanitize_heredoc failed to strip delimiter '{}' from input",
                delim
            );
        }
    }
});