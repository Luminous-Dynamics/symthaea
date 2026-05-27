// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![no_main]
use libfuzzer_sys::fuzz_target;
use symthaea_spore::security::validate_disk_path;

fuzz_target!(|data: &str| {
    match validate_disk_path(data) {
        Ok(path) => {
            assert!(
                path.starts_with("/dev/"),
                "Valid path must start with /dev/"
            );
            assert!(!path.contains(".."), "Path traversal must be rejected");
            assert!(!path.contains(";"), "Shell metacharacter must be rejected");
            let dev = &path[5..];
            assert!(
                dev.chars().all(|c| c.is_alphanumeric()),
                "Only alphanumeric after /dev/"
            );
        }
        Err(_) => {} // Rejection is always safe
    }
});