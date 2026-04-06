// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![no_main]
use libfuzzer_sys::fuzz_target;

/// Exact copy of validate_disk_path from ssh_relay.rs.
fn validate_disk_path(d: &str) -> Result<String, String> {
    let d = d.trim();
    if !d.starts_with("/dev/") {
        return Err("Disk must start with /dev/".into());
    }
    let dev = &d[5..];
    if dev.is_empty() {
        return Err("Invalid disk device: /dev/".into());
    }
    if !dev.chars().all(|c| c.is_alphanumeric()) {
        return Err(format!("Invalid disk device: {}", d));
    }
    let allowed_prefixes = ["sd", "vd", "nvme", "mmcblk", "xvd", "hd", "fd", "loop"];
    if !allowed_prefixes.iter().any(|p| dev.starts_with(p)) {
        return Err(format!("Unrecognized device class '{}'", d));
    }
    let blocked = ["/dev/null", "/dev/zero", "/dev/random", "/dev/urandom", "/dev/stdin", "/dev/stdout"];
    if blocked.contains(&d) {
        return Err("Not a valid disk device".into());
    }
    Ok(d.to_string())
}

fuzz_target!(|data: &str| {
    match validate_disk_path(data) {
        Ok(path) => {
            // Invariants for valid paths:
            assert!(path.starts_with("/dev/"), "Valid path must start with /dev/");
            assert!(!path.contains(".."), "Path traversal must be rejected");
            assert!(!path.contains(";"), "Shell metacharacter must be rejected");
            let dev = &path[5..];
            assert!(dev.chars().all(|c| c.is_alphanumeric()), "Only alphanumeric after /dev/");
        }
        Err(_) => {
            // Rejection is always safe
        }
    }
});
