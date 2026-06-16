// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Security validators for the NixForHumanity relay and installer.
//!
//! These functions form the security boundary between untrusted browser input
//! and shell commands executed on target machines. They are extracted here
//! for testability and fuzzing (see `fuzz/` directory).

/// Strip any line that exactly matches a heredoc delimiter from content.
///
/// Prevents heredoc escape injection: if an attacker embeds the delimiter
/// (e.g., `NIXCONF`) on its own line in browser-supplied Nix config,
/// it would break out of the heredoc and allow arbitrary shell execution.
pub fn sanitize_heredoc(content: &str, delimiter: &str) -> String {
    content
        .lines()
        .filter(|line| line.trim() != delimiter)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Validate disk device path — must match known block device patterns.
///
/// Prevents targeting arbitrary /dev/ nodes (symlinks, char devices, pseudo-devices).
/// Only allows alphanumeric characters after `/dev/` (no path traversal via `.` or `/`).
pub fn validate_disk_path(d: &str) -> Result<String, String> {
    let d = d.trim();
    if !d.starts_with("/dev/") {
        return Err("Disk must start with /dev/".into());
    }
    let dev = &d[5..];
    if dev.is_empty() {
        return Err("Invalid disk device: /dev/".into());
    }
    // Only allow alphanumeric (no path traversal via . or /)
    if !dev.chars().all(|c| c.is_alphanumeric()) {
        return Err(format!("Invalid disk device: {}", d));
    }
    // Whitelist known block device prefixes
    let allowed_prefixes = ["sd", "vd", "nvme", "mmcblk", "xvd", "hd", "fd", "loop"];
    if !allowed_prefixes.iter().any(|p| dev.starts_with(p)) {
        return Err(format!(
            "Unrecognized device class '{}'. Allowed: sd*, vd*, nvme*, mmcblk*, xvd*, hd*",
            d
        ));
    }
    // Block pseudo-devices
    let blocked = [
        "/dev/null",
        "/dev/zero",
        "/dev/random",
        "/dev/urandom",
        "/dev/stdin",
        "/dev/stdout",
    ];
    if blocked.contains(&d) {
        return Err("Not a valid disk device".into());
    }
    Ok(d.to_string())
}

/// Sanitize a string for safe use in shell commands and Nix config.
///
/// Only allows alphanumeric, hyphens, underscores, dots, and optionally forward slashes.
/// Rejects shell metacharacters (`; ` `` | & $ ' " \`), preventing command injection.
pub fn sanitize_input(s: &str, field_name: &str, allow_slashes: bool) -> Result<String, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err(format!("{} cannot be empty", field_name));
    }
    for ch in s.chars() {
        match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => {}
            '/' if allow_slashes => {}
            _ => {
                return Err(format!(
                    "{} contains invalid character '{}'",
                    field_name, ch
                ));
            }
        }
    }
    Ok(s.to_string())
}

/// Validate hostname — RFC 1123. Returns lowercase, defaults to "guardian" if empty.
pub fn validate_hostname(h: &str) -> Result<String, String> {
    let h = h.trim().to_lowercase();
    if h.is_empty() {
        return Ok("guardian".into());
    }
    if h.len() > 63 {
        return Err("Hostname too long (max 63)".into());
    }
    if !h
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err("Hostname can only contain lowercase letters, numbers, hyphens".into());
    }
    Ok(h)
}

/// Constant-time token comparison — prevents timing side-channel attacks.
pub fn token_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result = 0u8;
    for (x, y) in a.bytes().zip(b.bytes()) {
        result |= x ^ y;
    }
    result == 0
}

/// Validate a Nix expression by running `nix eval --pure-eval --expr` on it.
///
/// Pure eval disables network access, filesystem access outside the Nix store,
/// and `builtins.exec` — preventing malicious Nix expressions from exfiltrating
/// data or executing commands during evaluation.
///
/// Returns Ok(()) if the expression parses and evaluates without error,
/// or Err with the nix error output if evaluation fails.
///
/// Note: This is a best-effort check. Some valid NixOS module patterns may
/// fail pure eval (e.g., those that reference `<nixpkgs>`). Use alongside
/// rnix-parser for syntax validation.
#[cfg(not(target_arch = "wasm32"))]
pub fn validate_nix_pure_eval(nix_content: &str) -> Result<(), String> {
    use std::process::Command;

    // Wrap in a trivial evaluator — we just want to check it parses
    let expr = format!(
        "let config = {{}}; pkgs = {{}}; lib = {{}}; in builtins.tryEval ({})",
        nix_content
    );

    let output = Command::new("nix")
        .args(["eval", "--pure-eval", "--expr", &expr])
        .output()
        .map_err(|e| format!("Failed to run nix eval: {} (is nix installed?)", e))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Extract just the error message, not the full trace
        let msg = stderr
            .lines()
            .find(|l| l.contains("error:"))
            .unwrap_or("unknown evaluation error");
        Err(format!("Nix pure-eval failed: {}", msg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── sanitize_heredoc ──

    #[test]
    fn heredoc_strips_exact_delimiter() {
        let input = "line1\nNIXCONF\nline3";
        let result = sanitize_heredoc(input, "NIXCONF");
        assert!(result.contains("line1"));
        assert!(result.contains("line3"));
        assert!(!result.lines().any(|l| l.trim() == "NIXCONF"));
    }

    #[test]
    fn heredoc_preserves_partial_match() {
        let input = "NIXCONF_EXTRA = true;\nreal content";
        let result = sanitize_heredoc(input, "NIXCONF");
        assert!(result.contains("NIXCONF_EXTRA"));
    }

    #[test]
    fn heredoc_strips_indented() {
        let input = "line1\n  NIXCONF  \nline3";
        let result = sanitize_heredoc(input, "NIXCONF");
        assert!(!result.lines().any(|l| l.trim() == "NIXCONF"));
    }

    #[test]
    fn heredoc_empty_input() {
        assert_eq!(sanitize_heredoc("", "NIXCONF"), "");
    }

    // ── validate_disk_path ──

    #[test]
    fn disk_valid_devices() {
        assert!(validate_disk_path("/dev/sda").is_ok());
        assert!(validate_disk_path("/dev/nvme0n1").is_ok());
        assert!(validate_disk_path("/dev/vda").is_ok());
        assert!(validate_disk_path("/dev/mmcblk0").is_ok());
        assert!(validate_disk_path("  /dev/sda  ").unwrap() == "/dev/sda");
    }

    #[test]
    fn disk_rejects_invalid() {
        assert!(validate_disk_path("/tmp/sda").is_err());
        assert!(validate_disk_path("/dev/../etc/passwd").is_err());
        assert!(validate_disk_path("/dev/null").is_err());
        assert!(validate_disk_path("/dev/urandom").is_err());
        assert!(validate_disk_path("/dev/zz0").is_err());
        assert!(validate_disk_path("/dev/").is_err());
    }

    // ── sanitize_input ──

    #[test]
    fn sanitize_valid() {
        assert!(sanitize_input("my-host.name", "h", false).is_ok());
        assert!(sanitize_input("America/Chicago", "tz", true).is_ok());
    }

    #[test]
    fn sanitize_rejects_shell_metacharacters() {
        assert!(sanitize_input("foo;rm -rf /", "f", false).is_err());
        assert!(sanitize_input("foo`id`", "f", false).is_err());
        assert!(sanitize_input("foo$HOME", "f", false).is_err());
        assert!(sanitize_input("foo|bar", "f", false).is_err());
        assert!(sanitize_input("foo'bar", "f", false).is_err());
    }

    #[test]
    fn sanitize_slash_gating() {
        assert!(sanitize_input("America/Chicago", "tz", false).is_err());
        assert!(sanitize_input("America/Chicago", "tz", true).is_ok());
    }

    // ── validate_hostname ──

    #[test]
    fn hostname_valid() {
        assert_eq!(validate_hostname("my-host").unwrap(), "my-host");
        assert_eq!(validate_hostname("").unwrap(), "guardian");
        assert_eq!(validate_hostname("  MY-HOST  ").unwrap(), "my-host");
    }

    #[test]
    fn hostname_rejects_invalid() {
        assert!(validate_hostname("host;evil").is_err());
        assert!(validate_hostname(&"a".repeat(64)).is_err());
    }

    // ── token_eq ──

    #[test]
    fn token_comparison() {
        assert!(token_eq("sovereign", "sovereign"));
        assert!(!token_eq("sovereign", "Sovereign"));
        assert!(!token_eq("short", "longer"));
    }
}
