// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS Φ-Gate: Consciousness-Gated Action Execution
//!
//! Maps NixOS actions to consciousness thresholds and provides
//! rollback lookup for known NixOS operations.

use super::executor::SafetyLevel;
use std::borrow::Cow;

/// Get the rollback command for known NixOS operations.
///
/// Returns `Cow::Borrowed` for constant rollback strings and
/// `Cow::Owned` when the rollback depends on the original command's arguments.
pub fn get_nixos_rollback(command: &str) -> Option<Cow<'static, str>> {
    let cmd = command.trim().to_lowercase();

    if cmd.starts_with("nixos-rebuild switch") || cmd.starts_with("nixos-rebuild boot") {
        Some(Cow::Borrowed("nixos-rebuild switch --rollback"))
    } else if cmd.contains("nix-env -i") || cmd.contains("nix profile install") {
        let pkg = cmd.split_whitespace().last().unwrap_or("package");
        Some(Cow::Owned(format!(
            "nix-env -e {pkg} || nix profile remove {pkg}"
        )))
    } else if cmd.starts_with("nix build")
        || cmd.starts_with("nix develop")
        || cmd.starts_with("nix shell")
    {
        Some(Cow::Borrowed("exit")) // nix environments are ephemeral
    } else if cmd.starts_with("systemctl restart") || cmd.starts_with("systemctl stop") {
        let svc = cmd.split_whitespace().last().unwrap_or("service");
        Some(Cow::Owned(format!("systemctl start {svc}")))
    } else {
        None
    }
}

/// Classify command destructiveness based on common patterns
pub fn classify_command_destructiveness(command: &str) -> SafetyLevel {
    let cmd = command.trim().to_lowercase();

    // Destructive operations
    if cmd.contains("--purge")
        || cmd.contains("gc")
        || cmd.contains("delete")
        || cmd.contains("rm ")
        || cmd.contains("remove")
        || cmd.starts_with("nix-collect-garbage")
        || cmd.contains("wipe")
        || cmd.contains("format")
    {
        return SafetyLevel::Destructive;
    }

    // System-critical
    if cmd.starts_with("nixos-rebuild")
        || cmd.contains("switch")
        || cmd.starts_with("nix profile install")
        || cmd.starts_with("nix-env -i")
        || cmd.contains("upgrade")
        || cmd.contains("update")
    {
        return SafetyLevel::SystemCritical;
    }

    // User modify
    if cmd.starts_with("nix build")
        || cmd.starts_with("nix develop")
        || cmd.starts_with("nix shell")
        || cmd.contains("install")
    {
        return SafetyLevel::UserModify;
    }

    // Read-only
    if cmd.starts_with("nix search")
        || cmd.starts_with("nix-env -q")
        || cmd.starts_with("nixos-option")
        || cmd.contains("list")
        || cmd.contains("info")
        || cmd.contains("show")
        || cmd.contains("search")
        || cmd.starts_with("nix flake show")
        || cmd.starts_with("nix flake metadata")
    {
        return SafetyLevel::ReadOnly;
    }

    // Default to system-critical for unknown commands
    SafetyLevel::SystemCritical
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nixos_rollback_known() {
        assert!(get_nixos_rollback("nixos-rebuild switch").is_some());
        assert!(get_nixos_rollback("nix-env -i firefox").is_some());
        assert!(get_nixos_rollback("nix build .#package").is_some());
    }

    #[test]
    fn test_classify_destructiveness() {
        assert_eq!(
            classify_command_destructiveness("nix search nixpkgs firefox"),
            SafetyLevel::ReadOnly
        );
        assert_eq!(
            classify_command_destructiveness("nix-collect-garbage -d"),
            SafetyLevel::Destructive
        );
        assert_eq!(
            classify_command_destructiveness("nixos-rebuild switch"),
            SafetyLevel::SystemCritical
        );
    }

    #[test]
    fn test_classify_whitespace_handling() {
        // Leading/trailing whitespace should be trimmed
        assert_eq!(
            classify_command_destructiveness("  nix search nixpkgs vim  "),
            SafetyLevel::ReadOnly
        );
        assert_eq!(
            classify_command_destructiveness("\tnixos-rebuild switch\n"),
            SafetyLevel::SystemCritical
        );
    }

    #[test]
    fn test_classify_case_insensitive() {
        assert_eq!(
            classify_command_destructiveness("NIX SEARCH nixpkgs firefox"),
            SafetyLevel::ReadOnly
        );
        assert_eq!(
            classify_command_destructiveness("Nix-Collect-Garbage -d"),
            SafetyLevel::Destructive
        );
    }

    #[test]
    fn test_classify_profile_install_vs_env_install() {
        // Both nix profile install and nix-env -i should be SystemCritical
        assert_eq!(
            classify_command_destructiveness("nix profile install nixpkgs#firefox"),
            SafetyLevel::SystemCritical
        );
        assert_eq!(
            classify_command_destructiveness("nix-env -i firefox"),
            SafetyLevel::SystemCritical
        );
    }

    #[test]
    fn test_classify_empty_and_unknown() {
        // Empty string defaults to SystemCritical (unknown)
        assert_eq!(
            classify_command_destructiveness(""),
            SafetyLevel::SystemCritical
        );
        // Unknown command defaults to SystemCritical
        assert_eq!(
            classify_command_destructiveness("some-random-tool --flag"),
            SafetyLevel::SystemCritical
        );
    }

    #[test]
    fn test_classify_destructive_patterns() {
        assert_eq!(
            classify_command_destructiveness("nix store delete /nix/store/abc"),
            SafetyLevel::Destructive
        );
        assert_eq!(
            classify_command_destructiveness("rm -rf /nix/store/old"),
            SafetyLevel::Destructive
        );
        assert_eq!(
            classify_command_destructiveness("nix store gc --purge"),
            SafetyLevel::Destructive
        );
    }

    #[test]
    fn test_classify_user_modify_commands() {
        assert_eq!(
            classify_command_destructiveness("nix build .#mypackage"),
            SafetyLevel::UserModify
        );
        assert_eq!(
            classify_command_destructiveness("nix develop"),
            SafetyLevel::UserModify
        );
        assert_eq!(
            classify_command_destructiveness("nix shell nixpkgs#hello"),
            SafetyLevel::UserModify
        );
    }

    #[test]
    fn test_classify_read_only_variants() {
        assert_eq!(
            classify_command_destructiveness("nix-env -q"),
            SafetyLevel::ReadOnly
        );
        assert_eq!(
            classify_command_destructiveness("nixos-option services.nginx.enable"),
            SafetyLevel::ReadOnly
        );
        assert_eq!(
            classify_command_destructiveness("nix flake show github:NixOS/nixpkgs"),
            SafetyLevel::ReadOnly
        );
        assert_eq!(
            classify_command_destructiveness("nix flake metadata"),
            SafetyLevel::ReadOnly
        );
        assert_eq!(
            classify_command_destructiveness("systemd-analyze list-dependencies"),
            SafetyLevel::ReadOnly
        );
    }

    #[test]
    fn test_rollback_known_commands() {
        // Rebuild → rollback
        let rb = get_nixos_rollback("nixos-rebuild switch");
        assert!(rb.is_some());
        assert!(rb.unwrap().contains("rollback"));

        // nix-env install → remove
        let rb = get_nixos_rollback("nix-env -i firefox");
        assert!(rb.is_some());
        assert!(rb.unwrap().contains("firefox"));

        // nix profile install → remove
        let rb = get_nixos_rollback("nix profile install nixpkgs#vim");
        assert!(rb.is_some());
        assert!(rb.unwrap().contains("vim"));

        // systemctl restart → start
        let rb = get_nixos_rollback("systemctl restart nginx");
        assert!(rb.is_some());
        assert!(rb.unwrap().contains("start"));

        // Unknown → None
        assert!(get_nixos_rollback("echo hello").is_none());
    }

    #[test]
    fn test_rollback_cow_borrowed_for_constants() {
        use std::borrow::Cow;
        // Constant rollback strings should be Cow::Borrowed (zero allocation)
        let rb = get_nixos_rollback("nixos-rebuild switch").unwrap();
        assert!(
            matches!(rb, Cow::Borrowed(_)),
            "Constant rollback should be Cow::Borrowed"
        );

        // Dynamic rollback strings should be Cow::Owned
        let rb = get_nixos_rollback("nix-env -i firefox").unwrap();
        assert!(
            matches!(rb, Cow::Owned(_)),
            "Dynamic rollback should be Cow::Owned"
        );
    }

    #[test]
    fn test_rollback_ephemeral_commands() {
        // nix build/develop/shell are ephemeral
        let rb = get_nixos_rollback("nix build .#package");
        assert_eq!(rb.as_deref(), Some("exit"));
        let rb = get_nixos_rollback("nix develop");
        assert_eq!(rb.as_deref(), Some("exit"));
        let rb = get_nixos_rollback("nix shell nixpkgs#hello");
        assert_eq!(rb.as_deref(), Some("exit"));
    }
}
