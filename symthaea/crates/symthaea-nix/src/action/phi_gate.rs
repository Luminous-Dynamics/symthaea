//! NixOS Φ-Gate: Consciousness-Gated Action Execution
//!
//! Maps NixOS actions to consciousness thresholds and provides
//! rollback lookup for known NixOS operations.

use super::executor::SafetyLevel;

/// Get the rollback command for known NixOS operations.
pub fn get_nixos_rollback(command: &str) -> Option<String> {
    let cmd = command.trim().to_lowercase();

    if cmd.starts_with("nixos-rebuild switch") || cmd.starts_with("nixos-rebuild boot") {
        Some("nixos-rebuild switch --rollback".to_string())
    } else if cmd.contains("nix-env -i") || cmd.contains("nix profile install") {
        let pkg = cmd.split_whitespace().last().unwrap_or("package");
        Some(format!("nix-env -e {} || nix profile remove {}", pkg, pkg))
    } else if cmd.starts_with("nix build")
        || cmd.starts_with("nix develop")
        || cmd.starts_with("nix shell")
    {
        Some("exit".to_string()) // nix environments are ephemeral
    } else if cmd.starts_with("systemctl restart") || cmd.starts_with("systemctl stop") {
        let svc = cmd.split_whitespace().last().unwrap_or("service");
        Some(format!("systemctl start {}", svc))
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
}
