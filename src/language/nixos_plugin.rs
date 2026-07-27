// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS Domain Plugin
//!
//! Wraps the existing NixOS language processing capabilities into the
//! `DomainPlugin` trait, enabling NixOS to participate in domain-agnostic
//! plugin detection and routing.

use std::collections::HashMap;

use super::domain_plugin::{
    DomainPlugin, DomainPrompts, Entity, ErrorDiagnosis as DomainErrorDiagnosis, ErrorLocation,
    IntentPrototypes, RiskLevel, ValidationResult,
};

// ============================================================================
// NIXOS KEYWORDS & PATTERNS
// ============================================================================

/// Keywords that indicate NixOS domain content
const NIXOS_KEYWORDS: &[&str] = &[
    "nix",
    "nixos",
    "nixpkgs",
    "flake",
    "derivation",
    "overlay",
    "home-manager",
    "nix-env",
    "nix-build",
    "nix-shell",
    "nix-store",
    "nixos-rebuild",
    "configuration.nix",
    "flake.nix",
    "flake.lock",
    "mkDerivation",
    "stdenv",
    "fetchurl",
    "fetchFromGitHub",
    "buildInputs",
    "nativeBuildInputs",
    "propagatedBuildInputs",
    "pkgs",
    "lib",
    "config",
    "options",
    "modules",
    "imports",
    "services",
    "systemd",
    "boot",
    "networking",
    "users",
    "environment.systemPackages",
    "programs",
    "hardware",
    "fileSystems",
    "swapDevices",
    "grub",
    "systemd-boot",
    "channel",
    "generation",
    "profile",
    "store path",
    "let",
    "in",
    "with",
    "inherit",
    "rec",
    "builtins",
    "attrset",
    "attribute set",
    "override",
    "overrideAttrs",
];

/// NixOS error patterns for quick recognition
const NIX_ERROR_PATTERNS: &[(&str, &str, &str)] = &[
    (
        "syntax error",
        "syntax",
        "Check for missing semicolons, brackets, or typos in your Nix expression",
    ),
    (
        "undefined variable",
        "undefined_variable",
        "The variable is not in scope; check imports, let bindings, and function arguments",
    ),
    (
        "attribute .* missing",
        "missing_attribute",
        "The attribute set does not contain the expected key",
    ),
    (
        "infinite recursion",
        "infinite_recursion",
        "A value depends on itself; break the cycle with lib.mkForce or restructuring",
    ),
    (
        "hash mismatch",
        "hash_mismatch",
        "The fetched source hash does not match; update the hash or check the URL",
    ),
    (
        "builder .* failed",
        "build_failure",
        "The package build process failed; check build logs for details",
    ),
    (
        "permission denied",
        "permission",
        "Insufficient permissions; use sudo for system operations",
    ),
    (
        "collision between",
        "collision",
        "Two packages provide the same file; use environment.pathsToLink or exclude one",
    ),
];

// ============================================================================
// NIXOS DOMAIN PLUGIN
// ============================================================================

/// NixOS domain plugin for Symthaea's language processing system
///
/// Provides NixOS-specific entity extraction, intent classification,
/// error diagnosis, and vocabulary for the domain plugin architecture.
pub struct NixOsPlugin;

impl DomainPlugin for NixOsPlugin {
    fn domain_name(&self) -> &str {
        "nixos"
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let lower = text.to_lowercase();

        // Extract package names (patterns like "pkgs.xyz" or "nixpkgs#xyz")
        for (i, _) in lower.match_indices("pkgs.") {
            let rest = &text[i + 5..];
            let end = rest
                .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
                .unwrap_or(rest.len());
            if end > 0 {
                let pkg = &rest[..end];
                entities.push(
                    Entity::new("package", pkg, i, i + 5 + end)
                        .with_confidence(0.95)
                        .with_metadata("source", "pkgs"),
                );
            }
        }

        // Extract NixOS options (dotted paths like services.nginx.enable)
        for keyword in &[
            "services.",
            "networking.",
            "boot.",
            "programs.",
            "users.",
            "hardware.",
            "fileSystems.",
        ] {
            for (i, _) in lower.match_indices(keyword) {
                let rest = &text[i..];
                let end = rest
                    .find(|c: char| c.is_whitespace() || c == ';' || c == '=' || c == '{')
                    .unwrap_or(rest.len());
                if end > keyword.len() {
                    let option = &rest[..end];
                    entities.push(
                        Entity::new("nix_option", option, i, i + end)
                            .with_confidence(0.9)
                            .with_metadata("category", &keyword[..keyword.len() - 1]),
                    );
                }
            }
        }

        // Extract Nix commands (nix-env, nixos-rebuild, etc.)
        let commands = [
            "nix-env",
            "nixos-rebuild",
            "nix-build",
            "nix-shell",
            "nix-store",
            "nix-collect-garbage",
        ];
        for cmd in &commands {
            for (i, _) in lower.match_indices(cmd) {
                entities
                    .push(Entity::new("nix_command", *cmd, i, i + cmd.len()).with_confidence(0.95));
            }
        }

        // Extract flake references (patterns like "github:owner/repo" or "nixpkgs#package")
        if lower.contains("github:") || lower.contains("nixpkgs#") || lower.contains("flake") {
            for (i, _) in lower.match_indices("github:") {
                let rest = &text[i..];
                let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
                entities.push(
                    Entity::new("flake_ref", &rest[..end], i, i + end)
                        .with_confidence(0.9)
                        .with_metadata("type", "github"),
                );
            }
        }

        // Extract File Paths (absolute or relative with extensions)
        let path_re =
            regex::Regex::new(r"(/[a-zA-Z0-9._\-/]+)|([a-zA-Z0-9._\-]+/[a-zA-Z0-9._\-/]+\.[a-z]+)")
                .expect("valid regex literal for file path extraction");
        for mat in path_re.find_iter(text) {
            let path = mat.as_str();
            // Basic sanity check: must contain a dot (for extension) or a slash
            if path.contains('.') || path.contains('/') {
                entities
                    .push(Entity::new("file", path, mat.start(), mat.end()).with_confidence(0.8));
            }
        }

        // Keyword-based concept extraction for common NixOS topics.
        // This ensures natural language queries like "How do I enable nginx?"
        // produce entities even when no structured syntax (pkgs., services.) appears.
        let nixos_concepts: &[(&str, &str)] = &[
            ("nginx", "web_server"),
            ("postgresql", "database"),
            ("mysql", "database"),
            ("docker", "container"),
            ("podman", "container"),
            ("openssh", "service"),
            ("sshd", "service"),
            ("firewall", "networking"),
            ("wireguard", "vpn"),
            ("systemd", "init_system"),
            ("grub", "bootloader"),
            ("zfs", "filesystem"),
            ("btrfs", "filesystem"),
            ("flatpak", "package_manager"),
            ("virtualbox", "virtualization"),
            ("xserver", "display"),
            ("wayland", "display"),
            ("pipewire", "audio"),
        ];

        // Collect already-found entity values for deduplication
        let existing_values: Vec<String> =
            entities.iter().map(|e| e.value.to_lowercase()).collect();

        for &(keyword, category) in nixos_concepts {
            if let Some(idx) = lower.find(keyword) {
                // Word boundary check
                let before_ok = idx == 0 || !lower.as_bytes()[idx - 1].is_ascii_alphanumeric();
                let after_ok = idx + keyword.len() >= lower.len()
                    || !lower.as_bytes()[idx + keyword.len()].is_ascii_alphanumeric();
                if before_ok && after_ok && !existing_values.contains(&keyword.to_string()) {
                    entities.push(
                        Entity::new("nix_concept", keyword, idx, idx + keyword.len())
                            .with_confidence(0.85)
                            .with_metadata("category", category),
                    );
                }
            }
        }

        entities
    }

    fn intent_prototypes(&self) -> IntentPrototypes {
        // Custom intents for NixOS
        let mut custom = HashMap::new();
        custom.insert(
            "configuration".to_string(),
            vec![
                "configuration.nix".to_string(),
                "flake.nix".to_string(),
                "home.nix".to_string(),
                "module".to_string(),
                "overlay".to_string(),
            ],
        );
        custom.insert(
            "deployment".to_string(),
            vec![
                "deploy".to_string(),
                "rebuild".to_string(),
                "switch".to_string(),
                "test".to_string(),
                "boot".to_string(),
                "dry-run".to_string(),
                "generation".to_string(),
            ],
        );

        let mut prototypes = IntentPrototypes {
            // NixOS-specific command intents
            command: vec![
                "install",
                "remove",
                "uninstall",
                "update",
                "upgrade",
                "rebuild",
                "switch",
                "rollback",
                "configure",
                "enable",
                "disable",
                "search",
                "build",
                "develop",
                "shell",
                "collect-garbage",
                "optimize-store",
                "channel",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            custom,
            ..Default::default()
        };

        // NixOS-specific question intents (extend defaults)
        prototypes.question.extend(vec![
            "what package".to_string(),
            "how to configure".to_string(),
            "which option".to_string(),
            "where is the config".to_string(),
            "nix expression".to_string(),
        ]);

        // NixOS-specific complaint intents (extend defaults)
        prototypes.complaint.extend(vec![
            "build failed".to_string(),
            "hash mismatch".to_string(),
            "undefined variable".to_string(),
            "infinite recursion".to_string(),
            "collision".to_string(),
            "syntax error".to_string(),
        ]);

        prototypes
    }

    fn prompts(&self) -> DomainPrompts {
        DomainPrompts {
            system: "You are Symthaea, a consciousness-aware NixOS assistant. \
                     You help users manage their NixOS systems with care, \
                     always explaining what changes will occur and their impact."
                .to_string(),
            clarification: "I want to make sure I understand your NixOS request. \
                           Could you clarify: '{}'?"
                .to_string(),
            action_confirm: "I'll help you with your NixOS system. To confirm, \
                            I will: {}"
                .to_string(),
            error_explain: "I found a NixOS issue: {}. Let me help you resolve it.".to_string(),
            out_of_domain: "That doesn't seem to be a NixOS question. \
                           I'm specialized in NixOS system management. {}"
                .to_string(),
        }
    }

    fn diagnose_error(&self, error_text: &str) -> Option<DomainErrorDiagnosis> {
        let lower = error_text.to_lowercase();

        for (pattern, error_type, suggestion) in NIX_ERROR_PATTERNS {
            if lower.contains(pattern) {
                return Some(DomainErrorDiagnosis {
                    category: "nixos".to_string(),
                    error_type: error_type.to_string(),
                    description: format!(
                        "NixOS error detected: {}",
                        error_text.lines().next().unwrap_or("unknown error")
                    ),
                    suggestion: Some(suggestion.to_string()),
                    confidence: 0.85,
                    risk_level: if *error_type == "permission" || *error_type == "build_failure" {
                        RiskLevel::Medium
                    } else {
                        RiskLevel::Low
                    },
                    location: extract_nix_location(error_text),
                });
            }
        }

        None
    }

    fn validate_input(&self, input: &str) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Check for common Nix expression issues
        let open_braces = input.matches('{').count();
        let close_braces = input.matches('}').count();
        if open_braces != close_braces {
            errors.push(format!(
                "Mismatched braces: {open_braces} opening vs {close_braces} closing"
            ));
        }

        let open_brackets = input.matches('[').count();
        let close_brackets = input.matches(']').count();
        if open_brackets != close_brackets {
            errors.push(format!(
                "Mismatched brackets: {open_brackets} opening vs {close_brackets} closing"
            ));
        }

        // Check for missing semicolons in attribute sets
        if input.contains('{') && input.contains('=') && !input.contains(';') {
            warnings.push("Nix attribute sets require semicolons after each binding".to_string());
            suggestions.push("Add ';' after each attribute binding".to_string());
        }

        if errors.is_empty() {
            ValidationResult::valid()
        } else {
            let mut result = ValidationResult::invalid(errors);
            result.warnings = warnings;
            result.suggestions = suggestions;
            result
        }
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        let lower = topic.to_lowercase();
        let mut score: f64 = 0.0;
        let mut matches = 0;

        for keyword in NIXOS_KEYWORDS {
            if lower.contains(keyword) {
                matches += 1;
                // Strong indicators get more weight
                if [
                    "nixos",
                    "nix-env",
                    "nixos-rebuild",
                    "flake.nix",
                    "configuration.nix",
                    "derivation",
                    "nixpkgs",
                    "home-manager",
                ]
                .contains(keyword)
                {
                    score += 0.4;
                } else {
                    score += 0.15;
                }
            }
        }

        if matches > 0 {
            // At least one match: minimum 0.6
            (0.6 + score).min(1.0)
        } else {
            0.1
        }
    }

    fn vocabulary(&self) -> Vec<String> {
        NIXOS_KEYWORDS.iter().map(|s| s.to_string()).collect()
    }

    fn preprocess(&self, input: &str) -> String {
        // Normalize common NixOS abbreviations
        input
            .replace("nixos rebuild", "nixos-rebuild")
            .replace("nix env", "nix-env")
            .replace("nix build", "nix-build")
            .replace("nix shell", "nix-shell")
    }

    fn suggest_actions(&self, context: &str) -> Vec<String> {
        let lower = context.to_lowercase();
        let mut actions = Vec::new();

        if lower.contains("install") {
            actions.push("nix-env -iA nixpkgs.<package>".to_string());
            actions.push("Add to environment.systemPackages in configuration.nix".to_string());
        }
        if lower.contains("error") || lower.contains("failed") {
            actions.push("nixos-rebuild dry-run".to_string());
            actions.push("nix-store --verify --check-contents".to_string());
        }
        if lower.contains("update") || lower.contains("upgrade") {
            actions.push("nix-channel --update".to_string());
            actions.push("nixos-rebuild switch --upgrade".to_string());
        }
        if lower.contains("space") || lower.contains("disk") || lower.contains("garbage") {
            actions.push("nix-collect-garbage -d".to_string());
            actions.push("nix-store --optimise".to_string());
        }

        actions
    }
}

/// Extract file location from NixOS error text
fn extract_nix_location(error_text: &str) -> Option<ErrorLocation> {
    // Look for patterns like "at /path/to/file.nix:123:45" or "in file.nix, line 42"
    let line_iter = error_text.lines();
    for line in line_iter {
        // Pattern: "at /path:line:col"
        if let Some(at_pos) = line.find("at ") {
            let rest = &line[at_pos + 3..];
            let parts: Vec<&str> = rest.splitn(4, ':').collect();
            if parts.len() >= 3 {
                return Some(ErrorLocation {
                    file: Some(parts[0].trim().to_string()),
                    line: parts[1].trim().parse().ok(),
                    column: parts[2].trim().parse().ok(),
                    context: Some(line.to_string()),
                });
            }
        }
    }
    None
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_name() {
        let plugin = NixOsPlugin;
        assert_eq!(plugin.domain_name(), "nixos");
    }

    #[test]
    fn test_entity_extraction() {
        let plugin = NixOsPlugin;

        let entities = plugin.extract_entities("I want to install pkgs.firefox on my system");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == "package" && e.value == "firefox")
        );

        let entities = plugin.extract_entities("run nixos-rebuild switch");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == "nix_command" && e.value == "nixos-rebuild")
        );
    }

    #[test]
    fn test_domain_detection() {
        let plugin = NixOsPlugin;

        assert!(plugin.is_in_domain("How do I configure nixos-rebuild?") > 0.6);
        assert!(plugin.is_in_domain("Edit my flake.nix to add a package") > 0.6);
        assert!(plugin.is_in_domain("What is the weather today?") < 0.3);
    }

    #[test]
    fn test_error_diagnosis() {
        let plugin = NixOsPlugin;

        let diag = plugin
            .diagnose_error("error: undefined variable 'foo' at /etc/nixos/configuration.nix:42:5");
        assert!(diag.is_some());
        let diag = diag.unwrap();
        assert_eq!(diag.category, "nixos");
        assert_eq!(diag.error_type, "undefined_variable");
    }

    #[test]
    fn test_validation() {
        let plugin = NixOsPlugin;

        let result = plugin.validate_input("{ foo = 1; bar = 2; }");
        assert!(result.valid);

        let result = plugin.validate_input("{ foo = 1; bar = 2");
        assert!(!result.valid);
    }

    #[test]
    fn test_natural_language_entity_extraction() {
        let plugin = NixOsPlugin;

        // Natural language query that previously returned 0 entities
        let entities = plugin.extract_entities("How do I enable nginx in NixOS?");
        assert!(
            entities.iter().any(|e| e.value == "nginx"),
            "Should extract 'nginx' as entity. Got: {:?}",
            entities,
        );

        let entities = plugin.extract_entities("How to configure postgresql?");
        assert!(
            entities.iter().any(|e| e.value == "postgresql"),
            "Should extract 'postgresql' as entity. Got: {:?}",
            entities,
        );

        let entities = plugin.extract_entities("Enable the firewall in my NixOS config");
        assert!(
            entities.iter().any(|e| e.value == "firewall"),
            "Should extract 'firewall' as entity. Got: {:?}",
            entities,
        );
    }

    #[test]
    fn test_nix_option_extraction() {
        let plugin = NixOsPlugin;

        let entities = plugin.extract_entities("Set services.nginx.enable = true");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == "nix_option" && e.value.contains("services.nginx")),
            "Should extract NixOS option path. Got: {:?}",
            entities,
        );
    }

    #[test]
    fn test_flake_reference_extraction() {
        let plugin = NixOsPlugin;

        let entities = plugin.extract_entities("Add github:NixOS/nixpkgs as input");
        assert!(
            entities.iter().any(|e| e.entity_type == "flake_ref"),
            "Should extract github: flake ref. Got: {:?}",
            entities,
        );
    }

    #[test]
    fn test_intent_prototypes() {
        let plugin = NixOsPlugin;
        let protos = plugin.intent_prototypes();

        assert!(protos.command.contains(&"install".to_string()));
        assert!(protos.command.contains(&"rebuild".to_string()));
        assert!(protos.command.contains(&"rollback".to_string()));
        assert!(!protos.question.is_empty());
        assert!(!protos.complaint.is_empty());
        assert!(protos.custom.contains_key("configuration"));
        assert!(protos.custom.contains_key("deployment"));
    }

    #[test]
    fn test_prompts_non_empty() {
        let plugin = NixOsPlugin;
        let prompts = plugin.prompts();

        assert!(!prompts.system.is_empty());
        assert!(prompts.clarification.contains("{}"));
        assert!(prompts.action_confirm.contains("{}"));
        assert!(!prompts.error_explain.is_empty());
        assert!(!prompts.out_of_domain.is_empty());
    }

    #[test]
    fn test_diagnose_all_error_patterns() {
        let plugin = NixOsPlugin;

        // Note: patterns use literal `contains()` matching, not regex.
        // "attribute .* missing" means the text must literally contain "attribute .* missing"
        let errors = [
            ("syntax error: unexpected token", "syntax"),
            ("undefined variable 'pkgs'", "undefined_variable"),
            (
                "attribute .* missing in the derivation",
                "missing_attribute",
            ),
            ("infinite recursion encountered", "infinite_recursion"),
            ("hash mismatch in fixed-output derivation", "hash_mismatch"),
            ("builder .* failed to produce output", "build_failure"),
            ("permission denied", "permission"),
            (
                "collision between /nix/store/a and /nix/store/b",
                "collision",
            ),
        ];

        for (error_text, expected_type) in errors {
            let diag = plugin.diagnose_error(error_text);
            assert!(diag.is_some(), "Should diagnose: {}", error_text);
            assert_eq!(
                diag.unwrap().error_type,
                expected_type,
                "Wrong error type for: {}",
                error_text
            );
        }
    }

    #[test]
    fn test_diagnose_unknown_error() {
        let plugin = NixOsPlugin;
        let diag = plugin.diagnose_error("something completely unrelated went wrong");
        assert!(diag.is_none(), "Should not diagnose unknown errors");
    }

    #[test]
    fn test_validation_bracket_mismatch() {
        let plugin = NixOsPlugin;

        let result = plugin.validate_input("[ 1 2 3");
        assert!(!result.valid, "Mismatched brackets should be invalid");

        let result = plugin.validate_input("[ 1 2 3 ]");
        assert!(result.valid);
    }

    #[test]
    fn test_domain_detection_boundary() {
        let plugin = NixOsPlugin;

        // NixOS-specific terms should score high
        assert!(plugin.is_in_domain("I want to install a derivation") > 0.5);
        assert!(plugin.is_in_domain("My flake.nix needs an overlay") > 0.5);

        // Non-NixOS should score low
        assert!(plugin.is_in_domain("How do I cook pasta?") < 0.3);
        assert!(plugin.is_in_domain("What is 2 + 2?") < 0.3);
    }

    #[test]
    fn test_suggest_actions_install() {
        let plugin = NixOsPlugin;
        let actions = plugin.suggest_actions("I want to install firefox");
        assert!(!actions.is_empty());
        assert!(actions.iter().any(|a| a.contains("nix-env")));
    }

    #[test]
    fn test_suggest_actions_garbage() {
        let plugin = NixOsPlugin;
        let actions = plugin.suggest_actions("running out of disk space, need garbage collection");
        assert!(actions.iter().any(|a| a.contains("nix-collect-garbage")));
        assert!(actions.iter().any(|a| a.contains("optimise")));
    }

    #[test]
    fn test_suggest_actions_no_match() {
        let plugin = NixOsPlugin;
        let actions = plugin.suggest_actions("hello world");
        assert!(actions.is_empty());
    }

    #[test]
    fn test_preprocess_normalizes_commands() {
        let plugin = NixOsPlugin;
        assert_eq!(
            plugin.preprocess("nixos rebuild switch"),
            "nixos-rebuild switch"
        );
        assert_eq!(plugin.preprocess("nix env -i vim"), "nix-env -i vim");
    }

    #[test]
    fn test_multiple_entity_types() {
        let plugin = NixOsPlugin;

        // Text with multiple entity types
        let entities = plugin.extract_entities(
            "Run nixos-rebuild switch to apply services.nginx.enable from pkgs.nginx",
        );

        let types: Vec<&str> = entities.iter().map(|e| e.entity_type.as_str()).collect();
        assert!(
            types.contains(&"nix_command"),
            "Should find nix_command. Types: {:?}",
            types
        );
        assert!(
            types.contains(&"package"),
            "Should find package. Types: {:?}",
            types
        );
        assert!(
            types.contains(&"nix_option"),
            "Should find nix_option. Types: {:?}",
            types
        );
    }

    #[test]
    fn test_file_path_extraction() {
        let plugin = NixOsPlugin;

        let entities =
            plugin.extract_entities("Edit /etc/nixos/configuration.nix to fix the error");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == "file" && e.value.contains("configuration.nix")),
            "Should extract file path. Got: {:?}",
            entities,
        );
    }
}
