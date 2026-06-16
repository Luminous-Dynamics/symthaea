// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Safety Gateway - Unified Safety Interface

Provides a single facade over the different safety subsystems:
- AmygdalaActor: fast regex-based pre-cognitive veto
- SafetyGuardrails: hypervector-based forbidden subspace checking

This module does not execute actions itself. Instead it provides a
consistent, structured way for callers to ask:

- "Is this text safe?"
- "Is this command safe?"
- "Is this planned ActionIR safe?"
*/

use std::path::Path;

use super::{AmygdalaActor, ForbiddenCategory, SafetyGuardrails};
use crate::action::ActionIR;

/// What kind of thing is being checked for safety
#[derive(Debug)]
pub enum SafetyCheck<'a> {
    /// Natural language text (queries, prompts)
    Query(&'a str),
    /// Concrete shell / Nix command string
    Command(&'a str),
    /// Planned side-effect encoded as ActionIR
    Action(&'a ActionIR),
}

/// Structured safety decision
#[derive(Debug, Clone)]
pub struct SafetyDecision {
    /// Whether the input is allowed to proceed
    pub allowed: bool,
    /// Human-readable explanation (if any)
    pub message: Option<String>,
    /// High-level forbidden category (if matched)
    pub category: Option<ForbiddenCategory>,
}

impl SafetyDecision {
    /// Convenience helper: allowed without message or category
    pub fn allowed() -> Self {
        Self {
            allowed: true,
            message: None,
            category: None,
        }
    }

    /// Convenience helper: blocked with message and optional category
    pub fn blocked(message: String, category: Option<ForbiddenCategory>) -> Self {
        Self {
            allowed: false,
            message: Some(message),
            category,
        }
    }
}

/// Paths that should never be written to or deleted
const FORBIDDEN_PATHS: &[&str] = &[
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
];

/// Programs that should never be executed without careful review
const DANGEROUS_PROGRAMS: &[&str] = &["rm", "dd", "mkfs", "fdisk", "parted", "shred", "wipefs"];

/// Check if a path is within a forbidden system path
fn is_forbidden_path(path: &Path) -> bool {
    let normalized = normalize_path(path);
    let path_str = normalized.to_string_lossy();
    FORBIDDEN_PATHS
        .iter()
        .any(|forbidden| path_str.starts_with(forbidden))
}

fn normalize_path(path: &Path) -> std::path::PathBuf {
    use std::path::Component;
    let mut normalized = std::path::PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

/// Check if a program name is dangerous
fn is_dangerous_program(program: &str) -> bool {
    // Extract basename (e.g., "/usr/bin/rm" -> "rm")
    let basename = Path::new(program)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(program);

    DANGEROUS_PROGRAMS.contains(&basename)
}

/// SafetyGateway ties together fast-path regex safety and slower
/// hypervector-based guardrails into a single interface.
#[derive(Debug)]
pub struct SafetyGateway {
    amygdala: AmygdalaActor,
    guardrails: SafetyGuardrails,
}

impl SafetyGateway {
    /// Create a new gateway with default Amygdala and guardrail configuration
    pub fn new() -> Self {
        Self {
            amygdala: AmygdalaActor::new(),
            guardrails: SafetyGuardrails::new(),
        }
    }

    /// Check any supported input for safety
    pub fn check(&mut self, what: SafetyCheck<'_>) -> SafetyDecision {
        match what {
            SafetyCheck::Query(text) => self.check_text(text),
            SafetyCheck::Command(cmd) => self.check_command(cmd),
            SafetyCheck::Action(action) => self.check_action(action),
        }
    }

    /// Check a pre-encoded hypervector against the guardrails directly
    pub fn check_hv(&self, hv: &[f32]) -> SafetyDecision {
        if let Some(category) = self.guardrails.check(hv) {
            return SafetyDecision::blocked(
                format!("Blocked: Content matches forbidden category {category:?}"),
                Some(category),
            );
        }
        SafetyDecision::allowed()
    }

    /// Safety check for natural language text
    fn check_text(&mut self, text: &str) -> SafetyDecision {
        // Layer 1: Amygdala fast-path veto (regex-based)
        if let Some(msg) = self.amygdala.scan(text) {
            return SafetyDecision::blocked(msg, Some(ForbiddenCategory::DangerousCommand));
        }

        // Layer 2: Encode text to HV and check against forbidden subspaces
        let hv = self.encode_text_to_hv(text);
        if let Some(category) = self.guardrails.check(&hv) {
            return SafetyDecision::blocked(
                format!("Blocked: Semantic content matches forbidden category {category:?}"),
                Some(category),
            );
        }

        SafetyDecision::allowed()
    }

    /// Safety check for raw commands (bash/nix)
    fn check_command(&mut self, command: &str) -> SafetyDecision {
        // Layer 1: Amygdala fast-path veto (regex-based)
        if let Some(msg) = self.amygdala.scan(command) {
            return SafetyDecision::blocked(msg, Some(ForbiddenCategory::DangerousCommand));
        }

        // Layer 2: Check against forbidden subspaces
        let hv = self.encode_text_to_hv(command);
        if let Some(category) = self.guardrails.check(&hv) {
            return SafetyDecision::blocked(
                format!("Blocked: Command matches forbidden category {category:?}"),
                Some(category),
            );
        }

        SafetyDecision::allowed()
    }

    /// Safety check for planned side-effects (ActionIR)
    fn check_action(&mut self, action: &ActionIR) -> SafetyDecision {
        match action {
            ActionIR::DeleteFile { path } => {
                if is_forbidden_path(path) {
                    return SafetyDecision::blocked(
                        format!("Blocked: Cannot delete system path '{}'", path.display()),
                        Some(ForbiddenCategory::DangerousCommand),
                    );
                }
            }
            ActionIR::WriteFile { path, .. } => {
                if is_forbidden_path(path) {
                    return SafetyDecision::blocked(
                        format!("Blocked: Cannot write to system path '{}'", path.display()),
                        Some(ForbiddenCategory::SecurityRisk),
                    );
                }
            }
            ActionIR::RunCommand { program, args, .. } => {
                // Check the program name
                if is_dangerous_program(program) {
                    return SafetyDecision::blocked(
                        format!(
                            "Blocked: Dangerous program '{program}' requires explicit approval"
                        ),
                        Some(ForbiddenCategory::DangerousCommand),
                    );
                }

                // Also run the full command string through amygdala
                let full_cmd = format!("{} {}", program, args.join(" "));
                if let Some(msg) = self.amygdala.scan(&full_cmd) {
                    return SafetyDecision::blocked(msg, Some(ForbiddenCategory::DangerousCommand));
                }
            }
            ActionIR::Sequence(actions) => {
                // Check each sub-action
                for sub_action in actions {
                    let decision = self.check_action(sub_action);
                    if !decision.allowed {
                        return decision;
                    }
                }
            }
            // ReadFile, CreateDirectory, ListDirectory, NoOp are safe
            _ => {}
        }

        SafetyDecision::allowed()
    }

    /// Simple text-to-hypervector encoding via word hashing and bundling.
    ///
    /// This provides a basic semantic encoding for the HDC guardrails layer.
    /// Each word is hashed to produce a deterministic seed, which generates
    /// a random HV. All word HVs are bundled (averaged) to produce the text HV.
    fn encode_text_to_hv(&self, text: &str) -> Vec<f32> {
        let dim = self.guardrails.dimension();
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.is_empty() {
            return vec![0.0; dim];
        }

        let mut result = vec![0.0f32; dim];

        for word in &words {
            let seed = Self::hash_word(word);
            let mut state = seed;

            for i in 0..dim {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let val = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
                result[i] += val;
            }
        }

        // Normalize (average)
        let n = words.len() as f32;
        for v in &mut result {
            *v /= n;
        }

        result
    }

    /// Hash a word to a u64 seed
    fn hash_word(word: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        word.to_lowercase().hash(&mut h);
        h.finish()
    }

    /// Get a reference to the guardrails (for testing or direct HV checks)
    pub fn guardrails(&self) -> &SafetyGuardrails {
        &self.guardrails
    }

    /// Get a mutable reference to the guardrails
    pub fn guardrails_mut(&mut self) -> &mut SafetyGuardrails {
        &mut self.guardrails
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gateway_blocks_dangerous_command() {
        let mut gw = SafetyGateway::new();
        let decision = gw.check(SafetyCheck::Command("rm -rf /"));
        assert!(!decision.allowed);
        assert!(decision.category.is_some());
    }

    #[test]
    fn test_gateway_allows_safe_query() {
        let mut gw = SafetyGateway::new();
        let decision = gw.check(SafetyCheck::Query("What is the weather today?"));
        assert!(decision.allowed);
    }

    #[test]
    fn test_gateway_blocks_dangerous_action_delete() {
        let mut gw = SafetyGateway::new();
        let action = ActionIR::DeleteFile {
            path: "/etc/passwd".into(),
        };
        let decision = gw.check(SafetyCheck::Action(&action));
        assert!(!decision.allowed);
        assert_eq!(decision.category, Some(ForbiddenCategory::DangerousCommand));
    }

    #[test]
    fn test_gateway_blocks_dangerous_action_write() {
        let mut gw = SafetyGateway::new();
        let action = ActionIR::WriteFile {
            path: "/etc/shadow".into(),
            content: vec![0u8; 10],
            create_dirs: false,
        };
        let decision = gw.check(SafetyCheck::Action(&action));
        assert!(!decision.allowed);
        assert_eq!(decision.category, Some(ForbiddenCategory::SecurityRisk));
    }

    #[test]
    fn test_gateway_blocks_dangerous_action_run() {
        let mut gw = SafetyGateway::new();
        let action = ActionIR::RunCommand {
            program: "rm".to_string(),
            args: vec!["-rf".to_string(), "/".to_string()],
            env: Default::default(),
            working_dir: None,
        };
        let decision = gw.check(SafetyCheck::Action(&action));
        assert!(!decision.allowed);
    }

    #[test]
    fn test_gateway_allows_safe_action() {
        let mut gw = SafetyGateway::new();
        let action = ActionIR::ReadFile {
            path: "/tmp/test.txt".into(),
            encoding: None,
        };
        let decision = gw.check(SafetyCheck::Action(&action));
        assert!(decision.allowed);
    }

    #[test]
    fn test_gateway_blocks_dangerous_sequence() {
        let mut gw = SafetyGateway::new();
        let action = ActionIR::Sequence(vec![
            ActionIR::ReadFile {
                path: "/tmp/safe.txt".into(),
                encoding: None,
            },
            ActionIR::DeleteFile {
                path: "/boot/vmlinuz".into(),
            },
        ]);
        let decision = gw.check(SafetyCheck::Action(&action));
        assert!(!decision.allowed);
    }

    #[test]
    fn test_gateway_allows_noop() {
        let mut gw = SafetyGateway::new();
        let decision = gw.check(SafetyCheck::Action(&ActionIR::NoOp));
        assert!(decision.allowed);
    }

    #[test]
    fn test_gateway_hv_check_blocks_prototype() {
        let gw = SafetyGateway::new();
        let proto = gw
            .guardrails()
            .prototype(ForbiddenCategory::DangerousCommand)
            .unwrap()
            .to_vec();
        let decision = gw.check_hv(&proto);
        assert!(!decision.allowed);
    }

    #[test]
    fn test_text_encoding_produces_correct_dimension() {
        let gw = SafetyGateway::new();
        let hv = gw.encode_text_to_hv("hello world test");
        assert_eq!(hv.len(), gw.guardrails().dimension());
    }

    #[test]
    fn test_text_encoding_empty_input() {
        let gw = SafetyGateway::new();
        let hv = gw.encode_text_to_hv("");
        assert_eq!(hv.len(), gw.guardrails().dimension());
        assert!(hv.iter().all(|&v| v == 0.0));
    }
}
