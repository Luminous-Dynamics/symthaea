// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The validated six-section audit report schema, embedded at compile time.

/// Baked into the binary via `include_str!` — no runtime file dependency.
const TEMPLATE: &str = include_str!("../templates/audit_system_prompt.md");

const RUN_CHECK_DOC: &str = "- `run_check` — `{\"type\": \"run_check\", \"cmd\": \"...\"}` — run a \
    whitelisted read-only command (e.g. a compile check). Only the exact commands the operator \
    enabled are permitted; anything else is refused.";

/// Builds the system prompt. `focus` scopes the audit to a named subsystem when set.
/// `run_check_enabled` controls whether `run_check` is documented to the model at all —
/// when disabled, the model isn't even told the tool exists.
pub fn build_system_prompt(focus: Option<&str>, run_check_enabled: bool) -> String {
    let focus_clause = match focus {
        Some(f) => format!(
            "Scope this audit specifically to: {f}. Note in each section if you found \
             something relevant outside that scope, but keep the primary content focused there."
        ),
        None => "No specific focus was given — audit the whole repository.".to_string(),
    };
    let run_check_doc = if run_check_enabled { RUN_CHECK_DOC } else { "" };

    TEMPLATE
        .replace("{focus_clause}", &focus_clause)
        .replace("{run_check_doc}", run_check_doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focus_is_interpolated() {
        let prompt = build_system_prompt(Some("the Phi subsystem"), false);
        assert!(prompt.contains("the Phi subsystem"));
    }

    #[test]
    fn run_check_hidden_when_disabled() {
        let prompt = build_system_prompt(None, false);
        assert!(!prompt.contains("run_check"));
    }

    #[test]
    fn run_check_documented_when_enabled() {
        let prompt = build_system_prompt(None, true);
        assert!(prompt.contains("run_check"));
    }

    #[test]
    fn six_sections_present() {
        let prompt = build_system_prompt(None, false);
        for section in [
            "WIRED",
            "CLAIMED BUT DARK",
            "SAFETY-CRITICAL",
            "UNTESTED",
            "SHOULD DELETE",
            "SHOULD GATE",
        ] {
            assert!(prompt.contains(section), "missing section: {section}");
        }
    }

    #[test]
    fn no_leftover_placeholders() {
        let prompt = build_system_prompt(Some("x"), true);
        assert!(!prompt.contains("{focus_clause}"));
        assert!(!prompt.contains("{run_check_doc}"));
    }
}
