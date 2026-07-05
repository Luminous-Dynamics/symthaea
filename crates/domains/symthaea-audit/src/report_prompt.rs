// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The validated six-section audit report schema, embedded at compile time.

/// Baked into the binary via `include_str!` — no runtime file dependency.
const TEMPLATE: &str = include_str!("../templates/audit_system_prompt.md");
const VERIFY_TEMPLATE: &str = include_str!("../templates/verify_system_prompt.md");
const DIFF_REVIEW_TEMPLATE: &str = include_str!("../templates/diff_review_prompt.md");

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

/// Builds the verification-pass system prompt. Unlike [`build_system_prompt`], the
/// draft report to verify isn't interpolated here — it's seeded into the initial user
/// turn by the caller, so this prompt stays generic and reusable.
pub fn build_verification_prompt(run_check_enabled: bool) -> String {
    let run_check_doc = if run_check_enabled { RUN_CHECK_DOC } else { "" };
    VERIFY_TEMPLATE.replace("{run_check_doc}", run_check_doc)
}

/// Builds the diff-review system prompt. Like [`build_system_prompt`], `focus` adds an
/// optional framing clause; unlike it, the diff content itself is seeded into the
/// initial user turn by the caller (same pattern as [`build_verification_prompt`]'s
/// draft report), so this prompt stays generic and reusable across different diffs.
pub fn build_diff_review_prompt(focus: Option<&str>, run_check_enabled: bool) -> String {
    let focus_clause = match focus {
        Some(f) => format!(
            "Additional context on what this change is supposed to do: {f}. Use this to \
             judge SCOPE — does the diff actually match this stated purpose?"
        ),
        None => "No additional context on the change's intended purpose was given — judge \
                 SCOPE from the diff and commit history alone."
            .to_string(),
    };
    let run_check_doc = if run_check_enabled { RUN_CHECK_DOC } else { "" };
    DIFF_REVIEW_TEMPLATE
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

    #[test]
    fn verification_prompt_has_verdict_vocabulary_and_no_placeholders() {
        let prompt = build_verification_prompt(false);
        for word in ["VERIFIED", "UNVERIFIED", "CONTRADICTED"] {
            assert!(prompt.contains(word), "missing verdict word: {word}");
        }
        assert!(!prompt.contains("{run_check_doc}"));
        assert!(!prompt.contains("run_check"));
    }

    #[test]
    fn verification_prompt_documents_run_check_when_enabled() {
        let prompt = build_verification_prompt(true);
        assert!(prompt.contains("run_check"));
    }

    #[test]
    fn diff_review_prompt_has_five_sections_and_no_placeholders() {
        let prompt = build_diff_review_prompt(None, false);
        for section in [
            "SCOPE",
            "SAFETY-CRITICAL SURFACE",
            "TEST COVERAGE OF THE CHANGE",
            "DOCS VS. THE NEW REALITY",
            "RELEASE GATE",
        ] {
            assert!(prompt.contains(section), "missing section: {section}");
        }
        assert!(!prompt.contains("{focus_clause}"));
        assert!(!prompt.contains("{run_check_doc}"));
    }

    #[test]
    fn diff_review_prompt_interpolates_focus() {
        let prompt = build_diff_review_prompt(Some("fixes the login bug"), false);
        assert!(prompt.contains("fixes the login bug"));
    }

    #[test]
    fn diff_review_prompt_hides_run_check_when_disabled() {
        let prompt = build_diff_review_prompt(None, false);
        assert!(!prompt.contains("run_check"));
    }
}
