// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diff-review mode: run the agent loop over a specific set of changes (a git ref
//! range) instead of the whole repo, using the five-section diff-review report schema.
//!
//! This is the "trusted repo companion" idea — the same auditor machinery, run
//! continuously on diffs instead of once on a whole repo.

use anyhow::{Context, Result};

use crate::agent_loop::{self, RunOutcome};
use crate::llm_client::LlmClient;
use crate::report_prompt::build_diff_review_prompt;
use crate::tools::Sandbox;

/// Runs a diff review over `range` (a git ref range like `main..HEAD`) and returns the
/// resulting [`RunOutcome`]. Errors if the range doesn't resolve to any diff content —
/// callers should treat that as a usage error (bad range, or genuinely no changes), not
/// silently produce an empty report.
pub fn run_diff_review(
    client: &dyn LlmClient,
    sandbox: &Sandbox,
    range: &str,
    focus: Option<&str>,
    max_turns: usize,
) -> Result<RunOutcome> {
    let run_check_enabled = !sandbox.allow_exec().is_empty();
    let system_prompt = build_diff_review_prompt(focus, run_check_enabled);

    let diff = sandbox
        .git_diff_range(range)
        .with_context(|| format!("failed to diff range {range:?}"))?;
    if diff.trim().is_empty() {
        anyhow::bail!(
            "git diff {range:?} produced no output — check the range is correct and actually contains changes"
        );
    }
    let name_status = sandbox.git_diff_name_status(range).unwrap_or_default();

    let mut initial_context = String::new();
    if !name_status.trim().is_empty() {
        initial_context.push_str("## Changed files\n\n");
        initial_context.push_str(&name_status);
        initial_context.push_str("\n\n");
    }
    initial_context.push_str(&format!("## Full diff ({range})\n\n```diff\n{diff}\n```\n"));

    agent_loop::run(
        client,
        sandbox,
        &system_prompt,
        max_turns,
        Some(&initial_context),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::Turn;
    use std::fs;
    use std::process::Command;

    struct RecordingClient {
        responses: std::cell::RefCell<Vec<String>>,
        seen_first_turn: std::cell::RefCell<Option<String>>,
    }

    impl LlmClient for RecordingClient {
        fn complete(&self, _system_prompt: &str, history: &[Turn]) -> Result<String> {
            if self.seen_first_turn.borrow().is_none() {
                *self.seen_first_turn.borrow_mut() = history.first().map(|t| t.content.clone());
            }
            Ok(self.responses.borrow_mut().remove(0))
        }
        fn name(&self) -> &'static str {
            "recording"
        }
    }

    fn build_git_repo_with_two_commits() -> (tempfile::TempDir, std::path::PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("repo");
        fs::create_dir_all(&root).unwrap();
        let git = |args: &[&str]| {
            let status = Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(args)
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@t.com")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@t.com")
                .status()
                .unwrap();
            assert!(status.success(), "git {args:?} failed");
        };
        git(&["init", "-q"]);
        fs::write(root.join("a.rs"), "fn a() {}\n").unwrap();
        git(&["add", "a.rs"]);
        git(&["commit", "-q", "-m", "first"]);
        fs::write(root.join("a.rs"), "fn a() { /* changed */ }\n").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "second"]);
        let root = root.canonicalize().unwrap();
        (tmp, root)
    }

    #[test]
    fn diff_content_is_seeded_into_first_turn() {
        let (_tmp, root) = build_git_repo_with_two_commits();
        let sandbox = Sandbox::new(&root, vec![]).unwrap();
        let client = RecordingClient {
            responses: std::cell::RefCell::new(vec!["done <!-- AUDIT COMPLETE -->".to_string()]),
            seen_first_turn: std::cell::RefCell::new(None),
        };
        run_diff_review(&client, &sandbox, "HEAD~1..HEAD", None, 5).unwrap();
        let seen = client.seen_first_turn.borrow().clone().unwrap();
        assert!(seen.contains("a.rs"));
        assert!(seen.contains("changed"));
    }

    #[test]
    fn empty_range_is_an_error_not_a_silent_empty_report() {
        let (_tmp, root) = build_git_repo_with_two_commits();
        let sandbox = Sandbox::new(&root, vec![]).unwrap();
        let client = RecordingClient {
            responses: std::cell::RefCell::new(vec!["should not be called".to_string()]),
            seen_first_turn: std::cell::RefCell::new(None),
        };
        // HEAD..HEAD is always empty.
        let result = run_diff_review(&client, &sandbox, "HEAD..HEAD", None, 5);
        assert!(result.is_err());
    }
}
