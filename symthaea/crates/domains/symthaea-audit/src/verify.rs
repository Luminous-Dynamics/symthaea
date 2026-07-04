// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Self-verification second pass: re-runs the agent loop with a draft report seeded
//! as context and a system prompt asking it to check its own citations, rather than
//! trusting the first pass's claims at face value.
//!
//! This does not require a stronger model to be useful — even a weak model re-reading
//! one specific cited line and confirming or denying a narrow claim is a much easier
//! task than producing the original analysis, so it can catch shallow claims that
//! slipped through the first pass.

use anyhow::Result;

use crate::agent_loop::{self, RunOutcome};
use crate::llm_client::LlmClient;
use crate::report_prompt::build_verification_prompt;
use crate::tools::Sandbox;

/// Runs the verification pass over `draft_report` and returns its own [`RunOutcome`].
/// Callers should append (never overwrite) this to the original report — a failed or
/// truncated verification pass should never cause the original findings to be lost.
pub fn run_verification_pass(
    client: &dyn LlmClient,
    sandbox: &Sandbox,
    draft_report: &str,
    max_turns: usize,
) -> Result<RunOutcome> {
    let run_check_enabled = !sandbox.allow_exec().is_empty();
    let system_prompt = build_verification_prompt(run_check_enabled);
    let initial_context = format!("## Draft report to verify\n\n{draft_report}");
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

    #[test]
    fn draft_report_is_seeded_into_first_turn() {
        let tmp = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::new(tmp.path(), vec![]).unwrap();
        let client = RecordingClient {
            responses: std::cell::RefCell::new(vec!["done <!-- AUDIT COMPLETE -->".to_string()]),
            seen_first_turn: std::cell::RefCell::new(None),
        };
        let draft = "### WIRED\n\nSome claim at foo.rs:10";
        run_verification_pass(&client, &sandbox, draft, 5).unwrap();
        let seen = client.seen_first_turn.borrow().clone().unwrap();
        assert!(seen.contains("Some claim at foo.rs:10"));
    }

    #[test]
    fn verification_outcome_is_returned_not_merged_silently() {
        let tmp = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::new(tmp.path(), vec![]).unwrap();
        let client = RecordingClient {
            responses: std::cell::RefCell::new(vec![
                "## Verification Notes\nVERIFIED — x — checked <!-- AUDIT COMPLETE -->".to_string(),
            ]),
            seen_first_turn: std::cell::RefCell::new(None),
        };
        let outcome = run_verification_pass(&client, &sandbox, "draft", 5).unwrap();
        assert!(outcome.report.contains("Verification Notes"));
    }
}
