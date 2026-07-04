// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Turn-taking driver: LLM emits fenced tool-call JSON, we execute against the
//! [`Sandbox`], results feed back as the next turn.
//!
//! The fenced-block convention (` ```tool ` … ` ``` `, one JSON object per line) is
//! ported from `symthaea/src/bin/symthaea-repl.rs:929-1031`. It's protocol-agnostic —
//! plain text generation works identically over Ollama, Anthropic, or OpenAI — which is
//! exactly why it was chosen over each provider's native structured tool-calling API.

use anyhow::Result;

use crate::llm_client::{LlmClient, Role, Turn};
use crate::tools::Sandbox;

const END_OF_REPORT_MARKER: &str = "<!-- AUDIT COMPLETE -->";
/// Repeating the identical tool call this many times in a row aborts the run early
/// with a diagnostic, rather than burning the full turn budget on a stuck loop.
const STUCK_LOOP_THRESHOLD: usize = 3;
/// Only the most recent turns are sent on each call, so a long-running audit can't
/// silently exceed the model's context window purely from accumulated tool results.
const MAX_HISTORY_TURNS: usize = 24;

#[derive(Debug, Clone, PartialEq)]
pub enum ActionRequest {
    ReadFile {
        path: String,
    },
    ListDir {
        path: String,
        recursive: bool,
    },
    GrepRepo {
        pattern: String,
        glob: Option<String>,
    },
    GitLog {
        path: Option<String>,
        limit: u32,
    },
    GitStatus,
    GitDiff {
        path: Option<String>,
    },
    LocCount {
        path: Option<String>,
    },
    RunCheck {
        cmd: String,
    },
}

/// Parses one JSON object per line inside ` ```tool ` fenced blocks. Malformed lines
/// and unrecognized types are skipped with a stderr warning, never a hard error — a
/// misbehaving model shouldn't be able to crash the audit run.
pub fn parse_tool_calls(text: &str) -> Vec<ActionRequest> {
    let mut actions = Vec::new();
    let mut in_block = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```tool") {
            in_block = true;
            continue;
        }
        if in_block && trimmed.starts_with("```") {
            in_block = false;
            continue;
        }
        if !in_block || trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[TOOL WARN] skipping malformed tool line: {trimmed} — {e}");
                continue;
            }
        };
        let ty = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let str_field = |key: &str| value.get(key).and_then(|v| v.as_str()).map(str::to_string);
        match ty {
            "read_file" => {
                if let Some(path) = str_field("path") {
                    actions.push(ActionRequest::ReadFile { path });
                }
            }
            "list_dir" => {
                if let Some(path) = str_field("path") {
                    let recursive = value
                        .get("recursive")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    actions.push(ActionRequest::ListDir { path, recursive });
                }
            }
            "grep_repo" => {
                if let Some(pattern) = str_field("pattern") {
                    actions.push(ActionRequest::GrepRepo {
                        pattern,
                        glob: str_field("glob"),
                    });
                }
            }
            "git_log" => {
                let limit = value.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as u32;
                actions.push(ActionRequest::GitLog {
                    path: str_field("path"),
                    limit,
                });
            }
            "git_status" => actions.push(ActionRequest::GitStatus),
            "git_diff" => actions.push(ActionRequest::GitDiff {
                path: str_field("path"),
            }),
            "loc_count" => actions.push(ActionRequest::LocCount {
                path: str_field("path"),
            }),
            "run_check" => {
                if let Some(cmd) = str_field("cmd") {
                    actions.push(ActionRequest::RunCheck { cmd });
                }
            }
            other => {
                eprintln!("[TOOL WARN] unsupported tool type '{other}'");
            }
        }
    }
    actions
}

/// Executes one action against the sandbox and formats a `[TOOL RESULT]` string.
/// `run_check` is refused here — not just at the `Sandbox` whitelist layer — whenever
/// the sandbox's `allow_exec` is empty, so a process is never spawned even if a
/// prompt-injected instruction in the audited repo's own content tries to invoke it.
fn execute(sandbox: &Sandbox, action: &ActionRequest) -> String {
    match action {
        ActionRequest::ReadFile { path } => match sandbox.read_file(path) {
            Ok(preview) => {
                let truncation = if preview.truncated {
                    format!(" (truncated; {}B total)", preview.total_bytes)
                } else {
                    String::new()
                };
                format!(
                    "[TOOL RESULT] read_file {path:?}{truncation}:\n{}",
                    preview.content
                )
            }
            Err(e) => format!("[TOOL RESULT] read_file {path:?} failed: {e}"),
        },
        ActionRequest::ListDir { path, recursive } => match sandbox.list_dir(path, *recursive) {
            Ok(listing) => {
                let mut body = String::new();
                for entry in &listing.entries {
                    body.push_str(&format!("  {entry}\n"));
                }
                if listing.total_entries > listing.entries.len() {
                    body.push_str(&format!(
                        "  ... ({} more)\n",
                        listing.total_entries - listing.entries.len()
                    ));
                }
                format!(
                    "[TOOL RESULT] list_dir {path:?} ({} entries):\n{body}",
                    listing.total_entries
                )
            }
            Err(e) => format!("[TOOL RESULT] list_dir {path:?} failed: {e}"),
        },
        ActionRequest::GrepRepo { pattern, glob } => {
            match sandbox.grep_repo(pattern, glob.as_deref()) {
                Ok(out) => format!("[TOOL RESULT] grep_repo {pattern:?}:\n{out}"),
                Err(e) => format!("[TOOL RESULT] grep_repo {pattern:?} failed: {e}"),
            }
        }
        ActionRequest::GitLog { path, limit } => match sandbox.git_log(path.as_deref(), *limit) {
            Ok(out) => format!("[TOOL RESULT] git_log:\n{out}"),
            Err(e) => format!("[TOOL RESULT] git_log failed: {e}"),
        },
        ActionRequest::GitStatus => match sandbox.git_status() {
            Ok(out) => format!("[TOOL RESULT] git_status:\n{out}"),
            Err(e) => format!("[TOOL RESULT] git_status failed: {e}"),
        },
        ActionRequest::GitDiff { path } => match sandbox.git_diff(path.as_deref()) {
            Ok(out) => format!("[TOOL RESULT] git_diff:\n{out}"),
            Err(e) => format!("[TOOL RESULT] git_diff failed: {e}"),
        },
        ActionRequest::LocCount { path } => match sandbox.loc_count(path.as_deref()) {
            Ok(out) => format!("[TOOL RESULT] loc_count:\n{out}"),
            Err(e) => format!("[TOOL RESULT] loc_count failed: {e}"),
        },
        ActionRequest::RunCheck { cmd } => {
            if sandbox.allow_exec().is_empty() {
                format!(
                    "[TOOL WARN] run_check is disabled for this run (no --allow-exec whitelist); '{cmd}' was not executed"
                )
            } else {
                match sandbox.run_check(cmd) {
                    Ok(out) => format!("[TOOL RESULT] run_check {cmd:?}:\n{out}"),
                    Err(e) => format!("[TOOL RESULT] run_check {cmd:?} failed: {e}"),
                }
            }
        }
    }
}

pub struct RunOutcome {
    pub report: String,
    pub turns_used: usize,
    pub truncated_by_turn_budget: bool,
}

pub fn run(
    client: &dyn LlmClient,
    sandbox: &Sandbox,
    system_prompt: &str,
    max_turns: usize,
) -> Result<RunOutcome> {
    let mut history: Vec<Turn> = vec![Turn {
        role: Role::User,
        content: "Begin the audit.".to_string(),
    }];
    let mut last_action: Option<ActionRequest> = None;
    let mut repeat_count = 0usize;
    let mut last_response = String::new();

    for turn_index in 0..max_turns {
        let window_start = history.len().saturating_sub(MAX_HISTORY_TURNS);
        let response = client.complete(system_prompt, &history[window_start..])?;
        last_response = response.clone();
        history.push(Turn {
            role: Role::Assistant,
            content: response.clone(),
        });

        if response.contains(END_OF_REPORT_MARKER) {
            return Ok(RunOutcome {
                report: response,
                turns_used: turn_index + 1,
                truncated_by_turn_budget: false,
            });
        }

        let actions = parse_tool_calls(&response);
        if actions.is_empty() {
            // No tool calls and no end-of-report marker — treat the response itself as
            // the (possibly final) report rather than looping forever on nothing.
            return Ok(RunOutcome {
                report: response,
                turns_used: turn_index + 1,
                truncated_by_turn_budget: false,
            });
        }

        if let Some(first) = actions.first() {
            if last_action.as_ref() == Some(first) && actions.len() == 1 {
                repeat_count += 1;
            } else {
                repeat_count = 0;
            }
            last_action = Some(first.clone());
        }
        if repeat_count >= STUCK_LOOP_THRESHOLD {
            let banner = format!(
                "[AUDIT STOPPED EARLY] the same tool call repeated {STUCK_LOOP_THRESHOLD} times in a row; \
                 returning the last response as a partial report.\n\n{response}"
            );
            return Ok(RunOutcome {
                report: banner,
                turns_used: turn_index + 1,
                truncated_by_turn_budget: true,
            });
        }

        let mut results = String::new();
        for action in &actions {
            results.push_str(&execute(sandbox, action));
            results.push('\n');
        }
        history.push(Turn {
            role: Role::User,
            content: results,
        });
    }

    let banner = format!(
        "[AUDIT TRUNCATED] max-turns ({max_turns}) exhausted before an end-of-report marker was seen; \
         returning the last partial response below.\n\n{last_response}"
    );
    Ok(RunOutcome {
        report: banner,
        turns_used: max_turns,
        truncated_by_turn_budget: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_read_file() {
        let text = "```tool\n{\"type\": \"read_file\", \"path\": \"src/main.rs\"}\n```";
        let actions = parse_tool_calls(text);
        assert_eq!(
            actions,
            vec![ActionRequest::ReadFile {
                path: "src/main.rs".to_string()
            }]
        );
    }

    #[test]
    fn parses_list_dir_with_default_recursive() {
        let text = "```tool\n{\"type\": \"list_dir\", \"path\": \"src\"}\n```";
        let actions = parse_tool_calls(text);
        assert_eq!(
            actions,
            vec![ActionRequest::ListDir {
                path: "src".to_string(),
                recursive: false
            }]
        );
    }

    #[test]
    fn skips_malformed_json_without_panicking() {
        let text = "```tool\nnot json at all\n{\"type\": \"read_file\", \"path\": \"ok.rs\"}\n```";
        let actions = parse_tool_calls(text);
        assert_eq!(
            actions,
            vec![ActionRequest::ReadFile {
                path: "ok.rs".to_string()
            }]
        );
    }

    #[test]
    fn ignores_unfenced_json() {
        let text = "{\"type\": \"read_file\", \"path\": \"nope.rs\"}";
        let actions = parse_tool_calls(text);
        assert!(actions.is_empty());
    }

    #[test]
    fn parses_run_check() {
        let text = "```tool\n{\"type\": \"run_check\", \"cmd\": \"cargo check\"}\n```";
        let actions = parse_tool_calls(text);
        assert_eq!(
            actions,
            vec![ActionRequest::RunCheck {
                cmd: "cargo check".to_string()
            }]
        );
    }

    struct StubClient {
        responses: std::cell::RefCell<Vec<String>>,
    }

    impl LlmClient for StubClient {
        fn complete(&self, _system_prompt: &str, _history: &[Turn]) -> Result<String> {
            Ok(self.responses.borrow_mut().remove(0))
        }
        fn name(&self) -> &'static str {
            "stub"
        }
    }

    #[test]
    fn run_check_never_executes_without_allow_exec() {
        let tmp = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::new(tmp.path(), vec![]).unwrap();
        let client = StubClient {
            responses: std::cell::RefCell::new(vec![
                "```tool\n{\"type\": \"run_check\", \"cmd\": \"rm -rf /\"}\n```".to_string(),
                format!("done {END_OF_REPORT_MARKER}"),
            ]),
        };
        let outcome = run(&client, &sandbox, "system", 5).unwrap();
        assert!(outcome.report.contains(END_OF_REPORT_MARKER));
    }

    #[test]
    fn stuck_loop_stops_early() {
        let tmp = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::new(tmp.path(), vec![]).unwrap();
        let repeated = "```tool\n{\"type\": \"list_dir\", \"path\": \".\"}\n```".to_string();
        let client = StubClient {
            responses: std::cell::RefCell::new(vec![repeated.clone(); 10]),
        };
        let outcome = run(&client, &sandbox, "system", 40).unwrap();
        assert!(outcome.truncated_by_turn_budget);
        assert!(outcome.turns_used < 40);
    }

    #[test]
    fn max_turns_exhaustion_returns_partial_report_not_error() {
        let tmp = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::new(tmp.path(), vec![]).unwrap();
        let client = StubClient {
            responses: std::cell::RefCell::new(
                (0..3)
                    .map(|i| format!("```tool\n{{\"type\": \"loc_count\"}}\n```extra{i}"))
                    .collect(),
            ),
        };
        let outcome = run(&client, &sandbox, "system", 3).unwrap();
        assert!(outcome.truncated_by_turn_budget);
        assert!(outcome.report.contains("[AUDIT"));
    }
}
