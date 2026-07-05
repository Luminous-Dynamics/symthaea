// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Command-line interface definition.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// Standalone LLM-agent-driven architecture/trust auditor.
#[derive(Parser, Debug)]
#[command(name = "symthaea-audit", version, about)]
pub struct Cli {
    /// Path to the repository (or directory) to audit.
    pub target: PathBuf,

    /// Scope the audit to a named subsystem, e.g. "the Phi subsystem in src/cognitive_loop/".
    /// When omitted, the whole target is in scope.
    #[arg(long)]
    pub focus: Option<String>,

    /// Where to write the report. Defaults to a timestamped file in the current
    /// directory — never inside the audited repo, so repeated runs never mutate the
    /// target and never silently overwrite a prior report.
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Which LLM backend to use. `auto` detects from environment variables.
    #[arg(long, value_enum, default_value_t = LlmProviderArg::Auto)]
    pub llm_provider: LlmProviderArg,

    /// Override the model name for the selected backend.
    #[arg(long)]
    pub model: Option<String>,

    /// Base URL override for the Ollama backend (default: http://localhost:11434).
    #[arg(long)]
    pub ollama_base_url: Option<String>,

    /// Base URL override for the OpenAI-compatible backend (default: https://api.openai.com/v1).
    #[arg(long)]
    pub openai_base_url: Option<String>,

    /// Maximum number of agent turns before the run is stopped and the last partial
    /// response is returned (with a truncation banner).
    #[arg(long, default_value_t = 40)]
    pub max_turns: usize,

    /// Comma-separated whitelist of exact commands the agent may execute via the
    /// `run_check` tool, e.g. "cargo check,pytest --collect-only". Empty (default)
    /// disables `run_check` entirely — it is not even advertised to the model.
    #[arg(long)]
    pub allow_exec: Option<String>,

    /// Comma-separated repo-relative paths (files or directories) to read wholesale
    /// and hand to the model in its first turn, instead of making it plan a sequence
    /// of read_file/list_dir calls to discover the same content. The model keeps full
    /// tool access afterward. Aimed at smaller/local models, which are typically much
    /// weaker at multi-turn exploration planning than at reading a lot of text at once.
    #[arg(long)]
    pub single_shot_paths: Option<String>,

    /// Run a second pass after the main audit that re-checks each cited claim against
    /// the actual repository content, appending a "Verification Notes" section rather
    /// than replacing the original report.
    #[arg(long, default_value_t = false)]
    pub verify: bool,

    /// Disable the deterministic (non-LLM) pre-scan that flags functions called with a
    /// literal `None` argument in one place and a non-`None` argument at the same
    /// position elsewhere — on by default since it costs nothing and directly targets
    /// this tool's highest-value finding shape.
    #[arg(long, default_value_t = false)]
    pub no_hints: bool,

    /// Review a diff instead of auditing the whole repo — a git ref range, e.g.
    /// "main..HEAD" or "HEAD~3..HEAD". Switches to the five-section diff-review report
    /// (SCOPE / SAFETY-CRITICAL SURFACE / TEST COVERAGE / DOCS VS. THE NEW REALITY /
    /// RELEASE GATE) instead of the six-section whole-repo audit. The model still has
    /// full read-only tool access to the target's current state, not just the diff
    /// text. `--focus` becomes framing for the change's intended purpose in this mode.
    #[arg(long)]
    pub review_diff: Option<String>,
}

impl Cli {
    /// Parsed, trimmed, non-empty entries of `--allow-exec`. Empty when unset.
    pub fn allow_exec_list(&self) -> Vec<String> {
        split_comma_list(self.allow_exec.as_deref())
    }

    /// Parsed, trimmed, non-empty entries of `--single-shot-paths`. Empty when unset.
    pub fn single_shot_paths_list(&self) -> Vec<String> {
        split_comma_list(self.single_shot_paths.as_deref())
    }
}

fn split_comma_list(raw: Option<&str>) -> Vec<String> {
    raw.unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum LlmProviderArg {
    Auto,
    Ollama,
    Anthropic,
    Openai,
}

impl std::fmt::Display for LlmProviderArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            LlmProviderArg::Auto => "auto",
            LlmProviderArg::Ollama => "ollama",
            LlmProviderArg::Anthropic => "anthropic",
            LlmProviderArg::Openai => "openai",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_exec_list_splits_and_trims() {
        let cli = Cli {
            target: PathBuf::from("."),
            focus: None,
            out: None,
            llm_provider: LlmProviderArg::Auto,
            model: None,
            ollama_base_url: None,
            openai_base_url: None,
            max_turns: 40,
            allow_exec: Some(" cargo check , pytest --collect-only ,, ".to_string()),
            single_shot_paths: None,
            verify: false,
            no_hints: false,
            review_diff: None,
        };
        assert_eq!(
            cli.allow_exec_list(),
            vec![
                "cargo check".to_string(),
                "pytest --collect-only".to_string()
            ]
        );
    }

    #[test]
    fn allow_exec_list_empty_when_unset() {
        let cli = Cli {
            target: PathBuf::from("."),
            focus: None,
            out: None,
            llm_provider: LlmProviderArg::Auto,
            model: None,
            ollama_base_url: None,
            openai_base_url: None,
            max_turns: 40,
            allow_exec: None,
            single_shot_paths: None,
            verify: false,
            no_hints: false,
            review_diff: None,
        };
        assert!(cli.allow_exec_list().is_empty());
    }
}
