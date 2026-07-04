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
}

impl Cli {
    /// Parsed, trimmed, non-empty entries of `--allow-exec`. Empty when unset.
    pub fn allow_exec_list(&self) -> Vec<String> {
        self.allow_exec
            .as_deref()
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    }
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
        };
        assert!(cli.allow_exec_list().is_empty());
    }
}
