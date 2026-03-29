// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shell Completion Generation
//!
//! Generates shell completions for bash, zsh, fish, elvish, and PowerShell.

use clap::CommandFactory;
use clap_complete::Shell;

use super::commands::Cli;

/// Generate shell completions and write them to stdout.
pub fn generate_completions(shell: Shell) {
    let mut cmd = Cli::command();
    clap_complete::generate(shell, &mut cmd, "nix-mind", &mut std::io::stdout());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bash_completions_generate() {
        // Just verify it doesn't panic
        let mut cmd = Cli::command();
        let mut buf = Vec::new();
        clap_complete::generate(Shell::Bash, &mut cmd, "nix-mind", &mut buf);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_zsh_completions_generate() {
        let mut cmd = Cli::command();
        let mut buf = Vec::new();
        clap_complete::generate(Shell::Zsh, &mut cmd, "nix-mind", &mut buf);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_fish_completions_generate() {
        let mut cmd = Cli::command();
        let mut buf = Vec::new();
        clap_complete::generate(Shell::Fish, &mut cmd, "nix-mind", &mut buf);
        assert!(!buf.is_empty());
    }
}
