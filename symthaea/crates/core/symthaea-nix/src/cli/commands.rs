// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CLI Command Definitions and Dispatch
//!
//! Defines the top-level commands for the `nix-mind` CLI tool.
//! Natural language is the primary interface — subcommands exist
//! as convenience shortcuts for common goals.

use clap::{Parser, Subcommand};

/// nix-mind: A conscious NixOS management tool.
///
/// Natural language is the primary interface. Any argument without a
/// leading `--` is treated as a natural language request:
///
///   nix-mind "install firefox"
///   nix-mind "why did my rebuild fail?"
///   nix-mind "make my system faster"
///
/// Subcommands provide shortcuts for common operations.
#[derive(Parser, Debug)]
#[command(
    name = "nix-mind",
    version,
    about = "A conscious NixOS management tool powered by HDC and active inference",
    long_about = None,
)]
pub struct Cli {
    /// Natural language input (e.g. "install firefox", "why did my rebuild fail?").
    /// When provided, bypasses subcommands and routes through the cognitive core.
    #[arg(trailing_var_arg = true, allow_hyphen_values = false)]
    pub input: Vec<String>,

    /// Enable dry-run mode (no system changes, only show what would happen).
    #[arg(long, global = true)]
    pub dry_run: bool,

    /// Set verbosity level (0=quiet, 1=normal, 2=verbose, 3=debug).
    #[arg(short, long, global = true, default_value = "1")]
    pub verbose: u8,

    /// Output format.
    #[arg(long, global = true, default_value = "human")]
    pub format: OutputFormat,

    /// Override the consciousness level (Φ) for this session.
    /// Normally computed automatically.
    #[arg(long, global = true)]
    pub phi: Option<f64>,

    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Output format for CLI results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable output with consciousness metrics.
    Human,
    /// JSON output for scripting.
    Json,
    /// Minimal output (just the essential result).
    Minimal,
}

/// Subcommands for nix-mind.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Search packages and options in HDC space.
    Search {
        /// Search query.
        query: String,
        /// Search packages (default) or options.
        #[arg(long)]
        options: bool,
        /// Maximum results to show.
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },

    /// Rebuild the NixOS system.
    Rebuild {
        /// Rebuild mode.
        #[arg(value_enum, default_value = "switch")]
        mode: RebuildMode,
        /// Flake reference (e.g. ".#hostname").
        #[arg(long)]
        flake: Option<String>,
        /// Extra arguments to pass to nixos-rebuild.
        #[arg(last = true)]
        extra_args: Vec<String>,
    },

    /// Roll back to a previous generation.
    Rollback {
        /// Specific generation number (default: previous).
        generation: Option<u32>,
    },

    /// Observe the current system state.
    Observe {
        /// Show specific domain only.
        #[arg(value_enum)]
        domain: Option<ObserveDomain>,
    },

    /// Run system diagnostics.
    Doctor,

    /// List and manage NixOS generations.
    Generations {
        /// Show diff between generations.
        #[arg(long)]
        diff: bool,
        /// First generation for diff.
        #[arg(long)]
        from: Option<u32>,
        /// Second generation for diff.
        #[arg(long)]
        to: Option<u32>,
        /// Delete old generations.
        #[arg(long)]
        delete_older_than: Option<u32>,
    },

    /// Flake operations.
    Flake {
        /// Flake operation.
        #[command(subcommand)]
        op: FlakeCommand,
    },

    /// Intelligent garbage collection.
    Gc {
        /// Just analyze, don't actually collect.
        #[arg(long)]
        analyze: bool,
        /// Delete generations older than N days.
        #[arg(long)]
        older_than: Option<u32>,
        /// Aggressive mode: delete all old generations.
        #[arg(long)]
        aggressive: bool,
    },

    /// Manage systemd services.
    Service {
        /// Service operation.
        #[command(subcommand)]
        op: ServiceCommand,
    },

    /// Enter interactive conscious REPL.
    Repl,

    /// Generate shell completions.
    Completions {
        /// Shell to generate completions for.
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },

    /// Run health assessment on the system.
    Health {
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Predict future system state using LTC temporal model.
    Predict {
        /// Time horizons to predict (hours).
        #[arg(long, default_value = "1,6,24,168")]
        horizons: String,
    },

    /// Start post-rebuild watchdog monitor.
    Watch {
        /// Monitoring timeout in seconds.
        #[arg(long, default_value = "300")]
        timeout: u64,
        /// Check interval in seconds.
        #[arg(long, default_value = "10")]
        interval: u64,
    },

    /// Privacy-scrub a log file or stdin.
    Scrub {
        /// File to scrub (reads stdin if omitted).
        file: Option<String>,
    },

    /// Search NixOS knowledge base for solutions.
    Knowledge {
        /// Search query (error message or description).
        query: String,
        /// Maximum results to show.
        #[arg(short = 'n', long, default_value = "5")]
        limit: usize,
    },
}

/// Rebuild modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum RebuildMode {
    /// Apply and make default boot entry.
    Switch,
    /// Apply temporarily (no boot entry).
    Test,
    /// Set as default for next boot only.
    Boot,
}

/// Observation domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ObserveDomain {
    /// Systemd services.
    Services,
    /// Installed packages.
    Packages,
    /// Nix store.
    Store,
    /// System generations.
    Generations,
    /// Hardware info.
    Hardware,
    /// Flake inputs.
    Flakes,
}

/// Flake subcommands.
#[derive(Subcommand, Debug)]
pub enum FlakeCommand {
    /// Check the flake for errors.
    Check,
    /// Update all flake inputs.
    Update {
        /// Only update specific inputs.
        inputs: Vec<String>,
    },
    /// Show flake outputs.
    Show,
    /// Display flake metadata.
    Info,
}

/// Service subcommands.
#[derive(Subcommand, Debug)]
pub enum ServiceCommand {
    /// Show status of a service.
    Status {
        /// Service name.
        name: String,
    },
    /// Start a service.
    Start {
        /// Service name.
        name: String,
    },
    /// Stop a service.
    Stop {
        /// Service name.
        name: String,
    },
    /// Restart a service.
    Restart {
        /// Service name.
        name: String,
    },
    /// List failed services.
    Failed,
}

impl Cli {
    /// Whether the user provided natural language input (not a subcommand).
    pub fn has_natural_input(&self) -> bool {
        !self.input.is_empty() && self.command.is_none()
    }

    /// Get the natural language input as a single string.
    pub fn natural_input(&self) -> String {
        self.input.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_parse_natural_language() {
        let cli = Cli::parse_from(["nix-mind", "install", "firefox"]);
        assert!(cli.has_natural_input());
        assert_eq!(cli.natural_input(), "install firefox");
    }

    #[test]
    fn test_parse_search_subcommand() {
        let cli = Cli::parse_from(["nix-mind", "search", "editor"]);
        assert!(matches!(cli.command, Some(Command::Search { .. })));
    }

    #[test]
    fn test_parse_rebuild_switch() {
        let cli = Cli::parse_from(["nix-mind", "rebuild"]);
        if let Some(Command::Rebuild { mode, .. }) = cli.command {
            assert_eq!(mode, RebuildMode::Switch);
        } else {
            panic!("Expected Rebuild command");
        }
    }

    #[test]
    fn test_parse_dry_run_flag() {
        let cli = Cli::parse_from(["nix-mind", "--dry-run", "rebuild"]);
        assert!(cli.dry_run);
    }

    #[test]
    fn test_parse_gc_analyze() {
        let cli = Cli::parse_from(["nix-mind", "gc", "--analyze"]);
        if let Some(Command::Gc { analyze, .. }) = cli.command {
            assert!(analyze);
        } else {
            panic!("Expected Gc command");
        }
    }

    #[test]
    fn test_parse_generations_diff() {
        let cli = Cli::parse_from([
            "nix-mind",
            "generations",
            "--diff",
            "--from",
            "42",
            "--to",
            "43",
        ]);
        if let Some(Command::Generations { diff, from, to, .. }) = cli.command {
            assert!(diff);
            assert_eq!(from, Some(42));
            assert_eq!(to, Some(43));
        } else {
            panic!("Expected Generations command");
        }
    }

    #[test]
    fn test_parse_service_restart() {
        let cli = Cli::parse_from(["nix-mind", "service", "restart", "nginx"]);
        if let Some(Command::Service {
            op: ServiceCommand::Restart { name },
        }) = cli.command
        {
            assert_eq!(name, "nginx");
        } else {
            panic!("Expected Service Restart command");
        }
    }

    #[test]
    fn test_parse_repl() {
        let cli = Cli::parse_from(["nix-mind", "repl"]);
        assert!(matches!(cli.command, Some(Command::Repl)));
    }

    #[test]
    fn test_output_format_json() {
        let cli = Cli::parse_from(["nix-mind", "--format", "json", "search", "vim"]);
        assert_eq!(cli.format, OutputFormat::Json);
    }

    #[test]
    fn test_parse_health() {
        let cli = Cli::parse_from(["nix-mind", "health"]);
        assert!(matches!(cli.command, Some(Command::Health { json: false })));

        let cli = Cli::parse_from(["nix-mind", "health", "--json"]);
        assert!(matches!(cli.command, Some(Command::Health { json: true })));
    }

    #[test]
    fn test_parse_predict() {
        let cli = Cli::parse_from(["nix-mind", "predict"]);
        assert!(matches!(cli.command, Some(Command::Predict { .. })));
    }

    #[test]
    fn test_parse_watch() {
        let cli = Cli::parse_from(["nix-mind", "watch", "--timeout", "60", "--interval", "5"]);
        if let Some(Command::Watch {
            timeout, interval, ..
        }) = cli.command
        {
            assert_eq!(timeout, 60);
            assert_eq!(interval, 5);
        } else {
            panic!("Expected Watch command");
        }
    }

    #[test]
    fn test_parse_scrub() {
        let cli = Cli::parse_from(["nix-mind", "scrub", "/var/log/syslog"]);
        if let Some(Command::Scrub { file }) = cli.command {
            assert_eq!(file, Some("/var/log/syslog".to_string()));
        } else {
            panic!("Expected Scrub command");
        }
    }

    #[test]
    fn test_parse_knowledge() {
        let cli = Cli::parse_from(["nix-mind", "knowledge", "hash mismatch"]);
        if let Some(Command::Knowledge { query, limit }) = cli.command {
            assert_eq!(query, "hash mismatch");
            assert_eq!(limit, 5);
        } else {
            panic!("Expected Knowledge command");
        }
    }
}
