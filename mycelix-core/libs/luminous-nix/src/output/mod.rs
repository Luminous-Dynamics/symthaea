// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Output formatting for CLI results
//!
//! Provides beautiful table formatting and colored output for NixOS command results.

use colored::Colorize;
use tabled::{
    settings::{object::Columns, Alignment, Modify, Style, Width},
    Table, Tabled,
};

use crate::commands::{ExecutionResult, GenerationList, SearchResult, SystemStatus};
use crate::learning::LearningSystem;

/// Output formatter
pub struct Formatter {
    /// Whether to use colors
    use_color: bool,
    /// Whether to output JSON
    json_mode: bool,
}

impl Formatter {
    /// Create a new formatter
    pub fn new(use_color: bool, json_mode: bool) -> Self {
        Self {
            use_color,
            json_mode,
        }
    }

    /// Format search results
    pub fn format_search_result(&self, result: &SearchResult) {
        if self.json_mode {
            if let Ok(json) = serde_json::to_string_pretty(result) {
                println!("{}", json);
            }
            return;
        }

        if result.dry_run {
            println!("{}", "[DRY RUN] Would search for:".yellow());
            println!("  nix search nixpkgs {} --json", result.query);
            return;
        }

        if result.packages.is_empty() {
            println!("{}", "No packages found.".yellow());
            return;
        }

        #[derive(Tabled)]
        struct PackageRow {
            #[tabled(rename = "Package")]
            name: String,
            #[tabled(rename = "Version")]
            version: String,
            #[tabled(rename = "Description")]
            description: String,
        }

        let rows: Vec<PackageRow> = result
            .packages
            .iter()
            .map(|p| PackageRow {
                name: if self.use_color {
                    p.name.green().to_string()
                } else {
                    p.name.clone()
                },
                version: if self.use_color {
                    p.version.cyan().to_string()
                } else {
                    p.version.clone()
                },
                description: p
                    .description
                    .as_ref()
                    .map(|d| truncate_str(d, 50))
                    .unwrap_or_default(),
            })
            .collect();

        let table = Table::new(rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::single(0)).with(Width::wrap(30)))
            .with(Modify::new(Columns::single(2)).with(Width::wrap(50)))
            .to_string();

        println!("\n{}", table);
        println!(
            "\nFound {} packages matching '{}'",
            result.total_count,
            if self.use_color {
                result.query.cyan().to_string()
            } else {
                result.query.clone()
            }
        );
    }

    /// Format command execution result
    pub fn format_execution_result(&self, result: &ExecutionResult) {
        if self.json_mode {
            if let Ok(json) = serde_json::to_string_pretty(result) {
                println!("{}", json);
            }
            return;
        }

        if result.dry_run {
            println!("{}", "[DRY RUN] Would execute:".yellow());
            println!("  {}", result.command);
            return;
        }

        if result.success {
            println!(
                "{} {}",
                "✓".green(),
                if self.use_color {
                    "Command succeeded".green().to_string()
                } else {
                    "Command succeeded".to_string()
                }
            );
            if !result.stdout.is_empty() {
                println!("{}", result.stdout);
            }
            println!(
                "{}",
                format!("(completed in {}ms)", result.duration_ms).dimmed()
            );
        } else {
            println!(
                "{} {}",
                "✗".red(),
                if self.use_color {
                    "Command failed".red().to_string()
                } else {
                    "Command failed".to_string()
                }
            );
            if !result.stderr.is_empty() {
                eprintln!("{}", result.stderr);
            }
            if let Some(code) = result.exit_code {
                println!(
                    "{}",
                    format!("Exit code: {}", code).red()
                );
            }
        }
    }

    /// Format generations list
    pub fn format_generations(&self, result: &GenerationList) {
        if self.json_mode {
            if let Ok(json) = serde_json::to_string_pretty(result) {
                println!("{}", json);
            }
            return;
        }

        if result.dry_run {
            println!("{}", "[DRY RUN] Would list generations".yellow());
            return;
        }

        if result.generations.is_empty() {
            println!("{}", "No generations found.".yellow());
            return;
        }

        #[derive(Tabled)]
        struct GenerationRow {
            #[tabled(rename = "#")]
            number: String,
            #[tabled(rename = "Date")]
            date: String,
            #[tabled(rename = "Current")]
            current: String,
        }

        let rows: Vec<GenerationRow> = result
            .generations
            .iter()
            .map(|g| GenerationRow {
                number: if self.use_color && g.current {
                    g.number.to_string().green().bold().to_string()
                } else {
                    g.number.to_string()
                },
                date: g.date.clone(),
                current: if g.current {
                    if self.use_color {
                        "← current".green().to_string()
                    } else {
                        "← current".to_string()
                    }
                } else {
                    String::new()
                },
            })
            .collect();

        let table = Table::new(rows)
            .with(Style::rounded())
            .with(Modify::new(Columns::single(0)).with(Alignment::right()))
            .to_string();

        println!("\n{}", table);
        println!(
            "\n{} generations, current: #{}",
            result.generations.len(),
            result.current
        );
    }

    /// Format system status
    pub fn format_status(&self, status: &SystemStatus) {
        if self.json_mode {
            if let Ok(json) = serde_json::to_string_pretty(status) {
                println!("{}", json);
            }
            return;
        }

        println!("\n{}", "System Status".bold().underline());
        println!();

        let label_width = 20;
        print_kv("NixOS Version", &status.nixos_version, label_width, self.use_color);
        print_kv("Kernel", &status.kernel_version, label_width, self.use_color);
        print_kv(
            "Current Generation",
            &status.current_generation.to_string(),
            label_width,
            self.use_color,
        );
        print_kv(
            "Total Generations",
            &status.total_generations.to_string(),
            label_width,
            self.use_color,
        );

        if let Some(ref size) = status.store_size {
            print_kv("Nix Store Size", size, label_width, self.use_color);
        }

        println!();
    }

    /// Format learning system statistics
    pub fn format_learning_stats(&self, learning: &LearningSystem) {
        if self.json_mode {
            if let Ok(json) = serde_json::to_string_pretty(&learning.stats()) {
                println!("{}", json);
            }
            return;
        }

        let stats = learning.stats();

        println!("\n{}", "Learning Statistics".bold().underline());
        println!();

        let label_width = 25;
        print_kv("Total Interactions", &stats.total_interactions.to_string(), label_width, self.use_color);
        print_kv("Successful", &stats.successful.to_string(), label_width, self.use_color);
        print_kv("Failed", &stats.failed.to_string(), label_width, self.use_color);

        let success_rate = if stats.total_interactions > 0 {
            (stats.successful as f64 / stats.total_interactions as f64 * 100.0) as u32
        } else {
            0
        };
        print_kv("Success Rate", &format!("{}%", success_rate), label_width, self.use_color);

        println!();
    }
}

impl Default for Formatter {
    fn default() -> Self {
        Self::new(true, false)
    }
}

/// Print a key-value pair with aligned label
fn print_kv(key: &str, value: &str, width: usize, use_color: bool) {
    let label = if use_color {
        format!("{:>width$}:", key).dimmed().to_string()
    } else {
        format!("{:>width$}:", key)
    };
    println!("{} {}", label, value);
}

/// Truncate a string to max length with ellipsis
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world", 8), "hello...");
    }

    #[test]
    fn test_formatter_creation() {
        let formatter = Formatter::new(true, false);
        assert!(formatter.use_color);
        assert!(!formatter.json_mode);
    }
}
