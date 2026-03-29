// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Output formatting

use anyhow::{Context, Result};
use colored::*;
use comfy_table::{Cell, Table};
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub enum OutputMode {
    Table,
    Json,
    Plain,
}

impl OutputMode {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "table" => Ok(OutputMode::Table),
            "json" => Ok(OutputMode::Json),
            "plain" => Ok(OutputMode::Plain),
            _ => anyhow::bail!("Invalid output format: {}. Use 'table', 'json', or 'plain'", s),
        }
    }
}

/// Print success message
pub fn success(message: &str) {
    println!("{} {}", "✓".green().bold(), message);
}

/// Print error message
pub fn error(message: &str) {
    eprintln!("{} {}", "✗".red().bold(), message);
}

/// Print info message
pub fn info(message: &str) {
    println!("{} {}", "ℹ".blue().bold(), message);
}

/// Print warning message
pub fn warning(message: &str) {
    println!("{} {}", "⚠".yellow().bold(), message);
}

/// Print value as JSON
pub fn print_json<T: Serialize>(value: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(value)
        .context("Failed to serialize to JSON")?;
    println!("{}", json);
    Ok(())
}

/// Create a basic table
pub fn create_table(headers: &[&str]) -> Table {
    let mut table = Table::new();
    table.set_header(headers);
    table
}

/// Print a simple key-value table
pub fn print_key_value_table(items: &[(&str, String)]) {
    let mut table = create_table(&["Key", "Value"]);

    for (key, value) in items {
        table.add_row(vec![
            Cell::new(key.bold()),
            Cell::new(value),
        ]);
    }

    println!("{}", table);
}
