// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix-DeSci CLI
//!
//! Command-line interface for interacting with the Mycelix-DeSci API

use anyhow::Result;
use clap::{Parser, Subcommand};

mod client;
mod commands;
mod config;
mod output;

use client::ApiClient;
use config::Config;

#[derive(Parser)]
#[command(
    name = "mycelix",
    version,
    about = "Mycelix-DeSci CLI - Decentralized Science Platform",
    long_about = "Command-line interface for interacting with the Mycelix-DeSci decentralized science platform.\n\nManage scientific claims, query data, track trust scores, and more."
)]
struct Cli {
    /// API base URL
    #[arg(
        long,
        env = "MYCELIX_API_URL",
        default_value = "http://localhost:8080"
    )]
    api_url: String,

    /// Output format (json, table, plain)
    #[arg(long, short = 'o', default_value = "table")]
    output: String,

    /// Verbose output
    #[arg(long, short = 'v')]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage scientific claims
    #[command(subcommand)]
    Claims(commands::claims::ClaimsCommand),

    /// Query and search claims
    #[command(subcommand)]
    Query(commands::query::QueryCommand),

    /// Manage trust scores
    #[command(subcommand)]
    Trust(commands::trust::TrustCommand),

    /// System operations
    #[command(subcommand)]
    System(commands::system::SystemCommand),

    /// Show configuration
    Config,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load or create config
    let config = Config::load_or_default()?;

    // Create API client
    let api_url = if cli.api_url != "http://localhost:8080" {
        cli.api_url.clone()
    } else {
        config.api_url.clone().unwrap_or(cli.api_url.clone())
    };

    let client = ApiClient::new(&api_url)?;

    // Set output mode
    let output_mode = output::OutputMode::from_str(&cli.output)?;

    // Execute command
    match cli.command {
        Commands::Claims(cmd) => {
            commands::claims::execute(client, cmd, output_mode).await?;
        }
        Commands::Query(cmd) => {
            commands::query::execute(client, cmd, output_mode).await?;
        }
        Commands::Trust(cmd) => {
            commands::trust::execute(client, cmd, output_mode).await?;
        }
        Commands::System(cmd) => {
            commands::system::execute(client, cmd, output_mode).await?;
        }
        Commands::Config => {
            println!("Configuration:");
            println!("  API URL: {}", api_url);
            println!("  Config file: {}", config.config_path());
            if cli.verbose {
                println!("\nFull config:\n{:#?}", config);
            }
        }
    }

    Ok(())
}
