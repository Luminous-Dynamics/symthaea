// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix-DeSci CLI
//!
//! Command-line interface for interacting with Mycelix-DeSci

use clap::{Parser, Subcommand};
use mycelix_desci_core::{logging, Config, Error, Result};
use std::path::PathBuf;

mod commands;

#[derive(Parser)]
#[command(name = "mycelix-desci")]
#[command(about = "Mycelix-DeSci - Verifiable Infrastructure for Decentralized Science", long_about = None)]
#[command(version)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Log level (debug, info, warn, error)
    #[arg(short = 'l', long, global = true)]
    log_level: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize configuration
    Init {
        /// Output directory for configuration
        #[arg(short, long, default_value = ".mycelix")]
        output: PathBuf,
    },

    /// Calculate hash of a file or directory
    Hash {
        /// File or directory to hash
        path: PathBuf,

        /// Hash algorithm (blake3, sha256)
        #[arg(short = 'a', long, default_value = "blake3")]
        algorithm: String,
    },

    /// Upload a dataset and create a claim
    Upload {
        /// Dataset file to upload
        file: PathBuf,

        /// Epistemic tier (E0-E4)
        #[arg(short, long, default_value = "E0")]
        tier: String,

        /// Category (genomics, longevity, climate, etc.)
        #[arg(short, long)]
        category: String,

        /// Description of the dataset
        #[arg(short, long)]
        description: String,

        /// Provenance information
        #[arg(short, long)]
        provenance: Option<String>,

        /// License
        #[arg(short = 'L', long)]
        license: Option<String>,

        /// Keywords (comma-separated)
        #[arg(short, long)]
        keywords: Option<String>,
    },

    /// Verify a claim
    Verify {
        /// Claim ID to verify
        claim_id: String,

        /// Dataset file (optional, for hash verification)
        #[arg(short, long)]
        file: Option<PathBuf>,
    },

    /// Query claims
    Query {
        /// Category filter
        #[arg(short, long)]
        category: Option<String>,

        /// Minimum epistemic tier
        #[arg(short = 't', long)]
        min_tier: Option<String>,

        /// Keyword search
        #[arg(short, long)]
        keywords: Option<String>,

        /// Output format (json, text, table)
        #[arg(short = 'f', long, default_value = "table")]
        format: String,

        /// Maximum number of results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },

    /// Display information about a claim
    Info {
        /// Claim ID
        claim_id: String,

        /// Output format (json, text)
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
    },

    /// Display configuration
    Config {
        /// Show configuration
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,

    /// Validate configuration
    Validate,
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    // Load configuration
    let mut config = if let Some(config_path) = cli.config {
        Config::from_file(config_path)?
    } else {
        Config::load().unwrap_or_default()
    };

    // Override log level if specified
    if let Some(level) = cli.log_level {
        config.logging.level = level;
    }

    // Initialize logging
    logging::init(&config.logging);

    // Execute command
    match cli.command {
        Commands::Init { output } => commands::init::execute(output).await,
        Commands::Hash { path, algorithm } => commands::hash::execute(path, algorithm).await,
        Commands::Upload {
            file,
            tier,
            category,
            description,
            provenance,
            license,
            keywords,
        } => {
            commands::upload::execute(
                file,
                tier,
                category,
                description,
                provenance,
                license,
                keywords,
            )
            .await
        }
        Commands::Verify { claim_id, file } => commands::verify::execute(claim_id, file).await,
        Commands::Query {
            category,
            min_tier,
            keywords,
            format,
            limit,
        } => {
            commands::query::execute(category, min_tier, keywords, format, limit).await
        }
        Commands::Info { claim_id, format } => commands::info::execute(claim_id, format).await,
        Commands::Config { action } => match action {
            ConfigAction::Show => commands::config::show(config).await,
            ConfigAction::Validate => commands::config::validate(config).await,
        },
    }
}
