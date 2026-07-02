// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

mod commands;

#[derive(Parser)]
#[command(name = "mycelix")]
#[command(about = "Mycelix ERP CLI - Blockchain-auditable enterprise resource planning", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// API base URL
    #[arg(long, env = "MYCELIX_URL", default_value = "http://localhost:8080")]
    url: String,

    /// Output format (text, json)
    #[arg(long, default_value = "text")]
    format: OutputFormat,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Clone, Debug)]
enum OutputFormat {
    Text,
    Json,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(OutputFormat::Text),
            "json" => Ok(OutputFormat::Json),
            _ => Err(format!("Invalid format: {}", s)),
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a new Ed25519 keypair
    Keygen {
        /// Output file path
        #[arg(short, long, default_value = "keypair.json")]
        output: PathBuf,

        /// Use specific seed (hex string, 32 bytes)
        #[arg(long)]
        seed: Option<String>,
    },

    /// Ingest a supply chain event
    Ingest {
        /// Event file (JSON) - use "-" for stdin
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Read from stdin
        #[arg(long)]
        stdin: bool,
    },

    /// Get a claim by ID
    Get {
        /// Claim ID
        claim_id: String,
    },

    /// Get lineage for a batch
    Lineage {
        /// Batch ID
        batch_id: String,

        /// Output format (json, dot, mermaid)
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Maximum depth (0 = unlimited)
        #[arg(short, long, default_value = "0")]
        depth: usize,
    },

    /// Verify a Verifiable Credential JWT
    Verify {
        /// VC JWT string
        #[arg(long)]
        jwt: Option<String>,

        /// Read JWT from file
        #[arg(long)]
        file: Option<PathBuf>,
    },

    /// Import events from CSV file
    ImportCsv {
        /// CSV file path
        #[arg(short, long)]
        file: PathBuf,

        /// Skip header row
        #[arg(long, default_value = "true")]
        header: bool,

        /// Batch size for parallel ingestion
        #[arg(long, default_value = "10")]
        batch_size: usize,
    },

    /// Health check
    Health,

    /// Database operations
    #[command(subcommand)]
    Db(DbCommands),
}

#[derive(Subcommand)]
enum DbCommands {
    /// Run database migrations
    Migrate {
        /// Database URL
        #[arg(long, env = "DATABASE_URL", default_value = "sqlite://data/claims.db")]
        database_url: String,
    },

    /// Show database statistics
    Stats {
        /// Database URL
        #[arg(long, env = "DATABASE_URL", default_value = "sqlite://data/claims.db")]
        database_url: String,
    },

    /// Health check
    Check {
        /// Database URL
        #[arg(long, env = "DATABASE_URL", default_value = "sqlite://data/claims.db")]
        database_url: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up colored output
    if !cfg!(target_os = "windows") {
        colored::control::set_override(true);
    }

    let result = match &cli.command {
        Commands::Keygen { output, seed } => commands::keygen::run(output, seed.as_deref()).await,
        Commands::Ingest { file, stdin } => {
            commands::ingest::run(&cli.url, file.as_deref(), *stdin).await
        }
        Commands::Get { claim_id } => commands::get::run(&cli.url, claim_id).await,
        Commands::Lineage {
            batch_id,
            format,
            depth,
        } => commands::lineage::run(&cli.url, batch_id, format, *depth).await,
        Commands::Verify { jwt, file } => commands::verify::run(&cli.url, jwt.as_deref(), file.as_deref()).await,
        Commands::ImportCsv {
            file,
            header,
            batch_size,
        } => commands::import_csv::run(&cli.url, file, *header, *batch_size).await,
        Commands::Health => commands::health::run(&cli.url).await,
        Commands::Db(db_cmd) => match db_cmd {
            DbCommands::Migrate { database_url } => commands::db::migrate(database_url).await,
            DbCommands::Stats { database_url } => commands::db::stats(database_url).await,
            DbCommands::Check { database_url } => commands::db::health_check(database_url).await,
        },
    };

    match result {
        Ok(()) => {
            if cli.verbose {
                println!("{}", "✓ Command completed successfully".green());
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("{} {}", "✗ Error:".red().bold(), e);
            std::process::exit(1);
        }
    }
}
