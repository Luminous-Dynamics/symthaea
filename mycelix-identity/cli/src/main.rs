// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Identity CLI - Legacy Bridge Credential Import Tool
//!
//! This CLI tool enables institutions to import academic credentials from
//! legacy systems (CSV files) into the Mycelix Identity network.
//!
//! Features:
//! - CSV parsing with configurable field mapping
//! - DNS-DID verification with DNSSEC validation
//! - ZK commitment generation for privacy
//! - Batch import with progress tracking
//! - Dry-run mode for validation

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod csv_parser;
mod credential;
mod dns_did;
mod import;
mod pqc_commands;

use import::ImportConfig;

/// Mycelix Identity CLI - Academic Credential Import Tool
#[derive(Parser)]
#[command(name = "mycelix-credential")]
#[command(author = "Luminous Dynamics")]
#[command(version = "0.1.0")]
#[command(about = "Import academic credentials from legacy systems into Mycelix Identity")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Import credentials from a CSV file
    Import {
        /// Path to the CSV file
        #[arg(short, long)]
        file: PathBuf,

        /// Institution DID (e.g., did:dns:registrar.university.edu)
        #[arg(short, long)]
        institution_did: String,

        /// Institution name
        #[arg(short = 'n', long)]
        institution_name: String,

        /// Revocation registry ID
        #[arg(short, long)]
        revocation_registry: String,

        /// Output directory for generated credentials (JSON)
        #[arg(short, long, default_value = "./credentials")]
        output: PathBuf,

        /// Dry run - validate without creating credentials
        #[arg(long)]
        dry_run: bool,

        /// Skip DNS-DID verification
        #[arg(long)]
        skip_dns_verification: bool,

        /// Custom field mapping file (JSON)
        #[arg(long)]
        field_mapping: Option<PathBuf>,
    },

    /// Verify DNS-DID linkage for a domain
    VerifyDns {
        /// Domain to verify (e.g., registrar.university.edu)
        #[arg(short, long)]
        domain: String,

        /// Expected DID
        #[arg(short, long)]
        expected_did: Option<String>,

        /// Show detailed DNSSEC information
        #[arg(long)]
        verbose: bool,
    },

    /// Generate a sample CSV template
    Template {
        /// Output file path
        #[arg(short, long, default_value = "credentials_template.csv")]
        output: PathBuf,
    },

    /// Verify a credential's ZK commitment
    VerifyCommitment {
        /// Credential JSON file
        #[arg(short, long)]
        credential: PathBuf,

        /// Nonce (base64 encoded)
        #[arg(short, long)]
        nonce: String,
    },

    /// Show import batch status
    BatchStatus {
        /// Batch ID to check
        #[arg(short, long)]
        batch_id: String,
    },

    /// Verify a credential file (structure, ZK commitment, DNS-DID)
    Verify {
        /// Credential JSON file
        #[arg(short, long)]
        credential: PathBuf,

        /// Also verify DNS-DID linkage (requires network)
        #[arg(long)]
        check_dns: bool,

        /// Show full verification details
        #[arg(long)]
        verbose: bool,
    },

    /// Print DNS-DID setup instructions for an institution
    SetupDns {
        /// Domain name (e.g., registrar.university.edu)
        #[arg(short, long)]
        domain: String,

        /// Institution DID
        #[arg(short = 'i', long)]
        did: String,
    },

    /// Generate a PQC keypair (Ed25519, ML-DSA-65/87, SPHINCS+, or hybrid)
    Keygen {
        /// Algorithm: ed25519, ml-dsa-65, ml-dsa-87, slh-dsa-sha2-128s, hybrid-ed25519-ml-dsa-65
        #[arg(short, long)]
        algorithm: String,

        /// Output key file path (JSON)
        #[arg(short, long, default_value = "key.json")]
        output: PathBuf,
    },

    /// Sign a credential with a PQC key
    Sign {
        /// Key file (from keygen)
        #[arg(short, long)]
        key: PathBuf,

        /// Credential JSON file
        #[arg(short, long)]
        credential: PathBuf,

        /// Output signed credential path
        #[arg(short, long, default_value = "signed.json")]
        output: PathBuf,
    },

    /// Dual-sign a credential with Ed25519 + ML-DSA-65 (hybrid)
    HybridSign {
        /// Ed25519 key file
        #[arg(long)]
        ed25519_key: PathBuf,

        /// ML-DSA-65 key file
        #[arg(long)]
        pqc_key: PathBuf,

        /// Credential JSON file
        #[arg(short, long)]
        credential: PathBuf,

        /// Output signed credential path
        #[arg(short, long, default_value = "signed.json")]
        output: PathBuf,
    },

    /// Verify a PQC/hybrid signature on a credential
    PqcVerify {
        /// Signed credential JSON file
        #[arg(short, long)]
        credential: PathBuf,

        /// Key file for full cryptographic verification (from keygen)
        #[arg(short, long)]
        key: Option<PathBuf>,

        /// Ed25519 key file (for dual-key hybrid verification with hybrid-sign output)
        #[arg(long)]
        ed25519_key: Option<PathBuf>,

        /// ML-DSA-65 key file (for dual-key hybrid verification with hybrid-sign output)
        #[arg(long)]
        pqc_key: Option<PathBuf>,

        /// Show full verification details
        #[arg(long)]
        verbose: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Import {
            file,
            institution_did,
            institution_name,
            revocation_registry,
            output,
            dry_run,
            skip_dns_verification,
            field_mapping,
        } => {
            let config = ImportConfig {
                csv_path: file,
                institution_did,
                institution_name,
                revocation_registry,
                output_dir: output,
                dry_run,
                skip_dns_verification,
                field_mapping,
            };

            import::run_import(config).await?;
        }

        Commands::VerifyDns {
            domain,
            expected_did,
            verbose,
        } => {
            dns_did::verify_dns_did(&domain, expected_did.as_deref(), verbose).await?;
        }

        Commands::Template { output } => {
            csv_parser::generate_template(&output)?;
            println!("Template generated: {}", output.display());
        }

        Commands::VerifyCommitment { credential, nonce } => {
            credential::verify_commitment_cli(&credential, &nonce)?;
        }

        Commands::BatchStatus { batch_id } => {
            import::show_batch_status(&batch_id).await?;
        }

        Commands::Verify {
            credential,
            check_dns,
            verbose,
        } => {
            credential::verify_credential_cli(&credential, check_dns, verbose).await?;
        }

        Commands::SetupDns { domain, did } => {
            dns_did::print_setup_instructions(&domain, &did);
        }

        Commands::Keygen { algorithm, output } => {
            pqc_commands::keygen(&algorithm, &output)?;
        }

        Commands::Sign {
            key,
            credential,
            output,
        } => {
            pqc_commands::sign(&key, &credential, &output)?;
        }

        Commands::HybridSign {
            ed25519_key,
            pqc_key,
            credential,
            output,
        } => {
            pqc_commands::hybrid_sign(&ed25519_key, &pqc_key, &credential, &output)?;
        }

        Commands::PqcVerify {
            credential,
            key,
            ed25519_key,
            pqc_key,
            verbose,
        } => {
            pqc_commands::pqc_verify(
                &credential,
                key.as_deref(),
                ed25519_key.as_deref(),
                pqc_key.as_deref(),
                verbose,
            )?;
        }
    }

    Ok(())
}
