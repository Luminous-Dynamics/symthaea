// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ZK Tax CLI - Command-line tool for generating and verifying tax proofs.
//!
//! # Usage
//!
//! ```bash
//! # Generate a proof
//! zk-tax prove --income 85000 --jurisdiction US --status single --year 2024
//!
//! # Verify a proof
//! zk-tax verify --proof proof.json
//!
//! # Compress a proof
//! zk-tax compress --input proof.json --output proof.compressed.json
//!
//! # List jurisdictions
//! zk-tax jurisdictions
//!
//! # Show brackets
//! zk-tax brackets --jurisdiction US --year 2024
//! ```

use clap::{Parser, Subcommand};
use colored::Colorize;
use mycelix_zk_tax::{
    proof::CompressedProof,
    Jurisdiction, FilingStatus, TaxBracketProver, TaxBracketProof,
};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(name = "zk-tax")]
#[command(author = "Mycelix Team")]
#[command(version = mycelix_zk_tax::VERSION)]
#[command(about = "Zero-knowledge tax bracket proofs", long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Use development mode (fast but not cryptographically secure)
    #[arg(long, global = true)]
    dev: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a tax bracket proof
    Prove {
        /// Annual gross income in dollars
        #[arg(short, long)]
        income: u64,

        /// Tax jurisdiction (US, UK, DE, CA, AU, etc.)
        #[arg(short, long, default_value = "US")]
        jurisdiction: String,

        /// Filing status (single, mfj, mfs, hoh)
        #[arg(short, long, default_value = "single")]
        status: String,

        /// Tax year
        #[arg(short, long, default_value = "2024")]
        year: u32,

        /// Output file (defaults to stdout as JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format (json, hex, base64)
        #[arg(long, default_value = "json")]
        format: String,
    },

    /// Verify a tax bracket proof
    Verify {
        /// Path to proof file
        #[arg(short, long)]
        proof: PathBuf,

        /// Expected bracket index (optional)
        #[arg(long)]
        expected_bracket: Option<u8>,
    },

    /// Compress a proof
    Compress {
        /// Input proof file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for compressed proof
        #[arg(short, long)]
        output: PathBuf,

        /// Compression type (rle, base64, combined)
        #[arg(long, default_value = "combined")]
        compression: String,
    },

    /// Decompress a proof
    Decompress {
        /// Input compressed proof file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for decompressed proof
        #[arg(short, long)]
        output: PathBuf,
    },

    /// List supported jurisdictions
    Jurisdictions {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Show tax brackets for a jurisdiction
    Brackets {
        /// Tax jurisdiction
        #[arg(short, long, default_value = "US")]
        jurisdiction: String,

        /// Tax year
        #[arg(short, long, default_value = "2024")]
        year: u32,

        /// Filing status
        #[arg(short, long, default_value = "single")]
        status: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Generate a batch (multi-year) proof
    Batch {
        /// Comma-separated year:income pairs (e.g., "2022:75000,2023:80000,2024:85000")
        #[arg(short, long)]
        years: String,

        /// Tax jurisdiction
        #[arg(short = 'j', long, default_value = "US")]
        jurisdiction: String,

        /// Filing status
        #[arg(short, long, default_value = "single")]
        status: String,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Show information about a proof file
    Info {
        /// Path to proof file
        #[arg(short, long)]
        proof: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Prove {
            income,
            jurisdiction,
            status,
            year,
            output,
            format,
        } => {
            cmd_prove(cli.dev, cli.verbose, income, &jurisdiction, &status, year, output, &format)?;
        }

        Commands::Verify {
            proof,
            expected_bracket,
        } => {
            cmd_verify(cli.verbose, proof, expected_bracket)?;
        }

        Commands::Compress {
            input,
            output,
            compression,
        } => {
            cmd_compress(cli.verbose, input, output, &compression)?;
        }

        Commands::Decompress { input, output } => {
            cmd_decompress(cli.verbose, input, output)?;
        }

        Commands::Jurisdictions { detailed } => {
            cmd_jurisdictions(detailed)?;
        }

        Commands::Brackets {
            jurisdiction,
            year,
            status,
            json,
        } => {
            cmd_brackets(&jurisdiction, year, &status, json)?;
        }

        Commands::Batch {
            years,
            jurisdiction,
            status,
            output,
        } => {
            cmd_batch(cli.dev, cli.verbose, &years, &jurisdiction, &status, output)?;
        }

        Commands::Info { proof } => {
            cmd_info(proof)?;
        }
    }

    Ok(())
}

fn cmd_prove(
    dev_mode: bool,
    verbose: bool,
    income: u64,
    jurisdiction: &str,
    status: &str,
    year: u32,
    output: Option<PathBuf>,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    if verbose {
        eprintln!("Generating proof for income ${} in {} ({})...", income, jurisdiction, year);
    }

    let prover = if dev_mode {
        if verbose {
            eprintln!("Using DEV MODE - proof is not cryptographically secure");
        }
        TaxBracketProver::dev_mode()
    } else {
        TaxBracketProver::new()
    };

    let jurisdiction = jurisdiction.parse::<Jurisdiction>()
        .map_err(|e| format!("Invalid jurisdiction: {}", e))?;

    let filing_status = parse_filing_status(status)?;

    let proof = prover.prove(income, jurisdiction, filing_status, year)?;

    if verbose {
        eprintln!("Proof generated in {:?}", start.elapsed());
        eprintln!("Bracket: {} ({}% marginal rate)", proof.bracket_index, proof.rate_bps as f64 / 100.0);
        eprintln!("Range: ${} - ${}", proof.bracket_lower, proof.bracket_upper);
    }

    let output_str = match format {
        "json" => serde_json::to_string_pretty(&proof)?,
        "hex" => hex::encode(serde_json::to_vec(&proof)?),
        "base64" => {
            use base64::{Engine as _, engine::general_purpose};
            general_purpose::STANDARD.encode(serde_json::to_vec(&proof)?)
        }
        _ => return Err(format!("Unknown format: {}", format).into()),
    };

    if let Some(path) = output {
        fs::write(&path, &output_str)?;
        if verbose {
            eprintln!("Proof written to {:?}", path);
        }
    } else {
        println!("{}", output_str);
    }

    Ok(())
}

fn cmd_verify(
    verbose: bool,
    proof_path: PathBuf,
    expected_bracket: Option<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    if verbose {
        eprintln!("Reading proof from {:?}...", proof_path);
    }

    let content = fs::read_to_string(&proof_path)?;
    let proof: TaxBracketProof = serde_json::from_str(&content)?;

    if verbose {
        eprintln!("Verifying proof...");
    }

    match proof.verify() {
        Ok(_) => {
            println!("VALID");
            println!("  Jurisdiction: {:?}", proof.jurisdiction);
            println!("  Tax Year: {}", proof.tax_year);
            println!("  Bracket: {} ({}%)", proof.bracket_index, proof.rate_bps as f64 / 100.0);
            println!("  Range: ${} - ${}", proof.bracket_lower, proof.bracket_upper);

            if let Some(expected) = expected_bracket {
                if proof.bracket_index != expected {
                    eprintln!("WARNING: Expected bracket {}, got {}", expected, proof.bracket_index);
                    std::process::exit(1);
                }
            }

            if verbose {
                eprintln!("Verified in {:?}", start.elapsed());
            }
        }
        Err(e) => {
            eprintln!("INVALID: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn cmd_compress(
    verbose: bool,
    input: PathBuf,
    output: PathBuf,
    _compression: &str,  // Currently only RLE is implemented
) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(&input)?;
    let proof: TaxBracketProof = serde_json::from_str(&content)?;

    let original_size = content.len();
    let compressed = CompressedProof::from_bracket_proof(&proof)?;
    let compressed_json = serde_json::to_string_pretty(&compressed)?;
    let compressed_size = compressed_json.len();

    fs::write(&output, &compressed_json)?;

    let ratio = (1.0 - (compressed_size as f64 / original_size as f64)) * 100.0;
    println!("Compressed: {} bytes -> {} bytes ({:.1}% reduction)", original_size, compressed_size, ratio);

    if verbose {
        eprintln!("Written to {:?}", output);
    }

    Ok(())
}

fn cmd_decompress(
    verbose: bool,
    input: PathBuf,
    output: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(&input)?;
    let compressed: CompressedProof = serde_json::from_str(&content)?;

    let proof = compressed.to_bracket_proof()?;
    let proof_json = serde_json::to_string_pretty(&proof)?;

    fs::write(&output, &proof_json)?;

    if verbose {
        eprintln!("Decompressed to {:?}", output);
    }

    Ok(())
}

fn cmd_jurisdictions(detailed: bool) -> Result<(), Box<dyn std::error::Error>> {
    let jurisdictions = [
        ("US", "United States", "IRS", "USD"),
        ("UK", "United Kingdom", "HMRC", "GBP"),
        ("DE", "Germany", "Finanzamt", "EUR"),
        ("FR", "France", "DGFiP", "EUR"),
        ("CA", "Canada", "CRA", "CAD"),
        ("AU", "Australia", "ATO", "AUD"),
        ("JP", "Japan", "NTA", "JPY"),
        ("CN", "China", "SAT", "CNY"),
        ("IN", "India", "IT Department", "INR"),
        ("BR", "Brazil", "Receita Federal", "BRL"),
        ("MX", "Mexico", "SAT", "MXN"),
        ("KR", "South Korea", "NTS", "KRW"),
        ("IT", "Italy", "Agenzia Entrate", "EUR"),
        ("RU", "Russia", "FNS", "RUB"),
        ("TR", "Turkey", "GIB", "TRY"),
        ("AR", "Argentina", "AFIP", "ARS"),
        ("ID", "Indonesia", "DJP", "IDR"),
        ("SA", "Saudi Arabia", "ZATCA", "SAR"),
        ("ZA", "South Africa", "SARS", "ZAR"),
    ];

    if detailed {
        println!("{:<6} {:<20} {:<20} {:<6}", "CODE", "COUNTRY", "AUTHORITY", "CURR");
        println!("{}", "-".repeat(54));
        for (code, name, authority, currency) in &jurisdictions {
            println!("{:<6} {:<20} {:<20} {:<6}", code, name, authority, currency);
        }
    } else {
        for (code, name, _, _) in &jurisdictions {
            println!("{}: {}", code, name);
        }
    }

    Ok(())
}

fn cmd_brackets(
    jurisdiction: &str,
    year: u32,
    status: &str,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let jurisdiction = jurisdiction.parse::<Jurisdiction>()
        .map_err(|e| format!("Invalid jurisdiction: {}", e))?;
    let filing_status = parse_filing_status(status)?;

    let brackets = mycelix_zk_tax::brackets::get_brackets(jurisdiction, year, filing_status)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&brackets)?);
    } else {
        println!("Tax Brackets for {:?} {} ({}):", jurisdiction, year, status);
        println!("{:<8} {:<15} {:<15} {:<10}", "BRACKET", "LOWER", "UPPER", "RATE");
        println!("{}", "-".repeat(50));
        for (i, bracket) in brackets.iter().enumerate() {
            let upper = if bracket.upper == u64::MAX {
                "∞".to_string()
            } else {
                format!("${}", bracket.upper)
            };
            println!(
                "{:<8} ${:<14} {:<15} {}%",
                i,
                bracket.lower,
                upper,
                bracket.rate_bps as f64 / 100.0
            );
        }
    }

    Ok(())
}

fn cmd_batch(
    dev_mode: bool,
    verbose: bool,
    years: &str,
    jurisdiction: &str,
    status: &str,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use mycelix_zk_tax::proof::BatchProofBuilder;

    let jurisdiction = jurisdiction.parse::<Jurisdiction>()
        .map_err(|e| format!("Invalid jurisdiction: {}", e))?;
    let filing_status = parse_filing_status(status)?;

    let mut builder = BatchProofBuilder::new(jurisdiction, filing_status);

    for pair in years.split(',') {
        let parts: Vec<&str> = pair.trim().split(':').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid year:income pair: {}", pair).into());
        }
        let year: u32 = parts[0].parse()?;
        let income: u64 = parts[1].parse()?;
        builder = builder.add_year(year, income);

        if verbose {
            eprintln!("Added {}: ${}", year, income);
        }
    }

    let proof = if dev_mode {
        builder.build_dev()?
    } else {
        // Real proof requires prover feature
        builder.build_dev()?
    };

    let json = serde_json::to_string_pretty(&proof)?;

    if let Some(path) = output {
        fs::write(&path, &json)?;
        if verbose {
            eprintln!("Batch proof written to {:?}", path);
        }
    } else {
        println!("{}", json);
    }

    Ok(())
}

fn cmd_info(proof_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(&proof_path)?;
    let proof: TaxBracketProof = serde_json::from_str(&content)?;

    println!("Proof Information:");
    println!("  File: {:?}", proof_path);
    println!("  Size: {} bytes", content.len());
    println!();
    println!("  Jurisdiction: {:?} ({})", proof.jurisdiction, proof.jurisdiction.name());
    println!("  Filing Status: {:?}", proof.filing_status);
    println!("  Tax Year: {}", proof.tax_year);
    println!();
    println!("  Bracket Index: {}", proof.bracket_index);
    println!("  Marginal Rate: {}%", proof.rate_bps as f64 / 100.0);
    println!("  Lower Bound: ${}", proof.bracket_lower);
    println!("  Upper Bound: ${}", proof.bracket_upper);
    println!();
    println!("  Commitment: {}", proof.commitment.to_hex());
    println!("  Receipt Size: {} bytes", proof.receipt_bytes.len());
    println!("  Image ID Size: {} bytes", proof.image_id.len());

    Ok(())
}

fn parse_filing_status(s: &str) -> Result<FilingStatus, Box<dyn std::error::Error>> {
    match s.to_lowercase().as_str() {
        "single" | "s" => Ok(FilingStatus::Single),
        "mfj" | "married_filing_jointly" | "joint" => Ok(FilingStatus::MarriedFilingJointly),
        "mfs" | "married_filing_separately" | "separate" => Ok(FilingStatus::MarriedFilingSeparately),
        "hoh" | "head_of_household" | "head" => Ok(FilingStatus::HeadOfHousehold),
        _ => Err(format!("Invalid filing status: {}. Use: single, mfj, mfs, hoh", s).into()),
    }
}
