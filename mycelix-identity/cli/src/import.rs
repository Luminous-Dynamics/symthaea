// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CSV Import Orchestration
//!
//! Manages the full import workflow:
//! 1. Parse CSV file
//! 2. Validate records
//! 3. Verify DNS-DID (optional)
//! 4. Generate credentials with ZK commitments
//! 5. Output credentials as JSON files

use anyhow::{Context, Result};
use chrono::Utc;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use uuid::Uuid;

use crate::credential::{build_credential, AcademicCredential, CredentialConfig};
use crate::csv_parser::{self, FieldMapping, ParseResult};
use crate::dns_did::{self, DnsDidVerification};

/// Import configuration
pub struct ImportConfig {
    pub csv_path: PathBuf,
    pub institution_did: String,
    pub institution_name: String,
    pub revocation_registry: String,
    pub output_dir: PathBuf,
    pub dry_run: bool,
    pub skip_dns_verification: bool,
    pub field_mapping: Option<PathBuf>,
}

/// Import batch metadata
#[derive(serde::Serialize, serde::Deserialize)]
struct ImportBatch {
    batch_id: String,
    institution_did: String,
    institution_name: String,
    source_file: String,
    source_hash: String,
    import_timestamp: String,
    total_records: usize,
    successful: usize,
    failed: usize,
    dry_run: bool,
    credentials: Vec<String>,
    errors: Vec<ImportErrorRecord>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ImportErrorRecord {
    row: usize,
    field: String,
    message: String,
    code: String,
}

/// Run the import process
pub async fn run_import(config: ImportConfig) -> Result<()> {
    let start_time = std::time::Instant::now();

    println!();
    println!(
        "{}",
        style("╔══════════════════════════════════════════════════════════╗").cyan()
    );
    println!(
        "{}",
        style("║     Mycelix Identity - Legacy Bridge Credential Import   ║").cyan()
    );
    println!(
        "{}",
        style("╚══════════════════════════════════════════════════════════╝").cyan()
    );
    println!();

    if config.dry_run {
        println!(
            "{}",
            style("  ⚠ DRY RUN MODE - No credentials will be created").yellow().bold()
        );
        println!();
    }

    // Generate batch ID
    let batch_id = format!("batch-{}", Uuid::new_v4().to_string()[..8].to_string());
    println!("Batch ID: {}", style(&batch_id).green().bold());

    // Step 1: Load field mapping
    println!();
    println!(
        "{} Loading configuration...",
        style("[1/6]").bold().dim()
    );

    let mapping = if let Some(ref mapping_path) = config.field_mapping {
        println!("  Using custom mapping: {}", mapping_path.display());
        csv_parser::load_mapping(mapping_path)?
    } else {
        println!("  Using default field mapping");
        FieldMapping::default()
    };

    // Step 2: Parse CSV file
    println!();
    println!(
        "{} Parsing CSV file: {}",
        style("[2/6]").bold().dim(),
        config.csv_path.display()
    );

    // Calculate source file hash
    let source_data = fs::read(&config.csv_path).context("Failed to read CSV file")?;
    let source_hash = {
        let mut hasher = Sha256::new();
        hasher.update(&source_data);
        hex::encode(hasher.finalize())
    };
    println!("  Source hash: {}", &source_hash[..16]);

    let parse_result = csv_parser::parse_csv(&config.csv_path, &mapping)?;

    println!("  Total rows: {}", parse_result.total_rows);
    println!(
        "  Valid records: {}",
        style(parse_result.records.len()).green()
    );
    if !parse_result.errors.is_empty() {
        println!(
            "  Parse errors: {}",
            style(parse_result.errors.len()).red()
        );
    }

    // Step 3: DNS-DID Verification (optional)
    let dns_verification = if !config.skip_dns_verification {
        println!();
        println!(
            "{} Verifying DNS-DID linkage...",
            style("[3/6]").bold().dim()
        );

        // Extract domain from DID
        let domain = config
            .institution_did
            .strip_prefix("did:dns:")
            .unwrap_or(&config.institution_did);

        match dns_did::verify_dns_did(domain, Some(&config.institution_did), false).await {
            Ok(verification) => {
                if verification.verified {
                    println!(
                        "  DNS-DID verified: {}",
                        style("✓").green().bold()
                    );
                } else {
                    println!(
                        "  DNS-DID verification: {}",
                        style("⚠ Warning - DID not found in DNS").yellow()
                    );
                }
                Some(verification)
            }
            Err(e) => {
                println!(
                    "  DNS-DID verification: {} ({})",
                    style("⚠ Skipped").yellow(),
                    e
                );
                None
            }
        }
    } else {
        println!();
        println!(
            "{} Skipping DNS-DID verification (--skip-dns-verification)",
            style("[3/6]").bold().dim()
        );
        None
    };

    // Step 4: Prepare output directory
    println!();
    println!(
        "{} Preparing output directory...",
        style("[4/6]").bold().dim()
    );

    if !config.dry_run {
        fs::create_dir_all(&config.output_dir).context("Failed to create output directory")?;
    }
    println!("  Output: {}", config.output_dir.display());

    // Step 5: Generate credentials
    println!();
    println!(
        "{} Generating credentials...",
        style("[5/6]").bold().dim()
    );

    let credential_config = CredentialConfig::new(
        config.institution_did.clone(),
        config.institution_name.clone(),
        config.revocation_registry.clone(),
        dns_verification,
    );

    let pb = ProgressBar::new(parse_result.records.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("█▓░"),
    );

    let mut successful = 0;
    let mut failed = 0;
    let mut credential_ids: Vec<String> = Vec::new();
    let mut import_errors: Vec<ImportErrorRecord> = Vec::new();

    // Convert parse errors
    for error in &parse_result.errors {
        import_errors.push(ImportErrorRecord {
            row: error.row,
            field: error.field.clone(),
            message: error.message.clone(),
            code: error.code.clone(),
        });
    }

    for (index, record) in parse_result.records.iter().enumerate() {
        let revocation_index = index as u32;

        match build_credential(record, &credential_config, revocation_index, &batch_id) {
            Ok(credential) => {
                if !config.dry_run {
                    // Write credential to file
                    let filename = format!(
                        "{}_{}.json",
                        record.student_id,
                        credential.id.replace("urn:uuid:", "")
                    );
                    let filepath = config.output_dir.join(&filename);

                    let file = File::create(&filepath)?;
                    serde_json::to_writer_pretty(file, &credential)?;

                    credential_ids.push(credential.id.clone());
                }
                successful += 1;
            }
            Err(e) => {
                import_errors.push(ImportErrorRecord {
                    row: record.row_number,
                    field: "credential_generation".to_string(),
                    message: e.to_string(),
                    code: "GENERATION_ERROR".to_string(),
                });
                failed += 1;
            }
        }

        pb.inc(1);
    }

    pb.finish_with_message("Done");

    // Step 6: Write batch summary
    println!();
    println!(
        "{} Finalizing import...",
        style("[6/6]").bold().dim()
    );

    let batch = ImportBatch {
        batch_id: batch_id.clone(),
        institution_did: config.institution_did.clone(),
        institution_name: config.institution_name.clone(),
        source_file: config.csv_path.display().to_string(),
        source_hash,
        import_timestamp: Utc::now().to_rfc3339(),
        total_records: parse_result.total_rows,
        successful,
        failed: failed + parse_result.errors.len(),
        dry_run: config.dry_run,
        credentials: credential_ids,
        errors: import_errors,
    };

    if !config.dry_run {
        let batch_file = config.output_dir.join(format!("{}_summary.json", batch_id));
        let file = File::create(&batch_file)?;
        serde_json::to_writer_pretty(file, &batch)?;
        println!("  Batch summary: {}", batch_file.display());
    }

    // Print final summary
    let elapsed = start_time.elapsed();

    println!();
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!("{}", style("                    IMPORT SUMMARY").bold());
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!();
    println!(
        "  Batch ID:          {}",
        style(&batch_id).green().bold()
    );
    println!(
        "  Institution:       {}",
        config.institution_name
    );
    println!("  DID:               {}", config.institution_did);
    println!();
    println!(
        "  Total Records:     {}",
        style(parse_result.total_rows).bold()
    );
    println!(
        "  Successful:        {}",
        style(successful).green().bold()
    );
    println!(
        "  Failed:            {}",
        style(batch.failed).red().bold()
    );
    println!();
    println!(
        "  Elapsed Time:      {:.2}s",
        elapsed.as_secs_f64()
    );
    println!(
        "  Mode:              {}",
        if config.dry_run {
            style("DRY RUN").yellow()
        } else {
            style("PRODUCTION").green()
        }
    );
    println!();

    if batch.failed > 0 {
        println!(
            "{}",
            style("  ⚠ Some records failed. Check batch summary for details.").yellow()
        );
    }

    if config.dry_run {
        println!(
            "{}",
            style("  ℹ Run without --dry-run to create actual credentials.").dim()
        );
    } else {
        println!(
            "  Credentials saved to: {}",
            style(config.output_dir.display()).cyan()
        );
    }

    println!();

    Ok(())
}

/// Show batch status
pub async fn show_batch_status(batch_id: &str) -> Result<()> {
    println!("Batch status for: {}", batch_id);
    println!();
    println!(
        "{}",
        style("Note: Full batch status requires Holochain conductor connection.").dim()
    );
    println!(
        "{}",
        style("Looking for local batch summary files...").dim()
    );

    // Look for local batch summary
    let patterns = [
        format!("./{}_summary.json", batch_id),
        format!("./credentials/{}_summary.json", batch_id),
    ];

    for pattern in &patterns {
        if let Ok(file) = File::open(pattern) {
            let batch: ImportBatch = serde_json::from_reader(file)?;
            println!();
            println!("Found batch summary: {}", pattern);
            println!("  Institution: {}", batch.institution_name);
            println!("  Timestamp:   {}", batch.import_timestamp);
            println!("  Total:       {}", batch.total_records);
            println!("  Successful:  {}", batch.successful);
            println!("  Failed:      {}", batch.failed);
            println!("  Dry Run:     {}", batch.dry_run);
            return Ok(());
        }
    }

    println!();
    println!("Batch summary not found locally.");
    println!(
        "To check on-chain status, use the Holochain admin API or Observatory."
    );

    Ok(())
}
