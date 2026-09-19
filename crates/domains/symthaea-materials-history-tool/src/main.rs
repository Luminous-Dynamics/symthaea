// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::de::DeserializeOwned;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use symthaea_materials_corpus_audit::HistoricalBenchmarkTarget;
use symthaea_materials_historical_extraction::{
    CompactHistoricalCorpus, HistoricalExtractionReceipt, oqmd_v17_fe_co_zr_protocol,
};
use symthaea_materials_historical_import::{
    HistoricalImportExecutionReceipt, HistoricalImportProfile,
};
use symthaea_materials_history_tool::verify_historical_run;
use symthaea_materials_schema_inventory::MySqlSchemaInventory;
use symthaea_materials_snapshot_acquisition::{
    HistoricalSnapshotAcquisitionReceipt, hash_snapshot_stream,
};

#[derive(Debug, Parser)]
#[command(name = "symthaea-materials-history-tool")]
#[command(about = "Offline verifier for frozen historical materials benchmark artifacts")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Stream-hash one archive and emit exact SHA-256 plus byte count.
    HashSnapshot {
        /// Compressed historical archive path.
        archive: PathBuf,
    },
    /// Verify one complete OQMD v1.7 Fe/Co/Zr historical benchmark bundle.
    VerifyOqmdV17 {
        /// Exact compressed `qmdb__v1_7__052025.sql.gz` bytes.
        #[arg(long)]
        archive: PathBuf,
        /// MAG-DATA-006 acquisition receipt JSON.
        #[arg(long)]
        acquisition: PathBuf,
        /// MAG-DATA-008 import profile JSON.
        #[arg(long)]
        import_profile: PathBuf,
        /// MAG-DATA-008 import execution receipt JSON.
        #[arg(long)]
        import_receipt: PathBuf,
        /// MAG-DATA-009 canonical MySQL schema inventory JSON.
        #[arg(long)]
        schema_inventory: PathBuf,
        /// MAG-DATA-003 extraction receipt JSON.
        #[arg(long)]
        extraction_receipt: PathBuf,
        /// MAG-DATA-003 compact historical corpus JSON.
        #[arg(long)]
        compact_corpus: PathBuf,
        /// Historical benchmark target-set JSON array.
        #[arg(long)]
        targets: PathBuf,
        /// Optional create-new output path; stdout when omitted.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::HashSnapshot { archive } => {
            let file = File::open(&archive)
                .with_context(|| format!("failed to open snapshot {}", archive.display()))?;
            let identity = hash_snapshot_stream(BufReader::new(file))
                .context("failed to stream-hash historical snapshot")?;
            serde_json::to_writer_pretty(std::io::stdout().lock(), &identity)
                .context("failed to serialize snapshot identity")?;
            println!();
        }
        Command::VerifyOqmdV17 {
            archive,
            acquisition,
            import_profile,
            import_receipt,
            schema_inventory,
            extraction_receipt,
            compact_corpus,
            targets,
            output,
        } => {
            let protocol = oqmd_v17_fe_co_zr_protocol();
            let acquisition: HistoricalSnapshotAcquisitionReceipt = read_json(&acquisition)?;
            let import_profile: HistoricalImportProfile = read_json(&import_profile)?;
            let import_receipt: HistoricalImportExecutionReceipt = read_json(&import_receipt)?;
            let schema_inventory: MySqlSchemaInventory = read_json(&schema_inventory)?;
            let extraction_receipt: HistoricalExtractionReceipt = read_json(&extraction_receipt)?;
            let compact_corpus: CompactHistoricalCorpus = read_json(&compact_corpus)?;
            let targets: Vec<HistoricalBenchmarkTarget> = read_json(&targets)?;
            let archive_file = File::open(&archive)
                .with_context(|| format!("failed to open snapshot {}", archive.display()))?;
            let verifier_sha = current_executable_sha256()?;

            let verified = verify_historical_run(
                &verifier_sha,
                &protocol,
                BufReader::new(archive_file),
                &acquisition,
                &import_profile,
                &import_receipt,
                &schema_inventory,
                &extraction_receipt,
                &compact_corpus,
                &targets,
            )
            .context("historical run verification failed")?;

            match output {
                Some(path) => {
                    let file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&path)
                        .with_context(|| {
                            format!(
                                "failed to create new output {} (existing evidence is never overwritten)",
                                path.display()
                            )
                        })?;
                    let mut writer = BufWriter::new(file);
                    serde_json::to_writer_pretty(&mut writer, &verified)
                        .context("failed to serialize verified historical run")?;
                    writer.write_all(b"\n")?;
                    writer.flush()?;
                }
                None => {
                    serde_json::to_writer_pretty(std::io::stdout().lock(), &verified)
                        .context("failed to serialize verified historical run")?;
                    println!();
                }
            }
        }
    }
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("failed to parse JSON from {}", path.display()))
}

fn current_executable_sha256() -> Result<String> {
    let path = std::env::current_exe().context("failed to resolve current verifier executable")?;
    let file = File::open(&path)
        .with_context(|| format!("failed to open verifier executable {}", path.display()))?;
    let identity = hash_snapshot_stream(BufReader::new(file))
        .context("failed to hash verifier executable")?;
    Ok(identity.sha256)
}
