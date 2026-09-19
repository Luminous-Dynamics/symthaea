// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use symthaea_materials_historical_extraction::oqmd_v17_fe_co_zr_protocol;
use symthaea_materials_oqmd_acquisition::{
    OqmdHttpsAcquisitionProfile, bind_successful_acquisition_receipt,
    execute_https_acquisition,
};

#[derive(Debug, Parser)]
#[command(name = "symthaea-materials-oqmd-acquisition")]
#[command(about = "Evidence-bearing HTTPS acquisition for the frozen OQMD v1.7 benchmark")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Execute one preregistered OQMD v1.7 HTTPS acquisition.
    AcquireOqmdV17 {
        /// Preregistered acquisition profile JSON.
        #[arg(long)]
        profile: PathBuf,
        /// New absolute destination for the compressed archive.
        #[arg(long)]
        destination: PathBuf,
        /// New absolute destination for exact response-header bytes.
        #[arg(long)]
        response_headers: PathBuf,
        /// Canonical UTC timestamp `YYYY-MM-DDTHH:MM:SSZ` supplied by the orchestrator.
        #[arg(long)]
        acquired_at_utc: String,
        /// Create-new path for the complete attempt evidence, including failures.
        #[arg(long)]
        attempt_output: PathBuf,
        /// Create-new path for the successful MAG-DATA-006 receipt.
        #[arg(long)]
        receipt_output: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::AcquireOqmdV17 {
            profile,
            destination,
            response_headers,
            acquired_at_utc,
            attempt_output,
            receipt_output,
        } => {
            let protocol = oqmd_v17_fe_co_zr_protocol();
            let profile: OqmdHttpsAcquisitionProfile = read_json(&profile)?;
            let attempt = execute_https_acquisition(
                &protocol,
                &profile,
                &destination,
                &response_headers,
            )
            .context("OQMD HTTPS acquisition execution failed before a trustworthy attempt could be assembled")?;

            // Preserve the attempt before trying to promote it. A failed transfer is still
            // useful evidence and must not disappear because receipt promotion fails.
            write_new_json(&attempt_output, &attempt)?;

            let receipt = bind_successful_acquisition_receipt(
                &protocol,
                &profile,
                &attempt,
                &acquired_at_utc,
            )
            .context("acquisition attempt was preserved but did not qualify for a MAG-DATA-006 receipt")?;
            write_new_json(&receipt_output, &receipt)?;
        }
    }
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("failed to parse JSON from {}", path.display()))
}

fn write_new_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "failed to create new evidence file {} (existing evidence is never overwritten)",
                path.display()
            )
        })?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .with_context(|| format!("failed to serialize {}", path.display()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
