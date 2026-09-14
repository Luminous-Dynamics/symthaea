// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trusted-workflow adapter for the GWT-1 verified report projection candidate.
//!
//! This binary is deliberately not an authority boundary. It consumes bytes
//! that the surrounding workflow has already cryptographically verified and
//! bounded-admitted, reconstructs the scientific state, and writes a compact
//! Serialize-only projection. External authority exists only after the trusted
//! consumer workflow separately attests the deterministic report archive.

#[cfg(not(feature = "trusted-resolution-consumer"))]
fn main() {
    eprintln!(
        "gwt1_generate_verified_report_candidate requires --features trusted-resolution-consumer"
    );
    std::process::exit(2);
}

#[cfg(feature = "trusted-resolution-consumer")]
fn main() {
    if let Err(error) = trusted_main() {
        eprintln!("trusted GWT-1 report projection generation failed: {error}");
        std::process::exit(2);
    }
}

#[cfg(feature = "trusted-resolution-consumer")]
fn trusted_main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;
    use std::fs::OpenOptions;
    use std::io::{self, Write};
    use std::path::PathBuf;

    use symthaea_psych_bench::benchmarks::butlin::
        reconstruct_gwt1_verified_report_projection_candidate_v1;

    fn required_path(name: &str) -> Result<PathBuf, io::Error> {
        env::var_os(name).map(PathBuf::from).ok_or_else(|| {
            io::Error::other(format!("missing required environment variable {name}"))
        })
    }

    fn required_string(name: &str) -> Result<String, io::Error> {
        env::var(name)
            .map_err(|_| io::Error::other(format!("missing required environment variable {name}")))
    }

    let admitted_final_root = required_path("GWT1_ADMITTED_FINAL_ROOT")?;
    let admitted_direct_evidence_dir = required_path("GWT1_ADMITTED_DIRECT_EVIDENCE_DIR")?;
    let final_archive_sha256 = required_string("GWT1_FINAL_ARCHIVE_SHA256")?;
    let gh_executable = required_path("GWT1_TRUSTED_GH_PATH")?;
    let sha256sum_executable = required_path("GWT1_TRUSTED_SHA256SUM_PATH")?;
    let gh_config_dir = required_path("GWT1_TRUSTED_GH_CONFIG_DIR")?;
    let output_path = required_path("GWT1_VERIFIED_REPORT_OUTPUT")?;

    if !output_path.is_absolute() {
        return Err(io::Error::other("verified report output path must be absolute").into());
    }
    let parent = output_path
        .parent()
        .ok_or_else(|| io::Error::other("verified report output path has no parent"))?;
    let parent_metadata = std::fs::symlink_metadata(parent)?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(io::Error::other(
            "verified report output parent must be an existing non-symlink directory",
        )
        .into());
    }

    let projection = reconstruct_gwt1_verified_report_projection_candidate_v1(
        &admitted_final_root,
        &admitted_direct_evidence_dir,
        &final_archive_sha256,
        &gh_executable,
        &sha256sum_executable,
        &gh_config_dir,
    )?;
    let bytes = serde_json::to_vec(&projection)?;

    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)?;
    output.write_all(&bytes)?;
    output.sync_all()?;
    Ok(())
}
