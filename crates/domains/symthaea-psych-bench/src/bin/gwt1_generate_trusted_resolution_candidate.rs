// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trusted-workflow adapter for GWT-1 final-resolution candidate generation.
//!
//! This binary executes no candidate cognitive managers. It consumes persisted
//! direct evidence plus a separately attested promotion capsule, delegates all
//! scientific/authority semantics to the library, and writes deterministic JSON
//! candidate components. The surrounding final workflow is responsible for
//! independently verifying upstream package profiles and separately attesting
//! the complete deterministic final archive.

#[cfg(not(feature = "trusted-resolution-authority"))]
fn main() {
    eprintln!(
        "gwt1_generate_trusted_resolution_candidate requires --features trusted-resolution-authority"
    );
    std::process::exit(2);
}

#[cfg(feature = "trusted-resolution-authority")]
fn main() {
    if let Err(error) = trusted_main() {
        eprintln!("trusted GWT-1 final-resolution candidate generation failed: {error}");
        std::process::exit(2);
    }
}

#[cfg(feature = "trusted-resolution-authority")]
fn trusted_main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;
    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};

    use symthaea_psych_bench::benchmarks::butlin::generate_gwt1_trusted_resolution_candidate_v1;

    fn required_path(name: &str) -> Result<PathBuf, io::Error> {
        env::var_os(name).map(PathBuf::from).ok_or_else(|| {
            io::Error::other(format!("missing required environment variable {name}"))
        })
    }

    fn write_json<T: serde::Serialize>(
        path: &Path,
        value: &T,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = serde_json::to_vec(value)?;
        fs::write(path, bytes)?;
        Ok(())
    }

    let direct_evidence_dir = required_path("SYMTHAEA_GWT1_DIRECT_EVIDENCE_DIR")?;
    let promotion_capsule = required_path("GWT1_CAUSAL_PROMOTION_CAPSULE")?;
    let promotion_attestation_bundle = required_path("GWT1_CAUSAL_PROMOTION_ATTESTATION_BUNDLE")?;
    let gh_executable = required_path("GWT1_TRUSTED_GH_PATH")?;
    let sha256sum_executable = required_path("GWT1_TRUSTED_SHA256SUM_PATH")?;
    let gh_config_dir = required_path("GWT1_TRUSTED_GH_CONFIG_DIR")?;
    let output_dir = required_path("GWT1_FINAL_RESOLUTION_OUTPUT_DIR")?;

    if !output_dir.is_absolute() {
        return Err(io::Error::other("final-resolution output directory must be absolute").into());
    }
    if output_dir.exists() {
        if !output_dir.is_dir() || fs::read_dir(&output_dir)?.next().is_some() {
            return Err(io::Error::other("final-resolution output directory must be empty").into());
        }
    } else {
        fs::create_dir_all(&output_dir)?;
    }

    let candidate = generate_gwt1_trusted_resolution_candidate_v1(
        &direct_evidence_dir,
        &promotion_capsule,
        &promotion_attestation_bundle,
        &gh_executable,
        &sha256sum_executable,
        &gh_config_dir,
    )?;

    write_json(&output_dir.join("base_report.json"), &candidate.base_report)?;
    write_json(
        &output_dir.join("resolved_view_v2.json"),
        &candidate.resolved_view,
    )?;
    write_json(
        &output_dir.join("gwt1_evidence_disposition.json"),
        &candidate.disposition,
    )?;
    write_json(&output_dir.join("resolution_candidate.json"), &candidate)?;
    fs::write(
        output_dir.join("promotion_internal_verification.json"),
        candidate.promotion_attestation_verification_bytes(),
    )?;

    Ok(())
}
