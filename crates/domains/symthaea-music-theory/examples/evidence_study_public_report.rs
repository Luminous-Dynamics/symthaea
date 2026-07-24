// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Export the exact small-cell-suppressed public study projection from a bundle.

use std::error::Error;
use std::path::{Path, PathBuf};

use symthaea_music_theory::{
    CalibrationEvidenceBundle, audit_calibration_evidence_bundle,
    audit_calibration_study_public_summary,
};

fn main() -> Result<(), Box<dyn Error>> {
    let (bundle_path, write_path) = parse_arguments()?;
    let bundle: CalibrationEvidenceBundle = read_json(&bundle_path)?;
    let bundle_audit = audit_calibration_evidence_bundle(&bundle);
    if !bundle_audit.valid() {
        return Err(invalid_input(format!(
            "bundle failed audit with {} issues",
            bundle_audit.issues.len()
        ))
        .into());
    }
    let summary = &bundle.diagnostics.study_public_summary;
    let summary_audit =
        audit_calibration_study_public_summary(&bundle.diagnostics.study_judgment_links, summary);
    if !summary_audit.valid() {
        return Err(invalid_input(format!(
            "public study summary failed audit with {} issues",
            summary_audit.issues.len()
        ))
        .into());
    }
    let bytes = serde_json::to_vec_pretty(summary)?;
    if let Some(path) = write_path {
        atomic_bytes(&path, &bytes)?;
    } else {
        println!("{}", String::from_utf8(bytes)?);
    }
    Ok(())
}

fn parse_arguments() -> Result<(PathBuf, Option<PathBuf>), Box<dyn Error>> {
    let mut bundle = None;
    let mut write = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(next_value(&mut arguments, &argument)?)),
            "--write" => write = Some(PathBuf::from(next_value(&mut arguments, &argument)?)),
            "--help" | "-h" => {
                println!("usage: evidence_study_public_report --bundle PATH [--write PATH]");
                std::process::exit(0);
            }
            other => return Err(invalid_input(format!("unknown argument: {other}")).into()),
        }
    }
    Ok((
        bundle.ok_or_else(|| invalid_input("--bundle is required"))?,
        write,
    ))
}

fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    arguments
        .next()
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")).into())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        error
    })?;
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}
