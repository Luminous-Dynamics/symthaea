// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate a complete Humanoid release-evidence document.

use std::path::PathBuf;

use symthaea_humanoid::{HumanoidControlReleaseEvidence, certify_humanoid_control_release};

fn main() {
    let mut args = std::env::args_os().skip(1);
    let Some(evidence_path) = args.next().map(PathBuf::from) else {
        eprintln!("usage: symthaea_control_release <evidence.json> <certificate.json>");
        std::process::exit(2);
    };
    let Some(certificate_path) = args.next().map(PathBuf::from) else {
        eprintln!("output certificate path is required");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("unexpected extra arguments");
        std::process::exit(2);
    }
    let evidence: HumanoidControlReleaseEvidence = match std::fs::read(&evidence_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    {
        Some(evidence) => evidence,
        None => {
            eprintln!("failed to read or decode {}", evidence_path.display());
            std::process::exit(2);
        }
    };
    let certificate = certify_humanoid_control_release(&evidence);
    let output = match serde_json::to_vec_pretty(&certificate) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("failed to encode release certificate: {error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = std::fs::write(&certificate_path, output) {
        eprintln!("failed to write {}: {error}", certificate_path.display());
        std::process::exit(2);
    }
    if !certificate.accepted {
        for failure in &certificate.failures {
            eprintln!("release gate: {failure}");
        }
        std::process::exit(1);
    }
}
