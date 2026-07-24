// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Emit canonical bytes for externally signing a public governance export.

use std::error::Error;
use std::path::{Path, PathBuf};

use symthaea_music_theory::{
    CalibrationGovernanceExport, CalibrationGovernanceExportAttestationPayload,
    audit_calibration_governance_export_attestation_payload,
    build_calibration_governance_export_attestation_payload,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    let export: CalibrationGovernanceExport = read_json(&arguments.export)?;
    if let Some(payload_path) = arguments.audit {
        let payload: CalibrationGovernanceExportAttestationPayload = read_json(&payload_path)?;
        let report = audit_calibration_governance_export_attestation_payload(&export, &payload);
        write_json(arguments.write.as_deref(), &report)?;
        if !report.valid {
            std::process::exit(2);
        }
        return Ok(());
    }
    let payload = build_calibration_governance_export_attestation_payload(&export)?;
    if let Some(bytes_path) = arguments.write_bytes {
        atomic_bytes(&bytes_path, &payload.canonical_bytes())?;
    }
    write_json(arguments.write.as_deref(), &payload)
}

struct Arguments {
    export: PathBuf,
    audit: Option<PathBuf>,
    write: Option<PathBuf>,
    write_bytes: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut export = None;
        let mut audit = None;
        let mut write = None;
        let mut write_bytes = None;
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--export" => export = Some(PathBuf::from(next_value(&mut arguments, &argument)?)),
                "--audit" => audit = Some(PathBuf::from(next_value(&mut arguments, &argument)?)),
                "--write" => write = Some(PathBuf::from(next_value(&mut arguments, &argument)?)),
                "--write-bytes" => {
                    write_bytes = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--help" | "-h" => {
                    println!(
                        "usage: evidence_governance_attestation_payload --export EXPORT [--write PAYLOAD_JSON] [--write-bytes PAYLOAD_BIN]\n       evidence_governance_attestation_payload --export EXPORT --audit PAYLOAD_JSON [--write AUDIT_JSON]"
                    );
                    std::process::exit(0);
                }
                other => return Err(invalid_input(format!("unknown argument: {other}")).into()),
            }
        }
        Ok(Self {
            export: export.ok_or_else(|| invalid_input("--export is required"))?,
            audit,
            write,
            write_bytes,
        })
    }
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

fn write_json<T: serde::Serialize>(path: Option<&Path>, value: &T) -> Result<(), Box<dyn Error>> {
    let bytes = serde_json::to_vec_pretty(value)?;
    match path {
        Some(path) => atomic_bytes(path, &bytes)?,
        None => println!("{}", String::from_utf8(bytes)?),
    }
    Ok(())
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
