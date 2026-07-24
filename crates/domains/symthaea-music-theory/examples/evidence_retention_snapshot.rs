// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Export or audit an identity-free retention-compliance snapshot.

use std::error::Error;
use std::path::{Path, PathBuf};

use symthaea_music_theory::{
    CalibrationEvidenceBundle, CalibrationRetentionComplianceSnapshot,
    CalibrationStudyRetentionPolicy, CalibrationUnknownAttachmentEpochAction,
    audit_calibration_retention_compliance_snapshot,
    audit_calibration_retention_compliance_snapshot_against_bundle,
    build_calibration_retention_compliance_snapshot,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    if let Some(snapshot_path) = arguments.audit {
        let snapshot: CalibrationRetentionComplianceSnapshot = read_json(&snapshot_path)?;
        let report = match arguments.bundle {
            Some(bundle_path) => {
                let bundle: CalibrationEvidenceBundle = read_json(&bundle_path)?;
                audit_calibration_retention_compliance_snapshot_against_bundle(&bundle, &snapshot)
            }
            None => audit_calibration_retention_compliance_snapshot(&snapshot),
        };
        write_json(arguments.write.as_deref(), &report)?;
        if !report.valid() {
            std::process::exit(2);
        }
        return Ok(());
    }
    let bundle_path = arguments
        .bundle
        .ok_or_else(|| invalid_input("--bundle is required when exporting"))?;
    let current_epoch = arguments
        .current_epoch
        .ok_or_else(|| invalid_input("--current-epoch is required"))?;
    let maximum_retention_epochs = arguments
        .maximum_retention_epochs
        .ok_or_else(|| invalid_input("--maximum-retention-epochs is required"))?;
    let mut policy = CalibrationStudyRetentionPolicy::release(maximum_retention_epochs);
    if arguments.reject_unknown {
        policy.unknown_attachment_epoch_action =
            CalibrationUnknownAttachmentEpochAction::RejectEnforcement;
    }
    let bundle: CalibrationEvidenceBundle = read_json(&bundle_path)?;
    let snapshot = build_calibration_retention_compliance_snapshot(&bundle, policy, current_epoch)?;
    write_json(arguments.write.as_deref(), &snapshot)
}

struct Arguments {
    bundle: Option<PathBuf>,
    audit: Option<PathBuf>,
    current_epoch: Option<u64>,
    maximum_retention_epochs: Option<u64>,
    reject_unknown: bool,
    write: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = Self {
            bundle: None,
            audit: None,
            current_epoch: None,
            maximum_retention_epochs: None,
            reject_unknown: false,
            write: None,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--bundle" => {
                    values.bundle = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--audit" => {
                    values.audit = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--current-epoch" => {
                    values.current_epoch = Some(parse_u64(
                        &next_value(&mut arguments, &argument)?,
                        &argument,
                    )?)
                }
                "--maximum-retention-epochs" => {
                    values.maximum_retention_epochs = Some(parse_u64(
                        &next_value(&mut arguments, &argument)?,
                        &argument,
                    )?)
                }
                "--reject-unknown" => values.reject_unknown = true,
                "--write" => {
                    values.write = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--help" | "-h" => {
                    println!(
                        "usage: evidence_retention_snapshot --bundle BUNDLE --current-epoch N --maximum-retention-epochs N [--reject-unknown] [--write PATH]\n       evidence_retention_snapshot --audit SNAPSHOT [--bundle PRIVATE_BUNDLE] [--write PATH]"
                    );
                    std::process::exit(0);
                }
                other => return Err(invalid_input(format!("unknown argument: {other}")).into()),
            }
        }
        Ok(values)
    }
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, Box<dyn Error>> {
    value
        .parse::<u64>()
        .map_err(|_| invalid_input(format!("{flag} requires an unsigned integer")).into())
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
