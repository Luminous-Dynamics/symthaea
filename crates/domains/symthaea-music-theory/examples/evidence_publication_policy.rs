// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build, audit, or evaluate a governance-export publication policy.

use std::error::Error;
use std::path::{Path, PathBuf};

use symthaea_music_theory::{
    CalibrationGovernanceExport, CalibrationPublicationPolicy,
    audit_calibration_publication_policy, evaluate_calibration_publication,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    match arguments.command.as_str() {
        "build" => {
            let policy = match arguments.profile.as_deref().unwrap_or("release") {
                "release" => CalibrationPublicationPolicy::release(),
                "auditor" => CalibrationPublicationPolicy::auditor(),
                other => {
                    return Err(invalid_input(format!("unknown policy profile: {other}")).into());
                }
            };
            write_json(arguments.write.as_deref(), &policy)?;
        }
        "audit" => {
            let path = arguments
                .policy
                .ok_or_else(|| invalid_input("audit requires --policy"))?;
            let policy: CalibrationPublicationPolicy = read_json(&path)?;
            let valid = audit_calibration_publication_policy(&policy);
            write_json(
                arguments.write.as_deref(),
                &serde_json::json!({
                    "valid": valid,
                    "policy_sha256": policy.policy_sha256,
                }),
            )?;
            if !valid {
                std::process::exit(2);
            }
        }
        "evaluate" => {
            let policy_path = arguments
                .policy
                .ok_or_else(|| invalid_input("evaluate requires --policy"))?;
            let export_path = arguments
                .export
                .ok_or_else(|| invalid_input("evaluate requires --export"))?;
            let policy: CalibrationPublicationPolicy = read_json(&policy_path)?;
            let export: CalibrationGovernanceExport = read_json(&export_path)?;
            let decision = evaluate_calibration_publication(&export, &policy);
            write_json(arguments.write.as_deref(), &decision)?;
            if !decision.accepted {
                std::process::exit(2);
            }
        }
        other => return Err(invalid_input(format!("unknown command: {other}")).into()),
    }
    Ok(())
}

struct Arguments {
    command: String,
    profile: Option<String>,
    policy: Option<PathBuf>,
    export: Option<PathBuf>,
    write: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = std::env::args().skip(1);
        let command = values
            .next()
            .ok_or_else(|| invalid_input("command is required: build, audit, or evaluate"))?;
        let mut profile = None;
        let mut policy = None;
        let mut export = None;
        let mut write = None;
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--profile" => profile = Some(next_value(&mut values, &argument)?),
                "--policy" => policy = Some(PathBuf::from(next_value(&mut values, &argument)?)),
                "--export" => export = Some(PathBuf::from(next_value(&mut values, &argument)?)),
                "--write" => write = Some(PathBuf::from(next_value(&mut values, &argument)?)),
                "--help" | "-h" => {
                    println!(
                        "usage:\n  evidence_publication_policy build [--profile release|auditor] [--write PATH]\n  evidence_publication_policy audit --policy PATH [--write PATH]\n  evidence_publication_policy evaluate --policy PATH --export PATH [--write PATH]"
                    );
                    std::process::exit(0);
                }
                other => return Err(invalid_input(format!("unknown argument: {other}")).into()),
            }
        }
        Ok(Self {
            command,
            profile,
            policy,
            export,
            write,
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
