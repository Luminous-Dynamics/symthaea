// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build or audit exact before/after governance receipts.

use std::error::Error;
use std::path::{Path, PathBuf};

use symthaea_music_theory::{
    CalibrationEvidenceBundle, CalibrationGovernanceAction, CalibrationGovernanceReceipt,
    CalibrationStudyPrivacyRelease, CalibrationStudyRetentionReport,
    CalibrationStudyWithdrawalResult, audit_calibration_governance_receipt,
    audit_calibration_governance_receipt_transition, build_calibration_governance_receipt,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    if let Some(receipt_path) = arguments.audit {
        let receipt: CalibrationGovernanceReceipt = read_json(&receipt_path)?;
        let report = match (arguments.before, arguments.after) {
            (Some(before), Some(after)) => {
                let before: CalibrationEvidenceBundle = read_json(&before)?;
                let after: CalibrationEvidenceBundle = read_json(&after)?;
                audit_calibration_governance_receipt_transition(&before, &after, &receipt)
            }
            (None, None) => audit_calibration_governance_receipt(&receipt),
            _ => {
                return Err(invalid_input("--before and --after must be supplied together").into());
            }
        };
        write_json(arguments.write.as_deref(), &report)?;
        if !report.valid() {
            std::process::exit(2);
        }
        return Ok(());
    }
    let before_path = arguments
        .before
        .ok_or_else(|| invalid_input("--before is required"))?;
    let after_path = arguments
        .after
        .ok_or_else(|| invalid_input("--after is required"))?;
    let evidence_path = arguments
        .evidence
        .ok_or_else(|| invalid_input("--evidence is required"))?;
    let kind = arguments
        .kind
        .ok_or_else(|| invalid_input("--kind is required"))?;
    let before: CalibrationEvidenceBundle = read_json(&before_path)?;
    let after: CalibrationEvidenceBundle = read_json(&after_path)?;
    let (effective_epoch, action) = match kind.as_str() {
        "withdrawal" => {
            let result: CalibrationStudyWithdrawalResult = read_json(&evidence_path)?;
            let tombstone = &result.tombstone;
            (
                tombstone.effective_epoch,
                CalibrationGovernanceAction::StudyResponseWithdrawal {
                    signed_response_sha256: tombstone.signed_response_sha256.clone(),
                    tombstone_sha256: tombstone.tombstone_sha256.clone(),
                    reason: tombstone.reason,
                    removed_judgments: tombstone.removed_judgments,
                },
            )
        }
        "retention" => {
            let report: CalibrationStudyRetentionReport = read_json(&evidence_path)?;
            (
                report.current_epoch,
                CalibrationGovernanceAction::StudyRetentionEnforcement {
                    report_sha256: report.report_sha256,
                    removed_tombstone_sha256s: report
                        .removals
                        .iter()
                        .map(|removal| removal.tombstone_sha256.clone())
                        .collect(),
                    unknown_attachment_epochs: report.unknown_attachment_epochs.len(),
                    future_attachment_epochs: report.future_attachment_epochs.len(),
                },
            )
        }
        "privacy" => {
            let release: CalibrationStudyPrivacyRelease = read_json(&evidence_path)?;
            (
                release.release_epoch,
                CalibrationGovernanceAction::StudyPrivacyRelease {
                    query_id: release.query_id,
                    release_sha256: release.release_sha256,
                    epsilon_micros: release.epsilon_micros,
                    delta_parts_per_trillion: release.delta_parts_per_trillion,
                },
            )
        }
        other => return Err(invalid_input(format!("unknown governance kind: {other}")).into()),
    };
    let receipt = build_calibration_governance_receipt(
        arguments.sequence.unwrap_or(0),
        &before,
        &after,
        effective_epoch,
        action,
    );
    let audit = audit_calibration_governance_receipt_transition(&before, &after, &receipt);
    if !audit.valid() {
        return Err(invalid_input(format!(
            "receipt transition audit failed with {} issues",
            audit.issues.len()
        ))
        .into());
    }
    write_json(arguments.write.as_deref(), &receipt)
}

struct Arguments {
    kind: Option<String>,
    before: Option<PathBuf>,
    after: Option<PathBuf>,
    evidence: Option<PathBuf>,
    sequence: Option<u64>,
    audit: Option<PathBuf>,
    write: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = Self {
            kind: None,
            before: None,
            after: None,
            evidence: None,
            sequence: None,
            audit: None,
            write: None,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--kind" => values.kind = Some(next_value(&mut arguments, &argument)?),
                "--before" => {
                    values.before = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--after" => {
                    values.after = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--evidence" => {
                    values.evidence = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--sequence" => {
                    values.sequence = Some(
                        next_value(&mut arguments, &argument)?
                            .parse::<u64>()
                            .map_err(|_| {
                                invalid_input("--sequence requires an unsigned integer")
                            })?,
                    )
                }
                "--audit" => {
                    values.audit = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--write" => {
                    values.write = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--help" | "-h" => {
                    println!(
                        "usage: evidence_governance_receipt --kind withdrawal|retention|privacy --before BUNDLE --after BUNDLE --evidence RESULT [--sequence N] [--write PATH]\n       evidence_governance_receipt --audit RECEIPT [--before BUNDLE --after BUNDLE] [--write PATH]"
                    );
                    std::process::exit(0);
                }
                other => return Err(invalid_input(format!("unknown argument: {other}")).into()),
            }
        }
        Ok(values)
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
