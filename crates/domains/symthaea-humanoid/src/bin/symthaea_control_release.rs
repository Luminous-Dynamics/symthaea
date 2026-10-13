// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate a complete Humanoid release-evidence document.

use std::io::{self, Read};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use symthaea_humanoid::{HumanoidControlReleaseEvidence, certify_humanoid_control_release};

const MAX_EVIDENCE_BUNDLE_BYTES: usize = 256 * 1024 * 1024;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let Some(evidence_path) = args.next().map(PathBuf::from) else {
        eprintln!(
            concat!(
                "usage: symthaea_control_release <evidence.json> <certificate.json> ",
                "<model-artifact> <controller-artifact> <test-harness-artifact> ",
                "<scenario-config-artifact> <oracle-worker-artifact> ",
                "<trusted-evidence-bundle-sha256> <trusted-manifest-blake3> ",
                "<trusted-oracle-worker-sha256>"
            )
        );
        std::process::exit(2);
    };
    let Some(certificate_path) = args.next().map(PathBuf::from) else {
        eprintln!("output certificate path is required");
        std::process::exit(2);
    };
    let Some(model_artifact_path) = args.next().map(PathBuf::from) else {
        eprintln!("model artifact path is required");
        std::process::exit(2);
    };
    let Some(controller_artifact_path) = args.next().map(PathBuf::from) else {
        eprintln!("controller artifact path is required");
        std::process::exit(2);
    };
    let Some(test_harness_artifact_path) = args.next().map(PathBuf::from) else {
        eprintln!("test harness artifact path is required");
        std::process::exit(2);
    };
    let Some(scenario_config_artifact_path) = args.next().map(PathBuf::from) else {
        eprintln!("scenario configuration artifact path is required");
        std::process::exit(2);
    };
    let Some(oracle_worker_path) = args.next().map(PathBuf::from) else {
        eprintln!("oracle worker artifact path is required");
        std::process::exit(2);
    };
    let Some(trusted_evidence_bundle_sha256) =
        args.next().and_then(|value| value.into_string().ok())
    else {
        eprintln!("trusted evidence-bundle SHA-256 digest is required");
        std::process::exit(2);
    };
    let Some(trusted_manifest_digest) = args.next().and_then(|value| value.into_string().ok()) else {
        eprintln!("trusted scenario-manifest BLAKE3 digest is required");
        std::process::exit(2);
    };
    let Some(trusted_oracle_executable_sha256) =
        args.next().and_then(|value| value.into_string().ok())
    else {
        eprintln!("trusted oracle-worker SHA-256 digest is required");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("unexpected extra arguments");
        std::process::exit(2);
    }
    let evidence_file = match std::fs::File::open(&evidence_path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("failed to open evidence JSON {}: {error}", evidence_path.display());
            std::process::exit(2);
        }
    };
    let evidence_bytes = match read_bounded(evidence_file, MAX_EVIDENCE_BUNDLE_BYTES) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read bounded evidence JSON {}: {error}", evidence_path.display());
            std::process::exit(2);
        }
    };
    let actual_evidence_bundle_sha256 = format!("{:x}", Sha256::digest(&evidence_bytes));
    let evidence: HumanoidControlReleaseEvidence =
        match serde_json::from_slice(&evidence_bytes) {
            Ok(evidence) => evidence,
            Err(error) => {
                eprintln!("failed to decode evidence JSON {}: {error}", evidence_path.display());
                std::process::exit(2);
            }
        };
    let actual_model_sha256 = sha256_file(&model_artifact_path, "model");
    let actual_controller_sha256 = sha256_file(&controller_artifact_path, "controller");
    let actual_test_harness_sha256 = sha256_file(&test_harness_artifact_path, "test harness");
    let actual_scenario_config_sha256 =
        sha256_file(&scenario_config_artifact_path, "scenario configuration");
    let actual_oracle_sha256 = sha256_file(&oracle_worker_path, "oracle worker");
    let certificate = certify_humanoid_control_release(
        &evidence,
        &actual_evidence_bundle_sha256,
        &trusted_evidence_bundle_sha256,
        &actual_model_sha256,
        &actual_controller_sha256,
        &actual_test_harness_sha256,
        &actual_scenario_config_sha256,
        &actual_oracle_sha256,
        &trusted_manifest_digest,
        &trusted_oracle_executable_sha256,
    );
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

fn read_bounded<R: Read>(reader: R, maximum_bytes: usize) -> io::Result<Vec<u8>> {
    let read_limit = u64::try_from(maximum_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut reader = reader.take(read_limit);
    let mut bytes = Vec::with_capacity(maximum_bytes.min(64 * 1024));
    reader.read_to_end(&mut bytes)?;
    if bytes.len() > maximum_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("input exceeds the {maximum_bytes}-byte release limit"),
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn evidence_reader_enforces_limit_without_overreading_the_bundle() {
        assert_eq!(read_bounded(Cursor::new(b"1234"), 4).unwrap(), b"1234".to_vec());
        assert_eq!(
            read_bounded(Cursor::new(b"12345"), 4)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        assert!(read_bounded(Cursor::new(b""), 0).unwrap().is_empty());
    }
}

fn sha256_file(path: &Path, label: &str) -> String {
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("failed to open {label} artifact {}: {error}", path.display());
            std::process::exit(2);
        }
    };
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) => {
                eprintln!("failed to hash {label} artifact {}: {error}", path.display());
                std::process::exit(2);
            }
        };
        hasher.update(&buffer[..read]);
    }
    format!("{:x}", hasher.finalize())
}
