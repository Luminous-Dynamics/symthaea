// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verify externally produced Series 18 federation evidence.
//!
//! Usage:
//!   checkpoint_series18_federation_evaluator \
//!     CAMPAIGN.bin OPERATIONS_PLAN.bin SEALED_RESULTS.bin RESULT_KEY.bin \
//!     SEALED_FEDERATION_PLAN.bin FEDERATION_KEY.bin SEALED_REVOCATIONS.bin \
//!     ALLOCATIONS_DIR LAB_EVIDENCE_DIR LAB_KEYS_DIR VERIFIED_AT_UNIX_SECONDS
//!
//! Federation and lab key files contain exactly 48 bytes: a 16-byte public
//! identifier followed by a 32-byte secret. Lab key filenames are the lowercase
//! hexadecimal key identifier followed by `.key`.

#[cfg(unix)]
use std::fs::OpenOptions;
#[cfg(unix)]
use std::io::Read;
#[cfg(unix)]
use std::path::{Path, PathBuf};

#[cfg(unix)]
use symthaea_vocal_tract::{
    CheckpointOperationalTrustMetrics, CheckpointOperationalTrustRequirements,
    CheckpointPowerLossEvidenceAuthority, CheckpointPowerLossEvidenceKey,
    CheckpointPowerLossEvidenceKeyId, CheckpointPowerLossFederationArtifactKeyId,
    CheckpointPowerLossFederationAuthority, CheckpointPowerLossFederationKey,
    CheckpointPowerLossFederationKeyId, CheckpointPowerLossLabEvidenceAuthority,
    CheckpointPowerLossLabEvidenceKey, CheckpointPowerLossLabEvidenceKeyId,
    MAX_CHECKPOINT_POWER_LOSS_FEDERATION_BYTES, assemble_checkpoint_operational_trust_evidence,
    checkpoint_power_loss_sealed_artifact_digest, decode_checkpoint_power_loss_campaign,
    decode_checkpoint_power_loss_operations_plan,
    inspect_checkpoint_power_loss_federation_artifact,
    verify_and_apply_power_loss_federation_evidence,
};
#[cfg(unix)]
use zeroize::Zeroize;

#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let campaign_path = required_path(&mut arguments, "campaign artifact")?;
    let operations_path = required_path(&mut arguments, "operations-plan artifact")?;
    let result_evidence_path = required_path(&mut arguments, "sealed result evidence")?;
    let result_key_path = required_path(&mut arguments, "result authority key")?;
    let federation_plan_path = required_path(&mut arguments, "sealed federation plan")?;
    let federation_key_path = required_path(&mut arguments, "federation authority key")?;
    let revocations_path = required_path(&mut arguments, "sealed revocation list")?;
    let allocations_dir = required_path(&mut arguments, "allocation directory")?;
    let lab_evidence_dir = required_path(&mut arguments, "lab evidence directory")?;
    let lab_keys_dir = required_path(&mut arguments, "lab key directory")?;
    let verified_at = arguments
        .next()
        .ok_or("missing verification time")?
        .into_string()
        .map_err(|_| "verification time is not UTF-8")?
        .parse::<u64>()?;
    if arguments.next().is_some() {
        return Err("unexpected trailing arguments".into());
    }

    let campaign = decode_checkpoint_power_loss_campaign(&read_public_file(&campaign_path)?)?;
    let operations = decode_checkpoint_power_loss_operations_plan(
        &campaign,
        &read_public_file(&operations_path)?,
    )?;

    let (result_key_id, mut result_key_bytes) = read_private_key_file(&result_key_path)?;
    let result_authority =
        CheckpointPowerLossEvidenceAuthority::new(CheckpointPowerLossEvidenceKey::new(
            CheckpointPowerLossEvidenceKeyId::new(result_key_id)?,
            result_key_bytes,
        )?);
    result_key_bytes.zeroize();
    if result_authority.key_id() != campaign.power_loss_evidence_authority_key_id {
        return Err("result authority key does not match campaign".into());
    }
    let sealed_result_evidence = read_public_file(&result_evidence_path)?;
    let result_evidence = result_authority.open_campaign_evidence(
        &campaign,
        &sealed_result_evidence,
        campaign.power_loss_evidence_authority_key_id,
    )?;
    let sealed_result_evidence_digest =
        checkpoint_power_loss_sealed_artifact_digest(&sealed_result_evidence);

    let (federation_key_id, mut federation_key_bytes) =
        read_private_key_file(&federation_key_path)?;
    let federation_authority =
        CheckpointPowerLossFederationAuthority::new(CheckpointPowerLossFederationKey::new(
            CheckpointPowerLossFederationKeyId::new(federation_key_id)?,
            federation_key_bytes,
        )?);
    federation_key_bytes.zeroize();

    let sealed_federation_plan = read_public_file(&federation_plan_path)?;
    let sealed_revocations = read_public_file(&revocations_path)?;
    let sealed_allocations = read_public_directory(&allocations_dir)?;
    let sealed_lab_evidence = read_public_directory(&lab_evidence_dir)?;
    let mut lab_authorities = Vec::with_capacity(sealed_lab_evidence.len());
    let mut observed_lab_keys = std::collections::HashSet::new();
    for encoded in &sealed_lab_evidence {
        let inspection = inspect_checkpoint_power_loss_federation_artifact(encoded)?;
        let CheckpointPowerLossFederationArtifactKeyId::Lab(expected_key_id) = inspection.key_id
        else {
            return Err("lab evidence directory contains a non-lab artifact".into());
        };
        if !observed_lab_keys.insert(expected_key_id) {
            return Err("duplicate lab evidence key identifier".into());
        }
        let key_path = lab_keys_dir.join(format!("{}.key", hex(&expected_key_id.0)));
        let (observed_key_id, mut key_bytes) = read_private_key_file(&key_path)?;
        if observed_key_id != expected_key_id.0 {
            key_bytes.zeroize();
            return Err("lab key filename and embedded key identifier differ".into());
        }
        lab_authorities.push(CheckpointPowerLossLabEvidenceAuthority::new(
            CheckpointPowerLossLabEvidenceKey::new(
                CheckpointPowerLossLabEvidenceKeyId::new(observed_key_id)?,
                key_bytes,
            )?,
        ));
        key_bytes.zeroize();
    }

    let mut metrics = CheckpointOperationalTrustMetrics::default();
    metrics.power_loss_campaign_exercised = true;
    metrics.power_loss_planned_trials = campaign.trials.len();
    metrics.power_loss_completed_trials = result_evidence.results.len();
    let merged = verify_and_apply_power_loss_federation_evidence(
        &mut metrics,
        &federation_authority,
        &lab_authorities,
        &campaign,
        &operations,
        &result_evidence,
        sealed_result_evidence_digest,
        &sealed_federation_plan,
        &sealed_revocations,
        &sealed_allocations,
        &sealed_lab_evidence,
        verified_at,
    )?;
    let report = assemble_checkpoint_operational_trust_evidence(
        metrics,
        CheckpointOperationalTrustRequirements::series_18_delta(),
    );

    println!("schema={}", report.schema);
    println!("campaign_id={}", hex(&campaign.campaign_id));
    println!(
        "federation_id={}",
        hex(&merged.federation_plan.federation_id.0)
    );
    println!("federation_epoch={}", merged.federation_plan.epoch);
    println!(
        "verified_allocations={}",
        merged.summary.verified_allocations
    );
    println!(
        "verified_lab_bundles={}",
        merged.summary.verified_lab_bundles
    );
    println!("unique_labs={}", merged.summary.unique_labs);
    println!(
        "revocations_checked={}",
        merged.summary.revocation_entries_checked
    );
    println!(
        "maximum_clock_offset_seconds={}",
        merged.summary.maximum_clock_offset_seconds,
    );
    println!(
        "maximum_clock_uncertainty_seconds={}",
        merged.summary.maximum_clock_uncertainty_seconds,
    );
    for gate in &report.gates {
        if gate.required {
            println!(
                "gate={} status={:?} observed={:?} minimum={:?} detail={}",
                gate.name, gate.status, gate.observed, gate.required_minimum, gate.detail,
            );
        }
    }
    println!("series_18_federation_passed={}", report.passed());
    if !report.passed() {
        return Err("Series 18 federation evidence did not satisfy promotion gates".into());
    }
    Ok(())
}

#[cfg(not(unix))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("the Series 18 evaluator currently requires Unix private-file checks".into())
}

#[cfg(unix)]
fn required_path(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    name: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing {name}").into())
}

#[cfg(unix)]
fn read_public_directory(path: &Path) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err("unsafe federation artifact directory".into());
    }
    let mut paths = std::fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    if paths.is_empty() {
        return Err("federation artifact directory is empty".into());
    }
    paths.iter().map(|path| read_public_file(path)).collect()
}

#[cfg(unix)]
fn read_public_file(path: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::os::unix::fs::OpenOptionsExt;

    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let mut file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > MAX_CHECKPOINT_POWER_LOSS_FEDERATION_BYTES as u64
    {
        return Err("unsafe or oversized federation artifact".into());
    }
    let mut encoded = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut encoded)?;
    Ok(encoded)
}

#[cfg(unix)]
fn read_private_key_file(path: &Path) -> Result<([u8; 16], [u8; 32]), Box<dyn std::error::Error>> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};

    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let mut file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.uid() != symthaea_vocal_tract::effective_uid()
        || metadata.permissions().mode() & 0o077 != 0
        || metadata.len() != 48
    {
        return Err("unsafe Series 18 key file".into());
    }
    let mut encoded = [0u8; 48];
    file.read_exact(&mut encoded)?;
    let mut trailing = [0u8; 1];
    if file.read(&mut trailing)? != 0 {
        encoded.zeroize();
        return Err("Series 18 key file contains trailing bytes".into());
    }
    let mut key_id = [0u8; 16];
    key_id.copy_from_slice(&encoded[..16]);
    let mut key = [0u8; 32];
    key.copy_from_slice(&encoded[16..]);
    encoded.zeroize();
    Ok((key_id, key))
}

#[cfg(unix)]
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
