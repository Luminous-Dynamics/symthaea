// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verify externally produced Series 19 federation lifecycle evidence.
//!
//! Usage:
//!   checkpoint_series19_lifecycle_evaluator \
//!     PRIOR_CAMPAIGN.bin PRIOR_OPERATIONS.bin PRIOR_SEALED_PLAN.bin \
//!     PRIOR_FEDERATION_KEY.bin PRIOR_LAB_KEYS_DIR PRIOR_MERGES_DIR \
//!     NEXT_CAMPAIGN.bin NEXT_OPERATIONS.bin NEXT_SEALED_PLAN.bin \
//!     NEXT_FEDERATION_KEY.bin NEXT_LAB_KEYS_DIR CURRENT_MERGE.bin \
//!     SEALED_LIFECYCLE_EVIDENCE.bin

#[cfg(unix)]
use std::collections::HashSet;
#[cfg(unix)]
use std::fs::OpenOptions;
#[cfg(unix)]
use std::io::Read;
#[cfg(unix)]
use std::path::{Path, PathBuf};

#[cfg(unix)]
use symthaea_vocal_tract::{
    CheckpointOperationalTrustMetrics, CheckpointOperationalTrustRequirements,
    CheckpointPowerLossFederationAuthority, CheckpointPowerLossFederationKey,
    CheckpointPowerLossFederationKeyId, CheckpointPowerLossLabEvidenceAuthority,
    CheckpointPowerLossLabEvidenceKey, CheckpointPowerLossLabEvidenceKeyId,
    MAX_CHECKPOINT_POWER_LOSS_FEDERATION_BYTES, assemble_checkpoint_operational_trust_evidence,
    decode_checkpoint_power_loss_campaign, decode_checkpoint_power_loss_federation_merge,
    decode_checkpoint_power_loss_operations_plan,
    verify_and_apply_power_loss_federation_lifecycle_evidence,
};
#[cfg(unix)]
use zeroize::Zeroize;

#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let prior_campaign_path = required_path(&mut arguments, "prior campaign")?;
    let prior_operations_path = required_path(&mut arguments, "prior operations plan")?;
    let prior_plan_path = required_path(&mut arguments, "prior sealed federation plan")?;
    let prior_federation_key_path = required_path(&mut arguments, "prior federation key")?;
    let prior_lab_keys_dir = required_path(&mut arguments, "prior lab key directory")?;
    let prior_merges_dir = required_path(&mut arguments, "prior merge directory")?;
    let next_campaign_path = required_path(&mut arguments, "next campaign")?;
    let next_operations_path = required_path(&mut arguments, "next operations plan")?;
    let next_plan_path = required_path(&mut arguments, "next sealed federation plan")?;
    let next_federation_key_path = required_path(&mut arguments, "next federation key")?;
    let next_lab_keys_dir = required_path(&mut arguments, "next lab key directory")?;
    let current_merge_path = required_path(&mut arguments, "current merge")?;
    let lifecycle_path = required_path(&mut arguments, "sealed lifecycle evidence")?;
    if arguments.next().is_some() {
        return Err("unexpected trailing arguments".into());
    }

    let prior_campaign =
        decode_checkpoint_power_loss_campaign(&read_public_file(&prior_campaign_path)?)?;
    let prior_operations = decode_checkpoint_power_loss_operations_plan(
        &prior_campaign,
        &read_public_file(&prior_operations_path)?,
    )?;
    let next_campaign =
        decode_checkpoint_power_loss_campaign(&read_public_file(&next_campaign_path)?)?;
    let next_operations = decode_checkpoint_power_loss_operations_plan(
        &next_campaign,
        &read_public_file(&next_operations_path)?,
    )?;

    let prior_authority = read_federation_authority(&prior_federation_key_path)?;
    let next_authority = read_federation_authority(&next_federation_key_path)?;
    let prior_federation = prior_authority.open_plan(
        &prior_campaign,
        &prior_operations,
        &read_public_file(&prior_plan_path)?,
    )?;
    let next_federation = next_authority.open_plan(
        &next_campaign,
        &next_operations,
        &read_public_file(&next_plan_path)?,
    )?;

    let sealed_lifecycle = read_public_file(&lifecycle_path)?;
    let lifecycle = next_authority.open_lifecycle_evidence(&sealed_lifecycle)?;
    let mut prior_ids = HashSet::new();
    let mut next_ids = HashSet::new();
    for succession in &lifecycle.lab_successions {
        if !prior_ids.insert(succession.succession.prior_lab_evidence_key_id)
            || !next_ids.insert(succession.succession.next_lab_evidence_key_id)
        {
            return Err("duplicate lifecycle lab succession key".into());
        }
    }
    let prior_lab_authorities = load_lab_authorities(&prior_lab_keys_dir, &prior_ids)?;
    let next_lab_authorities = load_lab_authorities(&next_lab_keys_dir, &next_ids)?;

    let prior_merges = read_public_directory(&prior_merges_dir)?
        .iter()
        .map(|encoded| decode_checkpoint_power_loss_federation_merge(encoded))
        .collect::<Result<Vec<_>, _>>()?;
    let current_merge =
        decode_checkpoint_power_loss_federation_merge(&read_public_file(&current_merge_path)?)?;

    let mut metrics = CheckpointOperationalTrustMetrics::default();
    let summary = verify_and_apply_power_loss_federation_lifecycle_evidence(
        &mut metrics,
        &prior_authority,
        &next_authority,
        &prior_lab_authorities,
        &next_lab_authorities,
        &prior_campaign,
        &prior_operations,
        &prior_federation,
        &next_campaign,
        &next_operations,
        &next_federation,
        &prior_merges,
        &current_merge,
        &sealed_lifecycle,
    )?;
    let report = assemble_checkpoint_operational_trust_evidence(
        metrics,
        CheckpointOperationalTrustRequirements::series_19_delta(),
    );

    println!("schema={}", report.schema);
    println!("federation_id={}", hex(&summary.federation_id.0));
    println!("prior_epoch={}", summary.prior_epoch);
    println!("current_epoch={}", summary.current_epoch);
    println!("transition_sequence={}", summary.transition_sequence);
    println!(
        "authority_rotation_verified={}",
        summary.authority_rotation_verified
    );
    println!(
        "member_successions_verified={}",
        summary.member_successions_verified
    );
    println!("prior_merges_accounted={}", summary.prior_merges_accounted);
    println!("retained_merges={}", summary.retained_merges);
    println!("superseded_merges={}", summary.superseded_merges);
    println!("revoked_merges={}", summary.revoked_merges);
    println!("epoch_ledger_entries={}", summary.epoch_ledger_entries);
    println!(
        "epoch_ledger_head={}",
        hex(&summary.epoch_ledger_head_digest)
    );
    println!(
        "current_merge_digest={}",
        hex(&summary.current_merge_digest)
    );
    for gate in &report.gates {
        if gate.required {
            println!(
                "gate={} status={:?} observed={:?} minimum={:?} detail={}",
                gate.name, gate.status, gate.observed, gate.required_minimum, gate.detail,
            );
        }
    }
    println!("series_19_lifecycle_passed={}", report.passed());
    if !report.passed() {
        return Err("Series 19 lifecycle evidence did not satisfy promotion gates".into());
    }
    Ok(())
}

#[cfg(not(unix))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("the Series 19 evaluator currently requires Unix private-file checks".into())
}

#[cfg(unix)]
fn read_federation_authority(
    path: &Path,
) -> Result<CheckpointPowerLossFederationAuthority, Box<dyn std::error::Error>> {
    let (key_id, mut key_bytes) = read_private_key_file(path)?;
    let authority =
        CheckpointPowerLossFederationAuthority::new(CheckpointPowerLossFederationKey::new(
            CheckpointPowerLossFederationKeyId::new(key_id)?,
            key_bytes,
        )?);
    key_bytes.zeroize();
    Ok(authority)
}

#[cfg(unix)]
fn load_lab_authorities(
    directory: &Path,
    key_ids: &HashSet<CheckpointPowerLossLabEvidenceKeyId>,
) -> Result<Vec<CheckpointPowerLossLabEvidenceAuthority>, Box<dyn std::error::Error>> {
    let metadata = std::fs::symlink_metadata(directory)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err("unsafe lab key directory".into());
    }
    let mut authorities = Vec::with_capacity(key_ids.len());
    for key_id in key_ids {
        let path = directory.join(format!("{}.key", hex(&key_id.0)));
        let (observed_id, mut key_bytes) = read_private_key_file(&path)?;
        if observed_id != key_id.0 {
            key_bytes.zeroize();
            return Err("lab key filename and embedded key identifier differ".into());
        }
        authorities.push(CheckpointPowerLossLabEvidenceAuthority::new(
            CheckpointPowerLossLabEvidenceKey::new(*key_id, key_bytes)?,
        ));
        key_bytes.zeroize();
    }
    Ok(authorities)
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
        return Err("unsafe lifecycle artifact directory".into());
    }
    let mut paths = std::fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    if paths.is_empty() {
        return Err("lifecycle artifact directory is empty".into());
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
        return Err("unsafe or oversized lifecycle artifact".into());
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
        return Err("unsafe Series 19 key file".into());
    }
    let mut encoded = [0u8; 48];
    file.read_exact(&mut encoded)?;
    let mut trailing = [0u8; 1];
    if file.read(&mut trailing)? != 0 {
        encoded.zeroize();
        return Err("Series 19 key file contains trailing bytes".into());
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
