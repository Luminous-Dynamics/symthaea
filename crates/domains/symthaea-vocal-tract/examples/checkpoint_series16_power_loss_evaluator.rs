// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evaluate externally produced sudden-power-loss evidence.
//!
//! Usage:
//!   checkpoint_series16_power_loss_evaluator \
//!     CAMPAIGN.bin SEALED_EVIDENCE.bin PROFILE_KEY.bin RESULT_KEY.bin \
//!     PROFILE_ATTESTATION.bin [PROFILE_ATTESTATION.bin ...]
//!
//! Key files contain exactly 48 bytes: a 16-byte public key identifier followed
//! by a 32-byte secret key. On Unix they must be regular, effective-user-owned,
//! private files and are opened with `O_NOFOLLOW`.
//!
//! This executable does not synthesize passing trial results. Every profile and
//! result artifact must be authenticated under the authorities frozen in the
//! preregistered campaign.

#[cfg(unix)]
use std::fs::OpenOptions;
#[cfg(unix)]
use std::io::Read;
#[cfg(unix)]
use std::path::{Path, PathBuf};

#[cfg(unix)]
use symthaea_vocal_tract::{
    CheckpointPowerLossEvidenceAuthority, CheckpointPowerLossEvidenceKey,
    CheckpointPowerLossEvidenceKeyId, CheckpointPowerLossPromotionRequirements,
    CheckpointStorageProfileAttestationKey, CheckpointStorageProfileAttestationKeyId,
    CheckpointStorageProfileAuthority, assess_checkpoint_power_loss_campaign,
    decode_checkpoint_power_loss_campaign,
};
#[cfg(unix)]
use zeroize::Zeroize;

#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let campaign_path = required_path(&mut arguments, "campaign artifact path")?;
    let evidence_path = required_path(&mut arguments, "sealed evidence artifact path")?;
    let profile_key_path = required_path(&mut arguments, "profile authority key path")?;
    let result_key_path = required_path(&mut arguments, "result authority key path")?;
    let profile_paths = arguments.map(PathBuf::from).collect::<Vec<_>>();
    if profile_paths.is_empty() {
        return Err("at least one sealed storage-profile attestation is required".into());
    }

    let campaign = decode_checkpoint_power_loss_campaign(&std::fs::read(&campaign_path)?)?;
    let (profile_key_id, mut profile_key_bytes) = read_private_key_file(&profile_key_path)?;
    let profile_key = CheckpointStorageProfileAttestationKey::new(
        CheckpointStorageProfileAttestationKeyId::new(profile_key_id)?,
        profile_key_bytes,
    );
    profile_key_bytes.zeroize();
    let profile_authority = CheckpointStorageProfileAuthority::new(profile_key?);
    if profile_authority.key_id() != campaign.storage_profile_authority_key_id {
        return Err("profile authority key does not match the campaign".into());
    }
    let mut profiles = Vec::with_capacity(profile_paths.len());
    for path in &profile_paths {
        profiles.push(profile_authority.open_profile(
            &std::fs::read(path)?,
            campaign.storage_profile_authority_key_id,
        )?);
    }
    let observed_profile_digests = profiles
        .iter()
        .map(|profile| profile.digest())
        .collect::<Result<std::collections::HashSet<_>, _>>()?;
    let expected_profile_digests = campaign
        .storage_profiles
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    if observed_profile_digests != expected_profile_digests {
        return Err("authenticated profile set does not match the campaign".into());
    }

    let (result_key_id, mut result_key_bytes) = read_private_key_file(&result_key_path)?;
    let result_key = CheckpointPowerLossEvidenceKey::new(
        CheckpointPowerLossEvidenceKeyId::new(result_key_id)?,
        result_key_bytes,
    );
    result_key_bytes.zeroize();
    let result_authority = CheckpointPowerLossEvidenceAuthority::new(result_key?);
    if result_authority.key_id() != campaign.power_loss_evidence_authority_key_id {
        return Err("result authority key does not match the campaign".into());
    }
    let evidence = result_authority.open_campaign_evidence(
        &campaign,
        &std::fs::read(&evidence_path)?,
        campaign.power_loss_evidence_authority_key_id,
    )?;
    let report = assess_checkpoint_power_loss_campaign(
        &campaign,
        &evidence,
        CheckpointPowerLossPromotionRequirements::default(),
    )?;

    println!("schema={}", report.schema);
    println!("campaign_id={}", hex(&campaign.campaign_id));
    println!("campaign_digest={}", hex(&campaign.digest()?));
    println!("profile_authority_key_id={}", hex(&profile_key_id));
    println!("result_authority_key_id={}", hex(&result_key_id));
    println!("planned_trials={}", report.summary.planned_trials);
    println!("completed_trials={}", report.summary.completed_trials);
    println!("storage_profiles={}", report.summary.storage_profiles);
    println!(
        "process_crash_trials={}",
        report.summary.process_crash_trials
    );
    println!(
        "virtual_power_cut_trials={}",
        report.summary.virtual_power_cut_trials
    );
    println!(
        "physical_power_cut_trials={}",
        report.summary.physical_power_cut_trials
    );
    println!("clean_recoveries={}", report.summary.clean_recoveries);
    println!(
        "fail_closed_recoveries={}",
        report.summary.fail_closed_recoveries
    );
    println!("silent_corruptions={}", report.summary.silent_corruptions);
    for gate in &report.gates {
        println!(
            "gate={} required={} status={:?} observed={:?} minimum={:?} detail={}",
            gate.name,
            gate.required,
            gate.status,
            gate.observed,
            gate.required_minimum,
            gate.detail,
        );
    }
    println!("power_loss_promotion_passed={}", report.passed());
    if !report.passed() {
        return Err("physical power-loss campaign did not satisfy promotion gates".into());
    }
    Ok(())
}

#[cfg(not(unix))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("the Series 16 evaluator currently requires Unix private-file checks".into())
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
        return Err("unsafe Series 16 key file".into());
    }
    let mut encoded = [0u8; 48];
    file.read_exact(&mut encoded)?;
    let mut trailing = [0u8; 1];
    if file.read(&mut trailing)? != 0 {
        encoded.zeroize();
        return Err("Series 16 key file contains trailing bytes".into());
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
