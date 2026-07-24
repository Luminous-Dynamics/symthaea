// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verify externally produced Series 17 execution evidence.
//!
//! Usage:
//!   checkpoint_series17_operations_evaluator \
//!     CAMPAIGN.bin OPERATIONS_PLAN.bin SEALED_RESULTS.bin RESULT_KEY.bin \
//!     SEALED_OPERATIONS.bin OPERATIONS_KEY.bin
//!
//! Key files contain exactly 48 bytes: a 16-byte public identifier followed by
//! a 32-byte secret. Unix key files must be regular, private, effective-user-
//! owned files opened without following symlinks.

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
    CheckpointPowerLossEvidenceKeyId, CheckpointPowerLossOperationsAuthority,
    CheckpointPowerLossOperationsKey, CheckpointPowerLossOperationsKeyId,
    apply_authenticated_power_loss_operations_evidence,
    assemble_checkpoint_operational_trust_evidence, decode_checkpoint_power_loss_campaign,
    decode_checkpoint_power_loss_operations_plan,
};
#[cfg(unix)]
use zeroize::Zeroize;

#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let campaign_path = required_path(&mut arguments, "campaign artifact")?;
    let operations_plan_path = required_path(&mut arguments, "operations-plan artifact")?;
    let result_evidence_path = required_path(&mut arguments, "sealed result evidence")?;
    let result_key_path = required_path(&mut arguments, "result authority key")?;
    let operations_evidence_path = required_path(&mut arguments, "sealed operations evidence")?;
    let operations_key_path = required_path(&mut arguments, "operations authority key")?;
    if arguments.next().is_some() {
        return Err("unexpected trailing arguments".into());
    }

    let campaign = decode_checkpoint_power_loss_campaign(&std::fs::read(campaign_path)?)?;
    let operations = decode_checkpoint_power_loss_operations_plan(
        &campaign,
        &std::fs::read(operations_plan_path)?,
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
    let result_evidence = result_authority.open_campaign_evidence(
        &campaign,
        &std::fs::read(result_evidence_path)?,
        campaign.power_loss_evidence_authority_key_id,
    )?;

    let (operations_key_id, mut operations_key_bytes) =
        read_private_key_file(&operations_key_path)?;
    let operations_authority =
        CheckpointPowerLossOperationsAuthority::new(CheckpointPowerLossOperationsKey::new(
            CheckpointPowerLossOperationsKeyId::new(operations_key_id)?,
            operations_key_bytes,
        )?);
    operations_key_bytes.zeroize();
    if operations_authority.key_id() != operations.operations_authority_key_id {
        return Err("operations authority key does not match operations plan".into());
    }
    let operations_evidence = operations_authority.open_operations_evidence(
        &campaign,
        &operations,
        &result_evidence,
        &std::fs::read(operations_evidence_path)?,
    )?;

    let mut metrics = CheckpointOperationalTrustMetrics::default();
    apply_authenticated_power_loss_operations_evidence(
        &mut metrics,
        &campaign,
        &operations,
        &result_evidence,
        &operations_evidence,
    )?;
    let report = assemble_checkpoint_operational_trust_evidence(
        metrics,
        CheckpointOperationalTrustRequirements::series_17_delta(),
    );
    let summary = operations_evidence.summary(&campaign, &operations, &result_evidence)?;

    println!("schema={}", report.schema);
    println!("campaign_id={}", hex(&campaign.campaign_id));
    println!("campaign_digest={}", hex(&campaign.digest()?));
    println!(
        "operations_plan_digest={}",
        hex(&operations.digest(&campaign)?)
    );
    println!("result_authority_key_id={}", hex(&result_key_id));
    println!("operations_authority_key_id={}", hex(&operations_key_id));
    println!("planned_trials={}", summary.planned_trials);
    println!("completed_proofs={}", summary.completed_proofs);
    println!("unique_labs={}", summary.unique_labs);
    println!("resumed_trials={}", summary.resumed_trials);
    println!("quarantined_trials={}", summary.quarantined_trials);
    println!(
        "journal_concurrency_tests={}",
        operations_evidence.journal_concurrency_tests.len(),
    );
    for gate in &report.gates {
        if gate.required {
            println!(
                "gate={} status={:?} observed={:?} minimum={:?} detail={}",
                gate.name, gate.status, gate.observed, gate.required_minimum, gate.detail,
            );
        }
    }
    println!("series_17_operations_passed={}", report.passed());
    if !report.passed() {
        return Err("Series 17 operations evidence did not satisfy promotion gates".into());
    }
    Ok(())
}

#[cfg(not(unix))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("the Series 17 evaluator currently requires Unix private-file checks".into())
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
        return Err("unsafe Series 17 key file".into());
    }
    let mut encoded = [0u8; 48];
    file.read_exact(&mut encoded)?;
    let mut trailing = [0u8; 1];
    if file.read(&mut trailing)? != 0 {
        encoded.zeroize();
        return Err("Series 17 key file contains trailing bytes".into());
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
